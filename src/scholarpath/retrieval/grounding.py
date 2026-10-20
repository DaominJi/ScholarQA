"""Step 2 of Algorithm 2: taxonomy grounding (Section 4.1, Appendix A.3).

Each problem / method concept of the query is matched to the concept
descriptions of T_P / T_M by cosine similarity (``index.concept_matrix``).
The matched nodes are the *scopes* that confine the search:

* up to ``grounding_top_k`` concepts per query concept with similarity at
  least ``grounding_min_sim`` are kept, giving the matched sets U_q and M_q
  with grounding similarities g_u, g_m (Appendix A.3). Similarities are
  normalized so that the best match of each facet has g = 1, hence a query
  with a single matched pair has junction weights in {0, 1} (Section 4.3);
* an unspecified facet defaults to the taxonomy root;
* a concept absent from I (no description reaches ``grounding_min_sim``)
  becomes a *query-local anchor* attached to its ``anchor_top_k`` closest
  concepts. Because the link-prediction encoders read text descriptions, the
  anchor is encoded directly from its query description with
  ``LinkPredictor.encode_query_concept`` (Section 4.3) and registered with
  ``LinkPredictor.set_z``.

The neighbours of an anchor stand in for it as *scopes* of the facet only
when they are at least loosely related to it (cosine >= ``anchor_min_sim``,
by default half of ``grounding_min_sim``): an evidence query must not present
the papers of an unrelated concept as established findings for the asked
pair. When no neighbour qualifies, the facet is flagged ``unmatched`` and
defaults to the root like an unspecified facet. The anchor itself keeps all
``anchor_top_k`` neighbours: hypothesis queries attach it to them (its
premise paths start there, and Eq. (9) scores how well they actually
support it).

What the aspects target given these scopes (junction sub-concepts and
findings J(u_q, m_q) for evidence queries; baselines, unexplored candidates
and their families for discovery; the transfer premises of the given pair
for assessment) is resolved when the search space is built
(:func:`scholarpath.retrieval.search.build_search_space`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scholarpath.config import RetrievalConfig
from scholarpath.embeddings import Embedder, l2_normalize
from scholarpath.index.semantic_index import SemanticIndex, concept_key
from scholarpath.llm.mock import stable_hash
from scholarpath.retrieval.query_understanding import QueryAnalysis
from scholarpath.retrieval.units import ANCHOR_PREFIX
from scholarpath.types import HYPOTHESIS_ASSESSMENT, HYPOTHESIS_DISCOVERY

FACETS = ("problem", "method")


def other_facet(facet: str) -> str:
    return "method" if facet == "problem" else "problem"


@dataclass
class Anchor:
    """A query concept absent from I, attached to its closest taxonomy concepts."""

    anchor_id: str
    facet: str
    text: str
    vector: np.ndarray = field(repr=False)  # description embedding (document encoder)
    neighbors: List[Tuple[str, float]] = field(default_factory=list)  # (node id, cosine)


@dataclass
class Grounding:
    """Scopes (u_q, m_q) of a query with grounding similarities (Appendix A.3)."""

    problem_scopes: List[Tuple[str, float]]  # (node id, g in (0, 1]); best first
    method_scopes: List[Tuple[str, float]]
    anchors: Dict[str, Anchor] = field(default_factory=dict)
    similarities: Dict[str, float] = field(default_factory=dict)  # raw cosine of matched nodes
    unspecified: Dict[str, bool] = field(default_factory=dict)  # facet -> defaulted to the root
    facet_anchor: Dict[str, Optional[str]] = field(default_factory=dict)  # facet -> anchor id
    # facet -> the query names a concept of this facet, but nothing in I is close enough to scope it
    unmatched: Dict[str, bool] = field(default_factory=dict)

    def scopes(self, facet: str) -> List[Tuple[str, float]]:
        return self.problem_scopes if facet == "problem" else self.method_scopes

    def top(self, facet: str) -> str:
        return self.scopes(facet)[0][0]

    def anchor(self, facet: str) -> Optional[Anchor]:
        aid = self.facet_anchor.get(facet)
        return self.anchors.get(aid) if aid else None

    def to_dict(self) -> Dict[str, object]:
        return {
            "problem_scopes": [list(s) for s in self.problem_scopes],
            "method_scopes": [list(s) for s in self.method_scopes],
            "anchors": {
                k: {"facet": a.facet, "text": a.text, "neighbors": [list(n) for n in a.neighbors]}
                for k, a in self.anchors.items()
            },
            "unspecified": dict(self.unspecified),
            "unmatched": {f: True for f, v in self.unmatched.items() if v},
        }


def _normalize(scopes: Dict[str, float]) -> List[Tuple[str, float]]:
    if not scopes:
        return []
    top = max(scopes.values())
    top = top if top > 0 else 1.0
    return sorted(((n, max(1e-6, s / top)) for n, s in scopes.items()), key=lambda t: (-t[1], t[0]))


def _facet_texts(analysis: QueryAnalysis, facet: str) -> List[str]:
    texts = list(analysis.problem_concepts if facet == "problem" else analysis.method_concepts)
    if not texts and analysis.kind == HYPOTHESIS_ASSESSMENT and analysis.given_pair:
        texts = [analysis.given_pair[0] if facet == "problem" else analysis.given_pair[1]]
    return texts


def default_anchor_min_sim(config: RetrievalConfig) -> float:
    """Smallest cosine at which an anchor's neighbour may serve as a scope.

    ``config.anchor_min_sim`` when set, otherwise half of ``grounding_min_sim``.
    """
    value = getattr(config, "anchor_min_sim", None)
    return 0.5 * float(config.grounding_min_sim) if value is None else float(value)


def ground(
    analysis: QueryAnalysis,
    index: SemanticIndex,
    embedder: Embedder,
    config: Optional[RetrievalConfig] = None,
    link_predictor: Optional[object] = None,
    anchor_min_sim: Optional[float] = None,
) -> Grounding:
    """Algorithm 2, line 4: match the query concepts to scopes in T_P and T_M.

    ``anchor_min_sim`` overrides :func:`default_anchor_min_sim`, the cosine an
    anchor's neighbour needs to stand in for it as a scope.
    """
    config = config or RetrievalConfig()
    floor = default_anchor_min_sim(config) if anchor_min_sim is None else float(anchor_min_sim)
    out = Grounding(problem_scopes=[], method_scopes=[])
    for facet in FACETS:
        tax = index.taxonomy(facet)
        texts = _facet_texts(analysis, facet)
        # The candidate side of a discovery query only restricts where candidates are
        # searched; an unmatched description there means "anywhere" (the root).
        searched = analysis.kind == HYPOTHESIS_DISCOVERY and facet == (analysis.searched_side or "method")
        out.unmatched[facet] = False
        if not texts:
            _set(out, facet, [(tax.root_id, 1.0)])
            out.unspecified[facet] = True
            out.facet_anchor[facet] = None
            continue
        ids, mat = index.concept_matrix(facet)
        keep = [i for i, n in enumerate(ids) if n != tax.root_id]
        ids = [ids[i] for i in keep]
        mat = mat[keep]
        qv = l2_normalize(np.asarray(embedder.encode_queries(texts), dtype=np.float32))
        sims = mat @ qv.T if len(ids) else np.zeros((0, len(texts)), dtype=np.float32)
        matched: Dict[str, float] = {}
        anchor_id: Optional[str] = None
        for j, text in enumerate(texts):
            col = sims[:, j] if len(ids) else np.zeros(0)
            order = sorted(range(len(ids)), key=lambda i: (-float(col[i]), ids[i]))
            hits = [i for i in order[: config.grounding_top_k] if float(col[i]) >= config.grounding_min_sim]
            for i in hits:
                matched[ids[i]] = max(matched.get(ids[i], 0.0), float(col[i]))
            if not hits and not searched and order and anchor_id is None:
                anchor_id = _make_anchor(out, index, embedder, facet, text, ids, col, order, config, link_predictor)
        out.similarities.update(matched)
        if matched:
            _set(out, facet, _normalize(matched))
            out.unspecified[facet] = False
        else:
            proxies: Dict[str, float] = {}
            if anchor_id is not None:
                proxies = {n: s for n, s in out.anchors[anchor_id].neighbors if s > 0.0 and s >= floor}
            if proxies:
                _set(out, facet, _normalize(proxies))
                out.unspecified[facet] = False
            else:
                _set(out, facet, [(tax.root_id, 1.0)])
                out.unspecified[facet] = True
                out.unmatched[facet] = not searched
        out.facet_anchor[facet] = anchor_id if not matched else None
    return out


def _set(out: Grounding, facet: str, scopes: List[Tuple[str, float]]) -> None:
    if facet == "problem":
        out.problem_scopes = scopes
    else:
        out.method_scopes = scopes


def _make_anchor(
    out: Grounding,
    index: SemanticIndex,
    embedder: Embedder,
    facet: str,
    text: str,
    ids: Sequence[str],
    col: np.ndarray,
    order: Sequence[int],
    config: RetrievalConfig,
    link_predictor: Optional[object],
) -> str:
    """Create a query-local anchor attached to its top-k closest concepts (Section 4.1)."""
    neighbors = [(ids[i], float(col[i])) for i in order[: max(1, config.anchor_top_k)]]
    vec = l2_normalize(np.asarray(embedder.encode([text]), dtype=np.float32))[0]
    anchor_id = f"{ANCHOR_PREFIX}{facet[0].upper()}{stable_hash(facet, text) % 10 ** 10}"
    out.anchors[anchor_id] = Anchor(anchor_id=anchor_id, facet=facet, text=text, vector=vec, neighbors=neighbors)
    if link_predictor is not None:
        anc = np.stack([index.vector(concept_key(n)) for n, _ in neighbors]).astype(np.float32)
        z = link_predictor.encode_query_concept(facet, vec, anc)
        link_predictor.set_z(anchor_id, facet, z)
    return anchor_id
