"""Candidate units of budgeted support-path retrieval (Section 2.1, Section 4).

* A **direct path** e = (x_1 -> ... -> x_k) is a downward support path in the
  multi-granularity index I that starts at a concept and ends at a concept, a
  paper abstract or a section chunk; ``V(e)`` is the set of index keys on the
  path and ``l(e)`` the level of its terminal node.
* A **complementary path set** b = {e_C, e_P, e_M} supports a candidate
  hypothesis (Example 1): a concept path e_C from a bridge family to the
  candidate, a problem-side path e_P to papers on the problem and its
  baselines, and a method-side path e_M to papers on the mechanism and
  assumptions of the method.

Paths below a concept are routed canonically: from a start concept the path
descends to the first child (in taxonomy order) whose extent contains the
target paper, until no child does. Hence every (start, terminal) pair has a
single path, and a search region "below v" (Section 4.3) holds exactly the
paths routed through v.

A query concept absent from I is represented by a *query-local anchor*
(Section 4.1, Step 2) whose id starts with ``~``; it has no index key and is
attached to its closest taxonomy concepts (:mod:`scholarpath.retrieval.grounding`).

Besides index keys, ``V(z)`` may contain *virtual context nodes*
(``hyp:<candidate>`` and ``set:<uid>``): the lines that label a hypothesis and
list the premises of a path set in the assembled context. Treating them as
nodes lets the token cost C(S) account for every rendered line, so that the
context assembled in Step 6 never exceeds the budget B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from scholarpath.index.semantic_index import PAPER, SemanticIndex, chunk_key, concept_key, paper_key
from scholarpath.taxonomy.tree import Taxonomy

DIRECT = "direct"
COMPLEMENTARY = "complementary"
HYP_PREFIX = "hyp:"
SET_PREFIX = "set:"
ANCHOR_PREFIX = "~"


def is_anchor(node_id: Optional[str]) -> bool:
    """Whether ``node_id`` names a query-local anchor rather than a taxonomy concept."""
    return bool(node_id) and node_id.startswith(ANCHOR_PREFIX)


def key_level(key: str) -> str:
    """Level of an index key: concept | paper | chunk (virtual keys: their prefix)."""
    return key.split(":", 1)[0]


def key_id(key: str) -> str:
    return key.split(":", 1)[1]


def is_virtual(key: str) -> bool:
    return key.startswith(HYP_PREFIX) or key.startswith(SET_PREFIX)


@dataclass(frozen=True)
class DirectPath:
    """A downward support path; ``nodes`` are index keys from the start to the terminal."""

    nodes: Tuple[str, ...]
    scope: str  # id of the start concept (or of a query-local anchor)
    facet: str  # taxonomy the path descends in

    @property
    def end(self) -> str:
        return self.nodes[-1]

    @property
    def level(self) -> str:
        """Terminal level l(e) in {concept, paper, chunk}."""
        return key_level(self.end)

    @property
    def paper_id(self) -> Optional[str]:
        """Paper of the terminal node (paper or chunk paths)."""
        for key in reversed(self.nodes):
            if key.startswith(PAPER + ":"):
                return key_id(key)
        return None

    @property
    def uid(self) -> str:
        return f"D|{self.scope}|{self.end}"

    def extend(self, key: str) -> "DirectPath":
        return DirectPath(self.nodes + (key,), self.scope, self.facet)


@dataclass(frozen=True)
class ComplementaryPathSet:
    """b = {e_C, e_P, e_M} supporting the hypothesis that ``candidate`` pairs with ``known``."""

    candidate: str  # candidate concept c (or the anchor id of an assessed concept absent from I)
    candidate_facet: str
    known: str  # known endpoint (scope concept or anchor id)
    baseline: str  # m': the concept of the bridging paper closest to the candidate
    bridge: str  # bridge family f = LCA(m', c); e_C = f -> ... -> c (-> the anchor's neighbour)
    distance: int  # taxonomic distance d(m', c) of the bridge anchors
    phi: float  # bridge discount phi(b) in (0, 1]
    e_C: DirectPath
    e_P: DirectPath  # problem-side premise path
    e_M: DirectPath  # method-side premise path
    known_label: str = ""  # display name of the known endpoint (anchors have no index node)

    @property
    def attach(self) -> str:
        """Concept at which e_C ends: c itself, or the neighbour an anchored candidate is attached to."""
        return key_id(self.e_C.end)

    @property
    def known_path(self) -> DirectPath:
        return self.e_P if self.candidate_facet == "method" else self.e_M

    @property
    def candidate_path(self) -> DirectPath:
        return self.e_M if self.candidate_facet == "method" else self.e_P

    @property
    def uid(self) -> str:
        return f"H|{self.candidate}|{self.e_P.end}|{self.e_M.end}"

    @property
    def virtual_nodes(self) -> Tuple[str, str]:
        return (f"{HYP_PREFIX}{self.candidate}", f"{SET_PREFIX}{self.uid}")

    @property
    def nodes(self) -> Tuple[str, ...]:
        """V(b): distinct keys of e_C, e_P, e_M plus the two context-label nodes (ordered)."""
        seen: Dict[str, None] = {}
        for path in (self.e_C, self.e_P, self.e_M):
            for k in path.nodes:
                seen.setdefault(k, None)
        for k in self.virtual_nodes:
            seen.setdefault(k, None)
        return tuple(seen)

    @property
    def papers(self) -> List[str]:
        out = []
        for path in (self.e_P, self.e_M):
            pid = path.paper_id
            if pid is not None and pid not in out:
                out.append(pid)
        return out


class PathFactory:
    """Builds canonical downward paths over the taxonomies of an index (cached)."""

    def __init__(self, index: SemanticIndex, max_chunks_per_paper: int = 64):
        self.index = index
        self.max_chunks_per_paper = int(max_chunks_per_paper)
        self._route: Dict[Tuple[str, str, str], Tuple[str, ...]] = {}

    def taxonomy(self, facet: str) -> Taxonomy:
        return self.index.taxonomy(facet)

    def concept_chain(self, facet: str, ancestor: str, node: str) -> Tuple[str, ...]:
        """Concept keys from ``ancestor`` down to ``node`` (``ancestor`` must be an ancestor-or-self)."""
        path = self.taxonomy(facet).path(node)
        try:
            i = path.index(ancestor)
        except ValueError as err:
            raise ValueError(f"{ancestor!r} is not an ancestor of {node!r}") from err
        return tuple(concept_key(n) for n in path[i:])

    def route(self, facet: str, start: str, paper_id: str) -> Tuple[str, ...]:
        """Concept ids from ``start`` to the deepest concept reached by canonical routing of a paper."""
        key = (facet, start, paper_id)
        hit = self._route.get(key)
        if hit is not None:
            return hit
        tax = self.taxonomy(facet)
        out = [start]
        cur = start
        while True:
            nxt = None
            for c in tax.nodes[cur].children:
                if paper_id in tax.nodes[c].papers:
                    nxt = c
                    break
            if nxt is None:
                break
            out.append(nxt)
            cur = nxt
        res = tuple(out)
        self._route[key] = res
        return res

    def paper_path(self, facet: str, start: str, paper_id: str, prefix: Tuple[str, ...] = ()) -> DirectPath:
        """Path start -> ... -> paper (concepts on the canonical route, then the abstract)."""
        concepts = tuple(concept_key(n) for n in self.route(facet, start, paper_id))
        return DirectPath(prefix + concepts + (paper_key(paper_id),), start, facet)

    def chunks(self, paper_id: str) -> List[str]:
        """Chunk ids of a paper (at most ``max_chunks_per_paper``, in reading order)."""
        return list(self.index.chunks_of(paper_id)[: self.max_chunks_per_paper])

    def chunk_paths(self, path: DirectPath) -> List[DirectPath]:
        """Selective chunk expansions e+ of a paper path (Section 4.4)."""
        pid = path.paper_id
        if pid is None or path.level != PAPER:
            return []
        return [path.extend(chunk_key(c)) for c in self.chunks(pid)]

    def concept_path(self, facet: str, start: str, node: str, prefix: Tuple[str, ...] = ()) -> DirectPath:
        return DirectPath(prefix + self.concept_chain(facet, start, node), start, facet)


def residual_papers(tax: Taxonomy, node: str, papers: Iterable[str]) -> List[str]:
    """Papers routed to stop at ``node`` (no child contains them), sorted for determinism."""
    kids = tax.nodes[node].children
    return sorted(p for p in papers if not any(p in tax.nodes[c].papers for c in kids))
