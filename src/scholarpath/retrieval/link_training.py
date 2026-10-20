"""Temporally trained cross-taxonomy link prediction (Sec. 4.3, Eq. 8; App. B.5, Eq. 17-18).

Scorer (Eq. 8)
--------------
For a problem concept u and a method concept m,

    s_theta(u, m) = z_P(u)^T z_M(m),   z_P(u) = g_P(d_u, h_u),   z_M(m) = g_M(d_m, h_m),

where d_x is the embedding of the description of concept x and h_x the
descriptions of its taxonomy ancestors. Because g_P and g_M encode text rather
than node ids, an unfamiliar concept absent from the index (a query-local
anchor) is encoded directly from its description (Sec. 4.3).

:class:`TemporalLinkPredictor` implements each encoder g as a small residual
MLP over ``x = [d_x ; h_x ; c_x]``, with h_x the mean ancestor embedding and
``c_x = 1`` iff taxonomy context is present:

    g(x) = S x + W2 GELU(W1 x + b1) + b2.

S is initialised to a shared Gaussian random projection R (entries
N(0, 1/E)) of ``d_x + 0.5 h_x`` and W2 to zero. Before training,
``s_theta(u, m) = (d_u + 0.5 h_u)^T R R^T (d_m + 0.5 h_m)`` is therefore an
unbiased Johnson-Lindenstrauss estimate of the *unnormalised* dot product of
the context-smoothed descriptions: a warm start that is only roughly
rank-correlated with the scaled cosine prior of
:class:`~scholarpath.retrieval.link_prediction.EmbeddingLinkPredictor`
(Spearman about 0.5 for E = 32-128 and 768-d embeddings; no scale, no bias,
plus projection noise), not that prior itself. The MLP branch and S are then
trained with AdamW. When there is no temporal supervision at all,
:func:`train_link_predictor` returns the EmbeddingLinkPredictor prior
instead (:class:`FallbackLinkPredictor`). The subtree bound U(u, v) of
Eq. (12) and the caches (mu_v, rho_v) are inherited from
:class:`~scholarpath.retrieval.link_prediction.LinkPredictor`.

Temporally constructed supervision (Eq. 17)
-------------------------------------------
For a cutoff t, the historical index uses only the papers available by t, and
the papers of the window (t, t + Delta] supply the labels

    D+_{t,Delta} = {(u, m) : J_t(u, m) = {},  app_{(t, t+Delta]}(u, m) = 1},

i.e. problem-method pairs of the historical index that share no paper at the
cutoff but receive a verified application in the window. Window papers shared
by u and m only generate *candidate* annotations; an LLM verifies each one
from the paper's title and abstract (``benchmark.verify_application``).
See :class:`TemporalSnapshot` and :func:`build_supervision`.

Comparison sampling (Eq. 18)
----------------------------
The encoders minimise

    sum_{(u, m+, m-)} log(1 + exp[s(u, m-) - s(u, m+)])            (18)

together with its counterpart for ranking problems (u-, m+). See
:class:`ComparisonSampler` and :meth:`TemporalLinkPredictor.fit`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from scholarpath.config import LinkPredictionConfig
from scholarpath.data.heldout import add_years, normalize_date, paper_dates
from scholarpath.index.semantic_index import SemanticIndex, concept_key
from scholarpath.llm.base import LLMClient
from scholarpath.llm.parsing import JSONParseError
from scholarpath.prompts.benchmark import parse_verdicts, verify_application_request
from scholarpath.retrieval.link_prediction import EmbeddingLinkPredictor, LinkPredictor
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.text import truncate_tokens

log = logging.getLogger("scholarpath.link_training")

PROBLEM, METHOD = "problem", "method"
FACETS = (PROBLEM, METHOD)
Pair = Tuple[str, str]  # (problem concept id, method concept id)

# Cache of LLM verification verdicts: (paper id, problem id, method id) -> applies.
VerificationCache = MutableMapping[Tuple[str, str, str], bool]

PREDICTOR_JSON = "link_predictor.json"
PREDICTOR_NPZ = "link_predictor.npz"


def _other(facet: str) -> str:
    return METHOD if facet == PROBLEM else PROBLEM


# ---------------------------------------------------------------------------
# Historical index at a cutoff (Eq. 17)
# ---------------------------------------------------------------------------


class TemporalSnapshot:
    """Historical view of a semantic index at cutoff t, with its window (t, t + Delta].

    Approximation of "the historical index is constructed using only the
    papers available by t" (App. B.5): instead of rebuilding the taxonomies
    at every cutoff, the concepts of the final taxonomies are kept and their
    extents are restricted to papers dated on or before t,
    ``P_v^t = P_v ∩ {p : date(p) <= t}``. A concept belongs to the historical
    index iff its restricted extent is non-empty, and
    ``J_t(u, m) = P_u^t ∩ P_m^t``. Window papers are those dated in
    (t, t + Delta]; undated papers belong to neither set.

    App. B.5 requires "all cutoffs and horizons [to] lie within the indexed
    corpus": :attr:`exceeds_corpus` is true when the window ends after the
    latest indexed paper (``corpus_end``) or the cutoff precedes the earliest
    one, in which case the window is truncated and links that would be
    realized later in the window act as comparison negatives.
    """

    def __init__(
        self,
        index: SemanticIndex,
        cutoff: Any,
        horizon_years: float,
        dates: Optional[Mapping[str, Any]] = None,
    ):
        self.index = index
        self.cutoff = normalize_date(cutoff)
        if self.cutoff is None:
            raise ValueError("A cutoff date is required")
        self.horizon_years = float(horizon_years)
        self.end = add_years(self.cutoff, self.horizon_years)
        all_papers = set(index.problem.root.papers) | set(index.method.root.papers) | set(index.papers)
        self.dates: Dict[str, str] = {
            p: d for p, d in paper_dates(index.papers, dates).items() if p in all_papers
        }
        self.historical: FrozenSet[str] = frozenset(p for p, d in self.dates.items() if d <= self.cutoff)
        self.window: FrozenSet[str] = frozenset(
            p for p, d in self.dates.items() if self.cutoff < d <= self.end
        )
        self.num_undated = len(all_papers) - len(self.dates)
        self.corpus_start: Optional[str] = min(self.dates.values()) if self.dates else None
        self.corpus_end: Optional[str] = max(self.dates.values()) if self.dates else None
        self._hist_extent: Dict[str, Set[str]] = {}
        self._concepts: Dict[str, List[str]] = {}
        self._co: Dict[str, Set[str]] = {}
        self._window_co: Dict[str, Set[str]] = {}

    @property
    def exceeds_corpus(self) -> bool:
        """Does the cutoff or the window (t, t + Delta] fall outside the dated indexed corpus?"""
        if self.corpus_end is None or self.corpus_start is None:
            return True
        return self.end > self.corpus_end or self.cutoff < self.corpus_start

    # ----------------------------------------------------------- concepts
    def taxonomy(self, facet: str) -> Taxonomy:
        return self.index.taxonomy(facet)

    def historical_extent(self, node_id: str) -> Set[str]:
        """P_v^t: papers of concept v available by the cutoff."""
        ext = self._hist_extent.get(node_id)
        if ext is None:
            ext = {p for p in self.index.extent(node_id) if p in self.historical}
            self._hist_extent[node_id] = ext
        return ext

    def in_history(self, node_id: str) -> bool:
        return bool(self.historical_extent(node_id))

    def concepts(self, facet: str, include_root: bool = False) -> List[str]:
        """Concepts of the historical index (non-empty restricted extent)."""
        key = f"{facet}:{int(include_root)}"
        out = self._concepts.get(key)
        if out is None:
            tax = self.taxonomy(facet)
            out = [
                nid for nid in tax.nodes
                if (include_root or nid != tax.root_id) and self.in_history(nid)
            ]
            self._concepts[key] = out
        return out

    def shared(self, u: str, m: str) -> Set[str]:
        """J_t(u, m): papers available by t that study problem u with method m."""
        a, b = self.historical_extent(u), self.historical_extent(m)
        if len(a) > len(b):
            a, b = b, a
        return {p for p in a if p in b}

    def co_concepts(self, node_id: str) -> Set[str]:
        """Concepts x of the other taxonomy with J_t(node, x) non-empty (incl. ancestors)."""
        out = self._co.get(node_id)
        if out is None:
            other = _other(self.index.facet_of(node_id))
            out = set()
            for p in self.historical_extent(node_id):
                out |= self.index.paper_concepts(p, other)
            self._co[node_id] = out
        return out

    def window_co_concepts(self, node_id: str) -> Set[str]:
        """Concepts of the other taxonomy sharing at least one *window* paper with ``node_id``."""
        out = self._window_co.get(node_id)
        if out is None:
            other = _other(self.index.facet_of(node_id))
            out = set()
            for p in self.index.extent(node_id):
                if p in self.window:
                    out |= self.index.paper_concepts(p, other)
            self._window_co[node_id] = out
        return out

    def most_specific(self, paper_id: str, facet: str) -> List[str]:
        """Deepest *historical* concepts of a paper (no historical child also contains it)."""
        tax = self.taxonomy(facet)
        nodes = {n for n in self.index.paper_concepts(paper_id, facet) if n != tax.root_id and self.in_history(n)}
        return sorted(n for n in nodes if not any(c in nodes for c in tax.nodes[n].children))

    def summary(self) -> Dict[str, Any]:
        return {
            "cutoff": self.cutoff,
            "end": self.end,
            "historical_papers": len(self.historical),
            "window_papers": len(self.window),
            "undated_papers": self.num_undated,
            "corpus_start": self.corpus_start,
            "corpus_end": self.corpus_end,
            "window_exceeds_corpus": self.exceeds_corpus,
            "historical_problems": len(self.concepts(PROBLEM)),
            "historical_methods": len(self.concepts(METHOD)),
        }


def window_candidates(snapshot: TemporalSnapshot) -> Dict[Pair, List[str]]:
    """Candidate annotations of the window: pair -> window papers (sorted by date).

    Each window paper p generates the pairs (u, m) of its *most specific*
    historical problem and method concepts that share no paper at the cutoff
    (J_t(u, m) = {}). If the most specific pair already co-occurred, so did
    every ancestor pair (extents are closed upwards), hence restricting to the
    deepest historical concepts loses no new link while avoiding one label per
    ancestor combination.
    """
    cands: Dict[Pair, List[str]] = {}
    for p in sorted(snapshot.window, key=lambda x: (snapshot.dates[x], x)):
        probs = snapshot.most_specific(p, PROBLEM)
        meths = snapshot.most_specific(p, METHOD)
        for u in probs:
            co = snapshot.co_concepts(u)
            for m in meths:
                if m not in co:
                    cands.setdefault((u, m), []).append(p)
    return cands


def most_specific_pairs(pairs: Iterable[Pair], problem: Taxonomy, method: Taxonomy) -> List[Pair]:
    """Drop every pair dominated by a more specific pair of the set.

    ``(a, b)`` is dominated when another pair ``(u, m)`` of the set has
    ``u ⪯ a`` and ``m ⪯ b`` (⪯ = descendant-or-self): the more specific label
    already implies it.
    """
    pairs = set(pairs)
    dominated: Set[Pair] = set()
    for u, m in pairs:
        for a in problem.ancestors(u, include_self=True):
            for b in method.ancestors(m, include_self=True):
                if (a, b) != (u, m) and (a, b) in pairs:
                    dominated.add((a, b))
    return sorted(pairs - dominated)


# ---------------------------------------------------------------------------
# LLM verification of candidate annotations
# ---------------------------------------------------------------------------


def _concept_brief(index: SemanticIndex, nid: str) -> Dict[str, str]:
    node = index.concept(nid)
    return {"id": nid, "name": node.name, "definition": node.definition}


def verify_candidates(
    index: SemanticIndex,
    candidates: Mapping[Pair, Sequence[str]],
    llm: LLMClient,
    *,
    max_papers_per_pair: Optional[int] = 3,
    pairs_per_call: int = 12,
    abstract_tokens: int = 400,
    component: str = "link_training",
    cache: Optional[VerificationCache] = None,
) -> Tuple[Dict[Pair, List[str]], Dict[Pair, List[str]]]:
    """Verify candidate annotations with the LLM (application indicator of Eq. 17).

    For every pair, its window papers are checked in date order (at most
    ``max_papers_per_pair``) until one is verified. Each LLM call shows one
    paper with up to ``pairs_per_call`` of the pairs it generated. Returns
    ``(accepted pair -> verifying paper(s), rejected pair -> checked papers)``.
    ``cache`` memoises verdicts across cutoffs with overlapping windows.
    """
    cache = cache if cache is not None else {}
    limit = max_papers_per_pair if max_papers_per_pair is not None else max((len(v) for v in candidates.values()), default=0)
    accepted: Dict[Pair, List[str]] = {}
    checked: Dict[Pair, List[str]] = defaultdict(list)
    for r in range(max(0, int(limit))):
        active = [pair for pair, plist in candidates.items() if pair not in accepted and r < len(plist)]
        if not active:
            break
        pending: Dict[str, List[Pair]] = defaultdict(list)
        for pair in active:
            pid = candidates[pair][r]
            key = (pid, pair[0], pair[1])
            if key in cache:
                checked[pair].append(pid)
                if cache[key]:
                    accepted[pair] = [pid]
            else:
                pending[pid].append(pair)
        step = max(1, int(pairs_per_call))
        jobs: List[Tuple[str, List[Pair]]] = []
        for pid in sorted(pending):
            plist = sorted(pending[pid])
            for i in range(0, len(plist), step):
                jobs.append((pid, plist[i : i + step]))

        def run(job: Tuple[str, List[Pair]]) -> Dict[Pair, bool]:
            pid, pairs = job
            paper = index.papers.get(pid)
            brief = {
                "id": pid,
                "title": paper.title if paper else "",
                "abstract": truncate_tokens(paper.abstract or "", abstract_tokens) if paper else "",
            }
            keyed = [
                {"key": f"A{i}", "problem": _concept_brief(index, u), "method": _concept_brief(index, m)}
                for i, (u, m) in enumerate(pairs, 1)
            ]
            req = verify_application_request(brief, keyed, component=component)
            try:
                obj, _ = llm.complete_json(req)
            except JSONParseError as err:
                log.warning("Verification of %s failed to parse (%s); treating its candidates as rejected", pid, err)
                obj = {}
            verdicts = parse_verdicts(obj, [k["key"] for k in keyed])
            return {pair: verdicts[f"A{i}"] for i, pair in enumerate(pairs, 1)}

        for (pid, _), result in zip(jobs, llm.map(run, jobs)):
            for pair, ok in result.items():
                cache[(pid, pair[0], pair[1])] = bool(ok)
                checked[pair].append(pid)
                if ok and pair not in accepted:
                    accepted[pair] = [pid]
    rejected = {pair: list(checked.get(pair, [])) for pair in candidates if pair not in accepted}
    return accepted, rejected


# ---------------------------------------------------------------------------
# Supervision sets (Eq. 17)
# ---------------------------------------------------------------------------


@dataclass
class SupervisionSet:
    """D+_{t, Delta} for one cutoff, with its provenance."""

    cutoff: str
    end: str
    horizon_years: float
    positives: List[Pair]
    evidence: Dict[Pair, List[str]]  # positive -> window paper(s) supporting it
    candidates: Dict[Pair, List[str]]  # every candidate annotation -> window papers
    rejected: List[Pair]
    verified: bool  # True iff an LLM verified the candidates
    snapshot: TemporalSnapshot = field(repr=False)

    def summary(self) -> Dict[str, Any]:
        out = self.snapshot.summary()
        out.update({
            "horizon_years": self.horizon_years,
            "candidates": len(self.candidates),
            "positives": len(self.positives),
            "rejected": len(self.rejected),
            "verified": self.verified,
        })
        return out


def build_supervision(
    index: SemanticIndex,
    cutoff: Any,
    horizon_years: float,
    llm: Optional[LLMClient] = None,
    dates: Optional[Mapping[str, Any]] = None,
    *,
    verify: bool = True,
    max_papers_per_pair: Optional[int] = 3,
    component: str = "link_training",
    cache: Optional[VerificationCache] = None,
) -> SupervisionSet:
    """Construct D+_{t, Delta} (Eq. 17) for cutoff ``t`` and horizon ``Delta``.

    1. Historical index at t (:class:`TemporalSnapshot`).
    2. Candidate annotations: window papers shared by the most specific
       historical concepts u and m with J_t(u, m) = {}
       (:func:`window_candidates`).
    3. Application indicator: an LLM verifies each candidate from the paper's
       title and abstract (:func:`verify_candidates`). Without an LLM (or with
       ``verify=False``) every candidate is accepted and the set is flagged
       ``verified=False``.
    4. Verified pairs dominated by a more specific verified pair are dropped
       (:func:`most_specific_pairs`).

    A warning is logged when the cutoff or window falls outside the dated
    indexed corpus (App. B.5; :attr:`TemporalSnapshot.exceeds_corpus`).
    """
    snap = TemporalSnapshot(index, cutoff, horizon_years, dates)
    if snap.exceeds_corpus:
        log.warning(
            "Temporal supervision at %s: the window (%s, %s] is not within the indexed corpus [%s, %s] "
            "(App. B.5); links realized after the corpus end act as comparison negatives",
            snap.cutoff, snap.cutoff, snap.end, snap.corpus_start, snap.corpus_end,
        )
    cands = window_candidates(snap)
    if llm is not None and verify:
        accepted, rejected = verify_candidates(
            index, cands, llm, max_papers_per_pair=max_papers_per_pair, component=component, cache=cache
        )
        verified = True
    else:
        if cands:
            log.warning(
                "Temporal supervision at %s: no LLM verification; accepting %d candidate annotations unverified",
                snap.cutoff, len(cands),
            )
        accepted = {pair: list(papers) for pair, papers in cands.items()}
        rejected = {}
        verified = False
    positives = most_specific_pairs(accepted, index.problem, index.method)
    return SupervisionSet(
        cutoff=snap.cutoff,
        end=snap.end,
        horizon_years=snap.horizon_years,
        positives=positives,
        evidence={p: accepted[p] for p in positives},
        candidates={p: list(v) for p, v in cands.items()},
        rejected=sorted(rejected),
        verified=verified,
        snapshot=snap,
    )


def temporal_supervision(
    index: SemanticIndex,
    cutoff: Any,
    horizon_years: float,
    llm: Optional[LLMClient] = None,
    dates: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> List[Pair]:
    """The positive pairs D+_{t, Delta} of Eq. (17); see :func:`build_supervision`."""
    return build_supervision(index, cutoff, horizon_years, llm, dates, **kwargs).positives


_LATEST = "9999-12-31"


def static_supervision(
    index: SemanticIndex,
    positives: Iterable[Sequence[str]],
    cutoff: Any = None,
    dates: Optional[Mapping[str, Any]] = None,
) -> SupervisionSet:
    """Wrap plain (problem, method) positives in a :class:`SupervisionSet`.

    Supports the generic ``LinkPredictor.fit(index, positives)`` contract.
    The comparison candidates are the concepts of the historical index at
    ``cutoff`` (J_t(u, m) = {} for the anchor u). Without a cutoff every
    indexed paper counts as history (undated ones included), so comparisons
    are pairs that never share a paper. No window exists (Delta = 0), and the
    labels are marked unverified. Raises ``ValueError`` for ids that are not
    a (problem concept, method concept) pair of the index.
    """
    pairs = []
    for pair in positives:
        if len(pair) != 2:
            raise ValueError(f"Expected (problem id, method id) pairs, got {pair!r}")
        u, m = str(pair[0]), str(pair[1])
        if u not in index.problem.nodes or m not in index.method.nodes:
            raise ValueError(f"({u!r}, {m!r}) is not a (problem concept, method concept) pair of the index")
        pairs.append((u, m))
    if cutoff is None:
        known = paper_dates(index.papers, dates)
        all_papers = set(index.problem.root.papers) | set(index.method.root.papers) | set(index.papers)
        dates = {p: known.get(p, "0001-01-01") for p in all_papers}
        cutoff = _LATEST
    snap = TemporalSnapshot(index, cutoff, 0.0, dates)
    positives_sorted = sorted(set(pairs))
    return SupervisionSet(
        cutoff=snap.cutoff,
        end=snap.end,
        horizon_years=0.0,
        positives=positives_sorted,
        evidence={p: [] for p in positives_sorted},
        candidates={},
        rejected=[],
        verified=False,
        snapshot=snap,
    )


def _as_supervision(
    index: SemanticIndex,
    supervision: Any,
    cutoff: Any = None,
    dates: Optional[Mapping[str, Any]] = None,
) -> List[SupervisionSet]:
    """Normalise ``fit``'s input: supervision set(s), or plain (u, m) pairs (:func:`static_supervision`)."""
    if isinstance(supervision, SupervisionSet):
        return [supervision]
    items = list(supervision)
    if not items:
        return []
    if all(isinstance(s, SupervisionSet) for s in items):
        return items
    if all(isinstance(s, (tuple, list)) and len(s) == 2 and all(isinstance(x, str) for x in s) for s in items):
        return [static_supervision(index, items, cutoff, dates)]
    raise TypeError(
        "fit() expects SupervisionSet objects (build_supervision) or (problem id, method id) pairs, "
        f"got {type(items[0]).__name__}"
    )


# ---------------------------------------------------------------------------
# Comparison sampling (App. B.5)
# ---------------------------------------------------------------------------


def related_concepts(tax: Taxonomy, node_id: str, levels: int = 2) -> Set[str]:
    """Closely related alternatives of a concept: siblings, cousins and their subtrees.

    Takes the subtree of the ``levels``-th ancestor (never the root unless the
    concept is a child of the root, in which case only its siblings are
    returned) minus the concept's ancestors, itself and its descendants.
    """
    anc = tax.ancestors(node_id)
    if not anc:
        return set()
    if anc[0] == tax.root_id:
        return {c for c in tax.children(tax.root_id) if c != node_id}
    top = anc[min(levels, len(anc)) - 1]
    if top == tax.root_id:
        top = anc[-2]
    excluded = set(tax.ancestors(node_id, include_self=True)) | set(tax.descendants(node_id))
    return {n for n in tax.descendants(top) if n not in excluded}


class _Pool:
    """Admissible comparisons of one anchor at one cutoff, stored by exclusion.

    ``candidates`` holds the sorted rows of the historical concepts of the
    candidate facet (shared by every anchor of the cutoff) and ``blocked``
    the sorted positions in it that are not admissible. The i-th admissible
    position is ``i + #{j : blocked[j] - j <= i}``, so uniform draws need
    neither a materialised pool nor rejection sampling, and memory is
    O(#blocked) per anchor instead of O(#concepts) per positive.
    """

    __slots__ = ("candidates", "blocked", "_shift", "size")

    def __init__(self, candidates: np.ndarray, blocked: np.ndarray):
        self.candidates = candidates
        self.blocked = blocked
        self._shift = blocked - np.arange(len(blocked), dtype=blocked.dtype)
        self.size = int(len(candidates) - len(blocked))

    def rows(self, positions: np.ndarray) -> np.ndarray:
        """Rows of the admissible comparisons with the given ranks (0 <= rank < size)."""
        positions = np.asarray(positions, dtype=np.int64)
        return self.candidates[positions + np.searchsorted(self._shift, positions, side="right")].astype(np.int64)

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` uniform comparisons (without replacement when the pool is large enough)."""
        return self.rows(rng.choice(self.size, size=n, replace=self.size < n))

    def members(self) -> np.ndarray:
        """Every admissible comparison (for inspection and tests)."""
        return self.rows(np.arange(self.size, dtype=np.int64))


@dataclass
class _Item:
    direction: int  # 0: rank methods for a problem anchor; 1: rank problems for a method anchor
    anchor: int  # row in the anchor facet
    positive: int  # row in the candidate facet
    pool: int  # index into ComparisonSampler.pools (shared by the anchor's items at this cutoff)
    hard: np.ndarray  # closely related admissible comparisons (int32 rows in the candidate facet)


class ComparisonSampler:
    """Comparison examples for Eq. (18) and its problem-ranking counterpart (App. B.5).

    For each positive (u, m+) of every supervision set:

    * method ranking: comparison methods m- are drawn from the historical
      candidates of that cutoff, i.e. method concepts of the historical index
      with J_t(u, m-) = {}; a ``hard_fraction`` of them are closely related
      alternatives of m+ (siblings / cousins in T_M, :func:`related_concepts`);
    * problem ranking: symmetrically, problems u- with J_t(u-, m+) = {}.

    Excluded from the comparison pool are: every other known positive of the
    anchor in *any* training window, closed under ancestors (if (u', m') is a
    positive with u' ⪯ u then (u, a) for every a ⪰ m' is realized too), so that
    multiple known relationships stay positives instead of competing as
    negatives -- this closure also contains the positive and its ancestors;
    pairs co-occurring in a window paper of the same cutoff (their candidate
    annotation may have been rejected, so they are not clean negatives); and
    every concept in ``exclude_concepts`` (held-out endpoints, whose incident
    labels are also dropped). The pool depends only on the anchor and the
    cutoff, so it is stored once per (cutoff, direction, anchor) as a
    :class:`_Pool`.
    """

    def __init__(
        self,
        index: SemanticIndex,
        supervision: Sequence[SupervisionSet],
        negatives_per_positive: int = 8,
        hard_fraction: float = 0.5,
        exclude_concepts: Iterable[str] = (),
    ):
        self.index = index
        self.k = max(1, int(negatives_per_positive))
        self.hard_fraction = float(min(max(hard_fraction, 0.0), 1.0))
        self.exclude = set(exclude_concepts)
        self.rows: Dict[str, Dict[str, int]] = {}
        self.ids: Dict[str, List[str]] = {}
        for facet in FACETS:
            tax = index.taxonomy(facet)
            ids = [n for n in tax.nodes if n != tax.root_id]
            self.ids[facet] = ids
            self.rows[facet] = {n: i for i, n in enumerate(ids)}

        positives = [
            pair for s in supervision for pair in s.positives
            if pair[0] not in self.exclude and pair[1] not in self.exclude
        ]
        self.num_positive_labels = len(positives)
        self.positive_pairs = sorted(set(positives))
        realized: Dict[str, Set[str]] = defaultdict(set)
        for u, m in self.positive_pairs:
            anc_u = index.problem.ancestors(u, include_self=True)
            anc_m = index.method.ancestors(m, include_self=True)
            for a in anc_u:
                realized[a].update(anc_m)
            for b in anc_m:
                realized[b].update(anc_u)

        self.items: List[_Item] = []
        self.pools: List[_Pool] = []
        related_cache: Dict[str, Set[str]] = {}
        for s in supervision:
            snap = s.snapshot
            # Historical candidate rows per facet (shared by every anchor of this cutoff).
            cand_rows: Dict[str, np.ndarray] = {}
            position: Dict[str, Dict[int, int]] = {}
            for facet in FACETS:
                rows = self.rows[facet]
                arr = np.asarray(sorted(rows[c] for c in snap.concepts(facet) if c in rows), dtype=np.int32)
                cand_rows[facet] = arr
                position[facet] = {int(r): i for i, r in enumerate(arr)}
            pool_of: Dict[Tuple[int, str], int] = {}
            blocked_of: Dict[int, Set[int]] = {}
            for u, m in s.positives:
                if u in self.exclude or m in self.exclude:
                    continue
                for direction, (anchor, pos) in enumerate(((u, m), (m, u))):
                    cand_facet = METHOD if direction == 0 else PROBLEM
                    rows, pos_in = self.rows[cand_facet], position[cand_facet]
                    key = (direction, anchor)
                    if key not in pool_of:
                        blocked_ids = (
                            snap.co_concepts(anchor)
                            | snap.window_co_concepts(anchor)
                            | realized.get(anchor, set())
                            | self.exclude
                        )
                        blocked = sorted(
                            pos_in[rows[c]] for c in blocked_ids if c in rows and rows[c] in pos_in
                        )
                        pool_of[key] = len(self.pools)
                        blocked_of[pool_of[key]] = set(blocked)
                        self.pools.append(_Pool(cand_rows[cand_facet], np.asarray(blocked, dtype=np.int32)))
                    pool = self.pools[pool_of[key]]
                    if pool.size <= 0:
                        continue
                    if pos not in related_cache:
                        related_cache[pos] = related_concepts(index.taxonomy(cand_facet), pos)
                    blocked_pos = blocked_of[pool_of[key]]
                    hard = sorted(
                        rows[c] for c in related_cache[pos]
                        if c in rows and rows[c] in pos_in and pos_in[rows[c]] not in blocked_pos
                    )
                    anchor_facet = PROBLEM if direction == 0 else METHOD
                    self.items.append(_Item(
                        direction=direction,
                        anchor=self.rows[anchor_facet][anchor],
                        positive=rows[pos],
                        pool=pool_of[key],
                        hard=np.asarray(hard, dtype=np.int32),
                    ))

    def __len__(self) -> int:
        return len(self.items)

    def concepts(self) -> Dict[str, List[str]]:
        return self.ids

    def pool(self, item: _Item) -> np.ndarray:
        """All admissible comparison rows of an item (materialised; for inspection and tests)."""
        return self.pools[item.pool].members()

    def nbytes(self) -> int:
        """Memory held by the comparison pools and hard-negative arrays."""
        shared = {id(p.candidates): p.candidates.nbytes for p in self.pools}
        own = sum(p.blocked.nbytes + p._shift.nbytes for p in self.pools)
        return int(sum(shared.values()) + own + sum(it.hard.nbytes for it in self.items))

    def sample(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Draw ``negatives_per_positive`` comparisons per item (resampled every epoch)."""
        dirs, anchors, poss, negs = [], [], [], []
        n_hard_target = int(round(self.k * self.hard_fraction))
        for it in self.items:
            n_hard = min(n_hard_target, len(it.hard))
            chosen: List[np.ndarray] = []
            if n_hard:
                chosen.append(rng.choice(it.hard, size=n_hard, replace=False))
            n_rest = self.k - n_hard
            if n_rest:
                chosen.append(self.pools[it.pool].draw(rng, n_rest))
            neg = np.concatenate(chosen)
            dirs.append(np.full(len(neg), it.direction, dtype=np.int64))
            anchors.append(np.full(len(neg), it.anchor, dtype=np.int64))
            poss.append(np.full(len(neg), it.positive, dtype=np.int64))
            negs.append(neg.astype(np.int64))
        if not dirs:
            empty = np.zeros(0, dtype=np.int64)
            return {"direction": empty, "anchor": empty, "positive": empty, "negative": empty}
        return {
            "direction": np.concatenate(dirs),
            "anchor": np.concatenate(anchors),
            "positive": np.concatenate(poss),
            "negative": np.concatenate(negs),
        }


# ---------------------------------------------------------------------------
# Encoders g_P, g_M
# ---------------------------------------------------------------------------

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    t = np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x ** 3))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * _SQRT_2_OVER_PI * (1.0 + 3 * 0.044715 * x ** 2)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


_PARAM_NAMES = ("W1", "b1", "W2", "b2", "S")
_DECAYED = ("W1", "W2", "S")


class _Encoder:
    """g(x) = S x + W2 GELU(W1 x + b1) + b2 (NumPy, with manual gradients)."""

    def __init__(self, params: Dict[str, np.ndarray]):
        self.p = {k: np.asarray(params[k], dtype=np.float32) for k in _PARAM_NAMES}

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        p = self.p
        A = X @ p["W1"] + p["b1"]
        H = _gelu(A)
        Z = H @ p["W2"] + p["b2"] + X @ p["S"]
        return Z, (X, A, H)

    def backward(self, cache: Tuple[np.ndarray, np.ndarray, np.ndarray], dZ: np.ndarray) -> Dict[str, np.ndarray]:
        X, A, H = cache
        p = self.p
        dH = dZ @ p["W2"].T
        dA = dH * _gelu_grad(A)
        return {
            "W2": H.T @ dZ,
            "b2": dZ.sum(axis=0),
            "S": X.T @ dZ,
            "W1": X.T @ dA,
            "b1": dA.sum(axis=0),
        }


@contextlib.contextmanager
def _blas_threads(limit: Optional[int]) -> Iterator[None]:
    """Limit BLAS threads (via ``threadpoolctl``, a scikit-learn dependency) inside the block.

    The encoders are trained with many small matrix products, for which
    multi-threaded BLAS mostly adds synchronisation overhead (and stalls badly
    on a busy machine). ``None`` leaves the thread pools untouched; without
    ``threadpoolctl`` the limit is silently skipped.
    """
    if limit is None:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        yield
        return
    with threadpool_limits(limits=int(limit), user_api="blas"):
        yield


class _AdamW:
    """Adam with decoupled weight decay on the weight matrices."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float, weight_decay: float):
        self.params = params
        self.lr = float(lr)
        self.wd = float(weight_decay)
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: Mapping[str, np.ndarray]) -> None:
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.t += 1
        for k, g in grads.items():
            g = g.astype(np.float32)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            param = self.params[k]
            if k.split(".")[-1] in _DECAYED:
                param *= 1.0 - self.lr * self.wd
            param -= self.lr * mh / (np.sqrt(vh) + eps)


# ---------------------------------------------------------------------------
# Temporal link predictor
# ---------------------------------------------------------------------------


class EmptySupervisionError(ValueError):
    """Raised when temporal supervision yields no comparison example to train on."""


class TemporalLinkPredictor(LinkPredictor):
    """s_theta(u, m) = g_P(d_u, h_u)^T g_M(d_m, h_m), trained on temporal supervision (App. B.5)."""

    name = "temporal"

    def __init__(self, config: Optional[LinkPredictionConfig] = None, input_dim: Optional[int] = None):
        super().__init__()
        self.config = config or LinkPredictionConfig()
        self.input_dim: Optional[int] = None
        self.encoders: Dict[str, _Encoder] = {}
        self.loss_curve: List[float] = []
        self.report: Dict[str, Any] = {}
        if input_dim is not None:
            self.initialize(int(input_dim))

    # ------------------------------------------------------------ params
    @property
    def fitted(self) -> bool:
        return bool(self.encoders)

    def initialize(self, input_dim: int, seed: Optional[int] = None) -> "TemporalLinkPredictor":
        """Initialise g_P and g_M: S = random projection of d + 0.5 h, W2 = 0.

        The initial scorer is a Johnson-Lindenstrauss estimate of the
        unnormalised dot product of the context-smoothed descriptions (a warm
        start; see the module docstring), not the EmbeddingLinkPredictor prior.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        D, H, E = int(input_dim), int(cfg.hidden_dim), int(cfg.embed_dim)
        din = 2 * D + 1
        R = (rng.standard_normal((D, E)) / math.sqrt(E)).astype(np.float32)
        S = np.zeros((din, E), dtype=np.float32)
        S[:D] = R
        S[D : 2 * D] = 0.5 * R
        self.encoders = {}
        for facet in FACETS:
            self.encoders[facet] = _Encoder({
                "W1": (rng.standard_normal((din, H)) * math.sqrt(2.0 / (din + H))).astype(np.float32),
                "b1": np.zeros(H, dtype=np.float32),
                "W2": np.zeros((H, E), dtype=np.float32),
                "b2": np.zeros(E, dtype=np.float32),
                "S": S.copy(),
            })
        self.input_dim = D
        return self

    @staticmethod
    def features(description_vec: np.ndarray, ancestor_vecs: Optional[np.ndarray]) -> np.ndarray:
        """x = [d ; mean ancestor embedding ; context flag] (zeros and flag 0 without context)."""
        d = np.asarray(description_vec, dtype=np.float32).reshape(-1)
        anc = None if ancestor_vecs is None else np.asarray(ancestor_vecs, dtype=np.float32)
        if anc is not None and anc.size:
            h = anc.reshape(-1, d.shape[0]).mean(axis=0)
            flag = 1.0
        else:
            h = np.zeros_like(d)
            flag = 0.0
        return np.concatenate([d, h, np.array([flag], dtype=np.float32)])

    def encode_concept(self, facet: str, description_vec: np.ndarray, ancestor_vecs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TemporalLinkPredictor is not trained; call fit() or load() first")
        if facet not in self.encoders:
            raise KeyError(f"Unknown facet {facet!r}; expected 'problem' or 'method'")
        d = np.asarray(description_vec, dtype=np.float32).reshape(-1)
        if d.shape[0] != self.input_dim:
            raise ValueError(
                f"Description embedding has dimension {d.shape[0]}, but the predictor was trained on "
                f"{self.input_dim}-d embeddings (use the same embedder for training and inference)"
            )
        z, _ = self.encoders[facet].forward(self.features(d, ancestor_vecs)[None, :])
        return z[0].astype(np.float32)

    def prepare(self, index: SemanticIndex) -> "TemporalLinkPredictor":
        """Cache z_P / z_M for every concept and the subtree balls (mu_v, rho_v) of Eq. (12)."""
        with _blas_threads(1):  # one small product per concept
            super().prepare(index)
        return self

    # ---------------------------------------------------------- training
    def _feature_tables(self, index: SemanticIndex, ids: Mapping[str, List[str]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Per facet: (features with taxonomy context, description-only features), aligned with ``ids``."""
        tables = {}
        for facet in FACETS:
            tax = index.taxonomy(facet)
            ctx, desc = [], []
            for nid in ids[facet]:
                d = index.vector(concept_key(nid))
                anc_ids = tax.ancestors(nid)
                anc = np.stack([index.vector(concept_key(a)) for a in anc_ids]) if anc_ids else None
                ctx.append(self.features(d, anc))
                desc.append(self.features(d, None))
            dim = 2 * index.dim + 1
            tables[facet] = (
                np.stack(ctx) if ctx else np.zeros((0, dim), dtype=np.float32),
                np.stack(desc) if desc else np.zeros((0, dim), dtype=np.float32),
            )
        return tables

    def fit(  # type: ignore[override]
        self,
        index: SemanticIndex,
        supervision: Union[SupervisionSet, Sequence[SupervisionSet], Iterable[Sequence[str]]],
        exclude_concepts: Iterable[str] = (),
        *,
        cutoff: Any = None,
        dates: Optional[Mapping[str, Any]] = None,
        epochs: Optional[int] = None,
        batch_size: int = 64,
        hard_fraction: float = 0.5,
        seed: Optional[int] = None,
        blas_threads: Optional[int] = 1,
    ) -> "TemporalLinkPredictor":
        """Minimise Eq. (18) and its problem-ranking counterpart with AdamW.

        ``supervision`` is one or more :class:`SupervisionSet` (temporal
        supervision, Eq. 17; each set supplies the historical comparison
        candidates of its cutoff), or -- following the generic
        ``LinkPredictor.fit(index, positives)`` contract -- plain
        ``(problem id, method id)`` positives, wrapped by
        :func:`static_supervision` with comparisons drawn from the historical
        index at ``cutoff`` (default: the whole index).

        Comparison negatives are resampled every epoch (:class:`ComparisonSampler`);
        the taxonomy context of the *anchor* endpoint (the side a query-local
        concept would occupy) is omitted on a ``config.context_dropout``
        fraction of examples, so that the scorer also learns the
        description-only case. The mean loss of every epoch is recorded in
        :attr:`loss_curve`. Call :meth:`prepare` afterwards to cache z and the
        subtree balls of Eq. (12). ``blas_threads`` caps BLAS threads during
        training (``None`` = library default). Raises
        :class:`EmptySupervisionError` when there is no comparison example.
        """
        sets = _as_supervision(index, supervision, cutoff, dates)
        with _blas_threads(blas_threads):
            return self._fit(index, sets, exclude_concepts, epochs, batch_size, hard_fraction, seed)

    def _fit(
        self,
        index: SemanticIndex,
        supervision: Sequence[SupervisionSet],
        exclude_concepts: Iterable[str],
        epochs: Optional[int],
        batch_size: int,
        hard_fraction: float,
        seed: Optional[int],
    ) -> "TemporalLinkPredictor":
        cfg = self.config
        seed = cfg.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        if index.dim <= 0:
            raise ValueError("The semantic index has no embeddings")
        if not self.fitted:
            self.initialize(index.dim, seed=seed)
        elif self.input_dim != index.dim:
            raise ValueError(f"Predictor expects {self.input_dim}-d embeddings, index has {index.dim}-d")
        sampler = ComparisonSampler(
            index, supervision, cfg.negatives_per_positive, hard_fraction, exclude_concepts
        )
        if not len(sampler):
            raise EmptySupervisionError(
                "No comparison examples: the temporal supervision is empty (check the cutoffs, the horizon "
                "and the paper dates) or every positive involves an excluded concept"
            )
        tables = self._feature_tables(index, sampler.concepts())
        flat = {f"{facet}.{k}": self.encoders[facet].p[k] for facet in FACETS for k in _PARAM_NAMES}
        opt = _AdamW(flat, cfg.learning_rate, cfg.weight_decay)
        self.loss_curve = []
        n_epochs = int(cfg.epochs if epochs is None else epochs)
        for _ in range(n_epochs):
            ex = sampler.sample(rng)
            n = ex["direction"].shape[0]
            drop = rng.random(n) < float(cfg.context_dropout)
            order = rng.permutation(n)
            total = 0.0
            for s in range(0, n, max(1, int(batch_size))):
                idx = order[s : s + batch_size]
                loss, grads = self._batch_loss_and_grads(
                    tables, ex["direction"][idx], ex["anchor"][idx], ex["positive"][idx],
                    ex["negative"][idx], drop[idx],
                )
                opt.step(grads)
                total += loss * len(idx)
            self.loss_curve.append(total / max(1, n))
        self._z.clear()
        self._ball.clear()
        self.prepared_for = None
        return self

    def _batch_loss_and_grads(
        self,
        tables: Mapping[str, Tuple[np.ndarray, np.ndarray]],
        direction: np.ndarray,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        drop: np.ndarray,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Mean of log(1 + exp[s(a, c-) - s(a, c+)]) over the batch and its gradients."""
        B = direction.shape[0]
        E = int(self.config.embed_dim)
        za = np.zeros((B, E), dtype=np.float32)
        zp = np.zeros((B, E), dtype=np.float32)
        zn = np.zeros((B, E), dtype=np.float32)
        # Codes row * 2 + has_context; anchors may drop their taxonomy context.
        a_code = anchor * 2 + (~drop).astype(np.int64)
        c_code_p = positive * 2 + 1
        c_code_n = negative * 2 + 1
        state = {}
        for fi, facet in enumerate(FACETS):
            mask_a = direction == fi  # problem anchors when ranking methods (direction 0)
            mask_c = direction != fi  # candidates of this facet
            na, nc = int(mask_a.sum()), int(mask_c.sum())
            if na + nc == 0:
                continue
            codes = np.concatenate([a_code[mask_a], c_code_p[mask_c], c_code_n[mask_c]])
            uniq, inv = np.unique(codes, return_inverse=True)
            ctx_tab, desc_tab = tables[facet]
            rows = uniq // 2
            X = np.where((uniq % 2 == 1)[:, None], ctx_tab[rows], desc_tab[rows]).astype(np.float32)
            Z, cache = self.encoders[facet].forward(X)
            za[mask_a] = Z[inv[:na]]
            zp[mask_c] = Z[inv[na : na + nc]]
            zn[mask_c] = Z[inv[na + nc :]]
            state[facet] = (mask_a, mask_c, na, nc, inv, len(uniq), cache)
        delta = (za * zn).sum(axis=1) - (za * zp).sum(axis=1)
        loss = float(np.logaddexp(0.0, delta).mean())
        g = (_sigmoid(delta) / max(1, B)).astype(np.float32)[:, None]
        dza = g * (zn - zp)
        dzp = -g * za
        dzn = g * za
        grads: Dict[str, np.ndarray] = {}
        for facet, (mask_a, mask_c, na, nc, inv, n_uniq, cache) in state.items():
            dZ = np.zeros((n_uniq, E), dtype=np.float32)
            np.add.at(dZ, inv[:na], dza[mask_a])
            np.add.at(dZ, inv[na : na + nc], dzp[mask_c])
            np.add.at(dZ, inv[na + nc :], dzn[mask_c])
            for k, v in self.encoders[facet].backward(cache, dZ).items():
                grads[f"{facet}.{k}"] = v
        return loss, grads

    # --------------------------------------------------------------- io
    def save(self, path: Union[str, Path]) -> None:
        """Write ``link_predictor.json`` (kind "temporal") and ``link_predictor.npz`` into directory ``path``."""
        if not self.fitted:
            raise RuntimeError("Cannot save an untrained TemporalLinkPredictor")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": "temporal",
            "version": 1,
            "input_dim": self.input_dim,
            "context": "mean_ancestor_embedding",
            "config": dataclasses.asdict(self.config),
            "loss_curve": self.loss_curve,
            "report": self.report,
        }
        (path / PREDICTOR_JSON).write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        arrays = {f"{facet}.{k}": enc.p[k] for facet, enc in self.encoders.items() for k in _PARAM_NAMES}
        np.savez_compressed(path / PREDICTOR_NPZ, **arrays)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TemporalLinkPredictor":  # type: ignore[override]
        return load_predictor(path)


def load_predictor(path: Union[str, Path]) -> TemporalLinkPredictor:
    """Load a :class:`TemporalLinkPredictor` saved with :meth:`TemporalLinkPredictor.save`.

    Called by :meth:`LinkPredictor.load` for every non-"embedding" kind. The
    caches are not stored: call ``prepare(index)`` before scoring.
    """
    path = Path(path)
    meta = json.loads((path / PREDICTOR_JSON).read_text(encoding="utf-8"))
    kind = meta.get("kind")
    if kind != "temporal":
        raise ValueError(f"Unsupported link predictor kind {kind!r} in {path}")
    cfg_dict = dict(meta.get("config") or {})
    if "cutoffs" in cfg_dict:
        cfg_dict["cutoffs"] = tuple(cfg_dict["cutoffs"])
    known = {f.name for f in dataclasses.fields(LinkPredictionConfig)}
    config = LinkPredictionConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    model = TemporalLinkPredictor(config)
    with np.load(path / PREDICTOR_NPZ) as data:
        arrays = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
    model.encoders = {
        facet: _Encoder({k: arrays[f"{facet}.{k}"] for k in _PARAM_NAMES}) for facet in FACETS
    }
    model.input_dim = int(meta["input_dim"])
    model.loss_curve = [float(x) for x in meta.get("loss_curve", [])]
    model.report = dict(meta.get("report") or {})
    return model


# ---------------------------------------------------------------------------
# Ranking evaluation
# ---------------------------------------------------------------------------


def ranking_candidates(
    snapshot: TemporalSnapshot,
    anchors: Iterable[str],
    direction: str = METHOD,
    exclude_concepts: Iterable[str] = (),
) -> Dict[str, List[str]]:
    """Historical candidates of each anchor at a cutoff (the comparison pool of App. B.5).

    ``direction="method"``: anchors are problems and the candidates are
    method concepts m of the historical index with J_t(u, m) = {} that do not
    co-occur with u in a window paper (window co-occurrences are either the
    positives under evaluation or unverified, so they are neither ranked as
    negatives nor as competitors). ``direction="problem"`` exchanges roles.
    """
    exclude = set(exclude_concepts)
    out: Dict[str, List[str]] = {}
    for a in anchors:
        if a in out:
            continue
        blocked = snapshot.co_concepts(a) | snapshot.window_co_concepts(a) | exclude
        out[a] = [c for c in snapshot.concepts(direction) if c not in blocked]
    return out


def evaluate_ranking(
    predictor: LinkPredictor,
    pairs: Sequence[Pair],
    candidates: Union[Sequence[str], Mapping[str, Sequence[str]]],
    ks: Sequence[int] = (1, 3, 10),
    direction: str = METHOD,
) -> Dict[str, float]:
    """Filtered ranking metrics of held-out positives: MRR, Hit@k and AUC.

    ``pairs`` are (problem, method) positives. For ``direction="method"``
    each problem u ranks its positive m+ against ``candidates[u]`` (or the
    shared candidate list); for ``direction="problem"`` each method ranks its
    positive problem. Other positives of the same anchor are removed from
    the candidates (filtered setting), and ties count half (expected rank
    under random tie-breaking). A pair left without any competing candidate
    (anchor missing from ``candidates`` or every candidate filtered) carries
    no ranking information: it is excluded from the means and counted in
    ``n_no_candidates``; ``n`` is the number of ranked pairs. ``predictor``
    must be prepared on the index.
    """
    if direction not in (PROBLEM, METHOD):
        raise ValueError("direction must be 'method' or 'problem'")
    positives_of: Dict[str, Set[str]] = defaultdict(set)
    for u, m in pairs:
        if direction == METHOD:
            positives_of[u].add(m)
        else:
            positives_of[m].add(u)
    rr, aucs = [], []
    hits: Dict[int, List[float]] = {int(k): [] for k in ks}
    no_candidates = 0
    for u, m in pairs:
        anchor, pos = (u, m) if direction == METHOD else (m, u)
        cands = candidates.get(anchor, []) if isinstance(candidates, Mapping) else candidates
        negs = [c for c in cands if c != pos and c not in positives_of[anchor]]
        if not negs:
            no_candidates += 1
            continue
        score = (lambda c: predictor.score(anchor, c)) if direction == METHOD else (lambda c: predictor.score(c, anchor))
        s_pos = score(pos)
        s_neg = np.array([score(c) for c in negs], dtype=np.float64)
        greater = float((s_neg > s_pos).sum())
        ties = float((s_neg == s_pos).sum())
        rank = 1.0 + greater + 0.5 * ties
        rr.append(1.0 / rank)
        for k in hits:
            hits[k].append(1.0 if rank <= k else 0.0)
        aucs.append((len(negs) - greater - 0.5 * ties) / len(negs))
    out: Dict[str, float] = {"n": float(len(rr)), "n_no_candidates": float(no_candidates)}
    out["mrr"] = float(np.mean(rr)) if rr else float("nan")
    for k, v in hits.items():
        out[f"hit@{k}"] = float(np.mean(v)) if v else float("nan")
    out["auc"] = float(np.mean(aucs)) if aucs else float("nan")
    return out


def _evaluate_both(
    predictor: LinkPredictor, val: SupervisionSet, pairs: Sequence[Pair], exclude: Set[str], ks: Sequence[int]
) -> Dict[str, Dict[str, float]]:
    snap = val.snapshot
    return {
        "method_ranking": evaluate_ranking(
            predictor, pairs, ranking_candidates(snap, [u for u, _ in pairs], METHOD, exclude), ks, METHOD
        ),
        "problem_ranking": evaluate_ranking(
            predictor, pairs, ranking_candidates(snap, [m for _, m in pairs], PROBLEM, exclude), ks, PROBLEM
        ),
    }


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------


class FallbackLinkPredictor(EmbeddingLinkPredictor):
    """The untrained description prior, returned by :func:`train_link_predictor` without supervision.

    Behaves exactly like :class:`~scholarpath.retrieval.link_prediction.EmbeddingLinkPredictor`
    (saved with kind ``"embedding"``) and carries the training ``report``,
    whose ``"fallback"`` entry says why no temporal model was trained.
    """

    def __init__(self, report: Optional[Dict[str, Any]] = None, scale: float = 6.0, bias: float = -3.0):
        super().__init__(scale=scale, bias=bias)
        self.report: Dict[str, Any] = dict(report or {})

    def save(self, path: Union[str, Path]) -> None:
        super().save(path)
        meta_path = Path(path) / PREDICTOR_JSON
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["report"] = self.report
        meta_path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")


VALIDATION_RULE = "train on cutoffs whose window (t, t + Delta] ends on or before the validation cutoff"


def train_link_predictor(
    index: SemanticIndex,
    config: Optional[LinkPredictionConfig] = None,
    llm: Optional[LLMClient] = None,
    dates: Optional[Mapping[str, Any]] = None,
    exclude_concepts: Iterable[str] = (),
    *,
    validate: bool = True,
    refit: bool = True,
    strict: bool = False,
    batch_size: int = 64,
    hard_fraction: float = 0.5,
    max_papers_per_pair: Optional[int] = 3,
    ks: Sequence[int] = (1, 3, 10),
    supervision: Optional[Sequence[SupervisionSet]] = None,
) -> LinkPredictor:
    """Train the cross-taxonomy association model of Sec. 4.3 / App. B.5.

    1. Supervision D+_{t, Delta} for every cutoff t in ``config.cutoffs`` with
       Delta = ``config.horizon_years`` (Eq. 17), verified by ``llm`` when
       ``config.verify_with_llm`` (verdicts are shared across cutoffs).
       App. B.5 requires every cutoff and horizon to lie within the indexed
       corpus -- in the held-out evaluation, the reduced corpus of App. C.1 --
       so that no held-out target can supply a label; windows extending past
       the corpus are reported in ``report["windows_exceeding_corpus"]`` and
       logged (``strict=True`` raises instead).
    2. Validation (``validate``): the last cutoff t_v is held out, and a model
       is trained only on the cutoffs whose whole window ends by t_v
       (``t + Delta <= t_v``), so that no training label or comparison uses a
       paper published after t_v (overlapping windows would otherwise let one
       window paper supply both a training label and a validation label). It
       ranks the positives of t_v that are not training positives (filtered
       MRR, Hit@k and AUC for both ranking directions, alongside the untrained
       description-cosine prior). Skipped, with the reason, when no training
       window qualifies or there is nothing to rank.
    3. The returned model is trained on every cutoff (``refit``; otherwise the
       validation model is returned) and prepared on ``index``.

    ``exclude_concepts`` (held-out endpoints) never appear in a training
    example, as positive, anchor or comparison. Statistics are stored in
    ``predictor.report``. When the supervision yields no comparison example
    at all, the description prior is returned as a :class:`FallbackLinkPredictor`
    (``report["fallback"] = "empty_supervision"``, with a warning), or
    :class:`EmptySupervisionError` is raised when ``strict``.
    """
    config = config or LinkPredictionConfig()
    exclude = set(exclude_concepts)
    cutoffs = sorted({normalize_date(c) for c in config.cutoffs if normalize_date(c)})
    if supervision is None:
        if not cutoffs:
            raise ValueError("LinkPredictionConfig.cutoffs is empty")
        verifier = llm if config.verify_with_llm else None
        cache: Dict[Tuple[str, str, str], bool] = {}
        supervision = [
            build_supervision(
                index, c, config.horizon_years, verifier, dates,
                max_papers_per_pair=max_papers_per_pair, cache=cache,
            )
            for c in cutoffs
        ]
    supervision = sorted(supervision, key=lambda s: s.cutoff)
    outside = [s.cutoff for s in supervision if s.snapshot.exceeds_corpus]
    if outside and strict:
        raise ValueError(
            f"Temporal windows at cutoffs {outside} are not within the indexed corpus "
            f"(ends {supervision[0].snapshot.corpus_end}); App. B.5 requires every cutoff and horizon to lie "
            "within it"
        )
    report: Dict[str, Any] = {
        "kind": "temporal",
        "cutoffs": [s.cutoff for s in supervision],
        "horizon_years": float(config.horizon_years),
        "supervision": [s.summary() for s in supervision],
        "verified_with_llm": bool(supervision) and all(s.verified for s in supervision),
        "excluded_concepts": len(exclude),
        "windows_exceeding_corpus": outside,
    }
    fit_kw = dict(batch_size=batch_size, hard_fraction=hard_fraction)

    validation_model: Optional[TemporalLinkPredictor] = None
    if validate and len(supervision) >= 2:
        val = supervision[-1]
        train_sets = [s for s in supervision[:-1] if s.end <= val.cutoff]
        dropped = [s.cutoff for s in supervision[:-1] if s.end > val.cutoff]
        seen = {p for s in train_sets for p in s.positives}
        val_pairs = [p for p in val.positives if p not in seen and p[0] not in exclude and p[1] not in exclude]
        info: Dict[str, Any] = {
            "cutoff": val.cutoff,
            "rule": VALIDATION_RULE,
            "train_cutoffs": [s.cutoff for s in train_sets],
            "dropped_train_cutoffs": dropped,
            "pairs": len(val_pairs),
        }
        skipped = ""
        if not train_sets:
            skipped = "no training window ends by the validation cutoff"
        elif not val_pairs:
            skipped = "no held-out positives"
        else:
            try:
                validation_model = TemporalLinkPredictor(config).fit(index, train_sets, exclude, **fit_kw)
            except EmptySupervisionError:
                skipped = "no training positives"
        if validation_model is not None:
            validation_model.prepare(index)
            prior = EmbeddingLinkPredictor().prepare(index)
            info.update({
                "loss_curve": list(validation_model.loss_curve),
                **_evaluate_both(validation_model, val, val_pairs, exclude, ks),
                "embedding_prior": _evaluate_both(prior, val, val_pairs, exclude, ks),
            })
            mr = info["method_ranking"]
            log.info("Link-prediction validation at %s (train %s): %d pairs, MRR %.3f, AUC %.3f",
                     val.cutoff, info["train_cutoffs"], len(val_pairs), mr["mrr"], mr["auc"])
        else:
            info["skipped"] = skipped
        report["validation"] = info

    if validation_model is not None and not refit:
        model: TemporalLinkPredictor = validation_model
        train_cutoffs = list(report["validation"]["train_cutoffs"])
    else:
        try:
            model = TemporalLinkPredictor(config).fit(index, supervision, exclude, **fit_kw)
        except EmptySupervisionError:
            if strict:
                raise
            log.warning(
                "No temporal supervision at cutoffs %s (horizon %.1f y): returning the untrained description "
                "prior (EmbeddingLinkPredictor)", report["cutoffs"], float(config.horizon_years),
            )
            report["fallback"] = "empty_supervision"
            fallback = FallbackLinkPredictor(report)
            fallback.prepare(index)
            return fallback
        train_cutoffs = [s.cutoff for s in supervision]
    sampler_pairs = {p for s in supervision for p in s.positives if p[0] not in exclude and p[1] not in exclude}
    report["final"] = {
        "train_cutoffs": train_cutoffs,
        "positive_pairs": len(sampler_pairs),
        "loss_curve": list(model.loss_curve),
    }
    model.report = report
    model.prepare(index)
    return model


__all__ = [
    "ComparisonSampler",
    "EmptySupervisionError",
    "FallbackLinkPredictor",
    "SupervisionSet",
    "TemporalLinkPredictor",
    "TemporalSnapshot",
    "add_years",
    "build_supervision",
    "evaluate_ranking",
    "load_predictor",
    "most_specific_pairs",
    "normalize_date",
    "paper_dates",
    "ranking_candidates",
    "related_concepts",
    "static_supervision",
    "temporal_supervision",
    "train_link_predictor",
    "verify_candidates",
    "window_candidates",
]
