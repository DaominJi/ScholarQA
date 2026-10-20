"""Hierarchy-bounded subtree pruning and exact lazy search (Section 4.3, Algorithm 2 lines 5-23).

Candidate units are grouped into hierarchical *search regions* H. Each region
caches an adequacy upper bound r_bar_a(H) >= max_{z in H} r_hat_{a,z} and a
marginal-cost lower bound c_low(H | S) <= min_{z in H} Delta C(z | S), so that
(Eq. 7, Lemma 6)::

    delta_bar(H | S) = sum_{a in A_q(H)} w_a max{0, r_bar_a(H) - f_a(S)} / c_low(H | S)  >=  delta(z | S).

Regions
-------
Direct paths (evidence aspects, including the background baselines of hypothesis queries):

* ``ConceptRegion(v)``: the paths from a scope s routed through concept v and
  ending at or below it. r_bar uses the junction bound omega_bar_q(v) (one
  posting-list intersection per junction prunes all papers and chunks below v)
  and the cached balls of the concepts, abstracts and chunks below v.
  Expanding it yields the concept path s -> ... -> v, the paper paths that
  stop at v, one ``ChunkRegion`` per such paper and one ``ConceptRegion`` per
  child sharing papers with the junction.
* ``ChunkRegion(p)``: the chunk extensions e+ of a paper path (Section 4.4).

Complementary path sets (hypothesis aspects), H = (V_M(v), H_P, H_M):

* ``CandidateSubtreeRegion(v)``: path sets whose candidate lies below v; its
  bound is sigmoid(U(u_q, v)) phi_bar(H) min{s_bar_P(H_P), s_bar_M(H_M)}
  (Eq. 8, 12; phi_bar(H) is the largest bridge discount of a candidate below
  v), so uncompetitive subtrees stay deferred without scoring their concepts;
* ``CandidateRegion(c)``: path sets of one candidate (scores s_theta(u_q, c));
* ``BridgeRegion(c, d)``: path sets of c whose bridge anchors are at
  taxonomic distance d (constant bridge discount phi);
* ``PairGrid`` / ``PairRow``: a lazy, exact best-first enumeration of the
  (known-side witness, candidate-side witness) pairs of a bridge region. Both
  witness lists are sorted by premise adequacy, so by Eq. (9) a range of
  them is bounded by eta * min{s_K[first], s_C[first]}; its cost bound is the
  smallest abstract(+chunk) cost in the range (a range-minimum query).
  ``PairGrid`` holds a range of known-side witnesses (with all candidate-side
  ones) and ``PairRow`` one known-side witness with a range of candidate-side
  witnesses. A range is split at its cheapest witness, so when one premise
  is the bottleneck (the minimum of Eq. 9 is the same for the whole range)
  the pairs are produced cheapest first instead of being enumerated.

A discovery query starts from ``CandidateSubtreeRegion(m_q)``; an assessment
query fixes the pair, starts from ``CandidateRegion(m_q)`` and uses
eta = phi (no link-prediction factor). An assessed concept absent from I is a
query-local anchor: its path sets reach it through the neighbours it is
attached to (Section 4.1, Step 2).

Every unit of a region contains the region's ``req`` nodes and one node of
each of its alternative groups, so c_low(H | S) = sum of the unselected req
costs + sum of the minimum costs of the groups none of whose nodes is
selected yet (Appendix A.5: "the smallest terminal-node token cost in H, or
zero once one is selected"). An inverted index maps nodes (and structural
tokens: "a paper below v", "a concept below v") to the active entries, and
selecting a unit lowers the cost bounds of the entries sharing its new nodes
eagerly (Algorithm 2, line 21). The witness ranges of a bridge region keep
*refreshable* groups: a selection under their tokens lowers the group to
zero eagerly (keys stay upper bounds), and the exact range minimum given S
is recomputed when the region reaches the top of Q, together with its
numerator (line 12).

Instances of one aspect group are covered by at most one instance per unit
(see :mod:`scholarpath.retrieval.objective`), so a group contributes the
maximum, not the sum, of its instance terms to the numerator of Eq. (7); this
is a tightening of Lemma 6 that keeps the bound admissible.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from scholarpath.config import RetrievalConfig
from scholarpath.index.semantic_index import SemanticIndex, chunk_key, concept_key, paper_key
from scholarpath.retrieval.compose import ContextRenderer
from scholarpath.retrieval.grounding import FACETS, Anchor, Grounding, other_facet
from scholarpath.retrieval.link_prediction import sigmoid
from scholarpath.retrieval.objective import (
    CONCEPT_GROUP,
    FINDING_GROUP,
    AspectSpace,
    Coverage,
    SelectionState,
    delta,
)
from scholarpath.retrieval.query_understanding import QueryAnalysis
from scholarpath.retrieval.scoring import (
    AdequacyScorer,
    JunctionTable,
    balls_below,
    bridge_discount,
    chunk_ball,
    eta_factor,
    paper_balls,
)
from scholarpath.retrieval.units import (
    ComplementaryPathSet,
    DirectPath,
    PathFactory,
    is_anchor,
    key_id,
    key_level,
    residual_papers,
)
from scholarpath.types import EVIDENCE, HYPOTHESIS_DISCOVERY

Token = Union[str, Tuple[str, str, str]]
DISCOVERY = "discovery"
ASSESSMENT = "assessment"


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AltGroup:
    """Every unit of a region contains at least one node matched by these tokens.

    ``min_cost`` lower-bounds the cost of that node while no node matched by
    the tokens is selected (the group then contributes zero). A group with a
    ``refresh`` function can do better: ``refresh(state)`` returns a lower
    bound valid under the current selection, and the search re-evaluates it
    whenever the region reaches the top of Q after a selection under its tokens.
    """

    tokens: Tuple[Token, ...]
    min_cost: int
    refresh: Optional[Callable[[SelectionState], int]] = field(default=None, compare=False)

    def value(self, state: SelectionState, hit: bool) -> int:
        """Lower bound on the group's marginal cost under ``state`` (``hit``: a token is selected)."""
        if self.refresh is not None:
            return self.refresh(state)
        return 0 if hit else self.min_cost


@dataclass
class UnitEntry:
    """A scored candidate unit: z, V(z), A_q(z) with fixed adequacies."""

    uid: str
    unit: object  # DirectPath | ComplementaryPathSet
    nodes: Tuple[str, ...]
    coverage: Coverage
    meta: Dict[str, Any] = field(default_factory=dict)


class Region:
    """A search region H (Section 4.3)."""

    rid: str = ""
    req: Tuple[str, ...] = ()
    groups: Tuple[AltGroup, ...] = ()

    def numerator(self, space: "SearchSpace", state: SelectionState) -> float:
        """Numerator of Eq. (7) (an upper bound on the gain of every unit in H)."""
        raise NotImplementedError

    def expand(self, space: "SearchSpace") -> Tuple[List[UnitEntry], List["Region"]]:
        """Expand H one level into its root units and sub-regions (Algorithm 2, line 13)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Direct-path regions
# ---------------------------------------------------------------------------


class ConceptRegion(Region):
    """Direct paths from ``scope`` routed through concept ``node`` (ending at or below it)."""

    def __init__(self, space: "SearchSpace", facet: str, scope: str, node: str, prefix: Tuple[str, ...]):
        self.facet, self.scope, self.node = facet, scope, node
        self.rid = f"R|{scope}|{concept_key(node)}"
        self.req = prefix
        self.groups = ()
        self.omega_bar = space.omega_extent(facet, node)
        self._rbar: Optional[np.ndarray] = None

    def rbar(self, space: "SearchSpace") -> np.ndarray:
        if self._rbar is None:
            self._rbar = space.scorer.direct_bound(self.omega_bar, balls_below(space.index, self.node))
        return self._rbar

    def numerator(self, space: "SearchSpace", state: SelectionState) -> float:
        return space.direct_numerator(state, self.rbar(space), self.facet, self.req, below=self.node)

    def expand(self, space: "SearchSpace") -> Tuple[List[UnitEntry], List[Region]]:
        units: List[UnitEntry] = []
        regions: List[Region] = []
        tax = space.index.taxonomy(self.facet)
        concept_unit = space.direct_unit(DirectPath(self.req, self.scope, self.facet))
        if concept_unit is not None:
            units.append(concept_unit)
        for p in residual_papers(tax, self.node, tax.nodes[self.node].papers):
            if space.omega_paper(p) <= 0.0 or space.paths.route(self.facet, self.scope, p)[-1] != self.node:
                continue
            space.postings_scanned.add(p)
            path = DirectPath(self.req + (paper_key(p),), self.scope, self.facet)
            unit = space.direct_unit(path)
            if unit is not None:
                units.append(unit)
            if space.paths.chunks(p):
                regions.append(ChunkRegion(space, self.facet, self.scope, p, path.nodes))
        for c in tax.children(self.node):
            if space.omega_extent(self.facet, c) > 0.0:
                regions.append(ConceptRegion(space, self.facet, self.scope, c, self.req + (concept_key(c),)))
            else:
                space.postings_pruned_by_junction += len(tax.nodes[c].papers)
        return units, regions


class ChunkRegion(Region):
    """Chunk extensions e+ = (... -> p -> s) of the paper path ``prefix`` (Section 4.4)."""

    def __init__(self, space: "SearchSpace", facet: str, scope: str, paper_id: str, prefix: Tuple[str, ...]):
        self.facet, self.scope, self.paper_id = facet, scope, paper_id
        self.rid = f"C|{scope}|{paper_key(paper_id)}"
        self.req = prefix
        keys = tuple(chunk_key(c) for c in space.paths.chunks(paper_id))
        self.chunk_keys = keys
        self.groups = (AltGroup(keys, min(space.cost(k) for k in keys)),)
        self._rbar: Optional[np.ndarray] = None

    def rbar(self, space: "SearchSpace") -> np.ndarray:
        if self._rbar is None:
            omega = space.omega_paper(self.paper_id)
            self._rbar = space.scorer.direct_bound(omega, chunk_ball(space.index, self.paper_id))
        return self._rbar

    def numerator(self, space: "SearchSpace", state: SelectionState) -> float:
        return space.direct_numerator(state, self.rbar(space), self.facet, self.req, paper=self.paper_id)

    def expand(self, space: "SearchSpace") -> Tuple[List[UnitEntry], List[Region]]:
        units = []
        for key in self.chunk_keys:
            unit = space.direct_unit(DirectPath(self.req + (key,), self.scope, self.facet))
            if unit is not None:
                units.append(unit)
        return units, []


# ---------------------------------------------------------------------------
# Hypothesis regions (cross-taxonomy pruning, Eq. 8, 9, 12)
# ---------------------------------------------------------------------------


@dataclass
class HypothesisSpec:
    """Endpoints and candidate scope of a hypothesis query."""

    mode: str  # discovery | assessment
    known_facet: str
    cand_facet: str
    known_id: str  # concept id, or anchor id for a query-local anchor
    known_label: str
    roots: List[str]  # concepts the known-side paths start from
    known_extent: Set[str]
    cand_scope: str
    fixed_candidate: Optional[str]
    use_link: bool
    candidates: Set[str] = field(default_factory=set)
    has_candidate: Dict[str, bool] = field(default_factory=dict)
    baselines: Dict[str, Tuple[str, ...]] = field(default_factory=dict)  # known paper -> m' options
    d_min: Dict[str, int] = field(default_factory=dict)  # candidate-facet node -> min distance to a m'
    phi_bar: Dict[str, float] = field(default_factory=dict)  # subtree root v -> max_{c <= v} phi(d_min[c])
    cand_anchor: Optional[Anchor] = None  # assessed candidate absent from I (query-local anchor)
    studied: Set[str] = field(default_factory=set)  # J(u_q, m_q) of an assessed pair the literature studies

    @property
    def known_is_concept(self) -> bool:
        return len(self.roots) == 1 and self.roots[0] == self.known_id

    def roots_of(self, candidate: str) -> List[str]:
        """Concepts a candidate's paths reach it through: c itself, or an anchor's neighbours."""
        if self.cand_anchor is not None and candidate == self.cand_anchor.anchor_id:
            return [n for n, _ in self.cand_anchor.neighbors]
        return [candidate]


@dataclass
class Witness:
    """A premise path (paper or chunk level) with its adequacy s_hat (Eq. 9)."""

    score: float
    path: DirectPath
    baseline: Optional[str] = None  # m' (known side)
    bridge: Optional[str] = None  # f = LCA(m', c) (known side)
    attach: Optional[str] = None  # concept e_C ends at: c, or the neighbour of an anchored c (known side)


class RangeArgMin:
    """Sparse-table range-minimum queries over a fixed list (ties: smallest index)."""

    def __init__(self, values: Sequence[int]):
        self.values = list(values)
        n = len(self.values)
        self._table: List[List[int]] = [list(range(n))]
        k = 1
        while (1 << k) <= n:
            prev, half = self._table[-1], 1 << (k - 1)
            self._table.append([self._better(prev[i], prev[i + half]) for i in range(n - (1 << k) + 1)])
            k += 1

    def _better(self, a: int, b: int) -> int:
        return a if (self.values[a], a) <= (self.values[b], b) else b

    def argmin(self, lo: int, hi: int) -> int:
        """Index of the smallest value in ``values[lo:hi]`` (``lo < hi``)."""
        k = (hi - lo).bit_length() - 1
        row = self._table[k]
        return self._better(row[lo], row[hi - (1 << k)])

    def min(self, lo: int, hi: int) -> int:
        return self.values[self.argmin(lo, hi)]


class WitnessCosts:
    """Terminal (paper + chunk) costs of a sorted witness list with range-minimum queries under S."""

    def __init__(self, space: "SearchSpace", witnesses: Sequence[Witness]):
        self.cost = space.cost
        self.terms = [tuple(k for k in w.path.nodes if key_level(k) in ("paper", "chunk")) for w in witnesses]
        self.rmq = RangeArgMin([sum(self.cost(k) for k in t) for t in self.terms])
        self.by_paper: Dict[str, List[int]] = {}
        for j, w in enumerate(witnesses):
            self.by_paper.setdefault(paper_key(w.path.paper_id), []).append(j)

    def min(self, lo: int, hi: int, state: Optional[SelectionState] = None) -> int:
        """min_{j in [lo, hi)} of the unselected terminal cost of witness j (exact under ``state``)."""
        if state is None or not state.nodes:
            return self.rmq.min(lo, hi)
        touched = sorted(j for k in state.node_order if k in self.by_paper for j in self.by_paper[k] if lo <= j < hi)
        best: Optional[int] = None
        prev = lo
        for j in touched + [hi]:
            if prev < j:
                m = self.rmq.min(prev, j)
                best = m if best is None else min(best, m)
            if j < hi:
                m = sum(self.cost(k) for k in self.terms[j] if k not in state.nodes)
                best = m if best is None else min(best, m)
            prev = j + 1
        return best if best is not None else 0


@dataclass
class BridgeLists:
    """Witness lists of a bridge region sorted by premise adequacy, shared by its pair regions."""

    candidate: str
    distance: int
    eta: float
    phi: float
    known: List[Witness]
    cand: List[Witness]
    known_costs: WitnessCosts
    cand_costs: WitnessCosts


class CandidateSubtreeRegion(Region):
    """H = (V(v), H_K, H_C): path sets whose candidate lies in the subtree of v."""

    def __init__(self, space: "SearchSpace", node: str):
        hyp = space.hyp
        assert hyp is not None
        self.node = node
        self.rid = f"HS|{node}"
        self.req = space.hyp_req()
        self.groups = space.hyp_groups(node, include_concepts=True)
        self._bound: Optional[float] = None

    def rbar(self, space: "SearchSpace") -> float:
        """r_bar_{h,H} = sigmoid(U(u_q, v)) phi_bar(H) min{s_bar_P(H_P), s_bar_M(H_M)} (Eq. 8, 12; Lemma 6)."""
        if self._bound is None:
            eta = space.link_bound(self.node) * space.hyp.phi_bar.get(self.node, 0.0)
            premises = min(space.known_premise_bound(), space.cand_premise_bound(self.node)) if eta > 0 else 0.0
            self._bound = eta * premises
        return self._bound

    def numerator(self, space: "SearchSpace", state: SelectionState) -> float:
        g = space.aspects.hypothesis_group
        return 0.0 if g is None else g.weight * max(0.0, self.rbar(space))

    def expand(self, space: "SearchSpace") -> Tuple[List[UnitEntry], List[Region]]:
        hyp = space.hyp
        regions: List[Region] = []
        if self.node in hyp.candidates:
            regions.append(CandidateRegion(space, self.node))
        for c in space.index.taxonomy(hyp.cand_facet).children(self.node):
            if hyp.has_candidate.get(c):
                regions.append(CandidateSubtreeRegion(space, c))
        return [], regions


class CandidateRegion(Region):
    """Path sets of one candidate c (exact link-prediction factor, phi_bar from the closest bridge)."""

    def __init__(self, space: "SearchSpace", candidate: str):
        self.candidate = candidate
        self.rid = f"HC|{candidate}"
        self.req = space.cand_head(candidate)
        self.groups = space.hyp_groups(candidate, include_concepts=False)
        space.scored_candidates.add(candidate)  # its exact eta is computed here ("Scored", Table 7)
        self.prob = space.link_prob(candidate)
        self.phi_bar = bridge_discount(space.hyp.d_min.get(candidate, 10 ** 6), space.config)
        self._bound: Optional[float] = None

    def rbar(self, space: "SearchSpace") -> float:
        if self._bound is None:
            eta = eta_factor(self.prob, self.phi_bar)
            self._bound = eta * min(space.known_premise_bound(), space.cand_premise_bound(self.candidate))
        return self._bound

    def numerator(self, space: "SearchSpace", state: SelectionState) -> float:
        return space.hyp_numerator(state, self.candidate, self.rbar(space))

    def expand(self, space: "SearchSpace") -> Tuple[List[UnitEntry], List[Region]]:
        regions: List[Region] = []
        for d, items in sorted(space.bridge_groups(self.candidate).items()):
            regions.append(BridgeRegion(space, self, d, items))
        return [], regions


class BridgeRegion(Region):
    """Path sets of candidate c whose bridge anchors are at taxonomic distance d."""

    def __init__(self, space: "SearchSpace", parent: CandidateRegion, distance: int,
                 items: List[Tuple[str, str, str, str]]):
        self.candidate = parent.candidate
        self.distance = distance
        self.items = items  # (known paper, m', f, attachment concept)
        self.rid = f"HB|{self.candidate}|{distance}"
        self.req = parent.req
        self.groups = parent.groups
        self.phi = bridge_discount(distance, space.config)
        self.eta = eta_factor(parent.prob, self.phi)
        self._bound: Optional[float] = None

    def rbar(self, space: "SearchSpace") -> float:
        if self._bound is None:
            self._bound = self.eta * min(space.known_premise_bound(), space.cand_premise_bound(self.candidate))
        return self._bound

    def numerator(self, space: "SearchSpace", state: SelectionState) -> float:
        return space.hyp_numerator(state, self.candidate, self.rbar(space))

    def expand(self, space: "SearchSpace") -> Tuple[List[UnitEntry], List[Region]]:
        known: List[Witness] = []
        for pid, baseline, bridge, attach in self.items:
            for w in space.known_witnesses(pid):
                if w.score > 0.0:
                    known.append(Witness(w.score, w.path, baseline, bridge, attach))
        cand = [w for w in space.cand_witnesses(self.candidate) if w.score > 0.0]
        if not known or not cand or self.eta <= 0.0:
            return [], []
        known.sort(key=lambda w: (-w.score, w.path.end))
        lists = BridgeLists(self.candidate, self.distance, self.eta, self.phi, known, cand,
                            WitnessCosts(space, known), WitnessCosts(space, cand))
        return [], [PairGrid(space, self.req, lists, 0, len(known))]


class PairGrid(Region):
    """Pairs (i, j) with known-side witness i in [lo, hi) and any candidate-side witness j.

    Bound: eta * min{s_K[lo], s_C[0]} (both lists sorted by adequacy, Eq. 9);
    cost: the shared head plus the cheapest unselected known-side and
    candidate-side terminals of the ranges (refreshable groups). Expanding
    splits the range at its cheapest known-side witness m into ``PairRow(m)``
    and the two remaining ranges.
    """

    def __init__(self, space: "SearchSpace", req: Tuple[str, ...], lists: BridgeLists, lo: int, hi: int):
        self.lists, self.lo, self.hi = lists, lo, hi
        self.rid = f"HG|{lists.candidate}|{lists.distance}|{lo}|{hi}"
        self.req = req
        n_cand = len(lists.cand)
        self.groups = space.pair_groups(
            lists.candidate,
            AltGroup(space.known_paper_group().tokens, lists.known_costs.min(lo, hi),
                     lambda state: lists.known_costs.min(lo, hi, state)),
            AltGroup(space.cand_paper_group(lists.candidate).tokens, lists.cand_costs.min(0, n_cand),
                     lambda state: lists.cand_costs.min(0, n_cand, state)),
        )
        self.bound = lists.eta * min(lists.known[lo].score, lists.cand[0].score)

    def numerator(self, space: "SearchSpace", state: SelectionState) -> float:
        return space.hyp_numerator(state, self.lists.candidate, self.bound)

    def expand(self, space: "SearchSpace") -> Tuple[List[UnitEntry], List[Region]]:
        L = self.lists
        m = L.known_costs.rmq.argmin(self.lo, self.hi)
        regions: List[Region] = [PairRow(space, L, m, 0, len(L.cand))]
        if self.lo < m:
            regions.append(PairGrid(space, self.req, L, self.lo, m))
        if m + 1 < self.hi:
            regions.append(PairGrid(space, self.req, L, m + 1, self.hi))
        return [], regions


class PairRow(Region):
    """Pairs (i, j) with the known-side witness i fixed and candidate-side witness j in [lo, hi).

    Every node of e_C and of the known-side premise path is shared (``req``);
    the candidate-side terminal costs at least the unselected range minimum
    (refreshable group). Expanding emits the pair of the cheapest witness m
    and splits the rest of the range.
    """

    def __init__(self, space: "SearchSpace", lists: BridgeLists, i: int, lo: int, hi: int):
        self.lists, self.i, self.lo, self.hi = lists, i, lo, hi
        self.rid = f"HR|{lists.candidate}|{lists.distance}|{i}|{lo}|{hi}"
        kw = lists.known[i]
        e_c = space.paths.concept_chain(space.hyp.cand_facet, kw.bridge, kw.attach)
        seen: Dict[str, None] = {}
        for k in space.cand_head(lists.candidate) + e_c + kw.path.nodes:
            seen.setdefault(k, None)
        self.req = tuple(seen)
        tokens = space.cand_paper_group(lists.candidate).tokens
        costs = lists.cand_costs
        self.groups = (AltGroup(tokens, costs.min(lo, hi), lambda state: costs.min(lo, hi, state)),)
        self.bound = lists.eta * min(kw.score, lists.cand[lo].score)

    def numerator(self, space: "SearchSpace", state: SelectionState) -> float:
        return space.hyp_numerator(state, self.lists.candidate, self.bound)

    def expand(self, space: "SearchSpace") -> Tuple[List[UnitEntry], List[Region]]:
        L = self.lists
        m = L.cand_costs.rmq.argmin(self.lo, self.hi)
        regions: List[Region] = []
        if self.lo < m:
            regions.append(PairRow(space, L, self.i, self.lo, m))
        if m + 1 < self.hi:
            regions.append(PairRow(space, L, self.i, m + 1, self.hi))
        return [space.pair_unit(L, self.i, m)], regions


# ---------------------------------------------------------------------------
# Search space of one query
# ---------------------------------------------------------------------------


class SearchSpace:
    """Everything Algorithm 2 needs for one query: scopes, aspect instances, scorer, regions."""

    def __init__(
        self,
        index: SemanticIndex,
        analysis: QueryAnalysis,
        grounding: Grounding,
        config: RetrievalConfig,
        scorer: AdequacyScorer,
        aspects: AspectSpace,
        renderer: ContextRenderer,
        paths: PathFactory,
        link_predictor: Optional[object],
        hyp: Optional[HypothesisSpec],
        direct_roots: List[Tuple[str, str]],
    ):
        self.index = index
        self.analysis = analysis
        self.grounding = grounding
        self.config = config
        self.scorer = scorer
        self.aspects = aspects
        self.renderer = renderer
        self.paths = paths
        self.link_predictor = link_predictor
        self.hyp = hyp
        self.direct_roots = direct_roots
        self.postings_scanned: Set[str] = set()
        self.postings_pruned_by_junction = 0
        self.scored_candidates: Set[str] = set()  # candidates whose exact eta was computed
        self.link_evaluations = 0  # calls of s_theta(u_q, m)
        self._known_w: Dict[str, List[Witness]] = {}
        self._cand_w: Dict[str, List[Witness]] = {}
        self._cand_bound: Dict[str, float] = {}
        self._known_bound: Optional[float] = None
        self._known_group: Optional[AltGroup] = None
        self._target_anc: Dict[str, Set[str]] = {}
        self._antichain: Dict[str, bool] = {}
        for g in aspects.groups.values():
            if g.kind != CONCEPT_GROUP:
                continue
            tax = index.taxonomy(g.facet)
            for t in g.targets:
                self._target_anc.setdefault(t, set(tax.ancestors(key_id(t))))
            self._antichain[g.gid] = not any(
                key_id(s) in self._target_anc[t] for t in g.targets for s in g.targets if s != t
            )

    # ------------------------------------------------------------- basics
    @property
    def kind(self) -> str:
        return self.analysis.kind

    def cost(self, key: str) -> int:
        """c(x): token cost of a node as rendered in the context (Section 4.2)."""
        return self.renderer.cost(key)

    def tokens_of(self, key: str) -> List[Token]:
        """Inverted-index tokens touched when ``key`` joins V(S)."""
        out: List[Token] = [key]
        level = key_level(key)
        if level == "concept":
            nid = key_id(key)
            facet = self.index.facet_of(nid)
            out.extend(("sub", facet, a) for a in self.index.taxonomy(facet).ancestors(nid, include_self=True))
        elif level == "paper":
            pid = key_id(key)
            for facet in ("problem", "method"):
                out.extend(("ext", facet, v) for v in self.index.paper_concepts(pid, facet))
        return out

    def omega_extent(self, facet: str, node: str) -> float:
        j = self.scorer.junction
        return 1.0 if j is None else j.omega_extent(facet, node)

    def omega_paper(self, paper_id: str) -> float:
        j = self.scorer.junction
        return 1.0 if j is None else j.omega_paper(paper_id)

    def min_paper_cost(self, facet: str, node: str) -> int:
        """Smallest abstract cost below v (query-independent, cached per index; Section 4.3)."""
        return self.renderer.costs.min_paper(facet, node)

    def min_concept_cost(self, facet: str, node: str) -> int:
        """Smallest concept-line cost in the subtree of v (query-independent, cached per index)."""
        return self.renderer.costs.min_concept(facet, node)

    def top_regions(self) -> List[Region]:
        """Top-level regions under (u_q, m_q) (Algorithm 2, line 7)."""
        regions: List[Region] = []
        has_evidence = any(g.kind in (CONCEPT_GROUP, FINDING_GROUP) for g in self.aspects.groups.values())
        if has_evidence:
            for facet, scope in self.direct_roots:
                if self.omega_extent(facet, scope) > 0.0:
                    regions.append(ConceptRegion(self, facet, scope, scope, (concept_key(scope),)))
        hyp = self.hyp
        if hyp is not None and self.aspects.hypothesis_group is not None:
            if hyp.mode == DISCOVERY:
                if hyp.has_candidate.get(hyp.cand_scope):
                    regions.append(CandidateSubtreeRegion(self, hyp.cand_scope))
            elif hyp.fixed_candidate in hyp.candidates:
                regions.append(CandidateRegion(self, hyp.fixed_candidate))
        return regions

    # ------------------------------------------------------- direct paths
    def direct_unit(self, path: DirectPath) -> Optional[UnitEntry]:
        """Score a direct path (fixed adequacies); None if it supports no aspect."""
        rhat = self.scorer.direct_adequacy(path)
        cov = self.aspects.direct_coverage(path.nodes, path.paper_id, rhat)
        if not cov:
            return None
        return UnitEntry(path.uid, path, path.nodes, cov, {"level": path.level, "omega": self.scorer.omega(path)})

    def direct_numerator(
        self,
        state: SelectionState,
        rbar: np.ndarray,
        facet: str,
        prefix: Sequence[str],
        below: Optional[str] = None,
        paper: Optional[str] = None,
    ) -> float:
        """Numerator of Eq. (7) for a direct-path region (one term per aspect group)."""
        total = 0.0
        pset = set(prefix)
        for g in self.aspects.groups.values():
            if g.kind == CONCEPT_GROUP:
                if g.facet != facet:
                    continue
                r = float(rbar[g.base])
                if r <= 0.0:
                    continue
                hits = [t for t in g.targets if t in pset]
                reach = (
                    [t for t in g.targets if t not in pset and below in self._target_anc[t]]
                    if below is not None
                    else []
                )
                if not hits and not reach:
                    continue
                if self._antichain[g.gid]:
                    targets = hits[:1] if hits else reach
                    f_min = min(state.support(g.iid(t)) for t in targets)
                    total += g.weight * max(0.0, r - f_min)
                else:
                    total += g.weight * sum(max(0.0, r - state.support(g.iid(t))) for t in hits + reach)
            elif g.kind == FINDING_GROUP:
                r = float(rbar[g.base])
                if r <= 0.0:
                    continue
                f_min = state.support(g.iid(paper)) if paper is not None else 0.0
                total += g.weight * max(0.0, r - f_min)
        return total

    # --------------------------------------------------------- hypotheses
    def hyp_req(self) -> Tuple[str, ...]:
        hyp = self.hyp
        return (concept_key(hyp.known_id),) if hyp is not None and hyp.known_is_concept else ()

    def cand_head(self, candidate: str) -> Tuple[str, ...]:
        """Nodes shared by every path set of a candidate: known concept, c (unless anchored), header line."""
        concept = () if is_anchor(candidate) else (concept_key(candidate),)
        hyp_key = self.renderer.register_hypothesis(candidate, self.hyp.known_label)
        return self.hyp_req() + concept + (hyp_key,)

    def cand_paper_group(self, node: str) -> AltGroup:
        """Every candidate-side premise ends at (a chunk of) a paper below one of the candidate's roots."""
        cf = self.hyp.cand_facet
        roots = self.hyp.roots_of(node)
        return AltGroup(tuple(("ext", cf, r) for r in roots), min(self.min_paper_cost(cf, r) for r in roots))

    def known_paper_group(self) -> AltGroup:
        if self._known_group is None:
            kf = self.hyp.known_facet
            roots = self.hyp.roots
            self._known_group = AltGroup(
                tuple(("ext", kf, r) for r in roots), min(self.min_paper_cost(kf, r) for r in roots)
            )
        return self._known_group

    def pair_groups(self, candidate: str, known: AltGroup, cand: AltGroup) -> Tuple[AltGroup, ...]:
        """Alternative groups of a pair range: its known-side and candidate-side terminals (+ anchors)."""
        hyp = self.hyp
        groups: List[AltGroup] = []
        if is_anchor(candidate):  # e_C ends at one of the anchored candidate's neighbours
            keys = tuple(concept_key(r) for r in hyp.roots_of(candidate))
            groups.append(AltGroup(keys, min(self.cost(k) for k in keys)))
        groups.append(cand)
        groups.append(known)
        if not hyp.known_is_concept:
            keys = tuple(concept_key(r) for r in hyp.roots)
            groups.append(AltGroup(keys, min(self.cost(k) for k in keys)))
        return tuple(groups)

    def hyp_groups(self, node: str, include_concepts: bool) -> Tuple[AltGroup, ...]:
        hyp = self.hyp
        groups: List[AltGroup] = []
        if include_concepts:  # e_C ends at a candidate below v
            groups.append(AltGroup((("sub", hyp.cand_facet, node),), self.min_concept_cost(hyp.cand_facet, node)))
        elif is_anchor(node):  # e_C ends at one of the neighbours the anchored candidate is attached to
            keys = tuple(concept_key(r) for r in hyp.roots_of(node))
            groups.append(AltGroup(keys, min(self.cost(k) for k in keys)))
        groups.append(self.cand_paper_group(node))
        groups.append(self.known_paper_group())
        if not hyp.known_is_concept:
            keys = tuple(concept_key(r) for r in hyp.roots)
            groups.append(AltGroup(keys, min(self.cost(k) for k in keys)))
        return tuple(groups)

    def link_bound(self, node: str) -> float:
        """sigmoid(U(u_q, v)) >= sigmoid(s_theta(u_q, m)) for every m below v (Eq. 12, Prop. 4)."""
        hyp = self.hyp
        if not hyp.use_link:
            return 1.0
        lp = self.link_predictor
        if hyp.cand_facet == "method":
            return sigmoid(lp.upper_bound(hyp.known_id, node))
        return sigmoid(lp.upper_bound_problems(hyp.known_id, node))

    def link_prob(self, candidate: str) -> Optional[float]:
        """sigmoid(s_theta(u_q, m)) of Eq. (8), or None when the link prior is not used."""
        hyp = self.hyp
        if not hyp.use_link:
            return None
        self.link_evaluations += 1
        lp = self.link_predictor
        if hyp.cand_facet == "method":
            return float(lp.prob(hyp.known_id, candidate))
        return float(lp.prob(candidate, hyp.known_id))

    def known_premise_bound(self) -> float:
        if self._known_bound is None:
            kf = self.hyp.known_facet
            self._known_bound = max(
                self.scorer.premise_bound(paper_balls(self.index, r), kf) for r in self.hyp.roots
            )
        return self._known_bound

    def cand_premise_bound(self, node: str) -> float:
        """s_bar of the candidate-side witness subtree(s) below v (or below an anchor's neighbours)."""
        hit = self._cand_bound.get(node)
        if hit is None:
            cf = self.hyp.cand_facet
            hit = max(self.scorer.premise_bound(paper_balls(self.index, r), cf) for r in self.hyp.roots_of(node))
            self._cand_bound[node] = hit
        return hit

    def hyp_numerator(self, state: SelectionState, candidate: str, rbar: float) -> float:
        g = self.aspects.hypothesis_group
        if g is None:
            return 0.0
        return g.weight * max(0.0, rbar - state.support(g.iid(candidate)))

    def bridge_groups(self, candidate: str) -> Dict[int, List[Tuple[str, str, str, str]]]:
        """Known papers grouped by the distance d(m', c) of their closest baseline concept m'.

        Items are ``(paper, m', f = LCA(m', a), a)`` where a is the concept e_C
        ends at: the candidate itself, or, for a query-local anchor (attached as
        a child of its neighbours), the neighbour closest to m', in which case
        d(m', anchor) = d(m', a) + 1.
        """
        hyp = self.hyp
        tax = self.index.taxonomy(hyp.cand_facet)
        anchored = is_anchor(candidate)
        roots = hyp.roots_of(candidate)
        rank = {r: i for i, r in enumerate(roots)}
        ext_c = set() if anchored else tax.nodes[candidate].papers
        dist: Dict[Tuple[str, str], int] = {}
        out: Dict[int, List[Tuple[str, str, str, str]]] = {}
        for pid in sorted(hyp.baselines):
            if pid in ext_c:
                continue  # already a direct junction of the pair
            best: Optional[Tuple[int, int, str, str]] = None
            for m in hyp.baselines[pid]:
                for a in roots:
                    d = dist.get((m, a))
                    if d is None:
                        d = tax.distance(m, a) + (1 if anchored else 0)
                        dist[(m, a)] = d
                    key = (d, rank[a], m, a)
                    if best is None or key < best:
                        best = key
            if best is None or best[0] < 1 or best[0] > self.config.max_bridge_distance:
                continue
            d, _, m, a = best
            out.setdefault(d, []).append((pid, m, tax.lowest_common_ancestor(m, a), a))
        return out

    def _witnesses(self, facet: str, start: str, pid: str) -> List[Witness]:
        path = self.paths.paper_path(facet, start, pid)
        out = [Witness(self.scorer.premise_score(path, facet), path)]
        for ext in self.paths.chunk_paths(path):
            out.append(Witness(self.scorer.premise_score(ext, facet), ext))
        self.postings_scanned.add(pid)
        return out

    def known_witnesses(self, pid: str) -> List[Witness]:
        """Known-side premise paths to paper ``pid`` and its chunks (cached per query)."""
        hit = self._known_w.get(pid)
        if hit is None:
            hyp = self.hyp
            tax = self.index.taxonomy(hyp.known_facet)
            root = next(r for r in hyp.roots if pid in tax.nodes[r].papers)
            hit = self._witnesses(hyp.known_facet, root, pid)
            self._known_w[pid] = hit
        return hit

    def cand_witnesses(self, candidate: str) -> List[Witness]:
        """Candidate-side premise paths below c (or below an anchor's neighbours), sorted by adequacy."""
        hit = self._cand_w.get(candidate)
        if hit is None:
            hyp = self.hyp
            tax = self.index.taxonomy(hyp.cand_facet)
            hit = []
            seen: Set[str] = set()
            for root in hyp.roots_of(candidate):
                for pid in sorted(tax.nodes[root].papers):
                    if pid not in hyp.known_extent and pid not in seen:
                        seen.add(pid)
                        hit.extend(self._witnesses(hyp.cand_facet, root, pid))
            hit.sort(key=lambda w: (-w.score, w.path.end))
            self._cand_w[candidate] = hit
        return hit

    def pair_unit(self, lists: BridgeLists, i: int, j: int) -> UnitEntry:
        """The complementary path set of grid cell (i, j), scored by Eq. (9)."""
        hyp = self.hyp
        kw, cw = lists.known[i], lists.cand[j]
        e_c = self.paths.concept_path(hyp.cand_facet, kw.bridge, kw.attach)
        if hyp.cand_facet == "method":
            e_p, e_m, s_p, s_m = kw.path, cw.path, kw.score, cw.score
        else:
            e_p, e_m, s_p, s_m = cw.path, kw.path, cw.score, kw.score
        b = ComplementaryPathSet(
            candidate=lists.candidate, candidate_facet=hyp.cand_facet, known=hyp.known_id,
            baseline=kw.baseline, bridge=kw.bridge, distance=lists.distance, phi=lists.phi,
            e_C=e_c, e_P=e_p, e_M=e_m, known_label=hyp.known_label,
        )
        self.renderer.register_set(b)
        rhat = lists.eta * min(s_p, s_m)
        cov = self.aspects.hypothesis_coverage(lists.candidate, rhat)
        meta = {"s_P": s_p, "s_M": s_m, "eta": lists.eta, "phi": lists.phi, "distance": lists.distance,
                "r_hat": rhat}
        return UnitEntry(b.uid, b, b.nodes, cov, meta)

    # -------------------------------------------------------------- stats
    def postings_total(self) -> int:
        total: Set[str] = set()
        for facet, scope in self.direct_roots:
            total |= self.index.taxonomy(facet).nodes[scope].papers
        if self.hyp is not None:
            total |= self.hyp.known_extent
            total |= self.index.taxonomy(self.hyp.cand_facet).nodes[self.hyp.cand_scope].papers
        return len(total)


# ---------------------------------------------------------------------------
# Building the search space (Steps 2-3)
# ---------------------------------------------------------------------------


def build_search_space(
    analysis: QueryAnalysis,
    grounding: Grounding,
    index: SemanticIndex,
    config: RetrievalConfig,
    link_predictor: Optional[object] = None,
    renderer: Optional[ContextRenderer] = None,
) -> SearchSpace:
    """Resolve what the aspects target given the scopes and set up the top-level regions (Step 2).

    * Evidence queries: aspects target the junction sub-concepts and the
      findings J(u_q, m_q) (:func:`_instantiate_evidence`).
    * Hypothesis discovery / assessment of an unstudied pair: premise aspects
      form one hypothesis instance per unexplored candidate, and background
      aspects target the already-studied baselines m' of the known endpoint,
      i.e. the papers of J(u_q, m') and the concepts m' (:func:`_instantiate_background`).
    * Assessment of a pair the literature already studies (J(u_q, m_q) != empty,
      Section 2.1): no hypothesis is formed; every aspect targets J(u_q, m_q).
    """
    renderer = renderer or ContextRenderer(index)
    paths = PathFactory(index, config.max_chunks_per_paper)
    aspects = AspectSpace()
    hyp: Optional[HypothesisSpec] = None
    if analysis.kind == EVIDENCE:
        junction, direct_roots, observed = _evidence_scopes(index, grounding)
        _add_evidence_notes(analysis, grounding, renderer, observed)
        _instantiate_evidence(analysis.evidence_aspects, index, junction, direct_roots, aspects)
    else:
        hyp = _hypothesis_spec(analysis, grounding, index, config, link_predictor)
        _add_anchor_notes(index, hyp, renderer)
        direct_roots = [(hyp.known_facet, r) for r in hyp.roots]
        if hyp.studied:
            junction = JunctionTable(index, [(hyp.studied, 1.0)])
            direct_roots.append((hyp.cand_facet, hyp.fixed_candidate))
            _instantiate_evidence(list(enumerate(analysis.aspects)), index, junction, direct_roots, aspects)
        else:
            # Background evidence: the established baselines J(u_q, m') of the known endpoint, reached from
            # the known scope or, through the baseline concepts m', from the root of the candidate taxonomy.
            junction = JunctionTable(index, [(set(hyp.baselines), 1.0)])
            if hyp.baselines:
                direct_roots.append((hyp.cand_facet, index.taxonomy(hyp.cand_facet).root_id))
            _instantiate_background(analysis, hyp, junction, aspects)
            premise_weight = sum(a.weight for a in analysis.aspects if a.is_premise)
            aspects.add_hypothesis_group(premise_weight, len(hyp.candidates))
    scorer = AdequacyScorer(index, analysis, config, junction)
    return SearchSpace(index, analysis, grounding, config, scorer, aspects, renderer, paths, link_predictor,
                       hyp, direct_roots)


def _evidence_scopes(
    index: SemanticIndex, grounding: Grounding
) -> Tuple[JunctionTable, List[Tuple[str, str]], bool]:
    """Junctions J(u, m) over U_q x M_q weighted by g_u g_m, the scopes paths descend from, and
    whether any junction is observed (``False``: the fallback to the scope extents was used)."""
    pairs = [(u, m, gu * gm) for u, gu in grounding.problem_scopes for m, gm in grounding.method_scopes]
    junction = JunctionTable.from_pairs(index, pairs)
    observed = bool(junction.findings)
    if not observed:
        # No observed junction: fall back to the extents of the matched scopes of each facet.
        root_p, root_m = index.problem.root_id, index.method.root_id
        pairs = [(u, root_m, g) for u, g in grounding.problem_scopes] + [
            (root_p, m, g) for m, g in grounding.method_scopes
        ]
        junction = JunctionTable.from_pairs(index, pairs)
    roots: List[Tuple[str, str]] = []
    for facet in FACETS:
        if not grounding.unspecified.get(facet, True):
            roots.extend((facet, s) for s, _ in grounding.scopes(facet))
    if not roots:
        roots = [("problem", index.problem.root_id)]
    return junction, roots, observed


def _add_evidence_notes(
    analysis: QueryAnalysis, grounding: Grounding, renderer: ContextRenderer, observed: bool
) -> None:
    """Flag explicitly when the established findings do not concern the asked pair itself.

    This happens when a query concept matches nothing in I (not even loosely
    through a query-local anchor), or when no paper of the index studies the
    matched scopes together (J(u_q, m_q) is empty). The note is budgeted
    (:meth:`ContextRenderer.header_overhead`).
    """
    texts = {
        "problem": ", ".join(analysis.problem_concepts) or "the problem",
        "method": ", ".join(analysis.method_concepts) or "the method",
    }
    unmatched = [f for f in FACETS if grounding.unmatched.get(f)]
    if len(unmatched) == 2:
        renderer.add_note(
            f'no concept in the index matches "{texts["problem"]}" or "{texts["method"]}"; the findings below are '
            "the closest established findings in the index, not findings about the asked pair."
        )
    elif unmatched:
        facet = unmatched[0]
        anchor = grounding.anchor(facet)
        text = anchor.text if anchor is not None else texts[facet]
        renderer.add_note(
            f'no {facet} concept in the index matches "{text}"; the findings below concern the matched '
            f'{other_facet(facet)} concepts, not "{text}".'
        )
    elif not observed:
        renderer.add_note(
            f'no indexed paper studies "{texts["problem"]}" with "{texts["method"]}"; the findings below concern '
            "each side separately."
        )


def _add_anchor_notes(index: SemanticIndex, hyp: HypothesisSpec, renderer: ContextRenderer) -> None:
    """Label query-local anchors and state in the context where they are attached (Section 4.1, Step 2)."""
    anchors = []
    if not hyp.known_is_concept:
        anchors.append((hyp.known_id, hyp.known_label, hyp.known_facet, hyp.roots))
    if hyp.cand_anchor is not None:
        a = hyp.cand_anchor
        anchors.append((a.anchor_id, a.text, a.facet, [n for n, _ in a.neighbors]))
    for anchor_id, text, facet, neighbors in anchors:
        renderer.register_label(anchor_id, text)
        names = ", ".join(index.concept(n).name for n in neighbors)
        renderer.add_note(f'"{text}" is not in the index; it is attached to its closest {facet} concepts: {names}.')


def _instantiate_evidence(
    evidence_aspects: Sequence[Tuple[int, Any]],
    index: SemanticIndex,
    junction: JunctionTable,
    roots: List[Tuple[str, str]],
    aspects: AspectSpace,
) -> None:
    """Overview aspects target junction sub-concepts; empirical/mechanism aspects target findings."""
    for i, a in evidence_aspects:
        if a.role == "overview":
            per_scope: List[Tuple[str, str, List[str]]] = []
            for facet, s in roots:
                tax = index.taxonomy(facet)
                kids = [concept_key(c) for c in tax.children(s) if junction.omega_extent(facet, c) > 0.0]
                if not kids and junction.omega_extent(facet, s) > 0.0:
                    kids = [concept_key(s)]
                if kids:
                    per_scope.append((facet, s, kids))
            total = sum(len(k) for _, _, k in per_scope)
            if total:
                for facet, s, kids in per_scope:
                    aspects.add_concept_group(f"O{i}|{facet}|{s}", i, a.weight * len(kids) / total, kids, facet)
                continue
        aspects.add_finding_group(f"F{i}", i, a.weight, junction.findings)


def _instantiate_background(
    analysis: QueryAnalysis, hyp: HypothesisSpec, junction: JunctionTable, aspects: AspectSpace
) -> None:
    """Background aspects of a hypothesis query target the already-studied baselines (Step 2).

    Overview aspects target the baseline concepts m' (covered by a direct path
    through m' in the candidate taxonomy); empirical / mechanism aspects
    target the baseline findings J(u_q, m') (covered by a path ending at the
    paper or one of its chunks). Without baselines they fall back to the
    known endpoint concept.
    """
    concepts = sorted({m for ms in hyp.baselines.values() for m in ms})
    for i, a in analysis.evidence_aspects:
        if a.role == "overview" and concepts:
            aspects.add_concept_group(f"B{i}", i, a.weight, [concept_key(m) for m in concepts], hyp.cand_facet)
        elif junction.findings:
            aspects.add_finding_group(f"F{i}", i, a.weight, junction.findings)
        else:
            aspects.add_concept_group(f"B{i}", i, a.weight, [concept_key(r) for r in hyp.roots], hyp.known_facet)


def _multi_source_distances(tax, sources: Iterable[str]) -> Dict[str, int]:
    """Tree distance from every node to the closest source node (BFS over parent/child edges)."""
    dist: Dict[str, int] = {}
    queue = deque()
    for s in sorted(set(sources)):
        dist[s] = 0
        queue.append(s)
    while queue:
        v = queue.popleft()
        node = tax.nodes[v]
        nbrs = list(node.children) + ([node.parent] if node.parent is not None else [])
        for w in nbrs:
            if w not in dist:
                dist[w] = dist[v] + 1
                queue.append(w)
    return dist


def _encodes(link_predictor: object, node_id: str) -> bool:
    """Whether the link predictor holds z for ``node_id`` (a concept or a registered anchor)."""
    has = getattr(link_predictor, "has", None)
    if callable(has):
        return bool(has(node_id))
    try:
        link_predictor.z(node_id)  # type: ignore[attr-defined]
    except KeyError:
        return False
    return True


def _hypothesis_spec(
    analysis: QueryAnalysis,
    grounding: Grounding,
    index: SemanticIndex,
    config: RetrievalConfig,
    link_predictor: Optional[object],
) -> HypothesisSpec:
    """Known endpoint, candidate scope, baselines m' and unexplored candidates (Section 4.1, Step 2).

    The known endpoint is the top grounded scope of its facet (or its
    query-local anchor, whose paths start at the anchor's neighbours). For an
    assessment the candidate is the given concept, or its query-local anchor
    when it is absent from I (never a substituted neighbour).
    """
    discovery = analysis.kind == HYPOTHESIS_DISCOVERY
    cand_facet = (analysis.searched_side or "method") if discovery else "method"
    known_facet = other_facet(cand_facet)
    anchor = grounding.anchor(known_facet)
    known_tax = index.taxonomy(known_facet)
    cand_tax = index.taxonomy(cand_facet)
    if anchor is not None:
        known_id, label = anchor.anchor_id, anchor.text
        roots = [n for n, _ in anchor.neighbors]
    else:
        known_id = grounding.top(known_facet)
        label = known_tax.nodes[known_id].name
        roots = [known_id]
    known_extent: Set[str] = set()
    for r in roots:
        known_extent |= known_tax.nodes[r].papers
    cand_scope = grounding.top(cand_facet) if discovery else cand_tax.root_id
    cand_anchor = None if discovery else grounding.anchor(cand_facet)
    fixed = None
    if not discovery:
        fixed = cand_anchor.anchor_id if cand_anchor is not None else grounding.top(cand_facet)
    use_link = bool(config.use_link_prediction and discovery and link_predictor is not None)
    if use_link:  # the known endpoint (or its anchor) must be encoded by the predictor
        use_link = _encodes(link_predictor, known_id)
    # J(u_q, m_q) of an assessed pair of two indexed concepts.
    concept_pair = fixed is not None and cand_anchor is None and anchor is None and fixed != cand_tax.root_id
    studied = known_extent & cand_tax.nodes[fixed].papers if concept_pair else set()
    # Baselines m': the most specific candidate-facet concepts of every known paper
    # (for assessment, papers already in J(u_q, m_q) are direct evidence, not bridges).
    excluded = cand_tax.nodes[fixed].papers if fixed is not None and cand_anchor is None else set()
    baselines: Dict[str, Tuple[str, ...]] = {}
    for pid in sorted(known_extent):
        if pid in excluded:
            continue
        deep = tuple(n for n in cand_tax.deepest_nodes(pid) if n != cand_tax.root_id)
        if deep:
            baselines[pid] = deep
    sources = {m for ms in baselines.values() for m in ms}
    d_min = _multi_source_distances(cand_tax, sources) if sources else {}
    # J(u_q, m) is non-empty iff m holds a known paper: the union of the known papers' postings.
    applied: Set[str] = set()
    for pid in known_extent:
        applied |= index.paper_concepts(pid, cand_facet)

    def valid(v: str) -> bool:
        return v != cand_tax.root_id and 1 <= d_min.get(v, 10 ** 6) <= config.max_bridge_distance

    candidates: Set[str] = set()
    has_candidate: Dict[str, bool] = {}
    phi_bar: Dict[str, float] = {}
    if discovery:
        order = list(cand_tax.descendants(cand_scope, include_self=True))
        for v in order:
            if valid(v) and v not in applied:
                candidates.add(v)
        for v in reversed(order):  # post-order: does the subtree of v hold a candidate, and its best phi?
            kids = cand_tax.children(v)
            has_candidate[v] = v in candidates or any(has_candidate.get(c, False) for c in kids)
            own = bridge_discount(d_min[v], config) if v in candidates else 0.0
            phi_bar[v] = max([own] + [phi_bar.get(c, 0.0) for c in kids])
    elif cand_anchor is not None:
        # The anchor hangs below each of its neighbours: d(m', anchor) = min_a d(m', a) + 1.
        d_anchor = min((d_min.get(n, 10 ** 6) for n, _ in cand_anchor.neighbors), default=10 ** 6) + 1
        if d_anchor <= config.max_bridge_distance:
            d_min[fixed] = d_anchor
            candidates.add(fixed)
    elif not studied and valid(fixed):
        candidates.add(fixed)
    return HypothesisSpec(
        mode=DISCOVERY if discovery else ASSESSMENT,
        known_facet=known_facet,
        cand_facet=cand_facet,
        known_id=known_id,
        known_label=label,
        roots=roots,
        known_extent=known_extent,
        cand_scope=cand_scope,
        fixed_candidate=fixed,
        use_link=use_link,
        candidates=candidates,
        has_candidate=has_candidate,
        baselines=baselines,
        d_min=d_min,
        phi_bar=phi_bar,
        cand_anchor=cand_anchor,
        studied=studied,
    )


# ---------------------------------------------------------------------------
# Algorithm 2, lines 5-23: exact bound-guided lazy search
# ---------------------------------------------------------------------------


@dataclass
class Selection:
    """One selected unit with its marginal tokens Delta C(y | S) and density delta(y | S)."""

    entry: UnitEntry
    tokens: int
    density: float


@dataclass
class SearchResult:
    selections: List[Selection]
    state: SelectionState
    stats: Dict[str, Any]

    @property
    def units(self) -> List[object]:
        return [s.entry.unit for s in self.selections]

    @property
    def uids(self) -> List[str]:
        return [s.entry.uid for s in self.selections]


class _Entry:
    """Heap entry (unit or region) with lazily refreshed numerator and eagerly updated costs.

    ``c_marg`` = sum of ``req_cost`` (unselected shared nodes) + sum of
    ``group_val`` (current lower bounds of the alternative groups).
    """

    __slots__ = ("eid", "obj", "is_unit", "sort_id", "num", "c_marg", "c_dens", "version", "alive",
                 "req_cost", "group_of", "groups", "group_val", "stale")

    def __init__(self, eid: int, obj: Any, is_unit: bool):
        self.eid = eid
        self.obj = obj
        self.is_unit = is_unit
        self.sort_id = obj.uid if is_unit else obj.rid
        self.num = 0.0
        self.c_marg = 0
        self.c_dens = 0
        self.version = 0
        self.alive = True
        self.req_cost: Dict[Token, int] = {}
        self.group_of: Dict[Token, int] = {}  # one-shot tokens of the plain groups
        self.groups: Tuple[AltGroup, ...] = ()
        self.group_val: List[int] = []
        self.stale: Set[int] = set()  # refreshable groups lowered to zero since their last refresh


def _tie_key(density: float, is_unit: bool, sort_id: str) -> Tuple[float, int, str]:
    """Heap order: larger density first; regions before units on ties; then by id (consistent ties)."""
    return (-density, 1 if is_unit else 0, sort_id)


class BoundGuidedSearch:
    """Algorithm 2 (lines 5-23): best-first search over regions with admissible bounds.

    Selects exactly the same sequence of units as :class:`ExhaustiveGreedy`
    (Proposition 7): numerators are refreshed lazily when an entry reaches the
    top of the max-heap Q, denominators eagerly through an inverted index, and
    ties are broken by (regions first, then id). Refreshable cost groups are
    lowered to zero eagerly and re-tightened lazily, like the numerators.
    """

    name = "bound_guided"

    def __init__(self, space: SearchSpace):
        self.space = space

    def run(self, budget: int) -> SearchResult:
        space = self.space
        shared = bool(space.config.use_shared_cost_update)
        state = SelectionState(budget, space.cost)
        heap: List[Tuple[float, int, str, int, int]] = []
        entries: Dict[int, _Entry] = {}
        inverted: Dict[Token, Set[int]] = {}  # one-shot: req nodes and plain group tokens
        watchers: Dict[Token, Set[int]] = {}  # every selection under the token: refreshable groups
        hit_tokens: Set[Token] = set()
        stats = dict(regions_created=0, regions_expanded=0, regions_deferred=0, regions_discarded=0,
                     units_created=0, units_reinserted=0, units_discarded=0, heap_pops=0, key_raises=0,
                     cost_refreshes=0)
        selections: List[Selection] = []
        t0 = time.perf_counter()

        def push(e: _Entry, key: float) -> None:
            heapq.heappush(heap, _tie_key(key, e.is_unit, e.sort_id) + (e.version, e.eid))

        def add(obj: Any, is_unit: bool) -> None:
            e = _Entry(len(entries), obj, is_unit)
            entries[e.eid] = e
            req = obj.nodes if is_unit else obj.req
            e.groups = () if is_unit else tuple(obj.groups)
            c_static = 0
            for x in dict.fromkeys(req):
                c = space.cost(x)
                c_static += c
                if x not in state.nodes:
                    e.req_cost[x] = c
                    e.c_marg += c
                    inverted.setdefault(x, set()).add(e.eid)
            for gi, g in enumerate(e.groups):
                c_static += g.min_cost
                hit = any(t in hit_tokens for t in g.tokens)
                val = g.value(state, hit)
                e.group_val.append(val)
                e.c_marg += val
                for t in g.tokens:
                    if g.refresh is not None:
                        watchers.setdefault(t, set()).add(e.eid)
                    elif not hit:
                        e.group_of[t] = gi
                        inverted.setdefault(t, set()).add(e.eid)
            e.c_dens = e.c_marg if shared else c_static
            if is_unit:
                stats["units_created"] += 1
                e.num = state.gain(obj.coverage)
            else:
                stats["regions_created"] += 1
                e.num = obj.numerator(space, state)
            if e.num <= 0.0 or e.c_marg > state.remaining:
                e.alive = False  # zero gain or infeasible: both permanent
                stats["units_discarded" if is_unit else "regions_discarded"] += 1
                return
            push(e, delta(e.num, e.c_dens))

        def lower_group(e: _Entry, gi: int) -> None:
            e.c_marg -= e.group_val[gi]
            if shared:
                e.c_dens -= e.group_val[gi]
            e.group_val[gi] = 0

        def refresh_costs(e: _Entry) -> None:
            """Re-tighten refreshable groups lowered since the last refresh (valid under the current S)."""
            for gi in sorted(e.stale):
                val = e.groups[gi].value(state, True)
                e.group_val[gi] = val
                e.c_marg += val
                if shared:
                    e.c_dens += val
                stats["cost_refreshes"] += 1
            e.stale.clear()

        def peek() -> Optional[Tuple[float, int, str, int, int]]:
            while heap:
                top = heap[0]
                e = entries[top[4]]
                if e.alive and e.version == top[3]:
                    return top
                heapq.heappop(heap)
            return None

        def select(e: _Entry, density: float) -> None:
            entry: UnitEntry = e.obj
            tokens = e.c_marg
            new = state.apply(entry.nodes, entry.coverage)
            selections.append(Selection(entry, tokens, density))
            affected: Set[int] = set()
            for x in new:
                for t in space.tokens_of(x):
                    watching = watchers.get(t)
                    if watching:
                        dead = []
                        for eid in watching:
                            other = entries[eid]
                            if not other.alive:
                                dead.append(eid)
                                continue
                            for gi, g in enumerate(other.groups):
                                if g.refresh is not None and t in g.tokens and other.group_val[gi] > 0:
                                    lower_group(other, gi)
                                    other.stale.add(gi)
                                    affected.add(eid)
                        watching.difference_update(dead)
                    if t in hit_tokens:
                        continue
                    hit_tokens.add(t)
                    for eid in inverted.pop(t, ()):
                        other = entries[eid]
                        if not other.alive:
                            continue
                        c = other.req_cost.pop(t, None)
                        if c is not None:
                            other.c_marg -= c
                            if shared:
                                other.c_dens -= c
                        gi = other.group_of.pop(t, None)
                        if gi is not None and other.group_val[gi] > 0:
                            lower_group(other, gi)
                        affected.add(eid)
            for eid in sorted(affected):  # line 21: raise the keys of entries sharing Delta V
                other = entries[eid]
                if other.alive:
                    other.version += 1
                    push(other, delta(other.num, other.c_dens))
                    stats["key_raises"] += 1

        for region in space.top_regions():  # line 7
            add(region, False)
        while True:  # line 8
            top = peek()
            if top is None or -top[0] <= 0.0:
                break
            heapq.heappop(heap)  # line 9
            stats["heap_pops"] += 1
            e = entries[top[4]]
            nxt = peek()
            if not e.is_unit:  # lines 11-15
                region: Region = e.obj
                if e.stale:
                    refresh_costs(e)
                if e.c_marg > state.remaining:
                    e.alive = False
                    stats["regions_discarded"] += 1
                    continue
                e.num = region.numerator(space, state)
                if e.num <= 0.0:
                    e.alive = False
                    stats["regions_discarded"] += 1
                    continue
                key = delta(e.num, e.c_dens)
                if nxt is None or _tie_key(key, False, e.sort_id) <= nxt[:3]:
                    e.alive = False
                    stats["regions_expanded"] += 1
                    units, children = region.expand(space)
                    for u in units:
                        add(u, True)
                    for r in children:
                        add(r, False)
                else:
                    e.version += 1
                    push(e, key)
                    stats["regions_deferred"] += 1
                continue
            # lines 17-23
            unit: UnitEntry = e.obj
            if e.c_marg > state.remaining:
                e.alive = False
                stats["units_discarded"] += 1
                continue
            e.num = state.gain(unit.coverage)
            if e.num <= 0.0:
                e.alive = False
                stats["units_discarded"] += 1
                continue
            d = delta(e.num, e.c_dens)
            if nxt is None or _tie_key(d, True, e.sort_id) <= nxt[:3]:
                e.alive = False
                select(e, d)
            else:
                e.version += 1
                push(e, d)
                stats["units_reinserted"] += 1
        stats.update(_common_stats(space, state, selections, time.perf_counter() - t0, self.name))
        return SearchResult(selections, state, stats)


class ExhaustiveGreedy:
    """Exhaustive greedy scan (Table 7 "w/o bound-guided search").

    Enumerates every candidate unit in scope (by fully expanding all regions),
    then repeatedly selects the feasible unit of maximum delta(z | S) (ties by
    id) until no feasible unit has a positive gain.
    """

    name = "exhaustive"

    def __init__(self, space: SearchSpace):
        self.space = space

    def enumerate_units(self) -> List[UnitEntry]:
        units: List[UnitEntry] = []
        stack: List[Region] = list(reversed(self.space.top_regions()))
        while stack:
            region = stack.pop()
            us, rs = region.expand(self.space)
            units.extend(us)
            stack.extend(reversed(rs))
        return units

    def run(self, budget: int) -> SearchResult:
        space = self.space
        shared = bool(space.config.use_shared_cost_update)
        t0 = time.perf_counter()
        units = self.enumerate_units()
        state = SelectionState(budget, space.cost)
        marginal = [state.marginal_cost(u.nodes) for u in units]
        standalone = list(marginal)
        holders: Dict[str, List[int]] = {}
        for i, u in enumerate(units):
            for x in set(u.nodes):
                holders.setdefault(x, []).append(i)
        alive = list(range(len(units)))
        selections: List[Selection] = []
        evaluations = 0
        while alive:
            best: Optional[int] = None
            best_key: Optional[Tuple[float, str]] = None
            best_d = 0.0
            keep: List[int] = []
            remaining = state.remaining
            for i in alive:
                if marginal[i] > remaining:
                    continue  # infeasibility is permanent (proof of Prop. 7)
                g = state.gain(units[i].coverage)
                evaluations += 1
                if g <= 0.0:
                    continue  # zero gain is permanent (submodularity)
                keep.append(i)
                d = delta(g, marginal[i] if shared else standalone[i])
                key = (-d, units[i].uid)
                if best_key is None or key < best_key:
                    best, best_key, best_d = i, key, d
            if best is None:
                break
            entry = units[best]
            tokens = marginal[best]
            for x in state.apply(entry.nodes, entry.coverage):
                c = space.cost(x)
                for j in holders.get(x, ()):
                    marginal[j] -= c
            selections.append(Selection(entry, tokens, best_d))
            alive = [i for i in keep if i != best]
        stats = dict(units_created=len(units), evaluations=evaluations)
        stats.update(_common_stats(space, state, selections, time.perf_counter() - t0, self.name))
        return SearchResult(selections, state, stats)


def _common_stats(space: SearchSpace, state: SelectionState, selections: Sequence[Selection],
                  elapsed: float, algorithm: str) -> Dict[str, Any]:
    """Search statistics (Table 5, Table 7).

    * ``scored_candidates``: candidate concepts whose exact eta_hat_q
      (link prior and bridge discount) was computed, i.e. the "Scored" column
      of Table 7; ``candidate_pool`` is the number of unexplored candidates in
      scope (what an exhaustive scan scores) and ``link_evaluations`` the
      number of s_theta(u_q, m) calls (0 for assessment / w/o link prior).
    * ``scored_nodes``: distinct index nodes whose exact relevance was computed.
    * ``postings_pruned_by_junction``: papers below subtrees cut by the single
      posting-list intersection of Section 4.3; ``postings_pruned`` counts every
      paper in scope that was never scanned (junction pruning and deferral).
    """
    total = space.postings_total()
    scanned = len(space.postings_scanned)
    hyp = space.hyp
    return {
        "algorithm": algorithm,
        "selected": len(selections),
        "tokens_used": state.used,
        "coverage": state.value(),
        "scored_nodes": space.scorer.relevance.scored,
        "bound_evaluations": space.scorer.relevance.bound_evaluations,
        "scored_candidates": len(space.scored_candidates),
        "candidate_pool": len(hyp.candidates) if hyp is not None else 0,
        "link_evaluations": space.link_evaluations,
        "postings_total": total,
        "postings_scanned": scanned,
        "postings_pruned": max(0, total - scanned),
        "postings_pruned_by_junction": space.postings_pruned_by_junction,
        "junction_intersections": space.scorer.junction.intersections if space.scorer.junction else 0,
        "search_time_s": elapsed,
    }


def make_search(space: SearchSpace) -> Any:
    """BoundGuidedSearch, or ExhaustiveGreedy when ``use_bound_guided_search`` is off."""
    if space.config.use_bound_guided_search:
        return BoundGuidedSearch(space)
    return ExhaustiveGreedy(space)


# ---------------------------------------------------------------------------
# Appendix A.3: ranked join over grounded scopes
# ---------------------------------------------------------------------------


@dataclass
class RankedJoinResult:
    top: List[Tuple[str, float]]  # (paper id, score) in nonincreasing score
    pairs_processed: int
    pairs_total: int
    papers_scored: int
    postings_scanned: int
    postings_total: int

    @property
    def postings_pruned(self) -> int:
        return self.postings_total - self.postings_scanned


def ranked_join(
    index: SemanticIndex,
    pairs: Sequence[Tuple[str, str, float]],
    relevance: Callable[[str], float],
    k: int,
) -> RankedJoinResult:
    """Top-k papers under score(p) = omega_q(p) rel_a(p) (Eq. 11) with early termination (Prop. 3).

    ``pairs`` are grounded scope pairs ``(u, m, g_u g_m)`` and ``relevance``
    returns rel_a(p) in [0, 1]. Pairs are processed in nonincreasing weight;
    their postings are intersected and every unseen paper is scored with the
    weight of the first pair that exposes it (its junction weight). The join
    stops once k papers are held and the next pair's weight does not exceed
    the current k-th score.
    """
    order = sorted(pairs, key=lambda t: (-float(t[2]), t[0], t[1]))
    sizes = [len(index.extent(u)) + len(index.extent(m)) for u, m, _ in order]
    if k <= 0:
        return RankedJoinResult([], 0, len(order), 0, 0, sum(sizes))
    seen: Set[str] = set()
    heap: List[Tuple[float, str]] = []  # min-heap of the k best (score, paper id)
    scanned = 0
    processed = 0
    for idx, (u, m, w) in enumerate(order):
        w = float(w)
        if len(heap) >= k and w <= heap[0][0]:
            break
        processed += 1
        scanned += sizes[idx]
        for p in sorted(index.shared_papers(u, m)):
            if p in seen:
                continue
            seen.add(p)
            s = w * float(relevance(p))
            if len(heap) < k:
                heapq.heappush(heap, (s, p))
            elif (s, p) > heap[0]:
                heapq.heapreplace(heap, (s, p))
    top = sorted(((p, s) for s, p in heap), key=lambda t: (-t[1], t[0]))
    return RankedJoinResult(top, processed, len(order), len(seen), scanned, sum(sizes))


__all__ = [
    "AltGroup",
    "BoundGuidedSearch",
    "BridgeLists",
    "BridgeRegion",
    "CandidateRegion",
    "CandidateSubtreeRegion",
    "ChunkRegion",
    "ConceptRegion",
    "ExhaustiveGreedy",
    "HypothesisSpec",
    "PairGrid",
    "PairRow",
    "RangeArgMin",
    "RankedJoinResult",
    "Region",
    "SearchResult",
    "SearchSpace",
    "Selection",
    "UnitEntry",
    "Witness",
    "WitnessCosts",
    "build_search_space",
    "make_search",
    "ranked_join",
]
