"""Hierarchy-bounded subtree pruning and exact lazy search (Section 4.3, Algorithm 2 lines 5-23).

Candidate units are grouped into hierarchical *search regions* H. Each region
caches an adequacy upper bound r_bar_a(H) >= max_{z in H} r_hat_{a,z} and a
marginal-cost lower bound c_low(H | S) <= min_{z in H} Delta C(z | S), so that
(Eq. 7, Lemma 6)::

    delta_bar(H | S) = sum_{a in A_q(H)} w_a max{0, r_bar_a(H) - f_a(S)} / c_low(H | S)  >=  delta(z | S).

Regions
-------
Direct paths (evidence aspects):

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
  bound is sigmoid(U(u_q, v)) * min{s_bar_P(H_P), s_bar_M(H_M)} (Eq. 8, 12),
  so uncompetitive subtrees stay deferred without scoring their concepts;
* ``CandidateRegion(c)``: path sets of one candidate (scores s_theta(u_q, c));
* ``BridgeRegion(c, d)``: path sets of c whose bridge anchors are at
  taxonomic distance d (constant bridge discount phi);
* ``PairGrid`` / ``PairRow``: a lazy, exact best-first enumeration of
  (known-side witness, candidate-side witness) pairs over the two witness
  lists sorted by premise adequacy; since Eq. (9) takes a minimum, the bound
  of a grid cell (i, j) is eta * min{s_K[i], s_C[j]}.

A discovery query starts from ``CandidateSubtreeRegion(m_q)``; an assessment
query fixes the pair, starts from ``CandidateRegion(m_q)`` and uses
eta = phi (no link-prediction factor).

Every unit of a region contains the region's ``req`` nodes and one node of
each of its alternative groups, so c_low(H | S) = sum of the unselected req
costs + sum of the minimum costs of the groups none of whose nodes is
selected yet (Appendix A.5: "the smallest terminal-node token cost in H, or
zero once one is selected"). An inverted index maps nodes (and structural
tokens: "a paper below v", "a concept below v") to the active entries, and
selecting a unit lowers the cost bounds of the entries sharing its new nodes
eagerly (Algorithm 2, line 21).

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
from scholarpath.retrieval.grounding import Grounding, other_facet
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
    HYP_PREFIX,
    ComplementaryPathSet,
    DirectPath,
    PathFactory,
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
    """Every unit of a region contains at least one node matched by these tokens."""

    tokens: Tuple[Token, ...]
    min_cost: int


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

    @property
    def known_is_concept(self) -> bool:
        return len(self.roots) == 1 and self.roots[0] == self.known_id


@dataclass
class Witness:
    """A premise path (paper or chunk level) with its adequacy s_hat (Eq. 9)."""

    score: float
    path: DirectPath
    baseline: Optional[str] = None  # m' (known side)
    bridge: Optional[str] = None  # f = LCA(m', c) (known side)


@dataclass
class BridgeLists:
    """Sorted witness lists of a bridge region, shared by its pair regions."""

    candidate: str
    distance: int
    eta: float
    phi: float
    known: List[Witness]
    cand: List[Witness]


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
        if self._bound is None:
            eta = space.link_bound(self.node)  # phi_bar <= 1
            self._bound = eta * min(space.known_premise_bound(), space.cand_premise_bound(self.node))
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
        hyp_key = space.renderer.register_hypothesis(candidate, space.hyp.known_label)
        self.req = space.hyp_req() + (concept_key(candidate), hyp_key)
        self.groups = space.hyp_groups(candidate, include_concepts=False)
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
                 items: List[Tuple[str, str, str]]):
        self.candidate = parent.candidate
        self.distance = distance
        self.items = items  # (known paper, m', f)
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
        for pid, baseline, bridge in self.items:
            for w in space.known_witnesses(pid):
                if w.score > 0.0:
                    known.append(Witness(w.score, w.path, baseline, bridge))
        cand = [w for w in space.cand_witnesses(self.candidate) if w.score > 0.0]
        if not known or not cand or self.eta <= 0.0:
            return [], []
        known.sort(key=lambda w: (-w.score, w.path.end))
        lists = BridgeLists(self.candidate, self.distance, self.eta, self.phi, known, cand)
        return [], [PairGrid(space, self, lists, 0, 0)]


class PairGrid(Region):
    """Pairs (a, b) with a >= i, b >= j of the sorted witness lists (lazy best-first enumeration)."""

    def __init__(self, space: "SearchSpace", parent: Region, lists: BridgeLists, i: int, j: int):
        self.lists, self.i, self.j = lists, i, j
        self.rid = f"HG|{lists.candidate}|{lists.distance}|{i}|{j}"
        self.req = parent.req
        self.groups = parent.groups
        self.bound = lists.eta * min(lists.known[i].score, lists.cand[j].score)

    def numerator(self, space: "SearchSpace", state: SelectionState) -> float:
        return space.hyp_numerator(state, self.lists.candidate, self.bound)

    def expand(self, space: "SearchSpace") -> Tuple[List[UnitEntry], List[Region]]:
        L = self.lists
        units = [space.pair_unit(L, self.i, self.j)]
        regions: List[Region] = []
        if self.j + 1 < len(L.cand):
            regions.append(PairRow(space, self, L, self.i, self.j + 1))
        if self.i + 1 < len(L.known):
            regions.append(PairGrid(space, self, L, self.i + 1, self.j))
        return units, regions


class PairRow(Region):
    """Pairs (i, b) with b >= j: the known-side witness i is fixed."""

    def __init__(self, space: "SearchSpace", parent: Region, lists: BridgeLists, i: int, j: int):
        self.lists, self.i, self.j = lists, i, j
        self.rid = f"HR|{lists.candidate}|{lists.distance}|{i}|{j}"
        kw = lists.known[i]
        e_c = space.paths.concept_chain(space.hyp.cand_facet, kw.bridge, lists.candidate)
        seen: Dict[str, None] = {}
        head = space.hyp_req() + (concept_key(lists.candidate), f"{HYP_PREFIX}{lists.candidate}")
        for k in head + e_c + kw.path.nodes:
            seen.setdefault(k, None)
        self.req = tuple(seen)
        self.groups = (space.cand_paper_group(lists.candidate),)
        self.bound = lists.eta * min(kw.score, lists.cand[j].score)

    def numerator(self, space: "SearchSpace", state: SelectionState) -> float:
        return space.hyp_numerator(state, self.lists.candidate, self.bound)

    def expand(self, space: "SearchSpace") -> Tuple[List[UnitEntry], List[Region]]:
        L = self.lists
        units = [space.pair_unit(L, self.i, self.j)]
        regions: List[Region] = []
        if self.j + 1 < len(L.cand):
            regions.append(PairRow(space, self, L, self.i, self.j + 1))
        return units, regions


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
        self.link_scored = 0
        self._min_paper: Dict[Tuple[str, str], int] = {}
        self._min_concept: Dict[Tuple[str, str], int] = {}
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
        key = (facet, node)
        hit = self._min_paper.get(key)
        if hit is None:
            ext = self.index.taxonomy(facet).nodes[node].papers
            hit = min((self.cost(paper_key(p)) for p in ext), default=0)
            self._min_paper[key] = hit
        return hit

    def min_concept_cost(self, facet: str, node: str) -> int:
        key = (facet, node)
        hit = self._min_concept.get(key)
        if hit is None:
            tax = self.index.taxonomy(facet)
            hit = min(self.cost(concept_key(n)) for n in tax.descendants(node, include_self=True))
            self._min_concept[key] = hit
        return hit

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

    def cand_paper_group(self, node: str) -> AltGroup:
        cf = self.hyp.cand_facet
        return AltGroup((("ext", cf, node),), self.min_paper_cost(cf, node))

    def known_paper_group(self) -> AltGroup:
        if self._known_group is None:
            kf = self.hyp.known_facet
            roots = self.hyp.roots
            self._known_group = AltGroup(
                tuple(("ext", kf, r) for r in roots), min(self.min_paper_cost(kf, r) for r in roots)
            )
        return self._known_group

    def hyp_groups(self, node: str, include_concepts: bool) -> Tuple[AltGroup, ...]:
        hyp = self.hyp
        groups: List[AltGroup] = []
        if include_concepts:
            groups.append(AltGroup((("sub", hyp.cand_facet, node),), self.min_concept_cost(hyp.cand_facet, node)))
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
        self.link_scored += 1
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
        hit = self._cand_bound.get(node)
        if hit is None:
            hit = self.scorer.premise_bound(paper_balls(self.index, node), self.hyp.cand_facet)
            self._cand_bound[node] = hit
        return hit

    def hyp_numerator(self, state: SelectionState, candidate: str, rbar: float) -> float:
        g = self.aspects.hypothesis_group
        if g is None:
            return 0.0
        return g.weight * max(0.0, rbar - state.support(g.iid(candidate)))

    def bridge_groups(self, candidate: str) -> Dict[int, List[Tuple[str, str, str]]]:
        """Known papers grouped by the distance d(m', c) of their closest baseline concept m'."""
        hyp = self.hyp
        tax = self.index.taxonomy(hyp.cand_facet)
        ext_c = tax.nodes[candidate].papers
        dist: Dict[str, int] = {}
        out: Dict[int, List[Tuple[str, str, str]]] = {}
        for pid in sorted(hyp.baselines):
            if pid in ext_c:
                continue  # already a direct junction of the pair
            best: Optional[Tuple[int, str]] = None
            for m in hyp.baselines[pid]:
                d = dist.get(m)
                if d is None:
                    d = tax.distance(m, candidate)
                    dist[m] = d
                if best is None or (d, m) < best:
                    best = (d, m)
            if best is None or best[0] < 1 or best[0] > self.config.max_bridge_distance:
                continue
            d, m = best
            out.setdefault(d, []).append((pid, m, tax.lowest_common_ancestor(m, candidate)))
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
        """Candidate-side premise paths below c, sorted by adequacy (cached per candidate)."""
        hit = self._cand_w.get(candidate)
        if hit is None:
            hyp = self.hyp
            tax = self.index.taxonomy(hyp.cand_facet)
            hit = []
            for pid in sorted(tax.nodes[candidate].papers):
                if pid not in hyp.known_extent:
                    hit.extend(self._witnesses(hyp.cand_facet, candidate, pid))
            hit.sort(key=lambda w: (-w.score, w.path.end))
            self._cand_w[candidate] = hit
        return hit

    def pair_unit(self, lists: BridgeLists, i: int, j: int) -> UnitEntry:
        """The complementary path set of grid cell (i, j), scored by Eq. (9)."""
        hyp = self.hyp
        kw, cw = lists.known[i], lists.cand[j]
        e_c = self.paths.concept_path(hyp.cand_facet, kw.bridge, lists.candidate)
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
    """Resolve what the aspects target given the scopes and set up the top-level regions."""
    renderer = renderer or ContextRenderer(index)
    paths = PathFactory(index, config.max_chunks_per_paper)
    aspects = AspectSpace()
    hyp: Optional[HypothesisSpec] = None
    studied: Set[str] = set()
    if analysis.kind == EVIDENCE:
        junction, direct_roots = _evidence_scopes(index, grounding)
    else:
        hyp = _hypothesis_spec(analysis, grounding, index, config, link_predictor)
        direct_roots = [(hyp.known_facet, r) for r in hyp.roots]
        cand_tax = index.taxonomy(hyp.cand_facet)
        if hyp.fixed_candidate is not None and hyp.fixed_candidate != cand_tax.root_id:
            # Assessment of a pair that the literature already studies: J(u_q, m_q) is direct evidence.
            studied = hyp.known_extent & cand_tax.nodes[hyp.fixed_candidate].papers
        if studied:
            junction = JunctionTable(index, [(studied, 1.0)])
            direct_roots.append((hyp.cand_facet, hyp.fixed_candidate))
        else:
            junction = JunctionTable(index, [(hyp.known_extent, 1.0)])
    scorer = AdequacyScorer(index, analysis, config, junction)
    if analysis.kind == EVIDENCE or studied:
        _instantiate_evidence(analysis, index, junction, direct_roots, aspects)
    else:
        for i, a in analysis.evidence_aspects:  # background on the known endpoint
            aspects.add_concept_group(f"B{i}", i, a.weight, [concept_key(r) for r in hyp.roots], hyp.known_facet)
    if hyp is not None:
        premise_weight = sum(a.weight for a in analysis.aspects if a.is_premise)
        aspects.add_hypothesis_group(premise_weight, len(hyp.candidates))
    return SearchSpace(index, analysis, grounding, config, scorer, aspects, renderer, paths, link_predictor,
                       hyp, direct_roots)


def _evidence_scopes(index: SemanticIndex, grounding: Grounding) -> Tuple[JunctionTable, List[Tuple[str, str]]]:
    """Junctions J(u, m) over U_q x M_q weighted by g_u g_m, and the scopes paths descend from."""
    pairs = [(u, m, gu * gm) for u, gu in grounding.problem_scopes for m, gm in grounding.method_scopes]
    junction = JunctionTable.from_pairs(index, pairs)
    if not junction.findings:
        # No observed junction: fall back to the extents of the matched scopes of each facet.
        root_p, root_m = index.problem.root_id, index.method.root_id
        pairs = [(u, root_m, g) for u, g in grounding.problem_scopes] + [
            (root_p, m, g) for m, g in grounding.method_scopes
        ]
        junction = JunctionTable.from_pairs(index, pairs)
    roots: List[Tuple[str, str]] = []
    for facet in ("problem", "method"):
        if not grounding.unspecified.get(facet, True):
            roots.extend((facet, s) for s, _ in grounding.scopes(facet))
    if not roots:
        roots = [("problem", index.problem.root_id)]
    return junction, roots


def _instantiate_evidence(
    analysis: QueryAnalysis,
    index: SemanticIndex,
    junction: JunctionTable,
    roots: List[Tuple[str, str]],
    aspects: AspectSpace,
) -> None:
    """Overview aspects target junction sub-concepts; empirical/mechanism aspects target findings."""
    for i, a in analysis.evidence_aspects:
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


def _hypothesis_spec(
    analysis: QueryAnalysis,
    grounding: Grounding,
    index: SemanticIndex,
    config: RetrievalConfig,
    link_predictor: Optional[object],
) -> HypothesisSpec:
    """Known endpoint, candidate scope, baselines m' and unexplored candidates (Section 4.1, Step 2)."""
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
    fixed = None if discovery else grounding.top(cand_facet)
    use_link = bool(config.use_link_prediction and discovery and link_predictor is not None)
    if use_link:  # the known endpoint (or its anchor) must be encoded by the predictor
        try:
            if cand_facet == "method":
                link_predictor.upper_bound(known_id, cand_tax.root_id)
            else:
                link_predictor.upper_bound_problems(known_id, cand_tax.root_id)
        except KeyError:
            use_link = False
    # Baselines m': the most specific candidate-facet concepts of every known paper
    # (for assessment, papers already in J(u_q, m_q) are direct evidence, not bridges).
    excluded = cand_tax.nodes[fixed].papers if fixed is not None else set()
    baselines: Dict[str, Tuple[str, ...]] = {}
    for pid in sorted(known_extent):
        if pid in excluded:
            continue
        deep = tuple(n for n in cand_tax.deepest_nodes(pid) if n != cand_tax.root_id)
        if deep:
            baselines[pid] = deep
    sources = {m for ms in baselines.values() for m in ms}
    d_min = _multi_source_distances(cand_tax, sources) if sources else {}
    applied: Dict[str, bool] = {}

    def is_applied(v: str) -> bool:
        hit = applied.get(v)
        if hit is None:
            ext = cand_tax.nodes[v].papers
            small, large = (ext, known_extent) if len(ext) <= len(known_extent) else (known_extent, ext)
            hit = any(p in large for p in small)
            applied[v] = hit
        return hit

    def valid(v: str) -> bool:
        return v != cand_tax.root_id and 1 <= d_min.get(v, 10 ** 6) <= config.max_bridge_distance

    candidates: Set[str] = set()
    has_candidate: Dict[str, bool] = {}
    if discovery:
        order = list(cand_tax.descendants(cand_scope, include_self=True))
        for v in order:
            if valid(v) and not is_applied(v):
                candidates.add(v)
        for v in reversed(order):  # post-order flags: does the subtree of v contain a candidate?
            has_candidate[v] = v in candidates or any(has_candidate.get(c, False) for c in cand_tax.children(v))
    elif fixed is not None and valid(fixed):
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
    """Heap entry (unit or region) with lazily refreshed numerator and eagerly updated costs."""

    __slots__ = ("eid", "obj", "is_unit", "sort_id", "num", "c_marg", "c_dens", "version", "alive",
                 "req_cost", "group_of", "group_min", "group_hit")

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
        self.group_of: Dict[Token, int] = {}
        self.group_min: List[int] = []
        self.group_hit: List[bool] = []


def _tie_key(density: float, is_unit: bool, sort_id: str) -> Tuple[float, int, str]:
    """Heap order: larger density first; regions before units on ties; then by id (consistent ties)."""
    return (-density, 1 if is_unit else 0, sort_id)


class BoundGuidedSearch:
    """Algorithm 2 (lines 5-23): best-first search over regions with admissible bounds.

    Selects exactly the same sequence of units as :class:`ExhaustiveGreedy`
    (Proposition 7): numerators are refreshed lazily when an entry reaches the
    top of the max-heap Q, denominators eagerly through an inverted index, and
    ties are broken by (regions first, then id).
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
        inverted: Dict[Token, Set[int]] = {}
        hit_tokens: Set[Token] = set()
        stats = dict(regions_created=0, regions_expanded=0, regions_deferred=0, regions_discarded=0,
                     units_created=0, units_reinserted=0, units_discarded=0, heap_pops=0, key_raises=0)
        selections: List[Selection] = []
        t0 = time.perf_counter()

        def push(e: _Entry, key: float) -> None:
            heapq.heappush(heap, _tie_key(key, e.is_unit, e.sort_id) + (e.version, e.eid))

        def add(obj: Any, is_unit: bool) -> None:
            e = _Entry(len(entries), obj, is_unit)
            entries[e.eid] = e
            req = obj.nodes if is_unit else obj.req
            groups = () if is_unit else obj.groups
            c_static = 0
            for x in dict.fromkeys(req):
                c = space.cost(x)
                c_static += c
                if x not in state.nodes:
                    e.req_cost[x] = c
                    e.c_marg += c
                    inverted.setdefault(x, set()).add(e.eid)
            for gi, g in enumerate(groups):
                c_static += g.min_cost
                hit = any(t in hit_tokens for t in g.tokens)
                e.group_min.append(g.min_cost)
                e.group_hit.append(hit)
                if not hit:
                    e.c_marg += g.min_cost
                    for t in g.tokens:
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
                        if gi is not None and not other.group_hit[gi]:
                            other.group_hit[gi] = True
                            other.c_marg -= other.group_min[gi]
                            if shared:
                                other.c_dens -= other.group_min[gi]
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
    total = space.postings_total()
    scanned = len(space.postings_scanned)
    return {
        "algorithm": algorithm,
        "selected": len(selections),
        "tokens_used": state.used,
        "coverage": state.value(),
        "scored_nodes": space.scorer.relevance.scored,
        "bound_evaluations": space.scorer.relevance.bound_evaluations,
        "scored_candidates": space.link_scored,
        "postings_total": total,
        "postings_scanned": scanned,
        "postings_pruned": max(0, total - scanned),
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
    seen: Set[str] = set()
    heap: List[Tuple[float, str]] = []  # min-heap of the k best (score, -order via id)
    scanned = 0
    processed = 0
    for idx, (u, m, w) in enumerate(order):
        w = float(w)
        if k > 0 and len(heap) >= k and w <= heap[0][0]:
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
    "BridgeRegion",
    "CandidateRegion",
    "CandidateSubtreeRegion",
    "ChunkRegion",
    "ConceptRegion",
    "ExhaustiveGreedy",
    "HypothesisSpec",
    "PairGrid",
    "PairRow",
    "RankedJoinResult",
    "Region",
    "SearchResult",
    "SearchSpace",
    "Selection",
    "UnitEntry",
    "build_search_space",
    "make_search",
    "ranked_join",
]
