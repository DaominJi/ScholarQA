"""Redundancy-aware aspect coverage (Section 4.2, Eq. 5-6, Proposition 5).

The utility of a selection S of candidate units is the weighted maximum
adequacy::

    F_q(S) = sum_{a in A_q} w_a f_a(S),     f_a(S) = max_{z in S} r_hat_{a,z},   f_a(empty) = 0,   (5)

and units are prioritized by their marginal coverage gain per new token::

    delta(z | S) = sum_{a in A_q(z)} w_a max{0, r_hat_{a,z} - f_a(S)} / sum_{x in V(z) minus V(S)} c(x),   (6)

with delta = +inf when the marginal cost is zero and the gain positive, and
delta = 0 when the gain is zero (Appendix A.5). The token cost counts every
distinct node once, C(S) = sum_{x in V(S)} c(x), and a unit is feasible iff
its marginal cost fits the remaining budget B' = B - C(S).

Aspect instances
----------------
Step 2 lets the grounded scopes "set what the aspects target" (Section 4.1).
Each LLM aspect is therefore instantiated over its targets, and every
instance is an element of A_q with its own support f:

* evidence aspects with an *overview* role target the junction sub-concepts
  (a path covers the instance iff the target concept lies on it);
* evidence aspects with an *empirical* or *mechanism* role target the
  established findings, i.e. the papers of J(u_q, m_q) (a path covers the
  instance iff it ends at the paper or at one of its chunks);
* background aspects of a hypothesis query target the already-studied
  baselines of the known endpoint: overview aspects the baseline concepts m'
  (a path covers m' iff m' lies on it), empirical / mechanism aspects the
  findings J(u_q, m') (as above);
* the premise aspects of a hypothesis query define one hypothesis instance
  h_c per unexplored candidate c, covered by the path sets of c (Eq. 9). When
  an assessed pair is already studied (J(u_q, m_q) non-empty) no hypothesis is
  formed and every aspect targets the findings J(u_q, m_q).

The weight w_a of an aspect is split evenly over its targets, so the instance
weights still sum to one. Instances are organized in *groups* such that a
unit covers at most one instance of a group; region bounds (Eq. 7) use this
to take a maximum instead of a sum over a group (see ``search.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

INF = math.inf

CONCEPT_GROUP = "concept"  # explicit target keys (overview sub-concepts, baselines m', known endpoint)
FINDING_GROUP = "finding"  # papers of the junction J(u_q, m_q)
HYPOTHESIS_GROUP = "hypothesis"  # one instance per candidate c


@dataclass(frozen=True)
class InstanceGroup:
    """Aspect instances of which a single unit can cover at most one."""

    gid: str
    kind: str  # concept | finding | hypothesis
    base: int  # index of the LLM aspect (-1 for the hypothesis group)
    weight: float  # weight w of every instance in the group
    targets: FrozenSet[str] = frozenset()  # target keys (concept groups)
    facet: Optional[str] = None  # facet of the target concepts (concept groups)

    def iid(self, target: str) -> str:
        return f"{self.gid}|{target}"


@dataclass
class Coverage:
    """A_q(z) with the adequacies of unit z: ``(instance id, weight, r_hat)`` triples."""

    items: Tuple[Tuple[str, float, float], ...] = ()

    def __bool__(self) -> bool:
        return any(r > 0.0 and w > 0.0 for _, w, r in self.items)

    def as_dict(self) -> Dict[str, float]:
        return {iid: r for iid, _, r in self.items}


class AspectSpace:
    """The instantiated aspect set A_q of one query and the coverage rules of its units."""

    def __init__(self) -> None:
        self.groups: Dict[str, InstanceGroup] = {}
        self._by_target: Dict[str, List[InstanceGroup]] = {}  # concept-group target key -> groups
        self.finding_groups: List[InstanceGroup] = []
        self.findings: Mapping[str, float] = {}
        self.hypothesis_group: Optional[InstanceGroup] = None

    # ------------------------------------------------------------ building
    def add_concept_group(self, gid: str, base: int, total_weight: float, targets: Iterable[str],
                          facet: Optional[str]) -> Optional[InstanceGroup]:
        targets = frozenset(targets)
        if not targets or total_weight <= 0:
            return None
        g = InstanceGroup(gid, CONCEPT_GROUP, base, total_weight / len(targets), targets, facet)
        self.groups[gid] = g
        for t in targets:
            self._by_target.setdefault(t, []).append(g)
        return g

    def add_finding_group(
        self, gid: str, base: int, total_weight: float, findings: Mapping[str, float]
    ) -> Optional[InstanceGroup]:
        if not findings or total_weight <= 0:
            return None
        g = InstanceGroup(gid, FINDING_GROUP, base, total_weight / len(findings))
        self.groups[gid] = g
        self.finding_groups.append(g)
        self.findings = findings
        return g

    def add_hypothesis_group(self, total_weight: float, n_candidates: int) -> Optional[InstanceGroup]:
        if n_candidates <= 0 or total_weight <= 0:
            return None
        g = InstanceGroup("H", HYPOTHESIS_GROUP, -1, total_weight / n_candidates)
        self.groups["H"] = g
        self.hypothesis_group = g
        return g

    # ------------------------------------------------------------ coverage
    def concept_groups_of(self, key: str) -> List[InstanceGroup]:
        return self._by_target.get(key, [])

    def direct_coverage(self, nodes: Sequence[str], paper_id: Optional[str], rhat: Sequence[float]) -> Coverage:
        """A_q(e) of a direct path: concept targets on the path, plus the finding it ends at."""
        items: List[Tuple[str, float, float]] = []
        for key in nodes:
            for g in self._by_target.get(key, ()):
                items.append((g.iid(key), g.weight, float(rhat[g.base])))
        if paper_id is not None and paper_id in self.findings:
            for g in self.finding_groups:
                items.append((g.iid(paper_id), g.weight, float(rhat[g.base])))
        return Coverage(tuple(items))

    def hypothesis_coverage(self, candidate: str, rhat: float) -> Coverage:
        g = self.hypothesis_group
        if g is None:
            return Coverage()
        return Coverage(((g.iid(candidate), g.weight, float(rhat)),))


def delta(gain: float, cost: float) -> float:
    """Eq. (6) with the conventions of Appendix A.5."""
    if gain <= 0.0:
        return 0.0
    if cost <= 0:
        return INF
    return gain / cost


class SelectionState:
    """S, B' and the aspect support table f_a(S) of Algorithm 2 (lines 6, 19-20)."""

    def __init__(self, budget: int, cost: Callable[[str], int]):
        self.budget = int(budget)
        self.cost_of = cost
        self.f: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self.nodes: Set[str] = set()
        self.node_order: List[str] = []
        self.used = 0

    @property
    def remaining(self) -> int:
        """B' = B - C(S)."""
        return self.budget - self.used

    def support(self, iid: str) -> float:
        return self.f.get(iid, 0.0)

    def gain(self, coverage: Coverage) -> float:
        """Numerator of Eq. (6): sum_{a in A_q(z)} w_a max{0, r_hat_{a,z} - f_a(S)}."""
        g = 0.0
        f = self.f
        for iid, w, r in coverage.items:
            d = r - f.get(iid, 0.0)
            if d > 0.0:
                g += w * d
        return g

    def marginal_cost(self, nodes: Iterable[str]) -> int:
        """Delta C(z | S) = sum of c(x) over x in V(z) minus V(S)."""
        return sum(self.cost_of(x) for x in set(nodes) if x not in self.nodes)

    def standalone_cost(self, nodes: Iterable[str]) -> int:
        return sum(self.cost_of(x) for x in set(nodes))

    def apply(self, nodes: Sequence[str], coverage: Coverage) -> List[str]:
        """Select a unit: B' -= Delta C, V(S) grows by Delta V, f_a = max{f_a, r_hat_{a,y}} (lines 19-20)."""
        new = []
        for x in nodes:
            if x not in self.nodes:
                self.nodes.add(x)
                self.node_order.append(x)
                self.used += self.cost_of(x)
                new.append(x)
        for iid, w, r in coverage.items:
            self.weights[iid] = w
            if r > self.f.get(iid, 0.0):
                self.f[iid] = r
        return new

    def value(self) -> float:
        """F_q(S) of Eq. (5)."""
        return float(sum(self.weights[i] * v for i, v in self.f.items()))


def coverage_value(coverages: Iterable[Coverage]) -> float:
    """F_q(S) computed from scratch for a set of unit coverages (Eq. 5)."""
    best: Dict[str, float] = {}
    weight: Dict[str, float] = {}
    for cov in coverages:
        for iid, w, r in cov.items:
            weight[iid] = w
            if r > best.get(iid, 0.0):
                best[iid] = r
    return float(sum(weight[i] * v for i, v in best.items()))


__all__ = [
    "AspectSpace",
    "CONCEPT_GROUP",
    "Coverage",
    "FINDING_GROUP",
    "HYPOTHESIS_GROUP",
    "INF",
    "InstanceGroup",
    "SelectionState",
    "coverage_value",
    "delta",
]
