"""Adequacy scores and their admissible subtree bounds (Sections 4.3-4.4, Lemma 6).

Direct paths (Section 4.4, "Granularity-Weighted Adequacy of Direct Paths")::

    r_hat_{a,e} = omega_q(e) * s_hat_{a,q}(e),   s_hat_{a,q}(e) = gamma_{a,l(e)} * rel_a(e),
    rel_a(e)    = max{0, x_a^T x_end(e)}.

* ``gamma_{a,l}`` comes from ``config.granularity_weights[role][level]``
  (gamma_{a,concept} = 0 for aspects that require empirical support); with
  ``use_role_aware_granularity = False`` one weight (1.0) is used for every
  level and role (Table 7 ablation).
* ``omega_q(e)`` is the junction weight (Section 4.3, "Pruning Observed
  Junctions"; Appendix A.3): for matched scopes U_q x M_q with grounding
  similarities, a paper p has omega_q(p) = max{g_u g_m : p in J(u, m)}, a
  chunk inherits the weight of its paper, and a concept v has the largest
  weight of a junction sharing papers with its extent (zero otherwise).

Complementary path sets (Eq. 9, "Premise-Bottleneck Score")::

    r_hat_{h,b} = eta_hat_q(b) * min{ s_hat_{P,q}(e_P), s_hat_{M,q}(e_M) },
    eta_hat_q(b) = sigmoid(s_theta(u_q, m)) * phi(b)    (discovery)
                 = phi(b)                               (assessment),

where the premise adequacies s_hat = gamma * rel carry no junction weight and
"additional transfer premises specified by the query enter the same minimum"
(several premise aspects on one side contribute their minimum). The bridge
discount ``phi(b) = bridge_decay ** (d - 1)`` decays with the taxonomic
distance d of the bridge anchors and is zero beyond ``max_bridge_distance``.

Region bounds (Section 4.3, Lemma 6)::

    rel_bar_a(v) = min{1, max{0, x_a^T xbar_v + rho_v}}

from the cached center/radius of the index nodes below v (``index.ball``,
``index.paper_chunk_balls``), computed in O(d |A_q(H)|).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np

from scholarpath.config import RetrievalConfig
from scholarpath.index.semantic_index import CHUNK, CONCEPT, LEVELS, PAPER, Ball, SemanticIndex
from scholarpath.retrieval.link_prediction import sigmoid
from scholarpath.retrieval.query_understanding import QueryAnalysis
from scholarpath.retrieval.units import DirectPath, key_id

# Numerical slack added to every cached upper bound so that float32 rounding in
# ball centers/radii can never make a bound fall below an exact score.
BOUND_SLACK = 1e-5


class Granularity:
    """gamma_{a, level} by evidential role (Section 4.4)."""

    def __init__(self, config: RetrievalConfig):
        self.role_aware = bool(config.use_role_aware_granularity)
        self.weights = {r: dict(w) for r, w in config.granularity_weights.items()}

    def gamma(self, role: str, level: str) -> float:
        if not self.role_aware:
            return 1.0
        table = self.weights.get(role) or self.weights.get("empirical", {})
        return float(table.get(level, 0.0))


def bridge_discount(distance: int, config: RetrievalConfig) -> float:
    """phi(b) = bridge_decay ** (d - 1) for 1 <= d <= max_bridge_distance, else 0."""
    if distance < 1 or distance > config.max_bridge_distance:
        return 0.0
    return float(config.bridge_decay) ** (distance - 1)


class RelevanceCache:
    """rel_a(x) = max{0, x_a^T x} for all aspects at once, computed once per index node.

    ``scored`` counts the distinct index nodes whose exact relevance was
    computed (the "Scored" column of Table 7).
    """

    def __init__(self, index: SemanticIndex, aspect_matrix: np.ndarray):
        self.index = index
        self.X = np.asarray(aspect_matrix, dtype=np.float32)
        self._rel: Dict[str, np.ndarray] = {}
        self.bound_evaluations = 0

    @property
    def scored(self) -> int:
        return len(self._rel)

    def rel(self, key: str) -> np.ndarray:
        r = self._rel.get(key)
        if r is None:
            r = np.maximum(0.0, self.X @ self.index.vector(key)).astype(np.float64)
            self._rel[key] = r
        return r

    def rel_bound(self, ball: Optional[Ball]) -> np.ndarray:
        """Cauchy-Schwarz bound rel_bar_a over the nodes summarized by ``ball`` (Lemma 6)."""
        self.bound_evaluations += 1
        if ball is None or ball.count == 0:
            return np.zeros(self.X.shape[0])
        raw = self.X @ ball.center + float(ball.radius) + BOUND_SLACK
        return np.minimum(1.0, np.maximum(0.0, raw)).astype(np.float64)


class JunctionTable:
    """Junction weights omega_q over weighted junction sets (Section 4.3, Appendix A.3).

    ``junctions`` is a list of ``(papers, weight)``; for evidence queries the
    sets are J(u, m) for (u, m) in U_q x M_q with weight g_u g_m. Pairs are
    processed in nonincreasing weight, so the first junction that exposes a
    paper carries its maximum weight (proof of Proposition 3).
    """

    def __init__(self, index: SemanticIndex, junctions: Sequence[Tuple[Set[str], float]]):
        self.index = index
        self.junctions = sorted(((set(p), float(w)) for p, w in junctions if p), key=lambda t: -t[1])
        self.paper_weight: Dict[str, float] = {}
        for papers, w in self.junctions:
            for p in papers:
                if p not in self.paper_weight:
                    self.paper_weight[p] = w
        self._extent_cache: Dict[Tuple[str, str], float] = {}
        self.intersections = 0

    @classmethod
    def from_pairs(
        cls, index: SemanticIndex, pairs: Sequence[Tuple[str, str, float]]
    ) -> "JunctionTable":
        """Junctions J(u, m) of grounded scope pairs ``(u, m, g_u * g_m)``."""
        sets = [(index.shared_papers(u, m), w) for u, m, w in pairs]
        return cls(index, sets)

    @property
    def findings(self) -> Dict[str, float]:
        """Papers with a positive junction weight (the established findings) -> omega."""
        return self.paper_weight

    def omega_paper(self, paper_id: str) -> float:
        return self.paper_weight.get(paper_id, 0.0)

    def omega_extent(self, facet: str, node_id: str) -> float:
        """Largest junction weight sharing a paper with extent(v) (one posting intersection per junction)."""
        key = (facet, node_id)
        hit = self._extent_cache.get(key)
        if hit is not None:
            return hit
        ext = self.index.taxonomy(facet).nodes[node_id].papers
        out = 0.0
        for papers, w in self.junctions:
            self.intersections += 1
            small, large = (ext, papers) if len(ext) <= len(papers) else (papers, ext)
            if any(p in large for p in small):
                out = w
                break
        self._extent_cache[key] = out
        return out


class AdequacyScorer:
    """Exact adequacies of units and admissible bounds of regions for one query."""

    def __init__(
        self,
        index: SemanticIndex,
        analysis: QueryAnalysis,
        config: RetrievalConfig,
        junction: Optional[JunctionTable] = None,
    ):
        self.index = index
        self.analysis = analysis
        self.config = config
        self.granularity = Granularity(config)
        self.roles = [a.role for a in analysis.aspects]
        X = np.stack([a.vector for a in analysis.aspects]) if analysis.aspects else np.zeros((0, index.dim))
        self.relevance = RelevanceCache(index, X)
        self.junction = junction
        self.evidence = [i for i, _ in analysis.evidence_aspects]
        self.premises = {
            "problem": [i for i, _ in analysis.premise_aspects("problem")],
            "method": [i for i, _ in analysis.premise_aspects("method")],
        }
        n = len(self.roles)
        self._gamma = {lvl: np.array([self.granularity.gamma(self.roles[i], lvl) for i in range(n)]) for lvl in LEVELS}

    # ------------------------------------------------------------ direct paths
    def gamma(self, level: str) -> np.ndarray:
        return self._gamma[level]

    def omega(self, path: DirectPath) -> float:
        """Junction weight omega_q(e) of a direct path (depends on end(e) only)."""
        if self.junction is None:
            return 1.0
        level = path.level
        if level == CONCEPT:
            return self.junction.omega_extent(path.facet, key_id(path.end))
        pid = path.paper_id
        return self.junction.omega_paper(pid) if pid else 0.0

    def direct_adequacy(self, path: DirectPath) -> np.ndarray:
        """r_hat_{a,e} = omega_q(e) gamma_{a,l(e)} rel_a(e) for every aspect a (vector)."""
        w = self.omega(path)
        if w <= 0.0:
            return np.zeros(len(self.roles))
        return w * self._gamma[path.level] * self.relevance.rel(path.end)

    def direct_bound(self, omega_bar: float, balls: Dict[str, Optional[Ball]]) -> np.ndarray:
        """r_bar_a(H) = omega_bar max_level gamma_{a,level} rel_bar_a(ball_level) (Lemma 6)."""
        out = np.zeros(len(self.roles))
        if omega_bar <= 0.0:
            return out
        for level, ball in balls.items():
            if ball is None or ball.count == 0:
                continue
            out = np.maximum(out, self._gamma[level] * self.relevance.rel_bound(ball))
        return omega_bar * out

    # ---------------------------------------------------------------- premises
    def premise_score(self, path: DirectPath, side: str) -> float:
        """s_hat_{side,q}(e) = min over the side's premise aspects of gamma rel (no junction weight)."""
        idx = self.premises[side]
        if not idx:
            return 0.0
        vals = self._gamma[path.level][idx] * self.relevance.rel(path.end)[idx]
        return float(vals.min())

    def premise_bound(self, balls: Dict[str, Optional[Ball]], side: str) -> float:
        """s_bar_{side,q}(H_side) from the balls of the witness subtree (Lemma 6)."""
        idx = self.premises[side]
        if not idx:
            return 0.0
        best = np.zeros(len(idx))
        for level, ball in balls.items():
            if ball is None or ball.count == 0:
                continue
            best = np.maximum(best, self._gamma[level][idx] * self.relevance.rel_bound(ball)[idx])
        return float(best.min())


def eta_factor(prob: Optional[float], phi: float) -> float:
    """eta_hat_q(b) = sigmoid(s_theta) * phi (discovery) or phi (assessment / no link prior)."""
    return float(phi if prob is None else prob * phi)


def hypothesis_adequacy(eta: float, s_problem: float, s_method: float) -> float:
    """Premise-bottleneck score of Eq. (9)."""
    return float(eta * min(s_problem, s_method))


def balls_below(index: SemanticIndex, node_id: str) -> Dict[str, Optional[Ball]]:
    """Cached balls of the concepts, abstracts and chunks below a taxonomy node."""
    return {lvl: index.balls.get((node_id, lvl)) for lvl in LEVELS}


def paper_balls(index: SemanticIndex, node_id: str) -> Dict[str, Optional[Ball]]:
    """Balls of the abstracts and chunks below a node (premise witnesses end at papers/chunks)."""
    return {PAPER: index.balls.get((node_id, PAPER)), CHUNK: index.balls.get((node_id, CHUNK))}


def chunk_ball(index: SemanticIndex, paper_id: str) -> Dict[str, Optional[Ball]]:
    return {CHUNK: index.paper_chunk_balls.get(paper_id)}


__all__ = [
    "AdequacyScorer",
    "BOUND_SLACK",
    "Granularity",
    "JunctionTable",
    "RelevanceCache",
    "balls_below",
    "bridge_discount",
    "chunk_ball",
    "eta_factor",
    "hypothesis_adequacy",
    "paper_balls",
    "sigmoid",
]
