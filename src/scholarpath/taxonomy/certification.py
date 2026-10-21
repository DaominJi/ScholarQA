"""Audit-based certification of a node (Section 3.3, Equation 4).

After the repair of a cycle, the builder freezes the schema, the router
f_theta_v and the paper assignments of node v as *snapshot* j. For a paper p,
Z_{v,j}(p) = 1 if p has a missing concept or a routing error in snapshot j,
as judged by the LLM at the granularity of v. The verified papers D_{v,j}
(already read by the LLM) hold E_{D,v,j} known errors; the R = n_v - |D_{v,j}|
unverified papers R_{v,j} hold K_{v,j} unknown errors. Auditing ``a`` papers
drawn uniformly *without replacement* from R_{v,j} yields an error count

    Y_a ~ Hypergeom(R, K_{v,j}, a)

(conditional on the snapshot and the history). Inverting the hypergeometric
tail at the step level

    alpha_{v,j,a} = delta_v / [ j (j + 1) a (a + 1) ]

gives a (1 - alpha_{v,j,a}) upper confidence bound K^U_{v,j,a} on K_{v,j} and

    U_{v,j,a} = (E_{D,v,j} + K^U_{v,j,a}) / n_v                    (Eq. 4)

on the node-wise organization loss L_v. Because
sum_{j, a >= 1} alpha_{v,j,a} = delta_v, a union bound over every snapshot and
every audit step shows Pr(v certified while L_v > eps) <= delta_v, for *any*
data-dependent stopping rule; :class:`DeltaAllocator` splits the global delta
over nodes so that sum_v delta_v <= delta (Definition 1).

The module only depends on NumPy and SciPy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import hypergeom

# A judge maps a batch of audited paper ids to their error indicators Z(p).
Judge = Callable[[List[str]], Mapping[str, bool]]

#: Safety factor applied when capping an allocation by the remaining budget, so
#: that floating-point accumulation can never push sum_v delta_v above delta.
_BUDGET_SAFETY = 1.0 - 1e-9


# ---------------------------------------------------------------------------
# Hypergeometric upper confidence bound (Section 3.3)
# ---------------------------------------------------------------------------


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def hypergeom_tail(y: int, R: int, K: int, a: int) -> float:
    """P(Y <= y) for Y ~ Hypergeom(population R, K errors, a draws).

    Exact sum of the pmf terms C(K, i) C(R - K, a - i) / C(R, a) for
    i = max(0, a - R + K), ..., y, evaluated in log space (equal to
    ``scipy.stats.hypergeom.cdf(y, R, K, a)`` but without its per-call
    overhead, which matters because the audit evaluates the bound after
    every drawn paper).
    """
    y, R, K, a = int(y), int(R), int(K), int(a)
    lo = max(0, a - (R - K))
    hi = min(y, K, a)
    if hi < lo:
        return 0.0
    if hi >= min(K, a):
        return 1.0
    log_den = _log_comb(R, a)
    terms = [_log_comb(K, i) + _log_comb(R - K, a - i) - log_den for i in range(lo, hi + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * math.fsum(math.exp(t - top) for t in terms))


def hypergeom_tail_scipy(y: int, R: int, K: int, a: int) -> float:
    """Reference implementation through :mod:`scipy.stats`."""
    return float(hypergeom.cdf(int(y), int(R), int(K), int(a)))


def hypergeom_upper_bound(R: int, a: int, y: int, alpha: float) -> int:
    """Upper confidence bound K^U_{v,j,a} on the error count of R unverified papers.

    ``a`` papers were drawn uniformly without replacement from the ``R``
    unverified papers and ``y`` of them are errors. K^U is the largest
    K in {y, ..., R - (a - y)} (the values consistent with the sample) that
    is *not rejected* at level ``alpha`` by the one-sided test
    P(Y <= y | R, K, a) <= alpha. Since this tail probability is
    non-increasing in K, the non-rejected values form an interval starting at
    y and the bound is found by binary search.

    Validity: for the true K, {K^U < K} implies P(Y <= Y_obs | K) <= alpha,
    an event of probability at most alpha, so Pr(K <= K^U) >= 1 - alpha.

    Edge cases: ``R = 0`` gives 0; ``a = 0`` gives R (no information; e.g.
    R = 1 gives K^U_{v,j,0} = 1, so the last paper cannot be certified without
    being read, Section 3.3); ``a = R`` gives y (exhaustive inspection).
    ``alpha <= 0`` returns the deterministic bound R - (a - y).
    """
    R, a, y = int(R), int(a), int(y)
    if R < 0 or a < 0 or y < 0:
        raise ValueError(f"R, a, y must be non-negative (got R={R}, a={a}, y={y})")
    if a > R:
        raise ValueError(f"Cannot draw a={a} papers from R={R} without replacement")
    if y > a:
        raise ValueError(f"Observed errors y={y} exceed draws a={a}")
    if R == 0:
        return 0
    lo, hi = y, R - (a - y)
    if a == 0 or lo == hi:
        return hi
    if not alpha > 0.0:
        return hi
    if alpha >= 1.0:
        return lo
    if hypergeom_tail(y, R, hi, a) > alpha:
        return hi
    # Invariant: tail(lo) > alpha (tail(y) = 1), tail(hi) <= alpha.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if hypergeom_tail(y, R, mid, a) > alpha:
            lo = mid
        else:
            hi = mid
    return lo


def step_alpha(delta_v: float, j: int, a: int) -> float:
    """Step error level alpha_{v,j,a} = delta_v / [j (j+1) a (a+1)] (Section 3.3).

    ``j >= 1`` is the snapshot index and ``a >= 1`` the number of audited
    papers. Since sum_{j>=1} 1/(j(j+1)) = sum_{a>=1} 1/(a(a+1)) = 1, the levels
    sum to delta_v over all snapshots and audit steps.
    """
    if j < 1 or a < 1:
        raise ValueError(f"step_alpha needs j >= 1 and a >= 1 (got j={j}, a={a})")
    return float(delta_v) / (j * (j + 1.0) * a * (a + 1.0))


def certificate_bound(
    n_v: int, E_D: float, R: int, a: int, y: int, delta_v: float, j: int
) -> float:
    """U_{v,j,a} = (E_{D,v,j} + K^U_{v,j,a}) / n_v (Equation 4).

    With ``a = 0`` no unverified paper has been audited and every unverified
    paper is counted as a potential error (K^U = R); with ``R = 0`` the loss
    E_D / n_v is known exactly.
    """
    if n_v <= 0:
        raise ValueError("n_v must be positive")
    if a <= 0:
        k_upper = int(R)
    else:
        k_upper = hypergeom_upper_bound(R, a, y, step_alpha(delta_v, j, a))
    return (float(E_D) + k_upper) / float(n_v)


# ---------------------------------------------------------------------------
# Splitting delta across nodes (Definition 1: sum_v delta_v <= delta)
# ---------------------------------------------------------------------------


class DeltaAllocator:
    """Splits the taxonomy-wide failure probability delta over node episodes.

    Every episode calls :meth:`allocate` once (at InitNode) and receives
    delta_v. Spending is tracked and every allocation is capped by the
    remaining budget, so ``sum_v delta_v <= delta`` holds by construction
    regardless of how many nodes are created or how papers overlap between
    siblings (multi-label assignments). Strategies
    (``BuilderConfig.delta_allocation``):

    * ``"spend"``    -- delta_v = min(delta * n_v / (N * D), remaining / 2):
      proportional to the node size, where N is the repository size and D the
      maximum depth (the nodes of one level hold about N papers, and there are
      at most D levels of internal nodes);
    * ``"harmonic"`` -- delta_v = delta * 6 / (pi^2 k^2) for the k-th episode
      (sum_k 6 / (pi^2 k^2) = 1);
    * ``"uniform"``  -- delta_v = delta / max_nodes (zero once ``max_nodes``
      episodes have been allocated).
    """

    STRATEGIES = ("spend", "harmonic", "uniform")

    def __init__(
        self,
        delta: float,
        strategy: str = "spend",
        n_total: int = 1,
        max_depth: int = 5,
        max_nodes: int = 512,
    ):
        if not 0.0 < float(delta) < 1.0:
            raise ValueError(f"delta must lie in (0, 1), got {delta}")
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown delta allocation strategy {strategy!r}; use one of {self.STRATEGIES}")
        self.delta = float(delta)
        self.strategy = strategy
        self.n_total = max(1, int(n_total))
        self.max_depth = max(1, int(max_depth))
        self.max_nodes = max(1, int(max_nodes))
        self.allocations: List[float] = []

    @property
    def spent(self) -> float:
        return math.fsum(self.allocations)

    @property
    def remaining(self) -> float:
        return max(0.0, self.delta - self.spent)

    @property
    def count(self) -> int:
        return len(self.allocations)

    def allocate(self, n_v: int) -> float:
        """Allocate delta_v to the next node episode (of size ``n_v``)."""
        remaining = self.remaining
        k = self.count + 1
        if self.strategy == "spend":
            want = self.delta * float(n_v) / (self.n_total * self.max_depth)
            delta_v = min(want, 0.5 * remaining)
        elif self.strategy == "harmonic":
            want = self.delta * 6.0 / (math.pi ** 2 * k * k)
            delta_v = min(want, remaining * _BUDGET_SAFETY)
        else:  # uniform
            want = self.delta / self.max_nodes if k <= self.max_nodes else 0.0
            delta_v = min(want, remaining * _BUDGET_SAFETY)
        delta_v = max(0.0, float(delta_v))
        self.allocations.append(delta_v)
        return delta_v

    # Persistence (dynamic updates continue the same budget).
    def state_dict(self) -> Dict[str, Any]:
        return {
            "delta": self.delta,
            "strategy": self.strategy,
            "n_total": self.n_total,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "allocations": list(self.allocations),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DeltaAllocator":
        alloc = cls(
            float(state["delta"]),
            str(state.get("strategy", "spend")),
            int(state.get("n_total", 1)),
            int(state.get("max_depth", 5)),
            int(state.get("max_nodes", 512)),
        )
        alloc.allocations = [float(x) for x in state.get("allocations", [])]
        return alloc


# ---------------------------------------------------------------------------
# Stage 3: auditing one snapshot (Algorithm 1, lines 15-20)
# ---------------------------------------------------------------------------

# Reasons an audit stops.
CERTIFIED_STOP = "certified"
IMPOSSIBLE = "impossible"  # (E_D + y) / n_v > eps: no further draw can certify
IMPLAUSIBLE = "implausible"  # point estimate > abort_factor * eps after min_draws
EXHAUSTED = "exhausted"  # R_{v,j} = empty: exact loss known
BUDGET = "budget"  # max_draws reached


@dataclass
class AuditOutcome:
    """Result of auditing snapshot j of node v.

    ``audited`` maps every paper the judge read to its error indicator Z(p);
    papers the judge returned no verdict for are counted as errors
    (conservative) and listed in ``unjudged``. ``bound`` is U_{v,j,a} at
    ``draws`` = a with ``errors`` = y: the last step for an uncertified
    audit; for a certified one, the tightest U_{v,j,a} <= eps over the steps
    of the batch in which certification happened (that batch was judged in
    one LLM call, and all U_{v,j,a} hold simultaneously).
    """

    certified: bool
    bound: float
    audited: Dict[str, bool] = field(default_factory=dict)
    draws: int = 0
    errors: int = 0
    n_v: int = 0
    E_D: float = 0.0
    R: int = 0
    delta_v: float = 0.0
    snapshot: int = 1
    alpha: Optional[float] = None
    k_upper: int = 0
    stop_reason: str = ""
    unjudged: List[str] = field(default_factory=list)
    trace: List[Tuple[int, int, float]] = field(default_factory=list)  # (a, y, U_{v,j,a})

    @property
    def exhausted(self) -> bool:
        return self.stop_reason == EXHAUSTED

    @property
    def found_errors(self) -> bool:
        return self.errors > 0 or any(self.audited.values())

    @property
    def point_estimate(self) -> float:
        """Unbiased estimate (E_D + y R / a) / n_v of L_v for the snapshot."""
        if self.n_v <= 0:
            return 0.0
        if self.draws <= 0:
            return float(self.E_D) / self.n_v
        return (float(self.E_D) + self.errors * self.R / self.draws) / self.n_v

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["audited"] = {k: bool(v) for k, v in self.audited.items()}
        d["trace"] = [list(t) for t in self.trace]
        return d

    def summary(self) -> Dict[str, Any]:
        """Compact JSON-friendly record (no per-paper data)."""
        return {
            "certified": self.certified,
            "bound": self.bound,
            "n_v": self.n_v,
            "E_D": self.E_D,
            "R": self.R,
            "draws": self.draws,
            "errors": self.errors,
            "k_upper": self.k_upper,
            "alpha": self.alpha,
            "delta_v": self.delta_v,
            "snapshot": self.snapshot,
            "stop_reason": self.stop_reason,
            "n_audited": len(self.audited),
        }


def audit_snapshot(
    unverified: Sequence[str],
    judge: Judge,
    n_v: int,
    E_D: float,
    delta_v: float,
    j: int,
    epsilon: float,
    batch_size: int,
    rng: np.random.Generator,
    abort_factor: float = 1.5,
    min_draws: int = 8,
    max_draws: Optional[int] = None,
    min_certify_draws: int = 0,
) -> AuditOutcome:
    """Audit snapshot j of node v (Algorithm 1, lines 15-20).

    Draws uniform batches of ``batch_size`` papers without replacement from
    the unverified set R_{v,j} (a uniformly random permutation consumed in
    order), asks ``judge`` for their error indicators Z_{v,j}(p), and after
    every paper updates (a, y) and U_{v,j,a} (Eq. 4). The node is certified
    as soon as U_{v,j,a} <= epsilon. The audit stops without certification

    * when certification has become impossible, (E_D + y) / n_v > epsilon
      (U >= (E_D + y) / n_v for every later a);
    * when it is implausible: after ``min_draws`` papers the point estimate
      (E_D + y + (R - a) y / a) / n_v exceeds ``abort_factor * epsilon``;
    * when R_{v,j} is exhausted (the exact loss (E_D + y) / n_v is known);
    * when ``max_draws`` papers have been audited (budget cap).

    ``min_certify_draws`` forbids certifying before min(R, min_certify_draws)
    papers have been judged. The dynamic update (Section 3.4) uses it: once a
    keep condition fails, "the builder audits the new cohort", so the cohort
    audit must look at real papers even when U_{v,j,0} <= eps.

    The decision rule may depend on everything observed so far: the union
    bound over a (and over snapshots j, by the caller continuing j) keeps the
    certificate valid at level delta_v. That includes a minimum number of
    draws. Once certified, the remaining papers of the same (already judged)
    batch are still folded into (a, y) and the smallest bound is reported.
    """
    if n_v <= 0:
        raise ValueError("n_v must be positive")
    pool = sorted(set(unverified))
    R = len(pool)
    out = AuditOutcome(certified=False, bound=1.0, n_v=int(n_v), E_D=float(E_D), R=R,
                       delta_v=float(delta_v), snapshot=int(j))

    def bound_at(a: int, y: int) -> Tuple[float, int, Optional[float]]:
        if a <= 0:
            return (float(E_D) + R) / n_v, R, None
        alpha = step_alpha(delta_v, j, a)
        k_up = hypergeom_upper_bound(R, a, y, alpha)
        return (float(E_D) + k_up) / n_v, k_up, alpha

    # a = 0: the bound counts every unverified paper as an error.
    a_min = min(R, max(0, int(min_certify_draws)))
    out.bound, out.k_upper, out.alpha = bound_at(0, 0)
    out.trace.append((0, 0, out.bound))
    if out.bound <= epsilon and a_min == 0:
        out.certified, out.stop_reason = True, (EXHAUSTED if R == 0 else CERTIFIED_STOP)
        return out
    if float(E_D) / n_v > epsilon:
        out.stop_reason = IMPOSSIBLE if R > 0 else EXHAUSTED
        return out
    if R == 0:
        out.stop_reason = EXHAUSTED
        return out

    order = [pool[i] for i in rng.permutation(R)]
    cap = R if max_draws is None else max(0, min(R, int(max_draws)))
    a = y = 0
    pos = 0
    batch_size = max(1, int(batch_size))
    while pos < cap:
        batch = order[pos : min(pos + batch_size, cap)]
        pos += len(batch)
        verdicts = dict(judge(list(batch)) or {})
        for pid in batch:
            if pid in verdicts:
                z = bool(verdicts[pid])
            else:
                z = True  # no verdict: counted as an error (conservative)
                out.unjudged.append(pid)
            out.audited[pid] = z
            a += 1
            y += int(z)
            U, k_up, alpha = bound_at(a, y)
            out.trace.append((a, y, U))
            if out.certified:
                # The rest of an already-judged batch: every U_{v,j,a} holds
                # simultaneously (union bound over a), so keep the tightest.
                if U < out.bound:
                    out.draws, out.errors, out.bound, out.k_upper, out.alpha = a, y, U, k_up, alpha
                    if a == R:
                        out.stop_reason = EXHAUSTED
                continue
            out.draws, out.errors, out.bound, out.k_upper, out.alpha = a, y, U, k_up, alpha
            if U <= epsilon and a >= a_min:
                out.certified = True
                out.stop_reason = EXHAUSTED if a == R else CERTIFIED_STOP
        if out.certified:
            return out
        if a >= R:
            out.stop_reason = EXHAUSTED
            return out
        if (float(E_D) + y) / n_v > epsilon:
            out.stop_reason = IMPOSSIBLE
            return out
        if a >= min_draws and (float(E_D) + y + (R - a) * y / a) / n_v > abort_factor * epsilon:
            out.stop_reason = IMPLAUSIBLE
            return out
    out.stop_reason = BUDGET
    return out
