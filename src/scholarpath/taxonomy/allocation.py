"""Batch allocation and sizing for quality-targeted acquisition.

Section 3.2 and Appendix A.1. In round t the builder holds, for every
region g of node v, a Thompson-sampled regional loss curve
l~_{vg,t}(n) = predicted loss of region g after the LLM reads n more of its
u_{vg,t} unread papers. For a batch size z it chooses an allocation
b = (b_1, ..., b_G) with sum_g b_g = z minimising

    Phi_{v,t}(b) = sum_g pi_{vg} l~_{vg,t}(b_g),

a unit-weight multiple-choice knapsack problem, solved exactly for every
z <= beta_{v,t} by the recurrence (Eq. 10)

    F_g(z) = min_{0 <= n <= min(z, u_g)} { F_{g-1}(z - n) + pi_g l~_g(n) },
    F_0(0) = 0,  F_0(z) = inf  (z > 0),

in O(G beta^2) time (Proposition 1). With F(z) = F_G(z) and
Z_feas = {z in 1..beta : F(z) <= eps}, the batch size is (Eq. 3)

    z_plan = min Z_feas                                   if Z_feas != {},
             argmax_{1 <= z <= beta} (F(0) - F(z)) / C(z)  otherwise,

where C(z) is the expected token cost of reading z papers; the allocation
b* is recovered by backtracking. Because sampled slopes may be negative,
F need not decrease in z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np

TokenCost = Union[float, int, Sequence[float], np.ndarray, Callable[[int], float]]


@dataclass
class BatchPlan:
    """Planned acquisition batch of one round.

    ``z`` is z_plan (0 when nothing can be read), ``allocation`` is b*
    (papers per region), ``predicted_loss`` is F(z_plan), ``F`` holds
    F(0..beta) (``inf`` for infeasible sizes) and ``feasible`` tells whether
    z_plan was taken from Z_feas (the forecast reaches eps).
    """

    z: int
    allocation: np.ndarray
    predicted_loss: float
    F: np.ndarray
    feasible: bool = False
    choice: Optional[np.ndarray] = field(default=None, repr=False)  # (G, beta+1) argmin table

    @property
    def current_loss(self) -> float:
        """F(0) = sum_g pi_g l~_g(0), the current loss estimate L^_{v,t}."""
        return float(self.F[0]) if self.F.size else 0.0

    def allocation_for(self, z: int) -> np.ndarray:
        """Backtrack the optimal allocation for any batch size z <= beta."""
        if self.choice is None:
            raise ValueError("BatchPlan has no backtracking table")
        return _backtrack(self.choice, int(z))


def expected_token_cost(
    per_paper_tokens: float, per_call_tokens: float = 0.0, papers_per_call: int = 16
) -> Callable[[int], float]:
    """C(z): tokens of reading z papers in calls of at most ``papers_per_call``.

    Every call carries the shared prompt context (node definition, schema and
    instructions, ``per_call_tokens``) once, and each paper adds its own
    title/abstract and output tokens (``per_paper_tokens``).
    """
    ppc = max(1, int(papers_per_call))

    def cost(z: int) -> float:
        z = int(z)
        if z <= 0:
            return 0.0
        return z * float(per_paper_tokens) + math.ceil(z / ppc) * float(per_call_tokens)

    return cost


def _cost_vector(token_cost: TokenCost, beta: int) -> np.ndarray:
    if callable(token_cost):
        return np.array([float(token_cost(z)) for z in range(beta + 1)], dtype=float)
    if np.isscalar(token_cost):
        return float(token_cost) * np.arange(beta + 1, dtype=float)
    arr = np.asarray(token_cost, dtype=float).ravel()
    if arr.size < beta + 1:
        raise ValueError(f"token_cost has {arr.size} entries, need C(0..{beta})")
    return arr[: beta + 1]


def _backtrack(choice: np.ndarray, z: int) -> np.ndarray:
    G = choice.shape[0]
    alloc = np.zeros(G, dtype=int)
    rem = int(z)
    for g in range(G - 1, -1, -1):
        n = int(choice[g, rem])
        alloc[g] = n
        rem -= n
    if rem != 0:  # pragma: no cover - guarded by feasibility of z
        raise ValueError(f"Batch size {z} is infeasible")
    return alloc


def allocation_table(
    curves: Sequence[np.ndarray], shares: Sequence[float], beta: int
) -> tuple:
    """Run the recurrence of Eq. (10); returns (F(0..beta), choice table).

    ``curves[g][n]`` is l~_g(n) for n = 0..u_g (so ``len(curves[g]) - 1`` is
    the number u_g of unread papers in region g).
    """
    beta = max(0, int(beta))
    G = len(curves)
    if len(shares) != G:
        raise ValueError("curves and shares must have the same length")
    F_prev = np.full(beta + 1, np.inf)
    F_prev[0] = 0.0
    choice = np.zeros((G, beta + 1), dtype=int)
    for g in range(G):
        curve = np.asarray(curves[g], dtype=float).ravel()
        if curve.size == 0:
            raise ValueError(f"curve of region {g} is empty (needs at least l(0))")
        u = curve.size - 1
        weighted = float(shares[g]) * curve
        F_cur = np.full(beta + 1, np.inf)
        for z in range(beta + 1):
            m = min(z, u)
            # n = 0..m  ->  F_prev[z - n] + pi_g l_g(n)
            cand = F_prev[z - np.arange(m + 1)] + weighted[: m + 1]
            k = int(np.argmin(cand))  # first minimiser: fewest papers on ties
            F_cur[z] = cand[k]
            choice[g, z] = k
        F_prev = F_cur
    if G == 0:
        F_prev = np.full(beta + 1, np.inf)
        F_prev[0] = 0.0
    return F_prev, choice


def plan_batch(
    curves: Sequence[np.ndarray],
    shares: Sequence[float],
    beta: int,
    epsilon: float,
    token_cost: TokenCost = 1.0,
) -> BatchPlan:
    """Exact batch allocation and sizing (Eq. 3 and Eq. 10, Proposition 1).

    Args:
        curves: per-region sampled loss curves; ``curves[g][n]`` is the
            predicted regional loss after reading n more papers of region g,
            for n = 0..u_g.
        shares: region shares pi_g = |G_g| / n_v.
        beta: beta_{v,t} = min{beta, n_v - |S_{v,t}|}, the largest batch.
        epsilon: node-level error tolerance.
        token_cost: C(z) as a per-paper constant, a sequence C(0..beta) or a
            callable z -> C(z) (see :func:`expected_token_cost`).

    Returns:
        :class:`BatchPlan` with z_plan, b*, F(z_plan) and the full F.
    """
    beta = max(0, int(beta))
    F, choice = allocation_table(curves, shares, beta)
    G = len(curves)
    sizes = np.arange(beta + 1)
    finite = np.isfinite(F)
    candidates = sizes[1:][finite[1:]]
    if beta == 0 or candidates.size == 0:
        return BatchPlan(0, np.zeros(G, dtype=int), float(F[0]), F, False, choice)
    feas = candidates[F[candidates] <= epsilon]
    if feas.size:
        z = int(feas.min())
        feasible = True
    else:
        C = _cost_vector(token_cost, beta)
        ratio = np.full(beta + 1, -np.inf)
        c = np.maximum(C[candidates], 1e-12)
        ratio[candidates] = (F[0] - F[candidates]) / c
        z = int(np.argmax(ratio))  # first maximiser: smallest z on ties
        feasible = False
    return BatchPlan(z, _backtrack(choice, z), float(F[z]), F, feasible, choice)
