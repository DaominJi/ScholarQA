"""Tests for audit-based certification (Section 3.3, Eq. 4) and the delta split."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import hypergeom

from scholarpath.taxonomy.certification import (
    EXHAUSTED,
    IMPLAUSIBLE,
    IMPOSSIBLE,
    DeltaAllocator,
    audit_snapshot,
    certificate_bound,
    hypergeom_tail,
    hypergeom_tail_scipy,
    hypergeom_upper_bound,
    step_alpha,
)


def _exact_tail(y: int, R: int, K: int, a: int) -> Fraction:
    num = sum(math.comb(K, i) * math.comb(R - K, a - i) for i in range(0, min(y, K, a) + 1) if a - i <= R - K)
    return Fraction(num, math.comb(R, a))


def _brute_upper(R: int, a: int, y: int, alpha: float) -> int:
    """Largest non-rejected K, with the tail computed in exact rational arithmetic."""
    ok = [K for K in range(y, R - (a - y) + 1) if _exact_tail(y, R, K, a) > Fraction(alpha)]
    return max(ok)


# ---------------------------------------------------------------- K^U bound


def test_tail_matches_scipy():
    rng = np.random.default_rng(5)
    for _ in range(500):
        R = int(rng.integers(1, 2000))
        a, K = int(rng.integers(0, R + 1)), int(rng.integers(0, R + 1))
        y = int(rng.integers(0, min(a, 40) + 1))
        assert hypergeom_tail(y, R, K, a) == pytest.approx(hypergeom_tail_scipy(y, R, K, a), rel=1e-9, abs=1e-12)


def test_upper_bound_edge_cases():
    assert hypergeom_upper_bound(0, 0, 0, 0.05) == 0
    # a = 0: no information, every unverified paper may be wrong (paper: R = 1 -> K^U = 1).
    assert hypergeom_upper_bound(1, 0, 0, 0.05) == 1
    assert hypergeom_upper_bound(37, 0, 0, 0.05) == 37
    # a = R: exhaustive inspection gives the exact count.
    for y in (0, 3, 10):
        assert hypergeom_upper_bound(10, 10, y, 0.01) == y
    # alpha = 0 gives the deterministic bound R - (a - y).
    assert hypergeom_upper_bound(50, 10, 2, 0.0) == 42
    with pytest.raises(ValueError):
        hypergeom_upper_bound(5, 6, 0, 0.1)
    with pytest.raises(ValueError):
        hypergeom_upper_bound(5, 2, 3, 0.1)


def test_upper_bound_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        R = int(rng.integers(1, 40))
        a = int(rng.integers(0, R + 1))
        y = int(rng.integers(0, a + 1))
        alpha = float(rng.uniform(1e-4, 0.3))
        expect = R if a == 0 else _brute_upper(R, a, y, alpha)
        assert hypergeom_upper_bound(R, a, y, alpha) == expect


def test_upper_bound_monotonicity():
    R, alpha = 80, 0.05
    # Non-decreasing in the observed errors y (fixed a).
    vals = [hypergeom_upper_bound(R, 20, y, alpha) for y in range(0, 21)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    # Non-increasing in the number of clean draws (y = 0).
    vals = [hypergeom_upper_bound(R, a, 0, alpha) for a in range(0, R + 1)]
    assert all(b <= a for a, b in zip(vals, vals[1:]))
    # Smaller alpha (higher confidence) never gives a smaller bound.
    vals = [hypergeom_upper_bound(R, 25, 3, al) for al in (0.2, 0.1, 0.05, 0.01, 0.001)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("R,a,alpha", [(60, 15, 0.1), (200, 40, 0.05), (30, 29, 0.2)])
def test_upper_bound_exact_coverage(R, a, alpha):
    """Pr(K <= K^U) >= 1 - alpha for every K, computed exactly from the pmf."""
    ys = np.arange(0, a + 1)
    for K in range(0, R + 1, max(1, R // 15)):
        pmf = hypergeom.pmf(ys, R, K, a)
        covered = sum(p for y, p in zip(ys, pmf) if p > 0 and hypergeom_upper_bound(R, a, int(y), alpha) >= K)
        assert covered >= 1 - alpha - 1e-9, (K, covered)


def test_upper_bound_coverage_by_simulation():
    rng = np.random.default_rng(1)
    R, a, alpha, trials = 120, 25, 0.1, 2000
    for K in (0, 6, 12, 30, 60):
        pop = np.zeros(R, dtype=int)
        pop[:K] = 1
        hits = 0
        for _ in range(trials):
            y = int(rng.permutation(pop)[:a].sum())
            hits += hypergeom_upper_bound(R, a, y, alpha) >= K
        # 1 - alpha minus three Monte-Carlo standard errors
        assert hits / trials >= 1 - alpha - 3 * math.sqrt(alpha * (1 - alpha) / trials)


# ---------------------------------------------------------------- alpha / U


def test_step_alpha_sums_to_delta():
    delta_v = 0.01
    total = sum(step_alpha(delta_v, j, a) for j in range(1, 300) for a in range(1, 300))
    assert total <= delta_v
    assert total == pytest.approx(delta_v * (1 - 1 / 300) ** 2, rel=1e-9)
    assert step_alpha(0.06, 1, 1) == pytest.approx(0.06 / 4)
    with pytest.raises(ValueError):
        step_alpha(0.1, 0, 1)
    with pytest.raises(ValueError):
        step_alpha(0.1, 1, 0)


def test_certificate_bound_values():
    # a = 0: all R unverified papers count as errors.
    assert certificate_bound(100, 3, 10, 0, 0, 0.01, 1) == pytest.approx(0.13)
    # R = 0: exact loss.
    assert certificate_bound(100, 7, 0, 0, 0, 0.01, 1) == pytest.approx(0.07)
    # Generic: (E_D + K^U) / n_v with alpha_{v,j,a}.
    k_up = hypergeom_upper_bound(80, 30, 1, step_alpha(0.02, 2, 30))
    assert certificate_bound(100, 4, 80, 30, 1, 0.02, 2) == pytest.approx((4 + k_up) / 100)


# ------------------------------------------------------------- allocation


@pytest.mark.parametrize("strategy", DeltaAllocator.STRATEGIES)
def test_delta_allocator_never_exceeds_delta(strategy):
    rng = np.random.default_rng(2)
    alloc = DeltaAllocator(0.05, strategy, n_total=1000, max_depth=3, max_nodes=40)
    # Multi-label extents: node sizes sum to far more than n_total.
    for _ in range(500):
        dv = alloc.allocate(int(rng.integers(16, 1000)))
        assert dv >= 0.0
    assert math.fsum(alloc.allocations) <= 0.05
    assert alloc.spent <= alloc.delta and alloc.remaining >= 0.0
    restored = DeltaAllocator.from_state(alloc.state_dict())
    assert restored.spent == alloc.spent and restored.allocate(100) <= restored.delta - alloc.spent


def test_delta_allocator_strategies():
    a = DeltaAllocator(0.05, "spend", n_total=1000, max_depth=5)
    assert a.allocate(1000) == pytest.approx(0.05 / 5)
    h = DeltaAllocator(0.05, "harmonic")
    assert h.allocate(10) == pytest.approx(0.05 * 6 / math.pi ** 2)
    assert h.allocate(10) == pytest.approx(0.05 * 6 / (4 * math.pi ** 2))
    u = DeltaAllocator(0.05, "uniform", max_nodes=4)
    vals = [u.allocate(10) for _ in range(6)]
    assert vals[0] == pytest.approx(0.0125) and vals[-1] == 0.0
    with pytest.raises(ValueError):
        DeltaAllocator(0.05, "nope")


# ------------------------------------------------------------------ audit


def _judge_for(errors):
    def judge(batch):
        return {p: p in errors for p in batch}
    return judge


def test_audit_certifies_clean_population():
    pool = [f"p{i}" for i in range(400)]
    errors = set(pool[:8])  # 2% errors among the unverified
    out = audit_snapshot(pool, _judge_for(errors), n_v=500, E_D=5, delta_v=0.01, j=1, epsilon=0.1,
                         batch_size=8, rng=np.random.default_rng(0))
    assert out.certified and out.bound <= 0.1
    assert 0 < out.draws < 400 and len(out.audited) >= out.draws
    assert all(out.audited[p] == (p in errors) for p in out.audited)
    # The reported bound is Eq. (4) at the reported (a, y).
    assert out.bound == pytest.approx(certificate_bound(500, 5, 400, out.draws, out.errors, 0.01, 1))


def test_audit_stops_early_when_certification_implausible_or_impossible():
    pool = [f"p{i}" for i in range(300)]
    errors = set(pool[:120])  # 40% errors
    out = audit_snapshot(pool, _judge_for(errors), n_v=300, E_D=0, delta_v=0.01, j=1, epsilon=0.1,
                         batch_size=8, rng=np.random.default_rng(0), abort_factor=1.5, min_draws=8)
    assert not out.certified and out.stop_reason in (IMPLAUSIBLE, IMPOSSIBLE)
    assert out.draws < 40
    # Known errors alone already exceed eps: no draw at all.
    out = audit_snapshot(pool, _judge_for(set()), n_v=300, E_D=40, delta_v=0.01, j=1, epsilon=0.1,
                         batch_size=8, rng=np.random.default_rng(0))
    assert not out.certified and out.draws == 0 and out.stop_reason == IMPOSSIBLE


def test_audit_exhaustion_gives_exact_loss_and_unjudged_count_as_errors():
    pool = [f"p{i}" for i in range(12)]
    out = audit_snapshot(pool, _judge_for({"p3"}), n_v=40, E_D=1, delta_v=1e-6, j=3, epsilon=0.06,
                         batch_size=5, rng=np.random.default_rng(0))
    assert out.certified and out.stop_reason == EXHAUSTED and out.draws == 12
    assert out.bound == pytest.approx(2 / 40)
    # A judge that returns nothing: every drawn paper counts as an error.
    out = audit_snapshot(pool, lambda batch: {}, n_v=40, E_D=0, delta_v=0.01, j=1, epsilon=0.1,
                         batch_size=4, rng=np.random.default_rng(0))
    assert not out.certified and out.errors == out.draws > 0 and set(out.unjudged) <= set(pool)
    # Nothing unverified and no known errors: certified without drawing.
    out = audit_snapshot([], _judge_for(set()), n_v=40, E_D=0, delta_v=0.01, j=1, epsilon=0.1,
                         batch_size=4, rng=np.random.default_rng(0))
    assert out.certified and out.bound == 0.0 and out.draws == 0


def test_sequential_audit_false_certification_rate_below_delta():
    """Pr(certified while L_v > eps) <= delta_v despite sequential stopping."""
    rng = np.random.default_rng(3)
    n_v, eps, delta_v = 200, 0.1, 0.2
    pool = [f"p{i}" for i in range(n_v)]
    errors = set(pool[:23])  # L_v = 0.115 > eps
    trials, false_cert = 150, 0
    for _ in range(trials):
        out = audit_snapshot(pool, _judge_for(errors), n_v, 0, delta_v, 1, eps, 8, rng,
                             abort_factor=10.0, min_draws=1000)
        false_cert += out.certified
    assert false_cert / trials <= delta_v
