"""Tests for batch allocation / sizing (Eq. 3, Eq. 10, Prop. 1) and the response model (Eq. 13-14)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from scholarpath.taxonomy.allocation import BatchPlan, expected_token_cost, plan_batch
from scholarpath.taxonomy.response_model import (
    RegionDescriptor,
    RegionResponse,
    TransferredPrior,
    logit,
    region_dispersion,
    response_curve,
    sigmoid,
)


def _brute_force(curves, shares, beta):
    """min_b sum_g pi_g l_g(b_g) s.t. sum_g b_g = z, 0 <= b_g <= u_g, for every z."""
    best = np.full(beta + 1, np.inf)
    for b in itertools.product(*[range(len(c)) for c in curves]):
        z = sum(b)
        if z <= beta:
            val = sum(s * c[n] for s, c, n in zip(shares, curves, b))
            best[z] = min(best[z], val)
    return best


def _random_instance(rng, G_max=4, u_max=5):
    G = int(rng.integers(1, G_max + 1))
    curves = []
    for _ in range(G):
        u = int(rng.integers(0, u_max + 1))
        # Arbitrary (also non-monotone) curves: sampled slopes may be negative.
        curves.append(rng.uniform(0.0, 1.0, size=u + 1))
    shares = rng.dirichlet(np.ones(G))
    return curves, shares


def test_dp_is_exact_against_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(150):
        curves, shares = _random_instance(rng)
        beta = int(rng.integers(0, 12))
        plan = plan_batch(curves, shares, beta, epsilon=0.1, token_cost=1.0)
        brute = _brute_force(curves, shares, beta)
        finite = np.isfinite(brute)
        assert np.allclose(plan.F[finite], brute[finite])
        assert np.all(np.isinf(plan.F[~finite]))
        # Every backtracked allocation is feasible and attains F(z).
        for z in np.flatnonzero(finite):
            b = plan.allocation_for(int(z))
            assert b.sum() == z
            assert all(0 <= n <= len(c) - 1 for n, c in zip(b, curves))
            assert sum(s * c[n] for s, c, n in zip(shares, curves, b)) == pytest.approx(plan.F[z])


def test_batch_size_rule_eq3():
    rng = np.random.default_rng(1)
    for _ in range(150):
        curves, shares = _random_instance(rng)
        beta = int(rng.integers(1, 10))
        eps = float(rng.uniform(0.05, 0.6))
        cost = expected_token_cost(300.0, 500.0, 4)
        plan = plan_batch(curves, shares, beta, eps, cost)
        F = _brute_force(curves, shares, beta)
        sizes = [z for z in range(1, beta + 1) if np.isfinite(F[z])]
        if not sizes:
            assert plan.z == 0 and plan.allocation.sum() == 0
            continue
        feas = [z for z in sizes if F[z] <= eps]
        if feas:
            assert plan.feasible and plan.z == min(feas)
        else:
            ratios = {z: (F[0] - F[z]) / cost(z) for z in sizes}
            best = max(ratios.values())
            assert not plan.feasible
            assert ratios[plan.z] == pytest.approx(best)
            assert plan.z == min(z for z in sizes if np.isclose(ratios[z], best))
        assert plan.allocation.sum() == plan.z
        assert plan.predicted_loss == pytest.approx(F[plan.z])
        assert plan.current_loss == pytest.approx(F[0])


def test_plan_batch_edge_cases_and_cost_forms():
    curves = [np.array([0.5, 0.3, 0.2]), np.array([0.4, 0.1])]
    shares = [0.5, 0.5]
    plan = plan_batch(curves, shares, 0, 0.1)
    assert plan.z == 0 and plan.F.shape == (1,)
    # beta beyond the unread papers: infeasible sizes are +inf.
    plan = plan_batch(curves, shares, 5, 0.0)
    assert np.isinf(plan.F[4:]).all() and plan.z <= 3
    # Equivalent cost specifications give the same plan.
    p1 = plan_batch(curves, shares, 3, 0.0, 2.0)
    p2 = plan_batch(curves, shares, 3, 0.0, [0.0, 2.0, 4.0, 6.0])
    p3 = plan_batch(curves, shares, 3, 0.0, lambda z: 2.0 * z)
    assert p1.z == p2.z == p3.z and np.array_equal(p1.allocation, p3.allocation)
    # Feasible: the smallest batch predicted to reach eps.
    plan = plan_batch(curves, shares, 3, 0.30)  # F = [0.45, 0.30, 0.20, 0.15]
    assert plan.feasible and plan.z == 1 and list(plan.allocation) == [0, 1]
    plan = plan_batch(curves, shares, 3, 0.26)
    assert plan.feasible and plan.z == 2 and list(plan.allocation) == [1, 1]
    assert isinstance(plan, BatchPlan)


def test_expected_token_cost():
    c = expected_token_cost(100.0, 1000.0, papers_per_call=16)
    assert c(0) == 0.0 and c(1) == 1100.0 and c(16) == 2600.0 and c(17) == 3700.0


# --------------------------------------------------------- response model


def test_response_curve_shape():
    curve = response_curve(0.3, 0.2, [0.1, 0.2], 20)
    assert curve[0] == pytest.approx(0.3 + 0.7 * 0.2)
    assert np.all(np.diff(curve) < 0)  # positive slopes: loss decreases
    rising = response_curve(0.3, 0.2, [-0.1, 0.0], 5)
    assert np.all(np.diff(rising) > 0)  # negative slopes allowed (schema revisions)
    assert logit(sigmoid(1.3)) == pytest.approx(1.3)


def test_region_response_conjugate_update_and_thompson():
    resp = RegionResponse([0.05, 0.05], 0.05 ** 2 * np.eye(2), obs_noise_var=0.02 ** 2, m_hat=0.3, e_hat=0.2)
    rng = np.random.default_rng(0)
    c = resp.thompson_curve(10, rng)
    assert c.shape == (11,) and c[0] == pytest.approx(resp.current_loss)
    # Observed slopes of 0.3 per paper (large decreases in log-odds).
    for _ in range(20):
        m_b, e_b = 0.3, 0.2
        m_a = float(sigmoid(logit(m_b) - 0.3 * 4))
        e_a = float(sigmoid(logit(e_b) - 0.3 * 4))
        s = resp.observe(4, m_b, m_a, e_b, e_a)
        assert s == pytest.approx([0.3, 0.3])
    assert resp.mean == pytest.approx([0.3, 0.3], abs=0.01)
    assert np.all(np.linalg.eigvalsh(resp.cov) < 0.05 ** 2 / 10)
    assert resp.mean_observed_slope() == pytest.approx([0.3, 0.3])
    # Exact conjugate formula for one observation (diagonal case).
    r2 = RegionResponse([0.0, 0.0], np.eye(2), obs_noise_var=1.0)
    r2.observe(1, 0.5, float(sigmoid(-2.0)), 0.5, 0.5)
    assert r2.mean == pytest.approx([1.0, 0.0]) and r2.cov == pytest.approx(0.5 * np.eye(2))
    assert r2.observe(0, 0.5, 0.4, 0.5, 0.4) is None


def test_transferred_prior_learns_from_history():
    prior = TransferredPrior(prior_mean=0.05, prior_var=0.05 ** 2)
    h0 = RegionDescriptor(1, 0.2, 0.3, 4, 2, 0.1, 0.1)
    mean, cov = prior.predict(h0)
    assert mean == pytest.approx([0.05, 0.05]) and cov == pytest.approx(0.05 ** 2 * np.eye(2))
    # History: slope grows with the abstention rate, err slope constant.
    rng = np.random.default_rng(0)
    descs, slopes = [], []
    for _ in range(80):
        a = float(rng.uniform(0, 0.6))
        h = RegionDescriptor(float(rng.integers(0, 4)), float(rng.uniform(0.05, 0.5)), float(rng.uniform(0.2, 0.5)),
                             float(rng.integers(2, 8)), float(rng.integers(0, 20)), a, float(rng.uniform(0, 0.3)))
        descs.append(h)
        slopes.append([0.02 + 0.3 * a + rng.normal(0, 0.005), 0.08 + rng.normal(0, 0.005)])
    prior.update(descs, slopes)
    assert prior.n_observations == 80
    lo = RegionDescriptor(1, 0.2, 0.3, 4, 2, 0.05, 0.1)
    hi = RegionDescriptor(1, 0.2, 0.3, 4, 2, 0.55, 0.1)
    m_lo, c_lo = prior.predict(lo)
    m_hi, _ = prior.predict(hi)
    assert m_hi[0] - m_lo[0] == pytest.approx(0.3 * 0.5, abs=0.03)
    assert m_lo[1] == pytest.approx(0.08, abs=0.01) and m_hi[1] == pytest.approx(0.08, abs=0.01)
    assert c_lo[0, 0] < 0.05 ** 2  # history sharpens the prior
    # Round-trip and the ablation switch.
    clone = TransferredPrior.from_dict(prior.to_dict())
    assert clone.predict(hi)[0] == pytest.approx(m_hi)
    off = TransferredPrior(0.05, 0.05 ** 2, enabled=False)
    off.update(descs, slopes)
    assert off.predict(hi)[0] == pytest.approx([0.05, 0.05])


def test_region_dispersion_bounds():
    same = np.tile(np.array([[1.0, 0.0]]), (5, 1))
    assert region_dispersion(same) == pytest.approx(0.0)
    opposite = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert region_dispersion(opposite) == pytest.approx(1.0)
