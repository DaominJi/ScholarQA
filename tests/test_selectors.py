"""Tests for the static paper selectors of the selection baselines (Section 5.2)."""

from __future__ import annotations

import numpy as np
import pytest

from scholarpath.taxonomy.selectors import (
    SELECTION_METHODS,
    coverage_selection,
    facility_location,
    num_to_select,
    select,
)


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _clustered(sizes=(40, 25, 15, 10, 5), dim=32, spread=0.05, seed=0):
    """Well-separated clusters on the unit sphere; returns (X, labels)."""
    rng = np.random.default_rng(seed)
    centers = np.eye(dim)[: len(sizes)]
    X, y = [], []
    for j, n in enumerate(sizes):
        X.append(centers[j] + spread * rng.standard_normal((n, dim)))
        y.extend([j] * n)
    return _unit(np.concatenate(X)).astype(np.float32), np.array(y)


def _greedy_bruteforce(X: np.ndarray, k: int):
    K = np.maximum(X @ X.T, 0.0)
    cur = np.zeros(len(X))
    chosen = []
    for _ in range(k):
        gains = np.maximum(K - cur[:, None], 0.0).sum(axis=0)
        gains[chosen] = -1.0
        i = int(np.argmax(gains))
        chosen.append(i)
        cur = np.maximum(cur, K[:, i])
    return chosen


def test_num_to_select_is_ceiling_without_float_noise():
    assert num_to_select(15, 0.2) == 3  # 0.2 * 15 = 3.0000000000000004 in floating point
    assert num_to_select(16, 0.2) == 4
    assert num_to_select(3, 0.05) == 1
    assert num_to_select(10, 1.0) == 10
    assert num_to_select(0, 0.5) == 0 and num_to_select(10, 0.0) == 0


@pytest.mark.parametrize("method", SELECTION_METHODS)
def test_edge_cases(method):
    X, _ = _clustered()
    n = X.shape[0]
    assert select(X, 0, method) == []
    assert select(X, -3, method) == []
    assert select(X, n, method) == list(range(n))
    assert select(X, n + 5, method) == list(range(n))
    assert select(np.zeros((0, 4)), 3, method) == []
    out = select(X, 7, method, seed=3)
    assert len(out) == 7 and len(set(out)) == 7 and all(0 <= i < n for i in out)
    assert out == select(X, 7, method, seed=3)  # deterministic


@pytest.mark.parametrize("method", SELECTION_METHODS)
def test_duplicate_vectors_give_distinct_rows(method):
    rng = np.random.default_rng(1)
    X = np.repeat(rng.standard_normal((3, 6)), 5, axis=0)  # 3 distinct vectors, 15 rows
    out = select(X, 8, method)
    assert len(out) == 8 and len(set(out)) == 8
    # The first picks of the geometric selectors cover all three distinct vectors.
    if method != "random":
        assert {i // 5 for i in out[:3]} == {0, 1, 2}


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        select(np.eye(3), 1, "farthest")


def test_random_selection_depends_on_seed():
    X, _ = _clustered()
    assert select(X, 10, "random", seed=0) != select(X, 10, "random", seed=1)


def test_lazy_greedy_matches_plain_greedy():
    rng = np.random.default_rng(0)
    X = _unit(rng.standard_normal((300, 16))).astype(np.float32)
    expected = _greedy_bruteforce(X, 40)
    assert select(X, 40, "coverage") == expected
    # The on-the-fly path (no dense similarity matrix) selects the same rows.
    assert coverage_selection(X, 40, dense_max_rows=0) == expected


def test_facility_location_values():
    X, _ = _clustered()
    assert facility_location(X, []) == 0.0
    assert facility_location(X, range(X.shape[0])) == pytest.approx(X.shape[0], rel=1e-5)
    # Monotone: adding rows never decreases the objective.
    assert facility_location(X, [0, 50]) >= facility_location(X, [0]) - 1e-9


def test_coverage_beats_random_on_clustered_data():
    X, _ = _clustered()
    for k in (3, 5, 10):
        cov = facility_location(X, select(X, k, "coverage"))
        for seed in range(5):
            assert cov >= facility_location(X, select(X, k, "random", seed=seed)) - 1e-6


def test_diversity_spreads_points():
    X, y = _clustered()
    out = select(X, 5, "diversity")
    assert sorted(y[out].tolist()) == [0, 1, 2, 3, 4]  # one point per cluster
    # Deterministic start: the row nearest to the centroid of the data.
    assert out[0] == int(np.argmax(X @ X.mean(axis=0)))

    def covering_radius(idx):  # k-center objective: max_p min_{s in S} ||x_p - x_s||
        d2 = np.maximum(2.0 - 2.0 * (X @ X[idx].T), 0.0)
        return float(np.sqrt(d2.min(axis=1).max()))

    rand5 = [covering_radius(select(X, 5, "random", seed=s)) for s in range(5)]
    assert covering_radius(out) <= min(rand5) + 1e-6
    # Gonzalez: farthest-point traversal is within 2x the optimal radius, and
    # the optimum is at most the radius of any k-subset.
    div8 = covering_radius(select(X, 8, "diversity"))
    for seed in range(5):
        assert div8 <= 2.0 * covering_radius(select(X, 8, "random", seed=seed)) + 1e-6


def test_kmeans_returns_one_distinct_medoid_per_cluster():
    X, y = _clustered()
    out = select(X, 5, "kmeans", seed=0)
    assert len(set(out)) == 5
    assert sorted(y[out].tolist()) == [0, 1, 2, 3, 4]
    # Each pick is its cluster's member nearest to the cluster centroid,
    # listed from the largest to the smallest cluster.
    assert y[out].tolist() == [0, 1, 2, 3, 4]
    for i in out:
        members = np.flatnonzero(y == y[i])
        centroid = X[members].mean(axis=0)
        d = ((X[members] - centroid) ** 2).sum(axis=1)
        assert i == members[int(np.argmin(d))]
