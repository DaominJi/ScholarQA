"""Static paper selectors of the selection-based taxonomy baselines (Section 5.2).

At each internal node v, a node-wise selection baseline picks ``k = ⌈η n_v⌉``
papers of P_v for the LLM to read (one-off selection picks ``⌈η |P|⌉`` papers
once at the root, Appendix C.5). Four selectors are compared in the paper:

``"random"``
    uniform sampling without replacement.
``"coverage"``
    greedy maximization of the facility-location objective [88]

        F(S) = Σ_{p ∈ P_v} max_{s ∈ S} cos(x_p, x_s),

    a monotone submodular function, so the greedy solution is a
    (1 − 1/e)-approximation. Similarities are clipped at zero
    (``max(cos, 0)``), the standard non-negativity requirement of facility
    location, which leaves the objective unchanged for the non-negative
    similarities of SPECTER2 embeddings. The greedy loop uses *lazy
    evaluations* (Minoux's accelerated greedy): stale marginal gains are
    upper bounds by submodularity, so a candidate whose re-evaluated gain is
    still the largest key in the priority queue is the exact greedy choice.
``"kmeans"``
    k-means with ``k`` clusters; for every centroid, the member paper nearest
    to it (its medoid) is selected [18]. Clusters are visited from the most to
    the least populous, so the returned order lists the most representative
    papers first.
``"diversity"``
    farthest-point traversal (Gonzalez's k-center greedy [15, 32, 63]) under
    cosine distance ``1 − cos``. The traversal starts deterministically from
    the paper nearest to the centroid of P_v and repeatedly adds the paper
    whose distance to the selected set is largest.

All selectors L2-normalize the input rows (so inner products are cosine
similarities), return *distinct* row indices, and are deterministic given
``seed`` (only ``"random"`` and ``"kmeans"`` consume it; ties are broken by the
smallest row index). Edge cases: ``k <= 0`` returns ``[]`` and ``k >= n``
returns every row ``[0, ..., n-1]``; duplicate vectors never produce duplicate
indices. The individual ``*_selection`` functions expect unit-norm rows and
``0 < k < n``; :func:`select` normalizes the input and handles the edge cases.
"""

from __future__ import annotations

import heapq
import math
import warnings
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]

SELECTION_METHODS = ("random", "coverage", "kmeans", "diversity")

# Facility location keeps the full n x n float32 similarity matrix in memory
# when it fits in this budget (1 GiB: n <= 16384, which covers the 2,000-paper
# arXiv sets, the 3,044-paper TaxoBench pool and every node of up to 16K
# papers). Larger inputs recompute similarity columns on the fly from the
# vectors: the same objective and greedy rule with O(n d) work per candidate
# evaluation instead of O(n), so much slower; float32 near-ties between
# candidates may then resolve differently.
DENSE_MAX_BYTES = 1 << 30
DENSE_MAX_ROWS = int(math.isqrt(DENSE_MAX_BYTES // 4))
# Number of stale candidates re-evaluated together by the lazy greedy loop.
LAZY_BATCH = 16
# k-means: Lloyd iterations are capped (with k = ⌈η n_v⌉ up to n_v / 2 the
# clusters are tiny and Lloyd typically stops after a few iterations; the
# running time is dominated by the O(n k d) k-means++ seeding). Inputs above
# KMEANS_MINIBATCH_ROWS rows use mini-batch k-means.
KMEANS_MAX_ITER = 100
KMEANS_MINIBATCH_ROWS = 20000
# Blockwise similarity computations keep each temporary block at about this
# many float32 entries (2^25 = 128 MB).
BLOCK_ENTRIES = 1 << 25


def _block(n_other: int) -> int:
    """Rows (or columns) per block when the other side of the product has ``n_other`` entries."""
    return max(1, min(4096, BLOCK_ENTRIES // max(1, int(n_other))))


def num_to_select(n: int, eta: float) -> int:
    """Selection size ``⌈η n⌉`` (capped at ``n``), robust to float round-off.

    ``0.2 * 15`` evaluates to ``3.0000000000000004`` in floating point; a
    naive ``ceil`` would select 4 papers instead of 3.
    """
    n = int(n)
    if n <= 0 or eta <= 0:
        return 0
    return min(n, int(math.ceil(float(eta) * n - 1e-9)))


def _as_unit_rows(vectors: ArrayLike) -> np.ndarray:
    X = np.asarray(vectors, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1) if X.size else X.reshape(0, 0)
    if X.ndim != 2:
        raise ValueError(f"vectors must be a 2-D array, got shape {X.shape}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------


def random_selection(X: np.ndarray, k: int, seed: int = 0) -> List[int]:
    """Uniform sampling of ``k`` rows without replacement."""
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.choice(X.shape[0], size=k, replace=False)]


def facility_location(vectors: ArrayLike, selected: Sequence[int]) -> float:
    """Facility-location value F(S) = Σ_p max_{s∈S} max(cos(x_p, x_s), 0); F(∅) = 0."""
    X = _as_unit_rows(vectors)
    sel = np.asarray(list(selected), dtype=int)
    if X.shape[0] == 0 or sel.size == 0:
        return 0.0
    S = X[sel]
    total = 0.0
    step = _block(S.shape[0])
    for s in range(0, X.shape[0], step):
        block = X[s : s + step] @ S.T
        total += float(np.maximum(block.max(axis=1), 0.0).sum())
    return total


def _dense_similarity(X: np.ndarray) -> np.ndarray:
    """``max(X X^T, 0)`` as float32, built blockwise in place (peak ≈ n² floats + one block)."""
    n = X.shape[0]
    K = np.empty((n, n), dtype=np.float32)
    step = _block(n)
    for s in range(0, n, step):
        block = K[s : s + step]
        np.matmul(X[s : s + step], X.T, out=block)
        np.maximum(block, 0.0, out=block)
    return K


def coverage_selection(
    X: np.ndarray,
    k: int,
    seed: int = 0,
    dense_max_rows: int = DENSE_MAX_ROWS,
    lazy_batch: int = LAZY_BATCH,
) -> List[int]:
    """Lazy greedy maximization of the facility-location objective [88].

    Rows are returned in greedy order. ``seed`` is unused (the greedy is
    deterministic; ties go to the smallest index). Inputs of up to
    ``dense_max_rows`` rows use a precomputed similarity matrix; larger ones
    compute similarity columns on demand (see :data:`DENSE_MAX_BYTES`).
    """
    del seed
    n = X.shape[0]
    K = _dense_similarity(X) if n <= dense_max_rows else None

    def sim_cols(cols: np.ndarray) -> np.ndarray:
        """(n, len(cols)) clipped similarities of every row to the given rows."""
        if K is not None:
            return K[cols].T  # K is symmetric; row gathers are contiguous
        return np.maximum(X @ X[cols].T, 0.0)

    cur = np.zeros(n, dtype=np.float64)  # max_{s∈S} sim(p, s), 0 for S = ∅
    gains = np.empty(n, dtype=np.float64)
    step = _block(n)
    for s in range(0, n, step):
        cols = np.arange(s, min(n, s + step))
        gains[cols] = sim_cols(cols).sum(axis=0, dtype=np.float64)
    heap = [(-float(g), i) for i, g in enumerate(gains)]
    heapq.heapify(heap)
    fresh = np.zeros(n, dtype=np.int64)  # greedy round in which the key was computed
    selected: List[int] = []
    for r in range(k):
        while True:
            _, top = heap[0]
            if fresh[top] == r:  # exact gain that dominates all upper bounds
                heapq.heappop(heap)
                break
            cand: List[int] = []
            while heap and len(cand) < lazy_batch and fresh[heap[0][1]] != r:
                cand.append(heapq.heappop(heap)[1])
            idx = np.asarray(cand, dtype=int)
            g = np.maximum(sim_cols(idx) - cur[:, None], 0.0).sum(axis=0, dtype=np.float64)
            fresh[idx] = r
            for i, gi in zip(cand, g.tolist()):
                heapq.heappush(heap, (-gi, i))
        selected.append(int(top))
        cur = np.maximum(cur, sim_cols(np.asarray([top]))[:, 0])
    return selected


def _kmeans(X: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """(labels, centers) of k-means with k clusters (scikit-learn)."""
    try:
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as err:  # pragma: no cover - core dependency
        raise ImportError("The 'kmeans' selector requires scikit-learn (pip install scikit-learn).") from err
    with warnings.catch_warnings():
        # Fewer distinct points than clusters (duplicate papers) is handled below.
        warnings.simplefilter("ignore", ConvergenceWarning)
        # One k-means++ initialization (scikit-learn's ``n_init="auto"``) keeps the
        # selector affordable for k = ⌈η n_v⌉ in the hundreds.
        if X.shape[0] > KMEANS_MINIBATCH_ROWS:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=1, batch_size=4096,
                                 max_iter=KMEANS_MAX_ITER)
        else:
            km = KMeans(n_clusters=k, random_state=seed, n_init=1, max_iter=KMEANS_MAX_ITER)
        labels = km.fit_predict(X)
    return np.asarray(labels, dtype=int), np.asarray(km.cluster_centers_, dtype=np.float32)


def kmeans_selection(X: np.ndarray, k: int, seed: int = 0) -> List[int]:
    """Medoid nearest to each of ``k`` k-means centroids [18].

    Each centroid takes the member of its cluster nearest to it; clusters are
    disjoint, so these medoids are distinct. A centroid whose cluster is empty
    (possible when P_v has fewer than ``k`` distinct vectors) takes the nearest
    paper not selected yet, so exactly ``k`` distinct rows are returned.
    """
    n = X.shape[0]
    labels, centers = _kmeans(X, k, seed)
    sizes = np.bincount(labels, minlength=centers.shape[0])
    order = sorted(range(centers.shape[0]), key=lambda j: (-int(sizes[j]), j))
    taken = np.zeros(n, dtype=bool)
    picks: Dict[int, int] = {}
    for j in order:
        members = np.flatnonzero(labels == j)
        if members.size == 0:
            continue
        d = ((X[members] - centers[j]) ** 2).sum(axis=1)
        best = int(members[int(np.argmin(d))])
        picks[j] = best
        taken[best] = True
    for j in order:
        if j in picks:
            continue
        d = ((X - centers[j]) ** 2).sum(axis=1)
        d[taken] = np.inf
        best = int(np.argmin(d))
        picks[j] = best
        taken[best] = True
    return [picks[j] for j in order]


def diversity_selection(X: np.ndarray, k: int, seed: int = 0) -> List[int]:
    """Farthest-point traversal (k-center greedy) under cosine distance [15, 32, 63].

    Starts from the row nearest to the centroid (largest cosine with the mean
    direction). ``seed`` is unused (deterministic; ties go to the smallest index).
    """
    del seed
    centroid = X.mean(axis=0)
    start = int(np.argmax(X @ centroid)) if float(np.linalg.norm(centroid)) > 0 else 0
    selected = [start]
    min_dist = np.maximum(1.0 - X @ X[start], 0.0).astype(np.float64)
    min_dist[start] = -np.inf
    for _ in range(1, k):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.maximum(1.0 - X @ X[nxt], 0.0))
        min_dist[selected] = -np.inf
    return selected


_SELECTORS: Dict[str, Callable[[np.ndarray, int, int], List[int]]] = {
    "random": random_selection,
    "coverage": coverage_selection,
    "kmeans": kmeans_selection,
    "diversity": diversity_selection,
}


def select(vectors: ArrayLike, k: int, method: str, seed: int = 0) -> List[int]:
    """Select ``k`` distinct rows of ``vectors`` with the given method (Section 5.2).

    Args:
        vectors: (n, d) paper embeddings (rows are L2-normalized internally).
        k: number of rows to select; ``k <= 0`` -> ``[]``, ``k >= n`` -> all rows.
        method: ``"random"``, ``"coverage"``, ``"kmeans"`` or ``"diversity"``.
        seed: random seed (used by ``"random"`` and ``"kmeans"``).

    Returns:
        Row indices in selection order (greedy order for ``"coverage"`` /
        ``"diversity"``, decreasing cluster size for ``"kmeans"``, draw order
        for ``"random"``; ``list(range(n))`` when every row is selected).
    """
    name = (method or "").strip().lower()
    if name not in _SELECTORS:
        raise ValueError(f"Unknown selection method {method!r}; expected one of {SELECTION_METHODS}")
    X = _as_unit_rows(vectors)
    n = X.shape[0]
    k = int(k)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(range(n))
    out = _SELECTORS[name](X, k, int(seed))
    if len(out) != k or len(set(out)) != k:  # pragma: no cover - internal invariant
        raise RuntimeError(f"selector {name!r} returned {len(set(out))} distinct rows, expected {k}")
    return out


__all__ = [
    "SELECTION_METHODS",
    "coverage_selection",
    "diversity_selection",
    "facility_location",
    "kmeans_selection",
    "num_to_select",
    "random_selection",
    "select",
]
