"""Regions of a node as decision units (Section 3.2).

The papers P_v of node v are grouped by k-means in the embedding space into
G_v = min{max_regions, floor(n_v / papers_per_region)} regions
G_{v1}, ..., G_{vG_v} with shares pi_{vg} = |G_{vg}| / n_v (at least one
region). Regions serve only sampling and prediction: a concept may span
several regions, and each child builds its own regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


def num_regions(n_v: int, max_regions: int = 8, papers_per_region: int = 24) -> int:
    return max(1, min(int(max_regions), int(n_v) // max(1, int(papers_per_region))))


def kmeans_labels(X: np.ndarray, k: int, seed: int = 0, n_init: int = 4, max_iter: int = 100) -> np.ndarray:
    """k-means cluster labels (sklearn if available, NumPy Lloyd otherwise)."""
    n = X.shape[0]
    if k <= 1 or n <= k:
        return np.zeros(n, dtype=int) if k <= 1 else np.arange(n) % max(1, k)
    try:
        from sklearn.cluster import KMeans, MiniBatchKMeans

        if n > 20000:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=n_init, batch_size=4096)
        else:
            km = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter)
        return km.fit_predict(X).astype(int)
    except ImportError:  # pragma: no cover
        rng = np.random.default_rng(seed)
        best, best_inertia = None, np.inf
        for _ in range(n_init):
            C = X[rng.choice(n, k, replace=False)].copy()
            for _ in range(max_iter):
                d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
                lab = d.argmin(1)
                newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
                if np.allclose(newC, C):
                    break
                C = newC
            inertia = ((X - C[lab]) ** 2).sum()
            if inertia < best_inertia:
                best, best_inertia = lab, inertia
        return best


@dataclass
class Regions:
    """Partition of a node's papers into k-means regions."""

    paper_ids: List[str]
    labels: np.ndarray  # region index per paper (aligned with paper_ids)
    members: List[List[str]] = field(default_factory=list)

    @property
    def G(self) -> int:
        return len(self.members)

    @property
    def shares(self) -> np.ndarray:
        n = max(1, len(self.paper_ids))
        return np.array([len(m) / n for m in self.members], dtype=float)

    def region_of(self) -> Dict[str, int]:
        return {p: int(g) for p, g in zip(self.paper_ids, self.labels)}

    @classmethod
    def build(
        cls,
        paper_ids: Sequence[str],
        vectors: np.ndarray,
        max_regions: int = 8,
        papers_per_region: int = 24,
        seed: int = 0,
    ) -> "Regions":
        ids = list(paper_ids)
        k = num_regions(len(ids), max_regions, papers_per_region)
        labels = kmeans_labels(vectors, k, seed=seed) if len(ids) else np.zeros(0, dtype=int)
        # Re-index to contiguous non-empty regions.
        uniq = sorted(set(labels.tolist()))
        remap = {g: i for i, g in enumerate(uniq)}
        labels = np.array([remap[g] for g in labels.tolist()], dtype=int)
        members: List[List[str]] = [[] for _ in uniq]
        for p, g in zip(ids, labels.tolist()):
            members[g].append(p)
        return cls(ids, labels, members)
