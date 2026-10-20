"""Lightweight router f_theta_v (Section 3.1, Appendix C.3).

A two-layer MLP (hidden 256, GELU) trained with AdamW on LLM-labelled
(paper, child) pairs. Its input concatenates the paper embedding and the
child-definition embedding (plus their element-wise product, which lets a
small network express similarity); its output is the probability
f_theta_v(p, c) in [0, 1] that paper p belongs to child c.

A paper may belong to several children; a paper whose probabilities are all
below the threshold is placed in the residual leaf bottom_v.

The LLM-labelled papers are split (stratified by primary label) into a
training part, which trains the router, and a validation part, which
measures how often the router disagrees with the LLM (the validation
disagreement rate d_{v,t} of Section 3.2).

Implemented in NumPy so that the core library has no deep-learning
dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from scholarpath.config import RouterConfig

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    t = np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x ** 3))
    dt = (1 - t ** 2) * _SQRT_2_OVER_PI * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1 + t) + 0.5 * x * dt


def _single_blas_thread():
    """Context manager capping BLAS threads at 1 (tiny matmuls are far slower multi-threaded)."""
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(1, "blas")
    except Exception:  # pragma: no cover - threadpoolctl ships with scikit-learn
        import contextlib

        return contextlib.nullcontext()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def stratified_split(
    labels: Dict[str, Sequence[str]], fraction: float, seed: int = 0
) -> Tuple[List[str], List[str]]:
    """Split labelled papers into (train, validation) stratified by primary label.

    Strata with a single paper go to training. At least one paper is kept in
    validation when there are >= 5 labelled papers.
    """
    rng = np.random.default_rng(seed)
    strata: Dict[str, List[str]] = {}
    for pid in sorted(labels):
        kids = list(labels[pid])
        key = kids[0] if kids else "__residual__"
        strata.setdefault(key, []).append(pid)
    train, val = [], []
    for key in sorted(strata):
        ids = strata[key]
        rng.shuffle(ids)
        n_val = int(round(fraction * len(ids))) if len(ids) > 1 else 0
        val.extend(ids[:n_val])
        train.extend(ids[n_val:])
    if not val and len(labels) >= 5 and len(train) > 1:
        val.append(train.pop())
    return sorted(train), sorted(val)


@dataclass
class _Params:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


class MLPRouter:
    """f_theta(p, c) = sigmoid(W2 GELU(W1 [p; c; p*c] + b1) + b2)."""

    def __init__(self, config: Optional[RouterConfig] = None, features: str = "concat_product"):
        self.config = config or RouterConfig()
        self.features = features
        self.params: Optional[_Params] = None
        self.fitted = False
        self.prior_logit = 0.0

    # ----------------------------------------------------------- features
    def _feat(self, P: np.ndarray, C: np.ndarray) -> np.ndarray:
        if self.features == "concat":
            return np.concatenate([P, C], axis=1)
        return np.concatenate([P, C, P * C], axis=1)

    def _forward(self, X: np.ndarray):
        p = self.params
        h_pre = X @ p.W1 + p.b1
        h = _gelu(h_pre)
        logit = (h @ p.W2 + p.b2).ravel()
        return h_pre, h, logit

    # -------------------------------------------------------------- train
    def fit(
        self,
        paper_vecs: np.ndarray,
        child_vecs: np.ndarray,
        targets: Sequence[Set[int]],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "MLPRouter":
        """Train on every (paper i, child c) pair; ``targets[i]`` = set of positive child indices."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        n, C = paper_vecs.shape[0], child_vecs.shape[0]
        if n == 0 or C == 0:
            self.fitted = False
            return self
        ii, cc, yy = [], [], []
        for i in range(n):
            pos = targets[i]
            for c in range(C):
                ii.append(i)
                cc.append(c)
                yy.append(1.0 if c in pos else 0.0)
        ii, cc, y = np.array(ii), np.array(cc), np.array(yy, dtype=np.float32)
        X = self._feat(paper_vecs[ii], child_vecs[cc]).astype(np.float32)
        d = X.shape[1]
        H = cfg.hidden_dim
        self.params = _Params(
            W1=(rng.standard_normal((d, H)) * math.sqrt(2.0 / (d + H))).astype(np.float32),
            b1=np.zeros(H, dtype=np.float32),
            W2=(rng.standard_normal((H, 1)) * math.sqrt(2.0 / (H + 1))).astype(np.float32),
            b2=np.zeros(1, dtype=np.float32),
        )
        pos_rate = float(y.mean()) if y.size else 0.5
        self.prior_logit = math.log(max(pos_rate, 1e-3) / max(1 - pos_rate, 1e-3))
        self.params.b2[:] = self.prior_logit
        # Balance positives and negatives.
        w = np.where(y > 0.5, 0.5 / max(pos_rate, 1e-3), 0.5 / max(1 - pos_rate, 1e-3)).astype(np.float32)
        if sample_weight is not None:
            w = w * np.asarray(sample_weight, dtype=np.float32)[ii]
        with _single_blas_thread():
            self._adamw(X, y, w, rng)
        self.fitted = True
        return self

    def _adamw(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, rng: np.random.Generator) -> None:
        cfg = self.config
        p = self.params
        names = ("W1", "b1", "W2", "b2")
        m = {k: np.zeros_like(getattr(p, k)) for k in names}
        v = {k: np.zeros_like(getattr(p, k)) for k in names}
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        step = 0
        N = X.shape[0]
        bs = max(1, min(cfg.batch_size, N))
        for _ in range(cfg.epochs):
            order = rng.permutation(N)
            for s in range(0, N, bs):
                idx = order[s : s + bs]
                xb, yb, wb = X[idx], y[idx], w[idx]
                h_pre, h, logit = self._forward(xb)
                prob = _sigmoid(logit)
                g_logit = (wb * (prob - yb) / max(1.0, wb.sum()))[:, None]  # dL/dlogit
                grads = {
                    "W2": h.T @ g_logit,
                    "b2": g_logit.sum(axis=0),
                }
                g_h = (g_logit @ p.W2.T) * _gelu_grad(h_pre)
                grads["W1"] = xb.T @ g_h
                grads["b1"] = g_h.sum(axis=0)
                step += 1
                for k in names:
                    param = getattr(p, k)
                    g = grads[k].astype(np.float32).reshape(param.shape)
                    m[k] = b1_ * m[k] + (1 - b1_) * g
                    v[k] = b2_ * v[k] + (1 - b2_) * g * g
                    mh = m[k] / (1 - b1_ ** step)
                    vh = v[k] / (1 - b2_ ** step)
                    if k.startswith("W"):
                        param *= 1.0 - cfg.learning_rate * cfg.weight_decay  # decoupled decay
                    param -= cfg.learning_rate * mh / (np.sqrt(vh) + eps)

    # ------------------------------------------------------------ predict
    def predict_proba(self, paper_vecs: np.ndarray, child_vecs: np.ndarray) -> np.ndarray:
        """(n_papers, n_children) matrix of f_theta(p, c)."""
        n, C = paper_vecs.shape[0], child_vecs.shape[0]
        if n == 0 or C == 0:
            return np.zeros((n, C), dtype=np.float32)
        if not self.fitted:
            sims = paper_vecs @ child_vecs.T
            return _sigmoid(8.0 * (sims - sims.mean()))
        out = np.zeros((n, C), dtype=np.float32)
        with _single_blas_thread():
            for c in range(C):
                X = self._feat(paper_vecs, np.repeat(child_vecs[c : c + 1], n, axis=0))
                out[:, c] = _sigmoid(self._forward(X)[2])
        return out

    def assign(self, probs: np.ndarray, threshold: Optional[float] = None) -> List[Set[int]]:
        """Children with probability >= threshold; empty set = residual leaf."""
        thr = self.config.threshold if threshold is None else threshold
        return [set(np.flatnonzero(row >= thr).tolist()) for row in probs]


def disagreement(router_sets: Sequence[Set[int]], llm_sets: Sequence[Set[int]]) -> np.ndarray:
    """Per-paper indicator that the router's child set differs from the LLM label."""
    return np.array([set(a) != set(b) for a, b in zip(router_sets, llm_sets)], dtype=bool)
