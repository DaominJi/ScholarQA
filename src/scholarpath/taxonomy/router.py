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
        self.train_steps = 0

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
            self.train_steps = self._adamw(X, y, w, rng)
        self.fitted = True
        return self

    def _loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        prob = np.clip(_sigmoid(self._forward(X)[2]), 1e-7, 1 - 1e-7)
        return float(-(w * (y * np.log(prob) + (1 - y) * np.log(1 - prob))).sum() / max(1.0, float(w.sum())))

    def _adamw(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, rng: np.random.Generator) -> int:
        """Mini-batch AdamW (decoupled weight decay) on the weighted BCE loss.

        Runs ``epochs`` epochs but at least ``min_steps`` optimizer steps (small
        label sets would otherwise get only a handful of updates) and at most
        ``max_steps``; stops early once the epoch loss stops improving.
        Returns the number of optimizer steps taken.
        """
        cfg = self.config
        p = self.params
        names = ("W1", "b1", "W2", "b2")
        m = {k: np.zeros_like(getattr(p, k)) for k in names}
        v = {k: np.zeros_like(getattr(p, k)) for k in names}
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        lr, wd = float(cfg.learning_rate), float(cfg.weight_decay)
        step = 0
        N = X.shape[0]
        bs = max(1, min(cfg.batch_size, N))
        steps_per_epoch = int(math.ceil(N / bs))
        epochs = max(int(cfg.epochs), int(math.ceil(int(getattr(cfg, "min_steps", 0)) / steps_per_epoch)))
        max_steps = getattr(cfg, "max_steps", None)
        if max_steps:
            epochs = min(epochs, max(1, int(math.ceil(int(max_steps) / steps_per_epoch))))
        patience = int(getattr(cfg, "patience", 0) or 0)
        tol = float(getattr(cfg, "tolerance", 1e-4))
        min_epochs = int(math.ceil(int(getattr(cfg, "min_steps", 0)) / steps_per_epoch))
        best, stale = float("inf"), 0
        for epoch in range(epochs):
            order = rng.permutation(N)
            for s in range(0, N, bs):
                idx = order[s : s + bs]
                xb, yb, wb = X[idx], y[idx], w[idx]
                h_pre, h, logit = self._forward(xb)
                prob = _sigmoid(logit)
                g_logit = (wb * (prob - yb) / max(1.0, float(wb.sum())))[:, None].astype(np.float32)
                g_h = (g_logit @ p.W2.T) * _gelu_grad(h_pre)
                grads = {"W2": h.T @ g_logit, "b2": g_logit.sum(axis=0), "W1": xb.T @ g_h, "b1": g_h.sum(axis=0)}
                step += 1
                c1 = 1.0 - b1_ ** step
                c2 = 1.0 - b2_ ** step
                for k in names:
                    param = getattr(p, k)
                    g = grads[k].reshape(param.shape)
                    mk, vk = m[k], v[k]
                    mk *= b1_
                    mk += (1 - b1_) * g
                    vk *= b2_
                    vk += (1 - b2_) * (g * g)
                    if k.startswith("W") and wd:
                        param *= 1.0 - lr * wd  # decoupled decay
                    param -= (lr / c1) * mk / (np.sqrt(vk / c2) + eps)
            if patience and epoch + 1 >= min_epochs:
                loss = self._loss(X, y, w)
                if loss < best - tol:
                    best, stale = loss, 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
        return step

    # ---------------------------------------------------------------- io
    def state_dict(self) -> Dict[str, np.ndarray]:
        """Arrays + metadata needed to restore the router (see :meth:`from_state`)."""
        if self.params is None:
            return {"fitted": np.array(0)}
        return {
            "fitted": np.array(int(self.fitted)),
            "W1": self.params.W1, "b1": self.params.b1, "W2": self.params.W2, "b2": self.params.b2,
            "prior_logit": np.array(self.prior_logit),
            "features": np.array(self.features),
        }

    @classmethod
    def from_state(cls, state: Dict[str, np.ndarray], config: Optional[RouterConfig] = None) -> "MLPRouter":
        r = cls(config, features=str(state.get("features", "concat_product")))
        if int(state.get("fitted", 0)) and "W1" in state:
            r.params = _Params(
                W1=np.asarray(state["W1"], dtype=np.float32), b1=np.asarray(state["b1"], dtype=np.float32),
                W2=np.asarray(state["W2"], dtype=np.float32), b2=np.asarray(state["b2"], dtype=np.float32),
            )
            r.prior_logit = float(state.get("prior_logit", 0.0))
            r.fitted = True
        return r

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
