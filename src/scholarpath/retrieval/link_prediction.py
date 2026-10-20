"""Cross-taxonomy association model (Section 4.3, Eq. 8 and Eq. 12).

The link-prediction prior scores a (problem, method) pair by

    s_theta(u, m) = z_P(u)^T z_M(m),     z_P(u) = g_P(d_u, h_u),  z_M(m) = g_M(d_m, h_m),

where d_x is the embedding of concept x's description and h_x the
descriptions of its taxonomy ancestors. For bound-guided search over a
method subtree rooted at v, the index caches the center
mu_v = mean_{m ⪯ v} z_M(m) and radius rho_v = max_{m ⪯ v} ||z_M(m) - mu_v||_2,
so that by Cauchy-Schwarz every method m ⪯ v satisfies (Proposition 4)

    s_theta(u, m) <= z_P(u)^T mu_v + ||z_P(u)||_2 rho_v = U(u, v).        (12)

The same bound with the roles of z_P and z_M exchanged serves searches over
problem subtrees.

This module provides the abstract :class:`LinkPredictor` (scores, bounds and
caches, given an ``encode_concept`` implementation) and the untrained
:class:`EmbeddingLinkPredictor` fallback. The temporally trained encoders of
Appendix B.5 live in :mod:`scholarpath.retrieval.link_training`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from scholarpath.embeddings import l2_normalize
from scholarpath.index.semantic_index import SemanticIndex, concept_key


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class LinkPredictor:
    """Base class: subclasses implement :meth:`encode_concept`."""

    name = "base"

    def __init__(self) -> None:
        self._z: Dict[str, np.ndarray] = {}  # node id -> z (problem: z_P, method: z_M)
        self._facet: Dict[str, str] = {}
        self._ball: Dict[str, Tuple[np.ndarray, float]] = {}  # subtree root -> (mu_v, rho_v)
        self.prepared_for: Optional[int] = None

    # --------------------------------------------------------- encoding
    def encode_concept(self, facet: str, description_vec: np.ndarray, ancestor_vecs: np.ndarray) -> np.ndarray:
        """g_P / g_M: map a description embedding and its ancestors' embeddings to z."""
        raise NotImplementedError

    def encode_query_concept(
        self, facet: str, description_vec: np.ndarray, ancestor_vecs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Encode an unfamiliar concept absent from I (Section 4.3: query-local anchor)."""
        dim = description_vec.shape[-1]
        anc = ancestor_vecs if ancestor_vecs is not None else np.zeros((0, dim), dtype=np.float32)
        return self.encode_concept(facet, description_vec, anc)

    # ----------------------------------------------------------- caches
    def prepare(self, index: SemanticIndex) -> "LinkPredictor":
        """Compute z for every concept of the index and subtree balls (mu_v, rho_v)."""
        self._z.clear()
        self._facet.clear()
        self._ball.clear()
        for facet in ("problem", "method"):
            tax = index.taxonomy(facet)
            for nid in tax.nodes:
                desc = index.vector(concept_key(nid))
                anc_ids = tax.ancestors(nid)
                anc = (
                    np.stack([index.vector(concept_key(a)) for a in anc_ids])
                    if anc_ids
                    else np.zeros((0, desc.shape[0]), dtype=np.float32)
                )
                self._z[nid] = np.asarray(self.encode_concept(facet, desc, anc), dtype=np.float32)
                self._facet[nid] = facet
            for nid in tax.nodes:
                zs = np.stack([self._z[d] for d in tax.descendants(nid, include_self=True)])
                mu = zs.mean(axis=0)
                rho = float(np.sqrt(((zs - mu) ** 2).sum(axis=1)).max())
                self._ball[nid] = (mu.astype(np.float32), rho)
        self.prepared_for = id(index)
        return self

    def z(self, node_id: str) -> np.ndarray:
        return self._z[node_id]

    def z_problem(self, node_id: str) -> np.ndarray:
        return self._z[node_id]

    def z_method(self, node_id: str) -> np.ndarray:
        return self._z[node_id]

    def set_z(self, node_id: str, facet: str, z: np.ndarray) -> None:
        """Register a query-local anchor concept (not part of any subtree ball)."""
        self._z[node_id] = np.asarray(z, dtype=np.float32)
        self._facet[node_id] = facet

    # ----------------------------------------------------------- scores
    def score(self, u: str, m: str) -> float:
        """s_theta(u, m) = z_P(u)^T z_M(m) (Eq. 8)."""
        return float(np.dot(self._z[u], self._z[m]))

    def prob(self, u: str, m: str) -> float:
        return sigmoid(self.score(u, m))

    def ball(self, v: str) -> Tuple[np.ndarray, float]:
        return self._ball[v]

    def upper_bound(self, u: str, v: str) -> float:
        """U(u, v) >= s_theta(u, m) for every method m ⪯ v (Eq. 12)."""
        mu, rho = self._ball[v]
        zu = self._z[u]
        return float(np.dot(zu, mu) + np.linalg.norm(zu) * rho)

    def upper_bound_problems(self, m: str, v: str) -> float:
        """Roles exchanged: bound on s_theta(u, m) over problems u ⪯ v."""
        mu, rho = self._ball[v]
        zm = self._z[m]
        return float(np.dot(zm, mu) + np.linalg.norm(zm) * rho)

    def rank_methods(self, u: str, candidates: Sequence[str]) -> List[Tuple[str, float]]:
        return sorted(((m, self.score(u, m)) for m in candidates), key=lambda t: -t[1])

    # --------------------------------------------------------------- io
    def save(self, path: Union[str, Path]) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinkPredictor":
        path = Path(path)
        meta = json.loads((path / "link_predictor.json").read_text(encoding="utf-8"))
        kind = meta.get("kind", "embedding")
        if kind == "embedding":
            return EmbeddingLinkPredictor(scale=float(meta.get("scale", 6.0)), bias=float(meta.get("bias", -3.0)))
        from scholarpath.retrieval import link_training

        return link_training.load_predictor(path)


class EmbeddingLinkPredictor(LinkPredictor):
    """Untrained fallback: z from description embeddings (with ancestor context).

    s(u, m) = scale * cos(z_P(u), z_M(m)) + bias is realized by
    z = sqrt(scale) * normalize(d + 0.5 * mean(ancestors)) with an extra
    constant coordinate carrying the bias, so Eq. (12) still applies.
    """

    name = "embedding"

    def __init__(self, scale: float = 6.0, bias: float = -3.0):
        super().__init__()
        self.scale = float(scale)
        self.bias = float(bias)

    def encode_concept(self, facet: str, description_vec: np.ndarray, ancestor_vecs: np.ndarray) -> np.ndarray:
        d = np.asarray(description_vec, dtype=np.float32)
        if ancestor_vecs is not None and len(ancestor_vecs):
            d = d + 0.5 * np.asarray(ancestor_vecs, dtype=np.float32).mean(axis=0)
        d = l2_normalize(d[None, :])[0] * math.sqrt(self.scale)
        # Bias coordinate: z_P carries sign(bias) * sqrt(|bias|), z_M carries sqrt(|bias|).
        b = math.sqrt(abs(self.bias))
        extra = (math.copysign(b, self.bias) if facet == "problem" else b)
        return np.concatenate([d, np.array([extra], dtype=np.float32)])

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "link_predictor.json").write_text(
            json.dumps({"kind": "embedding", "scale": self.scale, "bias": self.bias}), encoding="utf-8"
        )
