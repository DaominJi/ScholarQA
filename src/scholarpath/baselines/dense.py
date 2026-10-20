"""Dense retrieval baselines (Section 5.3).

* Table 2 "SPECTER" [10]: SPECTER (v1) embeddings of the passage pool;
* Tables 3-5 "Dense retrieval" and the first stage of "Dense + reranker":
  SPECTER2 [71] embeddings (the same embedder that builds ScholarPath's
  semantic index; documents use the proximity adapter via
  ``embedder.encode`` and queries the ad-hoc query adapter via
  ``embedder.encode_queries``).

Passage embeddings live in an in-memory float32 matrix (optionally cached on
disk as ``.npy`` together with the pool fingerprint). Search is exact: the
query vector is multiplied with the matrix block by block and the top-k of
every block are merged, so memory stays bounded for ~10^6 passages.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from scholarpath.baselines.common import (
    DEFAULT_MAX_CANDIDATES,
    Candidate,
    PassagePool,
    QueryLike,
    as_query,
    distinct_papers,
    top_k_indices,
)
from scholarpath.baselines.packing import BudgetPacker
from scholarpath.config import EmbeddingConfig
from scholarpath.embeddings import CachedEmbedder, Embedder, build_embedder, l2_normalize
from scholarpath.retrieval.result import RetrievalResult

log = logging.getLogger(__name__)

#: sentence-transformers id of SPECTER (v1), Table 2 "SPECTER" [10].
SPECTER_MODEL = "sentence-transformers/allenai-specter"


def build_dense_embedder(variant: str = "specter2", config: Optional[EmbeddingConfig] = None) -> Embedder:
    """Embedder of a dense baseline.

    ``"specter2"`` -> the configured SPECTER2 embedder (``config.backend`` is
    forced to ``"specter2"`` unless it is ``"hashing"``, which keeps offline
    runs offline); ``"specter"`` -> SPECTER v1 through sentence-transformers
    (requires ``pip install sentence-transformers``); ``"config"`` -> exactly
    ``build_embedder(config)``.
    """
    import dataclasses

    config = config or EmbeddingConfig()
    variant = variant.lower()
    if variant == "config" or config.backend == "hashing":
        return build_embedder(config)
    if variant == "specter2":
        return build_embedder(dataclasses.replace(config, backend="specter2"))
    if variant == "specter":
        try:
            from scholarpath.embeddings import SentenceTransformerEmbedder

            base = SentenceTransformerEmbedder(SPECTER_MODEL, config.batch_size, config.device)
        except ImportError as err:  # pragma: no cover - optional dependency
            raise ImportError(
                "The SPECTER baseline needs sentence-transformers: pip install 'scholarpath[embeddings]'"
            ) from err
        return CachedEmbedder(base, cache_dir=config.cache_dir, namespace="specter-v1")  # pragma: no cover
    raise ValueError(f"Unknown dense variant {variant!r} (expected specter | specter2 | config)")


class DenseRetriever:
    """Exact inner-product (cosine) search over passage embeddings."""

    def __init__(
        self,
        pool: PassagePool,
        embedder: Embedder,
        name: str = "Dense retrieval",
        embeddings: Optional[np.ndarray] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        encode_batch: int = 4096,
        block_rows: int = 262_144,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ):
        self.pool = pool
        self.embedder = embedder
        self.name = name
        self.encode_batch = int(encode_batch)
        self.block_rows = int(block_rows)
        self.max_candidates = int(max_candidates)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._matrix: Optional[np.ndarray] = None
        if embeddings is not None:
            emb = np.asarray(embeddings, dtype=np.float32)
            if emb.shape[0] != len(pool):
                raise ValueError(f"{emb.shape[0]} embeddings for {len(pool)} units")
            self._matrix = emb

    # ------------------------------------------------------------ build
    @classmethod
    def from_index(cls, pool: PassagePool, index: Any, embedder: Embedder, **kwargs: Any) -> "DenseRetriever":
        """Reuse the SPECTER2 vectors of a :class:`SemanticIndex` for units it
        contains (``paper:<id>`` / ``chunk:<id>`` keys); others are encoded."""
        dim = index.dim or embedder.dim
        mat = np.zeros((len(pool), dim), dtype=np.float32)
        missing: List[int] = []
        for i, p in enumerate(pool):
            if index.has_vector(p.key):
                mat[i] = index.vector(p.key)
            else:
                missing.append(i)
        if missing:
            mat[missing] = embedder.encode(pool.search_texts(missing))
        return cls(pool, embedder, embeddings=mat, **kwargs)

    def _cache_files(self) -> Tuple[Path, Path]:
        assert self.cache_dir is not None
        tag = self.pool.fingerprint()[:16]
        return self.cache_dir / f"passages-{tag}.npy", self.cache_dir / f"passages-{tag}.json"

    def build(self) -> "DenseRetriever":
        """Encode the pool (or load the on-disk cache)."""
        if self._matrix is not None:
            return self
        if self.cache_dir is not None:
            npy, meta = self._cache_files()
            if npy.exists() and meta.exists():
                info = json.loads(meta.read_text())
                if info.get("fingerprint") == self.pool.fingerprint() and info.get("n") == len(self.pool):
                    self._matrix = np.load(npy, mmap_mode="r")
                    log.info("%s: loaded %d cached passage embeddings from %s", self.name, len(self.pool), npy)
                    return self
        n = len(self.pool)
        dim = int(getattr(self.embedder, "dim", 0)) or None
        chunks: List[np.ndarray] = []
        t0 = time.perf_counter()
        for s in range(0, n, self.encode_batch):
            idx = list(range(s, min(n, s + self.encode_batch)))
            chunks.append(np.asarray(self.embedder.encode(self.pool.search_texts(idx)), dtype=np.float32))
            if n > self.encode_batch:
                log.info("%s: encoded %d/%d passages (%.0fs)", self.name, idx[-1] + 1, n, time.perf_counter() - t0)
        mat = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, dim or 0), dtype=np.float32)
        self._matrix = l2_normalize(mat) if mat.size else mat
        if self.cache_dir is not None:
            npy, meta = self._cache_files()
            npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy, self._matrix)
            meta.write_text(json.dumps({"fingerprint": self.pool.fingerprint(), "n": n, "name": self.name}))
        return self

    def ensure_built(self) -> float:
        """Encode (or load) the passage matrix if needed; seconds spent (0 if built)."""
        if self._matrix is not None:
            return 0.0
        t0 = time.perf_counter()
        self.build()
        return time.perf_counter() - t0

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self.build()
        assert self._matrix is not None
        return self._matrix

    def vectors(self, indices: Sequence[int]) -> np.ndarray:
        return np.asarray(self.matrix[list(indices)], dtype=np.float32)

    # ----------------------------------------------------------- search
    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self.embedder.encode_queries([text])[0], dtype=np.float32)

    def search_vector(self, qvec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Exact top-``k`` by inner product, computed block by block."""
        M = self.matrix
        n = M.shape[0]
        if n == 0 or k <= 0:
            return []
        k = min(int(k), n)
        idx_parts: List[np.ndarray] = []
        sc_parts: List[np.ndarray] = []
        for s in range(0, n, self.block_rows):
            block = np.asarray(M[s : s + self.block_rows], dtype=np.float32) @ qvec
            top = top_k_indices(block, k)
            idx_parts.append(top + s)
            sc_parts.append(block[top])
        idx = np.concatenate(idx_parts)
        sc = np.concatenate(sc_parts)
        order = top_k_indices(sc, k)
        # top_k_indices breaks ties by position; positions are increasing in
        # the global index within and across blocks, so ties stay index-ordered.
        return [(int(idx[o]), float(sc[o])) for o in order]

    def search(self, text: str, k: int) -> List[Tuple[int, float]]:
        return self.search_vector(self.encode_query(text), k)

    def top_papers(self, query: QueryLike, k: int) -> List[str]:
        q = as_query(query)
        hits = self.search(q.text, max(4 * k, self.max_candidates))
        return distinct_papers(self.pool, (i for i, _ in hits), k)

    def retrieve(self, query: QueryLike, budget: int) -> RetrievalResult:
        t0 = time.perf_counter()
        index_s = self.ensure_built()  # offline passage encoding: not query latency
        q = as_query(query)
        hits = self.search(q.text, self.max_candidates)
        packer = BudgetPacker(budget)
        packer.extend(Candidate(self.pool[i], s, "dense") for i, s in hits)
        return packer.result(q.query_id, titles=self.pool.titles,
                             stats={"method": self.name, "index_build_s": index_s},
                             latency_s=max(0.0, time.perf_counter() - t0 - index_s))
