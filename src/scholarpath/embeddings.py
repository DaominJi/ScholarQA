"""Embedding backends.

Papers, concept definitions, section chunks and query aspects are embedded
with SPECTER2 (768-d) in the paper (Section 5.1). Two lightweight fallbacks
are provided so that everything runs without model downloads:

* :class:`HashingEmbedder` -- deterministic hashed bag-of-n-grams;
* :class:`SentenceTransformerEmbedder` -- any sentence-transformers model.

All embedders return L2-normalised float32 matrices, so inner products are
cosine similarities. :class:`CachedEmbedder` adds an on-disk cache keyed by
text hash. Missing optional dependencies of a backend raise
:class:`MissingDependencyError` with the install command.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from scholarpath.config import EmbeddingConfig
from scholarpath.text import tokenize


class MissingDependencyError(ImportError):
    """An optional dependency of the configured backend is not installed (the message says how to fix it)."""


_EMBEDDINGS_HINT = ('pip install -e ".[embeddings]" (torch, transformers, adapters, sentence-transformers), '
                    "or use the offline hashing embedder (--embedder hashing)")


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0.0 else float(np.dot(a, b) / denom)


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return l2_normalize(A, axis=1) @ l2_normalize(B, axis=1).T


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic hashed uni+bi-gram embedder (no model download).

    Uses sub-linear term frequency and three hash functions per n-gram to
    reduce collisions; vectors are L2-normalised.
    """

    def __init__(self, dim: int = 768, ngram_range: tuple = (1, 2), seed: int = 0):
        self.dim = int(dim)
        self.ngram_range = ngram_range
        self.seed = int(seed)

    def _hash(self, token: str, salt: int) -> tuple:
        h = hashlib.blake2b(f"{salt}:{token}".encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(h, "little", signed=False)
        return v % self.dim, 1.0 if (v >> 40) & 1 else -1.0

    def _vectorize(self, text: str) -> np.ndarray:
        toks = tokenize(text, remove_stopwords=True)
        vec = np.zeros(self.dim, dtype=np.float32)
        if not toks:
            return vec
        counts: Dict[str, int] = {}
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(toks) - n + 1):
                g = " ".join(toks[i : i + n])
                counts[g] = counts.get(g, 0) + 1
        for g, c in counts.items():
            w = (1.0 + np.log(c)) * (1.0 if " " not in g else 0.5)
            for salt in range(3):
                idx, sign = self._hash(g, self.seed + salt)
                vec[idx] += sign * w
        n = float(np.linalg.norm(vec))
        return vec / n if n > 0 else vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = self._vectorize(t or "")
        return out

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)


class SentenceTransformerEmbedder:  # pragma: no cover - needs model download
    def __init__(self, model_name: str, batch_size: int = 32, device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise MissingDependencyError(f"The {model_name} embedder needs sentence-transformers: "
                                         f"{_EMBEDDINGS_HINT}.") from err

        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.batch_size = int(batch_size)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        v = self._model.encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(v, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)


class Specter2Embedder:  # pragma: no cover - needs model download
    """SPECTER2 base model with task adapters (``adapters`` library).

    Documents (title [SEP] abstract, concept definitions, chunks) use the
    proximity adapter; short queries use the ad-hoc query adapter.
    """

    def __init__(
        self,
        model_name: str = "allenai/specter2_base",
        adapter: Optional[str] = "allenai/specter2",
        query_adapter: Optional[str] = "allenai/specter2_adhoc_query",
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as err:
            raise MissingDependencyError(f"SPECTER2 embeddings ({model_name}) need torch and transformers "
                                         f"(missing: {err.name or err}): {_EMBEDDINGS_HINT}.") from err

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._adapters: Dict[str, str] = {}
        try:
            from adapters import AutoAdapterModel

            self.model = AutoAdapterModel.from_pretrained(model_name)
            if adapter:
                self._adapters["doc"] = self.model.load_adapter(adapter, source="hf", set_active=True)
            if query_adapter:
                self._adapters["query"] = self.model.load_adapter(query_adapter, source="hf")
        except ImportError:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)

    def _encode(self, texts: Sequence[str], which: str) -> np.ndarray:
        torch = self._torch
        if which in self._adapters:
            self.model.set_active_adapters(self._adapters[which])
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t or "" for t in texts[i : i + self.batch_size]]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt", return_token_type_ids=False,
            ).to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state[:, 0, :]
            out.append(h.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.concatenate(out, axis=0))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "doc")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query")


class CachedEmbedder:
    """Wraps an embedder with an in-memory + optional on-disk cache."""

    def __init__(self, base: Embedder, cache_dir: Optional[str] = None, namespace: str = "default"):
        self.base = base
        self.dim = base.dim
        self._mem: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._dir = Path(cache_dir) / namespace if cache_dir else None

    @staticmethod
    def _key(kind: str, text: str) -> str:
        return hashlib.sha1(f"{kind}\x00{text}".encode("utf-8")).hexdigest()

    def _lookup(self, key: str) -> Optional[np.ndarray]:
        v = self._mem.get(key)
        if v is None and self._dir is not None:
            p = self._dir / key[:2] / f"{key}.npy"
            if p.exists():
                try:
                    v = np.load(p)
                    self._mem[key] = v
                except Exception:
                    v = None
        return v

    def _store(self, key: str, v: np.ndarray) -> None:
        self._mem[key] = v
        if self._dir is not None:
            p = self._dir / key[:2] / f"{key}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, v)

    def _run(self, texts: Sequence[str], kind: str) -> np.ndarray:
        texts = [t or "" for t in texts]
        keys = [self._key(kind, t) for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        missing: List[int] = []
        with self._lock:
            for i, k in enumerate(keys):
                v = self._lookup(k)
                if v is None:
                    missing.append(i)
                else:
                    out[i] = v
        if missing:
            uniq: Dict[str, int] = {}
            for i in missing:
                uniq.setdefault(texts[i], len(uniq))
            fn = self.base.encode_queries if kind == "query" else self.base.encode
            vecs = fn(list(uniq))
            with self._lock:
                for i in missing:
                    v = vecs[uniq[texts[i]]]
                    out[i] = v
                    self._store(keys[i], v)
        return out

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._run(texts, "doc")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._run(texts, "query")


def build_embedder(config: Optional[EmbeddingConfig] = None) -> Embedder:
    """Instantiate the embedder described by ``config`` (wrapped in a cache).

    Raises :class:`MissingDependencyError` (with the install command) when the
    backend's optional dependencies are missing.
    """
    config = config or EmbeddingConfig()
    backend = (config.backend or "hashing").lower()
    if backend == "hashing":
        base: Embedder = HashingEmbedder(dim=config.dim)
        return CachedEmbedder(base, cache_dir=None, namespace="hashing")
    if backend == "specter2":  # pragma: no cover - model download
        base = Specter2Embedder(
            model_name=config.model_name,
            adapter=config.adapter,
            query_adapter=config.query_adapter,
            batch_size=config.batch_size,
            max_length=config.max_length,
            device=config.device,
        )
    elif backend in {"sentence-transformers", "st"}:  # pragma: no cover - model download
        base = SentenceTransformerEmbedder(config.model_name, config.batch_size, config.device)
    else:
        raise ValueError(f"Unknown embedding backend {config.backend!r}")
    ns = hashlib.sha1(f"{backend}:{config.model_name}:{config.adapter}".encode()).hexdigest()[:12]
    return CachedEmbedder(base, cache_dir=config.cache_dir, namespace=ns)
