"""Okapi BM25 retrieval baseline [59] (Table 2, "BM25").

Scores every unit d of the passage pool for a query q by

    BM25(q, d) = sum_{t in q} qtf_t * idf_t * tf_{t,d} (k1 + 1) / (tf_{t,d} + k1 (1 - b + b |d| / avgdl)),
    idf_t      = ln(1 + (N - df_t + 0.5) / (df_t + 0.5)),

with the Lucene / Pyserini defaults k1 = 0.9, b = 0.4 (non-negative Lucene
idf; repeated query terms count once per occurrence, as in a Lucene bag-of-
words query). Documents and queries are tokenized with
:func:`scholarpath.text.tokenize` with stop-word removal.

The per-(document, term) weights are precomputed into a sparse CSC matrix
W (scipy.sparse), so a query is one sparse column slice and a
matrix-vector product: ``scores = W[:, q_terms] @ qtf`` -- fully vectorized
and fast enough for ~10^6 passages (148K papers with their section chunks).
"""

from __future__ import annotations

import json
import logging
import time
from array import array
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

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
from scholarpath.retrieval.result import RetrievalResult
from scholarpath.text import tokenize

log = logging.getLogger(__name__)

Tokenizer = Callable[[str], List[str]]


def default_tokenizer(text: str) -> List[str]:
    return tokenize(text, remove_stopwords=True)


class BM25Index:
    """Sparse Okapi BM25 index over a list of texts."""

    def __init__(self, k1: float = 0.9, b: float = 0.4, tokenizer: Optional[Tokenizer] = None):
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer: Tokenizer = tokenizer or default_tokenizer
        self.vocab: dict = {}
        self.W: Optional[sparse.csc_matrix] = None  # (n_docs, |V|) BM25 weights
        self.idf: Optional[np.ndarray] = None
        self.doc_len: Optional[np.ndarray] = None
        self.avgdl: float = 0.0

    @property
    def num_docs(self) -> int:
        return 0 if self.W is None else int(self.W.shape[0])

    def fit(self, texts: Iterable[str], log_every: int = 200_000) -> "BM25Index":
        vocab = self.vocab = {}
        indptr = array("q", [0])
        indices = array("i")
        tfs = array("f")
        lens = array("f")
        t0 = time.perf_counter()
        for n, text in enumerate(texts, 1):
            counts = Counter(self.tokenizer(text or ""))
            for tok, c in counts.items():
                j = vocab.get(tok)
                if j is None:
                    j = vocab[tok] = len(vocab)
                indices.append(j)
                tfs.append(c)
            indptr.append(len(indices))
            lens.append(sum(counts.values()))
            if log_every and n % log_every == 0:
                log.info("BM25: tokenized %d documents (%.0fs)", n, time.perf_counter() - t0)
        n_docs, n_terms = len(lens), len(vocab)
        ip = np.frombuffer(indptr, dtype=np.int64)
        ind = np.frombuffer(indices, dtype=np.int32)
        tf = np.frombuffer(tfs, dtype=np.float32)
        dl = np.frombuffer(lens, dtype=np.float32).copy()
        df = np.bincount(ind, minlength=n_terms).astype(np.float64)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.doc_len = dl
        self.avgdl = float(dl.mean()) if n_docs else 0.0
        row_len = np.repeat(dl, np.diff(ip)) if n_docs else np.zeros(0, dtype=np.float32)
        norm = self.k1 * (1.0 - self.b + self.b * row_len / max(self.avgdl, 1e-9))
        w = self.idf[ind] * tf * (self.k1 + 1.0) / (tf + norm)
        self.W = sparse.csr_matrix((w.astype(np.float32), ind, ip), shape=(n_docs, n_terms)).tocsc()
        return self

    # ---------------------------------------------------------- queries
    def query_terms(self, query: Union[str, Sequence[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """(term ids, query term frequencies) of the in-vocabulary query terms."""
        toks = self.tokenizer(query) if isinstance(query, str) else list(query)
        counts = Counter(t for t in toks if t in self.vocab)
        ids = np.array([self.vocab[t] for t in counts], dtype=np.int64)
        qtf = np.array([counts[t] for t in counts], dtype=np.float32)
        return ids, qtf

    def scores(self, query: Union[str, Sequence[str]]) -> np.ndarray:
        """BM25 score of every document (zeros if no query term is in the vocabulary)."""
        if self.W is None:
            raise RuntimeError("BM25Index is not fitted")
        ids, qtf = self.query_terms(query)
        if ids.size == 0:
            return np.zeros(self.num_docs, dtype=np.float32)
        return np.asarray(self.W[:, ids] @ qtf, dtype=np.float32).ravel()

    def match_counts(self, terms: Sequence[str]) -> np.ndarray:
        """Number of distinct ``terms`` occurring in each document."""
        if self.W is None:
            raise RuntimeError("BM25Index is not fitted")
        ids = sorted({self.vocab[t] for t in terms if t in self.vocab})
        if not ids:
            return np.zeros(self.num_docs, dtype=np.int32)
        sub = self.W[:, ids]
        return np.asarray((sub > 0).sum(axis=1), dtype=np.int32).ravel()

    # --------------------------------------------------------------- io
    def save(self, directory: Union[str, Path]) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(d / "bm25_weights.npz", self.W)
        np.save(d / "bm25_idf.npy", self.idf)
        np.save(d / "bm25_doc_len.npy", self.doc_len)
        (d / "bm25_vocab.json").write_text(json.dumps(self.vocab), encoding="utf-8")
        (d / "bm25_params.json").write_text(json.dumps({"k1": self.k1, "b": self.b, "avgdl": self.avgdl}))

    @classmethod
    def load(cls, directory: Union[str, Path], tokenizer: Optional[Tokenizer] = None) -> "BM25Index":
        d = Path(directory)
        params = json.loads((d / "bm25_params.json").read_text())
        idx = cls(params["k1"], params["b"], tokenizer)
        idx.W = sparse.load_npz(d / "bm25_weights.npz").tocsc()
        idx.idf = np.load(d / "bm25_idf.npy")
        idx.doc_len = np.load(d / "bm25_doc_len.npy")
        idx.vocab = json.loads((d / "bm25_vocab.json").read_text(encoding="utf-8"))
        idx.avgdl = float(params["avgdl"])
        return idx


class BM25Retriever:
    """BM25 over the passage pool, packed to the budget B (Table 2 "BM25")."""

    def __init__(
        self,
        pool: PassagePool,
        k1: float = 0.9,
        b: float = 0.4,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        index: Optional[BM25Index] = None,
        name: str = "BM25",
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.pool = pool
        self.name = name
        self.max_candidates = int(max_candidates)
        if index is not None:
            if index.num_docs != len(pool):
                raise ValueError(f"BM25 index has {index.num_docs} docs but the pool has {len(pool)} units")
            self.index = index
        else:
            self.index = BM25Index(k1, b, tokenizer).fit(pool.search_texts())

    def search(self, text: str, k: int) -> List[Tuple[int, float]]:
        """Top-``k`` (unit index, score) pairs with a positive score."""
        scores = self.index.scores(text)
        top = top_k_indices(scores, k, min_score=0.0)
        return [(int(i), float(scores[i])) for i in top]

    def top_papers(self, query: QueryLike, k: int) -> List[str]:
        q = as_query(query)
        hits = self.search(q.text, max(4 * k, self.max_candidates))
        return distinct_papers(self.pool, (i for i, _ in hits), k)

    def retrieve(self, query: QueryLike, budget: int) -> RetrievalResult:
        t0 = time.perf_counter()
        q = as_query(query)
        hits = self.search(q.text, self.max_candidates)
        packer = BudgetPacker(budget)
        packer.extend(Candidate(self.pool[i], s, "bm25") for i, s in hits)
        return packer.result(q.query_id, titles=self.pool.titles,
                             stats={"method": self.name, "hits": len(hits)},
                             latency_s=time.perf_counter() - t0)
