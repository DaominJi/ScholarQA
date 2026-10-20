"""Retrieval and QA baselines of Section 5.3.

Budgeted methods strictly enforce the context token budget B (Definition 2):

=========================  ================================================  =======================
name (paper)               class                                             used in
=========================  ================================================  =======================
BM25 [59]                  :class:`~.bm25.BM25Retriever`                      Table 2, Fig. 5
SPECTER [10]               :class:`~.dense.DenseRetriever` (SPECTER v1)       Table 2, Fig. 5
Dense + reranker [71]      :class:`~.rerank.DenseRerankRetriever`             Table 2, Fig. 5
RAPTOR [61]                :class:`~.raptor.RaptorRetriever`                  Table 2, Fig. 5
LightRAG [17]              :class:`~.lightrag.LightRAGRetriever`              Tables 2-5, Figs. 5-6
Keyword search             :class:`~.keyword.KeywordSearchRetriever`          Tables 3-5, Fig. 6
Dense retrieval            :class:`~.dense.DenseRetriever` (SPECTER2)         Tables 3-5, Fig. 6
Answer LLM, no retrieval   :class:`~.qa.NoRetrievalQA`                        Table 4
=========================  ================================================  =======================

:class:`~.qa.BudgetedQA` wraps any budgeted retriever with the shared single
composition call. Unbudgeted agent systems (PaperQA2, OpenScholar,
AgenticScholar, Codex) are adapted in :mod:`scholarpath.baselines.agents`.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from scholarpath.baselines.bm25 import BM25Index, BM25Retriever
from scholarpath.baselines.common import (
    ABSTRACT,
    CHUNK,
    DEFAULT_BUDGET,
    ENTITY,
    RELATION,
    SUMMARY,
    BudgetedRetriever,
    Candidate,
    Passage,
    PassagePool,
    as_query,
    ensure_built,
)
from scholarpath.baselines.dense import DenseRetriever, build_dense_embedder
from scholarpath.baselines.keyword import KeywordSearchRetriever
from scholarpath.baselines.lightrag import KnowledgeGraph, LightRAGConfig, LightRAGRetriever
from scholarpath.baselines.packing import BudgetPacker, build_result, pack_to_budget, render_context
from scholarpath.baselines.qa import BudgetedQA, NoRetrievalQA, compose
from scholarpath.baselines.raptor import RaptorConfig, RaptorRetriever, RaptorTree
from scholarpath.baselines.rerank import DenseRerankRetriever, sliding_window_rerank

#: Canonical method keys accepted by :func:`make_retriever`.
RETRIEVER_NAMES = ("bm25", "specter", "dense", "dense_rerank", "raptor", "lightrag", "keyword")


def make_retriever(
    method: str,
    pool: PassagePool,
    llm: Any = None,
    embedder: Any = None,
    dense: Optional[DenseRetriever] = None,
    **kwargs: Any,
) -> BudgetedRetriever:
    """Construct a budgeted baseline by key (see :data:`RETRIEVER_NAMES`).

    ``embedder`` is the dense model of the method (SPECTER for ``"specter"``,
    SPECTER2 otherwise); ``dense`` lets several methods share one built
    :class:`DenseRetriever` (first stage of the reranker, candidate pool of
    RAPTOR / LightRAG). Extra keyword arguments go to the constructor (and
    ``config`` objects for RAPTOR / LightRAG).
    """
    m = re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")
    if m == "bm25":
        return BM25Retriever(pool, **kwargs)
    if m in ("keyword", "keyword_search"):
        return KeywordSearchRetriever(pool, **kwargs)
    if m in ("specter", "dense", "specter2", "dense_retrieval"):
        if embedder is None:
            raise ValueError(f"{method} needs an embedder")
        default = "SPECTER" if m == "specter" else "Dense retrieval"
        return DenseRetriever(pool, embedder, name=kwargs.pop("name", default), **kwargs)
    if m in ("dense_rerank", "dense_reranker", "rerank"):
        if llm is None:
            raise ValueError("dense_rerank needs an LLM")
        if dense is None and embedder is None:
            raise ValueError("dense_rerank needs an embedder (SPECTER2) or a built dense retriever (dense=...)")
        dense = dense or DenseRetriever(pool, embedder)
        return DenseRerankRetriever(dense, llm, **kwargs)
    if m == "raptor":
        if llm is None or embedder is None:
            raise ValueError("raptor needs an LLM and an embedder")
        return RaptorRetriever(pool, llm, embedder, candidate_retriever=kwargs.pop("candidate_retriever", dense),
                               **kwargs)
    if m == "lightrag":
        if llm is None or embedder is None:
            raise ValueError("lightrag needs an LLM and an embedder")
        return LightRAGRetriever(pool, llm, embedder, candidate_retriever=kwargs.pop("candidate_retriever", dense),
                                 **kwargs)
    raise ValueError(f"Unknown baseline {method!r}; expected one of {RETRIEVER_NAMES}")


__all__ = [
    "ABSTRACT",
    "BM25Index",
    "BM25Retriever",
    "BudgetPacker",
    "BudgetedQA",
    "BudgetedRetriever",
    "CHUNK",
    "Candidate",
    "DEFAULT_BUDGET",
    "DenseRerankRetriever",
    "DenseRetriever",
    "ENTITY",
    "KeywordSearchRetriever",
    "KnowledgeGraph",
    "LightRAGConfig",
    "LightRAGRetriever",
    "NoRetrievalQA",
    "Passage",
    "PassagePool",
    "RELATION",
    "RETRIEVER_NAMES",
    "RaptorConfig",
    "RaptorRetriever",
    "RaptorTree",
    "SUMMARY",
    "as_query",
    "build_dense_embedder",
    "build_result",
    "compose",
    "ensure_built",
    "make_retriever",
    "pack_to_budget",
    "render_context",
    "sliding_window_rerank",
]
