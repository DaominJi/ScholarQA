"""Keyword search baseline (Tables 3-5, "Keyword search").

Lexical keyword search over paper titles and abstracts, the way a scholarly
search engine's keyword box works: the query's keywords
(:func:`scholarpath.text.keywords`, i.e. its most frequent non-stop-word
terms) are OR-ed, every paper whose title or abstract contains at least one
keyword matches, and matches are ranked by the number of distinct query
keywords they contain, ties broken by the BM25 score of the keywords over
title + abstract. The matching abstracts ("title. abstract" units) are packed
to the budget B.

No LLM call is made during retrieval, so a keyword-search QA query costs
exactly one LLM call (the composition call; Table 5).
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple, Union

import numpy as np

from scholarpath.baselines.bm25 import BM25Index
from scholarpath.baselines.common import (
    ABSTRACT,
    DEFAULT_MAX_CANDIDATES,
    Candidate,
    PapersLike,
    PassagePool,
    QueryLike,
    as_query,
)
from scholarpath.baselines.packing import BudgetPacker
from scholarpath.retrieval.result import RetrievalResult
from scholarpath.text import keywords


class KeywordSearchRetriever:
    """Boolean-OR keyword match over titles + abstracts, packed to B."""

    def __init__(
        self,
        source: Union[PassagePool, PapersLike],
        num_keywords: int = 8,
        k1: float = 0.9,
        b: float = 0.4,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        name: str = "Keyword search",
    ):
        if isinstance(source, PassagePool):
            self.pool = source.filter([ABSTRACT])
        else:
            self.pool = PassagePool.from_papers(source, include_chunks=False)
        self.num_keywords = int(num_keywords)
        self.max_candidates = int(max_candidates)
        self.name = name
        self.index = BM25Index(k1, b).fit(self.pool.search_texts())

    def query_keywords(self, text: str) -> List[str]:
        return keywords(text, k=self.num_keywords)

    def search(self, text: str, k: Optional[int] = None) -> List[Tuple[int, int, float]]:
        """Ranked ``(unit index, matched keyword count, BM25 score)`` triples."""
        kws = self.query_keywords(text)
        if not kws:
            return []
        matched = self.index.match_counts(kws)
        hits = np.flatnonzero(matched > 0)
        if hits.size == 0:
            return []
        bm25 = self.index.scores(kws)
        order = hits[np.lexsort((hits, -bm25[hits], -matched[hits]))]
        if k is not None:
            order = order[: int(k)]
        return [(int(i), int(matched[i]), float(bm25[i])) for i in order]

    def retrieve(self, query: QueryLike, budget: int) -> RetrievalResult:
        t0 = time.perf_counter()
        q = as_query(query)
        hits = self.search(q.text, self.max_candidates)
        packer = BudgetPacker(budget)
        packer.extend(Candidate(self.pool[i], m + s / (1.0 + s), "keyword") for i, m, s in hits)
        return packer.result(q.query_id, titles=self.pool.titles,
                             stats={"method": self.name, "keywords": self.query_keywords(q.text),
                                    "matches": len(hits)},
                             latency_s=time.perf_counter() - t0)
