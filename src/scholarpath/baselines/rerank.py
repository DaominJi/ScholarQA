"""Dense retrieval + LLM listwise reranker (Table 2, "Dense + reranker").

SPECTER2 retrieves the top-N candidate passages (default N = 50, after
deduplication), and DeepSeek-V4.0-Flash reranks them listwise with the
RankGPT permutation-generation prompt [71] in a sliding window (window 20,
step 10) that moves from the bottom of the list to the top, so that strong
passages can bubble up to the first positions. The reranked list is then
packed to the budget B; if the N reranked passages do not fill B, packing
continues with the remaining dense ranks.

With N = 50 this costs four reranking calls per query (windows
[30, 50), [20, 40), [10, 30), [0, 20)), billed to the ``rerank`` component.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

from scholarpath.baselines.common import Candidate, QueryLike, as_query
from scholarpath.baselines.dense import DenseRetriever
from scholarpath.baselines.packing import BudgetPacker
from scholarpath.llm.base import LLMClient
from scholarpath.prompts.baselines import listwise_rerank_request, parse_permutation
from scholarpath.retrieval.result import RetrievalResult


def sliding_windows(n: int, window: int = 20, step: int = 10) -> List[tuple]:
    """Window ``[start, end)`` positions of RankGPT's back-to-front pass.

    Unlike RankGPT's reference loop, the last window is clamped to start at 0
    so that the top positions are always reranked even when ``n`` is not
    aligned with the step.
    """
    if n <= 0:
        return []
    window = max(2, int(window))
    step = max(1, int(step))
    out = []
    end, start = n, max(0, n - window)
    while True:
        out.append((start, end))
        if start == 0:
            break
        end -= step
        start = max(0, start - step)
    return out


def sliding_window_rerank(
    llm: LLMClient,
    query: str,
    texts: Sequence[str],
    window: int = 20,
    step: int = 10,
    max_passage_tokens: int = 300,
    component: Optional[str] = "rerank",
) -> List[int]:
    """Rerank ``texts`` for ``query``; returns the new order as original indices."""
    order = list(range(len(texts)))
    for start, end in sliding_windows(len(texts), window, step):
        segment = order[start:end]
        if len(segment) < 2:
            continue
        req = listwise_rerank_request(query, [texts[i] for i in segment], max_passage_tokens, component=component)
        resp = llm.complete(req)
        perm = parse_permutation(resp.text, len(segment))
        order[start:end] = [segment[j] for j in perm]
    return order


class DenseRerankRetriever:
    """SPECTER2 top-N + RankGPT listwise reranking, packed to B."""

    def __init__(
        self,
        dense: DenseRetriever,
        llm: LLMClient,
        top_n: int = 50,
        window: int = 20,
        step: int = 10,
        max_passage_tokens: int = 300,
        name: str = "Dense + reranker",
    ):
        self.dense = dense
        self.pool = dense.pool
        self.llm = llm
        self.top_n = int(top_n)
        self.window = int(window)
        self.step = int(step)
        self.max_passage_tokens = int(max_passage_tokens)
        self.name = name

    def ensure_built(self) -> float:
        return self.dense.ensure_built()

    def retrieve(self, query: QueryLike, budget: int) -> RetrievalResult:
        t0 = time.perf_counter()
        index_s = self.ensure_built()
        q = as_query(query)
        hits = self.dense.search(q.text, self.dense.max_candidates)
        # Deduplicate before reranking so no window slot is wasted on copies.
        head: List[int] = []
        seen = set()
        tail: List[int] = []
        for i, _ in hits:
            dk = self.pool[i].dedup_key
            if dk in seen:
                continue
            seen.add(dk)
            (head if len(head) < self.top_n else tail).append(i)
        scores = dict(hits)
        before = self.llm.meter.component("rerank").calls
        order = sliding_window_rerank(
            self.llm, q.text, [self.pool[i].search_text for i in head], self.window, self.step,
            self.max_passage_tokens,
        )
        reranked = [head[j] for j in order]
        packer = BudgetPacker(budget)
        n = len(reranked)
        packer.extend(
            [Candidate(self.pool[i], float(n - r), "rerank") for r, i in enumerate(reranked)]
            + [Candidate(self.pool[i], scores[i], "dense") for i in tail]
        )
        calls = self.llm.meter.component("rerank").calls - before
        return packer.result(q.query_id, titles=self.pool.titles,
                             stats={"method": self.name, "reranked": n, "rerank_calls": calls,
                                    "index_build_s": index_s},
                             latency_s=max(0.0, time.perf_counter() - t0 - index_s))
