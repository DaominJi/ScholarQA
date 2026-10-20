"""End-to-end budgeted QA for the baselines (Section 5.3, Tables 3-5, Fig. 6).

Every budgeted method -- Keyword search, Dense retrieval, LightRAG and
ScholarPath -- packs at most B context tokens (default 8,192) and invokes
the answer LLM in a *single* composition call, built by the shared
:func:`scholarpath.prompts.answer.compose_answer_request`, so that systems
differ only in the context they retrieve. The unassisted answer LLM
("Answer LLM, no retrieval", Table 4) makes the same composition call with an
empty context.

Citations are restricted to papers present in the context
(:func:`~scholarpath.prompts.answer.parse_answer` with ``valid_ids``).

Per-query efficiency (Table 5) is read from the cost-meter diff of the
query: every LLM call made while answering (reranking, LightRAG keyword
extraction, composition, ...) counts; offline indexing (``index_build``,
e.g. LightRAG graphs / RAPTOR trees built on a per-query candidate pool, or
a lazily encoded dense index) and judge calls are excluded and reported
separately in ``Answer.metadata["excluded_usage"]``. Consistently, the wall
time a retriever spent building its index (``RetrievalResult.stats
["index_build_s"]``) is subtracted from ``Answer.latency_s`` and reported as
``Answer.metadata["excluded_latency_s"]``.

Constructor contract (docs/ARCHITECTURE.md Section 4.4):
``BudgetedQA(retriever, llm, config)`` with a
:class:`~scholarpath.config.RetrievalConfig` (``budget`` = B,
``answer_max_tokens``); ``budget=`` / ``answer_max_tokens=`` keywords
override the config.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Sequence

from scholarpath.baselines.common import (
    DEFAULT_BUDGET,
    EXCLUDED_QUERY_COMPONENTS,
    BudgetedRetriever,
    QueryLike,
    as_query,
    estimate_latency,
    index_build_seconds,
    split_usage,
)
from scholarpath.config import RetrievalConfig
from scholarpath.llm.base import LLMClient
from scholarpath.llm.parsing import JSONParseError
from scholarpath.prompts.answer import compose_answer_request, parse_answer
from scholarpath.retrieval.result import RetrievalResult
from scholarpath.types import Answer, Query

log = logging.getLogger(__name__)


def compose(
    llm: LLMClient,
    query: Query,
    retrieval: Optional[RetrievalResult],
    answer_max_tokens: int = 1536,
) -> Dict[str, Any]:
    """The single composition call over a (possibly empty) retrieved context.

    Returns ``{"text", "items", "error"}``; citations are restricted to the
    papers of the context (none when there is no context).
    """
    context = retrieval.context if retrieval is not None else ""
    sources = retrieval.sources if retrieval is not None else []
    valid = [s["paper_id"] for s in sources]
    req = compose_answer_request(query.text, context, query.kind, sources=sources, max_tokens=answer_max_tokens)
    try:
        obj, _ = llm.complete_json(req, expect=dict)
    except JSONParseError as err:
        log.warning("composition output for %s was not valid JSON: %s", query.query_id, err)
        return {"text": "", "items": [], "error": str(err)}
    text, items = parse_answer(obj, valid_ids=valid)
    return {"text": text, "items": items, "error": None}


class _MeteredQA:
    """Shared usage / latency bookkeeping of the QA wrappers."""

    name: str = "qa"

    def __init__(self, llm: LLMClient, exclude_components: Sequence[str] = EXCLUDED_QUERY_COMPONENTS):
        self.llm = llm
        self.exclude_components = tuple(exclude_components)

    def _finish(
        self,
        query: Query,
        composed: Dict[str, Any],
        retrieval: Optional[RetrievalResult],
        before: Dict[str, Dict[str, float]],
        t0: float,
        budget: Optional[int],
    ) -> Answer:
        wall = time.perf_counter() - t0
        diff = self.llm.meter.diff(before)
        counted, excluded = split_usage(diff, self.exclude_components)
        index_s = min(wall, index_build_seconds(retrieval))  # offline indexing triggered by this query
        meta: Dict[str, Any] = {
            "system": self.name,
            "budget": budget,
            "usage_by_component": {k: v.to_dict() for k, v in diff.items()},
            "excluded_usage": excluded.to_dict(),
            "excluded_latency_s": index_s,
            "wall_clock_s": wall,
            "llm_latency_s": counted.latency_s,
        }
        if composed.get("error"):
            meta["error"] = composed["error"]
        if retrieval is not None:
            from scholarpath.retrieval.compose import context_passages

            meta["retrieval_stats"] = retrieval.stats
            meta["context_papers"] = [s["paper_id"] for s in retrieval.sources]
            meta["retrieval_latency_s"] = retrieval.latency_s
            meta["passages"] = context_passages(retrieval)  # what the composer saw, for citation metrics
        return Answer(
            query_id=query.query_id,
            text=composed["text"],
            items=composed["items"],
            retrieved_papers=list(retrieval.ranked_papers) if retrieval is not None else [],
            context_tokens=retrieval.context_tokens if retrieval is not None else 0,
            llm_calls=counted.calls,
            input_tokens=counted.input_tokens,
            output_tokens=counted.output_tokens,
            cost_usd=counted.cost_usd,
            latency_s=estimate_latency(wall - index_s, counted),
            metadata=meta,
        )


def _check_config(config: Any) -> Optional[RetrievalConfig]:
    if config is None or isinstance(config, RetrievalConfig):
        return config
    raise TypeError(
        f"config must be a scholarpath.config.RetrievalConfig or None, not {type(config).__name__}; "
        "pass the budget as budget=<int>"
    )


def _resolve(value: Optional[int], config: Optional[RetrievalConfig], attr: str, default: int) -> int:
    if value is not None:
        return int(value)
    return int(getattr(config, attr)) if config is not None else int(default)


class BudgetedQA(_MeteredQA):
    """Retrieve <= B tokens with a budgeted retriever, then compose once.

    ``BudgetedQA(retriever, llm, config)`` follows the module contract; the
    keyword-only ``budget`` / ``answer_max_tokens`` override ``config``
    (defaults: B = 8,192, 1,536 answer tokens).
    """

    def __init__(
        self,
        retriever: BudgetedRetriever,
        llm: LLMClient,
        config: Optional[RetrievalConfig] = None,
        *,
        budget: Optional[int] = None,
        answer_max_tokens: Optional[int] = None,
        exclude_components: Sequence[str] = EXCLUDED_QUERY_COMPONENTS,
        name: Optional[str] = None,
    ):
        super().__init__(llm, exclude_components)
        config = _check_config(config)
        self.retriever = retriever
        self.config = config
        self.budget = _resolve(budget, config, "budget", DEFAULT_BUDGET)
        self.answer_max_tokens = _resolve(answer_max_tokens, config, "answer_max_tokens", 1536)
        self.name = name or getattr(retriever, "name", type(retriever).__name__)

    def retrieve(self, query: QueryLike, budget: Optional[int] = None) -> RetrievalResult:
        b = int(budget if budget is not None else self.budget)
        result = self.retriever.retrieve(as_query(query), b)
        if result.context_tokens > b:  # defensive: every baseline packs strictly within B
            raise RuntimeError(f"{self.name} exceeded the context budget: {result.context_tokens} > {b}")
        return result

    def answer(self, query: QueryLike, budget: Optional[int] = None) -> Answer:
        q = as_query(query)
        b = int(budget if budget is not None else self.budget)
        before = self.llm.meter.snapshot()
        t0 = time.perf_counter()
        retrieval = self.retrieve(q, b)
        composed = compose(self.llm, q, retrieval, self.answer_max_tokens)
        return self._finish(q, composed, retrieval, before, t0, b)


class NoRetrievalQA(_MeteredQA):
    """"Answer LLM, no retrieval" (Table 4): one composition call, empty context.

    ``NoRetrievalQA(llm, config)`` mirrors :class:`BudgetedQA` (only
    ``config.answer_max_tokens`` is used).
    """

    def __init__(
        self,
        llm: LLMClient,
        config: Optional[RetrievalConfig] = None,
        *,
        answer_max_tokens: Optional[int] = None,
        exclude_components: Sequence[str] = EXCLUDED_QUERY_COMPONENTS,
        name: str = "Answer LLM, no retrieval",
    ):
        super().__init__(llm, exclude_components)
        config = _check_config(config)
        self.config = config
        self.answer_max_tokens = _resolve(answer_max_tokens, config, "answer_max_tokens", 1536)
        self.name = name

    def answer(self, query: QueryLike, budget: Optional[int] = None) -> Answer:
        q = as_query(query)
        before = self.llm.meter.snapshot()
        t0 = time.perf_counter()
        composed = compose(self.llm, q, None, self.answer_max_tokens)
        return self._finish(q, composed, None, before, t0, budget)
