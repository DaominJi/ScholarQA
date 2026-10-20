"""SimulatedAgent: an offline, deterministic multi-step search agent.

**This is NOT one of the paper's baselines.** It exists to exercise the
unbudgeted-agent evaluation path (uniform task prompt, multi-call usage
accounting, robust output parsing, Answer records) in smoke tests and
offline dry runs, where PaperQA2 / OpenScholar / AgenticScholar / Codex are
not available. Its prompts are answered deterministically by
:class:`~scholarpath.llm.MockLLM`, but it also runs against a real backend.

Loop (all calls billed to the ``agent`` component):

1. *plan* -- one LLM call proposes search queries (``baselines.agent_plan``);
2. *search* -- each query is run against the corpus with BM25 and dense tools
   (fused by reciprocal rank fusion), collecting new passages;
3. *reflect* -- one LLM call per step decides whether to stop or which
   follow-up queries to issue (``baselines.agent_reflect``);
4. *answer* -- the uniform task prompt plus the collected evidence
   (``baselines.agent_task``) produces the final JSON records.

So a query costs at least three LLM calls, and the collected evidence is
not budgeted (only capped by ``max_evidence_tokens``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from scholarpath.baselines.agents.base import AgentAdapter, AgentConfig, AgentRun, LLMEndpoint
from scholarpath.baselines.bm25 import BM25Retriever
from scholarpath.baselines.common import PassagePool, reciprocal_rank_fusion, split_usage
from scholarpath.baselines.dense import DenseRetriever
from scholarpath.embeddings import Embedder
from scholarpath.llm.base import LLMClient
from scholarpath.llm.parsing import JSONParseError
from scholarpath.prompts.baselines import agent_plan_request, agent_reflect_request, agent_task_request
from scholarpath.types import Query


class SimulatedAgent(AgentAdapter):
    """Plan -> search (BM25 + dense) -> reflect -> answer, with an in-process LLM."""

    name = "SimulatedAgent"
    uses_endpoint = False

    def __init__(
        self,
        pool: PassagePool,
        llm: LLMClient,
        embedder: Optional[Embedder] = None,
        config: Optional[AgentConfig] = None,
        max_steps: int = 3,
        max_queries: int = 3,
        hits_per_search: int = 8,
        max_evidence_tokens: int = 16000,
        dense: Optional[DenseRetriever] = None,
        bm25: Optional[BM25Retriever] = None,
    ):
        super().__init__(config or AgentConfig(meter_with_proxy=False), valid_ids=pool.paper_ids)
        self.pool = pool
        self.llm = llm
        self.max_steps = int(max_steps)
        self.max_queries = int(max_queries)
        self.hits_per_search = int(hits_per_search)
        self.max_evidence_tokens = int(max_evidence_tokens)
        self.bm25 = bm25 or BM25Retriever(pool)
        self.dense = dense or (DenseRetriever(pool, embedder) if embedder is not None else None)

    def check_available(self) -> None:
        """Always available; builds the dense search tool's index before timing starts."""
        if self.dense is not None:
            self.dense.ensure_built()

    def _search(self, text: str) -> List[int]:
        rankings = [[i for i, _ in self.bm25.search(text, self.hits_per_search)]]
        if self.dense is not None:
            rankings.append([i for i, _ in self.dense.search(text, self.hits_per_search)])
        return reciprocal_rank_fusion(rankings)[: self.hits_per_search]

    def run(self, query: Query, prompt: str, endpoint: Optional[LLMEndpoint]) -> AgentRun:
        before = self.llm.meter.snapshot()
        try:
            plan, _ = self.llm.complete_json(agent_plan_request(query.text, self.max_queries), expect=dict)
        except JSONParseError:
            plan = {}
        queue: List[str] = [str(q) for q in (plan.get("queries") or [query.text])][: self.max_queries]
        issued: List[str] = []
        evidence: Dict[int, None] = {}
        tokens = 0
        for step in range(self.max_steps):
            if not queue:
                break
            text = queue.pop(0)
            issued.append(text)
            for i in self._search(text):
                if i in evidence:
                    continue
                cost = self.pool[i].cost
                if tokens + cost > self.max_evidence_tokens:
                    continue
                evidence[i] = None
                tokens += cost
            found = [self._record(i) for i in evidence]
            try:
                refl, _ = self.llm.complete_json(
                    agent_reflect_request(query.text, step, self.max_steps, found), expect=dict
                )
            except JSONParseError:
                refl = {"done": True}
            if refl.get("done"):
                break
            queue.extend(str(q) for q in (refl.get("next_queries") or []) if str(q) not in issued)
        records = [self._record(i) for i in evidence]
        final = self.llm.complete(agent_task_request(query.text, query.kind, records, self.config.max_records))
        counted, _ = split_usage(self.llm.meter.diff(before), exclude=("judge",))
        return AgentRun(
            output_text=final.text,
            llm_calls=counted.calls,
            input_tokens=counted.input_tokens,
            output_tokens=counted.output_tokens,
            cost_usd=counted.cost_usd,
            metadata={"search_queries": issued, "evidence_units": len(records), "evidence_tokens": tokens},
        )

    def _record(self, i: int) -> Dict[str, Any]:
        p = self.pool[i]
        return {"paper_id": p.paper_id, "title": self.pool.titles.get(p.paper_id, p.title), "text": p.body}
