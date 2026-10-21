"""ScholarPath query answering: Algorithm 2 end to end (Section 4).

    Step 1  analyze_query           one lightweight LLM call (optional)
    Step 2  ground                  scopes (u_q, m_q), query-local anchors
    Step 3  build_search_space      aspect instances, Eq. (5)-(6), top-level regions
    Step 4  BoundGuidedSearch       hierarchy-bounded subtree pruning (Eq. 7, 8, 12)
    Step 5  (inside the search)     granularity-adaptive scoring and chunk expansion (Eq. 9)
    Step 6  assemble_context +      one hierarchical context and ONE composition call
            compose_answer

The search itself issues no LLM call; the only calls are query
understanding (optional) and composition (Table 5).

Ablations (Table 7)
-------------------
* "w/o temporal link prediction": pass the untrained
  :class:`~scholarpath.retrieval.link_prediction.EmbeddingLinkPredictor`
  (text similarity of concept descriptions) instead of the temporally trained
  predictor; candidates are still scored, by the untrained prior.
* "w/o bound-guided search": ``config.use_bound_guided_search = False``
  (unpruned scan that scores every candidate node in scope; identical
  selections, Proposition 7; :class:`~scholarpath.retrieval.search.ExhaustiveGreedy`
  is the fully enumerating reference).
* "w/o role-aware granularity": ``config.use_role_aware_granularity = False``.
* "w/o shared-cost reuse update": ``config.use_shared_cost_update = False``.
* ``config.use_link_prediction = False`` is a further ablation that drops the
  prior altogether (eta_hat = phi).

``RetrievalResult.stats["scored_candidates"]`` is the "Scored" column.
``stats["studied_pair"]`` tells whether the asked pair itself is studied
(J(u_q, m_q) non-empty; ``None`` for discovery), and ``stats["tokenizer"]``
names the token counter behind the budget B and the node costs c(x).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Union

from scholarpath.config import RetrievalConfig
from scholarpath.embeddings import Embedder
from scholarpath.index.semantic_index import SemanticIndex, tokenizer_name
from scholarpath.llm import LLMClient, sum_usage
from scholarpath.retrieval.compose import ContextRenderer, NodeCostTable, assemble_context, compose_answer
from scholarpath.retrieval.grounding import Grounding, ground
from scholarpath.retrieval.link_prediction import EmbeddingLinkPredictor, LinkPredictor
from scholarpath.retrieval.query_understanding import QueryAnalysis, analyze_query
from scholarpath.retrieval.result import ESTABLISHED, HYPOTHESIS, RetrievalResult, RetrievedUnit
from scholarpath.retrieval.search import SearchResult, SearchSpace, Selection, build_search_space, make_search
from scholarpath.retrieval.units import ComplementaryPathSet, DirectPath
from scholarpath.text import count_tokens
from scholarpath.types import Answer, Query

log = logging.getLogger(__name__)

_MAX_SCORE = 1e12  # finite stand-in for delta = +inf (zero marginal cost) in serialized results


class ScholarPathRetriever:
    """Budgeted support-path retrieval over a multi-granularity semantic index."""

    name = "scholarpath"

    def __init__(
        self,
        index: SemanticIndex,
        llm: Optional[LLMClient],
        embedder: Embedder,
        config: Optional[RetrievalConfig] = None,
        link_predictor: Optional[LinkPredictor] = None,
    ):
        self.index = index
        self.llm = llm
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self._link_predictor = link_predictor
        if link_predictor is not None and getattr(link_predictor, "prepared_for", None) != id(index):
            link_predictor.prepare(index)
        # Query-independent node costs c(x) and subtree minima, shared by all queries (Section 4.3).
        self.costs = NodeCostTable(index)

    @property
    def link_predictor(self) -> LinkPredictor:
        """The cross-taxonomy prior s_theta (default: untrained embedding fallback, built lazily)."""
        if self._link_predictor is None:
            self._link_predictor = EmbeddingLinkPredictor().prepare(self.index)
        return self._link_predictor

    # ------------------------------------------------------------------ steps
    def analyze(self, query: Query) -> QueryAnalysis:
        return analyze_query(query, self.llm, self.embedder, self.config)

    def ground(self, analysis: QueryAnalysis) -> Grounding:
        lp = self.link_predictor if analysis.is_hypothesis else None
        return ground(analysis, self.index, self.embedder, self.config, lp)

    def renderer(self) -> ContextRenderer:
        """A fresh per-query renderer that prices index nodes through the shared cost table."""
        return ContextRenderer(self.index, self.costs)

    def search_space(self, analysis: QueryAnalysis, grounding: Grounding,
                     renderer: Optional[ContextRenderer] = None) -> SearchSpace:
        lp = self.link_predictor if analysis.is_hypothesis else None
        return build_search_space(analysis, grounding, self.index, self.config, lp, renderer or self.renderer())

    # ---------------------------------------------------------------- public
    def retrieve(self, query: Union[Query, str], budget: Optional[int] = None) -> RetrievalResult:
        """Steps 1-5 and context assembly; returns units, context (<= B tokens) and statistics."""
        query = query if isinstance(query, Query) else Query(query_id="q", text=str(query), kind="")
        budget = int(budget if budget is not None else self.config.budget)
        t0 = time.perf_counter()
        meter = self.llm.meter if self.llm is not None else None
        before = meter.snapshot() if meter is not None else {}
        analysis = self.analyze(query)
        grounding = self.ground(analysis)
        renderer = self.renderer()
        space = self.search_space(analysis, grounding, renderer)
        search_budget = max(0, budget - renderer.header_overhead())
        result = make_search(space).run(search_budget)

        candidates = self._rank_candidates(space, result)
        units = result.units
        context, sources = assemble_context(self.index, units, renderer, candidate_order=candidates)
        while units and count_tokens(context) > budget:  # tokenizer non-additivity safety net
            log.warning("Context exceeds the budget (%d > %d); dropping the last unit", count_tokens(context), budget)
            units = units[:-1]
            result.selections = result.selections[: len(units)]
            context, sources = assemble_context(self.index, units, renderer, candidate_order=candidates)
        retrieved = [self._to_retrieved(sel, renderer, space) for sel in result.selections]
        stats: Dict[str, Any] = dict(result.stats)
        if len(result.selections) != stats.get("selected"):
            stats["selected"] = len(result.selections)
            stats["tokens_used"] = sum(s.tokens for s in result.selections)
        stats.update(
            kind=analysis.kind,
            analysis=analysis.to_dict(),
            grounding=grounding.to_dict(),
            candidate_ids=candidates,
            candidate_scores={renderer.concept_name(c): self._support(space, result, c) for c in candidates},
            retrieval_llm_calls=sum_usage(meter.diff(before)).calls if meter is not None else 0,
            studied_pair=space.studied_pair,
            tokenizer=tokenizer_name(),
        )
        names = [renderer.concept_name(c) for c in candidates][: max(1, self.config.max_hypothesis_candidates)]
        return RetrievalResult(
            query_id=query.query_id,
            units=retrieved,
            context=context,
            context_tokens=count_tokens(context),
            budget=budget,
            ranked_papers=RetrievalResult.papers_in_order(retrieved),
            candidates=names,
            sources=sources,
            latency_s=time.perf_counter() - t0,
            stats=stats,
        )

    def answer(self, query: Union[Query, str], budget: Optional[int] = None) -> Answer:
        """Retrieve within B tokens, then compose the cited answer with one LLM call."""
        query = query if isinstance(query, Query) else Query(query_id="q", text=str(query), kind="")
        if self.llm is None:
            raise ValueError("ScholarPathRetriever.answer needs an LLM client for the composition call")
        before = self.llm.meter.snapshot()
        retrieval = self.retrieve(query, budget)
        answer = compose_answer(query, retrieval, self.llm, self.config)
        usage = sum_usage(self.llm.meter.diff(before))
        answer.llm_calls = usage.calls
        answer.input_tokens = usage.input_tokens
        answer.output_tokens = usage.output_tokens
        answer.cost_usd = usage.cost_usd
        answer.metadata["retrieval_stats"] = {
            k: v for k, v in retrieval.stats.items() if k not in ("analysis", "grounding")
        }
        return answer

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _support(space: SearchSpace, result: SearchResult, candidate: str) -> float:
        g = space.aspects.hypothesis_group
        return result.state.support(g.iid(candidate)) if g is not None else 0.0

    def _rank_candidates(self, space: SearchSpace, result: SearchResult) -> List[str]:
        """Hypothesis candidates by their final support f_h (ties: order of first selection)."""
        first: Dict[str, int] = {}
        for i, sel in enumerate(result.selections):
            unit = sel.entry.unit
            if isinstance(unit, ComplementaryPathSet):
                first.setdefault(unit.candidate, i)
        return sorted(first, key=lambda c: (-self._support(space, result, c), first[c]))

    def _to_retrieved(self, sel: Selection, renderer: ContextRenderer, space: SearchSpace) -> RetrievedUnit:
        entry = sel.entry
        unit = entry.unit
        text = "\n".join(renderer.line(k) for k in entry.nodes)
        score = min(sel.density, _MAX_SCORE)
        if isinstance(unit, DirectPath):
            return RetrievedUnit(
                key=entry.uid, kind="direct", text=text, tokens=sel.tokens, score=score,
                papers=[unit.paper_id] if unit.paper_id else [], nodes=list(entry.nodes),
                status=ESTABLISHED, aspects=entry.coverage.as_dict(),
                metadata={"level": unit.level, "scope": unit.scope, "facet": unit.facet,
                          "omega": entry.meta.get("omega")},
            )
        assert isinstance(unit, ComplementaryPathSet)
        return RetrievedUnit(
            key=entry.uid, kind="complementary", text=text, tokens=sel.tokens, score=score,
            papers=unit.papers, nodes=list(entry.nodes), status=HYPOTHESIS,
            candidate=renderer.concept_name(unit.candidate), candidate_id=unit.candidate,
            aspects=entry.coverage.as_dict(),
            metadata={"bridge": unit.bridge, "baseline": unit.baseline, "distance": unit.distance,
                      "phi": unit.phi, "known": unit.known, "e_C": list(unit.e_C.nodes),
                      "e_P": list(unit.e_P.nodes), "e_M": list(unit.e_M.nodes),
                      **{k: entry.meta[k] for k in ("s_P", "s_M", "eta", "r_hat") if k in entry.meta}},
        )
