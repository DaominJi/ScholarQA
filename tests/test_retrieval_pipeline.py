"""End-to-end tests of ScholarPathRetriever (Algorithm 2) on the synthetic benchmark with the mock LLM."""

from __future__ import annotations

import copy
import json

import pytest

from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.index import SemanticIndex
from scholarpath.retrieval import EmbeddingLinkPredictor, ScholarPathRetriever
from scholarpath.runtime import Runtime
from scholarpath.text import count_tokens
from scholarpath.types import EVIDENCE, HYPOTHESIS_DISCOVERY


@pytest.fixture(scope="module")
def rt():
    return Runtime.from_config()


@pytest.fixture(scope="module")
def bench():
    return make_synthetic_benchmark(n_papers=160, seed=7, n_heldout=4)


@pytest.fixture(scope="module")
def index(bench, rt):
    return SemanticIndex.build(bench.problem_gold, bench.method_gold, bench.corpus.papers, rt.embedder)


def _retriever(index, rt, **overrides):
    cfg = copy.deepcopy(rt.config.retrieval)
    cfg.max_chunks_per_paper = 2
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return ScholarPathRetriever(index, rt.llm, rt.embedder, cfg)


def test_evidence_queries_retrieve_gold_papers(index, bench, rt):
    r = _retriever(index, rt)
    recalls = []
    for q in bench.evidence_queries[:8]:
        res = r.retrieve(q, budget=8192)
        gold = set(q.gold_papers)
        recalls.append(len(gold & set(res.ranked_papers)) / len(gold))
        assert res.context_tokens == count_tokens(res.context) <= 8192
        assert res.stats["kind"] == EVIDENCE and res.stats["algorithm"] == "bound_guided"
        assert all(u.kind == "direct" and u.status == "established" for u in res.units)
        assert res.ranked_papers == res.papers_in_order(res.units)
        # Gold findings J(u_q, m_q) come before other papers.
        first = res.ranked_papers[: len(gold)]
        assert len(gold & set(first)) / len(gold) >= 0.8
    assert sum(recalls) / len(recalls) >= 0.95


@pytest.mark.parametrize("budget", [150, 400, 1200, 3000])
def test_budget_is_never_exceeded(index, bench, rt, budget):
    r = _retriever(index, rt)
    for q in bench.evidence_queries[:3] + bench.hypothesis_queries[:2]:
        res = r.retrieve(q, budget=budget)
        assert res.context_tokens <= budget
        assert res.stats["tokens_used"] <= budget
        assert sum(u.tokens for u in res.units) == res.stats["tokens_used"]


def test_hypothesis_queries_return_candidates_and_heldout_hits(index, bench, rt):
    r = _retriever(index, rt)
    hits = 0
    for q in bench.hypothesis_queries:
        res = r.retrieve(q, budget=8192)
        assert res.stats["kind"] == HYPOTHESIS_DISCOVERY
        assert res.candidates, q.query_id
        assert len(set(res.candidates)) == len(res.candidates)
        sets = [u for u in res.units if u.kind == "complementary"]
        assert sets and all(u.status == "hypothesis" and u.candidate for u in sets)
        # Every candidate is an unexplored pair: J(u_q, m) is empty.
        known = res.stats["grounding"]["problem_scopes"][0][0]
        for u in sets:
            assert not index.shared_papers(known, u.candidate_id)
            assert len(u.papers) == 2  # one problem-side and one method-side premise paper
        if q.metadata["heldout_counterpart"] in res.candidates[:10]:
            hits += 1
    assert hits >= 1


def test_answer_single_composition_call(index, bench, rt):
    r = _retriever(index, rt)
    q = bench.evidence_queries[0]
    ans = r.answer(q, budget=4096)
    assert ans.llm_calls == 2  # query understanding + composition; zero calls inside the search
    assert ans.items and all(it.status == "established" for it in ans.items)
    res = r.retrieve(q, budget=4096)
    assert res.stats["retrieval_llm_calls"] == 1
    cited = {c for it in ans.items for c in it.citations}
    assert cited and cited <= set(res.ranked_papers)
    assert ans.retrieved_papers == res.ranked_papers and ans.cost_usd > 0
    hq = bench.hypothesis_queries[0]
    hans = r.answer(hq, budget=4096)
    hyps = [it for it in hans.items if it.status == "hypothesis"]
    assert hyps and all(it.candidate for it in hyps)
    assert hyps[0].candidate == hans.metadata["candidates"][0]
    assert hyps[0].score is not None


def test_llm_free_query_understanding(index, bench, rt):
    r = _retriever(index, rt, llm_query_understanding=False)
    q = bench.evidence_queries[1]
    res = r.retrieve(q, budget=4096)
    assert res.stats["retrieval_llm_calls"] == 0
    assert len(res.stats["analysis"]["aspects"]) == 1
    assert set(q.gold_papers) & set(res.ranked_papers)
    assert r.answer(q, budget=4096).llm_calls == 1


def test_ablation_switches_change_behaviour(index, bench, rt):
    full = _retriever(index, rt)
    q_ev, q_hyp = bench.evidence_queries[2], bench.hypothesis_queries[0]
    base_ev, base_hyp = full.retrieve(q_ev, 1500), full.retrieve(q_hyp, 1500)

    # w/o bound-guided search: identical output (Prop. 7), more candidate units scored.
    exh = _retriever(index, rt, use_bound_guided_search=False)
    e_ev, e_hyp = exh.retrieve(q_ev, 1500), exh.retrieve(q_hyp, 1500)
    assert [u.key for u in e_ev.units] == [u.key for u in base_ev.units]
    assert [u.key for u in e_hyp.units] == [u.key for u in base_hyp.units]
    assert e_hyp.stats["algorithm"] == "exhaustive"
    assert e_hyp.stats["units_created"] >= base_hyp.stats["units_created"]
    # The unpruned scan scores every candidate node in scope (Table 7 "Scored" = candidate pool).
    assert e_hyp.stats["scored_candidates"] == e_hyp.stats["candidate_pool"]
    assert e_hyp.stats["scored_candidates"] >= base_hyp.stats["scored_candidates"]

    # "w/o temporal link prediction" (Table 7) swaps in another (untrained) prior: candidates are still
    # scored ("Scored" column), but with different link probabilities.
    other = copy.deepcopy(rt.config.retrieval)
    other.max_chunks_per_paper = 2
    untrained = ScholarPathRetriever(index, rt.llm, rt.embedder, other,
                                     link_predictor=EmbeddingLinkPredictor(scale=2.0, bias=-1.0)).retrieve(q_hyp, 1500)
    assert untrained.stats["scored_candidates"] == base_hyp.stats["scored_candidates"] > 0
    assert untrained.stats["candidate_scores"] != base_hyp.stats["candidate_scores"]
    # Dropping the prior altogether (eta = phi): no link evaluation, candidates still scored.
    nolink = _retriever(index, rt, use_link_prediction=False).retrieve(q_hyp, 1500)
    assert nolink.stats["link_evaluations"] == 0 < base_hyp.stats["link_evaluations"]
    assert nolink.stats["scored_candidates"] == base_hyp.stats["scored_candidates"]
    assert nolink.stats["candidate_scores"] != base_hyp.stats["candidate_scores"]

    # w/o role-aware granularity and w/o shared-cost reuse: different selections.
    changed = 0
    for q in bench.evidence_queries[:4] + bench.hypothesis_queries[:2]:
        ref = [u.key for u in full.retrieve(q, 1500).units]
        flat = [u.key for u in _retriever(index, rt, use_role_aware_granularity=False).retrieve(q, 1500).units]
        noshare = [u.key for u in _retriever(index, rt, use_shared_cost_update=False).retrieve(q, 1500).units]
        changed += int(flat != ref) + int(noshare != ref)
    assert changed >= 2


def test_node_costs_are_priced_once_per_index(index, bench, rt):
    """c(x) and the subtree minima behind c_low(H | S) are query-independent and shared across queries."""
    r = _retriever(index, rt)
    first, second = bench.hypothesis_queries[0], bench.hypothesis_queries[1]
    r.retrieve(first, budget=2048)
    papers = r.costs.priced("paper")
    assert papers > 0
    res = r.retrieve(second, budget=2048)
    assert r.costs.priced("paper") <= papers + len(res.ranked_papers)
    again = r.costs.priced()
    r.retrieve(first, budget=2048)
    assert r.costs.priced() == again  # a repeated query prices nothing
    space = r.search_space(r.analyze(first), r.ground(r.analyze(first)))
    assert space.renderer.costs is r.costs


def test_sources_attribute_shared_premise_papers_to_every_candidate(index, bench, rt):
    r = _retriever(index, rt)
    for q in bench.hypothesis_queries[:2]:
        res = r.retrieve(q, budget=4096)
        named = {s["candidate"] for s in res.sources if s["candidate"]}
        assert named == set(res.candidates)
        for u in res.units:
            if u.kind == "complementary":
                assert {(p, u.candidate) for p in u.papers} <= {(s["paper_id"], s["candidate"]) for s in res.sources}
        ans = r.answer(q, budget=4096)
        assert {it.candidate for it in ans.items if it.status == "hypothesis"} == set(res.candidates[:10])


def test_retrieval_result_is_json_serializable(index, bench, rt):
    r = _retriever(index, rt)
    for q in (bench.evidence_queries[0], bench.hypothesis_queries[0]):
        res = r.retrieve(q, budget=2048)
        blob = json.dumps(res.to_dict())
        assert res.query_id in blob
        assert res.sources and all({"paper_id", "title", "status"} <= set(s) for s in res.sources)
