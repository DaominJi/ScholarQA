"""Tests for held-out hypothesis metrics (Disc / Hit@10, LLM-judge scores) and the judge panel."""

from __future__ import annotations

import json
from typing import Callable, List

import pytest

from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.evaluation.hypothesis_metrics import (
    aggregate_hypothesis,
    discovery_hit,
    evaluate_hypotheses,
    expected_record,
    judge_hypotheses,
    proposed_ideas,
)
from scholarpath.evaluation.judges import Judge, JudgePanel, JudgeSpec, as_judge_llm
from scholarpath.llm import CostMeter, LLMClient, LLMRequest, LLMResponse
from scholarpath.runtime import Runtime
from scholarpath.types import HYPOTHESIS_DISCOVERY, Answer, AnswerItem, Query


class FakeLLM(LLMClient):
    name = "fake"

    def __init__(self, responder: Callable[[LLMRequest], str]):
        super().__init__(model="fake-judge", meter=CostMeter())
        self.responder = responder
        self.requests: List[LLMRequest] = []

    def _generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=self.responder(request), input_tokens=10, output_tokens=5)


def _query() -> Query:
    return Query("h1", "Which methods could address flood forecasting with sparse rainfall gauges?",
                 kind=HYPOTHESIS_DISCOVERY, benchmark="heldout", reference_answer="Gaussian Processes",
                 metadata={"heldout_counterpart": "Gaussian Processes", "heldout_aliases": ["GP regression"],
                           "rationale": "GPs interpolate sparse gauges with calibrated uncertainty.",
                           "known_endpoint": "Flood Forecasting", "setting": "new_method_existing_problem"})


def _answer(names) -> Answer:
    items = [AnswerItem("Background fact.", status="established", citations=["p0"])]
    items += [AnswerItem(f"Apply {n} to flood forecasting.", status="hypothesis", candidate=n,
                         rationale=f"{n} fits the problem.", citations=[f"p{i}"]) for i, n in enumerate(names, 1)]
    return Answer("h1", "", items=items)


def test_proposed_ideas_keep_rank_order_dedup_and_cut_at_k():
    ans = _answer([f"Method {i}" for i in range(12)] + ["method 0"])
    ideas = proposed_ideas(ans, k=10)
    assert [i.rank for i in ideas] == list(range(1, 11)) and ideas[0].name == "Method 0"
    assert all(i.name != "Background fact." for i in ideas)
    rec = expected_record(_query())
    assert rec.counterpart == "Gaussian Processes" and rec.aliases == ["GP regression"]
    # k <= 0 proposes nothing (regression: the cut-off used to be checked after appending).
    assert proposed_ideas(ans, k=0) == [] and len(proposed_ideas(ans, k=None)) == 12
    assert not discovery_hit(_answer(["Gaussian Processes"]), _query(), None, k=0).hit


def test_discovery_exact_match_short_circuits_without_judge_calls():
    def never(req: LLMRequest) -> str:
        raise AssertionError("judge must not be called on an exact match")

    res = discovery_hit(_answer(["Optimal Transport", "gaussian processes"]), _query(), FakeLLM(never))
    assert res.hit and res.method == "exact" and res.matched_rank == 2
    alias = discovery_hit(_answer(["GP Regression"]), _query(), None)
    assert alias.hit and alias.method == "exact"
    # Beyond the top 10 the counterpart is not credited.
    late = discovery_hit(_answer([f"Method {i}" for i in range(10)] + ["Gaussian Processes"]), _query(), None)
    assert not late.hit and len(late.candidates) == 10


def test_discovery_judged_match_uses_majority_vote():
    def judge_saying(ranks):
        return FakeLLM(lambda req: json.dumps({"matches": ranks, "explanation": "x"}))

    ans = _answer(["Optimal Transport", "Sparse variational GPs", "Evolutionary Search"])
    panel = JudgePanel([Judge("a", judge_saying([2])), Judge("b", judge_saying([2, 3])), Judge("c", judge_saying([]))])
    res = discovery_hit(ans, _query(), panel)
    assert res.hit and res.method == "judge" and res.matched_rank == 2
    assert res.votes == {"a": [2], "b": [2, 3], "c": []}
    req = panel["a"].llm.requests[0]
    assert req.component == "judge" and req.payload["counterpart"] == "Gaussian Processes"
    assert [c["name"] for c in req.payload["candidates"]] == ["Optimal Transport", "Sparse variational GPs",
                                                              "Evolutionary Search"]

    minority = JudgePanel([Judge("a", judge_saying([3])), Judge("b", judge_saying([])), Judge("c", judge_saying([99]))])
    assert not discovery_hit(ans, _query(), minority).hit
    assert not discovery_hit(Answer("h1", "No ideas."), _query(), panel).hit


def test_judge_panel_averages_over_ideas_then_judges():
    def scorer(base: float) -> FakeLLM:
        def respond(req: LLMRequest) -> str:
            bump = 10.0 if "Second" in req.payload["idea"] else 0.0
            return json.dumps({"feasibility": base + bump, "novelty": base, "soundness": base - 10, "explanation": "."})
        return FakeLLM(respond)

    panel = JudgePanel([Judge("gpt-6.0-sol", scorer(60)), Judge("opus-5.5", scorer(80)), Judge("grok-4.7", scorer(70))])
    res = judge_hypotheses(_answer(["First idea", "Second idea"]), _query(), panel)
    assert res.n_ideas == 2
    assert res.per_judge["gpt-6.0-sol"] == {"feasibility": 65.0, "novelty": 60.0, "soundness": 50.0}
    assert res.scores == {"feasibility": pytest.approx(75.0), "novelty": pytest.approx(70.0),
                          "soundness": pytest.approx(60.0)}
    assert res.avg == pytest.approx((75 + 70 + 60) / 3)
    assert all(len(j.llm.requests) == 2 for j in panel)  # one call per (judge, idea)

    empty = judge_hypotheses(Answer("h1", "Nothing proposed."), _query(), panel)
    assert empty.n_ideas == 0 and empty.avg == 0.0
    row = aggregate_hypothesis([discovery_hit(_answer(["Gaussian Processes"]), _query()),
                                discovery_hit(_answer(["X"]), _query())], [res, empty])
    assert row["Hit@10"] == pytest.approx(50.0)
    assert row["Feas."] == pytest.approx(37.5) and row["Avg."] == pytest.approx(res.avg / 2)
    assert row["opus-5.5.novelty"] == pytest.approx(40.0)


def test_failed_judges_abstain_instead_of_scoring_zero():
    good = json.dumps({"feasibility": 70, "novelty": 70, "soundness": 70})
    panel = JudgePanel([Judge("a", FakeLLM(lambda r: good)), Judge("b", FakeLLM(lambda r: good)),
                        Judge("c", FakeLLM(lambda r: "I cannot score this."))])
    res = judge_hypotheses(_answer(["Y"]), _query(), panel)
    assert res.failed_judges == ["c"] and set(res.per_judge) == {"a", "b"}
    assert res.avg == pytest.approx(70.0)  # not (70 + 70 + 0) / 3

    broken = JudgePanel([Judge("c", FakeLLM(lambda r: "no json"))])
    unscored = judge_hypotheses(_answer(["Y"]), _query(), broken)
    assert not unscored.scored and unscored.avg is None
    row = aggregate_hypothesis([], [res, unscored])
    assert row["Avg."] == pytest.approx(70.0) and row["n_judged"] == 1 and row["n_unscored"] == 1
    assert row["judge_failures"] == 2

    # Disc: an unparsable verdict abstains; the majority is taken over the judges that answered.
    def says(ranks):
        return FakeLLM(lambda req: json.dumps({"matches": ranks}))

    ans = _answer(["Optimal Transport", "Sparse variational GPs"])
    disc = discovery_hit(ans, _query(), JudgePanel([Judge("a", says([2])), Judge("b", FakeLLM(lambda r: "??")),
                                                    Judge("c", FakeLLM(lambda r: '{"verdict": 1}'))]))
    assert disc.hit and disc.matched_rank == 2 and disc.failed_judges == ["b", "c"] and disc.votes == {"a": [2]}
    none = discovery_hit(ans, _query(), JudgePanel([Judge("b", FakeLLM(lambda r: "??"))]))
    assert not none.hit and none.failed_judges == ["b"]


def test_mock_panel_keeps_judge_cost_out_of_system_cost():
    rt = Runtime.from_config()
    bench = make_synthetic_benchmark(n_papers=60, seed=4)
    panel = JudgePanel.mock()
    assert panel.names == ["gpt-6.0-sol", "opus-5.5", "grok-4.7"]
    answers = {}
    for q in bench.hypothesis_queries:
        target = q.metadata["heldout_counterpart"]
        answers[q.query_id] = Answer(q.query_id, "", items=[
            AnswerItem("Try transport.", status="hypothesis", candidate="Optimal Transport Couplings"),
            AnswerItem("Try it.", status="hypothesis", candidate=f"{target} variants", rationale="Fits.")])
    out = evaluate_hypotheses(answers, bench.hypothesis_queries, panel, panel)
    assert out["summary"]["Hit@10"] == pytest.approx(100.0)  # judged (non-exact) matches
    assert all(r["match_method"] == "judge" for r in out["rows"])
    assert 0.0 <= out["summary"]["Avg."] <= 100.0
    usage = panel.usage()
    assert all(u["calls"] > 0 for u in usage.values())
    assert panel.total_usage().calls == sum(u["calls"] for u in usage.values())
    assert rt.meter.total().calls == 0 and all(j.meter.total().calls == 0 for j in panel)  # "judge" excluded
    assert len({id(j.meter) for j in panel}) == 3  # one private meter per judge


def test_judge_specs_and_paper_default(monkeypatch):
    spec = JudgeSpec.from_dict({"name": "opus-5.5", "llm": {"provider": "mock", "model": "m", "cache_dir": None}})
    assert spec.to_dict()["llm"]["model"] == "m"
    # Fields left out take judge defaults, not the system's (cache, DeepSeek prices, deepseek model).
    remote = JudgeSpec.from_dict({"name": "opus-5.5", "llm": {"provider": "openai"}})
    assert remote.llm.model == "opus-5.5" and remote.llm.cache_dir == "cache/judge"
    assert remote.llm.price_input_per_m == 0.0 and remote.llm.price_output_per_m == 0.0
    assert JudgeSpec.from_dict({"name": "x"}) == JudgeSpec("x")
    priced = JudgeSpec.from_dict({"name": "x", "llm": {"provider": "openai", "price_input_per_m": 2.0}})
    assert priced.llm.price_input_per_m == 2.0
    panel = JudgePanel.from_specs([spec, {"name": "grok-4.7", "llm": {"provider": "mock", "temperature": 0.7}}])
    assert panel.names == ["opus-5.5", "grok-4.7"] and all(j.llm.temperature == 0.0 for j in panel)
    with pytest.raises(ValueError):
        JudgePanel([Judge("x", panel.primary.llm), Judge("x", panel.primary.llm)])

    monkeypatch.setenv("SCHOLARPATH_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("SCHOLARPATH_JUDGE_BASE_URL", "http://localhost:9/v1")
    paper = JudgePanel.paper_default(models={"grok-4.7": "x-ai/grok-4.7"}, cache_dir=None)
    assert [j.model for j in paper] == ["gpt-6.0-sol", "opus-5.5", "x-ai/grok-4.7"]
    assert all(j.llm.base_url == "http://localhost:9/v1" for j in paper)
    llm = Runtime.from_config().llm
    assert as_judge_llm(llm) is llm and as_judge_llm(JudgePanel.single(llm)) is llm
