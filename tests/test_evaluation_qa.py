"""Tests for the evidence-QA metrics: ResearchQA Rub / Cite, ScholarQA-CS Corr, citation P/R, Prometheus."""

from __future__ import annotations

import json
from typing import Callable, List

import pytest

from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.evaluation import qa_metrics as Q
from scholarpath.llm import CostMeter, LLMClient, LLMRequest, LLMResponse
from scholarpath.prompts.evaluation import (
    CS_CLAIMS_MAX_TOKENS,
    parse_attribution_label,
    parse_coverage_labels,
    parse_prometheus_score,
)
from scholarpath.runtime import Runtime
from scholarpath.types import Answer, AnswerItem, Query, RubricItem


class FakeLLM(LLMClient):
    """Scripted judge: ``responder(request) -> text``; records every request."""

    name = "fake"

    def __init__(self, responder: Callable[[LLMRequest], str]):
        super().__init__(model="fake-judge", meter=CostMeter())
        self.responder = responder
        self.requests: List[LLMRequest] = []

    def _generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=self.responder(request), input_tokens=10, output_tokens=2)


# ---------------------------------------------------------------- ResearchQA Rub


def test_coverage_label_parsing():
    assert parse_coverage_labels("Mostly\n\nCompletely\n", 2) == ["Mostly", "Completely"]
    assert parse_coverage_labels("1. mostly.\n- **Not at all**\nQ3: Barely", 3) == ["Mostly", "Not at all", "Barely"]
    assert parse_coverage_labels("Mostly\nCompletely", 3) is None  # wrong number of lines
    assert parse_coverage_labels("Mostly\nSomewhat", 2) is None  # unknown label
    assert Q.normalize_5scale(1) == 0.0 and Q.normalize_5scale(5) == 1.0


def test_rubric_coverage_batches_of_eight_and_type_pooling():
    labels = ["Completely", "Mostly", "Moderately", "Barely", "Not at all"]

    def responder(req: LLMRequest) -> str:
        n = len(req.payload["questions"])
        return "\n".join(labels[i % 5] for i in range(n))

    judge = FakeLLM(responder)
    rubric = [RubricItem(f"Does the response cover point {i}?", kind="Citation" if i in (0, 9) else "Other")
              for i in range(10)]
    query = Query("rq1", "question", rubric=rubric)
    res = Q.rubric_coverage(Answer("rq1", "answer text"), query, judge)
    assert [len(r.payload["questions"]) for r in judge.requests] == [8, 2]
    assert all(r.component == "judge" and r.temperature == 0.0 for r in judge.requests)
    first = [1.0, 0.75, 0.5, 0.25, 0.0, 1.0, 0.75, 0.5]
    second = [1.0, 0.75]
    assert res.coverage == pytest.approx(sum(first + second) / 10)
    assert res.type_scores("citation") == [1.0, 0.75]  # items 0 and 9
    agg = Q.aggregate_coverage([res, Q.CoverageResult("rq2", coverage=0.5)])
    assert agg["Rub"] == pytest.approx(100 * (res.coverage + 0.5) / 2)
    assert agg["coverage_citation"] == pytest.approx(87.5)
    assert judge.meter.total().calls == 0 and judge.meter.component("judge").calls == 2


def test_rubric_coverage_retries_then_drops_or_zeroes():
    calls = {"n": 0}

    def flaky(req: LLMRequest) -> str:
        calls["n"] += 1
        return "Mostly" if calls["n"] < 2 else "Mostly\nCompletely"  # malformed once, then valid

    res = Q.rubric_coverage("ans", ["a?", "b?"], FakeLLM(flaky), query_id="q")
    assert calls["n"] == 2 and res.coverage == pytest.approx((0.75 + 1.0) / 2) and not res.failed

    broken = FakeLLM(lambda req: "Mostly")
    dropped = Q.rubric_coverage("ans", ["a?", "b?"], broken, query_id="q")
    assert dropped.failed and dropped.coverage is None and len(broken.requests) == 3  # 3 attempts

    rubric = [f"item {i}?" for i in range(10)]
    zero_judge = FakeLLM(lambda req: "\n".join(["Completely"] * 8) if len(req.payload["questions"]) == 8 else "bad")
    zeroed = Q.rubric_coverage("ans", rubric, zero_judge, on_failure="zero")
    assert not zeroed.failed and zeroed.coverage == pytest.approx(8 / 10)
    agg = Q.aggregate_coverage([dropped, zeroed])
    assert agg["n"] == 1 and agg["n_failed"] == 1 and agg["Rub"] == pytest.approx(80.0)


def test_researchqa_end_to_end_with_mock_runtime():
    rt = Runtime.from_config()
    bench = make_synthetic_benchmark(n_papers=60, seed=2)
    queries = bench.evidence_queries[:3]
    answers = {}
    for q in queries:
        q.unaligned_gold = ["external-ref"]
        q.rubric = [RubricItem(f"Does the response discuss {q.metadata['method'].lower()}?", kind="Other"),
                    RubricItem(f"Does the response cite {bench.corpus[q.gold_papers[0]].title}?", kind="Citation")]
        items = [AnswerItem(f"{q.metadata['method']} is applied to {q.metadata['problem']}.", citations=[p])
                 for p in q.gold_papers[:2]]
        answers[q.query_id] = Answer(q.query_id, "", items=items)
    absent = Query("absent", "x", gold_papers=["g"], rubric=[RubricItem("Does it cite g?", kind="Citation")])
    ev = Q.evaluate_researchqa(answers, queries + [absent], rt.llm, papers=bench.corpus, missing="skip")
    assert ev.missing == ["absent"] and len(ev.rows) == 3 and ev.summary["n_missing"] == 1
    assert ev.summary["Rub"] is not None and 0.0 <= ev.summary["Rub"] <= 100.0
    assert ev.summary["coverage_citation"] == pytest.approx(100.0)  # titles appear in the reference list
    expected_cite = [Q.citation_f1(answers[q.query_id], q).f1 for q in queries]
    assert ev.summary["Cite"] == pytest.approx(100 * sum(expected_cite) / 3)
    assert rt.meter.component("judge").calls == 3 and rt.meter.total().calls == 0
    # missing="zero": a failed query scores 0 (as in Tables 2 and 4) instead of leaving the denominator.
    zero = Q.evaluate_researchqa(answers, queries + [absent], rt.llm, papers=bench.corpus, missing="zero")
    assert zero.missing == ["absent"] and len(zero.rows) == 4 and zero.rows[-1]["missing"]
    assert zero.summary["n"] == 4 and zero.summary["n_missing"] == 1
    assert zero.summary["Rub"] == pytest.approx(ev.summary["Rub"] * 3 / 4)
    assert zero.summary["Cite"] == pytest.approx(100 * sum(expected_cite) / 4)
    assert rt.meter.component("judge").calls == 6  # no judge call for the failed query


# ---------------------------------------------------------------- Cite vs. gold


def test_citation_f1_against_gold_papers():
    q = Query("q", "x", gold_papers=["a", "b", "c"], unaligned_gold=["u"])
    ans = Answer("q", "", items=[AnswerItem("s1", citations=["a", "x"]), AnswerItem("s2", citations=["b", "a"])])
    s = Q.citation_f1(ans, q)
    assert (s.precision, s.recall) == (pytest.approx(2 / 3), pytest.approx(2 / 4))
    assert s.f1 == pytest.approx(4 / 7)
    # Falls back to [id] markers in the text when items carry no citations.
    assert Q.cited_papers(Answer("q", "See [a, b] and [c]; also [b].")) == ["a", "b", "c"]
    # ... where valid ids keep bracketed non-citations out (regression: [Smith 2020], [0, 1] counted as papers).
    text_only = Answer("q", "As in [Smith 2020] and [2101.00001, Fig. 3], scores lie in [0, 1] [CLS].")
    assert Q.cited_papers(text_only) == ["2101.00001", "0", "1", "CLS"]
    assert Q.cited_papers(text_only, valid_ids={"2101.00001", "a"}) == ["2101.00001"]
    assert Q.cited_papers(text_only, valid_ids={"2101.00001": object()}.get) == ["2101.00001"]
    gold_q = Query("q", "x", gold_papers=["2101.00001"])
    assert Q.citation_f1(text_only, gold_q, valid_ids={"2101.00001"}).precision == 1.0
    # Structured citations are kept even outside the id set (a hallucinated id counts against precision).
    assert Q.cited_papers(ans, valid_ids={"a"}) == ["a", "x", "b"]
    assert Q.citation_prf([], ["a"]).f1 == 0.0
    agg = Q.aggregate_prf([s, Q.citation_prf(["a"], ["a"])])
    assert agg["Cite"] == pytest.approx(100 * (4 / 7 + 1.0) / 2)


# ---------------------------------------------------------------- ScholarQA-CS Corr


def _cs_config() -> dict:
    return {"name": "rubric_corpusqa_generic", "config": {
        "question": "Which datasets evaluate type inference?",
        "low_length": 300, "high_length": 600, "length_weight": 0.05, "expertise_weight": 0.05,
        "citations_weight": 0.2, "excerpts_weight": 0.1,
        "other_properties": [
            {"name": "most_important_item_0", "criterion": "Define type inference.", "weight": 0.4,
             "evidence": ["snippet A", "snippet B"]},
            {"name": "nice_to_have_item_0", "criterion": "Mention ManyTypes4Py.", "weight": 0.2,
             "evidence": ["snippet C"]},
        ]}}


def test_scholarqa_cs_correctness_components_and_weights():
    def responder(req: LLMRequest) -> str:
        if req.task == "evaluation.cs_property":
            crit = req.payload["criterion"]
            score = 8 if crit.startswith("The level of expertise") else 6 if "Define" in crit else 0
            return json.dumps({"score": score})
        if req.task == "evaluation.cs_evidence":
            return 'Sure: {"score": 1}'
        if req.task == "evaluation.cs_claims":
            return json.dumps({"claims": [
                {"claim_text": "c1", "citations": [{"citation_text": "[1]", "excerpts": ["e"]}]},
                {"claim_text": "c2", "citations": [{"citation_text": "[2]", "excerpts": []}]},
                {"claim_text": "c3", "citations": []},
                {"claim_text": "c4", "citations": [{"citation_text": "[3]", "excerpts": []}]}]})
        raise AssertionError(req.task)

    judge = FakeLLM(responder)
    answer = " ".join(["word"] * 450)
    s = Q.correctness(answer, _cs_config(), judge)
    assert s.weights["most_important_item_0"] == pytest.approx(0.2)
    assert s.weights["most_important_item_0_evidence"] == pytest.approx(0.2)
    assert s.components["length"] == pytest.approx(0.5)  # 450 words: halfway between 300 and 600
    assert s.components["expertise"] == pytest.approx(0.8)
    assert s.components["citations"] == pytest.approx(3 / 4) and s.components["excerpts"] == pytest.approx(1 / 3)
    assert s.components["most_important_item_0_evidence"] == pytest.approx(0.5)  # 1 of 2 snippets
    assert s.components["nice_to_have_item_0_evidence"] == 0.0  # criterion scored 0 -> evidence not scored
    evidence_calls = [r for r in judge.requests if r.task == "evaluation.cs_evidence"]
    assert len(evidence_calls) == 1
    rubric = 0.2 * 0.6 + 0.2 * 0.5 + 0.1 * 0.0 + 0.1 * 0.0
    static = 0.05 * 0.5 + 0.05 * 0.8 + 0.2 * 0.75 + 0.1 / 3
    assert s.score == pytest.approx(static + rubric)
    assert s.ann_score == pytest.approx(rubric / 0.6)
    assert Q.aggregate_correctness([s])["Corr"] == pytest.approx(100 * (static + rubric))

    # --rubrics only: evidence dropped, each criterion carries its full weight.
    only = Q.CSRubricConfig.from_dict(_cs_config()).restricted(use_snippets=False)
    assert only.weights()["most_important_item_0"] == pytest.approx(0.4)
    bad = _cs_config()
    bad["config"]["other_properties"][0]["weight"] = 0.1
    with pytest.raises(ValueError):
        Q.correctness(answer, bad, judge)


def test_scholarqa_cs_answers_are_judged_with_their_cited_passages():
    """convert_answer_scholar_cs.py format: [i] citations + References block with title and text."""
    passages = {"2101.00002": {"title": "Smith", "text": "GNNs predict binding affinity."},
                "2101.00003": "Docking scores correlate weakly with affinity."}
    ans = Answer("cs1", "GNNs predict binding affinity [2101.00002]. Docking is weaker [2101.00003, 2101.00002].",
                 items=[AnswerItem("x", citations=["2101.00002", "2101.00003"])], metadata={"passages": passages})
    rendered = Q.render_cs_answer(ans, Q.answer_passages(ans))
    assert rendered == ("GNNs predict binding affinity [0]. Docking is weaker [1][0]."
                        "\nReferences:\n[0] Smith GNNs predict binding affinity.\n\n"
                        "[1]  Docking scores correlate weakly with affinity.\n\n")

    seen: List[LLMRequest] = []

    def responder(req: LLMRequest) -> str:
        seen.append(req)
        if req.task == "evaluation.cs_claims":
            body, _, refs = req.payload["response"].partition("\nReferences:\n")
            has = "[0] Smith GNNs" in refs
            return json.dumps({"claims": [{"claim_text": "c", "citations": [
                {"citation_text": "[0]", "excerpts": ["GNNs predict binding affinity."] if has else []}]}]})
        return json.dumps({"score": 5})

    cfg = {"config": {"question": "q", "other_properties": [{"name": "a", "criterion": "c", "weight": 0.6}]}}
    with_refs = Q.correctness(ans, cfg, FakeLLM(responder))
    claims_req = [r for r in seen if r.task == "evaluation.cs_claims"][-1]
    assert "\nReferences:\n[0] Smith" in claims_req.payload["response"]
    assert claims_req.max_tokens == CS_CLAIMS_MAX_TOKENS == 4096  # official call is uncapped
    assert with_refs.components["excerpts"] == 1.0
    plain = Q.correctness(ans, cfg, FakeLLM(responder), references=False)
    assert plain.components["excerpts"] == 0.0 and with_refs.score - plain.score == pytest.approx(0.1)

    # The offline mock reads excerpts from the References block too.
    rt = Runtime.from_config()
    assert Q.correctness(ans, cfg, rt.llm).components["excerpts"] == 1.0
    assert Q.correctness(ans, cfg, rt.llm, references=False).components["excerpts"] == 0.0


def test_scholarqa_cs_config_from_query_rubric_and_mock_run():
    rt = Runtime.from_config()
    q = Query("cs1", "How is graph learning used for drug discovery?", benchmark="scholarqa_cs",
              rubric=[RubricItem("The answer should mention binding affinity.", weight=2.0),
                      RubricItem("The answer could mention docking.", weight=1.0)],
              metadata={"rubric_evidence": [["Binding affinity is predicted with GNNs."], []]})
    cfg = Q.CSRubricConfig.from_query(q)
    assert sum(p.weight for p in cfg.other_properties) == pytest.approx(0.6)
    assert sum(cfg.weights().values()) == pytest.approx(1.0)
    ans = Answer("cs1", "Graph neural networks predict binding affinity of ligands [p1]. Docking remains hard.")
    ev = Q.evaluate_scholarqa_cs({"cs1": ans, "cs2": ans}, [q, Query("cs2", q.text)], rt.llm)
    assert len(ev.rows) == 1  # duplicate question annotation scored once
    assert 0.0 <= ev.summary["Corr"] <= 100.0


# ---------------------------------------------------------------- Statement-level citation P / R


def test_citation_attribution_statement_level_with_fake_judge():
    ctxs = [{"title": "A", "text": "alpha facts"}, {"title": "B", "text": "beta facts"},
            {"title": "C", "text": "gamma facts"}, {"title": "D", "text": "delta facts"}]
    support = {"alpha": {"S1", "S3"}, "beta": {"S2"}, "gamma": {"S3"}, "delta": set()}

    def judge(passage: str, claim: str) -> float:
        tag = claim.split()[0]
        return float(any(tag in claims for key, claims in support.items() if key in passage))

    output = (
        "S1 is a statement long enough to be scored by the protocol [0]. "
        "S2 is a statement that cites two passages but only one matters [0][1]. "
        "S3 continues without its own citation marker but inherits the last ones. "
        "Short one [3]. "
        "S5 is a long statement citing an out-of-range passage index here [7]. "
        "S6 cites four passages but only three are used by the protocol [3][2][1][0]."
    )
    res = Q.citation_attribution(output, ctxs, judge)
    # Scored sentences: S1, S2, S3 (inherits [0][1]), S5, S6 (short one skipped).
    assert res.n_sentences == 5
    # Supported: S1 ([0]), S2 ([0][1] via beta), S3 ([0][1] via alpha); S5 out of range; S6 uses [3][2][1].
    assert res.recall == pytest.approx(3 / 5)
    # Citations: S1 1 + S2 2 + S3 2 + S6 3 = 8 (out-of-range S5 not counted).
    # Precise: S1 1; S2: [0] unnecessary (removable), [1] needed -> 1; S3: [0] supports alone, [1] removable -> 1.
    assert res.n_citations == 8
    assert res.precision == pytest.approx(3 / 8)
    agg = Q.aggregate_citation_attribution([res, None])
    assert agg["citation_rec"] == pytest.approx(60.0) and agg["citation_prec"] == pytest.approx(37.5)
    assert agg["Cite"] == pytest.approx(100 * Q.f1_score(0.375, 0.6))
    assert Q.citation_attribution("", ctxs, judge) is None
    assert Q.extract_citation_indices("x [1, 2] y [3]") == [1, 3]  # official regex quirk
    assert Q.remove_citations("Claim [1] here [2, 3] .") == "Claim here."


def test_number_citations_and_answer_attribution():
    papers = {"p1": "alpha facts", "p2": {"title": "Beta", "text": "beta facts"}}
    ans = Answer("q", "Alpha statement that is long enough to be scored here. [p1]\n"
                      "Beta statement that is long enough to be scored, too [p2, p1]. "
                      "See [Smith 2020] and [ghost] for a further discussion of these statements.",
                 items=[AnswerItem("x", citations=["p1", "p2", "ghost"])])
    out, ctxs, ids = Q.number_citations(ans, papers.get)
    assert ids == ["p1", "p2"] and ctxs[1] == {"title": "Beta", "text": "beta facts"} and ctxs[0] == {"text": "alpha facts"}
    assert Q.format_document(ctxs[0]) == "alpha facts"  # untitled passage, as exported (no "Title:" header)
    assert "here [0]." in out and "[1][0]" in out and "[Smith 2020]" in out and "[2]" in out
    judge = lambda passage, claim: float(claim.split()[0].lower() in passage)  # noqa: E731
    res = Q.answer_citation_attribution(ans, papers.get, judge)
    assert res.n_sentences == 3 and res.recall == pytest.approx(2 / 3)  # [ghost] is out of range
    assert res.n_citations == 3 and res.precision == pytest.approx(2 / 3)  # [0] is removable in sentence 2

    # Bracketed non-citations ([CLS], [MASK], intervals) are text, not out-of-range citations (regression).
    bert = {"2101.00001": {"title": "BERT", "text": "BERT prepends a CLS token used for sentence classification."}}
    cls = Answer("q", "BERT uses the [CLS] token representation for sentence classification tasks [2101.00001]. "
                      "Scores are normalized to the interval [0, 1] before the classifier uses them [2101.00001, Fig. 3].",
                 items=[AnswerItem("x", citations=["2101.00001"])])
    out2, ctxs2, ids2 = Q.number_citations(cls, bert.get)
    assert ids2 == ["2101.00001"] and len(ctxs2) == 1
    assert "the [CLS] token" in out2 and "interval (0, 1) before" in out2 and "them [0] (Fig. 3)." in out2
    cls_judge = lambda passage, claim: float("CLS" in passage)  # noqa: E731
    res2 = Q.answer_citation_attribution(cls, bert.get, cls_judge)
    assert res2.recall == 1.0 and res2.n_citations == 2 and res2.precision == 1.0
    assert res2.splitter == Q.sentence_splitter() in {"nltk", "regex"}
    # Markers are moved before a sentence-final period only, not after "et al." mid-sentence.
    etal = Answer("q", "Smith et al. [2101.00001] showed it. It holds. [2101.00001]")
    assert Q.number_citations(etal, bert.get)[0] == "Smith et al. [0] showed it. It holds [0]."
    # A text-only answer: only ids the passage lookup resolves are citations.
    text_only = Answer("q", cls.text)
    assert Q.number_citations(text_only, bert.get)[2] == ["2101.00001"]

    rt = Runtime.from_config()
    llm_judge = Q.LLMAttributionJudge(rt.llm)
    assert llm_judge("alpha beta gamma facts", "alpha beta gamma") == 1.0
    assert llm_judge("unrelated text", "alpha beta gamma") == 0.0
    assert rt.meter.component("judge").calls == 2
    # Negated labels never read as Attributable (regression: "Not attributable" -> Attributable).
    assert parse_attribution_label("Attributable.") == "Attributable"
    assert parse_attribution_label("Label: Extrapolatory") == "Extrapolatory"
    assert parse_attribution_label("Not attributable - the reference does not mention this.") is None
    assert parse_attribution_label("This claim is non-attributable; it is Contradictory.") == "Contradictory"
    assert parse_attribution_label("Unattributable") is None and parse_attribution_label("") is None


# ---------------------------------------------------------------- Prometheus + combination


def test_prometheus_parsing_grading_and_combination():
    assert parse_prometheus_score("Feedback: good. [RESULT] 4") == 4
    assert parse_prometheus_score("Feedback: ok [RESULT] 3.\n") == 3
    assert parse_prometheus_score("Feedback: the overall score is 2") == 2
    assert parse_prometheus_score("Feedback: no score [RESULT] great") is None
    assert Q.prometheus_to_100(5) == 100.0 and Q.prometheus_to_100(3) == 50.0

    rt = Runtime.from_config()
    q = Query("m1", "What forms a protein corona?", benchmark="scholarqa_multi",
              reference_answer="Lipid nanoparticles adsorb serum proteins forming a corona.")
    g = Q.prometheus_grade(Answer("m1", "Lipid nanoparticles adsorb serum proteins forming a corona [0]."), q, rt.llm)
    assert set(g.scores) == {"organization", "coverage", "relevance"} and all(s == 5 for s in g.scores.values())
    agg = Q.aggregate_prometheus([g, Q.PrometheusResult("m2", {"organization": 3, "coverage": None, "relevance": 1})])
    assert agg["organization"] == pytest.approx(4.0) and agg["coverage"] == pytest.approx(5.0)
    assert agg["average"] == pytest.approx((4.0 + 5.0 + 3.0) / 3) and agg["n_unparsed"] == 1
    assert agg["Corr"] == pytest.approx(100 * (agg["average"] - 1) / 4)

    combined = Q.scholarqabench_combined({"Corr": 60.0, "Cite": 50.0, "n": 100},
                                         {"Corr": 80.0, "Cite": 70.0, "n": 108})
    assert combined["Corr"] == pytest.approx((100 * 60 + 108 * 80) / 208)
    assert combined["Cite"] == pytest.approx((100 * 50 + 108 * 70) / 208) and combined["n"] == 208
    assert Q.scholarqabench_combined({"Corr": 60.0, "n": 100}, {"Corr": None, "n": 108})["Corr"] == 60.0


# ---------------------------------------------------------------- failed queries


def test_failed_queries_can_score_zero_on_scholarqabench():
    """missing="zero" keeps failed queries in the denominator (skipping them inflates a system's scores)."""
    rt = Runtime.from_config()
    rubric = [RubricItem("The answer should mention binding affinity.", weight=1.0)]
    cs = [Query(f"cs{i}", f"How is graph learning used for task {i}?", benchmark="scholarqa_cs", rubric=rubric)
          for i in range(2)]
    ans = {"cs0": Answer("cs0", "Graph neural networks predict binding affinity of ligands [p1].")}
    judge = lambda passage, claim: 1.0  # noqa: E731 - every citation supports its claim
    skip = Q.evaluate_scholarqa_cs(ans, cs, rt.llm, attribution_judge=judge)  # official scripts: skipped
    zero = Q.evaluate_scholarqa_cs(ans, cs, rt.llm, attribution_judge=judge, missing="zero")
    assert skip.summary["n"] == 1 and skip.summary["n_missing"] == 1 and len(skip.rows) == 1
    assert zero.summary["n"] == 2 and zero.summary["n_missing"] == 1
    assert zero.summary["Corr"] == pytest.approx(skip.summary["Corr"] / 2)
    assert zero.summary["citation_rec"] == pytest.approx(skip.summary["citation_rec"] / 2)
    assert [r["missing"] for r in zero.rows] == [False, True] and zero.rows[1]["Corr"] == 0.0

    multi = [Query(f"m{i}", "What forms a protein corona?", benchmark="scholarqa_multi",
                   reference_answer="Lipid nanoparticles adsorb serum proteins forming a corona.") for i in range(2)]
    ans = {"m0": Answer("m0", "Lipid nanoparticles adsorb serum proteins forming a corona [0].")}
    skip = Q.evaluate_scholarqa_multi(ans, multi, rt.llm)
    zero = Q.evaluate_scholarqa_multi(ans, multi, rt.llm, missing="zero")
    assert skip.summary["Corr"] == pytest.approx(100.0) and zero.summary["Corr"] == pytest.approx(50.0)
    assert zero.summary["n"] == 2 and zero.summary["n_missing"] == 1 and zero.rows[1]["Corr"] == 0.0
    with pytest.raises(ValueError):
        Q.evaluate_scholarqa_multi(ans, multi, rt.llm, missing="drop")


def test_scholarqa_cs_alternate_case_of_a_failed_question_is_scored():
    """As in rubric_eval.py, a question whose first case has no answer is scored through its alternate case."""
    rt = Runtime.from_config()
    rubric = [RubricItem("The answer should mention binding affinity.", weight=1.0)]
    q1 = Query("cs1", "How is graph learning used for drug discovery?", benchmark="scholarqa_cs", rubric=rubric)
    q2 = Query("cs2", q1.text, benchmark="scholarqa_cs", rubric=rubric)
    ans = {"cs2": Answer("cs2", "Graph neural networks predict binding affinity of ligands [p1].")}
    for missing in ("skip", "zero"):
        ev = Q.evaluate_scholarqa_cs(ans, [q1, q2], rt.llm, missing=missing)
        assert [r["query_id"] for r in ev.rows] == ["cs2"] and ev.missing == ["cs1"] and ev.summary["n"] == 1
