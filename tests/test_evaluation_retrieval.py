"""Tests for Recall@B / NDCG@B, efficiency accounting, reporting and official exporters."""

from __future__ import annotations

import csv
import importlib.util
import json
import math

import pytest

from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.evaluation.efficiency import EfficiencyTracker, measure, summarize_efficiency
from scholarpath.evaluation.official import (
    coverage_from_judgements,
    export_citation_eval,
    export_prometheus,
    export_researchqa_responses,
    export_scholarqa_cs,
)
from scholarpath.evaluation.reporting import (
    TABLE2,
    TABLE3,
    TABLE5,
    flatten_benchmarks,
    markdown_table,
    organization_table_row,
    sweep_long_rows,
    sweep_wide_rows,
    write_budget_sweep,
    write_jsonl,
    write_summary_csv,
    write_table,
)
from scholarpath.evaluation.retrieval_metrics import (
    canonical_benchmark,
    evaluate_retrieval,
    ndcg_at_budget,
    papers_within_budget,
    recall_at_budget,
)
from scholarpath.prompts.answer import compose_answer_request
from scholarpath.prompts.evaluation import rubric_coverage_request
from scholarpath.retrieval.result import RetrievalResult, RetrievedUnit
from scholarpath.runtime import Runtime
from scholarpath.types import Answer, AnswerItem, Query


def _log2(i: int) -> float:
    return math.log2(i)


# ---------------------------------------------------------------- Recall / NDCG


def test_recall_and_ndcg_hand_computed_with_unaligned_gold():
    ranked = ["a", "x", "a", "b"]  # duplicate "a" is ignored -> a, x, b
    gold = ["a", "b", "c"]
    num_gold = 4  # one unaligned gold citation stays in the denominator
    assert recall_at_budget(ranked, gold, num_gold) == pytest.approx(2 / 4)
    assert recall_at_budget(ranked, gold) == pytest.approx(2 / 3)

    dcg = 1 / _log2(2) + 1 / _log2(4)  # hits at ranks 1 and 3
    idcg_all = sum(1 / _log2(i + 1) for i in range(1, 5))  # four gold positions
    idcg_list = sum(1 / _log2(i + 1) for i in range(1, 4))  # list length 3
    assert ndcg_at_budget(ranked, gold, num_gold) == pytest.approx(dcg / idcg_all)
    assert ndcg_at_budget(ranked, gold, num_gold, ideal="list") == pytest.approx(dcg / idcg_list)

    # A perfect within-budget list reaches 1 only when every gold paper is aligned.
    assert ndcg_at_budget(["a", "b", "c"], gold) == pytest.approx(1.0)
    assert ndcg_at_budget(["a", "b", "c"], gold, num_gold) < 1.0
    assert recall_at_budget([], gold, num_gold) == 0.0 and ndcg_at_budget([], gold, num_gold) == 0.0
    assert recall_at_budget(["a"], [], 0) is None and ndcg_at_budget(["a"], []) is None
    with pytest.raises(ValueError):
        recall_at_budget(["a"], gold, num_gold=2)


def test_papers_within_budget_repacks_overlong_results():
    units = [RetrievedUnit("u1", "abstract", tokens=60, papers=["p1"]),
             RetrievedUnit("u2", "abstract", tokens=50, papers=["p2"]),
             RetrievedUnit("u3", "abstract", tokens=30, papers=["p3", "p1"])]
    res = RetrievalResult("q", units=units, context_tokens=140, ranked_papers=["p1", "p2", "p3"])
    assert papers_within_budget(res) == ["p1", "p2", "p3"]
    # B = 100: u1 (60) fits, u2 (50) does not, u3 (30) fits.
    assert papers_within_budget(res, budget=100) == ["p1", "p3"]


def test_evaluate_retrieval_macro_per_benchmark():
    queries = [
        Query("r1", "q", benchmark="ResearchQA", gold_papers=["a", "b"], unaligned_gold=["z"]),
        Query("r2", "q", benchmark="researchqa", gold_papers=["c"]),
        Query("s1", "q", benchmark="scholarqa_multi", gold_papers=["d", "e"]),
        Query("s2", "q", benchmark="scholarqa_cs"),  # no gold citations -> skipped
        Query("s3", "q", benchmark="scholarqa_multi", gold_papers=["f"]),  # no output -> zero
    ]
    results = {
        "r1": RetrievalResult("r1", ranked_papers=["a", "x", "b"], context_tokens=900),
        "r2": ["c"],
        "s1": Answer("s1", "", retrieved_papers=["e"]),
        "s2": ["a"],
    }
    ev = evaluate_retrieval(results, queries, budget=1000)
    assert ev.skipped == ["s2"] and ev.missing == ["s3"]
    assert set(ev.by_benchmark) == {"researchqa", "scholarqabench"}
    r_research = (100 * 2 / 3 + 100.0) / 2
    r_scholar = (100 * 1 / 2 + 0.0) / 2
    assert ev.by_benchmark["researchqa"]["R"] == pytest.approx(r_research)
    assert ev.by_benchmark["scholarqabench"]["R"] == pytest.approx(r_scholar)
    assert ev.macro["R"] == pytest.approx((r_research + r_scholar) / 2)
    n_s1 = (1 / _log2(2)) / (1 + 1 / _log2(3))  # "e" retrieved at rank 1 of 2 gold
    assert ev.by_benchmark["scholarqabench"]["N"] == pytest.approx(100 * n_s1 / 2)
    row = ev.table_row()
    assert row["researchqa.R"] == pytest.approx(r_research) and "macro.N" in row
    assert canonical_benchmark("ScholarQA-CS") == "scholarqabench"


# ---------------------------------------------------------------- Efficiency


def test_efficiency_tracker_excludes_judge_and_offline_components():
    rt = Runtime.from_config()
    tracker = EfficiencyTracker.from_config(rt.meter, rt.config.llm, system="ScholarPath")

    def answer_query() -> Answer:
        rt.llm.complete(compose_answer_request("How?", "ctx", sources=[{"paper_id": "p1", "title": "T"}]))
        rt.llm.complete(rubric_coverage_request("an answer", ["Does it?"]))  # judge: excluded
        rt.meter.record(10_000, 0, component="index_build")  # offline indexing: excluded
        return Answer("q1", "text")

    ans = tracker.measure("q1", answer_query)
    rec = tracker.records[-1]
    assert rec.calls == 1 and set(rec.components) == {"composition"}
    comp = rt.meter.component("composition")
    expected = (comp.input_tokens * 0.14 + comp.output_tokens * 0.28) / 1e6
    assert rec.cost_usd == pytest.approx(expected) and rec.tokens == comp.total_tokens
    assert rec.latency_s >= 0.0
    assert ans.llm_calls == 1 and ans.cost_usd == pytest.approx(expected)  # annotated

    tracker.record_answer(Answer("q2", "", llm_calls=3, input_tokens=2000, output_tokens=1000, latency_s=2.0))
    s = tracker.summary()
    assert s["Calls"] == pytest.approx(2.0) and s["Latency (s)"] >= 1.0
    assert s["Tokens (K)"] == pytest.approx((rec.tokens + 3000) / 2 / 1000)

    _, single = measure(lambda: rt.llm.complete(rubric_coverage_request("a", ["b?"])), rt.meter, query_id="q3")
    assert single.calls == 0 and single.cost_usd == 0.0
    assert summarize_efficiency([])["Calls"] is None


# ---------------------------------------------------------------- Reporting


def test_reporting_writes_tables_rows_and_sweeps(tmp_path):
    rows = {
        "BM25": {"researchqa.R": 46.8, "researchqa.N": 39.4},
        "ScholarPath": {"researchqa.R": 70.3, "researchqa.N": 61.8},
        "PaperQA2": {"researchqa.R": 50.0},
    }
    sections = [("Budgeted methods (B = 8,192)", ["BM25", "ScholarPath"]), ("Unbudgeted methods", ["PaperQA2"])]
    md = markdown_table(rows, TABLE2, sections)
    assert "| Method | ResearchQA R | ResearchQA N |" in md
    assert "**70.3**" in md and "**61.8**" in md and "46.8" in md and "*Unbudgeted methods*" in md

    path = write_table(tmp_path / "table2", rows, TABLE2, sections)
    assert path.suffix == ".md" and path.exists()
    with path.with_suffix(".csv").open() as f:
        table = list(csv.DictReader(f))
    assert [r["Method"] for r in table] == ["BM25", "ScholarPath", "PaperQA2"]
    assert table[1]["ResearchQA R"] == "70.3" and table[2]["ResearchQA N"] == ""

    eff = {"Keyword search": {"Calls": 1.0, "Cost ($)": 0.002}, "Codex": {"Calls": 34.8, "Cost ($)": 0.046}}
    md5 = markdown_table(eff, TABLE5)
    assert "**1.0**" in md5 and "**0.002**" in md5  # lower is better

    write_jsonl(tmp_path / "per_query.jsonl", [{"query_id": "q1", "R": 50.0, "nan": float("nan")}])
    line = json.loads((tmp_path / "per_query.jsonl").read_text().splitlines()[0])
    assert line == {"query_id": "q1", "R": 50.0, "nan": None}
    write_summary_csv(tmp_path / "summary.csv", {"ScholarPath": {"R": 70.3}, "BM25": {"R": 46.8, "N": 39.4}})
    assert (tmp_path / "summary.csv").read_text().splitlines()[0] == "system,R,N"

    sweep = {"ScholarPath": {2048: {"R": 58.0}, 8192: {"R": 70.3}}, "BM25": {2048: {"R": 35.0}, 8192: {"R": 46.8}}}
    files = write_budget_sweep(tmp_path / "fig5.csv", sweep)
    assert files["long"].read_text().count("\n") == 5 and files["R"].read_text().startswith("system,2048,8192")
    if importlib.util.find_spec("matplotlib") is None:
        return
    from scholarpath.evaluation.reporting import plot_budget_sweep

    assert plot_budget_sweep(sweep, "R", tmp_path / "fig5.png").exists()


def test_budget_sweep_survives_json_round_trip():
    sweep = {"ScholarPath": {2048: {"R": 58.0}, 8192: {"R": 70.3}}}
    restored = json.loads(json.dumps(sweep))  # budgets become "2048", "8192"
    assert sweep_wide_rows(restored, "R") == sweep_wide_rows(sweep, "R") == [
        {"system": "ScholarPath", "2048": 58.0, "8192": 70.3}]
    assert sweep_long_rows(restored)[0] == {"system": "ScholarPath", "budget": 2048, "metric": "R", "value": 58.0}


def test_table3_row_from_qa_and_organization_summaries():
    # Organization metrics report fractions under lower-case keys (organization_metrics.organization_scores).
    row = {**flatten_benchmarks({"researchqa": {"Rub": 74.2, "Cite": 68.3},
                                 "scholarqabench": {"Corr": 70.4, "Cite": 75.6}}),
           **organization_table_row({"ari": 0.558, "sem": 0.753, "n_queries": 72.0}, "taxobench"),
           **organization_table_row({"cat": 0.761, "assn": 0.698}, "CHIME")}
    assert row["taxobench.ARI"] == pytest.approx(55.8) and row["chime.Assn"] == pytest.approx(69.8)
    assert set(c.key for c in TABLE3.columns) <= set(row)
    md = markdown_table({"ScholarPath": row}, TABLE3, title=False)
    assert "| ScholarPath | 74.2 | 68.3 | 70.4 | 75.6 | 55.8 | 75.3 | 76.1 | 69.8 |" in md
    assert organization_table_row({"ari": None}, "taxobench") == {"taxobench.ARI": None, "taxobench.Sem": None}
    with pytest.raises(ValueError):
        organization_table_row({}, "researchqa")
    pytest.importorskip("sklearn")
    from scholarpath.evaluation import organization_metrics as O

    assert {k for keys in O.ORGANIZATION_METRICS.values() for k in keys} == {"ari", "sem", "cat", "assn"}


def test_package_exports_are_lazy_and_complete():
    import os
    import subprocess
    import sys
    from pathlib import Path

    import scholarpath.evaluation as E
    from scholarpath.evaluation import efficiency, hypothesis_metrics, judges, official, qa_metrics, reporting
    from scholarpath.evaluation import organization_metrics, retrieval_metrics, taxonomy_metrics

    for module in (retrieval_metrics, qa_metrics, hypothesis_metrics, judges, efficiency, reporting, official,
                   taxonomy_metrics, organization_metrics):
        missing = [n for n in module.__all__ if n not in E.__all__ and n not in ("PRF", "f1_score")]
        assert not missing, (module.__name__, missing)
    assert E.flatten_benchmarks is reporting.flatten_benchmarks and E.official_id is official.official_id
    assert E.PRF is qa_metrics.PRF  # name clashes resolve to the QA metrics
    assert E.TaxonomyScorer is taxonomy_metrics.TaxonomyScorer and E.soft_node_f1 is taxonomy_metrics.soft_node_f1
    assert "soft_node_f1" in E.__all__ and "adjusted_rand_index" in E.__all__
    with pytest.raises(AttributeError):
        E.no_such_symbol  # noqa: B018
    # Importing one submodule must not import the optional (sklearn-backed) metric modules.
    code = ("import sys, scholarpath.evaluation.retrieval_metrics; "
            "print(any(m in sys.modules for m in ('scholarpath.evaluation.organization_metrics', "
            "'scholarpath.evaluation.taxonomy_metrics', 'sklearn')))")
    src = str(Path(E.__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    assert out.strip() == "False"


# ---------------------------------------------------------------- Official formats


def test_official_exporters(tmp_path):
    bench = make_synthetic_benchmark(n_papers=40, seed=3)
    q = bench.evidence_queries[0]
    q.metadata["official_id"] = "174539437936812632-s1"
    q.reference_answer = "Expert answer."
    pid = q.gold_papers[0]
    ans = Answer(q.query_id, f"Finding about the topic is described in detail here. [{pid}] Unknown [nope]. "
                             "A [CLS] token and the range [0, 1] are not citations.",
                 items=[AnswerItem("Finding.", citations=[pid, "nope"])])
    answers = {q.query_id: ans}

    resp = export_researchqa_responses(answers, tmp_path / "responses.json", [q], papers=bench.corpus)
    assert list(resp) == ["174539437936812632-s1"]
    assert "References:" in resp["174539437936812632-s1"]["answer"] and bench.corpus[pid].title in json.dumps(resp)

    # ScholarQA-CS: the convert_answer_scholar_cs.py format (numbered citations + References block).
    rows = export_scholarqa_cs(answers, tmp_path / "qa" / "scholarpath.jsonl", [q], papers=bench.corpus)
    assert rows[0]["case_id"] == "174539437936812632-s1" and (tmp_path / "qa" / "scholarpath.jsonl").exists()
    paper = bench.corpus[pid]
    assert rows[0]["answer_text"].endswith(f"\nReferences:\n[0] {paper.title} {paper.abstract}\n\n")
    plain = export_scholarqa_cs(answers, tmp_path / "qa" / "plain.jsonl", [q], references=False)
    assert plain[0]["answer_text"] == ans.text

    items = export_citation_eval(answers, [q], tmp_path / "cite.json", papers=bench.corpus)
    out = items[0]["output"]
    # paper id -> 0-based index; a structured citation without a passage -> out of range; other brackets stay text.
    assert "here [0]." in out and "Unknown [1]." in out and pid not in out
    assert "[CLS]" in out and "(0, 1)" in out
    assert items[0]["ctxs"] == [{"title": paper.title, "text": paper.abstract}]

    untitled = Answer("u", "A finding that is described in enough detail here [p1].",
                      metadata={"passages": {"p1": "raw passage text"}})
    uq = Query("u", "question?")
    assert export_citation_eval({"u": untitled}, [uq], tmp_path / "u.json")[0]["ctxs"] == [{"text": "raw passage text"}]

    preds = export_prometheus(answers, [q], tmp_path / "preds.json", gold_path=tmp_path / "gold.json")
    gold = json.loads((tmp_path / "gold.json").read_text())
    assert len(preds) == len(gold) == 1 and gold[0]["output"] == "Expert answer."

    cov = coverage_from_judgements({"a": [{"rubric": "x", "score": "Completely"}, {"rubric": "y", "score": "Barely"}],
                                    "b": [{"rubric": "z", "score": "Mostly."}]})
    assert cov.per_answer["a"] == pytest.approx((1.0 + 0.25) / 2) and cov.per_answer["b"] == pytest.approx(0.75)
    assert cov.rub == pytest.approx(100 * (0.625 + 0.75) / 2)


def test_official_runners_with_stub_scripts(tmp_path):
    """The runners build the official command lines and parse their outputs (stub scripts, no network)."""
    from scholarpath.evaluation.official import (
        OfficialPaths,
        run_citation_eval,
        run_compute_coverage,
        run_prometheus_eval,
        run_rubric_eval,
    )

    rqa = tmp_path / "ResearchQA"
    rqa.mkdir()
    # dr-tulu style signature: the second parameter is the loaded response map.
    (rqa / "compute_coverage.py").write_text(
        "import json\n"
        "def compute_coverage(data_path, response_map, output_path=None, batch_size=8, model='gpt-4.1-mini'):\n"
        "    assert isinstance(response_map, dict) and batch_size == 8 and model == 'judge-x'\n"
        "    out = {k: [{'rubric': 'r', 'score': 'Mostly'}, {'rubric': 's', 'score': 'Completely'}] for k in response_map}\n"
        "    json.dump(out, open(output_path, 'w'))\n")
    bench = tmp_path / "ScholarQABench"
    (bench / "scripts").mkdir(parents=True)
    (bench / "scripts" / "citation_correctness_eval.py").write_text(
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "assert '--citations' in args and args[args.index('--at_most_citations') + 1] == '3'\n"
        "json.dump({'citation_rec': 60.0, 'citation_prec': 40.0}, open(args[args.index('--f') + 1] + '.score_post_fix', 'w'))\n")
    (bench / "scripts" / "rubric_eval.py").write_text(
        "import json, sys\n"
        "a = sys.argv[1:]\n"
        "assert '--rubrics' in a and '--snippets' in a\n"
        "res = {a[a.index('--qa-dir') + 1] + '/sp.jsonl': [{'case_id': 'c', 'scores': {'score': 0.5, 'ann_score': 0.4}}]}\n"
        "json.dump(res, open(a[a.index('--output') + 1], 'w'))\n")
    (bench / "scripts" / "prometheus_eval.py").write_text(
        "import json, os, sys\n"
        "a = sys.argv[1:]\n"
        "assert a[a.index('--aspects') + 1:a.index('--aspects') + 4] == ['organization', 'coverage', 'relevance']\n"
        "assert '-f' in a and '--load_vllm' in a\n"
        "json.dump({'summary': {'average': 4.0}}, open(os.path.join(a[a.index('-o') + 1], 'results.json'), 'w'))\n")
    paths = OfficialPaths(researchqa_dir=str(rqa), scholarqabench_dir=str(bench))

    (tmp_path / "data.json").write_text("[]")
    (tmp_path / "responses.json").write_text(json.dumps({"q1": {"answer": "a"}}))
    cov = run_compute_coverage(tmp_path / "data.json", tmp_path / "responses.json", tmp_path / "cov.json",
                               paths=paths, model="judge-x")
    assert cov.per_answer == {"q1": pytest.approx(0.875)} and cov.rub == pytest.approx(87.5)

    (tmp_path / "cite.json").write_text("[]")
    cite = run_citation_eval(tmp_path / "cite.json", paths=paths)
    assert cite["Cite"] == pytest.approx(2 * 40 * 60 / 100)

    rub = run_rubric_eval(tmp_path / "qa", tmp_path / "rubric_out.json", paths=paths)
    assert rub["sp"]["Corr"] == pytest.approx(50.0) and rub["sp"]["ann_score"] == pytest.approx(40.0)

    prom = run_prometheus_eval(tmp_path / "preds.json", tmp_path / "prom", gold_path=tmp_path / "gold.json",
                               paths=paths)
    assert prom["Corr"] == pytest.approx(75.0)
