"""run_official_evaluation / scripts/run_official_eval.py with stub official checkouts (no network)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from scholarpath.config import EvaluationConfig
from scholarpath.data.corpus import save_queries
from scholarpath.evaluation.official import OfficialPaths, export_researchqa_data, run_official_evaluation
from scholarpath.types import Answer, AnswerItem, Query, RubricItem

ROOT = Path(__file__).resolve().parents[1]


def _stub_checkouts(tmp_path: Path) -> OfficialPaths:
    rqa = tmp_path / "ResearchQA"
    rqa.mkdir()
    # Official signature: (data_path, response_map_path, output_path, batch_size, model); it reads the data file
    # with researchqa_loader and skips items without a response.
    (rqa / "compute_coverage.py").write_text(
        "import json\n"
        "def compute_coverage(data_path, response_map_path, output_path, batch_size=8, model='gpt-4.1-mini'):\n"
        "    items = json.load(open(data_path))\n"
        "    responses = json.load(open(response_map_path))\n"
        "    out = {}\n"
        "    for it in items:\n"
        "        assert set(it) >= {'id', 'general_domain', 'subdomain', 'field', 'query', 'date', 'rubric'}\n"
        "        if it['id'] in responses:\n"
        "            out[it['id']] = [{'rubric': r['rubric_item'], 'score': 'Completely'} for r in it['rubric']]\n"
        "    json.dump(out, open(output_path, 'w'))\n")
    bench = tmp_path / "ScholarQABench"
    (bench / "scripts").mkdir(parents=True)
    (bench / "scripts" / "citation_correctness_eval.py").write_text(
        "import json, sys\n"
        "a = sys.argv[1:]\n"
        "json.dump({'citation_rec': 80.0, 'citation_prec': 80.0}, open(a[a.index('--f') + 1] + '.score_post_fix', 'w'))\n")
    (bench / "scripts" / "rubric_eval.py").write_text(
        "import json, sys\n"
        "a = sys.argv[1:]\n"
        "qa = a[a.index('--qa-dir') + 1]\n"
        "src = a[a.index('--src-names') + 1]\n"
        "rows = [json.loads(l) for l in open(f'{qa}/{src}.jsonl')]\n"
        "res = {f'{qa}/{src}.jsonl': [{'case_id': r['case_id'], 'scores': {'score': 0.6, 'ann_score': 0.5}} for r in rows]}\n"
        "json.dump(res, open(a[a.index('--output') + 1], 'w'))\n")
    (bench / "scripts" / "prometheus_eval.py").write_text(
        "import json, os, sys\n"
        "a = sys.argv[1:]\n"
        "json.dump({'summary': {'average': 5.0}}, open(os.path.join(a[a.index('-o') + 1], 'results.json'), 'w'))\n")
    return OfficialPaths(researchqa_dir=str(rqa), scholarqabench_dir=str(bench))


def _queries():
    rqa = [Query(f"r{i}", f"question {i}?", benchmark="researchqa", gold_papers=["p1"],
                 rubric=[RubricItem("Does it explain x?", kind="other"), RubricItem("Does it cite p1?", kind="citation")],
                 metadata={"id": f"rqa-{i}", "domain": "cs", "rubric_types": [["Other"], ["Citation"]]})
           for i in range(2)]
    cs = [Query(f"c{i}", f"cs question {i}?", benchmark="scholarqa_cs", rubric=[RubricItem("x", weight=1.0)],
                metadata={"case_id": f"case-{i}"}) for i in range(2)]
    multi = [Query(f"m{i}", f"multi question {i}?", benchmark="scholarqa_multi", reference_answer="expert")
             for i in range(2)]
    return rqa, cs, multi


def _answer(qid: str) -> Answer:
    return Answer(qid, "", items=[AnswerItem("A long enough statement about the topic of the question here.",
                                             citations=["p1"])], metadata={"passages": {"p1": "text of p1"}})


def test_export_researchqa_data_matches_the_official_loader(tmp_path):
    rqa, _, _ = _queries()
    rec = export_researchqa_data(rqa, tmp_path / "data.json")[0]
    assert rec["id"] == "rqa-0" and rec["query"] == "question 0?" and rec["general_domain"] == "cs"
    assert rec["rubric"] == [{"rubric_item": "Does it explain x?", "type": ["Other"], "citation_metadata": None},
                             {"rubric_item": "Does it cite p1?", "type": ["Citation"], "citation_metadata": None}]


def test_official_paths_prefer_config_then_environment(monkeypatch):
    monkeypatch.setenv("SCHOLARPATH_RESEARCHQA_DIR", "/env/rqa")
    monkeypatch.setenv("SCHOLARPATH_SCHOLARQABENCH_DIR", "/env/bench")
    paths = OfficialPaths.from_config(EvaluationConfig(scholarqabench_dir="/cfg/bench"))
    assert paths.researchqa_dir == "/env/rqa" and paths.scholarqabench_dir == "/cfg/bench"
    assert OfficialPaths.from_config(EvaluationConfig(), researchqa_dir="/cli").researchqa_dir == "/cli"


def test_run_official_evaluation_counts_failed_queries_as_zero(tmp_path):
    paths = _stub_checkouts(tmp_path)
    rqa, cs, multi = _queries()
    one = {q.query_id: _answer(q.query_id) for q in (rqa[0], cs[0], multi[0])}  # the second query of each failed
    r = run_official_evaluation("researchqa", one, rqa, tmp_path / "r", paths, missing="skip")
    assert r["Rub"] == pytest.approx(100.0) and r["Cite"] == pytest.approx(100.0) and r["n_missing"] == 1
    r = run_official_evaluation("researchqa", one, rqa, tmp_path / "r0", paths)
    assert r["Rub"] == pytest.approx(50.0) and r["Cite"] == pytest.approx(50.0) and r["n"] == 2
    c = run_official_evaluation("scholarqa_cs", one, cs, tmp_path / "c", paths, system="sp")
    assert c["Corr"] == pytest.approx(30.0) and c["citation_prec"] == pytest.approx(40.0)
    assert c["Cite"] == pytest.approx(40.0)
    m = run_official_evaluation("scholarqa_multi", one, multi, tmp_path / "m", paths, missing="skip")
    assert m["Corr"] == pytest.approx(100.0) and m["Cite"] == pytest.approx(80.0)
    assert json.loads((tmp_path / "m" / "official_summary.json").read_text())["n_missing"] == 1
    with pytest.raises(ValueError):
        run_official_evaluation("taxobench", one, rqa, tmp_path / "x", paths)


def test_run_official_eval_script_on_cached_answers(tmp_path):
    from scholarpath.experiments.qa import RecordStore, RunRecord

    paths = _stub_checkouts(tmp_path)
    rqa, cs, multi = _queries()
    data = tmp_path / "data"
    save_queries(data / "researchqa" / "queries.jsonl", rqa)
    save_queries(data / "scholarqabench" / "cs.jsonl", cs)
    save_queries(data / "scholarqabench" / "multi.jsonl", multi)
    store = RecordStore(tmp_path / "answers")
    for bench, queries in (("researchqa", rqa), ("scholarqa_cs", cs), ("scholarqa_multi", multi)):
        for q in queries:
            store.append(RunRecord(q.query_id, bench, "scholarpath", 8192, answer=_answer(q.query_id)))
    spec = importlib.util.spec_from_file_location("run_official_eval", ROOT / "scripts" / "run_official_eval.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = tmp_path / "out"
    argv = ["--answers-dir", str(tmp_path / "answers"), "--data-dir", str(data), "--output-dir", str(out),
            "--researchqa-dir", paths.researchqa_dir, "--scholarqabench-dir", paths.scholarqabench_dir]
    assert script.main(argv) == 0
    table = json.loads((out / "summary.json").read_text())["table"]["scholarpath (B8192)"]
    assert table["researchqa.Rub"] == pytest.approx(100.0)
    assert table["scholarqabench.Corr"] == pytest.approx((2 * 60.0 + 2 * 100.0) / 4)
    assert (out / "researchqa" / "scholarpath" / "B8192" / "responses.json").exists()
    assert script.main(argv[:2] + ["--data-dir", str(data), "--benchmarks", "chime"]) == 2
    dry = tmp_path / "dry"
    assert script.main(argv[:4] + ["--output-dir", str(dry), "--export-only", "--budget", "8192"]) == 0
    assert (dry / "scholarqa_multi" / "scholarpath" / "B8192" / "prometheus_gold.json").exists()
    assert not (dry / "summary.json").exists()  # nothing was scored
