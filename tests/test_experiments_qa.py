"""Tests of the query-answering experiments (scholarpath.experiments.qa and the scripts of Tables 2-5, 7).

Everything runs offline on the synthetic processed-data layout
(scripts/make_synthetic_data.py) with the mock LLM, mock judges and the
hashing embedder, with tiny settings (few queries, budgets 1024 / 2048).
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from scholarpath.data.corpus import Corpus, load_queries
from scholarpath.experiments import qa
from scholarpath.index.semantic_index import SemanticIndex
from scholarpath.retrieval.result import RetrievalResult, RetrievedUnit
from scholarpath.runtime import Runtime
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.text import count_tokens
from scholarpath.types import Answer, AnswerItem, Query

ROOT = Path(__file__).resolve().parents[1]
# Keep the offline searches small: hypothesis candidates / chunks per paper (see run_retrieval_ablation.py).
FAST_RETRIEVAL = ["--set", "retrieval.max_hypothesis_candidates=6", "--set", "retrieval.max_chunks_per_paper=1"]
FAST_GRAPH = ["--graph-candidates", "6", "--raptor-clustering", "kmeans", "--raptor-no-umap"]


def load_script(name: str) -> Any:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_qa_test_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _single_thread():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover
        yield
        return
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module", autouse=True)
def _blas_single_thread():
    """Avoid BLAS over-subscription when several test processes share the CPU."""
    with _single_thread():
        yield


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> Dict[str, Path]:
    """Synthetic layout + semantic indexes (gold taxonomies) of arxiv_subfields and taxobench."""
    root = tmp_path_factory.mktemp("qa")
    data = root / "data"
    assert load_script("make_synthetic_data").main(["--out", str(data), "--n-papers", "120"]) == 0
    build = load_script("build_index")
    for corpus in ("arxiv_subfields", "taxobench"):
        assert build.main([
            "--data-dir", str(data), "--corpus", corpus, "--index-dir", str(root / "index"),
            "--problem-taxonomy", str(data / "gold" / "problem_taxonomy.json"),
            "--method-taxonomy", str(data / "gold" / "method_taxonomy.json"),
            "--output-dir", str(root / "results" / "build_index"),
        ]) == 0
    return {"root": root, "data": data, "index": root / "index", "results": root / "results"}


def _common(world: Dict[str, Path], name: str) -> List[str]:
    return ["--data-dir", str(world["data"]), "--index-dir", str(world["index"]),
            "--output-dir", str(world["results"] / name)]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parsing_helpers():
    assert qa.parse_budgets("2K,4K,6K,8192,16K") == [2048, 4096, 6144, 8192, 16384]
    assert qa.parse_budgets("none") == [] and qa.parse_budgets([1024]) == [1024]
    assert qa.resolve_system("Dense + reranker") == "dense_rerank"
    assert qa.resolve_system("SPECTER2") == "dense" and qa.resolve_system("ScholarPath") == "scholarpath"
    assert qa.resolve_systems("agents", qa.QA_SYSTEMS) == list(qa.AGENT_SYSTEMS)
    assert qa.resolve_benchmarks("scholarqabench") == ["scholarqa_cs", "scholarqa_multi"]
    assert qa.resolve_benchmarks(None, ["synthetic", "heldout"], default=qa.EVIDENCE_BENCHMARKS) == ["synthetic"]
    with pytest.raises(FileNotFoundError):
        qa.resolve_benchmarks("researchqa", ["synthetic"])
    assert qa.parse_budget_tag(qa.budget_tag(8192)) == 8192 and qa.parse_budget_tag(qa.budget_tag(None)) is None
    assert qa.resolve_kind("hypothesis") == "hypothesis_discovery" and qa.resolve_kind("auto") == ""
    assert [c.key for c in qa.table2_spec(["synthetic", "researchqa"]).columns][:2] == ["researchqa.R", "researchqa.N"]


def test_record_store_roundtrip(tmp_path):
    store = qa.RecordStore(tmp_path)
    unit = RetrievedUnit(key="u1", kind="direct", text="long text", tokens=12, papers=["p1"], nodes=["paper:p1"])
    ret = qa.compact_retrieval(RetrievalResult("q1", [unit], context="ctx", context_tokens=12, budget=100,
                                               ranked_papers=["p1"], stats={"analysis": {}, "selected": 1}))
    assert ret.context == "" and ret.units[0].text == "" and "analysis" not in ret.stats
    ans = Answer("q1", "text", [AnswerItem("claim", citations=["p1"])], retrieved_papers=["p1"], llm_calls=1)
    store.append(qa.RunRecord("q1", "synthetic", "scholarpath", 100, answer=ans, retrieval=ret,
                              efficiency={"query_id": "q1", "calls": 1}))
    store.append(qa.RunRecord("q2", "synthetic", "scholarpath", 100, error="boom"))
    store.append(qa.RunRecord("q2", "synthetic", "scholarpath", 100, answer=ans))  # last record wins
    loaded = store.load("synthetic", "scholarpath", 100)
    assert loaded["q1"].answer.items[0].citations == ["p1"] and loaded["q1"].retrieval.ranked_papers == ["p1"]
    assert loaded["q2"].ok and loaded["q2"].error is None
    assert store.entries()[0][:3] == ("synthetic", "scholarpath", 100)
    # An interrupted --resume under a new configuration: aggregate the latest configuration only.
    store.append(qa.RunRecord("q1", "synthetic", "scholarpath", 100, answer=ans, fingerprint="new"))
    latest = qa.RecordStore.load_latest(store.path("synthetic", "scholarpath", 100))
    assert list(latest) == ["q1"] and latest["q1"].fingerprint == "new"
    assert qa.RecordStore.load_latest(tmp_path / "missing.jsonl") == {}


def test_records_to_hierarchy_and_operator_summary():
    q = Query("o1", "Graph learning", metadata={"task": "organization", "style": "survey"})
    ans = Answer("o1", "", [AnswerItem("GNNs for molecules", citations=["p1", "p2", "bad"]),
                            AnswerItem("Contrastive", candidate="Graph contrastive learning", citations=["p3"]),
                            AnswerItem("no citations")])
    tax = qa.answer_hierarchy(ans, q, valid_ids=["p1", "p2", "p3"])
    assert sorted(tax.nodes[c].name for c in tax.children(tax.root_id)) == ["GNNs for molecules",
                                                                           "Graph contrastive learning"]
    assert tax.all_papers() == {"p1", "p2", "p3"}
    stored = Answer("o1", "", metadata={"hierarchy": tax.to_dict()})
    assert qa.answer_hierarchy(stored, q).all_papers() == {"p1", "p2", "p3"}
    assert qa.agent_task_query(q).text.startswith("Retrieve the key papers")
    # Sec. 5.3.3 / Table 7: candidate-node pruning = 1 - scored candidates / candidates in scope over the
    # hypothesis queries (evidence queries have no candidate pool); index nodes are a separate statistic.
    rows = [{"postings_total": 100, "postings_pruned": 90, "scored_nodes": 400, "search_ms": 2.0,
             "scored_candidates": 312, "candidate_pool": 4628, "scored_nodes_exhaustive": 500,
             "exhaustive_search_ms": 5.0, "same_selection": True},
            {"postings_total": 100, "postings_pruned": 80, "scored_nodes": 30, "search_ms": 4.0,
             "scored_candidates": 0, "candidate_pool": 0, "scored_nodes_exhaustive": 100,
             "exhaustive_search_ms": 5.0, "same_selection": True}]
    s = qa.summarize_operator_stats(rows)
    assert s["postings_pruned_pct"] == pytest.approx(85.0)
    assert s["candidate_nodes_pruned_pct"] == pytest.approx(93.258, abs=1e-3)  # the paper's 93.3%
    assert s["scored_candidates_mean"] == 312 and s["n_hypothesis"] == 1.0
    assert s["index_nodes_pruned_pct"] == pytest.approx(100.0 * (1 - 430 / 600))
    assert s["search_ms_mean"] == pytest.approx(3.0) and s["same_selection_rate"] == 1.0
    assert qa.summarize_operator_stats(rows[1:])["candidate_nodes_pruned_pct"] is None


# ---------------------------------------------------------------------------
# Experiment functions (library API)
# ---------------------------------------------------------------------------


def test_environment_runs_and_evaluation(world, tmp_path):
    rt = Runtime.from_config()
    layout = qa.DataLayout(world["data"])
    envs = qa.EnvironmentCache(rt, layout, qa.IndexStore(world["index"]),
                               qa.SystemOptions(graph_global_max_papers=0, graph_candidates=6,
                                                raptor_clustering="kmeans", raptor_use_umap=False))
    store = qa.RecordStore(tmp_path / "runs")
    queries = layout.load_queries("synthetic", limit=3)
    env = envs.for_benchmark("synthetic")
    recs = qa.run_queries(env, "scholarpath", queries, 1024, benchmark="synthetic", mode="retrieval", store=store,
                          fail_fast=True)
    for q in queries:
        r = recs[q.query_id].retrieval
        assert r.context_tokens <= 1024 and sum(u.tokens for u in r.units) <= 1024
        eff = recs[q.query_id].efficiency
        assert eff["calls"] >= 0 and eff["latency_s"] >= 0 and "composition_calls" in eff
    ev = qa.evaluate_retrieval_records(recs, queries, 1024, "synthetic")
    comb = qa.combine_retrieval({"synthetic": ev})
    assert 0.0 <= comb["macro"]["R"] <= 100.0 and set(comb["row"]) >= {"synthetic.R", "synthetic.N", "macro.R"}

    # End-to-end QA: one composition call per evidence answer (+ Step 1 query understanding for ScholarPath).
    answers = qa.run_queries(env, "dense", queries, 2048, benchmark="synthetic", store=store, fail_fast=True)
    for rec in answers.values():
        assert rec.answer.context_tokens <= 2048 and rec.efficiency["composition_calls"] == 1
    judges = qa.build_judges(rt.config.evaluation)
    res = qa.evaluate_benchmark("synthetic", qa.answers_of(answers), queries, judges, papers=env.index.papers)
    assert 0.0 <= res.summary["Cite"] <= 100.0

    # Organization queries: retrieve <= B, one hierarchy-composition call, ARI / Sem against the gold tree.
    org = layout.load_queries("taxobench", limit=2)
    tenv = envs.for_benchmark("taxobench")
    orecs = qa.run_queries(tenv, "scholarpath", org, 2048, benchmark="taxobench", store=store, fail_fast=True)
    for rec in orecs.values():
        assert rec.answer.metadata["task"] == "organization" and rec.answer.metadata["hierarchy"]["nodes"]
        assert rec.efficiency["composition_calls"] == 1
    ores = qa.evaluate_benchmark("taxobench", qa.answers_of(orecs), org, judges, papers=tenv.index.papers,
                                 embedder=rt.embedder, layout=layout)
    row = qa.qa_table_row({"taxobench": ores})
    assert set(row) == {"taxobench.ARI", "taxobench.Sem", "macro.Avg"}
    assert all(v is None or -100.0 <= v <= 100.0 for v in row.values())

    # Table 5 aggregation from the cached records.
    eff = qa.collect_efficiency([store], budget=2048)
    assert set(eff) == {"dense", "scholarpath"}
    s = qa.efficiency_summary(eff["dense"])
    assert s["Calls"] == pytest.approx(1.0) and s["Composition calls"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------


def test_build_index_with_the_scholarpath_builder(world):
    out = world["root"] / "index_small"
    assert load_script("build_index").main([
        "--data-dir", str(world["data"]), "--corpus", "arxiv_subfields", "--name", "small", "--limit-papers", "40",
        "--index-dir", str(out), "--set", "builder.max_depth=1", "--set", "builder.router.hidden_dim=16",
        "--set", "builder.router.epochs=5", "--set", "builder.router.min_steps=20",
        "--output-dir", str(world["results"] / "build_index_small"),
    ]) == 0
    summary = json.loads((out / "small" / "build_summary.json").read_text())
    assert summary["taxonomy_method"] == "scholarpath" and summary["index"]["papers"] == 40
    assert summary["llm_calls"] > 0 and summary["cost_usd"] > 0
    assert (out / "small" / "taxonomy_problem" / "summary.json").exists()


def _train_link_predictor(world: Dict[str, Path]) -> int:
    return load_script("train_link_predictor").main([
        "--index-dir", str(world["index"]), "--corpus", "arxiv_subfields",
        "--set", "link_prediction.cutoffs=[2016-01-01,2017-01-01,2018-01-01,2019-01-01]",
        "--set", "link_prediction.horizon_years=1", "--set", "link_prediction.epochs=3",
        "--set", "link_prediction.embed_dim=16", "--set", "link_prediction.hidden_dim=16",
        "--output-dir", str(world["results"] / "train_link_predictor"),
    ])


def test_train_link_predictor_script(world, capsys):
    assert _train_link_predictor(world) == 0
    lp_dir = world["index"] / "arxiv_subfields" / qa.LINK_PREDICTOR_DIR
    assert (lp_dir / "link_predictor.json").exists() and (lp_dir / "training_report.json").exists()
    assert qa.load_link_predictor(world["index"] / "arxiv_subfields") is not None
    assert "Link predictor" in capsys.readouterr().out


def test_run_retrieval_eval_script(world):
    args = _common(world, "rre") + ["--benchmarks", "synthetic,taxobench", "--budgets", "1024,2048", "--limit", "3",
                                    "--graph-global-max-papers", "0", "--no-plots"] + FAST_GRAPH
    script = load_script("run_retrieval_eval")
    assert script.main(args) == 0
    out = world["results"] / "rre"
    header = (out / "table2.csv").read_text().splitlines()[0]
    assert "TaxoBench R" in header and "Synthetic N" in header
    with open(out / "table2.csv", newline="") as f:
        methods = [row["Method"] for row in csv.DictReader(f)]
    assert methods == ["BM25", "SPECTER", "Dense + reranker", "RAPTOR", "LightRAG", "ScholarPath"]
    for path in (out / "per_query").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert 0.0 <= row["R"] / 100.0 <= 1.0 and 0.0 <= row["N"] / 100.0 <= 1.0
            assert not row["over_budget"]
    for budget in (1024, 2048):
        recs = qa.RecordStore(out / "retrievals").load("synthetic", "scholarpath", budget)
        assert len(recs) == 3 and all(r.retrieval.context_tokens <= budget for r in recs.values())
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["systems"]["ScholarPath"]) == {"1024", "2048"}
    assert (out / "fig5_Recall_B.csv").exists()
    # --resume reuses every cached retrieval: no cache file is rewritten.
    caches = sorted((out / "retrievals").glob("*/*/*.jsonl"))
    stamps = {p: p.stat().st_mtime_ns for p in caches}
    assert script.main(args + ["--resume"]) == 0
    assert {p: p.stat().st_mtime_ns for p in caches} == stamps


@pytest.fixture(scope="module")
def qa_runs(world) -> Dict[str, Path]:
    """run_agents.py (simulated agent) + run_qa_eval.py with its answers."""
    agents_out = world["results"] / "agents"
    assert load_script("run_agents").main(
        _common(world, "agents") + ["--agents", "simulated", "--limit", "2",
                                    "--benchmarks", "synthetic,taxobench,heldout"]) == 0
    qa_args = _common(world, "rqa") + ["--benchmarks", "synthetic,taxobench,heldout", "--budgets", "2048",
                                       "--sweep-budgets", "1024,2048", "--limit", "2", "--no-plots",
                                       "--answers-dir", str(agents_out)] + FAST_GRAPH + FAST_RETRIEVAL
    assert load_script("run_qa_eval").main(qa_args) == 0
    return {"agents": agents_out, "qa": world["results"] / "rqa", "qa_args": qa_args}  # type: ignore[dict-item]


def test_run_agents_outputs(qa_runs):
    store = qa.RecordStore(qa_runs["agents"] / "answers")
    for bench in ("synthetic", "taxobench", "heldout"):
        recs = store.load(bench, "simulated", None)
        assert len(recs) == 2 and all(r.ok and r.efficiency["calls"] >= 3 for r in recs.values())
    org = store.load("taxobench", "simulated", None)
    assert all(r.answer.metadata.get("hierarchy") for r in org.values())


def test_run_qa_eval_tables_and_resume(qa_runs):
    out = qa_runs["qa"]
    with open(out / "table3.csv", newline="") as f:
        rows3 = {r["System"]: r for r in csv.DictReader(f)}
    assert {"Keyword search", "Dense retrieval", "LightRAG", "ScholarPath", "SimulatedAgent (offline)"} <= set(rows3)
    assert rows3["ScholarPath"]["section"].startswith("Budgeted methods (B = 2,048)")
    assert rows3["SimulatedAgent (offline)"]["section"] == "Unbudgeted methods"
    assert {"TaxoBench ARI", "TaxoBench Sem", "Synthetic Cite"} <= set(rows3["ScholarPath"])
    with open(out / "table4.csv", newline="") as f:
        rows4 = {r["System"]: r for r in csv.DictReader(f)}
    assert {"Answer LLM, no retrieval", "ScholarPath", "SimulatedAgent (offline)"} <= set(rows4)
    for r in rows4.values():
        assert 0.0 <= float(r["Hit@10"]) <= 100.0
    fig6 = (out / "fig6_Disc.csv").read_text()
    assert "ScholarPath" in fig6 and "Answer LLM" not in fig6
    eff = [json.loads(line) for line in (out / "efficiency.jsonl").read_text().splitlines()]
    assert eff and all({"calls", "tokens", "cost_usd", "latency_s", "composition_calls"} <= set(r) for r in eff)
    judge_usage = json.loads((out / "judge_usage.json").read_text())
    assert judge_usage["hypothesis"]  # judges have their own meters
    # Answers of the ScholarPath runs stay within B.
    store = qa.RecordStore(out / "answers")
    for budget in (1024, 2048):
        for rec in store.load("heldout", "scholarpath", budget).values():
            assert rec.answer.context_tokens <= budget
    # Citation metrics judge each cited paper against the text the system saw (abstract and/or section
    # chunks), cached with the answer (Answer.metadata["passages"]), not only title + abstract.
    n_cited = 0
    for system in ("scholarpath", "dense", "keyword"):
        for rec in store.load("synthetic", system, 2048).values():
            passages = rec.answer.metadata["passages"]
            cited = {c for it in rec.answer.items for c in it.citations}
            assert cited <= set(passages) and all(passages[c]["text"] for c in cited), system
            n_cited += len(cited)
    assert n_cited > 0
    # --resume: cached answers are reused (no cache file rewritten).
    caches = sorted((out / "answers").glob("*/*/*.jsonl"))
    stamps = {p: p.stat().st_mtime_ns for p in caches}
    assert load_script("run_qa_eval").main(qa_runs["qa_args"] + ["--resume"]) == 0
    assert {p: p.stat().st_mtime_ns for p in caches} == stamps


def test_run_efficiency_script(world, qa_runs):
    out = world["results"] / "eff"
    assert load_script("run_efficiency").main(
        _common(world, "eff") + ["--inputs", f"{qa_runs['qa']},{qa_runs['agents']}", "--budget", "2048",
                                 "--operator-stats", "--exhaustive-check", "--benchmarks", "synthetic,heldout",
                                 "--limit", "2"] + FAST_RETRIEVAL) == 0
    with open(out / "table5.csv", newline="") as f:
        rows = {r["System"]: r for r in csv.DictReader(f)}
    assert {"Keyword search", "Dense retrieval", "LightRAG", "ScholarPath", "SimulatedAgent (offline)"} == set(rows)
    assert float(rows["Keyword search"]["Calls"]) == pytest.approx(1.0)
    details = json.loads((out / "efficiency_summary.json").read_text())
    sp = details["table5"]["ScholarPath"]
    assert sp["Composition calls"] == pytest.approx(1.0) and sp["Calls"] >= sp["Composition calls"]
    ops = json.loads((out / "operators.json").read_text())
    pooled = ops["summary"]["All queries"]
    assert 0.0 <= pooled["postings_pruned_pct"] <= 100.0 and 0.0 <= pooled["candidate_nodes_pruned_pct"] <= 100.0
    assert pooled["n_hypothesis"] == 2.0  # only the held-out queries have a candidate pool
    hyp_rows = ops["per_query"]["heldout"]
    expected = 100.0 * (1 - sum(r["scored_candidates"] for r in hyp_rows) / sum(r["candidate_pool"] for r in hyp_rows))
    assert pooled["candidate_nodes_pruned_pct"] == pytest.approx(expected)
    assert all(r["scored_candidates_exhaustive"] == r["candidate_pool"] for r in hyp_rows)  # exhaustive scores all
    assert pooled["same_selection_rate"] == 1.0  # Proposition 7: same units as the exhaustive scan


def _ensure_link_predictor(world: Dict[str, Path]) -> None:
    if not (world["index"] / "arxiv_subfields" / qa.LINK_PREDICTOR_DIR / "link_predictor.json").exists():
        assert _train_link_predictor(world) == 0


def test_run_retrieval_ablation_script(world):
    _ensure_link_predictor(world)
    out = world["results"] / "ablation"
    script = load_script("run_retrieval_ablation")
    assert script.main(_common(world, "ablation") + ["--budget", "2048", "--limit", "1"] + FAST_RETRIEVAL) == 0
    with open(out / "table7.csv", newline="") as f:
        rows = {r["Variant"]: r for r in csv.DictReader(f)}
    assert list(rows) == [qa.ABLATIONS[k][0] for k in qa.TABLE7_VARIANTS]
    full, exhaustive = rows["ScholarPath"], rows["w/o bound-guided search"]
    assert float(full["Scored"]) > 0 and float(full["Context (K)"]) <= 2.048
    # Bound-guided search is exact (Proposition 7): same answer quality, never fewer scored candidates when
    # exhaustive, which scores every candidate in scope ("Scored" = stats["scored_candidates"]).
    assert full["Hit@10"] == exhaustive["Hit@10"] and float(exhaustive["Scored"]) >= float(full["Scored"])
    summary = json.loads((out / "summary.json").read_text())["variants"]
    assert float(exhaustive["Scored"]) == summary["w/o bound-guided search"]["candidate_pool"]
    # "w/o temporal link prediction" swaps the trained s_theta for the text-similarity prior: still scored.
    no_lp = summary["w/o temporal link prediction"]
    assert no_lp["link_prior"] == "untrained" and no_lp["link_evaluations"] > 0 and no_lp["Scored"] > 0
    # Without a trained predictor the full row would equal that ablation: refused unless explicitly allowed.
    with pytest.raises(SystemExit):
        script.main(_common(world, "ablation_untrained") + ["--budget", "2048", "--limit", "1",
                                                            "--link-predictor", "none"] + FAST_RETRIEVAL)
    # Organization benchmarks read their gold trees from --data-dir.
    assert script.main(_common(world, "ablation_tb") + ["--budget", "1024", "--limit", "1", "--benchmarks",
                                                        "taxobench", "--variants", "scholarpath"]) == 0


def test_link_prior_of_the_ablation_variants(world):
    """Table 7 'w/o temporal link prediction' = untrained text prior; use_link_prediction=False drops the prior."""
    from scholarpath.retrieval.link_prediction import EmbeddingLinkPredictor

    _ensure_link_predictor(world)
    rt = Runtime.from_config()
    rt.config.retrieval.max_hypothesis_candidates = 6
    env = qa.QAEnvironment.load(world["index"] / "arxiv_subfields", rt)
    assert env.has_trained_link_predictor
    assert env.retriever("scholarpath").link_predictor is env.link_predictor
    assert isinstance(env.retriever("scholarpath_no_link_prediction").link_predictor, EmbeddingLinkPredictor)
    assert env.fingerprint("scholarpath") != env.fingerprint("scholarpath_no_link_prediction")
    q = qa.DataLayout(world["data"]).load_queries("heldout", limit=1)[0]
    untrained = env.retrieve("scholarpath_no_link_prediction", q, 2048).stats
    no_prior = env.retrieve("scholarpath_no_link_prior", q, 2048).stats
    assert untrained["link_evaluations"] > 0 and untrained["scored_candidates"] > 0
    assert no_prior["link_evaluations"] == 0


def _plant_emerging_concepts(data: Path, out: Path) -> Dict[str, str]:
    """Reference taxonomies over the complete synthetic corpus with planted new concepts (<= 4 papers).

    The synthetic held-out papers (published in 2024) realize unseen
    (sub-problem, method) pairs; one becomes a new problem adapting an
    existing method, the other a new method for an existing problem. Domains:
    the problem domain of a paper's sub-problem ("dom-emerging" for held-out papers).
    """
    full = Corpus.load(data / "arxiv_subfields_full" / "papers.jsonl.gz")
    P = Taxonomy.load(data / "gold" / "problem_taxonomy.json")
    M = Taxonomy.load(data / "gold" / "method_taxonomy.json")
    leaf_p = {P.nodes[n].name: n for n in P.leaves()}
    leaf_m = {M.nodes[n].name: n for n in M.leaves()}
    domains: Dict[str, str] = {}
    held: Dict[int, List[str]] = {}
    for p in full:
        if p.paper_id.startswith("held"):
            domains[p.paper_id] = "dom-emerging"
            held.setdefault(int(p.paper_id[4:6]), []).append(p.paper_id)
            P.assign(p.paper_id, leaf_p[p.metadata["subproblem"]])
            M.assign(p.paper_id, leaf_m[p.metadata["method"]])
        else:
            domains[p.paper_id] = "dom-" + P.nodes[P.parent(leaf_p[p.metadata["subproblem"]])].name
    s0 = full[held[0][0]].metadata["subproblem"]
    m1 = full[held[1][0]].metadata["method"]
    P.add_node(f"{s0} under Sensor Drift", "Forecasts that stay calibrated while sensors drift.",
               parent=leaf_p[s0], papers=held[0])
    M.add_node(f"Drift-Aware {m1}", "A variant that adapts its parameters online.", parent=leaf_m[m1],
               papers=held[1])
    out.mkdir(parents=True, exist_ok=True)
    P.save(out / "P.json")
    M.save(out / "M.json")
    (out / "domains.json").write_text(json.dumps(domains))
    return {"removed": ",".join(held[0] + held[1])}


def test_build_heldout_benchmark_script(world, tmp_path):
    import shutil

    data = tmp_path / "data"
    shutil.copytree(world["data"], data)
    planted = _plant_emerging_concepts(data, tmp_path / "planted")
    script = load_script("build_heldout_benchmark")
    common = ["--data-dir", str(data), "--output-dir", str(tmp_path / "out")]
    assert script.main(common + ["build", "--problem-taxonomy", str(tmp_path / "planted" / "P.json"),
                                 "--method-taxonomy", str(tmp_path / "planted" / "M.json"),
                                 "--domain-map", str(tmp_path / "planted" / "domains.json"),
                                 "--n-new-problem", "1", "--n-new-method", "1"]) == 0
    hd = data / "heldout"
    queries = load_queries(hd / "queries.jsonl")
    assert {q.metadata["setting"] for q in queries} == {"new_problem_existing_method", "new_method_existing_problem"}
    assert all(q.kind == "hypothesis_discovery" and q.metadata["heldout_counterpart"] for q in queries)
    reduced = Corpus.load(hd / "papers.jsonl.gz", chunks_path=hd / "chunks.jsonl.gz")
    removed = set(json.loads((hd / "removed_papers.json").read_text()))
    assert removed >= set(planted["removed"].split(",")) and not removed & set(reduced.ids)
    assert len(reduced) == len(json.loads((hd / "reduced_corpus_ids.json").read_text())) and reduced.num_chunks > 0
    # Two annotators accept the first query and reject the second.
    with open(hd / "annotation_sheet.csv", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    for i, row in enumerate(rows):
        for col in fields:
            if col.startswith(("a1_", "a2_")):
                row[col] = "1" if i == 0 else "0"
    with open(hd / "annotation_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    assert script.main(common + ["apply"]) == 0
    kept = load_queries(hd / "queries.jsonl")
    assert [q.query_id for q in kept] == [rows[0]["query_id"]]
    assert json.loads((hd / "annotation_report.json").read_text())["kept"] == 1
    # The reduced corpus is what the held-out benchmark is indexed and searched over.
    assert qa.DataLayout(data).load_corpus("heldout").ids == reduced.ids

    # App. C.1: the held-out index must not be built from taxonomies that saw the removed papers.
    build = load_script("build_index")
    args = ["--data-dir", str(data), "--corpus", "heldout", "--index-dir", str(tmp_path / "index"),
            "--problem-taxonomy", str(tmp_path / "planted" / "P.json"),
            "--method-taxonomy", str(tmp_path / "planted" / "M.json"), "--output-dir", str(tmp_path / "bi")]
    with pytest.raises(SystemExit):
        build.main(args)
    assert build.main(args + ["--allow-reference-taxonomy"]) == 0  # smoke-test override: restricted + pruned
    index = SemanticIndex.load(tmp_path / "index" / "heldout")
    names = [n.name for tax in (index.problem, index.method) for n in tax.nodes.values()]
    assert not [n for n in names if "Drift" in n]  # the planted held-out concepts are gone
    assert not removed & (index.problem.all_papers() | index.method.all_papers() | set(index.papers))
    restriction = json.loads((tmp_path / "index" / "heldout" / "build_summary.json").read_text())
    assert restriction["prebuilt_restriction"]["problem"]["concepts_removed"] >= 1


def test_restrict_taxonomy_drops_concepts_without_papers():
    tax = Taxonomy(facet="problem", root_name="Root")
    a = tax.add_node("A", papers=["p1", "p2"])
    b = tax.add_node("B", papers=["held1"])
    b1 = tax.add_node("B1", parent=b, papers=["held1"])
    c = tax.add_node("C")  # empty already: also dropped, its name could stem from removed papers
    out = qa.restrict_taxonomy(tax, ["p1", "p2", "p3"])
    assert set(out.nodes) == {out.root_id, a} and out.all_papers() == {"p1", "p2"}
    assert set(out.metadata["restriction"]["removed_concepts"]) == {b, b1, c}
    assert set(tax.nodes) == {tax.root_id, a, b, b1, c}  # the input is not modified
    assert set(qa.restrict_taxonomy(tax, ["p1"], drop_empty=False).nodes) == set(tax.nodes)


def test_heldout_corpus_never_falls_back_to_the_complete_corpus(world, tmp_path):
    """Paper layout (arxiv_subfields = complete corpus): no silent fallback that would leak held-out papers."""
    import shutil

    paper = tmp_path / "paper"
    shutil.copytree(world["data"] / "arxiv_subfields_full", paper / "arxiv_subfields")
    (paper / "heldout").mkdir()
    shutil.copy(world["data"] / "heldout" / "queries.jsonl", paper / "heldout" / "queries.jsonl")
    layout = qa.DataLayout(paper)
    assert not layout.is_synthetic_layout() and layout.complete_corpus_name() == "arxiv_subfields"
    with pytest.raises(FileNotFoundError):
        layout.load_corpus("heldout")
    store = qa.IndexStore(world["index"])  # has an arxiv_subfields index but no held-out index
    with pytest.raises(FileNotFoundError):
        store.resolve("heldout", layout)
    full = layout.load_corpus("arxiv_subfields", with_chunks=False)
    removed = [p for p in full.ids if p.startswith("held")]
    (paper / "heldout" / "removed_papers.json").write_text(json.dumps(removed))
    reduced = layout.load_corpus("heldout", with_chunks=False)
    assert removed and not set(removed) & set(reduced.ids) and len(reduced) == len(full) - len(removed)
    # The synthetic layout (whose arxiv_subfields already excludes the held-out papers) keeps its fallback.
    synthetic = qa.DataLayout(world["data"])
    assert synthetic.is_synthetic_layout() and store.resolve("heldout", synthetic) == "arxiv_subfields"


def test_ask_script(world, capsys):
    script = load_script("ask")
    base = ["--index-dir", str(world["index"]), "--corpus", "arxiv_subfields", "--budget", "1500"]
    assert script.main(base + ["How has bayesian optimization been used for cardinality estimation?"]) == 0
    text = capsys.readouterr().out
    assert "Answer:" in text and "Support paths" in text and "[established]" in text and "Usage:" in text
    assert script.main(base + ["--kind", "hypothesis", "--json", *FAST_RETRIEVAL,
                               "Which methods could be applied to flood forecasting?"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "hypothesis_discovery" and out["context_tokens"] <= 1500
    assert any(it["status"] == "hypothesis" for it in out["answer"]["items"])


def test_llm_free_step1_gives_one_call_per_query(world):
    """Table 5 reports one LLM call per ScholarPath query: Step 1 without its LLM call (``--no-query-llm``)."""
    rt = Runtime.from_config()
    options = qa.SystemOptions(retrieval_overrides={"llm_query_understanding": False})
    env = qa.QAEnvironment.load(world["index"] / "arxiv_subfields", rt, options=options)
    queries = qa.DataLayout(world["data"]).load_queries("synthetic", limit=2)
    recs = qa.run_queries(env, "scholarpath", queries, 2048, benchmark="synthetic", fail_fast=True)
    for rec in recs.values():
        assert rec.efficiency["calls"] == rec.efficiency["composition_calls"] == 1
        assert rec.answer.llm_calls == 1 and rec.answer.context_tokens <= 2048


def test_resume_regenerates_records_of_another_configuration(world, tmp_path):
    """--resume must not reuse records produced under other options (e.g. before --no-query-llm)."""
    store = qa.RecordStore(tmp_path / "runs")
    queries = qa.DataLayout(world["data"]).load_queries("synthetic", limit=2)
    calls = []
    for overrides in ({}, {}, {"llm_query_understanding": False}):
        rt = Runtime.from_config()
        env = qa.QAEnvironment.load(world["index"] / "arxiv_subfields", rt, link_predictor="none",
                                    options=qa.SystemOptions(retrieval_overrides=overrides))
        before = rt.meter.snapshot()
        recs = qa.run_queries(env, "scholarpath", queries, 2048, benchmark="synthetic", store=store, resume=True,
                              fail_fast=True)
        calls.append((sum(r.efficiency["calls"] for r in recs.values()), sum(
            u.calls for u in rt.meter.diff(before).values())))
        assert all(r.fingerprint == env.fingerprint("scholarpath") for r in recs.values())
    assert calls[0] == (4, 4)  # two calls per query (Step 1 + composition)
    assert calls[1] == (4, 0)  # same configuration: cached records reused, no new LLM call
    assert calls[2] == (2, 2)  # LLM-free Step 1: regenerated, one call per query (Table 5)


def test_table5_latency_is_uniform_with_a_warm_llm_cache(world, tmp_path):
    """Cached LLM calls count with their recorded latency for every system (Table 5 latency)."""
    import time as _time

    from scholarpath.llm.mock import MockLLM

    class SlowMock(MockLLM):
        def _generate(self, request):
            _time.sleep(0.05)
            return super()._generate(request)

    rt = Runtime.from_config()
    rt.llm = SlowMock(meter=rt.meter, cache_dir=str(tmp_path / "llm_cache"))
    env = qa.QAEnvironment.load(world["index"] / "arxiv_subfields", rt, link_predictor="none")
    q = qa.DataLayout(world["data"]).load_queries("synthetic", limit=1)[0]
    for system in ("lightrag", "scholarpath"):
        env.prepare(system)
        env.run_one(system, q, 2048, mode="retrieval", fail_fast=True)  # Table 2 run warms the cache
        rec = env.run_one(system, q, 2048, fail_fast=True)
        e = rec.efficiency
        assert e["calls"] == 2 and e["cached_calls"] == 1, system
        assert e["latency_s"] >= 0.1 - 1e-3 and e["cached_latency_s"] >= 0.05 - 1e-3, system
        assert e["latency_s"] == pytest.approx(e["wall_clock_s"] - e["excluded_latency_s"] + e["cached_latency_s"])
        assert rec.answer.latency_s == e["latency_s"]
    m = qa.efficiency_from_dict({"query_id": "q", "latency_s": 1.0, "components": {
        "composition": {"cached_latency_s": 0.5}, "query_understanding": {"cached_latency_s": 0.25}}})
    ans = Answer("q", "", metadata={"excluded_latency_s": 0.4})
    assert qa.efficiency_entry(m, ans)["latency_s"] == pytest.approx(1.0 - 0.4 + 0.75)
    assert qa.efficiency_entry(m, ans, metered=False)["latency_s"] == pytest.approx(1.0)  # agents: as reported


def test_run_agent_prices_with_the_backbone(tmp_path):
    from scholarpath.config import LLMConfig

    class FakeAgent:
        def __init__(self, cost):
            self.cost = cost

        def answer(self, query):
            return Answer(query.query_id, "x", [AnswerItem("claim", citations=["p1"])], llm_calls=3,
                          input_tokens=1_000_000, output_tokens=1_000_000, cost_usd=self.cost, latency_s=2.0)

    q = [Query("q1", "question")]
    cfg = LLMConfig(price_input_per_m=1.0, price_output_per_m=2.0)
    reported = qa.run_agent(FakeAgent(0.5), "codex", q, benchmark="b", llm_config=cfg)["q1"].efficiency
    assert reported["cost_usd"] == pytest.approx(0.5) and reported["latency_s"] == 2.0  # the adapter's cost
    priced = qa.run_agent(FakeAgent(0.0), "codex", q, benchmark="b", llm_config=cfg)["q1"].efficiency
    assert priced["cost_usd"] == pytest.approx(3.0)  # config.llm backbone prices
    store = qa.RecordStore(tmp_path)
    qa.run_agent(FakeAgent(0.5), "codex", q, benchmark="b", store=store, fingerprint="a")
    assert qa.run_agent(FakeAgent(0.7), "codex", q, benchmark="b", store=store, resume=True,
                        fingerprint="a")["q1"].efficiency["cost_usd"] == pytest.approx(0.5)  # reused
    assert qa.run_agent(FakeAgent(0.7), "codex", q, benchmark="b", store=store, resume=True,
                        fingerprint="b")["q1"].efficiency["cost_usd"] == pytest.approx(0.7)  # regenerated


def test_row_order_and_table3_macro():
    assert qa.order_systems(["codex", "agenticscholar", "openscholar", "paperqa2", "simulated"],
                            qa.TABLE34_AGENT_ORDER) == ["paperqa2", "openscholar", "agenticscholar", "codex",
                                                        "simulated"]
    assert qa.order_systems(["codex", "paperqa2", "openscholar"], qa.AGENT_SYSTEMS) == ["openscholar", "paperqa2",
                                                                                      "codex"]
    # Table 3's ScholarPath row: macro-average over the eight metrics = 70.7 (Sec. 5.3.2).
    cells = {"researchqa": {"Rub": 74.2, "Cite": 68.3}, "taxobench": {"taxobench.ARI": 55.8, "taxobench.Sem": 75.3},
             "chime": {"chime.Cat": 76.1, "chime.Assn": 69.8}}
    evals = {"researchqa": qa.BenchmarkEval("researchqa", "sp", 8192, cells["researchqa"]),
             "taxobench": qa.BenchmarkEval("taxobench", "sp", 8192, cells["taxobench"]),
             "chime": qa.BenchmarkEval("chime", "sp", 8192, cells["chime"]),
             "scholarqa_cs": qa.BenchmarkEval("scholarqa_cs", "sp", 8192, {"Corr": 75.6, "Cite": 70.4, "n": 100.0}),
             "scholarqa_multi": qa.BenchmarkEval("scholarqa_multi", "sp", 8192,
                                                 {"Corr": 75.6, "Cite": 70.4, "n": 108.0})}
    row = qa.qa_table_row(evals)
    assert row["macro.Avg"] == pytest.approx(70.7, abs=0.05)


def test_ask_format_lists_support_paths(world):
    rt = Runtime.from_config()
    env = qa.QAEnvironment.load(world["index"] / "arxiv_subfields", rt, link_predictor="none")
    res = qa.ask(env, "How have graph neural networks been used for drug-target interaction?", budget=800)
    assert res.retrieval.context_tokens <= 800 and count_tokens(res.retrieval.context) <= 800
    paths = qa.support_paths(env.index, res.retrieval)
    assert len(paths) == len(res.retrieval.units) and all("[established]" in p for p in paths)
    assert res.answer.llm_calls == 2  # Step 1 query understanding + one composition call
    text = qa.format_ask(res, env.index)
    assert "Support paths" in text and f"{res.answer.llm_calls} LLM call(s)" in text


@pytest.fixture(scope="module")
def paper_layout(world, tmp_path_factory) -> Dict[str, Path]:
    """The synthetic data in the paper's benchmark layout: ResearchQA, ScholarQA-CS / -Multi, TaxoBench, CHIME."""
    import shutil

    from scholarpath.data.corpus import save_queries
    from scholarpath.types import RubricItem

    root = tmp_path_factory.mktemp("paper_layout")
    data = root / "data"
    shutil.copytree(world["data"], data)
    ev = load_queries(data / "evidence" / "queries.jsonl")
    rubric = [RubricItem("Explains which method is applied to the problem", 1.0),
              RubricItem("Reports experimental findings", 1.0, kind="citation")]
    rqa = [Query(q.query_id.replace("syn", "rqa"), q.text, benchmark="researchqa", gold_papers=q.gold_papers,
                 unaligned_gold=["missing-paper"], rubric=list(rubric)) for q in ev[:2]]
    cs = [Query(q.query_id.replace("syn", "cs"), q.text, benchmark="scholarqa_cs", gold_papers=q.gold_papers,
                rubric=list(rubric), metadata={"rubric_evidence": [[f"{q.metadata['method']} is used"], []]})
          for q in ev[2:4]]
    multi = [Query(q.query_id.replace("syn", "multi"), q.text, benchmark="scholarqa_multi",
                   gold_papers=q.gold_papers, reference_answer=f"{q.metadata['method']} has been applied.")
             for q in ev[4:6]]
    save_queries(data / "researchqa" / "queries.jsonl", rqa)
    save_queries(data / "scholarqabench" / "cs.jsonl", cs)
    save_queries(data / "scholarqabench" / "multi.jsonl", multi)
    # CHIME: the TaxoBench-style trees as review hierarchies over their own pooled corpus.
    (data / "chime" / "gold").mkdir(parents=True)
    shutil.copy(data / "taxobench" / "papers.jsonl.gz", data / "chime" / "papers.jsonl.gz")
    chime = []
    for q in load_queries(data / "taxobench" / "queries.jsonl")[:2]:
        shutil.copy(data / "taxobench" / q.metadata["gold_taxonomy"], data / "chime" / "gold" / f"{q.query_id}.json")
        chime.append(Query(q.query_id.replace("taxobench", "chime"), q.text, benchmark="chime",
                           gold_papers=q.gold_papers,
                           metadata={"task": "organization", "style": "review",
                                     "gold_taxonomy": f"gold/{q.query_id}.json"}))
    save_queries(data / "chime" / "queries.jsonl", chime)
    (data / "evidence" / "queries.jsonl").unlink()  # paper layout: no synthetic evidence benchmark
    assert load_script("build_index").main([
        "--data-dir", str(data), "--corpus", "chime", "--index-dir", str(world["index"]),
        "--problem-taxonomy", str(data / "gold" / "problem_taxonomy.json"),
        "--method-taxonomy", str(data / "gold" / "method_taxonomy.json"),
        "--output-dir", str(root / "results" / "build_index"),
    ]) == 0
    return {"data": data, "results": root / "results"}


def test_paper_layout_tables_2_and_3(world, paper_layout):
    common = ["--data-dir", str(paper_layout["data"]), "--index-dir", str(world["index"]), "--limit", "2",
              "--no-plots"] + FAST_GRAPH + FAST_RETRIEVAL
    out2 = paper_layout["results"] / "rre"
    assert load_script("run_retrieval_eval").main(
        common + ["--output-dir", str(out2), "--systems", "bm25,scholarpath", "--budgets", "2048"]) == 0
    header2 = (out2 / "table2.csv").read_text().splitlines()[0].split(",")
    assert header2 == ["section", "Method", "ResearchQA R", "ResearchQA N", "ScholarQA R", "ScholarQA N",
                       "TaxoBench R", "TaxoBench N", "CHIME R", "CHIME N"]
    summary = json.loads((out2 / "summary.json").read_text())
    by_group = summary["systems"]["ScholarPath"]["2048"]["by_group"]
    assert by_group["scholarqabench"]["n"] == 4.0  # CS + Multi pooled into one column
    assert by_group["researchqa"]["R"] <= 100.0 * 8 / 9  # an unaligned gold paper stays in the denominator

    out3 = paper_layout["results"] / "rqa"
    assert load_script("run_qa_eval").main(
        common + ["--output-dir", str(out3), "--systems", "keyword,scholarpath", "--budgets", "2048",
                  "--sweep-budgets", "none", "--benchmarks", "evidence"]) == 0
    with open(out3 / "table3.csv", newline="") as f:
        rows = {r["System"]: r for r in csv.DictReader(f)}
    expected = ["ResearchQA Rub", "ResearchQA Cite", "ScholarQA Corr", "ScholarQA Cite", "TaxoBench ARI",
                "TaxoBench Sem", "CHIME Cat", "CHIME Assn", "Macro Avg."]
    assert list(rows["ScholarPath"])[2:] == expected
    sp = rows["ScholarPath"]
    assert float(sp["Macro Avg."]) == pytest.approx(sum(float(sp[c]) for c in expected[:-1]) / 8, abs=0.06)
    for row in rows.values():
        for col in expected:
            assert row[col] != "" and 0.0 <= float(row[col]) <= 100.0, (col, row[col])
    cs = json.loads((out3 / "eval" / "scholarqa_cs" / "scholarpath_B2048.json").read_text())["summary"]
    multi = json.loads((out3 / "eval" / "scholarqa_multi" / "scholarpath_B2048.json").read_text())["summary"]
    combined = float(rows["ScholarPath"]["ScholarQA Corr"])
    assert min(cs["Corr"], multi["Corr"]) - 1e-6 <= combined <= max(cs["Corr"], multi["Corr"]) + 1e-6
