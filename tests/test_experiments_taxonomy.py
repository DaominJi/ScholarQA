"""Tests for the taxonomy-construction experiments (Figs. 2-4, Tables 6, 8, 9, dynamic update).

Everything runs offline on the synthetic processed-data layout written by
``scripts/make_synthetic_data.py`` with the mock LLM and the hashing
embedder, a small router and small corpora, so the whole module takes a few
tens of seconds.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

from scholarpath.config import RouterConfig, ScholarPathConfig
from scholarpath.data.corpus import Corpus
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.experiments import common as C
from scholarpath.experiments import taxonomy as T
from scholarpath.llm.base import LLMRequest, chat
from scholarpath.runtime import Runtime
from scholarpath.taxonomy.tree import CERTIFIED, Taxonomy

REPO = Path(__file__).resolve().parents[1]
#: ScholarPath router and embedding settings of the quick runs.
QUICK_BUILDER = ["--set", "builder.router.hidden_dim=32", "--set", "builder.router.epochs=20",
                 "--set", "embedding.dim=128"]
#: ... plus small baseline routers (FULL trains none, so its cache key ignores these).
QUICK = QUICK_BUILDER + ["--set", "baseline.router.hidden_dim=32", "--min-router-steps", "100"]


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def quick_config() -> ScholarPathConfig:
    cfg = ScholarPathConfig()
    cfg.embedding.dim = 128
    cfg.builder.router = RouterConfig(hidden_dim=32, epochs=20)
    cfg.baseline.router = RouterConfig(hidden_dim=32)
    return cfg


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def in_unit_interval(value: Any) -> bool:
    x = float(value)
    return 0.0 <= x <= 1.0 + 1e-9


@pytest.fixture(scope="module")
def layout(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("syn")
    assert load_script("make_synthetic_data").main(["--out", str(out), "--n-papers", "160"]) == 0
    return out


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("cache")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_method_specs_roundtrip_and_labels():
    for name in ("scholarpath", "scholarpath/no_discovery", "full", "coverage@0.2", "one_off/kmeans@0.2"):
        assert T.MethodSpec.parse(name).key == name
    assert T.MethodSpec.parse("full").label() == "FULL (η = 1.0)"
    assert T.MethodSpec.parse("kmeans@0.05").family == "node_wise"
    assert T.MethodSpec.parse("one_off/random@0.2").family == "one_off"
    assert T.MethodSpec.scholarpath("no_snapshot_certification").label() == "w/o snapshot certification (η_v = 0.08)"
    with pytest.raises(ValueError):
        T.MethodSpec.scholarpath("no_such_ablation")
    with pytest.raises(ValueError):
        T.MethodSpec.selection("random", 0.0)
    keys = [m.key for m in T.construction_methods(("random",), (0.1, 0.2), (0.2,))]
    assert keys == ["scholarpath", "random@0.1", "random@0.2", "one_off/random@0.2"]
    assert [m.key for m in T.ablation_methods()][1:] == [f"scholarpath/{a}" for a in T.ABLATIONS]


def test_runner_configs_apply_ablations_and_seeds():
    rt = Runtime.from_config(quick_config())
    runner = T.ConstructionRunner.from_runtime(rt)
    cfg = runner.builder_config("method", 3, "no_bayesian_acquisition")
    assert cfg.facet == "method" and cfg.seed == 3 and cfg.router.seed == 3
    assert cfg.use_bayesian_acquisition is False and cfg.use_discovery is True
    for name, ablation in T.ABLATIONS.items():
        changed = runner.builder_config("problem", 0, name)
        assert all(getattr(changed, k) == v for k, v in ablation.overrides)
    b = runner.baseline_config("topic", 1, T.MethodSpec.parse("one_off/coverage@0.3"))
    assert (b.selector, b.eta, b.mode, b.facet, b.seed) == ("coverage", 0.3, "one_off", "topic", 1)
    assert runner.resolve_seed(T.FULL_METHOD, 7) == runner.reference_seed


def test_default_seeds_follow_the_config_section_of_each_method():
    cfg = quick_config()
    cfg.builder.seed, cfg.baseline.seed = 3, 5
    runner = T.ConstructionRunner.from_runtime(Runtime.from_config(cfg))
    selection = T.MethodSpec.parse("random@0.2")
    # baseline.seed reaches every selection baseline, not only FULL; builder.seed seeds ScholarPath.
    assert runner.resolve_seed(selection, None) == 5 and runner.resolve_seed(T.FULL_METHOD, None) == 5
    assert runner.resolve_seed(T.SCHOLARPATH_METHOD, None) == 3
    assert runner.resolve_seed(selection, 1) == 1 and runner.resolve_seed(T.FULL_METHOD, 1) == 5
    assert runner.baseline_config("problem", runner.resolve_seed(selection, None), selection).seed == 5
    # The global --seed is shorthand for the build seed(s); explicit --seeds / --build-seed win.
    ns = argparse.Namespace
    assert C.seeds_from_args(ns(seeds=None, seed=None)) is None
    assert C.seeds_from_args(ns(seeds=None, seed=7)) == [7]
    assert C.seeds_from_args(ns(seeds=[1, 2], seed=7)) == [1, 2]
    assert C.seed_from_args(ns(build_seed=None, seed=7)) == 7 and C.seed_from_args(ns(build_seed=2, seed=7)) == 2


def test_fresh_meter_isolates_and_merges():
    rt = Runtime.from_config()
    req = LLMRequest(task="answer.compose", messages=chat("s", "hello world"), component="acquisition")
    rt.llm.meter.record(10, 10, component="other")
    with C.fresh_meter(rt.llm) as meter:
        rt.llm.complete(req)
        assert set(meter.snapshot()) == {"acquisition"}
        inner = meter.total().cost_usd
    assert rt.llm.meter.component("acquisition").cost_usd == pytest.approx(inner)
    assert rt.llm.meter.component("other").calls == 1


def test_nested_subsets_are_nested_and_clipped():
    ids = [f"p{i:03d}" for i in range(50)]
    subsets = T.nested_subsets(ids, [10, 30, 30, 80], seed=1)
    assert list(subsets) == [10, 30, 50]
    assert set(subsets[10]) <= set(subsets[30]) <= set(subsets[50]) == set(ids)
    assert subsets == T.nested_subsets(reversed(ids), [10, 30, 80], seed=1)
    assert T.size_label(2000) == "2K" and T.size_label(148400) == "148.4K" and T.size_label(500) == "500"


def test_split_stream_month_and_size_cohorts():
    corpus = make_synthetic_benchmark(n_papers=80, seed=0).corpus
    split = T.split_stream(corpus, "2022-01-01")
    assert all(p.date < "2022-01-01" for p in split.base)
    streamed = split.streamed
    assert all(p.date >= "2022-01-01" for p in streamed)
    assert len(split.base) + len(streamed) == len(corpus)
    labels = [c.label for c in split.cohorts]
    assert labels == sorted(labels) and all(len(c.papers) for c in split.cohorts)
    assert all(p.date.startswith(c.label) for c in split.cohorts for p in c.papers)
    by_size = T.split_stream(corpus, "2022-01-01", cohort="size", batch_size=3, max_cohorts=2)
    assert [len(c.papers) for c in by_size.cohorts] == [3, 3]
    auto = T.split_stream(corpus, base_fraction=0.5)
    assert 0.3 * len(corpus) <= len(auto.base) <= 0.7 * len(corpus)


def test_truncate_and_restrict_gold():
    gold = make_synthetic_benchmark(n_papers=60, seed=0).problem_gold
    cut = T.truncate_taxonomy(gold, 1)
    assert cut.max_depth == 1 and len(cut.nodes) == 1 + len(gold.children(gold.root_id))
    assert cut.root.papers == gold.root.papers and gold.max_depth == 2
    keep = sorted(gold.root.papers)[:10]
    small = T.restrict_taxonomy(gold, keep)
    assert small.root.papers == set(keep)
    assert all(n.papers for nid, n in small.nodes.items())


def test_build_cache_memo_disk_and_reuse(tmp_path):
    rt = Runtime.from_config(quick_config())
    corpus = make_synthetic_benchmark(n_papers=60, seed=2).corpus
    ds = T.ConstructionDataset("tiny", "custom", corpus, facets=("problem",))
    cache = C.BuildCache(tmp_path)
    runner = T.ConstructionRunner.from_runtime(rt, cache, min_router_steps=50)
    method = T.MethodSpec.parse("random@0.5")
    first = runner.build(ds, "problem", method)
    assert runner.build(ds, "problem", method) is first
    assert cache.stats == {"built": 1, "memory_hits": 1, "disk_hits": 0, "reloaded": 0}
    key, _ = runner.build_key(ds, "problem", method)
    assert cache.contains(key)
    # A new process with --resume loads the stored build instead of rebuilding it.
    cache2 = C.BuildCache(tmp_path, reuse=True)
    again = T.ConstructionRunner.from_runtime(rt, cache2, min_router_steps=50).build(ds, "problem", method)
    assert cache2.stats["built"] == 0 and cache2.stats["disk_hits"] == 1
    assert again.cost_usd == pytest.approx(first.cost_usd)
    assert again.taxonomy.to_dict()["nodes"] == first.taxonomy.to_dict()["nodes"]
    assert len(again.episodes) == len(first.episodes) and set(again.routers) == set(first.routers)
    # Anything that changes the build changes the key: seed, papers, method, config.
    assert runner.build_key(ds, "problem", method, 1)[0] != key
    sub = T.ConstructionDataset("tiny", "custom", corpus.subset(corpus.ids[:40]), facets=("problem",))
    assert runner.build_key(sub, "problem", method)[0] != key
    other = T.ConstructionRunner.from_runtime(rt, cache, min_router_steps=60)
    assert other.build_key(ds, "problem", method)[0] != key


def test_full_cache_key_ignores_router_settings():
    """FULL trains no router: router settings must not fork its cache entry (one FULL per dataset / facet)."""
    ds = T.ConstructionDataset("tiny", "custom", make_synthetic_benchmark(n_papers=40, seed=2).corpus,
                               facets=("problem",))
    a = T.ConstructionRunner.from_runtime(Runtime.from_config(quick_config()), min_router_steps=100)
    cfg = quick_config()
    cfg.baseline.router = RouterConfig(hidden_dim=256, epochs=7, seed=3)
    b = T.ConstructionRunner.from_runtime(Runtime.from_config(cfg))
    assert a.build_key(ds, "problem", T.FULL_METHOD)[0] == b.build_key(ds, "problem", T.FULL_METHOD)[0]
    selection = T.MethodSpec.parse("coverage@0.2")
    assert a.build_key(ds, "problem", selection)[0] != b.build_key(ds, "problem", selection)[0]
    cfg2 = quick_config()
    cfg2.baseline.max_children = 5  # ... but every setting FULL does read still changes it
    c = T.ConstructionRunner.from_runtime(Runtime.from_config(cfg2), min_router_steps=100)
    assert c.build_key(ds, "problem", T.FULL_METHOD)[0] != a.build_key(ds, "problem", T.FULL_METHOD)[0]


def test_cache_key_fingerprints_paper_content_and_billing():
    """Same paper ids with different text (e.g. a re-prepared corpus) must not reuse a stored build."""
    runner = T.ConstructionRunner.from_runtime(Runtime.from_config(quick_config()))
    corpus = make_synthetic_benchmark(n_papers=40, seed=0).corpus
    edited = Corpus([dataclasses.replace(p, abstract=p.abstract + " Revised.") if i == 0 else p
                     for i, p in enumerate(corpus)], name=corpus.name)
    assert set(edited.ids) == set(corpus.ids)
    same = Corpus(list(reversed(list(corpus))), name="other name")
    fp = C.corpus_fingerprint(corpus)
    assert C.corpus_fingerprint(same) == fp and C.corpus_fingerprint(edited) != fp
    ds = T.ConstructionDataset("tiny", "custom", corpus, facets=("problem",))
    ds2 = T.ConstructionDataset("tiny", "custom", edited, facets=("problem",))
    for method in (T.FULL_METHOD, T.SCHOLARPATH_METHOD):
        assert runner.build_key(ds, "problem", method)[0] != runner.build_key(ds2, "problem", method)[0]
    for field_name, value in (("bill_cached_calls", False), ("json_retries", 0)):
        cfg = quick_config()
        setattr(cfg.llm, field_name, value)
        other = T.ConstructionRunner.from_runtime(Runtime.from_config(cfg))
        assert other.build_key(ds, "problem", T.FULL_METHOD)[0] != runner.build_key(ds, "problem", T.FULL_METHOD)[0]


def test_build_cache_evicts_and_reloads(tmp_path):
    rt = Runtime.from_config(quick_config())
    ds = T.ConstructionDataset("tiny", "custom", make_synthetic_benchmark(n_papers=50, seed=4).corpus,
                               facets=("problem",))
    cache = C.BuildCache(tmp_path, max_memory=1)
    runner = T.ConstructionRunner.from_runtime(rt, cache, min_router_steps=50)
    a = runner.build(ds, "problem", T.MethodSpec.parse("random@0.5"))
    runner.build(ds, "problem", T.FULL_METHOD)
    again = runner.build(ds, "problem", T.MethodSpec.parse("random@0.5"))  # evicted, reloaded from disk
    assert cache.stats == {"built": 2, "memory_hits": 0, "disk_hits": 0, "reloaded": 1}
    assert again is not a and again.cost_usd == pytest.approx(a.cost_usd)
    assert C.BuildCache(None).max_memory is None  # memory-only caches never evict


# ---------------------------------------------------------------------------
# Construction comparison (Figs. 2-3, Tables 8-9)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def comparison(layout, tmp_path_factory):
    rt = Runtime.from_config(quick_config())
    cache = C.BuildCache(tmp_path_factory.mktemp("cmp_cache"))
    runner = T.ConstructionRunner.from_runtime(rt, cache, min_router_steps=100)
    datasets = T.load_arxiv_datasets(layout, limit_papers=100, facets=("problem",))
    datasets.append(T.load_taxobench_dataset(layout, limit_papers=100))
    methods = [T.SCHOLARPATH_METHOD, T.MethodSpec.parse("random@0.2"), T.MethodSpec.parse("one_off/random@0.2")]
    rows = T.run_construction_comparison(datasets, methods, runner, seeds=[0, 1])
    return rt, cache, datasets, methods, rows


def test_comparison_builds_full_once_and_scores_in_range(comparison):
    _, cache, datasets, methods, rows = comparison
    assert [d.name for d in datasets] == ["synthetic", "taxobench"]
    full_keys = cache.built_keys("full")
    assert sorted((k.dataset, k.facet) for k in full_keys) == [("synthetic", "problem"), ("taxobench", "topic")]
    assert len(cache.built_keys("scholarpath")) == 4  # 2 datasets x 2 seeds
    assert len(rows) == 2 * (1 + len(methods) * 2)
    for r in rows:
        for k in T.METRICS:
            assert in_unit_interval(r[k]), (r["method"], k, r[k])
        assert r["cost_usd"] > 0 and r["relative_cost"] > 0
        assert r["full_cost_usd"] == pytest.approx(next(
            x["cost_usd"] for x in rows if x["method"] == "full" and x["dataset"] == r["dataset"]))
    full_rows = [r for r in rows if r["method"] == "full"]
    assert all(r["relative_cost"] == pytest.approx(1.0) for r in full_rows)
    arxiv_full = next(r for r in full_rows if r["benchmark"] == T.ARXIV)
    assert arxiv_full["reference"] == "full" and arxiv_full["node_f1"] == pytest.approx(1.0)
    tb_full = next(r for r in full_rows if r["benchmark"] == T.TAXOBENCH)
    assert tb_full["reference"] == "gold" and tb_full["facet"] == T.TOPIC_FACET
    sp = next(r for r in rows if r["method"] == "scholarpath")
    assert sp["cost_acquisition"] > 0 and sp["cost_full"] == 0 and sp["label"] == "ScholarPath"


def test_comparison_rerun_from_disk_is_identical(comparison, layout):
    rt, cache, datasets, methods, rows = comparison
    cache2 = C.BuildCache(cache.root, reuse=True)
    runner = T.ConstructionRunner.from_runtime(rt, cache2, min_router_steps=100)
    fresh = T.load_arxiv_datasets(layout, limit_papers=100, facets=("problem",))
    fresh.append(T.load_taxobench_dataset(layout, limit_papers=100))
    again = T.run_construction_comparison(fresh, methods, runner, seeds=[0, 1])
    assert cache2.stats["built"] == 0
    for a, b in zip(rows, again):
        assert (a["dataset"], a["method"], a["seed"]) == (b["dataset"], b["method"], b["seed"])
        for k in T.METRICS + ("cost_usd", "relative_cost", "calls"):
            assert a[k] == pytest.approx(b[k])


def test_summaries_figures_and_tables(comparison, tmp_path):
    _, _, _, _, rows = comparison
    summary = T.summarize_runs(rows)
    idx = {(r["benchmark"], r["facet"], r["method"]): r for r in summary}
    assert idx[(T.ARXIV, T.AVG_FACET, "scholarpath")]["n_runs"] == 2
    assert idx[(T.ARXIV, T.AVG_FACET, "full")]["relative_cost"] == pytest.approx(1.0)
    fig2 = T.figure2_rows(summary, ("random",), (0.2,))
    series = {(r["benchmark"], r["facet"], r["series"]) for r in fig2}
    assert (T.ARXIV, T.AVG_FACET, "ScholarPath") in series and (T.ARXIV, T.AVG_FACET, "Random") in series
    assert (T.TAXOBENCH, T.AVG_FACET, T.FULL_LABEL) in series  # FULL is a reference line vs. the gold only
    assert (T.ARXIV, T.AVG_FACET, T.FULL_LABEL) not in series
    assert all(0 <= r["node_f1"] <= 100 and 0 <= r["edge_f1"] <= 100 for r in fig2)
    fig3 = T.figure3_rows(summary, (0.2,), ("random",), bar_selector="random")
    bars = [(r["benchmark"], r["bar"]) for r in fig3]
    assert bars == [(b, x) for b in (T.ARXIV, T.TAXOBENCH) for x in ("Full", "η = 0.2", "ScholarPath")]
    assert all(r["relative_cost_pct"] == pytest.approx(100.0) for r in fig3 if r["bar"] == "Full")
    assert all(r["bar_selector"] == "random" for r in fig3 if r["eta"] == 0.2)

    rows8, sections8, spec8 = T.table8(summary)
    assert list(rows8) == ["Random (one-off)", "Random (node-wise)", "ScholarPath"]
    assert [s for s, _ in sections8] == ["One-off selection (η = 0.2)", "Node-wise selection (η = 0.2)", ""]
    assert rows8["ScholarPath"]["method.node"] is None  # only the problem facet was run
    md = T.render_table((rows8, sections8, spec8))
    assert "| ScholarPath |" in md and "*One-off selection (η = 0.2)*" in md and "**" in md
    rows9, _, _ = T.table9(summary)
    assert list(rows9)[0] == T.FULL_LABEL and rows9[T.FULL_LABEL]["cost"] == pytest.approx(1.0)
    path = T.write_table_files(tmp_path, "table9", T.table9(summary))
    assert path.exists() and read_csv(path.with_suffix(".csv"))[0]["Method"] == T.FULL_LABEL

    paths = T.write_construction_report(rows, tmp_path, selectors=("random",), etas=(0.2,), fig3_etas=(0.2,),
                                        plots=False)
    for name in ("runs.csv", "runs.jsonl", "summary.csv", "fig2.csv", "fig3.csv", "table8.md", "table9.md"):
        assert (tmp_path / name).exists(), name
    assert paths["table8"].name == "table8.md"


def test_figure3_bars_are_coverage_costs():
    """Fig. 3's eta bars are Coverage (20.4 / 24.6 at eta = 0.2 = Tables 8-9), not the selector mean."""
    table89 = {"arxiv": {"random": 0.201, "diversity": 0.203, "kmeans": 0.202, "coverage": 0.204},
               "taxobench": {"random": 0.241, "diversity": 0.245, "kmeans": 0.243, "coverage": 0.246}}
    summary = [{"benchmark": b, "facet": T.AVG_FACET, "method": f"{s}@0.2", "relative_cost": c}
               for b, costs in table89.items() for s, c in costs.items()]
    selectors = ("random", "diversity", "kmeans", "coverage")
    bars = {r["benchmark"]: r for r in T.figure3_rows(summary, (0.2,), selectors)}
    assert bars["arxiv"]["relative_cost_pct"] == pytest.approx(20.4)
    assert bars["taxobench"]["relative_cost_pct"] == pytest.approx(24.6)
    assert bars["arxiv"]["bar_selector"] == "coverage" and bars["arxiv"]["kmeans_pct"] == pytest.approx(20.2)
    mean = {r["benchmark"]: r for r in T.figure3_rows(summary, (0.2,), selectors, bar_selector="mean")}
    assert mean["arxiv"]["relative_cost_pct"] == pytest.approx(20.25)
    # A missing bar selector falls back to the mean of what was run, and says so.
    only_random = [r for r in summary if r["method"] == "random@0.2"]
    fallback = T.figure3_rows(only_random, (0.2,), ("random",))
    assert all(r["bar_selector"] == "mean" for r in fallback)
    # The Fig. 3 selector's bars are built even when it is not a Fig. 2 selector.
    keys = [m.key for m in T.construction_methods(("random",), (0.2,), (), fig3_selector="coverage",
                                                  fig3_etas=(0.5, 0.2))]
    assert keys == ["scholarpath", "random@0.2", "coverage@0.5", "coverage@0.2"]


def test_table9_bolds_full_and_the_best_budgeted_method():
    """Paper Table 9: FULL's F1 and ScholarPath's F1 and cost are bold; FULL's cost (1.000) is not."""
    paper = {"full": (87.6, 76.8, 1.0), "one_off/coverage@0.2": (79.2, 68.2, .234),
             "random@0.2": (75.1, 64.3, .241), "coverage@0.2": (82.4, 71.6, .246), "scholarpath": (84.8, 73.9, .114)}
    summary = [{"benchmark": T.TAXOBENCH, "facet": T.AVG_FACET, "method": m, "node_f1": n / 100,
                "edge_f1": e / 100, "relative_cost": c} for m, (n, e, c) in paper.items()]
    md = T.render_table(T.table9(summary))
    assert "| FULL (η = 1.0) | **87.6** | **76.8** | 1.000 |" in md
    assert "| ScholarPath | **84.8** | **73.9** | **0.114** |" in md
    assert "| Coverage (node-wise) | 82.4 | 71.6 | 0.246 |" in md


def test_render_table_bolds_overall_best_only():
    spec = T.table6_spec()
    rows = {"ScholarPath": {"node": 85.2, "edge": 74.1, "abstain": 4.7, "disagr": 7.2, "cost": 0.08},
            "w/o discovery (DISCOVER)": {"node": 80.5, "edge": 69.6, "abstain": 10.3, "disagr": 7.6, "cost": 0.072}}
    md = T.render_table((rows, [("", list(rows))], spec))
    assert "| ScholarPath | **85.2** | **74.1** | **4.7** | **7.2** | 0.080 |" in md
    assert "**0.072**" in md
    plain = T.render_table((rows, [], spec), bold_best=False)
    assert "**" not in plain.split("\n", 2)[2]  # only the title line is bold


# ---------------------------------------------------------------------------
# Scalability (Fig. 4) and dynamic update (Sec. 3.4)
# ---------------------------------------------------------------------------


def test_run_scalability_estimates_full_above_limit():
    rt = Runtime.from_config(quick_config())
    corpus = make_synthetic_benchmark(n_papers=90, seed=3).corpus
    runner = T.ConstructionRunner.from_runtime(rt, C.BuildCache(None), min_router_steps=50)
    rows = T.run_scalability(corpus, [45, 90], runner, etas=(0.5,), max_full_size=45, facets=("problem",))
    by = {(r["size"], r["method"]): r for r in rows}
    assert set(by) == {(45, "full"), (45, "scholarpath"), (45, "random@0.5"),
                       (90, "full_est"), (90, "scholarpath"), (90, "random@0.5")}
    measured = by[(45, "full")]
    assert measured["full_measured"] and measured["full_cost_usd"] == pytest.approx(measured["cost_usd"])
    est = by[(90, "full_est")]
    assert not est["full_measured"] and est["c_bar_size"] == 45
    assert est["cost_usd"] == pytest.approx(est["c_bar"] * by[(90, "scholarpath")]["internal_paper_count"])
    for n in (45, 90):
        sp = by[(n, "scholarpath")]
        assert sp["selected_ratio"] == pytest.approx(sp["cost_usd"] / sp["full_cost_usd"])
    summary = T.scalability_summary(rows)
    full90 = next(r for r in summary if r["size"] == 90 and r["method"] == "full")
    assert full90["full_estimated"] and full90["label"] == "Full (est.)"
    assert full90["selected_ratio_pct"] == pytest.approx(100.0)
    with pytest.raises(ValueError):
        T.run_scalability(corpus, [45, 90], runner, max_full_size=10, facets=("problem",))


def test_scalability_summary_ratio_of_totals_and_missing_costs():
    """Fig. 4(a) is Σ C_LLM / Σ C_Full over the facets (paper: $5.4 / $138.4 = 3.9%); NaN never counts as $0."""
    def row(facet, method, cost, full_cost, measured=True):
        return {"size": 148400, "facet": facet, "method": method, "label": method, "cost_usd": cost,
                "full_cost_usd": full_cost, "full_measured": measured,
                "selected_ratio": cost / full_cost if full_cost else None}

    rows = [row("problem", "full", 100.0, 100.0), row("method", "full", 38.4, 38.4),
            row("problem", "scholarpath", 3.0, 100.0), row("method", "scholarpath", 2.4, 38.4)]
    by = {r["method"]: r for r in T.scalability_summary(rows)}
    assert by["scholarpath"]["selected_ratio_pct"] == pytest.approx(100 * 5.4 / 138.4)
    assert by["scholarpath"]["mean_facet_ratio_pct"] == pytest.approx(50 * (3.0 / 100.0 + 2.4 / 38.4))
    assert by["scholarpath"]["total_cost_usd"] == pytest.approx(5.4)
    assert by["full"]["total_cost_usd"] == pytest.approx(138.4) and by["full"]["selected_ratio_pct"] == 100.0
    nan = float("nan")
    broken = [row(f, "full_est", nan, nan, measured=False) for f in ("problem", "method")]
    broken += [row(f, "scholarpath", 1.0, nan, measured=False) for f in ("problem", "method")]
    by = {r["method"]: r for r in T.scalability_summary(broken)}
    assert by["full"]["total_cost_usd"] is None and by["full"]["selected_ratio_pct"] is None
    assert by["scholarpath"]["total_cost_usd"] == pytest.approx(2.0)
    assert by["scholarpath"]["selected_ratio_pct"] is None


def test_run_dynamic_update_and_summary():
    rt = Runtime.from_config(quick_config())
    corpus = make_synthetic_benchmark(n_papers=150, seed=0).corpus
    split = T.split_stream(corpus, "2022-01-01", max_cohorts=3)
    base = T.ConstructionDataset("dyn", "dynamic", split.base, facets=("problem",))
    runner = T.ConstructionRunner.from_runtime(rt, C.BuildCache(None))
    base_before = copy.deepcopy(runner.build(base, "problem", T.SCHOLARPATH_METHOD).taxonomy.to_dict())
    run = T.run_dynamic_update(base, split.cohorts, runner, "problem")
    # The cached base build is not mutated; the updated taxonomy holds every streamed paper.
    assert run.base.taxonomy.to_dict() == base_before
    assert run.updated.taxonomy.root.papers == set(split.base.ids) | {p.paper_id for p in split.streamed}
    assert [r["cohort"] for r in run.cohorts] == [c.label for c in split.cohorts]
    for r in run.cohorts:
        assert r["zero_llm_cost"] == (r["calls"] == 0)
        assert r["cost_usd"] >= 0 and r["cumulative_cost_usd"] >= r["cost_usd"] - 1e-12
    for node in run.updated.taxonomy.nodes.values():
        if node.status == CERTIFIED:
            assert node.bound <= rt.config.builder.epsilon + 1e-12
    scratch_ds = T.ConstructionDataset("dyn+stream", "dynamic", Corpus(list(split.base) + split.streamed),
                                       facets=("problem",))
    scratch = runner.build(scratch_ds, "problem", T.SCHOLARPATH_METHOD)
    summary = T.dynamic_summary(run, scratch, T.make_scorer(rt.embedder))
    assert summary["papers_streamed"] == len(split.streamed) and summary["cohorts"] == len(split.cohorts)
    assert in_unit_interval(summary["vs_scratch_node_f1"]) and in_unit_interval(summary["vs_scratch_edge_f1"])
    assert summary["scratch_cost_usd"] == pytest.approx(scratch.cost_usd)


def test_replay_prior_reproduces_history():
    rt = Runtime.from_config(quick_config())
    corpus = make_synthetic_benchmark(n_papers=100, seed=1).corpus
    runner = T.ConstructionRunner.from_runtime(rt)
    ds = T.ConstructionDataset("prior", "custom", corpus, facets=("method",))
    result = runner.build(ds, "method", T.SCHOLARPATH_METHOD)
    prior = T.replay_prior(runner.builder_config("method", 0), result.episodes)
    expected = sum(len(e.region_slopes) for e in result.episodes if e.region_descriptors)
    assert prior.n_observations == expected


# ---------------------------------------------------------------------------
# Scripts (main() on the synthetic layout)
# ---------------------------------------------------------------------------


def test_script_run_taxonomy_experiments(layout, shared_cache, tmp_path):
    out = tmp_path / "exp"
    code = load_script("run_taxonomy_experiments").main([
        "--data-dir", str(layout), "--limit-papers", "100", "--facets", "problem", "--selectors", "random",
        "--etas", "0.2", "--fig3-etas", "0.2", "--no-plots", "--cache-dir", str(shared_cache),
        "--output-dir", str(out), *QUICK])
    assert code == 0
    for name in ("runs.csv", "runs.jsonl", "summary.csv", "fig2.csv", "fig3.csv", "table8.md", "table8.csv",
                 "table9.md", "table9.csv", "report.json", "config.yaml"):
        assert (out / name).exists(), name
    runs = read_csv(out / "runs.csv")
    for col in ("dataset", "facet", "method", "eta", "mode", "node_p", "node_r", "node_f1", "edge_p", "edge_r",
                "edge_f1", "cost_usd", "relative_cost", "tokens", "calls", "papers_read", "runtime_s"):
        assert col in runs[0], col
    # The Fig. 3 bars default to Coverage (built at --fig3-etas although --selectors is only random).
    assert {r["method"] for r in runs} == {"full", "scholarpath", "random@0.2", "one_off/random@0.2", "coverage@0.2"}
    assert all(math.isclose(float(r["relative_cost"]), 1.0) for r in runs if r["method"] == "full")
    assert all(in_unit_interval(r[k]) for r in runs for k in T.METRICS)
    report = json.loads((out / "report.json").read_text())
    assert report["cache"]["built"] == len(runs)  # every (dataset, facet, method) built exactly once
    assert "ScholarPath" in (out / "table9.md").read_text()
    summary = {(r["benchmark"], r["facet"], r["method"]): r for r in read_csv(out / "summary.csv")}
    for bar in read_csv(out / "fig3.csv"):
        if bar["bar"] == "η = 0.2":
            assert bar["bar_selector"] == "coverage"
            expected = 100 * float(summary[(bar["benchmark"], T.AVG_FACET, "coverage@0.2")]["relative_cost"])
            assert math.isclose(float(bar["relative_cost_pct"]), expected)


def test_script_ablation_reuses_cached_full(layout, shared_cache, tmp_path):
    out = tmp_path / "abl"
    # Different baseline-router flags than run_taxonomy_experiments (its documented quick command does the same).
    code = load_script("run_taxonomy_ablation").main([
        "--data-dir", str(layout), "--limit-papers", "100", "--facets", "problem",
        "--variants", "no_discovery,no_snapshot_certification", "--cache-dir", str(shared_cache), "--resume",
        "--output-dir", str(out), *QUICK_BUILDER])
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    # FULL and ScholarPath come from the cache written by run_taxonomy_experiments: FULL is never rebuilt.
    assert report["cache"]["disk_hits"] == 2 and report["cache"]["built"] == 2
    table = (out / "table6.md").read_text()
    assert "w/o discovery (DISCOVER)" in table and "η_v = 0.08" in table
    rows = read_csv(out / "table6.csv")
    assert [r["Variant"] for r in rows][0] == "ScholarPath"
    assert all(0.0 <= float(r["Node"]) <= 100.0 and float(r["Cost"]) > 0 for r in rows)


def test_script_build_taxonomy(layout, shared_cache, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the default output folder is relative: results/build_taxonomy/...
    script = load_script("build_taxonomy")
    gold_path = layout / "taxobench" / "gold_taxonomy.json"
    code = script.main([
        "--corpus", str(layout / "taxobench" / "papers.jsonl.gz"), "--facet", "topic", "--method", "kmeans",
        "--eta", "0.3", "--limit-papers", "80", "--reference", str(gold_path),
        "--cache-dir", str(shared_cache), *QUICK])
    assert code == 0
    out = tmp_path / "results" / "build_taxonomy" / "taxobench__kmeans@0.3"  # named after the method key
    for name in ("taxonomy.json", "summary.json", "usage.json", "episodes.jsonl"):
        assert (out / "topic" / name).exists(), name
    assert (out / "config.yaml").exists() and (out / "build.json").exists()
    build = json.loads((out / "build.json").read_text())
    assert build["method"] == "kmeans@0.3" and build["total_cost_usd"] > 0
    tax = Taxonomy.load(out / "topic" / "taxonomy.json")
    assert tax.facet == "topic"
    # --limit-papers: the reference is restricted to the sample, as in run_taxonomy_experiments.
    scores = build["facets"]["topic"]["scores"]
    assert in_unit_interval(scores["node_f1"])
    sample = C.load_corpus(layout / "taxobench" / "papers.jsonl.gz", 80, 0)
    expected = T.score_taxonomy(tax, T.restrict_taxonomy(Taxonomy.load(gold_path), sample.ids),
                                T.make_scorer(Runtime.from_config(quick_config()).embedder))
    assert scores["node_f1"] == pytest.approx(expected["node_f1"])
    assert scores["n_ref_nodes"] == expected["n_ref_nodes"]
    # Eta, mode and ablation all name distinct default folders.
    tags = {script.output_tag(T.MethodSpec.parse(k)) for k in ("coverage@0.1", "coverage@0.2", "one_off/coverage@0.2",
                                                                 "scholarpath", "scholarpath/no_discovery", "full")}
    assert tags == {"coverage@0.1", "coverage@0.2", "one_off-coverage@0.2", "scholarpath",
                    "scholarpath-no_discovery", "full"}


def test_script_run_scalability(layout, shared_cache, tmp_path):
    out = tmp_path / "scal"
    code = load_script("run_scalability").main([
        "--corpus", str(layout / "arxiv_subfields" / "papers.jsonl.gz"), "--sizes", "40,80", "--etas", "0.5",
        "--max-full-size", "40", "--facets", "problem", "--no-plots", "--cache-dir", str(shared_cache),
        "--output-dir", str(out), *QUICK])
    assert code == 0
    fig4 = read_csv(out / "fig4.csv")
    full = {int(r["size"]): r for r in fig4 if r["method"] == "full"}
    assert full[40]["full_estimated"] == "False" and full[80]["full_estimated"] == "True"
    assert all(math.isclose(float(r["selected_ratio_pct"]), 100.0) for r in full.values())
    assert {r["method"] for r in fig4} == {"full", "scholarpath", "random@0.5"}
    assert (out / "fig4_table.md").exists() and (out / "runs.csv").exists()


def test_script_run_dynamic_update(layout, shared_cache, tmp_path):
    out = tmp_path / "dyn"
    code = load_script("run_dynamic_update").main([
        "--corpus", str(layout / "arxiv_sets" / "synthetic.jsonl.gz"), "--cutoff", "2022-01-01",
        "--max-cohorts", "3", "--facets", "problem", "--compare-scratch", "--cache-dir", str(shared_cache),
        "--output-dir", str(out), *QUICK])
    assert code == 0
    cohorts = read_csv(out / "cohorts.csv")
    assert len(cohorts) == 3 and {"kept_zero_cost", "reaudited", "repaired", "cost_usd"} <= set(cohorts[0])
    summary = read_csv(out / "dynamic_summary.csv")[0]
    assert summary["facet"] == "problem" and in_unit_interval(summary["vs_scratch_node_f1"])
    assert (out / "updated" / "problem" / "taxonomy.json").exists()
    assert (out / "dynamic_summary_table.md").exists()
