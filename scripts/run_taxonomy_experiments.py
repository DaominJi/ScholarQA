#!/usr/bin/env python
"""Reproduce Fig. 2, Fig. 3 and Tables 8-9: taxonomy construction quality and cost (Section 5.2, App. C.5).

* **Fig. 2** -- soft node F1 and soft edge F1 vs. selection ratio
  eta in {0.05, 0.1, 0.2, 0.3, 0.4, 0.5} for the node-wise selectors Random /
  Coverage / K-means / Diversity, with ScholarPath (and, on TaxoBench, FULL)
  as reference lines.
* **Fig. 3** -- relative LLM construction cost C_LLM / C_Full of FULL,
  node-wise Coverage selection at eta = 0.5, 0.4, 0.2 (``--fig3-selector``;
  the paper's eta = 0.2 bars are the Coverage costs of Tables 8-9), and
  ScholarPath.
* **Tables 8-9** -- one-off vs. node-wise selection at eta = 0.2 on arXiv
  (vs. FULL) and TaxoBench (vs. the expert gold).

Benchmarks (``--benchmarks``):

* ``arxiv`` -- every arXiv set (``<data-dir>/arxiv_sets/*.jsonl.gz``; the
  paper's eight 2,000-paper sets): each method builds the problem *and* the
  method taxonomy, evaluated against FULL (eta = 1.0) of the same set and
  facet; results per facet and averaged over sets x facets.
* ``taxobench`` -- every method, FULL included, builds one topic-and-approach
  taxonomy over the pooled papers, evaluated against the merged expert gold
  taxonomy (``<data-dir>/taxobench/gold_taxonomy.json``).

Cost is the total billed LLM cost of a build (Eq. 15) relative to FULL on the
same dataset and facet. FULL is built once per (dataset, facet) and reused by
every comparison; all builds are cached in ``--cache-dir`` (shared with
``run_taxonomy_ablation.py`` and ``build_taxonomy.py``), and ``--resume``
reuses the builds of earlier runs. Build seeds: ``--seeds`` (or the global
``--seed``); FULL always uses ``baseline.seed``.

Outputs (``--output-dir``, default ``results/run_taxonomy_experiments``)::

    runs.{csv,jsonl}       one row per build: dataset, facet, method, eta, mode, node/edge P/R/F1,
                           cost_usd, relative_cost, tokens, calls, papers_read, runtime_s, ...
    summary.{csv,jsonl}    means per (benchmark, facet, method); facet "avg" pools sets x facets
    fig2.csv, fig2_<benchmark>_{node,edge}_f1.png
    fig3.csv, fig3.png
    table8.{md,csv}, table9.{md,csv}
    report.json            datasets, cache statistics, total LLM usage

Paper setting (DeepSeek-V4.0-Flash, SPECTER2; App. C.2 reports the merged gold with 4 levels)::

    python scripts/run_taxonomy_experiments.py --config configs/paper.yaml --gold-max-depth 4 --resume

Quick offline run on the synthetic layout (mock LLM, hashing embedder)::

    python scripts/make_synthetic_data.py --out /tmp/syn
    python scripts/run_taxonomy_experiments.py --data-dir /tmp/syn --limit-papers 120 \\
        --etas 0.1,0.2 --selectors coverage,random --fig3-etas 0.2 --min-router-steps 100 \\
        --set builder.router.hidden_dim=32 --set baseline.router.hidden_dim=32 --output-dir /tmp/syn_fig2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.evaluation.reporting import write_json  # noqa: E402
from scholarpath.experiments.common import (  # noqa: E402
    add_experiment_args,
    cache_from_args,
    csv_list,
    seeds_from_args,
)
from scholarpath.experiments.taxonomy import (  # noqa: E402
    ARXIV,
    DUAL_FACETS,
    FIG2_ETAS,
    FIG2_SELECTORS,
    FIG3_ETAS,
    FIG3_MEAN,
    FIG3_SELECTOR,
    TABLE89_ETA,
    TAXOBENCH,
    ConstructionRunner,
    construction_methods,
    load_arxiv_datasets,
    load_taxobench_dataset,
    make_scorer,
    render_table,
    run_construction_comparison,
    summarize_runs,
    table8,
    table9,
    write_construction_report,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402
from scholarpath.taxonomy.selectors import SELECTION_METHODS  # noqa: E402

log = logging.getLogger("run_taxonomy_experiments")
BENCHMARKS = (ARXIV, TAXOBENCH)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", type=csv_list(str), default=list(BENCHMARKS), help="arxiv,taxobench")
    ap.add_argument("--categories", type=csv_list(str), default=None,
                    help="arXiv sets to use (default: every set under <data-dir>/arxiv_sets)")
    ap.add_argument("--facets", type=csv_list(str), default=list(DUAL_FACETS),
                    help="arXiv facets (default: problem,method); TaxoBench always uses the topic facet")
    ap.add_argument("--selectors", type=csv_list(str), default=list(FIG2_SELECTORS),
                    help="node-wise / one-off selectors (default: coverage,kmeans,diversity,random)")
    ap.add_argument("--etas", type=csv_list(float), default=list(FIG2_ETAS),
                    help="node-wise ratios of Fig. 2 (the --fig3-etas and --table-eta ratios are built as well)")
    ap.add_argument("--fig3-etas", type=csv_list(float), default=list(FIG3_ETAS), help="node-wise bars of Fig. 3")
    ap.add_argument("--fig3-selector", choices=list(SELECTION_METHODS) + [FIG3_MEAN], default=FIG3_SELECTOR,
                    help="selector of the Fig. 3 bars (default coverage, as in the paper; built at --fig3-etas "
                         "even if it is not in --selectors); 'mean' averages --selectors")
    ap.add_argument("--table-eta", type=float, default=TABLE89_ETA, help="ratio of Tables 8-9 (default 0.2)")
    ap.add_argument("--no-one-off", action="store_true", help="skip one-off selection (Tables 8-9 lose a section)")
    ap.add_argument("--gold-max-depth", type=int, default=None,
                    help="cut the TaxoBench gold below this many levels under the common root (paper: 4)")
    ap.add_argument("--metric-text", choices=("name", "name_definition"), default="name",
                    help="node label embedded by the soft F1 metrics")
    add_experiment_args(ap)
    add_common_args(ap, default_output="results/run_taxonomy_experiments")
    args = ap.parse_args(argv)

    unknown = [b for b in args.benchmarks if b not in BENCHMARKS]
    if unknown:
        ap.error(f"unknown benchmarks {unknown}; expected a subset of {list(BENCHMARKS)}")
    bad = [s for s in args.selectors if s not in SELECTION_METHODS]
    if bad:
        ap.error(f"unknown selectors {bad}; expected a subset of {list(SELECTION_METHODS)}")
    bad = [f for f in args.facets if f not in DUAL_FACETS]
    if bad:
        ap.error(f"unknown arXiv facets {bad}; expected a subset of {list(DUAL_FACETS)}")

    rt = Runtime.from_args(args)
    cache = cache_from_args(args)
    runner = ConstructionRunner.from_runtime(rt, cache, min_router_steps=args.min_router_steps)
    datasets = []
    if ARXIV in args.benchmarks:
        datasets += load_arxiv_datasets(args.data_dir, args.categories, args.limit_papers, args.sample_seed,
                                        facets=args.facets)
    if TAXOBENCH in args.benchmarks:
        datasets.append(load_taxobench_dataset(args.data_dir, args.limit_papers, args.sample_seed,
                                               args.gold_max_depth))
    log.info("datasets: %s", ", ".join(f"{d.name} ({len(d.corpus)} papers, {'/'.join(d.facets)})" for d in datasets))

    etas = sorted(set(args.etas) | set(args.fig3_etas) | {args.table_eta})
    one_off = () if args.no_one_off else (args.table_eta,)
    methods = construction_methods(args.selectors, etas, one_off, fig3_selector=args.fig3_selector,
                                   fig3_etas=args.fig3_etas)
    seeds = seeds_from_args(args)
    rows = run_construction_comparison(datasets, methods, runner, seeds, make_scorer(rt.embedder, args.metric_text))
    out = rt.output_dir
    paths = write_construction_report(rows, out, selectors=args.selectors, etas=etas, fig3_etas=args.fig3_etas,
                                      fig3_selector=args.fig3_selector, table_eta=args.table_eta,
                                      plots=not args.no_plots)
    report: Dict[str, Any] = {
        "datasets": [{"name": d.name, "benchmark": d.benchmark, "papers": len(d.corpus), "facets": list(d.facets),
                      "reference": {f: d.reference_kind(f) for f in d.facets}} for d in datasets],
        "methods": [m.key for m in methods],
        "seeds": seeds,
        "fig3_selector": args.fig3_selector,
        "runs": len(rows),
        "cache": cache.summary(),
        "llm_usage": rt.cost_summary(),
        "outputs": paths,
    }
    write_json(out / "report.json", report)

    summary = summarize_runs(rows)
    for table in (table8(summary, args.table_eta), table9(summary, args.table_eta)):
        if table[0]:
            print(render_table(table))
    print(f"{len(rows)} runs ({cache.stats['built']} built, {cache.stats['disk_hits']} reused from disk); "
          f"outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
