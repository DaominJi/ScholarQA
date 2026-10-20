#!/usr/bin/env python
"""Reproduce Fig. 4: scalability of taxonomy construction across corpus sizes (Section 5.2.2).

Nested uniform samples (one seeded permutation; ``--sample-seed``) of
|P| in {2K, 4K, 6K, 8K, 10K, 50K, 148.4K} papers are drawn from arXiv-Subfields
(the largest size is the whole corpus). At each size the dual taxonomy
(problem + method) is built with ScholarPath and with fixed-ratio node-wise
selection at eta = 0.2 and 0.5 (``--selector``, default random), and the
script reports

* (a) the average selected ratio C_LLM / C_Full of the dual taxonomy (%):
  the total cost of both facets relative to FULL's (the paper's 3.9% at
  148.4K is $5.4 / $138.4); the mean of the per-facet ratios is kept as
  ``mean_facet_ratio_pct``; and
* (b) the total construction cost ($) of both facets, FULL included.

FULL (eta = 1) is run up to ``--max-full-size`` papers. Above it, running FULL
is prohibitive and C_Full is estimated as c̄ Σ_{v ∈ V_int} n_v: c̄ is the
per-paper-node cost measured on FULL at the largest size where it was run
(per facet; ``taxonomy.baselines.per_paper_node_cost``) and n_v are the routed
paper counts of the internal nodes of ScholarPath's taxonomy at that size
(``taxonomy.baselines.estimate_full_cost``). Where FULL was run, the measured
cost is used and the estimate is reported next to it for validation.

Outputs (``--output-dir``, default ``results/run_scalability``)::

    runs.{csv,jsonl}      one row per (size, facet, method): cost, C_Full (measured / estimated), ratio, ...
    fig4.{csv,jsonl}      per (size, method): selected_ratio_pct (Fig. 4a), total_cost_usd (Fig. 4b),
                          full_cost_usd, mean_facet_ratio_pct
    fig4_table.{md,csv}   the same as a table (one row per size)
    fig4a_selected_ratio.png, fig4b_total_cost.png
    report.json

Paper setting (DeepSeek-V4.0-Flash, SPECTER2)::

    python scripts/run_scalability.py --config configs/paper.yaml --resume

Quick offline run on the synthetic layout (mock LLM, hashing embedder)::

    python scripts/make_synthetic_data.py --out /tmp/syn
    python scripts/run_scalability.py --corpus /tmp/syn/arxiv_subfields/papers.jsonl.gz \\
        --sizes 60,120 --max-full-size 60 --min-router-steps 100 \\
        --set builder.router.hidden_dim=32 --set baseline.router.hidden_dim=32 --output-dir /tmp/syn_fig4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.evaluation.reporting import write_json  # noqa: E402
from scholarpath.experiments.common import (  # noqa: E402
    add_experiment_args,
    cache_from_args,
    csv_list,
    load_corpus,
    matplotlib_available,
    seed_from_args,
    write_rows,
)
from scholarpath.experiments.taxonomy import (  # noqa: E402
    DUAL_FACETS,
    SCALABILITY_ETAS,
    SCALABILITY_SIZES,
    ConstructionRunner,
    figure4_table,
    plot_figure4,
    render_table,
    run_scalability,
    scalability_summary,
    write_table_files,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402
from scholarpath.taxonomy.selectors import SELECTION_METHODS  # noqa: E402

log = logging.getLogger("run_scalability")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/arxiv_subfields/papers.jsonl.gz"),
                    help="arXiv-Subfields corpus (default: data/processed/arxiv_subfields/papers.jsonl.gz)")
    ap.add_argument("--sizes", type=csv_list(int), default=list(SCALABILITY_SIZES), help="corpus sizes |P|")
    ap.add_argument("--etas", type=csv_list(float), default=list(SCALABILITY_ETAS), help="node-wise ratios")
    ap.add_argument("--selector", choices=list(SELECTION_METHODS), default="random",
                    help="selector of the node-wise baselines (Fig. 4 does not name one; default random)")
    ap.add_argument("--max-full-size", type=int, default=10000,
                    help="run FULL up to this |P|; estimate C_Full above it (default 10000)")
    ap.add_argument("--facets", type=csv_list(str), default=list(DUAL_FACETS), help="default: problem,method")
    add_experiment_args(ap, data_dir=False, seeds=False)
    add_common_args(ap, default_output="results/run_scalability")
    args = ap.parse_args(argv)
    bad = [f for f in args.facets if f not in DUAL_FACETS]
    if bad:
        ap.error(f"unknown facets {bad}; expected a subset of {list(DUAL_FACETS)}")

    rt = Runtime.from_args(args)
    corpus = load_corpus(args.corpus, args.limit_papers, args.sample_seed, name="subfields")
    log.info("corpus %s: %d papers", args.corpus, len(corpus))
    cache = cache_from_args(args)
    runner = ConstructionRunner.from_runtime(rt, cache, min_router_steps=args.min_router_steps)
    rows = run_scalability(corpus, args.sizes, runner, etas=args.etas, selector=args.selector,
                           max_full_size=args.max_full_size, seed=seed_from_args(args), sample_seed=args.sample_seed,
                           facets=args.facets)
    summary = scalability_summary(rows)
    out = rt.output_dir
    paths = {"runs": write_rows(out, "runs", rows), "fig4": write_rows(out, "fig4", summary),
             "table": write_table_files(out, "fig4_table", figure4_table(summary), bold_best=False)}
    if not args.no_plots:
        if matplotlib_available():
            paths["plots"] = plot_figure4(summary, out)
        else:
            log.warning("matplotlib is not installed; skipping the figures (their CSV data is written)")
    write_json(out / "report.json", {"corpus": str(args.corpus), "papers": len(corpus), "sizes": args.sizes,
                                     "etas": args.etas, "selector": args.selector,
                                     "max_full_size": args.max_full_size, "cache": cache.summary(),
                                     "llm_usage": rt.cost_summary(), "outputs": paths})
    print(render_table(figure4_table(summary), bold_best=False))
    print(f"outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
