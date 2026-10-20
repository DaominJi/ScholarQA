#!/usr/bin/env python
"""Reproduce Table 6: ablation of taxonomy construction on the arXiv sets (App. C.4).

ScholarPath (eps = 0.10, 1 - delta = 0.95) is compared with six variants,
each switching off one component through the ``BuilderConfig`` ablation flags:

=========================================  ===========================================================
Variant (``--variants``)                    Change
=========================================  ===========================================================
``no_bayesian_acquisition``                 Stage 1 batches drawn uniformly at random (no Eq. 3 / 10)
``no_snapshot_certification``               no Stage 3 audit; stop at the fixed ratio eta_v = 0.08
``no_transferred_prior``                    fixed slope prior instead of the transferred prior (Eq. 14)
``no_discovery``                            no DISCOVER (missing concepts are not repaired)
``no_correction``                           no CORRECT (routing errors are not repaired)
``no_targeted_repair``                      no Stage 2 at all
=========================================  ===========================================================

Every variant builds the problem and the method taxonomy of every arXiv set
and is evaluated against FULL of the same set and facet. The table reports
soft node / edge F1 (%), the router abstention rate a_v (Abstain, %) and the
validation disagreement rate d_v (Disagr., %) -- both averaged over the
episodes of a build -- and the relative LLM cost C_LLM / C_Full, averaged
over sets and facets. FULL and ScholarPath come from the shared build cache
(``--cache-dir``), so with ``--resume`` they are reused from
``run_taxonomy_experiments.py`` (FULL's cache key ignores router settings, so
it is reused whatever ``baseline.router`` / ``--min-router-steps`` were;
ScholarPath is reused when the ``builder`` settings match).

Outputs (``--output-dir``, default ``results/run_taxonomy_ablation``)::

    runs.{csv,jsonl}, summary.{csv,jsonl}, table6.{md,csv}, report.json

Paper setting (DeepSeek-V4.0-Flash, SPECTER2)::

    python scripts/run_taxonomy_ablation.py --config configs/paper.yaml --resume

Quick offline run on the synthetic layout (mock LLM, hashing embedder)::

    python scripts/make_synthetic_data.py --out /tmp/syn
    python scripts/run_taxonomy_ablation.py --data-dir /tmp/syn --limit-papers 120 --facets problem \\
        --variants no_discovery,no_targeted_repair --set builder.router.hidden_dim=32 --output-dir /tmp/syn_tab6
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
    seeds_from_args,
    write_rows,
)
from scholarpath.experiments.taxonomy import (  # noqa: E402
    ABLATIONS,
    DUAL_FACETS,
    ConstructionRunner,
    ablation_methods,
    load_arxiv_datasets,
    make_scorer,
    render_table,
    run_construction_comparison,
    summarize_runs,
    table6,
    write_table_files,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402

log = logging.getLogger("run_taxonomy_ablation")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", type=csv_list(str), default=None,
                    help="arXiv sets to use (default: every set under <data-dir>/arxiv_sets)")
    ap.add_argument("--facets", type=csv_list(str), default=list(DUAL_FACETS), help="default: problem,method")
    ap.add_argument("--variants", type=csv_list(str), default=list(ABLATIONS),
                    help=f"ablations to run (default: all of {', '.join(ABLATIONS)})")
    ap.add_argument("--metric-text", choices=("name", "name_definition"), default="name",
                    help="node label embedded by the soft F1 metrics")
    add_experiment_args(ap, plots=False)
    add_common_args(ap, default_output="results/run_taxonomy_ablation")
    args = ap.parse_args(argv)
    bad = [v for v in args.variants if v not in ABLATIONS]
    if bad:
        ap.error(f"unknown variants {bad}; expected a subset of {list(ABLATIONS)}")
    bad = [f for f in args.facets if f not in DUAL_FACETS]
    if bad:
        ap.error(f"unknown facets {bad}; expected a subset of {list(DUAL_FACETS)}")

    rt = Runtime.from_args(args)
    cache = cache_from_args(args)
    runner = ConstructionRunner.from_runtime(rt, cache, min_router_steps=args.min_router_steps)
    datasets = load_arxiv_datasets(args.data_dir, args.categories, args.limit_papers, args.sample_seed,
                                   facets=args.facets)
    methods = ablation_methods(args.variants)
    seeds = seeds_from_args(args)
    rows = run_construction_comparison(datasets, methods, runner, seeds, make_scorer(rt.embedder, args.metric_text))
    summary = summarize_runs(rows)
    out = rt.output_dir
    table = table6(summary, variants=args.variants, config=rt.config.builder)
    paths = {"runs": write_rows(out, "runs", rows), "summary": write_rows(out, "summary", summary),
             "table6": write_table_files(out, "table6", table)}
    write_json(out / "report.json", {"datasets": [{"name": d.name, "papers": len(d.corpus)} for d in datasets],
                                     "facets": args.facets, "variants": args.variants, "seeds": seeds,
                                     "runs": len(rows), "cache": cache.summary(), "llm_usage": rt.cost_summary(),
                                     "outputs": paths})
    print(render_table(table))
    print(f"{len(rows)} runs ({cache.stats['built']} built, {cache.stats['disk_hits']} reused from disk); "
          f"outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
