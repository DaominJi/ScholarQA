#!/usr/bin/env python
"""Dynamic update study (Section 3.4, Appendix B.3): keep a certified taxonomy up to date as papers arrive.

ScholarPath first builds the taxonomy (problem and method facets by default)
on the papers first submitted before ``--cutoff`` (default: the date that
leaves ``--base-fraction`` of the dated papers before it). The later papers
are then streamed in date order through
:class:`~scholarpath.taxonomy.dynamic.DynamicUpdater`, in calendar-month
cohorts (default) or fixed-size batches (``--batch-size``).

At a certified node of size n with bound U_old, the frozen router assigns
the M arrivals at zero LLM cost and the bound becomes the worst case
U_new = (n U_old + M) / (n + M) (Eq. 16). The node stays certified without
any LLM call while U_new <= eps and the residual leaf stays below the
discovery threshold; otherwise the builder audits the new cohort and, if new
subtopics or routing drift appear, runs audit-guided repair (DISCOVER,
CORRECT) and certifies a new snapshot (Section 3.3).

Per cohort the script reports the nodes kept certified at zero LLM cost, the
nodes whose cohort was re-audited, re-certified, repaired or failed, new
concepts, and the LLM cost (absolute and cumulative). With
``--compare-scratch`` ScholarPath is also built from scratch on all papers
(base + stream) and the updated taxonomy is compared with it (soft node /
edge F1, cost).

Outputs (``--output-dir``, default ``results/run_dynamic_update``)::

    cohorts.{csv,jsonl}        one row per (facet, cohort)
    dynamic_summary.{csv,jsonl} one row per facet (totals; scratch comparison)
    dynamic_summary.{md}       the same as a table
    updated/<facet>/...        the updated BuildResult (taxonomy, episodes, usage, routers)
    report.json

Paper setting (DeepSeek-V4.0-Flash, SPECTER2; e.g. on the cs.DB arXiv set)::

    python scripts/run_dynamic_update.py --config configs/paper.yaml \\
        --corpus data/processed/arxiv_sets/cs.DB.jsonl.gz --cutoff 2024-01-01 --compare-scratch --resume

Quick offline run on the synthetic layout (mock LLM, hashing embedder)::

    python scripts/make_synthetic_data.py --out /tmp/syn
    python scripts/run_dynamic_update.py --corpus /tmp/syn/arxiv_sets/synthetic.jsonl.gz --cutoff 2022-01-01 \\
        --max-cohorts 3 --compare-scratch --set builder.router.hidden_dim=32 --set builder.router.epochs=20 \\
        --output-dir /tmp/syn_dynamic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.data.corpus import Corpus  # noqa: E402
from scholarpath.evaluation.reporting import Column, TableSpec, write_json  # noqa: E402
from scholarpath.experiments.common import (  # noqa: E402
    add_experiment_args,
    cache_from_args,
    csv_list,
    load_corpus,
    save_build_result,
    seed_from_args,
    write_rows,
)
from scholarpath.experiments.taxonomy import (  # noqa: E402
    DUAL_FACETS,
    SCHOLARPATH_METHOD,
    ConstructionDataset,
    ConstructionRunner,
    dynamic_summary,
    make_scorer,
    render_table,
    run_dynamic_update,
    split_stream,
    write_table_files,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402

log = logging.getLogger("run_dynamic_update")

SUMMARY_SPEC = TableSpec(
    "dynamic_update", "Dynamic update (Section 3.4): streamed cohorts vs. a from-scratch build.",
    [Column("cohorts", "Cohorts", digits=0), Column("papers_streamed", "Papers", digits=0),
     Column("zero_cost_cohorts", "Zero-cost cohorts", digits=0),
     Column("kept_zero_cost", "Kept (no LLM)", digits=0), Column("reaudited", "Re-audited", digits=0),
     Column("repaired", "Repaired", digits=0), Column("certified_fraction", "Certified frac.", digits=3),
     Column("base_cost_usd", "Base ($)", higher_is_better=False, digits=4),
     Column("update_cost_usd", "Update ($)", higher_is_better=False, digits=4),
     Column("scratch_cost_usd", "Scratch ($)", higher_is_better=False, digits=4),
     Column("vs_scratch_node_f1", "Node F1 vs. scratch", digits=3),
     Column("vs_scratch_edge_f1", "Edge F1 vs. scratch", digits=3)],
    first_header="Facet",
)


def corpus_name(path: Path) -> str:
    name = path.name.split(".jsonl")[0]
    return path.parent.name if name == "papers" else name


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/arxiv_sets/cs.DB.jsonl.gz"),
                    help="processed corpus with first-version dates (Paper JSONL)")
    ap.add_argument("--cutoff", default=None, help="ISO date; papers before it form the base repository")
    ap.add_argument("--base-fraction", type=float, default=0.8,
                    help="without --cutoff: share of dated papers in the base repository (default 0.8)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="stream fixed-size batches instead of calendar-month cohorts")
    ap.add_argument("--max-cohorts", type=int, default=None, help="stop after this many cohorts")
    ap.add_argument("--facets", type=csv_list(str), default=list(DUAL_FACETS), help="default: problem,method")
    ap.add_argument("--compare-scratch", action="store_true",
                    help="also build ScholarPath from scratch on base + streamed papers and compare")
    add_experiment_args(ap, data_dir=False, seeds=False, plots=False)
    add_common_args(ap, default_output="results/run_dynamic_update")
    args = ap.parse_args(argv)
    bad = [f for f in args.facets if f not in ("problem", "method", "topic")]
    if bad:
        ap.error(f"unknown facets {bad}")

    rt = Runtime.from_args(args)
    name = corpus_name(args.corpus)
    corpus = load_corpus(args.corpus, args.limit_papers, args.sample_seed, name=name)
    split = split_stream(corpus, args.cutoff, args.base_fraction, cohort="size" if args.batch_size else "month",
                         batch_size=args.batch_size, max_cohorts=args.max_cohorts)
    log.info("cutoff %s: %d base papers, %d cohorts / %d streamed papers (%d undated papers left out)",
             split.cutoff, len(split.base), len(split.cohorts), len(split.streamed), split.undated)
    if not split.cohorts:
        log.warning("no paper is dated on or after the cutoff %s; nothing to stream", split.cutoff)
    cache = cache_from_args(args)
    runner = ConstructionRunner.from_runtime(rt, cache, min_router_steps=args.min_router_steps)
    base = ConstructionDataset(name=f"{name}@{split.cutoff}", benchmark="dynamic", corpus=split.base,
                               facets=tuple(args.facets))
    scratch_ds = None
    if args.compare_scratch:
        everything = Corpus(list(split.base) + split.streamed, name=f"{name}@{split.cutoff}+stream")
        scratch_ds = ConstructionDataset(name=everything.name, benchmark="dynamic", corpus=everything,
                                         facets=tuple(args.facets))
    scorer = make_scorer(rt.embedder) if scratch_ds is not None else None

    out = rt.output_dir
    cohort_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    seed = seed_from_args(args)
    for facet in args.facets:
        run = run_dynamic_update(base, split.cohorts, runner, facet, seed)
        cohort_rows += run.cohorts
        save_build_result(run.updated, out / "updated" / facet)
        scratch = runner.build(scratch_ds, facet, SCHOLARPATH_METHOD, seed) if scratch_ds else None
        summaries.append({"cutoff": split.cutoff, **dynamic_summary(run, scratch, scorer)})

    table_rows = {s["facet"]: s for s in summaries}
    paths = {"cohorts": write_rows(out, "cohorts", cohort_rows),
             "summary": write_rows(out, "dynamic_summary", summaries),
             "table": write_table_files(out, "dynamic_summary_table", (table_rows, [], SUMMARY_SPEC), bold_best=False)}
    write_json(out / "report.json", {"corpus": str(args.corpus), "cutoff": split.cutoff, "base_papers": len(split.base),
                                     "cohorts": [c.label for c in split.cohorts], "streamed": len(split.streamed),
                                     "undated": split.undated, "facets": args.facets, "cache": cache.summary(),
                                     "llm_usage": rt.cost_summary(), "outputs": paths})
    print(render_table((table_rows, [], SUMMARY_SPEC), bold_best=False))
    print(f"outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
