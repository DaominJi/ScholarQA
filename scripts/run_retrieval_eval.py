#!/usr/bin/env python
"""Table 2 and Fig. 5: retrieval accuracy under a context token budget B (Sec. 5.3.1).

For every evidence benchmark (ResearchQA, ScholarQABench = ScholarQA-CS +
ScholarQA-Multi, TaxoBench, CHIME), every method packs deduplicated units up
to B tokens; the distinct source papers of the packed units, in selection
order, are scored against the gold source papers with Recall@B and NDCG@B
(unaligned gold citations stay in the denominator, App. C.2). Values are
means over a benchmark's queries; the macro average is the mean over the
benchmarks (Fig. 5).

Methods: BM25, SPECTER, Dense + reranker (SPECTER2 + listwise LLM reranker),
RAPTOR, LightRAG and ScholarPath. On arXiv-Subfields (148K papers) RAPTOR and
LightRAG run over a per-query candidate pool of the top ``--graph-candidates``
(default 100) SPECTER2 papers, since building a global tree / graph with LLM
summaries / extractions over every unit is prohibitive; corpora with at most
``--graph-global-max-papers`` (default 5,000) papers (TaxoBench, CHIME) get a
global tree / graph. Offline indexing calls are billed to ``index_build``.

Outputs (``--output-dir``, default ``results/run_retrieval_eval``)::

    retrievals/<benchmark>/<system>/B<budget>.jsonl   cached retrievals (+ per-query efficiency)
    per_query/<system>_B<budget>.jsonl                Recall@B / NDCG@B per query
    table2.md / table2.csv                            Table 2 at --table-budget (default 8,192)
    fig5.csv, fig5_Recall_B.csv, fig5_NDCG_B.csv      Fig. 5 (macro Recall@B / NDCG@B vs. B)
    fig5a_recall.png, fig5b_ndcg.png                  plots (if matplotlib is installed)
    summary.json

Paper setting (indexes built with build_index.py for arxiv_subfields, taxobench and chime)::

    python scripts/run_retrieval_eval.py --config configs/paper.yaml --resume

Quick offline run (synthetic layout, mock LLM + hashing embedder)::

    python scripts/run_retrieval_eval.py --data-dir data/synthetic --budgets 1024,2048 --limit 5 \\
        --graph-global-max-papers 0 --graph-candidates 8 --raptor-clustering kmeans --raptor-no-umap
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.evaluation.reporting import (  # noqa: E402
    markdown_table,
    plot_budget_sweep,
    write_budget_sweep,
    write_json,
    write_jsonl,
    write_table,
)
from scholarpath.experiments.qa import (  # noqa: E402
    BENCHMARKS,
    EVIDENCE_BENCHMARKS,
    PAPER_BUDGET,
    SWEEP_BUDGETS,
    SYSTEMS,
    TABLE2_SYSTEMS,
    DataLayout,
    EnvironmentCache,
    RecordStore,
    add_data_args,
    add_system_args,
    budget_tag,
    combine_retrieval,
    count_errors,
    environment_summary,
    evaluate_retrieval_records,
    index_store_from_args,
    matplotlib_available,
    parse_budgets,
    resolve_benchmarks,
    resolve_systems,
    run_queries,
    system_display,
    system_options_from_args,
    table2_spec,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402

log = logging.getLogger("run_retrieval_eval")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_output="results/run_retrieval_eval")
    add_data_args(ap)
    add_system_args(ap)
    ap.add_argument("--systems", default=None, help=f"comma list (default: {','.join(TABLE2_SYSTEMS)})")
    ap.add_argument("--budgets", default=",".join(str(b) for b in SWEEP_BUDGETS),
                    help="context budgets B (K = 1,024; default: the Fig. 5 sweep 2K,4K,6K,8K,16K)")
    ap.add_argument("--table-budget", type=int, default=PAPER_BUDGET,
                    help="budget of Table 2 (default 8,192; the largest budget run if absent)")
    ap.add_argument("--keep-context", action="store_true", help="also cache the rendered contexts")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    rt = Runtime.from_args(args)
    out: Path = rt.output_dir or Path("results/run_retrieval_eval")
    out.mkdir(parents=True, exist_ok=True)
    layout = DataLayout(args.data_dir)
    benchmarks = resolve_benchmarks(args.benchmarks, layout.available_benchmarks(), default=EVIDENCE_BENCHMARKS)
    if not benchmarks:
        ap.error(f"No evidence benchmark found under {args.data_dir}")
    systems = resolve_systems(args.systems, TABLE2_SYSTEMS)
    bad = [s for s in systems if not SYSTEMS[s].retrieves]
    if bad:
        ap.error(f"Systems without a budgeted retrieval cannot be evaluated here: {bad}")
    budgets = parse_budgets(args.budgets)
    if not budgets:
        ap.error("--budgets is empty")
    table_budget = args.table_budget if args.table_budget in budgets else max(budgets)

    options = system_options_from_args(args)
    envs = EnvironmentCache(rt, layout, index_store_from_args(args), options, args.link_predictor)
    store = RecordStore(out / "retrievals")
    queries = {b: layout.load_queries(b, args.limit) for b in benchmarks}
    log.info("Benchmarks: %s; systems: %s; budgets: %s", {b: len(q) for b, q in queries.items()},
             [system_display(s) for s in systems], budgets)

    evals: Dict[str, Dict[int, Dict[str, Any]]] = {s: {b: {} for b in budgets} for s in systems}
    errors: Dict[str, Dict[int, int]] = {s: {b: 0 for b in budgets} for s in systems}
    for bench in benchmarks:
        env = envs.for_benchmark(bench)
        group = BENCHMARKS[bench].group
        for system in systems:
            for budget in budgets:
                recs = run_queries(env, system, queries[bench], budget, benchmark=bench, mode="retrieval",
                                   store=store, resume=args.resume, fail_fast=args.fail_fast,
                                   keep_context=args.keep_context)
                evals[system][budget][bench] = evaluate_retrieval_records(recs, queries[bench], budget, group)
                errors[system][budget] += count_errors(recs)

    summary: Dict[str, Any] = {"environment": environment_summary(rt, options),
                               "benchmarks": {b: len(q) for b, q in queries.items()},
                               "budgets": budgets, "table_budget": table_budget, "systems": {}}
    table_rows: Dict[str, Dict[str, Any]] = {}
    sweep: Dict[str, Dict[int, Dict[str, Any]]] = {}
    groups: List[str] = []
    for system in systems:
        name = system_display(system)
        summary["systems"][name] = {}
        for budget in budgets:
            comb = combine_retrieval(evals[system][budget])
            rows = [r for ev in evals[system][budget].values() for r in ev.rows]
            write_jsonl(out / "per_query" / f"{system}_{budget_tag(budget)}.jsonl", rows)
            tokens = [r["context_tokens"] for r in rows if r.get("context_tokens") is not None]
            comb.update(errors=errors[system][budget], over_budget=sum(1 for r in rows if r["over_budget"]),
                        max_context_tokens=max(tokens) if tokens else 0,
                        mean_context_tokens=(sum(tokens) / len(tokens)) if tokens else 0.0)
            summary["systems"][name][str(budget)] = comb
            if comb["macro"]:
                sweep.setdefault(name, {})[budget] = {"Recall@B": comb["macro"]["R"], "NDCG@B": comb["macro"]["N"]}
            if budget == table_budget:
                table_rows[name] = comb["row"]
                groups.extend(comb["by_group"])
            if comb["over_budget"]:
                log.error("%s exceeded the budget B = %d on %d queries", name, budget, comb["over_budget"])
    spec = table2_spec(groups, table_budget)
    write_table(out / "table2.md", table_rows, spec)
    if sweep:
        write_budget_sweep(out / "fig5.csv", sweep, metrics=["Recall@B", "NDCG@B"])
        if len(budgets) > 1 and not args.no_plots and matplotlib_available():
            names = [system_display(s) for s in systems if system_display(s) in sweep][:8]
            plot_budget_sweep(sweep, "Recall@B", out / "fig5a_recall.png", ylabel="Macro Recall@B (%)",
                              title="Fig. 5(a) Macro Recall@B", systems=names)
            plot_budget_sweep(sweep, "NDCG@B", out / "fig5b_ndcg.png", ylabel="Macro NDCG@B (%)",
                              title="Fig. 5(b) Macro NDCG@B", systems=names)
    write_json(out / "summary.json", summary)
    print(markdown_table(table_rows, spec))
    if len(budgets) > 1:
        print("Fig. 5 (macro Recall@B / NDCG@B, %):")
        for name, by_b in sweep.items():
            cells = ", ".join(f"{b}: {v['Recall@B']:.1f}/{v['NDCG@B']:.1f}" for b, v in sorted(by_b.items()))
            print(f"  {name}: {cells}")
    print(f"Results written to {out}")
    n_err = sum(v for by_b in errors.values() for v in by_b.values())
    if n_err:
        print(f"Warning: {n_err} failed queries (counted as retrieving nothing); see the log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
