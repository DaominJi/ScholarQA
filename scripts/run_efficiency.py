#!/usr/bin/env python
"""Table 5: per-query efficiency of end-to-end QA, and the operator statistics of Sec. 5.3.3.

Aggregates the per-query efficiency rows saved with every cached answer by
run_qa_eval.py (budgeted methods at ``--budget``, default B = 8,192) and
run_agents.py (unbudgeted agents) into Table 5: LLM calls, tokens (K), cost ($,
DeepSeek-V4.0-Flash pricing) and end-to-end latency (s) per query. Judge
calls and offline indexing (RAPTOR / LightRAG ``index_build``) are excluded.
The latency of every budgeted system is defined identically (wall time minus
offline indexing, plus the recorded latency of calls served from the LLM
response cache), so a warm cache never favours one system; for the cleanest
timings run the systems with ``--set llm.cache_dir=null``.

Paper inconsistency. Table 5 reports 1.0 call per query for ScholarPath
("zero retrieval LLM calls and a single composition call"), whereas
Algorithm 2, Step 1 makes a lightweight query-understanding call
(``retrieval.llm_query_understanding``, default true). ``table5_detail``
therefore lists both the total calls and the composition-only calls; run
run_qa_eval.py with ``--no-query-llm`` (= ``--set
retrieval.llm_query_understanding=false``) for the LLM-free Step 1 that
matches Table 5's call count (cached records of the other setting are not
reused: every record carries a configuration fingerprint).

Operator statistics (Sec. 5.3.3: "prune 88.1% of postings and 93.3% of
candidate nodes (8.7 ms and 11.4 ms per query)"), from the search statistics
of ScholarPath's retrievals (``--operator-stats`` re-runs the retrieval
without composition on the given benchmarks; otherwise the statistics stored
with cached ScholarPath answers are summarized):

* postings pruned -- papers in the extents of the grounded scopes that the
  search never scanned (junction intersections and the early-terminated
  join, Sec. 4.3 / App. A.3), pooled over queries;
* candidate nodes pruned -- 1 - scored candidates / candidates in scope over
  the hypothesis queries (``stats["scored_candidates"]`` /
  ``stats["candidate_pool"]``; Table 7: 1 - 312 / 4,628 = 93.3%);
* the search time and the full retrieval time per query (ms).

``--exhaustive-check`` also re-runs each query with the exhaustive greedy
scan to verify Proposition 7 (same selected units) and to report the index
nodes it scores.

Outputs (``--output-dir``, default ``results/run_efficiency``)::

    table5.md/.csv            Calls / Tokens (K) / Cost ($) / Latency (s)
    table5_detail.md/.csv     + composition-only and other online calls, #queries
    operators.json/.md        operator statistics per benchmark and pooled
    efficiency_summary.json

Paper setting::

    python scripts/run_efficiency.py --inputs results/run_qa_eval,results/run_agents \\
        --config configs/paper.yaml --operator-stats

Quick offline run (after the quick runs of run_qa_eval.py / run_agents.py)::

    python scripts/run_efficiency.py --inputs results/run_qa_eval,results/run_agents --budget 2048 \\
        --data-dir data/synthetic --operator-stats --exhaustive-check --limit 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.evaluation.reporting import (  # noqa: E402
    TABLE5,
    Column,
    TableSpec,
    markdown_table,
    write_json,
    write_table,
)
from scholarpath.experiments.qa import (  # noqa: E402
    AGENT_SYSTEMS,
    BENCHMARKS,
    EVIDENCE_BENCHMARKS,
    HYPOTHESIS_BENCHMARKS,
    NO_RETRIEVAL,
    PAPER_BUDGET,
    QA_SYSTEMS,
    SCHOLARPATH,
    SYSTEMS,
    TABLE5_DETAIL,
    DataLayout,
    EnvironmentCache,
    RecordStore,
    add_data_args,
    add_system_args,
    budget_sections,
    collect_efficiency,
    csv_list,
    efficiency_summary,
    environment_summary,
    index_store_from_args,
    measure_operator_stats,
    operator_rows_from_records,
    order_systems,
    resolve_benchmarks,
    resolve_systems,
    summarize_operator_stats,
    system_display,
    system_options_from_args,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402

log = logging.getLogger("run_efficiency")

OPERATORS = TableSpec(
    "operators", "Operator statistics of ScholarPath's search (Sec. 5.3.3).",
    [Column("postings_pruned_pct", "Postings pruned (%)"),
     Column("candidate_nodes_pruned_pct", "Candidate nodes pruned (%)"),
     Column("scored_candidates_mean", "Scored candidates", higher_is_better=False),
     Column("candidate_pool_mean", "Candidates in scope", higher_is_better=False),
     Column("search_ms_mean", "Search (ms)", higher_is_better=False, digits=2),
     Column("retrieval_ms_mean", "Retrieval (ms)", higher_is_better=False, digits=2),
     Column("index_nodes_pruned_pct", "Index nodes pruned vs. exhaustive (%)"),
     Column("same_selection_rate", "Same units as exhaustive", digits=2),
     Column("n", "Queries", digits=0),
     Column("n_hypothesis", "Hypothesis queries", digits=0)],
    first_header="Benchmark",
)


def stores_of(inputs: List[Path]) -> List[RecordStore]:
    out = []
    for d in inputs:
        root = d / "answers" if (d / "answers").is_dir() else d
        if not root.is_dir():
            log.warning("Skipping missing input %s", d)
            continue
        out.append(RecordStore(root))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_output="results/run_efficiency")
    add_data_args(ap, run_flags=False)
    add_system_args(ap)
    ap.add_argument("--inputs", default="results/run_qa_eval,results/run_agents",
                    help="comma list of run directories (with answers/) or answer roots")
    ap.add_argument("--budget", type=int, default=PAPER_BUDGET, help="budget of the budgeted rows (default 8,192)")
    ap.add_argument("--systems", default=None,
                    help="systems to report (default: Keyword search, Dense retrieval, LightRAG, ScholarPath + "
                         "every agent found; 'all' adds every cached system)")
    ap.add_argument("--operator-stats", action="store_true",
                    help="measure ScholarPath's operator statistics live (needs the indexes)")
    ap.add_argument("--exhaustive-check", action="store_true",
                    help="with --operator-stats: also run the exhaustive scan per query to verify Proposition 7")
    args = ap.parse_args(argv)

    rt = Runtime.from_args(args)
    options = system_options_from_args(args)
    out: Path = rt.output_dir or Path("results/run_efficiency")
    out.mkdir(parents=True, exist_ok=True)
    stores = stores_of([Path(p) for p in csv_list(args.inputs)])
    bench_filter = resolve_benchmarks(args.benchmarks) if args.benchmarks else None

    if args.systems is None or args.systems.strip().lower() == "all":
        found = sorted({s for st in stores for _, s, _, _ in st.entries() if s in SYSTEMS})
        if args.systems is None:
            found = [s for s in found if s in QA_SYSTEMS or SYSTEMS[s].family == "agent"]
        # Paper row order: the budgeted methods, then the agents in Table 5's order.
        systems = order_systems(found, list(QA_SYSTEMS) + [NO_RETRIEVAL] + list(AGENT_SYSTEMS) + list(SYSTEMS))
    else:
        systems = resolve_systems(args.systems, QA_SYSTEMS)
    rows_by_system = collect_efficiency(stores, budget=args.budget, systems=systems, benchmarks=bench_filter)
    summaries = {system_display(s): efficiency_summary(rows_by_system[s]) for s in systems if s in rows_by_system}
    budgeted = [system_display(s) for s in systems if s in rows_by_system and SYSTEMS[s].budgeted]
    unbudgeted = [system_display(s) for s in systems if s in rows_by_system and not SYSTEMS[s].budgeted]
    sections = budget_sections(budgeted, unbudgeted, args.budget)
    result: Dict[str, Any] = {"environment": environment_summary(rt, options), "budget": args.budget,
                              "inputs": [str(s.root) for s in stores], "table5": summaries}
    if summaries:
        write_table(out / "table5.md", summaries, TABLE5, sections)
        write_table(out / "table5_detail.md", summaries, TABLE5_DETAIL, sections)
        print(markdown_table(summaries, TABLE5, sections))
        print(markdown_table(summaries, TABLE5_DETAIL, sections))
        sp = summaries.get(system_display(SCHOLARPATH))
        if sp and sp.get("Calls") is not None:
            print(f"Note: ScholarPath makes {sp['Calls']:.2f} LLM calls per query, of which "
                  f"{sp['Composition calls']:.2f} composition call(s); the rest is the Step-1 query-understanding "
                  "call of Algorithm 2 (Table 5 reports 1.0; run_qa_eval.py --no-query-llm drops it).")
    else:
        print(f"No cached efficiency rows found in {[str(s.root) for s in stores]} at B = {args.budget}")

    # ------------------------------------------------------------ operator statistics
    op_rows: Dict[str, List[Dict[str, Any]]] = {}
    if args.operator_stats:
        layout = DataLayout(args.data_dir)
        benchmarks = resolve_benchmarks(args.benchmarks, layout.available_benchmarks(),
                                        default=EVIDENCE_BENCHMARKS + HYPOTHESIS_BENCHMARKS)
        envs = EnvironmentCache(rt, layout, index_store_from_args(args), options, args.link_predictor)
        for bench in benchmarks:
            queries = layout.load_queries(bench, args.limit)
            op_rows[bench] = measure_operator_stats(envs.for_benchmark(bench), queries, args.budget,
                                                    exhaustive=args.exhaustive_check)
    else:
        for st in stores:
            for bench, system, budget, path in st.entries():
                if system == SCHOLARPATH and budget == args.budget and (bench_filter is None or bench in bench_filter):
                    cached = RecordStore.load_latest(path).values()
                    op_rows.setdefault(bench, []).extend(operator_rows_from_records(cached))
    if op_rows:
        ops = {BENCHMARKS[b].display if b in BENCHMARKS else b: summarize_operator_stats(r) for b, r in op_rows.items()}
        ops["All queries"] = summarize_operator_stats([r for rows in op_rows.values() for r in rows])
        write_table(out / "operators.md", ops, OPERATORS)
        write_json(out / "operators.json", {"summary": ops, "per_query": op_rows,
                                            "source": "live" if args.operator_stats else "cached answers"})
        print(markdown_table(ops, OPERATORS))
        result["operators"] = ops
    write_json(out / "efficiency_summary.json", result)
    print(f"Results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
