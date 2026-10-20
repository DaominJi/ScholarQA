#!/usr/bin/env python
"""Tables 3-4 and Fig. 6: end-to-end QA on evidence and hypothesis queries (Sec. 5.3.2).

Budgeted methods -- Keyword search, Dense retrieval (SPECTER2), LightRAG and
ScholarPath, plus the unassisted answer LLM on hypothesis queries -- pack at
most B context tokens (default 8,192) and invoke the backbone LLM in ONE
composition call (Algorithm 2, Step 6; the same call for every method).
Organization queries (TaxoBench, CHIME) compose a topic hierarchy from the
same <= B context instead (prompts/organize.py). Answers of the unbudgeted
agents (run_agents.py) are evaluated with ``--answers-dir``; without an
explicit ``--systems``, ``--answers-dir`` implies ``--eval-only`` for the
budgeted systems (their cached answers in ``--output-dir`` are evaluated, none
is generated), so evaluating the agents never regenerates or replaces the
budgeted answers.

Metrics (Sec. 5.3 "Metrics", App. C.3):

* Table 3 -- ResearchQA: Rub (rubric coverage, official judge protocol) and
  Cite (citation F1 vs. gold papers); ScholarQABench: Corr and Cite, the CS
  (rubric score) and Multi (Prometheus-2) subsets averaged weighted by their
  numbers of queries, with every citation judged against the text of that
  paper the system actually had in its context (``Answer.metadata
  ["passages"]``); TaxoBench: ARI and Sem; CHIME: Cat and Assn; plus the
  macro-average over the eight metrics quoted in Sec. 5.3.2;
* Table 4 -- held-out hypothesis queries: Hit@10 (Disc) and the
  Feasibility / Novelty / Soundness scores (0-100) of three LLM judges and
  their average;
* Fig. 6 -- Disc of the budgeted methods for B in {2K, 4K, 6K, 8K, 16K}.

Judges come from ``config.evaluation`` (mock judges offline): the three
hypothesis judges (``judge_models``), the official ResearchQA / ScholarQA-CS
judge models, Prometheus-2 7B, and the statement-level citation judge
(``attribution_judge``: ``nli`` = AttrScore-Flan-T5-XL as in the official
ScholarQABench script, ``llm`` = the CS judge with the AttrScore prompt). Each
judge has a private cost meter, so judge spending is reported separately
(``judge_usage.json``) and never enters the per-query efficiency rows that
run_efficiency.py aggregates into Table 5.

Outputs (``--output-dir``, default ``results/run_qa_eval``)::

    answers/<benchmark>/<system>/B<budget>.jsonl   cached answers + per-query efficiency (Table 5 input)
    eval/<benchmark>/<system>_<tag>.json           per-query scores and summary
    table1.md/.csv                                 Table 1 (benchmark statistics)
    table3.md/.csv, table4.md/.csv                 Tables 3-4 at --table-budget (default 8,192)
    fig6.csv, fig6_Disc.csv, fig6_disc.png         Fig. 6
    efficiency.jsonl                               flat per-query efficiency rows
    judge_usage.json, summary.json

Every cached answer carries a fingerprint of the configuration that produced
it (system variant, retrieval / system options such as ``--no-query-llm``,
link predictor, LLM, embedder, index); ``--resume`` regenerates the answers
of any other configuration (appending them), and a run without ``--resume``
refuses to replace a cache file that holds answers of another configuration
unless ``--force`` is given. ``--eval-only`` warns when the cached answers
were produced under another configuration than the current ``--config``.

Failed or missing answers score 0 on every metric (as in Recall@B, the
organization metrics and Disc), and ScholarQABench weights its subsets by
their numbers of queries (100 / 108); the "Failed" column of Tables 3-4
counts them (shown only when some system has failures). Agent rows follow
the paper's order (PaperQA2, OpenScholar, AgenticScholar, Codex). Table 1
(queries, corpus size and gold papers per query of every benchmark) is
written as ``table1.md``.

Paper setting::

    python scripts/run_qa_eval.py --config configs/paper.yaml --set evaluation.judge_provider=openai \\
        --set evaluation.attribution_judge=nli --resume
    # after run_agents.py: evaluate the agents next to the cached budgeted answers (no regeneration)
    python scripts/run_qa_eval.py --config configs/paper.yaml --set evaluation.judge_provider=openai \\
        --set evaluation.attribution_judge=nli --answers-dir results/run_agents --eval-only

Quick offline run (synthetic layout, mock LLM and judges)::

    python scripts/run_qa_eval.py --data-dir data/synthetic --budgets 2048 --sweep-budgets 1024,2048 --limit 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.evaluation.reporting import (  # noqa: E402
    TABLE4,
    markdown_table,
    plot_budget_sweep,
    table_rows,
    write_budget_sweep,
    write_csv,
    write_json,
    write_jsonl,
    write_table,
)
from scholarpath.experiments.qa import (  # noqa: E402
    BENCHMARKS,
    EVIDENCE_BENCHMARKS,
    HYPOTHESIS_BENCHMARKS,
    NO_RETRIEVAL,
    PAPER_BUDGET,
    SWEEP_BUDGETS,
    SYSTEMS,
    TABLE1,
    TABLE34_AGENT_ORDER,
    TABLE4_SYSTEMS,
    BenchmarkEval,
    CacheConflictError,
    DataLayout,
    EnvironmentCache,
    RecordStore,
    RunRecord,
    add_data_args,
    add_system_args,
    answers_of,
    budget_sections,
    budget_tag,
    build_judges,
    check_cache_conflicts,
    count_errors,
    environment_summary,
    evaluate_benchmark,
    index_store_from_args,
    matplotlib_available,
    order_systems,
    organization_embedder,
    parse_budgets,
    qa_table_row,
    resolve_benchmarks,
    resolve_systems,
    run_queries,
    system_display,
    system_options_from_args,
    table1_rows,
    table3_spec,
    unanswered_cell,
    with_failed_column,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402

log = logging.getLogger("run_qa_eval")


def answer_stores(dirs: List[Path]) -> List[RecordStore]:
    """Record stores of ``--answers-dir`` (a run directory with ``answers/`` or the answers root itself)."""
    out = []
    for d in dirs:
        root = d / "answers" if (d / "answers").is_dir() else d
        if not root.is_dir():
            raise FileNotFoundError(f"--answers-dir {d} does not exist")
        out.append(RecordStore(root))
    return out


def plan_runs(benchmarks: List[str], systems: List[str], budgets: List[int],
              sweep: List[int]) -> List[Tuple[str, str, int]]:
    """(benchmark, system, budget) runs: Tables 3-4 at ``budgets``; Fig. 6 sweep on hypothesis queries.

    The unassisted answer LLM only answers hypothesis queries (Table 4) and is
    not part of the budget sweep (Fig. 6 plots the four budgeted methods).
    """
    plan: List[Tuple[str, str, int]] = []
    for bench in benchmarks:
        hyp = BENCHMARKS[bench].is_hypothesis
        for system in systems:
            if system == NO_RETRIEVAL and not hyp:
                continue
            bs = list(budgets)
            if hyp and system != NO_RETRIEVAL:
                bs += [b for b in sweep if b not in bs]
            plan.extend((bench, system, b) for b in bs)
    return plan


def write_table1(out: Path, benchmarks: List[str], queries: Dict[str, List[Any]], envs: EnvironmentCache) -> None:
    """Table 1: queries, corpus size and gold papers per query of every benchmark run."""
    sizes: Dict[str, Optional[int]] = {}
    for bench in benchmarks:
        try:
            sizes[bench] = len(envs.papers_for(bench))
        except (FileNotFoundError, OSError, ValueError) as err:
            log.warning("Table 1: corpus of %s unavailable (%s)", bench, err)
            sizes[bench] = None
    rows, sections = table1_rows({b: queries[b] for b in benchmarks}, sizes)
    if not rows:
        return
    text = markdown_table(rows, TABLE1, sections, bold_best=False)
    (out / "table1.md").write_text(text, encoding="utf-8")
    write_csv(out / "table1.csv", table_rows(rows, TABLE1, sections),
              fieldnames=["section", TABLE1.first_header] + [c.header for c in TABLE1.columns])
    print(text)


def warn_foreign_fingerprints(envs: EnvironmentCache, bench: str, system: str, budget: Optional[int],
                              records: Dict[str, RunRecord]) -> None:
    """``--eval-only``: warn when cached answers were produced under another configuration than ``--config``.

    The judges, the organization embedder and the corpus come from the
    current configuration; evaluating paper-setting answers under the offline
    default configuration (e.g. a forgotten ``--config``) scores them with
    the mock judges.
    """
    try:
        fp = envs.for_benchmark(bench).fingerprint(system)
    except FileNotFoundError:
        return
    foreign = sorted({str(r.fingerprint) for r in records.values() if r.fingerprint != fp})
    if foreign:
        log.warning("%s / %s / %s: the cached answers were produced under another configuration (fingerprint %s, "
                    "current %s); pass the --config / --set options of the run that produced them",
                    bench, system_display(system), budget_tag(budget), ", ".join(foreign), fp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_output="results/run_qa_eval")
    add_data_args(ap)
    add_system_args(ap)
    ap.add_argument("--systems", default=None, help=f"budgeted systems (default: {','.join(TABLE4_SYSTEMS)})")
    ap.add_argument("--budgets", default=str(PAPER_BUDGET), help="budgets of Tables 3-4 (default 8192)")
    ap.add_argument("--table-budget", type=int, default=PAPER_BUDGET,
                    help="budget reported in Tables 3-4 (default 8,192; the largest budget run if absent)")
    ap.add_argument("--sweep-budgets", default=",".join(str(b) for b in SWEEP_BUDGETS),
                    help="Fig. 6 budgets for hypothesis benchmarks ('none' to skip)")
    ap.add_argument("--answers-dir", type=Path, action="append", default=[],
                    help="evaluate saved answers of unbudgeted agents (output of run_agents.py; repeatable); "
                         "implies --eval-only unless --systems is given")
    ap.add_argument("--eval-only", action="store_true", help="evaluate cached answers only (no generation)")
    ap.add_argument("--skip-eval", action="store_true", help="generate answers only")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    rt = Runtime.from_args(args)
    out: Path = rt.output_dir or Path("results/run_qa_eval")
    out.mkdir(parents=True, exist_ok=True)
    layout = DataLayout(args.data_dir)
    benchmarks = resolve_benchmarks(args.benchmarks, layout.available_benchmarks(),
                                    default=EVIDENCE_BENCHMARKS + HYPOTHESIS_BENCHMARKS)
    if not benchmarks:
        ap.error(f"No benchmark found under {args.data_dir}")
    systems = resolve_systems(args.systems, TABLE4_SYSTEMS)
    agents = [s for s in systems if SYSTEMS[s].family == "agent"]
    if agents:
        ap.error(f"Agents {agents} are run with run_agents.py; pass their answers with --answers-dir")
    budgets = parse_budgets(args.budgets)
    sweep = parse_budgets(args.sweep_budgets)
    if not budgets:
        ap.error("--budgets is empty")
    table_budget = args.table_budget if args.table_budget in budgets else max(budgets)
    eval_only = bool(args.eval_only)
    if args.answers_dir and args.systems is None and not eval_only:
        # Evaluating agents must never regenerate (and replace) the expensive budgeted answers.
        log.info("--answers-dir without --systems: evaluating cached answers only (implies --eval-only)")
        eval_only = True

    options = system_options_from_args(args)
    envs = EnvironmentCache(rt, layout, index_store_from_args(args), options, args.link_predictor)
    store = RecordStore(out / "answers")
    queries = {b: layout.load_queries(b, args.limit) for b in benchmarks}
    plan = plan_runs(benchmarks, systems, budgets, sweep)
    log.info("Benchmarks: %s; %d (benchmark, system, budget) runs", {b: len(q) for b, q in queries.items()}, len(plan))
    write_table1(out, benchmarks, queries, envs)

    # ---------------------------------------------------------------- generation
    records: Dict[Tuple[str, str, Optional[int]], Dict[str, RunRecord]] = {}
    if eval_only:
        for bench, system, budget in plan:
            recs = store.load_latest(store.path(bench, system, budget))
            if not recs:
                log.warning("--eval-only: no cached answers for %s / %s / %s (not evaluated)", bench,
                            system_display(system), budget_tag(budget))
                continue
            warn_foreign_fingerprints(envs, bench, system, budget, recs)
            records[(bench, system, budget)] = recs
    else:
        try:
            check_cache_conflicts(store, [(b, s, bud, envs.for_benchmark(b).fingerprint(s)) for b, s, bud in plan],
                                  resume=args.resume, force=args.force)
        except CacheConflictError as err:
            log.error("%s", err)
            return 2
        for bench, system, budget in plan:
            env = envs.for_benchmark(bench)
            records[(bench, system, budget)] = run_queries(env, system, queries[bench], budget, benchmark=bench,
                                                           mode="qa", store=store, resume=args.resume,
                                                           fail_fast=args.fail_fast, force=args.force)
    eff_rows = [dict(r.efficiency, system=s, benchmark=b, budget=bud, query_id=qid)
                for (b, s, bud), recs in records.items() for qid, r in recs.items() if r.efficiency is not None]
    write_jsonl(out / "efficiency.jsonl", eff_rows)
    agent_records: Dict[Tuple[str, str, Optional[int]], Dict[str, RunRecord]] = {}
    for st in answer_stores(args.answers_dir):
        for bench in benchmarks:
            for system in st.systems(bench):
                recs = st.load_latest(st.path(bench, system, None))
                if recs:
                    agent_records[(bench, system, None)] = {q.query_id: recs[q.query_id] for q in queries[bench]
                                                            if q.query_id in recs}
    if args.skip_eval:
        print(f"Answers written to {out / 'answers'} (evaluation skipped)")
        return 0

    # ---------------------------------------------------------------- evaluation
    judges = build_judges(rt.config.evaluation, seed=rt.config.seed)
    org_embedder = None
    evals: Dict[Tuple[str, str, Optional[int]], BenchmarkEval] = {}
    for key, recs in list(records.items()) + list(agent_records.items()):
        bench, system, budget = key
        spec = BENCHMARKS[bench]
        if spec.is_organization and org_embedder is None:
            org_embedder = organization_embedder(rt)
        ev = evaluate_benchmark(bench, answers_of(recs), queries[bench], judges, papers=envs.papers_for(bench),
                                embedder=org_embedder, layout=layout,
                                discovery_k=rt.config.evaluation.discovery_k, system=system, budget=budget)
        ev.summary["n_errors"] = float(count_errors(recs))
        evals[key] = ev
        write_json(out / "eval" / bench / f"{system}_{budget_tag(budget)}.json", ev.to_dict())

    # -------------------------------------------------------------------- tables
    budgeted = [s for s in systems if any(k[1] == s for k in evals)]
    unbudgeted = order_systems(sorted({k[1] for k in agent_records}), TABLE34_AGENT_ORDER)
    sections = budget_sections([system_display(s) for s in budgeted],
                               [system_display(s) for s in unbudgeted], table_budget)
    evidence = [b for b in benchmarks if not BENCHMARKS[b].is_hypothesis]
    hypothesis = [b for b in benchmarks if BENCHMARKS[b].is_hypothesis]
    summary: Dict[str, Any] = {"environment": environment_summary(rt, options), "table_budget": table_budget,
                               "benchmarks": {b: len(q) for b, q in queries.items()}, "results": {}}
    for (bench, system, budget), ev in evals.items():
        by_system = summary["results"].setdefault(bench, {}).setdefault(system_display(system), {})
        by_system[budget_tag(budget)] = ev.summary

    def pick(system: str, bench: str) -> Optional[BenchmarkEval]:
        b = None if system in unbudgeted and (bench, system, None) in evals else table_budget
        return evals.get((bench, system, b))

    rows3: Dict[str, Dict[str, Any]] = {}
    groups: List[str] = []
    for system in budgeted + unbudgeted:
        per = {b: pick(system, b) for b in evidence}
        per = {b: e for b, e in per.items() if e is not None}
        if per:
            row = rows3[system_display(system)] = qa_table_row(per)
            row["n_failed"], _, row["Failed"] = unanswered_cell(per.values())
            groups.extend(BENCHMARKS[b].group for b in per)
    if rows3:
        spec3 = with_failed_column(table3_spec(groups), rows3)
        write_table(out / "table3.md", rows3, spec3, sections)
        print(markdown_table(rows3, spec3, sections))
    rows4: Dict[str, Dict[str, Any]] = {}
    sweep_rows: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for bench in hypothesis:
        for system in budgeted + unbudgeted:
            ev = pick(system, bench)
            if ev is not None:
                row = rows4[system_display(system)] = dict(ev.summary)
                row["n_failed"], _, row["Failed"] = unanswered_cell([ev])
        for (b, system, budget), ev in evals.items():
            if b == bench and budget is not None and system != NO_RETRIEVAL:
                sweep_rows.setdefault(system_display(system), {})[budget] = {"Disc": ev.summary.get("Hit@10")}
    if rows4:
        spec4 = with_failed_column(TABLE4, rows4)
        write_table(out / "table4.md", rows4, spec4, sections)
        print(markdown_table(rows4, spec4, sections))
    if sweep_rows and any(len(v) > 1 for v in sweep_rows.values()):
        write_budget_sweep(out / "fig6.csv", sweep_rows, metrics=["Disc"])
        if not args.no_plots and matplotlib_available():
            names = [n for n in (system_display(s) for s in budgeted) if n in sweep_rows][:8]
            plot_budget_sweep(sweep_rows, "Disc", out / "fig6_disc.png", ylabel="Disc (%)",
                              title="Fig. 6: hypothesis QA Disc vs. B", systems=names)
        print("Fig. 6 (Disc, %): " + "; ".join(
            f"{n}: " + ", ".join(f"{b}={v['Disc']:.1f}" for b, v in sorted(by_b.items()) if v["Disc"] is not None)
            for n, by_b in sweep_rows.items()))
    write_json(out / "judge_usage.json", judges.usage())
    write_json(out / "summary.json", summary)
    n_err = sum(count_errors(r) for r in list(records.values()) + list(agent_records.values()))
    if n_err:
        print(f"Warning: {n_err} failed queries (no answer, scored 0; 'Failed' column of Tables 3-4); see the log")
    print(f"Results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
