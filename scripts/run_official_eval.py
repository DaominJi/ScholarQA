#!/usr/bin/env python
"""Table 3 with the official evaluation scripts of ResearchQA and ScholarQABench (App. C.3).

App. C.3: "ResearchQA rubric coverage and ScholarQA-CS correctness and
citation F1 use the official evaluation scripts of their benchmarks, while
ScholarQA-Multi is scored by Prometheus-v2-7B". ``run_qa_eval.py`` computes
Table 3 with faithful in-process re-implementations of those protocols
(``scholarpath.evaluation.qa_metrics``), which run with any judge backend.
This script scores the same saved answers with the official scripts
themselves, through ``scholarpath.evaluation.official``:

* ResearchQA -- ``compute_coverage.py`` (Rub; gpt-4.1-mini). The official
  repository has no citation metric; Cite is the citation F1 against the
  gold papers, as in ``run_qa_eval.py``.
* ScholarQA-CS -- ``scripts/rubric_eval.py --rubrics --snippets`` (Corr;
  gpt-4-turbo through litellm) and ``scripts/citation_correctness_eval.py
  --citations`` (Cite; AttrScore-Flan-T5-XL).
* ScholarQA-Multi -- ``scripts/prometheus_eval.py`` (Corr; Prometheus-2 7B
  through vLLM, greedy decoding) and ``citation_correctness_eval.py`` (Cite).

Requirements: local checkouts of github.com/realliyifei/ResearchQA and
github.com/AkariAsai/ScholarQABench (``evaluation.researchqa_dir`` /
``evaluation.scholarqabench_dir``, ``--researchqa-dir`` /
``--scholarqabench-dir``, or ``$SCHOLARPATH_RESEARCHQA_DIR`` /
``$SCHOLARPATH_SCHOLARQABENCH_DIR``); a Python environment with their
requirements (``$SCHOLARPATH_EVAL_PYTHON``, default: this interpreter);
``$OPENAI_API_KEY`` for their judges. ResearchQA's ``compute_coverage.py``
calls the pre-1.0 ``openai.ChatCompletion`` API; pass ``--coverage-script``
with the openai>=1.0 port of rlresearch/dr-tulu
(``agent/evaluation/research_qa_eval/compute_coverage.py``) or patch it.

Failed queries (no answer in the cache) count as 0, as in ``run_qa_eval.py``
(``--missing skip`` reports the official numbers over the answered queries).
ScholarQABench combines CS and Multi weighted by their numbers of queries.

Outputs (``--output-dir``, default ``results/run_official_eval``)::

    <benchmark>/<system>/<tag>/...        official inputs, raw outputs, official_summary.json
    official_rows.csv                     one row per (benchmark, system, budget)
    official.md / official.csv            Rub / Cite (ResearchQA), Corr / Cite (ScholarQABench) per system
    summary.json

Paper setting (answers of run_qa_eval.py and run_agents.py)::

    python scripts/run_official_eval.py --config configs/paper.yaml \\
        --answers-dir results/run_qa_eval/answers,results/run_agents/answers \\
        --researchqa-dir ../ResearchQA --scholarqabench-dir ../ScholarQABench
    # add --budget 8192 to score only the runs at B = 8,192 (this drops the unbudgeted agents)

Dry run (export the official input files only, no script is executed)::

    python scripts/run_official_eval.py --data-dir data/synthetic --answers-dir results/run_qa_eval/answers \\
        --benchmarks researchqa --export-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.evaluation.official import (  # noqa: E402
    OFFICIAL_BENCHMARKS,
    OfficialEvalError,
    OfficialPaths,
    export_citation_eval,
    export_prometheus,
    export_researchqa_data,
    export_researchqa_responses,
    export_scholarqa_cs,
    run_official_evaluation,
)
from scholarpath.evaluation.qa_metrics import MISSING_POLICIES, scholarqabench_combined  # noqa: E402
from scholarpath.evaluation.reporting import Column, TableSpec, write_csv, write_json, write_table  # noqa: E402
from scholarpath.experiments.qa import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DataLayout,
    RecordStore,
    answers_of,
    budget_tag,
)
from scholarpath.runtime import add_common_args, config_from_args, setup_logging  # noqa: E402

log = logging.getLogger("run_official_eval")

OFFICIAL_TABLE = TableSpec(
    "official", "Table 3 (ResearchQA / ScholarQABench columns) with the official evaluation scripts (%).",
    [Column("researchqa.Rub", "Rub", "ResearchQA"), Column("researchqa.Cite", "Cite", "ResearchQA"),
     Column("scholarqabench.Corr", "Corr", "ScholarQA"), Column("scholarqabench.Cite", "Cite", "ScholarQA")],
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--answers-dir", default="results/run_qa_eval/answers",
                   help="comma-separated answer caches (<benchmark>/<system>/<B|unbudgeted>.jsonl)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="processed data layout (queries)")
    p.add_argument("--benchmarks", default=",".join(OFFICIAL_BENCHMARKS),
                   help=f"subset of {','.join(OFFICIAL_BENCHMARKS)}")
    p.add_argument("--systems", default=None, help="comma-separated systems (default: every cached system)")
    p.add_argument("--budget", default=None,
                   help="only this budget (e.g. 8192, or 'unbudgeted'); default: every cached budget")
    p.add_argument("--researchqa-dir", default=None, help="ResearchQA checkout (overrides the config)")
    p.add_argument("--scholarqabench-dir", default=None, help="ScholarQABench checkout (overrides the config)")
    p.add_argument("--eval-python", default=None, help="interpreter with the official requirements")
    p.add_argument("--coverage-script", type=Path, default=None,
                   help="compute_coverage.py to run (e.g. the dr-tulu openai>=1.0 port)")
    p.add_argument("--no-citations", action="store_true", help="skip citation_correctness_eval.py")
    p.add_argument("--no-vllm", action="store_true", help="run prometheus_eval.py without --load_vllm")
    p.add_argument("--missing", choices=MISSING_POLICIES, default="zero",
                   help="failed queries count as 0 (default) or are skipped as by the official scripts")
    p.add_argument("--timeout", type=float, default=None, help="timeout of each official script (seconds)")
    p.add_argument("--export-only", action="store_true", help="write the official input files and stop")
    add_common_args(p, default_output="results/run_official_eval")
    return p.parse_args(argv)


def _budget_filter(value: Optional[str]) -> Any:
    if value is None:
        return "all"
    if value.strip().lower() == "unbudgeted":
        return None
    return int(value)


def export_inputs(benchmark: str, answers: Dict[str, Any], queries: List[Any], out: Path, system: str) -> None:
    """The official input files of one run, without executing any script."""
    answered = [q for q in queries if q.query_id in answers]
    if benchmark == "researchqa":
        export_researchqa_data(queries, out / "researchqa_data.json")
        export_researchqa_responses(answers, out / "responses.json", queries=answered)
    elif benchmark == "scholarqa_cs":
        export_scholarqa_cs(answers, out / "qa" / f"{system}.jsonl", queries=answered)
        export_citation_eval(answers, answered, out / "cs_citations.json")
    else:
        export_prometheus(answers, answered, out / "prometheus_preds.json", gold_path=out / "prometheus_gold.json")
        export_citation_eval(answers, answered, out / "multi_citations.json")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    config = config_from_args(args)
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    unknown = [b for b in benchmarks if b not in OFFICIAL_BENCHMARKS]
    if unknown:
        print(f"run_official_eval.py: error: no official script for {unknown}; "
              f"choose from {','.join(OFFICIAL_BENCHMARKS)}", file=sys.stderr)
        return 2
    systems = {s.strip() for s in args.systems.split(",")} if args.systems else None
    budget = _budget_filter(args.budget)
    layout = DataLayout(args.data_dir)
    paths = OfficialPaths.from_config(config.evaluation, researchqa_dir=args.researchqa_dir,
                                      scholarqabench_dir=args.scholarqabench_dir, python=args.eval_python)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for root in [r.strip() for r in str(args.answers_dir).split(",") if r.strip()]:
        for bench, system, b, path in RecordStore(root).entries():
            if bench in benchmarks and (systems is None or system in systems) and (budget == "all" or b == budget):
                runs.append((bench, system, b, path))
    if not runs:
        print(f"run_official_eval.py: error: no cached answers of {benchmarks} under {args.answers_dir}",
              file=sys.stderr)
        return 2

    queries_of = {}
    rows: List[Dict[str, Any]] = []
    failures = 0
    for bench, system, b, path in runs:
        if bench not in queries_of:
            queries_of[bench] = layout.load_queries(bench)
        queries = queries_of[bench]
        answers = answers_of(RecordStore.load_latest(path))
        out = out_root / bench / system / budget_tag(b)
        log.info("%s / %s / %s: %d of %d queries answered", bench, system, budget_tag(b), len(answers), len(queries))
        if args.export_only:
            export_inputs(bench, answers, queries, out, system)
            continue
        try:
            summary = run_official_evaluation(
                bench, answers, queries, out, paths, system=system,
                researchqa_model=config.evaluation.researchqa_judge_model, coverage_script=args.coverage_script,
                prometheus_model=config.evaluation.prometheus_model, load_vllm=not args.no_vllm,
                citations=not args.no_citations, missing=args.missing, timeout=args.timeout)
        except OfficialEvalError as err:
            failures += 1
            log.error("%s / %s / %s failed: %s", bench, system, budget_tag(b), err)
            continue
        rows.append({"benchmark": bench, "system": system, "budget": budget_tag(b),
                     **{k: v for k, v in summary.items() if not isinstance(v, dict) and k not in ("benchmark", "system")}})

    if args.export_only:
        print(f"Official input files written under {out_root}")
        return 0
    table: Dict[str, Dict[str, Any]] = {}
    for system, tag in sorted({(r["system"], r["budget"]) for r in rows}):
        of = {r["benchmark"]: r for r in rows if r["system"] == system and r["budget"] == tag}
        rqa = of.get("researchqa", {})
        sqa = scholarqabench_combined(of.get("scholarqa_cs", {}), of.get("scholarqa_multi", {}))
        table[f"{system} ({tag})"] = {"researchqa.Rub": rqa.get("Rub"), "researchqa.Cite": rqa.get("Cite"),
                                      "scholarqabench.Corr": sqa.get("Corr"), "scholarqabench.Cite": sqa.get("Cite")}
    write_csv(out_root / "official_rows.csv", rows)
    write_table(out_root / "official.md", table, OFFICIAL_TABLE)
    write_json(out_root / "summary.json", {"rows": rows, "table": table, "failures": failures,
                                           "missing_policy": args.missing})
    print((out_root / "official.md").read_text(encoding="utf-8"))
    print(f"Results written to {out_root}")
    return 1 if failures or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
