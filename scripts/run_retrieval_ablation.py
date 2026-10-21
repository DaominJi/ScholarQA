#!/usr/bin/env python
"""Table 7: ablation of evidence-path retrieval on the held-out hypothesis queries (App. C.4).

ScholarPath at B = 8,192 against four variants, each answering every
held-out query with the same single composition call:

* w/o temporal link prediction   -- the temporally trained link predictor s_theta
  (Eq. 8, App. B.5) is replaced by the untrained text-similarity prior
  (EmbeddingLinkPredictor: cosine of concept descriptions); candidates are
  still scored ("structural analogies over text similarity", App. C.4);
* w/o bound-guided search        -- ``use_bound_guided_search=False``: the exhaustive
  greedy scan; selects exactly the same units (Proposition 7) but scores every
  candidate in scope. Cost note: the current scan
  (``scholarpath.retrieval.search.ExhaustiveGreedy``) materializes every
  complementary (known-side x candidate-side witness) path pair up front and
  rescans them at each selection, so its time grows roughly quadratically
  with the corpus (about 0.07 s vs. 1 s at 120 papers and 0.6 s vs. 93 s at
  960 papers per query for the bound-guided vs. exhaustive search, offline
  configuration); on the 148K-paper held-out corpus run this row on a
  ``--limit``-ed query set or expect long run times;
* w/o role-aware granularity     -- ``use_role_aware_granularity=False``: gamma = 1 for
  every level and role (Sec. 4.4);
* w/o shared-cost reuse update   -- ``use_shared_cost_update=False``: the marginal cost of
  a unit is its standalone cost, ignoring nodes already in V(S) (Eq. 6).

``--variants ablations`` adds ``scholarpath_no_link_prior`` (not a paper row):
``use_link_prediction=False``, i.e. no link prior at all (eta_hat = phi).

Columns: Hit@10 (Disc), the three-judge average (Judge Avg., 0-100), the mean
context size in thousands of tokens (Context (K)), and the mean number of
scored candidate nodes (Scored: candidate concepts whose exact eta_hat the
search computed, ``RetrievalResult.stats["scored_candidates"]``; the paper's
312 of 4,628). summary.json also lists the candidate pool in scope and the
index nodes scored (``scored_nodes``).

The full model needs the link predictor trained on the reduced held-out
corpus (``train_link_predictor.py --corpus heldout``); without it the
"ScholarPath" row would itself be the "w/o temporal link prediction" variant,
so the script stops unless ``--allow-untrained-prior`` is given.

Outputs (``--output-dir``, default ``results/run_retrieval_ablation``)::

    answers/<benchmark>/<variant>/B<budget>.jsonl, eval/<benchmark>/<variant>.json
    table7.md / table7.csv, summary.json

Paper setting (held-out index + link predictor trained on the reduced corpus)::

    python scripts/run_retrieval_ablation.py --config configs/paper.yaml \\
        --set evaluation.judge_provider=openai --resume

Quick offline run (synthetic layout; train the link predictor first or allow the untrained prior)::

    python scripts/run_retrieval_ablation.py --data-dir data/synthetic --budget 2048 --limit 2 --allow-untrained-prior
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.evaluation.reporting import markdown_table, write_json, write_table  # noqa: E402
from scholarpath.experiments.qa import (  # noqa: E402
    ABLATIONS,
    BENCHMARKS,
    HYPOTHESIS_BENCHMARKS,
    PAPER_BUDGET,
    SYSTEMS,
    TABLE7,
    TABLE7_VARIANTS,
    CacheConflictError,
    DataLayout,
    EnvironmentCache,
    RecordStore,
    add_data_args,
    add_system_args,
    answers_of,
    build_judges,
    check_cache_conflicts,
    count_errors,
    environment_summary,
    evaluate_benchmark,
    index_store_from_args,
    organization_embedder,
    resolve_benchmarks,
    resolve_systems,
    run_queries,
    system_display,
    system_options_from_args,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402

log = logging.getLogger("run_retrieval_ablation")


def _mean_stat(answers: List[Any], key: str) -> Optional[float]:
    """Mean of a ScholarPath search statistic over the answers that report it."""
    vals = [float(st[key]) for st in ((a.metadata or {}).get("retrieval_stats") or {} for a in answers) if key in st]
    return float(np.mean(vals)) if vals else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_output="results/run_retrieval_ablation")
    add_data_args(ap)
    add_system_args(ap)
    ap.add_argument("--variants", default=",".join(TABLE7_VARIANTS),
                    help=f"comma list of {list(ABLATIONS)} or 'table7' / 'ablations' (default: the Table 7 rows)")
    ap.add_argument("--budget", type=int, default=PAPER_BUDGET, help="context budget B (default 8,192)")
    ap.add_argument("--allow-untrained-prior", action="store_true",
                    help="run even if the index has no trained link predictor (then the full model and "
                         "'w/o temporal link prediction' coincide; offline runs only)")
    args = ap.parse_args(argv)

    rt = Runtime.from_args(args)
    out: Path = rt.output_dir or Path("results/run_retrieval_ablation")
    out.mkdir(parents=True, exist_ok=True)
    layout = DataLayout(args.data_dir)
    benchmarks = resolve_benchmarks(args.benchmarks, layout.available_benchmarks(), default=HYPOTHESIS_BENCHMARKS)
    if not benchmarks:
        ap.error(f"No held-out hypothesis queries under {args.data_dir} (run build_heldout_benchmark.py)")
    variants = resolve_systems(args.variants, list(TABLE7_VARIANTS))
    bad = [v for v in variants if v not in ABLATIONS]
    if bad:
        ap.error(f"Not ScholarPath variants: {bad}; expected {list(ABLATIONS)}")
    options = system_options_from_args(args)
    envs = EnvironmentCache(rt, layout, index_store_from_args(args), options, args.link_predictor)
    uses_trained = any(SYSTEMS[v].link_prior == "trained" for v in variants)
    for bench in benchmarks:
        env = envs.for_benchmark(bench)
        if uses_trained and BENCHMARKS[bench].is_hypothesis and not env.has_trained_link_predictor:
            msg = (f"index {env.name!r} has no trained link predictor: the full model would use the untrained "
                   f"text-similarity prior, i.e. 'w/o temporal link prediction'. Train it with "
                   f"scripts/train_link_predictor.py --corpus {env.name}")
            if not args.allow_untrained_prior:
                ap.error(msg + " (or pass --allow-untrained-prior)")
            log.warning(msg)
    store = RecordStore(out / "answers")
    try:  # never replace cached answers of another configuration without --resume / --force
        check_cache_conflicts(store, [(b, v, args.budget, envs.for_benchmark(b).fingerprint(v))
                                      for b in benchmarks for v in variants], resume=args.resume, force=args.force)
    except CacheConflictError as err:
        log.error("%s", err)
        return 2
    judges = build_judges(rt.config.evaluation, seed=rt.config.seed)
    org_embedder = None
    rows: Dict[str, Dict[str, Any]] = {}
    summary: Dict[str, Any] = {"environment": environment_summary(rt, options), "budget": args.budget, "variants": {}}
    for variant in variants:
        recs_all = {}
        evals = []
        for bench in benchmarks:
            queries = layout.load_queries(bench, args.limit)
            env = envs.for_benchmark(bench)
            recs = run_queries(env, variant, queries, args.budget, benchmark=bench, mode="qa", store=store,
                               resume=args.resume, fail_fast=args.fail_fast, force=args.force)
            if BENCHMARKS[bench].is_organization and org_embedder is None:
                org_embedder = organization_embedder(rt)
            ev = evaluate_benchmark(bench, answers_of(recs), queries, judges, papers=env.index.papers,
                                    embedder=org_embedder, layout=layout,
                                    discovery_k=rt.config.evaluation.discovery_k, system=variant, budget=args.budget)
            write_json(out / "eval" / bench / f"{variant}.json", ev.to_dict())
            evals.append((len(queries), ev))
            recs_all.update({f"{bench}/{k}": v for k, v in recs.items()})
        answers = [r.answer for r in recs_all.values() if r.answer is not None and r.error is None]
        hyp_answers = [a for a in answers if ((a.metadata or {}).get("retrieval_stats") or {}).get("candidate_pool")]
        n_total = sum(n for n, _ in evals)

        def weighted(key: str) -> Any:
            vals = [(n, ev.summary.get(key)) for n, ev in evals if ev.summary.get(key) is not None]
            tot = sum(n for n, _ in vals)
            return sum(n * float(v) for n, v in vals) / tot if tot else None

        row = {
            "Hit@10": weighted("Hit@10"),
            "Judge Avg.": weighted("Avg."),
            "Context (K)": float(np.mean([a.context_tokens for a in answers])) / 1000.0 if answers else None,
            # Table 7 "Scored": candidate concepts whose exact eta_hat was computed (hypothesis queries).
            "Scored": _mean_stat(hyp_answers, "scored_candidates"),
        }
        name = system_display(variant)
        rows[name] = row
        summary["variants"][name] = {
            **row, "queries": n_total, "errors": count_errors(recs_all),
            "candidate_pool": _mean_stat(hyp_answers, "candidate_pool"),
            "link_evaluations": _mean_stat(hyp_answers, "link_evaluations"),
            "scored_nodes": _mean_stat(answers, "scored_nodes"),
            "link_prior": SYSTEMS[variant].link_prior,
            "per_benchmark": {ev.benchmark: ev.summary for _, ev in evals},
        }
    write_table(out / "table7.md", rows, TABLE7)
    write_json(out / "summary.json", summary)
    write_json(out / "judge_usage.json", judges.usage())
    print(markdown_table(rows, TABLE7))
    print(f"Results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
