#!/usr/bin/env python
"""Train the cross-taxonomy link predictor s_theta of an index (Sec. 4.3, Eq. 8; App. B.5, Eq. 17-18).

For every cutoff t of ``config.link_prediction.cutoffs`` and horizon
Delta = ``horizon_years``, the historical index at t supplies problem-method
pairs that share no paper at t; papers of the window (t, t + Delta] generate
candidate application labels, which the LLM verifies (``verify_with_llm``;
Eq. 17). The encoders g_P, g_M are trained with the pairwise ranking loss of
Eq. (18). The last cutoff is held out for validation (filtered MRR / Hit@k /
AUC of the window's positives against the untrained description prior), then
the model is refitted on every cutoff and saved next to the index
(``<index>/link_predictor/``), where run_qa_eval.py / ask.py pick it up.

For the held-out hypothesis evaluation, train on the REDUCED corpus index
(``--corpus heldout``): every cutoff and horizon must lie within that corpus
(App. B.5), so no held-out target can supply a label; windows extending past
the corpus are reported (``--strict`` makes them an error).

Paper setting::

    python scripts/train_link_predictor.py --config configs/paper.yaml --corpus heldout
    python scripts/train_link_predictor.py --config configs/paper.yaml --corpus arxiv_subfields

Quick offline run (synthetic data, mock LLM verification)::

    python scripts/train_link_predictor.py --index-dir artifacts/index --corpus arxiv_subfields \\
        --set link_prediction.cutoffs=[2018-01-01,2019-01-01,2020-01-01,2021-01-01] \\
        --set link_prediction.horizon_years=1 --set link_prediction.epochs=5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.data.corpus import write_json  # noqa: E402
from scholarpath.experiments.qa import DEFAULT_INDEX_ROOT, LINK_PREDICTOR_DIR, split_csv  # noqa: E402
from scholarpath.index.semantic_index import SemanticIndex  # noqa: E402
from scholarpath.llm import sum_usage  # noqa: E402
from scholarpath.retrieval.link_training import train_link_predictor  # noqa: E402
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402

log = logging.getLogger("train_link_predictor")


def _concepts(spec: str) -> List[str]:
    """Comma list, or ``@file`` with one concept id per line (or a JSON list)."""
    if not spec:
        return []
    if spec.startswith("@"):
        text = Path(spec[1:]).read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            return [str(x) for x in data]
        except json.JSONDecodeError:
            return [line.strip() for line in text.splitlines() if line.strip()]
    return split_csv(spec)


def _fmt(metrics: Dict[str, Any]) -> str:
    keys = ("mrr", "hit@1", "hit@3", "hit@10", "auc")
    return ", ".join(f"{k.upper()} {metrics[k]:.3f}" for k in keys if isinstance(metrics.get(k), float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_output="results/train_link_predictor")
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_ROOT, help="index root")
    ap.add_argument("--corpus", default="arxiv_subfields", help="index name under --index-dir")
    ap.add_argument("--index-path", type=Path, default=None, help="explicit index directory (overrides --corpus)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <index>/link_predictor)")
    ap.add_argument("--exclude-concepts", default="",
                    help="concept ids never used for training (comma list or @file), e.g. held-out endpoints")
    ap.add_argument("--no-validate", action="store_true", help="skip the held-out-cutoff validation")
    ap.add_argument("--no-refit", action="store_true", help="return the validation model instead of refitting")
    ap.add_argument("--strict", action="store_true", help="error if a window exceeds the corpus or no supervision")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hard-fraction", type=float, default=0.5, help="share of hard (related) comparison negatives")
    ap.add_argument("--max-papers-per-pair", type=int, default=3, help="window papers verified per candidate pair")
    args = ap.parse_args(argv)

    rt = Runtime.from_args(args)
    index_dir = args.index_path or (args.index_dir / args.corpus)
    index = SemanticIndex.load(index_dir)
    out = args.out or (index_dir / LINK_PREDICTOR_DIR)
    cfg = rt.config.link_prediction
    log.info("Training the link predictor on %s (%d papers; cutoffs %s, horizon %.1f y, LLM verification %s)",
             index_dir, len(index.papers), list(cfg.cutoffs), cfg.horizon_years, cfg.verify_with_llm)
    before = rt.meter.snapshot()
    model = train_link_predictor(
        index, cfg, rt.llm, exclude_concepts=_concepts(args.exclude_concepts),
        validate=not args.no_validate, refit=not args.no_refit, strict=args.strict,
        batch_size=args.batch_size, hard_fraction=args.hard_fraction, max_papers_per_pair=args.max_papers_per_pair,
    )
    model.save(out)
    usage = rt.meter.diff(before)
    report: Dict[str, Any] = dict(getattr(model, "report", {}) or {})
    report["model"] = type(model).__name__
    report["index_dir"] = str(index_dir)
    report["output_dir"] = str(out)
    report["verification_usage"] = {k: v.to_dict() for k, v in usage.items()}
    report["verification_cost_usd"] = sum_usage(usage).cost_usd
    write_json(Path(out) / "training_report.json", report)
    if rt.output_dir is not None:
        rt.write_json(f"{Path(index_dir).name}_training_report.json", report)

    print(f"Link predictor ({report['model']}) saved to {out}")
    for s in report.get("supervision", []) or []:
        print(f"  cutoff {s.get('cutoff')}: {s.get('positives', 0)} positive pairs "
              f"({s.get('candidates', 0)} candidates, {s.get('rejected', 0)} rejected, verified={s.get('verified')})")
    final = report.get("final") or {}
    if final:
        print(f"  training pairs: {final.get('positive_pairs')} (cutoffs {final.get('train_cutoffs')})")
    val = report.get("validation") or {}
    if val.get("skipped"):
        print(f"  validation skipped: {val['skipped']}")
    elif val:
        print(f"  validation at {val.get('cutoff')} ({val.get('pairs')} pairs, trained on {val.get('train_cutoffs')}):")
        for direction in ("method_ranking", "problem_ranking"):
            if direction in val:
                prior = (val.get("embedding_prior") or {}).get(direction, {})
                print(f"    {direction}: {_fmt(val[direction])}  | untrained prior: {_fmt(prior)}")
    if report.get("fallback"):
        print(f"  no temporal supervision ({report['fallback']}): saved the untrained description prior")
    if report.get("windows_exceeding_corpus"):
        print(f"  warning: windows exceeding the corpus at cutoffs {report['windows_exceeding_corpus']}")
    print(f"  verification cost: ${report['verification_cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
