#!/usr/bin/env python
"""Build the Held-Out Hypothesis benchmark of Sec. 5.3 / App. C.1 (Table 1: 47 queries = 24 + 23).

Sub-command ``build`` runs App. C.1 up to the annotation stage:

1. **Reference dual taxonomy over the COMPLETE corpus** (before any removal):
   built with ScholarPath's builder (Algorithm 1), or given as prebuilt
   taxonomies (``--problem-taxonomy`` / ``--method-taxonomy``) or the
   taxonomies of an existing index (``--reference-index``).
2. **Targets** (``select_targets``): emerging cross-domain ideas whose
   source papers all postdate the answer LLM's knowledge cutoff
   (``--min-source-date``, the cutoff date of the backbone) -- a *new*
   endpoint (a concept with at most ``--max-extent`` = 4 papers) combined with
   an *existing* endpoint of another domain (primary arXiv category, or
   ``--domain-map``): ``--n-new-problem`` 24 "new problem / existing method"
   and ``--n-new-method`` 23 "new method / existing problem" targets with
   pairwise disjoint supporting papers and no shared endpoint;
   ``--oversample`` samples extra targets so that the post-validation counts
   can reach the quotas; ``--seed`` fixes the sampling order.
3. **Corpus reduction** (``reduce_corpus``): every supporting paper and the
   papers published after them are removed before any index is built.
4. **Queries and expected answers** (``generate_queries``, one LLM call per
   target from the source paper's title and abstract) with the automatic
   leakage screen, and the supporting evidence left in the reduced corpus.
5. The two-annotator validation sheet (``annotation_sheet.csv``).

Outputs in ``--out`` (default ``<data-dir>/heldout``): ``queries.jsonl`` (all
candidate queries until ``apply``), ``candidates.jsonl``,
``removed_papers.json``, ``reduced_corpus_ids.json``, ``targets.json``,
``stats.json``, ``annotation_sheet.csv``, and the reduced corpus
``papers.jsonl.gz`` + ``chunks.jsonl.gz`` -- index it next with
``build_index.py --corpus heldout`` and train the link predictor on it.

Sub-command ``apply`` keeps the queries that pass all three conditions
(no answer leakage, the source realizes the expected answer, the reduced
corpus keeps related evidence; ``final_*`` columns resolve disagreements),
trimmed to the per-setting quotas in seeded selection order, and writes them
to ``queries.jsonl`` with ``annotation_report.json`` (counts, agreement).
``--auto-accept`` fills the sheet from the automatic leakage screen instead
(smoke tests only: the paper's protocol uses two human annotators).

Paper setting (``--min-source-date``: the knowledge cutoff of the answer LLM,
DeepSeek-V4.0-Flash in the paper; App. C.1 requires every source paper to
postdate it)::

    python scripts/build_heldout_benchmark.py --config configs/paper.yaml build --oversample 1.5 \\
        --min-source-date <answer-LLM knowledge cutoff, YYYY-MM-DD>
    # ... two annotators fill data/processed/heldout/annotation_sheet.csv ...
    python scripts/build_heldout_benchmark.py apply
    python scripts/build_index.py --config configs/paper.yaml --corpus heldout

Quick offline run (synthetic layout; targets need small "emerging" concepts in
the reference taxonomies and a domain per paper, e.g. planted ones)::

    python scripts/build_heldout_benchmark.py build --data-dir data/synthetic \\
        --problem-taxonomy P.json --method-taxonomy M.json --domain-map domains.json --n-new-problem 2 --n-new-method 2
    python scripts/build_heldout_benchmark.py apply --data-dir data/synthetic --auto-accept
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.data.corpus import Corpus, load_queries, save_queries, write_json  # noqa: E402
from scholarpath.data.heldout import (  # noqa: E402
    ANNOTATORS,
    CONDITIONS,
    REDUCTIONS,
    annotation_agreement,
    apply_annotations,
    build_heldout_benchmark,
    export_annotation_sheet,
    load_heldout_benchmark,
    save_heldout_benchmark,
)
from scholarpath.experiments.qa import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DataLayout,
    domain_lookup,
    embed_papers,
    restrict_taxonomy,
    write_reduced_corpus,
)
from scholarpath.index.semantic_index import SemanticIndex  # noqa: E402
from scholarpath.llm import sum_usage  # noqa: E402
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402
from scholarpath.taxonomy.tree import Taxonomy  # noqa: E402

log = logging.getLogger("build_heldout_benchmark")


def _reference_taxonomies(args: argparse.Namespace, rt: Runtime, corpus: Corpus, out: Path):
    """The reference dual taxonomy over the complete corpus (App. C.1, "Targets")."""
    if args.problem_taxonomy is not None:
        return (restrict_taxonomy(Taxonomy.load(args.problem_taxonomy), corpus.ids),
                restrict_taxonomy(Taxonomy.load(args.method_taxonomy), corpus.ids), {"source": "prebuilt"})
    if args.reference_index is not None:
        idx = SemanticIndex.load(args.reference_index)
        missing = set(corpus.ids) - set(idx.papers)
        if missing:
            log.warning("The reference index lacks %d papers of the complete corpus", len(missing))
        return idx.problem, idx.method, {"source": str(args.reference_index)}
    from scholarpath.taxonomy.builder import build_dual_taxonomies

    log.info("Building the reference dual taxonomy over the complete corpus (%d papers)", len(corpus))
    before = rt.meter.snapshot()
    problem, method = build_dual_taxonomies(rt.llm, rt.embedder, rt.config.builder, corpus,
                                            paper_vectors=embed_papers(corpus, rt.embedder))
    problem.save(out / "reference" / "problem")
    method.save(out / "reference" / "method")
    usage = sum_usage(rt.meter.diff(before))
    return problem.taxonomy, method.taxonomy, {"source": "scholarpath", "cost_usd": usage.cost_usd,
                                               "llm_calls": usage.calls}


def cmd_build(args: argparse.Namespace, rt: Runtime) -> int:
    layout = DataLayout(args.data_dir)
    if args.papers is not None:
        corpus = Corpus.load(args.papers, chunks_path=args.chunks, name="complete")
    else:
        corpus = layout.load_corpus(args.corpus or layout.complete_corpus_name())
    out: Path = args.out or layout.path("heldout")
    out.mkdir(parents=True, exist_ok=True)
    problem, method, ref_info = _reference_taxonomies(args, rt, corpus, out)
    domains = domain_lookup(args.domain_map, corpus.papers)
    kwargs: Dict[str, Any] = {}
    if not args.min_source_date and rt.config.llm.knowledge_cutoff:
        args.min_source_date = str(rt.config.llm.knowledge_cutoff)  # default: the answer LLM's cutoff
    if args.min_source_date:
        kwargs["min_source_date"] = args.min_source_date
    else:
        log.warning("No --min-source-date and no llm.knowledge_cutoff: App. C.1 requires every target's source "
                    "paper to postdate the answer LLM's knowledge cutoff; pass the cutoff date so that "
                    "pre-cutoff ideas cannot be selected")
    bm = build_heldout_benchmark(
        corpus, (problem, method), rt.llm, primary_category_of=domains,
        n_new_problem=args.n_new_problem, n_new_method=args.n_new_method, oversample=args.oversample,
        max_extent=args.max_extent, seed=rt.config.seed, reduction=args.reduction,
        min_existing_papers=args.min_existing_papers, provide_counterpart=args.provide_counterpart,
        drop_leaking=args.drop_leaking, **kwargs,
    )
    bm.stats["reference_taxonomies"] = ref_info
    bm.stats["min_source_date"] = args.min_source_date
    bm.stats["complete_corpus"] = {"name": corpus.name, "papers": len(corpus)}
    bm.stats["validated"] = False
    paths = save_heldout_benchmark(bm, out)
    save_queries(out / "candidates.jsonl", bm.queries)
    sheet = export_annotation_sheet(bm.queries, out / "annotation_sheet.csv", corpus)
    reduced = write_reduced_corpus(corpus, bm.reduced_ids, out)
    if rt.output_dir is not None:
        rt.write_json("heldout_build_stats.json", {k: v for k, v in bm.stats.items() if k != "targets"})
    settings = Counter(q.metadata.get("setting") for q in bm.queries)
    print(json.dumps({
        "queries": len(bm.queries), "per_setting": dict(settings), "targets": bm.stats["targets_per_setting"],
        "requested": bm.stats["requested"], "removed_papers": len(bm.removed_ids),
        "reduced_papers": len(bm.reduced_ids), "auto_leaking": bm.stats["auto_leaking"],
        "reference": ref_info,
    }, indent=2, default=str))
    if not bm.queries:
        print("No target found: the reference taxonomies need concepts with at most --max-extent papers whose "
              "counterpart belongs to another domain (see scholarpath/data/heldout.py).")
    print(f"Wrote {paths['queries']}, {sheet}, {reduced['papers']}")
    print(f"Next: annotate {sheet}, then run `build_heldout_benchmark.py apply`, and index the reduced corpus "
          "with `build_index.py --corpus heldout`.")
    return 0


def auto_fill_sheet(sheet: Path, out: Path) -> Path:
    """Fill both annotators' columns from the automatic leakage screen (smoke tests only)."""
    with open(sheet, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        ok = str(row.get("auto_leak", "0")).strip() in ("0", "", "false", "False")
        for a in ANNOTATORS:
            for c in CONDITIONS:
                row[f"{a}_{c}"] = "1" if ok else "0"
        row["notes"] = "auto-accepted from the automatic leakage screen (not a human annotation)"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out


def cmd_apply(args: argparse.Namespace, rt: Runtime) -> int:
    layout = DataLayout(args.data_dir)
    out: Path = args.out or layout.path("heldout")
    bm = load_heldout_benchmark(out)
    cand_path = out / "candidates.jsonl"
    candidates = load_queries(cand_path) if cand_path.exists() else bm.queries
    sheet: Path = args.sheet or (out / "annotation_sheet.csv")
    if args.auto_accept:
        sheet = auto_fill_sheet(sheet, out / "annotation_sheet.auto.csv")
        log.warning("Auto-accepting queries from the leakage screen (%s); the paper's protocol uses two annotators",
                    sheet)
    quota = None if args.no_quota else (bm.stats.get("requested") or None)
    kept = apply_annotations(candidates, sheet, quota=quota)
    save_queries(out / "queries.jsonl", kept)
    report: Dict[str, Any] = {
        "sheet": str(sheet), "auto_accepted": bool(args.auto_accept), "candidates": len(candidates),
        "kept": len(kept), "per_setting": dict(Counter(q.metadata.get("setting") for q in kept)), "quota": quota,
        "agreement": annotation_agreement(sheet),
    }
    write_json(out / "annotation_report.json", report)
    stats_path = out / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats.update(validated=True, validated_queries=len(kept), annotation=report)
        write_json(stats_path, stats)
    print(json.dumps(report, indent=2, default=str))
    print(f"Wrote {len(kept)} validated queries to {out / 'queries.jsonl'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_output="results/build_heldout_benchmark")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="processed data root")
    ap.add_argument("--out", type=Path, default=None, help="benchmark directory (default: <data-dir>/heldout)")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="targets -> reduced corpus -> queries -> annotation sheet")
    b.add_argument("--corpus", default=None,
                   help="complete corpus of the layout (default: arxiv_subfields_full if present, "
                        "else arxiv_subfields)")
    b.add_argument("--papers", type=Path, default=None, help="explicit complete corpus (papers JSONL)")
    b.add_argument("--chunks", type=Path, default=None, help="section chunks of --papers")
    b.add_argument("--problem-taxonomy", type=Path, default=None, help="prebuilt reference problem taxonomy")
    b.add_argument("--method-taxonomy", type=Path, default=None, help="prebuilt reference method taxonomy")
    b.add_argument("--reference-index", type=Path, default=None,
                   help="use the taxonomies of an index built over the complete corpus")
    b.add_argument("--domain-map", default=None,
                   help="paper domains: JSON {paper_id: domain} or metadata:<key> (default: primary category)")
    b.add_argument("--n-new-problem", type=int, default=24)
    b.add_argument("--n-new-method", type=int, default=23)
    b.add_argument("--oversample", type=float, default=1.0, help=">= 1: extra targets for annotator rejections")
    b.add_argument("--max-extent", type=int, default=4, help="a new endpoint holds at most this many papers")
    b.add_argument("--min-existing-papers", type=int, default=2,
                   help="earlier papers of the existing endpoint outside the target domain")
    b.add_argument("--reduction", choices=REDUCTIONS, default="global")
    b.add_argument("--min-source-date", default=None,
                   help="knowledge cutoff of the answer LLM (YYYY-MM-DD): only targets whose source paper is "
                        "on/after it (App. C.1); warns when absent")
    b.add_argument("--provide-counterpart", action="store_true",
                   help="give the counterpart to the query-writing LLM (otherwise it extracts it)")
    b.add_argument("--drop-leaking", action="store_true", help="drop queries flagged by the leakage screen")
    a = sub.add_parser("apply", help="keep the validated queries of the filled annotation sheet")
    a.add_argument("--sheet", type=Path, default=None, help="filled sheet (default: <out>/annotation_sheet.csv)")
    a.add_argument("--no-quota", action="store_true", help="keep every validated query (no 24 / 23 trimming)")
    a.add_argument("--auto-accept", action="store_true",
                   help="fill the sheet from the leakage screen (smoke tests; not the paper's protocol)")
    args = ap.parse_args(argv)
    if args.command == "build" and (args.problem_taxonomy is None) != (args.method_taxonomy is None):
        ap.error("--problem-taxonomy and --method-taxonomy go together")
    rt = Runtime.from_args(args)
    return cmd_build(args, rt) if args.command == "build" else cmd_apply(args, rt)


if __name__ == "__main__":
    sys.exit(main())
