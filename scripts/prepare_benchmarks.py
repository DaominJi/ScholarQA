#!/usr/bin/env python
"""Preprocess every benchmark into ``data/processed/`` and report corpus coverage (Sec. 5.3, App. C.2).

Outputs (docs/ARCHITECTURE.md, Sec. 5)::

    <out>/researchqa/queries.jsonl                 # aligned to arXiv-Subfields
    <out>/scholarqabench/{cs,multi}.jsonl          # aligned to arXiv-Subfields
    <out>/taxobench/{papers.jsonl.gz, gold_taxonomy.json, gold/<id>.json, queries.jsonl}
    <out>/chime/{papers.jsonl.gz, gold/<review_id>.json, queries.jsonl}
    <out>/coverage_report.json                     # App. C.2 "Coverage"

ResearchQA and ScholarQABench gold citations are aligned to the
arXiv-Subfields corpus (``--corpus``, built by ``prepare_arxiv.py``) by arXiv
id, Semantic Scholar id (``--s2-papers`` attaches S2 ids to arXiv papers),
DOI and exact normalized title, as in App. C.2 (``--fuzzy`` additionally
enables the stricter opt-in fuzzy-title stage, reported separately as
``fuzzy_title``); ScholarQA-CS evidence quotes are attributed to corpus
papers by searching abstracts and, if available, full-text chunks
(``--chunks``). These two benchmarks need the corpus: without it the script
stops, unless ``--allow-unaligned`` keeps every gold citation unaligned.
``coverage_report.json`` is updated in place, so benchmarks can be prepared
in separate runs.

Examples::

    python scripts/prepare_benchmarks.py
    python scripts/prepare_benchmarks.py --only taxobench,chime --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.data import chime, researchqa, scholarqabench, taxobench  # noqa: E402
from scholarpath.data.alignment import PaperAligner, QuoteAttributor, attach_s2_ids  # noqa: E402
from scholarpath.data.corpus import Corpus, read_json, save_queries, write_json  # noqa: E402

log = logging.getLogger("prepare_benchmarks")
BENCHMARKS = ("researchqa", "scholarqabench", "taxobench", "chime")


def _first_existing(*paths: Path) -> Optional[Path]:
    return next((p for p in paths if p is not None and p.exists()), None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"), help="raw data root (download_data.py layout)")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--only", default=",".join(BENCHMARKS), help="comma-separated benchmarks")
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/arxiv_subfields/papers.jsonl.gz"),
                    help="alignment target (arXiv-Subfields)")
    ap.add_argument("--chunks", type=Path, default=None,
                    help="full-text chunks of the corpus for quote attribution (default: <corpus dir>/chunks.jsonl.gz)")
    ap.add_argument("--s2-papers", type=Path, default=None, help="S2AG `papers` shards to attach S2 ids")
    ap.add_argument("--fuzzy", action="store_true",
                    help="also align near-identical titles (opt-in; App. C.2 uses ids and exact normalized titles)")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.92, help="difflib ratio of the --fuzzy stage")
    ap.add_argument("--allow-unaligned", action="store_true",
                    help="write ResearchQA / ScholarQABench queries even without the alignment corpus")
    ap.add_argument("--cs-unattributed", choices=list(scholarqabench.UNATTRIBUTED_MODES), default="item",
                    help="ScholarQA-CS: one unaligned gold entry per rubric item (default) or per evidence quote")
    ap.add_argument("--researchqa", type=Path, default=None, help="ResearchQA test_mini file/dir (default: <raw>/researchqa)")
    ap.add_argument("--researchqa-extra-gold", type=Path, default=None,
                    help="optional JSON(L) {id, references:[...]} with survey-section citations")
    ap.add_argument("--scholarqabench", type=Path, default=None, help="ScholarQABench repo (default: <raw>/scholarqabench)")
    ap.add_argument("--taxobench", type=Path, default=None, help="TaxoBench repo or data.jsonl (default: <raw>/taxobench)")
    ap.add_argument("--chime", type=Path, default=None, help="CHIME repo or resources dir (default: <raw>/chime)")
    ap.add_argument("--chime-annotations", type=Path, default=None,
                    help="CHIME expert annotation tables (default: <raw>/chime_annotations)")
    ap.add_argument("--chime-gold-studies", choices=["annotated", "assigned", "all"], default="annotated")
    ap.add_argument("--taxobench-max-depth", type=int, default=None,
                    help="truncate the stored merged TaxoBench gold to N levels below the common root (default: "
                         "keep all 7; the experiments cut it at evaluation time, run_taxonomy_experiments.py "
                         "--gold-max-depth 4 by default, as in App. C.2)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N queries/topics per benchmark")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    only = [b.strip() for b in args.only.split(",") if b.strip()]
    unknown = set(only) - set(BENCHMARKS)
    if unknown:
        raise SystemExit(f"Unknown benchmark(s) {sorted(unknown)}; choose from {BENCHMARKS}")

    report: Dict[str, Any] = {}
    corpus: Optional[Corpus] = None
    aligner: Optional[PaperAligner] = None
    alignment: Dict[str, Any] = {"corpus": None}
    needs_corpus = sorted({"researchqa", "scholarqabench"} & set(only))
    if needs_corpus:
        if args.corpus.exists():
            corpus = Corpus.load(args.corpus)
            if args.s2_papers is not None:
                n = attach_s2_ids(corpus, args.s2_papers)
                log.info("attached Semantic Scholar ids to %d papers", n)
            aligner = PaperAligner(corpus, fuzzy=args.fuzzy, fuzzy_threshold=args.fuzzy_threshold)
            alignment = {"corpus": str(args.corpus), **aligner.describe()}
            log.info("alignment target: %s (%d papers)", args.corpus, len(corpus))
        elif args.allow_unaligned:
            log.warning("Alignment corpus %s not found: gold citations of %s stay unaligned (--allow-unaligned)",
                        args.corpus, ", ".join(needs_corpus))
        else:
            log.error("Alignment corpus %s not found; %s gold citations are aligned to arXiv-Subfields. Build it with "
                      "`python scripts/prepare_arxiv.py --snapshot data/raw/arxiv` (or pass --corpus), or rerun with "
                      "--allow-unaligned to keep all gold citations unaligned, or exclude them with --only.",
                      args.corpus, " and ".join(needs_corpus))
            return 2

    if "researchqa" in only:
        src = args.researchqa or args.raw / "researchqa"
        queries, cov = researchqa.load_researchqa(src, aligner, limit=args.limit,
                                                  extra_gold_path=args.researchqa_extra_gold)
        save_queries(args.out / "researchqa" / "queries.jsonl", queries)
        report["researchqa"] = {"queries": len(queries), "rubric_items": researchqa.rubric_item_count(queries),
                                "gold_per_query": round(sum(q.num_gold for q in queries) / max(1, len(queries)), 2),
                                "coverage": cov.to_dict(), "alignment": alignment,
                                "paper_reported": researchqa.PAPER_STATS}
        log.info("ResearchQA: %d queries, coverage %s", len(queries), cov)

    if "scholarqabench" in only:
        root = args.scholarqabench or args.raw / "scholarqabench"
        attributor = None
        if corpus is not None:
            cases = scholarqabench.read_cs_cases(root)
            attributor = QuoteAttributor(scholarqabench.cs_evidence_quotes(cases))
            chunks = args.chunks or args.corpus.parent / "chunks.jsonl.gz"
            n = attributor.scan_corpus(corpus, chunks if chunks.exists() else None)
            log.info("quote attribution: scanned corpus%s (%d texts with hits)",
                     " + chunks" if chunks.exists() else " abstracts only", n)
        cs, cov_cs, origins = scholarqabench.load_scholarqa_cs(root, aligner=aligner, attributor=attributor,
                                                                limit=args.limit, unattributed=args.cs_unattributed)
        multi, cov_multi = scholarqabench.load_scholarqa_multi(root, aligner, limit=args.limit)
        save_queries(args.out / "scholarqabench" / "cs.jsonl", cs)
        save_queries(args.out / "scholarqabench" / "multi.jsonl", multi)
        total = type(cov_cs)().merge(cov_cs).merge(cov_multi)
        n_q = len(cs) + len(multi)
        report["scholarqabench"] = {
            "queries": {"cs": len(cs), "multi": len(multi)},
            "gold_per_query": round(sum(q.num_gold for q in cs + multi) / max(1, n_q), 2),
            "coverage": total.to_dict(), "coverage_cs": cov_cs.to_dict(), "coverage_multi": cov_multi.to_dict(),
            "cs_evidence_origins": origins, "cs_unattributed": args.cs_unattributed, "alignment": alignment,
            "paper_reported": scholarqabench.PAPER_STATS,
        }
        log.info("ScholarQABench: %d + %d queries, coverage %s", len(cs), len(multi), total)

    if "taxobench" in only:
        src = args.taxobench or args.raw / "taxobench"
        data = taxobench.load_taxobench(src, merged_max_depth=args.taxobench_max_depth, limit=args.limit)
        data.save(args.out / "taxobench")
        report["taxobench"] = data.stats
        log.info("TaxoBench: %d surveys, %d distinct papers", data.stats["surveys"], data.stats["distinct_papers"])

    if "chime" in only:
        src = args.chime or args.raw / "chime"
        ann = args.chime_annotations or _first_existing(args.raw / "chime_annotations")
        data = chime.build_chime(src, ann, gold_studies=args.chime_gold_studies, limit=args.limit)
        data.save(args.out / "chime")
        report["chime"] = data.stats
        log.info("CHIME: %d topics, %d studies (corrected=%s)", data.stats["topics"], data.stats["distinct_studies"],
                 data.stats["corrected"])

    report_path = args.out / "coverage_report.json"
    merged: Dict[str, Any] = {}
    if report_path.exists():
        try:
            merged = read_json(report_path)
        except (OSError, ValueError) as err:
            log.warning("Ignoring unreadable %s (%s)", report_path, err)
    merged.update(report)  # benchmarks of this run replace their previous entries
    write_json(report_path, merged)
    print("\nCorpus coverage (App. C.2):")
    for name, info in report.items():
        cov = info.get("coverage", {})
        print(f"  {name:15s} {cov.get('aligned', 0):>7,}/{cov.get('total', 0):<7,} aligned "
              f"({100 * cov.get('coverage', 0):5.1f}%)   gold/query {info.get('gold_per_query')}")
    if args.verbose:
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
