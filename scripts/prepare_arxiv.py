#!/usr/bin/env python
"""Build the arXiv sets and the arXiv-Subfields corpus (Sec. 5.2, App. C.2).

Streams the Kaggle metadata snapshot once and writes::

    <out>/arxiv_sets/<category>.jsonl.gz       # 2,000 papers per category (uniform, fixed seed)
    <out>/arxiv_subfields/papers.jsonl.gz      # all papers of the eight primary categories
    <out>/arxiv_subfields/stats.json

Examples::

    python scripts/prepare_arxiv.py --snapshot data/raw/arxiv
    python scripts/prepare_arxiv.py --snapshot data/raw/arxiv/arxiv-metadata-oai-snapshot.json --until 2026-09-30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.data.arxiv import ARXIV_SET_CATEGORIES, PAPER_SUBFIELDS_SIZE, write_arxiv_corpora  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, default=Path("data/raw/arxiv"),
                    help="arxiv-metadata-oai-snapshot.json[.gz] / arxiv.zip, or the directory holding it")
    ap.add_argument("--out", type=Path, default=Path("data/processed"), help="processed data root")
    ap.add_argument("--categories", default=",".join(ARXIV_SET_CATEGORIES),
                    help="comma-separated primary categories (default: the paper's eight)")
    ap.add_argument("--n-per-set", type=int, default=2000, help="papers per arXiv set (default 2000)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    ap.add_argument("--until", default=None, help="inclusive ISO cutoff on the first-version date")
    ap.add_argument("--limit", type=int, default=None, help="read only the first N snapshot records")
    ap.add_argument("--no-subfields", action="store_true", help="only build the arXiv sets")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    report = write_arxiv_corpora(args.snapshot, args.out, categories=cats, n_per_set=args.n_per_set,
                                 seed=args.seed, until=args.until, limit=args.limit,
                                 write_subfields=not args.no_subfields)
    info = report.to_dict()
    print(json.dumps(info, indent=2))
    print(f"\narXiv sets: {', '.join(f'{c}={n}' for c, n in report.sets.items())}")
    print(f"arXiv-Subfields: {report.subfields_size:,} papers (paper's September-2026 snapshot: {PAPER_SUBFIELDS_SIZE:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
