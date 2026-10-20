#!/usr/bin/env python
"""Download the raw datasets of the paper into ``data/raw/<name>/``.

Examples::

    python scripts/download_data.py --list
    python scripts/download_data.py --dry-run                 # show what would run
    python scripts/download_data.py                           # default set (no S2 API key needed)
    python scripts/download_data.py --only taxobench,chime    # a subset
    S2_API_KEY=... python scripts/download_data.py --only s2orc_v2 --max-shards 2

Downloads are idempotent (finished datasets are skipped unless ``--force``).
Hugging Face, Kaggle and arxiv.org may be unreachable behind restrictive
proxies; failures print manual instructions, and every preprocessing script
also accepts pre-downloaded files (see data/README.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.data.download import DATASETS, DEFAULT_DATASETS, DEFAULT_RAW_ROOT, download_dataset  # noqa: E402


def parse_names(values):
    names = []
    for v in values or []:
        names.extend(x.strip() for x in v.split(",") if x.strip())
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}. Available: {', '.join(DATASETS)}")
    return names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_RAW_ROOT, help="raw data root (default: data/raw)")
    ap.add_argument("--only", action="append", metavar="NAME[,NAME]", help="datasets to download")
    ap.add_argument("--all", action="store_true", help="also download the optional datasets (S2AG, ResearchQA repo)")
    ap.add_argument("--dry-run", action="store_true", help="print the planned commands only")
    ap.add_argument("--force", action="store_true", help="re-download finished datasets")
    ap.add_argument("--max-shards", type=int, default=None, help="S2AG datasets: download at most N shards")
    ap.add_argument("--list", action="store_true", help="list the available datasets and exit")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.list:
        for name, spec in DATASETS.items():
            flag = "default" if spec.default else "optional"
            print(f"{name:18s} {spec.kind:7s} {flag:9s} {spec.description}")
        return 0
    names = parse_names(args.only) or (list(DATASETS) if args.all else list(DEFAULT_DATASETS))
    results = []
    for name in names:
        res = download_dataset(name, args.root, dry_run=args.dry_run, force=args.force, max_shards=args.max_shards)
        results.append(res)
        if not args.json:
            print(f"[{res.status:7s}] {name} -> {res.target}")
            if res.status in ("planned", "failed"):
                for cmd in res.commands:
                    print(f"           $ {cmd}")
            if res.status == "failed":
                print(f"           {res.message}")
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
