#!/usr/bin/env python
"""Fetch full texts and write section chunks for a corpus (Sec. 2.1).

Sources are tried in ``--sources`` order (default ``local,s2orc,gcs,arxiv``);
sources without configuration (no ``--local-dir`` / ``--s2orc``) are
skipped. Chunk sizes come from ``IndexConfig`` (``--config``, overridable with
``--set index.chunk_max_tokens=...``). The run is resumable: rerunning the
command continues where it stopped (``<out stem>.state.jsonl``).

Examples::

    # arXiv-Subfields from the public GCS PDF mirror (PyMuPDF required)
    python scripts/prepare_fulltext.py --corpus data/processed/arxiv_subfields/papers.jsonl.gz \\
        --sources gcs --cache-dir data/fulltext/cache --limit 100
    # S2ORC v2 shards first, then local files, then GCS
    python scripts/prepare_fulltext.py --s2orc data/raw/s2orc_v2 --local-dir data/fulltext/local \\
        --sources s2orc,local,gcs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.config import ScholarPathConfig  # noqa: E402
from scholarpath.data.corpus import Corpus  # noqa: E402
from scholarpath.data.download import USER_AGENT  # noqa: E402
from scholarpath.data.fulltext import DEFAULT_SOURCE_ORDER, FulltextExtractor, make_sources  # noqa: E402
from scholarpath.runtime import DEFAULT_CONFIG  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/arxiv_subfields/papers.jsonl.gz"))
    ap.add_argument("--out", type=Path, default=None, help="chunks file (default: <corpus dir>/chunks.jsonl.gz)")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCE_ORDER), help="comma-separated source order")
    ap.add_argument("--local-dir", type=Path, default=Path("data/fulltext/local"))
    ap.add_argument("--s2orc", type=Path, action="append", default=[], help="S2ORC v2 shard file or directory")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/fulltext/cache"), help="downloaded PDFs / e-prints")
    ap.add_argument("--arxiv-interval", type=float, default=3.0, help="seconds between arxiv.org requests")
    ap.add_argument("--user-agent", default=USER_AGENT)
    ap.add_argument("--ids", type=Path, default=None, help="optional file with one paper id per line (subset)")
    ap.add_argument("--limit", type=int, default=None, help="process at most N papers")
    ap.add_argument("--retry-missing", action="store_true", help="retry papers that had no full text before")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    config = ScholarPathConfig.load(args.config).override(args.overrides)
    corpus = Corpus.load(args.corpus)
    papers = list(corpus)
    if args.ids is not None:
        wanted = {l.strip() for l in args.ids.read_text(encoding="utf-8").splitlines() if l.strip()}
        papers = [p for p in papers if p.paper_id in wanted]
    out = args.out or args.corpus.parent / "chunks.jsonl.gz"
    sources = make_sources([s.strip() for s in args.sources.split(",") if s.strip()], local_dir=args.local_dir,
                           s2orc_paths=args.s2orc or None, cache_dir=args.cache_dir, user_agent=args.user_agent,
                           arxiv_interval_s=args.arxiv_interval)
    if not sources:
        raise SystemExit("No usable full-text source; see --help")
    extractor = FulltextExtractor.from_index_config(sources, config.index)
    stats = extractor.run(papers, out, limit=args.limit, retry_missing=args.retry_missing)
    info = stats.to_dict()
    info.update({"corpus": str(args.corpus), "out": str(out), "sources": [s.name for s in sources],
                 "chunk_max_tokens": config.index.chunk_max_tokens, "chunk_min_tokens": config.index.chunk_min_tokens})
    stats_path = out.with_name(out.name.split(".")[0] + ".stats.json")
    stats_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
