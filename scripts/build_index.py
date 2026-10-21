#!/usr/bin/env python
"""Build the multi-granularity semantic index I of a corpus (Sec. 2.1; corpora of Table 1).

    corpus (+ section chunks) -> dual taxonomies (T_P, T_M) -> SemanticIndex.build
    -> <index-dir>/<name>/   index files (taxonomies, papers, chunks, embeddings, token costs),
                             taxonomy_problem/ and taxonomy_method/ (build results: summary, usage, episodes),
                             build_summary.json (sizes, wall times, LLM cost per component of Eq. 15)

Corpora (Table 1, App. C.1-C.2): ``arxiv_subfields`` (ResearchQA, ScholarQABench;
148,400 papers), ``heldout`` (the reduced arXiv-Subfields corpus written by
build_heldout_benchmark.py), ``taxobench`` (3,044 pooled papers), ``chime``
(2,146 pooled studies); or any ``--papers`` / ``--chunks`` files with ``--name``.

Taxonomies: ScholarPath's node-wise confidence-bounded builder (Algorithm 1;
default), a construction baseline of Sec. 5.2 (``--taxonomy full``,
``--taxonomy kmeans@0.2``, ...), or prebuilt taxonomy JSON files
(``--problem-taxonomy`` / ``--method-taxonomy``; extents are restricted to
the corpus and concepts left without papers are dropped). An existing index
is kept unless ``--force``.

Held-out corpus (App. C.1: the held-out papers are "removed before any index
is built"): its taxonomies must be built over the reduced corpus. Prebuilt
taxonomies that reference papers outside it (e.g. the reference taxonomies
of build_heldout_benchmark.py, built over the complete corpus) are refused
unless ``--allow-reference-taxonomy`` (offline runs): then the concepts of
the removed papers are dropped, but the names and definitions of the
remaining concepts may still reflect the held-out papers.

Paper setting (DeepSeek-V4.0-Flash + SPECTER2, configs/paper.yaml)::

    python scripts/build_index.py --config configs/paper.yaml --corpus arxiv_subfields
    python scripts/build_index.py --config configs/paper.yaml --corpus taxobench
    python scripts/build_index.py --config configs/paper.yaml --corpus chime
    python scripts/build_index.py --config configs/paper.yaml --corpus heldout

Quick offline run (mock LLM + hashing embedder on the synthetic layout)::

    python scripts/make_synthetic_data.py --out data/synthetic
    python scripts/build_index.py --data-dir data/synthetic --corpus arxiv_subfields \\
        --problem-taxonomy data/synthetic/gold/problem_taxonomy.json \\
        --method-taxonomy data/synthetic/gold/method_taxonomy.json
    python scripts/build_index.py --data-dir data/synthetic --corpus taxobench --limit-papers 60 \\
        --set builder.max_depth=1   # ScholarPath builder on a small sample
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.data.corpus import Corpus  # noqa: E402
from scholarpath.experiments.qa import (  # noqa: E402
    CORPORA,
    DEFAULT_DATA_DIR,
    DEFAULT_INDEX_ROOT,
    DataLayout,
    IndexStore,
    build_index,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402
from scholarpath.taxonomy.tree import Taxonomy  # noqa: E402

log = logging.getLogger("build_index")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_output="results/build_index")
    g = ap.add_argument_group("corpus")
    g.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="processed data root")
    g.add_argument("--corpus", choices=sorted(CORPORA), default=None,
                   help="corpus of the data layout (default: arxiv_subfields unless --papers)")
    g.add_argument("--papers", type=Path, default=None, help="explicit papers JSONL(.gz) instead of --corpus")
    g.add_argument("--chunks", type=Path, default=None, help="section chunks JSONL(.gz) of --papers")
    g.add_argument("--name", default=None, help="index name (default: the corpus name)")
    g.add_argument("--limit-papers", type=int, default=None, help="index only the first N papers (quick runs)")
    t = ap.add_argument_group("taxonomies")
    t.add_argument("--taxonomy", default="scholarpath",
                   help="'scholarpath' (Algorithm 1) or a Sec. 5.2 baseline: full, kmeans@0.2, "
                        "one_off/coverage@0.2, ...")
    t.add_argument("--problem-taxonomy", type=Path, default=None, help="prebuilt problem taxonomy (Taxonomy JSON)")
    t.add_argument("--method-taxonomy", type=Path, default=None, help="prebuilt method taxonomy (Taxonomy JSON)")
    t.add_argument("--allow-reference-taxonomy", action="store_true",
                   help="held-out corpus only: accept prebuilt taxonomies that reference removed papers "
                        "(restricted to the corpus; offline runs)")
    o = ap.add_argument_group("output")
    o.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_ROOT, help="index root (<index-dir>/<name>/)")
    o.add_argument("--force", action="store_true", help="rebuild an existing index")
    args = ap.parse_args(argv)
    if (args.problem_taxonomy is None) != (args.method_taxonomy is None):
        ap.error("--problem-taxonomy and --method-taxonomy go together")

    rt = Runtime.from_args(args)
    if args.papers is not None:
        name = args.name or args.papers.name.split(".")[0]
        corpus = Corpus.load(args.papers, limit=args.limit_papers, chunks_path=args.chunks, name=name)
    else:
        corpus_name = args.corpus or "arxiv_subfields"
        name = args.name or corpus_name
        corpus = DataLayout(args.data_dir).load_corpus(corpus_name, limit=args.limit_papers)
    store = IndexStore(args.index_dir)
    out = store.path(name)
    if store.exists(name) and not args.force:
        log.info("Index %s already exists at %s (use --force to rebuild)", name, out)
        print(json.dumps({"name": name, "index_dir": str(out), "skipped": True}))
        return 0

    log.info("Indexing %s: %d papers, %d chunks (taxonomies: %s)", name, len(corpus), corpus.num_chunks,
             "prebuilt" if args.problem_taxonomy else args.taxonomy)
    heldout = args.papers is None and (args.corpus or "arxiv_subfields") == "heldout"
    try:
        build = build_index(
            corpus, rt,
            taxonomy=args.taxonomy,
            problem_taxonomy=Taxonomy.load(args.problem_taxonomy) if args.problem_taxonomy else None,
            method_taxonomy=Taxonomy.load(args.method_taxonomy) if args.method_taxonomy else None,
            name=name,
            allow_foreign_papers=not heldout or args.allow_reference_taxonomy,
        )
    except ValueError as err:
        if not heldout:
            raise
        ap.error(f"{err} -- the held-out index must not see the removed papers (App. C.1); "
                 "pass --allow-reference-taxonomy to restrict them anyway")
    build.save(out)
    summary = dict(build.summary, index_dir=str(out))
    if rt.output_dir is not None:
        rt.write_json(f"{name}_build_summary.json", summary)
    idx = build.summary["index"]
    print(json.dumps({
        "name": name, "index_dir": str(out), "papers": idx["papers"], "chunks": idx["chunks"],
        "problem_nodes": idx["problem"]["num_nodes"], "method_nodes": idx["method"]["num_nodes"],
        "taxonomy_method": build.summary["taxonomy_method"], "llm_calls": build.summary["llm_calls"],
        "cost_usd": round(build.summary["cost_usd"], 6), "seconds": round(build.summary["seconds"]["total"], 2),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
