#!/usr/bin/env python
"""Write the synthetic repository in the processed-data layout used by every script.

The synthetic data (``scholarpath.data.synthetic``) is artificial: it only
exists so that the whole pipeline -- taxonomy construction, index building,
link-prediction training, held-out benchmark construction, retrieval and QA
evaluation -- can be exercised offline with the mock LLM and the hashing
embedder (see ``scripts/smoke_test.sh``). It does not reproduce any number of
the paper.

Layout written under ``--out`` (default ``data/synthetic``), mirroring
``data/processed``::

    arxiv_sets/synthetic.jsonl.gz          # construction corpus (no chunks)
    arxiv_subfields/papers.jsonl.gz        # QA corpus (held-out papers removed)
    arxiv_subfields/chunks.jsonl.gz        # section chunks of the QA corpus
    arxiv_subfields_full/papers.jsonl.gz   # QA corpus *with* the held-out papers
    arxiv_subfields_full/chunks.jsonl.gz
    evidence/queries.jsonl                 # evidence queries (gold = J(u, m), two rubric items each)
    heldout/queries.jsonl                  # held-out hypothesis queries (planted)
    heldout_reference/{problem_taxonomy.json, method_taxonomy.json, domains.json}
    taxobench/{papers.jsonl.gz, gold_taxonomy.json, gold/<id>.json, queries.jsonl}
    gold/{problem_taxonomy.json, method_taxonomy.json}

``heldout_reference/`` holds planted *reference* taxonomies over the complete
corpus (``arxiv_subfields_full``) and a domain per paper, the inputs of the
held-out benchmark construction of App. C.1: the first held-out pair becomes a
small "emerging" problem concept (new problem / existing method) and the
second a small "emerging" method concept (new method / existing problem),
whose papers postdate the rest of the corpus and belong to a new domain::

    python scripts/build_heldout_benchmark.py --data-dir data/synthetic build --force \\
        --problem-taxonomy data/synthetic/heldout_reference/problem_taxonomy.json \\
        --method-taxonomy data/synthetic/heldout_reference/method_taxonomy.json \\
        --domain-map data/synthetic/heldout_reference/domains.json --n-new-problem 1 --n-new-method 1

Example::

    python scripts/make_synthetic_data.py --out data/synthetic --n-papers 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.data.corpus import Corpus, save_queries, write_json  # noqa: E402
from scholarpath.data.synthetic import SyntheticBenchmark, make_synthetic_benchmark  # noqa: E402
from scholarpath.taxonomy.tree import Taxonomy  # noqa: E402
from scholarpath.types import EVIDENCE, Query, RubricItem  # noqa: E402

#: Domain of the planted held-out papers (a new domain: their counterparts come from established ones).
EMERGING_DOMAIN = "emerging"


def with_rubrics(queries: List[Query]) -> List[Query]:
    """Two ResearchQA-style rubric items per synthetic evidence query (content + citation), so that the
    rubric-coverage path (Rub) runs offline."""
    for q in queries:
        method, problem = q.metadata.get("method", ""), q.metadata.get("problem", "")
        q.rubric = [
            RubricItem(text=f"Does the response explain how {method} is applied to {problem}?", kind="Other"),
            RubricItem(text=f"Does the response cite at least one paper that uses {method} for {problem}?",
                       kind="Citation"),
        ]
    return queries


def heldout_reference(bench: SyntheticBenchmark) -> Tuple[Taxonomy, Taxonomy, Dict[str, str]]:
    """Planted reference taxonomies over the complete corpus and a domain map (App. C.1 inputs).

    The gold taxonomies plus the held-out papers at their (sub-problem, method)
    leaves; held-out pair 0 gets a new problem concept below its sub-problem
    and pair 1 a new method concept below its method, each holding only that
    pair's papers (<= 4: an emerging idea). Regular papers belong to the
    domain of their sub-problem, held-out papers to :data:`EMERGING_DOMAIN`.
    """
    P = Taxonomy.from_dict(bench.problem_gold.to_dict())
    M = Taxonomy.from_dict(bench.method_gold.to_dict())
    leaf_p = {P.nodes[n].name: n for n in P.leaves()}
    leaf_m = {M.nodes[n].name: n for n in M.leaves()}
    domains: Dict[str, str] = {}
    for p in bench.corpus:
        leaf = leaf_p[bench.pair_of[p.paper_id][0]]
        domains[p.paper_id] = P.nodes[P.parent(leaf)].name
    by_pair: Dict[int, List[str]] = {}
    for p in bench.heldout_papers:
        s, m = bench.pair_of[p.paper_id]
        P.assign(p.paper_id, leaf_p[s])
        M.assign(p.paper_id, leaf_m[m])
        domains[p.paper_id] = EMERGING_DOMAIN
        by_pair.setdefault(bench.heldout_pairs.index((s, m)), []).append(p.paper_id)
    if 0 in by_pair:
        s, _ = bench.heldout_pairs[0]
        P.add_node(f"{s} under Sensor Drift", f"{s} when the input sensors drift over time.",
                   parent=leaf_p[s], papers=by_pair[0])
    if 1 in by_pair:
        _, m = bench.heldout_pairs[1]
        M.add_node(f"Drift-Aware {m}", f"A {m.lower()} variant that adapts its parameters online.",
                   parent=leaf_m[m], papers=by_pair[1])
    return P, M, domains


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--n-papers", type=int, default=240)
    ap.add_argument("--n-heldout", type=int, default=2, help="held-out (problem, method) pairs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    bench = make_synthetic_benchmark(n_papers=args.n_papers, n_heldout=args.n_heldout, seed=args.seed)
    out: Path = args.out
    corpus = bench.corpus
    full = Corpus(list(corpus) + list(bench.heldout_papers), name="synthetic_full")

    corpus.save(out / "arxiv_sets" / "synthetic.jsonl.gz", include_chunks=False)
    corpus.save(out / "arxiv_subfields" / "papers.jsonl.gz", include_chunks=False)
    corpus.save_chunks(out / "arxiv_subfields" / "chunks.jsonl.gz")
    full.save(out / "arxiv_subfields_full" / "papers.jsonl.gz", include_chunks=False)
    full.save_chunks(out / "arxiv_subfields_full" / "chunks.jsonl.gz")
    save_queries(out / "evidence" / "queries.jsonl", with_rubrics(bench.evidence_queries))
    save_queries(out / "heldout" / "queries.jsonl", bench.hypothesis_queries)
    bench.problem_gold.save(out / "gold" / "problem_taxonomy.json")
    bench.method_gold.save(out / "gold" / "method_taxonomy.json")
    ref_p, ref_m, domains = heldout_reference(bench)
    ref_p.save(out / "heldout_reference" / "problem_taxonomy.json")
    ref_m.save(out / "heldout_reference" / "method_taxonomy.json")
    write_json(out / "heldout_reference" / "domains.json", domains)

    # TaxoBench-style organization benchmark: one "survey" per problem domain.
    tb = out / "taxobench"
    corpus.save(tb / "papers.jsonl.gz", include_chunks=False)
    merged = Taxonomy.from_dict(bench.problem_gold.to_dict())
    merged.facet = "topic"
    merged.save(tb / "gold_taxonomy.json")
    queries = []
    pg = bench.problem_gold
    for i, dom in enumerate(pg.children(pg.root_id)):
        sub = Taxonomy(facet="topic", root_name=pg.nodes[dom].name, root_definition=pg.nodes[dom].definition)
        for leaf in pg.children(dom):
            nid = sub.add_node(pg.nodes[leaf].name, pg.nodes[leaf].definition)
            for p in sorted(pg.extent(leaf)):
                sub.assign(p, nid)
        sub.save(tb / "gold" / f"{i}.json")
        topic = pg.nodes[dom].name
        queries.append(Query(
            query_id=f"taxobench-{i}",
            text=topic,
            kind=EVIDENCE,
            benchmark="taxobench",
            gold_papers=sorted(pg.extent(dom)),
            metadata={"task": "organization", "style": "survey", "mode": "deep_research", "survey_id": i,
                      "topic": topic, "instruction": f"Organize the literature on {topic} into a topic hierarchy.",
                      "gold_taxonomy": f"gold/{i}.json"},
        ))
    save_queries(tb / "queries.jsonl", queries)
    write_json(out / "stats.json", {
        "papers": len(corpus), "heldout_papers": len(bench.heldout_papers),
        "chunks": corpus.num_chunks, "evidence_queries": len(bench.evidence_queries),
        "hypothesis_queries": len(bench.hypothesis_queries), "taxobench_queries": len(queries),
        "heldout_pairs": bench.heldout_pairs,
    })
    print(f"Wrote synthetic data ({len(corpus)} papers, {len(bench.evidence_queries)} evidence queries, "
          f"{len(bench.hypothesis_queries)} hypothesis queries) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
