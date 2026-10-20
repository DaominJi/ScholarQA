#!/usr/bin/env python
"""Build one taxonomy (or the dual problem / method pair) on a corpus; print the tree and its LLM cost.

Methods (``--method``):

* ``scholarpath`` -- node-wise confidence-bounded construction (Algorithm 1,
  Section 3; eps = 0.10, 1 - delta = 0.95 by default). ``--ablation`` selects
  a Table 6 variant (e.g. ``no_discovery``).
* ``full`` -- exhaustive LLM construction (FULL, eta = 1; Section 5.2).
* ``random`` / ``coverage`` / ``kmeans`` / ``diversity`` -- selection baselines
  (Section 5.2) at ratio ``--eta``, node-wise (``--mode node_wise``, default) or
  one-off (``--mode one_off``, App. C.5).

``--facet dual`` (default) builds the problem taxonomy T_P and the method
taxonomy T_M (Section 2); ``--facet topic`` builds the single
topic-and-approach taxonomy used on TaxoBench. ``--reference`` scores the
result against a reference taxonomy (soft node / edge F1, Section 5.2),
prepared as in ``run_taxonomy_experiments.py``: ``--reference-max-depth`` cuts
it below that many levels, and with ``--limit-papers`` it is restricted to the
sampled papers (concepts left empty are dropped).

Outputs under ``--output-dir`` (default ``results/build_taxonomy/<corpus>__<method key>``,
e.g. ``cs.DB__scholarpath``, ``cs.DB__coverage@0.2``, ``cs.DB__one_off-kmeans@0.2``,
``cs.DB__scholarpath-no_discovery``)::

    <facet>/{taxonomy.json, summary.json, usage.json, episodes.jsonl, routers.*}   # BuildResult.save
    build.json                                                         # per-facet cost and scores
    config.yaml

Every build is also stored in the shared build cache (``--cache-dir``,
default ``results/taxonomy_cache``); ``--resume`` reuses a stored build.

Paper setting (DeepSeek-V4.0-Flash, SPECTER2)::

    python scripts/build_taxonomy.py --config configs/paper.yaml \\
        --corpus data/processed/arxiv_sets/cs.DB.jsonl.gz --facet dual --method scholarpath

Quick offline run (mock LLM, hashing embedder)::

    python scripts/make_synthetic_data.py --out /tmp/syn
    python scripts/build_taxonomy.py --corpus /tmp/syn/arxiv_sets/synthetic.jsonl.gz --method scholarpath \\
        --limit-papers 120 --set builder.router.hidden_dim=32
    python scripts/build_taxonomy.py --corpus /tmp/syn/taxobench/papers.jsonl.gz --facet topic \\
        --method coverage --eta 0.2 --reference /tmp/syn/taxobench/gold_taxonomy.json \\
        --set baseline.router.hidden_dim=32 --min-router-steps 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.evaluation.reporting import write_json  # noqa: E402
from scholarpath.experiments.common import (  # noqa: E402
    add_experiment_args,
    cache_from_args,
    load_corpus,
    save_build_result,
    seed_from_args,
)
from scholarpath.experiments.taxonomy import (  # noqa: E402
    ABLATIONS,
    DUAL_FACETS,
    FACET_ROOTS,
    ConstructionDataset,
    ConstructionRunner,
    MethodSpec,
    make_scorer,
    restrict_taxonomy,
    score_taxonomy,
    truncate_taxonomy,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402
from scholarpath.taxonomy.baselines import FULL, MODES  # noqa: E402
from scholarpath.taxonomy.selectors import SELECTION_METHODS  # noqa: E402
from scholarpath.taxonomy.tree import Taxonomy  # noqa: E402

FACETS = ("problem", "method", "topic", "dual")
METHODS = ("scholarpath", FULL) + tuple(SELECTION_METHODS)


def corpus_name(path: Path) -> str:
    """``arxiv_sets/cs.DB.jsonl.gz`` -> ``cs.DB``; ``taxobench/papers.jsonl.gz`` -> ``taxobench``."""
    name = path.name
    for suffix in (".jsonl.gz", ".jsonl", ".json.gz", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name == "papers" and path.parent.name:
        return path.parent.name
    return name


def method_spec(args: argparse.Namespace, rt: Runtime) -> MethodSpec:
    if args.method == "scholarpath":
        return MethodSpec.scholarpath(args.ablation)
    if args.ablation:
        raise SystemExit("--ablation only applies to --method scholarpath")
    if args.method == FULL:
        return MethodSpec.full()
    eta = rt.config.baseline.eta if args.eta is None else args.eta
    mode = rt.config.baseline.mode if args.mode is None else args.mode
    return MethodSpec.selection(args.method, eta, mode)


def output_tag(method: MethodSpec) -> str:
    """Folder-safe method key: ``coverage@0.2``, ``one_off-kmeans@0.2``, ``scholarpath-no_discovery``."""
    return method.key.replace("/", "-")


def load_reference(path: Path, max_depth: Optional[int], paper_ids: Optional[List[str]]) -> Taxonomy:
    """Reference taxonomy cut at ``max_depth`` and, for a ``--limit-papers`` sample, restricted to its papers."""
    reference = truncate_taxonomy(Taxonomy.load(path), max_depth)
    return restrict_taxonomy(reference, paper_ids) if paper_ids is not None else reference


def cost_line(result: Any) -> str:
    usage = result.total_usage
    parts = [f"${result.cost_usd:.4f}", f"{usage.calls} LLM calls", f"{usage.total_tokens:,} tokens",
             f"{sum(result.papers_read.values()):,} paper reads", f"{result.runtime_s:.1f}s"]
    breakdown = {k: v for k, v in result.cost_breakdown().items() if v > 0}
    if breakdown:
        parts.append("breakdown " + ", ".join(f"{k}=${v:.4f}" for k, v in breakdown.items()))
    return "; ".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/arxiv_sets/cs.DB.jsonl.gz"),
                    help="processed corpus (Paper JSONL, optionally gzipped)")
    ap.add_argument("--facet", choices=FACETS, default="dual",
                    help="problem, method, topic (TaxoBench), or dual = problem + method (default)")
    ap.add_argument("--method", choices=METHODS, default="scholarpath")
    ap.add_argument("--eta", type=float, default=None,
                    help="selection ratio of a baseline (default: config baseline.eta)")
    ap.add_argument("--mode", choices=MODES, default=None, help="node_wise or one_off (default: config baseline.mode)")
    ap.add_argument("--ablation", choices=list(ABLATIONS), default=None, help="ScholarPath variant of Table 6")
    ap.add_argument("--root-name", default=None, help="root concept name (single facet only)")
    ap.add_argument("--root-definition", default=None, help="root concept definition (single facet only)")
    ap.add_argument("--reference", type=Path, default=None,
                    help="reference taxonomy JSON (e.g. TaxoBench gold_taxonomy.json) to report soft node / edge F1")
    ap.add_argument("--reference-max-depth", type=int, default=None,
                    help="cut the reference below this many levels under its root (TaxoBench paper setting: 4)")
    ap.add_argument("--print-depth", type=int, default=3, help="deepest level printed (-1: all)")
    add_experiment_args(ap, data_dir=False, seeds=False, plots=False)
    add_common_args(ap)
    args = ap.parse_args(argv)

    if not args.corpus.exists():
        ap.error(f"corpus {args.corpus} does not exist (see data/README.md or scripts/make_synthetic_data.py)")
    facets = list(DUAL_FACETS) if args.facet == "dual" else [args.facet]
    if (args.root_name or args.root_definition) and len(facets) != 1:
        ap.error("--root-name / --root-definition need a single --facet")
    name = corpus_name(args.corpus)

    rt = Runtime.from_args(args)
    method = method_spec(args, rt)
    if rt.output_dir is None:  # named after the method key, whose eta / mode may come from the config
        rt.output_dir = Path("results") / "build_taxonomy" / f"{name}__{output_tag(method)}"
        rt.output_dir.mkdir(parents=True, exist_ok=True)
        rt.config.save(rt.output_dir / "config.yaml")
    corpus = load_corpus(args.corpus, args.limit_papers, args.sample_seed, name=name)
    dataset = ConstructionDataset(name=name, benchmark="custom", corpus=corpus, facets=tuple(facets))
    roots = None
    if args.root_name or args.root_definition:
        default_name, default_def = FACET_ROOTS.get(facets[0], ("Root", ""))
        roots = {facets[0]: (args.root_name or default_name, args.root_definition or default_def)}
    runner = ConstructionRunner.from_runtime(rt, cache_from_args(args), min_router_steps=args.min_router_steps,
                                             roots=roots)
    reference = None
    if args.reference is not None:
        reference = load_reference(args.reference, args.reference_max_depth,
                                   corpus.ids if args.limit_papers is not None else None)
    scorer = make_scorer(rt.embedder) if reference is not None else None

    print(f"Corpus {args.corpus} ({len(corpus):,} papers); method {method.key}; facets {', '.join(facets)}")
    report: Dict[str, Any] = {"corpus": str(args.corpus), "n_papers": len(corpus), "method": method.key, "facets": {}}
    depth = None if args.print_depth is not None and args.print_depth < 0 else args.print_depth
    for facet in facets:
        result = runner.build(dataset, facet, method, seed_from_args(args))
        save_build_result(result, rt.output_dir / facet)
        entry: Dict[str, Any] = {"summary": result.summary(), "output": str(rt.output_dir / facet)}
        tax = result.taxonomy
        print(f"\n=== {facet} taxonomy ({len(tax.nodes) - 1} concepts, depth {tax.max_depth}) ===")
        print(tax.pretty(max_depth=depth))
        print(f"cost: {cost_line(result)}")
        if reference is not None:
            scores = score_taxonomy(result.taxonomy, reference, scorer)
            entry["scores"] = scores
            print(f"vs. {args.reference}: soft node F1 {100 * scores['node_f1']:.1f}, "
                  f"soft edge F1 {100 * scores['edge_f1']:.1f}")
        report["facets"][facet] = entry
    total = sum(e["summary"]["cost_usd"] for e in report["facets"].values())
    report["total_cost_usd"] = total
    report["cache"] = runner.cache.summary()
    write_json(rt.output_dir / "build.json", report)
    print(f"\nTotal LLM cost: ${total:.4f}. Outputs in {rt.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
