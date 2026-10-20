#!/usr/bin/env python
"""Run the unbudgeted agent systems of Tables 3-5 over a benchmark's local corpus (Sec. 5.3, App. C.3).

PaperQA2, OpenScholar, AgenticScholar and Codex run over the same local
corpus as the budgeted methods (the corpus of the benchmark's semantic
index), with the same LLM backbone (``config.llm``; DeepSeek-V4.0-Flash in the
paper) and no external web access. "The local corpus is mounted as JSONL
shard files indexed by paper ID and primary category" (App. C.3): the corpus
is exported once per index with ``export_corpus_shards``. Every system
receives the identical JSON-schema task prompt and must return ranked
records with cited paper IDs and an explicit established / hypothesis status;
AgenticScholar is equipped with ScholarPath's taxonomies. Their token budgets
cannot be controlled, so they form the "Unbudgeted methods" rows.

Organization queries (TaxoBench, CHIME, Deep Research mode) give the agent
the benchmark's instruction ("Retrieve the key papers on the topic ... and
organize them into a topic hierarchy ..."); the ranked records are read as
the categories of a one-level hierarchy for ARI / Sem / Cat / Assn.

Usage (calls, tokens, cost, latency) is metered per query by the local
metering proxy (or reported by the system) and saved with every answer in the
layout run_qa_eval.py (``--answers-dir``) and run_efficiency.py consume; the
cost is priced with the backbone prices of ``config.llm`` (Table 5), and
``--resume`` only reuses answers of the same agent configuration::

    answers/<benchmark>/<agent>/unbudgeted.jsonl
    corpus_shards/<index>/...      exported corpus
    summary.json

The ``simulated`` agent (plan -> BM25 + dense search -> reflect -> answer
with the in-process LLM) is NOT a paper baseline; it exercises this path
offline.

Paper setting (external systems installed, see scholarpath/baselines/agents/*.py)::

    python scripts/run_agents.py --config configs/paper.yaml --agents openscholar,paperqa2,agenticscholar,codex \\
        --openscholar-dir ~/OpenScholar --agenticscholar-url http://localhost:8800/run --resume

Quick offline run::

    python scripts/run_agents.py --data-dir data/synthetic --agents simulated --limit 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.evaluation.reporting import write_json  # noqa: E402
from scholarpath.experiments.qa import (  # noqa: E402
    AGENT_SYSTEMS,
    EVIDENCE_BENCHMARKS,
    HYPOTHESIS_BENCHMARKS,
    SIMULATED,
    SYSTEMS,
    AgentOptions,
    DataLayout,
    EnvironmentCache,
    RecordStore,
    SystemOptions,
    add_data_args,
    agent_fingerprint,
    build_agent,
    count_errors,
    efficiency_summary,
    environment_summary,
    index_corpus,
    index_store_from_args,
    resolve_benchmarks,
    resolve_systems,
    run_agent,
    system_display,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402

log = logging.getLogger("run_agents")

#: Agents that read the exported JSONL shards (the others index the corpus in their own format).
_NEEDS_EXPORT = ("codex", "agenticscholar")


def main(argv=None) -> int:
    from scholarpath.baselines.agents import AgentUnavailableError, export_corpus_shards

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_output="results/run_agents")
    add_data_args(ap)
    ap.add_argument("--agents", default=",".join(AGENT_SYSTEMS),
                    help=f"comma list of {list(AGENT_SYSTEMS) + [SIMULATED]} (default: the four paper agents)")
    ap.add_argument("--export-dir", type=Path, default=None,
                    help="where the corpus shards are exported (default: <output-dir>/corpus_shards)")
    g = ap.add_argument_group("agents")
    g.add_argument("--workdir", default=None, help="scratch directory of the agents")
    g.add_argument("--max-records", type=int, default=10, help="records requested by the task prompt")
    g.add_argument("--timeout", type=float, default=3600.0, help="per-query timeout of external agents (s)")
    g.add_argument("--no-proxy", action="store_true", help="do not meter agents through the local proxy")
    g.add_argument("--openscholar-dir", default=None, help="OpenScholar checkout")
    g.add_argument("--agenticscholar-url", default=None, help="AgenticScholar HTTP endpoint")
    g.add_argument("--agenticscholar-fn", default=None, help="AgenticScholar Python callable module:function")
    g.add_argument("--codex-binary", default="codex")
    g.add_argument("--codex-isolation", choices=("unshare", "none"), default="unshare")
    g.add_argument("--paperqa-embedding", default=None, help="PaperQA2 embedding model name")
    args = ap.parse_args(argv)

    rt = Runtime.from_args(args)
    out: Path = rt.output_dir or Path("results/run_agents")
    out.mkdir(parents=True, exist_ok=True)
    layout = DataLayout(args.data_dir)
    benchmarks = resolve_benchmarks(args.benchmarks, layout.available_benchmarks(),
                                    default=EVIDENCE_BENCHMARKS + HYPOTHESIS_BENCHMARKS)
    agents = resolve_systems(args.agents, AGENT_SYSTEMS)
    bad = [a for a in agents if SYSTEMS[a].family != "agent"]
    if bad:
        ap.error(f"Not agent systems: {bad} (budgeted systems are run by run_qa_eval.py)")
    options = AgentOptions(workdir=args.workdir, max_records=args.max_records, timeout_s=args.timeout,
                           meter_with_proxy=not args.no_proxy, openscholar_dir=args.openscholar_dir,
                           agenticscholar_url=args.agenticscholar_url, agenticscholar_fn=args.agenticscholar_fn,
                           codex_binary=args.codex_binary, codex_isolation=args.codex_isolation,
                           paperqa_embedding=args.paperqa_embedding)
    # The agents only need the corpus, the passage pool and the SPECTER2 dense retriever of each index.
    envs = EnvironmentCache(rt, layout, index_store_from_args(args), SystemOptions(), link_predictor="none")
    store = RecordStore(out / "answers")
    export_root = args.export_dir or (out / "corpus_shards")
    exported: Dict[str, Path] = {}
    built: Dict[Tuple[str, str], Any] = {}
    unavailable: Dict[str, str] = {}
    summary: Dict[str, Any] = {"environment": environment_summary(rt), "results": {}, "unavailable": unavailable}
    try:
        for bench in benchmarks:
            queries = layout.load_queries(bench, args.limit)
            env = envs.for_benchmark(bench)
            for key in agents:
                if key in unavailable:
                    continue
                export_dir = None
                if key in _NEEDS_EXPORT:
                    export_dir = exported.get(env.name)
                    if export_dir is None:
                        export_dir = exported[env.name] = export_root / env.name
                        if not (export_dir / "manifest.json").exists():
                            log.info("Exporting the %s corpus to %s", env.name, export_dir)
                            export_corpus_shards(index_corpus(env.index, env.name), export_dir)
                try:
                    agent = built.get((key, env.name))
                    if agent is None:
                        agent = built[(key, env.name)] = build_agent(key, env, options, export_dir)
                    recs = run_agent(agent, key, queries, benchmark=bench, store=store, resume=args.resume,
                                     fail_fast=args.fail_fast, valid_ids=env.index.papers.keys(),
                                     llm_config=rt.config.llm, fingerprint=agent_fingerprint(key, env, options))
                except AgentUnavailableError as err:
                    log.error("%s is unavailable: %s", system_display(key), err)
                    unavailable[key] = str(err)
                    continue
                rows = [r.efficiency for r in recs.values() if r.efficiency is not None and r.error is None]
                summary["results"].setdefault(bench, {})[system_display(key)] = {
                    "queries": len(queries), "answered": sum(1 for r in recs.values() if r.ok),
                    "errors": count_errors(recs), "efficiency": efficiency_summary(rows)}
                eff = summary["results"][bench][system_display(key)]["efficiency"]
                print(f"{bench} / {system_display(key)}: {len(recs)} answers, {count_errors(recs)} errors, "
                      f"{(eff.get('Calls') or 0):.1f} calls, {(eff.get('Tokens (K)') or 0):.1f}K tokens per query")
    finally:
        for agent in built.values():
            close = getattr(agent, "close", None)
            if callable(close):
                close()
    write_json(out / "summary.json", summary)
    print(f"Answers written to {out / 'answers'}")
    if unavailable:
        print("Unavailable agents: " + "; ".join(f"{system_display(k)}: {v}" for k, v in unavailable.items()))
    ran = [k for k in agents if k not in unavailable]
    return 0 if ran else 2


if __name__ == "__main__":
    sys.exit(main())
