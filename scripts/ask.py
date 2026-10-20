#!/usr/bin/env python
"""Ask ScholarPath a question (one-shot or interactive) over a built semantic index.

Algorithm 2 end to end: Step 1 query understanding (one lightweight LLM call
unless ``--set retrieval.llm_query_understanding=false``), taxonomy grounding,
hierarchy-bounded support-path search within the context budget B, and ONE
composition call. The answer is printed with its established / hypothesis
labels and citations, followed by the selected support paths
(concept -> paper -> section chunk for evidence; bridge, problem-side and
method-side paths of each complementary path set for hypotheses) and the
LLM calls, tokens, cost and latency of the query.

The query kind is detected by Step 1 (``--kind auto``) or forced:
``evidence``, ``hypothesis`` (discovery: candidate methods / problems that
were never combined) or ``assessment`` (is a given problem-method pairing
plausible?). In interactive mode, ``:kind <kind>``, ``:budget <B>`` and
``:quit`` change the settings or leave.

Paper setting::

    python scripts/ask.py --config configs/paper.yaml --corpus arxiv_subfields \\
        "Which methods could detect hidden disease superspreaders from incomplete contact-tracing logs?"

Quick offline run (synthetic index built by build_index.py)::

    python scripts/ask.py --index-dir artifacts/index --corpus arxiv_subfields \\
        "How has bayesian optimization been used for cardinality estimation?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scholarpath.experiments.qa import (  # noqa: E402
    DEFAULT_INDEX_ROOT,
    QUERY_KIND_CHOICES,
    QAEnvironment,
    SystemOptions,
    ask,
    format_ask,
    resolve_kind,
    support_paths,
)
from scholarpath.runtime import Runtime, add_common_args  # noqa: E402

log = logging.getLogger("ask")


def _answer(env: QAEnvironment, question: str, kind: str, budget, args, n: int) -> None:
    result = ask(env, question, kind=kind, budget=budget, query_id=f"ask-{n}")
    if args.json:
        print(json.dumps({
            "question": question,
            "kind": result.retrieval.stats.get("kind"),
            "answer": result.answer.to_dict(),
            "support_paths": support_paths(env.index, result.retrieval),
            "context_tokens": result.retrieval.context_tokens,
            "budget": result.retrieval.budget,
        }, indent=2, default=str))
    else:
        print(format_ask(result, env.index, show_paths=args.max_paths))
    sys.stdout.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("question", nargs="*", help="question (omit for an interactive session)")
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_ROOT, help="index root")
    ap.add_argument("--corpus", default="arxiv_subfields", help="index name under --index-dir")
    ap.add_argument("--index-path", type=Path, default=None, help="explicit index directory")
    ap.add_argument("--link-predictor", default="auto", help="'auto', 'none' or a directory")
    ap.add_argument("--kind", choices=QUERY_KIND_CHOICES, default="auto", help="query kind (default: detected)")
    ap.add_argument("--budget", type=int, default=None, help="context token budget B (default: config, 8,192)")
    ap.add_argument("--max-paths", type=int, default=20, help="support paths shown")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--no-query-llm", action="store_true",
                    help="LLM-free Step 1 (one LLM call per question: the composition call)")
    args = ap.parse_args(argv)

    rt = Runtime.from_args(args)
    if not args.verbose:
        logging.getLogger("scholarpath").setLevel(logging.WARNING)
    index_dir = args.index_path or (args.index_dir / args.corpus)
    options = SystemOptions(retrieval_overrides={"llm_query_understanding": False} if args.no_query_llm else {})
    env = QAEnvironment.load(index_dir, rt, link_predictor=args.link_predictor, options=options)
    kind, budget = args.kind, args.budget
    if args.question:
        _answer(env, " ".join(args.question), kind, budget, args, 0)
        return 0
    print(f"ScholarPath over {index_dir} ({len(env.index.papers):,} papers). "
          "Commands: :kind auto|evidence|hypothesis|assessment, :budget N, :quit")
    n = 0
    while True:
        try:
            line = input("? ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            break
        if line.startswith(":kind"):
            try:
                value = line.split(maxsplit=1)[1]
                resolve_kind(value)
                kind = value
                print(f"kind = {kind}")
            except (IndexError, ValueError) as err:
                print(f"usage: :kind {'|'.join(QUERY_KIND_CHOICES)} ({err})")
            continue
        if line.startswith(":budget"):
            try:
                budget = int(line.split(maxsplit=1)[1])
                print(f"budget = {budget}")
            except (IndexError, ValueError):
                print("usage: :budget <tokens>")
            continue
        n += 1
        _answer(env, line, kind, budget, args, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
