"""Query-answering experiments of Section 5.3 (Tables 2-5 and 7, Figs. 5-6, App. C.1-C.4).

This module holds the reusable, unit-tested logic behind the query-answering
scripts; the scripts themselves only parse arguments and choose output paths:

=====================================  ==========================================================
script                                 reproduces
=====================================  ==========================================================
``scripts/build_index.py``             multi-granularity semantic index I (Sec. 2.1) per corpus
``scripts/train_link_predictor.py``    temporal link predictor s_theta (Sec. 4.3, App. B.5)
``scripts/build_heldout_benchmark.py`` Held-Out Hypothesis benchmark (Sec. 5.3, App. C.1)
``scripts/run_retrieval_eval.py``      Table 2 (Recall@B / NDCG@B at B = 8,192) and Fig. 5
``scripts/run_qa_eval.py``             Table 3 (evidence QA), Table 4 (hypothesis QA), Fig. 6
``scripts/run_agents.py``              unbudgeted agents of Tables 3-5 (PaperQA2, OpenScholar, ...)
``scripts/run_efficiency.py``          Table 5 and the operator statistics of Sec. 5.3.3
``scripts/run_retrieval_ablation.py``  Table 7 (App. C.4, "Query Answering Ablation")
``scripts/ask.py``                     interactive / one-shot question answering
=====================================  ==========================================================

Building blocks
---------------
* **Benchmarks and corpora** (:data:`BENCHMARKS`, :data:`CORPORA`,
  :class:`DataLayout`). Table 1: ResearchQA (776 queries) and ScholarQABench
  (ScholarQA-CS 100 + ScholarQA-Multi 108) are indexed and searched over
  arXiv-Subfields (148,400 papers); TaxoBench (72) and CHIME (100) over their
  own pooled corpora (3,044 and 2,146 papers); the held-out hypothesis
  benchmark (47 queries) over the *reduced* arXiv-Subfields corpus of
  App. C.1. Organization queries (TaxoBench, CHIME) carry
  ``metadata["task"] == "organization"``: their answer is a topic hierarchy.
* **Indexes** (:class:`IndexStore`, :func:`build_index`): one
  :class:`~scholarpath.index.semantic_index.SemanticIndex` per corpus under
  ``artifacts/index/<corpus>/``, with the trained link predictor next to it
  (``link_predictor/``).
* **Systems** (:data:`SYSTEMS`, :class:`QAEnvironment`): ScholarPath
  (Algorithm 2), the budgeted baselines of :mod:`scholarpath.baselines`, the
  unassisted answer LLM, the Table 7 ablation variants of ScholarPath, and the
  unbudgeted agent adapters (:func:`build_agent`). Every budgeted method packs
  at most B context tokens and composes with ONE LLM call (Sec. 5.3,
  "Baselines"); organization queries use the hierarchy-composition call of
  :mod:`scholarpath.prompts.organize` over the same <= B context.
* **Runs** (:func:`run_queries`, :func:`run_agent`, :class:`RecordStore`):
  per-query answers / retrievals are cached as JSONL together with their
  per-query efficiency (LLM calls, tokens, cost, latency; Table 5), so that
  evaluation can be re-run without regenerating, and interrupted runs resume.
* **Evaluation** (:func:`evaluate_retrieval_records`, :func:`evaluate_benchmark`,
  :class:`JudgeSet`): Recall@B / NDCG@B (Table 2), the official QA protocols
  (Table 3), Disc and the three-judge scores (Table 4). Judges own private
  cost meters, so judge spending never mixes with system cost.
* **Efficiency** (:func:`collect_efficiency`, :func:`efficiency_summary`,
  :func:`measure_operator_stats`): Table 5 and the Sec. 5.3.3 statistics
  (postings pruned by the early-terminated join, candidate nodes pruned by
  hierarchy-bounded search, ms per query).

Paper inconsistency (Table 5 vs. Algorithm 2)
---------------------------------------------
Table 5 reports 1.0 LLM call per query for ScholarPath ("zero retrieval LLM
calls and a single composition call", Sec. 5.3.3), whereas Algorithm 2,
Step 1 issues one lightweight query-understanding call
(``RetrievalConfig.llm_query_understanding``, default ``True``), i.e. two
calls per query. Efficiency summaries therefore report both the total calls
and the composition-only calls; the scripts' ``--no-query-llm`` flag (or
``--set retrieval.llm_query_understanding=false``) runs the LLM-free Step 1
that matches Table 5's call count.

Reproducibility of cached runs
------------------------------
Every cached record carries a fingerprint of the configuration that produced
it (:meth:`QAEnvironment.fingerprint`: system variant, retrieval / system
options, link predictor, LLM, embedder and index); ``--resume`` reuses a
record only when the fingerprint matches, and a run without ``--resume``
refuses to replace records of another configuration
(:class:`CacheConflictError`) unless ``--force`` is given. Indexes record
their embedder (:func:`embedding_identity`); query vectors of another
embedder are refused (:class:`EmbeddingMismatchError`). Failed or missing
answers score 0 on every metric (:func:`score_missing_as_zero`). Table 5 latencies are computed
uniformly for every budgeted system (:func:`online_latency`): wall time minus
offline indexing, plus the recorded latency of calls served from the LLM
response cache, so that a warm cache never favours one system.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from scholarpath.baselines.common import (
    EXCLUDED_QUERY_COMPONENTS,
    PassagePool,
    index_build_seconds,
    split_usage,
)
from scholarpath.config import EmbeddingConfig, EvaluationConfig, LLMConfig, RetrievalConfig
from scholarpath.data.corpus import Corpus, load_queries, read_json, write_json
from scholarpath.embeddings import l2_normalize
from scholarpath.evaluation.efficiency import EfficiencyTracker, QueryEfficiency, summarize_efficiency
from scholarpath.evaluation.judges import JudgePanel, JudgeSpec
from scholarpath.evaluation.reporting import Column, TableSpec
from scholarpath.evaluation.retrieval_metrics import RetrievalEvaluation, evaluate_retrieval
from scholarpath.experiments.common import _json_default, matplotlib_available, restrict_taxonomy, split_csv
from scholarpath.index.semantic_index import SemanticIndex, parse_key
from scholarpath.llm import sum_usage
from scholarpath.retrieval.link_prediction import LinkPredictor
from scholarpath.retrieval.result import RetrievalResult, RetrievedUnit
from scholarpath.runtime import Runtime
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.types import (
    EVIDENCE,
    HYPOTHESIS_ASSESSMENT,
    HYPOTHESIS_DISCOVERY,
    Answer,
    AnswerItem,
    Chunk,
    Paper,
    Query,
)

log = logging.getLogger("scholarpath.experiments.qa")

PathLike = Union[str, Path]

#: Context token budget B of the paper (Definition 2, Sec. 5.1).
PAPER_BUDGET = 8192
#: Budgets of Figs. 5-6 ("B varies from 2K to 16K").
SWEEP_BUDGETS = (2048, 4096, 6144, 8192, 16384)
#: Default processed-data directory (``data/README.md``) and index root.
DEFAULT_DATA_DIR = Path("data") / "processed"
DEFAULT_INDEX_ROOT = Path("artifacts") / "index"
#: Sub-directory of an index directory holding the trained link predictor.
LINK_PREDICTOR_DIR = "link_predictor"
#: Budget tag of unbudgeted runs (agents).
UNBUDGETED = "unbudgeted"


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

SCHOLARPATH = "scholarpath"
NO_RETRIEVAL = "no_retrieval"
SIMULATED = "simulated"


@dataclass(frozen=True)
class SystemSpec:
    """A compared system: key (CLI / file names), display name (tables) and family.

    ``family`` is ``"scholarpath"`` (Algorithm 2 or one of its Table 7
    variants), ``"baseline"`` (a budgeted retriever of
    :mod:`scholarpath.baselines`), ``"no_retrieval"`` (the unassisted answer
    LLM of Table 4) or ``"agent"`` (an unbudgeted agent system).

    ``link_prior`` (ScholarPath variants) selects the cross-taxonomy prior
    s_theta of Eq. (8): ``"trained"`` = the temporally trained link predictor
    of the index (App. B.5; the untrained prior when none was trained), or
    ``"untrained"`` = the text-similarity prior
    :class:`~scholarpath.retrieval.link_prediction.EmbeddingLinkPredictor`
    (Table 7, "w/o temporal link prediction").
    """

    key: str
    display: str
    family: str
    overrides: Tuple[Tuple[str, Any], ...] = ()  # RetrievalConfig overrides (ScholarPath variants)
    link_prior: str = "trained"

    @property
    def retrieves(self) -> bool:
        """Whether the system returns a budgeted :class:`RetrievalResult` (Table 2 / Fig. 5)."""
        return self.family in ("scholarpath", "baseline")

    @property
    def budgeted(self) -> bool:
        return self.family != "agent"


#: ScholarPath variants (App. C.4): key -> (display name, RetrievalConfig overrides, link prior).
#:
#: * "w/o temporal link prediction" keeps the link prior but replaces the
#:   temporally trained s_theta by the untrained text-similarity prior
#:   (EmbeddingLinkPredictor), so candidates are still scored ("confirming the
#:   importance of structural analogies over text similarity", App. C.4;
#:   Table 7 still reports 348 scored candidates);
#: * "w/o bound-guided search" is the unpruned scan that scores every candidate
#:   node in scope (same selections, Proposition 7);
#: * ``scholarpath_no_link_prior`` (not a Table 7 row) drops the prior
#:   altogether (``use_link_prediction=False``: eta_hat = phi).
ABLATIONS: Dict[str, Tuple[str, Tuple[Tuple[str, Any], ...], str]] = {
    SCHOLARPATH: ("ScholarPath", (), "trained"),
    "scholarpath_no_link_prediction": ("w/o temporal link prediction", (), "untrained"),
    "scholarpath_no_bound_guided_search": ("w/o bound-guided search", (("use_bound_guided_search", False),),
                                           "trained"),
    "scholarpath_no_role_aware_granularity": ("w/o role-aware granularity",
                                              (("use_role_aware_granularity", False),), "trained"),
    "scholarpath_no_shared_cost_update": ("w/o shared-cost reuse update", (("use_shared_cost_update", False),),
                                          "trained"),
    "scholarpath_no_link_prior": ("w/o link prior (eta = phi)", (("use_link_prediction", False),), "trained"),
}
#: The rows of Table 7 (paper order).
TABLE7_VARIANTS = (SCHOLARPATH, "scholarpath_no_link_prediction", "scholarpath_no_bound_guided_search",
                   "scholarpath_no_role_aware_granularity", "scholarpath_no_shared_cost_update")

SYSTEMS: Dict[str, SystemSpec] = {
    **{k: SystemSpec(k, d, "scholarpath", o, lp) for k, (d, o, lp) in ABLATIONS.items()},
    "bm25": SystemSpec("bm25", "BM25", "baseline"),
    "specter": SystemSpec("specter", "SPECTER", "baseline"),
    "dense_rerank": SystemSpec("dense_rerank", "Dense + reranker", "baseline"),
    "raptor": SystemSpec("raptor", "RAPTOR", "baseline"),
    "lightrag": SystemSpec("lightrag", "LightRAG", "baseline"),
    "keyword": SystemSpec("keyword", "Keyword search", "baseline"),
    "dense": SystemSpec("dense", "Dense retrieval", "baseline"),
    NO_RETRIEVAL: SystemSpec(NO_RETRIEVAL, "Answer LLM, no retrieval", "no_retrieval"),
    "paperqa2": SystemSpec("paperqa2", "PaperQA2", "agent"),
    "openscholar": SystemSpec("openscholar", "OpenScholar", "agent"),
    "agenticscholar": SystemSpec("agenticscholar", "AgenticScholar", "agent"),
    "codex": SystemSpec("codex", "Codex", "agent"),
    SIMULATED: SystemSpec(SIMULATED, "SimulatedAgent (offline)", "agent"),
}

#: Default rows of Table 2 / Fig. 5 (paper order).
TABLE2_SYSTEMS = ("bm25", "specter", "dense_rerank", "raptor", "lightrag", SCHOLARPATH)
#: Budgeted rows of Tables 3-5 and Fig. 6; Table 4 adds the unassisted answer LLM.
QA_SYSTEMS = ("keyword", "dense", "lightrag", SCHOLARPATH)
TABLE4_SYSTEMS = (NO_RETRIEVAL,) + QA_SYSTEMS
#: The unbudgeted agents of Tables 3-5 (paper order of Table 5).
AGENT_SYSTEMS = ("openscholar", "paperqa2", "agenticscholar", "codex")
#: Row order of the unbudgeted agents in Tables 3-4 (Table 5 uses :data:`AGENT_SYSTEMS`).
TABLE34_AGENT_ORDER = ("paperqa2", "openscholar", "agenticscholar", "codex")

_SYSTEM_ALIASES = {
    "scholar_path": SCHOLARPATH, "sp": SCHOLARPATH, "spectre": "specter", "specter1": "specter",
    "specter2": "dense", "dense_retrieval": "dense", "dense_reranker": "dense_rerank", "rerank": "dense_rerank",
    "dense_plus_reranker": "dense_rerank", "keyword_search": "keyword", "answer_llm": NO_RETRIEVAL,
    "llm": NO_RETRIEVAL, "none": NO_RETRIEVAL, "answer_llm_no_retrieval": NO_RETRIEVAL, "paper_qa2": "paperqa2",
    "paperqa": "paperqa2", "agentic_scholar": "agenticscholar", "open_scholar": "openscholar",
    "simulatedagent": SIMULATED,
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def resolve_system(name: str) -> str:
    """Canonical system key of a CLI name, key, alias or display name (``"Dense + reranker"``)."""
    s = _slug(name)
    if s in SYSTEMS:
        return s
    if s in _SYSTEM_ALIASES:
        return _SYSTEM_ALIASES[s]
    for key, spec in SYSTEMS.items():
        if _slug(spec.display) == s:
            return key
    raise ValueError(f"Unknown system {name!r}; expected one of {sorted(SYSTEMS)}")


def system_display(key: str) -> str:
    spec = SYSTEMS.get(key)
    return spec.display if spec is not None else key


def resolve_systems(names: Union[str, Sequence[str], None], default: Sequence[str]) -> List[str]:
    """Expand a comma list (or sequence) of system names; ``"all"`` keeps ``default``.

    Aliases: ``agents`` (the four paper agents), ``table7`` (the Table 7
    rows) and ``ablations`` (every ScholarPath variant).
    """
    if names is None:
        return list(default)
    items = split_csv(names) if isinstance(names, str) else [str(n) for n in names]
    if not items or items == ["all"]:
        return list(default)
    out: List[str] = []
    for n in items:
        if _slug(n) == "agents":
            out.extend(AGENT_SYSTEMS)
        elif _slug(n) == "table7":
            out.extend(TABLE7_VARIANTS)
        elif _slug(n) == "ablations":
            out.extend(ABLATIONS)
        else:
            out.append(resolve_system(n))
    return list(dict.fromkeys(out))


def order_systems(keys: Iterable[str], order: Sequence[str]) -> List[str]:
    """``keys`` in the paper's row order ``order``, followed by any other keys in their given order."""
    keys = list(dict.fromkeys(keys))
    order = list(dict.fromkeys(order))
    return [k for k in order if k in keys] + [k for k in keys if k not in order]


# ---------------------------------------------------------------------------
# Benchmarks and corpora (Table 1, App. C.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusSpec:
    """A retrieval corpus of the processed-data layout (paths relative to the data dir)."""

    name: str
    papers: str
    chunks: Optional[str] = None
    description: str = ""


CORPORA: Dict[str, CorpusSpec] = {
    "arxiv_subfields": CorpusSpec("arxiv_subfields", "arxiv_subfields/papers.jsonl.gz",
                                  "arxiv_subfields/chunks.jsonl.gz", "arXiv-Subfields (148,400 papers)"),
    "arxiv_subfields_full": CorpusSpec("arxiv_subfields_full", "arxiv_subfields_full/papers.jsonl.gz",
                                       "arxiv_subfields_full/chunks.jsonl.gz",
                                       "complete corpus incl. held-out papers (synthetic layout)"),
    "taxobench": CorpusSpec("taxobench", "taxobench/papers.jsonl.gz", None, "TaxoBench pooled corpus (3,044 papers)"),
    "chime": CorpusSpec("chime", "chime/papers.jsonl.gz", None, "CHIME pooled studies (2,146 papers)"),
    "heldout": CorpusSpec("heldout", "heldout/papers.jsonl.gz", "heldout/chunks.jsonl.gz",
                          "reduced arXiv-Subfields corpus of App. C.1"),
}

#: Evaluation protocols of the benchmarks (Sec. 5.3 "Metrics", App. C.3).
PROTOCOL_RESEARCHQA = "researchqa"  # Rub (rubric coverage) + Cite (citation F1 vs. gold)
PROTOCOL_CS = "scholarqa_cs"  # Corr (rubric score) + Cite (statement-level citation F1)
PROTOCOL_MULTI = "scholarqa_multi"  # Corr (Prometheus-2) + Cite
PROTOCOL_ORGANIZATION = "organization"  # TaxoBench ARI / Sem, CHIME Cat / Assn
PROTOCOL_HYPOTHESIS = "hypothesis"  # Disc (Hit@10) + three-judge Feas. / Nov. / Sound.


@dataclass(frozen=True)
class BenchmarkSpec:
    """A benchmark: its query file, retrieval corpus, table group and evaluation protocol."""

    key: str
    display: str
    queries: str
    corpus: str
    group: str
    protocol: str

    @property
    def is_hypothesis(self) -> bool:
        return self.protocol == PROTOCOL_HYPOTHESIS

    @property
    def is_organization(self) -> bool:
        return self.protocol == PROTOCOL_ORGANIZATION


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "researchqa": BenchmarkSpec("researchqa", "ResearchQA", "researchqa/queries.jsonl", "arxiv_subfields",
                                "researchqa", PROTOCOL_RESEARCHQA),
    "scholarqa_cs": BenchmarkSpec("scholarqa_cs", "ScholarQA-CS", "scholarqabench/cs.jsonl", "arxiv_subfields",
                                  "scholarqabench", PROTOCOL_CS),
    "scholarqa_multi": BenchmarkSpec("scholarqa_multi", "ScholarQA-Multi", "scholarqabench/multi.jsonl",
                                     "arxiv_subfields", "scholarqabench", PROTOCOL_MULTI),
    "taxobench": BenchmarkSpec("taxobench", "TaxoBench", "taxobench/queries.jsonl", "taxobench", "taxobench",
                               PROTOCOL_ORGANIZATION),
    "chime": BenchmarkSpec("chime", "CHIME", "chime/queries.jsonl", "chime", "chime", PROTOCOL_ORGANIZATION),
    "heldout": BenchmarkSpec("heldout", "Held-Out Hypothesis", "heldout/queries.jsonl", "heldout", "heldout",
                             PROTOCOL_HYPOTHESIS),
    # Evidence queries of the synthetic smoke-test layout (scripts/make_synthetic_data.py).
    "synthetic": BenchmarkSpec("synthetic", "Synthetic evidence", "evidence/queries.jsonl", "arxiv_subfields",
                               "synthetic", PROTOCOL_RESEARCHQA),
}

#: Evidence benchmarks of Tables 2-3 (``synthetic`` only in the smoke-test layout).
EVIDENCE_BENCHMARKS = ("researchqa", "scholarqa_cs", "scholarqa_multi", "taxobench", "chime", "synthetic")
HYPOTHESIS_BENCHMARKS = ("heldout",)
_BENCHMARK_ALIASES = {
    "scholarqabench": ("scholarqa_cs", "scholarqa_multi"),
    "scholarqa": ("scholarqa_cs", "scholarqa_multi"),
    "cs": ("scholarqa_cs",),
    "multi": ("scholarqa_multi",),
    "evidence": EVIDENCE_BENCHMARKS,
    "hypothesis": HYPOTHESIS_BENCHMARKS,
    "held_out": ("heldout",),
    "all": tuple(BENCHMARKS),
}

#: Table groups (columns of Tables 2-3), their display names and Table 3 metrics.
GROUP_ORDER = ("researchqa", "scholarqabench", "taxobench", "chime", "synthetic")
GROUP_DISPLAY = {"researchqa": "ResearchQA", "scholarqabench": "ScholarQA", "taxobench": "TaxoBench",
                 "chime": "CHIME", "synthetic": "Synthetic", "heldout": "Held-Out"}
GROUP_QA_METRICS = {"researchqa": ("Rub", "Cite"), "scholarqabench": ("Corr", "Cite"),
                    "taxobench": ("ARI", "Sem"), "chime": ("Cat", "Assn"), "synthetic": ("Rub", "Cite")}


def resolve_benchmarks(names: Union[str, Sequence[str], None], available: Optional[Sequence[str]] = None,
                       default: Optional[Sequence[str]] = None) -> List[str]:
    """Expand benchmark names / aliases (``scholarqabench`` = CS + Multi, ``evidence``, ``all``).

    With ``names=None`` the ``default`` keys are used. When ``available`` is
    given, benchmarks whose query file is missing are dropped (with a log
    message) unless they were requested explicitly by key.
    """
    explicit = names is not None
    items = (split_csv(names) if isinstance(names, str) else list(names)) if names is not None else list(
        default if default is not None else BENCHMARKS)
    out: List[str] = []
    for n in items:
        s = _slug(n)
        if s in _BENCHMARK_ALIASES:
            out.extend(_BENCHMARK_ALIASES[s])
        elif s in BENCHMARKS:
            out.append(s)
        else:
            raise ValueError(f"Unknown benchmark {n!r}; expected one of {sorted(BENCHMARKS)} or "
                             f"{sorted(_BENCHMARK_ALIASES)}")
    out = list(dict.fromkeys(out))
    if available is not None:
        avail = set(available)
        missing = [b for b in out if b not in avail]
        requested = {_slug(n) for n in items} if explicit else set()
        hard = [b for b in missing if b in requested]
        if hard:
            raise FileNotFoundError(f"No query file for benchmark(s) {hard}; run the data preparation scripts "
                                    "(see data/README.md)")
        if missing:
            log.info("Skipping benchmarks without query files: %s", missing)
        out = [b for b in out if b in avail]
    return out


class DataLayout:
    """Paths of the processed-data layout (``data/README.md``; also the synthetic layout)."""

    def __init__(self, root: PathLike = DEFAULT_DATA_DIR):
        self.root = Path(root)

    def path(self, rel: str) -> Path:
        return self.root / rel

    def query_file(self, benchmark: str) -> Path:
        return self.path(BENCHMARKS[benchmark].queries)

    def benchmark_dir(self, benchmark: str) -> Path:
        return self.query_file(benchmark).parent

    def has_benchmark(self, benchmark: str) -> bool:
        return self.query_file(benchmark).exists()

    def available_benchmarks(self) -> List[str]:
        return [b for b in BENCHMARKS if self.has_benchmark(b)]

    def load_queries(self, benchmark: str, limit: Optional[int] = None) -> List[Query]:
        """Queries of a benchmark in file order (the first ``limit`` with ``limit``)."""
        path = self.query_file(benchmark)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; prepare the benchmark first (data/README.md)")
        queries = load_queries(path)
        return queries[:limit] if limit is not None else queries

    def has_corpus(self, name: str) -> bool:
        spec = CORPORA.get(name)
        return spec is not None and self.path(spec.papers).exists()

    def is_synthetic_layout(self) -> bool:
        """The smoke-test layout of ``scripts/make_synthetic_data.py``.

        It ships the complete corpus as ``arxiv_subfields_full`` and an
        ``arxiv_subfields`` corpus from which its held-out papers are already
        removed; in the paper layout ``arxiv_subfields`` IS the complete corpus.
        """
        return self.has_corpus("arxiv_subfields_full")

    def complete_corpus_name(self) -> str:
        """The complete arXiv-Subfields corpus before the App. C.1 reduction.

        In the paper layout this is ``arxiv_subfields``; the synthetic layout
        additionally ships ``arxiv_subfields_full`` because its
        ``arxiv_subfields`` already excludes the synthetic held-out papers.
        """
        return "arxiv_subfields_full" if self.is_synthetic_layout() else "arxiv_subfields"

    def heldout_reduction_available(self) -> bool:
        """Whether the layout describes the App. C.1 reduction (reduced corpus, its ids, or the removed ids)."""
        return (self.has_corpus("heldout") or self.path("heldout/reduced_corpus_ids.json").exists()
                or self.path("heldout/removed_papers.json").exists())

    def load_corpus(self, name: str, limit: Optional[int] = None, with_chunks: bool = True) -> Corpus:
        """Load a corpus by name (see :data:`CORPORA`).

        ``heldout`` (the reduced corpus of App. C.1: "every supporting paper
        and the papers published after them are removed before any index is
        built") resolves to (1) ``heldout/papers.jsonl.gz`` written by
        ``build_heldout_benchmark.py``, else (2) the complete corpus restricted
        to ``heldout/reduced_corpus_ids.json``, else (3) the complete corpus
        minus ``heldout/removed_papers.json``, else (4) in the synthetic
        layout only, ``arxiv_subfields`` as is (its held-out papers are already
        removed). Otherwise the reduction is unknown and a
        :class:`FileNotFoundError` is raised, since the complete corpus would
        leak the held-out papers.
        """
        if name == "heldout" and not self.has_corpus("heldout"):
            ids_file = self.path("heldout/reduced_corpus_ids.json")
            removed_file = self.path("heldout/removed_papers.json")
            if ids_file.exists() or removed_file.exists():
                complete = self.load_corpus(self.complete_corpus_name(), with_chunks=with_chunks)
                if ids_file.exists():
                    ids = [str(x) for x in read_json(ids_file)]
                else:
                    removed = {str(x) for x in read_json(removed_file)}
                    ids = [pid for pid in complete.ids if pid not in removed]
                    log.info("Held-out corpus = %s minus %d removed papers (%s)", complete.name,
                             len(complete) - len(ids), removed_file)
                reduced = complete.subset(ids, name="heldout")
                return reduced.subset(reduced.ids[:limit], name="heldout") if limit is not None else reduced
            if self.is_synthetic_layout():
                log.info("Synthetic layout: the held-out corpus is arxiv_subfields (held-out papers already removed)")
                return self.load_corpus("arxiv_subfields", limit, with_chunks)
            raise FileNotFoundError(
                f"No reduced held-out corpus under {self.path('heldout')} (papers.jsonl.gz, reduced_corpus_ids.json "
                "or removed_papers.json); the complete corpus would leak the held-out papers. Run "
                "scripts/build_heldout_benchmark.py build first (App. C.1).")
        spec = CORPORA.get(name)
        if spec is None:
            raise KeyError(f"Unknown corpus {name!r}; expected one of {sorted(CORPORA)} (or pass --papers)")
        papers = self.path(spec.papers)
        if not papers.exists():
            raise FileNotFoundError(f"{papers} not found; prepare the corpus first (data/README.md)")
        chunks = self.path(spec.chunks) if (with_chunks and spec.chunks) else None
        return Corpus.load(papers, limit=limit, chunks_path=chunks, name=name)


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------


class IndexStore:
    """Semantic indexes under ``<root>/<corpus>/`` (``SemanticIndex.save`` layout).

    ``overrides`` maps a corpus name to an explicit index directory.
    """

    def __init__(self, root: PathLike = DEFAULT_INDEX_ROOT, overrides: Optional[Mapping[str, PathLike]] = None):
        self.root = Path(root)
        self.overrides = {k: Path(v) for k, v in (overrides or {}).items()}

    def path(self, name: str) -> Path:
        return self.overrides.get(name, self.root / name)

    def exists(self, name: str) -> bool:
        return (self.path(name) / "problem_taxonomy.json").exists()

    def resolve(self, name: str, layout: Optional[DataLayout] = None) -> str:
        """Index name used for corpus ``name``.

        The held-out corpus falls back to the ``arxiv_subfields`` index only
        in the synthetic layout without any reduction files (its
        ``arxiv_subfields`` already excludes the held-out papers); otherwise a
        missing held-out index is an error, since searching the complete
        corpus would leak the held-out papers.
        """
        if self.exists(name) or name in self.overrides:
            return name
        if (name == "heldout" and layout is not None and layout.is_synthetic_layout()
                and not layout.heldout_reduction_available()):
            if self.exists("arxiv_subfields"):
                log.warning("No held-out index; using the arxiv_subfields index (synthetic layout)")
                return "arxiv_subfields"
        raise FileNotFoundError(f"No index for corpus {name!r} at {self.path(name)}; "
                                f"run: python scripts/build_index.py --corpus {name}")

    def load(self, name: str) -> SemanticIndex:
        return SemanticIndex.load(self.path(name))


def link_predictor_dir(index_dir: PathLike) -> Path:
    return Path(index_dir) / LINK_PREDICTOR_DIR


def load_link_predictor(index_dir: PathLike, spec: Optional[str] = "auto") -> Optional[LinkPredictor]:
    """The trained link predictor of an index (``spec``: ``auto`` | ``none`` | a directory).

    ``auto`` loads ``<index_dir>/link_predictor/`` when present; otherwise
    ScholarPath falls back to the untrained description prior
    (:class:`~scholarpath.retrieval.link_prediction.EmbeddingLinkPredictor`).
    """
    if spec is None or str(spec).lower() in ("none", "off", "false"):
        return None
    path = link_predictor_path(index_dir, spec)
    if path is None:
        return None
    lp = LinkPredictor.load(path)
    lp.source_id = link_predictor_identity(path)  # type: ignore[attr-defined]  # cache fingerprints
    return lp


def link_predictor_path(index_dir: PathLike, spec: Optional[str] = "auto") -> Optional[Path]:
    """Directory of the link predictor selected by ``spec`` (see :func:`load_link_predictor`), or ``None``."""
    if spec is None or str(spec).lower() in ("none", "off", "false"):
        return None
    path = link_predictor_dir(index_dir) if str(spec).lower() == "auto" else Path(spec)
    if not (path / "link_predictor.json").exists():
        if str(spec).lower() != "auto":
            raise FileNotFoundError(f"No link predictor at {path}")
        return None
    return path


def link_predictor_identity(path: PathLike) -> str:
    """Content hash of a saved link predictor (part of the cache fingerprint of ScholarPath runs)."""
    h = hashlib.sha1()
    for f in sorted(Path(path).iterdir()):
        if f.is_file() and f.name != "training_report.json":
            h.update(f.name.encode("utf-8"))
            h.update(f.read_bytes())
    return f"trained:{h.hexdigest()[:16]}"


def embed_papers(corpus: Corpus, embedder: Any, batch_size: int = 1024) -> Dict[str, np.ndarray]:
    """Unit-norm embeddings of the paper summaries (title + abstract), shared by builder and index."""
    ids = corpus.ids
    out: Dict[str, np.ndarray] = {}
    for s in range(0, len(ids), batch_size):
        part = ids[s : s + batch_size]
        vecs = l2_normalize(np.asarray(embedder.encode(corpus.texts(part)), dtype=np.float32))
        out.update(zip(part, vecs))
    return out


def embedding_identity(config: EmbeddingConfig) -> Dict[str, Any]:
    """What makes paper and query vectors comparable: backend, model / adapter (or the hashing dimension)."""
    backend = str(config.backend or "hashing").lower()
    backend = "sentence-transformers" if backend == "st" else backend
    out: Dict[str, Any] = {"backend": backend}
    if backend == "hashing":
        out["dim"] = int(config.dim)
    else:
        out["model"] = config.model_name
    if backend == "specter2":
        out["adapter"] = config.adapter
    return out


class EmbeddingMismatchError(ValueError):
    """The runtime embedder differs from the one that encoded the index's paper / chunk vectors."""


def embedding_mismatches(index_identity: Mapping[str, Any], runtime_identity: Mapping[str, Any]) -> List[str]:
    """Fields on which an index's embedder and the runtime's differ (only fields recorded by both)."""
    out = []
    for key in ("backend", "model", "adapter", "dim"):
        if key in index_identity and key in runtime_identity and index_identity[key] != runtime_identity[key]:
            out.append(f"{key}: index {index_identity[key]!r} vs. runtime {runtime_identity[key]!r}")
    return out


@dataclass
class IndexBuild:
    """Output of :func:`build_index`."""

    index: SemanticIndex
    problem: Optional[Any] = None  # BuildResult of T_P (None for prebuilt taxonomies)
    method: Optional[Any] = None  # BuildResult of T_M
    summary: Dict[str, Any] = field(default_factory=dict)

    def save(self, directory: PathLike) -> Path:
        d = Path(directory)
        self.index.save(d)
        if self.problem is not None:
            self.problem.save(d / "taxonomy_problem")
        if self.method is not None:
            self.method.save(d / "taxonomy_method")
        write_json(d / "build_summary.json", self.summary)
        return d


def build_index(
    corpus: Corpus,
    runtime: Runtime,
    *,
    taxonomy: str = "scholarpath",
    problem_taxonomy: Optional[Taxonomy] = None,
    method_taxonomy: Optional[Taxonomy] = None,
    name: str = "index",
    allow_foreign_papers: bool = True,
) -> IndexBuild:
    """Corpus -> dual taxonomies (T_P, T_M) -> multi-granularity semantic index I (Sec. 2.1).

    ``taxonomy`` selects the construction method: ``"scholarpath"``
    (Algorithm 1 via :func:`~scholarpath.taxonomy.builder.build_dual_taxonomies`,
    the paper's setting) or a baseline name of
    :func:`~scholarpath.taxonomy.baselines.parse_method` (``"full"``,
    ``"kmeans@0.2"``, ...). Prebuilt ``problem_taxonomy`` / ``method_taxonomy``
    skip construction; they are restricted to the corpus' papers and concepts
    left without papers are dropped (:func:`restrict_taxonomy`). With
    ``allow_foreign_papers=False`` a prebuilt taxonomy that references papers
    outside the corpus is refused instead -- the held-out index must not be
    built from taxonomies that saw the held-out papers (App. C.1). Paper
    embeddings are computed once and shared by the builder and the index.
    The summary records sizes, wall times and the LLM cost per component
    (Eq. 15).
    """
    rt = runtime
    if (problem_taxonomy is None) != (method_taxonomy is None):
        raise ValueError("Pass both a problem and a method taxonomy, or neither")
    restriction: Dict[str, Any] = {}
    if problem_taxonomy is not None and method_taxonomy is not None:
        ids = set(corpus.ids)
        foreign = (problem_taxonomy.all_papers() | method_taxonomy.all_papers()) - ids
        if foreign and not allow_foreign_papers:
            raise ValueError(
                f"The prebuilt taxonomies reference {len(foreign)} papers outside the corpus {corpus.name!r} "
                f"(e.g. {sorted(foreign)[:3]}); build the taxonomies over this corpus instead")
    t0 = time.perf_counter()
    before = rt.meter.snapshot()
    vectors = embed_papers(corpus, rt.embedder)
    t_embed = time.perf_counter() - t0
    problem_res = method_res = None
    t1 = time.perf_counter()
    if problem_taxonomy is not None and method_taxonomy is not None:
        method_name = "prebuilt"
        P = restrict_taxonomy(problem_taxonomy, corpus.ids)
        M = restrict_taxonomy(method_taxonomy, corpus.ids)
        restriction = {"problem": P.metadata.get("restriction"), "method": M.metadata.get("restriction")}
        for facet, info in restriction.items():
            if info and (info["papers_removed"] or info["concepts_removed"]):
                log.warning("Prebuilt %s taxonomy restricted to the corpus: %d foreign papers and %d concepts "
                            "without papers in the corpus removed", facet, info["papers_removed"],
                            info["concepts_removed"])
    elif _slug(taxonomy) == "scholarpath":
        from scholarpath.taxonomy.builder import build_dual_taxonomies

        method_name = "scholarpath"
        problem_res, method_res = build_dual_taxonomies(rt.llm, rt.embedder, rt.config.builder, corpus,
                                                        paper_vectors=vectors)
        P, M = problem_res.taxonomy, method_res.taxonomy
    else:
        from scholarpath.taxonomy.baselines import baseline_config, build_dual_baselines

        method_name = taxonomy
        cfg = baseline_config(taxonomy, rt.config.baseline)
        problem_res, method_res = build_dual_baselines(rt.llm, rt.embedder, cfg, corpus, paper_vectors=vectors)
        P, M = problem_res.taxonomy, method_res.taxonomy
    t_tax = time.perf_counter() - t1
    t2 = time.perf_counter()
    index = SemanticIndex.build(P, M, corpus.papers, rt.embedder, rt.config.index, paper_embeddings=vectors)
    t_index = time.perf_counter() - t2
    usage = rt.meter.diff(before)
    total = sum_usage(usage)
    summary = {
        "name": name,
        "corpus": corpus.name,
        "taxonomy_method": method_name,
        "index": index.stats(),
        "llm": {"provider": rt.config.llm.provider, "model": rt.config.llm.model},
        "embedding": embedding_identity(rt.config.embedding),
        "cost_usd": total.cost_usd,
        "llm_calls": total.calls,
        "tokens": total.total_tokens,
        "usage_by_component": {k: v.to_dict() for k, v in usage.items()},
        "seconds": {"embedding": t_embed, "taxonomies": t_tax, "index": t_index,
                    "total": time.perf_counter() - t0},
    }
    if restriction:
        summary["prebuilt_restriction"] = restriction
    for facet, res in (("problem", problem_res), ("method", method_res)):
        if res is not None:
            summary[f"{facet}_build"] = res.summary()
    index.metadata.update({"name": name, "corpus": corpus.name, "taxonomy_method": method_name,
                           "papers": len(corpus), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "embedding": embedding_identity(rt.config.embedding)})
    return IndexBuild(index, problem_res, method_res, summary)


def index_corpus(index: SemanticIndex, name: str = "index") -> Corpus:
    """The indexed papers with their section chunks re-attached (for agent corpus exports)."""
    papers = []
    for pid, p in index.papers.items():
        q = Paper.from_dict(p.to_dict(include_chunks=False))
        q.chunks = [index.chunks[c] for c in index.chunks_of(pid)]
        papers.append(q)
    return Corpus(papers, name=name)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def is_organization_query(query: Query) -> bool:
    """TaxoBench / CHIME queries: the answer is a topic hierarchy (Sec. 5.3)."""
    return str((query.metadata or {}).get("task", "")).lower() == "organization"


def organization_style(query: Query) -> str:
    """``"survey"`` (TaxoBench) or ``"review"`` (CHIME) framing of the hierarchy prompt."""
    style = str((query.metadata or {}).get("style", "") or "")
    if style in ("survey", "review"):
        return style
    return "review" if "chime" in (query.benchmark or "").lower() else "survey"


def hierarchy_items(tax: Taxonomy) -> List[AnswerItem]:
    """One established item per category, citing the papers attached to it."""
    items = []
    for nid in tax.preorder():
        if nid == tax.root_id:
            continue
        node = tax.nodes[nid]
        items.append(AnswerItem(text=" > ".join(tax.path_names(nid)[1:]) or node.name, status="established",
                                citations=sorted(tax.direct_papers(nid))))
    return items


def answer_hierarchy(answer: Optional[Answer], query: Query, valid_ids: Optional[Iterable[str]] = None) -> Taxonomy:
    """Predicted hierarchy of an organization answer.

    Budgeted systems store the parsed hierarchy in
    ``answer.metadata["hierarchy"]``. Agent answers (uniform JSON-schema
    records, App. C.3) are read as a one-level hierarchy whose categories are
    the ranked records (named by their candidate or claim) and whose papers
    are the records' citations (:func:`records_to_hierarchy`). A missing
    answer gives an empty hierarchy (scored 0).
    """
    if answer is None:
        return Taxonomy(facet="topic", root_name=query.text)
    data = (answer.metadata or {}).get("hierarchy")
    if isinstance(data, Mapping) and data.get("nodes"):
        return Taxonomy.from_dict(dict(data))
    return records_to_hierarchy(answer, query.text, valid_ids)


def records_to_hierarchy(answer: Answer, root_name: str, valid_ids: Optional[Iterable[str]] = None) -> Taxonomy:
    """One-level topic hierarchy from ranked answer records (see :func:`answer_hierarchy`)."""
    valid = set(valid_ids) if valid_ids is not None else None
    tax = Taxonomy(facet="topic", root_name=root_name)
    names: Dict[str, str] = {}
    for item in answer.items:
        cites = [c for c in item.citations if valid is None or c in valid]
        if not cites:
            continue
        name = " ".join((item.candidate or item.text or "").split())[:120] or "Category"
        key = name.lower()
        nid = names.get(key)
        if nid is None:
            nid = names[key] = tax.add_node(name)
        for c in cites:
            tax.assign(c, nid)
    return tax


def render_hierarchy(tax: Taxonomy) -> str:
    """Markdown outline of a hierarchy with ``[paper_id]`` citations."""
    lines: List[str] = [f"# {tax.root.name}"]
    for nid in tax.preorder():
        if nid == tax.root_id:
            continue
        node = tax.nodes[nid]
        cites = " ".join(f"[{p}]" for p in sorted(tax.direct_papers(nid)))
        lines.append(f"{'  ' * (node.depth - 1)}- {node.name} {cites}".rstrip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Systems over one index
# ---------------------------------------------------------------------------


@dataclass
class SystemOptions:
    """Knobs of the compared systems that the paper leaves to the implementation.

    ``graph_candidates`` / ``graph_global_max_papers``: RAPTOR and LightRAG
    build their tree / graph offline over the whole corpus when it has at
    most ``graph_global_max_papers`` papers (TaxoBench, CHIME); on larger
    corpora (arXiv-Subfields, 148K papers) indexing every unit with LLM
    summaries / extractions is prohibitive, so they run over a per-query
    candidate pool of the top ``graph_candidates`` papers of the SPECTER2
    dense retriever (``candidate_retriever`` + ``candidate_papers``). This
    is a protocol deviation: the paper's baselines index the corpus globally
    offline, whereas here the per-query tree / graph construction (LLM
    summaries / extractions over the candidates) runs at query time. It is
    billed to ``index_build`` and kept out of Table 5's online Calls /
    Tokens / Cost / Latency, as offline indexing would be, but recorded per
    query (``index_build_calls`` / ``_tokens`` / ``_cost_usd`` and
    ``excluded_latency_s`` of :func:`efficiency_entry`) and reported in the
    "Index-build" columns of ``table5_detail`` (run_efficiency.py).
    """

    graph_candidates: int = 100
    graph_global_max_papers: int = 5000
    raptor_clustering: str = "gmm"
    raptor_use_umap: bool = True
    rerank_top_n: int = 50
    organize_max_tokens: int = 2048
    retrieval_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    """One query's output for one (benchmark, system, budget): answer and/or retrieval, efficiency, error.

    ``fingerprint`` identifies the configuration that produced the record
    (:meth:`QAEnvironment.fingerprint`); ``--resume`` only reuses records
    whose fingerprint matches the current configuration.
    """

    query_id: str
    benchmark: str
    system: str
    budget: Optional[int]
    answer: Optional[Answer] = None
    retrieval: Optional[RetrievalResult] = None
    efficiency: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    fingerprint: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and (self.answer is not None or self.retrieval is not None)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"query_id": self.query_id, "benchmark": self.benchmark, "system": self.system,
                             "budget": self.budget, "error": self.error, "fingerprint": self.fingerprint}
        if self.answer is not None:
            d["answer"] = self.answer.to_dict()
        if self.retrieval is not None:
            d["retrieval"] = self.retrieval.to_dict()
        if self.efficiency is not None:
            d["efficiency"] = dict(self.efficiency)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunRecord":
        ans = d.get("answer")
        ret = d.get("retrieval")
        budget = d.get("budget")
        return cls(
            query_id=str(d["query_id"]),
            benchmark=str(d.get("benchmark", "")),
            system=str(d.get("system", "")),
            budget=int(budget) if budget is not None else None,
            answer=Answer.from_dict(dict(ans)) if isinstance(ans, Mapping) else None,
            retrieval=retrieval_from_dict(ret) if isinstance(ret, Mapping) else None,
            efficiency=dict(d["efficiency"]) if isinstance(d.get("efficiency"), Mapping) else None,
            error=d.get("error"),
            fingerprint=d.get("fingerprint"),
        )


def compact_retrieval(result: RetrievalResult, keep_context: bool = False) -> RetrievalResult:
    """Copy of a retrieval result without unit texts / context (enough for every metric)."""
    units = [dataclasses.replace(u, text=u.text if keep_context else "") for u in result.units]
    stats = {k: v for k, v in (result.stats or {}).items() if k not in ("analysis", "grounding")} \
        if not keep_context else dict(result.stats or {})
    return dataclasses.replace(result, units=units, context=result.context if keep_context else "", stats=stats)


def retrieval_from_dict(d: Mapping[str, Any]) -> RetrievalResult:
    data = dict(d)
    units = [RetrievedUnit(**u) for u in data.pop("units", []) or []]
    names = {f.name for f in dataclasses.fields(RetrievalResult)}
    return RetrievalResult(units=units, **{k: v for k, v in data.items() if k in names})


def budget_tag(budget: Optional[int]) -> str:
    return UNBUDGETED if budget is None else f"B{int(budget)}"


def parse_budget_tag(tag: str) -> Optional[int]:
    if tag == UNBUDGETED:
        return None
    m = re.fullmatch(r"B(\d+)", tag)
    if m is None:
        raise ValueError(f"Bad budget tag {tag!r}")
    return int(m.group(1))


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Stable short hash of a JSON-serialisable configuration (cache validity of run records)."""
    blob = json.dumps(config, sort_keys=True, default=_json_default, ensure_ascii=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class RecordStore:
    """JSONL caches ``<root>/<benchmark>/<system>/<B<budget>|unbudgeted>.jsonl`` of :class:`RunRecord`.

    Records are appended (and flushed) one query at a time, so an
    interrupted run keeps its progress; when a file holds several records of
    a query the last one wins. ``run_qa_eval.py``, ``run_agents.py`` and
    ``run_efficiency.py`` share this layout.
    """

    def __init__(self, root: PathLike):
        self.root = Path(root)

    def path(self, benchmark: str, system: str, budget: Optional[int]) -> Path:
        return self.root / benchmark / system / f"{budget_tag(budget)}.jsonl"

    def load(self, benchmark: str, system: str, budget: Optional[int]) -> Dict[str, RunRecord]:
        return self.load_file(self.path(benchmark, system, budget))

    @staticmethod
    def load_file(path: PathLike) -> Dict[str, RunRecord]:
        out: Dict[str, RunRecord] = {}
        path = Path(path)
        if not path.exists():
            return out
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = RunRecord.from_dict(json.loads(line))
                except (ValueError, KeyError, TypeError) as err:  # e.g. a line cut by a crash
                    log.warning("Skipping unreadable record %s:%d (%s)", path, line_no, err)
                    continue
                out[rec.query_id] = rec
        return out

    @staticmethod
    def load_latest(path: PathLike) -> Dict[str, RunRecord]:
        """Records of the latest configuration only (the fingerprint of the last record in the file).

        A file can mix configurations when a ``--resume`` run under new
        options was interrupted; aggregating such a file would average two
        configurations, so the older records are dropped with a warning.
        """
        records = RecordStore.load_file(path)
        fps = {r.fingerprint for r in records.values()}
        if len(fps) <= 1:
            return records
        latest = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        latest = json.loads(line).get("fingerprint")
                    except ValueError:
                        continue
        kept = {qid: r for qid, r in records.items() if r.fingerprint == latest}
        log.warning("%s mixes records of %d configurations (an interrupted --resume?); using the %d records of the "
                    "latest one", path, len(fps), len(kept))
        return kept

    def append(self, record: RunRecord) -> None:
        path = self.path(record.benchmark, record.system, record.budget)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False, default=_json_default) + "\n")
            f.flush()

    def reset(self, benchmark: str, system: str, budget: Optional[int]) -> None:
        path = self.path(benchmark, system, budget)
        if path.exists():
            path.unlink()

    def entries(self) -> List[Tuple[str, str, Optional[int], Path]]:
        """``(benchmark, system, budget, path)`` of every cache file under the root."""
        out = []
        if not self.root.exists():
            return out
        for path in sorted(self.root.glob("*/*/*.jsonl")):
            try:
                budget = parse_budget_tag(path.stem)
            except ValueError:
                continue
            out.append((path.parent.parent.name, path.parent.name, budget, path))
        return out

    def systems(self, benchmark: str) -> List[str]:
        return sorted({s for b, s, _, _ in self.entries() if b == benchmark})

    def foreign_records(self, benchmark: str, system: str, budget: Optional[int],
                        fingerprint: Optional[str]) -> Dict[str, int]:
        """Successful cached records produced under another configuration: ``{fingerprint: count}``."""
        out: Dict[str, int] = {}
        for r in self.load_file(self.path(benchmark, system, budget)).values():
            if r.ok and r.fingerprint != fingerprint:
                key = str(r.fingerprint)
                out[key] = out.get(key, 0) + 1
        return out


class CacheConflictError(RuntimeError):
    """A run without ``--resume`` would delete cached records of another configuration.

    Answers of the paper setting are expensive (LLM calls, agents with
    1-hour timeouts); a run that forgot ``--config`` must not silently replace
    them with answers of the offline default configuration. The scripts map
    this error to ``--force``.
    """


def check_cache_conflicts(store: RecordStore, runs: Iterable[Tuple[str, str, Optional[int], Optional[str]]],
                          *, resume: bool = False, force: bool = False) -> None:
    """Refuse (:class:`CacheConflictError`) to replace cached records of another configuration.

    ``runs`` are ``(benchmark, system, budget, fingerprint)`` tuples of the
    planned runs. Without ``resume`` a run replaces its cache file; if that
    file holds successful records whose fingerprint differs from the current
    configuration, the run is refused unless ``force``. (With ``resume``,
    records of the current configuration are reused and records of another
    one are regenerated and *appended*, so the older lines stay in the file.)
    """
    if resume or force:
        return
    conflicts = []
    for bench, system, budget, fp in runs:
        foreign = store.foreign_records(bench, system, budget, fp)
        if foreign:
            n = sum(foreign.values())
            conflicts.append(f"{store.path(bench, system, budget)} ({n} answers, fingerprint "
                             f"{', '.join(sorted(foreign))}; current {fp})")
    if conflicts:
        raise CacheConflictError(
            "Refusing to replace cached records produced under another configuration:\n  "
            + "\n  ".join(conflicts)
            + "\nRerun with the configuration that produced them (e.g. --config configs/paper.yaml) and --resume, "
              "evaluate them with --eval-only, write to another --output-dir, or pass --force to replace them.")


class QAEnvironment:
    """One semantic index with every compared system over it (built lazily and cached).

    * ScholarPath (and its Table 7 variants): :class:`~scholarpath.retrieval.pipeline.ScholarPathRetriever`
      with the index's trained link predictor (if any);
    * budgeted baselines over the shared passage pool of the index (abstracts
      + section chunks; :meth:`PassagePool.from_index`): BM25, SPECTER (v1),
      Dense retrieval (SPECTER2, reusing the index vectors), Dense + reranker,
      RAPTOR, LightRAG, Keyword search, wrapped by
      :class:`~scholarpath.baselines.qa.BudgetedQA` for end-to-end QA;
    * the unassisted answer LLM (:class:`~scholarpath.baselines.qa.NoRetrievalQA`).
    """

    def __init__(
        self,
        index: SemanticIndex,
        runtime: Runtime,
        *,
        name: str = "index",
        link_predictor: Optional[LinkPredictor] = None,
        options: Optional[SystemOptions] = None,
        cache_dir: Optional[PathLike] = None,
    ):
        self.index = index
        self.runtime = runtime
        self.name = name
        self.link_predictor = link_predictor
        if link_predictor is None:
            self.link_predictor_id = "untrained"
        else:
            self.link_predictor_id = str(getattr(link_predictor, "source_id", None)
                                         or f"in-memory:{type(link_predictor).__name__}:{id(link_predictor)}")
        self.options = options or SystemOptions()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.index_dir: Optional[Path] = None
        self.warned_untrained_prior = False
        self._embedder_checked = False
        self._pool: Optional[PassagePool] = None
        self._dense: Any = None
        self._retrievers: Dict[str, Any] = {}
        self._qa: Dict[str, Any] = {}

    @classmethod
    def load(
        cls,
        index_dir: PathLike,
        runtime: Runtime,
        *,
        name: Optional[str] = None,
        link_predictor: Optional[str] = "auto",
        options: Optional[SystemOptions] = None,
    ) -> "QAEnvironment":
        index_dir = Path(index_dir)
        index = SemanticIndex.load(index_dir)
        lp = load_link_predictor(index_dir, link_predictor)
        env = cls(index, runtime, name=name or index_dir.name, link_predictor=lp, options=options,
                  cache_dir=index_dir / "baselines")
        env.index_dir = index_dir
        log.info("Loaded index %s (%d papers, %d chunks; link predictor: %s)", index_dir, len(index.papers),
                 len(index.chunks), type(lp).__name__ if lp is not None else "untrained prior")
        return env

    # ------------------------------------------------------------- basics
    @property
    def llm(self) -> Any:
        return self.runtime.llm

    @property
    def embedder(self) -> Any:
        return self.runtime.embedder

    def index_embedding(self) -> Optional[Dict[str, Any]]:
        """The embedder that encoded the index (index metadata, else ``build_summary.json``; ``None`` if unknown)."""
        ident = (self.index.metadata or {}).get("embedding")
        if isinstance(ident, Mapping):
            return dict(ident)
        if self.index_dir is not None and (self.index_dir / "build_summary.json").exists():
            try:
                ident = read_json(self.index_dir / "build_summary.json").get("embedding")
            except (OSError, ValueError):
                ident = None
            if isinstance(ident, Mapping):
                return dict(ident)
        return None

    def check_embedder(self) -> None:
        """Refuse to encode queries with another embedder than the index's (:class:`EmbeddingMismatchError`).

        SPECTER2 and the offline hashing embedder are both 768-d, so a query
        vector of the wrong embedder would silently be compared with the
        index vectors (e.g. ``run_efficiency.py --operator-stats`` without
        ``--config configs/paper.yaml`` on a paper-setting index).
        """
        if self._embedder_checked:
            return
        ident = self.index_embedding()
        if ident is None:
            log.debug("Index %s does not record its embedder; skipping the embedder check", self.name)
        else:
            diffs = embedding_mismatches(ident, embedding_identity(self.runtime.config.embedding))
            if diffs:
                raise EmbeddingMismatchError(
                    f"Index {self.name} was built with another embedder than the current configuration "
                    f"({'; '.join(diffs)}): query vectors would not be comparable with the index vectors. "
                    "Pass the --config / --set options the index was built with (e.g. --config configs/paper.yaml).")
        self._embedder_checked = True

    @property
    def pool(self) -> PassagePool:
        if self._pool is None:
            self._pool = PassagePool.from_index(self.index, name=self.name)
        return self._pool

    @property
    def titles(self) -> Dict[str, str]:
        return {pid: p.title for pid, p in self.index.papers.items()}

    @property
    def dense(self) -> Any:
        """SPECTER2 dense retriever over the pool (reuses the index's paper / chunk vectors)."""
        if self._dense is None:
            from scholarpath.baselines.dense import DenseRetriever

            self.check_embedder()
            self._dense = DenseRetriever.from_index(self.pool, self.index, self.embedder)
        return self._dense

    @property
    def per_query_graph(self) -> bool:
        """Whether RAPTOR / LightRAG use per-query candidate pools (see :class:`SystemOptions`)."""
        return len(self.index.papers) > int(self.options.graph_global_max_papers)

    @property
    def has_trained_link_predictor(self) -> bool:
        """Whether a temporally trained s_theta (App. B.5) is loaded (else the untrained text prior is used)."""
        from scholarpath.retrieval.link_prediction import EmbeddingLinkPredictor

        return self.link_predictor is not None and not isinstance(self.link_predictor, EmbeddingLinkPredictor)

    def retrieval_config(self, system: str) -> RetrievalConfig:
        cfg = copy.deepcopy(self.runtime.config.retrieval)
        for k, v in list(self.options.retrieval_overrides.items()) + list(SYSTEMS[system].overrides):
            if not hasattr(cfg, k):
                raise AttributeError(f"RetrievalConfig has no field {k!r}")
            setattr(cfg, k, v)
        return cfg

    # ------------------------------------------------------- fingerprints
    def system_config(self, system: str) -> Dict[str, Any]:
        """Everything that determines a system's outputs and efficiency on this index (cache fingerprint).

        Common: the index identity, the backbone LLM (model, decoding,
        prices, seed), the embedder and the hierarchy-call budget. ScholarPath
        variants add the effective :class:`RetrievalConfig` (incl. overrides
        such as ``--no-query-llm``) and the link prior; baselines add their
        own knobs (:class:`SystemOptions`).
        """
        spec = SYSTEMS[system]
        cfg = self.runtime.config
        llm = cfg.llm
        out: Dict[str, Any] = {
            "system": system,
            "index": {"name": self.name, "papers": len(self.index.papers), "chunks": len(self.index.chunks),
                      "built_at": (self.index.metadata or {}).get("built_at")},
            "llm": {k: getattr(llm, k, None) for k in ("provider", "model", "temperature", "max_output_tokens",
                                                       "price_input_per_m", "price_output_per_m",
                                                       "bill_cached_calls", "seed")},
            "embedding": {"backend": cfg.embedding.backend, "model": cfg.embedding.model_name},
            "seed": cfg.seed,
            "organize_max_tokens": int(self.options.organize_max_tokens),
        }
        rc = self.retrieval_config(system)
        if spec.family == "scholarpath":
            retrieval = dataclasses.asdict(rc)
            retrieval.pop("budget", None)  # the budget is part of the cache key
            out["retrieval"] = retrieval
            out["link_prior"] = self.link_predictor_id if spec.link_prior == "trained" else "untrained"
        else:
            out["answer_max_tokens"] = int(rc.answer_max_tokens)
            o = self.options
            if system == "dense_rerank":
                out["rerank_top_n"] = int(o.rerank_top_n)
            if system in ("raptor", "lightrag"):
                out["graph"] = {"per_query": self.per_query_graph,
                                "candidates": int(o.graph_candidates) if self.per_query_graph else None}
            if system == "raptor":
                out["raptor"] = {"clustering": o.raptor_clustering, "umap": bool(o.raptor_use_umap)}
        return out

    def fingerprint(self, system: str) -> str:
        """Short hash of :meth:`system_config` stored with every cached record."""
        return config_fingerprint(self.system_config(system))

    # -------------------------------------------------------- retrievers
    def retriever(self, system: str) -> Any:
        """The budgeted retriever of ``system`` (``.retrieve(query, budget) -> RetrievalResult``)."""
        spec = SYSTEMS[system]
        if not spec.retrieves:
            raise ValueError(f"{spec.display} does not retrieve a budgeted context")
        r = self._retrievers.get(system)
        if r is None:
            r = self._retrievers[system] = self._make_retriever(system)
        return r

    def _make_retriever(self, system: str) -> Any:
        from scholarpath import baselines as B

        spec = SYSTEMS[system]
        llm, emb, opts = self.llm, self.embedder, self.options
        if spec.family == "scholarpath" or system in ("raptor", "lightrag"):
            self.check_embedder()  # query (and node) vectors must live in the index's embedding space
        if spec.family == "scholarpath":
            from scholarpath.retrieval.pipeline import ScholarPathRetriever

            # "w/o temporal link prediction" (Table 7): no trained predictor -> the retriever falls back to the
            # untrained text-similarity prior (EmbeddingLinkPredictor); candidates are still scored.
            lp = self.link_predictor if spec.link_prior == "trained" else None
            return ScholarPathRetriever(self.index, llm, emb, self.retrieval_config(system), lp)
        if system == "bm25":
            return B.BM25Retriever(self.pool)
        if system == "keyword":
            return B.KeywordSearchRetriever(self.pool)
        if system == "dense":
            return self.dense
        if system == "specter":
            specter = B.build_dense_embedder("specter", self.runtime.config.embedding)
            cache = self.cache_dir / "specter" if self.cache_dir is not None else None
            return B.DenseRetriever(self.pool, specter, name="SPECTER", cache_dir=cache)
        if system == "dense_rerank":
            return B.DenseRerankRetriever(self.dense, llm, top_n=opts.rerank_top_n)
        per_query = self.per_query_graph
        cand = self.dense if per_query else None
        k = int(opts.graph_candidates) if per_query else None
        if system == "raptor":
            cfg = B.RaptorConfig(clustering=opts.raptor_clustering, use_umap=opts.raptor_use_umap,
                                 candidate_papers=k, seed=self.runtime.config.seed)
            return B.RaptorRetriever(self.pool, llm, emb, cfg, candidate_retriever=cand)
        if system == "lightrag":
            return B.LightRAGRetriever(self.pool, llm, emb, B.LightRAGConfig(candidate_papers=k),
                                       candidate_retriever=cand)
        raise ValueError(f"No retriever for system {system!r}")

    def retrieve(self, system: str, query: Query, budget: int) -> RetrievalResult:
        return self.retriever(system).retrieve(query, int(budget))

    def prepare(self, system: str) -> float:
        """Construct ``system`` and its offline indexes before any query is timed; returns the seconds spent.

        Retriever construction (BM25 / dense indexes, ScholarPath's node-cost
        table and link-prior preparation) and lazily built offline indexes
        (``ensure_built``: dense passage encodings, global RAPTOR trees /
        LightRAG graphs, billed to ``index_build``) are offline work that must
        not enter the first query's latency (Table 5).
        """
        from scholarpath.baselines.common import ensure_built

        t0 = time.perf_counter()
        spec = SYSTEMS[system]
        if spec.retrieves:
            r = self.retriever(system)
            if spec.family == "scholarpath":
                _ = r.link_predictor  # builds the untrained prior lazily when no predictor was passed
            else:
                with self.llm.meter.scope("index_build"):
                    ensure_built(r)
        return time.perf_counter() - t0

    # ---------------------------------------------------------- answering
    def answer(self, system: str, query: Query, budget: Optional[int]) -> Answer:
        """End-to-end answer: retrieve <= B tokens, then ONE composition call (Sec. 5.3)."""
        spec = SYSTEMS[system]
        if spec.family == "agent":
            raise ValueError("Agents are run with run_agent(), not QAEnvironment.answer()")
        b = int(budget if budget is not None else self.runtime.config.retrieval.budget)
        if is_organization_query(query):
            return self.organize(system, query, b)
        if spec.family == "scholarpath":
            ans = self.retriever(system).answer(query, b)
            ans.metadata["system"] = spec.display
            ans.metadata["budget"] = b
            return ans
        cfg = self.retrieval_config(system)
        if system == NO_RETRIEVAL:
            from scholarpath.baselines.qa import NoRetrievalQA

            qa = self._qa.get(system) or NoRetrievalQA(self.llm, cfg)
        else:
            from scholarpath.baselines.qa import BudgetedQA

            qa = self._qa.get(system) or BudgetedQA(self.retriever(system), self.llm, cfg, name=spec.display)
        self._qa[system] = qa
        return qa.answer(query, b)

    def organize(self, system: str, query: Query, budget: int) -> Answer:
        """Organization query: retrieve <= B tokens, then one hierarchy-composition call."""
        from scholarpath.prompts.organize import organize

        spec = SYSTEMS[system]
        meter = self.llm.meter
        before = meter.snapshot()
        t0 = time.perf_counter()
        retrieval = None if system == NO_RETRIEVAL else self.retrieve(system, query, budget)
        context = retrieval.context if retrieval is not None else ""
        sources = retrieval.sources if retrieval is not None else []
        tax = organize(self.llm, query.text, context, sources, max_tokens=self.options.organize_max_tokens,
                       style=organization_style(query))
        wall = time.perf_counter() - t0
        index_s = min(wall, index_build_seconds(retrieval))
        counted, excluded = split_usage(meter.diff(before), EXCLUDED_QUERY_COMPONENTS)
        meta: Dict[str, Any] = {
            "task": "organization", "system": spec.display, "budget": budget, "hierarchy": tax.to_dict(),
            "excluded_usage": excluded.to_dict(), "excluded_latency_s": index_s, "wall_clock_s": wall,
        }
        if retrieval is not None:
            meta["retrieval_stats"] = {k: v for k, v in retrieval.stats.items() if k not in ("analysis", "grounding")}
            meta["context_papers"] = [s["paper_id"] for s in retrieval.sources]
        return Answer(
            query_id=query.query_id,
            text=render_hierarchy(tax),
            items=hierarchy_items(tax),
            retrieved_papers=list(retrieval.ranked_papers) if retrieval is not None else [],
            context_tokens=retrieval.context_tokens if retrieval is not None else 0,
            llm_calls=counted.calls,
            input_tokens=counted.input_tokens,
            output_tokens=counted.output_tokens,
            cost_usd=counted.cost_usd,
            latency_s=online_latency(wall, index_s, counted.cached_latency_s),
            metadata=meta,
        )

    # ------------------------------------------------------------ one query
    def run_one(
        self,
        system: str,
        query: Query,
        budget: Optional[int],
        *,
        mode: str = "qa",
        tracker: Optional[EfficiencyTracker] = None,
        keep_context: bool = False,
        fail_fast: bool = False,
        benchmark: str = "",
    ) -> RunRecord:
        """Answer (``mode="qa"``) or retrieve (``mode="retrieval"``) one query with efficiency tracking.

        The record's efficiency row (:func:`efficiency_entry`) uses the same
        latency definition for every system; an answer's ``latency_s`` is set
        to it. Call :meth:`prepare` first so that system construction is not
        timed (``run_queries`` does).
        """
        if mode not in ("qa", "retrieval"):
            raise ValueError("mode must be 'qa' or 'retrieval'")
        tracker = tracker or EfficiencyTracker(self.llm.meter, system=system)
        record = RunRecord(query.query_id, benchmark, system, budget, fingerprint=self.fingerprint(system))
        retrieval_full: Optional[RetrievalResult] = None
        before = self.llm.meter.snapshot()
        with tracker.track(query.query_id) as holder:
            try:
                if mode == "retrieval":
                    retrieval_full = self.retrieve(system, query, int(budget))
                else:
                    record.answer = self.answer(system, query, budget)
            except Exception as err:  # noqa: BLE001 - one failing query must not abort a benchmark run
                if fail_fast:
                    raise
                log.exception("%s failed on %s", system_display(system), query.query_id)
                record.error = f"{type(err).__name__}: {err}"
        if retrieval_full is not None:
            record.retrieval = compact_retrieval(retrieval_full, keep_context)
        index_usage = self.llm.meter.diff(before).get("index_build")
        record.efficiency = efficiency_entry(holder["record"], record.answer, record.retrieval,
                                             index_build=index_usage)
        if record.answer is not None:
            record.answer.latency_s = float(record.efficiency["latency_s"])
        return record


def online_latency(wall_s: float, excluded_s: float = 0.0, cached_latency_s: float = 0.0) -> float:
    """End-to-end online latency of one query (Table 5), identical for every budgeted system.

    ``wall_s`` minus the offline index construction the query triggered
    (``excluded_s``: RAPTOR / LightRAG per-query indexing), plus the recorded
    latency of the LLM calls served from the response cache
    (``cached_latency_s``; :attr:`UsageRecord.cached_latency_s`), which return
    instantly but would have taken that long. With the cache disabled
    (``--set llm.cache_dir=null``) the last term is zero.
    """
    return max(0.0, float(wall_s) - max(0.0, float(excluded_s))) + max(0.0, float(cached_latency_s))


def _excluded_seconds(answer: Optional[Answer], retrieval: Optional[RetrievalResult]) -> float:
    """Offline indexing seconds a query triggered (reported by BudgetedQA / organize, or the retrieval stats)."""
    if answer is not None:
        md = answer.metadata or {}
        if md.get("excluded_latency_s") is not None:
            return max(0.0, float(md["excluded_latency_s"]))
        stats = md.get("retrieval_stats") or {}
        try:
            return max(0.0, float(stats.get("index_build_s", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0
    return index_build_seconds(retrieval)


def efficiency_entry(
    measured: QueryEfficiency,
    answer: Optional[Answer] = None,
    retrieval: Optional[RetrievalResult] = None,
    *,
    metered: bool = True,
    index_build: Optional[Any] = None,
) -> Dict[str, Any]:
    """Per-query efficiency row (Table 5).

    Calls / tokens / cost come from the cost-meter diff of the query with
    ``judge``, ``index_build``, ``link_training`` and ``benchmark`` excluded
    (:data:`~scholarpath.evaluation.efficiency.EXCLUDED_COMPONENTS`).
    For metered runs (``metered=True``: budgeted systems measured by
    :class:`EfficiencyTracker`), ``latency_s`` is :func:`online_latency` of
    the measured wall time -- the same definition for ScholarPath, the
    baselines and organization answers, robust to a warm LLM response cache;
    ``wall_clock_s`` keeps the raw wall time. Unmetered rows (agents recorded
    with :meth:`EfficiencyTracker.record_answer`) keep the latency the system
    reported. ``composition_calls`` counts the calls of the ``composition``
    component, ``retrieval_calls`` every other online call.

    ``index_build`` (the ``index_build`` usage the query triggered: RAPTOR /
    LightRAG trees and graphs built at query time over a per-query candidate
    pool on large corpora, :class:`SystemOptions`) is kept out of the online
    columns but recorded as ``index_build_calls`` / ``index_build_tokens`` /
    ``index_build_cost_usd`` next to ``excluded_latency_s``, so that
    ``table5_detail`` can report this per-query construction work.
    """
    d = measured.to_dict()
    d["wall_clock_s"] = measured.latency_s
    if metered:
        comps = measured.components or {}
        cached_s = sum(float((c or {}).get("cached_latency_s", 0.0) or 0.0) for c in comps.values())
        excluded_s = _excluded_seconds(answer, retrieval)
        d["excluded_latency_s"] = excluded_s
        d["cached_latency_s"] = cached_s
        d["latency_s"] = online_latency(measured.latency_s, excluded_s, cached_s)
    comp = (measured.components or {}).get("composition", {})
    d["composition_calls"] = int(comp.get("calls", 0) or 0)
    d["retrieval_calls"] = int(measured.calls) - d["composition_calls"]
    if metered:
        ib = index_build
        d["index_build_calls"] = int(getattr(ib, "calls", 0) or 0)
        d["index_build_tokens"] = int(getattr(ib, "total_tokens", 0) or 0)
        d["index_build_cost_usd"] = float(getattr(ib, "cost_usd", 0.0) or 0.0)
    return d


def run_queries(
    env: QAEnvironment,
    system: str,
    queries: Sequence[Query],
    budget: Optional[int],
    *,
    benchmark: str,
    mode: str = "qa",
    store: Optional[RecordStore] = None,
    resume: bool = False,
    fail_fast: bool = False,
    keep_context: bool = False,
    log_every: int = 25,
    force: bool = False,
) -> Dict[str, RunRecord]:
    """Run one system over a benchmark's queries at one budget, caching every record.

    With ``resume`` the records already in ``store`` are reused when they
    succeeded AND were produced under the current configuration
    (:meth:`QAEnvironment.fingerprint`); failed queries and records of
    another configuration (e.g. before ``--no-query-llm`` or a retrained link
    predictor) are regenerated, the latter with a warning (the new records
    are appended; :meth:`RecordStore.load_latest` reads the latest
    configuration). Without ``resume`` the cache file is replaced -- unless
    it holds records of another configuration, which raises
    :class:`CacheConflictError` (``force=True`` replaces them anyway).
    Queries are processed sequentially so that per-query cost-meter diffs
    are exact.
    """
    fp = env.fingerprint(system)
    done: Dict[str, RunRecord] = {}
    if store is not None:
        check_cache_conflicts(store, [(benchmark, system, budget, fp)], resume=resume, force=force)
        if resume:
            cached = store.load(benchmark, system, budget)
            wanted = {q.query_id for q in queries}
            stale = [qid for qid, r in cached.items() if qid in wanted and r.ok and r.fingerprint != fp]
            if stale:
                log.warning("%s / %s / %s: %d cached records were produced under a different configuration "
                            "(fingerprint %s); regenerating them", benchmark, system_display(system),
                            budget_tag(budget), len(stale), fp)
            done = {qid: r for qid, r in cached.items() if r.fingerprint == fp}
        else:
            store.reset(benchmark, system, budget)
    _warn_untrained_prior(env, system, queries)
    tracker = EfficiencyTracker(env.llm.meter, system=system)
    out: Dict[str, RunRecord] = {}
    todo = [q for q in queries if not (q.query_id in done and done[q.query_id].ok)]
    if len(todo) < len(queries):
        log.info("%s / %s / %s: reusing %d cached records", benchmark, system_display(system), budget_tag(budget),
                 len(queries) - len(todo))
    for q in queries:
        if q.query_id in done and done[q.query_id].ok:
            out[q.query_id] = done[q.query_id]
    if todo:
        setup_s = env.prepare(system)
        if setup_s > 1.0:
            log.info("%s: system set up in %.1fs (not part of per-query latency)", system_display(system), setup_s)
    t0 = time.perf_counter()
    for i, q in enumerate(todo, 1):
        rec = env.run_one(system, q, budget, mode=mode, tracker=tracker, keep_context=keep_context,
                          fail_fast=fail_fast, benchmark=benchmark)
        if store is not None:
            store.append(rec)
        out[q.query_id] = rec
        if i == len(todo) or (log_every and i % log_every == 0):
            log.info("%s / %s / %s: %d/%d queries (%.1fs)", benchmark, system_display(system), budget_tag(budget),
                     i, len(todo), time.perf_counter() - t0)
    return {q.query_id: out[q.query_id] for q in queries if q.query_id in out}


_HYPOTHESIS_KINDS = (HYPOTHESIS_DISCOVERY, HYPOTHESIS_ASSESSMENT)


def _warn_untrained_prior(env: QAEnvironment, system: str, queries: Sequence[Query]) -> None:
    """Warn (once per environment) when ScholarPath answers hypothesis queries without a trained s_theta.

    The temporally trained link predictor (App. B.5) is part of the full
    model; without it the "trained" variants fall back to the untrained
    text-similarity prior, i.e. Table 7's "w/o temporal link prediction".
    """
    spec = SYSTEMS[system]
    if spec.family != "scholarpath" or spec.link_prior != "trained" or env.has_trained_link_predictor:
        return
    if not any(q.kind in _HYPOTHESIS_KINDS for q in queries):
        return
    if not env.warned_untrained_prior:
        env.warned_untrained_prior = True
        log.warning("Index %s has no trained link predictor: ScholarPath answers hypothesis queries with the "
                    "untrained text-similarity prior (= 'w/o temporal link prediction'); train one with "
                    "scripts/train_link_predictor.py --corpus %s", env.name, env.name)


def answers_of(records: Mapping[str, RunRecord]) -> Dict[str, Answer]:
    return {qid: r.answer for qid, r in records.items() if r.answer is not None and r.error is None}


def retrievals_of(records: Mapping[str, RunRecord]) -> Dict[str, RetrievalResult]:
    return {qid: r.retrieval for qid, r in records.items() if r.retrieval is not None and r.error is None}


def count_errors(records: Mapping[str, RunRecord]) -> int:
    return sum(1 for r in records.values() if r.error is not None)


# ---------------------------------------------------------------------------
# Unbudgeted agents (Sec. 5.3, App. C.3)
# ---------------------------------------------------------------------------


@dataclass
class AgentOptions:
    """Settings of the unbudgeted agent adapters (see :mod:`scholarpath.baselines.agents`)."""

    workdir: Optional[str] = None
    max_records: int = 10
    timeout_s: float = 3600.0
    meter_with_proxy: bool = True
    shard_size: int = 5000
    openscholar_dir: Optional[str] = None
    agenticscholar_url: Optional[str] = None
    agenticscholar_fn: Optional[str] = None  # "package.module:function"
    codex_binary: str = "codex"
    codex_isolation: str = "unshare"
    codex_wire_api: str = "chat"  # the paper's Codex v0.42 + a chat-completions backbone endpoint
    paperqa_embedding: Optional[str] = None


def import_callable(spec: str) -> Callable[..., Any]:
    """Resolve ``"package.module:function"``."""
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError(f"Expected 'module:function', got {spec!r}")
    fn = getattr(importlib.import_module(module), attr)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def build_agent(key: str, env: QAEnvironment, options: AgentOptions, export_dir: Optional[PathLike] = None) -> Any:
    """Agent adapter ``key`` over the corpus of ``env``, with the shared LLM backbone (App. C.3).

    ``export_dir`` holds the JSONL corpus shards
    (:func:`~scholarpath.baselines.agents.export_corpus_shards`) mounted for
    the agents that read files (Codex; AgenticScholar receives its path).
    """
    from scholarpath.baselines import agents as A

    llm_cfg: LLMConfig = env.runtime.config.llm
    if key == SIMULATED:
        cfg = A.AgentConfig(meter_with_proxy=False, max_records=options.max_records)
        return A.SimulatedAgent(env.pool, env.llm, env.embedder, cfg, dense=env.dense)
    cfg = A.AgentConfig.from_llm_config(llm_cfg, timeout_s=options.timeout_s, max_records=options.max_records,
                                        workdir=options.workdir, meter_with_proxy=options.meter_with_proxy,
                                        corpus_dir=str(export_dir) if export_dir else None)
    if key == "paperqa2":
        kw = {"embedding": options.paperqa_embedding} if options.paperqa_embedding else {}
        return A.PaperQA2Agent(index_corpus(env.index, env.name), cfg, **kw)
    if key == "openscholar":
        return A.OpenScholarAgent(env.pool, checkout_dir=options.openscholar_dir, config=cfg,
                                  initial_retriever=env.dense)
    if key == "agenticscholar":
        fn = import_callable(options.agenticscholar_fn) if options.agenticscholar_fn else None
        if fn is None and not options.agenticscholar_url:
            raise A.AgentUnavailableError(
                "AgenticScholar needs an implementation: --agenticscholar-fn module:function or "
                "--agenticscholar-url http://host/endpoint (see scholarpath/baselines/agents/agenticscholar.py)")
        return A.AgenticScholarAgent(index=env.index, config=cfg, fn=fn, endpoint_url=options.agenticscholar_url)
    if key == "codex":
        if export_dir is None:
            raise ValueError("Codex needs the exported corpus shards (export_dir)")
        return A.CodexAgent(export_dir, cfg, binary=options.codex_binary, isolation=options.codex_isolation,
                            wire_api=options.codex_wire_api, valid_ids=list(env.index.papers))
    raise ValueError(f"Unknown agent {key!r}; expected one of {AGENT_SYSTEMS + (SIMULATED,)}")


def agent_task_query(query: Query) -> Query:
    """The query an agent receives: organization queries carry their Deep Research instruction.

    TaxoBench queries store the instruction in ``metadata["instruction"]``;
    for other organization queries an equivalent one is formed from the
    topic. The uniform JSON-schema task prompt is added by the adapter.
    """
    if not is_organization_query(query):
        return query
    md = query.metadata or {}
    text = md.get("instruction")
    if not text:
        if organization_style(query) == "review":
            text = (f"Retrieve the studies on the topic \"{query.text}\" and organize them into a category "
                    "hierarchy, as in a systematic review on this topic.")
        else:
            text = (f"Retrieve the key papers on the topic \"{query.text}\" and organize them into a topic "
                    "hierarchy, as in the taxonomy of a survey on this topic.")
    return dataclasses.replace(query, text=str(text))


def agent_fingerprint(key: str, env: QAEnvironment, options: AgentOptions) -> str:
    """Cache fingerprint of an agent run: agent, index, backbone LLM and the agent options."""
    llm = env.runtime.config.llm
    # workdir is scratch space; the default Codex wire API is omitted so that earlier fingerprints stay valid.
    opts = {k: v for k, v in dataclasses.asdict(options).items()
            if k != "workdir" and not (k == "codex_wire_api" and v == "chat")}
    return config_fingerprint({
        "system": key,
        "index": {"name": env.name, "papers": len(env.index.papers),
                  "built_at": (env.index.metadata or {}).get("built_at")},
        "llm": {k: getattr(llm, k, None) for k in ("provider", "model", "temperature", "max_output_tokens",
                                                   "price_input_per_m", "price_output_per_m", "seed")},
        "options": opts,
    })


def run_agent(
    agent: Any,
    system: str,
    queries: Sequence[Query],
    *,
    benchmark: str,
    store: Optional[RecordStore] = None,
    resume: bool = False,
    fail_fast: bool = False,
    valid_ids: Optional[Iterable[str]] = None,
    llm_config: Optional[LLMConfig] = None,
    fingerprint: Optional[str] = None,
    force: bool = False,
) -> Dict[str, RunRecord]:
    """Run an unbudgeted agent over a benchmark; records use the budget tag ``unbudgeted``.

    Agents report their own usage (metering proxy or in-process meter), so
    efficiency rows come from the answers
    (:meth:`EfficiencyTracker.record_answer`) and keep the latency the agent
    reported. Their cost is the one the adapter computed (it prices the
    proxy-metered tokens with ``config.llm``'s backbone prices, or uses the
    cost the system reported); answers without a cost are priced at
    ``llm_config``'s prices (Table 5 uses the backbone pricing for every
    system, App. C.3). For organization queries the records are additionally
    stored as a hierarchy (:func:`records_to_hierarchy`). With ``resume``,
    records are reused only if their ``fingerprint`` matches
    (:func:`agent_fingerprint`); without it, cached answers of another
    configuration raise :class:`CacheConflictError` unless ``force``. An
    :class:`~scholarpath.baselines.agents.AgentUnavailableError` aborts the run.
    """
    from scholarpath.baselines.agents import AgentUnavailableError

    done: Dict[str, RunRecord] = {}
    if store is not None:
        check_cache_conflicts(store, [(benchmark, system, None, fingerprint)], resume=resume, force=force)
        if resume:
            cached = store.load(benchmark, system, None)
            stale = [qid for qid, r in cached.items() if r.ok and r.fingerprint != fingerprint]
            if stale:
                log.warning("%s / %s: %d cached answers were produced under a different configuration; "
                            "regenerating them", benchmark, system_display(system), len(stale))
            done = {qid: r for qid, r in cached.items() if r.fingerprint == fingerprint}
        else:
            store.reset(benchmark, system, None)
    meter = getattr(getattr(agent, "llm", None), "meter", None)
    meter = meter if meter is not None else _null_meter()
    tracker = (EfficiencyTracker.from_config(meter, llm_config, system=system) if llm_config is not None
               else EfficiencyTracker(meter, system=system))
    valid = list(valid_ids) if valid_ids is not None else None
    out: Dict[str, RunRecord] = {}
    for q in queries:
        if q.query_id in done and done[q.query_id].ok:
            out[q.query_id] = done[q.query_id]
            continue
        rec = RunRecord(q.query_id, benchmark, system, None, fingerprint=fingerprint)
        try:
            ans = agent.answer(agent_task_query(q))
            ans.query_id = q.query_id
            ans.metadata.setdefault("system", system_display(system))
            if is_organization_query(q):
                ans.metadata["task"] = "organization"
                ans.metadata["hierarchy"] = records_to_hierarchy(ans, q.text, valid).to_dict()
            rec.answer = ans
            eff = tracker.record_answer(ans)
            if ans.cost_usd:  # the adapter's cost (backbone prices or the system's own report)
                eff.cost_usd = float(ans.cost_usd)
            rec.efficiency = efficiency_entry(eff, ans, metered=False)
        except AgentUnavailableError:
            raise
        except Exception as err:  # noqa: BLE001
            if fail_fast:
                raise
            log.exception("%s failed on %s", system_display(system), q.query_id)
            rec.error = f"{type(err).__name__}: {err}"
        if store is not None:
            store.append(rec)
        out[q.query_id] = rec
    return out


def _null_meter() -> Any:
    from scholarpath.llm import CostMeter

    return CostMeter()


# ---------------------------------------------------------------------------
# Judges (Sec. 5.3 "Metrics", App. C.3)
# ---------------------------------------------------------------------------


@dataclass
class JudgeSet:
    """Every judge of the QA evaluation; each judge owns a private cost meter.

    * ``hypothesis`` -- the three-judge panel of Table 4 (GPT-6.0-Sol,
      Opus-5.5, Grok-4.7): Disc matching and Feas. / Nov. / Sound. scores;
    * ``researchqa`` -- the rubric-coverage judge of ResearchQA's
      ``compute_coverage.py`` (gpt-4.1-mini);
    * ``scholarqa_cs`` -- the ScholarQA-CS rubric judge (gpt-4-turbo);
    * ``prometheus`` -- Prometheus-2 7B for ScholarQA-Multi;
    * ``attribution`` -- the statement-level citation judge of ScholarQABench
      Cite (AttrScore NLI, or an LLM with the AttrScore prompt).
    """

    hypothesis: JudgePanel
    researchqa: JudgePanel
    scholarqa_cs: JudgePanel
    prometheus: JudgePanel
    attribution: Optional[Callable[[str, str], float]] = None

    def usage(self) -> Dict[str, Any]:
        """Judge usage per role and judge (reported separately from system cost)."""
        roles = ("hypothesis", "researchqa", "scholarqa_cs", "prometheus")
        return {role: getattr(self, role).usage() for role in roles}


def _judge_spec(name: str, model: str, cfg: EvaluationConfig, seed: int = 0) -> JudgeSpec:
    provider = (cfg.judge_provider or "mock").lower()
    if provider == "mock":
        return JudgeSpec(name, LLMConfig(provider="mock", model=f"mock-{name}", cache_dir=None, seed=seed,
                                         price_input_per_m=0.0, price_output_per_m=0.0))
    return JudgeSpec(name, LLMConfig(provider=provider, model=model, base_url=cfg.judge_base_url,
                                     api_key_env=cfg.judge_api_key_env, temperature=0.0,
                                     cache_dir=cfg.judge_cache_dir, price_input_per_m=0.0, price_output_per_m=0.0))


def build_judges(cfg: Optional[EvaluationConfig] = None, seed: int = 0) -> JudgeSet:
    """Judges from ``config.evaluation`` (mock judges when ``judge_provider == "mock"``)."""
    from scholarpath.evaluation.qa_metrics import AttrScoreNLI, LLMAttributionJudge

    cfg = cfg or EvaluationConfig()
    hyp = JudgePanel.from_specs([_judge_spec(m, m, cfg, seed) for m in cfg.judge_models])
    rqa = JudgePanel.from_specs([_judge_spec(cfg.researchqa_judge_model, cfg.researchqa_judge_model, cfg, seed)])
    cs = JudgePanel.from_specs([_judge_spec(cfg.scholarqa_cs_judge_model, cfg.scholarqa_cs_judge_model, cfg, seed)])
    prom = JudgePanel.from_specs([_judge_spec(cfg.prometheus_model, cfg.prometheus_model, cfg, seed)])
    mode = (cfg.attribution_judge or "llm").lower()
    attribution: Optional[Callable[[str, str], float]]
    if mode == "nli" and (cfg.judge_provider or "mock").lower() != "mock":
        attribution = AttrScoreNLI()
    elif mode in ("none", "off"):
        attribution = None
    else:
        attribution = LLMAttributionJudge(cs)
    return JudgeSet(hyp, rqa, cs, prom, attribution)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkEval:
    """Evaluation of one (benchmark, system, budget): per-query rows and a summary."""

    benchmark: str
    system: str
    budget: Optional[int]
    summary: Dict[str, Any] = field(default_factory=dict)
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_retrieval_records(
    records: Mapping[str, RunRecord], queries: Sequence[Query], budget: Optional[int], group: str
) -> RetrievalEvaluation:
    """Recall@B / NDCG@B of one benchmark (Table 2): ranked within-budget papers vs. gold.

    Unaligned gold citations stay in the denominator (``Query.num_gold``);
    failed queries count as retrieving nothing.
    """
    return evaluate_retrieval(retrievals_of(records), queries, budget=budget, group=lambda q: group)


def combine_retrieval(evals: Mapping[str, RetrievalEvaluation]) -> Dict[str, Any]:
    """Pool per-benchmark evaluations into table groups and a macro average (Table 2, Fig. 5).

    Group values are means over the group's queries (ScholarQA-CS and
    ScholarQA-Multi pooled into "ScholarQA"); ``macro`` is the mean over
    groups ("macro-average Recall@B", Sec. 5.3.1). Values are in percent.
    """
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for ev in evals.values():
        for row in ev.rows:
            by_group.setdefault(row["benchmark"], []).append(row)
    groups = {g: {"R": float(np.mean([r["R"] for r in rows])), "N": float(np.mean([r["N"] for r in rows])),
                  "n": float(len(rows))} for g, rows in by_group.items() if rows}
    out: Dict[str, Any] = {"by_group": groups, "macro": {}, "row": {}}
    for g, v in groups.items():
        out["row"][f"{g}.R"] = v["R"]
        out["row"][f"{g}.N"] = v["N"]
    if groups:
        out["macro"] = {"R": float(np.mean([v["R"] for v in groups.values()])),
                        "N": float(np.mean([v["N"] for v in groups.values()])),
                        "n": float(sum(v["n"] for v in groups.values()))}
        out["row"]["macro.R"] = out["macro"]["R"]
        out["row"]["macro.N"] = out["macro"]["N"]
    return out


_ORG_TASK = {"taxobench": "taxobench", "chime": "chime"}


def organization_embedder(runtime: Runtime) -> Any:
    """Label embedder of the organization metrics.

    TaxoBench's Sem uses all-MiniLM-L6-v2 (its paper, App. A.3.1); CHIME's
    Cat / Assn use the same sentence embedder here. Offline runs (hashing
    backend) and installations without sentence-transformers fall back to
    the runtime embedder.
    """
    if runtime.config.embedding.backend == "hashing":
        return runtime.embedder
    try:
        from scholarpath.evaluation.organization_metrics import load_taxobench_embedder

        return load_taxobench_embedder()
    except Exception as err:  # noqa: BLE001 - optional dependency / model download
        log.warning("all-MiniLM-L6-v2 unavailable (%s); organization metrics use the runtime embedder", err)
        return runtime.embedder


def evaluate_organization(
    spec: BenchmarkSpec,
    answers: Mapping[str, Answer],
    queries: Sequence[Query],
    *,
    gold_dir: PathLike,
    titles: Mapping[str, str],
    embedder: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """TaxoBench ARI / Sem or CHIME Cat / Assn, macro-averaged over queries (Table 3).

    The predicted hierarchy (:func:`answer_hierarchy`) is scored against the
    query's gold tree (``metadata["gold_taxonomy"]``, relative to the
    benchmark directory). TaxoBench aligns papers by normalized title, so the
    corpus titles are passed for both trees. Missing answers score 0.
    """
    from scholarpath.evaluation.organization_metrics import ORGANIZATION_METRICS, macro_average, organization_scores
    from scholarpath.evaluation.reporting import organization_table_row

    task = _ORG_TASK.get(spec.group, "taxobench")
    rows: List[Dict[str, Any]] = []
    for q in queries:
        rel = (q.metadata or {}).get("gold_taxonomy")
        if not rel:
            log.warning("Organization query %s has no gold taxonomy; skipped", q.query_id)
            continue
        gold = Taxonomy.load(Path(gold_dir) / str(rel))
        pred = answer_hierarchy(answers.get(q.query_id), q, titles.keys())
        scores = organization_scores(pred, gold, task, embedder, titles=dict(titles), gold_titles=dict(titles))
        rows.append({"query_id": q.query_id, "benchmark": spec.key, "missing": q.query_id not in answers,
                     "pred_categories": len(pred) - 1, "pred_papers": len(pred.all_papers()), **scores})
    keys = list(ORGANIZATION_METRICS[task]) + [k for k in ("ari_e2e", "sem_expert", "paper_recall", "cat_precision",
                                                         "cat_recall", "assn_precision", "assn_recall")
                                               if any(k in r for r in rows)]
    summary: Dict[str, Any] = dict(macro_average(rows, keys)) if rows else {"n_queries": 0.0}
    summary.update(organization_table_row(summary, task))
    summary["n"] = float(len(rows))
    return summary, rows


def score_missing_as_zero(
    summary: Mapping[str, Any],
    counts: Mapping[str, Optional[float]],
    *,
    n_missing: int,
    n_expected: int,
) -> Dict[str, Any]:
    """Fold failed / missing answers into the means as 0 (Table 3 evidence metrics).

    The official ResearchQA / ScholarQABench scripts average over the answers
    they receive; a system that fails or times out on a query (an agent with
    a 1-hour timeout, a crashed retrieval) would then be averaged over its
    successful queries only, while Recall@B, the organization metrics and
    Disc count the same failure as 0. Every metric in ``counts`` (key ->
    number of answers it was averaged over) is rescaled to the mean over
    those answers plus ``n_missing`` zeros: ``v * n / (n + n_missing)``. This
    is exact for arithmetic means and for Cite = F1(mean precision, mean
    recall), whose precision and recall both scale by the same factor. A
    metric without any scored answer becomes 0 when answers are missing.

    The answered-only values are kept as ``<key>_answered``; ``n`` becomes
    ``n_expected`` (the benchmark's number of queries, which weights
    ScholarQA-CS / -Multi in :func:`qa_table_row`), ``n_scored`` the number
    of answered queries and ``n_missing`` the failed / missing ones.
    """
    out: Dict[str, Any] = dict(summary)
    for key, n in counts.items():
        value = summary.get(key)
        n = float(n or 0.0)
        if value is None:
            if n_missing > 0 and n == 0.0:
                out[key] = 0.0
            continue
        out[f"{key}_answered"] = value
        if n_missing > 0:
            out[key] = float(value) * n / (n + n_missing)
    out["n_scored"] = float(summary.get("n") or 0.0)
    out["n_missing"] = float(n_missing)
    out["n"] = float(n_expected)
    return out


def _missing_rows(query_ids: Iterable[str], benchmark: str, keys: Sequence[str]) -> List[Dict[str, Any]]:
    """Per-query rows of failed / missing answers (every reported metric 0)."""
    return [{"query_id": qid, "benchmark": benchmark, "missing": True, **{k: 0.0 for k in keys}}
            for qid in query_ids]


def _first_per_text(query_ids: Sequence[str], queries: Sequence[Query]) -> List[str]:
    """One query id per distinct question text (ScholarQA-CS scores doubly annotated questions once)."""
    text_of = {q.query_id: q.text for q in queries}
    seen: Dict[str, str] = {}
    for qid in query_ids:
        seen.setdefault(text_of.get(qid, qid), qid)
    return list(seen.values())


def evaluate_benchmark(
    benchmark: str,
    answers: Mapping[str, Answer],
    queries: Sequence[Query],
    judges: JudgeSet,
    *,
    papers: Mapping[str, Paper],
    embedder: Any = None,
    layout: Optional[DataLayout] = None,
    discovery_k: int = 10,
    system: str = "",
    budget: Optional[int] = None,
) -> BenchmarkEval:
    """Evaluate the answers of one benchmark with its official protocol (Tables 3-4).

    * ResearchQA (and the synthetic evidence set): Rub (rubric coverage) and
      Cite (citation F1 against the gold source papers);
    * ScholarQA-CS: Corr (rubric score) and Cite (statement-level citation F1);
    * ScholarQA-Multi: Corr (Prometheus-2 organization / coverage /
      relevance) and Cite;
    * TaxoBench / CHIME: :func:`evaluate_organization`;
    * Held-out hypothesis queries: Disc (Hit@10) and the three-judge
      Feasibility / Novelty / Soundness scores (0-100) and their average.

    A query without an answer in ``answers`` (failed or missing) scores 0
    on every metric (:func:`score_missing_as_zero`; the organization and
    hypothesis protocols already do so), so systems that fail are not
    averaged over their successful queries only.
    """
    from scholarpath.evaluation import hypothesis_metrics as H
    from scholarpath.evaluation import qa_metrics as QM

    spec = BENCHMARKS[benchmark]
    out = BenchmarkEval(benchmark, system, budget)
    cite_keys = ("Cite", "citation_prec", "citation_rec") if judges.attribution is not None else ()
    if spec.protocol == PROTOCOL_RESEARCHQA:
        ev = QM.evaluate_researchqa(answers, queries, judges.researchqa, papers=papers)
        n_rows = float(len(ev.rows))
        out.summary = score_missing_as_zero(
            ev.summary, {"Rub": ev.summary.get("n"), "Cite": n_rows, "cite_P": n_rows, "cite_R": n_rows},
            n_missing=len(ev.missing), n_expected=len(queries))
        out.rows = ev.rows + _missing_rows(ev.missing, benchmark, ("Rub", "Cite"))
    elif spec.protocol == PROTOCOL_CS:
        ev = QM.evaluate_scholarqa_cs(answers, queries, judges.scholarqa_cs, attribution_judge=judges.attribution,
                                      papers=papers)
        # Doubly annotated questions are scored once (official rubric_eval.py): count distinct questions.
        n_expected = len({q.text for q in queries})
        n_cite = float(sum(1 for r in ev.rows if "citation_prec" in r))
        counts: Dict[str, Optional[float]] = {"Corr": ev.summary.get("n"), "ann_score": ev.summary.get("n")}
        counts.update({k: n_cite for k in cite_keys})
        out.summary = score_missing_as_zero(ev.summary, counts, n_missing=max(0, n_expected - len(ev.rows)),
                                            n_expected=n_expected)
        scored_ids = {r["query_id"] for r in ev.rows}
        scored_texts = {q.text for q in queries if q.query_id in scored_ids}
        missing = _first_per_text([q.query_id for q in queries if q.text not in scored_texts], queries)
        out.rows = ev.rows + _missing_rows(missing, benchmark, ("Corr",) + cite_keys[:1])
    elif spec.protocol == PROTOCOL_MULTI:
        ev = QM.evaluate_scholarqa_multi(answers, queries, judges.prometheus, attribution_judge=judges.attribution,
                                         papers=papers)
        n_cite = float(sum(1 for r in ev.rows if "citation_prec" in r))
        counts = {"Corr": ev.summary.get("n")}
        counts.update({k: n_cite for k in cite_keys})
        out.summary = score_missing_as_zero(ev.summary, counts, n_missing=len(ev.missing), n_expected=len(queries))
        out.rows = ev.rows + _missing_rows(ev.missing, benchmark, ("Corr",) + cite_keys[:1])
    elif spec.protocol == PROTOCOL_ORGANIZATION:
        layout = layout or DataLayout()
        titles = {pid: p.title for pid, p in papers.items()}
        out.summary, out.rows = evaluate_organization(spec, answers, queries, gold_dir=layout.benchmark_dir(benchmark),
                                                      titles=titles, embedder=embedder)
    elif spec.protocol == PROTOCOL_HYPOTHESIS:
        titles = {pid: p.title for pid, p in papers.items()}
        res = H.evaluate_hypotheses(answers, queries, judges.hypothesis, judges.hypothesis, k=discovery_k,
                                    paper_titles=titles)
        out.summary, out.rows = dict(res["summary"]), res["rows"]
    else:  # pragma: no cover - guarded by BENCHMARKS
        raise ValueError(f"Unknown protocol {spec.protocol!r}")
    out.summary.setdefault("n_queries", float(len(queries)))
    n_answered = sum(1 for q in queries if q.query_id in answers)
    out.summary["n_answered"] = float(n_answered)
    out.summary["n_unanswered"] = float(len(queries) - n_answered)
    return out


#: Column of Tables 3-4 counting failed / missing answers (scored 0), shown only when some system has any.
FAILED_COLUMN = Column("Failed", "Failed", higher_is_better=False)


def unanswered_cell(evals: Iterable[BenchmarkEval]) -> Tuple[int, int, str]:
    """``(failed, total, "failed/total")`` over a system's evaluations (failed = no answer, scored 0)."""
    failed = total = 0
    for ev in evals:
        total += int(ev.summary.get("n_queries") or 0)
        failed += int(ev.summary.get("n_unanswered") or 0)
    return failed, total, f"{failed}/{total}"


def with_failed_column(spec: TableSpec, rows: Mapping[str, Mapping[str, Any]]) -> TableSpec:
    """``spec`` plus :data:`FAILED_COLUMN` if any row reports failed answers (``row["n_failed"] > 0``)."""
    if not any(int(r.get("n_failed") or 0) > 0 for r in rows.values()):
        return spec
    return dataclasses.replace(spec, columns=list(spec.columns) + [FAILED_COLUMN])


def qa_table_row(evals: Mapping[str, BenchmarkEval]) -> Dict[str, Any]:
    """Table 3 cells of one system from its per-benchmark evaluations.

    ScholarQABench combines its CS and Multi subsets weighted by their
    numbers of queries (App. C.3; :func:`~scholarpath.evaluation.qa_metrics.scholarqabench_combined`).
    ``macro.Avg`` is the macro-average over the reported metrics (Sec. 5.3.2:
    "the highest macro-average across all eight metrics (70.7%)"); it is
    computed over the metrics present (all eight in the paper setting).
    """
    from scholarpath.evaluation.qa_metrics import scholarqabench_combined

    row: Dict[str, Any] = {}
    cs = evals.get("scholarqa_cs")
    multi = evals.get("scholarqa_multi")
    if cs is not None or multi is not None:
        comb = scholarqabench_combined(cs.summary if cs else {}, multi.summary if multi else {})
        row["scholarqabench.Corr"] = comb.get("Corr")
        row["scholarqabench.Cite"] = comb.get("Cite")
    for key, ev in evals.items():
        spec = BENCHMARKS[key]
        if spec.protocol == PROTOCOL_RESEARCHQA:
            row[f"{spec.group}.Rub"] = ev.summary.get("Rub")
            row[f"{spec.group}.Cite"] = ev.summary.get("Cite")
        elif spec.protocol == PROTOCOL_ORGANIZATION:
            row.update({k: v for k, v in ev.summary.items() if k.startswith(f"{spec.group}.")})
    metrics = [f"{g}.{m}" for g, ms in GROUP_QA_METRICS.items() for m in ms]
    values = [float(row[k]) for k in metrics if row.get(k) is not None]
    row["macro.Avg"] = float(np.mean(values)) if values else None
    return row


def _group_columns(groups: Iterable[str], metrics_of: Callable[[str], Sequence[str]]) -> List[Column]:
    present = list(dict.fromkeys(groups))
    ordered = [g for g in GROUP_ORDER if g in present] + [g for g in present if g not in GROUP_ORDER]
    return [Column(f"{g}.{m}", m, GROUP_DISPLAY.get(g, g)) for g in ordered for m in metrics_of(g)]


def table2_spec(groups: Iterable[str], budget: int = PAPER_BUDGET) -> TableSpec:
    """Table 2 shaped spec (R / N per benchmark group present)."""
    return TableSpec("table2", f"Table 2: Retrieval accuracy (%) on evidence queries (B = {budget:,}).",
                     _group_columns(groups, lambda g: ("R", "N")), first_header="Method")


def table3_spec(groups: Iterable[str]) -> TableSpec:
    """Table 3 shaped spec (metrics per benchmark group present) plus the macro-average of Sec. 5.3.2."""
    cols = _group_columns(groups, lambda g: GROUP_QA_METRICS.get(g, ("Rub", "Cite")))
    cols.append(Column("macro.Avg", "Avg.", "Macro"))
    return TableSpec("table3", "Table 3: End-to-end QA results (%) on evidence queries.", cols,
                     first_header="System")


TABLE5_DETAIL = TableSpec(
    "table5_detail", "Table 5 (detail): per-query LLM calls split into composition and other online calls; "
                     "index-build columns = tree / graph construction triggered at query time (RAPTOR / LightRAG "
                     "per-query candidate pools), excluded from the online columns.",
    [Column("Calls", "Calls", higher_is_better=False),
     Column("Composition calls", "Composition calls", higher_is_better=False),
     Column("Retrieval calls", "Retrieval calls", higher_is_better=False),
     Column("Tokens (K)", "Tokens (K)", higher_is_better=False),
     Column("Cost ($)", "Cost ($)", higher_is_better=False, digits=4),
     Column("Latency (s)", "Latency (s)", higher_is_better=False, digits=2),
     Column("Index-build calls", "Index-build calls", higher_is_better=False),
     Column("Index-build tokens (K)", "Index-build tokens (K)", higher_is_better=False),
     Column("Index-build cost ($)", "Index-build cost ($)", higher_is_better=False, digits=4),
     Column("Index-build latency (s)", "Index-build latency (s)", higher_is_better=False, digits=2),
     Column("n", "Queries", higher_is_better=False, digits=0)],
)

TABLE7 = TableSpec(
    "table7", "Table 7: Ablation of evidence-path retrieval on hypothesis queries.",
    [Column("Hit@10", "Hit@10"), Column("Judge Avg.", "Judge Avg."),
     Column("Context (K)", "Context (K)", higher_is_better=False),
     Column("Scored", "Scored", higher_is_better=False, digits=0)],
    first_header="Variant",
)


TABLE1 = TableSpec(
    "table1", "Table 1: Statistics of the QA benchmarks.",
    [Column("Queries", "Queries", digits=0), Column("Corpus", "Corpus", digits=0), Column("Gold/Q", "Gold/Q")],
    first_header="Benchmark",
)

#: Held-out settings of App. C.1 and their Table 1 row labels.
HELDOUT_SETTING_ROWS = (("new_problem_existing_method", "– New Prob. / Exist. Meth."),
                        ("new_method_existing_problem", "– New Meth. / Exist. Prob."))


def query_gold_count(query: Query) -> Optional[int]:
    """Gold papers of one query for Table 1's Gold/Q.

    Evidence queries: the gold citations including the unaligned ones
    (:attr:`Query.num_gold`, kept in every denominator, App. C.2). Held-out
    hypothesis queries have no gold paper in the *reduced* corpus -- the
    papers that realize the expected answer are removed (App. C.1) -- so
    their count is the target's supporting papers J(u, m) on the complete
    corpus (``metadata["supporting_papers"]``). The paper does not define
    Gold/Q for the held-out rows (4.3 / 4.2 / 4.5); this is the closest
    quantity the construction records. ``None`` (left out of the mean) for a
    hypothesis query without that record (e.g. the planted synthetic ones).
    """
    if query.num_gold:
        return int(query.num_gold)
    md = query.metadata or {}
    if "supporting_papers" in md:
        return len(md.get("supporting_papers") or [])
    return None if query.kind in _HYPOTHESIS_KINDS else 0


def table1_rows(
    queries: Mapping[str, Sequence[Query]],
    corpus_sizes: Mapping[str, Optional[int]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, List[str]]]]:
    """Table 1 (benchmark statistics): queries, corpus size and mean gold papers per query.

    ``queries`` maps benchmark keys to their (possibly ``--limit``-ed) queries
    and ``corpus_sizes`` to the number of papers each benchmark is searched
    over (the indexed corpus: for the held-out rows the *reduced* corpus,
    whereas the paper's Table 1 lists the complete 148,400). ScholarQA-CS and
    ScholarQA-Multi are pooled into one ScholarQABench row; the held-out
    benchmark gets one row per setting.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    evidence: List[str] = []
    hypothesis: List[str] = []
    by_group: Dict[str, List[str]] = {}
    for key in queries:
        by_group.setdefault(BENCHMARKS[key].group, []).append(key)

    def row(qs: Sequence[Query], keys: Sequence[str]) -> Dict[str, Any]:
        sizes = [corpus_sizes.get(k) for k in keys if corpus_sizes.get(k) is not None]
        gold = [g for g in (query_gold_count(q) for q in qs) if g is not None]
        return {"Queries": len(qs), "Corpus": max(sizes) if sizes else None,
                "Gold/Q": float(np.mean(gold)) if gold else None}

    ordered = [g for g in GROUP_ORDER if g in by_group] + [g for g in by_group if g not in GROUP_ORDER]
    for group in ordered:
        keys = by_group[group]
        qs = [q for k in keys for q in queries[k]]
        if all(BENCHMARKS[k].is_hypothesis for k in keys):
            name = "Held-Out Total" if group == "heldout" else f"{GROUP_DISPLAY.get(group, group)} Total"
            rows[name] = row(qs, keys)
            hypothesis.append(name)
            for setting, label in HELDOUT_SETTING_ROWS:
                sub = [q for q in qs if (q.metadata or {}).get("setting") == setting]
                if sub:
                    rows[label] = row(sub, keys)
                    hypothesis.append(label)
        else:
            name = "ScholarQABench" if group == "scholarqabench" else (
                BENCHMARKS[keys[0]].display if len(keys) == 1 else GROUP_DISPLAY.get(group, group))
            rows[name] = row(qs, keys)
            evidence.append(name)
    sections = [(t, names) for t, names in (("Evidence Queries", evidence), ("Hypothesis Queries", hypothesis))
                if names]
    return rows, sections


def budget_sections(budgeted: Sequence[str], unbudgeted: Sequence[str], budget: int) -> List[Tuple[str, List[str]]]:
    """Table sections "Budgeted methods (B = 8,192)" / "Unbudgeted methods" of Tables 3-5."""
    return [(f"Budgeted methods (B = {budget:,})", list(budgeted)), ("Unbudgeted methods", list(unbudgeted))]


# ---------------------------------------------------------------------------
# Efficiency (Table 5, Sec. 5.3.3)
# ---------------------------------------------------------------------------

_QE_FIELDS = {f.name for f in dataclasses.fields(QueryEfficiency)}


def efficiency_from_dict(d: Mapping[str, Any]) -> QueryEfficiency:
    return QueryEfficiency(**{k: v for k, v in d.items() if k in _QE_FIELDS})


def collect_efficiency(
    stores: Sequence[RecordStore],
    *,
    budget: Optional[int] = PAPER_BUDGET,
    systems: Optional[Sequence[str]] = None,
    benchmarks: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Per-query efficiency rows per system from cached runs.

    Budgeted systems are read at ``budget``; agents from their
    ``unbudgeted`` runs. Failed queries are left out, and so are records of
    an older configuration in a file that mixes configurations
    (:meth:`RecordStore.load_latest`). When the same (benchmark, system,
    budget) appears in several stores, the later store wins.
    """
    found: Dict[Tuple[str, str], Path] = {}
    for store in stores:
        for bench, system, b, path in store.entries():
            if benchmarks is not None and bench not in benchmarks:
                continue
            if systems is not None and system not in systems:
                continue
            spec = SYSTEMS.get(system)
            want = None if (spec is not None and spec.family == "agent") else budget
            if b != want:
                continue
            found[(bench, system)] = path
    out: Dict[str, List[Dict[str, Any]]] = {}
    for (bench, system), path in sorted(found.items()):
        for rec in RecordStore.load_latest(path).values():
            if rec.error is not None or rec.efficiency is None:
                continue
            row = dict(rec.efficiency)
            row.update(benchmark=bench, system=system, query_id=rec.query_id)
            out.setdefault(system, []).append(row)
    return out


def efficiency_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """Table 5 row (means per query) plus the composition / other-call split.

    ``Calls`` counts every online LLM call (incl. ScholarPath's Step 1
    query-understanding call when enabled); ``Composition calls`` counts the
    composition calls only (the "1.0" of Table 5 for budgeted methods);
    ``Retrieval calls`` = Calls - Composition calls.
    """
    summary = dict(summarize_efficiency([efficiency_from_dict(r) for r in rows]))
    n = len(rows)
    if n:
        comp = [float(r.get("composition_calls", 0) or 0) for r in rows]
        summary["Composition calls"] = sum(comp) / n
        summary["Retrieval calls"] = (summary["Calls"] or 0.0) - summary["Composition calls"]
    else:
        summary["Composition calls"] = summary["Retrieval calls"] = None
    # Query-time index construction (RAPTOR / LightRAG per-query candidate pools), excluded from the columns above.
    for key, out_key, scale_by in (("index_build_calls", "Index-build calls", 1.0),
                                   ("index_build_tokens", "Index-build tokens (K)", 1e-3),
                                   ("index_build_cost_usd", "Index-build cost ($)", 1.0),
                                   ("excluded_latency_s", "Index-build latency (s)", 1.0)):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        summary[out_key] = scale_by * sum(vals) / len(vals) if vals else None
    return summary


def operator_rows_from_records(records: Iterable[RunRecord]) -> List[Dict[str, Any]]:
    """ScholarPath search statistics stored with cached answers or retrievals."""
    rows = []
    for rec in records:
        stats = None
        if rec.retrieval is not None:
            stats = rec.retrieval.stats
        elif rec.answer is not None:
            stats = (rec.answer.metadata or {}).get("retrieval_stats")
        if stats and "postings_total" in stats:
            rows.append(_operator_row(rec.query_id, stats, None))
    return rows


def _operator_row(query_id: str, stats: Mapping[str, Any], exhaustive: Optional[Mapping[str, Any]],
                  retrieval_s: Optional[float] = None) -> Dict[str, Any]:
    def num(d: Mapping[str, Any], key: str) -> int:
        return int(d.get(key, 0) or 0)

    row = {
        "query_id": query_id,
        "kind": stats.get("kind"),
        "postings_total": num(stats, "postings_total"),
        "postings_pruned": num(stats, "postings_pruned"),
        # Candidate concepts of hypothesis queries: scored exactly vs. in scope ("Scored" of Table 7).
        "scored_candidates": num(stats, "scored_candidates"),
        "candidate_pool": num(stats, "candidate_pool"),
        "link_evaluations": num(stats, "link_evaluations"),
        "scored_nodes": num(stats, "scored_nodes"),
        "search_ms": 1000.0 * float(stats.get("search_time_s", 0.0) or 0.0),
        "selected": num(stats, "selected"),
    }
    if retrieval_s is not None:
        row["retrieval_ms"] = 1000.0 * float(retrieval_s)
    if exhaustive is not None:
        row["scored_nodes_exhaustive"] = num(exhaustive, "scored_nodes")
        row["scored_candidates_exhaustive"] = num(exhaustive, "scored_candidates")
        row["exhaustive_search_ms"] = 1000.0 * float(exhaustive.get("search_time_s", 0.0) or 0.0)
        row["same_selection"] = exhaustive.get("_units") == stats.get("_units") if "_units" in stats else None
    return row


def measure_operator_stats(
    env: QAEnvironment, queries: Sequence[Query], budget: int = PAPER_BUDGET, exhaustive: bool = False
) -> List[Dict[str, Any]]:
    """Run ScholarPath retrieval (no composition) and collect its operator statistics (Sec. 5.3.3).

    Per query (from :attr:`RetrievalResult.stats` of the search):

    * postings -- papers in the extents of the grounded scopes; those the
      search never scans were pruned by junction intersections and the
      early-terminated join (Sec. 4.3, App. A.3): ``postings_pruned`` /
      ``postings_total``;
    * candidate nodes -- for hypothesis queries, the candidate concepts in
      scope (``candidate_pool``, what an exhaustive scan scores) and those
      whose exact eta_hat was computed by the hierarchy-bounded search
      (``scored_candidates``, the "Scored" column of Table 7; Table 7 / App.
      C.4: 312 of 4,628 candidate nodes, i.e. 93.3% pruned);
    * the search time (ms) and the full retrieval time.

    ``exhaustive`` additionally re-runs every query with the exhaustive
    greedy scan (Table 7 "w/o bound-guided search") to check Proposition 7
    (``same_selection``: both select the same units) and to report the index
    nodes the exhaustive scan scores; it is not needed for the pruning shares.
    """
    rows = []
    env.prepare(SCHOLARPATH)
    bound = env.retriever(SCHOLARPATH)
    exh = None
    if exhaustive:
        env.prepare("scholarpath_no_bound_guided_search")
        exh = env.retriever("scholarpath_no_bound_guided_search")
    for q in queries:
        r = bound.retrieve(q, int(budget))
        stats = dict(r.stats)
        stats["_units"] = [u.key for u in r.units]
        ex_stats = None
        if exh is not None:
            e = exh.retrieve(q, int(budget))
            ex_stats = dict(e.stats)
            ex_stats["_units"] = [u.key for u in e.units]
        rows.append(_operator_row(q.query_id, stats, ex_stats, r.latency_s))
    return rows


def summarize_operator_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """Pooled operator statistics of Sec. 5.3.3: % postings pruned, % candidate nodes pruned, ms per query.

    * ``postings_pruned_pct`` = 100 (1 - sum postings scanned / sum postings in
      scope), pooled over every query;
    * ``candidate_nodes_pruned_pct`` = 100 (1 - sum scored_candidates / sum
      candidate_pool), pooled over the hypothesis queries (those with a
      candidate pool), e.g. 100 (1 - 312 / 4,628) = 93.3 (Table 7);
      ``scored_candidates_mean`` is Table 7's "Scored";
    * with the exhaustive check: ``index_nodes_pruned_pct`` (index nodes whose
      exact relevance was computed, bound-guided vs. exhaustive) and
      ``same_selection_rate`` (Proposition 7).
    """
    if not rows:
        return {"n": 0.0}
    tot = sum(int(r.get("postings_total", 0) or 0) for r in rows)
    pruned = sum(int(r.get("postings_pruned", 0) or 0) for r in rows)
    out: Dict[str, Optional[float]] = {
        "n": float(len(rows)),
        "postings_pruned_pct": 100.0 * pruned / tot if tot else None,
        "postings_total_mean": tot / len(rows),
        "scored_nodes_mean": float(np.mean([r.get("scored_nodes", 0) or 0 for r in rows])),
        "search_ms_mean": float(np.mean([r.get("search_ms", 0.0) or 0.0 for r in rows])),
    }
    if all("retrieval_ms" in r for r in rows):
        out["retrieval_ms_mean"] = float(np.mean([r["retrieval_ms"] for r in rows]))
    hyp = [r for r in rows if int(r.get("candidate_pool", 0) or 0) > 0]
    out["n_hypothesis"] = float(len(hyp))
    if hyp:
        pool = sum(int(r["candidate_pool"]) for r in hyp)
        scored = sum(int(r.get("scored_candidates", 0) or 0) for r in hyp)
        out["candidate_nodes_pruned_pct"] = 100.0 * (1.0 - scored / pool)
        out["scored_candidates_mean"] = scored / len(hyp)
        out["candidate_pool_mean"] = pool / len(hyp)
    else:
        out["candidate_nodes_pruned_pct"] = None
    ex = [r for r in rows if "scored_nodes_exhaustive" in r]
    if ex:
        total_ex = sum(r["scored_nodes_exhaustive"] for r in ex)
        out["scored_nodes_exhaustive_mean"] = total_ex / len(ex)
        out["index_nodes_pruned_pct"] = (100.0 * (1.0 - sum(r.get("scored_nodes", 0) for r in ex) / total_ex)
                                         if total_ex else None)
        out["exhaustive_search_ms_mean"] = float(np.mean([r["exhaustive_search_ms"] for r in ex]))
        same = [r["same_selection"] for r in ex if r.get("same_selection") is not None]
        out["same_selection_rate"] = float(np.mean(same)) if same else None
    return out


# ---------------------------------------------------------------------------
# Held-out benchmark helpers (App. C.1)
# ---------------------------------------------------------------------------


def domain_lookup(spec: Optional[str], papers: Mapping[str, Paper]) -> Optional[Any]:
    """Domain of a paper for target sampling (App. C.1: the primary arXiv category).

    ``spec``: ``None`` -> primary category (the default of
    :mod:`scholarpath.data.heldout`); ``"metadata:<key>"`` -> ``paper.metadata[key]``;
    otherwise a JSON file mapping paper id -> domain.
    """
    if not spec:
        return None
    if spec.startswith("metadata:"):
        key = spec.split(":", 1)[1]
        return {pid: (p.metadata or {}).get(key) for pid, p in papers.items()}
    data = read_json(spec)
    if not isinstance(data, Mapping):
        raise ValueError(f"{spec} must map paper ids to domains")
    return {str(k): (str(v) if v is not None else None) for k, v in data.items()}


def write_reduced_corpus(corpus: Corpus, reduced_ids: Iterable[str], out_dir: PathLike) -> Dict[str, Path]:
    """Write the reduced corpus of App. C.1 (``papers.jsonl.gz`` + ``chunks.jsonl.gz``) for indexing."""
    out = Path(out_dir)
    reduced = corpus.subset(sorted(set(reduced_ids)), name="heldout")
    paths = {"papers": out / "papers.jsonl.gz", "chunks": out / "chunks.jsonl.gz"}
    reduced.save(paths["papers"], include_chunks=False)
    reduced.save_chunks(paths["chunks"])
    return paths


# ---------------------------------------------------------------------------
# Interactive answering (scripts/ask.py)
# ---------------------------------------------------------------------------

QUERY_KIND_CHOICES = ("auto", "evidence", "hypothesis", "assessment")
_KIND_MAP = {"auto": "", "evidence": EVIDENCE, "hypothesis": HYPOTHESIS_DISCOVERY,
             "discovery": HYPOTHESIS_DISCOVERY, "hypothesis_discovery": HYPOTHESIS_DISCOVERY,
             "assessment": HYPOTHESIS_ASSESSMENT, "hypothesis_assessment": HYPOTHESIS_ASSESSMENT}


def resolve_kind(kind: Optional[str]) -> str:
    """Query kind for ``Query.kind`` ("" lets Step 1 decide)."""
    k = _slug(kind or "auto")
    if k not in _KIND_MAP:
        raise ValueError(f"Unknown query kind {kind!r}; expected one of {sorted(_KIND_MAP)}")
    return _KIND_MAP[k]


@dataclass
class AskResult:
    query: Query
    answer: Answer
    retrieval: RetrievalResult


def ask(env: QAEnvironment, question: str, kind: str = "auto", budget: Optional[int] = None,
        query_id: str = "ask") -> AskResult:
    """Answer one question with ScholarPath and keep its retrieval (support paths) for display.

    Same calls as :meth:`ScholarPathRetriever.answer`: Steps 1-5 within B,
    then one composition call (Algorithm 2, line 25).
    """
    from scholarpath.retrieval.compose import compose_answer

    q = Query(query_id=query_id, text=question, kind=resolve_kind(kind))
    retriever = env.retriever(SCHOLARPATH)
    meter = env.llm.meter
    before = meter.snapshot()
    t0 = time.perf_counter()
    retrieval = retriever.retrieve(q, budget)
    answer = compose_answer(q, retrieval, env.llm, retriever.config)
    usage = sum_usage(meter.diff(before))
    answer.llm_calls, answer.input_tokens, answer.output_tokens = usage.calls, usage.input_tokens, usage.output_tokens
    answer.cost_usd = usage.cost_usd
    answer.latency_s = time.perf_counter() - t0
    return AskResult(q, answer, retrieval)


def node_label(index: SemanticIndex, key: str) -> Optional[str]:
    """Human-readable label of an index node key (``None`` for virtual context-label nodes)."""
    try:
        level, raw = parse_key(key)
    except ValueError:
        return None
    if level == "concept":
        try:
            facet = index.facet_of(raw)
        except KeyError:  # e.g. a query-local anchor attached to the taxonomy for this query only
            return f"anchor:{raw}"
        return f"{'P' if facet == 'problem' else 'M'}:{index.concept(raw).name}"
    if level == "paper":
        p = index.papers.get(raw)
        return f"\"{p.title}\" [{raw}]" if p is not None else f"[{raw}]"
    ch: Optional[Chunk] = index.chunks.get(raw)
    return f"§{ch.heading} [{ch.paper_id}]" if ch is not None else f"[{raw}]"


def support_paths(index: SemanticIndex, retrieval: RetrievalResult) -> List[str]:
    """One line per selected unit: direct paths concept -> paper -> chunk, or path sets b = {e_C, e_P, e_M}."""
    lines = []
    for i, u in enumerate(retrieval.units, 1):
        if u.kind == "complementary":
            md = u.metadata or {}

            def path(keys: Sequence[str]) -> str:
                return " -> ".join(lbl for lbl in (node_label(index, k) for k in keys) if lbl) or "-"

            lines.append(f"{i:>2}. [hypothesis] {u.candidate}: bridge {path(md.get('e_C', []))} | problem side "
                         f"{path(md.get('e_P', []))} | method side {path(md.get('e_M', []))} ({u.tokens} tok)")
        else:
            labels = [lbl for lbl in (node_label(index, k) for k in u.nodes) if lbl]
            lines.append(f"{i:>2}. [established] {' -> '.join(labels)} ({u.tokens} tok)")
    return lines


def format_ask(result: AskResult, index: SemanticIndex, show_paths: int = 20) -> str:
    """Plain-text report: answer with established / hypothesis labels, support paths, usage."""
    a, r = result.answer, result.retrieval
    kind = (r.stats or {}).get("kind") or result.query.kind or EVIDENCE
    lines = [f"Question: {result.query.text}", f"Kind: {kind}{' (auto)' if not result.query.kind else ''}", "",
             "Answer:", a.text.strip() or "(empty)", ""]
    if a.items:
        lines.append("Statements:")
        for i, it in enumerate(a.items, 1):
            head = f"{it.candidate}: " if it.candidate else ""
            cites = " ".join(f"[{c}]" for c in it.citations)
            lines.append(f"{i:>2}. [{it.status}] {head}{it.text} {cites}".rstrip())
            if it.rationale:
                lines.append(f"     rationale: {it.rationale}")
        lines.append("")
    if r.candidates:
        lines.append("Ranked hypothesis candidates: " + ", ".join(r.candidates[:10]))
        lines.append("")
    paths = support_paths(index, r)
    lines.append(f"Support paths ({len(paths)} units, {r.context_tokens:,} / {r.budget:,} context tokens):")
    lines.extend(paths[:show_paths])
    if len(paths) > show_paths:
        lines.append(f"    ... {len(paths) - show_paths} more")
    lines.append("")
    lines.append(f"Usage: {a.llm_calls} LLM call(s), {a.input_tokens + a.output_tokens:,} tokens, "
                 f"${a.cost_usd:.4f}, {a.latency_s:.2f} s")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI helpers shared by the scripts
# ---------------------------------------------------------------------------


def parse_budgets(text: Union[str, Sequence[int], None]) -> List[int]:
    """Budgets from ``"2K,4K,6K,8192,16K"`` (K = 1,024 tokens, as in Figs. 5-6); ``none`` -> []."""
    if text is None:
        return []
    if not isinstance(text, str):
        return [int(b) for b in text]
    if text.strip().lower() in ("", "none", "off"):
        return []
    out = []
    for t in split_csv(text):
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kK]?)", t)
        if m is None:
            raise argparse.ArgumentTypeError(f"Bad budget {t!r}")
        value = float(m.group(1)) * (1024 if m.group(2) else 1)
        if value <= 0 or not math.isfinite(value):
            raise argparse.ArgumentTypeError(f"Budget must be positive, got {t!r}")
        out.append(int(value))
    return list(dict.fromkeys(out))


def _key_value(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {text!r}")
    k, v = text.split("=", 1)
    return k.strip(), v.strip()


def add_data_args(parser: argparse.ArgumentParser, *, benchmarks: bool = True, limit: bool = True,
                  run_flags: bool = True) -> Any:
    """``--data-dir``, ``--index-dir``, ``--index NAME=PATH``, ``--benchmarks``, ``--limit``,
    and (``run_flags``) ``--resume`` / ``--fail-fast``."""
    g = parser.add_argument_group("data")
    g.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="processed data root (data/README.md; or the synthetic layout)")
    g.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_ROOT,
                   help="root of the semantic indexes (<index-dir>/<corpus>/, built by build_index.py)")
    g.add_argument("--index", dest="index_overrides", action="append", default=[], type=_key_value,
                   metavar="CORPUS=PATH", help="explicit index directory of a corpus (repeatable)")
    if benchmarks:
        g.add_argument("--benchmarks", default=None,
                       help="comma list of benchmarks or aliases (scholarqabench, evidence, hypothesis, all); "
                            "default: every benchmark with a query file")
    if limit:
        g.add_argument("--limit", type=int, default=None, help="first N queries per benchmark (quick runs)")
    if run_flags:
        g.add_argument("--resume", action="store_true",
                       help="reuse cached per-query records (failed queries are retried)")
        g.add_argument("--force", action="store_true",
                       help="without --resume: replace cached records even if another configuration produced "
                            "them (refused by default)")
        g.add_argument("--fail-fast", action="store_true", help="abort on the first failing query")
    return g


def add_system_args(parser: argparse.ArgumentParser) -> Any:
    """Knobs of :class:`SystemOptions` and the link predictor."""
    g = parser.add_argument_group("systems")
    g.add_argument("--link-predictor", default="auto",
                   help="'auto' (<index>/link_predictor if trained), 'none' (untrained prior) or a directory")
    g.add_argument("--graph-candidates", type=int, default=100,
                   help="RAPTOR / LightRAG per-query candidate pool: top-k SPECTER2 papers (large corpora)")
    g.add_argument("--graph-global-max-papers", type=int, default=5000,
                   help="corpora with at most this many papers get one global RAPTOR tree / LightRAG graph")
    g.add_argument("--raptor-clustering", choices=("gmm", "kmeans"), default="gmm")
    g.add_argument("--raptor-no-umap", action="store_true", help="reduce with PCA instead of UMAP")
    g.add_argument("--rerank-top-n", type=int, default=50, help="Dense + reranker: SPECTER2 top-N reranked")
    g.add_argument("--organize-max-tokens", type=int, default=2048, help="output tokens of the hierarchy call")
    g.add_argument("--no-query-llm", action="store_true",
                   help="LLM-free Step 1 (retrieval.llm_query_understanding=false): ScholarPath then makes exactly "
                        "one LLM call per query, the count reported in Table 5")
    return g


def system_options_from_args(args: argparse.Namespace) -> SystemOptions:
    return SystemOptions(
        graph_candidates=int(getattr(args, "graph_candidates", 100)),
        graph_global_max_papers=int(getattr(args, "graph_global_max_papers", 5000)),
        raptor_clustering=str(getattr(args, "raptor_clustering", "gmm")),
        raptor_use_umap=not bool(getattr(args, "raptor_no_umap", False)),
        rerank_top_n=int(getattr(args, "rerank_top_n", 50)),
        organize_max_tokens=int(getattr(args, "organize_max_tokens", 2048)),
        retrieval_overrides={"llm_query_understanding": False} if getattr(args, "no_query_llm", False) else {},
    )


def index_store_from_args(args: argparse.Namespace) -> IndexStore:
    return IndexStore(args.index_dir, dict(getattr(args, "index_overrides", []) or []))


class EnvironmentCache:
    """Loads one :class:`QAEnvironment` per corpus on demand (benchmarks sharing a corpus share it)."""

    def __init__(self, runtime: Runtime, layout: DataLayout, store: IndexStore,
                 options: Optional[SystemOptions] = None, link_predictor: Optional[str] = "auto"):
        self.runtime = runtime
        self.layout = layout
        self.store = store
        self.options = options or SystemOptions()
        self.link_predictor = link_predictor
        self._envs: Dict[str, QAEnvironment] = {}
        self._names: Dict[str, str] = {}

    def for_benchmark(self, benchmark: str) -> QAEnvironment:
        return self.for_corpus(BENCHMARKS[benchmark].corpus)

    def for_corpus(self, corpus: str) -> QAEnvironment:
        name = self._names.get(corpus)
        if name is None:
            name = self._names[corpus] = self.store.resolve(corpus, self.layout)
        env = self._envs.get(name)
        if env is None:
            env = self._envs[name] = QAEnvironment.load(self.store.path(name), self.runtime, name=name,
                                                        link_predictor=self.link_predictor, options=self.options)
        return env

    def papers_for(self, benchmark: str) -> Dict[str, Paper]:
        """Paper records of a benchmark's corpus (from its index, else from the data layout)."""
        corpus = BENCHMARKS[benchmark].corpus
        try:
            return dict(self.for_corpus(corpus).index.papers)
        except FileNotFoundError:
            return dict(self.layout.load_corpus(corpus, with_chunks=False).papers)


def environment_summary(runtime: Runtime, options: Optional[SystemOptions] = None) -> Dict[str, Any]:
    """Backends, system options and the Table 5 call-count switch, stored with every result."""
    cfg = runtime.config
    overrides = dict(options.retrieval_overrides) if options is not None else {}
    return {
        "llm": {"provider": cfg.llm.provider, "model": cfg.llm.model},
        "embedding": {"backend": cfg.embedding.backend, "model": cfg.embedding.model_name},
        "judges": {"provider": cfg.evaluation.judge_provider, "models": list(cfg.evaluation.judge_models)},
        "llm_query_understanding": overrides.get("llm_query_understanding", cfg.retrieval.llm_query_understanding),
        "system_options": dataclasses.asdict(options) if options is not None else None,
        "seed": cfg.seed,
    }


__all__ = [
    "ABLATIONS",
    "AGENT_SYSTEMS",
    "AgentOptions",
    "AskResult",
    "BENCHMARKS",
    "BenchmarkEval",
    "BenchmarkSpec",
    "CORPORA",
    "CacheConflictError",
    "CorpusSpec",
    "DataLayout",
    "EVIDENCE_BENCHMARKS",
    "EmbeddingMismatchError",
    "EnvironmentCache",
    "FAILED_COLUMN",
    "IndexBuild",
    "IndexStore",
    "JudgeSet",
    "PAPER_BUDGET",
    "QAEnvironment",
    "QA_SYSTEMS",
    "RecordStore",
    "RunRecord",
    "SWEEP_BUDGETS",
    "SYSTEMS",
    "SystemOptions",
    "SystemSpec",
    "TABLE1",
    "TABLE2_SYSTEMS",
    "TABLE34_AGENT_ORDER",
    "TABLE4_SYSTEMS",
    "TABLE5_DETAIL",
    "TABLE7",
    "TABLE7_VARIANTS",
    "add_data_args",
    "add_system_args",
    "agent_fingerprint",
    "agent_task_query",
    "answer_hierarchy",
    "ask",
    "budget_sections",
    "build_agent",
    "build_index",
    "build_judges",
    "check_cache_conflicts",
    "collect_efficiency",
    "combine_retrieval",
    "compact_retrieval",
    "config_fingerprint",
    "domain_lookup",
    "efficiency_entry",
    "efficiency_summary",
    "embedding_identity",
    "embedding_mismatches",
    "evaluate_benchmark",
    "evaluate_organization",
    "evaluate_retrieval_records",
    "format_ask",
    "is_organization_query",
    "link_predictor_identity",
    "link_predictor_path",
    "load_link_predictor",
    "matplotlib_available",
    "measure_operator_stats",
    "online_latency",
    "operator_rows_from_records",
    "order_systems",
    "parse_budgets",
    "qa_table_row",
    "query_gold_count",
    "records_to_hierarchy",
    "resolve_benchmarks",
    "resolve_kind",
    "resolve_system",
    "resolve_systems",
    "restrict_taxonomy",
    "run_agent",
    "run_queries",
    "score_missing_as_zero",
    "split_csv",
    "summarize_operator_stats",
    "support_paths",
    "table1_rows",
    "table2_spec",
    "table3_spec",
    "unanswered_cell",
    "with_failed_column",
    "write_reduced_corpus",
]
