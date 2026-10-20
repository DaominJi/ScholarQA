"""Shared building blocks of the Section 5.3 retrieval and QA baselines.

Budgeted methods (Section 5.3, "Baselines") strictly enforce a user-specified
context token budget B (Definition 2; default 8,192). Every budgeted
retriever implements :class:`BudgetedRetriever` and returns the foundation
:class:`~scholarpath.retrieval.result.RetrievalResult`, so that retrieval
accuracy (Table 2, Fig. 5: Recall@B / NDCG@B over ``ranked_papers``) and
end-to-end QA (Tables 3-5: one composition call over ``context``) are
evaluated identically for ScholarPath and the baselines.

Retrievable units
-----------------
The flat baselines (BM25, SPECTER, Dense + reranker, keyword search) and the
leaves of RAPTOR / text units of LightRAG are drawn from one *passage pool*
built from the same multi-granularity data ScholarPath indexes (Section 2.1):

* one **abstract** unit per paper, ``"<title>. <abstract>"``, keyed
  ``paper:<paper_id>`` (same key as the paper-summary node of the
  :class:`~scholarpath.index.semantic_index.SemanticIndex`);
* one **chunk** unit per section chunk of the full text, keyed
  ``chunk:<chunk_id>`` (same key as the index's paper-detail node).

Each unit carries its source paper id(s) and its token cost c(x)
(:func:`scholarpath.text.count_tokens` of the rendered block that enters the
context). Derived units of RAPTOR (``summary``) and LightRAG (``entity`` /
``relation``) use the same :class:`Passage` record with several source papers.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

try:  # Python >= 3.8
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

from scholarpath.data.corpus import Corpus
from scholarpath.llm.accounting import UsageRecord
from scholarpath.retrieval.result import RetrievalResult
from scholarpath.text import count_tokens
from scholarpath.types import EVIDENCE, Chunk, Paper, Query

log = logging.getLogger(__name__)

# Unit kinds.
ABSTRACT = "abstract"
CHUNK = "chunk"
SUMMARY = "summary"  # RAPTOR summary node
ENTITY = "entity"  # LightRAG entity description
RELATION = "relation"  # LightRAG relation description

#: Default context token budget B (Definition 2, Section 5.1).
DEFAULT_BUDGET = 8192

#: Default depth of the candidate ranking handed to the packer. At B = 8,192
#: and 100-500 tokens per unit, a few dozen units fill the budget, so 2,000
#: ranks are never exhausted in practice; the cap only bounds the work of
#: skip-and-continue packing when the budget is almost full.
DEFAULT_MAX_CANDIDATES = 2000


QueryLike = Union[Query, str]


def as_query(query: QueryLike, query_id: Optional[str] = None) -> Query:
    """Coerce a string into a :class:`~scholarpath.types.Query` (evidence kind)."""
    if isinstance(query, Query):
        return query
    text = str(query)
    qid = query_id or "q-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return Query(query_id=qid, text=text, kind=EVIDENCE)


@runtime_checkable
class BudgetedRetriever(Protocol):
    """A retrieval baseline that packs at most ``budget`` context tokens."""

    name: str

    def retrieve(self, query: QueryLike, budget: int) -> RetrievalResult:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Passages
# ---------------------------------------------------------------------------


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").lower().split())


@dataclass(eq=False)
class Passage:
    """One retrievable context unit.

    ``papers`` lists the source paper ids (one for abstracts / chunks, several
    for RAPTOR summaries and LightRAG graph units, ordered by contribution).
    ``title`` is the paper title for abstracts / chunks and a label (e.g.
    ``"Entity: GNN (method)"``) for derived units. ``index_text`` overrides the
    text used for lexical / dense indexing (:attr:`search_text`).
    """

    key: str
    kind: str
    papers: List[str]
    title: str = ""
    heading: str = ""
    body: str = ""
    index_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _cost: Optional[int] = field(default=None, init=False, repr=False)

    @property
    def paper_id(self) -> str:
        return self.papers[0] if self.papers else ""

    @property
    def search_text(self) -> str:
        """Text indexed by BM25 / dense retrieval (mirrors ``SemanticIndex.text``)."""
        if self.index_text is not None:
            return self.index_text
        if self.kind == ABSTRACT:
            return f"{self.title}. {self.body}" if self.body else self.title
        if self.kind == CHUNK:
            return f"{self.heading}\n{self.body}" if self.heading else self.body
        return f"{self.title}\n{self.body}" if self.title else self.body

    def render(self) -> str:
        """The block this unit contributes to the context, in the same
        ``[paper_id] ...`` citation style as ScholarPath's context assembly."""
        cite = f"[{', '.join(self.papers)}] " if self.papers else ""  # uncited derived units: no id list
        if self.kind == CHUNK and self.heading:
            header = f"{cite}{self.title} -- {self.heading}"
        else:
            header = f"{cite}{self.title}".rstrip()
        return f"{header}\n{self.body}" if self.body else header

    @property
    def cost(self) -> int:
        """Token cost c(x) of the rendered block (cached)."""
        if self._cost is None:
            self._cost = max(1, count_tokens(self.render()))
        return self._cost

    @property
    def dedup_key(self) -> str:
        """Identity used to drop duplicate texts during packing."""
        return hashlib.sha1(_normalize_ws(self.search_text).encode("utf-8")).hexdigest()


@dataclass
class Candidate:
    """A ranked candidate unit handed to the packer."""

    passage: Passage
    score: float = 0.0
    source: str = ""  # ranking that produced it (e.g. "bm25", "local", "global")


CandidateLike = Union[Candidate, Passage, Tuple[Passage, float]]


def as_candidate(c: CandidateLike) -> Candidate:
    if isinstance(c, Candidate):
        return c
    if isinstance(c, Passage):
        return Candidate(c, 0.0)
    p, s = c
    return Candidate(p, float(s))


# ---------------------------------------------------------------------------
# Passage pool
# ---------------------------------------------------------------------------

PapersLike = Union[Corpus, Mapping[str, Paper], Iterable[Paper]]


def _iter_papers(papers: Any) -> Iterator[Paper]:
    if hasattr(papers, "papers") and isinstance(papers.papers, Mapping):
        yield from papers.papers.values()  # Corpus or SemanticIndex
    elif isinstance(papers, Mapping):
        yield from papers.values()
    else:
        yield from papers


class PassagePool:
    """An ordered pool of :class:`Passage` units with lookup helpers."""

    def __init__(self, passages: Sequence[Passage], titles: Optional[Mapping[str, str]] = None, name: str = "pool"):
        self.name = name
        self.passages: List[Passage] = list(passages)
        if titles is None:
            titles = {}
            for p in self.passages:
                if p.kind in (ABSTRACT, CHUNK) and p.papers:
                    titles.setdefault(p.paper_id, p.title)
        self.titles: Dict[str, str] = dict(titles)
        self._key_to_idx: Dict[str, int] = {}
        for i, p in enumerate(self.passages):
            if p.key in self._key_to_idx:
                raise ValueError(f"Duplicate passage key {p.key!r}")
            self._key_to_idx[p.key] = i
        self._by_paper: Optional[Dict[str, List[int]]] = None
        self._fingerprint: Optional[str] = None

    # ----------------------------------------------------------- basics
    def __len__(self) -> int:
        return len(self.passages)

    def __getitem__(self, i: int) -> Passage:
        return self.passages[i]

    def __iter__(self) -> Iterator[Passage]:
        return iter(self.passages)

    @property
    def keys(self) -> List[str]:
        return [p.key for p in self.passages]

    def index_of(self, key: str) -> Optional[int]:
        return self._key_to_idx.get(key)

    def get(self, key: str) -> Optional[Passage]:
        i = self._key_to_idx.get(key)
        return None if i is None else self.passages[i]

    def search_texts(self, indices: Optional[Sequence[int]] = None) -> List[str]:
        if indices is None:
            return [p.search_text for p in self.passages]
        return [self.passages[i].search_text for i in indices]

    @property
    def paper_ids(self) -> List[str]:
        return list(self.by_paper)

    @property
    def by_paper(self) -> Dict[str, List[int]]:
        """paper id -> indices of its units (abstract first, then chunks in order)."""
        if self._by_paper is None:
            idx: Dict[str, List[int]] = {}
            for i, p in enumerate(self.passages):
                for pid in p.papers[:1]:
                    idx.setdefault(pid, []).append(i)
            self._by_paper = idx
        return self._by_paper

    def indices_for_papers(self, paper_ids: Iterable[str], kinds: Optional[Sequence[str]] = None) -> List[int]:
        """Indices of the units of ``paper_ids`` (in the given paper order)."""
        out: List[int] = []
        allowed = set(kinds) if kinds else None
        for pid in paper_ids:
            for i in self.by_paper.get(pid, []):
                if allowed is None or self.passages[i].kind in allowed:
                    out.append(i)
        return out

    def subset(self, indices: Sequence[int], name: Optional[str] = None) -> "PassagePool":
        return PassagePool([self.passages[i] for i in indices], titles=self.titles, name=name or self.name)

    def filter(self, kinds: Sequence[str], name: Optional[str] = None) -> "PassagePool":
        allowed = set(kinds)
        return self.subset([i for i, p in enumerate(self.passages) if p.kind in allowed], name=name)

    def fingerprint(self) -> str:
        """Stable hash of the ordered unit keys (validates on-disk caches)."""
        if self._fingerprint is None:
            h = hashlib.sha1()
            for p in self.passages:
                h.update(p.key.encode("utf-8"))
                h.update(b"\x00")
            self._fingerprint = h.hexdigest()
        return self._fingerprint

    def stats(self) -> Dict[str, Any]:
        kinds: Dict[str, int] = {}
        for p in self.passages:
            kinds[p.kind] = kinds.get(p.kind, 0) + 1
        return {"units": len(self.passages), "papers": len(self.by_paper), "kinds": kinds}

    # ------------------------------------------------------ constructors
    @classmethod
    def from_papers(
        cls,
        papers: PapersLike,
        include_abstracts: bool = True,
        include_chunks: bool = True,
        paper_ids: Optional[Iterable[str]] = None,
        chunks: Optional[Mapping[str, Sequence[Chunk]]] = None,
        chunk_title_in_search: bool = False,
        name: str = "pool",
    ) -> "PassagePool":
        """Build the pool from papers (a :class:`~scholarpath.data.corpus.Corpus`,
        a ``{paper_id: Paper}`` mapping, or an iterable of papers).

        ``chunks`` optionally supplies the section chunks per paper (otherwise
        ``Paper.chunks`` is used). With ``chunk_title_in_search`` the paper
        title is prepended to the indexed text of chunks.
        """
        wanted = set(paper_ids) if paper_ids is not None else None
        passages: List[Passage] = []
        titles: Dict[str, str] = {}
        for paper in _iter_papers(papers):
            pid = paper.paper_id
            if wanted is not None and pid not in wanted:
                continue
            titles[pid] = paper.title
            if include_abstracts:
                passages.append(Passage(f"paper:{pid}", ABSTRACT, [pid], title=paper.title, body=paper.abstract))
            if include_chunks:
                pchunks = chunks.get(pid, []) if chunks is not None else paper.chunks
                for ch in sorted(pchunks, key=lambda c: c.order):
                    if not (ch.text or "").strip():
                        continue
                    p = Passage(f"chunk:{ch.chunk_id}", CHUNK, [pid], title=paper.title, heading=ch.heading,
                                body=ch.text)
                    if chunk_title_in_search:
                        p.index_text = f"{paper.title}. {p.search_text}"
                    passages.append(p)
        return cls(passages, titles=titles, name=name)

    @classmethod
    def from_corpus(cls, corpus: PapersLike, **kwargs: Any) -> "PassagePool":
        """Alias of :meth:`from_papers`."""
        return cls.from_papers(corpus, **kwargs)

    @classmethod
    def from_index(cls, index: Any, **kwargs: Any) -> "PassagePool":
        """Build the pool from a :class:`~scholarpath.index.semantic_index.SemanticIndex`
        (whose paper records are stored without chunks)."""
        chunks: Dict[str, List[Chunk]] = {}
        for pid, cids in index.paper_chunks.items():
            chunks[pid] = [index.chunks[c] for c in cids]
        kwargs.setdefault("name", "index")
        return cls.from_papers(index.papers, chunks=chunks, **kwargs)


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def top_k_indices(scores: np.ndarray, k: int, min_score: Optional[float] = None) -> np.ndarray:
    """Indices of the ``k`` largest scores, sorted by (score desc, index asc).

    ``min_score`` keeps only entries with ``score > min_score``. Uses
    ``argpartition`` so the cost is O(n + k log k).
    """
    scores = np.asarray(scores)
    if scores.ndim != 1 or k <= 0 or scores.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if min_score is not None:
        cand = np.flatnonzero(scores > min_score)
        sub = scores[cand]
    else:
        cand = None
        sub = scores
    n = sub.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = min(int(k), n)
    if k < n:
        part = np.argpartition(-sub, k - 1)[:k]
    else:
        part = np.arange(n)
    order = part[np.lexsort((part, -sub[part]))]
    return (cand[order] if cand is not None else order).astype(np.int64)


def distinct_papers(pool: PassagePool, indices: Iterable[int], k: Optional[int] = None) -> List[str]:
    """Distinct source papers of ranked unit indices, in rank order."""
    seen: Dict[str, None] = {}
    for i in indices:
        for pid in pool[int(i)].papers:
            seen.setdefault(pid, None)
            if k is not None and len(seen) >= k:
                return list(seen)
    return list(seen)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Any]], k: int = 60) -> List[Any]:
    """Reciprocal rank fusion of several rankings (used by the simulated agent)."""
    scores: Dict[Any, float] = {}
    first: Dict[Any, Tuple[int, int]] = {}
    for r_i, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
            first.setdefault(item, (rank, r_i))
    return sorted(scores, key=lambda x: (-scores[x], first[x]))


# ---------------------------------------------------------------------------
# Usage / latency helpers shared by BudgetedQA and the agents
# ---------------------------------------------------------------------------

#: Cost components excluded from per-query usage: evaluation judges, and the
#: offline index construction of RAPTOR / LightRAG (Table 5 counts only the
#: online calls: LightRAG = keyword extraction + composition).
EXCLUDED_QUERY_COMPONENTS: Tuple[str, ...] = ("judge", "index_build")


def split_usage(
    diff: Mapping[str, UsageRecord], exclude: Sequence[str] = EXCLUDED_QUERY_COMPONENTS
) -> Tuple[UsageRecord, UsageRecord]:
    """(counted usage, excluded usage) of a :meth:`CostMeter.diff`."""
    counted, excluded = UsageRecord(), UsageRecord()
    for name, rec in diff.items():
        (excluded if name in exclude else counted).add(rec)
    return counted, excluded


def ensure_built(retriever: Any) -> float:
    """Build a lazily indexed retriever now; returns the seconds spent (0 if built).

    Retrievers with lazy offline indexing (dense passage encoding, RAPTOR
    trees, LightRAG graphs) expose ``ensure_built()``; they call it at the
    start of :meth:`~BudgetedRetriever.retrieve`, report the time as
    ``RetrievalResult.stats["index_build_s"]`` and exclude it from
    ``RetrievalResult.latency_s``, so that per-query latency (Table 5) never
    includes offline index construction.
    """
    fn = getattr(retriever, "ensure_built", None)
    return float(fn()) if callable(fn) else 0.0


def index_build_seconds(result: Optional[RetrievalResult]) -> float:
    """Offline index-construction time a retrieval reported (``stats["index_build_s"]``)."""
    if result is None:
        return 0.0
    try:
        return max(0.0, float(result.stats.get("index_build_s", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def estimate_latency(wall_s: float, usage: UsageRecord) -> float:
    """End-to-end latency of a query given its wall-clock time and LLM usage.

    Fresh LLM calls are already part of the wall-clock time. Calls served from
    the on-disk response cache return instantly but are billed with their
    originally recorded latency (``LLMConfig.bill_cached_calls``); their share
    of the recorded LLM latency is added back so that latencies stay
    reproducible across cached re-runs.
    """
    if usage.calls <= 0 or usage.cached_calls <= 0:
        return float(wall_s)
    cached_share = usage.cached_calls / float(usage.calls)
    return float(wall_s) + cached_share * float(usage.latency_s)

