"""Core data records shared by every ScholarPath module.

The multi-granularity semantic index (Section 2.1) has three levels:

* **concepts** -- nodes of the problem taxonomy T_P and the method taxonomy
  T_M (see :mod:`scholarpath.taxonomy.tree`);
* **paper summaries** -- one :class:`Paper` per paper, carrying title and
  abstract;
* **paper details** -- :class:`Chunk` records obtained by splitting the full
  text by section.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Chunk:
    """A section chunk of a paper's full text (paper-detail level)."""

    chunk_id: str
    paper_id: str
    heading: str
    text: str
    order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(d["chunk_id"]),
            paper_id=str(d["paper_id"]),
            heading=str(d.get("heading", "")),
            text=str(d.get("text", "")),
            order=int(d.get("order", 0)),
        )


@dataclass
class Paper:
    """A paper of the scholarly repository P (paper-summary level)."""

    paper_id: str
    title: str
    abstract: str = ""
    year: Optional[int] = None
    date: Optional[str] = None  # ISO date of first version, e.g. "2021-03-04"
    categories: List[str] = field(default_factory=list)
    primary_category: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    arxiv_id: Optional[str] = None
    s2_id: Optional[str] = None
    doi: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Title + abstract (the paper summary)."""
        if self.abstract:
            return f"{self.title}. {self.abstract}"
        return self.title

    def to_dict(self, include_chunks: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not include_chunks:
            d["chunks"] = []
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Paper":
        chunks = [Chunk.from_dict(c) for c in d.get("chunks", []) or []]
        year = d.get("year")
        return cls(
            paper_id=str(d["paper_id"]),
            title=str(d.get("title", "") or ""),
            abstract=str(d.get("abstract", "") or ""),
            year=int(year) if year not in (None, "") else None,
            date=d.get("date"),
            categories=list(d.get("categories", []) or []),
            primary_category=d.get("primary_category"),
            authors=list(d.get("authors", []) or []),
            arxiv_id=d.get("arxiv_id"),
            s2_id=d.get("s2_id"),
            doi=d.get("doi"),
            chunks=chunks,
            metadata=dict(d.get("metadata", {}) or {}),
        )


# Query kinds (Section 2.1, "Support Paths").
EVIDENCE = "evidence"
HYPOTHESIS_DISCOVERY = "hypothesis_discovery"
HYPOTHESIS_ASSESSMENT = "hypothesis_assessment"
QUERY_KINDS = (EVIDENCE, HYPOTHESIS_DISCOVERY, HYPOTHESIS_ASSESSMENT)


@dataclass
class RubricItem:
    text: str
    weight: float = 1.0
    kind: str = "content"  # content | citation | ... (ResearchQA: comma-joined official type list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Query:
    """A benchmark query with its gold annotations."""

    query_id: str
    text: str
    kind: str = EVIDENCE
    benchmark: str = ""
    gold_papers: List[str] = field(default_factory=list)  # aligned corpus paper ids
    unaligned_gold: List[str] = field(default_factory=list)  # gold refs not found in the corpus
    rubric: List[RubricItem] = field(default_factory=list)
    reference_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_gold(self) -> int:
        """Gold count including unaligned citations (kept in the denominator)."""
        return len(self.gold_papers) + len(self.unaligned_gold)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Query":
        rubric = [
            RubricItem(**r) if isinstance(r, dict) else RubricItem(text=str(r))
            for r in d.get("rubric", []) or []
        ]
        return cls(
            query_id=str(d["query_id"]),
            text=str(d["text"]),
            kind=str(d.get("kind", EVIDENCE)),
            benchmark=str(d.get("benchmark", "")),
            gold_papers=[str(x) for x in d.get("gold_papers", []) or []],
            unaligned_gold=[str(x) for x in d.get("unaligned_gold", []) or []],
            rubric=rubric,
            reference_answer=d.get("reference_answer"),
            metadata=dict(d.get("metadata", {}) or {}),
        )


@dataclass
class Citation:
    paper_id: str
    chunk_id: Optional[str] = None
    status: str = "established"  # "established" | "hypothesis"


@dataclass
class AnswerItem:
    """One statement of a composed answer: an established finding or a hypothesis."""

    text: str
    status: str = "established"  # established | hypothesis
    citations: List[str] = field(default_factory=list)  # paper ids
    candidate: Optional[str] = None  # proposed counterpart concept (hypotheses)
    rationale: Optional[str] = None
    score: Optional[float] = None


@dataclass
class Answer:
    """System output for a query."""

    query_id: str
    text: str
    items: List[AnswerItem] = field(default_factory=list)
    retrieved_papers: List[str] = field(default_factory=list)  # ranked
    context_tokens: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cited_papers(self) -> List[str]:
        seen: Dict[str, None] = {}
        for it in self.items:
            for c in it.citations:
                seen.setdefault(c, None)
        return list(seen)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Answer":
        items = [AnswerItem(**it) for it in d.get("items", []) or []]
        kwargs = {k: v for k, v in d.items() if k != "items"}
        return cls(items=items, **kwargs)


def unique(seq: Sequence[str]) -> List[str]:
    """Order-preserving de-duplication."""
    seen: Dict[str, None] = {}
    for x in seq:
        seen.setdefault(x, None)
    return list(seen)
