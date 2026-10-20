"""Retrieval output records shared by ScholarPath and every retrieval baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

ESTABLISHED = "established"
HYPOTHESIS = "hypothesis"


@dataclass
class RetrievedUnit:
    """One selected context unit.

    For ScholarPath a unit is a direct path or a complementary path set
    (Section 2.1); for baselines it is a passage, abstract, summary node,
    entity/relation description, etc.
    """

    key: str
    kind: str  # direct | complementary | passage | abstract | summary | graph | ...
    text: str = ""
    tokens: int = 0  # marginal tokens this unit added to the context
    score: float = 0.0  # selection score (e.g. gain per new token)
    papers: List[str] = field(default_factory=list)  # source papers cited by this unit
    nodes: List[str] = field(default_factory=list)  # index node keys (ScholarPath)
    status: str = ESTABLISHED  # established | hypothesis
    candidate: Optional[str] = None  # proposed counterpart concept (hypotheses)
    candidate_id: Optional[str] = None
    aspects: Dict[str, float] = field(default_factory=dict)  # aspect -> adequacy
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    query_id: str
    units: List[RetrievedUnit] = field(default_factory=list)  # in selection order
    context: str = ""  # rendered context passed to the answer composer
    context_tokens: int = 0  # C(S) <= B
    budget: int = 0
    ranked_papers: List[str] = field(default_factory=list)  # distinct papers in selection order
    candidates: List[str] = field(default_factory=list)  # ranked hypothesis candidates (names)
    sources: List[Dict[str, Any]] = field(default_factory=list)  # structured sources for the composer
    latency_s: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def papers_in_order(units: List[RetrievedUnit]) -> List[str]:
        seen: Dict[str, None] = {}
        for u in units:
            for p in u.papers:
                seen.setdefault(p, None)
        return list(seen)
