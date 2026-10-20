"""Step 1 of Algorithm 2: query understanding and aspect extraction (Section 4.1).

One lightweight LLM call (``prompts.retrieval``, component
``query_understanding``) identifies the problem and method concepts of q and
splits its information need into aspects A_q. Each aspect a has

* an importance weight w_a > 0 with sum_a w_a = 1,
* a unit-norm embedding x_a (``embedder.encode_queries``), and
* an evidential role in {overview, empirical, mechanism} that sets how much
  detail it needs (granularity weights gamma_{a, level}, Section 4.4).

For hypothesis queries each aspect also carries a *target*: the problem-side
or method-side premise of Eq. (9), or background evidence.

``config.llm_query_understanding = False`` selects an LLM-free fallback that
uses the whole query as a single aspect (and as the concept text of both
facets, so that grounding can still locate scopes by similarity).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from scholarpath.config import RetrievalConfig
from scholarpath.embeddings import Embedder, l2_normalize
from scholarpath.llm import JSONParseError, LLMClient
from scholarpath.prompts.retrieval import (
    TARGET_EVIDENCE,
    TARGET_METHOD_PREMISE,
    TARGET_PROBLEM_PREMISE,
    parse_query_understanding,
    query_understanding_request,
)
from scholarpath.types import EVIDENCE, HYPOTHESIS_ASSESSMENT, HYPOTHESIS_DISCOVERY, QUERY_KINDS, Query

log = logging.getLogger(__name__)

HYPOTHESIS_KINDS = (HYPOTHESIS_DISCOVERY, HYPOTHESIS_ASSESSMENT)


@dataclass
class Aspect:
    """One aspect a of the information need A_q."""

    text: str
    weight: float  # w_a > 0, sum over A_q is one
    role: str  # overview | empirical | mechanism
    vector: np.ndarray = field(repr=False)  # unit-norm x_a
    target: str = TARGET_EVIDENCE  # evidence | problem_premise | method_premise

    @property
    def is_premise(self) -> bool:
        return self.target in (TARGET_PROBLEM_PREMISE, TARGET_METHOD_PREMISE)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "weight": self.weight, "role": self.role, "target": self.target}


@dataclass
class QueryAnalysis:
    """Output of Step 1."""

    query_id: str
    text: str
    kind: str
    problem_concepts: List[str]
    method_concepts: List[str]
    aspects: List[Aspect]
    given_pair: Optional[Tuple[str, str]] = None  # (problem, method) for assessment
    searched_side: Optional[str] = None  # "method" | "problem" for discovery
    used_llm: bool = False

    @property
    def is_hypothesis(self) -> bool:
        return self.kind in HYPOTHESIS_KINDS

    @property
    def evidence_aspects(self) -> List[Tuple[int, Aspect]]:
        return [(i, a) for i, a in enumerate(self.aspects) if a.target == TARGET_EVIDENCE]

    def premise_aspects(self, side: str) -> List[Tuple[int, Aspect]]:
        """Premise aspects of the ``"problem"`` or ``"method"`` side (Eq. 9)."""
        target = TARGET_PROBLEM_PREMISE if side == "problem" else TARGET_METHOD_PREMISE
        return [(i, a) for i, a in enumerate(self.aspects) if a.target == target]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "kind": self.kind,
            "problem_concepts": list(self.problem_concepts),
            "method_concepts": list(self.method_concepts),
            "aspects": [a.to_dict() for a in self.aspects],
            "given_pair": list(self.given_pair) if self.given_pair else None,
            "searched_side": self.searched_side,
            "used_llm": self.used_llm,
        }


def _as_query(query: Union[Query, str]) -> Query:
    if isinstance(query, Query):
        return query
    return Query(query_id="q", text=str(query), kind="")


def _fallback_spec(text: str, kind: str) -> Dict[str, Any]:
    """LLM-free Step 1: the whole query is the single aspect and the concept text of both facets."""
    if kind in HYPOTHESIS_KINDS:
        # One aspect used for both premises of Eq. (9).
        aspects = [
            {"text": text, "role": "empirical", "weight": 0.5, "target": TARGET_PROBLEM_PREMISE},
            {"text": text, "role": "mechanism", "weight": 0.5, "target": TARGET_METHOD_PREMISE},
        ]
    else:
        aspects = [{"text": text, "role": "empirical", "weight": 1.0, "target": TARGET_EVIDENCE}]
    return {
        "kind": kind,
        "searched_side": "method" if kind == HYPOTHESIS_DISCOVERY else None,
        "problem_concepts": [text],
        "method_concepts": [text] if kind != HYPOTHESIS_DISCOVERY else [],
        "given_pair": (text, text) if kind == HYPOTHESIS_ASSESSMENT else None,
        "aspects": aspects,
    }


def _complete_spec(spec: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Guarantee a usable aspect set: hypothesis queries need both premises (Eq. 9)."""
    kind = spec["kind"]
    aspects = list(spec["aspects"])
    if kind in HYPOTHESIS_KINDS:
        targets = {a["target"] for a in aspects}
        if TARGET_PROBLEM_PREMISE not in targets:
            aspects.append({"text": text, "role": "empirical", "weight": 0.4, "target": TARGET_PROBLEM_PREMISE})
        if TARGET_METHOD_PREMISE not in targets:
            aspects.append({"text": text, "role": "mechanism", "weight": 0.4, "target": TARGET_METHOD_PREMISE})
        if kind == HYPOTHESIS_DISCOVERY and spec.get("searched_side") is None:
            spec["searched_side"] = (
                "problem" if spec["method_concepts"] and not spec["problem_concepts"] else "method"
            )
        if kind == HYPOTHESIS_ASSESSMENT and spec.get("given_pair") is None:
            p = spec["problem_concepts"][0] if spec["problem_concepts"] else text
            m = spec["method_concepts"][0] if spec["method_concepts"] else text
            spec["given_pair"] = (p, m)
    if not aspects:
        aspects = [{"text": text, "role": "empirical", "weight": 1.0, "target": TARGET_EVIDENCE}]
    spec["aspects"] = aspects
    return spec


def analyze_query(
    query: Union[Query, str],
    llm: Optional[LLMClient],
    embedder: Embedder,
    config: Optional[RetrievalConfig] = None,
) -> QueryAnalysis:
    """Algorithm 2, lines 1-2: parse q into concepts and weighted, role-tagged aspects.

    The benchmark-provided ``Query.kind`` is preferred when set. With
    ``config.llm_query_understanding`` disabled (or ``llm=None``) no LLM call
    is made and the whole query is used as one aspect.
    """
    config = config or RetrievalConfig()
    q = _as_query(query)
    kind_hint = q.kind if q.kind in QUERY_KINDS else None
    used_llm = False
    spec: Optional[Dict[str, Any]] = None
    if config.llm_query_understanding and llm is not None:
        request = query_understanding_request(q.text, kind_hint=kind_hint, max_aspects=config.max_aspects)
        try:
            obj, _ = llm.complete_json(request, expect=dict)
            spec = parse_query_understanding(obj, kind_hint=kind_hint, max_aspects=config.max_aspects)
            used_llm = True
        except JSONParseError as err:  # pragma: no cover - depends on the backend
            log.warning("Query understanding failed for %s (%s); using the LLM-free fallback", q.query_id, err)
    if spec is None:
        spec = _fallback_spec(q.text, kind_hint or EVIDENCE)
    spec = _complete_spec(spec, q.text)

    raw = spec["aspects"]
    total = float(sum(a["weight"] for a in raw)) or 1.0
    vecs = l2_normalize(np.asarray(embedder.encode_queries([a["text"] for a in raw]), dtype=np.float32))
    aspects = [
        Aspect(text=a["text"], weight=float(a["weight"]) / total, role=a["role"], vector=vecs[i], target=a["target"])
        for i, a in enumerate(raw)
    ]
    return QueryAnalysis(
        query_id=q.query_id,
        text=q.text,
        kind=spec["kind"],
        problem_concepts=list(spec["problem_concepts"]),
        method_concepts=list(spec["method_concepts"]),
        aspects=aspects,
        given_pair=tuple(spec["given_pair"]) if spec.get("given_pair") else None,
        searched_side=spec.get("searched_side"),
        used_llm=used_llm,
    )
