"""Query-understanding prompt for budgeted support-path retrieval (Algorithm 2, Step 1).

Step 1 of Algorithm 2 (Section 4.1) uses *one lightweight LLM call* to

* identify the problem and method concepts mentioned by the query q,
* decide the query kind (evidence / hypothesis discovery / hypothesis
  assessment), the searched side of a discovery query (candidate methods for
  a problem, or candidate problems for a method) and the given pair of an
  assessment query, and
* split the information need into aspects A_q, each with an importance weight
  w_a > 0 (weights sum to one) and an evidential role in
  {overview, empirical, mechanism} that sets how much detail it needs.

For hypothesis queries every aspect additionally names its *target*: the
problem-side premise (properties, data and baselines of the problem), the
method-side premise (mechanism and assumptions of the method), or background
evidence. Premise aspects enter the premise-bottleneck score of Eq. (9).

The answer composition prompt (Step 6) lives in :mod:`scholarpath.prompts.answer`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scholarpath.llm.base import LLMRequest, chat
from scholarpath.text import STOPWORDS, keywords, tokenize
from scholarpath.types import EVIDENCE, HYPOTHESIS_ASSESSMENT, HYPOTHESIS_DISCOVERY, QUERY_KINDS

ROLES = ("overview", "empirical", "mechanism")
TARGET_EVIDENCE = "evidence"
TARGET_PROBLEM_PREMISE = "problem_premise"
TARGET_METHOD_PREMISE = "method_premise"
TARGETS = (TARGET_EVIDENCE, TARGET_PROBLEM_PREMISE, TARGET_METHOD_PREMISE)

SYSTEM = (
    "You analyze research questions for a literature retrieval engine that indexes papers under a "
    "PROBLEM taxonomy (what is studied: tasks, application domains, phenomena) and a METHOD taxonomy "
    "(how it is studied: techniques, models, algorithms). Answer with a single JSON object."
)

_KIND_GUIDE = (
    '* "evidence": the user asks what the literature has ESTABLISHED about a problem and/or method '
    "(e.g. \"How have graph neural networks been used in drug discovery?\").\n"
    '* "hypothesis_discovery": the user asks for promising but UNSTUDIED directions, i.e. candidate '
    "methods for a problem (searched_side = \"method\") or candidate problems for a method "
    "(searched_side = \"problem\") (e.g. \"Which other methods could suit air-quality forecasting?\").\n"
    '* "hypothesis_assessment": the user asks whether a GIVEN problem-method pairing is plausible '
    "(e.g. \"Could graph contrastive learning work for air-quality forecasting?\")."
)

_ASPECT_GUIDE = (
    "Split the information need into at most {max_aspects} aspects. Each aspect has\n"
    "  - text: a short self-contained description of what evidence is needed,\n"
    "  - role: \"overview\" (structure of the area / sub-topics; concept summaries suffice),\n"
    "          \"empirical\" (concrete applications and results; needs paper abstracts),\n"
    "          \"mechanism\" (technical mechanism or assumption to verify; needs full-text detail),\n"
    "  - weight: importance in (0, 1]; weights are renormalized to sum to one,\n"
    "  - target: \"evidence\" for evidence queries; for hypothesis queries use\n"
    "          \"problem_premise\" (properties, data and baselines of the problem side),\n"
    "          \"method_premise\" (mechanism and assumptions of the method side), or\n"
    "          \"evidence\" (background on what is already established).\n"
    "Hypothesis queries must contain at least one problem_premise and one method_premise aspect."
)


def query_understanding_request(
    query: str,
    kind_hint: Optional[str] = None,
    max_aspects: int = 6,
    component: Optional[str] = "query_understanding",
) -> LLMRequest:
    """Build the single Step-1 call (task ``retrieval.query_understanding``).

    ``kind_hint`` is the benchmark-provided query kind when known; the model is
    asked to keep it.
    """
    hint = (
        f'The query kind is known to be "{kind_hint}"; keep it.\n'
        if kind_hint in QUERY_KINDS
        else "Decide the query kind:\n" + _KIND_GUIDE + "\n"
    )
    user = (
        f"Research question: {query}\n\n{hint}\n"
        "Identify the problem concepts and method concepts the question mentions, using the wording a "
        "taxonomy would use (short noun phrases, no question words). Leave a list empty if the question "
        "does not mention that facet. For hypothesis_assessment also give the pair being judged.\n\n"
        + _ASPECT_GUIDE.format(max_aspects=max_aspects)
        + "\n\nAnswer JSON:\n"
        '{"kind": "evidence" | "hypothesis_discovery" | "hypothesis_assessment",\n'
        ' "searched_side": "method" | "problem" | null,\n'
        ' "problem_concepts": ["..."], "method_concepts": ["..."],\n'
        ' "given_pair": {"problem": "...", "method": "..."} | null,\n'
        ' "aspects": [{"text": "...", "role": "overview" | "empirical" | "mechanism",\n'
        '              "weight": 0.3, "target": "evidence" | "problem_premise" | "method_premise"}]}'
    )
    payload = {"query": query, "kind_hint": kind_hint, "max_aspects": int(max_aspects)}
    return LLMRequest(
        task="retrieval.query_understanding",
        messages=chat(SYSTEM, user),
        payload=payload,
        component=component,
        max_tokens=600,
        json_mode=True,
    )


def _clean_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out: List[str] = []
    for v in value:
        s = str(v).strip()
        if s and s.lower() not in {"none", "null", "n/a"} and s not in out:
            out.append(s)
    return out


def parse_query_understanding(
    obj: Any, kind_hint: Optional[str] = None, max_aspects: int = 6
) -> Dict[str, Any]:
    """Validate and normalize the Step-1 JSON.

    Returns a dict with keys ``kind``, ``searched_side``, ``problem_concepts``,
    ``method_concepts``, ``given_pair`` (tuple or None) and ``aspects``
    (list of ``{"text", "role", "weight", "target"}``, weights unnormalized).
    """
    obj = obj if isinstance(obj, dict) else {}
    kind = str(obj.get("kind") or "").strip().lower()
    if kind_hint in QUERY_KINDS:
        kind = kind_hint
    elif kind not in QUERY_KINDS:
        kind = EVIDENCE
    side = obj.get("searched_side")
    side = str(side).strip().lower() if side else None
    if side not in ("method", "problem"):
        side = None
    pair = obj.get("given_pair")
    given: Optional[Tuple[str, str]] = None
    if isinstance(pair, dict) and pair.get("problem") and pair.get("method"):
        given = (str(pair["problem"]).strip(), str(pair["method"]).strip())
    elif isinstance(pair, (list, tuple)) and len(pair) == 2 and all(pair):
        given = (str(pair[0]).strip(), str(pair[1]).strip())
    aspects: List[Dict[str, Any]] = []
    for a in obj.get("aspects", []) or []:
        if not isinstance(a, dict) or not str(a.get("text", "")).strip():
            continue
        role = str(a.get("role", "empirical")).strip().lower()
        if role not in ROLES:
            role = "mechanism" if role.startswith(("mech", "assum", "verif")) else (
                "overview" if role.startswith("over") else "empirical")
        target = str(a.get("target", TARGET_EVIDENCE)).strip().lower()
        if target not in TARGETS:
            target = TARGET_PROBLEM_PREMISE if target.startswith("prob") else (
                TARGET_METHOD_PREMISE if target.startswith("meth") else TARGET_EVIDENCE)
        if kind == EVIDENCE:
            target = TARGET_EVIDENCE
        try:
            weight = float(a.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if not weight > 0:
            weight = 1.0
        aspects.append({"text": str(a["text"]).strip(), "role": role, "weight": weight, "target": target})
    return {
        "kind": kind,
        "searched_side": side if kind == HYPOTHESIS_DISCOVERY else None,
        "problem_concepts": _clean_list(obj.get("problem_concepts")),
        "method_concepts": _clean_list(obj.get("method_concepts")),
        "given_pair": given if kind == HYPOTHESIS_ASSESSMENT else None,
        "aspects": aspects[: max(1, int(max_aspects))],
    }


# ---------------------------------------------------------------------------
# Offline mock: deterministic pattern-based parser
# ---------------------------------------------------------------------------

_EVIDENCE_PATTERNS: Sequence[re.Pattern] = (
    # "How has/have X been used/applied for/to/in Y?"
    re.compile(
        r"^\s*how\s+(?:has|have|is|are|was|were|do|does|did|can)\s+(?P<method>.+?)\s+(?:been\s+)?"
        r"(?:used|applied|adapted|employed|leveraged|utilized)\s+(?:for|to|in|on)\s+(?P<problem>.+?)\s*[?.!]*\s*$",
        re.IGNORECASE,
    ),
    # "What methods/approaches are used for Y?"
    re.compile(
        r"^\s*what\s+(?:methods|approaches|techniques|models)\s+(?:are|have been)\s+(?:used|applied|proposed)"
        r"\s+(?:for|to|in)\s+(?P<problem>.+?)\s*[?.!]*\s*$",
        re.IGNORECASE,
    ),
)
_DISCOVER_METHOD = re.compile(
    r"^\s*(?:which|what)\s+(?:other\s+|new\s+|alternative\s+)?(?:methods|approaches|techniques|models)"
    r"\s+(?:could|might|can|would)\s+(?:be\s+applied\s+to|be\s+used\s+for|suit|solve|address|help(?:\s+with)?)"
    r"\s+(?P<problem>.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_DISCOVER_PROBLEM = re.compile(
    r"^\s*(?:which|what)\s+(?:other\s+|new\s+)?(?:problems|tasks|applications|domains)\s+(?:could|might|can|would)"
    r"\s+(?P<method>.+?)\s+(?:be\s+applied\s+to|solve|address|help(?:\s+with)?|be\s+used\s+for)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_ASSESS = re.compile(
    r"^\s*(?:could|can|would|might|is|does)\s+(?P<method>.+?)\s+(?:be\s+applied\s+to|work\s+for|help(?:\s+with)?|"
    r"suit|be\s+used\s+for|improve)\s+(?P<problem>.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_GENERIC_METHOD_WORDS = {"other", "methods", "method", "approaches", "techniques", "models", "new", "alternative"}


def _short(text: str, k: int = 6) -> str:
    toks = [t for t in tokenize(text) if t not in STOPWORDS] or tokenize(text)
    return " ".join(toks[:k])


def _mock_parse(query: str, kind_hint: Optional[str]) -> Dict[str, Any]:
    problem: Optional[str] = None
    method: Optional[str] = None
    kind = kind_hint if kind_hint in QUERY_KINDS else None
    side: Optional[str] = None
    m = _DISCOVER_METHOD.match(query)
    if m:
        problem, kind, side = m.group("problem"), kind or HYPOTHESIS_DISCOVERY, "method"
    if problem is None:
        m = _DISCOVER_PROBLEM.match(query)
        if m:
            method, kind, side = m.group("method"), kind or HYPOTHESIS_DISCOVERY, "problem"
    if problem is None and method is None:
        for pat in _EVIDENCE_PATTERNS:
            m = pat.match(query)
            if m:
                gd = m.groupdict()
                problem, method = gd.get("problem"), gd.get("method")
                kind = kind or EVIDENCE
                break
    if problem is None and method is None:
        m = _ASSESS.match(query)
        if m:
            problem, method = m.group("problem"), m.group("method")
            kind = kind or HYPOTHESIS_ASSESSMENT
    kind = kind or EVIDENCE
    if problem is None and method is None:
        problem = " ".join(keywords(query, k=6)) or query
    if method and set(tokenize(method)) <= _GENERIC_METHOD_WORDS:
        method = None
    if kind == HYPOTHESIS_DISCOVERY and side is None:
        side = "problem" if (method and not problem) else "method"
    return {"problem": problem.strip() if problem else None, "method": method.strip() if method else None,
            "kind": kind, "side": side}


def _mock_query_understanding(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    query = str(payload.get("query", ""))
    parsed = _mock_parse(query, payload.get("kind_hint"))
    kind, problem, method, side = parsed["kind"], parsed["problem"], parsed["method"], parsed["side"]
    p = problem or "the problem"
    mth = method or "the method"
    if kind == EVIDENCE:
        aspects = [
            {"text": f"Overview of the sub-topics of {p} addressed with {mth}", "role": "overview",
             "weight": 0.25, "target": TARGET_EVIDENCE},
            {"text": f"Empirical applications and results of {mth} for {p}", "role": "empirical",
             "weight": 0.45, "target": TARGET_EVIDENCE},
            {"text": f"How {mth} is adapted to {p}: mechanism and assumptions", "role": "mechanism",
             "weight": 0.30, "target": TARGET_EVIDENCE},
        ]
    elif kind == HYPOTHESIS_DISCOVERY and side == "problem":
        aspects = [
            {"text": f"Established applications of {mth}", "role": "overview", "weight": 0.2,
             "target": TARGET_EVIDENCE},
            {"text": f"Problems with structure and data that {mth} can model: {_short(query)}",
             "role": "empirical", "weight": 0.4, "target": TARGET_PROBLEM_PREMISE},
            {"text": f"Mechanism and assumptions of {mth}", "role": "mechanism", "weight": 0.4,
             "target": TARGET_METHOD_PREMISE},
        ]
    else:  # discovery of methods, or assessment of a given pair
        aspects = [
            {"text": f"Existing approaches and baselines for {p}", "role": "overview", "weight": 0.2,
             "target": TARGET_EVIDENCE},
            {"text": f"Properties, data and baselines of {p}", "role": "empirical", "weight": 0.4,
             "target": TARGET_PROBLEM_PREMISE},
            {"text": f"Mechanism and assumptions of {mth if method else 'methods'} that can model {p}",
             "role": "mechanism", "weight": 0.4, "target": TARGET_METHOD_PREMISE},
        ]
    max_aspects = int(payload.get("max_aspects", 6) or 6)
    return {
        "kind": kind,
        "searched_side": side if kind == HYPOTHESIS_DISCOVERY else None,
        "problem_concepts": [problem] if problem else [],
        "method_concepts": [method] if method else [],
        "given_pair": {"problem": p, "method": mth} if kind == HYPOTHESIS_ASSESSMENT else None,
        "aspects": aspects[:max_aspects],
    }


MOCK_HANDLERS = {"retrieval.query_understanding": _mock_query_understanding}
