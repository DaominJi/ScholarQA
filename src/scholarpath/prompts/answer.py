"""Answer composition prompt (Algorithm 2, Step 6; Section 5.3).

Every budgeted method -- ScholarPath, keyword search, dense retrieval,
LightRAG, and the unassisted LLM -- packs at most B context tokens and
invokes the backbone LLM in a *single* composition call with this prompt,
so that end-to-end comparisons differ only in the retrieved context.

The answer is a JSON object whose items are labelled as an
``established`` finding or a ``hypothesis`` and cite source papers by id.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scholarpath.llm.base import LLMRequest, chat
from scholarpath.types import EVIDENCE, HYPOTHESIS_ASSESSMENT, HYPOTHESIS_DISCOVERY, AnswerItem

SYSTEM = (
    "You are a meticulous research assistant. You answer questions about the scientific literature "
    "using ONLY the provided context, cite source papers by their ids in square brackets, and clearly "
    "separate established findings (directly supported by cited papers) from hypotheses (unstudied "
    "directions supported by indirect evidence). Answer with a single JSON object."
)

_KIND_INSTRUCTIONS = {
    EVIDENCE: (
        "This is an EVIDENCE query: synthesize what the literature has established. Organize the answer "
        "by the sub-topics in the context, state concrete methods and findings, and cite every claim. "
        "All items must have status \"established\"."
    ),
    HYPOTHESIS_DISCOVERY: (
        "This is a HYPOTHESIS query: propose promising but UNSTUDIED research directions (e.g. candidate "
        "methods for the problem, or candidate problems for the method). For each candidate, give the "
        "candidate name, a rationale grounded in indirect evidence (what the target problem requires, and "
        "what the candidate's mechanism and assumptions provide), and cite the supporting papers on both "
        "sides. Rank candidates from most to least promising (at most 10). Items proposing candidates must "
        "have status \"hypothesis\"; background facts may be \"established\"."
    ),
    HYPOTHESIS_ASSESSMENT: (
        "This is a HYPOTHESIS ASSESSMENT query: judge whether the given problem-method pairing is "
        "plausible. Check each transfer premise against the cited evidence on the problem side and the "
        "method side, state which premises are supported or unverified, and give an overall verdict. The "
        "verdict item has status \"hypothesis\"."
    ),
}


def compose_answer_request(
    query: str,
    context: str,
    kind: str = EVIDENCE,
    sources: Sequence[Dict[str, Any]] = (),
    max_tokens: int = 1536,
    component: Optional[str] = "composition",
) -> LLMRequest:
    """Build the single composition call.

    ``sources`` lists the citable sources present in ``context``
    (``{"paper_id", "title", "status", "candidate"}``); it is rendered in the
    context already and is passed in the payload for the offline mock.
    """
    instructions = _KIND_INSTRUCTIONS.get(kind, _KIND_INSTRUCTIONS[EVIDENCE])
    ctx = context.strip() or "(no retrieved context: answer from your own knowledge and cite nothing)"
    user = (
        f"Question: {query}\n\n{instructions}\n\nContext:\n{ctx}\n\n"
        "Answer JSON:\n"
        '{"answer": "<well-organized answer in markdown with [paper_id] citations>",\n'
        ' "items": [{"text": "<one claim or proposal>", "status": "established" | "hypothesis",\n'
        '            "citations": ["<paper_id>", ...], "candidate": "<candidate concept or null>",\n'
        '            "rationale": "<why (hypotheses only)>"}]}'
    )
    payload = {"query": query, "kind": kind, "sources": list(sources), "has_context": bool(context.strip())}
    return LLMRequest(
        task="answer.compose",
        messages=chat(SYSTEM, user),
        payload=payload,
        component=component,
        max_tokens=max_tokens,
        json_mode=True,
    )


_CITE_RE = re.compile(r"\[([^\[\]]+?)\]")


def parse_answer(obj: Any, valid_ids: Optional[Sequence[str]] = None) -> Tuple[str, List[AnswerItem]]:
    """Parse the composer's JSON into (answer text, items).

    Citations not in ``valid_ids`` (when given) are dropped, so that systems
    are only credited for papers actually present in their context.
    """
    valid = set(valid_ids) if valid_ids is not None else None
    if not isinstance(obj, dict):
        return str(obj), []
    text = str(obj.get("answer", "") or "")
    items: List[AnswerItem] = []
    for it in obj.get("items", []) or []:
        if not isinstance(it, dict):
            continue
        cites = it.get("citations") or []
        if isinstance(cites, str):
            cites = [c.strip() for c in cites.split(",")]
        cites = [str(c).strip().strip("[]") for c in cites if str(c).strip()]
        if valid is not None:
            cites = [c for c in cites if c in valid]
        status = str(it.get("status", "established")).lower()
        items.append(
            AnswerItem(
                text=str(it.get("text", "")),
                status="hypothesis" if status.startswith("hyp") else "established",
                citations=list(dict.fromkeys(cites)),
                candidate=(str(it["candidate"]) if it.get("candidate") not in (None, "", "null") else None),
                rationale=(str(it["rationale"]) if it.get("rationale") else None),
            )
        )
    if not items and text:
        cites = [c.strip() for m in _CITE_RE.findall(text) for c in m.split(",")]
        if valid is not None:
            cites = [c for c in cites if c in valid]
        items.append(AnswerItem(text=text, citations=list(dict.fromkeys(cites))))
    return text, items


# ---------------------------------------------------------------------------
# Offline mock
# ---------------------------------------------------------------------------


def _mock_compose(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    kind = payload.get("kind", EVIDENCE)
    sources = payload.get("sources", []) or []
    items: List[Dict[str, Any]] = []
    if kind == EVIDENCE:
        for s in sources[:12]:
            items.append({"text": f"{s.get('title', s['paper_id'])} reports relevant findings.",
                          "status": "established", "citations": [s["paper_id"]]})
    else:
        seen: Dict[str, List[str]] = {}
        for s in sources:
            cand = s.get("candidate")
            if cand:
                seen.setdefault(cand, []).append(s["paper_id"])
        for cand, cites in list(seen.items())[:10]:
            items.append({"text": f"{cand} could be applied to this problem.", "status": "hypothesis",
                          "citations": cites[:4], "candidate": cand,
                          "rationale": "Its mechanism matches properties of the target problem."})
        if not items:
            for s in sources[:5]:
                items.append({"text": f"See {s['paper_id']}.", "status": "established",
                              "citations": [s["paper_id"]]})
    answer = "\n".join(f"- {it['text']} " + " ".join(f"[{c}]" for c in it["citations"]) for it in items)
    return {"answer": answer or "No evidence found in the context.", "items": items}


MOCK_HANDLERS = {"answer.compose": _mock_compose}
