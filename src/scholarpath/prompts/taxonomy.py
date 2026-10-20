"""Prompts for LLM-based taxonomy construction (Section 3).

Three tasks are used by the node-wise builder and by every baseline:

* ``taxonomy.propose_children`` -- read a batch of papers of node v, propose
  (or refine) the children of v with concise definitions (the *schema* of v),
  and label the papers that were read;
* ``taxonomy.label_papers`` -- given the schema of v, label papers with the
  matching children (multi-label) or flag a *missing concept*; optionally
  suggest new children (schema update);
* ``taxonomy.discover`` -- DISCOVER (Section 3.3): read papers from the
  residual pool, then assign each to an existing child, refine a child
  definition, or add a new child.

All errors are judged "at the granularity of v": a paper is correctly
organized at v if it is assigned to every child of v it belongs to.

Children are referred to inside prompts by short keys (``C1``, ``C2``, ...;
new children ``N1``, ``N2``, ...), which :mod:`scholarpath.taxonomy.llm_ops`
maps back to taxonomy node ids.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scholarpath.llm.base import LLMRequest, chat
from scholarpath.text import STOPWORDS, tokenize

FACET_GUIDE = {
    "problem": (
        "the RESEARCH PROBLEMS the papers address: tasks, application domains, phenomena or "
        "questions under study (WHAT is studied), independent of the technique used"
    ),
    "method": (
        "the METHODS the papers use or propose: techniques, models, algorithms, theoretical or "
        "experimental tools (HOW it is studied), independent of the application"
    ),
    "topic": (
        "the research TOPICS AND APPROACHES of the papers, the way a survey's taxonomy organizes "
        "its cited works"
    ),
}

SYSTEM = (
    "You are an expert research librarian who organizes scientific papers into precise, "
    "non-overlapping concept taxonomies. Always answer with a single JSON object and nothing else."
)


def _paper_lines(papers: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for p in papers:
        abstract = (p.get("abstract") or "").replace("\n", " ")
        lines.append(f"[{p['id']}] {p.get('title', '')}\n    {abstract}")
    return "\n".join(lines)


def _children_lines(children: Sequence[Dict[str, Any]]) -> str:
    if not children:
        return "(none yet)"
    return "\n".join(f"{c['key']}: {c['name']} -- {c.get('definition', '')}" for c in children)


def _node_header(facet: str, node: Dict[str, Any]) -> str:
    path = " > ".join(node.get("path", [])) or node.get("name", "Root")
    definition = node.get("definition") or "(root of the repository)"
    return (
        f"We are organizing papers by {FACET_GUIDE.get(facet, facet)}.\n"
        f"Current concept: {node.get('name')}\nPath from root: {path}\nDefinition: {definition}\n"
    )


def propose_children_request(
    facet: str,
    node: Dict[str, Any],
    papers: Sequence[Dict[str, Any]],
    existing: Sequence[Dict[str, Any]] = (),
    min_children: int = 2,
    max_children: int = 8,
    component: Optional[str] = None,
) -> LLMRequest:
    payload = {
        "facet": facet,
        "node": node,
        "papers": list(papers),
        "existing": list(existing),
        "min_children": min_children,
        "max_children": max_children,
    }
    existing_txt = _children_lines(existing)
    user = (
        _node_header(facet, node)
        + f"\nExisting children of this concept:\n{existing_txt}\n\n"
        f"Papers of this concept (id, title, abstract):\n{_paper_lines(papers)}\n\n"
        f"Task: propose the sub-concepts (children) of '{node.get('name')}' that organize these papers "
        f"by {facet}. Keep useful existing children (refer to them by their key and you may sharpen their "
        f"definition), and add new ones (keys N1, N2, ...) only for sub-topics the existing children miss. "
        f"Produce between {min_children} and {max_children} children in total, each one level more specific "
        f"than the current concept, mutually distinguishable, and each with a one-sentence definition that "
        f"states what belongs to it. Then label every paper above with ALL children it belongs to "
        f"(multi-label allowed); use an empty list only if no child fits.\n\n"
        "Answer JSON:\n"
        '{"children": [{"key": "C1 or N1", "name": "...", "definition": "..."}],\n'
        ' "labels": [{"paper_id": "...", "children": ["C1", "N2"]}]}'
    )
    return LLMRequest(
        task="taxonomy.propose_children",
        messages=chat(SYSTEM, user),
        payload=payload,
        component=component,
        json_mode=True,
    )


def label_papers_request(
    facet: str,
    node: Dict[str, Any],
    children: Sequence[Dict[str, Any]],
    papers: Sequence[Dict[str, Any]],
    allow_new: bool = False,
    component: Optional[str] = None,
) -> LLMRequest:
    payload = {
        "facet": facet,
        "node": node,
        "children": list(children),
        "papers": list(papers),
        "allow_new": bool(allow_new),
    }
    new_clause = (
        "If a paper belongs to this concept but fits none of the children, set \"missing_concept\": true "
        "and propose the missing child in \"new_children\" (keys N1, N2, ...; reuse one new child for "
        "papers sharing the same missing sub-topic) and list its key in the paper's children.\n"
        if allow_new
        else "If a paper fits none of the children, return an empty list and set \"missing_concept\": true.\n"
    )
    user = (
        _node_header(facet, node)
        + f"\nChildren (sub-concepts) of this concept:\n{_children_lines(children)}\n\n"
        f"Papers to label:\n{_paper_lines(papers)}\n\n"
        "Task: for every paper, list ALL children it belongs to (multi-label allowed), judged at the "
        f"granularity of '{node.get('name')}'.\n" + new_clause + "\nAnswer JSON:\n"
        '{"labels": [{"paper_id": "...", "children": ["C1"], "missing_concept": false}],\n'
        ' "new_children": [{"key": "N1", "name": "...", "definition": "..."}]}'
    )
    return LLMRequest(
        task="taxonomy.label_papers",
        messages=chat(SYSTEM, user),
        payload=payload,
        component=component,
        json_mode=True,
    )


def discover_request(
    facet: str,
    node: Dict[str, Any],
    children: Sequence[Dict[str, Any]],
    papers: Sequence[Dict[str, Any]],
    component: Optional[str] = None,
) -> LLMRequest:
    payload = {"facet": facet, "node": node, "children": list(children), "papers": list(papers)}
    user = (
        _node_header(facet, node)
        + f"\nCurrent children of this concept:\n{_children_lines(children)}\n\n"
        "The papers below could not be placed into these children by an automatic router, or an audit "
        f"flagged them as needing a concept the schema lacks:\n{_paper_lines(papers)}\n\n"
        "Task (repair the schema): for each paper decide whether it (a) belongs to an existing child "
        "(possibly because that child's definition is too narrow -- then refine the definition), or (b) "
        "needs a NEW child that the schema is missing. Add new children (keys N1, N2, ...) only for genuine "
        "sub-topics at the same granularity as the existing children; group papers that share a missing "
        "sub-topic under one new child. Then label every paper with all children it belongs to.\n\n"
        "Answer JSON:\n"
        '{"new_children": [{"key": "N1", "name": "...", "definition": "..."}],\n'
        ' "refined": [{"key": "C2", "definition": "..."}],\n'
        ' "labels": [{"paper_id": "...", "children": ["C2", "N1"]}]}'
    )
    return LLMRequest(
        task="taxonomy.discover",
        messages=chat(SYSTEM, user),
        payload=payload,
        component=component,
        json_mode=True,
    )


# ---------------------------------------------------------------------------
# Offline mock handlers (lexical heuristics; see scholarpath.llm.mock)
# ---------------------------------------------------------------------------


def _content_tokens(text: str, banned: set) -> List[str]:
    return [t for t in tokenize(text, remove_stopwords=True) if t not in banned and len(t) > 2 and not t.isdigit()]


def _banned(node: Dict[str, Any]) -> set:
    banned = set(STOPWORDS)
    for name in node.get("path", []) + [node.get("name", "")]:
        banned.update(tokenize(name))
    banned.update({"paper", "papers", "propose", "proposed", "study", "studies", "results", "using", "based",
                   "approach", "method", "methods", "problem", "problems", "novel", "show", "task", "work",
                   "scalable", "efficient", "robust", "unified", "towards", "revisiting", "learning", "fast",
                   "improved", "improving", "effective", "simple", "general", "deep", "large"})
    return banned


_PROBLEM_CUES = ("study", "studies", "address", "problem", "task", "challenge", "concerns", "forecast", "predict")
_METHOD_CUES = ("propose", "model", "method", "algorithm", "approach", "based on", "framework", "technique", "use")


def _facet_text(p: Dict[str, Any], facet: str) -> Tuple[str, str]:
    """(title part, abstract part) most indicative of the facet.

    Heuristic: in "<Method> for <Problem>" titles the method precedes " for "
    and the problem follows it; abstract sentences are filtered by cue words.
    """
    title = p.get("title", "") or ""
    abstract = p.get("abstract", "") or ""
    if facet not in ("problem", "method"):
        return title, abstract
    low = title.lower()
    cut = low.find(" for ")
    if cut >= 0:
        before, after = title[:cut], title[cut + 5 :]
        after = re.split(r"\bwith\b|:", after, maxsplit=1)[0]
        title_part = after if facet == "problem" else before
    else:
        title_part = title
    cues = _PROBLEM_CUES if facet == "problem" else _METHOD_CUES
    sents = [s for s in re.split(r"(?<=[.!?])\s+", abstract) if any(c in s.lower() for c in cues)]
    if facet == "problem":
        sents = [s.split(" where ")[0] for s in sents]
    return title_part, " ".join(sents) or abstract


def _paper_tokens(p: Dict[str, Any], banned: set, facet: str = "topic") -> List[str]:
    # Titles carry the most signal; weight them twice.
    title_part, abstract_part = _facet_text(p, facet)
    title = _content_tokens(title_part, banned)
    return title + title + _content_tokens(abstract_part, banned)


def _child_terms(c: Dict[str, Any]) -> set:
    terms = set(c.get("terms") or [])
    if not terms:
        terms = set(tokenize(c.get("name", ""), remove_stopwords=True))
    return {t for t in terms if len(t) > 2}


def _match(paper_toks: List[str], children: Sequence[Dict[str, Any]]) -> List[str]:
    toks = set(paper_toks)
    scores = []
    for c in children:
        terms = _child_terms(c)
        s = len(terms & toks) / max(1, len(terms))
        scores.append((s, c["key"]))
    best = max((s for s, _ in scores), default=0.0)
    if best <= 0:
        return []
    return [k for s, k in scores if s >= 0.999 * best and s > 0]


def _cover_terms(
    papers: Sequence[Dict[str, Any]], banned: set, k: int, exclude: set, facet: str = "topic"
) -> List[str]:
    """Greedy max-coverage choice of k distinctive terms."""
    docs = [set(_paper_tokens(p, banned, facet)) - exclude for p in papers]
    df: Dict[str, int] = {}
    for d in docs:
        for t in d:
            df[t] = df.get(t, 0) + 1
    n = max(1, len(docs))
    cands = [t for t, c in df.items() if c >= 2 and c <= max(2, int(0.8 * n))]
    chosen: List[str] = []
    covered = set()
    for _ in range(k):
        best, best_gain = None, 0
        for t in sorted(cands):
            if t in chosen:
                continue
            gain = sum(1 for i, d in enumerate(docs) if t in d and i not in covered)
            if gain > best_gain:
                best, best_gain = t, gain
        if best is None:
            break
        chosen.append(best)
        covered |= {i for i, d in enumerate(docs) if best in d}
    return chosen


def _mock_propose(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    """Propose children by greedy term coverage; refinement batches add children
    only for sub-topics the existing children miss (papers they cannot match)."""
    node = payload.get("node", {})
    papers = payload.get("papers", [])
    existing = payload.get("existing", [])
    facet = payload.get("facet", "topic")
    lo, hi = int(payload.get("min_children", 2)), int(payload.get("max_children", 8))
    banned = _banned(node)
    exclude = set()
    for c in existing:
        exclude |= _child_terms(c)
    children = [dict(c) for c in existing]
    if existing:
        unmatched = [p for p in papers if not _match(_paper_tokens(p, banned, facet), children)]
        room = max(0, hi - len(children))
        terms = _cover_terms(unmatched, banned, room, exclude, facet) if (room and len(unmatched) >= 2) else []
    else:
        target = max(lo, min(hi, int(round(math.sqrt(max(1, len(papers)) / 1.5)))))
        terms = _cover_terms(papers, banned, target, exclude, facet)
    for i, t in enumerate(terms, 1):
        children.append({
            "key": f"N{i}",
            "name": t.title(),
            "definition": f"Papers whose {facet} centers on {t}.",
            "terms": [t],
        })
    # Fewer than min_children distinctive sub-topics: return what exists (the
    # builders treat a schema with < 2 usable children as a leaf).
    labels = [{"paper_id": p["id"], "children": _match(_paper_tokens(p, banned, facet), children)} for p in papers]
    out_children = [{"key": c["key"], "name": c["name"], "definition": c.get("definition", "")} for c in children]
    return {"children": out_children, "labels": labels}


def _mock_label(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    node = payload.get("node", {})
    banned = _banned(node)
    children = payload.get("children", [])
    facet = payload.get("facet", "topic")
    labels, new_children = [], []
    unmatched = []
    for p in payload.get("papers", []):
        toks = _paper_tokens(p, banned, facet)
        keys = _match(toks, children)
        labels.append({"paper_id": p["id"], "children": keys, "missing_concept": not keys})
        if not keys:
            unmatched.append((p, toks))
    if payload.get("allow_new") and len(unmatched) >= 2:
        exclude = set()
        for c in children:
            exclude |= _child_terms(c)
        terms = _cover_terms([p for p, _ in unmatched], banned, 2, exclude, facet)
        for i, t in enumerate(terms, 1):
            new_children.append({"key": f"N{i}", "name": t.title(),
                                 "definition": f"Papers centered on {t}.", "terms": [t]})
        for lab in labels:
            if lab["missing_concept"]:
                p = next(p for p, _ in unmatched if p["id"] == lab["paper_id"])
                keys = _match(_paper_tokens(p, banned, facet), new_children)
                lab["children"] = keys
    return {"labels": labels,
            "new_children": [{k: c[k] for k in ("key", "name", "definition")} for c in new_children]}


def _mock_discover(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    out = _mock_label(dict(payload, allow_new=True), request)
    return {"new_children": out["new_children"], "refined": [],
            "labels": [{"paper_id": l["paper_id"], "children": l["children"]} for l in out["labels"]]}


MOCK_HANDLERS = {
    "taxonomy.propose_children": _mock_propose,
    "taxonomy.label_papers": _mock_label,
    "taxonomy.discover": _mock_discover,
}
