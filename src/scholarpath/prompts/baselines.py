"""Prompts of the Section 5.3 baselines and the uniform agent task prompt.

Tasks (``LLMRequest.task``) and their cost components:

====================================  ======================  ===============================================
task                                  component               used by
====================================  ======================  ===============================================
``baselines.listwise_rerank``         ``rerank``              Dense + reranker (RankGPT sliding window [71])
``baselines.raptor_summarize``        ``index_build``         RAPTOR tree construction [61]
``baselines.lightrag_extract``        ``index_build``         LightRAG entity / relation extraction [17]
``baselines.lightrag_glean``          ``index_build``         LightRAG gleaning pass ("MANY entities were missed")
``baselines.lightrag_glean_check``    ``index_build``         LightRAG "continue gleaning?" check
``baselines.lightrag_summarize``      ``index_build``         LightRAG merged-description summary
``baselines.lightrag_keywords``       ``query_understanding`` LightRAG query keyword extraction (1 call / query)
``baselines.agent_task``              ``agent``               uniform JSON-schema task prompt (App. C.3)
``baselines.agent_plan``              ``agent``               offline SimulatedAgent planning step (not a baseline)
``baselines.agent_reflect``           ``agent``               offline SimulatedAgent reflection step (not a baseline)
====================================  ======================  ===============================================

The RankGPT, RAPTOR and LightRAG prompts follow the wording of the original
implementations; LightRAG's tuple-delimited record format is replaced by an
equivalent JSON object so that it can be parsed with
:func:`scholarpath.llm.extract_json` (see the module docstring of
:mod:`scholarpath.baselines.lightrag`).

Every external agent of Table 3/4 (PaperQA2, OpenScholar, AgenticScholar,
Codex) receives the *identical* task text rendered by
:func:`render_agent_task`: a JSON-schema prompt requiring ranked candidate
records with cited paper IDs and an explicit status tag (``established`` vs.
``hypothesis``), as described in Appendix C.3.

``MOCK_HANDLERS`` answers every task deterministically from its payload so
that all baselines run offline with :class:`scholarpath.llm.MockLLM`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scholarpath.llm.base import LLMRequest, chat
from scholarpath.text import keywords, tokenize, truncate_tokens
from scholarpath.types import EVIDENCE, HYPOTHESIS_ASSESSMENT, HYPOTHESIS_DISCOVERY

# ---------------------------------------------------------------------------
# 1. Listwise reranking (RankGPT, Sun et al. 2023; "Dense + reranker" [71])
# ---------------------------------------------------------------------------

RERANK_SYSTEM = (
    "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
)


def _clean_passage(text: str, max_tokens: int) -> str:
    return truncate_tokens(" ".join((text or "").split()), max_tokens)


def listwise_rerank_request(
    query: str,
    passages: Sequence[str],
    max_passage_tokens: int = 300,
    component: Optional[str] = "rerank",
) -> LLMRequest:
    """RankGPT permutation-generation prompt for one window of passages.

    The message layout reproduces RankGPT's multi-turn format: an instruction,
    one user turn per passage ``[i] text`` acknowledged by the assistant, and a
    final instruction asking for ``[] > []`` output.
    """
    texts = [_clean_passage(p, max_passage_tokens) for p in passages]
    n = len(texts)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": RERANK_SYSTEM},
        {
            "role": "user",
            "content": f"I will provide you with {n} passages, each indicated by number identifier []. \n"
            f"Rank the passages based on their relevance to query: {query}.",
        },
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]
    for i, t in enumerate(texts, 1):
        messages.append({"role": "user", "content": f"[{i}] {t}"})
        messages.append({"role": "assistant", "content": f"Received passage [{i}]."})
    messages.append(
        {
            "role": "user",
            "content": f"Search Query: {query}. \nRank the {n} passages above based on their relevance to the "
            "search query. The passages should be listed in descending order using identifiers. The most "
            "relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. "
            "Only response the ranking results, do not say any word or explain.",
        }
    )
    return LLMRequest(
        task="baselines.listwise_rerank",
        messages=messages,
        payload={"query": query, "passages": texts},
        component=component,
        max_tokens=8 * n + 16,
        temperature=0.0,
    )


def parse_permutation(text: str, n: int) -> List[int]:
    """Parse a RankGPT answer (``"[3] > [1] > [2]"``) into a 0-based permutation.

    Mirrors RankGPT's ``clean_response`` / ``remove_duplicate``: every digit run
    is an identifier, duplicates and out-of-range ids are dropped, and the
    passages the model omitted are appended in their original order.
    """
    nums = [int(x) - 1 for x in re.findall(r"\d+", text or "")]
    seen: Dict[int, None] = {}
    for i in nums:
        if 0 <= i < n:
            seen.setdefault(i, None)
    order = list(seen)
    order += [i for i in range(n) if i not in seen]
    return order


# ---------------------------------------------------------------------------
# 2. RAPTOR cluster summarization [61]
# ---------------------------------------------------------------------------

RAPTOR_SYSTEM = "You are a Summarizing Text Portal"


def raptor_cluster_text(texts: Sequence[str], max_input_tokens: int = 3500) -> str:
    """The (token-capped) cluster text a RAPTOR summary is written from."""
    return truncate_tokens("\n\n".join(t.strip() for t in texts if t and t.strip()), max_input_tokens)


def raptor_cluster_coverage(texts: Sequence[str], max_input_tokens: int = 3500) -> List[bool]:
    """Per text: does (at least the start of) it reach the summarization prompt?

    The cluster text is a prefix-truncation of the joined texts, so a text is
    covered iff it starts inside the kept prefix. Empty texts are never covered.
    """
    kept = len(raptor_cluster_text(texts, max_input_tokens))
    out: List[bool] = []
    offset = 0
    first = True
    for t in texts:
        t = (t or "").strip()
        if not t:
            out.append(False)
            continue
        if not first:
            offset += 2  # "\n\n" separator
        out.append(offset < kept)
        offset += len(t)
        first = False
    return out


def raptor_summarize_request(
    texts: Sequence[str],
    max_tokens: int = 100,
    max_input_tokens: int = 3500,
    component: Optional[str] = "index_build",
) -> LLMRequest:
    """RAPTOR's summarization prompt for one cluster of tree nodes.

    ``max_input_tokens`` caps the concatenated cluster text (RAPTOR bounds a
    cluster by ``max_length_in_cluster`` = 3,500 tokens before summarizing);
    :func:`raptor_cluster_coverage` tells which texts reach the prompt.
    """
    context = raptor_cluster_text(texts, max_input_tokens)
    user = f"Write a summary of the following, including as many key details as possible: {context}:"
    return LLMRequest(
        task="baselines.raptor_summarize",
        messages=chat(RAPTOR_SYSTEM, user),
        payload={"texts": list(texts), "max_tokens": int(max_tokens)},
        component=component,
        max_tokens=max_tokens,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# 3. LightRAG [17]: graph extraction, description summaries, query keywords
# ---------------------------------------------------------------------------

DEFAULT_ENTITY_TYPES: Tuple[str, ...] = (
    "research problem",
    "method",
    "model",
    "dataset",
    "metric",
    "task",
    "concept",
)

LIGHTRAG_EXTRACT_SYSTEM = (
    "You are a knowledge graph specialist responsible for extracting entities and relationships from "
    "scientific text. Always answer with a single JSON object."
)

_LIGHTRAG_JSON_FORMAT = (
    '{"entities": [{"entity_name": "...", "entity_type": "...", "entity_description": "..."}],\n'
    ' "relationships": [{"source_entity": "...", "target_entity": "...", "relationship_description": "...",\n'
    '                    "relationship_keywords": "...", "relationship_strength": 1-10}],\n'
    ' "content_keywords": ["..."]}'
)


def _lightrag_extract_user(text: str, entity_types: Sequence[str]) -> str:
    types = ", ".join(entity_types)
    return (
        "---Goal---\n"
        "Given a text document that is potentially relevant to this activity and a list of entity types, "
        "identify all entities of those types from the text and all relationships among the identified "
        "entities.\n\n"
        "---Steps---\n"
        "1. Identify all entities. For each identified entity, extract the following information:\n"
        "- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.\n"
        f"- entity_type: One of the following types: [{types}]\n"
        "- entity_description: Comprehensive description of the entity's attributes and activities\n"
        "2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that "
        "are *clearly related* to each other. For each pair of related entities, extract the following "
        "information:\n"
        "- source_entity: name of the source entity, as identified in step 1\n"
        "- target_entity: name of the target entity, as identified in step 1\n"
        "- relationship_description: explanation as to why you think the source entity and the target entity "
        "are related to each other\n"
        "- relationship_strength: a numeric score indicating strength of the relationship between the source "
        "entity and target entity\n"
        "- relationship_keywords: one or more high-level key words that summarize the overarching nature of "
        "the relationship, focusing on concepts or themes rather than specific details\n"
        "3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire "
        "text (content_keywords).\n"
        f"4. Return the output as one JSON object:\n{_LIGHTRAG_JSON_FORMAT}\n\n"
        "---Real Data---\n"
        f"Entity_types: [{types}]\nText:\n{text}\n\nOutput:"
    )


def lightrag_extract_request(
    text: str,
    entity_types: Sequence[str] = DEFAULT_ENTITY_TYPES,
    max_input_tokens: int = 1200,
    component: Optional[str] = "index_build",
) -> LLMRequest:
    """LightRAG entity / relationship extraction for one text unit."""
    text = truncate_tokens(text, max_input_tokens)
    return LLMRequest(
        task="baselines.lightrag_extract",
        messages=chat(LIGHTRAG_EXTRACT_SYSTEM, _lightrag_extract_user(text, entity_types)),
        payload={"text": text, "entity_types": list(entity_types)},
        component=component,
        json_mode=True,
        temperature=0.0,
    )


def lightrag_glean_request(
    text: str,
    previous_output: str,
    entity_types: Sequence[str] = DEFAULT_ENTITY_TYPES,
    max_input_tokens: int = 1200,
    component: Optional[str] = "index_build",
) -> LLMRequest:
    """LightRAG gleaning pass: the extraction dialogue continued with
    "MANY entities and relationships were missed in the last extraction"."""
    text = truncate_tokens(text, max_input_tokens)
    messages = chat(LIGHTRAG_EXTRACT_SYSTEM, _lightrag_extract_user(text, entity_types))
    messages.append({"role": "assistant", "content": previous_output})
    messages.append(
        {
            "role": "user",
            "content": "MANY entities and relationships were missed in the last extraction. Add them below "
            f"using the same JSON format (only the NEW ones):\n{_LIGHTRAG_JSON_FORMAT}",
        }
    )
    return LLMRequest(
        task="baselines.lightrag_glean",
        messages=messages,
        payload={"text": text, "entity_types": list(entity_types), "previous": previous_output},
        component=component,
        json_mode=True,
        temperature=0.0,
    )


def lightrag_glean_check_request(
    text: str,
    history: str,
    entity_types: Sequence[str] = DEFAULT_ENTITY_TYPES,
    max_input_tokens: int = 1200,
    component: Optional[str] = "index_build",
) -> LLMRequest:
    """LightRAG's loop check between gleaning rounds (answer YES or NO)."""
    text = truncate_tokens(text, max_input_tokens)
    messages = chat(LIGHTRAG_EXTRACT_SYSTEM, _lightrag_extract_user(text, entity_types))
    messages.append({"role": "assistant", "content": history})
    messages.append(
        {
            "role": "user",
            "content": "It appears some entities and relationships may have still been missed. Answer YES | NO "
            "if there are still entities or relationships that need to be added.",
        }
    )
    return LLMRequest(
        task="baselines.lightrag_glean_check",
        messages=messages,
        payload={"text": text, "history": history},
        component=component,
        max_tokens=4,
        temperature=0.0,
    )


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _strip_name(value: Any) -> str:
    return str(value or "").strip().strip('"').strip("'").strip()


def parse_extraction(obj: Any) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], List[str]]:
    """Normalize an extraction answer into (entities, relations, content keywords).

    Entities: ``{"name", "type", "description"}``; relations:
    ``{"source", "target", "description", "keywords", "weight"}``. Records with
    missing names are dropped; numeric strengths default to 1.0 (LightRAG's
    default edge weight).
    """
    if not isinstance(obj, dict):
        return [], [], []
    entities: List[Dict[str, str]] = []
    for e in _as_list(obj.get("entities")):
        if not isinstance(e, dict):
            continue
        name = _strip_name(e.get("entity_name") or e.get("name"))
        if not name:
            continue
        entities.append(
            {
                "name": name,
                "type": _strip_name(e.get("entity_type") or e.get("type") or "unknown").lower(),
                "description": str(e.get("entity_description") or e.get("description") or "").strip(),
            }
        )
    relations: List[Dict[str, Any]] = []
    for r in _as_list(obj.get("relationships") or obj.get("relations")):
        if not isinstance(r, dict):
            continue
        src = _strip_name(r.get("source_entity") or r.get("source") or r.get("src"))
        tgt = _strip_name(r.get("target_entity") or r.get("target") or r.get("tgt"))
        if not src or not tgt or src.upper() == tgt.upper():
            continue
        kw = r.get("relationship_keywords") or r.get("keywords") or ""
        if isinstance(kw, list):
            kw = ", ".join(str(k) for k in kw)
        try:
            weight = float(r.get("relationship_strength", r.get("weight", 1.0)))
        except (TypeError, ValueError):
            weight = 1.0
        relations.append(
            {
                "source": src,
                "target": tgt,
                "description": str(r.get("relationship_description") or r.get("description") or "").strip(),
                "keywords": str(kw).strip(),
                "weight": weight,
            }
        )
    ck = obj.get("content_keywords") or []
    if isinstance(ck, str):
        ck = [c.strip() for c in ck.split(",")]
    return entities, relations, [str(c).strip() for c in ck if str(c).strip()]


def lightrag_summarize_request(
    entity_name: str,
    descriptions: Sequence[str],
    max_tokens: int = 500,
    component: Optional[str] = "index_build",
) -> LLMRequest:
    """LightRAG's ``summarize_entity_descriptions`` prompt for merged fragments."""
    desc_list = "\n".join(f"- {d}" for d in descriptions)
    user = (
        "You are a helpful assistant responsible for generating a comprehensive summary of the data provided "
        "below. Given one or two entities, and a list of descriptions, all related to the same entity or group "
        "of entities. Please concatenate all of these into a single, comprehensive description. Make sure to "
        "include information collected from all the descriptions. If the provided descriptions are "
        "contradictory, please resolve the contradictions and provide a single, coherent summary. Make sure it "
        "is written in third person, and include the entity names so we have the full context.\n\n"
        f"#######\n---Data---\nEntities: {entity_name}\nDescription List:\n{desc_list}\n#######\nOutput:"
    )
    return LLMRequest(
        task="baselines.lightrag_summarize",
        messages=[{"role": "user", "content": user}],
        payload={"name": entity_name, "descriptions": list(descriptions), "max_tokens": int(max_tokens)},
        component=component,
        max_tokens=max_tokens,
        temperature=0.0,
    )


def lightrag_keywords_request(query: str, component: Optional[str] = "query_understanding") -> LLMRequest:
    """LightRAG's query keyword extraction (high-level / low-level keywords)."""
    user = (
        "---Role---\n"
        "You are a helpful assistant tasked with identifying both high-level and low-level keywords in the "
        "user's query.\n\n"
        "---Goal---\n"
        "Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching "
        "concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.\n\n"
        "---Instructions---\n"
        "- Output the keywords in JSON format.\n"
        '- The JSON should have two keys: "high_level_keywords" for overarching concepts or themes, and '
        '"low_level_keywords" for specific entities or details.\n\n'
        f"---Real Data---\nQuery: {query}\n\nOutput:"
    )
    return LLMRequest(
        task="baselines.lightrag_keywords",
        messages=[{"role": "user", "content": user}],
        payload={"query": query},
        component=component,
        json_mode=True,
        max_tokens=256,
        temperature=0.0,
    )


def parse_keywords(obj: Any) -> Tuple[List[str], List[str]]:
    """(high-level keywords, low-level keywords) from the keyword answer."""
    if not isinstance(obj, dict):
        return [], []

    def norm(v: Any) -> List[str]:
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",")]
        return [str(x).strip() for x in (v or []) if str(x).strip()]

    return norm(obj.get("high_level_keywords")), norm(obj.get("low_level_keywords"))


# ---------------------------------------------------------------------------
# 4. Uniform agent task prompt (Appendix C.3) and SimulatedAgent steps
# ---------------------------------------------------------------------------

#: JSON schema every agent system must answer with (Appendix C.3).
AGENT_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["answer", "records"],
    "properties": {
        "answer": {
            "type": "string",
            "description": "Well-organized answer in markdown that cites papers as [paper_id].",
        },
        "records": {
            "type": "array",
            "description": "Ranked candidate records, most relevant / most promising first.",
            "items": {
                "type": "object",
                "required": ["rank", "claim", "status", "paper_ids"],
                "properties": {
                    "rank": {"type": "integer", "minimum": 1},
                    "claim": {"type": "string", "description": "One finding or one proposed hypothesis."},
                    "candidate": {
                        "type": ["string", "null"],
                        "description": "Proposed counterpart concept (method or problem) for hypotheses.",
                    },
                    "status": {"type": "string", "enum": ["established", "hypothesis"]},
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of corpus papers supporting the record.",
                    },
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}

AGENT_SYSTEM = (
    "You are an automated research agent working over a local corpus of scientific papers. You may only use "
    "papers from this corpus as evidence and must cite them by their paper IDs."
)

_AGENT_KIND = {
    EVIDENCE: (
        "This is an EVIDENCE query: synthesize what the literature in the corpus has established. Each record "
        "is one established finding, with status \"established\" and the IDs of the papers that report it."
    ),
    HYPOTHESIS_DISCOVERY: (
        "This is a HYPOTHESIS query: propose promising but UNSTUDIED research directions (e.g. candidate methods "
        "for the problem, or candidate problems for the method) that no corpus paper has realized yet. Each "
        "record proposes one candidate (field \"candidate\"), has status \"hypothesis\", explains in "
        "\"rationale\" why the indirect evidence supports it, and cites the corpus papers on both the problem "
        "side and the method side. Rank candidates from most to least promising."
    ),
    HYPOTHESIS_ASSESSMENT: (
        "This is a HYPOTHESIS ASSESSMENT query: judge whether the given problem-method pairing is plausible. "
        "Records state which transfer premises are supported (status \"established\", with citations) and the "
        "overall verdict (status \"hypothesis\")."
    ),
}


def render_agent_task(query: str, kind: str = EVIDENCE, max_records: int = 10) -> str:
    """The identical task text given to every agent system (Appendix C.3)."""
    instructions = _AGENT_KIND.get(kind, _AGENT_KIND[EVIDENCE])
    return (
        f"Research question: {query}\n\n{instructions}\n\n"
        f"Return at most {int(max_records)} records, ranked. Cite ONLY paper IDs that exist in the local corpus "
        "(never invent IDs), and tag every record with an explicit status: \"established\" (directly reported "
        "by the cited papers) or \"hypothesis\" (not yet studied; supported only by indirect evidence).\n\n"
        "Your final answer must be a single JSON object that validates against this JSON schema:\n"
        f"{json.dumps(AGENT_OUTPUT_SCHEMA, indent=1)}"
    )


def _evidence_lines(evidence: Sequence[Dict[str, Any]], max_tokens_each: int = 160) -> str:
    lines = []
    for e in evidence:
        snippet = truncate_tokens(" ".join(str(e.get("text", "")).split()), max_tokens_each)
        lines.append(f"[{e['paper_id']}] {e.get('title', '')}\n{snippet}")
    return "\n\n".join(lines)


def agent_task_request(
    query: str,
    kind: str = EVIDENCE,
    evidence: Optional[Sequence[Dict[str, Any]]] = None,
    max_records: int = 10,
    max_tokens: int = 2048,
    component: Optional[str] = "agent",
) -> LLMRequest:
    """The uniform task prompt as an LLM request, optionally followed by the
    evidence an in-process agent collected with its tools
    (``{"paper_id", "title", "text"}`` records)."""
    messages = chat(AGENT_SYSTEM, render_agent_task(query, kind, max_records))
    ev = list(evidence or [])
    if ev:
        messages.append({"role": "user", "content": "Search results collected so far:\n\n" + _evidence_lines(ev)})
    return LLMRequest(
        task="baselines.agent_task",
        messages=messages,
        payload={
            "query": query,
            "kind": kind,
            "max_records": int(max_records),
            "evidence": [{"paper_id": e["paper_id"], "title": e.get("title", "")} for e in ev],
        },
        component=component,
        max_tokens=max_tokens,
        json_mode=True,
    )


def agent_plan_request(query: str, max_queries: int = 3, component: Optional[str] = "agent") -> LLMRequest:
    """SimulatedAgent step 1: plan corpus search queries."""
    user = (
        f"Research question: {query}\n\nYou can search the local corpus with keyword (BM25) and semantic (dense) "
        f"search tools. Plan up to {int(max_queries)} search queries that together cover the question.\n"
        'Answer JSON: {"queries": ["..."]}'
    )
    return LLMRequest(
        task="baselines.agent_plan",
        messages=chat(AGENT_SYSTEM, user),
        payload={"query": query, "max_queries": int(max_queries)},
        component=component,
        json_mode=True,
        max_tokens=256,
    )


def agent_reflect_request(
    query: str,
    step: int,
    max_steps: int,
    found: Sequence[Dict[str, Any]],
    component: Optional[str] = "agent",
) -> LLMRequest:
    """SimulatedAgent step 2..k: decide whether to stop or issue more searches."""
    user = (
        f"Research question: {query}\n\nAfter search step {step + 1} of at most {max_steps}, you collected "
        f"{len(found)} passages:\n\n{_evidence_lines(found[:20], 60)}\n\n"
        "Decide whether the evidence suffices. If not, propose follow-up search queries.\n"
        'Answer JSON: {"done": true | false, "next_queries": ["..."]}'
    )
    return LLMRequest(
        task="baselines.agent_reflect",
        messages=chat(AGENT_SYSTEM, user),
        payload={
            "query": query,
            "step": int(step),
            "max_steps": int(max_steps),
            "found": [{"paper_id": f["paper_id"], "title": f.get("title", "")} for f in found],
        },
        component=component,
        json_mode=True,
        max_tokens=256,
    )


# ---------------------------------------------------------------------------
# Offline mock handlers (deterministic lexical heuristics)
# ---------------------------------------------------------------------------


def _mock_rerank(payload: Dict[str, Any], request: LLMRequest) -> str:
    q = set(tokenize(payload.get("query", ""), remove_stopwords=True))
    scored = []
    for i, text in enumerate(payload.get("passages", [])):
        toks = tokenize(text, remove_stopwords=True)
        overlap = len(q & set(toks))
        tf = sum(1 for t in toks if t in q)
        scored.append((-overlap, -tf, i))
    scored.sort()
    return " > ".join(f"[{i + 1}]" for _, _, i in scored)


def _first_sentence(text: str) -> str:
    text = " ".join((text or "").split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return m.group(1) if m else text


def _mock_summarize(payload: Dict[str, Any], request: LLMRequest) -> str:
    texts = [t for t in payload.get("texts", []) if t]
    kws = keywords(" ".join(texts), k=6)
    lead = " ".join(_first_sentence(t) for t in texts[:3])
    out = f"Summary of {len(texts)} passages on {', '.join(kws)}. {lead}"
    return truncate_tokens(out, int(payload.get("max_tokens", 100)))


def _mock_extract(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    text = payload.get("text", "")
    types = payload.get("entity_types") or list(DEFAULT_ENTITY_TYPES)
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    kws = keywords(text, k=6)
    entities = []
    for j, kw in enumerate(kws):
        sent = next((s for s in sentences if kw in s.lower()), sentences[0] if sentences else text)
        entities.append(
            {"entity_name": kw.title(), "entity_type": types[j % len(types)], "entity_description": sent}
        )
    relations = []
    for a, b in zip(kws, kws[1:]):
        relations.append(
            {
                "source_entity": a.title(),
                "target_entity": b.title(),
                "relationship_description": f"{a.title()} is discussed together with {b.title()}.",
                "relationship_keywords": f"{a} {b}",
                "relationship_strength": 5,
            }
        )
    return {"entities": entities, "relationships": relations, "content_keywords": kws[:3]}


def _mock_glean(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    return {"entities": [], "relationships": [], "content_keywords": []}


def _mock_glean_check(payload: Dict[str, Any], request: LLMRequest) -> str:
    return "NO"


def _mock_lightrag_summarize(payload: Dict[str, Any], request: LLMRequest) -> str:
    descs = list(dict.fromkeys(payload.get("descriptions", [])))
    return truncate_tokens(f"{payload.get('name', '')}: " + " ".join(descs), int(payload.get("max_tokens", 500)))


def _mock_keywords(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    kws = keywords(payload.get("query", ""), k=6)
    high = [" ".join(kws[i : i + 2]) for i in range(0, min(len(kws), 4), 2)]
    return {"high_level_keywords": high, "low_level_keywords": kws}


def _mock_agent_task(payload: Dict[str, Any], request: LLMRequest) -> str:
    kind = payload.get("kind", EVIDENCE)
    by_paper: Dict[str, Dict[str, Any]] = {}
    for e in payload.get("evidence", []) or []:
        by_paper.setdefault(e["paper_id"], e)
    evidence = list(by_paper.values())
    k = int(payload.get("max_records", 10))
    records = []
    for rank, e in enumerate(evidence[:k], 1):
        if kind == EVIDENCE:
            records.append(
                {"rank": rank, "claim": f"{e.get('title', e['paper_id'])} reports relevant findings.",
                 "candidate": None, "status": "established", "paper_ids": [e["paper_id"]]}
            )
        else:
            cand = " ".join(keywords(e.get("title", ""), k=3)).title() or e["paper_id"]
            records.append(
                {"rank": rank, "claim": f"{cand} could be applied to this problem.", "candidate": cand,
                 "status": "hypothesis", "paper_ids": [e["paper_id"]],
                 "rationale": "Its mechanism matches properties of the target problem."}
            )
    answer = "\n".join(f"- {r['claim']} [{r['paper_ids'][0]}]" for r in records)
    obj = {"answer": answer or "No supporting evidence was found in the corpus.", "records": records}
    # Agents often wrap their JSON in prose and code fences; so does the mock.
    return "Here is the final answer.\n```json\n" + json.dumps(obj, indent=1) + "\n```"


def _mock_agent_plan(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    q = payload.get("query", "")
    kws = keywords(q, k=6)
    queries = [q]
    if kws:
        queries.append(" ".join(kws[:4]))
    if len(kws) > 4:
        queries.append(" ".join(kws[2:]))
    return {"queries": queries[: int(payload.get("max_queries", 3))]}


def _mock_agent_reflect(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    step, max_steps = int(payload.get("step", 0)), int(payload.get("max_steps", 3))
    found = payload.get("found", []) or []
    done = step + 1 >= min(2, max_steps) or len(found) >= 30
    nxt: List[str] = []
    if not done:
        titles = " ".join(f.get("title", "") for f in found[:5])
        nxt = [" ".join(keywords(payload.get("query", "") + " " + titles, k=5))]
    return {"done": bool(done), "next_queries": nxt}


MOCK_HANDLERS = {
    "baselines.listwise_rerank": _mock_rerank,
    "baselines.raptor_summarize": _mock_summarize,
    "baselines.lightrag_extract": _mock_extract,
    "baselines.lightrag_glean": _mock_glean,
    "baselines.lightrag_glean_check": _mock_glean_check,
    "baselines.lightrag_summarize": _mock_lightrag_summarize,
    "baselines.lightrag_keywords": _mock_keywords,
    "baselines.agent_task": _mock_agent_task,
    "baselines.agent_plan": _mock_agent_plan,
    "baselines.agent_reflect": _mock_agent_reflect,
}
