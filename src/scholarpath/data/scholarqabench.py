"""ScholarQABench: ScholarQA-CS and ScholarQA-Multi (Sec. 5.3, App. C.2).

App. C.2: *"ScholarQABench combines ScholarQA-CS (100 computer-science
queries with PhD-authored rubric items and supporting quotes) and
ScholarQA-Multi (108 multi-paper synthesis queries with expert-written,
citation-backed reference answers)."* Coverage: 2,375 of 2,454
expert-cited papers aligned to arXiv-Subfields (96.8%).

**ScholarQA-CS** (``data/scholarqa_cs/test_configs_snippets.json``): 110
test cases over 100 unique questions (10 are double-annotated). The
official scorer (``scripts/rubric_eval.py``) scores only the first case
seen per question unless ``--agreement`` is passed; we keep that case and
store the other annotation under ``metadata["alternate_cases"]``. The full
``metric_config`` is kept so the official rubric evaluation can be run.
Rubric items are the ``other_properties`` criteria (weights sum to 0.6 per
case; ``kind`` = ``most_important`` / ``nice_to_have``) with their evidence
quotes in ``metadata["evidence"]``.

ScholarQA-CS has no explicit citation list; its *expert-cited papers* are
the sources of the rubric evidence quotes. We reconstruct them per quote
(:func:`classify_evidence`), using the system answers the annotators worked
from (``output_snippets.jsonl``) when available:

* a quote matching a snippet of a citation in the Ai2 ScholarQA ("Nora")
  answer -> that citation's Semantic Scholar ``corpus_id``;
* a quote found in the Nora answer text -> the citations marked next to it
  (``(k, Author et al., YYYY)``);
* a quote found in another system's answer (Perplexity, GPT, Claude) is
  model-generated text, not a paper, and is not a gold source;
* any other quote is a verbatim excerpt of a paper the expert read; it is
  attributed to the arXiv-Subfields paper whose text contains it
  (:class:`~scholarpath.data.alignment.QuoteAttributor`) or, failing that,
  kept as an unaligned gold entry ``quote:<hash>``.

**ScholarQA-Multi** (``data/scholarqa_multi/human_answers.json``): 108
questions with expert answers (``output``) and their citations (``ctxs``:
title, text, S2 url, year, authors). Gold source papers are the ``ctxs``,
aligned by Semantic Scholar id / title; **citation markers in the answers
are 0-based** (``[0]`` refers to ``ctxs[0]``), recorded in
``metadata["citation_marker_base"]``.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from scholarpath.data.alignment import (
    CoverageStats,
    PaperAligner,
    QuoteAttributor,
    Reference,
    quote_key,
    resolve_references,
)
from scholarpath.data.corpus import read_jsonl
from scholarpath.types import EVIDENCE, Query, RubricItem

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

CS_BENCHMARK = "scholarqa_cs"
MULTI_BENCHMARK = "scholarqa_multi"
CS_FILE = "data/scholarqa_cs/test_configs_snippets.json"
CS_SOURCES_FILE = "data/scholarqa_cs/output_snippets.jsonl"
MULTI_FILE = "data/scholarqa_multi/human_answers.json"
PAPER_STATS = {"cs_queries": 100, "multi_queries": 108, "expert_cited": 2454, "aligned": 2375, "coverage": 0.968}

NORA = "Nora"  # Ai2 ScholarQA system answers (structured, with S2 corpus ids)
_MARKER_RE = re.compile(r"\(\d+,\s[^()]*?\d{4}[a-z]?\)")
_WORDS_RE = re.compile(r"[a-z0-9]+")


def _resolve(root: PathLike, rel: str) -> Path:
    root = Path(root)
    if root.is_file():
        return root
    for cand in (root / rel, root / Path(rel).name, root / Path(rel).relative_to("data")):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{rel} not found under {root}; run `python scripts/download_data.py --only scholarqabench`")


def _norm(text: str) -> str:
    return " ".join(_WORDS_RE.findall(str(text or "").lower()))


# ---------------------------------------------------------------------------
# ScholarQA-CS
# ---------------------------------------------------------------------------


def read_cs_cases(path: PathLike) -> List[Dict[str, Any]]:
    """The 110 test cases of ``test_configs_snippets.json`` (or the repo root)."""
    data = json.loads(_resolve(path, CS_FILE).read_text(encoding="utf-8"))
    for c in data:
        if "initial_prompt" not in c or "metric_config" not in c:
            raise ValueError(f"Not a ScholarQA-CS test case: {str(c)[:200]}")
    return data


@dataclass
class SourceAnswers:
    """System answers shown to the annotators of one question (``output_snippets.jsonl``)."""

    texts: Dict[str, str] = field(default_factory=dict)  # system -> normalized answer text
    nora_sections: List[str] = field(default_factory=list)  # raw Nora section texts (with markers)
    nora_citations: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # marker id -> citation


def read_cs_sources(path: PathLike) -> Dict[str, SourceAnswers]:
    """Index ``output_snippets.jsonl`` by normalized question (rows of double-annotated questions merged)."""
    out: Dict[str, SourceAnswers] = {}
    for row in read_jsonl(_resolve(path, CS_SOURCES_FILE)):
        sa = out.setdefault(_norm(row.get("question", "")), SourceAnswers())
        for src in row.get("sources") or []:
            name, ans = str(src.get("name", "")), src.get("answer_txt")
            if isinstance(ans, str):
                sa.texts[name] = (sa.texts.get(name, "") + " " + _norm(ans)).strip()
                continue
            parts = []
            for sec in ans or []:
                text = str(sec.get("text", ""))
                parts.append(f"{sec.get('title', '')} {sec.get('tldr', '')} {text}")
                if name == NORA:
                    sa.nora_sections.append(text)
                    for cit in sec.get("citations") or []:
                        sa.nora_citations.setdefault(str(cit.get("id", "")), cit)
            sa.texts[name] = (sa.texts.get(name, "") + " " + _norm(" ".join(parts))).strip()
    return out


@dataclass
class EvidenceSource:
    """Where one rubric evidence quote comes from (see :func:`classify_evidence`)."""

    quote: str
    origin: str  # nora_citation | nora_answer | llm_answer | paper_quote | paper_quote_attributed
    references: List[Reference] = field(default_factory=list)
    system: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"key": quote_key(self.quote), "origin": self.origin, "system": self.system,
                "references": [r.to_dict() for r in self.references]}


def _nora_ref(cit: Mapping[str, Any]) -> Reference:
    return Reference(s2_corpus_id=cit.get("corpus_id"), source="nora_citation")


def classify_evidence(quote: str, sources: Optional[SourceAnswers], min_words: int = 6) -> EvidenceSource:
    """Classify an evidence quote by its origin (module docstring)."""
    nq = _norm(quote)
    if sources is None or len(nq.split()) < min_words:
        return EvidenceSource(quote, "paper_quote")
    for cit in sources.nora_citations.values():
        for snip in cit.get("snippets") or []:
            ns = _norm(snip)
            if len(ns.split()) >= min_words and (ns in nq or nq in ns):
                return EvidenceSource(quote, "nora_citation", [_nora_ref(cit)], NORA)
    probe = " ".join(nq.split()[:12])
    if NORA in sources.texts and probe in sources.texts[NORA]:
        refs = []
        flat_q = " ".join(quote.split())
        head = flat_q[:40]
        body = flat_q.rstrip(" .;:")
        for sec in sources.nora_sections:
            flat = " ".join(sec.split())
            pos = flat.find(head)
            if pos < 0:
                continue
            # Markers inside the quoted span and those directly following it (Nora cites at sentence end).
            end = pos + len(body)
            markers = _MARKER_RE.findall(flat[pos:end])
            trailing = re.match(r"[\s.;,]*((?:\(\d+,\s[^()]*?\d{4}[a-z]?\)[\s;,]*)+)", flat[end:])
            if trailing:
                markers += _MARKER_RE.findall(trailing.group(1))
            for marker in markers:
                cit = sources.nora_citations.get(marker)
                if cit is not None and cit.get("corpus_id"):
                    refs.append(_nora_ref(cit))
            break
        return EvidenceSource(quote, "nora_answer", refs, NORA)
    for system, text in sources.texts.items():
        if system != NORA and probe in text:
            return EvidenceSource(quote, "llm_answer", [], system)
    return EvidenceSource(quote, "paper_quote")


def cs_case_to_query(case: Mapping[str, Any], sources: Optional[SourceAnswers] = None,
                     aligner: Optional[PaperAligner] = None,
                     attributor: Optional[QuoteAttributor] = None) -> Tuple[Query, CoverageStats, Counter]:
    """Convert one ScholarQA-CS case; returns the query, its gold coverage and evidence-origin counts.

    Gold entries: corpus papers of the Nora citations behind evidence quotes
    (aligned by S2 corpus id) and of the attributed paper quotes; unaligned
    entries are the unmatched citations (``s2corpus:<id>``) and the
    unattributed paper quotes (``quote:<hash>``). Quotes taken from LLM
    answers are not gold sources.
    """
    cfg = case["metric_config"].get("config", {})
    props = cfg.get("other_properties") or []
    rubric, evidence = [], []
    for prop in props:
        name = str(prop.get("name", ""))
        kind = "nice_to_have" if name.startswith("nice_to_have") else "most_important"
        rubric.append(RubricItem(text=str(prop.get("criterion", "")), weight=float(prop.get("weight", 0.0)), kind=kind))
        evidence.append([str(e) for e in prop.get("evidence") or []])
    origins: Counter = Counter()
    refs: List[Reference] = []
    attributed: Dict[str, None] = {}
    unattributed: Dict[str, None] = {}
    classified: List[EvidenceSource] = []
    for quote in dict.fromkeys(q for ev in evidence for q in ev):
        src = classify_evidence(quote, sources)
        if src.origin == "paper_quote":
            pid = attributor.attribution(quote)[0] if attributor is not None else None
            if pid is not None:
                src.origin = "paper_quote_attributed"
                attributed[pid] = None
            else:
                unattributed[quote_key(quote)] = None
        origins[src.origin] += 1
        classified.append(src)
        refs.extend(src.references)
    res = resolve_references(refs, aligner)
    gold = list(dict.fromkeys(list(res.gold_papers) + list(attributed)))
    unaligned = list(res.unaligned_gold) + list(unattributed)
    stats = CoverageStats(total=len(gold) + len(unaligned), aligned=len(gold), by_method=dict(res.stats.by_method))
    extra = len(gold) - len(res.gold_papers)
    if extra:
        stats.by_method["quote_attribution"] = extra
    query = Query(
        query_id=str(case.get("case_id")),
        text=str(case["initial_prompt"]).strip(),
        kind=EVIDENCE,
        benchmark=CS_BENCHMARK,
        gold_papers=gold,
        unaligned_gold=unaligned,
        rubric=rubric,
        metadata={
            "case_id": case.get("case_id"),
            "annotator": case.get("annotator"),
            "agreement": case.get("agreement"),
            "metric_config": case["metric_config"],
            "rubric_names": [str(p.get("name", "")) for p in props],
            "evidence": evidence,
            "evidence_sources": [s.to_dict() for s in classified],
            "gold_source": "evidence_quotes",
        },
    )
    return query, stats, origins


def cs_evidence_quotes(cases: Sequence[Mapping[str, Any]]) -> List[str]:
    """All distinct evidence quotes of the given cases (input of a :class:`QuoteAttributor`)."""
    out: Dict[str, None] = {}
    for c in cases:
        for prop in c["metric_config"].get("config", {}).get("other_properties") or []:
            for e in prop.get("evidence") or []:
                out[str(e)] = None
    return list(out)


def dedupe_cs_cases(cases: Sequence[Mapping[str, Any]]) -> List[Tuple[Mapping[str, Any], List[Mapping[str, Any]]]]:
    """Group cases by question: ``[(first case, [other annotations]), ...]`` in file order (official scorer)."""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for c in cases:
        groups.setdefault(" ".join(str(c["initial_prompt"]).split()), []).append(c)
    return [(g[0], g[1:]) for g in groups.values()]


def load_scholarqa_cs(
    path: PathLike,
    sources_path: Optional[PathLike] = None,
    aligner: Optional[PaperAligner] = None,
    attributor: Optional[QuoteAttributor] = None,
    dedupe: bool = True,
    limit: Optional[int] = None,
) -> Tuple[List[Query], CoverageStats, Dict[str, int]]:
    """ScholarQA-CS queries (one per unique question when ``dedupe``).

    ``path`` is the ScholarQABench root or ``test_configs_snippets.json``;
    ``sources_path`` the root or ``output_snippets.jsonl`` (defaults to the
    same root when present). Returns queries, gold coverage and counts of
    evidence-quote origins.
    """
    cases = read_cs_cases(path)
    if sources_path is None and Path(path).is_dir():
        try:
            sources_path = _resolve(path, CS_SOURCES_FILE)
        except FileNotFoundError:
            sources_path = None
    sources = read_cs_sources(sources_path) if sources_path is not None else {}
    groups = dedupe_cs_cases(cases) if dedupe else [(c, []) for c in cases]
    if limit is not None:
        groups = groups[:limit]
    queries: List[Query] = []
    stats = CoverageStats()
    origins: Counter = Counter()
    for case, others in groups:
        q, st, org = cs_case_to_query(case, sources.get(_norm(case["initial_prompt"])), aligner, attributor)
        q.metadata["alternate_cases"] = [{"case_id": o.get("case_id"), "annotator": o.get("annotator"),
                                          "agreement": o.get("agreement"), "metric_config": o["metric_config"]}
                                         for o in others]
        queries.append(q)
        stats.merge(st)
        origins.update(org)
    return queries, stats, dict(origins)


def cs_test_configs(queries: Sequence[Query], include_alternates: bool = False) -> List[Dict[str, Any]]:
    """Rebuild ``test_configs_snippets.json`` entries for the official ``rubric_eval.py`` from queries."""
    out = []
    for q in queries:
        md = q.metadata
        out.append({"initial_prompt": q.text, "metric_config": md["metric_config"], "case_id": md.get("case_id"),
                    "annotator": md.get("annotator"), "agreement": md.get("agreement")})
        if include_alternates:
            for alt in md.get("alternate_cases", []):
                out.append({"initial_prompt": q.text, **alt})
    return out


# ---------------------------------------------------------------------------
# ScholarQA-Multi
# ---------------------------------------------------------------------------


def read_multi(path: PathLike) -> List[Dict[str, Any]]:
    data = json.loads(_resolve(path, MULTI_FILE).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data", [])
    for r in data:
        if "input" not in r or "ctxs" not in r:
            raise ValueError(f"Not a ScholarQA-Multi record: {str(r)[:200]}")
    return data


def ctx_reference(ctx: Mapping[str, Any]) -> Reference:
    """Reference of one expert citation (title, S2 url -> paperId, year, first author)."""
    authors = ctx.get("authors") or []
    first = authors[0].get("name") if authors and isinstance(authors[0], Mapping) else (authors[0] if authors else None)
    return Reference(title=" ".join(str(ctx.get("title") or "").split()) or None, url=ctx.get("url") or None,
                     year=ctx.get("year"), first_author=first, source="ctxs")


def cited_indices(answer: str, n_ctxs: int) -> List[int]:
    """Indices of ``ctxs`` cited in an expert answer (0-based markers ``[0]``, ``[1, 3]``)."""
    found: Dict[int, None] = {}
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer or ""):
        for x in m.group(1).split(","):
            i = int(x)
            if 0 <= i < n_ctxs:
                found[i] = None
    return sorted(found)


def multi_record_to_query(record: Mapping[str, Any], aligner: Optional[PaperAligner] = None) -> Tuple[Query, CoverageStats]:
    ctxs = list(record.get("ctxs") or [])
    res = resolve_references([ctx_reference(c) for c in ctxs], aligner)
    query = Query(
        query_id=str(record["id"]),
        text=str(record["input"]).strip(),
        kind=EVIDENCE,
        benchmark=MULTI_BENCHMARK,
        gold_papers=res.gold_papers,
        unaligned_gold=res.unaligned_gold,
        reference_answer=record.get("output"),
        metadata={
            "id": record["id"],
            "subject": record.get("subject"),
            "annotator": record.get("annotator"),
            "citation_marker_base": 0,
            "cited_ctx_indices": cited_indices(str(record.get("output") or ""), len(ctxs)),
            "ctxs": [{"title": c.get("title"), "url": c.get("url"), "year": c.get("year"),
                      "text": c.get("text")} for c in ctxs],
            "citations": [m.to_dict() for m in res.matches],
        },
    )
    return query, res.stats


def load_scholarqa_multi(path: PathLike, aligner: Optional[PaperAligner] = None,
                         limit: Optional[int] = None) -> Tuple[List[Query], CoverageStats]:
    """ScholarQA-Multi queries (``path``: ScholarQABench root or ``human_answers.json``)."""
    queries, stats = [], CoverageStats()
    for i, rec in enumerate(read_multi(path)):
        if limit is not None and i >= limit:
            break
        q, st = multi_record_to_query(rec, aligner)
        queries.append(q)
        stats.merge(st)
    return queries, stats
