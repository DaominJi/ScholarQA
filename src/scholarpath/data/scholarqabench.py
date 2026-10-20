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
(:class:`EvidenceClassifier`), using the system answers the annotators worked
from (``output_snippets.jsonl``) when available. Quotes and answers are
compared by word 5-gram shingles (:func:`~scholarpath.data.alignment.words`
splits words glued by a lost markdown line break, e.g. ``"### Emphasizing
Foundational ConceptsOne of ..."``), so a quote is recognised wherever it
starts and however it was copied:

* a quote overlapping a snippet of a citation in the Ai2 ScholarQA ("Nora")
  answer (overlap coefficient ``>= min_fraction``) -> that citation's
  Semantic Scholar ``corpus_id``;
* a quote whose shingles are covered ``>= min_fraction`` by the Nora answer
  text -> the citations marked inside and right after the matched span
  (``(k, Author et al., YYYY)`` markers, or inline ``<Paper corpusId=...>``
  tags in newer answers; Nora's ``(LLM MEMORY, YYYY)`` statements cite no paper);
* a quote covered ``>= min_fraction`` by another system's answer
  (Perplexity, GPT, Claude) is model-generated text, not a paper, and is
  not a gold source;
* any other quote is a verbatim excerpt of a paper the expert read; it is
  attributed to the arXiv-Subfields paper whose text contains it
  (:class:`~scholarpath.data.alignment.QuoteAttributor`). Quotes attributed
  to the same paper count once. Unattributed quotes cannot be told apart by
  source, so by default the unattributed quotes of one rubric item are
  pooled into a single unaligned gold entry ``quote:<hash>``
  (``unattributed="item"``; ``"quote"`` counts every quote). Pooling avoids
  counting several excerpts of one absent paper several times, at the cost
  of under-counting items that quote several absent papers.

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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from scholarpath.data.alignment import (
    CoverageStats,
    PaperAligner,
    QuoteAttributor,
    Reference,
    quote_key,
    resolve_references,
    shingles,
    word_spans,
    words,
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
# Nora citation markers: "(k, Author et al., YYYY)" or, in newer answers, an inline tag
# '<Paper corpusId="267740476" paperTitle="(115, Taylor et al., 2024)" isShortName></Paper>'.
_CITE = r"(?:<Paper\b[^>]*>(?:\s*</Paper>)?|\(\d+,\s[^()]*?\d{4}[a-z]?\))"
_CITE_RE = re.compile(_CITE)
_TRAILING_CITES_RE = re.compile(r"[\s.;,]*((?:" + _CITE + r"[\s;,]*)+)")
_TAG_CORPUS_ID_RE = re.compile(r'corpusId="(\d+)"')
_WORDS_RE = re.compile(r"[a-z0-9]+")
UNATTRIBUTED_MODES = ("item", "quote")


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

    texts: Dict[str, str] = field(default_factory=dict)  # system -> answer text (Nora sections joined)
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
                sa.texts[name] = (sa.texts.get(name, "") + "\n\n" + ans).strip()
                continue
            parts = []
            for sec in ans or []:
                text = str(sec.get("text", ""))
                parts.append(f"{sec.get('title', '')}\n{sec.get('tldr', '')}\n{text}")
                if name == NORA:
                    sa.nora_sections.append(text)
                    for cit in sec.get("citations") or []:
                        sa.nora_citations.setdefault(str(cit.get("id", "")), cit)
            sa.texts[name] = (sa.texts.get(name, "") + "\n\n" + "\n\n".join(parts)).strip()
    return out


@dataclass
class EvidenceSource:
    """Where one rubric evidence quote comes from (see :class:`EvidenceClassifier`)."""

    quote: str
    origin: str  # nora_citation | nora_answer | llm_answer | paper_quote | paper_quote_attributed
    references: List[Reference] = field(default_factory=list)
    system: Optional[str] = None
    score: float = 0.0  # shingle overlap with the matched source
    gold_key: Optional[str] = None  # gold entry the quote contributes to (paper quotes)

    def to_dict(self) -> Dict[str, Any]:
        return {"key": quote_key(self.quote), "origin": self.origin, "system": self.system,
                "score": round(self.score, 3), "gold_key": self.gold_key,
                "references": [r.to_dict() for r in self.references]}


def _nora_ref(cit: Mapping[str, Any]) -> Reference:
    return Reference(s2_corpus_id=cit.get("corpus_id"), source="nora_citation")


class EvidenceClassifier:
    """Classify the rubric evidence quotes of one question by their origin (module docstring).

    Parameters
    ----------
    sources:
        The system answers of the question (``None``: every quote is a paper quote).
    shingle:
        Word n-gram size of the shingles.
    min_fraction:
        Minimum share of a quote's shingles found in a system answer (or
        overlap coefficient with a Nora citation snippet) for the quote to be
        assigned to that source.
    min_words:
        Quotes (and snippets) with fewer words are not matched.
    """

    def __init__(self, sources: Optional[SourceAnswers], shingle: int = 5, min_fraction: float = 0.5,
                 min_words: int = 6):
        self.k = int(shingle)
        self.min_fraction = float(min_fraction)
        self.min_words = max(int(min_words), self.k)
        self.sources = sources
        self.systems: Dict[str, Set[Tuple[str, ...]]] = {}
        self.snippets: List[Tuple[Mapping[str, Any], Set[Tuple[str, ...]]]] = []
        self.sections: List[Tuple[str, List[str], List[Tuple[int, int]]]] = []
        if sources is None:
            return
        self.systems = {name: shingles(words(text), self.k) for name, text in sources.texts.items()}
        for cit in sources.nora_citations.values():
            for snip in cit.get("snippets") or []:
                toks = words(str(snip))
                if len(toks) >= self.min_words:
                    self.snippets.append((cit, shingles(toks, self.k)))
        for sec in sources.nora_sections:
            tokens, spans = word_spans(sec)
            self.sections.append((sec, tokens, spans))

    def classify(self, quote: str) -> EvidenceSource:
        tokens = words(quote)
        if self.sources is None or len(tokens) < self.min_words:
            return EvidenceSource(quote, "paper_quote")
        grams = shingles(tokens, self.k)
        # 1. A snippet of a Nora citation (paper text shown with the Nora answer).
        best_cit, best = None, 0.0
        for cit, sg in self.snippets:
            score = len(grams & sg) / min(len(grams), len(sg))
            if score > best:
                best_cit, best = cit, score
        if best_cit is not None and best >= self.min_fraction:
            return EvidenceSource(quote, "nora_citation", [_nora_ref(best_cit)], NORA, best)
        # 2. Text of a system answer (ties prefer Nora, whose citations identify papers).
        scores = {name: len(grams & sg) / len(grams) for name, sg in self.systems.items()}
        if scores:
            system = max(scores, key=lambda n: (scores[n], n == NORA, n))
            if scores[system] >= self.min_fraction:
                if system == NORA:
                    return EvidenceSource(quote, "nora_answer", self._nora_markers(grams), NORA, scores[system])
                return EvidenceSource(quote, "llm_answer", [], system, scores[system])
        return EvidenceSource(quote, "paper_quote")

    def _nora_markers(self, grams: Set[Tuple[str, ...]]) -> List[Reference]:
        """Citations marked inside the matched span of the Nora answer and directly after it."""
        k = self.k
        best: Tuple[int, Optional[str], int, int] = (0, None, 0, 0)
        for raw, tokens, spans in self.sections:
            pos = [i for i in range(len(tokens) - k + 1) if tuple(tokens[i:i + k]) in grams]
            if not pos:
                continue
            # Densest run of matched positions (gaps <= k) = the copied span.
            runs: List[List[int]] = [[pos[0]]]
            for i in pos[1:]:
                if i - runs[-1][-1] <= k:
                    runs[-1].append(i)
                else:
                    runs.append([i])
            run = max(runs, key=len)
            if len(run) > best[0]:
                best = (len(run), raw, spans[run[0]][0], spans[run[-1] + k - 1][1])
        _, raw, start, end = best
        if raw is None:
            return []
        inside = [m for m in _CITE_RE.finditer(raw) if m.start() < end and m.end() > start]
        cites = [m.group() for m in inside]
        trailing = _TRAILING_CITES_RE.match(raw, max([end] + [m.end() for m in inside]))
        if trailing:
            cites += _CITE_RE.findall(trailing.group(1))
        refs: Dict[str, Reference] = {}
        for cite in cites:
            ref = self._cite_reference(cite)
            if ref is not None:
                refs.setdefault(ref.key, ref)
        return list(refs.values())

    def _cite_reference(self, cite: str) -> Optional[Reference]:
        """Reference of one Nora marker (``corpusId`` of an inline tag, else the citation list entry)."""
        if cite.startswith("<"):
            m = _TAG_CORPUS_ID_RE.search(cite)
            return Reference(s2_corpus_id=m.group(1), source="nora_citation") if m else None
        cit = self.sources.nora_citations.get(cite) if self.sources is not None else None
        return _nora_ref(cit) if cit is not None and cit.get("corpus_id") else None


def classify_evidence(quote: str, sources: Optional[SourceAnswers], min_fraction: float = 0.5,
                      min_words: int = 6) -> EvidenceSource:
    """Classify one evidence quote (convenience wrapper around :class:`EvidenceClassifier`)."""
    return EvidenceClassifier(sources, min_fraction=min_fraction, min_words=min_words).classify(quote)


def cs_case_to_query(case: Mapping[str, Any], sources: Optional[SourceAnswers] = None,
                     aligner: Optional[PaperAligner] = None,
                     attributor: Optional[QuoteAttributor] = None,
                     unattributed: str = "item",
                     classifier: Optional[EvidenceClassifier] = None) -> Tuple[Query, CoverageStats, Counter]:
    """Convert one ScholarQA-CS case; returns the query, its gold coverage and evidence-origin counts.

    Gold entries: corpus papers of the Nora citations behind evidence quotes
    (aligned by S2 corpus id) and of the attributed paper quotes (one entry
    per paper); unaligned entries are the unmatched citations
    (``s2corpus:<id>``) and the unattributed paper quotes, pooled per rubric
    item (``unattributed="item"``) or one per quote (``"quote"``). Quotes
    taken from LLM answers are not gold sources.
    """
    if unattributed not in UNATTRIBUTED_MODES:
        raise ValueError(f"unattributed must be one of {UNATTRIBUTED_MODES}, got {unattributed!r}")
    classifier = classifier if classifier is not None else EvidenceClassifier(sources)
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
    unattributed_keys: Dict[str, None] = {}
    classified: List[EvidenceSource] = []
    seen: Set[str] = set()
    for item_quotes in evidence:
        item_key: Optional[str] = None
        for quote in item_quotes:
            qk = quote_key(quote)
            if qk in seen:  # the same quote supporting several items
                continue
            seen.add(qk)
            src = classifier.classify(quote)
            if src.origin == "paper_quote":
                pid, frac = attributor.attribution(quote) if attributor is not None else (None, 0.0)
                if pid is not None:
                    src.origin, src.score, src.gold_key = "paper_quote_attributed", frac, pid
                    attributed[pid] = None
                else:
                    if unattributed == "quote" or item_key is None:
                        item_key = qk
                    src.gold_key = item_key
                    unattributed_keys[item_key] = None
            origins[src.origin] += 1
            classified.append(src)
            refs.extend(src.references)
    res = resolve_references(refs, aligner)
    gold = list(dict.fromkeys(list(res.gold_papers) + list(attributed)))
    unaligned = list(res.unaligned_gold) + list(unattributed_keys)
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
            "unattributed_pooling": unattributed,
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
    unattributed: str = "item",
    min_fraction: float = 0.5,
) -> Tuple[List[Query], CoverageStats, Dict[str, int]]:
    """ScholarQA-CS queries (one per unique question when ``dedupe``).

    ``path`` is the ScholarQABench root or ``test_configs_snippets.json``;
    ``sources_path`` the root or ``output_snippets.jsonl`` (defaults to the
    same root when present). ``unattributed`` and ``min_fraction`` are
    passed to :func:`cs_case_to_query` / :class:`EvidenceClassifier`.
    Returns queries, gold coverage and counts of evidence-quote origins.
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
        classifier = EvidenceClassifier(sources.get(_norm(case["initial_prompt"])), min_fraction=min_fraction)
        q, st, org = cs_case_to_query(case, aligner=aligner, attributor=attributor, unattributed=unattributed,
                                      classifier=classifier)
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
