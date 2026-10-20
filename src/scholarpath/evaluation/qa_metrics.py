"""End-to-end QA metrics on the evidence benchmarks (Section 5.3, Table 3, App. C.3).

"For end-to-end QA, we report each evidence benchmark's official metrics:
rubric coverage (Rub) and citation F1 (Cite) on ResearchQA; combined
correctness (Corr) and citation F1 (Cite) on ScholarQABench" (Section 5.3).
Appendix C.3: "ResearchQA rubric coverage and ScholarQA-CS correctness and
citation F1 use the official evaluation scripts of their benchmarks, while
ScholarQA-Multi is scored by Prometheus-v2-7B (temperature 0) for
organization, coverage, and relevance; the ScholarQABench scores ... average
the two subsets weighted by their numbers of queries."

This module re-implements those protocols in-process (so they run with any
:class:`~scholarpath.llm.LLMClient` judge, including the offline mock), and
:mod:`scholarpath.evaluation.official` exports predictions for, and runs, the
official scripts themselves.

ResearchQA (github.com/realliyifei/ResearchQA ``compute_coverage.py``;
olmo-eval ``researchqa`` task; dr-tulu port):

* :func:`rubric_coverage` -- rubric items are judged in batches of 8 with the
  official prompt; the judge outputs one of five ordinal labels per item
  (Not at all / Barely / Moderately / Mostly / Completely -> 1..5), each
  normalized to (s-1)/4; an answer's coverage is the mean over its items and
  "Rub" is the mean over answers (x100). A batch whose output has the wrong
  number of lines is retried (3 attempts in total) and, as in the official
  script, the whole answer is then dropped (``on_failure="drop"``);
  ``on_failure="zero"`` instead scores the items of that batch 0 like
  olmo-eval. Coverage restricted to ``Citation``-type items (pooled over
  items, olmo-eval's ``coverage_citation``) and the other rubric types are
  reported too.
* :func:`citation_f1` -- "Cite". The official repository has no separate
  citation metric, so Cite is defined here as the F1 of the answer's cited
  papers against the gold source papers of the query: precision over the
  distinct cited papers, recall over ``Query.num_gold`` (unaligned gold
  citations stay in the denominator, App. C.2); macro-averaged over queries
  like ScholarQABench's ``compute_citation_coverage`` (``evidence_f1``).

ScholarQABench (github.com/AkariAsai/ScholarQABench, MIT License):

* :func:`correctness` -- ScholarQA-CS rubric score of ``scripts/rubric_eval.py``
  + ``scripts/scholarqacs_rubric.py``: weighted sum of a length score, a 0-10
  expertise score, the fraction of claims with citations, the fraction of
  citations with excerpts, and per rubric item a 0-10 criterion score and an
  evidence-snippet recall (each half of the item weight; the evidence score is
  0 whenever the criterion score is 0). ``score`` (all components) is Corr;
  ``ann_score`` is the rubric part rescaled by 1/0.6. Answers are judged in
  the official input format of ``scripts/convert_answer_scholar_cs.py``
  (:func:`render_cs_answer`): numbered citations followed by a
  ``References:`` block with the title and text of every cited passage.
* :func:`citation_attribution` -- statement-level citation precision / recall
  of ``scripts/citation_correctness_eval.py`` (AutoAIS with the AttrScore NLI
  model ``osunlp/attrscore-flan-t5-xl``): the output is split into
  sentences, sentences shorter than 50 characters are skipped, a sentence
  without citations inherits the previous sentence's citations, at most 3
  citations are used per sentence, a sentence is supported iff its joint
  cited passages entail it (recall), and a citation is precise unless it is
  individually unsupportive *and* removable without losing support
  (ALCE-style precision). Citation F1 = 2PR/(P+R) of the averaged P and R.
* :func:`prometheus_grade` -- ScholarQA-Multi Prometheus-2 absolute grading
  (``scripts/prometheus_eval.py`` with ``rubrics/prometheus_rubrics_v8.json``)
  for organization, coverage and relevance, with the expert answer as the
  score-5 reference; 1-5 scores are mapped to 0-100 by (s-1)/4 x 100.
* :func:`scholarqabench_combined` -- query-count-weighted average of the CS and
  Multi subsets.

Units: per-item functions return fractions in [0, 1] (Prometheus: 1-5);
``aggregate_*`` / ``evaluate_*`` summaries are in percent (the paper's units).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from scholarpath.evaluation.judges import JudgeLike, as_judge_llm
from scholarpath.llm import LLMClient
from scholarpath.prompts import evaluation as P
from scholarpath.types import Answer, Paper, Query, RubricItem, unique

log = logging.getLogger(__name__)

AnswerLike = Union[Answer, str]
PaperLookup = Any  # Mapping[str, Paper] or scholarpath.data.corpus.Corpus (anything with .get(pid))


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else None


def _pct(value: Optional[float]) -> Optional[float]:
    return None if value is None else 100.0 * value


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean 2PR/(P+R) (0 when P + R = 0)."""
    return 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Answer rendering helpers
# ---------------------------------------------------------------------------

_ID_MARKER_RE = re.compile(r"\[([^\[\]]+)\]")
_WS_RE = re.compile(r"\s")

#: The known paper ids: a set / mapping / :class:`~scholarpath.data.corpus.Corpus` (``pid in ids``),
#: an object with ``.get(pid)`` or a callable ``pid -> object | None`` (a paper or passage lookup;
#: ``None`` = unknown), or any other iterable of ids.
IdSet = Any


def _split_marker(content: str) -> List[str]:
    return [t.strip() for t in re.split(r"[,;]", content) if t.strip()]


def _id_predicate(valid_ids: Optional[IdSet]) -> Optional[Callable[[str], bool]]:
    """``pid -> bool`` membership test for :data:`IdSet` (``None`` when no id set is given)."""
    if valid_ids is None:
        return None
    if hasattr(valid_ids, "__contains__"):
        return lambda pid: pid in valid_ids
    if hasattr(valid_ids, "get"):
        return lambda pid: valid_ids.get(pid) is not None
    if callable(valid_ids):
        return lambda pid: valid_ids(pid) is not None
    ids = {str(v) for v in valid_ids}
    return lambda pid: pid in ids


def _marker_tokens(content: str) -> List[str]:
    """Whitespace-free tokens of a bracket marker -- the only tokens that can be paper ids."""
    return [t for t in _split_marker(content) if not _WS_RE.search(t)]


def answer_text(answer: AnswerLike) -> str:
    """The answer as the judge sees it: ``Answer.text``, or its items when the text is empty."""
    if isinstance(answer, str):
        return answer
    if answer.text and answer.text.strip():
        return answer.text
    lines = []
    for it in answer.items:
        cites = "".join(f"[{c}]" for c in it.citations)
        lines.append(f"{it.text} {cites}".strip())
    return "\n".join(lines)


def cited_papers(answer: AnswerLike, valid_ids: Optional[IdSet] = None) -> List[str]:
    """Distinct cited paper ids, in order of first citation.

    Uses the structured item citations of an :class:`Answer` as they are
    (a citation of a paper outside the corpus stays in, so it counts against
    precision). Without structured citations (text-only answers, e.g. from
    external agents) it falls back to the whitespace-free tokens of ``[id]`` /
    ``[id1, id2]`` markers in the text; ``valid_ids`` (e.g. the corpus) then
    restricts those tokens to real paper ids, so bracketed text such as
    ``[CLS]``, ``[Smith 2020]`` or the interval ``[0, 1]`` is not mistaken for
    a citation. Pass it whenever text-only answers are scored.
    """
    ids: List[str] = list(answer.cited_papers) if isinstance(answer, Answer) else []
    if not ids:
        valid = _id_predicate(valid_ids)
        ids = unique(t for m in _ID_MARKER_RE.findall(answer_text(answer)) for t in _marker_tokens(m)
                     if valid is None or valid(t))
    return ids


def format_reference(paper: Paper) -> str:
    """``First-Author et al. (Year). Title.`` -- lets judges match "Does the response cite X (Year)?"."""
    parts = []
    if paper.authors:
        first = str(paper.authors[0]).strip()
        surname = first.split(",")[0].strip() if "," in first else (first.split()[-1] if first.split() else first)
        parts.append(surname + (" et al." if len(paper.authors) > 1 else ""))
    year = paper.year or (int(paper.date[:4]) if paper.date and paper.date[:4].isdigit() else None)
    if year:
        parts.append(f"({year})")
    head = " ".join(parts)
    title = paper.title.strip().rstrip(".")
    return f"{head}. {title}." if head else f"{title}."


def render_answer(answer: AnswerLike, papers: Optional[PaperLookup] = None, references: bool = True) -> str:
    """Answer text plus a ``References:`` list of the cited papers (author, year, title).

    Our answers cite corpus ids (``[2101.00001]``); appending the reference
    list gives an LLM judge the bibliographic information that rubric items
    such as "Does the response cite Desneux et al. (2007) (title: ...)?" ask
    for. The same rendering is applied to every system.
    """
    text = answer_text(answer)
    if not references or papers is None:
        return text
    lines = []
    for pid in cited_papers(answer):
        paper = papers.get(pid)
        if paper is not None:
            lines.append(f"[{pid}] {format_reference(paper)}")
    return text + ("\n\nReferences:\n" + "\n".join(lines) if lines else "")


# ---------------------------------------------------------------------------
# (a) ResearchQA rubric coverage ("Rub")
# ---------------------------------------------------------------------------

RESEARCHQA_BATCH_SIZE = 8
RESEARCHQA_MAX_ATTEMPTS = 3  # @retry(tries=3) in compute_coverage.py
RUBRIC_TYPES = ("citation", "comparison", "example", "impact", "limitation", "other")


def normalize_5scale(score: float) -> float:
    """(s - 1) / 4: maps the ordinal 1..5 judge scale to [0, 1]."""
    return (float(score) - 1.0) / 4.0


def rubric_item_types(item: Union[RubricItem, str]) -> Tuple[str, ...]:
    """Lower-cased ResearchQA rubric types of an item (``RubricItem.kind``, e.g. ``"Citation,Other"``)."""
    if isinstance(item, str):
        return ()
    return tuple(unique(t for t in re.split(r"[\s,;|/+]+", (item.kind or "").lower()) if t))


@dataclass
class RubricJudgement:
    rubric: str
    label: Optional[str]  # one of COVERAGE_LABELS; None when the batch failed
    score: Optional[float]  # (s-1)/4 in [0, 1]
    types: Tuple[str, ...] = ()


@dataclass
class CoverageResult:
    """Coverage of one answer (``coverage`` is ``None`` when the answer was dropped)."""

    query_id: str
    judgements: List[RubricJudgement] = field(default_factory=list)
    coverage: Optional[float] = None
    failed: bool = False
    failed_batches: int = 0
    judge_calls: int = 0

    def type_scores(self, rubric_type: str) -> List[float]:
        t = rubric_type.lower()
        return [j.score for j in self.judgements if t in j.types and j.score is not None]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["judgements"] = [asdict(j) for j in self.judgements]
        return d


def _rubric_of(rubric: Union[Query, Sequence[Union[RubricItem, str]]]) -> List[RubricItem]:
    items = rubric.rubric if isinstance(rubric, Query) else list(rubric)
    return [it if isinstance(it, RubricItem) else RubricItem(text=str(it)) for it in items]


def rubric_coverage(
    answer: AnswerLike,
    rubric: Union[Query, Sequence[Union[RubricItem, str]]],
    judge: JudgeLike,
    *,
    query_id: Optional[str] = None,
    batch_size: int = RESEARCHQA_BATCH_SIZE,
    max_attempts: int = RESEARCHQA_MAX_ATTEMPTS,
    on_failure: str = "drop",
    model: Optional[str] = None,
    papers: Optional[PaperLookup] = None,
    references: bool = True,
) -> CoverageResult:
    """ResearchQA rubric coverage of one answer (official ``compute_coverage`` protocol).

    ``rubric`` is a :class:`Query` (its ``rubric``) or a list of rubric items.
    ``judge`` is the judge client (official default model ``gpt-4.1-mini`` --
    pass ``model`` to override the client's model per request). With
    ``papers``, the answer is rendered with a reference list
    (:func:`render_answer`).
    """
    if on_failure not in {"drop", "zero"}:
        raise ValueError("on_failure must be 'drop' (official script) or 'zero' (olmo-eval)")
    llm = as_judge_llm(judge)
    items = _rubric_of(rubric)
    qid = query_id or (rubric.query_id if isinstance(rubric, Query) else "")
    response = render_answer(answer, papers, references) if isinstance(answer, Answer) else str(answer)
    result = CoverageResult(query_id=qid)
    if not items:
        log.warning("Query %s has no rubric items; coverage is undefined", qid)
        return result
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        request = P.rubric_coverage_request(response, [it.text for it in batch], model=model)
        labels: Optional[List[str]] = None
        for attempt in range(max_attempts):
            try:
                resp = llm.complete(request, _salt=f"retry{attempt}" if attempt else "")
            except Exception as err:  # transport failure: the official script skips the answer
                log.error("Judge call failed for %s: %s", qid, err)
                break
            result.judge_calls += 1
            labels = P.parse_coverage_labels(resp.text, len(batch))
            if labels is not None:
                break
            log.warning("Expected %d labels for %s, got malformed output (attempt %d/%d)",
                        len(batch), qid, attempt + 1, max_attempts)
        if labels is None:
            result.failed_batches += 1
            if on_failure == "drop":
                log.error("Dropping answer %s after %d malformed judge outputs", qid, max_attempts)
                result.failed = True
                result.judgements = []
                result.coverage = None
                return result
            labels_or_none: List[Optional[str]] = [None] * len(batch)
        else:
            labels_or_none = list(labels)
        for it, lab in zip(batch, labels_or_none):
            score = normalize_5scale(P.COVERAGE_SCORES[lab]) if lab is not None else 0.0
            result.judgements.append(RubricJudgement(it.text, lab, score, rubric_item_types(it)))
    result.coverage = _mean(j.score for j in result.judgements if j.score is not None)
    return result


def aggregate_coverage(results: Sequence[CoverageResult]) -> Dict[str, Optional[float]]:
    """``Rub`` = mean coverage over scored answers (x100) plus pooled per-type coverage.

    Dropped answers are excluded from the mean, as in the official script
    (their count is reported as ``n_failed``). ``coverage_<type>`` pools the
    item scores of that rubric type over all scored answers (olmo-eval).
    """
    scored = [r for r in results if r.coverage is not None]
    out: Dict[str, Optional[float]] = {
        "Rub": _pct(_mean(r.coverage for r in scored)),  # type: ignore[misc]
        "n": float(len(scored)),
        "n_failed": float(sum(1 for r in results if r.failed)),
    }
    types = unique(t for r in scored for j in r.judgements for t in j.types)
    for t in unique(list(RUBRIC_TYPES) + types):
        vals = [s for r in scored for s in r.type_scores(t)]
        if vals:
            out[f"coverage_{t}"] = _pct(_mean(vals))
    return out


# ---------------------------------------------------------------------------
# (b) Citation F1 against gold source papers (ResearchQA "Cite")
# ---------------------------------------------------------------------------


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    n_pred: int = 0
    n_gold: int = 0
    hits: int = 0


def citation_prf(cited: Iterable[str], gold: Iterable[str], num_gold: Optional[int] = None) -> PRF:
    """Precision over distinct cited papers, recall over ``num_gold`` (>= aligned gold)."""
    pred = unique(str(c) for c in cited)
    gold_set = {str(g) for g in gold}
    n_gold = len(gold_set) if num_gold is None else max(int(num_gold), len(gold_set))
    hits = sum(1 for p in pred if p in gold_set)
    precision = hits / len(pred) if pred else 0.0
    recall = hits / n_gold if n_gold else 0.0
    return PRF(precision, recall, f1_score(precision, recall), len(pred), n_gold, hits)


def citation_f1(answer: AnswerLike, query: Query, valid_ids: Optional[IdSet] = None) -> PRF:
    """ResearchQA Cite of one answer: cited papers vs. ``query.gold_papers`` / ``query.num_gold``.

    ``valid_ids`` (e.g. the corpus) filters the text-marker fallback of
    :func:`cited_papers` for answers without structured citations.
    """
    return citation_prf(cited_papers(answer, valid_ids), query.gold_papers, query.num_gold)


def aggregate_prf(scores: Sequence[PRF]) -> Dict[str, Optional[float]]:
    """Macro averages over queries (percent): ``Cite`` is the mean per-query F1."""
    return {
        "Cite": _pct(_mean(s.f1 for s in scores)),
        "cite_P": _pct(_mean(s.precision for s in scores)),
        "cite_R": _pct(_mean(s.recall for s in scores)),
        "n": float(len(scores)),
    }


# ---------------------------------------------------------------------------
# (c) ScholarQA-CS rubric correctness ("Corr")
# ---------------------------------------------------------------------------

STATIC_COMPONENTS = ("length", "expertise", "citations", "excerpts")
CS_RUBRIC_SHARE = 0.6  # rubric items carry 60% of the total weight (``ann_score`` divides by 0.6)


@dataclass
class CSRubricProperty:
    name: str
    criterion: str = ""
    weight: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class CSRubricConfig:
    """``CorpusQaRubricConfig`` of scholarqacs_rubric.py (defaults identical)."""

    question: str
    low_length: int = 300
    high_length: int = 600
    length_weight: float = 0.05
    expertise_weight: float = 0.05
    citations_weight: float = 0.2
    excerpts_weight: float = 0.1
    other_properties: List[CSRubricProperty] = field(default_factory=list)
    # The official config hard-codes "gpt-4-turbo"; None uses the judge client's model.
    model_name: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CSRubricConfig":
        """From an official ``metric_config`` (``{"name", "config"}``) or its ``config`` dict."""
        if "config" in d and isinstance(d["config"], Mapping):
            d = d["config"]
        props = [
            CSRubricProperty(
                name=str(p["name"]),
                criterion=str(p.get("criterion", "") or ""),
                weight=float(p.get("weight", 0.0)),
                evidence=[str(e) for e in (p.get("evidence") or [])],
            )
            for p in d.get("other_properties", []) or []
        ]
        kw = {k: d[k] for k in ("low_length", "high_length", "length_weight", "expertise_weight",
                                "citations_weight", "excerpts_weight", "model_name") if k in d}
        return cls(question=str(d.get("question", "")), other_properties=props, **kw)

    @classmethod
    def from_query(cls, query: Query) -> "CSRubricConfig":
        """Rubric config of a ScholarQA-CS query.

        Uses ``query.metadata["metric_config"]`` (the official test-case field)
        or ``["rubric_config"]`` when present. Otherwise the config is built
        from ``query.rubric`` (criterion = item text, evidence from
        ``metadata["rubric_evidence"][i]``), rescaling the item weights to sum
        to 0.6 as in every official test case.
        """
        md = query.metadata or {}
        for key in ("metric_config", "rubric_config"):
            if isinstance(md.get(key), Mapping):
                cfg = cls.from_dict(md[key])
                if not cfg.question:
                    cfg.question = query.text
                return cfg
        evidence = md.get("rubric_evidence") or []
        props = []
        for i, it in enumerate(query.rubric):
            ev = evidence[i] if i < len(evidence) and isinstance(evidence[i], (list, tuple)) else []
            props.append(CSRubricProperty(name=f"rubric_item_{i}", criterion=it.text, weight=float(it.weight),
                                          evidence=[str(e) for e in ev]))
        total = sum(p.weight for p in props)
        if props and total > 0:
            for p in props:
                p.weight = CS_RUBRIC_SHARE * p.weight / total
        return cls(question=query.text, other_properties=props)

    def restricted(self, use_rubrics: bool = True, use_snippets: bool = True) -> "CSRubricConfig":
        """``--rubrics`` / ``--snippets`` flags of rubric_eval.py (blank criteria / drop evidence)."""
        props = [CSRubricProperty(p.name, p.criterion if use_rubrics else "", p.weight,
                                  list(p.evidence) if use_snippets else []) for p in self.other_properties]
        return CSRubricConfig(self.question, self.low_length, self.high_length, self.length_weight,
                              self.expertise_weight, self.citations_weight, self.excerpts_weight, props,
                              self.model_name)

    def weights(self) -> Dict[str, float]:
        """Component weights of ``score_output``; they must sum to 1."""
        w = {"length": self.length_weight, "expertise": self.expertise_weight,
             "citations": self.citations_weight, "excerpts": self.excerpts_weight}
        for p in self.other_properties:
            if p.criterion:
                w[p.name] = p.weight / (2.0 if p.evidence else 1.0)
        for p in self.other_properties:
            if p.evidence:
                w[f"{p.name}_evidence"] = p.weight / (2.0 if p.criterion else 1.0)
        if abs(sum(w.values()) - 1.0) > 1e-6:
            raise ValueError(f"ScholarQA-CS component weights sum to {sum(w.values()):.6f}, expected 1 "
                             f"(question: {self.question[:80]!r})")
        return w


@dataclass
class CSScore:
    query_id: str
    score: float  # total weighted score in [0, 1] (Corr)
    ann_score: float  # rubric-only part / 0.6
    components: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)


def length_score(response: str, low_length: int = 300, high_length: int = 600) -> float:
    """1 up to ``low_length`` words, linearly down to 0 at ``high_length`` words."""
    words = len(response.split())
    return 1.0 - (max(min(high_length, words), low_length) - low_length) / (high_length - low_length)


def _score_property(llm: LLMClient, question: str, response: str, criterion: str, model: Optional[str]) -> float:
    resp = llm.complete(P.cs_property_request(question, response, criterion, model=model))
    obj = P.extract_json_object(resp.text)
    if not obj:
        return 0.0
    try:
        return float(obj["score"]) / 10.0
    except (KeyError, TypeError, ValueError):
        log.warning("Malformed criterion score %r", obj)
        return 0.0


def _score_evidence(llm: LLMClient, response: str, evidence: Sequence[str], model: Optional[str]) -> float:
    resp = llm.complete(P.cs_evidence_request(response, evidence, model=model))
    obj = P.extract_json_object(resp.text)
    if not obj:
        return 0.0
    try:
        return min(float(obj["score"]) / len(evidence), 1.0)
    except (KeyError, TypeError, ValueError):
        log.warning("Malformed evidence count %r", obj)
        return 0.0


def _score_citations_excerpts(llm: LLMClient, response: str, model: Optional[str]) -> Dict[str, float]:
    resp = llm.complete(P.cs_claims_request(response, model=model))
    obj = P.extract_json_object(resp.text)
    if not obj:
        return {"citations": 0.0, "excerpts": 0.0}
    try:
        claims = obj["claims"]
        citation_score = sum(1 for c in claims if c["citations"]) / max(len(claims), 1)
        all_cites = [ct for c in claims for ct in c["citations"]]
        excerpt_score = sum(1 for ct in all_cites if ct["excerpts"]) / max(len(all_cites), 1)
    except (KeyError, TypeError) as err:
        log.warning("Could not extract citations and excerpts: %s", err)
        return {"citations": 0.0, "excerpts": 0.0}
    return {"citations": float(citation_score), "excerpts": float(excerpt_score)}


def render_cs_answer(answer: AnswerLike, passage_fn: "PassageLookup", valid_ids: Optional[IdSet] = None) -> str:
    """ScholarQA-CS ``answer_text`` as produced by ScholarQABench ``scripts/convert_answer_scholar_cs.py``.

    The official pipeline converts a system output (text with 0-based ``[i]``
    citations + its ``ctxs``) by appending ``"\\nReferences:\\n"`` and, for
    every cited ctx in ctx order, ``"[k] <title> <text>\\n\\n"`` (``k``
    enumerates the cited ctxs). The rubric judge reads these references when
    it splits claims / citations / excerpts, so the ``excerpts`` component
    (10% of the score) depends on them. Paper-id citations are first converted
    to ``[i]`` markers with :func:`number_citations` (markers stay where they
    are); ``passage_fn`` gives the cited text (the context the system saw).
    """
    output, ctxs, _ = number_citations(answer, passage_fn, attach_trailing=False, valid_ids=valid_ids)
    cited = {int(n) for m in _GROUP_CITE_RE.findall(output) for n in m.split(",")}
    used = [c for i, c in enumerate(ctxs) if i in cited]
    references = "\nReferences:\n"
    for k, ctx in enumerate(used):
        references += "[{0}] {1} {2}\n\n".format(k, ctx.get("title", ""), ctx.get("text", ""))
    return output + references


def has_passage_source(
    answer: Answer, papers: Optional[PaperLookup] = None, passage_fn: Optional["PassageLookup"] = None
) -> bool:
    """Can the cited passages of ``answer`` be looked up (``passage_fn``, ``papers`` or stored passages)?"""
    stored = answer.metadata.get("passages") if isinstance(answer.metadata, Mapping) else None
    return passage_fn is not None or papers is not None or bool(stored)


def correctness(
    answer: AnswerLike,
    query: Union[Query, CSRubricConfig, Mapping[str, Any]],
    judge: JudgeLike,
    *,
    use_rubrics: bool = True,
    use_snippets: bool = True,
    model: Optional[str] = None,
    query_id: Optional[str] = None,
    references: bool = True,
    papers: Optional[PaperLookup] = None,
    passage_fn: Optional["PassageLookup"] = None,
) -> CSScore:
    """ScholarQA-CS correctness of one answer (``RubricCorpusQaGenericMetric.score_output``).

    ``query`` is a ScholarQA-CS :class:`Query`, a :class:`CSRubricConfig` or
    an official ``metric_config`` dict. ``model`` overrides the judge model
    (official: ``gpt-4-turbo``).

    An :class:`Answer` is scored in the official input format
    (:func:`render_cs_answer`: numbered citations + ``References:`` block with
    the cited passages), the passages coming from ``passage_fn`` (paper id ->
    passage), else ``answer.metadata["passages"]`` / ``papers``
    (:func:`answer_passages`). ``references=False``, or an answer with no
    passage source at all, scores the raw answer text. A string answer is
    assumed to be in the official format already and is scored as is.
    """
    if isinstance(query, Query):
        cfg = CSRubricConfig.from_query(query)
        qid = query_id or query.query_id
    elif isinstance(query, CSRubricConfig):
        cfg, qid = query, query_id or ""
    else:
        cfg, qid = CSRubricConfig.from_dict(query), query_id or ""
    cfg = cfg.restricted(use_rubrics, use_snippets)
    weights = cfg.weights()
    llm = as_judge_llm(judge)
    model = model or cfg.model_name
    response = answer_text(answer)
    if references and isinstance(answer, Answer):
        if has_passage_source(answer, papers, passage_fn):
            response = render_cs_answer(answer, passage_fn or answer_passages(answer, papers))
        else:
            log.warning("No passages for the citations of %s (pass papers / passage_fn or set "
                        "answer.metadata['passages']); scoring it without a References block", qid)

    comp: Dict[str, float] = {"length": length_score(response, cfg.low_length, cfg.high_length)}
    comp["expertise"] = _score_property(llm, cfg.question, response, P.EXPERTISE_CRITERION, model)
    comp.update(_score_citations_excerpts(llm, response, model))
    for p in cfg.other_properties:
        if p.criterion:
            comp[p.name] = _score_property(llm, cfg.question, response, p.criterion, model)
        if p.evidence:
            # Evidence recall is only scored when the criterion is absent or scored > 0.
            comp[f"{p.name}_evidence"] = (
                _score_evidence(llm, response, p.evidence, model) if (not p.criterion or comp.get(p.name)) else 0.0
            )
    assert set(comp) == set(weights)
    score = sum(weights[k] * comp[k] for k in weights)
    ann = sum(weights[k] * comp[k] for k in weights if k not in STATIC_COMPONENTS) / CS_RUBRIC_SHARE
    return CSScore(query_id=qid, score=float(score), ann_score=float(ann), components=comp, weights=weights)


def aggregate_correctness(scores: Sequence[CSScore]) -> Dict[str, Optional[float]]:
    """``Corr`` = mean total score (x100); ``ann_score`` = mean rubric-only score (x100)."""
    return {
        "Corr": _pct(_mean(s.score for s in scores)),
        "ann_score": _pct(_mean(s.ann_score for s in scores)),
        "n": float(len(scores)),
    }


# ---------------------------------------------------------------------------
# (d) Statement-level citation precision / recall (citation_correctness_eval.py)
# ---------------------------------------------------------------------------

AttributionJudge = Callable[[str, str], float]  # (joint passage, claim) -> 1.0 if attributable else 0.0
PassageLookup = Callable[[str], Any]  # paper id -> str | {"title", "text"} | (title, text) | Paper | None

ATTRSCORE_MODEL = "osunlp/attrscore-flan-t5-xl"
AT_MOST_CITATIONS = 3
MIN_SENTENCE_CHARS = 50


class AttrScoreNLI:
    """AutoAIS judge of the official script: ``osunlp/attrscore-flan-t5-xl``.

    Loaded lazily through ``transformers`` (bfloat16, ``device_map="auto"``,
    slow tokenizer, greedy decoding of at most 10 tokens); a claim is
    supported iff the decoded label equals ``"Attributable"``.
    """

    def __init__(self, model_name: str = ATTRSCORE_MODEL, device_map: Optional[str] = "auto",
                 dtype: str = "bfloat16", max_new_tokens: int = 10):
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens)
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:  # pragma: no cover - needs torch, transformers and the model weights
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as err:
            raise ImportError(
                "AttrScoreNLI needs torch and transformers (pip install -e '.[embeddings]'); "
                "alternatively use LLMAttributionJudge."
            ) from err
        log.info("Loading %s ...", self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, torch_dtype=getattr(torch, self.dtype), device_map=self.device_map)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

    def __call__(self, passage: str, claim: str) -> float:  # pragma: no cover - needs model weights
        self._load()
        import torch

        ids = self._tokenizer(P.attribution_prompt(passage, claim), return_tensors="pt").input_ids
        ids = ids.to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(ids, max_new_tokens=self.max_new_tokens)
        label = self._tokenizer.decode(out[0], skip_special_tokens=True)
        return 1.0 if label == "Attributable" else 0.0


class LLMAttributionJudge:
    """Fallback attribution judge: the AttrScore prompt answered by an instruction-following LLM."""

    def __init__(self, judge: JudgeLike, model: Optional[str] = None):
        self.llm = as_judge_llm(judge)
        self.model = model

    def __call__(self, passage: str, claim: str) -> float:
        resp = self.llm.complete(P.attribution_request(passage, claim, model=self.model))
        return 1.0 if P.parse_attribution_label(resp.text) == "Attributable" else 0.0


_ABBREV_RE = re.compile(
    r"\b(e\.g|i\.e|et al|etc|vs|cf|Figs?|Eqs?|Sec|Secs|Tab|Ref|Refs|No|approx|Dr|Prof|Mr|Ms|resp)\.", re.IGNORECASE)
_SENT_BOUNDARY = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"'(\[]?[A-Z0-9#*-])")
_HOLD = "\u0000"
_NLTK_AVAILABLE: Optional[bool] = None


def _regex_sentences(text: str) -> List[str]:
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + _HOLD, text)
    return [part.replace(_HOLD, ".").strip() for part in _SENT_BOUNDARY.split(protected) if part.strip()]


def split_sentences(text: str) -> List[str]:
    """Sentence split with ``nltk.sent_tokenize`` (as the official script), else a regex fallback.

    Install ``nltk`` and its ``punkt`` data for official-protocol numbers;
    a warning is logged once when the fallback is used, and
    :func:`sentence_splitter` / :attr:`CitationAttribution.splitter` record
    which splitter produced a result. The fallback splits after ``.!?``
    (optionally followed by a closing quote or bracket) and whitespace when
    the next sentence starts with an upper-case letter, a digit, or a
    bracket; it protects common abbreviations (``e.g.``, ``et al.``,
    ``Fig.``) and, like punkt, does not split at bare line breaks.
    """
    global _NLTK_AVAILABLE
    if _NLTK_AVAILABLE is not False:
        try:
            from nltk import sent_tokenize

            sents = list(sent_tokenize(text or ""))
            _NLTK_AVAILABLE = True
            return sents
        except (ImportError, LookupError) as err:
            log.warning("nltk.sent_tokenize unavailable (%s); statement-level citation scores use a regex "
                        "sentence splitter and may differ from the official script. Install nltk and run "
                        "nltk.download('punkt').", err)
            _NLTK_AVAILABLE = False
    return _regex_sentences(text or "")


def sentence_splitter() -> str:
    """``"nltk"`` or ``"regex"``: the splitter :func:`split_sentences` uses (probing nltk on first use)."""
    if _NLTK_AVAILABLE is None:
        split_sentences("")
    return "nltk" if _NLTK_AVAILABLE else "regex"


_NUM_CITE_RE = re.compile(r"\[\d+")
_REMOVE_CITE_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")
_GROUP_CITE_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")


def extract_citation_indices(sentence: str) -> List[int]:
    """Citation indices of a sentence exactly as the official ``re.findall(r"\\[\\d+", ...)``.

    Note that a grouped marker ``[1, 2]`` only yields ``1``; :func:`number_citations`
    therefore renders every citation in its own bracket.
    """
    return [int(m[1:]) for m in _NUM_CITE_RE.findall(sentence)]


def remove_citations(text: str) -> str:
    """Strip ``[n]`` / ``[n, m]`` markers (official ``remove_citations``)."""
    cleaned = _REMOVE_CITE_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned.replace(" .", ".").replace(" ,", ",")


def format_document(doc: Mapping[str, Any]) -> str:
    """``Title: <title>\\n<text>`` (official ``_format_document``)."""
    if "sent" in doc:
        return "Title: %s\n%s" % (doc.get("title", ""), doc["sent"])
    if "title" in doc:
        return "Title: %s\n%s" % (doc["title"], doc.get("text", ""))
    return str(doc.get("text", ""))


@dataclass
class CitationAttribution:
    """Statement-level citation recall / precision of one answer (fractions)."""

    recall: float
    precision: float
    n_sentences: int  # scored sentences (>= 50 characters)
    n_citations: int  # citations entering the precision denominator
    n_supported: int
    n_cited_papers: int
    sentences: List[Dict[str, Any]] = field(default_factory=list)
    query_id: str = ""
    splitter: str = "nltk"  # sentence splitter used ("nltk" as the official script, or "regex")


def citation_attribution(
    output: str,
    ctxs: Sequence[Mapping[str, Any]],
    judge: AttributionJudge,
    *,
    at_most_citations: Optional[int] = AT_MOST_CITATIONS,
    min_sentence_chars: int = MIN_SENTENCE_CHARS,
    query_id: str = "",
) -> Optional[CitationAttribution]:
    """Official AutoAIS citation recall / precision of one output with 0-based ``[i]`` markers.

    Returns ``None`` when the output has no sentence (the official script
    skips such items). ``judge(passage, claim)`` returns 1.0 iff the passage
    supports the claim; identical calls are memoized.
    """
    memo: Dict[Tuple[str, str], float] = {}

    def nli(passage: str, claim: str) -> float:
        key = (passage, claim)
        if key not in memo:
            memo[key] = 1.0 if float(judge(passage, claim)) >= 0.5 else 0.0
        return memo[key]

    sents = split_sentences(output or "")
    if not sents:
        return None
    targets = [remove_citations(s).strip() for s in sents]
    # Official ``extract_citations``: distinct ids of [n] / [n, m] markers (reported only).
    cited_ids = {int(n) for m in _GROUP_CITE_RE.findall(output or "") for n in m.split(",")}
    entail = 0.0
    entail_prec = 0.0
    total_citations = 0
    total_sents = 0
    previous: Optional[List[int]] = None
    log_rows: List[Dict[str, Any]] = []
    for sid, sent in enumerate(sents):
        if len(sent) < min_sentence_chars:
            continue
        total_sents += 1
        target = targets[sid]
        joint = -1.0
        ref = extract_citation_indices(sent)
        inherited = False
        if not ref and previous is not None:
            ref, inherited = list(previous), True
        joint_passage = ""
        if not ref:
            joint = 0.0
        elif any(r >= len(ctxs) for r in ref):
            joint = 0.0
        else:
            previous = list(ref)
            if at_most_citations is not None:
                ref = ref[:at_most_citations]
            total_citations += len(ref)
            joint_passage = "\n".join(format_document(ctxs[i]) for i in ref if i >= 0)
        if joint == -1.0:
            joint = nli(joint_passage, target)
        entail += joint
        necessary = 0
        if joint and len(ref) > 1:
            for pid in ref:
                if nli(format_document(ctxs[pid]), target):
                    necessary += 1
                    continue
                rest = list(ref)
                rest.remove(pid)
                if not nli("\n".join(format_document(ctxs[i]) for i in rest), target):
                    necessary += 1  # the citation is needed for the joint support
        else:
            necessary = int(joint)
        entail_prec += necessary
        log_rows.append({"sentence": sent, "claim": target, "citations": ref, "inherited": inherited,
                         "supported": bool(joint), "precise_citations": necessary})
    recall = entail / total_sents if total_sents > 0 else 0.0
    precision = entail_prec / total_citations if total_citations > 0 else 0.0
    return CitationAttribution(recall=recall, precision=precision, n_sentences=total_sents,
                               n_citations=total_citations, n_supported=int(entail),
                               n_cited_papers=len(cited_ids), sentences=log_rows, query_id=query_id,
                               splitter=sentence_splitter())


def _as_ctx(value: Any) -> Optional[Dict[str, str]]:
    """Normalise a passage to an official ctx: ``{"title", "text"}``, or ``{"text"}`` when untitled.

    A ``"sent"`` field is stored as ``"text"``. The same dict is formatted
    in-process (:func:`format_document`) and written by
    :func:`~scholarpath.evaluation.official.export_citation_eval`, so both
    paths give the attribution judge identical passages.
    """
    if value is None:
        return None
    if isinstance(value, Paper):
        return {"title": value.title, "text": value.abstract}
    if isinstance(value, str):
        return {"text": value}
    if isinstance(value, Mapping):
        body = value.get("text", value.get("sent"))
        if body is None:
            return None
        ctx = {"title": str(value["title"])} if value.get("title") is not None else {}
        ctx["text"] = str(body)
        return ctx
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return {"title": str(value[0]), "text": str(value[1])}
    raise TypeError(f"Unsupported passage type {type(value).__name__}")


def answer_passages(answer: Answer, papers: Optional[PaperLookup] = None) -> PassageLookup:
    """Passage lookup for an answer: the context text the system saw, else title + abstract.

    ``answer.metadata["passages"]`` (``{paper_id: text | {"title", "text"}}``)
    takes precedence -- it should hold the evidence actually packed into the
    system's context; otherwise the corpus abstract of the paper is used.
    """
    stored = answer.metadata.get("passages") if isinstance(answer.metadata, Mapping) else None

    def lookup(pid: str) -> Any:
        if isinstance(stored, Mapping) and pid in stored:
            return stored[pid]
        if papers is not None:
            return papers.get(pid)
        return None

    return lookup


# Markers after sentence-final punctuation, moved before it -- only at a sentence boundary
# (end of text, a line break, or a capitalised next sentence), never after "et al." in mid-sentence.
_TRAILING_CITES_RE = re.compile(r"([.!?])((?:[ \t]*\[\d+\])+)(?=[ \t]*(?:$|\n)|\s+[\"'(\[]?[A-Z0-9])")


def number_citations(
    answer: AnswerLike,
    passage_fn: PassageLookup,
    attach_trailing: bool = True,
    valid_ids: Optional[IdSet] = None,
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    """Convert ``[paper_id]`` citations to the official 0-based ``[i]`` format.

    Returns ``(output, ctxs, paper_ids)``. A whitespace-free token of a
    bracket marker (``[id]``, ``[id1, id2]``, ``[id, Fig. 3]``) is a citation
    only if it is a known paper id: a structured citation of the
    :class:`Answer` items, a key of ``answer.metadata["passages"]``, a member
    of ``valid_ids``, or an id ``passage_fn`` resolves. Other bracketed text
    (``[CLS]``, ``[MASK]``, the interval ``[0, 1]``, ``[Smith 2020]``) is kept
    as text -- in parentheses when it starts with a digit, so that the
    official ``\\[\\d+`` regex does not read it as a citation -- and the
    non-id part of a mixed marker is kept the same way (``[id, Fig. 3]`` ->
    ``[0] (Fig. 3)``).

    Cited papers with a passage get indices ``0..K-1`` in order of first
    citation (``ctxs[i]`` is their passage, ``paper_ids[i]`` their id); known
    ids without a passage get out-of-range indices, which the official script
    scores as unsupported. Every citation is rendered in its own bracket
    (``[0][2]``) because the official regex only reads the first id of
    ``[0, 2]``. With ``attach_trailing``, markers placed after the
    sentence-final punctuation (``... finding. [0] Next ...``) are moved before
    it so that they stay with their sentence after splitting; markers after
    a mid-sentence period (``Smith et al. [0] showed``) stay where they are.
    """
    text = answer_text(answer)
    known = set(answer.cited_papers) if isinstance(answer, Answer) else set()
    stored = answer.metadata.get("passages") if isinstance(answer, Answer) else None
    if isinstance(stored, Mapping):
        known.update(str(k) for k in stored)
    in_valid = _id_predicate(valid_ids)
    passages: Dict[str, Optional[Dict[str, str]]] = {}

    def passage(pid: str) -> Optional[Dict[str, str]]:
        if pid not in passages:
            passages[pid] = _as_ctx(passage_fn(pid))
        return passages[pid]

    def is_citation(tok: str) -> bool:
        return tok in known or (in_valid is not None and in_valid(tok)) or passage(tok) is not None

    def split(content: str) -> Tuple[List[str], List[str]]:
        toks = _split_marker(content)
        cites = [t for t in toks if not _WS_RE.search(t) and is_citation(t)]
        return cites, [t for t in toks if t not in cites]

    order: List[str] = unique(t for m in _ID_MARKER_RE.findall(text) for t in split(m)[0])
    ctxs: List[Dict[str, str]] = []
    index: Dict[str, int] = {}
    unresolved: List[str] = []
    for pid in order:
        ctx = passage(pid)
        if ctx is None:
            unresolved.append(pid)
            continue
        index[pid] = len(ctxs)
        ctxs.append(ctx)
    for j, pid in enumerate(unresolved):
        index[pid] = len(ctxs) + j

    def as_text(content: str) -> str:
        return f"({content})" if content[:1].isdigit() else f"[{content}]"

    def repl(m: "re.Match[str]") -> str:
        cites, rest = split(m.group(1))
        if not cites:
            return as_text(m.group(1))
        marks = "".join(f"[{index[t]}]" for t in cites)
        return marks + (f" ({', '.join(rest)})" if rest else "")

    output = _ID_MARKER_RE.sub(repl, text)
    if attach_trailing:
        output = _TRAILING_CITES_RE.sub(lambda m: m.group(2).rstrip() + m.group(1), output)
    paper_ids = [pid for pid in order if pid in index and index[pid] < len(ctxs)]
    return output, ctxs, paper_ids


def answer_citation_attribution(
    answer: Answer,
    passage_fn: PassageLookup,
    judge: AttributionJudge,
    *,
    at_most_citations: Optional[int] = AT_MOST_CITATIONS,
    min_sentence_chars: int = MIN_SENTENCE_CHARS,
) -> Optional[CitationAttribution]:
    """:func:`citation_attribution` of an :class:`Answer` citing paper ids."""
    output, ctxs, _ = number_citations(answer, passage_fn)
    return citation_attribution(output, ctxs, judge, at_most_citations=at_most_citations,
                                min_sentence_chars=min_sentence_chars, query_id=answer.query_id)


def aggregate_citation_attribution(results: Sequence[Optional[CitationAttribution]]) -> Dict[str, Optional[float]]:
    """``citation_prec`` / ``citation_rec`` = means over answers (x100); ``Cite`` = 2PR/(P+R)."""
    scored = [r for r in results if r is not None]
    p = _mean(r.precision for r in scored)
    r_ = _mean(r.recall for r in scored)
    return {
        "citation_prec": _pct(p),
        "citation_rec": _pct(r_),
        "Cite": _pct(f1_score(p, r_)) if p is not None and r_ is not None else None,
        "n": float(len(scored)),
    }


# ---------------------------------------------------------------------------
# (e) Prometheus-2 absolute grading (ScholarQA-Multi)
# ---------------------------------------------------------------------------

PROMETHEUS_ASPECTS = ("organization", "coverage", "relevance")


@dataclass
class PrometheusResult:
    query_id: str
    scores: Dict[str, Optional[int]] = field(default_factory=dict)  # aspect -> 1..5 (None if unparsable)
    feedback: Dict[str, str] = field(default_factory=dict)

    def mean(self) -> Optional[float]:
        return _mean(s for s in self.scores.values() if s is not None)


def prometheus_to_100(score: Optional[float]) -> Optional[float]:
    """Map a 1-5 Prometheus score to 0-100 via (s - 1) / 4 x 100."""
    return None if score is None else 100.0 * normalize_5scale(score)


def prometheus_grade(
    answer: AnswerLike,
    query: Query,
    judge: JudgeLike,
    *,
    aspects: Sequence[str] = PROMETHEUS_ASPECTS,
    instruction: str = P.PROMETHEUS_INSTRUCTION,
    reference: Optional[str] = None,
    use_reference: bool = True,
    model: Optional[str] = None,
) -> PrometheusResult:
    """Prometheus-2 absolute grading of one answer for each aspect (temperature 0).

    The reference answer (score 5) is ``reference`` or
    ``query.reference_answer`` (the ScholarQA-Multi expert answer); without
    one the no-reference template is used. The response is the answer text
    without a reference list (the official script appends none).
    """
    llm = as_judge_llm(judge)
    ref = reference if reference is not None else query.reference_answer
    if use_reference and ref is None:
        log.warning("Query %s has no reference answer; using the no-reference Prometheus template", query.query_id)
    ref = ref if use_reference else None
    instr = f"{instruction} {query.text}"
    response = answer_text(answer)
    out = PrometheusResult(query_id=query.query_id)
    for aspect in aspects:
        resp = llm.complete(P.prometheus_request(instr, response, ref, aspect, model=model))
        out.scores[aspect] = P.parse_prometheus_score(resp.text)
        text = resp.text
        out.feedback[aspect] = text[text.find("Feedback: ") + len("Feedback: "):] if "Feedback: " in text else text
    return out


def aggregate_prometheus(
    results: Sequence[PrometheusResult], aspects: Optional[Sequence[str]] = None
) -> Dict[str, Optional[float]]:
    """Per-aspect means of valid scores (1-5), their average, and the 0-100 mapping.

    Mirrors ``extract_score``: unparsable scores are excluded from an
    aspect's mean and ``average`` is the mean of the aspect means. ``Corr``
    is ``average`` mapped to 0-100 (the Multi-subset correctness).
    """
    aspects = list(aspects or unique(a for r in results for a in r.scores))
    out: Dict[str, Optional[float]] = {}
    means = []
    for a in aspects:
        m = _mean(r.scores[a] for r in results if r.scores.get(a) is not None)
        out[a] = m
        out[f"{a}_100"] = prometheus_to_100(m)
        if m is not None:
            means.append(m)
    avg = _mean(means)
    out["average"] = avg
    out["Corr"] = prometheus_to_100(avg)
    out["n"] = float(len(results))
    out["n_unparsed"] = float(sum(1 for r in results for a in aspects if r.scores.get(a) is None))
    return out


# ---------------------------------------------------------------------------
# (f) ScholarQABench = CS + Multi weighted by query counts
# ---------------------------------------------------------------------------


def scholarqabench_combined(
    cs_scores: Mapping[str, Optional[float]],
    multi_scores: Mapping[str, Optional[float]],
    metrics: Sequence[str] = ("Corr", "Cite"),
) -> Dict[str, Optional[float]]:
    """Average the ScholarQA-CS and ScholarQA-Multi subsets weighted by their numbers of queries.

    Each input is a summary with ``"n"`` (number of scored queries) and the
    metric keys (``Corr``: CS rubric score / Multi Prometheus score, both
    0-100; ``Cite``: citation F1). A metric missing from one subset is
    taken from the other alone.
    """
    n_cs = float(cs_scores.get("n") or 0.0)
    n_multi = float(multi_scores.get("n") or 0.0)
    out: Dict[str, Optional[float]] = {"n": n_cs + n_multi, "n_cs": n_cs, "n_multi": n_multi}
    for m in metrics:
        parts = [(n, s.get(m)) for n, s in ((n_cs, cs_scores), (n_multi, multi_scores))
                 if s.get(m) is not None and n > 0]
        total = sum(n for n, _ in parts)
        out[m] = sum(n * float(v) for n, v in parts) / total if total > 0 else None
    return out


# ---------------------------------------------------------------------------
# Benchmark drivers
# ---------------------------------------------------------------------------


@dataclass
class QAEvaluation:
    rows: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Optional[float]] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)


def _map(llm: LLMClient, fn: Callable[[Any], Any], items: Sequence[Any], max_workers: Optional[int]) -> List[Any]:
    return llm.map(fn, items, max_workers=max_workers)


def evaluate_researchqa(
    answers: Mapping[str, Answer],
    queries: Sequence[Query],
    judge: JudgeLike,
    *,
    papers: Optional[PaperLookup] = None,
    references: bool = True,
    on_failure: str = "drop",
    model: Optional[str] = None,
    max_workers: Optional[int] = 8,
) -> QAEvaluation:
    """ResearchQA Rub (rubric coverage) and Cite (citation F1 vs. gold papers).

    Queries without an answer are skipped, as in the official script, and
    listed in ``missing``. ``papers`` (the corpus) also restricts the
    text-marker fallback of Cite to real paper ids.
    """
    llm = as_judge_llm(judge)
    todo = [q for q in queries if q.query_id in answers]
    ev = QAEvaluation(missing=[q.query_id for q in queries if q.query_id not in answers])
    coverages = _map(llm, lambda q: rubric_coverage(answers[q.query_id], q, llm, papers=papers,
                                                   references=references, on_failure=on_failure,
                                                   model=model), todo, max_workers)
    cites = [citation_f1(answers[q.query_id], q, valid_ids=papers) for q in todo]
    for q, cov, cite in zip(todo, coverages, cites):
        ev.rows.append({
            "query_id": q.query_id, "benchmark": "researchqa",
            "Rub": _pct(cov.coverage), "coverage_failed": cov.failed,
            "coverage_citation": _pct(_mean(cov.type_scores("citation"))),
            "Cite": 100.0 * cite.f1, "cite_P": 100.0 * cite.precision, "cite_R": 100.0 * cite.recall,
            "num_cited": cite.n_pred, "num_gold": cite.n_gold,
        })
    ev.summary = {**aggregate_coverage(coverages), **{k: v for k, v in aggregate_prf(cites).items() if k != "n"}}
    ev.summary["n_missing"] = float(len(ev.missing))
    return ev


def evaluate_scholarqa_cs(
    answers: Mapping[str, Answer],
    queries: Sequence[Query],
    judge: JudgeLike,
    *,
    attribution_judge: Optional[AttributionJudge] = None,
    papers: Optional[PaperLookup] = None,
    passage_fn: Optional[Callable[[Answer], PassageLookup]] = None,
    dedup_questions: bool = True,
    references: bool = True,
    model: Optional[str] = None,
    max_workers: Optional[int] = 8,
) -> QAEvaluation:
    """ScholarQA-CS Corr (rubric score) and, with an ``attribution_judge``, statement-level Cite.

    With ``dedup_questions`` only the first test case of a doubly annotated
    question is scored (rubric_eval.py without ``--agreement``). Answers are
    scored with their cited passages appended as in the official
    ``convert_answer_scholar_cs.py`` (:func:`render_cs_answer`;
    ``references=False`` disables it); passages come from ``passage_fn(answer)``
    or :func:`answer_passages` (``answer.metadata["passages"]``, else ``papers``).
    """
    llm = as_judge_llm(judge)
    seen = set()
    todo: List[Query] = []
    ev = QAEvaluation()
    for q in queries:
        if q.query_id not in answers:
            ev.missing.append(q.query_id)
            continue
        if dedup_questions and q.text in seen:
            continue
        seen.add(q.text)
        todo.append(q)

    def lookup(a: Answer) -> Optional[PassageLookup]:
        if passage_fn is not None:
            return passage_fn(a)
        return answer_passages(a, papers) if has_passage_source(a, papers, None) else None

    scores = _map(llm, lambda q: correctness(answers[q.query_id], q, llm, model=model, references=references,
                                             passage_fn=lookup(answers[q.query_id])), todo, max_workers)
    attributions: List[Optional[CitationAttribution]] = []
    if attribution_judge is not None:
        for q in todo:
            a = answers[q.query_id]
            attributions.append(answer_citation_attribution(a, lookup(a) or answer_passages(a), attribution_judge))
    for i, (q, s) in enumerate(zip(todo, scores)):
        row: Dict[str, Any] = {"query_id": q.query_id, "benchmark": "scholarqa_cs", "Corr": 100.0 * s.score,
                               "ann_score": 100.0 * s.ann_score}
        if attributions and attributions[i] is not None:
            row.update(citation_prec=100.0 * attributions[i].precision, citation_rec=100.0 * attributions[i].recall,
                       splitter=attributions[i].splitter)
        ev.rows.append(row)
    ev.summary = aggregate_correctness(scores)
    if attributions:
        cite = aggregate_citation_attribution(attributions)
        ev.summary.update({k: v for k, v in cite.items() if k != "n"})
    return ev


def evaluate_scholarqa_multi(
    answers: Mapping[str, Answer],
    queries: Sequence[Query],
    judge: JudgeLike,
    *,
    attribution_judge: Optional[AttributionJudge] = None,
    papers: Optional[PaperLookup] = None,
    passage_fn: Optional[Callable[[Answer], PassageLookup]] = None,
    aspects: Sequence[str] = PROMETHEUS_ASPECTS,
    model: Optional[str] = None,
    max_workers: Optional[int] = 8,
) -> QAEvaluation:
    """ScholarQA-Multi Corr (Prometheus organization / coverage / relevance, 0-100) and Cite."""
    llm = as_judge_llm(judge)
    todo = [q for q in queries if q.query_id in answers]
    ev = QAEvaluation(missing=[q.query_id for q in queries if q.query_id not in answers])
    grades = _map(llm, lambda q: prometheus_grade(answers[q.query_id], q, llm, aspects=aspects, model=model),
                  todo, max_workers)
    attributions: List[Optional[CitationAttribution]] = []
    if attribution_judge is not None:
        for q in todo:
            a = answers[q.query_id]
            lookup = passage_fn(a) if passage_fn is not None else answer_passages(a, papers)
            attributions.append(answer_citation_attribution(a, lookup, attribution_judge))
    for i, (q, g) in enumerate(zip(todo, grades)):
        row: Dict[str, Any] = {"query_id": q.query_id, "benchmark": "scholarqa_multi",
                               "Corr": prometheus_to_100(g.mean()), **{a: g.scores.get(a) for a in aspects}}
        if attributions and attributions[i] is not None:
            row.update(citation_prec=100.0 * attributions[i].precision, citation_rec=100.0 * attributions[i].recall,
                       splitter=attributions[i].splitter)
        ev.rows.append(row)
    ev.summary = aggregate_prometheus(grades, aspects)
    if attributions:
        cite = aggregate_citation_attribution(attributions)
        ev.summary.update({k: v for k, v in cite.items() if k != "n"})
    return ev


__all__ = [
    "AT_MOST_CITATIONS",
    "ATTRSCORE_MODEL",
    "AttrScoreNLI",
    "AttributionJudge",
    "CSRubricConfig",
    "CSRubricProperty",
    "CSScore",
    "CitationAttribution",
    "CoverageResult",
    "LLMAttributionJudge",
    "PRF",
    "PROMETHEUS_ASPECTS",
    "PrometheusResult",
    "QAEvaluation",
    "RubricJudgement",
    "aggregate_citation_attribution",
    "aggregate_correctness",
    "aggregate_coverage",
    "aggregate_prf",
    "aggregate_prometheus",
    "answer_citation_attribution",
    "answer_passages",
    "answer_text",
    "citation_attribution",
    "citation_f1",
    "citation_prf",
    "cited_papers",
    "correctness",
    "evaluate_researchqa",
    "evaluate_scholarqa_cs",
    "evaluate_scholarqa_multi",
    "extract_citation_indices",
    "f1_score",
    "format_document",
    "format_reference",
    "has_passage_source",
    "length_score",
    "normalize_5scale",
    "number_citations",
    "prometheus_grade",
    "prometheus_to_100",
    "remove_citations",
    "render_answer",
    "render_cs_answer",
    "rubric_coverage",
    "rubric_item_types",
    "scholarqabench_combined",
    "sentence_splitter",
    "split_sentences",
]
