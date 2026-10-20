"""Held-out hypothesis query metrics: Disc (Hit@10) and LLM-judge scores (Table 4, Fig. 6).

Section 5.3: "On held-out hypothesis queries we report discovery rate (Disc,
the percentage of queries whose ground-truth held-out counterpart is
uncovered among the proposed hypotheses) and an LLM-judged evaluation
averaged across three state-of-the-art LLMs (GPT-6.0-Sol, Opus-5.5,
Grok-4.7) scoring each proposed idea on Feasibility (Feas.), Novelty (Nov.),
and Technical Soundness (Sound.) on a 0-100 scale."

Appendix C.3: "Disc counts a query as solved when the proposed candidates of
the answer realize the held-out counterpart of its expected answer record
(Appendix C.1), and three judges score each proposed idea and its rationale
for Feasibility, Novelty, and Technical Soundness on a 0-100 scale; Table 4
reports the average over the three judges."

The expected answer record of a held-out query (App. C.1) is "the held-out
counterpart concept and a one-sentence rationale from the source paper"; it
is read from ``query.metadata["heldout_counterpart"]`` /
``query.metadata["rationale"]`` (falling back to ``query.reference_answer``).

* :func:`discovery_hit` -- the proposed ideas are the answer items labelled
  ``hypothesis`` (or carrying a ``candidate``), in answer order, de-duplicated
  by candidate name and cut at k = 10 (Table 4 column "Hit@10"). An exact
  match of normalized names (or of a listed alias) short-circuits; otherwise
  the judge decides whether a candidate realizes the counterpart (majority
  vote when a multi-judge panel is given).
* :func:`judge_hypotheses` -- every judge scores every proposed idea (with its
  rationale) on the three criteria; scores are averaged over ideas, then over
  judges; ``Avg.`` is the mean of the three criteria.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from scholarpath.evaluation.judges import Judge, JudgeLike, as_panel
from scholarpath.llm import JSONParseError
from scholarpath.prompts import evaluation as P
from scholarpath.text import normalize_title
from scholarpath.types import Answer, AnswerItem, Query

log = logging.getLogger(__name__)

HIT_K = 10
CRITERIA = P.HYPOTHESIS_CRITERIA  # ("feasibility", "novelty", "soundness")
TABLE4_COLUMNS = {"feasibility": "Feas.", "novelty": "Nov.", "soundness": "Sound."}


@dataclass
class ProposedIdea:
    """One proposed hypothesis of an answer, in rank order."""

    rank: int  # 1-based
    name: str  # candidate concept (or the item text when no candidate is given)
    text: str
    rationale: str = ""
    citations: List[str] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return {"name": self.name, "text": self.text, "rationale": self.rationale}


@dataclass
class ExpectedRecord:
    """Expected answer record of a held-out hypothesis query (Appendix C.1)."""

    counterpart: str
    rationale: str = ""
    aliases: List[str] = field(default_factory=list)
    known_endpoint: Optional[str] = None
    setting: Optional[str] = None


def is_proposal(item: AnswerItem) -> bool:
    """An answer item proposing a hypothesis (status ``hypothesis`` or a ``candidate``)."""
    return item.status == "hypothesis" or bool(item.candidate)


def proposed_ideas(answer: Answer, k: Optional[int] = HIT_K) -> List[ProposedIdea]:
    """Top-``k`` distinct proposed ideas of an answer, in the answer's ranking order (``k=None``: all)."""
    if k is not None and k <= 0:
        return []
    ideas: List[ProposedIdea] = []
    seen = set()
    for item in answer.items:
        if not is_proposal(item):
            continue
        name = (item.candidate or item.text or "").strip()
        key = normalize_title(name)
        if not key or key in seen:
            continue
        seen.add(key)
        ideas.append(ProposedIdea(rank=len(ideas) + 1, name=name, text=item.text or name,
                                  rationale=item.rationale or "", citations=list(item.citations)))
        if k is not None and len(ideas) >= k:
            break
    return ideas


def expected_record(query: Query) -> ExpectedRecord:
    """Read the held-out counterpart, its rationale and aliases from a query."""
    md = query.metadata or {}
    counterpart = md.get("heldout_counterpart") or md.get("counterpart") or query.reference_answer
    if not counterpart:
        raise ValueError(f"Query {query.query_id} has no held-out counterpart (metadata['heldout_counterpart'] "
                         "or reference_answer)")
    aliases = md.get("heldout_aliases") or md.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    return ExpectedRecord(
        counterpart=str(counterpart),
        rationale=str(md.get("rationale") or md.get("heldout_rationale") or ""),
        aliases=[str(a) for a in aliases],
        known_endpoint=md.get("known_endpoint"),
        setting=md.get("setting"),
    )


# ---------------------------------------------------------------------------
# Disc (Hit@10)
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryResult:
    query_id: str
    hit: bool
    matched_rank: Optional[int] = None  # 1-based rank of the first realizing idea
    method: str = "none"  # "exact" | "judge" | "none"
    candidates: List[str] = field(default_factory=list)
    votes: Dict[str, List[int]] = field(default_factory=dict)  # judge -> matched ranks (answered judges)
    counterpart: str = ""
    failed_judges: List[str] = field(default_factory=list)  # judges whose verdict could not be parsed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _judge_matches(
    judge: Judge, query: Query, record: ExpectedRecord, ideas: Sequence[ProposedIdea]
) -> Optional[List[int]]:
    """1-based ranks of the ideas ``judge`` says realize the counterpart; ``None`` if unparsable."""
    request = P.hypothesis_match_request(
        query.text, record.counterpart, record.rationale, [i.to_payload() for i in ideas],
        known_endpoint=record.known_endpoint, setting=record.setting,
    )
    try:
        obj, _ = judge.llm.complete_json(request, expect=dict)
    except JSONParseError as err:
        log.warning("Unparsable match judgement from %s for %s: %s", judge.name, query.query_id, err)
        return None
    raw = obj.get("matches")
    if isinstance(raw, (int, str)):
        raw = [raw]
    if not isinstance(raw, list):
        log.warning("Match judgement from %s for %s has no 'matches' list: %r", judge.name, query.query_id, obj)
        return None
    ranks = []
    for r in raw:
        try:
            r = int(r)
        except (TypeError, ValueError):
            continue
        if 1 <= r <= len(ideas):
            ranks.append(r)
    return sorted(set(ranks))


def discovery_hit(answer: Answer, query: Query, judge: Optional[JudgeLike] = None, k: int = HIT_K) -> DiscoveryResult:
    """Is the held-out counterpart realized among the top-``k`` proposed ideas?

    An exact normalized-name match against the counterpart (or an alias)
    short-circuits without any judge call. Otherwise each judge of ``judge``
    (a client, a :class:`Judge` or a :class:`JudgePanel`) lists the ideas that
    realize the counterpart, and an idea counts when a strict majority of the
    judges that answered select it; a judge whose verdict cannot be parsed
    (after the client's JSON retries) abstains and is listed in
    ``failed_judges`` instead of voting "no match". With ``judge=None`` only
    exact matches count.
    """
    record = expected_record(query)
    ideas = proposed_ideas(answer, k)
    out = DiscoveryResult(query_id=query.query_id, hit=False, candidates=[i.name for i in ideas],
                          counterpart=record.counterpart)
    targets = {normalize_title(t) for t in [record.counterpart, *record.aliases] if normalize_title(t)}
    for idea in ideas:
        if normalize_title(idea.name) in targets:
            out.hit, out.matched_rank, out.method = True, idea.rank, "exact"
            return out
    if not ideas or judge is None:
        return out
    panel = as_panel(judge)
    verdicts = panel.map(lambda j: _judge_matches(j, query, record, ideas))
    out.votes = {name: ranks for name, ranks in verdicts.items() if ranks is not None}
    out.failed_judges = [name for name, ranks in verdicts.items() if ranks is None]
    if not out.votes:
        log.warning("No judge returned a usable Disc verdict for %s; counted as not discovered", query.query_id)
        return out
    need = len(out.votes) // 2 + 1
    for idea in ideas:
        if sum(idea.rank in ranks for ranks in out.votes.values()) >= need:
            out.hit, out.matched_rank, out.method = True, idea.rank, "judge"
            break
    return out


def discovery_rate(results: Sequence[DiscoveryResult]) -> Optional[float]:
    """Disc in percent: share of queries whose counterpart was discovered."""
    if not results:
        return None
    return 100.0 * sum(1 for r in results if r.hit) / len(results)


# ---------------------------------------------------------------------------
# LLM-judge scores: Feasibility / Novelty / Technical Soundness (0-100)
# ---------------------------------------------------------------------------


@dataclass
class HypothesisJudgement:
    """Judge scores of one answer: per judge (averaged over ideas) and overall."""

    query_id: str
    n_ideas: int
    per_judge: Dict[str, Dict[str, float]] = field(default_factory=dict)  # answered judges only
    per_idea: Dict[str, List[Optional[Dict[str, float]]]] = field(default_factory=dict)  # judge -> idea scores
    scores: Dict[str, float] = field(default_factory=dict)  # criterion -> mean over answered judges
    failed_judges: List[str] = field(default_factory=list)  # judges with no parsable score for any idea

    @property
    def scored(self) -> bool:
        """False when no judge returned a usable score (the query is left out of the aggregate)."""
        return bool(self.scores)

    @property
    def avg(self) -> Optional[float]:
        """Table 4 "Avg." -- mean of the three criteria (``None`` if no judge answered)."""
        if not self.scored:
            return None
        return sum(self.scores[c] for c in CRITERIA) / len(CRITERIA)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["avg"] = self.avg
        return d


def _score_idea(
    judge: Judge, query: Query, idea: ProposedIdea, titles: Mapping[str, str]
) -> Optional[Dict[str, float]]:
    evidence = [titles.get(c, c) for c in idea.citations] if titles else []
    statement = f"{idea.name}: {idea.text}" if idea.text != idea.name else idea.name
    request = P.hypothesis_score_request(query.text, statement, idea.rationale, evidence=evidence,
                                         judge_name=judge.name)
    try:
        obj, _ = judge.llm.complete_json(request, expect=dict)
    except JSONParseError as err:
        log.warning("Unparsable idea score from %s for %s: %s", judge.name, query.query_id, err)
        return None
    scores = P.parse_hypothesis_scores(obj)
    if scores is None:
        log.warning("Incomplete idea score from %s for %s: %r", judge.name, query.query_id, obj)
    return scores


def judge_hypotheses(
    answer: Answer,
    query: Query,
    judges: JudgeLike,
    k: int = HIT_K,
    empty_score: float = 0.0,
    paper_titles: Optional[Mapping[str, str]] = None,
) -> HypothesisJudgement:
    """Score each proposed idea with every judge; average over ideas, then over judges.

    Ideas are the top-``k`` proposals of :func:`proposed_ideas`. An answer
    without proposals receives ``empty_score`` on every criterion (the system
    proposed nothing). Ideas whose judgement cannot be parsed are left out of
    that judge's mean; a judge with no parsable score for any idea is a judge
    failure, not a property of the answer: it is left out of the average over
    judges and listed in ``failed_judges`` (if every judge fails, ``scores``
    is empty and :func:`aggregate_hypothesis` skips the query). Transport
    errors are raised. ``paper_titles`` (paper id -> title) lists the cited
    papers to the judge.
    """
    panel = as_panel(judges)
    ideas = proposed_ideas(answer, k)
    out = HypothesisJudgement(query_id=query.query_id, n_ideas=len(ideas))
    if not ideas:
        out.per_judge = {name: {c: float(empty_score) for c in CRITERIA} for name in panel.names}
        out.scores = {c: float(empty_score) for c in CRITERIA}
        return out
    titles = dict(paper_titles or {})
    per_idea = panel.map(lambda j: [_score_idea(j, query, idea, titles) for idea in ideas])
    out.per_idea = per_idea
    for name, scored in per_idea.items():
        valid = [s for s in scored if s is not None]
        if not valid:
            out.failed_judges.append(name)
            continue
        out.per_judge[name] = {c: sum(s[c] for s in valid) / len(valid) for c in CRITERIA}
    if out.failed_judges:
        log.warning("Judges %s returned no usable score for %s; averaging over %d judge(s)",
                    out.failed_judges, query.query_id, len(out.per_judge))
    if out.per_judge:
        out.scores = {c: sum(v[c] for v in out.per_judge.values()) / len(out.per_judge) for c in CRITERIA}
    return out


def aggregate_hypothesis(
    discovery: Sequence[DiscoveryResult] = (),
    judgements: Sequence[HypothesisJudgement] = (),
) -> Dict[str, Optional[float]]:
    """Table 4 row: ``Hit@10`` (= Disc, %), ``Feas.``, ``Nov.``, ``Sound.``, ``Avg.`` (0-100).

    Judge scores are means over queries of the per-query (judge-averaged)
    scores; per-judge means are reported as ``<judge>.<criterion>``. Queries
    that no judge could score are left out (``n_unscored``); judge failures
    are counted in ``judge_failures`` (Disc: abstentions, scores: failed
    judge x query pairs).
    """
    out: Dict[str, Optional[float]] = {"Hit@10": discovery_rate(discovery), "n_disc": float(len(discovery))}
    if discovery:
        out["disc_judge_failures"] = float(sum(len(d.failed_judges) for d in discovery))
    if judgements:
        scored = [j for j in judgements if j.scored]
        n = len(scored)
        for c in CRITERIA:
            out[TABLE4_COLUMNS[c]] = sum(j.scores[c] for j in scored) / n if n else None
        out["Avg."] = sum(j.avg for j in scored) / n if n else None  # type: ignore[misc]
        judges = sorted({name for j in scored for name in j.per_judge})
        for name in judges:
            vals = [j.per_judge[name] for j in scored if name in j.per_judge]
            for c in CRITERIA:
                out[f"{name}.{c}"] = sum(v[c] for v in vals) / len(vals)
        out["n_judged"] = float(n)
        out["n_unscored"] = float(len(judgements) - n)
        out["judge_failures"] = float(sum(len(j.failed_judges) for j in judgements))
        out["ideas_per_query"] = sum(j.n_ideas for j in scored) / n if n else None
    return out


def evaluate_hypotheses(
    answers: Mapping[str, Answer],
    queries: Sequence[Query],
    disc_judge: Optional[JudgeLike],
    score_judges: Optional[JudgeLike] = None,
    k: int = HIT_K,
    paper_titles: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Disc and judge scores for a set of held-out hypothesis queries.

    A query without an answer counts as not discovered and receives zero
    judge scores (the system produced nothing). Returns
    ``{"rows": [...], "summary": {...}}``.
    """
    discovery: List[DiscoveryResult] = []
    judgements: List[HypothesisJudgement] = []
    rows: List[Dict[str, Any]] = []
    for q in queries:
        ans = answers.get(q.query_id) or Answer(query_id=q.query_id, text="")
        d = discovery_hit(ans, q, disc_judge, k=k)
        discovery.append(d)
        row: Dict[str, Any] = {"query_id": q.query_id, "benchmark": q.benchmark, "hit": d.hit,
                               "matched_rank": d.matched_rank, "match_method": d.method,
                               "counterpart": d.counterpart, "candidates": d.candidates,
                               "disc_failed_judges": d.failed_judges, "missing": q.query_id not in answers}
        if score_judges is not None:
            j = judge_hypotheses(ans, q, score_judges, k=k, paper_titles=paper_titles)
            judgements.append(j)
            row.update({TABLE4_COLUMNS[c]: j.scores.get(c) for c in CRITERIA})
            row["Avg."] = j.avg
            row["failed_judges"] = j.failed_judges
        rows.append(row)
    return {"rows": rows, "summary": aggregate_hypothesis(discovery, judgements)}


__all__ = [
    "CRITERIA",
    "DiscoveryResult",
    "ExpectedRecord",
    "HIT_K",
    "HypothesisJudgement",
    "ProposedIdea",
    "TABLE4_COLUMNS",
    "aggregate_hypothesis",
    "discovery_hit",
    "discovery_rate",
    "evaluate_hypotheses",
    "expected_record",
    "is_proposal",
    "judge_hypotheses",
    "proposed_ideas",
]
