"""Held-Out Hypothesis benchmark (Sec. 5.3; App. C.1).

For hypothesis queries no existing benchmark provides ground-truth rationales
and indirect evidence for unstudied research directions, so the paper builds
a *Held-Out Hypothesis* benchmark of 47 queries over arXiv-Subfields:
24 "new problem / existing method" and 23 "new method / existing problem".
This module implements every step of App. C.1:

1. **Targets** (:func:`select_targets`). Targets are sampled on the dual
   taxonomy built over the *complete* corpus before any removal. A target is
   an emerging cross-domain idea combining a *new* endpoint -- a concept whose
   extent holds at most four papers -- with an *existing* endpoint of another
   domain: (1) a new problem proposed in a target domain and addressed by
   adapting an existing method from another domain, or (2) a new method
   proposed to solve an existing problem in a new domain. Larger concepts are
   excluded because removing them would erase a substantial part of the
   literature. Sampled targets have pairwise disjoint supporting papers and
   share no endpoint, so removing one target never removes the evidence from
   which another could be predicted. "Existing endpoint of another domain"
   is checked against the evidence that *survives the corpus reduction*: the
   endpoint's papers that remain in the reduced corpus and predate the source
   must be dominated by a domain other than the target domain and include at
   least ``min_existing_papers`` papers outside it (:class:`_EvidenceGuard`).
2. **Corpus reduction** (:func:`reduce_corpus`). All supporting papers of the
   sampled targets, together with the papers published after them, are
   removed before any index is built; every compared system indexes and
   searches the same reduced corpus. The paper's 24 / 23 queries are the
   counts *after* validation, so :func:`build_heldout_benchmark` can
   oversample targets (``oversample``), and :func:`apply_annotations` keeps the
   first validated queries per setting in the seeded selection order
   (``quota``).
3. **Queries and expected answers** (:func:`generate_queries`). The source of
   a target is the earliest of its supporting papers. An LLM receives the
   source's title and abstract, the setting, and the label and definition of
   the new endpoint, writes a query in the voice of a researcher who knows
   the new endpoint but not the answer, and extracts the expected answer
   record (held-out counterpart + one-sentence rationale).
4. **Validation and leakage control** (:func:`check_leakage`,
   :func:`supporting_evidence`, :func:`export_annotation_sheet`,
   :func:`apply_annotations`). Two annotators check that the query does not
   reveal its answer, that the source paper realizes the expected answer, and
   that the reduced corpus still contains related evidence (e.g. applications
   of the counterpart to neighbouring problems or methods), so that the
   answer is predictable rather than a guess. Queries failing any condition
   are discarded; disagreements are resolved by discussion.

:func:`build_heldout_benchmark` runs steps 1-4 (up to the annotation sheet),
and :func:`save_heldout_benchmark` / :func:`load_heldout_benchmark` read and
write ``data/processed/heldout/``.

Domains are primary arXiv categories (``Paper.primary_category``, else the
first listed category as in App. C.2, or a caller-supplied mapping).
"""

from __future__ import annotations

import bisect
import csv
import dataclasses
import datetime as _dt
import json
import logging
import math
import numbers
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from scholarpath.data.corpus import Corpus, load_queries, read_json, save_queries, write_json
from scholarpath.llm.base import LLMClient
from scholarpath.llm.parsing import JSONParseError
from scholarpath.prompts.benchmark import (
    COUNTERPART_FACET,
    NEW_FACET,
    NEW_METHOD,
    NEW_PROBLEM,
    SETTINGS,
    content_terms,
    heldout_query_request,
    parse_heldout_query,
)
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.text import STOPWORDS, tokenize
from scholarpath.types import HYPOTHESIS_DISCOVERY, Paper, Query

log = logging.getLogger("scholarpath.heldout")

PathLike = Union[str, Path]
DEFAULT_HELDOUT_DIR = Path("data") / "processed" / "heldout"
HELDOUT_FILES = {
    "queries": "queries.jsonl",
    "removed": "removed_papers.json",
    "reduced": "reduced_corpus_ids.json",
    "targets": "targets.json",
    "stats": "stats.json",
}
# Annotation conditions of App. C.1 (each must hold for a query to be kept).
CONDITIONS = ("no_reveal", "source_realizes", "predictable")
CONDITION_TEXT = {
    "no_reveal": "the query does not reveal its answer",
    "source_realizes": "the source paper realizes the expected answer",
    "predictable": "the reduced corpus still contains related evidence, so the answer is predictable rather than a guess",
}
ANNOTATORS = ("a1", "a2")
# Corpus reduction modes (see reduce_corpus).
REDUCTIONS = ("global", "per_target")

CategoryFn = Union[Callable[[str], Optional[str]], Mapping[str, Optional[str]], None]


# ---------------------------------------------------------------------------
# Paper records and dates
# ---------------------------------------------------------------------------

def _papers(papers: Any) -> Dict[str, Any]:
    """``paper_id -> record`` from a dict, a :class:`~scholarpath.data.corpus.Corpus`, or an iterable of papers."""
    if papers is None:
        return {}
    inner = getattr(papers, "papers", None)
    if isinstance(inner, Mapping):
        return dict(inner)
    if isinstance(papers, Mapping):
        return dict(papers)
    return {p.paper_id: p for p in papers}


# YYYY, YYYY-MM, YYYY-MM-DD (separators - / .), optionally followed by a time part.
_DATE_RE = re.compile(r"^(\d{4})(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?(?:[T\s].*)?$")
# Compact YYYYMMDD, optionally followed by a time part.
_COMPACT_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})(?:[T\s].*)?$")


def normalize_date(value: Any) -> Optional[str]:
    """ISO ``YYYY-MM-DD`` for a date-like value; ``None`` if missing.

    Accepts ``date``/``datetime`` objects, integral years (``2021``,
    ``2021.0``), and strings ``"2021"``, ``"2021-03"`` (also ``2021/03``,
    ``2021.03``), ``"2021-03-04"``, ``"20210304"``, each optionally followed
    by a time (``"2021-03-04T10:00:00Z"``, ``"2021-03-04 10:00"``). Missing
    month/day default to 1 (a year-only date is placed at the start of the
    year). Anything else -- including trailing garbage, invalid months or
    days, and non-integral numbers -- raises ``ValueError`` instead of being
    silently truncated, since a misplaced date moves papers across cutoffs.
    """
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, bool):
        raise ValueError(f"Unrecognised date {value!r}")
    if isinstance(value, numbers.Real):
        if isinstance(value, float) and math.isnan(value):
            return None
        if not float(value).is_integer():
            raise ValueError(f"Unrecognised date {value!r} (a number must be an integral year)")
        return f"{int(value):04d}-01-01"
    s = str(value).strip()
    if not s:
        return None
    m = _COMPACT_DATE_RE.match(s) or _DATE_RE.match(s)
    if not m:
        raise ValueError(f"Unrecognised date {value!r}")
    year, month, day = int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1)
    try:
        return _dt.date(year, month, day).isoformat()
    except ValueError as err:
        raise ValueError(f"Unrecognised date {value!r}: {err}") from None


def add_years(date: Any, years: float) -> str:
    """``date + years`` as an ISO date (calendar years when ``years`` is integral)."""
    iso = normalize_date(date)
    if iso is None:
        raise ValueError("add_years needs a date")
    d = _dt.date.fromisoformat(iso)
    if float(years).is_integer():
        y = d.year + int(years)
        try:
            return d.replace(year=y).isoformat()
        except ValueError:  # 29 February -> 28 February
            return d.replace(year=y, day=28).isoformat()
    return (d + _dt.timedelta(days=int(round(float(years) * 365.2425)))).isoformat()


def paper_dates(papers: Any = None, dates: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """``paper_id -> ISO date`` from paper records, overridden by an explicit ``dates`` mapping.

    A record contributes ``Paper.date`` (falling back to ``Paper.year``); a
    mapping value may also be a plain date string. Undated papers are omitted.
    """
    out: Dict[str, str] = {}
    for pid, rec in _papers(papers).items():
        if rec is None or isinstance(rec, (str, numbers.Integral, _dt.date)):
            d = normalize_date(rec)
        else:  # a Paper (or paper-like record)
            d = normalize_date(getattr(rec, "date", None)) or normalize_date(getattr(rec, "year", None))
        if d is not None:
            out[str(pid)] = d
    for pid, value in (dates or {}).items():
        d = normalize_date(value)
        if d is not None:
            out[str(pid)] = d
    return out


# ---------------------------------------------------------------------------
# Dual taxonomy access
# ---------------------------------------------------------------------------


class _Dual:
    """Uniform access to (T_P, T_M) from a SemanticIndex, a (problem, method) pair, or any object
    with ``problem`` / ``method`` taxonomy attributes."""

    def __init__(self, source: Any):
        if isinstance(source, (tuple, list)) and len(source) == 2:
            self.problem, self.method = source
        else:
            self.problem, self.method = source.problem, source.method
        if not isinstance(self.problem, Taxonomy) or not isinstance(self.method, Taxonomy):
            raise TypeError("Expected a SemanticIndex, a (problem, method) taxonomy pair, or an object with "
                            "'problem' and 'method' taxonomies")

    def taxonomy(self, facet: str) -> Taxonomy:
        return self.problem if facet == "problem" else self.method

    def paper_concepts(self, paper_id: str, facet: str) -> Set[str]:
        return self.taxonomy(facet).paper_nodes(paper_id)


def _category_lookup(papers: Mapping[str, Any], primary_category_of: CategoryFn) -> Callable[[str], Optional[str]]:
    """Domain of a paper: caller mapping/function, else ``Paper.primary_category``, else first category."""
    if callable(primary_category_of):
        return primary_category_of  # type: ignore[return-value]
    if isinstance(primary_category_of, Mapping):
        mapping = primary_category_of
        return lambda pid: mapping.get(pid)

    def lookup(pid: str) -> Optional[str]:
        p = papers.get(pid)
        if not isinstance(p, Paper):
            return None
        return p.primary_category or (p.categories[0] if p.categories else None)

    return lookup


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass
class HeldoutTarget:
    """An emerging cross-domain idea (new endpoint, existing endpoint) held out from the corpus."""

    target_id: str
    setting: str  # NEW_PROBLEM | NEW_METHOD
    problem_id: str
    method_id: str
    supporting_papers: List[str]  # J(u, m) on the full taxonomy, sorted by (date, id)
    source_paper: str  # earliest supporting paper
    source_date: str
    target_domain: Optional[str]  # primary category of the source paper
    # Home domain of the existing endpoint: the dominant primary category of its papers
    # published before ``evidence_before`` (and kept by the reduction, after select_targets).
    existing_domain: Optional[str]
    new_extent: int  # |P_new| on the full taxonomy (<= max_extent)
    existing_extent: int
    existing_prior_papers: int  # those earlier papers of the existing endpoint outside the target domain
    followup_papers: List[str] = field(default_factory=list)  # endpoint papers dated on/after the source
    new_name: str = ""
    new_definition: str = ""
    new_path: List[str] = field(default_factory=list)
    existing_name: str = ""
    existing_definition: str = ""
    existing_path: List[str] = field(default_factory=list)
    existing_target_domain_papers: int = 0  # those earlier papers inside the target domain
    # Existing-endpoint evidence is counted on papers dated strictly before this date: the source
    # date for a candidate, and after select_targets the date bounding what the reduction keeps
    # (the global reduction cutoff, or the source date under the per-target reduction).
    evidence_before: str = ""
    reduction: str = ""  # reduction mode the evidence was checked against ("" = none)
    selection_rank: int = -1  # acceptance order of the seeded sampler (used to trim oversampling)

    @property
    def new_facet(self) -> str:
        return NEW_FACET[self.setting]

    @property
    def existing_facet(self) -> str:
        return COUNTERPART_FACET[self.setting]

    @property
    def new_endpoint(self) -> str:
        return self.problem_id if self.setting == NEW_PROBLEM else self.method_id

    @property
    def existing_endpoint(self) -> str:
        return self.method_id if self.setting == NEW_PROBLEM else self.problem_id

    @property
    def endpoints(self) -> Tuple[str, str]:
        return self.problem_id, self.method_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HeldoutTarget":
        names = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in names})


class _DomainTimeline:
    """Dated papers of one concept grouped by domain, for "published before d" counts.

    Undated papers and papers without a domain are left out: they can neither
    be placed before a source nor attributed to a domain.
    """

    __slots__ = ("by_domain",)

    def __init__(self, extent: Iterable[str], dates: Mapping[str, str], cat: Callable[[str], Optional[str]]):
        groups: Dict[str, List[Tuple[str, str]]] = {}
        for p in extent:
            d = dates.get(p)
            c = cat(p) if d is not None else None
            if c is not None:
                groups.setdefault(c, []).append((d, p))
        self.by_domain: Dict[str, List[Tuple[str, str]]] = {c: sorted(v) for c, v in sorted(groups.items())}

    def counts(self, before: str) -> Dict[str, int]:
        """Papers per domain dated strictly before ``before``."""
        out: Dict[str, int] = {}
        for c, v in self.by_domain.items():
            k = bisect.bisect_left(v, (before, ""))
            if k:
                out[c] = k
        return out

    def papers(self, before: str) -> Iterator[Tuple[str, str]]:
        """``(paper id, domain)`` of the papers dated strictly before ``before`` (deterministic order)."""
        for c, v in self.by_domain.items():
            for _, p in v[: bisect.bisect_left(v, (before, ""))]:
                yield p, c


def existing_endpoint_domain(counts: Mapping[str, int], target_domain: Optional[str]) -> Tuple[Optional[str], int, int]:
    """Home domain of an existing endpoint from its per-domain paper counts.

    Returns ``(home, n_other, n_target)``: ``n_target`` papers lie in the
    target domain and ``n_other`` outside it. ``home`` is the dominant domain
    (most papers; ties broken by name, so the result never depends on set
    iteration order), or ``None`` when there is no evidence or the target
    domain is itself (co-)dominant -- the endpoint is then not "of another
    domain" (App. C.1; e.g. m* of C.6 is a signal-processing method, eess.SP,
    adapted to a q-bio.PE problem).
    """
    n_target = int(counts.get(target_domain, 0)) if target_domain is not None else 0
    n_other = int(sum(v for c, v in counts.items() if c != target_domain))
    ranked = sorted(((c, int(v)) for c, v in counts.items() if v > 0), key=lambda kv: (-kv[1], kv[0]))
    if not ranked or n_target >= ranked[0][1]:
        return None, n_other, n_target
    return ranked[0][0], n_other, n_target


def enumerate_target_candidates(
    taxonomies: Any,
    papers_or_dates: Any,
    primary_category_of: CategoryFn = None,
    max_extent: int = 4,
    *,
    min_extent: int = 1,
    min_existing_papers: int = 2,
    maximal_only: bool = True,
    require_new_first: bool = True,
    min_source_date: Optional[str] = None,
) -> List[HeldoutTarget]:
    """Every admissible target of the dual taxonomy (before sampling).

    New endpoint v (either facet): a non-root concept with
    ``min_extent <= |P_v| <= max_extent``. With ``maximal_only`` only the
    largest such concepts are used (the parent's extent exceeds
    ``max_extent``), so a new idea is represented once rather than by every
    concept of its small subtree. With ``require_new_first`` the new endpoint
    must not predate its source paper (no paper of P_v is earlier), i.e. it is
    *proposed* by the target.

    Existing endpoint x (other facet, non-root): shares papers with v, has
    ``|P_x| > max_extent``, and is *of another domain*: among its papers
    published before the source, the dominant primary category is not the
    target domain (the source's primary category) and at least
    ``min_existing_papers`` lie outside the target domain
    (:func:`existing_endpoint_domain`). Among the admissible x of a given v,
    the most specific one is kept (deepest; ties by larger support, then id),
    so every new endpoint yields at most one candidate.

    The existing-endpoint statistics of a candidate are relative to its own
    source date; :func:`select_targets` re-checks them against the evidence
    that survives the corpus reduction of the selected set.
    """
    dual = _Dual(taxonomies)
    papers = _papers(papers_or_dates) if papers_or_dates is not None else {}
    dates = paper_dates(papers)
    cat = _category_lookup(papers, primary_category_of)
    min_source = normalize_date(min_source_date) if min_source_date else None
    timelines: Dict[Tuple[str, str], _DomainTimeline] = {}
    out: List[HeldoutTarget] = []
    for setting in SETTINGS:
        new_facet, old_facet = NEW_FACET[setting], COUNTERPART_FACET[setting]
        T_new, T_old = dual.taxonomy(new_facet), dual.taxonomy(old_facet)
        for v in sorted(T_new.nodes):
            if v == T_new.root_id:
                continue
            ext = T_new.extent(v)
            if not (min_extent <= len(ext) <= max_extent):
                continue
            parent = T_new.parent(v)
            if maximal_only and parent is not None and len(T_new.extent(parent)) <= max_extent:
                continue
            if any(p not in dates for p in ext):
                continue  # cannot order an undated idea in time
            first_date = min(dates[p] for p in ext)
            co: Dict[str, Set[str]] = {}
            for p in ext:
                for x in dual.paper_concepts(p, old_facet):
                    if x != T_old.root_id:
                        co.setdefault(x, set()).add(p)
            # Most specific counterpart first: deepest, then larger support, then smallest id;
            # the first admissible one is kept.
            for x, J in sorted(co.items(), key=lambda kv: (-T_old.nodes[kv[0]].depth, -len(kv[1]), kv[0])):
                support = sorted(J, key=lambda p: (dates[p], p))
                source, src_date = support[0], dates[support[0]]
                if require_new_first and first_date < src_date:
                    continue
                if min_source is not None and src_date < min_source:
                    continue
                ext_x = T_old.extent(x)
                domain = cat(source)
                if len(ext_x) <= max_extent or domain is None:
                    continue
                # Papers of x dated before the source: none of them is in P_v, since the source is
                # the earliest paper of J(v, x) = P_v ∩ P_x.
                tl = timelines.get((old_facet, x))
                if tl is None:
                    tl = timelines[(old_facet, x)] = _DomainTimeline(ext_x, dates, cat)
                home, n_other, n_target = existing_endpoint_domain(tl.counts(src_date), domain)
                if home is None or n_other < min_existing_papers:
                    continue
                u, m = (v, x) if setting == NEW_PROBLEM else (x, v)
                out.append(HeldoutTarget(
                    target_id="",
                    setting=setting,
                    problem_id=u,
                    method_id=m,
                    supporting_papers=support,
                    source_paper=source,
                    source_date=src_date,
                    target_domain=domain,
                    existing_domain=home,
                    new_extent=len(ext),
                    existing_extent=len(ext_x),
                    existing_prior_papers=n_other,
                    followup_papers=sorted(
                        q for q in (ext | ext_x) if q not in J and q in dates and dates[q] >= src_date
                    ),
                    new_name=T_new.nodes[v].name,
                    new_definition=T_new.nodes[v].definition,
                    new_path=T_new.path_names(v),
                    existing_name=T_old.nodes[x].name,
                    existing_definition=T_old.nodes[x].definition,
                    existing_path=T_old.path_names(x),
                    existing_target_domain_papers=n_target,
                    evidence_before=src_date,
                ))
                break
    return out


class _EvidenceGuard:
    """Keeps the existing endpoint of every selected target *existing* after the corpus reduction.

    App. C.1 requires an existing endpoint of another domain and a reduced
    corpus that "still contains related evidence ... so that the answer is
    predictable rather than a guess"; removing one target must not remove the
    evidence from which another could be predicted. A target therefore stays
    admissible only if the papers of its existing endpoint that (i) survive
    the reduction of the *whole* selected set and (ii) predate its source
    still have a dominant domain other than the target domain and at least
    ``min_existing_papers`` papers outside it (:func:`existing_endpoint_domain`).

    * ``"global"``: the reduction keeps the papers dated before
      ``c = min(source dates)``. Supporting papers are never earlier than
      their own source, hence never earlier than c, so the surviving evidence
      of target s is its existing endpoint's papers dated before
      ``min(c, source_s) = c``. Accepting an earlier source lowers c, so every
      accepted target is re-checked.
    * ``"per_target"``: the reduction removes each target's supporting and
      follow-up papers; the surviving evidence of s is its existing
      endpoint's papers dated before its source minus every removed paper
      (tracked incrementally with an inverted index paper -> targets).
    """

    def __init__(
        self,
        dual: _Dual,
        dates: Mapping[str, str],
        cat: Callable[[str], Optional[str]],
        mode: str,
        min_existing_papers: int,
    ):
        if mode not in REDUCTIONS:
            raise ValueError(f"Unknown reduction mode {mode!r}; expected one of {REDUCTIONS}")
        self.dual, self.dates, self.cat, self.mode = dual, dates, cat, mode
        self.k = int(min_existing_papers)
        self.chosen: List[HeldoutTarget] = []
        self.cutoff: Optional[str] = None  # global: min source date of the accepted targets
        self.removed: Set[str] = set()  # per_target: papers removed by the accepted targets
        self.alive: List[Counter] = []  # per_target: surviving evidence per domain, per accepted target
        self.inv: Dict[str, List[int]] = {}  # per_target: evidence paper -> accepted targets using it
        self._timelines: Dict[Tuple[str, str], _DomainTimeline] = {}

    def _timeline(self, t: HeldoutTarget) -> _DomainTimeline:
        key = (t.existing_facet, t.existing_endpoint)
        tl = self._timelines.get(key)
        if tl is None:
            ext = self.dual.taxonomy(t.existing_facet).extent(t.existing_endpoint)
            tl = self._timelines[key] = _DomainTimeline(ext, self.dates, self.cat)
        return tl

    def _ok(self, counts: Mapping[str, int], t: HeldoutTarget) -> bool:
        home, n_other, _ = existing_endpoint_domain(counts, t.target_domain)
        return home is not None and n_other >= self.k

    @staticmethod
    def _removal(t: HeldoutTarget) -> Set[str]:
        return set(t.supporting_papers) | set(t.followup_papers)

    def _surviving(self, t: HeldoutTarget, extra_removed: Set[str]) -> Counter:
        alive: Counter = Counter()
        for p, dom in self._timeline(t).papers(t.source_date):
            if p not in self.removed and p not in extra_removed:
                alive[dom] += 1
        return alive

    def _losses(self, new_removed: Iterable[str]) -> Dict[int, Counter]:
        out: Dict[int, Counter] = {}
        for p in new_removed:
            for i in self.inv.get(p, ()):
                out.setdefault(i, Counter())[self.cat(p)] += 1
        return out

    def admits(self, t: HeldoutTarget) -> bool:
        """Would every target stay admissible if ``t`` were added to the selection?"""
        if self.mode == "global":
            c = t.source_date if self.cutoff is None else min(self.cutoff, t.source_date)
            if not self._ok(self._timeline(t).counts(c), t):
                return False
            if self.cutoff is not None and c < self.cutoff:
                return all(self._ok(self._timeline(s).counts(c), s) for s in self.chosen)
            return True
        new_removed = self._removal(t) - self.removed
        if not self._ok(self._surviving(t, new_removed), t):
            return False
        return all(self._ok(self.alive[i] - lost, self.chosen[i]) for i, lost in self._losses(new_removed).items())

    def accept(self, t: HeldoutTarget) -> None:
        idx = len(self.chosen)
        self.chosen.append(t)
        if self.mode == "global":
            self.cutoff = t.source_date if self.cutoff is None else min(self.cutoff, t.source_date)
            return
        new_removed = self._removal(t) - self.removed
        for i, lost in self._losses(new_removed).items():
            self.alive[i] = self.alive[i] - lost
        self.removed |= new_removed
        alive: Counter = Counter()
        for p, dom in self._timeline(t).papers(t.source_date):
            if p not in self.removed:
                alive[dom] += 1
                self.inv.setdefault(p, []).append(idx)
        self.alive.append(alive)

    def final(self, idx: int) -> Tuple[Dict[str, int], str]:
        """Surviving evidence counts of the ``idx``-th accepted target and the date bounding them."""
        t = self.chosen[idx]
        if self.mode == "global":
            assert self.cutoff is not None
            return self._timeline(t).counts(self.cutoff), self.cutoff
        return dict(self.alive[idx]), t.source_date


def select_targets(
    index: Any,
    papers_or_dates: Any,
    primary_category_of: CategoryFn = None,
    n_new_problem: int = 24,
    n_new_method: int = 23,
    max_extent: int = 4,
    seed: int = 0,
    candidates: Optional[Sequence[HeldoutTarget]] = None,
    *,
    reduction: str = "global",
    min_existing_papers: int = 2,
    **kwargs: Any,
) -> List[HeldoutTarget]:
    """Sample held-out targets on the complete-corpus dual taxonomy (App. C.1, "Targets").

    ``index`` is a :class:`~scholarpath.index.SemanticIndex`, a
    ``(problem, method)`` taxonomy pair, or any object with ``problem`` and
    ``method`` taxonomies, built over the complete corpus *before* removal.
    ``papers_or_dates`` supplies the papers (or a ``paper_id -> date``
    mapping, in which case ``primary_category_of`` is required).

    Candidates (:func:`enumerate_target_candidates` with
    ``min_existing_papers`` and the extra ``kwargs``; or precomputed
    ``candidates``) are visited in a seeded random order and accepted
    greedily while

    * the per-setting quota is open;
    * the supporting papers are disjoint from those of every accepted target,
      and neither endpoint is an endpoint of an accepted target;
    * every accepted target, including the new one, keeps an existing
      endpoint of another domain in the corpus reduced with ``reduction``
      (the mode later passed to :func:`reduce_corpus`; see
      :class:`_EvidenceGuard`), so no target's answer disappears from the
      reduced corpus.

    Fewer targets are returned (with a warning) when the candidates run out.
    The returned targets are copies: ``existing_domain``,
    ``existing_prior_papers``, ``existing_target_domain_papers`` and
    ``evidence_before`` describe the evidence that survives the reduction,
    ``selection_rank`` is the acceptance order, and target ids are
    ``np-XXX`` / ``nm-XXX`` in order of source date. ``n_new_problem`` /
    ``n_new_method`` are the numbers of *sampled* targets; the paper's 24 / 23
    are counts after validation (see ``oversample`` in
    :func:`build_heldout_benchmark`).
    """
    if candidates is None:
        candidates = enumerate_target_candidates(
            index, papers_or_dates, primary_category_of, max_extent,
            min_existing_papers=min_existing_papers, **kwargs,
        )
    papers = _papers(papers_or_dates) if papers_or_dates is not None else {}
    guard = _EvidenceGuard(_Dual(index), paper_dates(papers), _category_lookup(papers, primary_category_of),
                           reduction, min_existing_papers)
    rng = random.Random(seed)
    order = sorted(candidates, key=lambda t: (t.setting, t.problem_id, t.method_id))
    rng.shuffle(order)
    quota = {NEW_PROBLEM: int(n_new_problem), NEW_METHOD: int(n_new_method)}
    used_papers: Set[str] = set()
    used_endpoints: Set[str] = set()
    rejected_by_reduction = 0
    for t in order:
        if quota[t.setting] <= 0:
            continue
        if used_papers.intersection(t.supporting_papers):
            continue
        if used_endpoints.intersection(t.endpoints):
            continue
        if not guard.admits(t):
            rejected_by_reduction += 1
            continue
        guard.accept(dataclasses.replace(t))
        quota[t.setting] -= 1
        used_papers.update(t.supporting_papers)
        used_endpoints.update(t.endpoints)
        if all(q <= 0 for q in quota.values()):
            break
    for setting, missing in quota.items():
        if missing > 0:
            log.warning("Only %d %s targets available (%d short; %d candidates rejected because the reduction "
                        "would erase an existing endpoint's evidence)",
                        sum(t.setting == setting for t in guard.chosen), setting, missing, rejected_by_reduction)
    for rank, t in enumerate(guard.chosen):
        counts, before = guard.final(rank)
        home, n_other, n_target = existing_endpoint_domain(counts, t.target_domain)
        t.existing_domain, t.existing_prior_papers, t.existing_target_domain_papers = home, n_other, n_target
        t.evidence_before, t.reduction, t.selection_rank = before, reduction, rank
    prefix = {NEW_PROBLEM: "np", NEW_METHOD: "nm"}
    out: List[HeldoutTarget] = []
    for setting in SETTINGS:
        group = sorted((t for t in guard.chosen if t.setting == setting), key=lambda t: (t.source_date, t.source_paper))
        for k, t in enumerate(group):
            t.target_id = f"{prefix[setting]}-{k:03d}"
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Corpus reduction
# ---------------------------------------------------------------------------


def reduce_corpus(
    corpus: Any,
    targets: Sequence[HeldoutTarget],
    mode: str = "global",
    dates: Optional[Mapping[str, Any]] = None,
    drop_undated: bool = False,
) -> Tuple[Corpus, List[str]]:
    """Remove the targets' supporting papers and the papers published after them.

    ``mode="global"`` (default; App. C.1 "together with the papers published
    after them"): remove every supporting paper plus every paper dated on or
    after the *earliest* source date over all targets, so the reduced corpus
    is a strict temporal prefix of the literature before any held-out idea.

    ``mode="per_target"``: remove every supporting paper plus, for each
    target, the papers of either endpoint's extent (on the full taxonomy)
    dated on or after that target's source date (``followup_papers``); the
    rest of the recent literature stays. Use it when the targets span a long
    period.

    Undated papers cannot be placed in time and are kept unless
    ``drop_undated``. Returns ``(reduced corpus, sorted removed ids)``.
    Use the same ``mode`` (and dates) as :func:`select_targets`, which only
    accepts targets whose existing endpoint keeps cross-domain evidence under
    that reduction.
    """
    if mode not in REDUCTIONS:
        raise ValueError(f"Unknown reduction mode {mode!r}; expected one of {REDUCTIONS}")
    papers = _papers(corpus)
    date_map = paper_dates(papers, dates)
    removed: Set[str] = set()
    for t in targets:
        removed.update(t.supporting_papers)
    if mode == "global":
        if targets:
            cutoff = min(t.source_date for t in targets)
            removed.update(p for p, d in date_map.items() if d >= cutoff)
    else:  # per_target
        for t in targets:
            removed.update(t.followup_papers)
    if drop_undated:
        removed.update(p for p in papers if p not in date_map)
    removed &= set(papers)
    name = getattr(corpus, "name", "corpus")
    reduced = Corpus((p for pid, p in papers.items() if pid not in removed), name=f"{name}-reduced")
    return reduced, sorted(removed)


# ---------------------------------------------------------------------------
# Leakage control
# ---------------------------------------------------------------------------


@dataclass
class LeakageReport:
    """Automatic leakage check of one query (a screening aid for the annotators)."""

    leaked: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


def acronyms(name: str, abstract: str = "") -> Set[str]:
    """Plausible acronyms of a concept name (upper case).

    Initials of the name's words (``Graph Neural Networks`` -> ``GNN``),
    upper-case tokens inside the name, and a parenthesised abbreviation that
    follows the name in the abstract (``blind source separation (BSS)``).
    """
    out: Set[str] = set()
    words = [w for w in _WORD_RE.findall(name or "") if w.lower() not in STOPWORDS]
    initials = "".join(w[0] for w in words).upper()
    if len(initials) >= 2:
        out.add(initials)
    for w in _WORD_RE.findall(name or ""):
        if len(w) >= 2 and sum(ch.isupper() for ch in w) >= 2:
            out.add(_singular_upper(w))
    if name and abstract:
        pat = re.compile(re.escape(name.strip()) + r"s?\s*\(([A-Za-z][A-Za-z0-9\-]{1,12})\)", re.IGNORECASE)
        for m in pat.finditer(abstract):
            out.add(_singular_upper(m.group(1)))
    return out


def _singular_upper(word: str) -> str:
    """``GNNs`` -> ``GNN``; ``BSS`` stays ``BSS`` (only a lower-case plural 's' is dropped)."""
    if len(word) > 2 and word.endswith("s"):
        word = word[:-1]
    return word.upper()


def _ngrams(tokens: Sequence[str], n: int) -> Set[Tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _subtokens(text: str) -> Set[str]:
    """Lower-case word tokens, also split at hyphens, plus their forms without a plural 's'."""
    out: Set[str] = set()
    for tok in tokenize(text or ""):
        for part in [tok] + tok.split("-"):
            if part:
                out.add(part)
                if len(part) > 3 and part.endswith("s"):
                    out.add(part[:-1])
    return out


def check_leakage(
    query: str,
    counterpart_names: Sequence[str],
    abstract: str = "",
    allowed_text: str = "",
    ngram: int = 5,
    term_fraction: float = 2.0 / 3.0,
) -> LeakageReport:
    """Does a held-out query reveal its answer? (App. C.1, "Validation and Leakage Control")

    A heuristic screen for the annotators. For each counterpart name it flags

    (i) the name appearing as a phrase;
    (ii) a contiguous part of the name (two or more tokens) appearing in the
         query and holding at least two *distinctive* terms ("blind source
         separation" for "blind source separation for cascading point
         processes");
    (iii) at least ``term_fraction`` of the name's distinctive terms, in any
         order or inflection (2 of 3 for "Blind Source Separation");
    (iv) an acronym of the name (:func:`acronyms`): upper-case anywhere, and
         for acronyms of three or more letters that are not stopwords also in
         lower case or inside hyphenated words ("bss-style");

    and, independently of the names, (v) a distinctive phrase of the abstract:
    a shared ``ngram``-token sequence with at least two content words.

    Distinctive terms are the stemmed content words of the name without
    generic words ("learning", "model", "network", ...) and without the terms
    of ``allowed_text`` -- the new endpoint's label and definition, which the
    query is meant to describe. Acronyms and abstract phrases occurring in
    ``allowed_text`` are allowed for the same reason.
    """
    reasons: List[str] = []
    q_tokens = tokenize(query)
    q_norm = " " + " ".join(q_tokens) + " "
    q_terms = content_terms(query, drop_generic=False)
    q_upper = {_singular_upper(w) for w in _WORD_RE.findall(query) if sum(ch.isupper() for ch in w) >= 2}
    q_lower = _subtokens(query)
    allowed_terms = content_terms(allowed_text, drop_generic=False) if allowed_text else set()
    allowed_lower = _subtokens(allowed_text) if allowed_text else set()
    for name in [n for n in counterpart_names if n and n.strip()]:
        n_tokens = tokenize(name)
        if n_tokens and f" {' '.join(n_tokens)} " in q_norm:
            reasons.append(f"names the counterpart '{name}'")
            continue
        terms = content_terms(name, drop_generic=True) - allowed_terms
        span = _shared_span(n_tokens, q_norm, terms)
        if span:
            reasons.append(f"names part of the counterpart ('{span}' of '{name}')")
            continue
        if terms:
            hit = terms & q_terms
            need = max(1, math.ceil(term_fraction * len(terms) - 1e-9))
            if len(hit) >= need:
                reasons.append(f"uses the counterpart's terms {sorted(hit)} ('{name}')")
                continue
        hit_acr = sorted(
            a for a in acronyms(name, abstract)
            if len(a) >= 2 and a.lower() not in allowed_lower and (
                a in q_upper or (len(a) >= 3 and a.lower() not in STOPWORDS and a.lower() in q_lower)
            )
        )
        if hit_acr:
            reasons.append(f"uses the counterpart acronym {hit_acr}")
    if abstract and ngram > 0:
        allowed = _ngrams(tokenize(allowed_text), ngram) if allowed_text else set()
        shared = _ngrams(q_tokens, ngram) & _ngrams(tokenize(abstract), ngram)
        shared = {
            g for g in shared
            if g not in allowed and sum(1 for t in g if t not in STOPWORDS and len(t) > 2) >= 2
        }
        if shared:
            phrase = " ".join(sorted(shared)[0])
            reasons.append(f"copies an abstract phrase ('{phrase}')")
    return LeakageReport(leaked=bool(reasons), reasons=reasons)


def _shared_span(name_tokens: Sequence[str], q_norm: str, distinctive: Set[str]) -> str:
    """Longest contiguous part of a name (>= 2 tokens, >= 2 distinctive terms) occurring in the query."""
    n = len(name_tokens)
    for length in range(n - 1, 1, -1):
        for i in range(n - length + 1):
            part = list(name_tokens[i : i + length])
            text = " ".join(part)
            if f" {text} " in q_norm and len(content_terms(text, drop_generic=True) & distinctive) >= 2:
                return text
    return ""


# ---------------------------------------------------------------------------
# Queries and expected answers
# ---------------------------------------------------------------------------


def _target_request(target: HeldoutTarget, papers: Mapping[str, Any], provide_counterpart: bool, component: str):
    src = papers.get(target.source_paper)
    if src is None:
        raise KeyError(f"Source paper {target.source_paper!r} of target {target.target_id} is not in the corpus")
    source = {"id": src.paper_id, "title": src.title, "abstract": src.abstract, "date": target.source_date}
    new = {"facet": target.new_facet, "name": target.new_name, "definition": target.new_definition,
           "path": list(target.new_path)}
    counterpart = None
    if provide_counterpart:
        counterpart = {"facet": target.existing_facet, "name": target.existing_name,
                       "definition": target.existing_definition}
    return heldout_query_request(source, target.setting, new, counterpart, component=component)


def generate_queries(
    llm: LLMClient,
    targets: Sequence[HeldoutTarget],
    papers: Any,
    *,
    provide_counterpart: bool = False,
    component: str = "benchmark",
    benchmark: str = "heldout",
) -> List[Query]:
    """Write one hypothesis-discovery query per target (App. C.1, "Queries and Expected Answers").

    ``papers`` is the complete corpus (it must still contain the source
    papers). Each LLM call (task ``benchmark.heldout_query``, billed to
    ``component``) receives the source paper's title and abstract, the
    setting, and the new endpoint's label and definition; with
    ``provide_counterpart`` it additionally receives the existing endpoint's
    reference label as an alignment hint (not part of the paper's protocol).

    The returned :class:`~scholarpath.types.Query` objects have kind
    ``hypothesis_discovery``; ``reference_answer`` is the extracted
    counterpart (the expected answer record used by the Disc metric), and
    ``metadata`` holds the setting, the known (new) endpoint, the held-out
    counterpart concept of the reference taxonomy, the rationale, the source
    paper, the supporting papers and the automatic leakage report.
    ``gold_papers`` stays empty: the supporting papers are removed from the
    reduced corpus. Targets whose LLM output cannot be parsed are skipped.
    """
    pmap = _papers(papers)
    targets = list(targets)
    requests = [_target_request(t, pmap, provide_counterpart, component) for t in targets]

    def run(req):
        try:
            obj, _ = llm.complete_json(req)
            return parse_heldout_query(obj)
        except JSONParseError as err:
            log.warning("Held-out query generation failed to parse: %s", err)
            return "", "", ""

    queries: List[Query] = []
    for t, (text, counterpart, rationale) in zip(targets, llm.map(run, requests)):
        if not text:
            log.warning("Target %s produced no query; skipped", t.target_id)
            continue
        counterpart = counterpart or t.existing_name
        src = pmap[t.source_paper]
        leak = check_leakage(
            text,
            [counterpart, t.existing_name],
            abstract=src.abstract or "",
            allowed_text=f"{t.new_name}. {t.new_definition}",
        )
        queries.append(Query(
            query_id=f"{benchmark}-{t.target_id}",
            text=text,
            kind=HYPOTHESIS_DISCOVERY,
            benchmark=benchmark,
            gold_papers=[],
            reference_answer=counterpart,
            metadata={
                "setting": t.setting,
                "target_id": t.target_id,
                "known_endpoint": t.new_name,
                "known_endpoint_id": t.new_endpoint,
                "known_endpoint_facet": t.new_facet,
                "known_endpoint_definition": t.new_definition,
                "heldout_counterpart": counterpart,
                "heldout_counterpart_id": t.existing_endpoint,
                "heldout_counterpart_facet": t.existing_facet,
                "heldout_counterpart_label": t.existing_name,
                "heldout_counterpart_definition": t.existing_definition,
                "rationale": rationale,
                "source_paper": t.source_paper,
                "source_date": t.source_date,
                "source_title": src.title,
                "supporting_papers": list(t.supporting_papers),
                "target_domain": t.target_domain,
                "existing_domain": t.existing_domain,
                "existing_prior_papers": t.existing_prior_papers,
                "evidence_before": t.evidence_before,
                "selection_rank": t.selection_rank,
                "leakage": leak.to_dict(),
            },
        ))
    return queries


def supporting_evidence(
    taxonomies: Any,
    target: HeldoutTarget,
    reduced_ids: Iterable[str],
    primary_category_of: CategoryFn = None,
    papers: Any = None,
    max_ids: int = 10,
) -> Dict[str, Any]:
    """Related evidence for a target that survives in the reduced corpus (annotation aid).

    Counts, over the reduced corpus and on the complete-corpus taxonomy:

    * ``counterpart_papers`` -- papers of the held-out counterpart concept,
      and ``counterpart_cross_domain_papers`` those outside the target domain;
    * ``neighbor_papers`` -- papers of the new endpoint's neighbours (its
      parent's extent minus its own);
    * ``counterpart_on_neighbors`` -- applications of the counterpart to
      those neighbouring concepts (siblings), and ``counterpart_on_cousins``
      the same one level higher (grandparent's extent minus parent's);
    * ``new_endpoint_papers`` -- papers of the new endpoint that remain
      (0 under the global reduction: the query presents it as unfamiliar).

    ``evidence_papers`` lists up to ``max_ids`` neighbour applications
    (siblings first) for the annotators to inspect.
    """
    dual = _Dual(taxonomies)
    R = set(reduced_ids)
    T_new, T_old = dual.taxonomy(target.new_facet), dual.taxonomy(target.existing_facet)
    v, x = target.new_endpoint, target.existing_endpoint
    ext_v = T_new.extent(v)
    cp = T_old.extent(x) & R
    pmap = _papers(papers) if papers is not None else {}
    cat = _category_lookup(pmap, primary_category_of)
    parent = T_new.parent(v)
    grand = T_new.parent(parent) if parent is not None else None
    sib = (T_new.extent(parent) - ext_v) & R if parent is not None else set()
    cousin = (T_new.extent(grand) - T_new.extent(parent)) & R if grand is not None else set()
    on_sib = cp & sib
    on_cousin = cp & cousin
    dates = paper_dates(pmap)
    ordered = sorted(on_sib, key=lambda p: (dates.get(p, ""), p), reverse=True) + sorted(
        on_cousin, key=lambda p: (dates.get(p, ""), p), reverse=True
    )
    cross: Optional[int] = None  # unknown without domain information
    if pmap or primary_category_of is not None:
        domains = [cat(p) for p in cp]
        cross = sum(1 for c in domains if c is not None and c != target.target_domain)
    return {
        "counterpart_papers": len(cp),
        "counterpart_cross_domain_papers": cross,
        "neighbor_papers": len(sib),
        "counterpart_on_neighbors": len(on_sib),
        "counterpart_on_cousins": len(on_cousin),
        "new_endpoint_papers": len(ext_v & R),
        "evidence_papers": ordered[:max_ids],
    }


# ---------------------------------------------------------------------------
# Annotation sheet (two annotators, three conditions)
# ---------------------------------------------------------------------------

_INFO_COLUMNS = (
    "query_id", "setting", "query", "known_endpoint", "known_endpoint_definition", "expected_counterpart",
    "counterpart_concept", "counterpart_definition", "rationale", "source_paper_id", "source_date",
    "source_title", "source_abstract", "auto_leak", "auto_leak_reasons", "counterpart_papers",
    "counterpart_cross_domain_papers", "neighbor_papers", "counterpart_on_neighbors", "counterpart_on_cousins",
    "evidence_papers",
)


def annotation_columns() -> List[str]:
    """Columns of the annotation sheet: query info, ``a1_*``/``a2_*`` judgements, ``final_*`` and notes."""
    judge = [f"{a}_{c}" for a in ANNOTATORS for c in CONDITIONS]
    final = [f"final_{c}" for c in CONDITIONS]
    return list(_INFO_COLUMNS) + judge + final + ["notes"]


def _sheet_row(q: Query, abstracts: Mapping[str, str]) -> Dict[str, Any]:
    md = q.metadata
    ev = md.get("evidence", {}) or {}
    leak = md.get("leakage", {}) or {}
    row: Dict[str, Any] = {
        "query_id": q.query_id,
        "setting": md.get("setting", ""),
        "query": q.text,
        "known_endpoint": md.get("known_endpoint", ""),
        "known_endpoint_definition": md.get("known_endpoint_definition", ""),
        "expected_counterpart": q.reference_answer or md.get("heldout_counterpart", ""),
        "counterpart_concept": md.get("heldout_counterpart_label", ""),
        "counterpart_definition": md.get("heldout_counterpart_definition", ""),
        "rationale": md.get("rationale", ""),
        "source_paper_id": md.get("source_paper", ""),
        "source_date": md.get("source_date", ""),
        "source_title": md.get("source_title", ""),
        "source_abstract": abstracts.get(md.get("source_paper", ""), md.get("source_abstract", "")),
        "auto_leak": int(bool(leak.get("leaked", False))),
        "auto_leak_reasons": "; ".join(leak.get("reasons", []) or []),
        "evidence_papers": " ".join(ev.get("evidence_papers", []) or []),
    }
    for k in ("counterpart_papers", "counterpart_cross_domain_papers", "neighbor_papers",
              "counterpart_on_neighbors", "counterpart_on_cousins"):
        row[k] = ev.get(k, "")
    for col in annotation_columns():
        row.setdefault(col, "")
    return row


def export_annotation_sheet(queries: Sequence[Query], path: PathLike, papers: Any = None) -> Path:
    """Write the two-annotator validation sheet (CSV/TSV by suffix, else JSONL).

    One row per query with everything an annotator needs -- the query, the
    known endpoint, the expected answer record, the source paper (title and,
    when ``papers`` is given, abstract), the automatic leakage screen and the
    precomputed supporting-evidence counts of :func:`supporting_evidence` --
    plus empty judgement columns ``a1_<condition>``, ``a2_<condition>`` for the
    three conditions of App. C.1 (:data:`CONDITIONS`), ``final_<condition>``
    for disagreements resolved by discussion, and ``notes``. Judgements are
    1/0 (also yes/no, true/false, pass/fail).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pmap = _papers(papers) if papers is not None else {}
    abstracts = {pid: (p.abstract or "") for pid, p in pmap.items()}
    rows = [_sheet_row(q, abstracts) for q in queries]
    cols = annotation_columns()
    if path.suffix.lower() in (".csv", ".tsv"):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
    else:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({c: r.get(c, "") for c in cols}, ensure_ascii=False) + "\n")
    return path


_TRUE = {"1", "y", "yes", "true", "t", "pass", "ok", "keep", "✓"}
_FALSE = {"0", "n", "no", "false", "f", "fail", "drop", "reject", "x", "✗"}


def parse_judgement(value: Any) -> Optional[bool]:
    """1/0, yes/no, true/false, pass/fail -> bool; blank -> None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value != 0
    s = str(value).strip().lower()
    if not s:
        return None
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"Cannot parse annotation value {value!r} (use 1/0, yes/no, true/false or pass/fail)")


@dataclass
class AnnotationDecision:
    """Resolved judgement of one query across the three conditions."""

    query_id: str
    conditions: Dict[str, Optional[bool]]  # resolved value per condition (None = unresolved)
    disagreements: List[str] = field(default_factory=list)  # conditions where a1 != a2
    unresolved: List[str] = field(default_factory=list)  # no final value and no agreement
    notes: str = ""

    @property
    def keep(self) -> bool:
        return not self.unresolved and all(v is True for v in self.conditions.values())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["keep"] = self.keep
        return d


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() in (".csv", ".tsv"):
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_annotations(path: PathLike) -> Dict[str, AnnotationDecision]:
    """Resolve each condition: ``final_<c>`` if filled, else the annotators' agreed value, else unresolved."""
    decisions: Dict[str, AnnotationDecision] = {}
    for row in _read_rows(Path(path)):
        qid = str(row.get("query_id", "")).strip()
        if not qid:
            continue
        resolved: Dict[str, Optional[bool]] = {}
        disagreements, unresolved = [], []
        for c in CONDITIONS:
            votes = [parse_judgement(row.get(f"{a}_{c}")) for a in ANNOTATORS]
            final = parse_judgement(row.get(f"final_{c}"))
            if None not in votes and votes[0] != votes[1]:
                disagreements.append(c)
            if final is not None:
                resolved[c] = final
            elif None not in votes and votes[0] == votes[1]:
                resolved[c] = votes[0]
            else:
                resolved[c] = None
                unresolved.append(c)
        decisions[qid] = AnnotationDecision(qid, resolved, disagreements, unresolved, str(row.get("notes", "") or ""))
    return decisions


def annotation_agreement(path: PathLike) -> Dict[str, Dict[str, float]]:
    """Raw agreement and Cohen's kappa of the two annotators per condition."""
    rows = _read_rows(Path(path))
    out: Dict[str, Dict[str, float]] = {}
    for c in CONDITIONS:
        pairs = []
        for row in rows:
            a, b = (parse_judgement(row.get(f"{x}_{c}")) for x in ANNOTATORS)
            if a is not None and b is not None:
                pairs.append((a, b))
        n = len(pairs)
        if not n:
            out[c] = {"n": 0.0, "agreement": float("nan"), "kappa": float("nan")}
            continue
        po = sum(a == b for a, b in pairs) / n
        pa = sum(a for a, _ in pairs) / n
        pb = sum(b for _, b in pairs) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)
        out[c] = {"n": float(n), "agreement": po, "kappa": kappa}
    return out


def apply_annotations(
    queries: Sequence[Query],
    path: PathLike,
    quota: Optional[Mapping[str, int]] = None,
) -> List[Query]:
    """Keep the queries that pass all three conditions (App. C.1).

    A condition passes when its ``final_<c>`` column says so, or -- without a
    final value -- when both annotators agree that it passes. Queries with an
    unresolved disagreement, a failed condition, or no row in the sheet are
    discarded. The decision is stored in ``query.metadata["annotation"]``.

    ``quota`` (setting -> count, e.g. ``benchmark.stats["requested"]`` =
    24 / 23 as in the paper) trims an oversampled benchmark: per setting, the
    first ``quota[setting]`` validated queries in the seeded selection order
    (``metadata["selection_rank"]``) are kept; settings missing from ``quota``
    are not trimmed. A warning is logged when fewer queries pass than the
    quota asks for. The input order is preserved.
    """
    decisions = read_annotations(path)
    kept: List[Query] = []
    dropped = Counter()
    for q in queries:
        d = decisions.get(q.query_id)
        if d is None:
            dropped["not annotated"] += 1
            continue
        q.metadata["annotation"] = d.to_dict()
        if d.keep:
            kept.append(q)
        elif d.unresolved:
            dropped["unresolved"] += 1
        else:
            dropped["failed"] += 1
    if quota:
        def rank(q: Query) -> Tuple[float, str]:
            r = q.metadata.get("selection_rank")
            return (float(r) if isinstance(r, numbers.Real) and r >= 0 else math.inf, q.query_id)

        chosen: Set[str] = set()
        used: Counter = Counter()
        for q in sorted(kept, key=rank):
            setting = q.metadata.get("setting")
            limit = quota.get(setting) if setting is not None else None
            if limit is None or used[setting] < int(limit):
                chosen.add(q.query_id)
                used[setting] += 1
            else:
                dropped["over quota"] += 1
        kept = [q for q in kept if q.query_id in chosen]
        for setting, limit in quota.items():
            if used[setting] < int(limit):
                log.warning("Only %d validated %s queries (quota %d): sample more targets (oversample)",
                            used[setting], setting, int(limit))
    if dropped:
        log.info("Annotation: kept %d of %d queries (%s)", len(kept), len(queries), dict(dropped))
    return kept


# ---------------------------------------------------------------------------
# Orchestration and I/O
# ---------------------------------------------------------------------------


class HeldoutBenchmark(NamedTuple):
    """Output of :func:`build_heldout_benchmark` (targets are in ``stats["targets"]``)."""

    queries: List[Query]
    reduced_ids: List[str]
    removed_ids: List[str]
    stats: Dict[str, Any]

    @property
    def targets(self) -> List[HeldoutTarget]:
        return [HeldoutTarget.from_dict(d) for d in self.stats.get("targets", [])]


def build_heldout_benchmark(
    corpus: Any,
    index: Any,
    llm: LLMClient,
    *,
    primary_category_of: CategoryFn = None,
    n_new_problem: int = 24,
    n_new_method: int = 23,
    oversample: float = 1.0,
    max_extent: int = 4,
    seed: int = 0,
    reduction: str = "global",
    min_existing_papers: int = 2,
    provide_counterpart: bool = False,
    drop_leaking: bool = False,
    component: str = "benchmark",
    **target_kwargs: Any,
) -> HeldoutBenchmark:
    """Construct the Held-Out Hypothesis benchmark up to the annotation stage (App. C.1).

    ``corpus`` is the complete corpus and ``index`` its dual taxonomy (a
    SemanticIndex or ``(problem, method)`` pair) built before any removal.
    Steps: :func:`select_targets` (checked against ``reduction``) ->
    :func:`reduce_corpus` (same ``reduction``) -> :func:`generate_queries`
    (with the automatic leakage screen) -> :func:`supporting_evidence` on the
    reduced corpus (stored in ``query.metadata["evidence"]``).

    ``n_new_problem`` / ``n_new_method`` are the final, *post-validation*
    counts (24 / 23 in the paper). ``ceil(n * oversample)`` targets are
    sampled per setting and the corpus is reduced for all of them, so that
    after :func:`apply_annotations` with ``quota=stats["requested"]`` the first
    validated queries of each setting (in seeded selection order) form the
    benchmark; with ``oversample=1`` (default) validation can only lower the
    counts. Extra ``target_kwargs`` (e.g. ``min_source_date``,
    ``require_new_first``) go to :func:`enumerate_target_candidates`.

    As a safety net, a query whose held-out counterpart has no paper left in
    the reduced corpus (it cannot be predicted) is dropped and listed in
    ``stats["unsupported_targets"]``; target selection prevents this by
    construction. With ``drop_leaking`` the queries flagged by the automatic
    leakage check are dropped right away; otherwise they go to the annotators
    flagged. Export the sheet with :func:`export_annotation_sheet` and keep the
    validated queries with :func:`apply_annotations`.
    """
    if oversample < 1.0:
        raise ValueError("oversample must be >= 1")
    papers = _papers(corpus)
    before = llm.meter.snapshot()
    sampled = {
        NEW_PROBLEM: int(math.ceil(n_new_problem * oversample - 1e-9)),
        NEW_METHOD: int(math.ceil(n_new_method * oversample - 1e-9)),
    }
    candidates = enumerate_target_candidates(
        index, papers, primary_category_of, max_extent, min_existing_papers=min_existing_papers, **target_kwargs
    )
    targets = select_targets(
        index, papers, primary_category_of, sampled[NEW_PROBLEM], sampled[NEW_METHOD], max_extent, seed,
        candidates=candidates, reduction=reduction, min_existing_papers=min_existing_papers,
    )
    reduced, removed = reduce_corpus(corpus, targets, mode=reduction)
    queries = generate_queries(llm, targets, papers, provide_counterpart=provide_counterpart, component=component)
    by_id = {t.target_id: t for t in targets}
    unsupported: List[str] = []
    for q in queries:
        t = by_id[q.metadata["target_id"]]
        q.metadata["evidence"] = supporting_evidence(index, t, reduced.ids, primary_category_of, papers)
        if q.metadata["evidence"]["counterpart_papers"] == 0:
            unsupported.append(t.target_id)
    if unsupported:
        log.warning("Dropping %d queries whose counterpart has no paper in the reduced corpus: %s",
                    len(unsupported), unsupported)
        drop = set(unsupported)
        queries = [q for q in queries if q.metadata["target_id"] not in drop]
    n_leaking = sum(bool(q.metadata["leakage"]["leaked"]) for q in queries)
    if drop_leaking:
        queries = [q for q in queries if not q.metadata["leakage"]["leaked"]]
    usage = llm.meter.diff(before).get(component)
    stats: Dict[str, Any] = {
        "corpus_papers": len(papers),
        "candidates": {s: sum(c.setting == s for c in candidates) for s in SETTINGS},
        "targets_per_setting": {s: sum(t.setting == s for t in targets) for s in SETTINGS},
        "requested": {NEW_PROBLEM: n_new_problem, NEW_METHOD: n_new_method},
        "sampled": sampled,
        "oversample": oversample,
        "max_extent": max_extent,
        "min_existing_papers": min_existing_papers,
        "seed": seed,
        "reduction": reduction,
        "reduction_cutoff": min((t.source_date for t in targets), default=None),
        "supporting_papers": sum(len(t.supporting_papers) for t in targets),
        "removed_papers": len(removed),
        "reduced_papers": len(reduced),
        "queries": len(queries),
        "unsupported_targets": unsupported,
        "auto_leaking": n_leaking,
        "dropped_leaking": n_leaking if drop_leaking else 0,
        "provide_counterpart": provide_counterpart,
        "llm_usage": usage.to_dict() if usage is not None else None,
        "targets": [t.to_dict() for t in targets],
    }
    return HeldoutBenchmark(queries, sorted(reduced.ids), removed, stats)


def save_heldout_benchmark(benchmark: HeldoutBenchmark, directory: PathLike = DEFAULT_HELDOUT_DIR) -> Dict[str, Path]:
    """Write ``queries.jsonl``, ``removed_papers.json``, ``reduced_corpus_ids.json``,
    ``targets.json`` (and ``stats.json``) into ``directory``."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    paths = {k: d / v for k, v in HELDOUT_FILES.items()}
    save_queries(paths["queries"], benchmark.queries)
    write_json(paths["removed"], list(benchmark.removed_ids))
    write_json(paths["reduced"], list(benchmark.reduced_ids))
    write_json(paths["targets"], list(benchmark.stats.get("targets", [])))
    write_json(paths["stats"], {k: v for k, v in benchmark.stats.items() if k != "targets"})
    return paths


def load_heldout_benchmark(directory: PathLike = DEFAULT_HELDOUT_DIR) -> HeldoutBenchmark:
    """Read a benchmark written by :func:`save_heldout_benchmark`."""
    d = Path(directory)
    stats: Dict[str, Any] = {}
    if (d / HELDOUT_FILES["stats"]).exists():
        stats = dict(read_json(d / HELDOUT_FILES["stats"]))
    if (d / HELDOUT_FILES["targets"]).exists():
        stats["targets"] = list(read_json(d / HELDOUT_FILES["targets"]))
    return HeldoutBenchmark(
        queries=load_queries(d / HELDOUT_FILES["queries"]),
        reduced_ids=[str(x) for x in read_json(d / HELDOUT_FILES["reduced"])],
        removed_ids=[str(x) for x in read_json(d / HELDOUT_FILES["removed"])],
        stats=stats,
    )


def load_targets(directory: PathLike = DEFAULT_HELDOUT_DIR) -> List[HeldoutTarget]:
    return [HeldoutTarget.from_dict(t) for t in read_json(Path(directory) / HELDOUT_FILES["targets"])]


__all__ = [
    "AnnotationDecision",
    "CONDITIONS",
    "DEFAULT_HELDOUT_DIR",
    "HeldoutBenchmark",
    "HeldoutTarget",
    "LeakageReport",
    "NEW_METHOD",
    "NEW_PROBLEM",
    "REDUCTIONS",
    "acronyms",
    "annotation_agreement",
    "annotation_columns",
    "apply_annotations",
    "build_heldout_benchmark",
    "check_leakage",
    "enumerate_target_candidates",
    "existing_endpoint_domain",
    "export_annotation_sheet",
    "generate_queries",
    "load_heldout_benchmark",
    "load_targets",
    "parse_judgement",
    "read_annotations",
    "reduce_corpus",
    "save_heldout_benchmark",
    "select_targets",
    "supporting_evidence",
]
