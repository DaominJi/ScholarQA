"""Budget packing shared by every budgeted baseline (Section 5.3).

Each budgeted retriever ranks candidate units and *packs its selected
deduplicated units up to B tokens* (Section 5.3, "Baselines"). Packing walks
the ranking once:

* a unit whose key or (whitespace-normalized) text was already packed is a
  duplicate and is skipped;
* a unit is added only if the total context cost stays ``<= B``; otherwise
  it is skipped and the walk continues, so that smaller units further down
  the ranking can still use the remaining budget (skip-and-continue);
* the cost of a unit is the token count of its rendered block
  (:meth:`Passage.render`) plus the separator joining it to the context.

The rendered context uses the ``[paper_id] title`` citation style of
ScholarPath's context assembly, and ``RetrievalResult.ranked_papers`` lists
the distinct source papers in rank (= packing) order, which is what Recall@B
and NDCG@B are computed over (Table 2, Fig. 5).

After rendering, the context is re-tokenized as a whole and trailing units
are dropped should the whole-string count exceed B (possible only with
sub-word tokenizers whose counts are not additive), so the budget is never
violated: ``count_tokens(result.context) <= B`` always holds.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from scholarpath.baselines.common import Candidate, CandidateLike, Passage, as_candidate
from scholarpath.retrieval.result import ESTABLISHED, RetrievalResult, RetrievedUnit
from scholarpath.text import count_tokens

SEPARATOR = "\n\n"


class BudgetPacker:
    """Incremental skip-and-continue packer with deduplication.

    Several groups of candidates can be packed one after the other into the
    same budget (LightRAG packs entities, relations and text units with
    per-group token limits); deduplication state is shared across groups.
    """

    def __init__(self, budget: int, dedup: bool = True, separator: str = SEPARATOR):
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = int(budget)
        self.dedup = bool(dedup)
        self.separator = separator
        self._sep_cost = count_tokens(separator)
        self.units: List[RetrievedUnit] = []
        self.used = 0
        self._keys: Dict[str, None] = {}
        self._texts: Dict[str, None] = {}
        self.offered = 0
        self.skipped_duplicate = 0
        self.skipped_budget = 0

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def marginal_cost(self, passage: Passage) -> int:
        return passage.cost + (self._sep_cost if self.units else 0)

    def is_duplicate(self, passage: Passage) -> bool:
        return passage.key in self._keys or passage.dedup_key in self._texts

    def offer(
        self,
        candidate: CandidateLike,
        group_used: int = 0,
        group_limit: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> int:
        """Try to add one unit; returns the tokens added (0 if skipped)."""
        cand: Candidate = as_candidate(candidate)
        p = cand.passage
        self.offered += 1
        if self.dedup and self.is_duplicate(p):
            self.skipped_duplicate += 1
            return 0
        cost = self.marginal_cost(p)
        if self.used + cost > self.budget or (group_limit is not None and group_used + cost > group_limit):
            self.skipped_budget += 1
            return 0
        self.units.append(
            RetrievedUnit(
                key=p.key,
                kind=p.kind,
                text=p.render(),
                tokens=cost,
                score=float(cand.score),
                papers=list(p.papers),
                status=ESTABLISHED,
                metadata={"rank": rank, "source": cand.source, **p.metadata},
            )
        )
        self.used += cost
        self._keys[p.key] = None
        self._texts[p.dedup_key] = None
        return cost

    def extend(
        self,
        candidates: Iterable[CandidateLike],
        limit: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> int:
        """Walk a ranking and pack what fits; ``limit`` caps this group's tokens.

        Returns the number of tokens this group added.
        """
        group_used = 0
        cap = self.budget if limit is None else min(self.budget, int(limit))
        for rank, c in enumerate(candidates):
            if max_candidates is not None and rank >= max_candidates:
                break
            if self.remaining <= 0 or group_used >= cap:
                break
            group_used += self.offer(c, group_used=group_used, group_limit=limit, rank=rank)
        return group_used

    def result(
        self,
        query_id: str,
        titles: Optional[Mapping[str, str]] = None,
        ranked_papers: Optional[Sequence[str]] = None,
        stats: Optional[Dict[str, Any]] = None,
        latency_s: float = 0.0,
    ) -> RetrievalResult:
        stats = dict(stats or {})
        stats.update(
            {
                "candidates_offered": self.offered,
                "skipped_duplicate": self.skipped_duplicate,
                "skipped_budget": self.skipped_budget,
            }
        )
        return build_result(query_id, self.units, self.budget, titles=titles, ranked_papers=ranked_papers,
                            stats=stats, latency_s=latency_s, separator=self.separator)


def pack_to_budget(
    candidates: Iterable[CandidateLike],
    budget: int,
    dedup: bool = True,
    max_candidates: Optional[int] = None,
) -> List[RetrievedUnit]:
    """Pack ranked candidates into at most ``budget`` tokens (see module docstring).

    ``candidates`` may be :class:`~scholarpath.baselines.common.Candidate`
    objects, bare :class:`~scholarpath.baselines.common.Passage` objects, or
    ``(passage, score)`` pairs, in rank order.
    """
    packer = BudgetPacker(budget, dedup=dedup)
    packer.extend(candidates, max_candidates=max_candidates)
    return packer.units


def render_context(units: Sequence[RetrievedUnit], separator: str = SEPARATOR) -> str:
    return separator.join(u.text for u in units)


def build_result(
    query_id: str,
    units: List[RetrievedUnit],
    budget: int,
    titles: Optional[Mapping[str, str]] = None,
    ranked_papers: Optional[Sequence[str]] = None,
    stats: Optional[Dict[str, Any]] = None,
    latency_s: float = 0.0,
    separator: str = SEPARATOR,
) -> RetrievalResult:
    """Render packed units into a :class:`RetrievalResult` (context, sources, ranking).

    ``ranked_papers`` overrides the default ranking (distinct source papers of
    the units in packing order); papers not present in the context are
    dropped from the override.
    """
    units = list(units)
    context = render_context(units, separator)
    tokens = count_tokens(context)
    while units and tokens > budget:  # strict guarantee for non-additive tokenizers
        units.pop()
        context = render_context(units, separator)
        tokens = count_tokens(context)
    in_context = RetrievalResult.papers_in_order(units)
    if ranked_papers is not None:
        present = set(in_context)
        ranking = [p for p in dict.fromkeys(ranked_papers) if p in present]
        ranking += [p for p in in_context if p not in set(ranking)]
    else:
        ranking = in_context
    titles = titles or {}
    sources = [
        {"paper_id": pid, "title": titles.get(pid, ""), "status": ESTABLISHED, "candidate": None}
        for pid in ranking
    ]
    return RetrievalResult(
        query_id=query_id,
        units=units,
        context=context,
        context_tokens=tokens,
        budget=int(budget),
        ranked_papers=ranking,
        candidates=[],
        sources=sources,
        latency_s=float(latency_s),
        stats=dict(stats or {}),
    )
