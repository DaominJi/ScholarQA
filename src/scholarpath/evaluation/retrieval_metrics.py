"""Retrieval accuracy under a context token budget: Recall@B and NDCG@B (Section 5.3).

"For retrieval accuracy, we report Recall@B (R) and NDCG@B (N) over gold
source papers" (Section 5.3; Table 2, Fig. 5). A method's ranked output at
budget B is the list of *distinct source papers* of the context units it
selected, in selection order, restricted to units whose total token cost fits
in B (Definition 2: C(S) <= B). Gains are binary: a paper is relevant iff it
is a gold source paper of the query that is aligned to the corpus.

Unaligned gold citations -- gold references that could not be matched to a
corpus paper (Appendix C.2) -- "are retained in the denominator so that
reported metrics reflect end-to-end corpus coverage": they count towards
``num_gold`` (``Query.num_gold``) but can never be retrieved.

Definitions (``L`` = number of ranked distinct papers within B,
``G`` = aligned gold set, ``n_gold = |G| + #unaligned``)::

    Recall@B = |{p_1..p_L} ∩ G| / n_gold
    DCG@B    = sum_{i=1..L} [p_i ∈ G] / log2(i + 1)
    IDCG@B   = sum_{i=1..min(n_gold, D)} 1 / log2(i + 1)
    NDCG@B   = DCG@B / IDCG@B

The ideal depth ``D`` defaults to ``n_gold`` ("all_gold": the ideal ranking
places every gold paper first, as trec_eval's ``ndcg`` does for a run equal
to the within-budget list), so NDCG@B, like Recall@B, grows as a larger B
admits more gold papers (Fig. 5) and a system cannot reach 1.0 by retrieving
very few papers. ``ideal="list"`` uses ``D = L`` instead (NDCG@k with k equal
to the list length).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

from scholarpath.retrieval.result import RetrievalResult
from scholarpath.types import Answer, Query, unique

RankedInput = Union[RetrievalResult, Answer, Sequence[str]]


def _dedup(ranked: Iterable[str]) -> List[str]:
    return unique([str(p) for p in ranked if p is not None and str(p) != ""])


def recall_at_budget(ranked_papers: Sequence[str], gold: Iterable[str], num_gold: Optional[int] = None) -> Optional[float]:
    """Recall@B in [0, 1] (``None`` when the query has no gold papers).

    ``ranked_papers`` are the papers within the budget (duplicates ignored);
    ``num_gold`` is the denominator including unaligned gold citations
    (``Query.num_gold``); it defaults to ``len(gold)`` and must not be smaller.
    """
    gold_set: Set[str] = {str(g) for g in gold}
    n = len(gold_set) if num_gold is None else int(num_gold)
    if n < len(gold_set):
        raise ValueError(f"num_gold={n} is smaller than the number of aligned gold papers ({len(gold_set)})")
    if n <= 0:
        return None
    hits = sum(1 for p in _dedup(ranked_papers) if p in gold_set)
    return hits / n


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain sum_i g_i / log2(i + 1) with 1-based ranks."""
    return float(sum(g / math.log2(i + 2) for i, g in enumerate(gains)))


def ideal_dcg(num_relevant: int, depth: Optional[int] = None) -> float:
    """DCG of the ideal binary ranking with ``num_relevant`` relevant items, cut at ``depth``."""
    k = num_relevant if depth is None else min(num_relevant, int(depth))
    return dcg([1.0] * max(0, k))


def ndcg_at_budget(
    ranked_papers: Sequence[str],
    gold: Iterable[str],
    num_gold: Optional[int] = None,
    ideal: str = "all_gold",
) -> Optional[float]:
    """NDCG@B in [0, 1] with binary gains (``None`` when there is no gold paper).

    ``ideal="all_gold"`` (default): IDCG over ``num_gold`` positions (unaligned
    gold included); ``ideal="list"``: IDCG truncated at the length of the
    ranked list. See the module docstring.
    """
    if ideal not in {"all_gold", "list"}:
        raise ValueError(f"ideal must be 'all_gold' or 'list', got {ideal!r}")
    gold_set: Set[str] = {str(g) for g in gold}
    n = len(gold_set) if num_gold is None else int(num_gold)
    if n < len(gold_set):
        raise ValueError(f"num_gold={n} is smaller than the number of aligned gold papers ({len(gold_set)})")
    if n <= 0:
        return None
    ranked = _dedup(ranked_papers)
    gains = [1.0 if p in gold_set else 0.0 for p in ranked]
    idcg = ideal_dcg(n, None if ideal == "all_gold" else len(ranked))
    if idcg <= 0.0:
        return 0.0
    return dcg(gains) / idcg


def papers_within_budget(result: RetrievalResult, budget: Optional[int] = None) -> List[str]:
    """Distinct source papers of ``result`` that fit in the budget, in rank order.

    ``result.ranked_papers`` is used when present (the systems already pack at
    most B tokens). When ``budget`` is given and the result exceeds it, the
    units are re-packed greedily in selection order (a unit is kept iff the
    running marginal-token total stays <= B) and their papers are returned;
    this enforces Definition 2 for results produced without a budget.
    """
    over = budget is not None and result.context_tokens > budget and bool(result.units)
    if result.ranked_papers and not over:
        return _dedup(result.ranked_papers)
    if not result.units:
        return _dedup(result.ranked_papers)
    if budget is None:
        return RetrievalResult.papers_in_order(result.units)
    used = 0
    kept = []
    for unit in result.units:
        cost = max(0, int(unit.tokens))
        if used + cost > budget:
            continue
        used += cost
        kept.append(unit)
    return RetrievalResult.papers_in_order(kept)


def ranked_papers_of(output: RankedInput, budget: Optional[int] = None) -> List[str]:
    """Ranked within-budget papers of a :class:`RetrievalResult`, an :class:`Answer`, or a list."""
    if isinstance(output, RetrievalResult):
        return papers_within_budget(output, budget)
    if isinstance(output, Answer):
        return _dedup(output.retrieved_papers)
    if isinstance(output, str):
        raise TypeError("Expected a sequence of paper ids, got a string")
    return _dedup(output)


# ---------------------------------------------------------------------------
# Benchmark-level evaluation (Table 2, Fig. 5)
# ---------------------------------------------------------------------------

#: Canonical benchmark keys and the column names used in Tables 2-3.
BENCHMARK_DISPLAY = {
    "researchqa": "ResearchQA",
    "scholarqabench": "ScholarQA",
    "taxobench": "TaxoBench",
    "chime": "CHIME",
}


def canonical_benchmark(name: str, merge_scholarqa: bool = True) -> str:
    """Canonical benchmark key: lower-case alphanumerics; ScholarQA-CS/-Multi -> ``scholarqabench``."""
    key = "".join(ch for ch in (name or "").lower() if ch.isalnum())
    if merge_scholarqa and key.startswith("scholarqa"):
        return "scholarqabench"
    return key or "unknown"


@dataclass
class RetrievalEvaluation:
    """Per-query rows and macro averages (percent) of Recall@B / NDCG@B."""

    rows: List[Dict[str, Any]] = field(default_factory=list)
    by_benchmark: Dict[str, Dict[str, float]] = field(default_factory=dict)
    macro: Dict[str, float] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)  # queries without gold papers
    missing: List[str] = field(default_factory=list)  # queries without a system output

    def table_row(self) -> Dict[str, float]:
        """Flat ``{"<benchmark>.R": .., "<benchmark>.N": .., "macro.R": ..}`` (Table 2 / reporting)."""
        out: Dict[str, float] = {}
        for bench, vals in self.by_benchmark.items():
            out[f"{bench}.R"] = vals["R"]
            out[f"{bench}.N"] = vals["N"]
        if self.macro:
            out["macro.R"] = self.macro["R"]
            out["macro.N"] = self.macro["N"]
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"by_benchmark": self.by_benchmark, "macro": self.macro, "skipped": self.skipped,
                "missing": self.missing, "n_rows": len(self.rows)}


def evaluate_retrieval(
    results: Mapping[str, RankedInput],
    queries: Sequence[Query],
    budget: Optional[int] = None,
    ideal: str = "all_gold",
    group: Optional[Callable[[Query], str]] = None,
    missing_as_zero: bool = True,
) -> RetrievalEvaluation:
    """Recall@B and NDCG@B for every query, averaged per benchmark and across benchmarks.

    ``results`` maps ``query_id`` to the system output. Queries without gold
    papers are skipped (the metrics are undefined); queries without an output
    count as zero when ``missing_as_zero`` (a failed run retrieves nothing).
    Per-benchmark values are means over queries; the ``macro`` row is the mean
    of the per-benchmark means (Fig. 5 "macro Recall@B"). All values are in
    percent. ``group`` maps a query to its benchmark (default:
    :func:`canonical_benchmark` of ``Query.benchmark``).
    """
    group = group or (lambda q: canonical_benchmark(q.benchmark))
    ev = RetrievalEvaluation()
    per_bench: Dict[str, List[Dict[str, Any]]] = {}
    for q in queries:
        if q.num_gold <= 0:
            ev.skipped.append(q.query_id)
            continue
        out = results.get(q.query_id)
        if out is None:
            ev.missing.append(q.query_id)
            if not missing_as_zero:
                continue
            ranked: List[str] = []
        else:
            ranked = ranked_papers_of(out, budget)
        r = recall_at_budget(ranked, q.gold_papers, q.num_gold)
        n = ndcg_at_budget(ranked, q.gold_papers, q.num_gold, ideal=ideal)
        tokens = out.context_tokens if isinstance(out, RetrievalResult) else (
            out.context_tokens if isinstance(out, Answer) else None)
        bench = group(q)
        gold = set(q.gold_papers)
        row = {
            "query_id": q.query_id,
            "benchmark": bench,
            "num_gold": q.num_gold,
            "num_aligned_gold": len(gold),
            "num_retrieved": len(ranked),
            "hits": sum(1 for p in ranked if p in gold),
            "R": 100.0 * float(r or 0.0),
            "N": 100.0 * float(n or 0.0),
            "context_tokens": tokens,
            "budget": budget,
            "over_budget": bool(budget is not None and tokens is not None and tokens > budget),
            "missing": out is None,
        }
        ev.rows.append(row)
        per_bench.setdefault(bench, []).append(row)
    for bench, rows in per_bench.items():
        ev.by_benchmark[bench] = {
            "R": sum(r["R"] for r in rows) / len(rows),
            "N": sum(r["N"] for r in rows) / len(rows),
            "n": float(len(rows)),
        }
    if ev.by_benchmark:
        vals = list(ev.by_benchmark.values())
        ev.macro = {"R": sum(v["R"] for v in vals) / len(vals), "N": sum(v["N"] for v in vals) / len(vals),
                    "n": float(sum(v["n"] for v in vals))}
    return ev


__all__ = [
    "BENCHMARK_DISPLAY",
    "RetrievalEvaluation",
    "canonical_benchmark",
    "dcg",
    "evaluate_retrieval",
    "ideal_dcg",
    "ndcg_at_budget",
    "papers_within_budget",
    "ranked_papers_of",
    "recall_at_budget",
]
