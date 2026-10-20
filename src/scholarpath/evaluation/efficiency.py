"""Per-query efficiency of end-to-end QA (Table 5, Section 5.3.3).

Table 5 reports, per query, the number of online LLM calls, the tokens
consumed (input + output, in thousands), the monetary cost in dollars, and
the end-to-end latency in seconds. Costs use the pricing of the shared
backbone, DeepSeek-V4.0-Flash ($0.14 / 1M input and $0.28 / 1M output
tokens, Appendix C.3; ``LLMConfig.price_*_per_m``).

:class:`EfficiencyTracker` measures one query at a time: it snapshots the
system's :class:`~scholarpath.llm.CostMeter` before the call, takes
:meth:`CostMeter.diff` afterwards, and times the call with a wall clock.
Components that are not online system cost are excluded: ``judge``
(evaluation), ``index_build`` and ``link_training`` (offline indexing of
LightRAG / RAPTOR and temporal link-predictor training, Appendix B.5), and
``benchmark`` (held-out benchmark construction). Queries must be measured
sequentially on a meter that no concurrent work bills to.

Note on caching: calls served from the LLM response cache are still counted
(``bill_cached_calls``) but take ~0 s of wall clock; for latency numbers run
with the cache disabled (``--set llm.cache_dir=null``). The sum of the
per-call latencies recorded by the meter (which replays the original latency
of cached calls) is reported as ``llm_latency_s``.

Unbudgeted agents that call an external tool (e.g. the Codex CLI) and fill
``Answer.llm_calls / input_tokens / output_tokens / latency_s`` themselves
can be recorded with :meth:`EfficiencyTracker.record_answer`.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar

from scholarpath.config import LLMConfig
from scholarpath.llm import CostMeter, UsageRecord
from scholarpath.types import Answer

T = TypeVar("T")

#: Components never counted as online system cost.
EXCLUDED_COMPONENTS = ("judge", "index_build", "link_training", "benchmark")
#: Table 5 columns.
TABLE5_COLUMNS = ("Calls", "Tokens (K)", "Cost ($)", "Latency (s)")


@dataclass
class QueryEfficiency:
    """LLM usage and latency of answering one query."""

    query_id: str
    system: str = ""
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0  # wall clock
    llm_latency_s: float = 0.0  # sum of per-call latencies recorded by the meter
    components: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tokens"] = self.tokens
        return d


def price(input_tokens: int, output_tokens: int, price_input_per_m: float, price_output_per_m: float) -> float:
    """Dollar cost of a token count at per-1M-token prices."""
    return (input_tokens * price_input_per_m + output_tokens * price_output_per_m) / 1_000_000.0


class EfficiencyTracker:
    """Collects :class:`QueryEfficiency` records for one system (one Table 5 row)."""

    def __init__(
        self,
        meter: CostMeter,
        system: str = "",
        price_input_per_m: Optional[float] = None,
        price_output_per_m: Optional[float] = None,
        exclude: Sequence[str] = EXCLUDED_COMPONENTS,
    ):
        self.meter = meter
        self.system = system
        # Prices default to the meter's (the config's DeepSeek-V4.0-Flash pricing).
        self.price_input_per_m = meter.price_input_per_m if price_input_per_m is None else float(price_input_per_m)
        self.price_output_per_m = meter.price_output_per_m if price_output_per_m is None else float(price_output_per_m)
        self.exclude = tuple(exclude)
        self.records: List[QueryEfficiency] = []

    @classmethod
    def from_config(cls, meter: CostMeter, llm_config: LLMConfig, system: str = "", **kw: Any) -> "EfficiencyTracker":
        """Tracker priced with ``llm_config.price_input_per_m`` / ``price_output_per_m``."""
        return cls(meter, system=system, price_input_per_m=llm_config.price_input_per_m,
                   price_output_per_m=llm_config.price_output_per_m, **kw)

    # ----------------------------------------------------------- recording
    def _record(self, query_id: str, usage: Dict[str, UsageRecord], wall: float) -> QueryEfficiency:
        kept = {k: v for k, v in usage.items() if k not in self.exclude}
        total = UsageRecord()
        for rec in kept.values():
            total.add(rec)
        rec = QueryEfficiency(
            query_id=query_id,
            system=self.system,
            calls=total.calls,
            cached_calls=total.cached_calls,
            input_tokens=total.input_tokens,
            output_tokens=total.output_tokens,
            cost_usd=price(total.input_tokens, total.output_tokens, self.price_input_per_m, self.price_output_per_m),
            latency_s=float(wall),
            llm_latency_s=total.latency_s,
            components={k: v.to_dict() for k, v in kept.items()},
        )
        self.records.append(rec)
        return rec

    @contextlib.contextmanager
    def track(self, query_id: str) -> Iterator[Dict[str, QueryEfficiency]]:
        """Context manager measuring the enclosed block; the record is in ``holder["record"]``."""
        holder: Dict[str, QueryEfficiency] = {}
        before = self.meter.snapshot()
        t0 = time.perf_counter()
        try:
            yield holder
        finally:
            wall = time.perf_counter() - t0
            holder["record"] = self._record(query_id, self.meter.diff(before), wall)

    def measure(self, query_id: str, fn: Callable[..., T], *args: Any, annotate: bool = True, **kwargs: Any) -> T:
        """Run ``fn(*args, **kwargs)`` for one query and record its usage.

        With ``annotate``, an :class:`Answer` result whose usage fields are
        unset gets them filled from the measurement.
        """
        with self.track(query_id) as holder:
            result = fn(*args, **kwargs)
        if annotate and isinstance(result, Answer):
            annotate_answer(result, holder["record"])
        return result

    def record_answer(self, answer: Answer, latency_s: Optional[float] = None) -> QueryEfficiency:
        """Record usage reported by the answer itself (external agents that bypass the meter)."""
        rec = QueryEfficiency(
            query_id=answer.query_id,
            system=self.system,
            calls=int(answer.llm_calls),
            input_tokens=int(answer.input_tokens),
            output_tokens=int(answer.output_tokens),
            cost_usd=price(answer.input_tokens, answer.output_tokens, self.price_input_per_m, self.price_output_per_m),
            latency_s=float(answer.latency_s if latency_s is None else latency_s),
        )
        self.records.append(rec)
        return rec

    # ----------------------------------------------------------- reporting
    def summary(self) -> Dict[str, Optional[float]]:
        return summarize_efficiency(self.records)

    def rows(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.records]


def annotate_answer(answer: Answer, record: QueryEfficiency) -> Answer:
    """Fill unset usage fields of ``answer`` from a measurement (in place)."""
    if not answer.llm_calls:
        answer.llm_calls = record.calls
    if not answer.input_tokens:
        answer.input_tokens = record.input_tokens
    if not answer.output_tokens:
        answer.output_tokens = record.output_tokens
    if not answer.cost_usd:
        answer.cost_usd = record.cost_usd
    if not answer.latency_s:
        answer.latency_s = record.latency_s
    return answer


def measure(
    fn: Callable[..., T],
    meter: CostMeter,
    *args: Any,
    query_id: str = "",
    exclude: Sequence[str] = EXCLUDED_COMPONENTS,
    price_input_per_m: Optional[float] = None,
    price_output_per_m: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[T, QueryEfficiency]:
    """Measure a single call: returns ``(result, QueryEfficiency)``."""
    tracker = EfficiencyTracker(meter, price_input_per_m=price_input_per_m,
                                price_output_per_m=price_output_per_m, exclude=exclude)
    result = tracker.measure(query_id, fn, *args, annotate=False, **kwargs)
    return result, tracker.records[-1]


def summarize_efficiency(records: Sequence[QueryEfficiency]) -> Dict[str, Optional[float]]:
    """Table 5 row: per-query means of calls, tokens (K), cost ($) and latency (s)."""
    n = len(records)
    if n == 0:
        return {c: None for c in TABLE5_COLUMNS}

    def mean(f: Callable[[QueryEfficiency], float]) -> float:
        return sum(f(r) for r in records) / n

    return {
        "Calls": mean(lambda r: r.calls),
        "Tokens (K)": mean(lambda r: r.tokens) / 1000.0,
        "Cost ($)": mean(lambda r: r.cost_usd),
        "Latency (s)": mean(lambda r: r.latency_s),
        "input_tokens": mean(lambda r: r.input_tokens),
        "output_tokens": mean(lambda r: r.output_tokens),
        "cached_calls": mean(lambda r: r.cached_calls),
        "llm_latency_s": mean(lambda r: r.llm_latency_s),
        "total_cost_usd": sum(r.cost_usd for r in records),
        "n": float(n),
    }


__all__ = [
    "EXCLUDED_COMPONENTS",
    "EfficiencyTracker",
    "QueryEfficiency",
    "TABLE5_COLUMNS",
    "annotate_answer",
    "measure",
    "price",
    "summarize_efficiency",
]
