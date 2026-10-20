"""LLM cost accounting.

Every LLM call is attributed to a *component* so that the cost breakdowns of
the paper can be reproduced:

* taxonomy construction (Appendix B.2, Eq. 15):
  ``acquisition``, ``auditing``, ``discovery``, ``correction``,
  ``relabeling`` (plus ``full`` / ``selection`` for the baselines);
* query answering (Table 5): ``query_understanding``, ``composition``,
  ``rerank``, ``agent``, ...;
* evaluation: ``judge`` (never counted towards system cost).

The meter is thread-safe and supports nested scopes::

    with meter.scope("auditing"):
        llm.complete(...)     # billed to "auditing" unless the request says otherwise
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional


# Components of C_LLM (Eq. 15).
BUILD_COMPONENTS = ("acquisition", "auditing", "discovery", "correction", "relabeling")


@dataclass
class UsageRecord:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0  # recorded latency of all calls (cached calls replay their original latency)
    cached_latency_s: float = 0.0  # part of latency_s replayed from the response cache

    def add(self, other: "UsageRecord") -> None:
        self.calls += other.calls
        self.cached_calls += other.cached_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.latency_s += other.latency_s
        self.cached_latency_s += other.cached_latency_s

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


class CostMeter:
    """Accumulates LLM usage per component."""

    def __init__(self, price_input_per_m: float = 0.14, price_output_per_m: float = 0.28):
        self.price_input_per_m = float(price_input_per_m)
        self.price_output_per_m = float(price_output_per_m)
        self._lock = threading.RLock()
        self._records: Dict[str, UsageRecord] = {}
        self._local = threading.local()

    # ------------------------------------------------------------- scoping
    def _stack(self) -> List[str]:
        st = getattr(self._local, "stack", None)
        if st is None:
            st = []
            self._local.stack = st
        return st

    @property
    def current_component(self) -> str:
        st = self._stack()
        return st[-1] if st else "default"

    @contextlib.contextmanager
    def scope(self, component: str) -> Iterator[None]:
        st = self._stack()
        st.append(component)
        try:
            yield
        finally:
            st.pop()

    # ----------------------------------------------------------- recording
    def price(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.price_input_per_m + output_tokens * self.price_output_per_m
        ) / 1_000_000.0

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_s: float = 0.0,
        component: Optional[str] = None,
        cached: bool = False,
    ) -> float:
        comp = component or self.current_component
        cost = self.price(input_tokens, output_tokens)
        with self._lock:
            rec = self._records.setdefault(comp, UsageRecord())
            rec.calls += 1
            rec.cached_calls += int(bool(cached))
            rec.input_tokens += int(input_tokens)
            rec.output_tokens += int(output_tokens)
            rec.cost_usd += cost
            rec.latency_s += float(latency_s)
            if cached:
                rec.cached_latency_s += float(latency_s)
        return cost

    # ------------------------------------------------------------- queries
    def component(self, name: str) -> UsageRecord:
        with self._lock:
            rec = self._records.get(name)
            out = UsageRecord()
            if rec is not None:
                out.add(rec)
            return out

    def total(self, exclude: tuple = ("judge",)) -> UsageRecord:
        out = UsageRecord()
        with self._lock:
            for name, rec in self._records.items():
                if name in exclude:
                    continue
                out.add(rec)
        return out

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {k: v.to_dict() for k, v in sorted(self._records.items())}

    def diff(self, before: Dict[str, Dict[str, float]]) -> Dict[str, UsageRecord]:
        """Usage accumulated since ``before = meter.snapshot()``."""
        now = self.snapshot()
        out: Dict[str, UsageRecord] = {}
        for name, d in now.items():
            b = before.get(name, {})
            out[name] = UsageRecord(
                calls=int(d["calls"] - b.get("calls", 0)),
                cached_calls=int(d["cached_calls"] - b.get("cached_calls", 0)),
                input_tokens=int(d["input_tokens"] - b.get("input_tokens", 0)),
                output_tokens=int(d["output_tokens"] - b.get("output_tokens", 0)),
                cost_usd=float(d["cost_usd"] - b.get("cost_usd", 0.0)),
                latency_s=float(d["latency_s"] - b.get("latency_s", 0.0)),
                cached_latency_s=float(d.get("cached_latency_s", 0.0) - b.get("cached_latency_s", 0.0)),
            )
        return {k: v for k, v in out.items() if v.calls > 0}

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def merge(self, other: "CostMeter") -> None:
        with self._lock:
            for name, rec in other._records.items():
                self._records.setdefault(name, UsageRecord()).add(rec)


def sum_usage(records: Dict[str, UsageRecord], exclude: tuple = ("judge",)) -> UsageRecord:
    out = UsageRecord()
    for name, rec in records.items():
        if name not in exclude:
            out.add(rec)
    return out
