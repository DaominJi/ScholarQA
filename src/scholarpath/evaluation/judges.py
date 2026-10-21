"""LLM judge panel (Section 5.3, Appendix C.3).

Held-out hypothesis answers are scored by three state-of-the-art LLMs
(GPT-6.0-Sol, Opus-5.5, Grok-4.7) and the reported number is the average over
the three judges; ResearchQA rubric coverage and ScholarQA-CS correctness use
the single judge model of their official scripts (``gpt-4.1-mini`` and
``gpt-4-turbo``), and ScholarQA-Multi uses Prometheus-2 7B. "The judge models
and prompts are fixed across all systems" (Appendix C.3).

A :class:`JudgePanel` holds one :class:`Judge` per model. Every judge owns a
private :class:`~scholarpath.llm.CostMeter`, so judge spending can be reported
separately and never mixes with the system cost of Table 5; in addition, every
judge request is billed to the ``"judge"`` component, which
:meth:`CostMeter.total` excludes by default even when a judge shares the
system's client. Judges always run at temperature 0 and reuse the on-disk LLM
response cache, so re-scoring the same outputs is free and deterministic.

Example (paper setting, OpenAI-compatible endpoints such as OpenRouter)::

    panel = JudgePanel.from_specs([
        {"name": "gpt-6.0-sol", "llm": {"provider": "openai", "model": "<provider id>",
                                         "base_url": "https://openrouter.ai/api/v1",
                                         "api_key_env": "OPENROUTER_API_KEY"}},
        ...
    ])

or ``JudgePanel.paper_default()`` with ``$SCHOLARPATH_JUDGE_BASE_URL`` /
``$SCHOLARPATH_JUDGE_API_KEY`` set. ``JudgePanel.mock()`` gives an offline
panel for dry runs.
"""

from __future__ import annotations

import dataclasses
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, TypeVar, Union

from scholarpath.config import LLMConfig
from scholarpath.llm import CostMeter, LLMClient, UsageRecord, build_llm

T = TypeVar("T")

JUDGE_COMPONENT = "judge"

#: The three hypothesis judges of Table 4 / Appendix C.3 (paper names; set the
#: provider-specific model identifiers through :class:`JudgeSpec`).
PAPER_JUDGES = ("gpt-6.0-sol", "opus-5.5", "grok-4.7")
#: Official judge of ResearchQA ``compute_coverage.py``.
RESEARCHQA_JUDGE_MODEL = "gpt-4.1-mini"
#: Hard-coded judge of ScholarQABench ``scholarqacs_rubric.py``.
SCHOLARQA_CS_JUDGE_MODEL = "gpt-4-turbo"
#: Prometheus-2 7B used for ScholarQA-Multi (Appendix C.3).
PROMETHEUS_MODEL = "prometheus-eval/prometheus-7b-v2.0"
#: Default on-disk cache for judge responses (kept apart from system calls).
DEFAULT_JUDGE_CACHE = "cache/judge"


@dataclass
class JudgeSpec:
    """Name + backend configuration of one judge model.

    Judge prices default to 0 (unknown): set ``price_input_per_m`` /
    ``price_output_per_m`` in the spec to report judge spending in dollars
    (token counts are always reported). Judge spending never enters the
    system cost of Table 5 either way.
    """

    name: str
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(
        provider="mock", model="mock-judge", cache_dir=None, price_input_per_m=0.0, price_output_per_m=0.0))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "JudgeSpec":
        """From ``{"name": ..., "llm": {...LLMConfig fields...}}``.

        Fields left out of ``llm`` take judge defaults rather than the system
        defaults of :class:`~scholarpath.config.LLMConfig`: ``model`` = the
        judge name, ``cache_dir`` = :data:`DEFAULT_JUDGE_CACHE` (``None`` for
        the mock provider) and prices 0. Without ``llm`` the offline mock
        judge of the dataclass default is used.
        """
        name = str(d["name"])
        llm = d.get("llm")
        if isinstance(llm, LLMConfig):
            return cls(name=name, llm=llm)
        if not llm:
            return cls(name=name)
        fields = dict(llm)
        fields.setdefault("model", name)
        fields.setdefault("cache_dir", None if fields.get("provider", "mock") == "mock" else DEFAULT_JUDGE_CACHE)
        fields.setdefault("price_input_per_m", 0.0)
        fields.setdefault("price_output_per_m", 0.0)
        return cls(name=name, llm=LLMConfig(**fields))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "llm": dataclasses.asdict(self.llm)}


def build_judge_llm(spec: JudgeSpec, meter: Optional[CostMeter] = None) -> LLMClient:
    """Instantiate a judge backend: temperature 0, private cost meter, cached.

    For OpenAI-compatible judges without an explicit ``base_url`` the endpoint
    is read from ``$SCHOLARPATH_JUDGE_BASE_URL`` (falling back to the system
    endpoint variables of :class:`~scholarpath.llm.openai_compat.OpenAICompatibleLLM`).
    """
    cfg = dataclasses.replace(spec.llm, temperature=0.0)
    if cfg.provider != "mock" and cfg.base_url is None and os.environ.get("SCHOLARPATH_JUDGE_BASE_URL"):
        cfg = dataclasses.replace(cfg, base_url=os.environ["SCHOLARPATH_JUDGE_BASE_URL"])
    meter = meter or CostMeter(cfg.price_input_per_m, cfg.price_output_per_m)
    return build_llm(cfg, meter)


@dataclass
class Judge:
    """One judge model: a name, its LLM client and its private cost meter."""

    name: str
    llm: LLMClient

    @property
    def meter(self) -> CostMeter:
        return self.llm.meter

    @property
    def model(self) -> str:
        return self.llm.model

    def usage(self) -> UsageRecord:
        """Usage billed to the ``"judge"`` component of this judge's meter."""
        return self.meter.component(JUDGE_COMPONENT)


JudgeLike = Union["JudgePanel", Judge, LLMClient]


class JudgePanel:
    """A fixed list of judges whose scores are averaged (Table 4)."""

    def __init__(self, judges: Sequence[Judge], max_workers: int = 8):
        if not judges:
            raise ValueError("A JudgePanel needs at least one judge.")
        names = [j.name for j in judges]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate judge names: {names}")
        self.judges: List[Judge] = list(judges)
        self.max_workers = int(max_workers)

    # ------------------------------------------------------------ builders
    @classmethod
    def from_specs(cls, specs: Sequence[Union[JudgeSpec, Mapping[str, Any]]], max_workers: int = 8) -> "JudgePanel":
        built = []
        for s in specs:
            spec = s if isinstance(s, JudgeSpec) else JudgeSpec.from_dict(s)
            built.append(Judge(spec.name, build_judge_llm(spec)))
        return cls(built, max_workers=max_workers)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "JudgePanel":
        """Load ``{"judges": [{"name": ..., "llm": {...}}, ...]}`` from YAML or JSON."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        specs = data.get("judges", data) if isinstance(data, Mapping) else data
        return cls.from_specs(specs)

    @classmethod
    def paper_default(
        cls,
        models: Optional[Mapping[str, str]] = None,
        base_url: Optional[str] = None,
        api_key_env: str = "SCHOLARPATH_JUDGE_API_KEY",
        cache_dir: Optional[str] = DEFAULT_JUDGE_CACHE,
        provider: str = "openai",
        prices: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> "JudgePanel":
        """The three judges of the paper behind one OpenAI-compatible endpoint.

        ``models`` maps each paper judge name to the provider's model id
        (identity by default); ``prices`` maps it to ``(input, output)`` USD
        per 1M tokens (0 when not given, i.e. judge $ is not reported).
        """
        models = dict(models or {})
        prices = dict(prices or {})
        specs = []
        for name in PAPER_JUDGES:
            p_in, p_out = prices.get(name, (0.0, 0.0))
            specs.append(JudgeSpec(name, LLMConfig(
                provider=provider, model=models.get(name, name), base_url=base_url, api_key_env=api_key_env,
                temperature=0.0, cache_dir=cache_dir, price_input_per_m=float(p_in), price_output_per_m=float(p_out))))
        return cls.from_specs(specs)

    @classmethod
    def mock(cls, names: Sequence[str] = PAPER_JUDGES, seed: int = 0) -> "JudgePanel":
        """Offline panel of :class:`~scholarpath.llm.MockLLM` judges (dry runs)."""
        specs = [JudgeSpec(n, LLMConfig(provider="mock", model=f"mock-{n}", cache_dir=None, seed=seed,
                                        price_input_per_m=0.0, price_output_per_m=0.0)) for n in names]
        return cls.from_specs(specs)

    @classmethod
    def single(cls, llm: LLMClient, name: Optional[str] = None) -> "JudgePanel":
        """Wrap an existing client as a one-judge panel (requests still billed to ``"judge"``)."""
        return cls([Judge(name or llm.model, llm)])

    # ------------------------------------------------------------- access
    @property
    def names(self) -> List[str]:
        return [j.name for j in self.judges]

    @property
    def primary(self) -> Judge:
        """First judge (used where a protocol calls for a single judge)."""
        return self.judges[0]

    def __len__(self) -> int:
        return len(self.judges)

    def __iter__(self) -> Iterator[Judge]:
        return iter(self.judges)

    def __getitem__(self, name: str) -> Judge:
        for j in self.judges:
            if j.name == name:
                return j
        raise KeyError(name)

    # --------------------------------------------------------------- usage
    def usage(self) -> Dict[str, Dict[str, float]]:
        """Per-judge LLM usage (calls, tokens, $) -- reported separately from system cost."""
        return {j.name: j.usage().to_dict() for j in self.judges}

    def total_usage(self) -> UsageRecord:
        out = UsageRecord()
        seen = set()
        for j in self.judges:
            if id(j.meter) in seen:
                continue
            seen.add(id(j.meter))
            out.add(j.usage())
        return out

    # ------------------------------------------------------------ execution
    def map(self, fn: Callable[[Judge], T]) -> Dict[str, T]:
        """Run ``fn(judge)`` for every judge (concurrently); returns ``{name: result}``."""
        workers = max(1, min(self.max_workers, len(self.judges)))
        if workers == 1:
            return {j.name: fn(j) for j in self.judges}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(fn, self.judges))
        return {j.name: r for j, r in zip(self.judges, results)}


def as_panel(judge: JudgeLike) -> JudgePanel:
    """Coerce a panel, a single :class:`Judge`, or an :class:`LLMClient` to a panel."""
    if isinstance(judge, JudgePanel):
        return judge
    if isinstance(judge, Judge):
        return JudgePanel([judge])
    if isinstance(judge, LLMClient):
        return JudgePanel.single(judge)
    raise TypeError(f"Expected a JudgePanel, Judge or LLMClient, got {type(judge).__name__}")


def as_judge_llm(judge: JudgeLike) -> LLMClient:
    """The client of a single-judge protocol (the panel's primary judge)."""
    if isinstance(judge, LLMClient):
        return judge
    if isinstance(judge, Judge):
        return judge.llm
    if isinstance(judge, JudgePanel):
        return judge.primary.llm
    raise TypeError(f"Expected a JudgePanel, Judge or LLMClient, got {type(judge).__name__}")


__all__ = [
    "DEFAULT_JUDGE_CACHE",
    "JUDGE_COMPONENT",
    "Judge",
    "JudgeLike",
    "JudgePanel",
    "JudgeSpec",
    "PAPER_JUDGES",
    "PROMETHEUS_MODEL",
    "RESEARCHQA_JUDGE_MODEL",
    "SCHOLARQA_CS_JUDGE_MODEL",
    "as_judge_llm",
    "as_panel",
    "build_judge_llm",
]
