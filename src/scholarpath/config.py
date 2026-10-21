"""Configuration dataclasses for every ScholarPath component.

Defaults follow the implementation details reported in the paper
(Section 5.1 "Implementation" and Appendix C.3):

* DeepSeek-V4.0-Flash as the unified LLM backbone
  ($0.14 / 1M input tokens, $0.28 / 1M output tokens);
* SPECTER2 (768-d) paper / concept embeddings;
* router: two-layer MLP (hidden 256, GELU, AdamW) on concatenated paper and
  child-definition embeddings with a 20% stratified validation split;
* G_v = min{8, floor(n_v / 24)} k-means regions per node;
* minimum node size 16, maximum depth 5;
* node-level error tolerance eps = 0.10, confidence 1 - delta = 0.95;
* discovery sampling ratio eta_A = 0.02, batch cap beta = 16,
  neighbour count kappa = 3;
* context token budget B = 8,192.

Every config can be loaded from / saved to YAML or JSON
(see ``configs/default.yaml``) and overridden from the command line with
``--set section.field=value``. Values from files and overrides are checked
against the field types (``"1e-3"`` becomes a float, ``"0.1x"`` or an unknown
field raises :class:`ConfigError`), so typos fail at start-up rather than deep
inside a run.
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union


class ConfigError(ValueError):
    """Invalid configuration: a value of the wrong type or a malformed ``--set`` override."""


class UnknownConfigField(ConfigError, AttributeError, KeyError):
    """Unknown config section or field (also an ``AttributeError`` / ``KeyError`` for older callers)."""

    def __str__(self) -> str:  # KeyError.__str__ would quote the message
        return str(self.args[0]) if self.args else ""


# ---------------------------------------------------------------------------
# LLM backbone
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """LLM backend settings.

    ``provider``:
      * ``"openai"`` -- any OpenAI-compatible chat-completions endpoint
        (DeepSeek, OpenRouter, vLLM, ...). Credentials come from the
        environment variable named by ``api_key_env``.
      * ``"mock"``   -- deterministic offline heuristics (no network); used by
        offline runs and dry runs of every script.
    """

    provider: str = "mock"
    model: str = "deepseek-v4-flash"
    base_url: Optional[str] = None  # falls back to $SCHOLARPATH_LLM_BASE_URL
    api_key_env: str = "SCHOLARPATH_LLM_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    timeout_s: float = 120.0
    max_retries: int = 4
    json_retries: int = 2
    # Pricing in USD per 1M tokens (DeepSeek-V4.0-Flash, Appendix C.3).
    price_input_per_m: float = 0.14
    price_output_per_m: float = 0.28
    # On-disk response cache (None disables). Cached calls are still billed
    # in the cost meter with ``cached=True`` so cost numbers are reproducible.
    cache_dir: Optional[str] = "cache/llm"
    bill_cached_calls: bool = True
    # Knowledge cutoff of the answer LLM (ISO date). App. C.1: held-out source
    # papers must postdate it (default of build_heldout_benchmark --min-source-date).
    knowledge_cutoff: Optional[str] = None
    seed: int = 0


@dataclass
class EmbeddingConfig:
    """Embedding backend settings.

    ``backend``:
      * ``"specter2"`` -- allenai/specter2_base with the proximity adapter
        (768-d), loaded through the ``adapters`` library (falls back to
        sentence-transformers if ``adapters`` is unavailable);
      * ``"sentence-transformers"`` -- any sentence-transformers model;
      * ``"hashing"`` -- deterministic hashing embedder (no model download),
        used for offline runs.
    """

    backend: str = "hashing"
    model_name: str = "allenai/specter2_base"
    adapter: Optional[str] = "allenai/specter2"
    query_adapter: Optional[str] = "allenai/specter2_adhoc_query"
    dim: int = 768
    batch_size: int = 32
    max_length: int = 512
    device: Optional[str] = None
    cache_dir: Optional[str] = "cache/embeddings"


# ---------------------------------------------------------------------------
# Section 3: node-wise confidence-bounded taxonomy builder
# ---------------------------------------------------------------------------


@dataclass
class RouterConfig:
    """Lightweight router f_theta_v (Section 3.1, Appendix C.3)."""

    hidden_dim: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-2
    epochs: int = 60
    min_steps: int = 250  # small label sets still get enough AdamW updates
    max_steps: Optional[int] = 6000
    patience: int = 3  # early stop when the epoch loss stops improving (0 disables)
    tolerance: float = 1e-3
    batch_size: int = 32
    validation_fraction: float = 0.2
    threshold: float = 0.5
    seed: int = 0


@dataclass
class BuilderConfig:
    """Hyper-parameters of Algorithm 1 (Section 3 and Appendix B)."""

    # Which facet the taxonomy organizes: "problem", "method", or "topic"
    # (combined topic-and-approach taxonomy used on TaxoBench).
    facet: str = "problem"
    epsilon: float = 0.10  # node-level error tolerance
    delta: float = 0.05  # taxonomy-wide failure probability (1 - delta = 0.95)
    # How delta is split across nodes so that sum_v delta_v <= delta:
    #   "spend"    -- delta_v = min(delta * n_v / (N * max_depth), remaining / 2)
    #   "harmonic" -- delta_v = delta * 6 / (pi^2 k^2) for the k-th episode
    #   "uniform"  -- delta_v = delta / max_nodes
    delta_allocation: str = "spend"
    max_nodes: int = 512
    batch_cap: int = 16  # beta
    kappa: int = 3  # neighbours relabelled by CORRECT
    eta_A: float = 0.02  # discovery sampling ratio
    s_max: int = 32  # cap on discovery draws per invocation
    min_node_size: int = 16
    max_depth: int = 5
    max_regions: int = 8  # G_v = min{max_regions, floor(n_v / papers_per_region)}
    papers_per_region: int = 24
    seed_per_region: int = 2  # seed batch = seed_per_region papers per region
    min_children: int = 2
    max_children: int = 8
    llm_papers_per_call: int = 16  # papers labelled in one LLM call
    max_rounds: int = 12  # acquisition rounds per cycle
    max_cycles: int = 6  # repair/certification cycles per node
    audit_batch_size: int = 8
    # Stage 2 trigger (Alg. 1 line 12), both measured in papers of P_v against
    # max{repair_pool_floor, repair_pool_fraction * n_v}: DISCOVER when the
    # residual pool (bottom_v plus missing-concept papers) reaches it, CORRECT
    # when the number of misrouted papers implied by the held-out disagreements
    # (|pool| / |held-out| x unverified placed papers) reaches it.
    repair_pool_fraction: float = 0.02
    repair_pool_floor: int = 2
    # Stop auditing a snapshot early when the audited error rate makes
    # certification implausible (point estimate > epsilon * this factor).
    audit_abort_factor: float = 1.5
    audit_min_draws: int = 8
    # Response-model / prior (Appendix B.1).
    prior_slope_mean: float = 0.05
    prior_slope_var: float = 0.05 ** 2
    obs_noise_var: float = 0.02 ** 2
    use_transferred_prior: bool = True
    # Ablation switches (Table 6).
    use_bayesian_acquisition: bool = True
    use_snapshot_certification: bool = True
    fixed_ratio: float = 0.08  # eta_v when snapshot certification is disabled
    use_discovery: bool = True
    use_correction: bool = True
    use_targeted_repair: bool = True
    # Budget cap: if reached, the node is returned as unresolved.
    max_llm_papers_per_node: Optional[int] = None
    router: RouterConfig = field(default_factory=RouterConfig)
    seed: int = 0


@dataclass
class BaselineBuilderConfig:
    """Selection-based baselines of Section 5.2 (FULL, node-wise, one-off)."""

    selector: str = "full"  # full | random | coverage | kmeans | diversity
    eta: float = 0.2  # selection ratio
    mode: str = "node_wise"  # node_wise | one_off
    facet: str = "problem"
    min_node_size: int = 16
    max_depth: int = 5
    min_children: int = 2
    max_children: int = 8
    llm_papers_per_call: int = 16
    router: RouterConfig = field(default_factory=RouterConfig)
    seed: int = 0


# ---------------------------------------------------------------------------
# Section 4: budgeted support-path retrieval
# ---------------------------------------------------------------------------


@dataclass
class RetrievalConfig:
    """Hyper-parameters of Algorithm 2 (Section 4)."""

    budget: int = 8192  # context token budget B
    grounding_top_k: int = 3  # scopes matched per query concept
    grounding_min_sim: float = 0.2
    anchor_top_k: int = 3  # neighbours of a query-local anchor concept
    # Cosine an anchor's neighbour needs to act as a scope (None: 0.5 * grounding_min_sim).
    anchor_min_sim: Optional[float] = None
    # A secondary grounding match is kept as an alternative scope when its cosine
    # is at least this fraction of the best match's cosine.
    grounding_rel_margin: float = 0.5
    max_aspects: int = 6
    # Granularity weights gamma_{a, level} per evidential role (Section 4.4).
    # Roles: overview, empirical, mechanism. Levels: concept, paper, chunk.
    granularity_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "overview": {"concept": 0.9, "paper": 0.8, "chunk": 0.5},
            "empirical": {"concept": 0.0, "paper": 0.85, "chunk": 0.9},
            "mechanism": {"concept": 0.0, "paper": 0.6, "chunk": 1.0},
        }
    )
    # Bridge discount phi(b) = bridge_decay ** (taxonomic distance - 1).
    bridge_decay: float = 0.8
    max_bridge_distance: int = 4
    max_hypothesis_candidates: int = 64
    max_chunks_per_paper: int = 64
    # Ablation switches (Table 7).
    use_link_prediction: bool = True
    use_bound_guided_search: bool = True
    use_role_aware_granularity: bool = True
    use_shared_cost_update: bool = True
    # Step 1 can use one lightweight LLM call or an embedding-only parser.
    llm_query_understanding: bool = True
    answer_max_tokens: int = 1536


@dataclass
class LinkPredictionConfig:
    """Cross-taxonomy association model (Section 4.3, Appendix B.5)."""

    embed_dim: int = 128
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    negatives_per_positive: int = 8
    context_dropout: float = 0.2  # omit taxonomy context on this fraction
    horizon_years: float = 2.0  # Delta
    cutoffs: Tuple[str, ...] = ("2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01")
    verify_with_llm: bool = True
    seed: int = 0


@dataclass
class IndexConfig:
    """Multi-granularity semantic index construction (Section 2.1)."""

    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 32
    use_fulltext: bool = True


@dataclass
class EvaluationConfig:
    """LLM judges and official evaluation scripts (Section 5.3, App. C.3).

    Judge calls are billed to the "judge" component of a *separate* meter and
    never count towards system cost.
    """

    # Hypothesis judges (Table 4): three state-of-the-art LLMs, scores averaged.
    judge_models: Tuple[str, ...] = ("gpt-6.0-sol", "opus-5.5", "grok-4.7")
    judge_provider: str = "mock"  # "openai" for real judges (OpenAI-compatible endpoint)
    judge_base_url: Optional[str] = None  # falls back to $SCHOLARPATH_JUDGE_BASE_URL
    judge_api_key_env: str = "SCHOLARPATH_JUDGE_API_KEY"
    judge_cache_dir: Optional[str] = "cache/judge"
    # Official benchmark judges.
    researchqa_judge_model: str = "gpt-4.1-mini"
    scholarqa_cs_judge_model: str = "gpt-4-turbo"
    prometheus_model: str = "prometheus-eval/prometheus-7b-v2.0"
    # Statement-level citation judge of ScholarQABench Cite: "nli" is the
    # AttrScore-Flan-T5-XL model of the official citation_correctness_eval.py
    # (configs/paper.yaml); "llm" asks the ScholarQA-CS judge with the AttrScore
    # prompt; "none" disables Cite. Mock judges always use "llm".
    attribution_judge: str = "llm"
    discovery_k: int = 10  # Hit@10
    # Local checkouts of the official repositories, used by
    # scripts/run_official_eval.py (scholarpath.evaluation.official); when unset,
    # $SCHOLARPATH_RESEARCHQA_DIR / $SCHOLARPATH_SCHOLARQABENCH_DIR are used.
    researchqa_dir: Optional[str] = None
    scholarqabench_dir: Optional[str] = None


@dataclass
class ScholarPathConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    builder: BuilderConfig = field(default_factory=BuilderConfig)
    baseline: BaselineBuilderConfig = field(default_factory=BaselineBuilderConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    link_prediction: LinkPredictionConfig = field(default_factory=LinkPredictionConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 0

    # ------------------------------------------------------------------ io
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScholarPathConfig":
        return _from_dict(cls, data or {})

    @classmethod
    def load(cls, path: Union[str, Path, None]) -> "ScholarPathConfig":
        if path is None:
            return cls()
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            try:
                data = yaml.safe_load(text) or {}
            except yaml.YAMLError as err:
                raise ConfigError(f"not valid YAML: {err}") from None
        else:
            try:
                data = json.loads(text)
            except ValueError as err:
                raise ConfigError(f"not valid JSON: {err}") from None
        return cls.from_dict(data)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def override(self, assignments: Iterable[str]) -> "ScholarPathConfig":
        """Apply ``section.field=value`` overrides (values parsed as YAML, then type-checked).

        Nested dict entries are addressed the same way, e.g.
        ``retrieval.granularity_weights.overview.chunk=0.4``. Raises
        :class:`UnknownConfigField` for an unknown section / field and
        :class:`ConfigError` for a value that does not fit the field's type.
        """
        import yaml

        for item in assignments or []:
            if "=" not in item:
                raise ConfigError(f"Override must look like section.field=value, got {item!r}")
            key, raw = item.split("=", 1)
            key = key.strip()
            try:
                value = _plain(yaml.safe_load(raw))
            except yaml.YAMLError as err:
                raise ConfigError(f"Cannot parse the value of {key}: {err}") from None
            parts = key.split(".")
            target: Any = self
            tp: Any = ScholarPathConfig
            for i, part in enumerate(parts):
                dotted = ".".join(parts[: i + 1])
                last = i == len(parts) - 1
                if dataclasses.is_dataclass(target):
                    names = [f.name for f in dataclasses.fields(target)]
                    if part not in names:
                        raise UnknownConfigField(f"Unknown config field: {dotted}{_suggest(part, names)}")
                    tp = _field_types(type(target))[part]
                    if last:
                        setattr(target, part, _coerce(value, tp, key, raw=raw))
                    else:
                        target = getattr(target, part)
                elif isinstance(target, dict):
                    if part not in target:
                        raise UnknownConfigField(f"Unknown config key: {dotted}{_suggest(part, list(target))}")
                    args = typing.get_args(tp)
                    tp = args[1] if len(args) == 2 else type(target[part])
                    if last:
                        target[part] = _coerce(value, tp, key, raw=raw)
                    else:
                        target = target[part]
                else:
                    raise UnknownConfigField(f"Unknown config field: {key} ({'.'.join(parts[:i])} is a value)")
        return self

    def mock_judge_warning(self) -> Optional[str]:
        """A warning when a real LLM backend is paired with the offline mock judges, else ``None``.

        Judge-based scores (ResearchQA Rub, ScholarQA Corr / Cite, Prometheus,
        Disc and the Feas. / Nov. / Sound. judges) are then produced by the
        deterministic ``MockLLM`` and are not comparable with the paper.
        """
        if (self.llm.provider or "mock").lower() == "mock":
            return None
        if (self.evaluation.judge_provider or "mock").lower() != "mock":
            return None
        return ("llm.provider is %r but evaluation.judge_provider is 'mock': every LLM judge (Rub, Corr, "
                "Prometheus, citation attribution, Disc, Feas./Nov./Sound.) is the offline MockLLM, so "
                "judge-based scores are not comparable with the paper. Use configs/paper.yaml or "
                "--set evaluation.judge_provider=openai." % self.llm.provider)


def _plain(value: Any) -> Any:
    """Convert YAML-parsed dates back to ISO strings (recursively)."""
    import datetime as _dt

    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _from_dict(cls: Any, data: Mapping[str, Any], prefix: str = "") -> Any:
    if not dataclasses.is_dataclass(cls):
        return data
    if not isinstance(data, Mapping):
        raise ConfigError(f"{prefix.rstrip('.') or cls.__name__} must be a mapping of fields, got {data!r}")
    data = _plain(dict(data))
    kwargs: Dict[str, Any] = {}
    names = [f.name for f in dataclasses.fields(cls)]
    types = _field_types(cls)
    for name, value in data.items():
        if name not in names:
            raise UnknownConfigField(f"Unknown config field {prefix + name!r} ({cls.__name__}){_suggest(name, names)}")
        kwargs[name] = _coerce(value, types[name], f"{prefix}{name}")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Type checking of config values
# ---------------------------------------------------------------------------

_NULLS = {"null", "none", "~", ""}
_TRUE = {"true", "yes", "on", "y", "1"}
_FALSE = {"false", "no", "off", "n", "0"}


@functools.lru_cache(maxsize=None)
def _field_types(cls: Any) -> Dict[str, Any]:
    return typing.get_type_hints(cls)


def _suggest(name: str, candidates: Iterable[str]) -> str:
    close = difflib.get_close_matches(name, list(candidates), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _coerce(value: Any, tp: Any, name: str, raw: Optional[str] = None) -> Any:
    """Check ``value`` against the field type ``tp`` and convert it where unambiguous.

    ``raw`` is the unparsed text of a ``--set`` override: a ``str`` field keeps
    it verbatim when YAML turned it into a bool / number (``off`` -> ``"off"``).
    """
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Union:
        options = [a for a in args if a is not type(None)]
        nullable = len(options) < len(args)
        if nullable and (value is None or (isinstance(value, str) and value.strip().lower() in _NULLS)):
            return None
        if len(options) == 1:
            return _coerce(value, options[0], name, raw)
        return value
    if tp is Any:
        return value
    if dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if value is None:
            return tp()
        if isinstance(value, Mapping):
            return _from_dict(tp, value, prefix=f"{name}.")
        raise ConfigError(f"{name} is a config section; set its fields ({name}.<field>=value)")
    if value is None:
        raise ConfigError(f"{name} cannot be null")
    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE
        raise ConfigError(f"{name} expects true or false, got {value!r}")
    if tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            try:
                return int(text)
            except ValueError:
                try:
                    number = float(text)
                except ValueError:
                    number = None
                if number is not None and number.is_integer():
                    return int(number)
        raise ConfigError(f"{name} expects an integer, got {value!r}")
    if tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return float(value.strip())  # YAML reads 1e-3 as a string
            except ValueError:
                pass
        raise ConfigError(f"{name} expects a number, got {value!r}")
    if tp is str:
        if isinstance(value, str):
            return value
        if raw is not None and isinstance(value, (bool, int, float)):
            return raw.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        raise ConfigError(f"{name} expects a string, got {value!r} (quote the value)")
    if origin is tuple:
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{name} expects a list, got {value!r}")
        elem = args[0] if len(args) == 2 and args[1] is Ellipsis else Any
        return tuple(_coerce(v, elem, f"{name}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{name} expects a mapping, got {value!r}")
        vt = args[1] if len(args) == 2 else Any
        return {str(k): _coerce(v, vt, f"{name}.{k}") for k, v in value.items()}
    return value
