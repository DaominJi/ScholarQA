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
``--set section.field=value``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union


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
        the smoke tests and for dry runs of every script.
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
        used for smoke tests.
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
    negatives_per_positive: int = 3
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
    # Stage 2 trigger: repair when the residual pool or the disagreement pool
    # exceeds this fraction of n_v (or the absolute floor).
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
    attribution_judge: str = "llm"  # "llm" | "nli" (osunlp/attrscore-flan-t5-xl)
    discovery_k: int = 10  # Hit@10
    # Official checkouts (optional; enable scripts/run_official_eval-style runs).
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

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
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
        """Apply ``section.field=value`` overrides (values parsed as YAML)."""
        import yaml

        for item in assignments or []:
            if "=" not in item:
                raise ValueError(f"Override must look like section.field=value, got {item!r}")
            key, raw = item.split("=", 1)
            value = _plain(yaml.safe_load(raw))
            target: Any = self
            parts = key.strip().split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            if not hasattr(target, parts[-1]):
                raise AttributeError(f"Unknown config field: {key}")
            current = getattr(target, parts[-1])
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(target, parts[-1], value)
        return self


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


def _from_dict(cls: Any, data: Mapping[str, Any]) -> Any:
    if not dataclasses.is_dataclass(cls):
        return data
    data = _plain(dict(data))
    kwargs: Dict[str, Any] = {}
    hints = {f.name: f for f in dataclasses.fields(cls)}
    for name, value in data.items():
        if name not in hints:
            raise KeyError(f"Unknown field {name!r} for {cls.__name__}")
        f = hints[name]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default  # type: ignore[misc]
        if dataclasses.is_dataclass(default) and isinstance(value, Mapping):
            kwargs[name] = _from_dict(type(default), value)
        elif isinstance(default, tuple) and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def resolve_path(path: Optional[str], root: Optional[Path] = None) -> Optional[Path]:
    """Resolve a (possibly relative) path against ``root`` (default: cwd)."""
    if path is None:
        return None
    p = Path(os.path.expanduser(path))
    if not p.is_absolute() and root is not None:
        p = root / p
    return p
