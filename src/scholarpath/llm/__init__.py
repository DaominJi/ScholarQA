"""LLM backends, cost accounting, and JSON parsing."""

from __future__ import annotations

from typing import Optional

from scholarpath.config import LLMConfig
from scholarpath.llm.accounting import BUILD_COMPONENTS, CostMeter, UsageRecord, sum_usage
from scholarpath.llm.base import LLMClient, LLMRequest, LLMResponse, chat
from scholarpath.llm.mock import MockLLM, register_mock_handler, stable_hash
from scholarpath.llm.parsing import JSONParseError, extract_json


def build_llm(config: Optional[LLMConfig] = None, meter: Optional[CostMeter] = None) -> LLMClient:
    """Instantiate the LLM backend described by ``config``."""
    config = config or LLMConfig()
    meter = meter or CostMeter(config.price_input_per_m, config.price_output_per_m)
    common = dict(
        meter=meter,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        cache_dir=config.cache_dir,
        bill_cached_calls=config.bill_cached_calls,
        json_retries=config.json_retries,
    )
    provider = (config.provider or "mock").lower()
    if provider == "mock":
        common["cache_dir"] = None  # mock is deterministic and free; never cache
        return MockLLM(model="mock", seed=config.seed, **common)
    if provider in {"openai", "deepseek", "openrouter", "vllm", "openai_compatible"}:
        from scholarpath.llm.openai_compat import OpenAICompatibleLLM

        return OpenAICompatibleLLM(
            model=config.model,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            **common,
        )
    raise ValueError(f"Unknown LLM provider {config.provider!r}")


__all__ = [
    "BUILD_COMPONENTS",
    "CostMeter",
    "JSONParseError",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "MockLLM",
    "UsageRecord",
    "build_llm",
    "chat",
    "extract_json",
    "register_mock_handler",
    "stable_hash",
    "sum_usage",
]
