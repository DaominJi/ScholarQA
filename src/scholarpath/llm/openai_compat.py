"""OpenAI-compatible chat-completions backend (DeepSeek, OpenRouter, vLLM, ...).

Uses the ``openai`` package when installed and falls back to a dependency-free
``urllib`` implementation otherwise.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from scholarpath.llm.accounting import CostMeter
from scholarpath.llm.base import LLMClient, LLMRequest, LLMResponse


class OpenAICompatibleLLM(LLMClient):
    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "SCHOLARPATH_LLM_API_KEY",
        timeout_s: float = 120.0,
        max_retries: int = 4,
        meter: Optional[CostMeter] = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, meter=meter, **kwargs)
        self.base_url = (
            base_url
            or os.environ.get("SCHOLARPATH_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.deepseek.com/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                f"No API key found. Set ${api_key_env} (or $OPENAI_API_KEY), "
                "or use the offline backend with --set llm.provider=mock."
            )
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self._client = None
        try:  # pragma: no cover - optional dependency
            import openai

            self._client = openai.OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s, max_retries=0
            )
        except Exception:
            self._client = None

    def _body(self, request: LLMRequest) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": request.model or self.model,
            "messages": request.messages,
            "temperature": self.temperature if request.temperature is None else request.temperature,
            "max_tokens": request.max_tokens or self.max_output_tokens,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if self._client is not None:  # pragma: no cover - network
            resp = self._client.chat.completions.create(**body)
            return resp.model_dump()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:  # pragma: no cover - network
            return json.loads(r.read().decode("utf-8"))

    def _generate(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - network
        body = self._body(request)
        delay = 2.0
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                data = self._post(body)
                choice = data["choices"][0]["message"]
                usage = data.get("usage") or {}
                return LLMResponse(
                    text=choice.get("content") or "",
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    latency_s=time.perf_counter() - t0,
                    model=data.get("model", body["model"]),
                    raw=None,
                )
            except Exception as err:  # retry on transient errors
                last = err
                if body.get("response_format") and "response_format" in str(err):
                    body.pop("response_format", None)  # backend lacks JSON mode
                if attempt < self.max_retries:
                    time.sleep(delay + random.random())
                    delay *= 2
        raise RuntimeError(f"LLM request failed after {self.max_retries + 1} attempts: {last}")
