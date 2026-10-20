"""LLM client abstraction shared by every ScholarPath component.

All LLM calls go through :class:`LLMClient`, which provides

* a uniform request type (:class:`LLMRequest`) carrying the *task* name, the
  rendered chat messages, and the structured *payload* the prompt was built
  from (the offline :class:`~scholarpath.llm.mock.MockLLM` answers from the
  payload, real backends from the messages);
* cost accounting per component (:class:`~scholarpath.llm.accounting.CostMeter`);
* an optional on-disk response cache keyed by (model, messages, params);
* JSON-mode helpers with parse-retry (:meth:`LLMClient.complete_json`);
* thread-pooled batch execution (:meth:`LLMClient.map`).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from scholarpath.llm.accounting import CostMeter
from scholarpath.llm.parsing import JSONParseError, extract_json
from scholarpath.text import count_tokens


@dataclass
class LLMRequest:
    """A single chat-completion request.

    ``task``      -- prompt identifier (e.g. ``"taxonomy.propose_children"``);
                     used by the mock backend and for logging.
    ``messages``  -- OpenAI-style ``[{"role": ..., "content": ...}]``.
    ``payload``   -- the structured inputs the prompt was rendered from.
    ``component`` -- cost bucket; defaults to the meter's current scope.
    """

    task: str
    messages: List[Dict[str, str]]
    payload: Dict[str, Any] = field(default_factory=dict)
    component: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    json_mode: bool = False
    model: Optional[str] = None  # override the client's default model

    @property
    def prompt_text(self) -> str:
        return "\n\n".join(m.get("content", "") for m in self.messages)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float = 0.0
    cached: bool = False
    model: str = ""
    cost_usd: float = 0.0
    raw: Optional[Dict[str, Any]] = None


def chat(system: str, user: str) -> List[Dict[str, str]]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs


class _DiskCache:
    """Tiny JSON-file cache (one file per key, sharded by prefix)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        p = self._path(key)
        with self._lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(value), encoding="utf-8")
            os.replace(tmp, p)


class LLMClient:
    """Base class; subclasses implement :meth:`_generate`."""

    name = "base"

    def __init__(
        self,
        model: str = "unknown",
        meter: Optional[CostMeter] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        cache_dir: Optional[str] = None,
        bill_cached_calls: bool = True,
        json_retries: int = 2,
        max_workers: int = 8,
    ):
        self.model = model
        self.meter = meter or CostMeter()
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.cache = _DiskCache(Path(cache_dir)) if cache_dir else None
        self.bill_cached_calls = bool(bill_cached_calls)
        self.json_retries = int(json_retries)
        self.max_workers = int(max_workers)

    # ------------------------------------------------------------ backend
    def _generate(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - abstract
        raise NotImplementedError

    # -------------------------------------------------------------- public
    def _cache_key(self, request: LLMRequest, salt: str = "") -> str:
        blob = json.dumps(
            {
                "model": request.model or self.model,
                "backend": self.name,
                "messages": request.messages,
                "temperature": self.temperature if request.temperature is None else request.temperature,
                "max_tokens": request.max_tokens or self.max_output_tokens,
                "json": request.json_mode,
                "salt": salt,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def complete(self, request: LLMRequest, _salt: str = "") -> LLMResponse:
        """Run one request (cache-aware) and bill it to the cost meter."""
        component = request.component or self.meter.current_component
        key = self._cache_key(request, _salt) if self.cache is not None else None
        if key is not None:
            hit = self.cache.get(key)
            if hit is not None:
                resp = LLMResponse(
                    text=hit["text"],
                    input_tokens=int(hit["input_tokens"]),
                    output_tokens=int(hit["output_tokens"]),
                    latency_s=0.0,
                    cached=True,
                    model=hit.get("model", self.model),
                )
                if self.bill_cached_calls:
                    resp.cost_usd = self.meter.record(
                        resp.input_tokens, resp.output_tokens, float(hit.get("latency_s", 0.0)),
                        component=component, cached=True,
                    )
                return resp
        t0 = time.perf_counter()
        resp = self._generate(request)
        resp.latency_s = resp.latency_s or (time.perf_counter() - t0)
        if resp.input_tokens <= 0:
            resp.input_tokens = count_tokens(request.prompt_text)
        if resp.output_tokens <= 0:
            resp.output_tokens = count_tokens(resp.text)
        resp.cost_usd = self.meter.record(
            resp.input_tokens, resp.output_tokens, resp.latency_s, component=component
        )
        if key is not None:
            self.cache.put(
                key,
                {
                    "text": resp.text,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "latency_s": resp.latency_s,
                    "model": resp.model or self.model,
                    "task": request.task,
                },
            )
        return resp

    def complete_json(
        self, request: LLMRequest, expect: Optional[type] = dict
    ) -> Tuple[Any, LLMResponse]:
        """Run a request and parse a JSON value from its output.

        On a parse failure the request is retried (with a cache salt so the
        retry is not served from cache) up to ``json_retries`` times; each
        retry is billed. Raises :class:`JSONParseError` if all attempts fail.
        """
        request.json_mode = True
        last_err: Optional[Exception] = None
        last_resp: Optional[LLMResponse] = None
        for attempt in range(self.json_retries + 1):
            resp = self.complete(request, _salt=f"retry{attempt}" if attempt else "")
            last_resp = resp
            try:
                return extract_json(resp.text, expect=expect), resp
            except JSONParseError as err:
                last_err = err
        raise JSONParseError(f"[{request.task}] {last_err}") from last_err

    def map(
        self,
        fn: Callable[[Any], Any],
        items: Sequence[Any],
        max_workers: Optional[int] = None,
    ) -> List[Any]:
        """Apply ``fn`` (which typically issues LLM calls) to items concurrently.

        The meter scope of the calling thread is propagated to the workers.
        """
        items = list(items)
        if not items:
            return []
        workers = max(1, min(max_workers or self.max_workers, len(items)))
        if workers == 1:
            return [fn(x) for x in items]
        scope = list(self.meter._stack())

        def run(x: Any) -> Any:
            st = self.meter._stack()
            st[:] = scope
            try:
                return fn(x)
            finally:
                st[:] = []

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(run, items))
