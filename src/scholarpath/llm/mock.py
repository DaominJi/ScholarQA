"""Deterministic offline LLM backend.

The mock answers each request from its structured ``payload`` using cheap
lexical / embedding heuristics, so that every pipeline and script in the
repository can be exercised end to end without network access or API keys
(``--set llm.provider=mock``). Its outputs are *not* meant to reproduce the
paper's numbers -- use a real backend for that.

Handlers are registered per task name. Each prompt module under
:mod:`scholarpath.prompts` exposes a ``MOCK_HANDLERS`` dict
``{task_name: handler}`` with ``handler(payload, request) -> str | dict | list``;
they are discovered lazily on first use. Additional handlers can be added with
:func:`register_mock_handler`.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import pkgutil
import threading
from typing import Any, Callable, Dict, Optional

from scholarpath.llm.accounting import CostMeter
from scholarpath.llm.base import LLMClient, LLMRequest, LLMResponse
from scholarpath.text import count_tokens

log = logging.getLogger(__name__)

MockHandler = Callable[[Dict[str, Any], LLMRequest], Any]

_HANDLERS: Dict[str, MockHandler] = {}
_LOADED = False
_LOCK = threading.Lock()


def register_mock_handler(task: str, handler: MockHandler) -> None:
    _HANDLERS[task] = handler


def _load_handlers() -> None:
    global _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        import scholarpath.prompts as prompts_pkg

        for mod in pkgutil.iter_modules(prompts_pkg.__path__):
            try:
                module = importlib.import_module(f"scholarpath.prompts.{mod.name}")
            except Exception as err:  # a broken prompt module must not disable all mocks
                log.warning("MockLLM: skipping prompt module %s (%s)", mod.name, err)
                continue
            for task, handler in getattr(module, "MOCK_HANDLERS", {}).items():
                _HANDLERS.setdefault(task, handler)
        _LOADED = True


def stable_hash(*parts: Any) -> int:
    """Deterministic 64-bit hash of arbitrary JSON-serialisable parts."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(), "little")


class MockLLM(LLMClient):
    name = "mock"

    def __init__(self, model: str = "mock", meter: Optional[CostMeter] = None, seed: int = 0, **kwargs: Any):
        super().__init__(model=model, meter=meter, **kwargs)
        self.seed = int(seed)

    def _generate(self, request: LLMRequest) -> LLMResponse:
        _load_handlers()
        handler = _HANDLERS.get(request.task)
        if handler is None:
            raise KeyError(
                f"MockLLM has no handler for task {request.task!r}. "
                "Add one to the MOCK_HANDLERS dict of the prompt module that defines it."
            )
        out = handler(dict(request.payload or {}), request)
        text = out if isinstance(out, str) else json.dumps(out)
        return LLMResponse(
            text=text,
            input_tokens=count_tokens(request.prompt_text),
            output_tokens=count_tokens(text),
            latency_s=0.0,
            model=self.model,
        )
