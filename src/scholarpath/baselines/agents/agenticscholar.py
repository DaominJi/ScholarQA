"""AgenticScholar adapter [42] (Tables 3-5, "AgenticScholar").

AgenticScholar explores hypotheses by organizing papers into taxonomies and
scoring problem-method pairs with an LLM. For a fair comparison the paper
equips it "with the same constructed taxonomies as ScholarPath"
(Section 5.3): this adapter serializes the problem and method taxonomies of
ScholarPath's :class:`~scholarpath.index.semantic_index.SemanticIndex`
(concept names, definitions, hierarchy and paper extents, via
:meth:`Taxonomy.to_dict`) and hands them, with the uniform task prompt, the
exported corpus shards, and the (metered) LLM endpoint, to an external
AgenticScholar implementation.

AgenticScholar is not distributed as a package, so the implementation is
pluggable:

* ``fn``: a Python callable ``fn(request: dict) -> dict | str``, or
* ``endpoint_url``: an HTTP endpoint that receives the request as a JSON
  POST body and returns JSON.

Request (JSON)::

    {"query_id", "question", "kind", "prompt",            # prompt = uniform task text
     "output_schema",                                     # AGENT_OUTPUT_SCHEMA
     "taxonomies": {"problem": "<path>.json", "method": "<path>.json"},
     "taxonomy_data": {"problem": {...}, "method": {...}},   # only with inline_taxonomies=True
     "corpus": {"shard_dir": "<export dir>"},
     "llm": {"base_url", "api_key", "model"}}

Response: ``{"output": <final JSON or text>, "usage": {"llm_calls",
"input_tokens", "output_tokens", "cost_usd"}}`` (``usage`` optional; the
metering proxy measures calls made through ``llm.base_url``) or just the
output string.
"""

from __future__ import annotations

import json
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

from scholarpath.baselines.agents.base import AgentAdapter, AgentConfig, AgentRun, AgentUnavailableError, LLMEndpoint
from scholarpath.prompts.baselines import AGENT_OUTPUT_SCHEMA
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.types import Query


class AgenticScholarAgent(AgentAdapter):
    """Adapter around an external AgenticScholar implementation."""

    name = "AgenticScholar"

    def __init__(
        self,
        problem_taxonomy: Optional[Taxonomy] = None,
        method_taxonomy: Optional[Taxonomy] = None,
        index: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
        fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        endpoint_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        inline_taxonomies: bool = False,
        valid_ids: Optional[Iterable[str]] = None,
    ):
        if index is not None:
            problem_taxonomy = problem_taxonomy or index.problem
            method_taxonomy = method_taxonomy or index.method
            valid_ids = valid_ids if valid_ids is not None else list(index.papers)
        if valid_ids is None and problem_taxonomy is not None:
            method_papers = method_taxonomy.all_papers() if method_taxonomy else set()
            valid_ids = sorted(problem_taxonomy.all_papers() | method_papers)
        super().__init__(config, valid_ids=valid_ids)
        self.problem_taxonomy = problem_taxonomy
        self.method_taxonomy = method_taxonomy
        self.fn = fn
        self.endpoint_url = endpoint_url
        self.headers = dict(headers or {})
        self.inline_taxonomies = bool(inline_taxonomies)
        self._tax_files: Optional[Dict[str, str]] = None
        self._workdir: Optional[Path] = None

    def check_available(self) -> None:
        if self.fn is None and not self.endpoint_url:
            raise AgentUnavailableError(
                "AgenticScholar is not bundled with ScholarPath: pass fn=<callable> or endpoint_url=<http url> "
                "pointing at an AgenticScholar implementation (request/response protocol documented in "
                "scholarpath.baselines.agents.agenticscholar)."
            )
        if self.problem_taxonomy is None or self.method_taxonomy is None:
            raise AgentUnavailableError(
                "AgenticScholar must be given ScholarPath's constructed taxonomies "
                "(pass index=<SemanticIndex> or problem_taxonomy/method_taxonomy)."
            )

    @property
    def workdir(self) -> Path:
        if self._workdir is None:
            base = self.config.workdir
            self._workdir = Path(base) / "agenticscholar" if base else Path(tempfile.mkdtemp(prefix="agenticscholar-"))
            self._workdir.mkdir(parents=True, exist_ok=True)
        return self._workdir

    def taxonomy_files(self) -> Dict[str, str]:
        """Serialize the shared taxonomies once (``Taxonomy.save`` JSON)."""
        if self._tax_files is None:
            assert self.problem_taxonomy is not None and self.method_taxonomy is not None
            files = {}
            for facet, tax in (("problem", self.problem_taxonomy), ("method", self.method_taxonomy)):
                path = self.workdir / f"{facet}_taxonomy.json"
                tax.save(path)
                files[facet] = str(path)
            self._tax_files = files
        return self._tax_files

    def build_request(self, query: Query, prompt: str, endpoint: Optional[LLMEndpoint]) -> Dict[str, Any]:
        req: Dict[str, Any] = {
            "query_id": query.query_id,
            "question": query.text,
            "kind": query.kind,
            "prompt": prompt,
            "output_schema": AGENT_OUTPUT_SCHEMA,
            "taxonomies": self.taxonomy_files(),
            "corpus": {"shard_dir": self.config.corpus_dir},
            "llm": ({"base_url": endpoint.base_url, "api_key": endpoint.api_key, "model": endpoint.model}
                    if endpoint is not None else None),
        }
        if self.inline_taxonomies:
            assert self.problem_taxonomy is not None and self.method_taxonomy is not None
            req["taxonomy_data"] = {"problem": self.problem_taxonomy.to_dict(),
                                    "method": self.method_taxonomy.to_dict()}
        return req

    def endpoint(self) -> LLMEndpoint:
        # A user-supplied in-process callable may not need the backend at all
        # (e.g. tests); only require credentials when they exist.
        if self.fn is not None and not self.config.api_key(required=False):
            return LLMEndpoint(self.config.resolved_base_url(), "", self.config.model)
        return super().endpoint()

    def run(self, query: Query, prompt: str, endpoint: Optional[LLMEndpoint]) -> AgentRun:
        request = self.build_request(query, prompt, endpoint)
        t0 = time.perf_counter()
        if self.fn is not None:
            response: Any = self.fn(request)
        else:
            body = json.dumps(request).encode("utf-8")
            http_req = urllib.request.Request(
                str(self.endpoint_url), data=body, method="POST",
                headers={"Content-Type": "application/json", **self.headers},
            )
            with urllib.request.urlopen(http_req, timeout=self.config.timeout_s) as r:
                response = json.loads(r.read().decode("utf-8"))
        latency = time.perf_counter() - t0
        return _to_run(response, latency)


def _to_run(response: Any, latency: float) -> AgentRun:
    if isinstance(response, str):
        return AgentRun(output_text=response, metadata={"latency_s": latency})
    if not isinstance(response, dict):
        raise TypeError(f"AgenticScholar returned {type(response).__name__}; expected dict or str")
    output: Union[str, Any] = response.get("output", response.get("answer", ""))
    if not isinstance(output, str):
        output = json.dumps(output)
    usage = response.get("usage") or {}

    def opt_int(key: str) -> Optional[int]:
        return int(usage[key]) if usage.get(key) is not None else None

    return AgentRun(
        output_text=output,
        llm_calls=opt_int("llm_calls"),
        input_tokens=opt_int("input_tokens"),
        output_tokens=opt_int("output_tokens"),
        cost_usd=float(usage["cost_usd"]) if usage.get("cost_usd") is not None else None,
        metadata={"latency_s": latency},
    )
