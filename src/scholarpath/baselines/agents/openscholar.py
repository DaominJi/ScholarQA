"""OpenScholar adapter [3] (Tables 3-5, "OpenScholar").

OpenScholar (Asai et al., 2024) answers with a retrieve -> rerank ->
generate -> self-feedback -> citation-verification pipeline. The adapter
shells out to a local checkout of https://github.com/AkariAsai/OpenScholar
with **our corpus as its datastore**:

1. :meth:`OpenScholarAgent.prepare_datastore` writes every passage of the
   shared pool (abstracts and section chunks) as ``passages.jsonl``
   (``id``, ``title``, ``text``, ``paper_id``) -- the datastore format of
   OpenScholar's retriever;
2. for each query, the initial top-``initial_k`` passages are retrieved from
   that datastore (by default with the SPECTER2 dense retriever over the
   same pool, standing in for OpenScholar's bi-encoder datastore search) and
   written as the ``ctxs`` of the input file, whose ``input`` is the uniform
   task prompt;
3. OpenScholar's ``run.py`` reranks them with its cross-encoder, generates,
   runs self-feedback and post-hoc attribution, and writes its output file;
4. the final ``output`` is parsed with
   :func:`~scholarpath.baselines.agents.base.parse_agent_output`; numeric
   citations ``[n]`` are mapped back to the paper ids of OpenScholar's
   (reranked) ``ctxs``.

The Semantic Scholar API retriever of OpenScholar (``--ss_retriever``) is
NOT enabled: external web access is disabled for every agent. LLM calls go
through the metering proxy via ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` (the
``openai`` client honours both), with the backbone as ``--model_name``.

Setup::

    git clone https://github.com/AkariAsai/OpenScholar && cd OpenScholar
    pip install -r requirements.txt

then pass ``checkout_dir``. ``command_template`` (a list of arguments with
``{placeholders}``) can be adjusted to the flags of the checked-out version.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from scholarpath.baselines.agents.base import AgentAdapter, AgentConfig, AgentRun, AgentUnavailableError, LLMEndpoint
from scholarpath.baselines.common import PassagePool
from scholarpath.types import Query

log = logging.getLogger(__name__)

#: Default command (OpenScholar ``run.py`` pipeline with reranking, feedback
#: and post-hoc citation attribution; no Semantic Scholar retrieval).
DEFAULT_COMMAND: List[str] = [
    "{python}", "run.py",
    "--input_file", "{input_file}",
    "--output_file", "{output_file}",
    "--model_name", "{model}",
    "--api", "openai",
    "--api_key_fp", "{api_key_file}",
    "--use_contexts",
    "--ranking_ce",
    "--reranker", "OpenScholar/OpenScholar_Reranker",
    "--top_n", "{top_n}",
    "--llm_inference",
    "--zero_shot",
    "--feedback",
    "--posthoc_at",
    "--min_citation", "1",
    "--norm_cite",
    "--max_per_paper", "3",
]

_NUM_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class OpenScholarAgent(AgentAdapter):
    """Shell-out adapter for an OpenScholar checkout."""

    name = "OpenScholar"

    def __init__(
        self,
        pool: PassagePool,
        checkout_dir: Optional[Union[str, Path]] = None,
        config: Optional[AgentConfig] = None,
        initial_retriever: Optional[Any] = None,
        initial_k: int = 100,
        top_n: int = 10,
        command_template: Optional[Sequence[str]] = None,
        python: Optional[str] = None,
        citation_base: int = 0,
        extra_env: Optional[Dict[str, str]] = None,
    ):
        super().__init__(config, valid_ids=pool.paper_ids)
        self.pool = pool
        self.checkout_dir = Path(checkout_dir) if checkout_dir else None
        self.initial_retriever = initial_retriever
        self.initial_k = int(initial_k)
        self.top_n = int(top_n)
        self.command_template = list(command_template or DEFAULT_COMMAND)
        self.python = python or sys.executable
        self.citation_base = int(citation_base)
        self.extra_env = dict(extra_env or {})
        self._workdir: Optional[Path] = None

    def check_available(self) -> None:
        if self.checkout_dir is None or not self.checkout_dir.is_dir():
            raise AgentUnavailableError(
                "OpenScholar checkout not found. Clone https://github.com/AkariAsai/OpenScholar, install its "
                "requirements, and pass checkout_dir=<path> (see scholarpath.baselines.agents.openscholar)."
            )
        script = self.command_template[1] if len(self.command_template) > 1 else ""
        if script.endswith(".py") and not (self.checkout_dir / script).exists():
            raise AgentUnavailableError(f"{script} not found in the OpenScholar checkout {self.checkout_dir}")
        if self.initial_retriever is None:
            raise AgentUnavailableError(
                "OpenScholarAgent needs an initial_retriever over the corpus datastore "
                "(e.g. scholarpath.baselines.dense.DenseRetriever over the same passage pool)."
            )

    @property
    def workdir(self) -> Path:
        if self._workdir is None:
            base = self.config.workdir
            self._workdir = Path(base) / "openscholar" if base else Path(tempfile.mkdtemp(prefix="openscholar-"))
            self._workdir.mkdir(parents=True, exist_ok=True)
        return self._workdir

    def prepare_datastore(self, out_dir: Optional[Union[str, Path]] = None) -> Path:
        """Write the pool as OpenScholar datastore passages (``passages.jsonl``)."""
        d = Path(out_dir) if out_dir else self.workdir / "datastore"
        d.mkdir(parents=True, exist_ok=True)
        path = d / "passages.jsonl"
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                for p in self.pool:
                    f.write(json.dumps({"id": p.key, "paper_id": p.paper_id, "title": p.title,
                                        "text": p.search_text}, ensure_ascii=False) + "\n")
        return path

    def _ctxs(self, query: Query) -> List[Dict[str, Any]]:
        hits = self.initial_retriever.search(query.text, self.initial_k)
        out = []
        for i, score in hits:
            p = self.pool[i]
            out.append({"id": p.key, "paper_id": p.paper_id, "title": f"{p.title} [{p.paper_id}]",
                        "text": p.search_text, "score": float(score)})
        return out

    def build_command(self, input_file: Path, output_file: Path, api_key_file: Path, model: str) -> List[str]:
        values = {"python": self.python, "input_file": str(input_file), "output_file": str(output_file),
                  "model": model, "api_key_file": str(api_key_file), "top_n": str(self.top_n),
                  "checkout_dir": str(self.checkout_dir)}
        return [arg.format(**values) for arg in self.command_template]

    def run(self, query: Query, prompt: str, endpoint: Optional[LLMEndpoint]) -> AgentRun:
        assert endpoint is not None
        self.prepare_datastore()
        qdir = Path(tempfile.mkdtemp(prefix=f"q-{re.sub(r'[^A-Za-z0-9_-]+', '_', query.query_id)}-",
                                     dir=self.workdir))
        ctxs = self._ctxs(query)
        input_file, output_file, key_file = qdir / "input.jsonl", qdir / "output.json", qdir / "api_key.txt"
        record = {"id": query.query_id, "input": prompt, "question": query.text, "ctxs": ctxs}
        input_file.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        key_file.write_text(endpoint.api_key, encoding="utf-8")
        os.chmod(key_file, 0o600)
        env = dict(os.environ)
        env.update({"OPENAI_BASE_URL": endpoint.base_url, "OPENAI_API_BASE": endpoint.base_url,
                    "OPENAI_API_KEY": endpoint.api_key, **self.extra_env})
        cmd = self.build_command(input_file, output_file, key_file, endpoint.model)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(self.checkout_dir), env=env, capture_output=True, text=True,
                              timeout=self.config.timeout_s)
        (qdir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (qdir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"OpenScholar failed (exit {proc.returncode}); see {qdir / 'stderr.log'}:\n"
                               f"{(proc.stderr or '')[-2000:]}")
        result = read_openscholar_output(output_file)
        output = str(result.get("output") or result.get("final_output") or "")
        out_ctxs = result.get("ctxs") or ctxs
        cited = map_numeric_citations(output, out_ctxs, self.citation_base)
        return AgentRun(output_text=output, cited_hint=cited,
                        metadata={"latency_s": time.perf_counter() - t0, "run_dir": str(qdir),
                                  "num_ctxs": len(out_ctxs)})


def read_openscholar_output(path: Union[str, Path]) -> Dict[str, Any]:
    """First result record of an OpenScholar output file (JSON list/object or JSONL)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except ValueError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]
    if isinstance(data, list):
        if not data:
            raise ValueError(f"empty OpenScholar output {path}")
        return dict(data[0])
    return dict(data)


def map_numeric_citations(output: str, ctxs: Sequence[Dict[str, Any]], base: int = 0) -> List[str]:
    """Paper ids of the ``[n]`` citations of an OpenScholar answer, in order."""
    out: List[str] = []
    for group in _NUM_CITE.findall(output or ""):
        for n in group.split(","):
            j = int(n.strip()) - base
            if 0 <= j < len(ctxs):
                pid = ctxs[j].get("paper_id") or ctxs[j].get("id")
                if pid and str(pid) not in out:
                    out.append(str(pid))
    return out
