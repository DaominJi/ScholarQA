"""PaperQA2 adapter [72] (Tables 3-5, "PaperQA2").

Runs the ``paper-qa`` package (PaperQA2 agent: paper search -> evidence
gathering -> answer generation) over the local corpus with the configured
OpenAI-compatible backbone:

* every corpus paper is written as a text document ``<paper_id>.txt``
  (title, paper id, abstract and section chunks) with a ``manifest.csv``
  that fixes ``docname = paper_id`` and the title / citation, and
  ``parsing.use_doc_details = False`` so that PaperQA2 never queries external
  metadata services (Crossref / Semantic Scholar) -- external web access is
  disabled for every agent;
* the agent LLM, the answer LLM and the evidence-summary LLM all point at the
  backbone through LiteLLM's OpenAI-compatible provider: model name
  ``openai/<model>`` with ``api_base`` = the metering proxy (or the backend);
* the question is the uniform JSON-schema task prompt
  (:func:`~scholarpath.prompts.baselines.render_agent_task`).

Setup::

    pip install "paper-qa>=5"          # or: pip install -e ".[agents]"
    pip install sentence-transformers  # for the default local "st-" embedding

The text embedding model is configurable (``embedding``); the backbone
provider has no embedding endpoint, so a local sentence-transformers model is
used by default in PaperQA2's hybrid (dense + sparse) mode.
"""

from __future__ import annotations

import csv
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from scholarpath.baselines.agents.base import AgentAdapter, AgentConfig, AgentRun, AgentUnavailableError, LLMEndpoint
from scholarpath.data.corpus import Corpus
from scholarpath.types import Paper, Query

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "The PaperQA2 baseline needs the 'paper-qa' package (>= 5): pip install 'paper-qa>=5' "
    "(or pip install -e '.[agents]'). The default embedding additionally needs sentence-transformers."
)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(paper_id: str) -> str:
    return _SAFE.sub("_", paper_id).strip("._") or "paper"


def paper_document(paper: Paper, include_chunks: bool = True) -> str:
    """Plain-text document PaperQA2 indexes for one paper."""
    parts = [f"Title: {paper.title}", f"Paper ID: {paper.paper_id}", "", "Abstract", paper.abstract or ""]
    if include_chunks:
        for ch in sorted(paper.chunks, key=lambda c: c.order):
            parts += ["", ch.heading or "Section", ch.text]
    return "\n".join(parts).strip() + "\n"


class PaperQA2Agent(AgentAdapter):
    """PaperQA2 over the local corpus with the configured LLM backbone."""

    name = "PaperQA2"

    def __init__(
        self,
        corpus: Union[Corpus, Dict[str, Paper], Iterable[Paper]],
        config: Optional[AgentConfig] = None,
        embedding: str = "hybrid-st-multi-qa-MiniLM-L6-cos-v1",
        include_chunks: bool = True,
        settings_overrides: Optional[Dict[str, Any]] = None,
    ):
        papers = list(corpus) if not isinstance(corpus, dict) else list(corpus.values())
        super().__init__(config, valid_ids=[p.paper_id for p in papers])
        self.papers = papers
        self.embedding = embedding
        self.include_chunks = bool(include_chunks)
        self.settings_overrides = dict(settings_overrides or {})
        self._workdir: Optional[Path] = None
        self._docname_to_id: Dict[str, str] = {}

    def check_available(self) -> None:
        try:
            import paperqa  # noqa: F401
        except ImportError as err:
            raise AgentUnavailableError(INSTALL_HINT) from err

    # --------------------------------------------------------- documents
    @property
    def workdir(self) -> Path:
        if self._workdir is None:
            base = self.config.workdir
            self._workdir = Path(base) / "paperqa2" if base else Path(tempfile.mkdtemp(prefix="paperqa2-"))
            self._workdir.mkdir(parents=True, exist_ok=True)
        return self._workdir

    def prepare_documents(self) -> Path:
        """Write ``papers/<id>.txt`` and ``papers/manifest.csv`` (idempotent)."""
        paper_dir = self.workdir / "papers"
        manifest = paper_dir / "manifest.csv"
        if manifest.exists():
            self._docname_to_id = {safe_filename(p.paper_id): p.paper_id for p in self.papers}
            return paper_dir
        paper_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for p in self.papers:
            fname = safe_filename(p.paper_id)
            (paper_dir / f"{fname}.txt").write_text(paper_document(p, self.include_chunks), encoding="utf-8")
            self._docname_to_id[fname] = p.paper_id
            rows.append({"file_location": f"{fname}.txt", "doi": "", "title": p.title, "key": fname,
                         "docname": fname, "citation": f"{p.title} [{p.paper_id}]"})
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file_location", "doi", "title"])
            writer.writeheader()
            writer.writerows(rows)
        return paper_dir

    def build_settings(self, endpoint: LLMEndpoint) -> Any:
        """PaperQA2 ``Settings`` pointing every LLM role at the backbone."""
        from paperqa import Settings

        paper_dir = self.prepare_documents()
        llm_name = f"openai/{endpoint.model}"
        llm_config = {
            "model_list": [
                {
                    "model_name": llm_name,
                    "litellm_params": {"model": llm_name, "api_base": endpoint.base_url,
                                       "api_key": endpoint.api_key, "temperature": 0.0},
                }
            ]
        }
        data: Dict[str, Any] = {
            "llm": llm_name,
            "llm_config": llm_config,
            "summary_llm": llm_name,
            "summary_llm_config": llm_config,
            "embedding": self.embedding,
            "temperature": 0.0,
            "parsing": {"use_doc_details": False},
            "agent": {
                "agent_llm": llm_name,
                "agent_llm_config": llm_config,
                "index": {
                    "paper_directory": str(paper_dir),
                    "manifest_file": str(paper_dir / "manifest.csv"),
                    "index_directory": str(self.workdir / "index"),
                },
            },
        }
        _deep_update(data, self.settings_overrides)
        try:
            return Settings(**data)
        except Exception:  # older paper-qa 5.x: paper_directory at the top level
            idx = data["agent"].pop("index")
            data["paper_directory"] = idx["paper_directory"]
            data["agent"].pop("agent_llm_config", None)
            return Settings(**data)

    # --------------------------------------------------------------- run
    def run(self, query: Query, prompt: str, endpoint: Optional[LLMEndpoint]) -> AgentRun:
        assert endpoint is not None
        from paperqa import ask

        settings = self.build_settings(endpoint)
        response = ask(prompt, settings=settings)
        session = getattr(response, "session", response)
        text = getattr(session, "raw_answer", None) or getattr(session, "answer", "") or ""
        formatted = getattr(session, "formatted_answer", "") or text
        cited: List[str] = []
        for ctx in getattr(session, "contexts", []) or []:
            doc = getattr(getattr(ctx, "text", None), "doc", None)
            docname = str(getattr(doc, "docname", "") or "")
            pid = self._docname_to_id.get(docname)
            if pid and (docname in formatted or pid in formatted) and pid not in cited:
                cited.append(pid)
        tokens = getattr(session, "token_counts", {}) or {}
        inp = sum(int(v[0]) for v in tokens.values() if isinstance(v, (list, tuple)) and v)
        out = sum(int(v[1]) for v in tokens.values() if isinstance(v, (list, tuple)) and len(v) > 1)
        cost = getattr(session, "cost", None)
        return AgentRun(
            output_text=text if "{" in text else formatted,
            input_tokens=inp or None,
            output_tokens=out or None,
            cost_usd=float(cost) if cost is not None else None,
            cited_hint=cited,
            metadata={"formatted_answer": formatted},
        )


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
