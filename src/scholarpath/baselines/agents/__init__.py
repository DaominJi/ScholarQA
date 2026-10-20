"""Unbudgeted agent systems of Section 5.3 (Tables 3-5) over the local corpus.

=================  ==========================================================  =========================
system             adapter                                                     external dependency
=================  ==========================================================  =========================
PaperQA2 [72]      :class:`~.paperqa2.PaperQA2Agent`                            ``paper-qa`` package
OpenScholar [3]    :class:`~.openscholar.OpenScholarAgent`                      OpenScholar checkout
AgenticScholar     :class:`~.agenticscholar.AgenticScholarAgent`                callable / HTTP endpoint
Codex [51]         :class:`~.codex.CodexAgent`                                  ``codex`` CLI (v0.42)
(test only)        :class:`~.simulated.SimulatedAgent`                          none (MockLLM-compatible)
=================  ==========================================================  =========================

All adapters share :class:`~.base.AgentAdapter`: identical JSON-schema task
prompt, robust output parsing (:func:`~.base.parse_agent_output`), and usage
metering through :class:`~.base.MeteringProxy`. Missing external
dependencies raise :class:`~.base.AgentUnavailableError` at run time (never
at import time).
"""

from __future__ import annotations

from scholarpath.baselines.agents.agenticscholar import AgenticScholarAgent
from scholarpath.baselines.agents.base import (
    AgentAdapter,
    AgentConfig,
    AgentRun,
    AgentUnavailableError,
    LLMEndpoint,
    MeteringProxy,
    corpus_ids,
    normalize_paper_id,
    parse_agent_output,
)
from scholarpath.baselines.agents.codex import CodexAgent, codex_config_toml, parse_codex_events
from scholarpath.baselines.agents.corpus_export import (
    CorpusExport,
    export_corpus_shards,
    load_corpus_shards,
    read_paper,
)
from scholarpath.baselines.agents.openscholar import OpenScholarAgent
from scholarpath.baselines.agents.paperqa2 import PaperQA2Agent
from scholarpath.baselines.agents.simulated import SimulatedAgent

__all__ = [
    "AgentAdapter",
    "AgentConfig",
    "AgentRun",
    "AgentUnavailableError",
    "AgenticScholarAgent",
    "CodexAgent",
    "CorpusExport",
    "LLMEndpoint",
    "MeteringProxy",
    "OpenScholarAgent",
    "PaperQA2Agent",
    "SimulatedAgent",
    "codex_config_toml",
    "corpus_ids",
    "export_corpus_shards",
    "load_corpus_shards",
    "normalize_paper_id",
    "parse_agent_output",
    "parse_codex_events",
    "read_paper",
]
