"""Evaluation metrics, LLM judges, efficiency accounting and reporting (Sections 5.2-5.3).

* :mod:`.retrieval_metrics`   -- Recall@B / NDCG@B over gold source papers (Table 2, Fig. 5);
* :mod:`.qa_metrics`          -- ResearchQA Rub / Cite, ScholarQA-CS Corr, statement-level
  citation F1, ScholarQA-Multi Prometheus scores, ScholarQABench combination (Table 3);
* :mod:`.hypothesis_metrics`  -- Disc (Hit@10) and Feasibility / Novelty / Soundness judge
  scores on held-out hypothesis queries (Table 4, Fig. 6);
* :mod:`.judges`              -- the judge panel (GPT-6.0-Sol, Opus-5.5, Grok-4.7);
* :mod:`.efficiency`          -- per-query calls / tokens / cost / latency (Table 5);
* :mod:`.reporting`           -- per-query JSONL, summary CSVs, Markdown tables, budget sweeps;
* :mod:`.official`            -- exporters to, and runners of, the official evaluation scripts;
* :mod:`.taxonomy_metrics`    -- soft node / edge F1 (Section 5.2);
* :mod:`.organization_metrics` -- ARI, semantic path similarity, category soft F1, assignment F1.

Every symbol is resolved lazily (PEP 562) on first access, so importing this
package (or one of its submodules) does not import the others. The public
names of the first seven modules are listed in ``__all__``. A few symbols of
:mod:`.taxonomy_metrics` and :mod:`.organization_metrics` are re-exported as
well, but only if their module imports cleanly: they are left out of
``__all__`` so that ``from scholarpath.evaluation import *`` never depends on
them, and accessing one whose module fails to import raises
:class:`AttributeError` naming the import error.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

_LAZY: Dict[str, List[str]] = {
    "retrieval_metrics": [
        "BENCHMARK_DISPLAY", "RetrievalEvaluation", "canonical_benchmark", "dcg", "evaluate_retrieval",
        "ideal_dcg", "ndcg_at_budget", "papers_within_budget", "ranked_papers_of", "recall_at_budget",
    ],
    "qa_metrics": [
        "AT_MOST_CITATIONS", "ATTRSCORE_MODEL", "AttrScoreNLI", "AttributionJudge", "CSRubricConfig",
        "CSRubricProperty", "CSScore", "CitationAttribution", "CoverageResult", "LLMAttributionJudge", "PRF",
        "PROMETHEUS_ASPECTS", "PrometheusResult", "QAEvaluation", "RubricJudgement",
        "aggregate_citation_attribution", "aggregate_correctness", "aggregate_coverage", "aggregate_prf",
        "aggregate_prometheus", "answer_citation_attribution", "answer_passages", "answer_text",
        "citation_attribution", "citation_f1", "citation_prf", "cited_papers", "correctness",
        "evaluate_researchqa", "evaluate_scholarqa_cs", "evaluate_scholarqa_multi", "extract_citation_indices",
        "f1_score", "format_document", "format_reference", "has_passage_source", "length_score",
        "normalize_5scale", "number_citations", "prometheus_grade", "prometheus_to_100", "remove_citations",
        "render_answer", "render_cs_answer", "rubric_coverage", "rubric_item_types", "scholarqabench_combined",
        "sentence_splitter", "split_sentences",
    ],
    "hypothesis_metrics": [
        "CRITERIA", "DiscoveryResult", "ExpectedRecord", "HIT_K", "HypothesisJudgement", "ProposedIdea",
        "TABLE4_COLUMNS", "aggregate_hypothesis", "discovery_hit", "discovery_rate", "evaluate_hypotheses",
        "expected_record", "is_proposal", "judge_hypotheses", "proposed_ideas",
    ],
    "judges": [
        "DEFAULT_JUDGE_CACHE", "JUDGE_COMPONENT", "Judge", "JudgeLike", "JudgePanel", "JudgeSpec", "PAPER_JUDGES",
        "PROMETHEUS_MODEL", "RESEARCHQA_JUDGE_MODEL", "SCHOLARQA_CS_JUDGE_MODEL", "as_judge_llm", "as_panel",
        "build_judge_llm",
    ],
    "efficiency": [
        "EXCLUDED_COMPONENTS", "EfficiencyTracker", "QueryEfficiency", "TABLE5_COLUMNS", "annotate_answer",
        "measure", "price", "summarize_efficiency",
    ],
    "reporting": [
        "Column", "ORGANIZATION_COLUMNS", "PAPER_TABLES", "TABLE2", "TABLE3", "TABLE4", "TABLE5", "TableSpec",
        "flatten_benchmarks", "format_value", "markdown_table", "normalize_sweep", "organization_table_row",
        "plot_budget_sweep", "sweep_long_rows", "sweep_wide_rows", "table_rows", "write_budget_sweep",
        "write_csv", "write_json", "write_jsonl", "write_run", "write_summary_csv", "write_table",
    ],
    "official": [
        "OfficialCoverage", "OfficialEvalError", "OfficialPaths", "coverage_from_judgements",
        "export_citation_eval", "export_prometheus", "export_researchqa_responses", "export_scholarqa_cs",
        "official_id", "run_citation_eval", "run_compute_coverage", "run_prometheus_eval", "run_rubric_eval",
    ],
}

# Section 5.2 / organization metrics live in modules maintained separately; their
# symbols are re-exported only when the module imports cleanly (see __getattr__).
_OPTIONAL: Dict[str, List[str]] = {
    "taxonomy_metrics": ["soft_node_f1", "soft_edge_f1", "taxonomy_scores"],
    "organization_metrics": [
        "adjusted_rand_index", "semantic_path_similarity", "category_soft_f1", "assignment_f1",
        "organization_scores", "ORGANIZATION_METRICS",
    ],
}

_SYMBOL_TO_MODULE: Dict[str, str] = {}
for _mod, _names in _LAZY.items():
    for _name in _names:
        _SYMBOL_TO_MODULE.setdefault(_name, _mod)
_OPTIONAL_TO_MODULE: Dict[str, str] = {n: m for m, names in _OPTIONAL.items() for n in names}


def _resolve_optional(name: str) -> Any:
    module_name = f"{__name__}.{_OPTIONAL_TO_MODULE[name]}"
    try:
        module = importlib.import_module(module_name)
    except Exception as err:  # noqa: BLE001 - an optional module may be missing or broken
        raise AttributeError(f"{__name__}.{name} is unavailable: {module_name} failed to import ({err!r})") from err
    if not hasattr(module, name):
        raise AttributeError(f"{__name__}.{name} is unavailable: {module_name} does not define it")
    return getattr(module, name)


def __getattr__(name: str) -> Any:
    if name in _SYMBOL_TO_MODULE:
        value = getattr(importlib.import_module(f"{__name__}.{_SYMBOL_TO_MODULE[name]}"), name)
    elif name in _OPTIONAL_TO_MODULE:
        value = _resolve_optional(name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_SYMBOL_TO_MODULE) | set(_OPTIONAL_TO_MODULE))


__all__ = sorted(_SYMBOL_TO_MODULE)
