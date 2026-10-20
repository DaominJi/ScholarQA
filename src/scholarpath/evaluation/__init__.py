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
* :mod:`.taxonomy_metrics`    -- soft node / edge F1 (Section 5.2; :class:`TaxonomyScorer`);
* :mod:`.organization_metrics` -- ARI, semantic path similarity, category soft F1, assignment F1
  (TaxoBench / CHIME organization queries, Table 3).

Every symbol is resolved lazily (PEP 562) on first access, so importing this
package (or one of its submodules) does not import the others, and the
scikit-learn-backed taxonomy / organization metrics are only loaded when used.
``__all__`` lists the public names of every module; on a name clash (``PRF``,
``f1_score``) the QA-metrics definition is the package-level one.
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
        "sentence_splitter", "split_sentences", "MISSING_POLICIES",
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
        "OFFICIAL_BENCHMARKS", "OfficialCoverage", "OfficialEvalError", "OfficialPaths", "coverage_from_judgements",
        "export_citation_eval", "export_prometheus", "export_researchqa_data", "export_researchqa_responses",
        "export_scholarqa_cs",
        "official_id", "run_citation_eval", "run_compute_coverage", "run_official_evaluation",
        "run_prometheus_eval", "run_rubric_eval",
    ],
    "taxonomy_metrics": [
        "AGGREGATES", "COMBINERS", "SimilarityTables", "TEXT_MODES", "TaxonomyScorer", "soft_edge_f1",
        "soft_edge_prf", "soft_node_f1", "soft_node_prf", "taxonomy_scores",
    ],
    "organization_metrics": [
        "ARI_E2E", "ARI_INTERSECTION", "DEFAULT_CATEGORY_THRESHOLD", "MISSING_LABEL", "ORGANIZATION_METRICS",
        "PUBLIC_RULE", "STRICT_RULE", "TAXOBENCH_EMBEDDING_MODEL", "TAXOBENCH_THRESHOLD", "NodeSimilarity",
        "PaperAlignment", "PaperMatch", "adjusted_rand_index", "align_categories", "align_papers",
        "ancestor_chains", "assignment_f1", "assignment_prf", "benchmark_title_similarity", "category_soft_f1",
        "category_soft_prf", "chain_alignment_cost", "leaf_clustering_scores", "load_taxobench_embedder",
        "macro_average", "merge_hierarchies", "normalize_benchmark_title", "organization_scores",
        "paper_attachments", "semantic_path_scores", "semantic_path_similarity", "subtree", "to_taxobench_tree",
    ],
}

_SYMBOL_TO_MODULE: Dict[str, str] = {}
for _mod, _names in _LAZY.items():
    for _name in _names:
        _SYMBOL_TO_MODULE.setdefault(_name, _mod)


def __getattr__(name: str) -> Any:
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{_SYMBOL_TO_MODULE[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_SYMBOL_TO_MODULE))


__all__ = sorted(_SYMBOL_TO_MODULE)
