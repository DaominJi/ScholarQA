"""Corpus containers, dataset loaders, and benchmark preprocessing.

* :mod:`.corpus`         -- :class:`Corpus`, JSONL I/O, query files
* :mod:`.synthetic`      -- synthetic benchmark for tests and smoke runs
* :mod:`.arxiv`          -- arXiv sets and the arXiv-Subfields corpus (Sec. 5.2, App. C.2)
* :mod:`.fulltext`       -- full text -> section chunks (Sec. 2.1)
* :mod:`.alignment`      -- aligning benchmark citations to corpus papers (App. C.2 "Coverage")
* :mod:`.researchqa`     -- ResearchQA test_mini
* :mod:`.scholarqabench` -- ScholarQA-CS and ScholarQA-Multi
* :mod:`.taxobench`      -- TaxoBench pooled corpus, gold taxonomies, organization queries
* :mod:`.chime`          -- CHIME human-corrected hierarchies, study corpus, organization queries
* :mod:`.download`       -- raw dataset downloads (``scripts/download_data.py``)
"""

# Corpus I/O first: other packages import these names from ``scholarpath.data``.
from scholarpath.data.corpus import Corpus, load_queries, read_json, read_jsonl, save_queries, write_json, write_jsonl
from scholarpath.data.alignment import (
    AlignmentMatch,
    CoverageStats,
    PaperAligner,
    QuoteAttributor,
    Reference,
    attach_s2_ids,
    parse_arxiv_id,
    resolve_references,
)
from scholarpath.data.arxiv import (
    ARXIV_SET_CATEGORIES,
    build_arxiv_sets,
    build_subfields_corpus,
    is_withdrawn,
    iter_arxiv_papers,
    record_to_paper,
    write_arxiv_corpora,
)
from scholarpath.data.chime import ChimeData, build_chime, parse_generated_hierarchy
from scholarpath.data.download import DATASETS, download_dataset, read_json_records
from scholarpath.data.fulltext import FulltextExtractor, latex_to_sections, make_sources, s2orc_sections, sections_to_chunks
from scholarpath.data.researchqa import load_researchqa, researchqa_to_queries
from scholarpath.data.scholarqabench import EvidenceClassifier, load_scholarqa_cs, load_scholarqa_multi
from scholarpath.data.taxobench import TaxoBenchData, load_taxobench, merge_gold_taxonomies

__all__ = [
    # corpus I/O
    "Corpus", "load_queries", "read_json", "read_jsonl", "save_queries", "write_json", "write_jsonl",
    # alignment
    "AlignmentMatch", "CoverageStats", "PaperAligner", "QuoteAttributor", "Reference", "attach_s2_ids",
    "parse_arxiv_id", "resolve_references",
    # arXiv
    "ARXIV_SET_CATEGORIES", "build_arxiv_sets", "build_subfields_corpus", "is_withdrawn", "iter_arxiv_papers",
    "record_to_paper", "write_arxiv_corpora",
    # full text
    "FulltextExtractor", "latex_to_sections", "make_sources", "s2orc_sections", "sections_to_chunks",
    # benchmarks
    "ChimeData", "build_chime", "parse_generated_hierarchy", "load_researchqa", "researchqa_to_queries",
    "EvidenceClassifier", "load_scholarqa_cs", "load_scholarqa_multi", "TaxoBenchData", "load_taxobench", "merge_gold_taxonomies",
    # downloads
    "DATASETS", "download_dataset", "read_json_records",
]
