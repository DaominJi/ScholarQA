"""Tests for the benchmark loaders (Sec. 5.3, App. C.2) and the download helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scholarpath.data import chime, download, researchqa, scholarqabench, taxobench
from scholarpath.data.alignment import PaperAligner, QuoteAttributor
from scholarpath.data.arxiv import build_subfields_corpus
from scholarpath.data.corpus import Corpus, load_queries
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.text import normalize_title
from scholarpath.types import EVIDENCE

FIX = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def subfields():
    """A miniature arXiv-Subfields corpus built from the snapshot fixture."""
    return build_subfields_corpus(FIX / "arxiv" / "arxiv-metadata-sample.json")


@pytest.fixture(scope="module")
def aligner(subfields):
    return PaperAligner(subfields)


# ---------------------------------------------------------------------------
# ResearchQA
# ---------------------------------------------------------------------------


def test_researchqa_queries_rubrics_and_gold(aligner):
    queries, cov = researchqa.load_researchqa(FIX / "researchqa", aligner)
    assert [q.query_id for q in queries] == ["100000000000000001-s1", "100000000000000002-s3", "100000000000000003-s2"]
    q = queries[0]
    assert q.kind == EVIDENCE and q.benchmark == "researchqa"
    assert [r.kind for r in q.rubric] == ["limitation", "comparison,example", "citation", "citation", "citation",
                                          "citation", "citation"]
    assert all(r.weight == 1.0 for r in q.rubric)
    assert q.metadata["rubric_types"][1] == ["Comparison", "Example"]
    assert q.metadata["date"] == "2021-06-05" and q.metadata["raw_date"] == "2021-6-5"
    assert q.metadata["domain"] == "Engineering" and q.metadata["field"] == "Databases"
    # arXiv URL -> 2101.00001; exact title -> 2101.00002; the repeated Smith citation counts once;
    # DOI-only / unknown titles stay in the denominator.
    assert q.gold_papers == ["2101.00001", "2101.00002"]
    assert q.unaligned_gold == ["doi:10.1145/5555.6666", "title:access path selection in a relational database management system"]
    assert q.num_gold == 4
    # citation_metadata missing -> title parsed from the rubric text (balanced parentheses) -> fuzzy containment
    assert queries[1].gold_papers == ["math/0601001"] and queries[1].metadata["date"] == "2019-11-20"
    assert queries[2].num_gold == 0 and queries[2].metadata["date"] == "2022-01-01"
    assert (cov.total, cov.aligned) == (5, 3)
    assert researchqa.rubric_item_count(queries) == 10


def test_researchqa_helpers_and_extra_gold(tmp_path):
    assert researchqa.normalize_date("2021-1-12") == "2021-01-12"
    assert researchqa.normalize_date("2021-3") == "2021-03-01"
    assert researchqa.normalize_date("") is None
    text = ('Does the response cite the research (titles: Wang et al. (2020) (title: Synergistic effects of nano-silica) '
            'and "Reduction of heavy metals leaching by adding nano-SiO2")?')
    assert researchqa.titles_from_rubric_text(text) == ["Synergistic effects of nano-silica",
                                                        "Reduction of heavy metals leaching by adding nano-SiO2"]
    assert researchqa.item_types({"type": "Citation, Example"}) == ["Citation", "Example"]
    extra = tmp_path / "extra.jsonl"
    extra.write_text(json.dumps({"id": "100000000000000003-s2", "references": [{"arxiv_id": "2107.00009"}]}) + "\n")
    queries, _ = researchqa.load_researchqa(FIX / "researchqa" / "test_mini.json", limit=3, extra_gold_path=extra)
    assert queries[2].unaligned_gold == ["arxiv:2107.00009"]  # no aligner -> unaligned but counted


# ---------------------------------------------------------------------------
# ScholarQABench
# ---------------------------------------------------------------------------


def test_scholarqa_cs_dedup_rubric_and_evidence_origins(subfields, aligner):
    root = FIX / "scholarqabench"
    cases = scholarqabench.read_cs_cases(root)
    attributor = QuoteAttributor(scholarqabench.cs_evidence_quotes(cases))
    attributor.scan_corpus(subfields)
    queries, cov, origins = scholarqabench.load_scholarqa_cs(root, aligner=aligner, attributor=attributor)
    assert [q.query_id for q in queries] == ["case-q1-a", "case-q2"]  # first annotation per question kept
    q1, q2 = queries
    assert [a["case_id"] for a in q1.metadata["alternate_cases"]] == ["case-q1-b"]
    assert [r.kind for r in q1.rubric] == ["most_important", "most_important", "nice_to_have"]
    assert abs(sum(r.weight for r in q1.rubric) - 0.6) < 1e-9
    assert q1.metadata["metric_config"]["name"] == "rubric_corpusqa_generic"
    by_origin = {s["origin"] for s in q1.metadata["evidence_sources"]}
    assert by_origin == {"nora_citation", "paper_quote_attributed", "nora_answer", "paper_quote", "llm_answer"}
    nora_answer = [s for s in q1.metadata["evidence_sources"] if s["origin"] == "nora_answer"][0]
    assert [r["s2_corpus_id"] for r in nora_answer["references"]] == ["199543209"]  # only the adjacent marker
    # attributed quote -> 2101.00001; Nora citations (no S2 ids in the toy corpus) and the unknown quote unaligned
    assert q1.gold_papers == ["2101.00001"]
    assert set(q1.unaligned_gold) == {"s2corpus:53221591", "s2corpus:199543209",
                                      [s["key"] for s in q1.metadata["evidence_sources"] if s["origin"] == "paper_quote"][0]}
    assert q2.gold_papers == ["2101.00002"] and len(q2.unaligned_gold) == 1  # short quote stays unaligned
    assert origins["llm_answer"] == 1 and cov.total == 6 and cov.aligned == 2
    configs = scholarqabench.cs_test_configs(queries, include_alternates=True)
    assert len(configs) == 3 and configs[0]["initial_prompt"] == q1.text


def test_scholarqa_cs_without_sources_or_dedupe():
    queries, cov, origins = scholarqabench.load_scholarqa_cs(
        FIX / "scholarqabench" / "data" / "scholarqa_cs" / "test_configs_snippets.json", dedupe=False)
    assert len(queries) == 3 and set(origins) == {"paper_quote"}
    assert cov.aligned == 0 and all(not q.gold_papers for q in queries)


def test_scholarqa_multi(aligner):
    queries, cov = scholarqabench.load_scholarqa_multi(FIX / "scholarqabench", aligner)
    q = queries[0]
    assert q.benchmark == "scholarqa_multi" and q.reference_answer.startswith("Learned estimators")
    assert q.metadata["citation_marker_base"] == 0 and q.metadata["cited_ctx_indices"] == [0, 1, 2, 3]
    assert q.gold_papers == ["2101.00001", "2101.00002"]  # duplicated ctx counted once
    assert q.unaligned_gold == ["s2:" + "c" * 40]
    assert queries[1].unaligned_gold == ["s2corpus:210983829"]  # CorpusId stored as title
    assert (cov.total, cov.aligned) == (4, 2)
    assert scholarqabench.cited_indices("see [0], [2, 5] and [9]", 6) == [0, 2, 5]


# ---------------------------------------------------------------------------
# TaxoBench
# ---------------------------------------------------------------------------


def test_taxobench_pooled_corpus_and_gold(tmp_path):
    data = taxobench.load_taxobench(FIX / "taxobench")
    assert data.stats["surveys"] == 3 and data.stats["citations"] == 10
    assert data.stats["distinct_papers"] == 7  # duplicate titles across surveys merged by normalized title
    gnn = [p for p in data.corpus if normalize_title(p.title) == "graph neural networks for traffic forecasting"]
    assert len(gnn) == 1 and gnn[0].metadata["surveys"] == [0, 1]
    assert gnn[0].paper_id == taxobench.taxobench_paper_id("Graph neural networks for traffic forecasting.")
    assert gnn[0].metadata["title_variants"] == ["Graph neural networks for traffic forecasting."]
    merged = data.merged
    root = merged.root
    assert root.name == taxobench.MERGED_ROOT_NAME
    tops = [merged[c].name for c in root.children]
    assert tops == ["Large Models for Time Series - A Survey", "Retrieval-Augmented Question Answering",
                    "Neural Information Retrieval"]  # survey roots relabelled by survey_topic
    # papers are extents, not nodes; a paper cited by two surveys belongs to both
    names = {n.name for n in merged.nodes.values()}
    assert not names & {p.title for p in data.corpus}
    assert merged.paper_nodes(gnn[0].paper_id) >= {root.children[0], root.children[1]}
    assert taxobench.taxonomy_stats(merged)["concept_nodes"] == 3 + 6 + 3 + 2
    assert len(root.papers) == 7
    t0 = data.survey_taxonomies[0]
    assert t0.root.name == "Large Models for Time Series - A Survey" and t0.max_depth == 3
    assert not t0.metadata.get("unmatched_leaf_titles")
    trunc = taxobench.merge_gold_taxonomies(taxobench.read_taxobench(FIX / "taxobench"), max_depth=2,
                                            include_papers=False)
    assert trunc.max_depth == 2 and len(trunc.root.papers) == 0
    q = data.queries[1]
    assert q.text == "Retrieval-Augmented Question Answering" and q.kind == EVIDENCE
    assert q.metadata["task"] == "organization" and q.metadata["survey_id"] == 1 and q.metadata["style"] == "survey"
    assert len(q.gold_papers) == 4 and gnn[0].paper_id in q.gold_papers
    paths = data.save(tmp_path)
    assert Taxonomy.load(paths["gold_taxonomy"]).to_dict() == merged.to_dict()
    assert (tmp_path / "gold" / "2.json").exists() and len(load_queries(paths["queries"])) == 3
    assert len(Corpus.load(paths["papers"])) == 7


# ---------------------------------------------------------------------------
# CHIME
# ---------------------------------------------------------------------------

STYLES = {
    "aspect": """[Response 2]:
Aspect 1: Treatments (Claim 0, 1)
1.1: Drugs (Claims: 0)
1.2: Surgery (Claim 1)
Aspect 2: Outcomes (Claim 2)""",
    "dash": """[Response 2]:
- Treatments (Claim 0, 1)
  - Drugs (Claim 0)
    - Antibiotics (Claim 0)
  - Surgery (Claim 1)
- Outcomes (Claim 2-4)""",
    "plain_numbered": """[Response 2]:
Treatments (Claim 0, 1)
1. Drugs
   1.1. Antibiotics (Claim 0)
2. Surgery (Claim 1)

Outcomes (Claim 2)""",
}


@pytest.mark.parametrize("style", sorted(STYLES))
def test_chime_hierarchy_parser_styles(style):
    aspects = chime.parse_generated_hierarchy(STYLES[style])
    assert [a.name for a in aspects] == ["Treatments", "Outcomes"]
    treat = aspects[0]
    assert treat.claims == [0, 1]
    assert [c.name for c in treat.children] == ["Drugs", "Surgery"]
    assert treat.all_claims() == {0, 1}
    if style == "dash":
        assert treat.children[0].children[0].name == "Antibiotics"
        assert aspects[1].claims == [2, 3, 4]  # ranges are expanded
    assert chime.add_uncategorized(aspects, 6)[-1].claims == ([5] if style == "dash" else [3, 4, 5])


def test_chime_claim_helpers():
    assert chime.parse_claim_ids("Name (Claim 1, 3-5)") == ("Name", [1, 3, 4, 5])
    assert chime.parse_claim_ids("Name (Papers: 2 and 7)") == ("Name", [2, 7])
    assert chime.clean_claim("New Finding:  Effect  was Large.") == "effect was large."


def test_chime_with_annotations(tmp_path):
    data = chime.build_chime(FIX / "chime", FIX / "chime_annotations")
    st = data.stats
    assert st["corrected"] and st["topics"] == 2 and st["gold_hierarchies"] == 2
    assert st["dropped_edges"] == 1 and st["rejected_assignments"] == 2 and st["unmatched_annotations"] == 0
    assert st["incoherent_sibling_groups"] == 1
    tax = data.gold["CD900001"]
    names = [tax[n].name for n in tax.bfs()]
    assert "Population subgroups" not in names  # hierarchy not annotated by the experts
    assert "Long-term effects" not in names  # parent-child edge rejected -> subtree dropped
    onset = next(n for n in tax.nodes.values() if n.name == "Impact on smoking onset")
    assert onset.papers == {"pmid:11000001"}  # claim 2 rejected for this category
    assert tax.metadata["hierarchy_ids"] == [0]
    music = next(n for n in data.gold["CD900002"].nodes.values() if n.name == "Music")
    assert music.papers == set()
    q1, q2 = data.queries
    assert q1.query_id == "chime-CD900001" and q1.metadata["task"] == "organization"
    assert q1.metadata["review_pmid"] == "90000001" and q1.metadata["style"] == "review"
    assert q1.gold_papers == ["pmid:11000001", "pmid:11000002", "pmid:11000003", "pmid:11000004"]
    assert "pmid:11000002" in q2.gold_papers  # a study included by two reviews
    shared = data.corpus["pmid:11000002"]
    assert shared.metadata["reviews"] == ["CD900001", "CD900002"] and shared.title.startswith("Parental monitoring")
    assert len(data.corpus) == 7
    data.save(tmp_path)
    assert Taxonomy.load(tmp_path / "gold" / "CD900002.json").root.name.startswith("Interventions for improving sleep")
    assert len(load_queries(tmp_path / "queries.jsonl")) == 2


def test_chime_annotation_split_directories(tmp_path):
    """HF-style layout: one directory per dataset with per-split files (plus metadata files to ignore)."""
    import csv

    for src in sorted((FIX / "chime_annotations").glob("*.csv")):
        rows = list(csv.DictReader(src.open(newline="")))
        d = tmp_path / src.stem / "data"
        d.mkdir(parents=True)
        (tmp_path / src.stem / "README.md").write_text("# card")
        (tmp_path / src.stem / "dataset_infos.json").write_text("{}")
        for split in ("train", "test_ID"):
            part = [r for r in rows if r["split"] == split]
            with (d / f"{split}-00000-of-00001.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "split"])
                w.writeheader()
                w.writerows({k: v for k, v in r.items() if k != "split"} for r in part)
    tables = chime.find_annotation_tables(tmp_path)
    assert set(tables) == set(chime.ANNOTATION_TABLES)
    rows = chime.read_table(tables["chime-claim-category"])
    assert len(rows) == 7 and {r["split"] for r in rows} == {"train", "test_ID"}
    data = chime.build_chime(FIX / "chime", tmp_path)
    assert data.stats["rejected_assignments"] == 2 and data.stats["dropped_edges"] == 1


def test_chime_fallback_and_options():
    data = chime.build_chime(FIX / "chime")  # no annotations -> uncorrected generated hierarchies
    assert not data.stats["corrected"] and data.stats["gold_hierarchies"] == 4
    assert len(data.queries[0].gold_papers) == 5
    every = chime.build_chime(FIX / "chime", topics="all", gold_studies="all", include_uncategorized=True)
    assert every.stats["topics"] == 3 and len(every.corpus) == 11 - 1  # one study shared by two reviews
    with pytest.raises(FileNotFoundError):
        chime.build_chime(FIX / "chime", FIX / "missing", require_annotations=True)
    with pytest.raises(ValueError):
        chime.build_chime(FIX / "chime", gold_studies="some")


# ---------------------------------------------------------------------------
# Downloads and scripts
# ---------------------------------------------------------------------------


def test_download_dry_run_plans(tmp_path):
    for name in download.DATASETS:
        res = download.download_dataset(name, tmp_path, dry_run=True)
        assert res.status == "planned" and res.commands, name
    assert download.plan_download("taxobench", tmp_path)[0].startswith("git clone https://github.com/KongLongGeFDU/TaxoBench")
    assert "test_mini-00000-of-00001.parquet" in download.plan_download("researchqa", tmp_path)[0]
    assert "kaggle datasets download -d Cornell-University/arxiv" in download.plan_download("arxiv", tmp_path)[0]
    assert len(download.plan_download("chime_annotations", tmp_path)) == 3
    with pytest.raises(KeyError):
        download.download_dataset("nope", tmp_path, dry_run=True)


def test_download_idempotence_and_clear_failures(tmp_path, monkeypatch):
    target = tmp_path / "taxobench"
    target.mkdir()
    (target / download.MARKER).write_text("{}")
    assert download.download_dataset("taxobench", tmp_path).status == "skipped"
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    res = download.download_dataset("s2orc_v2", tmp_path)
    assert res.status == "failed" and "S2_API_KEY" in res.message
    monkeypatch.setattr(download.shutil, "which", lambda _: None)
    monkeypatch.setitem(sys.modules, "kagglehub", None)  # import fails
    res = download.download_dataset("arxiv", tmp_path)
    assert res.status == "failed" and "kaggle" in res.message.lower() and "data/raw/arxiv" in res.message


def test_download_script_dry_run(capsys):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import download_data
    finally:
        sys.path.pop(0)
    assert download_data.main(["--dry-run", "--json", "--only", "taxobench,chime"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [r["name"] for r in out] == ["taxobench", "chime"] and all(r["status"] == "planned" for r in out)
    assert download_data.main(["--list"]) == 0
    with pytest.raises(SystemExit):
        download_data.main(["--only", "unknown"])


def test_prepare_benchmarks_script(tmp_path):
    corpus_path = tmp_path / "subfields.jsonl.gz"
    build_subfields_corpus(FIX / "arxiv" / "arxiv-metadata-sample.json").save(corpus_path)
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import prepare_benchmarks
    finally:
        sys.path.pop(0)
    out = tmp_path / "processed"
    rc = prepare_benchmarks.main(["--raw", str(FIX), "--out", str(out), "--corpus", str(corpus_path),
                                  "--researchqa", str(FIX / "researchqa"),
                                  "--chime-annotations", str(FIX / "chime_annotations")])
    assert rc == 0
    report = json.loads((out / "coverage_report.json").read_text())
    assert set(report) == {"researchqa", "scholarqabench", "taxobench", "chime"}
    assert report["researchqa"]["coverage"]["aligned"] == 3
    assert report["scholarqabench"]["queries"] == {"cs": 2, "multi": 2}
    assert report["taxobench"]["coverage"]["coverage"] == 1.0 and report["chime"]["corrected"]
    for rel in ("researchqa/queries.jsonl", "scholarqabench/cs.jsonl", "scholarqabench/multi.jsonl",
                "taxobench/gold_taxonomy.json", "taxobench/papers.jsonl.gz", "chime/queries.jsonl",
                "chime/papers.jsonl.gz", "chime/gold/CD900001.json"):
        assert (out / rel).exists(), rel
