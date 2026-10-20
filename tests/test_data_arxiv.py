"""Tests for the arXiv corpora (Sec. 5.2, App. C.2) and full-text section chunking (Sec. 2.1)."""

from __future__ import annotations

import gzip
import io
import json
import random
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from scholarpath.config import IndexConfig
from scholarpath.data.arxiv import (
    ARXIV_SET_CATEGORIES,
    UniformSampler,
    build_arxiv_sets,
    build_subfields_corpus,
    first_version_date,
    is_withdrawn,
    iter_arxiv_papers,
    iter_snapshot_records,
    load_arxiv_sets,
    record_to_paper,
    top_level_group,
    write_arxiv_corpora,
)
from scholarpath.data.corpus import Corpus, read_jsonl
from scholarpath.data.fulltext import (
    FulltextExtractor,
    FulltextSource,
    LocalSource,
    S2ORCSource,
    clean_sections,
    gcs_pdf_url,
    latex_to_sections,
    make_sources,
    read_latex_source,
    s2orc_sections,
    safe_paper_id,
    sections_to_chunks,
    text_to_sections,
)
from scholarpath.types import Paper

FIX = Path(__file__).parent / "fixtures"
SNAPSHOT = FIX / "arxiv" / "arxiv-metadata-sample.json"
FULLTEXT = FIX / "fulltext"
REPO = Path(__file__).resolve().parents[1]


def records():
    return {r["id"]: r for r in iter_snapshot_records(SNAPSHOT)}


# ---------------------------------------------------------------------------
# Snapshot records -> Paper
# ---------------------------------------------------------------------------


def test_eight_categories_span_seven_groups():
    assert len(ARXIV_SET_CATEGORIES) == 8
    assert len({top_level_group(c) for c in ARXIV_SET_CATEGORIES}) == 7
    assert top_level_group("astro-ph.GA") == top_level_group("physics.med-ph") == "Physics"


def test_record_to_paper_fields():
    recs = records()
    assert len(recs) == 12  # blank and malformed lines are skipped
    p = record_to_paper(recs["2101.00001"])
    assert p.paper_id == p.arxiv_id == "2101.00001"
    assert p.title == "Learned Cardinality Estimation for Join Queries"
    assert p.abstract == "We study learned cardinality estimation for multi-way join queries in relational databases."
    assert p.categories == ["cs.DB", "cs.LG"] and p.primary_category == "cs.DB"
    assert p.authors == ["Alice Smith", "Bob Jones Jr."]
    assert p.date == "2021-01-01" and p.year == 2021  # first version, not the latest
    assert p.doi == "10.1145/1234567.1234568"  # first DOI listed
    assert p.metadata["num_versions"] == 2 and p.metadata["latest_version"] == "v2"
    old = record_to_paper(recs["math/0601001"])
    assert old.paper_id == "math/0601001" and old.date == "2006-01-01" and old.primary_category == "math.OC"


def test_first_version_date_fallbacks():
    assert first_version_date({"id": "2103.00001", "versions": []}) == "2021-03-01"
    assert first_version_date({"id": "hep-th/9901001"}) == "1999-01-01"
    assert first_version_date({"id": "x", "update_date": "2020-05-05"}) == "2020-05-05"
    v = [{"version": "v2", "created": "Mon, 1 Mar 2021 10:00:00 GMT"}, {"version": "v1", "created": "Fri, 1 Jan 2021 10:00:00 GMT"}]
    assert first_version_date({"id": "2101.1", "versions": v}) == "2021-01-01"


def test_withdrawn_rule():
    recs = records()
    assert is_withdrawn(recs["2103.00005"])  # comments: "This paper has been withdrawn by the author ..."
    assert is_withdrawn(recs["2104.00006"])  # abstract: "This paper has been withdrawn."
    assert not is_withdrawn(recs["2107.00009"])  # mentions drugs "withdrawn from the market"
    assert not is_withdrawn(recs["2101.00001"])
    for comment in ("Withdrawn", "withdrawn due to an error in Lemma 2", "Paper withdrawn by arXiv administrators",
                    "The authors have withdrawn this paper"):
        assert is_withdrawn({"comments": comment}), comment
    assert not is_withdrawn({"comments": "12 pages; extended discussion of withdrawn drugs"})


def test_iter_papers_primary_any_until():
    prim = {p.paper_id: c for c, p in iter_arxiv_papers(SNAPSHOT)}
    assert set(prim) == {"2101.00001", "2101.00002", "2102.00003", "2103.00004", "math/0601001", "2106.00008",
                         "2107.00009", "2609.00010", "1501.00011"}
    assert prim["2107.00009"] == "stat.AP"  # primary = first listed category, not cs.DB
    anyc = {p.paper_id: c for c, p in iter_arxiv_papers(SNAPSHOT, match="any")}
    assert anyc["2105.00007"] == "cs.DB"  # cs.LG primary, cs.DB cross-list
    early = {p.paper_id for _, p in iter_arxiv_papers(SNAPSHOT, until="2026-01-01")}
    assert "2609.00010" not in early and "2101.00001" in early
    withdrawn = {p.paper_id for _, p in iter_arxiv_papers(SNAPSHOT, include_withdrawn=True)}
    assert {"2103.00005", "2104.00006"} <= withdrawn
    every = list(iter_arxiv_papers(SNAPSHOT, categories=None))
    assert len(every) == 10


def test_gzip_and_zip_snapshots(tmp_path):
    gz = tmp_path / "arxiv-metadata-oai-snapshot.json.gz"
    with gzip.open(gz, "wb") as f:
        f.write(SNAPSHOT.read_bytes())
    zp = tmp_path / "arxiv.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.write(SNAPSHOT, "arxiv-metadata-oai-snapshot.json")
    plain = [p.paper_id for _, p in iter_arxiv_papers(SNAPSHOT)]
    assert [p.paper_id for _, p in iter_arxiv_papers(gz)] == plain
    assert [p.paper_id for _, p in iter_arxiv_papers(zp)] == plain
    assert [p.paper_id for _, p in iter_arxiv_papers(tmp_path)] == plain  # directory: snapshot found inside
    with pytest.raises(FileNotFoundError):
        list(iter_snapshot_records(tmp_path / "missing.json"))


def test_uniform_sampler_is_order_independent_and_uniform():
    papers = [Paper(f"id{i:02d}", "t") for i in range(20)]
    a, b = UniformSampler(5, seed=7), UniformSampler(5, seed=7)
    for p in papers:
        a.offer(p)
    for p in random.Random(1).sample(papers, len(papers)):
        b.offer(p)
    assert [p.paper_id for p in a.result()] == [p.paper_id for p in b.result()]
    counts = {p.paper_id: 0 for p in papers}
    for seed in range(400):
        s = UniformSampler(5, seed=seed)
        for p in papers:
            s.offer(p)
        for p in s.result():
            counts[p.paper_id] += 1
    assert all(55 <= c <= 145 for c in counts.values()), counts  # expectation 100


def test_build_sets_and_subfields(tmp_path):
    sets = build_arxiv_sets(SNAPSHOT, n_per_set=2, seed=0)
    assert set(sets) == set(ARXIV_SET_CATEGORIES)
    assert len(sets["cs.DB"]) == 2 and len(sets["math.OC"]) == 2 and len(sets["econ.EM"]) == 1
    assert all(p.primary_category == "cs.DB" for p in sets["cs.DB"])
    assert [p.paper_id for p in build_arxiv_sets(SNAPSHOT, n_per_set=2, seed=0)["cs.DB"]] == \
        [p.paper_id for p in sets["cs.DB"]]
    # shuffling the snapshot does not change the sample
    lines = [l for l in SNAPSHOT.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(3).shuffle(lines)
    shuffled = tmp_path / "shuffled.json"
    shuffled.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert [p.paper_id for p in build_arxiv_sets(shuffled, n_per_set=2, seed=0)["cs.DB"]] == \
        [p.paper_id for p in sets["cs.DB"]]
    sub = build_subfields_corpus(SNAPSHOT)
    assert len(sub) == 9 and "2103.00005" not in sub and "2105.00007" not in sub


def test_write_arxiv_corpora(tmp_path):
    report = write_arxiv_corpora(SNAPSHOT, tmp_path, n_per_set=3, seed=1, until="2025-12-31")
    sub = Corpus.load(tmp_path / "arxiv_subfields" / "papers.jsonl.gz")
    assert len(sub) == report.subfields_size == 8
    sets = load_arxiv_sets(tmp_path / "arxiv_sets")
    assert len(sets["cs.DB"]) == 3 and "econ.EM" in sets and len(sets["econ.EM"]) == 0
    stats = json.loads((tmp_path / "arxiv_subfields" / "stats.json").read_text())
    assert stats["scan"]["withdrawn_in_categories"] == 2 and stats["scan"]["after_cutoff"] == 1
    assert stats["subfields_size"] == 8 and len(stats["top_level_groups"]) == 7


# ---------------------------------------------------------------------------
# Full text and chunking
# ---------------------------------------------------------------------------


def test_latex_sections_with_inputs_comments_and_floats():
    main, files = read_latex_source(FULLTEXT / "latex")
    secs = dict(latex_to_sections(main, files))
    assert list(secs) == ["Abstract", "Introduction", "Method", "Method > Model", "Method > Training"]
    assert "Not a section" not in " ".join(secs)  # commented-out heading
    assert "Achilles heel" in secs["Introduction"] and "[cite]" in secs["Introduction"]
    assert "(Categorical values are hashed.)" in secs["Method"]  # footnote inlined
    assert "[Figure] Architecture of the estimator." in secs["Method > Model"]
    assert "$\\mathcal{L}" in secs["Method > Model"]  # math kept verbatim
    assert "- Joins: are encoded" in secs["Method > Model"]
    assert all("reviewers" not in b for b in secs.values())  # acknowledgements dropped


def test_latex_appendix_after_bibliography_is_kept():
    tex = (r"\begin{document}\section{Intro}Text body here.\bibliographystyle{plain}\bibliography{refs}"
           r"\appendix\section{Proofs}Proof of the theorem."
           r"\begin{thebibliography}{9}\bibitem{a} A. Author. Title.\end{thebibliography}\end{document}")
    assert latex_to_sections(tex) == [("Intro", "Text body here."), ("Proofs", "Proof of the theorem.")]


def test_local_source_prefers_latex_and_latest_version(tmp_path):
    body = "1 Introduction\nThe {v} version of the text with enough words to form a section body.\n"
    (tmp_path / "2102.00003v1.txt").write_text(body.format(v="first"))
    (tmp_path / "2102.00003v2.txt").write_text(body.format(v="second"))
    (tmp_path / "2102.00004.txt").write_text(body.format(v="plain"))
    (tmp_path / "2102.00004.tex").write_text(r"\section{Intro}LaTeX source wins over text.")
    (tmp_path / "notes.json").write_text("{}")  # ignored
    src = LocalSource(tmp_path)
    doc = src.fetch(Paper("2102.00003", "t"))
    assert doc.kind == "text" and "second" in doc.sections[0][1]
    doc = src.fetch(Paper("2102.00004", "t"))
    assert doc.kind == "latex" and doc.sections == [("Intro", "LaTeX source wins over text.")]
    assert src.fetch(Paper("2102.00005", "t")) is None


def test_latex_source_archives(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in (FULLTEXT / "latex").rglob("*.tex"):
            tar.add(path, arcname=str(path.relative_to(FULLTEXT / "latex")))
    main, files = read_latex_source(buf.getvalue())
    assert "sections/method.tex" in files and "\\documentclass" in main
    single = gzip.compress(b"\\documentclass{article}\\begin{document}\\section{Intro}Hello world text.\\end{document}")
    main2, _ = read_latex_source(single)
    assert latex_to_sections(main2) == [("Intro", "Hello world text.")]
    assert read_latex_source(b"%PDF-1.5 binary") is None


def test_text_and_s2orc_sections():
    secs = text_to_sections((FULLTEXT / "local" / "math_0601001.txt").read_text())
    assert [h for h, _ in secs] == ["Preamble", "Abstract", "Introduction", "Semidefinite Relaxations", "Experiments"]
    assert [h for h, _ in clean_sections(secs)] == ["Introduction", "Semidefinite Relaxations", "Experiments"]
    recs = list(read_jsonl(FULLTEXT / "s2orc_v2-sample.jsonl"))
    s2 = s2orc_sections(recs[1])  # string offsets, JSON-encoded annotations, references dropped
    assert [h for h, _ in s2] == ["1. Introduction", "2. Methods"]
    assert s2[1][1].startswith("We solve a basis pursuit problem")
    assert s2orc_sections(recs[0]) == [("Body", "Intro text.")]
    legacy = {"content": {"text": "Intro\nBody text here.", "annotations": {"sectionheader": json.dumps([{"start": 0, "end": 5}])}}}
    assert s2orc_sections(legacy) == [("Intro", "Body text here.")]


def test_sections_to_chunks_ids_and_sizes():
    main, files = read_latex_source(FULLTEXT / "latex")
    chunks = sections_to_chunks("2101.00002", latex_to_sections(main, files), max_tokens=40, min_tokens=5)
    assert [c.chunk_id for c in chunks] == [f"2101.00002#{i}" for i in range(len(chunks))]
    assert [c.order for c in chunks] == list(range(len(chunks)))
    assert all(c.paper_id == "2101.00002" for c in chunks)
    assert not any(c.heading == "Abstract" for c in chunks)  # front matter dropped (abstract is the paper level)
    cfg = IndexConfig()
    default = sections_to_chunks("x", latex_to_sections(main, files), cfg.chunk_max_tokens, cfg.chunk_min_tokens)
    assert 1 <= len(default) <= len(chunks)


def test_urls_and_ids():
    assert gcs_pdf_url("2601.12369", 1) == \
        "https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf/2601/2601.12369v1.pdf"
    assert gcs_pdf_url("math/0601001", 2) == \
        "https://storage.googleapis.com/arxiv-dataset/arxiv/math/pdf/0601/0601001v2.pdf"
    assert gcs_pdf_url("hep-th/9901001v3").endswith("/arxiv/hep-th/pdf/9901/9901001v3.pdf")
    assert safe_paper_id("math/0601001") == "math_0601001"


class _FailingSource(FulltextSource):
    name = "flaky"

    def __init__(self):
        self.calls = 0

    def fetch(self, paper):
        self.calls += 1
        raise OSError("network down")


def _papers():
    return [Paper("2101.00002", "Index Tuning", arxiv_id="2101.00002"),
            Paper("math/0601001", "Convex Relaxations", arxiv_id="math/0601001"),
            Paper("1501.00011", "Radar Signal Denoising", arxiv_id="1501.00011"),
            Paper("2102.00003", "No full text anywhere", arxiv_id="2102.00003")]


def test_extractor_sources_stats_and_resume(tmp_path):
    flaky = _FailingSource()
    sources = [LocalSource(FULLTEXT / "local"), S2ORCSource(FULLTEXT / "s2orc_v2-sample.jsonl"), flaky]
    ex = FulltextExtractor(sources, max_tokens=60, min_tokens=5)
    out = tmp_path / "chunks.jsonl.gz"
    stats = ex.run(_papers(), out)
    assert stats.with_fulltext == 3 and stats.abstract_only == 1
    assert stats.succeeded == {"local": 2, "s2orc": 1}
    assert stats.attempted == {"local": 4, "s2orc": 2, "flaky": 1} and flaky.calls == 1
    assert stats.kinds == {"latex": 1, "text": 1, "s2orc": 1}
    chunks = list(read_jsonl(out))
    assert {c["paper_id"] for c in chunks} == {"2101.00002", "math/0601001", "1501.00011"}
    assert all(c["chunk_id"].startswith(c["paper_id"] + "#") for c in chunks)
    # Resume: nothing is fetched again.
    again = FulltextExtractor([_FailingSource()]).run(_papers(), out)
    assert again.resumed == 4 and again.chunks == 0 and again.with_fulltext == 3
    corpus = Corpus(_papers())
    assert corpus.attach_chunks(out) == len(chunks)  # the foundation can read the appended gzip
    # Retrying papers without full text only touches those.
    retry = FulltextExtractor([flaky]).run(_papers(), out, retry_missing=True)
    assert retry.attempted == {"flaky": 1}


def test_extractor_repairs_interrupted_runs(tmp_path):
    out = tmp_path / "chunks.jsonl.gz"
    ex = FulltextExtractor([LocalSource(FULLTEXT / "local")], max_tokens=60, min_tokens=5)
    ex.run(_papers()[:1], out)
    n_ok = len(list(read_jsonl(out)))
    # Simulate a crash: chunks of a paper were written but its state line was not, then a truncated tail.
    with gzip.open(out, "at", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": "math/0601001#0", "paper_id": "math/0601001", "heading": "H",
                            "text": "partial", "order": 0}) + "\n")
    with open(out, "ab") as f:
        f.write(gzip.compress(b'{"chunk_id": "x#0", "paper_id": "x"')[:-8])
    stats = ex.run(_papers(), out)
    chunks = list(read_jsonl(out))  # readable again
    assert stats.resumed == 1 and stats.succeeded == {"local": 1}
    assert sum(c["paper_id"] == "math/0601001" for c in chunks) == stats.chunks
    assert sum(c["paper_id"] == "2101.00002" for c in chunks) == n_ok
    assert not any(c["text"] == "partial" for c in chunks)


def test_make_sources_skips_unconfigured(tmp_path):
    names = [s.name for s in make_sources(("local", "s2orc", "gcs", "arxiv"), local_dir=tmp_path / "nope")]
    assert names == ["gcs", "arxiv"]
    with pytest.raises(ValueError):
        make_sources(("ftp",))


def test_pdf_extraction_when_pymupdf_available(tmp_path):
    fitz = pytest.importorskip("fitz")
    from scholarpath.data.fulltext import pdf_to_sections

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1 Introduction\nWe study joins in depth here.\n2 Method\nWe learn estimators.")
    data = doc.tobytes()
    secs = pdf_to_sections(data)
    assert [h for h, _ in secs][-2:] == ["Introduction", "Method"]


def test_prepare_arxiv_script(tmp_path, capsys):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import prepare_arxiv
    finally:
        sys.path.pop(0)
    assert prepare_arxiv.main(["--snapshot", str(SNAPSHOT), "--out", str(tmp_path), "--n-per-set", "2"]) == 0
    assert (tmp_path / "arxiv_sets" / "cs.DB.jsonl.gz").exists()
    assert "arXiv-Subfields: 9 papers" in capsys.readouterr().out


def test_prepare_fulltext_script(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import prepare_fulltext
    finally:
        sys.path.pop(0)
    corpus_path = tmp_path / "papers.jsonl.gz"
    Corpus(_papers()).save(corpus_path)
    rc = prepare_fulltext.main(["--corpus", str(corpus_path), "--sources", "local,s2orc",
                                "--local-dir", str(FULLTEXT / "local"), "--s2orc", str(FULLTEXT / "s2orc_v2-sample.jsonl"),
                                "--config", str(REPO / "configs" / "default.yaml")])
    assert rc == 0
    stats = json.loads((tmp_path / "chunks.stats.json").read_text())
    assert stats["with_fulltext"] == 3 and stats["abstract_only"] == 1
    shutil.rmtree(tmp_path)
