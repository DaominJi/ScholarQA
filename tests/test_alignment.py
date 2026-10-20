"""Tests for corpus alignment (App. C.2 "Coverage"): ids, titles, fuzzy titles, coverage, quote attribution."""

from __future__ import annotations

import json

import pytest

from scholarpath.data.alignment import (
    CoverageStats,
    PaperAligner,
    QuoteAttributor,
    Reference,
    attach_s2_ids,
    fuzzy_title_score,
    parse_arxiv_id,
    parse_doi,
    parse_s2_corpus_id,
    parse_s2_id,
    resolve_references,
    split_arxiv_version,
    title_key,
)
from scholarpath.data.corpus import Corpus, write_jsonl
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.types import Chunk, Paper, Query

SHA = "90b3eb94139217e45986958d1a1083a1ad8dbb5c"


@pytest.mark.parametrize("text,expected", [
    ("2101.00001", "2101.00001"),
    ("2101.00001v3", "2101.00001"),
    ("arXiv:2101.00001v2", "2101.00001"),
    ("https://arxiv.org/abs/2101.00001v1", "2101.00001"),
    ("https://arxiv.org/pdf/2101.00001v1.pdf", "2101.00001"),
    ("10.48550/arXiv.2301.10140", "2301.10140"),
    ("0704.0001", "0704.0001"),
    ("hep-th/9901001v2", "hep-th/9901001"),
    ("math.GT/0309136", "math/0309136"),
    ("https://arxiv.org/abs/math/0601001", "math/0601001"),
    ("and/0301001", None),  # not an arXiv archive
    ("10.1103/PhysRevD.76.013009", None),
    ("2113.00001", None),  # month 13
    (None, None),
])
def test_parse_arxiv_id(text, expected):
    assert parse_arxiv_id(text) == expected


def test_parse_arxiv_id_strict_requires_context():
    assert parse_arxiv_id("https://journal.org/view/2101.00001", strict=True) is None
    assert parse_arxiv_id("https://arxiv.org/abs/2101.00001", strict=True) == "2101.00001"
    assert parse_arxiv_id("2101.00001", strict=True) == "2101.00001"


def test_other_identifier_parsers():
    assert split_arxiv_version("2101.00001v12") == ("2101.00001", 12)
    assert split_arxiv_version("hep-th/9901001v2") == ("hep-th/9901001", 2)
    assert split_arxiv_version("2101.00001") == ("2101.00001", None)
    assert parse_doi("https://doi.org/10.1103/PhysRevD.76.013009.") == "10.1103/physrevd.76.013009"
    assert parse_doi("doi:10.1145/1234.5678)") == "10.1145/1234.5678"
    assert parse_doi("no doi here") is None
    assert parse_s2_id(f"https://www.semanticscholar.org/paper/{SHA}") == SHA
    assert parse_s2_id(f"https://www.semanticscholar.org/paper/Some-Title/{SHA}") == SHA
    assert parse_s2_id(SHA.upper()) == SHA
    assert parse_s2_corpus_id("CorpusId:12345") == "12345"
    assert parse_s2_corpus_id(987) == "987"
    assert parse_s2_corpus_id("https://www.semanticscholar.org/p/4242") == "4242"


def test_reference_parses_urls_and_keys():
    r = Reference(title="Some Title", url="https://arxiv.org/abs/2101.00001v2")
    assert r.arxiv_id == "2101.00001" and r.key == "arxiv:2101.00001"
    r = Reference(title="T", url=f"https://www.semanticscholar.org/paper/{SHA}")
    assert r.s2_id == SHA and r.key == f"s2:{SHA}"
    r = Reference(title="T", url="https://doi.org/10.1145/5555.6666")
    assert r.doi == "10.1145/5555.6666" and r.key == "doi:10.1145/5555.6666"
    r = Reference(title="210983829")  # ScholarQA-Multi stores some CorpusIds as titles
    assert r.title is None and r.s2_corpus_id == "210983829"
    r = Reference(title="Fast $k$-means Clustering!", url="http://journal.example.org/article/view/40757", year="2019")
    assert r.arxiv_id is None and r.year == 2019 and r.key == "title:fast k means clustering"
    assert Reference.from_dict(r.to_dict()).key == r.key


def test_title_keys_and_fuzzy_score():
    assert title_key("Fast $k$-means with $\\alpha$-divergences") == "fast k means with alpha divergences"
    assert fuzzy_title_score("graph neural networks", "graph neural networks") == 1.0
    # containment with enough tokens -> 1.0; short titles never match by containment
    assert fuzzy_title_score("convex relaxations of scheduling problems",
                             "convex relaxations of scheduling problems an lp view") == 1.0
    assert fuzzy_title_score("attention", "attention is all you need") < 0.92
    assert 0.92 <= fuzzy_title_score("learned cardinality estimation for join queries",
                                     "learned cardinality estimaton for join queries") < 1.0


@pytest.fixture()
def corpus():
    papers = [
        Paper("2101.00001", "Learned Cardinality Estimation for Join Queries", year=2021, arxiv_id="2101.00001",
              doi="10.1145/1234567.1234568"),
        Paper("2101.00002", "Index Tuning with Reinforcement Learning", year=2021, arxiv_id="2101.00002",
              s2_id=SHA, metadata={"s2_corpus_id": "53221591"}),
        Paper("math/0601001", "Convex Relaxations of Scheduling Problems", year=2006, arxiv_id="math/0601001"),
        Paper("1801.00003", "Fast $k$-means Clustering", year=2018, arxiv_id="1801.00003"),
        Paper("1901.00004", "A Survey", year=2019, arxiv_id="1901.00004"),
        Paper("2001.00005", "A Survey", year=2020, arxiv_id="2001.00005"),
    ]
    return Corpus(papers, name="toy")


def test_aligner_cascade(corpus):
    al = PaperAligner(corpus)
    cases = [
        (Reference(url="https://arxiv.org/pdf/2101.00001v2"), "2101.00001", "arxiv"),
        (Reference(arxiv_id="math/0601001v3"), "math/0601001", "arxiv"),
        (Reference(url=f"https://www.semanticscholar.org/paper/{SHA}"), "2101.00002", "s2"),
        (Reference(s2_corpus_id=53221591), "2101.00002", "s2_corpus"),
        (Reference(doi="10.1145/1234567.1234568"), "2101.00001", "doi"),
        (Reference(title="index tuning with reinforcement-learning"), "2101.00002", "title"),
        (Reference(title="Fast k-means clustering"), "1801.00003", "title"),  # LaTeX math unwrapped
        (Reference(title="Learned cardinality estimaton for join queries"), "2101.00001", "fuzzy_title"),
        (Reference(title="Convex Relaxations of Scheduling Problems (an LP view)"), "math/0601001", "fuzzy_title"),
        (Reference(title="Access Path Selection in a Relational Database Management System"), None, None),
    ]
    for ref, pid, method in cases:
        m = al.align(ref)
        assert (m.paper_id, m.method) == (pid, method), ref


def test_aligner_ambiguity_year_and_options(corpus):
    al = PaperAligner(corpus)
    m = al.align(Reference(title="A survey", year=2020))
    assert m.paper_id == "2001.00005" and m.ambiguous
    m = al.align(Reference(title="A survey"))
    assert m.paper_id == "1901.00004" and m.ambiguous  # deterministic: smallest id
    strict = PaperAligner(corpus, year_tolerance=1)
    assert strict.align(Reference(title="Index Tuning with Reinforcement Learning", year=2010)).paper_id is None
    nofuzzy = PaperAligner(corpus, fuzzy=False)
    assert nofuzzy.align(Reference(title="Learned cardinality estimaton for join queries")).paper_id is None
    assert len(al) == len(corpus)


def test_resolve_dedupes_and_counts_coverage(corpus):
    al = PaperAligner(corpus)
    refs = [
        Reference(url="https://arxiv.org/abs/2101.00001v2", title="Learned Cardinality Estimation for Join Queries"),
        Reference(title="Learned cardinality estimation for join queries."),  # same paper, other key
        Reference(title="Unknown Paper About Something"),
        Reference(title="Unknown paper about something"),  # same key -> counted once
        {"title": "Index Tuning with Reinforcement Learning"},
    ]
    res = al.resolve(refs)
    assert res.gold_papers == ["2101.00001", "2101.00002"]
    assert res.unaligned_gold == ["title:unknown paper about something"]
    assert (res.stats.total, res.stats.aligned) == (3, 2)
    q = Query("q", "text", gold_papers=res.gold_papers, unaligned_gold=res.unaligned_gold)
    assert q.num_gold == res.stats.total  # unaligned citations stay in the denominator
    no_aligner = resolve_references(refs)
    assert no_aligner.gold_papers == [] and no_aligner.stats.coverage == 0.0


def test_coverage_stats_merge_and_format():
    a = CoverageStats(total=10, aligned=9, by_method={"arxiv": 9})
    b = CoverageStats(total=5, aligned=3, by_method={"title": 2, "arxiv": 1})
    a.merge(b)
    assert (a.total, a.aligned, a.unaligned) == (15, 12, 3)
    assert a.by_method == {"arxiv": 10, "title": 2}
    assert abs(a.coverage - 0.8) < 1e-12 and "80.0%" in str(a)
    assert CoverageStats().coverage == 1.0
    assert a.to_dict()["coverage"] == 0.8


def test_aligner_on_synthetic_corpus_scales_with_blocking():
    bench = make_synthetic_benchmark(n_papers=200, seed=3)
    al = PaperAligner(bench.corpus)
    papers = list(bench.corpus)[:40]
    refs = [Reference(title=p.title.lower().replace(" and ", " & ")) for p in papers]
    matches, stats = al.align_many(refs)
    # Templated synthetic titles collide, so check that every match is a paper with an equivalent title.
    for m, p in zip(matches, papers):
        assert m.paper_id is not None
        assert fuzzy_title_score(title_key(bench.corpus[m.paper_id].title), title_key(p.title)) >= 0.92
    assert stats.coverage == 1.0


def test_attach_s2_ids(tmp_path, corpus):
    shard = tmp_path / "papers-0000.jsonl.gz"
    write_jsonl(shard, [
        {"corpusid": 111, "externalids": {"ArXiv": "2101.00001", "DOI": None, "CorpusId": "111"},
         "url": "https://www.semanticscholar.org/paper/" + "a" * 40, "title": "x"},
        {"corpusid": 222, "externalids": {"ArXiv": None, "DOI": "10.1/zzz"}, "url": "", "title": "y"},
        {"corpusid": 333, "externalids": {"ArXiv": "math/0601001", "DOI": "10.5555/ABC"},
         "url": "https://www.semanticscholar.org/paper/" + "b" * 40, "title": "z"},
    ])
    n = attach_s2_ids(corpus, tmp_path)
    assert n == 2
    assert corpus["2101.00001"].s2_id == "a" * 40 and corpus["2101.00001"].metadata["s2_corpus_id"] == "111"
    assert corpus["math/0601001"].doi == "10.5555/abc"
    al = PaperAligner(corpus)
    assert al.align(Reference(s2_corpus_id="333")).paper_id == "math/0601001"


def test_quote_attributor():
    corpus = Corpus([
        Paper("p1", "Join estimation", "We study learned cardinality estimation for multi-way join queries in "
                                       "relational databases."),
        Paper("p2", "Other", "Completely different text about protein folding and molecular dynamics.",
              chunks=[Chunk("p2#0", "p2", "Method", "Histograms assume attribute value independence which rarely "
                                                       "holds for real world correlated datasets at scale.", 0)]),
    ])
    quotes = [
        "We study learned cardinality estimation for multi-way join queries in relational databases.",
        "histograms assume attribute value independence, which rarely holds for real-world correlated datasets",
        "Too short to attribute.",
        "A long quote that appears nowhere in the corpus at all, not even partially, really.",
    ]
    qa = QuoteAttributor(quotes)
    assert qa.scan_corpus(corpus) >= 2
    assert qa.attribution(quotes[0])[0] == "p1"
    pid, frac = qa.attribution(quotes[1])
    assert pid == "p2" and frac == 1.0
    assert qa.attribution(quotes[2]) == (None, 0.0)
    assert qa.attribution(quotes[3])[0] is None


def test_quote_attributor_scans_chunk_files(tmp_path):
    path = tmp_path / "chunks.jsonl.gz"
    write_jsonl(path, [{"chunk_id": "x#0", "paper_id": "x", "heading": "H",
                        "text": "the solver tunes the sparsity level by cross validation on held out pulses"}])
    qa = QuoteAttributor(["We tune the sparsity level by cross validation on held-out pulses."])
    qa.scan_corpus(Corpus([]), chunks_path=path)
    pid, frac = qa.attribution("We tune the sparsity level by cross validation on held-out pulses.")
    assert pid == "x" and 0.6 <= frac <= 1.0
    assert json.loads(json.dumps(qa.quotes))  # serialisable
