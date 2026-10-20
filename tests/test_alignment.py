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
    surname,
    title_key,
    titles_compatible,
    version_tokens,
    word_spans,
    words,
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


@pytest.mark.parametrize("text,expected", [
    ("https://link.springer.com/content/pdf/10.1007/s10994-020-05000-1.pdf", "10.1007/s10994-020-05000-1"),
    ("https://www.biorxiv.org/content/10.1101/2020.01.01.123456v1.full", "10.1101/2020.01.01.123456"),
    ("https://www.medrxiv.org/content/10.1101/2021.02.03.21251000v2.full.pdf", "10.1101/2021.02.03.21251000"),
    ("https://onlinelibrary.wiley.com/doi/full/10.1002/anie.201912345?af=R", "10.1002/anie.201912345"),
    ("https://www.tandfonline.com/doi/full/10.1080/12345678.2019.1234567#abstract", "10.1080/12345678.2019.1234567"),
    ("https://www.frontiersin.org/articles/10.3389/fnins.2020.00001/full", "10.3389/fnins.2020.00001"),
    ("https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0123456&type=printable",
     "10.1371/journal.pone.0123456"),
    ("https://doi.org/10.1000%2Fabc", "10.1000/abc"),
    ("10.1145/3318464.3389700", "10.1145/3318464.3389700"),
])
def test_parse_doi_strips_url_parts(text, expected):
    assert parse_doi(text) == expected
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
    # no containment shortcut: a title contained in a longer one is just a ratio
    assert fuzzy_title_score("attention is all you need", "attention is all you need in speech separation") < 0.92
    assert 0.92 <= fuzzy_title_score("learned cardinality estimation for join queries",
                                     "learned cardinality estimaton for join queries") < 1.0
    assert version_tokens("qwen2 technical report") == {"qwen2"}
    assert version_tokens("textbooks are all you need ii phi 1 5 technical report") == {"ii", "1", "5"}
    assert version_tokens("x ray imaging of a v shaped part i") == set()
    assert titles_compatible("learned cardinality estimaton for join queries",
                             "learned cardinality estimation for join queries")
    assert titles_compatible("multi task learning", "multitask learning")
    assert titles_compatible("convex optimisation methods", "convex optimization methods")
    assert not titles_compatible("non autoregressive neural machine translation",
                                 "semi autoregressive neural machine translation")
    assert not titles_compatible("qwen2 technical report", "qwen technical report")
    assert [surname(n) for n in ("Smith, John", "John Smith", "J. Smith", "Kroiß", None)] == \
        ["smith", "smith", "smith", "kroi", None]


def test_words_split_glued_markdown_and_track_spans():
    text = "### Emphasizing Foundational ConceptsOne of the key ideas: ManyTypes4Py, café"
    assert words(text) == ["emphasizing", "foundational", "concepts", "one", "of", "the", "key", "ideas",
                           "many", "types4py", "cafe"]
    tokens, spans = word_spans(text)
    assert tokens == words(text)
    assert [text[a:b] for a, b in spans][2:4] == ["Concepts", "One"] and text[slice(*spans[-1])] == "café"


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
    al = PaperAligner(corpus)  # the paper's rule: identifiers and exact normalized title only
    cases = [
        (Reference(url="https://arxiv.org/pdf/2101.00001v2"), "2101.00001", "arxiv"),
        (Reference(arxiv_id="math/0601001v3"), "math/0601001", "arxiv"),
        (Reference(url=f"https://www.semanticscholar.org/paper/{SHA}"), "2101.00002", "s2"),
        (Reference(s2_corpus_id=53221591), "2101.00002", "s2_corpus"),
        (Reference(doi="10.1145/1234567.1234568"), "2101.00001", "doi"),
        (Reference(url="https://dl.acm.org/doi/pdf/10.1145/1234567.1234568?download=true"), "2101.00001", "doi"),
        (Reference(title="index tuning with reinforcement-learning"), "2101.00002", "title"),
        (Reference(title="Fast k-means clustering"), "1801.00003", "title"),  # LaTeX math unwrapped
        (Reference(title="Learned cardinality estimaton for join queries"), None, None),  # typo: no fuzzy by default
        (Reference(title="Convex Relaxations of Scheduling Problems (an LP view)"), None, None),
        (Reference(title="Access Path Selection in a Relational Database Management System"), None, None),
    ]
    for ref, pid, method in cases:
        m = al.align(ref)
        assert (m.paper_id, m.method) == (pid, method), ref
    assert al.describe()["methods"] == ["arxiv", "s2", "s2_corpus", "doi", "pmid", "title"]


def test_opt_in_fuzzy_stage(corpus):
    al = PaperAligner(corpus, fuzzy=True)
    assert "fuzzy_title" in al.describe()["methods"]
    m = al.align(Reference(title="Learned cardinality estimaton for join queries"))
    assert (m.paper_id, m.method) == ("2101.00001", "fuzzy_title") and 0.92 <= m.score < 1.0
    # containment never aligns: the longer title is a different (absent) paper
    assert al.align(Reference(title="Convex Relaxations of Scheduling Problems (an LP view)")).paper_id is None
    # year and first-author disagreement reject a fuzzy match (exact titles are unaffected)
    assert al.align(Reference(title="Learned cardinality estimaton for join queries", year=2010)).paper_id is None
    assert al.align(Reference(title="Learned cardinality estimaton for join queries", year=2022)).paper_id == "2101.00001"
    authored = PaperAligner([Paper("p1", "Learned Cardinality Estimation for Join Queries", authors=["Alice Smith"])],
                            fuzzy=True)
    typo = "Learned cardinality estimaton for join queries"
    assert authored.align(Reference(title=typo, first_author="Smith, A.")).paper_id == "p1"
    assert authored.align(Reference(title=typo, first_author="Bob Jones")).paper_id is None


def test_fuzzy_never_aligns_absent_papers_to_versions_or_extensions():
    """Regression: citations of papers missing from the corpus must stay unaligned (App. C.2 denominator)."""
    present = ["Qwen Technical Report", "Textbooks Are All You Need II: phi-1.5 technical report",
               "Baichuan-Omni-1.5 Technical Report", "Attention Is All You Need in Speech Separation",
               "Neural networks for time series", "CMMLU: Measuring massive multitask language understanding in Chinese",
               "Llama 2: Open Foundation and Fine-Tuned Chat Models", "Semi-Autoregressive Neural Machine Translation"]
    corpus = Corpus([Paper(f"p{i}", t) for i, t in enumerate(present)])
    absent = ["Qwen2 Technical Report", "Textbooks Are All You Need", "Baichuan-Omni Technical Report",
              "Attention is all you need", "Neural networks for time series forecasting",
              "Measuring Massive Multitask Language Understanding", "Llama 3: Open Foundation and Fine-Tuned Chat Models",
              "Non-Autoregressive Neural Machine Translation"]
    for fuzzy in (False, True):
        al = PaperAligner(corpus, fuzzy=fuzzy)
        matches, stats = al.align_many(Reference(title=t) for t in absent)
        assert [m.paper_id for m in matches] == [None] * len(absent), fuzzy
        assert stats.coverage == 0.0
        assert al.align(Reference(title="Qwen technical report.")).paper_id == "p0"  # exact normalized title


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
    al = PaperAligner(bench.corpus, fuzzy=True)
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
