"""Tests for the budgeted retrieval / QA baselines of Section 5.3 (offline: MockLLM + hashing embedder)."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from scholarpath.baselines import (
    ABSTRACT,
    CHUNK,
    ENTITY,
    RELATION,
    SUMMARY,
    BM25Index,
    BM25Retriever,
    BudgetedQA,
    BudgetedRetriever,
    Candidate,
    DenseRerankRetriever,
    DenseRetriever,
    KeywordSearchRetriever,
    LightRAGConfig,
    LightRAGRetriever,
    NoRetrievalQA,
    Passage,
    PassagePool,
    RaptorConfig,
    RaptorRetriever,
    make_retriever,
    pack_to_budget,
)
from scholarpath.baselines.packing import build_result
from scholarpath.config import RetrievalConfig
from scholarpath.baselines.lightrag import _round_robin
from scholarpath.baselines.raptor import RaptorTree
from scholarpath.baselines.rerank import sliding_window_rerank, sliding_windows
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.index import SemanticIndex
from scholarpath.prompts.baselines import parse_permutation
from scholarpath.runtime import Runtime
from scholarpath.text import count_tokens, tokenize


@pytest.fixture(scope="module")
def bench():
    return make_synthetic_benchmark(n_papers=48, seed=3)


@pytest.fixture(scope="module")
def pool(bench):
    return PassagePool.from_corpus(bench.corpus)


@pytest.fixture()
def rt():
    return Runtime.from_config()


@pytest.fixture(scope="module")
def query(bench):
    return bench.evidence_queries[0]


# --------------------------------------------------------------------- pool


def test_passage_pool_units(bench, pool):
    n_chunks = sum(len(p.chunks) for p in bench.corpus)
    assert len(pool) == len(bench.corpus) + n_chunks
    assert pool.stats()["kinds"] == {ABSTRACT: len(bench.corpus), CHUNK: n_chunks}
    paper = next(iter(bench.corpus))
    ab = pool.get(f"paper:{paper.paper_id}")
    assert ab.kind == ABSTRACT and ab.papers == [paper.paper_id]
    assert ab.search_text == paper.text  # "title. abstract"
    assert ab.render().startswith(f"[{paper.paper_id}] {paper.title}")
    assert ab.cost == count_tokens(ab.render())
    ch = pool.get(f"chunk:{paper.chunks[1].chunk_id}")
    assert ch.kind == CHUNK and ch.paper_id == paper.paper_id and paper.chunks[1].heading in ch.render()
    assert pool.indices_for_papers([paper.paper_id]) == pool.by_paper[paper.paper_id]
    assert len(pool.by_paper[paper.paper_id]) == 1 + len(paper.chunks)


def test_pool_from_index_matches_corpus(bench, rt, pool):
    idx = SemanticIndex.build(bench.problem_gold, bench.method_gold, bench.corpus.papers, rt.embedder)
    ipool = PassagePool.from_index(idx)
    assert sorted(ipool.keys) == sorted(pool.keys)
    dense = DenseRetriever.from_index(ipool, idx, rt.embedder)
    i = ipool.index_of(f"paper:{next(iter(bench.corpus)).paper_id}")
    assert np.allclose(dense.matrix[i], idx.vector(ipool[i].key))


# ------------------------------------------------------------------ packing


def _p(key: str, text: str, pid: str = "p") -> Passage:
    return Passage(key, ABSTRACT, [pid], title=f"T{key}", body=text)


def test_pack_never_exceeds_budget_and_deduplicates():
    big = _p("big", "word " * 400, "p1")
    a = _p("a", "alpha beta gamma", "p2")
    a_dup_key = _p("a", "something else entirely", "p3")
    a_dup_text = Passage("a2", ABSTRACT, ["p4"], title="Ta", body="alpha  beta GAMMA")  # same normalized text
    b = _p("b", "delta epsilon", "p5")
    budget = 60
    units = pack_to_budget([(a, 3.0), big, a_dup_key, a_dup_text, Candidate(b, 1.0)], budget)
    keys = [u.key for u in units]
    assert keys == ["a", "b"]  # big skipped (too large), later small unit still packed; duplicates dropped
    assert sum(u.tokens for u in units) <= budget
    res = build_result("q", units, budget)
    assert count_tokens(res.context) == res.context_tokens <= budget
    assert res.ranked_papers == ["p2", "p5"] and [s["paper_id"] for s in res.sources] == ["p2", "p5"]


@pytest.mark.parametrize("budget", [0, 50, 300, 1000, 4096])
def test_pack_budget_sweep(pool, budget):
    cands = [(pool[i], -i) for i in range(len(pool))] + [(pool[0], 0.0)]
    units = pack_to_budget(cands, budget)
    res = build_result("q", units, budget)
    assert res.context_tokens <= budget and count_tokens(res.context) <= budget
    assert len({u.key for u in units}) == len(units)
    if budget >= 1000:
        assert len(units) >= 3


# --------------------------------------------------------------------- BM25


def test_bm25_matches_reference_formula():
    texts = ["graph neural network message passing", "gaussian process kernel", "graph kernel methods graph"]
    idx = BM25Index(k1=0.9, b=0.4).fit(texts)
    q = "graph kernel"
    docs = [tokenize(t, remove_stopwords=True) for t in texts]
    avgdl = sum(map(len, docs)) / len(docs)
    expected = []
    for d in docs:
        s = 0.0
        for t in tokenize(q, remove_stopwords=True):
            df = sum(1 for x in docs if t in x)
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            tf = d.count(t)
            s += idf * tf * 1.9 / (tf + 0.9 * (1 - 0.4 + 0.4 * len(d) / avgdl))
        expected.append(s)
    assert np.allclose(idx.scores(q), expected, atol=1e-5)
    assert list(idx.match_counts(["graph", "kernel", "absent"])) == [1, 1, 2]


def test_bm25_ranks_exact_match_first(pool, query):
    bm25 = BM25Retriever(pool)
    target = pool[17]
    hits = bm25.search(target.search_text, 5)
    assert hits[0][0] == 17
    res = bm25.retrieve(query, 2048)
    assert res.context_tokens <= 2048 and res.units
    assert res.ranked_papers[0] in query.gold_papers
    assert isinstance(bm25, BudgetedRetriever)


def test_bm25_save_load(tmp_path, pool):
    a = BM25Retriever(pool)
    a.index.save(tmp_path / "bm25")
    b = BM25Retriever(pool, index=BM25Index.load(tmp_path / "bm25"))
    q = pool[3].search_text
    assert a.search(q, 10) == b.search(q, 10)


# -------------------------------------------------------------------- dense


def test_dense_returns_nearest_neighbour(pool, rt, query):
    dense = DenseRetriever(pool, rt.embedder, block_rows=7)  # several blocks
    target = pool[42]
    hits = dense.search(target.search_text, 10)
    assert hits[0][0] == 42 and hits[0][1] == pytest.approx(1.0, abs=1e-5)
    qv = dense.encode_query(query.text)
    full = dense.matrix @ qv
    brute = np.sort(full)[::-1][:10]
    got = dense.search(query.text, 10)
    # Exact top-k up to float32 rounding (blocked products may break exact ties differently).
    assert np.allclose([s for _, s in got], brute, atol=1e-5)
    assert all(abs(full[i] - s) < 1e-5 for i, s in got)
    res = dense.retrieve(query, 1500)
    assert res.context_tokens <= 1500 and set(query.gold_papers) & set(res.ranked_papers)


def test_dense_disk_cache(tmp_path, pool, rt):
    a = DenseRetriever(pool, rt.embedder, cache_dir=tmp_path).build()
    b = DenseRetriever(pool, rt.embedder, cache_dir=tmp_path).build()
    assert np.allclose(np.asarray(a.matrix), np.asarray(b.matrix))
    assert list(tmp_path.glob("passages-*.npy"))


# ------------------------------------------------------------------ rerank


def test_sliding_windows_cover_top():
    assert sliding_windows(50, 20, 10) == [(30, 50), (20, 40), (10, 30), (0, 20)]
    assert sliding_windows(45, 20, 10)[-1] == (0, 15)
    assert sliding_windows(12, 20, 10) == [(0, 12)]
    assert parse_permutation("[3] > [1] > [3] > [9]", 4) == [2, 0, 1, 3]


def test_listwise_reranker_reorders_with_mock(rt):
    texts = ["nothing relevant here", "kernel only", "graph neural kernel message passing"]
    order = sliding_window_rerank(rt.llm, "graph neural message passing kernel", texts, window=2, step=1)
    assert order[0] == 2 and sorted(order) == [0, 1, 2]
    assert rt.meter.component("rerank").calls == 2


def test_dense_rerank_retriever(pool, rt, query):
    dense = DenseRetriever(pool, rt.embedder)
    rr = DenseRerankRetriever(dense, rt.llm, top_n=50, window=20, step=10)
    res = rr.retrieve(query, 2048)
    assert res.stats["rerank_calls"] == 4 and rt.meter.component("rerank").calls == 4
    assert res.context_tokens <= 2048
    assert [u.metadata["source"] for u in res.units][0] == "rerank"
    qa = BudgetedQA(rr, rt.llm, budget=2048)
    ans = qa.answer(query)
    assert ans.llm_calls == 5  # 4 reranking windows + 1 composition call


# ----------------------------------------------------------------- keyword


def test_keyword_search(pool, rt, query):
    kw = KeywordSearchRetriever(pool)
    assert all(p.kind == ABSTRACT for p in kw.pool)
    hits = kw.search(query.text)
    counts = [m for _, m, _ in hits]
    assert counts == sorted(counts, reverse=True) and counts[0] >= 2
    res = kw.retrieve(query, 2048)
    assert all(u.kind == ABSTRACT for u in res.units) and res.context_tokens <= 2048
    assert rt.meter.total().calls == 0  # no LLM call during retrieval
    ans = BudgetedQA(kw, rt.llm, budget=2048).answer(query)
    assert ans.llm_calls == 1


# ------------------------------------------------------------------ RAPTOR


def test_raptor_builds_multilayer_tree(pool, rt, query, tmp_path):
    cfg = RaptorConfig(clustering="kmeans", kmeans_cluster_size=6, reduction_dim=4)
    rap = RaptorRetriever(pool, rt.llm, rt.embedder, config=cfg).build()
    tree = rap.tree
    sizes = tree.layer_sizes()
    assert len(sizes) >= 3 and sizes[0] == len(pool) and sizes[1] < sizes[0] and sizes[2] < sizes[1]
    summaries = tree.summary_nodes
    assert rt.meter.component("index_build").calls == len(summaries) > 0
    top = max(summaries, key=lambda n: n.layer)
    assert len(top.leaves) > len(top.children)
    res = rap.retrieve(query, 2048)
    assert res.context_tokens <= 2048
    kinds = {u.kind for u in res.units}
    assert kinds <= {ABSTRACT, CHUNK, SUMMARY}
    leaf_papers = {n.node_id: set(n.paper_counts) for n in tree.nodes}
    qv = rt.embedder.encode_queries([query.text])[0]
    n_summaries = 0
    for u in res.units:
        assert u.papers and set(u.papers) <= set(pool.paper_ids)
        if u.kind == SUMMARY:  # cites <= 5 of its covered leaves' papers, most query-similar first
            n_summaries += 1
            node = tree.nodes[int(u.key.split(":")[2])]
            covered = set().union(*(leaf_papers[lf] for lf in node.cited_leaves))
            assert len(u.papers) == min(5, len(covered)) and set(u.papers) <= covered
            best = {p: max(float(tree.embeddings[lf] @ qv) for lf in node.cited_leaves if p in leaf_papers[lf])
                    for p in u.papers}
            assert [best[p] for p in u.papers] == sorted(best.values(), reverse=True)
    assert n_summaries > 0
    # A summary credits a bounded number of papers: Recall@B cannot exceed what the units cite.
    assert len(res.ranked_papers) <= 5 * len(res.units)
    capped = RaptorRetriever(pool, rt.llm, rt.embedder, config=RaptorConfig(max_sources_per_summary=2), tree=tree)
    assert all(len(u.papers) <= 2 for u in capped.retrieve(query, 2048).units)
    union = RaptorRetriever(pool, rt.llm, rt.embedder, config=RaptorConfig(max_sources_per_summary=None), tree=tree)
    for u in union.retrieve(query, 2048).units:  # ablation: the uncapped union of covered leaf papers
        if u.kind == SUMMARY:
            node = tree.nodes[int(u.key.split(":")[2])]
            assert set(u.papers) == set().union(*(leaf_papers[lf] for lf in node.cited_leaves))
    tree.save(tmp_path / "raptor")
    loaded = RaptorTree.load(tmp_path / "raptor")
    assert loaded.layer_sizes() == sizes and np.allclose(loaded.embeddings, tree.embeddings)


def test_raptor_summary_citations_stay_bounded_at_full_budget(pool, rt, query):
    """Regression: summaries used to credit every leaf paper of their subtree, so a
    large-budget RAPTOR context "retrieved" (almost) the whole corpus."""
    cfg = RaptorConfig(clustering="kmeans", kmeans_cluster_size=6, reduction_dim=4)
    rap = RaptorRetriever(pool, rt.llm, rt.embedder, config=cfg).build()
    res = rap.retrieve(query, 8192)
    leaves_cited = sum(len(u.papers) for u in res.units if u.kind != SUMMARY)
    summaries = [u for u in res.units if u.kind == SUMMARY]
    assert summaries and all(1 <= len(u.papers) <= 5 for u in summaries)
    assert len(res.ranked_papers) <= leaves_cited + 5 * len(summaries)
    top = max(rap.tree.summary_nodes, key=lambda n: len(n.paper_counts))
    assert len(top.paper_counts) > 5  # the uncapped union would have credited all of these


def test_raptor_forced_split_respects_cluster_cap(rt):
    """Regression: a cluster that re-clustering cannot shrink is bisected until every
    part fits ``max_cluster_tokens``, so no leaf is silently cut from a summary prompt."""
    from scholarpath.baselines.raptor import RaptorBuilder

    body = ("graph neural networks for molecular property prediction " * 40).strip()
    passages = [Passage(f"chunk:c{i}", CHUNK, [f"p{i}"], title=f"T{i}", heading="Method", body=f"{body} v{i}")
                for i in range(24)]
    cap = 1800
    cfg = RaptorConfig(clustering="gmm", max_clusters=3, reduction_dim=3, max_cluster_tokens=cap)
    builder = RaptorBuilder(rt.llm, rt.embedder, cfg)
    tree = builder.build(passages)
    layer1 = [n for n in tree.summary_nodes if n.layer == 1]
    assert layer1 and len(layer1) < len(passages)
    for n in layer1:
        assert len(n.children) == 1 or sum(count_tokens(tree.nodes[c].text) for c in n.children) <= cap
        assert n.cited_leaves == n.leaves  # every leaf reached the summarizer
    # Bisection of near-duplicates yields balanced parts, not one leaf peeled off at a time.
    tokens = [count_tokens(p.search_text) for p in passages]
    emb = rt.embedder.encode([p.search_text for p in passages])
    parts = builder._split(list(range(18)), tokens, emb)
    assert sorted(m for part in parts for m in part) == list(range(18))
    assert all(sum(tokens[m] for m in part) <= cap for part in parts)
    assert len(parts) <= 2 * math.ceil(sum(tokens[:18]) / cap) and min(len(part) for part in parts) >= 2


def test_raptor_summary_coverage_of_truncated_prompts():
    from scholarpath.prompts.baselines import raptor_cluster_coverage

    texts = ["a" * 400, "", "b" * 400, "c" * 400]  # 100 tokens each (chars / 4 without tiktoken)
    cov = raptor_cluster_coverage(texts, max_input_tokens=150)
    assert cov[0] and not cov[1] and cov[2] is (count_tokens("a" * 400) < 150) and not cov[3]
    assert raptor_cluster_coverage(texts, 10_000) == [True, False, True, True]


def test_raptor_gmm_and_candidate_pool(pool, rt, query):
    cfg = RaptorConfig(clustering="gmm", max_clusters=4, reduction_dim=3, candidate_papers=6)
    dense = DenseRetriever(pool, rt.embedder)
    rap = RaptorRetriever(pool, rt.llm, rt.embedder, config=cfg, candidate_retriever=dense)
    res = rap.retrieve(query, 1024)
    assert res.stats["candidate_papers"] == 6
    assert len(set(res.ranked_papers)) <= 6 and res.context_tokens <= 1024
    calls = rt.meter.component("index_build").calls
    assert calls > 0
    rap.retrieve(query, 1024)  # cached tree: no new summaries
    assert rt.meter.component("index_build").calls == calls


# ---------------------------------------------------------------- LightRAG


def test_lightrag_graph_and_two_query_calls(bench, rt, query):
    small = PassagePool.from_corpus(bench.corpus, paper_ids=sorted(bench.corpus.ids)[:16])
    lr = LightRAGRetriever(small, rt.llm, rt.embedder, LightRAGConfig(min_similarity=0.0)).build()
    g = lr.graph
    assert len(g.entities) > 0 and len(g.relations) > 0
    for e in g.entities.values():
        assert e.sources and all(small.get(s) is not None for s in e.sources)
    for (a, b), r in g.relations.items():
        assert a in g.entities and b in g.entities and r.weight > 0 and r.sources
    n_units = len(small)
    build_calls = rt.meter.component("index_build").calls
    assert build_calls >= 2 * n_units  # extraction + one gleaning pass per unit
    before = rt.meter.snapshot()
    ans = BudgetedQA(lr, rt.llm, budget=2048).answer(query)
    diff = rt.meter.diff(before)
    assert ans.llm_calls == 2
    assert diff["query_understanding"].calls == 1 and diff["composition"].calls == 1
    assert "index_build" not in diff
    res = lr.retrieve(query, 2048)
    kinds = [u.kind for u in res.units]
    assert res.context_tokens <= 2048 and {ENTITY, RELATION} & set(kinds)
    ent_tokens = sum(u.tokens for u in res.units if u.kind == ENTITY)
    assert ent_tokens <= int(lr.config.entity_budget_fraction * 2048)
    # Graph section first, then text units (LightRAG context order).
    first_text = next((i for i, k in enumerate(kinds) if k in (ABSTRACT, CHUNK)), len(kinds))
    assert all(k in (ABSTRACT, CHUNK) for k in kinds[first_text:])


def test_lightrag_graph_citations_follow_query_relevance(bench, rt, query):
    small = PassagePool.from_corpus(bench.corpus, paper_ids=sorted(bench.corpus.ids)[:16])
    lr = LightRAGRetriever(small, rt.llm, rt.embedder, LightRAGConfig(min_similarity=0.0)).build()
    res = lr.retrieve(query, 4096)
    g = lr.graph
    # The query's merged local/global text-unit ranking (from the keywords the query used).
    _, _, local_units = lr._local(lr.view, res.stats["low_level_keywords"])
    _, _, global_units = lr._global(lr.view, res.stats["high_level_keywords"])
    text_rank = {k: r for r, k in enumerate(_round_robin(local_units, global_units))}
    graph_units = [u for u in res.units if u.kind in (ENTITY, RELATION)]
    assert graph_units
    for u in graph_units:
        name = u.key.split(":", 1)[1]
        sources = (g.entities[name].sources if u.kind == ENTITY
                   else g.relations[tuple(name.split("|", 1))].sources)
        assert 1 <= len(u.papers) <= lr.config.max_sources_per_unit
        ranked_sources = sorted((s for s in sources if s in text_rank), key=text_rank.get)
        if ranked_sources:  # the most relevant packed source unit is cited first
            assert u.papers[0] == small.get(ranked_sources[0]).paper_id
    # max_sources_per_unit = 0: only text units are credited (graph blocks carry no ids).
    lr.config.max_sources_per_unit = 0
    res0 = lr.retrieve(query, 4096)
    text_papers = {p for u in res0.units if u.kind in (ABSTRACT, CHUNK) for p in u.papers}
    assert set(res0.ranked_papers) == text_papers
    assert all(not u.papers and not u.text.startswith("[") for u in res0.units if u.kind in (ENTITY, RELATION))


def test_lightrag_candidate_pool_reuses_extractions(pool, rt, query):
    dense = DenseRetriever(pool, rt.embedder)
    lr = LightRAGRetriever(pool, rt.llm, rt.embedder, LightRAGConfig(candidate_papers=4, min_similarity=0.0),
                           candidate_retriever=dense)
    res = lr.retrieve(query, 1024)
    assert res.stats["candidate_papers"] == 4
    n = len(lr.extractions)
    assert n == len(pool.indices_for_papers(dense.top_papers(query, 4)))
    lr.retrieve(query, 1024)
    assert len(lr.extractions) == n


# ---------------------------------------------------------------------- QA


@pytest.mark.parametrize("method", ["bm25", "keyword", "dense"])
def test_budgeted_qa_cites_only_context_papers(method, pool, rt, bench):
    retriever = make_retriever(method, pool, rt.llm, rt.embedder)
    qa = BudgetedQA(retriever, rt.llm, budget=1500)
    for q in bench.evidence_queries[:3]:
        ans = qa.answer(q)
        ctx = set(ans.metadata["context_papers"])
        assert ans.items and ans.cited_papers
        assert set(ans.cited_papers) <= ctx
        assert ans.llm_calls == 1 and ans.context_tokens <= 1500
        assert ans.cost_usd > 0 and ans.input_tokens > ans.context_tokens
        assert ans.retrieved_papers and set(ans.retrieved_papers) == ctx


def test_budgeted_qa_hypothesis_query(pool, rt, bench):
    q = bench.hypothesis_queries[0]
    ans = BudgetedQA(make_retriever("dense", pool, rt.llm, rt.embedder), rt.llm, budget=1024).answer(q)
    assert ans.llm_calls == 1 and set(ans.cited_papers) <= set(ans.metadata["context_papers"])


def test_qa_latency_excludes_offline_index_build(pool, rt, query, monkeypatch):
    """Regression: per-query graph construction (billed to index_build) used to be
    excluded from the call counts but included in Answer.latency_s."""
    dense = DenseRetriever(pool, rt.embedder)
    lr = LightRAGRetriever(pool, rt.llm, rt.embedder, LightRAGConfig(candidate_papers=4, min_similarity=0.0),
                           candidate_retriever=dense)
    original = lr._build_view

    def slow_build_view(indices):
        time.sleep(0.4)
        return original(indices)

    monkeypatch.setattr(lr, "_build_view", slow_build_view)
    ans = BudgetedQA(lr, rt.llm, budget=1024).answer(query)
    assert ans.llm_calls == 2 and ans.metadata["excluded_usage"]["calls"] > 0
    assert ans.metadata["excluded_latency_s"] >= 0.4
    assert ans.metadata["retrieval_stats"]["index_build_s"] >= 0.4
    assert ans.latency_s < ans.metadata["wall_clock_s"] - 0.39
    assert ans.metadata["retrieval_latency_s"] < ans.metadata["retrieval_stats"]["index_build_s"]
    # Lazily encoded dense index: the first query reports the encoding as index build.
    fresh = DenseRetriever(pool, rt.embedder)
    first = fresh.retrieve(query, 512)
    assert first.stats["index_build_s"] > 0 and fresh.retrieve(query, 512).stats["index_build_s"] == 0.0


def test_budgeted_qa_contract_signature(pool, rt, query):
    """docs/ARCHITECTURE.md 4.4: BudgetedQA(retriever, llm, config); answer(query, budget)."""
    retriever = make_retriever("bm25", pool)
    qa = BudgetedQA(retriever, rt.llm, RetrievalConfig(budget=900, answer_max_tokens=256))
    assert qa.budget == 900 and qa.answer_max_tokens == 256
    ans = qa.answer(query)
    assert ans.context_tokens <= 900 and ans.metadata["budget"] == 900
    assert qa.answer(query, 400).context_tokens <= 400
    assert BudgetedQA(retriever, rt.llm, RetrievalConfig(budget=900), budget=300).budget == 300
    assert BudgetedQA(retriever, rt.llm).budget == 8192
    with pytest.raises(TypeError, match="RetrievalConfig"):
        BudgetedQA(retriever, rt.llm, 2048)
    assert NoRetrievalQA(rt.llm, RetrievalConfig(answer_max_tokens=128)).answer_max_tokens == 128


def test_no_retrieval_qa_single_call(rt, query):
    ans = NoRetrievalQA(rt.llm).answer(query)
    assert ans.llm_calls == 1 and ans.context_tokens == 0
    assert ans.cited_papers == [] and ans.retrieved_papers == []
    assert rt.meter.component("composition").calls == 1


def test_make_retriever_names(pool, rt):
    assert make_retriever("specter", pool, rt.llm, rt.embedder).name == "SPECTER"
    assert make_retriever("dense", pool, rt.llm, rt.embedder).name == "Dense retrieval"
    assert make_retriever("Dense + reranker", pool, rt.llm, rt.embedder).name == "Dense + reranker"
    assert make_retriever("raptor", pool, rt.llm, rt.embedder).name == "RAPTOR"
    assert make_retriever("lightrag", pool, rt.llm, rt.embedder).name == "LightRAG"
    with pytest.raises(ValueError):
        make_retriever("nope", pool)
    with pytest.raises(ValueError, match="embedder"):
        make_retriever("dense_rerank", pool, rt.llm)
