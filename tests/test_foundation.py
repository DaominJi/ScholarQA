"""Tests for the shared foundation: config, text, LLM layer, taxonomy tree, index, router."""

from __future__ import annotations

import numpy as np
import pytest

from scholarpath.config import ScholarPathConfig
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.index import SemanticIndex, concept_key, paper_key
from scholarpath.llm import CostMeter, extract_json
from scholarpath.prompts.answer import compose_answer_request, parse_answer
from scholarpath.runtime import Runtime
from scholarpath.taxonomy.llm_ops import TaxonomyLLM, resolve_new_children
from scholarpath.taxonomy.router import MLPRouter, stratified_split
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.text import chunk_sections, count_tokens, normalize_title, split_sections


@pytest.fixture(scope="module")
def bench():
    return make_synthetic_benchmark(n_papers=120, seed=1)


@pytest.fixture()
def rt():
    return Runtime.from_config()


def test_config_roundtrip_and_override(tmp_path):
    cfg = ScholarPathConfig()
    cfg.override(["retrieval.budget=4096", "builder.epsilon=0.2", "link_prediction.cutoffs=[2020-01-01]"])
    assert cfg.retrieval.budget == 4096 and cfg.builder.epsilon == 0.2
    assert cfg.link_prediction.cutoffs == ("2020-01-01",)
    path = tmp_path / "c.yaml"
    cfg.save(path)
    assert ScholarPathConfig.load(path).to_dict() == cfg.to_dict()
    with pytest.raises(AttributeError):
        cfg.override(["retrieval.nope=1"])


def test_text_utils():
    assert normalize_title("Graph  Neural Networks: A {Survey}!") == "graph neural networks a survey"
    secs = split_sections("Intro text\n1 Introduction\nhello\n2. Method\nwe do x\nReferences\n[1] a")
    assert [h for h, _ in secs] == ["Preamble", "Introduction", "Method"]
    long = [("Method", "\n\n".join(["word " * 200] * 5))]
    pieces = chunk_sections(long, max_tokens=300, min_tokens=10)
    assert len(pieces) > 1 and all(count_tokens(t) <= 300 for _, t in pieces)


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('noise {"a": [1, 2,],} trailing') == {"a": [1, 2]}
    assert extract_json("x [1, 2] y", expect=list) == [1, 2]


def test_cost_meter_scopes():
    m = CostMeter(0.14, 0.28)
    with m.scope("acquisition"):
        m.record(1_000_000, 0)
        with m.scope("auditing"):
            m.record(0, 1_000_000)
    assert abs(m.component("acquisition").cost_usd - 0.14) < 1e-9
    assert abs(m.component("auditing").cost_usd - 0.28) < 1e-9
    m.record(10, 10, component="judge")
    assert abs(m.total().cost_usd - 0.42) < 1e-9  # judge excluded


def test_taxonomy_tree_ops(tmp_path):
    t = Taxonomy("problem", "Root")
    a = t.add_node("A", "a")
    b = t.add_node("B", "b", parent=a)
    t.assign("p1", b)
    assert "p1" in t.extent(t.root_id) and t.paper_nodes("p1") == {t.root_id, a, b}
    assert t.deepest_nodes("p1") == [b] and t.distance(b, t.root_id) == 2
    t.save(tmp_path / "t.json")
    t2 = Taxonomy.load(tmp_path / "t.json")
    assert t2.to_dict() == t.to_dict()
    t.remove_subtree(a)
    assert len(t) == 1


def test_index_build_save_load(tmp_path, bench, rt):
    idx = SemanticIndex.build(bench.problem_gold, bench.method_gold, bench.corpus.papers, rt.embedder)
    u = bench.problem_gold.children(bench.problem_gold.root_id)[0]
    m = bench.method_gold.children(bench.method_gold.root_id)[0]
    assert idx.shared_papers(u, m) == idx.extent(u) & idx.extent(m)
    x = idx.vector(paper_key(next(iter(idx.extent(u)))))
    assert idx.ball(u, "paper").upper_cos(x) >= float(x @ x) - 1e-5
    idx.save(tmp_path / "idx")
    idx2 = SemanticIndex.load(tmp_path / "idx")
    assert idx2.stats() == idx.stats()
    assert np.allclose(idx2.vector(concept_key(u)), idx.vector(concept_key(u)))


def test_taxonomy_llm_and_router(bench, rt):
    papers = bench.corpus.papers
    tax = Taxonomy("problem", "Root", root_papers=papers.keys())
    ops = TaxonomyLLM(rt.llm, papers, "problem")
    ids = sorted(papers)
    with rt.meter.scope("acquisition"):
        specs, labels = ops.propose_children(tax, tax.root_id, ids[:40])
    resolve_new_children(tax, tax.root_id, specs, labels, max_children=8)
    kids = tax.children(tax.root_id)
    assert 2 <= len(kids) <= 8
    assert all(c in kids for lab in labels.values() for c in lab.children)
    assert rt.meter.component("acquisition").calls >= 1
    lab = {p: l.children for p, l in labels.items()}
    tr, va = stratified_split(lab, 0.2, seed=0)
    assert set(tr) | set(va) == set(lab) and not set(tr) & set(va)
    vecs = rt.embedder.encode([papers[p].text for p in tr])
    cvecs = rt.embedder.encode([tax.nodes[c].text for c in kids])
    router = MLPRouter(rt.config.builder.router).fit(vecs, cvecs, [{kids.index(c) for c in lab[p]} for p in tr])
    probs = router.predict_proba(vecs, cvecs)
    assert probs.shape == (len(tr), len(kids)) and np.all((probs >= 0) & (probs <= 1))


def test_compose_prompt_mock(rt):
    req = compose_answer_request("How?", "ctx", "evidence", sources=[{"paper_id": "p1", "title": "T"}])
    obj, _ = rt.llm.complete_json(req)
    text, items = parse_answer(obj, valid_ids=["p1"])
    assert items and items[0].citations == ["p1"]
    assert rt.meter.component("composition").calls == 1
