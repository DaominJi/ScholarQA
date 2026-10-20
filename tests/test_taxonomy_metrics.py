"""Tests for soft node / edge F1 (Section 5.2 "Metrics")."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

import numpy as np
import pytest

from scholarpath.config import BaselineBuilderConfig, ScholarPathConfig
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.runtime import Runtime
from scholarpath.evaluation.taxonomy_metrics import (
    TaxonomyScorer,
    soft_edge_f1,
    soft_edge_prf,
    soft_node_f1,
    soft_node_prf,
    taxonomy_scores,
)
from scholarpath.taxonomy.baselines import SelectionBuilder
from scholarpath.taxonomy.tree import Taxonomy


class TableEmbedder:
    """Embedder with hand-set label vectors (unknown texts map to zero vectors)."""

    def __init__(self, table: Dict[str, Sequence[float]], dim: int = 3):
        self.table = {k: np.asarray(v, dtype=np.float32) for k, v in table.items()}
        self.dim = dim
        self.calls: List[str] = []

    def encode(self, texts):
        self.calls.extend(texts)
        return np.stack([self.table.get(t, np.zeros(self.dim, dtype=np.float32)) for t in texts])

    encode_queries = encode


@pytest.fixture(scope="module")
def rt():
    cfg = ScholarPathConfig()
    cfg.embedding.dim = 256
    return Runtime.from_config(cfg)


@pytest.fixture(scope="module")
def bench():
    return make_synthetic_benchmark(n_papers=160, seed=2)


def _copy(tax: Taxonomy) -> Taxonomy:
    return Taxonomy.from_dict(tax.to_dict())


def _move(tax: Taxonomy, node: str, new_parent: str) -> None:
    old = tax.parent(node)
    tax.nodes[old].children.remove(node)
    tax.nodes[new_parent].children.append(node)
    tax.nodes[node].parent = new_parent


def _random_taxonomy(rng: random.Random, vocab: Sequence[str], n_nodes: int) -> Taxonomy:
    tax = Taxonomy("topic", "Root")
    ids = [tax.root_id]
    for _ in range(n_nodes):
        ids.append(tax.add_node(" ".join(rng.sample(list(vocab), 2)), parent=rng.choice(ids)))
    return tax


def test_hand_computed_scores():
    s = 1.0 / np.sqrt(2.0)
    emb = TableEmbedder({"A": [1, 0, 0], "A2": [1, 0, 0], "B": [0, 1, 0], "C": [0, s, s]})
    pred = Taxonomy("topic", "Root")
    a = pred.add_node("A")
    pred.add_node("B", parent=a)  # Root -> A -> B
    gold = Taxonomy("topic", "Gold root")
    gold.add_node("A2")
    gold.add_node("C")  # Root -> A2, Root -> C
    out = taxonomy_scores(pred, gold, emb, fuzzy_threshold=0.9)
    # Nodes: A~A2 = 1, B~C = 1/sqrt(2) on both sides.
    assert out["node_p"] == pytest.approx((1 + s) / 2) and out["node_r"] == pytest.approx((1 + s) / 2)
    assert out["node_f1"] == pytest.approx((1 + s) / 2)
    # Edges: (Root, A) matches (Root, A2) with min(1, 1); (A, B) has a non-root parent
    # and matches nothing; (Root, C) is best matched by (Root, A) with min(1, s(A, C) = 0).
    assert out["edge_p"] == pytest.approx(0.5) and out["edge_r"] == pytest.approx(0.5)
    assert out["edge_f1"] == pytest.approx(0.5)
    assert out["node_fuzzy_p"] == pytest.approx(0.5) and out["node_fuzzy_r"] == pytest.approx(0.5)
    assert out["edge_fuzzy_f1"] == pytest.approx(0.5)
    assert (out["n_pred_nodes"], out["n_gold_nodes"], out["n_pred_edges"], out["n_gold_edges"]) == (2, 2, 2, 2)
    # The "mean" combiner credits (A, B) with (s(A, Root*) + s(B, C)) / 2 = s / 2, and
    # (Root*, C) with (s(Root, Root*) + s(A, C)) / 2 = 1 / 2.
    mean = soft_edge_prf(pred, gold, emb, combine="mean")
    assert mean.precision == pytest.approx((1 + s / 2) / 2) and mean.recall == pytest.approx(0.75)
    # One-to-one matching: A<->A2 and B<->C for nodes; (Root, A)<->(Root*, A2) for edges.
    matched = TaxonomyScorer(emb, aggregate="matching").score(pred, gold)
    assert matched["node_f1"] == pytest.approx((1 + s) / 2) and matched["edge_f1"] == pytest.approx(0.5)


def test_identical_taxonomies_score_one(rt, bench):
    gold = bench.problem_gold
    for kw in ({}, {"aggregate": "matching"}, {"text": "name_definition"}, {"combine": "product"}):
        out = taxonomy_scores(gold, _copy(gold), rt.embedder, fuzzy_threshold=0.8, **kw)
        for key in ("node_p", "node_r", "node_f1", "edge_p", "edge_r", "edge_f1", "node_fuzzy_f1", "edge_fuzzy_f1"):
            assert out[key] == pytest.approx(1.0), (kw, key)
    # Roots are excluded by default, so differently named roots do not matter.
    renamed = _copy(gold)
    renamed.root.name = "Completely different root label"
    assert soft_node_f1(renamed, gold, rt.embedder) == pytest.approx(1.0)
    assert soft_edge_f1(renamed, gold, rt.embedder) == pytest.approx(1.0)
    assert soft_node_f1(renamed, gold, rt.embedder, include_root=True) < 1.0


def test_built_taxonomy_scores_lower_than_gold(rt, bench):
    gold = bench.problem_gold
    full = SelectionBuilder(rt.llm, rt.embedder, BaselineBuilderConfig(selector="full"), bench.corpus).build(
        "Synthetic Repository"
    )
    out = taxonomy_scores(full.taxonomy, gold, rt.embedder)
    perfect = taxonomy_scores(gold, gold, rt.embedder)
    assert 0.0 < out["node_f1"] < perfect["node_f1"] == pytest.approx(1.0)
    assert 0.0 <= out["edge_f1"] < perfect["edge_f1"]
    assert out["edge_f1"] <= out["node_f1"] + 1e-12


def test_perturbed_tree_keeps_nodes_but_loses_edges(rt, bench):
    gold = bench.problem_gold
    pert = _copy(gold)
    tops = pert.children(pert.root_id)
    _move(pert, pert.children(tops[0])[0], tops[1])  # re-parent one sub-problem
    out = taxonomy_scores(pert, gold, rt.embedder)
    assert out["node_f1"] == pytest.approx(1.0)
    assert out["edge_f1"] < 1.0
    assert out["edge_p"] <= out["node_p"] and out["edge_r"] <= out["node_r"]


def test_precision_recall_asymmetry(rt, bench):
    gold = bench.problem_gold
    partial = _copy(gold)
    for top in partial.children(partial.root_id)[:2]:
        partial.remove_subtree(top)
    prf = soft_node_prf(partial, gold, rt.embedder)
    assert prf.precision == pytest.approx(1.0) and prf.recall < 1.0
    rev = soft_node_prf(gold, partial, rt.embedder)
    assert rev.recall == pytest.approx(1.0) and rev.precision == pytest.approx(prf.recall)


def test_edge_f1_never_exceeds_node_f1(rt):
    rng = random.Random(0)
    vocab = ["graph", "neural", "drug", "flood", "query", "index", "kernel", "sparse", "transport", "mutation"]
    for _ in range(10):
        a = _random_taxonomy(rng, vocab, rng.randint(3, 14))
        b = _random_taxonomy(rng, vocab, rng.randint(3, 14))
        for kw in ({}, {"combine": "product"}, {"aggregate": "matching"}):
            scorer = TaxonomyScorer(rt.embedder, fuzzy_threshold=0.5, **kw)
            out = scorer.score(a, b)
            assert out["edge_f1"] <= out["node_f1"] + 1e-9
            assert out["edge_fuzzy_f1"] <= out["node_fuzzy_f1"] + 1e-9


def test_options_and_conventions(rt, bench):
    gold = bench.problem_gold
    n_top = len(gold.children(gold.root_id))
    out = taxonomy_scores(gold, gold, rt.embedder, root_edges=False)
    assert out["n_gold_edges"] == len(gold.edges()) - n_top and out["edge_f1"] == pytest.approx(1.0)
    root_only = Taxonomy("problem", "Root")
    assert taxonomy_scores(root_only, Taxonomy("problem", "Other"), rt.embedder)["node_f1"] == 1.0
    empty = taxonomy_scores(root_only, gold, rt.embedder)
    assert empty["node_f1"] == 0.0 and empty["edge_f1"] == 0.0 and empty["n_pred_nodes"] == 0
    # Nested-dict input (e.g. a gold taxonomy file) is accepted.
    nested = gold.to_nested(include_papers=False)
    assert soft_node_f1(nested, gold, rt.embedder) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        TaxonomyScorer(rt.embedder, text="title")
    with pytest.raises(ValueError):
        TaxonomyScorer(rt.embedder, combine="max")


def test_definition_text_mode():
    emb = TableEmbedder({"X": [1, 0, 0], "X: about a": [0, 1, 0], "X: about b": [0, 0, 1]})
    a, b = Taxonomy("topic", "R"), Taxonomy("topic", "R")
    a.add_node("X", "about a")
    b.add_node("X", "about b")
    assert soft_node_f1(a, b, emb) == pytest.approx(1.0)
    assert soft_node_f1(a, b, emb, text="name_definition") == pytest.approx(0.0)
    assert soft_node_f1(a, b, emb, text=lambda node: node.name.lower()) == pytest.approx(0.0)  # unknown -> zero


def test_label_embeddings_are_cached():
    emb = TableEmbedder({"A": [1, 0, 0], "B": [0, 1, 0]})
    a = Taxonomy("topic", "R")
    a.add_node("A")
    a.add_node("B")
    scorer = TaxonomyScorer(emb)
    scorer.score(a, a)
    scorer.score(a, _copy(a))
    assert sorted(emb.calls) == ["A", "B", "R"]  # each distinct label embedded once
    shared: Dict[str, np.ndarray] = {}
    soft_node_f1(a, a, emb, cache=shared)
    soft_edge_f1(a, a, emb, cache=shared)
    assert set(shared) == {"A", "B", "R"} and len(emb.calls) == 6
