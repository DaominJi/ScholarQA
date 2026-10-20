"""Tests for the organization benchmarks (Section 5.3): the hierarchy composition
prompt (prompts/organize.py) and the TaxoBench / CHIME metrics
(evaluation/organization_metrics.py: ARI, Sem, Cat, Assn)."""

from __future__ import annotations

import json
import random
from typing import Any, Callable, Dict, Iterable, Optional

import pytest

from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.evaluation import organization_metrics as om
from scholarpath.llm.base import LLMResponse
from scholarpath.prompts.organize import MOCK_HANDLERS, STYLES, TASK, organize, organize_request, parse_hierarchy
from scholarpath.runtime import Runtime
from scholarpath.taxonomy.tree import Taxonomy

# Toy example shipped with the public TaxoBench scorer (examples/toy_*.jsonl).
TOY_REFERENCE = {
    "name": "Toy LLM Systems",
    "subtopics": [
        {"name": "Retrieval", "papers": ["Retrieval Augmented Generation"]},
        {"name": "Agents", "papers": ["Agent Tool Use"]},
        {"name": "Evaluation", "papers": ["Long Context Evaluation"]},
    ],
}
TOY_PREDICTION = {
    "name": "Toy LLM Systems",
    "subtopics": [
        {"name": "Retrieval Methods", "papers": ["Retrieval Augmented Generation"]},
        {"name": "Agentic Systems", "papers": ["Agent Tool Use"]},
        {"name": "Benchmarks", "papers": ["Long Context Evaluation"]},
    ],
}


@pytest.fixture(scope="module")
def bench():
    return make_synthetic_benchmark(n_papers=48, seed=3)


@pytest.fixture(scope="module")
def rt():
    return Runtime.from_config()


def _remap(tax: Taxonomy, fn: Callable[[str], Optional[str]]) -> Taxonomy:
    """Copy ``tax`` with every paper id replaced by ``fn(id)`` (dropped if None)."""

    def rec(node: Dict[str, Any]) -> Dict[str, Any]:
        papers = [fn(p) for p in node["papers"]]
        return {"name": node["name"], "papers": [p for p in papers if p is not None],
                "children": [rec(c) for c in node["children"]]}

    return Taxonomy.from_nested(rec(tax.to_nested()))


def _labels(tax: Taxonomy) -> Dict[str, str]:
    return {p: tax.nodes[nodes[0]].name for p, nodes in om.paper_attachments(tax).items()}


# ---------------------------------------------------------------------------
# TaxoBench: ARI and Sem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_embedder", [True, False])
def test_identical_hierarchies_score_one(bench, rt, use_embedder):
    gold = bench.problem_gold
    pred = _remap(gold, lambda p: p)
    emb = rt.embedder if use_embedder else None
    tb = om.organization_scores(pred, gold, "taxobench", emb, titles=bench.corpus)
    assert tb["ari"] == pytest.approx(1.0) and tb["ari_intersection"] == pytest.approx(1.0)
    assert tb["sem"] == pytest.approx(1.0) and tb["sem_expert"] == pytest.approx(1.0)
    assert tb["v_measure"] == pytest.approx(1.0) and tb["paper_recall"] == 1.0 and tb["paper_precision"] == 1.0
    if emb is None:  # Cat / Assn are embedding-based: no implicit string fallback
        with pytest.raises(ValueError, match="embedder"):
            om.organization_scores(pred, gold, "chime")
        ch = om.organization_scores(pred, gold, "chime", similarity=om.NodeSimilarity())
    else:
        ch = om.organization_scores(pred, gold, "chime", emb)
    assert ch["cat"] == pytest.approx(1.0) and ch["assn"] == pytest.approx(1.0)


def test_shuffled_assignment_lowers_ari(bench, rt):
    gold = bench.problem_gold
    ids = sorted(gold.all_papers())
    perm = ids[:]
    random.Random(0).shuffle(perm)
    mapping = dict(zip(ids, perm))
    pred = _remap(gold, mapping.get)
    tb = om.organization_scores(pred, gold, "taxobench", rt.embedder, titles=bench.corpus)
    assert tb["ari"] < 0.3 and tb["paper_recall"] == 1.0
    assert tb["sem"] < 1.0
    ch = om.organization_scores(pred, gold, "chime", rt.embedder)
    assert ch["cat"] == pytest.approx(1.0)  # same category names
    assert ch["assn"] < 0.8  # but studies sit in the wrong categories


def test_missing_papers_and_ari_views(bench, rt):
    gold = bench.problem_gold
    ids = sorted(gold.all_papers())
    dropped = set(ids[::4])
    pred = _remap(gold, lambda p: None if p in dropped else p)
    kw = dict(titles=bench.corpus)
    ari_cond = om.adjusted_rand_index(pred, gold, **kw)  # default: ARI∩ on retrieved expert papers (Table 3)
    ari_e2e = om.adjusted_rand_index(pred, gold, include_missing=True, **kw)  # unretrieved papers share ⊥
    assert ari_cond == pytest.approx(1.0) and ari_e2e < 1.0
    scores = om.organization_scores(pred, gold, "taxobench", rt.embedder, **kw)
    assert scores["ari"] == scores["ari_intersection"] == pytest.approx(1.0)
    assert scores["ari_e2e"] == pytest.approx(ari_e2e) and scores["v_measure_e2e"] < 1.0
    e2e = om.organization_scores(pred, gold, "taxobench", rt.embedder, ari_view="e2e", **kw)
    assert e2e["ari"] == pytest.approx(ari_e2e) and e2e["v_measure"] == pytest.approx(scores["v_measure_e2e"])
    with pytest.raises(ValueError):
        om.organization_scores(pred, gold, "taxobench", ari_view="both", **kw)
    sem = om.semantic_path_scores(pred, gold, embedder=rt.embedder, **kw)
    frac = 1 - len(dropped) / len(ids)
    assert sem["sem_path"] == pytest.approx(1.0)  # Eq. 13 averages over aligned pairs
    assert sem["sem_path_expert"] == pytest.approx(frac)  # unretrieved papers score 0
    assert om.semantic_path_similarity(pred, gold, rt.embedder, average="expert", **kw) == pytest.approx(frac)
    assert sem["n_aligned"] == len(ids) - len(dropped) and sem["n_reference"] == len(ids)
    assn = om.assignment_prf(pred, gold, rt.embedder)
    assert assn.precision == pytest.approx(1.0) and assn.recall == pytest.approx(frac)

    empty = _remap(gold, lambda p: None)
    scores = om.organization_scores(empty, gold, "taxobench", rt.embedder, **kw)
    assert scores["ari"] == scores["ari_e2e"] == 0.0 and scores["sem"] == 0.0 and scores["n_aligned"] == 0
    assert om.organization_scores(empty, gold, "chime", rt.embedder)["assn"] == 0.0


def test_taxobench_toy_example_matches_public_scorer():
    gold = Taxonomy.from_nested(TOY_REFERENCE, children_key="subtopics")
    pred = parse_hierarchy(TOY_PREDICTION)
    scores = om.organization_scores(pred, gold, "taxobench")
    # Values of taxobench-score v0.2.0 (clustering_metrics / sem_path_score with its string fallback).
    assert scores["ari"] == pytest.approx(1.0)
    assert scores["sem"] == pytest.approx(0.7518518518518519)
    assert om.to_taxobench_tree(pred) == TOY_PREDICTION


def test_title_matching_threshold_and_rules():
    assert om.normalize_benchmark_title("Few-Shot Learners!") == "few shot learners"
    assert om.normalize_benchmark_title("Few-Shot Learners!", rule="strict") == "fewshot learners"
    gold = ["Attention Is All You Need", "BERT", "Deep Residual Learning for Image Recognition",
            "Graph Neural Networks: A Review",
            "Neural Machine Translation by Jointly Learning to Align and Translate"]
    pred = ["attention is all you need.", "BERT: Pre-training of Deep Bidirectional Transformers",
            "Deep Residual Learning for Image Recognitions", "Graph Neural Networks for Text",
            "Neural machine translation by jointly learning to align & translate"]

    def matched(**kw):
        return [m.pred_id is not None for m in om.align_papers(gold, pred, **kw).matches]

    # Public scorer: containment counts as 1.0, difflib ratio >= 0.92 otherwise.
    assert matched() == [True, True, True, False, True]
    assert matched(threshold=0.5) == [True, True, True, True, True]
    # Paper-era rule: exact after normalisation, or 0.6 <= s < 1 with strict containment.
    assert matched(rule="strict") == [True, False, True, False, False]
    assert matched(rule="strict", threshold=0.5) == [True, False, True, True, True]
    with pytest.raises(ValueError):
        om.align_papers(gold, pred, rule="fuzzy")

    # Greedy one-to-one: a predicted paper is used at most once (first expert paper wins).
    al = om.align_papers(["Graph Attention Networks", "Graph Attention Networks."], ["graph attention networks"])
    assert [m.pred_id for m in al.matches] == ["graph attention networks", None]
    assert al.recall == 0.5 and al.precision == 1.0


def test_prediction_ids_resolved_to_titles(bench):
    """Predicted corpus ids are aligned to expert papers given only by title (raw TaxoBench trees)."""
    gold_ids = bench.problem_gold
    corpus = bench.corpus
    gold_titles = _remap(gold_ids, lambda p: corpus[p].title)
    pred = _remap(gold_ids, lambda p: p)
    scores = om.organization_scores(pred, gold_titles, "taxobench", titles=corpus)
    assert scores["ari"] == pytest.approx(1.0) and scores["paper_recall"] == 1.0
    assert om.adjusted_rand_index(pred, gold_titles) == 0.0  # ids are not titles: nothing aligns
    lookup = {p.paper_id: p.title for p in corpus}
    assert om.adjusted_rand_index(pred, gold_titles, titles=lookup.get) == pytest.approx(1.0)


def test_chain_alignment_cost_dp():
    exact = lambda a, b: 1.0 if a == b else 0.0  # noqa: E731
    assert om.chain_alignment_cost(["R", "A", "B"], ["R", "A", "B"], exact) == 0.0
    # Shorter chain matched to an ordered subsequence; one unmatched node costs lambda = 1.
    assert om.chain_alignment_cost(["R", "B"], ["R", "A", "B"], exact) == pytest.approx(1.0)
    assert om.chain_alignment_cost(["R", "A", "B"], ["R", "B"], exact, unmatched_penalty=0.5) == pytest.approx(0.5)
    # Order matters: (B, A) cannot be aligned to (A, B) for free.
    assert om.chain_alignment_cost(["B", "A"], ["A", "B"], exact) == pytest.approx(2.0)
    assert om.chain_alignment_cost([], ["A", "B"], exact) == pytest.approx(2.0)
    half = lambda a, b: 0.5  # noqa: E731
    assert om.chain_alignment_cost(["X"], ["Y", "Z"], half) == pytest.approx(1.5)


def test_multi_placed_papers():
    gold = Taxonomy.from_nested({"name": "R", "children": [
        {"name": "A", "papers": ["p1", "p2", "p3"]}, {"name": "B", "papers": ["p4", "p5", "p6"]}]})
    pred = Taxonomy.from_nested({"name": "R", "children": [
        {"name": "A", "papers": ["p1", "p2", "p3"]},
        {"name": "B", "papers": ["p4", "p5", "p6", "p1"]},
        {"name": "C", "children": [{"name": "Deep", "papers": ["p2"]}]}]})
    att = om.paper_attachments(pred)
    assert [pred.nodes[n].name for n in att["p1"]] == ["A", "B"]
    assert [pred.nodes[n].name for n in att["p2"]] == ["A", "Deep"]
    # Labels use the first most specific category in pre-order, so ARI is unaffected.
    assert om.adjusted_rand_index(pred, gold) == pytest.approx(1.0)
    # Sem takes the best of the candidate ancestor chains (TaxoBench App. A.4).
    assert om.semantic_path_similarity(pred, gold) == pytest.approx(1.0)
    assert len(om.ancestor_chains(pred, "p2")) == 2


# ---------------------------------------------------------------------------
# CHIME: Cat and Assn
# ---------------------------------------------------------------------------


def _aspect(name: str, children: Dict[str, Iterable[str]]) -> Taxonomy:
    return Taxonomy.from_nested({"name": name, "children": [
        {"name": c, "papers": list(ps)} for c, ps in children.items()]})


def test_chime_aspects_duplicates_and_threshold(rt):
    a1 = _aspect("Family smoking prevention programs", {"Smoking onset": ["s1", "s2"], "Adverse events": ["s3"]})
    a2 = _aspect("Nicotine replacement therapy", {"Abstinence rates": ["s4", "s5", "s6"], "Adverse events": ["s7"]})
    gold = om.merge_hierarchies([a1, a2], root_name="Smoking cessation review")
    assert [gold.nodes[c].name for c in gold.children(gold.root_id)] == [a1.root.name, a2.root.name]
    assert om.subtree(gold, gold.children(gold.root_id)[1]).to_nested(include_papers=True)["children"] == \
        a2.to_nested()["children"]
    pred = om.merge_hierarchies([a1, a2], root_name="Topic")
    # Gold given as a list of aspect hierarchies is merged; equally named categories are aligned by path.
    scores = om.organization_scores(pred, [a1, a2], "chime", rt.embedder)
    assert scores["cat"] == pytest.approx(1.0) and scores["assn"] == pytest.approx(1.0)
    assert scores["n_aligned_categories"] == 6

    # Swapping the two "Adverse events" studies is only visible to Assn.
    swapped = om.merge_hierarchies([
        _aspect(a1.root.name, {"Smoking onset": ["s1", "s2"], "Adverse events": ["s7"]}),
        _aspect(a2.root.name, {"Abstinence rates": ["s4", "s5", "s6"], "Adverse events": ["s3"]})], "Topic")
    sw = om.assignment_prf(swapped, gold, rt.embedder)
    assert om.category_soft_f1(swapped, gold, rt.embedder) == pytest.approx(1.0)
    assert sw.precision == pytest.approx(10 / 14) and sw.recall == pytest.approx(10 / 14)
    assert om.assignment_f1(swapped, gold, rt.embedder) == pytest.approx(sw.f1)
    assert om.assignment_f1(swapped, gold, rt.embedder, mode="direct") == pytest.approx(5 / 7)

    # Unrelated category names: soft F1 near 0 and nothing aligned above the threshold.
    other = _aspect("Unrelated", {"Quantum chromodynamics": ["s1", "s2", "s3"], "Galaxy formation": ["s4"]})
    assert om.category_soft_f1(other, gold, rt.embedder) < 0.2
    assert om.align_categories(other, gold, rt.embedder) == {}
    assert om.assignment_f1(other, gold, rt.embedder) == 0.0
    # Any (a, b) -> float callable works as the node similarity.
    exact = lambda a, b: float(a == b)  # noqa: E731
    assert om.category_soft_f1(pred, gold, similarity=exact) == pytest.approx(1.0)


def test_assignment_f1_paper_map_and_irrelevant_studies(rt):
    gold = _aspect("Review", {"Exercise": ["s1", "s2"], "Diet": ["s3", "s4"]})
    pred = _aspect("Review", {"Exercise": ["x1", "x2", "noise"], "Diet": ["x3", "x4"]})
    pmap = {"x1": "s1", "x2": "s2", "x3": "s3", "x4": "s4"}
    res = om.assignment_prf(pred, gold, rt.embedder, paper_map=pmap)
    assert res.recall == pytest.approx(1.0) and res.precision == pytest.approx(4 / 5)
    assert om.assignment_f1(pred, gold, rt.embedder) == 0.0


def test_category_alignment_thresholds_before_matching():
    """Sub-threshold similarities must not steer the one-to-one matching of Assn (review regression)."""
    table = {("X", "Y1"): 0.50, ("X", "Y2"): 0.49, ("Z", "Y1"): 0.49}
    sim = lambda a, b: 1.0 if a == b else table.get((a, b), 0.0)  # noqa: E731
    gold = Taxonomy.from_nested({"name": "R", "children": [
        {"name": "Y1", "papers": ["s1", "s2"]}, {"name": "Y2", "papers": ["s3"]}]})
    pred = Taxonomy.from_nested({"name": "R", "children": [
        {"name": "X", "papers": ["s1", "s2"]}, {"name": "Z", "papers": ["s3"]}]})
    for tie_break in (0.0, 1e-3):
        al = om.align_categories(pred, gold, similarity=sim, tie_break=tie_break)
        assert {pred.nodes[k].name: gold.nodes[v].name for k, v in al.items()} == {"X": "Y1"}
    res = om.assignment_prf(pred, gold, similarity=sim)
    # (s1, Y1) and (s2, Y1) are correct; (s3, Z) sits in an unaligned category (a false positive and a miss).
    assert res.precision == pytest.approx(2 / 3) and res.recall == pytest.approx(2 / 3)
    assert om.align_categories(pred, gold, similarity=sim, threshold=0.6) == {}


def test_chime_metrics_require_embedding_similarity(rt):
    gold = Taxonomy.from_nested({"name": "Exercise for hypertension", "children": [
        {"name": "Aerobic exercise interventions", "children": [{"name": "Blood pressure outcomes",
                                                                 "papers": ["s1", "s2"]}]},
        {"name": "Dietary co-interventions", "papers": ["s3", "s4"]}]})
    degenerate = Taxonomy.from_nested({"name": "Q", "children": [
        {"name": "e", "children": [{"name": "o", "papers": ["s1", "s2"]}]}, {"name": "i", "papers": ["s3", "s4"]}]})
    for fn in (om.category_soft_prf, om.category_soft_f1, om.align_categories, om.assignment_prf, om.assignment_f1):
        with pytest.raises(ValueError, match="embedder"):
            fn(degenerate, gold)
    with pytest.raises(ValueError, match="embedder"):
        om.organization_scores(degenerate, gold, "chime")
    scores = om.organization_scores(degenerate, gold, "chime", rt.embedder)
    assert scores["cat"] < 0.2 and scores["assn"] == 0.0
    assert om.organization_scores(gold, gold, "chime", rt.embedder)["assn"] == pytest.approx(1.0)
    # The TaxoBench string fallback scores containment as 1 ("e" in "Aerobic ..."): opt-in only, never implicit.
    assert om.organization_scores(degenerate, gold, "chime", similarity=om.NodeSimilarity())["cat"] == 1.0


def test_cat_matches_soft_node_f1_and_scores_are_floats(bench, rt):
    tm = pytest.importorskip("scholarpath.evaluation.taxonomy_metrics")
    if not hasattr(tm, "soft_node_prf"):
        pytest.skip("taxonomy_metrics.soft_node_prf not available")
    pred, gold = bench.method_gold, bench.problem_gold
    cat = om.category_soft_prf(pred, gold, rt.embedder)
    ref = tm.soft_node_prf(pred, gold, rt.embedder)
    assert (cat.precision, cat.recall, cat.f1) == pytest.approx((ref.precision, ref.recall, ref.f1), abs=1e-6)
    for value in (om.category_soft_f1(pred, gold, rt.embedder), om.assignment_f1(pred, gold, rt.embedder)):
        assert isinstance(value, float) and 0.0 <= value <= 1.0
    empty = Taxonomy("topic")
    assert om.category_soft_f1(empty, empty, rt.embedder) == tm.soft_node_f1(empty, empty, rt.embedder) == 1.0
    assert om.category_soft_f1(empty, gold, rt.embedder) == 0.0


def test_macro_average():
    rows = [{"ari": 1.0, "sem": 0.5, "n": 3}, {"ari": 0.0, "sem": float("nan")}, {"ari": 0.5, "label": "x"}]
    avg = om.macro_average(rows)
    assert avg["ari"] == pytest.approx(0.5) and avg["sem"] == pytest.approx(0.5) and avg["n"] == 3.0
    assert "label" not in avg and avg["n_queries"] == 3.0
    table3 = om.macro_average(rows, keys=om.ORGANIZATION_METRICS["taxobench"])
    assert table3 == {"ari": 0.5, "sem": 0.5, "n_queries": 3.0}
    with pytest.raises(ValueError):
        om.organization_scores(Taxonomy("topic"), Taxonomy("topic"), "unknown-benchmark")


# ---------------------------------------------------------------------------
# Composition prompt and parser
# ---------------------------------------------------------------------------


def test_parse_hierarchy_robustness():
    valid = ["p1", "p2", "p3", "p4", "p5"]
    text = (
        "Sure! Here is the hierarchy:\n```json\n"
        '{"hierarchy": {"title": "Topic", "subtopics": ['
        '{"category": "1. Graph methods", "paper_ids": ["[p1]", {"paper_id": "p2"}, "p9"],},'
        '{"name": "Empty", "papers": []},'
        '"Kernels [p3, p4]"]}}\n```'
    )
    tax = parse_hierarchy(text, valid_ids=valid)
    assert tax.facet == "topic" and tax.root.name == "Topic"
    assert {tax.nodes[c].name: sorted(tax.nodes[c].papers) for c in tax.children(tax.root_id)} == {
        "Graph methods": ["p1", "p2"], "Kernels": ["p3", "p4"]}
    assert tax.metadata["dropped_ids"] == ["p9"]

    # Name-keyed mappings, CHIME-style nested lists, top-level lists.
    keyed = parse_hierarchy({"Survey": {"Methods": {"GNN": ["p1"], "Kernels": {"papers": ["p2"]}}}}, valid)
    assert keyed.root.name == "Survey" and keyed.pretty(show_sizes=False).split("\n")[1:] == [
        "  - Methods", "    - GNN", "    - Kernels"]
    nested = parse_hierarchy([["Aspect A", ["p1", "p2"], [["Sub", ["p1"], []]]], ["Aspect B", ["p3"], []]], valid)
    assert nested.deepest_nodes("p1") and nested.nodes[nested.deepest_nodes("p1")[0]].name == "Sub"
    assert nested.nodes[nested.deepest_nodes("p2")[0]].name == "Aspect A"

    # Markdown outline fallback when no JSON is present.
    outline = (
        "# My topic\n## Graph methods [p1]\n- GNNs\n  - [p2] A GNN paper\n  - [p3] Another\n"
        "## Other\n1. Misc [p4, p5]"
    )
    otax = parse_hierarchy(outline, valid_ids=valid)
    assert otax.root.name == "My topic" and otax.all_papers() == set(valid)
    assert otax.nodes[otax.deepest_nodes("p2")[0]].name == "GNNs"

    # Depth cap, garbage, empty output.
    deep = parse_hierarchy({"name": "R", "children": [{"name": "A", "children": [
        {"name": "B", "children": [{"name": "C", "papers": ["p1"]}]}]}]}, max_depth=2)
    assert deep.max_depth == 2 and deep.nodes[deep.deepest_nodes("p1")[0]].name == "B"
    for bad in (None, 42, "no hierarchy here", "", {"unexpected": 1}):
        t = parse_hierarchy(bad, valid_ids=valid, root_name="Q")
        assert len(t) == 1 and not t.all_papers()
    kept = parse_hierarchy({"name": "R", "children": [{"name": "Empty"}]}, prune_empty=False)
    assert len(kept) == 2


class _ReplyLLM:
    """Stub backbone returning a fixed raw reply: exercises the text that organize() actually parses."""

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return LLMResponse(text=self.text, input_tokens=1, output_tokens=1)


SOURCES = [{"paper_id": f"p{i}", "title": f"Paper {i}"} for i in range(1, 6)]
FULL = {"name": "Topic", "children": [
    {"name": "Graph methods", "children": [{"name": "GNNs", "papers": ["p1", "p2"]},
                                           {"name": "Kernels", "papers": ["p3"]}]},
    {"name": "Sequence models", "children": [{"name": "RNNs", "papers": ["p4", "p5"]}]}]}
FULL_SHAPE = {"Topic": ["p1", "p2", "p3", "p4", "p5"], "Graph methods": ["p1", "p2", "p3"], "GNNs": ["p1", "p2"],
              "Kernels": ["p3"], "Sequence models": ["p4", "p5"], "RNNs": ["p4", "p5"]}
ID_VARIANTS = {"name": "Topic", "children": [
    {"name": "Graph methods", "children": [{"name": "GNNs", "papers": "p1, P2"},
                                           {"name": "Kernels", "children": ["p3: A kernel paper"]}]},
    {"name": "Sequence models", "children": [{"name": "RNNs", "children": [
        {"paper_id": "p4", "title": "An RNN paper"}, {"id": "[p5]", "title": "Another one"}]}]}]}


def _shape(tax: Taxonomy) -> Dict[str, Any]:
    return {tax.nodes[n].name: sorted(tax.nodes[n].papers) for n in tax.bfs()}


@pytest.mark.parametrize("reply", [
    json.dumps(FULL),
    json.dumps(FULL["children"]),  # top-level list of categories as raw text
    "Here it is:\n```json\n" + json.dumps(FULL) + "\n```\nI used every paper [1].",
    'Format:\n```json\n{"name": "<root>", "children": [{"name": "<category>", "papers": ["<paper_id>"]}]}\n```\n'
    "Answer:\n```json\n" + json.dumps(FULL) + "\n```",
    json.dumps(FULL).replace('}]}, {"name": "Sequence', '}]} {"name": "Sequence'),  # missing comma
    json.dumps(FULL).replace('["p3"]}', '["p3"}'),  # missing closer
    "Note [the ids below are from the context:\n" + json.dumps(FULL),  # unclosed bracket in prose
    json.dumps({"topic": "Topic", "hierarchy": {"children": FULL["children"]}}),  # wrapper naming the root
    json.dumps({"title": "Topic", "taxonomy": FULL}),
    json.dumps(ID_VARIANTS),  # "p1, P2", "p3: title" child, paper dicts among the children
], ids=["plain", "top-list", "prose-fence", "example-fence", "missing-comma", "missing-closer", "prose-bracket",
        "wrapper-topic", "wrapper-title", "id-variants"])
def test_organize_parses_raw_replies(reply):
    llm = _ReplyLLM(reply)
    tax = organize(llm, "Topic", "ctx", SOURCES)
    assert llm.calls == 1 and tax.metadata["llm_calls"] == 1
    assert _shape(tax) == FULL_SHAPE and tax.metadata["dropped_ids"] == []


def test_parse_truncated_and_outline_replies():
    text = json.dumps(FULL)
    cut = text[: text.index('"p5"')]  # reply cut at max_tokens inside the last category
    tax = parse_hierarchy(cut, valid_ids=[s["paper_id"] for s in SOURCES], root_name="Q")
    assert _shape(tax) == {**FULL_SHAPE, "Topic": ["p1", "p2", "p3", "p4"], "Sequence models": ["p4"], "RNNs": ["p4"]}
    for at in ('"Kernels"', '"papers": ["p3"', '"Sequence models"'):
        part = parse_hierarchy(text[: text.index(at) + 3], valid_ids=[s["paper_id"] for s in SOURCES])
        assert part.root.name == "Topic" and {"p1", "p2"} <= part.all_papers()

    # Bracketed numeric ids (e.g. PMIDs) must not be read as a bare JSON list instead of the outline.
    outline = "# Exercise review\n## Aerobic training [101, 102]\n## Diet [103]\n- [104] A diet trial\n"
    otax = parse_hierarchy(outline, valid_ids=["101", "102", "103", "104"])
    assert _shape(otax) == {"Exercise review": ["101", "102", "103", "104"], "Aerobic training": ["101", "102"],
                            "Diet": ["103", "104"]}
    # A bare id list is still used when nothing else is available.
    assert parse_hierarchy("Papers: [101, 102]", valid_ids=["101", "102"], root_name="Q").all_papers() == {"101", "102"}


def test_organize_request_rendering():
    sources = [{"paper_id": "p1", "title": "Graph Networks"}, {"paper_id": "p2", "title": "Kernels"}]
    req = organize_request("Graph learning", "[p1] Graph Networks ...", sources, max_tokens=999)
    assert req.task == TASK == "organize.hierarchy" and TASK in MOCK_HANDLERS
    assert req.component == "composition" and req.max_tokens == 999 and req.json_mode
    assert STYLES["survey"] in req.prompt_text and "most specific" in req.prompt_text.lower()
    assert "Papers available for placement" not in req.prompt_text
    assert req.payload["sources"] == sources
    review = organize_request("Smoking", "ctx", sources, style="review", list_sources=True, component="agent")
    assert STYLES["review"] in review.prompt_text and "[p2] Kernels" in review.prompt_text
    assert review.component == "agent"
    with pytest.raises(ValueError):
        organize_request("Q", "ctx", sources, style="poem")


def test_mock_organize_returns_valid_hierarchy(bench, rt):
    papers = list(bench.corpus)[:24]
    sources = [{"paper_id": p.paper_id, "title": p.title} for p in papers]
    context = "\n".join(f"[{p.paper_id}] {p.title}. {p.abstract}" for p in papers)
    before = rt.meter.component("composition").calls
    tax = organize(rt.llm, "Machine learning for scientific problems", context, sources)
    assert rt.meter.component("composition").calls == before + 1  # exactly ONE composition call
    assert tax.all_papers() == {s["paper_id"] for s in sources}  # every source placed, nothing invented
    assert len(tax.children(tax.root_id)) >= 2 and tax.metadata["llm_calls"] == 1
    # Deterministic, and the raw JSON reply parses to the same hierarchy.
    obj, _ = rt.llm.complete_json(organize_request("Machine learning for scientific problems", context, sources))
    again = parse_hierarchy(obj, valid_ids=[s["paper_id"] for s in sources])
    assert json.dumps(again.to_nested(), sort_keys=True) == json.dumps(tax.to_nested(), sort_keys=True)

    # End to end against the gold problem taxonomy restricted to these papers.
    keep = {s["paper_id"] for s in sources}
    gold = _remap(bench.problem_gold, lambda p: p if p in keep else None)
    tb = om.organization_scores(tax, gold, "taxobench", rt.embedder, titles=bench.corpus)
    ch = om.organization_scores(tax, gold, "chime", rt.embedder)
    for value in (tb["ari"], tb["sem"], ch["cat"], ch["assn"]):
        assert -1.0 <= value <= 1.0
    assert tb["paper_recall"] == 1.0 and 0.0 < tb["sem"] <= 1.0 and 0.0 < ch["cat"] <= 1.0

    # Exported TaxoBench trees (titles) parse back to the same clustering.
    exported = om.to_taxobench_tree(tax, titles=bench.corpus)
    back = parse_hierarchy(exported)
    title = {p.paper_id: p.title for p in papers}
    assert _labels(back) == {title[p]: name for p, name in _labels(tax).items()}
    empty = organize(rt.llm, "Nothing", "", [])
    assert len(empty) == 1 and not empty.all_papers()
