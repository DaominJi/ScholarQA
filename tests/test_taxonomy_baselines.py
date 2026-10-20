"""Tests for the FULL / node-wise / one-off selection baselines (Section 5.2, App. B.2, C.5)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from scholarpath.config import BaselineBuilderConfig, RouterConfig, ScholarPathConfig
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.runtime import Runtime
from scholarpath.taxonomy.baselines import (
    COMPONENT_FULL,
    COMPONENT_SELECTION,
    TERM_NO_SPLIT,
    TERM_SPLIT,
    SelectionBuilder,
    baseline_config,
    build_dual_baselines,
    estimate_full_cost,
    internal_paper_count,
    method_name,
    parse_method,
    per_paper_node_cost,
)
from scholarpath.taxonomy.selectors import num_to_select
from scholarpath.taxonomy.tree import LEAF

# A small router keeps the tests fast; test_default_router_* checks the
# repository's default RouterConfig (hidden 256) separately.
ROUTER = RouterConfig(hidden_dim=32, epochs=200, learning_rate=3e-3)


def _config(selector: str, eta: float = 0.2, mode: str = "node_wise", **kw) -> BaselineBuilderConfig:
    return BaselineBuilderConfig(selector=selector, eta=eta, mode=mode, router=ROUTER, **kw)


@pytest.fixture(scope="module")
def rt():
    cfg = ScholarPathConfig()
    cfg.embedding.dim = 128
    return Runtime.from_config(cfg)


@pytest.fixture(scope="module")
def bench():
    return make_synthetic_benchmark(n_papers=240, seed=0)


@pytest.fixture(scope="module")
def vecs(rt, bench):
    papers = bench.corpus.papers
    return dict(zip(papers, rt.embedder.encode([p.text for p in papers.values()])))


@pytest.fixture(scope="module")
def builds(rt, bench, vecs):
    def run(cfg: BaselineBuilderConfig, **kw):
        return SelectionBuilder(rt.llm, rt.embedder, cfg, bench.corpus, vecs, **kw).build("Synthetic Repository")

    return {
        "full": run(_config("full")),
        "coverage@0.5": run(_config("coverage", 0.5)),
        "random@0.1": run(_config("random", 0.1)),
        "kmeans@0.2": run(_config("kmeans", 0.2)),
        "diversity@0.2": run(_config("diversity", 0.2)),
        "one_off/random@0.2": run(_config("random", 0.2, "one_off")),
    }


# ---------------------------------------------------------------------------
# Names and configuration
# ---------------------------------------------------------------------------


def test_method_names_roundtrip():
    assert method_name("full", 0.3) == "full"
    assert method_name("kmeans", 0.2) == "kmeans@0.2"
    assert method_name("coverage", 0.05, "one_off") == "one_off/coverage@0.05"
    for name in ("full", "kmeans@0.2", "one_off/diversity@0.5", "random@0.05"):
        sel, eta, mode = parse_method(name)
        assert method_name(sel, eta, mode) == name
    cfg = baseline_config("one_off/kmeans@0.3", BaselineBuilderConfig(max_depth=3))
    assert (cfg.selector, cfg.eta, cfg.mode, cfg.max_depth) == ("kmeans", 0.3, "one_off", 3)
    with pytest.raises(ValueError):
        parse_method("kmeans")


def test_invalid_configs_are_rejected(rt, bench):
    for bad in (_config("farthest"), _config("random", 0.0), _config("random", 1.5), _config("random", 0.2, "lazy")):
        with pytest.raises(ValueError):
            SelectionBuilder(rt.llm, rt.embedder, bad, bench.corpus)


def test_method_and_components(builds):
    for name, res in builds.items():
        assert res.method == name
        assert res.taxonomy.metadata["method"] == name
    assert set(builds["full"].usage) == {COMPONENT_FULL}
    assert set(builds["full"].papers_read) == {COMPONENT_FULL}
    for name in ("coverage@0.5", "random@0.1", "one_off/random@0.2"):
        assert set(builds[name].usage) == {COMPONENT_SELECTION}
        assert builds[name].cost_usd > 0


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["full", "coverage@0.5", "random@0.1", "kmeans@0.2", "diversity@0.2", "one_off/random@0.2"]
)
def test_every_paper_is_organized(builds, bench, name):
    res = builds[name]
    tax = res.taxonomy
    cfg = BaselineBuilderConfig()
    assert tax.root.papers == set(bench.corpus.papers)
    for nid, node in tax.nodes.items():
        assert node.depth <= cfg.max_depth
        if node.children:
            # P_v = (union of the children's extents) + residual leaf; nothing is lost.
            union = set().union(*(tax.nodes[c].papers for c in node.children))
            assert union | node.residual == node.papers
            assert not (union & node.residual)
            assert 2 <= len(node.children) <= cfg.max_children  # no single-child chains
            assert all(tax.nodes[c].papers for c in node.children)  # empty children pruned
            assert node.depth < cfg.max_depth and node.size >= cfg.min_node_size
        else:
            assert node.status == LEAF
            assert not node.residual  # the residual leaf only exists below a split
    # Every paper sits in some most-specific concept (possibly the root's residual).
    assert all(tax.deepest_nodes(p) for p in bench.corpus.papers)


def test_episodes_and_routers(builds):
    for name in ("coverage@0.5", "kmeans@0.2", "one_off/random@0.2"):
        res = builds[name]
        tax = res.taxonomy
        internal = set(tax.internal_nodes())
        assert internal <= {e.node_id for e in res.episodes}
        assert set(res.routers) == internal
        for nid, router in res.routers.items():
            assert router.child_ids == tax.children(nid)
            assert router.child_vecs.shape[0] == len(router.child_ids)
        for e in res.episodes:
            assert e.metadata["selected"] == e.verified
            assert e.metadata["termination"] == (TERM_SPLIT if e.node_id in internal else TERM_NO_SPLIT)
            if e.node_id in internal:
                assert e.metadata["routed"] == e.n_v - e.verified
        # A frozen router places new papers into existing children (or the residual).
        root_router = res.routers[tax.root_id]
        routed = root_router.route(np.eye(1, root_router.child_vecs.shape[1], dtype=np.float32))
        assert set(routed[0]) <= set(tax.children(tax.root_id))
    assert builds["full"].routers == {}


# ---------------------------------------------------------------------------
# Reading semantics and cost
# ---------------------------------------------------------------------------


def test_full_reads_every_paper_twice_per_level(builds, bench):
    res = builds["full"]
    n = len(bench.corpus.papers)
    root = res.episodes[0]
    assert root.node_id == res.taxonomy.root_id and root.n_v == n
    assert root.papers_read == {COMPONENT_FULL: 2 * n}  # propose pass + routing pass
    for e in res.episodes:
        assert e.papers_read == {COMPONENT_FULL: 2 * e.n_v}
        assert e.verified == e.n_v and e.metadata["routed"] == 0
    assert res.papers_read[COMPONENT_FULL] == 2 * sum(e.n_v for e in res.episodes)
    # Per level, the internal nodes of FULL read their whole extents twice.
    by_depth = {}
    for e in res.episodes:
        by_depth.setdefault(e.depth, [0, 0])
        by_depth[e.depth][0] += e.papers_read[COMPONENT_FULL]
        by_depth[e.depth][1] += e.n_v
    assert all(reads == 2 * size for reads, size in by_depth.values())


@pytest.mark.parametrize("name,eta", [("coverage@0.5", 0.5), ("random@0.1", 0.1), ("kmeans@0.2", 0.2)])
def test_node_wise_reads_eta_fraction(builds, name, eta):
    res = builds[name]
    total_n = sum(e.n_v for e in res.episodes)
    for e in res.episodes:
        assert e.verified == num_to_select(e.n_v, eta)  # ceil(eta * n_v) papers selected at v
        assert e.papers_read == {COMPONENT_SELECTION: 2 * e.verified}
        assert e.n_v * eta >= 1.0 - 1e-9  # sub-sampling stops when n_v < 1 / eta
    reads = res.papers_read[COMPONENT_SELECTION] / 2
    assert eta * total_n <= reads <= eta * total_n + len(res.episodes)


def test_stop_rule_n_below_inverse_eta(rt, bench, vecs):
    # With min_node_size out of the way, sub-sampling stops exactly when n_v < 1 / eta.
    def episodes(n_papers: int, selector: str, eta: float):
        ids = sorted(bench.corpus.papers)[:n_papers]
        cfg = _config(selector, eta, min_node_size=2, max_depth=1)
        return SelectionBuilder(rt.llm, rt.embedder, cfg, bench.corpus, vecs).build(paper_ids=ids).episodes

    assert episodes(19, "random", 0.05) == []  # 19 < 1 / 0.05: the root is a leaf, nothing is read
    assert episodes(18, "kmeans", 0.05) == []
    assert [e.verified for e in episodes(20, "random", 0.05)] == [1]  # 20 = 1 / 0.05: one paper is read
    assert [e.verified for e in episodes(18, "random", 0.1)] == [2]  # 18 >= 1 / 0.1
    assert [e.verified for e in episodes(18, "full", 1.0)] == [18]  # FULL has no such rule


def test_one_off_samples_once_at_the_root(builds, bench):
    res = builds["one_off/random@0.2"]
    n = len(bench.corpus.papers)
    k = num_to_select(n, 0.2)
    assert res.metadata["one_off_sample"] == k
    assert res.episodes[0].verified == k  # the whole sample is read at the root
    assert res.metadata["distinct_papers_read"] == k  # deeper nodes only re-read sampled papers
    by_id = {e.node_id: e for e in res.episodes}
    tax = res.taxonomy
    for e in res.episodes[1:]:
        parent = by_id[tax.parent(e.node_id)]
        assert e.verified <= min(parent.verified, e.n_v)
        assert e.papers_read == {COMPONENT_SELECTION: 2 * e.verified}
    # Node-wise re-sampling reads more papers than the fixed root sample.
    assert res.metadata["distinct_papers_read"] < builds["kmeans@0.2"].metadata["distinct_papers_read"]


def test_cost_ordering(builds):
    full = builds["full"].cost_usd
    assert full > builds["coverage@0.5"].cost_usd > builds["random@0.1"].cost_usd > 0
    # Relative cost C_LLM / C_Full tracks the selection ratio (Fig. 3).
    assert 0.3 < builds["coverage@0.5"].relative_cost(builds["full"]) < 0.8
    assert builds["random@0.1"].relative_cost(builds["full"]) < 0.3
    assert 0.1 < builds["one_off/random@0.2"].relative_cost(builds["full"]) < 0.35


def test_estimate_full_cost_reproduces_measured_cost(builds):
    full = builds["full"]
    c_bar = per_paper_node_cost(full)
    assert c_bar > 0
    assert estimate_full_cost(full, c_bar) == pytest.approx(full.cost_usd, rel=1e-9)
    assert estimate_full_cost(full.taxonomy, c_bar) == pytest.approx(full.cost_usd, rel=1e-9)
    tokens_bar = per_paper_node_cost(full, measure="tokens")
    assert estimate_full_cost(full, tokens_bar) == pytest.approx(full.total_tokens, rel=1e-9)
    # Normalizing by every processed node (incl. expansions that ended as leaves) gives a lower c_bar.
    assert per_paper_node_cost(full, nodes="processed") <= c_bar
    other = builds["coverage@0.5"]
    assert estimate_full_cost(other, c_bar) == pytest.approx(c_bar * internal_paper_count(other.taxonomy))
    assert other.metadata["internal_paper_count"] == internal_paper_count(other.taxonomy)


def test_default_router_keeps_cost_proportional_to_eta(rt, bench, vecs, builds):
    # Regression: with the default RouterConfig (60 epochs) the routers of the
    # small label sets S_v were trained for ~60 steps, over-assigned P_v \ S_v
    # (Σ n_v 1614 vs 296 for FULL) and pushed C_LLM / C_Full to 0.88 at eta = 0.2.
    cfg = BaselineBuilderConfig(selector="kmeans", eta=0.2)
    assert cfg.router == RouterConfig()
    builder = SelectionBuilder(rt.llm, rt.embedder, cfg, bench.corpus, vecs)
    res = builder.build("Synthetic Repository")
    full = builds["full"]
    assert res.relative_cost(full) < 0.35
    assert internal_paper_count(res.taxonomy) <= 1.25 * internal_paper_count(full.taxonomy)
    assert all(e.disagreement <= 0.25 for e in res.episodes)
    # The floor only raises the epochs of small label sets.
    steps = builder.min_router_steps
    assert builder._router_config(30).epochs == steps  # one batch per epoch
    assert builder._router_config(64 * steps).epochs == cfg.router.epochs
    unfloored = SelectionBuilder(rt.llm, rt.embedder, cfg, bench.corpus, vecs, min_router_steps=0)
    assert unfloored._router_config(30) == cfg.router


class _OneChildLLM(SelectionBuilder):
    """The LLM puts every paper it reads into the first proposed child only."""

    def _induce_and_label(self, ctx, v, read):
        labels = super()._induce_and_label(ctx, v, read)
        assert len(ctx.tax.children(v)) >= 2
        first = ctx.tax.children(v)[:1]
        return {p: list(first) for p in labels}


class _NoMatchLLM(SelectionBuilder):
    """The LLM finds no proposed child matching any paper it reads."""

    def _induce_and_label(self, ctx, v, read):
        return {p: [] for p in super()._induce_and_label(ctx, v, read)}


@pytest.mark.parametrize("selector,eta", [("full", 1.0), ("random", 0.5)])
def test_single_child_split_becomes_a_leaf(rt, bench, vecs, selector, eta):
    cfg = _config(selector, eta, max_depth=2)
    res = _OneChildLLM(rt.llm, rt.embedder, cfg, bench.corpus, vecs).build()
    tax, root = res.taxonomy, res.episodes[0]
    assert len(tax) == 1 and tax.root.status == LEAF and not tax.root.residual
    assert root.metadata["termination"] == TERM_NO_SPLIT and root.metadata["unsplit_children"] == 1
    assert root.n_children == 0 and res.routers == {} and root.verified > 0
    # min_split_children=1 keeps the (degenerate) single-child chain.
    chain = _OneChildLLM(rt.llm, rt.embedder, cfg, bench.corpus, vecs, min_split_children=1).build()
    only = chain.taxonomy.children(chain.taxonomy.root_id)
    assert len(only) == 1 and chain.taxonomy.nodes[only[0]].papers == chain.taxonomy.root.papers


@pytest.mark.parametrize("selector,eta", [("full", 1.0), ("random", 0.5)])
def test_failed_split_leaves_no_residual_on_the_leaf(rt, bench, vecs, selector, eta):
    n = len(bench.corpus.papers)
    for min_split in (1, 2):
        res = _NoMatchLLM(rt.llm, rt.embedder, _config(selector, eta), bench.corpus, vecs,
                          min_split_children=min_split).build()
        tax, root = res.taxonomy, res.episodes[0]
        assert len(tax) == 1 and tax.root.status == LEAF
        assert tax.root.residual == set()  # not P_v: a leaf has no residual leaf
        assert root.abstention == 0.0 and root.metadata["residual"] == 0
        assert root.metadata["termination"] == TERM_NO_SPLIT
        assert root.metadata["unsplit_children"] == 0 and root.metadata["unsplit_residual"] == n


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_deterministic_rebuild(rt, bench, vecs, builds):
    again = SelectionBuilder(rt.llm, rt.embedder, _config("kmeans", 0.2), bench.corpus, vecs).build(
        "Synthetic Repository"
    )
    assert again.taxonomy.to_dict() == builds["kmeans@0.2"].taxonomy.to_dict()
    assert again.cost_usd == pytest.approx(builds["kmeans@0.2"].cost_usd)


def test_paper_subset_facet_and_lazy_embeddings(rt, bench):
    ids = sorted(bench.corpus.papers)[:120]
    cfg = _config("coverage", 0.3, max_depth=1)
    res = SelectionBuilder(rt.llm, rt.embedder, cfg, bench.corpus).build("Methods", paper_ids=ids, facet="method")
    tax = res.taxonomy
    assert tax.facet == "method" and tax.root_id.startswith("M")
    assert tax.root.papers == set(ids)
    assert tax.max_depth <= 1 and len(res.episodes) == 1
    with pytest.raises(KeyError):
        SelectionBuilder(rt.llm, rt.embedder, cfg, bench.corpus).build(paper_ids=["missing-paper"])


def test_full_options(rt, bench, vecs):
    cfg = _config("full", max_depth=2)
    pruned = SelectionBuilder(rt.llm, rt.embedder, cfg, bench.corpus, vecs).build()
    kept = SelectionBuilder(rt.llm, rt.embedder, cfg, bench.corpus, vecs, fit_full_routers=True,
                            prune_empty_children=False).build()
    assert pruned.routers == {} and set(kept.routers) == set(kept.taxonomy.internal_nodes())
    # Pruning only removes children that received no paper (they are never expanded).
    empty = [n for n in kept.taxonomy.nodes.values() if not n.papers]
    assert len(kept.taxonomy) - len(empty) == len(pruned.taxonomy)
    assert sorted(n.size for n in kept.taxonomy.nodes.values() if n.papers) == sorted(
        n.size for n in pruned.taxonomy.nodes.values()
    )
    assert kept.cost_usd == pytest.approx(pruned.cost_usd)


def test_build_dual_baselines(rt, bench, vecs):
    cfg = dataclasses.replace(_config("random", 0.2), max_depth=1)
    problem, method = build_dual_baselines(rt.llm, rt.embedder, cfg, bench.corpus, vecs)
    assert problem.taxonomy.facet == "problem" and method.taxonomy.facet == "method"
    assert problem.taxonomy.root.papers == method.taxonomy.root.papers == set(bench.corpus.papers)
