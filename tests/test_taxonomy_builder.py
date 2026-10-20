"""End-to-end tests of the node-wise confidence-bounded builder (Algorithm 1) and of the repair helpers."""

from __future__ import annotations

import contextlib
import json
import math

import numpy as np
import pytest

from scholarpath.config import RouterConfig, ScholarPathConfig
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.embeddings import l2_normalize
from scholarpath.llm import BUILD_COMPONENTS
from scholarpath.runtime import Runtime
from scholarpath.taxonomy import CERTIFIED, LEAF, OPEN, UNRESOLVED, Taxonomy
from scholarpath.taxonomy.builder import STATE_KEY, NodeWiseBuilder, build_dual_taxonomies
from scholarpath.taxonomy.certification import DeltaAllocator
from scholarpath.taxonomy.repair import (
    correction_targets,
    discovery_draws,
    discovery_sample_size,
    relabel_candidates,
    stratified_allocation,
)
from scholarpath.taxonomy.response_model import RegionDescriptor, TransferredPrior


@contextlib.contextmanager
def single_thread():
    """Avoid BLAS over-subscription when several test processes share the CPU."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover
        yield
        return
    with threadpool_limits(limits=1):
        yield


def fast_config(**builder_overrides) -> ScholarPathConfig:
    """Offline config with a small router that learns quickly on the mock labels."""
    cfg = ScholarPathConfig()
    cfg.embedding.dim = 256
    cfg.builder.router = RouterConfig(epochs=30, hidden_dim=64, learning_rate=1e-2, batch_size=32)
    for k, v in builder_overrides.items():
        setattr(cfg.builder, k, v)
    return cfg


@pytest.fixture(scope="module")
def bench():
    return make_synthetic_benchmark(n_papers=200, seed=0)


@pytest.fixture(scope="module")
def dual(bench):
    cfg = fast_config(max_depth=2)
    rt = Runtime.from_config(cfg)
    with single_thread():
        problem, method = build_dual_taxonomies(rt.llm, rt.embedder, cfg.builder, bench.corpus)
    return cfg, rt, problem, method


def _check_invariants(result, epsilon, delta, n_papers):
    tax: Taxonomy = result.taxonomy
    root = tax.root
    assert root.size == n_papers
    assert len(tax.children(tax.root_id)) >= 2 and len(tax) >= 3
    node_deltas = []
    for nid, node in tax.nodes.items():
        assert node.status in (OPEN, CERTIFIED, UNRESOLVED, LEAF)
        if node.children:
            union = set().union(*(tax.nodes[c].papers for c in node.children))
            # Every paper of an internal node is assigned to a child or to its residual leaf.
            assert node.papers == union | node.residual
            assert not (union & node.residual)
            assert all(tax.nodes[c].papers for c in node.children)  # no empty child
            assert nid in result.routers
        if node.status == CERTIFIED:
            assert node.bound is not None and node.bound <= epsilon + 1e-12
            cert = node.metadata[STATE_KEY]["certificate"]
            assert cert["certified"] and cert["n_v"] == node.size
        if node.delta is not None:
            node_deltas.append(node.delta)
    assert math.fsum(node_deltas) <= delta
    assert result.metadata["delta_spent"] <= delta
    check_episodes(result)


def check_episodes(result):
    """At most one (the latest) episode per node of the taxonomy; the summary matches the tree."""
    tax = result.taxonomy
    ids = [e.node_id for e in result.episodes]
    assert len(ids) == len(set(ids)) and set(ids) <= set(tax.nodes)
    certified = {nid for nid, node in tax.nodes.items() if node.status == CERTIFIED}
    assert {e.node_id for e in result.episodes if e.certified} == certified
    assert result.summary()["certified_nodes"] == len(certified)


def test_dual_build_end_to_end(dual, bench):
    cfg, rt, problem, method = dual
    for result, facet in ((problem, "problem"), (method, "method")):
        assert result.taxonomy.facet == facet
        _check_invariants(result, cfg.builder.epsilon, cfg.builder.delta, len(bench.corpus))
        assert result.episodes and result.episodes[0].node_id == result.taxonomy.root_id
        assert any(e.certified for e in result.episodes)
        assert "acquisition" in result.usage and set(result.usage) <= set(BUILD_COMPONENTS)
        assert set(result.papers_read) <= set(BUILD_COMPONENTS)
        assert result.papers_read["acquisition"] > 0 and result.cost_usd > 0 and result.runtime_s > 0
        for e in result.episodes:
            assert 0 < e.verified <= e.n_v and 0.0 <= e.abstention <= 1.0 and 0.0 <= e.disagreement <= 1.0
            assert e.metadata["termination"] in ("certified", "max_cycles", "budget", "no_schema")
            if e.metadata["termination"] == "no_schema":  # seed read, no usable split: a leaf
                assert result.taxonomy.nodes[e.node_id].status == LEAF
    # Facet-specific taxonomies over the same papers.
    assert problem.taxonomy.pretty() != method.taxonomy.pretty()
    # Usage is per build: both builds together account for every call of the meter.
    assert problem.total_usage.calls + method.total_usage.calls == rt.meter.total().calls


def test_certified_nodes_persist_state_for_dynamic_updates(dual):
    _, _, problem, _ = dual
    tax = problem.taxonomy
    certified = [n for n in tax.nodes.values() if n.status == CERTIFIED]
    assert certified
    for node in certified:
        meta = node.metadata[STATE_KEY]
        assert meta["next_snapshot"] >= 2 and meta["n"] == node.size
        assert set(meta["verified"]) <= node.papers
        assert set(meta["audited"]) <= set(meta["verified"])
        assert all(set(kids) <= set(node.children) for kids in meta["verified"].values())
        assert node.delta and node.delta > 0
        # Verified papers keep their LLM labels as assignments.
        for pid, kids in meta["verified"].items():
            placed = {c for c in node.children if pid in tax.nodes[c].papers}
            assert placed == set(kids)
            assert (pid in node.residual) == (not kids)


def _paper_vectors(rt, corpus, ids):
    """Paper embeddings exactly as the builder computes them (L2-normalised title + abstract)."""
    return l2_normalize(np.asarray(rt.embedder.encode([corpus[p].text for p in ids]), dtype=np.float32))


def test_routers_and_save(tmp_path, dual, bench):
    _, rt, problem, method = dual
    checked = 0
    for result in (problem, method):
        tax = result.taxonomy
        for nid, nr in result.routers.items():
            node = tax.nodes[nid]
            assert set(nr.child_ids) == set(node.children)
            ids = sorted(node.papers)
            routes = nr.route(_paper_vectors(rt, bench.corpus, ids))
            verified = node.metadata[STATE_KEY]["verified"]
            agree = 0
            for pid, route in zip(ids, routes):
                placed = sorted(c for c in node.children if pid in tax.nodes[c].papers)
                if pid in verified:  # verified papers keep their LLM labels
                    agree += set(route) == set(verified[pid])
                else:  # the frozen router reproduces the stored assignment of every unverified paper
                    assert sorted(route) == placed and (not route) == (pid in node.residual)
                    checked += 1
            if nid == tax.root_id:  # trained on these labels: mostly agrees with them
                assert agree >= 0.7 * len(verified)
    assert checked > 0
    tax = problem.taxonomy
    problem.save(tmp_path / "problem")
    loaded = Taxonomy.load(tmp_path / "problem" / "taxonomy.json")
    assert loaded.to_dict() == tax.to_dict()
    summary = json.loads((tmp_path / "problem" / "summary.json").read_text())
    assert summary["certified_nodes"] == sum(e.certified for e in problem.episodes)
    assert (tmp_path / "problem" / "episodes.jsonl").read_text().count("\n") == len(problem.episodes)


# ------------------------------------------------------------------ ablations

# Every ablation of Table 6 runs and keeps the invariants. The effect of use_transferred_prior
# (only visible with a non-empty history H) is checked by test_transferred_prior_flag_sets_region_priors.
ABLATIONS = [
    {"use_bayesian_acquisition": False},
    {"use_snapshot_certification": False},
    {"use_transferred_prior": False},
    {"use_discovery": False},
    {"use_correction": False},
    {"use_targeted_repair": False},
]


@pytest.fixture(scope="module")
def small_bench():
    return make_synthetic_benchmark(n_papers=160, seed=3)


def _root_only(small_bench, **overrides):
    cfg = fast_config(max_depth=1, **overrides)
    rt = Runtime.from_config(cfg)
    builder = NodeWiseBuilder(rt.llm, rt.embedder, cfg.builder, small_bench.corpus)
    with single_thread():
        result = builder.build("Root", "All problems.", facet="problem")
    return cfg, builder, result


@pytest.mark.parametrize("flags", ABLATIONS, ids=lambda f: next(iter(f)))
def test_ablation_flags_run(small_bench, flags):
    cfg, builder, result = _root_only(small_bench, **flags)
    _check_invariants(result, cfg.builder.epsilon, cfg.builder.delta, len(small_bench.corpus))
    root = result.taxonomy.root
    ep = result.episodes[0]
    if flags.get("use_snapshot_certification") is False:
        assert "auditing" not in result.usage
        assert root.status == OPEN and ep.metadata["termination"] == "fixed_ratio"
        assert ep.verified >= math.ceil(cfg.builder.fixed_ratio * root.size)
    if flags.get("use_bayesian_acquisition") is False:
        assert all("forecast" not in r for r in ep.metadata["rounds_log"])
    if flags.get("use_targeted_repair") is False:
        assert "discovery" not in result.usage and "correction" not in result.usage
    if flags.get("use_discovery") is False:
        assert "discovery" not in result.usage
    if flags.get("use_correction") is False:
        assert "correction" not in result.usage


@pytest.mark.parametrize("use_prior", [True, False])
def test_transferred_prior_flag_sets_region_priors(small_bench, use_prior):
    """use_transferred_prior=False (Table 6) ignores a non-empty history H; True uses Eq. (14)."""
    cfg = fast_config(max_depth=1, use_transferred_prior=use_prior)
    b = cfg.builder
    prior = TransferredPrior(b.prior_slope_mean, b.prior_slope_var)  # enabled: only the builder flag differs
    # History of earlier episodes whose regions responded much faster than the fixed prior.
    rng = np.random.default_rng(0)
    descs = [RegionDescriptor(depth=0, share=float(rng.uniform(0.1, 0.3)), dispersion=float(rng.uniform(0.3, 0.7)),
                              child_count=4, papers_read=float(rng.integers(2, 10)),
                              abstention=float(rng.uniform(0, 0.3)), disagreement=float(rng.uniform(0, 0.3)))
             for _ in range(12)]
    prior.update(descs, [[0.6 + 0.01 * k, 0.4] for k in range(12)])
    calls = []
    predict = prior.predict

    def spy(h):
        out = predict(h)
        calls.append(out)
        return out

    prior.predict = spy
    rt = Runtime.from_config(cfg)
    builder = NodeWiseBuilder(rt.llm, rt.embedder, b, small_bench.corpus, prior=prior)
    ids = sorted(small_bench.corpus.papers)
    tax = Taxonomy(facet="problem", root_name="Root", root_definition="", root_papers=ids)
    allocator = DeltaAllocator(b.delta, b.delta_allocation, len(ids), b.max_depth, b.max_nodes)
    with single_thread():
        state = builder.init_node(tax, tax.root_id, builder.make_ops("problem"), allocator)
    assert state.responses and len(state.responses) == state.regions.G
    fixed_mean, fixed_cov = prior.fixed_prior()
    if use_prior:
        assert len(calls) == len(state.responses)
        for resp, (mean, _) in zip(state.responses, calls):
            np.testing.assert_allclose(resp.mean, mean)
            assert resp.mean.min() > 0.3  # transferred from H, far from the fixed prior mean 0.05
    else:
        assert not calls
        for resp in state.responses:
            np.testing.assert_allclose(resp.mean, fixed_mean)
            np.testing.assert_allclose(resp.cov, fixed_cov, atol=1e-9)


def test_all_cost_components_when_repair_pools_reach_threshold(small_bench):
    """A weak router leaves disagreements and residual papers: DISCOVER and CORRECT both run (Eq. 15)."""
    cfg = fast_config(max_depth=1, repair_pool_floor=1, repair_pool_fraction=0.0, max_cycles=3)
    cfg.builder.router = RouterConfig(epochs=3, hidden_dim=16, learning_rate=1e-2, batch_size=32)
    rt = Runtime.from_config(cfg)
    builder = NodeWiseBuilder(rt.llm, rt.embedder, cfg.builder, small_bench.corpus)
    with single_thread():
        result = builder.build("Root", "", facet="problem")
    _check_invariants(result, cfg.builder.epsilon, cfg.builder.delta, len(small_bench.corpus))
    assert set(BUILD_COMPONENTS) <= set(result.usage)
    ep = result.episodes[0]
    # CORRECT labels at most kappa unread neighbours per disagreeing (held-out) paper.
    assert ep.papers_read["correction"] <= cfg.builder.kappa * ep.verified
    assert result.cost_breakdown()["correction"] > 0


def test_budget_cap_returns_unresolved_and_is_respected(small_bench):
    cfg, builder, result = _root_only(small_bench, max_llm_papers_per_node=20)
    ep = result.episodes[0]
    assert sum(ep.papers_read.values()) <= 20
    assert result.taxonomy.root.status in (UNRESOLVED, CERTIFIED)
    if result.taxonomy.root.status == UNRESOLVED:
        assert ep.metadata["termination"] == "budget" and result.taxonomy.root.bound is None


def test_build_is_deterministic_and_feeds_prior(small_bench):
    cfg = fast_config(max_depth=1)
    runs, priors = [], []
    for _ in range(2):
        rt = Runtime.from_config(cfg)
        prior = TransferredPrior(cfg.builder.prior_slope_mean, cfg.builder.prior_slope_var)
        builder = NodeWiseBuilder(rt.llm, rt.embedder, cfg.builder, small_bench.corpus, prior=prior)
        with single_thread():
            runs.append(builder.build("Root", "", facet="problem"))
        priors.append(prior)
    assert runs[0].taxonomy.to_dict() == runs[1].taxonomy.to_dict()
    # Completed episodes with observed responses are added to the history H.
    observed = sum(len(e.region_slopes) for e in runs[0].episodes)
    assert priors[0].n_observations == observed == priors[1].n_observations


def test_build_on_subset_and_unknown_ids(small_bench):
    cfg = fast_config(max_depth=1, min_node_size=1000)
    rt = Runtime.from_config(cfg)
    builder = NodeWiseBuilder(rt.llm, rt.embedder, cfg.builder, small_bench.corpus)
    ids = sorted(small_bench.corpus.papers)[:50]
    res = builder.build("Root", paper_ids=ids, facet="problem")
    assert res.taxonomy.root.papers == set(ids) and res.taxonomy.root.status == LEAF and not res.episodes
    assert rt.meter.total().calls == 0
    # The builder keeps a private copy of the repository (dynamic updates add papers to it).
    assert builder.papers is not small_bench.corpus.papers and builder.papers == small_bench.corpus.papers
    with pytest.raises(KeyError):
        builder.build("Root", paper_ids=["nope"])


# ------------------------------------------------------------ repair helpers


def test_discovery_draws_proportional_stratified():
    rng = np.random.default_rng(0)
    assert discovery_sample_size(0, 0.02, 32) == 0
    assert discovery_sample_size(10, 0.02, 32) == 1
    assert discovery_sample_size(10_000, 0.02, 32) == 32
    # Exactly min{ceil(eta_A |pool|), s_max}: s_max = 0 or eta_A = 0 switches DISCOVER sampling off.
    assert discovery_sample_size(50, 0.02, 0) == 0 and discovery_sample_size(50, 0.0, 32) == 0
    assert discovery_draws({0: ["a", "b", "c"]}, 0.02, 0, rng) == []
    assert stratified_allocation([50, 30, 20], 10) == [5, 3, 2]
    assert stratified_allocation([1, 1, 98], 5) == [1, 1, 5]  # rounding up adds at most G - 1
    pool = {0: [f"a{i}" for i in range(500)], 1: [f"b{i}" for i in range(300)], 2: [f"c{i}" for i in range(200)]}
    draws = discovery_draws(pool, eta_A=0.02, s_max=32, rng=rng)
    s_disc = 20
    assert s_disc <= len(draws) <= s_disc + len(pool) - 1 and len(set(draws)) == len(draws)
    counts = [sum(d.startswith(p) for d in draws) for p in "abc"]
    assert counts == [10, 6, 4]
    assert len(discovery_draws(pool, 0.02, 32, rng, max_draws=5)) == 5


def test_discovery_sample_detects_missing_concept_prop2():
    """A concept held by a fraction rho of the pool is missed w.p. <= (1 - rho)^s."""
    rng = np.random.default_rng(1)
    pool = {g: [f"g{g}p{i}" for i in range(100)] for g in range(4)}
    concept = {p for g in pool for p in pool[g][:5]}  # rho = 0.05
    s_disc, trials = discovery_sample_size(400, 0.02, 32), 2000
    missed = sum(not (set(discovery_draws(pool, 0.02, 32, rng)) & concept) for _ in range(trials))
    assert missed / trials <= (1 - 0.05) ** s_disc + 0.03


def test_correction_targets_nearest_unread_same_region():
    vecs = {
        "d": np.array([1.0, 0.0]), "n1": np.array([0.99, 0.14]), "n2": np.array([0.9, 0.44]),
        "n3": np.array([0.0, 1.0]), "far": np.array([0.995, 0.1]), "read": np.array([1.0, 0.0]),
    }
    vecs = {k: v / np.linalg.norm(v) for k, v in vecs.items()}
    region = {"d": 0, "n1": 0, "n2": 0, "n3": 0, "far": 1, "read": 0}
    out = correction_targets(["d"], region, vecs, unread=["n1", "n2", "n3", "far"], kappa=2)
    assert out == ["n1", "n2"]  # nearest two, same region, unread only
    assert correction_targets(["d"], region, vecs, ["n1"], kappa=0) == []


def test_relabel_candidates():
    verified = {"m": ((), True), "r": (("c1",), False), "near": (("c2",), False), "far": (("c2",), False)}
    child_vecs = {"c1": np.array([1.0, 0.0]), "c2": np.array([0.0, 1.0]), "new": np.array([0.7071, 0.7071])}
    paper_vecs = {"near": np.array([0.6, 0.8]), "far": np.array([0.0, 1.0])}
    out = relabel_candidates(verified, ["new"], ["c1"], paper_vecs, child_vecs)
    assert out == ["m", "near", "r"]
    assert relabel_candidates(verified, [], [], paper_vecs, child_vecs, exclude=["m"]) == []
