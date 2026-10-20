"""Tests for dynamic updates of a certified taxonomy (Section 3.4, App. B.3, Eq. 16)."""

from __future__ import annotations

import copy
import dataclasses
import json
import math

import numpy as np
import pytest

from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.embeddings import l2_normalize
from scholarpath.runtime import Runtime
from scholarpath.taxonomy import CERTIFIED, Taxonomy
from scholarpath.taxonomy.builder import STATE_KEY, NodeWiseBuilder
from scholarpath.taxonomy.dynamic import DynamicUpdater, absorbable_arrivals, updated_bound
from scholarpath.taxonomy.result import BuildResult, EpisodeRecord

from test_taxonomy_builder import check_episodes, fast_config, single_thread


@pytest.fixture(scope="module")
def built():
    bench = make_synthetic_benchmark(n_papers=200, seed=0)
    cfg = fast_config(max_depth=2)
    rt = Runtime.from_config(cfg)
    builder = NodeWiseBuilder(rt.llm, rt.embedder, cfg.builder, bench.corpus)
    with single_thread():
        result = builder.build("Root", "All research problems.", facet="problem")
    assert result.taxonomy.root.status == CERTIFIED
    return cfg, rt, builder, result, bench


def _fresh_copy(built):
    """An independent (builder, result) pair so each test mutates its own taxonomy."""
    cfg, rt, builder, result, bench = built
    result = copy.deepcopy(result)
    b2 = NodeWiseBuilder(rt.llm, rt.embedder, cfg.builder, dict(builder.papers),
                         paper_vectors=builder.paper_vectors, prior=copy.deepcopy(builder.prior))
    return b2, result


def _clean_route(builder, result, pid):
    """True if the frozen routers place ``pid`` into some child at every internal node on its way down."""
    tax = result.taxonomy
    frontier = [tax.root_id]
    vec = builder.vectors([pid])
    while frontier:
        nid = frontier.pop()
        router = result.routers.get(nid)
        if router is None or not tax.children(nid):
            continue
        kids = router.route(vec)[0]
        if not kids:
            return False
        frontier.extend(kids)
    return True


def _duplicates(papers, prefix):
    return [dataclasses.replace(p, paper_id=f"{prefix}{i:04d}", chunks=[]) for i, p in enumerate(papers)]


def _check_tree(result, epsilon):
    """Partition invariant, valid certificates for the current node sizes, consistent episodes."""
    tax = result.taxonomy
    for nid, node in tax.nodes.items():
        if node.children:
            union = set().union(*(tax.nodes[c].papers for c in node.children))
            assert node.papers == union | node.residual, nid
            assert nid in result.routers, nid
        if node.status == CERTIFIED:
            assert node.bound <= epsilon + 1e-12
            assert node.metadata[STATE_KEY]["certificate"]["n_v"] == node.size
    check_episodes(result)


def _slack_certificates(tax, bound=0.02):
    """Give every certificate slack (as for nodes certified at U_old = 0.02).

    Eq. (16) only depends on (n, U_old). The audit certifies at the first a
    with U <= eps, so real certificates sit just below eps and a node
    certified there cannot absorb any arrival.
    """
    for node in tax.nodes.values():
        if node.status == CERTIFIED:
            node.metadata[STATE_KEY]["certificate"]["bound"] = bound
            node.bound = bound


def test_eq16_arithmetic():
    assert updated_bound(1000, 0.06, 44) <= 0.10 < updated_bound(1000, 0.06, 45)
    assert absorbable_arrivals(1000, 0.06, 0.10) == 44  # example of Section 3.4
    assert absorbable_arrivals(100, 0.12, 0.10) == 0
    assert updated_bound(50, 0.1, 0) == pytest.approx(0.1)


def test_small_arrival_batch_keeps_certificates_at_zero_llm_cost(built):
    cfg, rt, _, _, bench = built
    builder, result = _fresh_copy(built)
    tax = result.taxonomy
    eps = cfg.builder.epsilon
    _slack_certificates(tax)
    n_old = tax.root.size
    # Arrivals: copies (new ids) of papers that the frozen routers place cleanly.
    clean = [p for pid, p in sorted(bench.corpus.papers.items()) if _clean_route(builder, result, pid)][:3]
    assert len(clean) == 3
    M = min(len(clean), absorbable_arrivals(n_old, 0.02, eps))
    arrivals = _duplicates(clean[:M], "dup")
    calls_before = rt.meter.total().calls
    report = DynamicUpdater(builder, result).add_papers(arrivals)
    assert report.llm_calls == 0 and rt.meter.total().calls == calls_before
    assert tax.root_id in report.kept and not report.reaudited and not report.repaired
    assert report.n_new == M and report.arrivals[tax.root_id] == M
    root_cert = tax.root.metadata[STATE_KEY]["certificate"]
    assert root_cert["n_v"] == n_old + M == tax.root.size
    assert tax.root.bound == pytest.approx(updated_bound(n_old, 0.02, M)) and tax.root.bound <= eps
    for a in arrivals:
        assert a.paper_id in tax.root.papers and len(tax.paper_nodes(a.paper_id)) >= 2
    _check_tree(result, eps)
    # Re-adding the same papers is a no-op.
    again = DynamicUpdater(builder, result).add_papers(arrivals)
    assert again.n_new == 0 and len(again.skipped) == M and again.llm_calls == 0


def test_large_arrival_batch_triggers_audit_and_recertification(built):
    cfg, rt, _, _, _ = built
    builder, result = _fresh_copy(built)
    tax = result.taxonomy
    eps = cfg.builder.epsilon
    root_meta = tax.root.metadata[STATE_KEY]
    j_old, delta_root, n_old = root_meta["next_snapshot"], tax.root.delta, tax.root.size
    new = make_synthetic_benchmark(n_papers=120, seed=11).corpus
    arrivals = _duplicates([new.papers[p] for p in sorted(new.papers)], "new")
    assert updated_bound(n_old, tax.root.bound, len(arrivals)) > eps
    with single_thread():
        report = DynamicUpdater(builder, result).add_papers(arrivals)
    assert tax.root_id in report.reaudited
    assert report.llm_calls > 0 and "auditing" in report.usage and report.papers_read.get("auditing", 0) > 0
    assert tax.root.size == n_old + len(arrivals)
    # The snapshot index continues (alpha_{v,j,a} keeps summing to <= delta_v); delta_v is unchanged.
    assert tax.root.metadata[STATE_KEY]["next_snapshot"] > j_old and tax.root.delta == delta_root
    # Every internal node still partitions its papers into children and the residual leaf,
    # and every certificate still satisfies U <= eps with n = current node size.
    for nid, node in tax.nodes.items():
        if node.children:
            union = set().union(*(tax.nodes[c].papers for c in node.children))
            assert node.papers == union | node.residual
        if node.status == CERTIFIED:
            assert node.bound <= eps + 1e-12
            assert node.metadata[STATE_KEY]["certificate"]["n_v"] == node.size
    if tax.root_id in report.recertified:
        assert tax.root.status == CERTIFIED
    else:
        assert tax.root_id in report.failed
    assert result.metadata["updates"][-1]["n_new"] == len(arrivals)
    # Episodes run by the update replace the superseded ones in result.episodes.
    _check_tree(result, eps)
    for ep in report.episodes:
        if ep.node_id in tax.nodes:
            assert any(e is ep for e in result.episodes)
    if tax.root_id in report.repaired:
        root_ep = next(e for e in result.episodes if e.node_id == tax.root_id)
        assert root_ep.metadata["dynamic_update"] and "bound" in root_ep.metadata["trigger"]
        assert any(e["node_id"] == tax.root_id for e in result.metadata["superseded_episodes"])


def test_residual_crossing_discovery_threshold_triggers_real_audit_and_repair(built):
    """U_new <= eps, but the arrivals grow a residual leaf at or above the discovery threshold:
    the cohort is audited with real draws, and the errors it confirms trigger DISCOVER (Sec. 3.4)."""
    cfg, rt, _, _, _ = built
    builder, result = _fresh_copy(built)
    tax = result.taxonomy
    eps = cfg.builder.epsilon
    _slack_certificates(tax)
    root, n_old = tax.root, tax.root.size
    # Arrivals that the frozen root router places in the residual leaf bottom_v.
    pool = make_synthetic_benchmark(n_papers=400, seed=7).corpus
    cands = [pool.papers[p] for p in sorted(pool.papers)]
    vecs = l2_normalize(np.asarray(rt.embedder.encode([p.text for p in cands]), dtype=np.float32))
    residual_cands = [p for p, r in zip(cands, result.routers[tax.root_id].route(vecs)) if not r]
    M = max(1, int(math.ceil(builder.repair_threshold(n_old + 8) - len(root.residual))))
    M = min(M, cfg.builder.audit_min_draws)
    assert len(residual_cands) >= M
    arrivals = _duplicates(residual_cands[:M], "res")
    # Preconditions: the bound alone would keep the certificate; the residual condition fails.
    assert updated_bound(n_old, 0.02, M) <= eps
    assert len(root.residual) + M >= builder.repair_threshold(n_old + M)
    with single_thread():
        report = DynamicUpdater(builder, result).add_papers(arrivals)
    assert tax.root_id not in report.kept
    assert tax.root_id in report.reaudited and tax.root_id in report.repaired
    assert report.llm_calls > 0 and report.papers_read.get("auditing", 0) >= M
    assert "discovery" in report.usage  # Stage 2 DISCOVER on the residual pool
    root_ep = next(e for e in result.episodes if e.node_id == tax.root_id)
    cohort = root_ep.metadata["cohort_audit"]
    assert root_ep.metadata["trigger"] == ["residual"]
    assert cohort["draws"] == M and cohort["errors"] == M  # residual-routed arrivals are errors
    assert tax.root.size == n_old + M
    _check_tree(result, eps)


def test_update_after_save_and_reload_rebuilds_routers(built, tmp_path):
    """BuildResult.save drops routers: they are rebuilt from the persisted labels (zero LLM cost)."""
    cfg, rt, builder0, result0, bench = built
    eps = cfg.builder.epsilon
    result0.save(tmp_path / "build")
    summary = json.loads((tmp_path / "build" / "summary.json").read_text())
    episodes = [
        EpisodeRecord(**{k: v for k, v in json.loads(line).items() if k != "ratio"})
        for line in (tmp_path / "build" / "episodes.jsonl").read_text().splitlines()
    ]
    loaded = BuildResult(taxonomy=Taxonomy.load(tmp_path / "build" / "taxonomy.json"), method=result0.method,
                         episodes=episodes, metadata=summary["metadata"])
    internal = sorted(nid for nid, node in loaded.taxonomy.nodes.items() if node.children)
    assert internal and not loaded.routers
    corpus = bench.corpus
    n_corpus = len(corpus)
    builder = NodeWiseBuilder(rt.llm, rt.embedder, cfg.builder, corpus, paper_vectors=builder0.paper_vectors)
    with pytest.raises(ValueError, match="rebuild_routers"):
        DynamicUpdater(builder, loaded, rebuild_routers=False)
    new = make_synthetic_benchmark(n_papers=30, seed=9).corpus
    arrivals = _duplicates([new.papers[p] for p in sorted(new.papers)], "reload")
    calls = rt.meter.total().calls
    updater = DynamicUpdater(builder, loaded)
    assert rt.meter.total().calls == calls  # rebuilding routers costs no LLM call
    with single_thread():
        report = updater.add_papers(arrivals)
    assert report.rebuilt_routers == internal and set(internal) <= set(loaded.routers)
    tax = loaded.taxonomy
    assert tax.root.size == len(result0.taxonomy.root.papers) + len(arrivals)
    for a in arrivals:
        assert a.paper_id in tax.root.papers
    _check_tree(loaded, eps)
    # The caller's Corpus is not mutated by the update.
    assert len(corpus) == n_corpus and all(a.paper_id not in corpus for a in arrivals)
