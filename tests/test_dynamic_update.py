"""Tests for dynamic updates of a certified taxonomy (Section 3.4, App. B.3, Eq. 16)."""

from __future__ import annotations

import copy
import dataclasses

import pytest

from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.runtime import Runtime
from scholarpath.taxonomy import CERTIFIED
from scholarpath.taxonomy.builder import STATE_KEY, NodeWiseBuilder
from scholarpath.taxonomy.dynamic import DynamicUpdater, absorbable_arrivals, updated_bound

from test_taxonomy_builder import fast_config, single_thread


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
    # Give every certificate slack (as for a node certified at U_old = 0.02): Eq. (16) only
    # depends on (n, U_old), and a node certified exactly at eps cannot absorb any arrival.
    for node in tax.nodes.values():
        if node.status == CERTIFIED:
            node.metadata[STATE_KEY]["certificate"]["bound"] = 0.02
            node.bound = 0.02
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
