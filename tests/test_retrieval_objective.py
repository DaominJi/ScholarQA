"""Tests for Step 1 and the objective/scoring of budgeted support-path retrieval (Sections 4.1, 4.2, 4.4)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from scholarpath.config import RetrievalConfig
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.index import SemanticIndex, paper_key
from scholarpath.prompts.retrieval import parse_query_understanding
from scholarpath.retrieval.objective import Coverage, SelectionState, coverage_value, delta
from scholarpath.retrieval.query_understanding import analyze_query
from scholarpath.retrieval.scoring import Granularity, RelevanceCache, bridge_discount, hypothesis_adequacy
from scholarpath.runtime import Runtime
from scholarpath.types import EVIDENCE, HYPOTHESIS_ASSESSMENT, HYPOTHESIS_DISCOVERY, Query


@pytest.fixture(scope="module")
def rt():
    return Runtime.from_config()


# ---------------------------------------------------------------------------
# Eq. (5)-(6), Proposition 5
# ---------------------------------------------------------------------------


def test_delta_conventions():
    assert delta(0.0, 0) == 0.0  # zero gain -> 0 even at zero cost
    assert delta(0.0, 10) == 0.0
    assert delta(0.5, 0) == math.inf  # all nodes already selected: taken at no cost
    assert delta(1.0, 4) == pytest.approx(0.25)


def test_eq6_shared_cost_and_gain_arithmetic():
    costs = {"concept:u1": 5, "concept:u3": 7, "paper:p2": 40, "chunk:p2#1": 30, "paper:p3": 35}
    st = SelectionState(budget=100, cost=costs.__getitem__)
    e = ("concept:u1", "concept:u3", "paper:p2")
    e_plus = e + ("chunk:p2#1",)
    cov_e = Coverage((("emp|p2", 0.5, 0.6), ("mech|p2", 0.5, 0.3)))
    cov_plus = Coverage((("emp|p2", 0.5, 0.65), ("mech|p2", 0.5, 0.9)))
    # Before selection: the chunk path pays for every node on it.
    assert st.marginal_cost(e_plus) == 82
    assert st.gain(cov_plus) == pytest.approx(0.5 * 0.65 + 0.5 * 0.9)
    st.apply(e, cov_e)
    assert st.used == 52 and st.remaining == 48
    assert st.value() == pytest.approx(0.5 * 0.6 + 0.5 * 0.3)
    # Extending the selected paper path costs only the chunk tokens and gains only the upgrade.
    assert st.marginal_cost(e_plus) == 30
    gain = st.gain(cov_plus)
    assert gain == pytest.approx(0.5 * (0.65 - 0.6) + 0.5 * (0.9 - 0.3))
    assert delta(gain, st.marginal_cost(e_plus)) == pytest.approx(gain / 30)
    # A redundant unit (r <= f) gains nothing.
    assert st.gain(Coverage((("emp|p2", 0.5, 0.55),))) == 0.0
    # Shared nodes are counted once in C(S).
    st.apply(("concept:u1", "paper:p3"), Coverage((("emp|p3", 0.5, 0.4),)))
    assert st.used == 52 + 35


def _random_instance(rng: random.Random, n_units=7, n_aspects=5, n_nodes=9):
    aspects = [f"a{i}" for i in range(n_aspects)]
    weights = {a: rng.random() + 0.1 for a in aspects}
    units = []
    for _ in range(n_units):
        covered = rng.sample(aspects, rng.randint(1, n_aspects))
        cov = Coverage(tuple((a, weights[a], rng.random()) for a in covered))
        nodes = tuple(rng.sample([f"n{j}" for j in range(n_nodes)], rng.randint(1, 4)))
        units.append((nodes, cov))
    costs = {f"n{j}": rng.randint(1, 20) for j in range(n_nodes)}
    return units, costs


@pytest.mark.parametrize("seed", range(6))
def test_coverage_is_monotone_submodular_and_matches_eq6(seed):
    rng = random.Random(seed)
    units, costs = _random_instance(rng)
    idx = list(range(len(units)))
    F = lambda S: coverage_value(units[i][1] for i in S)  # noqa: E731
    C = lambda S: sum(costs[x] for x in {x for i in S for x in units[i][0]})  # noqa: E731
    for _ in range(25):
        B = set(rng.sample(idx, rng.randint(0, len(idx) - 1)))
        A = set(rng.sample(sorted(B), rng.randint(0, len(B)))) if B else set()
        z = rng.choice([i for i in idx if i not in B])
        gain_A = F(A | {z}) - F(A)
        gain_B = F(B | {z}) - F(B)
        assert gain_A >= -1e-12 and gain_B >= -1e-12  # monotone
        assert gain_A >= gain_B - 1e-12  # submodular (A subset of B)
        st = SelectionState(10 ** 6, costs.__getitem__)
        for i in sorted(B):
            st.apply(units[i][0], units[i][1])
        assert st.value() == pytest.approx(F(B))
        assert st.gain(units[z][1]) == pytest.approx(gain_B)  # numerator of Eq. (6)
        assert st.marginal_cost(units[z][0]) == C(B | {z}) - C(B)  # denominator of Eq. (6)


# ---------------------------------------------------------------------------
# Eq. (9) and Appendix B.4: premise bottleneck
# ---------------------------------------------------------------------------


def test_premise_bottleneck_directs_expansion_to_weaker_premise():
    eta = 0.7
    st = SelectionState(10 ** 6, lambda x: 1)
    base = hypothesis_adequacy(eta, 0.9, 0.4)  # p4 abstract strong, p1 abstract weak
    st.apply(("b",), Coverage((("h|m3", 1.0, base),)))
    stronger = hypothesis_adequacy(eta, 0.95, 0.4)  # extend e_P: min unchanged
    weaker = hypothesis_adequacy(eta, 0.9, 0.8)  # extend the limiting e_M
    assert st.gain(Coverage((("h|m3", 1.0, stronger),))) == 0.0
    assert weaker == pytest.approx(2 * base)  # "doubles r_hat_{h,b}"
    assert st.gain(Coverage((("h|m3", 1.0, weaker),))) == pytest.approx(base)
    # Tie at the minimum: single-side extensions gain nothing, the joint extension does.
    st2 = SelectionState(10 ** 6, lambda x: 1)
    st2.apply(("b",), Coverage((("h", 1.0, hypothesis_adequacy(eta, 0.5, 0.5)),)))
    assert st2.gain(Coverage((("h", 1.0, hypothesis_adequacy(eta, 0.9, 0.5)),))) == 0.0
    assert st2.gain(Coverage((("h", 1.0, hypothesis_adequacy(eta, 0.5, 0.9)),))) == 0.0
    assert st2.gain(Coverage((("h", 1.0, hypothesis_adequacy(eta, 0.9, 0.8)),))) > 0.0


def test_bridge_discount_and_granularity():
    cfg = RetrievalConfig()
    assert bridge_discount(1, cfg) == 1.0
    assert bridge_discount(2, cfg) == pytest.approx(cfg.bridge_decay)
    assert bridge_discount(cfg.max_bridge_distance + 1, cfg) == 0.0
    assert bridge_discount(0, cfg) == 0.0
    g = Granularity(cfg)
    assert g.gamma("empirical", "concept") == 0.0  # concepts never substitute for papers
    assert g.gamma("mechanism", "chunk") > g.gamma("mechanism", "paper")
    assert g.gamma("overview", "concept") > g.gamma("overview", "chunk")
    cfg.use_role_aware_granularity = False
    flat = Granularity(cfg)
    assert {flat.gamma(r, lvl) for r in ("overview", "empirical", "mechanism")
            for lvl in ("concept", "paper", "chunk")} == {1.0}


def test_relevance_ball_bound_is_admissible(rt):
    bench = make_synthetic_benchmark(n_papers=60, seed=2)
    index = SemanticIndex.build(bench.problem_gold, bench.method_gold, bench.corpus.papers, rt.embedder)
    X = rt.embedder.encode_queries(["graph neural networks for drug discovery", "pollution sensor forecasting"])
    cache = RelevanceCache(index, X)
    for nid in bench.problem_gold.nodes:
        bound = cache.rel_bound(index.ball(nid, "paper"))
        for p in index.extent(nid):
            assert np.all(cache.rel(paper_key(p)) <= bound + 1e-9)  # Cauchy-Schwarz (Lemma 6)
    assert cache.scored > 0


# ---------------------------------------------------------------------------
# Step 1: query understanding
# ---------------------------------------------------------------------------


def test_analyze_query_with_mock_llm(rt):
    q = Query("q1", "How have graph neural networks been used for drug-target interaction?", EVIDENCE)
    a = analyze_query(q, rt.llm, rt.embedder, rt.config.retrieval)
    assert a.used_llm and a.kind == EVIDENCE
    assert a.method_concepts == ["graph neural networks"] and a.problem_concepts == ["drug-target interaction"]
    assert sum(x.weight for x in a.aspects) == pytest.approx(1.0)
    assert {x.role for x in a.aspects} == {"overview", "empirical", "mechanism"}
    assert all(abs(float(np.linalg.norm(x.vector)) - 1.0) < 1e-5 for x in a.aspects)
    assert rt.meter.component("query_understanding").calls >= 1

    h = Query("q2", "Which other methods could be applied to air-quality forecasting?", HYPOTHESIS_DISCOVERY)
    b = analyze_query(h, rt.llm, rt.embedder, rt.config.retrieval)
    assert b.kind == HYPOTHESIS_DISCOVERY and b.searched_side == "method"
    assert b.problem_concepts == ["air-quality forecasting"] and not b.method_concepts
    assert b.premise_aspects("problem") and b.premise_aspects("method")

    c = analyze_query("Could graph contrastive learning be applied to air-quality forecasting?",
                      rt.llm, rt.embedder, rt.config.retrieval)
    assert c.kind == HYPOTHESIS_ASSESSMENT
    assert c.given_pair == ("air-quality forecasting", "graph contrastive learning")


def test_analyze_query_llm_free_fallback(rt):
    cfg = RetrievalConfig(llm_query_understanding=False)
    before = rt.meter.total().calls
    q = Query("q", "graph neural networks for drug discovery", EVIDENCE)
    a = analyze_query(q, rt.llm, rt.embedder, cfg)
    assert rt.meter.total().calls == before and not a.used_llm
    assert len(a.aspects) == 1 and a.aspects[0].text == q.text and a.aspects[0].weight == pytest.approx(1.0)
    h = analyze_query(Query("h", "new methods for flood forecasting", HYPOTHESIS_DISCOVERY), None, rt.embedder, cfg)
    assert h.premise_aspects("problem") and h.premise_aspects("method")
    assert sum(x.weight for x in h.aspects) == pytest.approx(1.0)


def test_parse_query_understanding_normalizes():
    obj = {"kind": "weird", "problem_concepts": "flood forecasting", "method_concepts": [None, "none", "GP"],
           "aspects": [{"text": "a", "role": "Mechanistic", "weight": -2, "target": "method side"},
                       {"text": " ", "role": "overview"}, "junk"]}
    out = parse_query_understanding(obj, kind_hint=HYPOTHESIS_DISCOVERY)
    assert out["kind"] == HYPOTHESIS_DISCOVERY
    assert out["problem_concepts"] == ["flood forecasting"] and out["method_concepts"] == ["GP"]
    assert out["aspects"] == [{"text": "a", "role": "mechanism", "weight": 1.0, "target": "method_premise"}]
    ev = parse_query_understanding({"aspects": [{"text": "x", "target": "problem_premise"}]})
    assert ev["kind"] == EVIDENCE and ev["aspects"][0]["target"] == "evidence"
