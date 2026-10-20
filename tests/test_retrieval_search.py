"""Tests for the bound-guided search of Algorithm 2 (Section 4.3-4.4, Appendix A.3-A.5, B.4)."""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

import numpy as np
import pytest

from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.index import SemanticIndex
from scholarpath.retrieval.grounding import default_anchor_min_sim, ground
from scholarpath.retrieval.objective import Coverage, SelectionState, delta
from scholarpath.retrieval.pipeline import ScholarPathRetriever
from scholarpath.retrieval.query_understanding import Aspect, QueryAnalysis
from scholarpath.retrieval.search import (
    BoundGuidedSearch,
    CandidateSubtreeRegion,
    ExhaustiveGreedy,
    RangeArgMin,
    Region,
    Witness,
    WitnessCosts,
    build_search_space,
    ranked_join,
)
from scholarpath.retrieval.units import ComplementaryPathSet, DirectPath
from scholarpath.runtime import Runtime
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.types import EVIDENCE, HYPOTHESIS_ASSESSMENT, HYPOTHESIS_DISCOVERY, Chunk, Paper, Query


@pytest.fixture(scope="module")
def rt():
    return Runtime.from_config()


# ---------------------------------------------------------------------------
# Example 1 / Fig. 1: the toy repository of the paper
# ---------------------------------------------------------------------------


def example1_index(embedder) -> SemanticIndex:
    """Fig. 1(a): problems u1-u5, methods m1-m5, papers p1-p5 with chunks s_{1,1}, s_{2,1}."""
    P = Taxonomy("problem", "Research problems", "All research problems.")
    P.add_node("Drug discovery", "finding drug candidates and molecules for therapeutic targets", node_id="u1")
    P.add_node("Environmental forecasting", "forecasting environmental conditions such as weather and pollution",
               node_id="u2")
    P.add_node("Drug-target interaction", "predicting binding between drug molecules and protein targets",
               parent="u1", node_id="u3")
    P.add_node("Molecular property prediction", "predicting toxicity and solubility of molecules",
               parent="u1", node_id="u4")
    P.add_node("Air-quality forecasting", "forecasting pollution levels from sensor networks", parent="u2",
               node_id="u5")
    M = Taxonomy("method", "Research methods", "All research methods.")
    M.add_node("Graph learning", "learning representations of graph-structured data", node_id="m1")
    M.add_node("Federated learning", "training models across decentralized clients without sharing data",
               node_id="m2")
    M.add_node("Evolutionary search", "population-based optimization with mutation and crossover", node_id="m5")
    M.add_node("Graph contrastive learning",
               "self-supervised graph representation learning by contrasting augmented graph views",
               parent="m1", node_id="m3")
    M.add_node("Graph neural networks", "neural message passing over nodes and edges of graphs", parent="m1",
               node_id="m4")
    papers = [
        Paper("p1", "Graph contrastive learning for drug-target binding",
              "We pretrain drug-target interaction models with graph contrastive learning on augmented molecular "
              "graph views.",
              chunks=[Chunk("p1#1", "p1", "Method", "Our graph contrastive learning assumes that node and edge "
                            "dropping augmentations preserve graph semantics; contrastive views are built by graph "
                            "augmentation.", 1)]),
        Paper("p2", "Graph neural networks for drug-target and molecular property prediction",
              "Graph neural networks and graph contrastive pretraining predict drug-target binding and molecular "
              "toxicity.",
              chunks=[Chunk("p2#1", "p2", "Method", "We adapt neural message passing to molecular graphs: atoms are "
                            "nodes, bonds are edges, and messages aggregate neighbour atom features.", 1)]),
        Paper("p3", "Message passing networks predict molecular properties",
              "A graph neural network predicts toxicity and solubility of molecules."),
        Paper("p4", "Graph neural networks for air-quality forecasting over sensor networks",
              "Sensor networks are modeled as graphs and graph neural networks forecast pollution levels at "
              "monitoring stations.",
              chunks=[Chunk("p4#1", "p4", "Experiments", "Experiments on city sensor networks show lower pollution "
                            "forecasting error.", 1)]),
        Paper("p5", "Federated learning for pollution forecasting",
              "Federated learning trains pollution forecasting models across distributed sensor clients."),
    ]
    links = {"p1": (["u3"], ["m3"]), "p2": (["u3", "u4"], ["m3", "m4"]), "p3": (["u4"], ["m4"]),
             "p4": (["u5"], ["m4"]), "p5": (["u5"], ["m2"])}
    for pid, (ps, ms) in links.items():
        for n in ps:
            P.assign(pid, n)
        for n in ms:
            M.assign(pid, n)
    return SemanticIndex.build(P, M, papers, embedder)


@pytest.fixture(scope="module")
def toy(rt):
    return example1_index(rt.embedder)


def test_example1_evidence_query_q1(rt, toy):
    """q1: J(u1, m4) = {p2, p3}; concept paths to the sub-problems u3, u4; findings cite p2 and p3."""
    assert toy.shared_papers("u1", "m4") == {"p2", "p3"}
    r = ScholarPathRetriever(toy, rt.llm, rt.embedder, rt.config.retrieval)
    res = r.retrieve(Query("q1", "How have graph neural networks been used in drug discovery?", EVIDENCE), 1024)
    assert res.stats["grounding"]["problem_scopes"][0][0] == "u1"
    assert res.stats["grounding"]["method_scopes"][0][0] == "m4"
    assert {"p2", "p3"} <= set(res.ranked_papers[:3])
    assert all(u.kind == "direct" and u.status == "established" for u in res.units)
    nodes = {k for u in res.units for k in u.nodes}
    assert {"concept:u3", "concept:u4"} <= nodes  # the sub-problems addressed in the literature
    for pid in res.ranked_papers:  # shared nodes appear once in the context
        assert res.context.count(f"- [{pid}] ") == 1
    assert res.context_tokens <= 1024


def test_example1_hypothesis_query_q2_traces_appendix_b4(rt, toy):
    """q2: candidate GCL (m3) supported by e_C = m1 -> m3, e_P = u5 -> p4, e_M = m3 -> p1 (-> s_{1,1})."""
    assert not toy.shared_papers("u5", "m3")
    r = ScholarPathRetriever(toy, rt.llm, rt.embedder, rt.config.retrieval)
    q2 = Query("q2", "Which other methods could be applied to air-quality forecasting?", HYPOTHESIS_DISCOVERY)
    res = r.retrieve(q2, 1024)
    assert res.candidates[0] == "Graph contrastive learning"
    sets = [u for u in res.units if u.kind == "complementary"]
    first = sets[0]
    assert first.status == "hypothesis" and first.candidate_id == "m3"
    assert first.metadata["e_C"] == ["concept:m1", "concept:m3"]
    assert first.metadata["e_P"] == ["concept:u5", "paper:p4"]
    assert first.metadata["e_M"][:2] == ["concept:m3", "paper:p1"]
    assert first.metadata["e_C"] and first.metadata["bridge"] == "m1" and first.metadata["baseline"] == "m4"
    # Selective chunk expansion goes to the weaker premise (Eq. 9, App. B.4).
    weak_side = "e_M" if first.metadata["s_M"] < first.metadata["s_P"] else "e_P"
    upgrades = [u for u in sets[1:] if u.candidate_id == "m3"]
    for u in upgrades:
        assert u.metadata[weak_side][-1].startswith("chunk:")
    assert "## Hypothesis: Graph contrastive learning for Air-quality forecasting" in res.context
    assert res.context_tokens <= 1024


def test_example1_hypothesis_assessment(rt, toy):
    """Assessment fixes the pair: eta_hat = phi(b) (no link-prediction factor, Eq. 9)."""
    r = ScholarPathRetriever(toy, rt.llm, rt.embedder, rt.config.retrieval)
    q = Query("qa", "Could graph contrastive learning be applied to air-quality forecasting?", HYPOTHESIS_ASSESSMENT)
    res = r.retrieve(q, 1024)
    assert res.stats["kind"] == HYPOTHESIS_ASSESSMENT and res.candidates == ["Graph contrastive learning"]
    sets = [u for u in res.units if u.kind == "complementary"]
    assert sets and sets[0].metadata["e_P"][-1] == "paper:p4"
    phi = rt.config.retrieval.bridge_decay ** (sets[0].metadata["distance"] - 1)
    assert sets[0].metadata["eta"] == pytest.approx(phi)
    assert res.stats["scored_candidates"] == 1 and res.stats["link_evaluations"] == 0


@pytest.mark.parametrize("text, method", [
    ("Could graph neural networks be applied to drug discovery?", "m4"),  # J(u1, m4) = {p2, p3}
    ("Could graph learning be applied to air-quality forecasting?", "m1"),  # J(u5, m1) = {p4}: m1 is above m4
])
def test_assessment_of_a_studied_pair_returns_established_findings(rt, toy, text, method):
    """Sec. 2.1: hypotheses concern pairs with J(u, m) empty; a studied pair is answered by J(u_q, m_q)."""
    r = ScholarPathRetriever(toy, rt.llm, rt.embedder, rt.config.retrieval)
    res = r.retrieve(Query("qs", text, HYPOTHESIS_ASSESSMENT), 1024)
    problem = res.stats["grounding"]["problem_scopes"][0][0]
    assert res.stats["grounding"]["method_scopes"][0][0] == method
    junction = toy.shared_papers(problem, method)
    assert junction and res.units and not res.candidates
    assert all(u.kind == "direct" and u.status == "established" for u in res.units)
    assert set(res.ranked_papers) <= junction
    assert "Hypothesis:" not in res.context


def _analysis(rt, specs, problem, method, text="q") -> QueryAnalysis:
    vecs = rt.embedder.encode_queries([t for t, _, _ in specs])
    total = sum(w for _, w, _ in specs)
    aspects = [Aspect(t, w / total, role, vecs[i]) for i, (t, w, role) in enumerate(specs)]
    return QueryAnalysis("q", text, EVIDENCE, problem, method, aspects)


def _select(rt, index, analysis, budget, cfg=None):
    cfg = cfg or rt.config.retrieval
    g = ground(analysis, index, rt.embedder, cfg)
    return BoundGuidedSearch(build_search_space(analysis, g, index, cfg)).run(budget)


def test_selective_chunk_expansion_only_when_detail_is_needed(rt, toy):
    """App. B.4: keep p2's abstract when it suffices; pay for s_{2,1} only for the message-passing detail."""
    empirical = ("graph neural networks predict drug-target binding and molecular toxicity", 1.0, "empirical")
    mechanism = ("neural message passing adapted to molecular graphs with atoms as nodes and bonds as edges",
                 1.0, "mechanism")
    res_a = _select(rt, toy, _analysis(rt, [empirical], ["drug discovery"], ["graph neural networks"]), 2000)
    ends_a = [s.entry.unit.end for s in res_a.selections]
    assert "paper:p2" in ends_a and "chunk:p2#1" not in ends_a
    res_b = _select(rt, toy, _analysis(rt, [empirical, mechanism], ["drug discovery"], ["graph neural networks"]),
                    2000)
    sel = {s.entry.unit.end: s for s in res_b.selections}
    assert "chunk:p2#1" in sel and "paper:p2" in sel
    order = [s.entry.unit.end for s in res_b.selections]
    assert order.index("paper:p2") < order.index("chunk:p2#1")
    # The extension e+ costs only the chunk tokens once e is selected.
    space = build_search_space(_analysis(rt, [empirical, mechanism], ["drug discovery"], ["graph neural networks"]),
                               ground(_analysis(rt, [empirical], ["drug discovery"], ["graph neural networks"]),
                                      toy, rt.embedder, rt.config.retrieval), toy, rt.config.retrieval)
    assert sel["chunk:p2#1"].tokens == space.cost("chunk:p2#1")


# ---------------------------------------------------------------------------
# Lemma 6 and Proposition 7 on the synthetic repository
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def syn(rt):
    bench = make_synthetic_benchmark(n_papers=80, seed=5, n_heldout=3)
    index = SemanticIndex.build(bench.problem_gold, bench.method_gold, bench.corpus.papers, rt.embedder)
    cfg = copy.deepcopy(rt.config.retrieval)
    cfg.max_chunks_per_paper = 2
    retriever = ScholarPathRetriever(index, rt.llm, rt.embedder, cfg)
    queries = bench.evidence_queries[:4] + bench.hypothesis_queries
    prepared = [(q, retriever.analyze(q)) for q in queries]
    prepared = [(q, a, retriever.ground(a)) for q, a in prepared]
    return bench, index, retriever, prepared


def _c_low(space, region: Region, state: SelectionState, hit: set) -> int:
    total = sum(space.cost(x) for x in dict.fromkeys(region.req) if x not in state.nodes)
    for g in region.groups:
        total += g.value(state, any(t in hit for t in g.tokens))
    return total


def _check_region_bounds(space, state: SelectionState) -> int:
    """Lemma 6: every unit z below a region H has gain <= num(H), Delta C >= c_low(H), delta <= delta_bar."""
    hit = {t for x in state.nodes for t in space.tokens_of(x)}
    checked = 0
    # Iterative post-order over the region tree: (region, children, units).
    stack: List[Tuple[Region, bool]] = [(r, False) for r in space.top_regions()]
    agg: Dict[int, Tuple[float, float, float]] = {}
    pending: Dict[int, List[Region]] = {}
    own: Dict[int, list] = {}
    while stack:
        region, done = stack.pop()
        if not done:
            units, children = region.expand(space)
            own[id(region)] = units
            pending[id(region)] = children
            stack.append((region, True))
            stack.extend((c, False) for c in children)
            continue
        max_gain, min_cost, max_delta = 0.0, float("inf"), 0.0
        for u in own.pop(id(region)):
            g = state.gain(u.coverage)
            c = state.marginal_cost(u.nodes)
            max_gain, min_cost = max(max_gain, g), min(min_cost, c)
            max_delta = max(max_delta, delta(g, c))
        for child in pending.pop(id(region)):
            cg, cc, cd = agg.pop(id(child))
            max_gain, min_cost, max_delta = max(max_gain, cg), min(min_cost, cc), max(max_delta, cd)
        num = region.numerator(space, state)
        c_low = _c_low(space, region, state, hit)
        assert max_gain <= num + 1e-12, (region.rid, max_gain, num)
        assert min_cost >= c_low, (region.rid, min_cost, c_low)
        assert max_delta <= delta(num, c_low) + 1e-12, region.rid
        agg[id(region)] = (max_gain, min_cost, max_delta)
        checked += 1
    return checked


def test_lemma6_region_bounds_are_admissible(rt, syn):
    _, index, retriever, prepared = syn
    total = 0
    for q, a, g in prepared:
        space = retriever.search_space(a, g)
        state = SelectionState(10 ** 6, space.cost)
        total += _check_region_bounds(space, state)
        # Re-check under a nonempty selection S (f_a > 0, shared nodes selected).
        partial = BoundGuidedSearch(retriever.search_space(a, g)).run(700)
        space2 = retriever.search_space(a, g)
        state2 = SelectionState(10 ** 6, space2.cost)
        for sel in partial.selections:
            if isinstance(sel.entry.unit, ComplementaryPathSet):
                space2.renderer.register_set(sel.entry.unit)
            state2.apply(sel.entry.nodes, sel.entry.coverage)
        total += _check_region_bounds(space2, state2)
    assert total > 100


@pytest.mark.parametrize("budget", [250, 900, 4000])
def test_prop7_bound_guided_equals_exhaustive_greedy(rt, syn, budget):
    _, index, retriever, prepared = syn
    for q, a, g in prepared:
        bgs = BoundGuidedSearch(retriever.search_space(a, g)).run(budget)
        exh = ExhaustiveGreedy(retriever.search_space(a, g)).run(budget)
        assert bgs.uids == exh.uids, q.query_id
        assert [s.tokens for s in bgs.selections] == [s.tokens for s in exh.selections]
        assert bgs.state.used == exh.state.used <= budget
        assert bgs.stats["units_created"] <= exh.stats["units_created"]


def test_prop7_holds_without_shared_cost_update(rt, syn):
    _, index, retriever, prepared = syn
    cfg = copy.deepcopy(retriever.config)
    cfg.use_shared_cost_update = False
    for q, a, g in prepared:
        bgs = BoundGuidedSearch(build_search_space(a, g, index, cfg, retriever.link_predictor)).run(1200)
        exh = ExhaustiveGreedy(build_search_space(a, g, index, cfg, retriever.link_predictor)).run(1200)
        assert bgs.uids == exh.uids and bgs.state.used <= 1200


def test_weaker_premise_rule_on_synthetic_hypotheses(rt, syn):
    """A selected chunk upgrade of a path set always extends the premise that limited its paper-level set."""
    _, index, retriever, prepared = syn
    seen_upgrade = False
    for q, a, g in prepared:
        if not a.is_hypothesis:
            continue
        res = BoundGuidedSearch(retriever.search_space(a, g)).run(6000)
        base: Dict[Tuple[str, str, str], dict] = {}
        for sel in res.selections:
            b = sel.entry.unit
            if not isinstance(b, ComplementaryPathSet):
                continue
            key = (b.candidate, b.e_P.paper_id, b.e_M.paper_id)
            p_chunk, m_chunk = b.e_P.level == "chunk", b.e_M.level == "chunk"
            if not p_chunk and not m_chunk:
                base[key] = sel.entry.meta
            elif p_chunk != m_chunk and key in base:
                seen_upgrade = True
                meta = base[key]
                if p_chunk:
                    assert meta["s_P"] <= meta["s_M"] + 1e-12
                else:
                    assert meta["s_M"] <= meta["s_P"] + 1e-12
    assert seen_upgrade


def test_subtree_bound_uses_the_best_bridge_discount_below_v(rt, syn):
    """eta_bar(H) = sigma(U(u_q, v)) phi_bar(H), phi_bar(H) = max bridge discount of the candidates below v."""
    _, index, retriever, prepared = syn
    cfg = retriever.config
    checked = 0
    for q, a, g in prepared:
        if not a.is_hypothesis:
            continue
        space = retriever.search_space(a, g)
        hyp = space.hyp
        tax = index.taxonomy(hyp.cand_facet)
        for v, flag in hyp.has_candidate.items():
            below = [c for c in tax.descendants(v, include_self=True) if c in hyp.candidates]
            assert flag == bool(below)
            best = max((cfg.bridge_decay ** (hyp.d_min[c] - 1) for c in below), default=0.0)
            assert hyp.phi_bar[v] == pytest.approx(best)
            if flag:
                region = CandidateSubtreeRegion(space, v)
                expected = space.link_bound(v) * best * min(space.known_premise_bound(), space.cand_premise_bound(v))
                assert region.rbar(space) == pytest.approx(expected)
                checked += 1
    assert checked


def test_range_minimum_queries_under_a_selection():
    """Cost bounds of the pair ranges: range minima of terminal costs, exact under the current S."""
    rng = random.Random(3)
    values = [rng.randint(1, 50) for _ in range(37)]
    rmq = RangeArgMin(values)
    for lo in range(len(values)):
        for hi in range(lo + 1, len(values) + 1):
            m = rmq.argmin(lo, hi)
            assert values[m] == min(values[lo:hi]) and m == values.index(values[m], lo, hi)
    costs = {f"paper:p{i}": rng.randint(5, 40) for i in range(8)}
    costs.update({f"chunk:p{i}#0": rng.randint(5, 40) for i in range(8)})
    witnesses = []
    for i in range(8):
        witnesses.append(Witness(0.0, DirectPath(("concept:c", f"paper:p{i}"), "c", "method")))
        witnesses.append(Witness(0.0, DirectPath(("concept:c", f"paper:p{i}", f"chunk:p{i}#0"), "c", "method")))
    rng.shuffle(witnesses)
    space = type("Space", (), {"cost": staticmethod(costs.__getitem__)})()
    wc = WitnessCosts(space, witnesses)
    state = SelectionState(10 ** 6, costs.__getitem__)
    for step in range(4):
        for lo in range(len(witnesses)):
            for hi in range(lo + 1, len(witnesses) + 1):
                brute = min(sum(costs[k] for k in w.path.nodes[1:] if k not in state.nodes) for w in witnesses[lo:hi])
                assert wc.min(lo, hi, state) == brute
        state.apply((f"paper:p{2 * step}",) + ((f"chunk:p{2 * step}#0",) if step % 2 else ()), Coverage())


def test_junction_pruning_is_reported(rt, syn):
    _, index, retriever, prepared = syn
    q, a, g = next(t for t in prepared if t[1].kind == EVIDENCE)
    stats = BoundGuidedSearch(retriever.search_space(a, g)).run(10 ** 6).stats
    assert stats["postings_pruned_by_junction"] > 0
    assert stats["postings_scanned"] + stats["postings_pruned"] == stats["postings_total"]


# ---------------------------------------------------------------------------
# Prop. 7 / Lemma 6 on deeper random taxonomies with multi-label and residual papers
# ---------------------------------------------------------------------------

_WORDS = ("graph neural network contrastive learning drug target molecule property pollution sensor forecasting "
          "federated client privacy evolutionary search mutation transformer attention image segmentation protein "
          "folding diffusion model reinforcement policy reward time series anomaly detection kernel bayesian "
          "optimization causal inference language retrieval summarization").split()


def _random_index(embedder, seed: int, n_papers: int = 45, n_concepts: int = 11) -> SemanticIndex:
    rng = random.Random(seed)

    def text(k: int) -> str:
        return " ".join(rng.choice(_WORDS) for _ in range(k))

    def taxonomy(facet: str) -> Taxonomy:
        tax = Taxonomy(facet, "Root", f"all {facet}s")
        frontier = [tax.root_id]
        while len(tax.nodes) < n_concepts and frontier:
            parent = rng.choice(frontier)
            if len(tax.nodes[parent].children) >= 3:
                frontier.remove(parent)
                continue
            frontier.append(tax.add_node(f"{text(2)} {facet[0]}{len(tax.nodes)}", text(6), parent=parent))
        return tax

    P, M = taxonomy("problem"), taxonomy("method")
    papers = []
    for i in range(n_papers):
        pid = f"x{i}"
        chunks = [Chunk(f"{pid}#{j}", pid, rng.choice(["Method", "Experiments"]), text(15), j)
                  for j in range(rng.randint(0, 2))]
        papers.append(Paper(pid, text(5), text(20), chunks=chunks))
        for tax in (P, M):  # one or two labels anywhere in the tree (internal labels leave residual papers)
            for nid in rng.sample([n for n in tax.nodes if n != tax.root_id], rng.choice([1, 1, 2])):
                tax.assign(pid, nid)
    return SemanticIndex.build(P, M, papers, embedder)


@pytest.mark.parametrize("seed", range(3))
def test_prop7_on_random_multilabel_taxonomies(rt, seed):
    index = _random_index(rt.embedder, seed)
    cfg = copy.deepcopy(rt.config.retrieval)
    cfg.max_chunks_per_paper = 2
    r = ScholarPathRetriever(index, rt.llm, rt.embedder, cfg)
    rng = random.Random(seed)
    pn = [index.problem.nodes[n].name for n in index.problem.nodes if n != index.problem.root_id]
    mn = [index.method.nodes[n].name for n in index.method.nodes if n != index.method.root_id]
    queries = [
        Query("e1", f"How has {rng.choice(mn)} been used for {rng.choice(pn)}?", EVIDENCE),
        Query("e2", f"What methods are used for {rng.choice(pn)}?", EVIDENCE),
        Query("h1", f"Which other methods could be applied to {rng.choice(pn)}?", HYPOTHESIS_DISCOVERY),
        Query("h2", f"Which problems could {rng.choice(mn)} be applied to?", HYPOTHESIS_DISCOVERY),
        Query("a1", f"Could {rng.choice(mn)} be applied to {rng.choice(pn)}?", HYPOTHESIS_ASSESSMENT),
        Query("a2", f"Could quantum annealing hardware be applied to {rng.choice(pn)}?", HYPOTHESIS_ASSESSMENT),
    ]
    for q in queries:
        a = r.analyze(q)
        g = r.ground(a)
        for budget in (300, 3000):
            bgs = BoundGuidedSearch(r.search_space(a, g)).run(budget)
            exh = ExhaustiveGreedy(r.search_space(a, g)).run(budget)
            assert bgs.uids == exh.uids, (seed, q.query_id, budget)
            assert bgs.state.used == exh.state.used <= budget
        res = r.retrieve(q, 600)
        assert res.context_tokens <= 600
        if seed == 0:  # Lemma 6 on every region of the query
            space = r.search_space(a, g)
            assert _check_region_bounds(space, SelectionState(10 ** 6, space.cost))


# ---------------------------------------------------------------------------
# Appendix A.3: ranked join with early termination (Proposition 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_ranked_join_returns_exact_topk(rt, syn, seed):
    bench, index, _, _ = syn
    rng = random.Random(seed)
    P = [n for n in index.problem.nodes if n != index.problem.root_id]
    M = [n for n in index.method.nodes if n != index.method.root_id]
    pairs = [(u, m, rng.random()) for u in rng.sample(P, 4) for m in rng.sample(M, 4)]
    rel = {p: rng.random() for p in index.papers}
    for k in (1, 3, 8):
        res = ranked_join(index, pairs, rel.__getitem__, k)
        omega: Dict[str, float] = {}
        for u, m, w in pairs:
            for p in index.shared_papers(u, m):
                omega[p] = max(omega.get(p, 0.0), w)
        brute = sorted((w * rel[p] for p, w in omega.items()), reverse=True)[:k]
        assert [s for _, s in res.top] == pytest.approx(brute)
        assert res.pairs_processed <= res.pairs_total


def test_ranked_join_terminates_early(rt, toy):
    pairs = [("u1", "m4", 1.0), ("u1", "m3", 0.3), ("u5", "m2", 0.2)]
    res = ranked_join(toy, pairs, lambda p: 1.0, k=1)
    assert res.top[0][1] == pytest.approx(1.0) and res.top[0][0] in {"p2", "p3"}
    assert res.pairs_processed == 1 and res.postings_pruned > 0
    empty = ranked_join(toy, pairs, lambda p: 1.0, k=0)
    assert empty.top == [] and empty.pairs_processed == 0 and empty.postings_pruned == empty.postings_total


# ---------------------------------------------------------------------------
# Step 2: query-local anchors
# ---------------------------------------------------------------------------


def test_unfamiliar_concept_becomes_query_local_anchor(rt, toy):
    r = ScholarPathRetriever(toy, rt.llm, rt.embedder, rt.config.retrieval)
    q = Query("qa", "Which other methods could be applied to zebrafish morphogenesis?", HYPOTHESIS_DISCOVERY)
    a = r.analyze(q)
    g = r.ground(a)
    anchor = g.anchor("problem")
    assert anchor is not None and len(anchor.neighbors) == rt.config.retrieval.anchor_top_k
    # Unrelated neighbours (cosine below anchor_min_sim) do not become scopes: the facet is unmatched.
    assert max(sim for _, sim in anchor.neighbors) < default_anchor_min_sim(rt.config.retrieval)
    assert g.unmatched["problem"] and g.problem_scopes == [(toy.problem.root_id, 1.0)]
    z = r.link_predictor.z(anchor.anchor_id)  # encoded from its description (Section 4.3)
    assert np.all(np.isfinite(z))
    res = r.retrieve(q, 1024)
    assert res.context_tokens <= 1024
    # The hypothesis side still attaches the anchor to its top-k neighbours, and says so in the context.
    assert res.context.startswith('Note: "zebrafish morphogenesis" is not in the index')
    for u in res.units:
        if u.kind == "complementary":
            assert u.metadata["known"] == anchor.anchor_id


def test_anchor_neighbours_scope_evidence_only_when_related(rt, toy):
    r = ScholarPathRetriever(toy, rt.llm, rt.embedder, rt.config.retrieval)
    # "contrastive pretraining" is absent from I but close to GCL (m3): m3 stands in for it.
    near = r.retrieve(Query("e1", "How has contrastive pretraining been used for drug discovery?", EVIDENCE), 1024)
    g = near.stats["grounding"]
    assert g["anchors"] and [s for s, _ in g["method_scopes"]] == ["m3"] and not g["unmatched"]
    assert set(near.ranked_papers) <= toy.shared_papers("u1", "m3") and not near.context.startswith("Note:")
    # An unrelated absent method: no neighbour stands in, so no other method's papers are cited for the pair;
    # the findings of the matched problem are returned under an explicit note.
    far = r.retrieve(Query("e2", "How has quantum annealing hardware been used for drug discovery?", EVIDENCE), 1024)
    g = far.stats["grounding"]
    assert g["unmatched"] == {"method": True} and g["method_scopes"] == [[toy.method.root_id, 1.0]]
    assert far.context.startswith('Note: no method concept in the index matches "quantum annealing hardware"')
    assert set(far.ranked_papers) <= toy.extent("u1") and far.context_tokens <= 1024
    # The note's tokens are reserved from the budget.
    for budget in (60, 150, 250):
        tight = r.retrieve(Query("e2", far.stats["analysis"]["text"], EVIDENCE), budget)
        assert tight.context_tokens <= budget
        assert tight.context.startswith("Note:") == bool(tight.units)


def test_assessment_of_a_concept_absent_from_the_index_keeps_the_asked_pair(rt, toy):
    """The given method is a query-local anchor, never silently replaced by its closest neighbour."""
    r = ScholarPathRetriever(toy, rt.llm, rt.embedder, rt.config.retrieval)
    q = Query("qa", "Could spiking neuromorphic chips be applied to air-quality forecasting?", HYPOTHESIS_ASSESSMENT)
    a = r.analyze(q)
    g = r.ground(a)
    anchor = g.anchor("method")
    assert anchor is not None
    res = r.retrieve(q, 1024)
    assert res.candidates == ["spiking neuromorphic chips"]
    assert "## Hypothesis: spiking neuromorphic chips for Air-quality forecasting" in res.context
    assert "## Hypothesis: Graph learning" not in res.context
    sets = [u for u in res.units if u.kind == "complementary"]
    neighbours = [n for n, _ in anchor.neighbors]
    assert sets and all(u.candidate_id == anchor.anchor_id for u in sets)
    for u in sets:
        assert u.metadata["e_C"][-1] in {f"concept:{n}" for n in neighbours}  # attached to a neighbour
        assert u.metadata["e_M"][0] in {f"concept:{n}" for n in neighbours}  # method side via a neighbour
        assert u.metadata["e_P"][0] == "concept:u5" and u.metadata["eta"] == pytest.approx(
            rt.config.retrieval.bridge_decay ** (u.metadata["distance"] - 1))
    assert res.stats["link_evaluations"] == 0 and res.context_tokens <= 1024
    space_a, space_b = r.search_space(a, g), r.search_space(a, g)
    for budget in (200, 1024):
        assert BoundGuidedSearch(space_a).run(budget).uids == ExhaustiveGreedy(space_b).run(budget).uids


def test_discovery_retrieves_the_already_studied_baselines(rt, toy):
    """Step 2: discovery aspects also target the baselines m' with J(u_q, m') non-empty (here m4 and m2)."""
    r = ScholarPathRetriever(toy, rt.llm, rt.embedder, rt.config.retrieval)
    q2 = Query("q2", "Which other methods could be applied to air-quality forecasting?", HYPOTHESIS_DISCOVERY)
    a = r.analyze(q2)
    space = r.search_space(a, r.ground(a))
    assert space.hyp.baselines == {"p4": ("m4",), "p5": ("m2",)}
    targets = {t for grp in space.aspects.groups.values() if grp.kind == "concept" for t in grp.targets}
    assert targets == {"concept:m4", "concept:m2"}
    res = r.retrieve(q2, 2048)
    direct = [u for u in res.units if u.kind == "direct"]
    assert {"p4", "p5"} <= {p for u in direct for p in u.papers}
    assert all(u.status == "established" for u in direct)
    assert res.candidates[0] == "Graph contrastive learning"
    for pid in ("p4", "p5"):  # shared nodes appear once even though p4 is also a premise of the hypothesis
        assert res.context.count(f"- [{pid}] ") == 1


def test_problem_side_discovery(rt, toy):
    """Roles exchanged (App. A.4): candidate problems for a given method, here GCL (m3)."""
    r = ScholarPathRetriever(toy, rt.llm, rt.embedder, rt.config.retrieval)
    q3 = Query("q3", "Which problems could graph contrastive learning be applied to?", HYPOTHESIS_DISCOVERY)
    a = r.analyze(q3)
    assert a.searched_side == "problem"
    space = r.search_space(a, r.ground(a))
    assert space.hyp.known_id == "m3" and space.hyp.candidates == {"u2", "u5"}
    res = r.retrieve(q3, 1024)
    assert set(res.candidates) == {"Environmental forecasting", "Air-quality forecasting"}
    sets = [u for u in res.units if u.kind == "complementary"]
    assert sets
    for u in sets:
        assert not toy.shared_papers(u.candidate_id, "m3")
        assert u.metadata["e_M"][0] == "concept:m3"  # known (method) side
        assert u.metadata["e_P"][0] == f"concept:{u.candidate_id}"  # candidate (problem) side
    # Every candidate reaches the composed answer even though both share their premise papers.
    ans = r.answer(q3, 1024)
    assert {it.candidate for it in ans.items if it.status == "hypothesis"} == set(res.candidates)
