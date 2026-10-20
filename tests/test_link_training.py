"""Tests for temporally trained cross-taxonomy link prediction (Sec. 4.3 Eq. 8/12; App. B.5 Eq. 17-18)."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import pytest

from scholarpath.config import LinkPredictionConfig
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.index import SemanticIndex, concept_key
from scholarpath.retrieval.link_prediction import EmbeddingLinkPredictor, LinkPredictor
from scholarpath.retrieval.link_training import (
    ComparisonSampler,
    EmptySupervisionError,
    FallbackLinkPredictor,
    TemporalLinkPredictor,
    add_years,
    build_supervision,
    evaluate_ranking,
    load_predictor,
    most_specific_pairs,
    normalize_date,
    ranking_candidates,
    related_concepts,
    static_supervision,
    temporal_supervision,
    train_link_predictor,
)
from scholarpath.runtime import Runtime
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.types import Paper

FAST = dict(embed_dim=32, hidden_dim=64, epochs=12, learning_rate=2e-3)


@pytest.fixture(scope="module")
def rt():
    return Runtime.from_config()


@pytest.fixture(scope="module")
def synth(rt):
    bench = make_synthetic_benchmark(n_papers=200, seed=0)
    index = SemanticIndex.build(bench.problem_gold, bench.method_gold, bench.corpus.papers, rt.embedder)
    return bench, index


# ---------------------------------------------------------------------------
# A synthetic world with learnable cross-taxonomy structure
# ---------------------------------------------------------------------------

PROBLEM_TRAITS = ["seasonal", "networked"]
METHOD_TRAITS = ["oscillations", "relations"]


def trait_world(rt, seed: int = 0) -> Tuple[SemanticIndex, Dict[str, str]]:
    """Synthetic repository whose problem-method links follow description structure.

    The synthetic gold taxonomies are kept, but each leaf definition receives
    one latent trait, phrased with *different* words on the two sides
    ("data with seasonal structure" vs "designed for oscillations structure"),
    and papers are re-assigned so that a sub-problem is only ever solved by
    methods of the same trait. Half of the compatible pairs emerge in
    2015-2019 (training windows), the rest in 2021-2024 (held-out window).
    One undated-pair background paper per leaf makes every concept exist
    from 2014 on. The description-cosine prior carries no signal here, so
    ranking held-out pairs requires learning the structure from earlier
    windows through the encoders.
    """
    bench = make_synthetic_benchmark(n_papers=320, seed=seed)
    papers = {p.paper_id: p for p in list(bench.corpus) + list(bench.heldout_papers)}

    def skeleton(gold: Taxonomy, words: List[str], verb: str):
        tax = Taxonomy(gold.facet, gold.root.name, gold.root.definition)
        leaves = []
        for top in gold.children(gold.root_id):
            t = tax.add_node(gold[top].name, gold[top].definition)
            for leaf in gold.children(top):
                leaves.append((gold[leaf].name, t))
        out = []
        for i, (name, parent) in enumerate(sorted(leaves)):
            out.append((tax.add_node(name, f"{name}: {verb} {words[i % 2]} structure.", parent=parent), i % 2))
        return tax, out

    ptax, pleaves = skeleton(bench.problem_gold, PROBLEM_TRAITS, "data with")
    mtax, mleaves = skeleton(bench.method_gold, METHOD_TRAITS, "designed for")
    compat = [(u, m) for u, tu in pleaves for m, tm in mleaves if tu == tm]
    rng = random.Random(seed)
    rng.shuffle(compat)
    half = len(compat) // 2
    emerge = {pair: rng.randint(2015, 2019) if i < half else rng.randint(2021, 2024) for i, pair in enumerate(compat)}
    dates: Dict[str, str] = {}
    ids = sorted(papers)
    rng.shuffle(ids)
    for i, pid in enumerate(ids):
        u, m = compat[i % len(compat)]
        dates[pid] = f"{rng.randint(emerge[(u, m)], 2024)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        ptax.assign(pid, u)
        mtax.assign(pid, m)
    for tax, leaves in ((ptax, pleaves), (mtax, mleaves)):
        for nid, _ in leaves:
            pid = f"bg-{nid}"
            papers[pid] = Paper(pid, f"Background on {tax[nid].name}", date="2014-06-01")
            dates[pid] = "2014-06-01"
            tax.assign(pid, nid)
    return SemanticIndex.build(ptax, mtax, papers, rt.embedder), dates


@pytest.fixture(scope="module")
def traits(rt):
    return trait_world(rt, seed=0)


# ---------------------------------------------------------------------------
# Dates and supervision (Eq. 17)
# ---------------------------------------------------------------------------


def test_date_helpers():
    assert normalize_date("2021") == "2021-01-01"
    assert normalize_date("2021-3") == "2021-03-01"
    assert normalize_date("2021-03-04T10:00:00Z") == "2021-03-04"
    assert normalize_date(2020) == "2020-01-01" and normalize_date(None) is None
    assert add_years("2019-01-01", 2) == "2021-01-01"
    assert add_years("2020-02-29", 1) == "2021-02-28"
    assert add_years("2020-01-01", 0.5) == "2020-07-02"
    # compact and dotted formats are parsed, never truncated to January 1
    assert normalize_date("20210304") == "2021-03-04"
    assert normalize_date("2021.07") == "2021-07-01"
    assert normalize_date("2021-03-04 10:00") == "2021-03-04" and normalize_date(2021.0) == "2021-01-01"
    for bad in ("2021-03-04x", "2107.01234", "2021-13-01", "abc", 2021.5):
        with pytest.raises(ValueError):
            normalize_date(bad)


@pytest.mark.parametrize("cutoff", ["2016-01-01", "2017-06-01", "2019-01-01"])
def test_supervision_never_shares_a_paper_before_cutoff(synth, cutoff):
    bench, index = synth
    rt = Runtime.from_config()
    sup = build_supervision(index, cutoff, 2.0, rt.llm)
    dates = {p.paper_id: p.date for p in bench.corpus}
    end = add_years(cutoff, 2.0)
    assert sup.verified and sup.end == end
    assert rt.meter.component("link_training").calls > 0 or not sup.candidates
    assert set(sup.positives) <= set(sup.candidates)
    for u, m in sup.positives:
        shared = index.extent(u) & index.extent(m)
        assert not any(dates[p] <= cutoff for p in shared)  # J_t(u, m) is empty
        assert sup.snapshot.in_history(u) and sup.snapshot.in_history(m)
        for p in sup.evidence[(u, m)]:  # verified application inside (t, t + Delta]
            assert p in shared and cutoff < dates[p] <= end
    # most specific pairs only: no positive is implied by a more specific positive
    assert most_specific_pairs(sup.positives, index.problem, index.method) == sorted(sup.positives)
    assert temporal_supervision(index, cutoff, 2.0, rt.llm) == sup.positives


def test_supervision_recovers_synthetic_heldout_pairs(rt):
    """The 2024 pairs of make_synthetic_benchmark are exactly the new links of the 2024 window."""
    bench = make_synthetic_benchmark(n_papers=200, n_heldout=3, seed=2)
    papers = {p.paper_id: p for p in list(bench.corpus) + list(bench.heldout_papers)}
    problem = Taxonomy.from_dict(bench.problem_gold.to_dict())
    method = Taxonomy.from_dict(bench.method_gold.to_dict())
    leaf_p = {problem[n].name: n for n in problem.leaves()}
    leaf_m = {method[n].name: n for n in method.leaves()}
    for p in bench.heldout_papers:
        s, m = bench.pair_of[p.paper_id]
        problem.assign(p.paper_id, leaf_p[s])
        method.assign(p.paper_id, leaf_m[m])
    index = SemanticIndex.build(problem, method, papers, rt.embedder)
    sup = build_supervision(index, "2023-12-31", 1.0, rt.llm)
    names = {(index.concept(u).name, index.concept(m).name) for u, m in sup.positives}
    assert names == set(bench.heldout_pairs)
    assert all(p.startswith("held") for ev in sup.evidence.values() for p in ev)


def test_supervision_without_llm_is_flagged(synth, rt):
    _, index = synth
    unverified = build_supervision(index, "2016-01-01", 2.0, llm=None)
    verified = build_supervision(index, "2016-01-01", 2.0, rt.llm)
    assert not unverified.verified and verified.verified
    assert set(verified.positives) <= set(unverified.positives) | {
        p for p in verified.positives if p not in unverified.candidates
    }
    assert len(unverified.positives) >= len(verified.positives)
    assert set(unverified.candidates) == set(verified.candidates)
    # LLM verification rejects candidates whose paper does not name both concepts
    assert set(verified.rejected) | set(verified.evidence) == set(verified.candidates)


def test_most_specific_pairs_drops_ancestor_pairs(synth):
    _, index = synth
    leaf_p = index.problem.leaves()[0]
    leaf_m = index.method.leaves()[0]
    parent_p, parent_m = index.problem.parent(leaf_p), index.method.parent(leaf_m)
    pairs = [(leaf_p, leaf_m), (parent_p, leaf_m), (leaf_p, parent_m), (parent_p, parent_m)]
    assert most_specific_pairs(pairs, index.problem, index.method) == [(leaf_p, leaf_m)]


# ---------------------------------------------------------------------------
# Comparison sampling (App. B.5)
# ---------------------------------------------------------------------------


def test_comparison_sampler_respects_constraints(synth):
    _, index = synth
    sets = [build_supervision(index, c, 1.0) for c in ("2015-07-01", "2016-01-01", "2016-07-01")]
    exclude = {sets[1].positives[0][1]}
    sampler = ComparisonSampler(index, sets, negatives_per_positive=6, hard_fraction=0.5, exclude_concepts=exclude)
    assert len(sampler) > 0
    ids = sampler.concepts()
    positives = {p for s in sets for p in s.positives}
    facet_of = {0: ("problem", "method"), 1: ("method", "problem")}
    for item in sampler.items:
        a_facet, c_facet = facet_of[item.direction]
        anchor, pos = ids[a_facet][item.anchor], ids[c_facet][item.positive]
        u, m = (anchor, pos) if item.direction == 0 else (pos, anchor)
        assert (u, m) in positives and u not in exclude and m not in exclude
        pool = {ids[c_facet][i] for i in sampler.pool(item)}
        assert not pool & exclude and pos not in pool
        tax = index.taxonomy(c_facet)
        assert not pool & set(tax.ancestors(pos))
        # comparisons are known non-links at some cutoff and never other known positives of the anchor
        for c in pool:
            pair = (anchor, c) if item.direction == 0 else (c, anchor)
            assert pair not in positives
        assert {ids[c_facet][i] for i in item.hard} <= pool & related_concepts(tax, pos)
    batch = sampler.sample(np.random.default_rng(0))
    assert batch["negative"].shape[0] == 6 * len(sampler)
    # every drawn comparison is admissible for its item
    offset = 0
    for item in sampler.items:
        drawn = set(batch["negative"][offset : offset + 6].tolist())
        assert drawn <= set(sampler.pool(item).tolist())
        offset += 6


def test_comparison_pool_matches_explicit_definition(synth):
    """The exclusion-encoded pools equal the explicit pool of App. B.5 at each cutoff."""
    _, index = synth
    checked = 0
    for cutoff in ("2015-07-01", "2016-01-01", "2016-07-01"):
        sup = build_supervision(index, cutoff, 1.0)
        snap = sup.snapshot
        sampler = ComparisonSampler(index, [sup], negatives_per_positive=4)
        ids = sampler.concepts()
        realized = {}  # known positives closed under ancestors, per anchor
        for u, m in sup.positives:
            for a in index.problem.ancestors(u, include_self=True):
                realized.setdefault(a, set()).update(index.method.ancestors(m, include_self=True))
            for b in index.method.ancestors(m, include_self=True):
                realized.setdefault(b, set()).update(index.problem.ancestors(u, include_self=True))
        for item in sampler.items:
            a_facet, c_facet = ("problem", "method") if item.direction == 0 else ("method", "problem")
            anchor = ids[a_facet][item.anchor]
            blocked = snap.co_concepts(anchor) | snap.window_co_concepts(anchor) | realized.get(anchor, set())
            expected = sorted(c for c in snap.concepts(c_facet) if c not in blocked)
            assert sorted(ids[c_facet][i] for i in sampler.pool(item)) == expected
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pool_rank_mapping_is_exact(seed):
    from scholarpath.retrieval.link_training import _Pool

    rng = np.random.default_rng(seed)
    candidates = np.sort(rng.choice(1000, size=300, replace=False)).astype(np.int32)
    blocked = np.sort(rng.choice(300, size=int(rng.integers(0, 299)), replace=False)).astype(np.int32)
    pool = _Pool(candidates, blocked)
    expected = np.delete(candidates, blocked)
    assert pool.size == len(expected)
    assert np.array_equal(pool.members(), expected)
    draws = pool.draw(rng, 50)
    assert set(draws.tolist()) <= set(expected.tolist())


def test_comparison_sampler_memory_is_per_anchor():
    """Pools are stored by exclusion: memory does not grow with #positives x #concepts."""
    import types

    def tax(facet, a, b, c):
        t = Taxonomy(facet, "Root", "")
        leaves = []
        for i in range(a):
            x = t.add_node(f"{facet}{i}")
            for j in range(b):
                y = t.add_node(f"{facet}{i}.{j}", parent=x)
                leaves += [t.add_node(f"{facet}{i}.{j}.{k}", parent=y) for k in range(c)]
        return t, leaves

    P, pl = tax("problem", 6, 6, 6)
    M, ml = tax("method", 6, 8, 8)
    rng = random.Random(0)
    pos = sorted({(rng.choice(pl), rng.choice(ml)) for _ in range(800)})
    concepts = {"problem": [n for n in P.nodes if n != P.root_id], "method": [n for n in M.nodes if n != M.root_id]}
    idx = types.SimpleNamespace(problem=P, method=M, taxonomy=lambda f: P if f == "problem" else M)
    snap = types.SimpleNamespace(concepts=lambda f: concepts[f], co_concepts=lambda n: set(),
                                 window_co_concepts=lambda n: set())
    sampler = ComparisonSampler(idx, [types.SimpleNamespace(positives=pos, snapshot=snap)])
    explicit = sum(len(sampler.pool(it)) for it in sampler.items) * 8  # one int64 pool per item
    assert len(sampler) == 2 * len(pos)
    assert sampler.nbytes() < 0.2 * explicit


def test_comparison_pool_has_no_historical_cooccurrence(synth):
    _, index = synth
    sup = build_supervision(index, "2016-01-01", 2.0)
    sampler = ComparisonSampler(index, [sup], negatives_per_positive=4)
    ids = sampler.concepts()
    for item in sampler.items:
        a_facet = "problem" if item.direction == 0 else "method"
        c_facet = "method" if item.direction == 0 else "problem"
        anchor = ids[a_facet][item.anchor]
        for i in sampler.pool(item):
            c = ids[c_facet][i]
            u, m = (anchor, c) if item.direction == 0 else (c, anchor)
            assert not sup.snapshot.shared(u, m)  # J_t(u, m-) is empty


# ---------------------------------------------------------------------------
# Training (Eq. 18), ranking, Eq. (12) bound, persistence
# ---------------------------------------------------------------------------


def test_loss_decreases_and_heldout_window_positives_rank_high(rt, traits):
    index, dates = traits
    train = [build_supervision(index, c, 1.0, None, dates) for c in ("2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01")]
    held = build_supervision(index, "2020-01-01", 5.0, None, dates)  # window covers the 2024 papers
    seen = {p for s in train for p in s.positives}
    pairs = [p for p in held.positives if p not in seen]
    assert len(seen) >= 8 and len(pairs) >= 8
    model = TemporalLinkPredictor(LinkPredictionConfig(epochs=20)).fit(index, train)
    assert len(model.loss_curve) == 20
    assert model.loss_curve[-1] < 0.5 * model.loss_curve[0]
    model.prepare(index)
    cands = ranking_candidates(held.snapshot, [u for u, _ in pairs])
    trained = evaluate_ranking(model, pairs, cands)
    prior = evaluate_ranking(EmbeddingLinkPredictor().prepare(index), pairs, cands)
    assert trained["auc"] >= 0.8, trained
    assert trained["mrr"] > prior["mrr"] + 0.2, (trained, prior)
    probs = evaluate_ranking(model, pairs, ranking_candidates(held.snapshot, [m for _, m in pairs], "problem"),
                             direction="problem")
    assert probs["auc"] >= 0.7, probs


def test_evaluate_ranking_filtered_metrics():
    class Fixed(LinkPredictor):
        def score(self, u, m):
            return {"m1": 3.0, "m2": 2.0, "m3": 1.0, "m4": 0.0}[m]

    pairs = [("u", "m2"), ("u", "m1")]
    r = evaluate_ranking(Fixed(), pairs, ["m1", "m2", "m3", "m4"], ks=(1, 2))
    # filtered: m1 does not compete with m2 (both positives of u) -> both rank first
    assert r["mrr"] == 1.0 and r["hit@1"] == 1.0 and r["auc"] == 1.0
    r2 = evaluate_ranking(Fixed(), [("u", "m3")], ["m1", "m2", "m3", "m4"], ks=(1, 3))
    assert r2["mrr"] == pytest.approx(1 / 3) and r2["hit@1"] == 0.0 and r2["hit@3"] == 1.0
    assert r2["auc"] == pytest.approx(1 / 3)
    # pairs without any competing candidate carry no ranking information and are not perfect hits
    r3 = evaluate_ranking(Fixed(), [("u", "m3"), ("u2", "m1"), ("u3", "m2")], {"u": ["m1", "m2", "m3", "m4"],
                                                                             "u2": [], "u3": ["m2"]})
    assert r3["n"] == 1.0 and r3["n_no_candidates"] == 2.0 and r3["mrr"] == pytest.approx(1 / 3)
    empty = evaluate_ranking(Fixed(), [("u", "m1")], {"u": []})
    assert empty["n"] == 0.0 and np.isnan(empty["mrr"]) and np.isnan(empty["auc"])


def test_train_link_predictor_report_bound_and_roundtrip(synth, rt, tmp_path):
    _, index = synth
    cfg = LinkPredictionConfig(cutoffs=("2015-07-01", "2016-01-01", "2016-07-01", "2017-01-01"),
                               horizon_years=1.0, **FAST)
    excluded = {index.method.leaves()[0]}
    model = train_link_predictor(index, cfg, rt.llm, exclude_concepts=excluded)
    rep = model.report
    assert isinstance(model, TemporalLinkPredictor) and "fallback" not in rep
    assert rep["verified_with_llm"] and len(rep["supervision"]) == 4 and rep["windows_exceeding_corpus"] == []
    # only windows that end by the validation cutoff train the validation model
    assert rep["validation"]["train_cutoffs"] == ["2015-07-01", "2016-01-01"]
    assert rep["validation"]["dropped_train_cutoffs"] == ["2016-07-01"]
    assert {"method_ranking", "problem_ranking", "embedding_prior"} <= set(rep["validation"])
    assert len(rep["final"]["loss_curve"]) == cfg.epochs
    assert rt.meter.component("link_training").calls > 0

    # Eq. (12): U(u, v) bounds s(u, m) for every method m below v (and roles exchanged).
    for u in index.problem.nodes:
        for v in index.method.nodes:
            best = max(model.score(u, m) for m in index.method.descendants(v, include_self=True))
            assert best <= model.upper_bound(u, v) + 1e-4
    for m in index.method.nodes:
        for v in index.problem.nodes:
            best = max(model.score(u, m) for u in index.problem.descendants(v, include_self=True))
            assert best <= model.upper_bound_problems(m, v) + 1e-4

    model.save(tmp_path / "lp")
    loaded = LinkPredictor.load(tmp_path / "lp")
    assert isinstance(loaded, TemporalLinkPredictor) and loaded.report["cutoffs"] == rep["cutoffs"]
    loaded.prepare(index)
    u, m = index.problem.leaves()[1], index.method.leaves()[1]
    assert loaded.score(u, m) == pytest.approx(model.score(u, m), rel=1e-5, abs=1e-5)
    assert np.allclose(load_predictor(tmp_path / "lp").encoders["method"].p["S"], model.encoders["method"].p["S"])


def test_query_local_anchor_and_errors(synth):
    _, index = synth
    model = TemporalLinkPredictor(LinkPredictionConfig(**FAST))
    with pytest.raises(RuntimeError):
        model.encode_query_concept("problem", np.zeros(index.dim, dtype=np.float32))
    sets = [build_supervision(index, c, 1.0) for c in ("2015-07-01", "2016-01-01")]
    model.fit(index, sets).prepare(index)
    u = index.problem.leaves()[0]
    d = index.vector(concept_key(u))
    z_desc = model.encode_query_concept("problem", d)  # description only (no taxonomy context)
    z_ctx = model.z_problem(u)
    assert z_desc.shape == z_ctx.shape == (FAST["embed_dim"],)
    model.set_z("anchor:q", "problem", z_desc)
    m = index.method.leaves()[0]
    assert np.isfinite(model.score("anchor:q", m))
    with pytest.raises(ValueError):
        model.encode_query_concept("problem", np.zeros(index.dim + 1, dtype=np.float32))
    with pytest.raises(ValueError):
        TemporalLinkPredictor(LinkPredictionConfig(**FAST)).fit(index, [])


# ---------------------------------------------------------------------------
# Regressions: validation leakage, corpus extent, empty supervision, fit contract
# ---------------------------------------------------------------------------


def _assert_no_training_paper_after_validation_cutoff(index, sets, report, dates):
    val = report["validation"]
    by_cutoff = {s.cutoff: s for s in sets}
    for c in val["train_cutoffs"]:
        s = by_cutoff[c]
        assert s.end <= val["cutoff"]  # the whole training window precedes the validation cutoff
        for papers in s.evidence.values():
            assert all(dates[p] <= val["cutoff"] for p in papers)


def test_validation_does_not_train_on_the_validation_window(rt, traits):
    """Overlapping windows (cutoffs 1 y apart, Delta = 2 y) must not leak validation papers into training."""
    index, dates = traits
    cfg = LinkPredictionConfig(cutoffs=("2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"),
                               horizon_years=2.0, verify_with_llm=False, **FAST)
    sets = [build_supervision(index, c, cfg.horizon_years, None, dates) for c in cfg.cutoffs]
    model = train_link_predictor(index, cfg, dates=dates, supervision=sets)
    val = model.report["validation"]
    assert val["train_cutoffs"] == ["2015-01-01", "2016-01-01"] and val["dropped_train_cutoffs"] == ["2017-01-01"]
    assert "method_ranking" in val and val["pairs"] > 0
    _assert_no_training_paper_after_validation_cutoff(index, sets, model.report, dates)
    # the dropped window did contain papers of the validation window
    dropped = {s.cutoff: s for s in sets}["2017-01-01"]
    assert any(dates[p] > val["cutoff"] for ev in dropped.evidence.values() for p in ev)


def test_validation_leak_regression_one_paper_two_labels(rt):
    """One window paper yields an ancestor label at t1 and a specific label at t2 > t1 (reviewer probe)."""
    P, M = Taxonomy("problem", "Root", ""), Taxonomy("method", "Root", "")
    U = P.add_node("Traffic forecasting", "forecasting road traffic")
    u = P.add_node("Traffic forecasting under sensor drift", "drift", parent=U)
    other = P.add_node("Other problem", "x")
    m, m2 = M.add_node("Optimal transport", "transport plans"), M.add_node("Kalman filtering", "state estimation")
    papers = {}
    for pid, date, pu, pm in [("old1", "2015-01-01", U, m2), ("old2", "2016-01-01", U, m2),
                              ("mthist", "2015-03-01", other, m), ("u_first", "2021-06-01", u, m2),
                              ("p", "2022-06-01", u, m)]:
        papers[pid] = Paper(pid, f"{P[pu].name} with {M[pm].name}", f"We use {M[pm].name} for {P[pu].name}.",
                            date=date)
        P.assign(pid, pu)
        M.assign(pid, pm)
    index = SemanticIndex.build(P, M, papers, rt.embedder)
    cfg = LinkPredictionConfig(embed_dim=8, hidden_dim=8, epochs=2)  # default cutoffs 2019-2022, Delta = 2 y
    sets = [build_supervision(index, c, cfg.horizon_years, None) for c in cfg.cutoffs]
    leaky = {s.cutoff for s in sets if any("p" in ev for ev in s.evidence.values())}
    assert leaky == {"2021-01-01", "2022-01-01"}  # the same paper labels two cutoffs
    model = train_link_predictor(index, cfg, supervision=sets)
    val = model.report["validation"]
    assert "2021-01-01" in val["dropped_train_cutoffs"] and "2021-01-01" not in val["train_cutoffs"]
    assert "method_ranking" not in val and val["skipped"] == "no training positives"


def test_windows_outside_the_corpus_are_reported(synth, caplog):
    _, index = synth
    with caplog.at_level("WARNING", logger="scholarpath.link_training"):
        sup = build_supervision(index, "2023-06-01", 2.0)
    assert sup.snapshot.exceeds_corpus and sup.summary()["window_exceeds_corpus"]
    assert sup.summary()["corpus_end"] < sup.end
    assert any("not within the indexed corpus" in r.getMessage() for r in caplog.records)
    assert not build_supervision(index, "2016-01-01", 1.0).snapshot.exceeds_corpus
    cfg = LinkPredictionConfig(cutoffs=("2016-01-01", "2023-06-01"), horizon_years=2.0, **FAST)
    with pytest.raises(ValueError, match="App. B.5"):
        train_link_predictor(index, cfg, strict=True)


def test_empty_supervision_falls_back_to_the_prior(synth, rt, tmp_path):
    """Default config on the stock synthetic benchmark: no labels -> description prior, not a crash."""
    _, index = synth
    model = train_link_predictor(index, LinkPredictionConfig(), rt.llm)
    assert isinstance(model, FallbackLinkPredictor) and isinstance(model, EmbeddingLinkPredictor)
    assert model.report["fallback"] == "empty_supervision"
    u, m = index.problem.leaves()[0], index.method.leaves()[0]
    assert model.score(u, m) == pytest.approx(EmbeddingLinkPredictor().prepare(index).score(u, m))
    model.save(tmp_path / "fb")
    assert isinstance(LinkPredictor.load(tmp_path / "fb"), EmbeddingLinkPredictor)
    with pytest.raises(EmptySupervisionError):
        train_link_predictor(index, LinkPredictionConfig(), supervision=[], strict=True)


def test_fit_accepts_plain_positive_pairs(synth):
    """LinkPredictor.fit(index, positives) contract (ARCHITECTURE 4.3)."""
    _, index = synth
    leaves_p, leaves_m = index.problem.leaves(), index.method.leaves()
    pairs = [(leaves_p[i], leaves_m[j]) for i, j in [(0, 0), (1, 2), (2, 1), (3, 3)]]
    model = TemporalLinkPredictor(LinkPredictionConfig(**FAST)).fit(index, pairs).prepare(index)
    assert len(model.loss_curve) == FAST["epochs"] and model.loss_curve[-1] < model.loss_curve[0]
    sup = static_supervision(index, pairs)
    assert sup.positives == sorted(pairs) and not sup.verified and not sup.snapshot.window
    at_cutoff = static_supervision(index, pairs, cutoff="2016-01-01")
    assert at_cutoff.snapshot.cutoff == "2016-01-01"
    with pytest.raises(TypeError):
        TemporalLinkPredictor(LinkPredictionConfig(**FAST)).fit(index, [1, 2])
    with pytest.raises(ValueError):
        TemporalLinkPredictor(LinkPredictionConfig(**FAST)).fit(index, [(leaves_m[0], leaves_p[0])])
