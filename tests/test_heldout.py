"""Tests for the Held-Out Hypothesis benchmark construction (App. C.1)."""

from __future__ import annotations

import csv
import json
from typing import Dict, List

import pytest

from scholarpath.data.corpus import Corpus
from scholarpath.data.heldout import (
    CONDITIONS,
    NEW_METHOD,
    NEW_PROBLEM,
    acronyms,
    annotation_agreement,
    apply_annotations,
    build_heldout_benchmark,
    check_leakage,
    enumerate_target_candidates,
    existing_endpoint_domain,
    export_annotation_sheet,
    generate_queries,
    load_heldout_benchmark,
    load_targets,
    read_annotations,
    reduce_corpus,
    save_heldout_benchmark,
    select_targets,
    supporting_evidence,
)
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.prompts.benchmark import counterpart_from_title, heldout_query_request
from scholarpath.runtime import Runtime
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.text import tokenize
from scholarpath.types import HYPOTHESIS_DISCOVERY

EMERGING = "dom-emerging"


class World:
    """Complete corpus + reference dual taxonomy with planted emerging ideas.

    Built from the synthetic benchmark: its held-out 2024 papers realize four
    (sub-problem, method) pairs never seen before. New concepts are planted:

    * NP0 -- new problem (pair 0) adapting an existing method;
    * NM1 -- new method (pair 1) for an existing problem;
    * NP2a / NP2b -- two new problems whose only counterpart is the same
      method (pair 2): they share an endpoint;
    * NM3 (pair 3, both papers) and NP3 (one paper of pair 3): they share a
      supporting paper;
    * BIG -- a 5-paper "emerging" problem, too large to be a target.
    """

    def __init__(self, seed: int = 0):
        bench = make_synthetic_benchmark(n_papers=200, n_heldout=4, seed=seed)
        self.bench = bench
        self.corpus = Corpus(list(bench.corpus) + list(bench.heldout_papers), name="world")
        self.problem = Taxonomy.from_dict(bench.problem_gold.to_dict())
        self.method = Taxonomy.from_dict(bench.method_gold.to_dict())
        leaf_p = {self.problem[n].name: n for n in self.problem.leaves()}
        leaf_m = {self.method[n].name: n for n in self.method.leaves()}
        domain_of_leaf = {n: self.problem[self.problem.parent(n)].name for n in self.problem.leaves()}
        self.category: Dict[str, str] = {}
        for p in bench.corpus:
            self.category[p.paper_id] = "dom-" + domain_of_leaf[leaf_p[bench.pair_of[p.paper_id][0]]]
        held: Dict[int, List[str]] = {}
        for p in bench.heldout_papers:
            self.category[p.paper_id] = EMERGING
            j = int(p.paper_id[4:6])
            held.setdefault(j, []).append(p.paper_id)
            s, m = bench.pair_of[p.paper_id]
            # Held-out papers keep their existing endpoints in the reference taxonomy.
            self.problem.assign(p.paper_id, leaf_p[s])
            self.method.assign(p.paper_id, leaf_m[m])
        self.pairs = bench.heldout_pairs
        s0, m0 = self.pairs[0]
        s1, m1 = self.pairs[1]
        s2, m2 = self.pairs[2]
        s3, m3 = self.pairs[3]
        self.np0 = self.problem.add_node(f"{s0} under Sensor Drift", "Forecasts that stay calibrated while sensors drift.",
                                         parent=leaf_p[s0], papers=held[0])
        self.nm1 = self.method.add_node(f"Drift-Aware {m1}", "A variant that adapts its parameters online.",
                                        parent=leaf_m[m1], papers=held[1])
        self.np2a = self.problem.add_node(f"{s2} at Night", "Nocturnal variant of the task.",
                                          parent=leaf_p[s2], papers=held[2][:1])
        self.np2b = self.problem.add_node(f"{s2} in Winter", "Seasonal variant of the task.",
                                          parent=leaf_p[s2], papers=held[2][1:])
        self.nm3 = self.method.add_node(f"Sparse {m3}", "A sparsified variant.", parent=leaf_m[m3], papers=held[3])
        self.np3 = self.problem.add_node(f"{s3} on Edge Devices", "Runs under tight memory budgets.",
                                         parent=leaf_p[s3], papers=held[3][1:])
        late = sorted((p for p in bench.corpus if p.year == 2023), key=lambda p: p.paper_id)[:5]
        self.big_papers = [p.paper_id for p in late]
        self.big = self.problem.add_node("Big Emerging Problem", "Too large to be held out.",
                                         parent=self.problem.root_id, papers=self.big_papers)
        self.taxonomies = (self.problem, self.method)
        self.leaf_p, self.leaf_m = leaf_p, leaf_m

    def cat(self, pid: str):
        return self.category.get(pid)


@pytest.fixture(scope="module")
def world() -> World:
    return World(seed=0)


@pytest.fixture()
def rt():
    return Runtime.from_config()


def _endpoints(targets) -> List[str]:
    return [e for t in targets for e in t.endpoints]


def test_candidates_respect_definition(world):
    cands = enumerate_target_candidates(world.taxonomies, world.corpus, world.cat, max_extent=4)
    news = {t.new_endpoint for t in cands}
    assert world.big not in news  # extent 5 > max_extent
    assert {world.np0, world.nm1, world.np2a, world.np2b, world.nm3, world.np3} <= news
    for t in cands:
        tax_new = world.problem if t.new_facet == "problem" else world.method
        tax_old = world.method if t.new_facet == "problem" else world.problem
        assert 1 <= len(tax_new.extent(t.new_endpoint)) <= 4
        assert len(tax_old.extent(t.existing_endpoint)) > 4
        # supporting papers = J(u, m); the source is the earliest one
        assert set(t.supporting_papers) == world.problem.extent(t.problem_id) & world.method.extent(t.method_id)
        dates = {p: world.corpus[p].date for p in t.supporting_papers}
        assert t.source_paper == min(t.supporting_papers, key=lambda p: (dates[p], p))
        assert t.target_domain == EMERGING and t.existing_domain != EMERGING
        assert t.existing_prior_papers >= 2
    by_new = {t.new_endpoint: t for t in cands}
    assert by_new[world.np0].setting == NEW_PROBLEM
    assert by_new[world.np0].method_id == world.leaf_m[world.pairs[0][1]]  # most specific counterpart
    assert by_new[world.nm1].setting == NEW_METHOD
    assert by_new[world.nm1].problem_id == world.leaf_p[world.pairs[1][0]]
    # Only the maximal small concept of a small subtree is a candidate.
    tiny = world.problem.add_node("Tiny Refinement", "", parent=world.np0, papers=world.problem.extent(world.np0))
    try:
        news = {t.new_endpoint for t in enumerate_target_candidates(world.taxonomies, world.corpus, world.cat)}
        assert world.np0 in news and tiny not in news
    finally:
        world.problem.remove_subtree(tiny)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_select_targets_disjoint_and_no_shared_endpoints(world, seed):
    targets = select_targets(world.taxonomies, world.corpus, world.cat, n_new_problem=10, n_new_method=10,
                             max_extent=4, seed=seed)
    assert targets
    papers = [p for t in targets for p in t.supporting_papers]
    assert len(papers) == len(set(papers))  # pairwise disjoint supporting papers
    ends = _endpoints(targets)
    assert len(ends) == len(set(ends))  # no shared endpoint
    news = {t.new_endpoint for t in targets}
    assert world.big not in news
    assert not ({world.np2a, world.np2b} <= news)  # same counterpart method
    assert not (world.np3 in news and world.nm3 in news)  # shared supporting paper
    assert len({t.target_id for t in targets}) == len(targets)


def test_select_targets_quota_and_seed(world):
    t1 = select_targets(world.taxonomies, world.corpus, world.cat, n_new_problem=1, n_new_method=1, seed=3)
    assert sorted(t.setting for t in t1) == sorted([NEW_PROBLEM, NEW_METHOD])
    t2 = select_targets(world.taxonomies, world.corpus, world.cat, n_new_problem=1, n_new_method=1, seed=3)
    assert [t.to_dict() for t in t1] == [t.to_dict() for t in t2]


def test_select_targets_with_dates_mapping(world):
    dates = {p.paper_id: p.date for p in world.corpus}
    targets = select_targets(world.taxonomies, dates, world.category, n_new_problem=2, n_new_method=2, seed=0)
    assert targets and all(t.target_domain == EMERGING for t in targets)


def test_reduce_corpus_global_and_per_target(world):
    targets = select_targets(world.taxonomies, world.corpus, world.cat, n_new_problem=2, n_new_method=2, seed=0)
    reduced, removed = reduce_corpus(world.corpus, targets)
    cutoff = min(t.source_date for t in targets)
    removed_set = set(removed)
    for t in targets:
        assert set(t.supporting_papers) <= removed_set
    for p in world.corpus:
        assert (p.paper_id in removed_set) == (p.paper_id in {x for t in targets for x in t.supporting_papers}
                                               or p.date >= cutoff)
    assert set(reduced.ids) | removed_set == set(world.corpus.ids)
    assert not set(reduced.ids) & removed_set
    # every remaining paper predates the earliest held-out idea
    assert all(p.date < cutoff for p in reduced)

    reduced_t, removed_t = reduce_corpus(world.corpus, targets, mode="per_target")
    expected = {x for t in targets for x in t.supporting_papers + t.followup_papers}
    assert set(removed_t) == expected
    for t in targets:
        assert all(world.corpus[p].date >= t.source_date for p in t.followup_papers)
    with pytest.raises(ValueError):
        reduce_corpus(world.corpus, targets, mode="nope")


def test_leakage_checks():
    abstract = ("We study flood forecasting, where rainfall remains hard to model. "
                "We adapt graph neural networks (GNNs) with hydrological message passing to river gauges.")
    names = ["Graph Neural Networks"]
    assert check_leakage("Could graph neural networks help here?", names).leaked
    assert check_leakage("Would GNNs be a good fit?", names, abstract).leaked
    assert check_leakage("Which models use graph neural message passing?", names).leaked  # terms graph+neural
    copied = check_leakage("What could add hydrological message passing to river gauges?", ["Kriging"], abstract)
    assert copied.leaked and "abstract phrase" in copied.reasons[0]
    clean = check_leakage("Forecasting floods from drifting sensors is hard. Which existing methods could help?",
                          names, abstract, allowed_text="Flood Forecasting under Sensor Drift")
    assert not clean.leaked
    assert {"GNN", "BSS"} <= acronyms("Graph Neural Networks") | acronyms(
        "blind source separation", "We use blind source separation (BSS) on logs.")


def test_prompt_mock_title_heuristic():
    title = "A Scalable Graph Neural Networks Approach for Flood Forecasting with rainfall and gnn"
    assert counterpart_from_title(title, "method") == "Graph Neural Networks"
    assert counterpart_from_title(title, "problem") == "Flood Forecasting"
    with pytest.raises(ValueError):
        heldout_query_request({"title": "t"}, "bad-setting", {"name": "x"})


@pytest.mark.parametrize("provide_counterpart", [False, True])
def test_generate_queries_mock_no_leak(world, rt, provide_counterpart):
    targets = select_targets(world.taxonomies, world.corpus, world.cat, n_new_problem=3, n_new_method=2, seed=0)
    queries = generate_queries(rt.llm, targets, world.corpus, provide_counterpart=provide_counterpart)
    assert len(queries) == len(targets)
    assert rt.meter.component("benchmark").calls == len(targets)
    for q, t in zip(queries, targets):
        md = q.metadata
        assert q.kind == HYPOTHESIS_DISCOVERY and q.gold_papers == []
        assert md["setting"] == t.setting and md["known_endpoint_id"] == t.new_endpoint
        assert md["heldout_counterpart_id"] == t.existing_endpoint
        assert md["heldout_counterpart_label"] == t.existing_name
        assert md["source_paper"] == t.source_paper and md["supporting_papers"] == t.supporting_papers
        assert q.reference_answer and md["rationale"]
        # the expected answer names the existing endpoint; the query never does
        assert set(tokenize(t.existing_name)) <= set(tokenize(q.reference_answer)) | {"approach"}
        assert md["leakage"]["leaked"] is False, md["leakage"]
        assert " ".join(tokenize(t.existing_name)) not in " ".join(tokenize(q.text))


def test_annotation_round_trip_csv_and_jsonl(world, rt, tmp_path):
    targets = select_targets(world.taxonomies, world.corpus, world.cat, n_new_problem=2, n_new_method=2, seed=0)
    queries = generate_queries(rt.llm, targets, world.corpus)
    assert len(queries) == 4
    sheet = export_annotation_sheet(queries, tmp_path / "sheet.csv", papers=world.corpus)
    with open(sheet, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["query_id"] for r in rows] == [q.query_id for q in queries]
    assert rows[0]["source_abstract"]
    # q0: both pass; q1: disagreement on 'predictable' resolved by discussion;
    # q2: unresolved disagreement; q3: both annotators fail 'no_reveal'.
    for r in rows:
        for a in ("a1", "a2"):
            for c in CONDITIONS:
                r[f"{a}_{c}"] = "1"
    rows[1]["a2_predictable"] = "0"
    rows[1]["final_predictable"] = "yes"
    rows[2]["a1_source_realizes"] = "no"
    rows[3]["a1_no_reveal"] = rows[3]["a2_no_reveal"] = "0"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    decisions = read_annotations(sheet)
    assert decisions[queries[1].query_id].disagreements == ["predictable"]
    assert decisions[queries[2].query_id].unresolved == ["source_realizes"]
    kept = apply_annotations(queries, sheet)
    assert [q.query_id for q in kept] == [queries[0].query_id, queries[1].query_id]
    assert kept[0].metadata["annotation"]["keep"] is True
    agree = annotation_agreement(sheet)
    assert agree["no_reveal"]["agreement"] == 1.0 and agree["predictable"]["agreement"] == 0.75

    jl = export_annotation_sheet(queries, tmp_path / "sheet.jsonl")
    recs = [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines()]
    for rec in recs:
        rec.update({f"{a}_{c}": True for a in ("a1", "a2") for c in CONDITIONS})
    recs[0]["a1_predictable"] = recs[0]["a2_predictable"] = False
    jl.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    assert [q.query_id for q in apply_annotations(queries, jl)] == [q.query_id for q in queries[1:]]


def test_build_save_load_benchmark(world, rt, tmp_path):
    bm = build_heldout_benchmark(world.corpus, world.taxonomies, rt.llm, primary_category_of=world.cat,
                                 n_new_problem=2, n_new_method=2, seed=1)
    targets = bm.targets
    assert len(bm.queries) == len(targets) == sum(bm.stats["targets_per_setting"].values())
    assert set(bm.removed_ids) >= {p for t in targets for p in t.supporting_papers}
    assert not set(bm.reduced_ids) & set(bm.removed_ids)
    assert bm.stats["llm_usage"]["calls"] == len(targets)
    for q, t in zip(bm.queries, targets):
        ev = q.metadata["evidence"]
        assert ev["new_endpoint_papers"] == 0  # the new endpoint is unfamiliar in the reduced corpus
        assert ev["counterpart_papers"] > 0
        assert set(ev["evidence_papers"]) <= set(bm.reduced_ids)
        assert ev == supporting_evidence(world.taxonomies, t, bm.reduced_ids, world.cat, world.corpus)
    paths = save_heldout_benchmark(bm, tmp_path / "heldout")
    assert {p.name for p in paths.values()} >= {"queries.jsonl", "removed_papers.json",
                                                 "reduced_corpus_ids.json", "targets.json"}
    loaded = load_heldout_benchmark(tmp_path / "heldout")
    assert [q.to_dict() for q in loaded.queries] == [q.to_dict() for q in bm.queries]
    assert loaded.removed_ids == bm.removed_ids and loaded.reduced_ids == bm.reduced_ids
    assert [t.to_dict() for t in load_targets(tmp_path / "heldout")] == [t.to_dict() for t in targets]
    assert loaded.stats["reduction_cutoff"] == bm.stats["reduction_cutoff"]


# ---------------------------------------------------------------------------
# Regressions: evidence after reduction, cross-domain rule, determinism, leakage
# ---------------------------------------------------------------------------


class Mini:
    """Hand-built dual taxonomy for targeted selection scenarios."""

    def __init__(self):
        self.P, self.M = Taxonomy("problem", "Root", ""), Taxonomy("method", "Root", "")
        self.fam = self.P.add_node("Family", "")
        self.oldp = self.P.add_node("Old problem", "", parent=self.fam)
        self.mfam = self.M.add_node("Method family", "")
        self.papers: List = []
        self.cat: Dict[str, str] = {}

    def add(self, pid, date, domain, problem, method):
        from scholarpath.types import Paper

        self.papers.append(Paper(pid, f"Paper {pid}", "An abstract.", date=date))
        self.cat[pid] = domain
        self.P.assign(pid, problem)
        self.M.assign(pid, method)

    @property
    def corpus(self) -> Corpus:
        return Corpus(self.papers)


def _later_target_world(nested: bool) -> Mini:
    """Target A (source 2020) and target B (source 2023) whose method's evidence dates from 2021-2022.

    ``nested=False``: B's method is a sibling of A's (the global reduction at 2020 erases its evidence).
    ``nested=True``: B's method is a child of A's, so A's follow-up papers (A's method extent dated
    after 2020) include B's evidence (the per-target reduction erases it too).
    """
    w = Mini()
    xa = w.M.add_node("Existing method A", "", parent=w.mfam)
    xb = w.M.add_node("Existing method B", "", parent=xa if nested else w.mfam)
    other = w.M.add_node("Other method", "", parent=w.mfam)
    for i in range(5):
        w.add(f"a{i}", f"2015-0{i + 1}-01", "src", w.oldp, xa)
    for i in range(4):
        w.add(f"b{i}", f"2021-0{i + 1}-01", "src", w.oldp, xb)
    for i in range(6):
        w.add(f"f{i}", f"2014-0{i + 1}-01", "src", w.oldp, other)
    va = w.P.add_node("New problem A", "", parent=w.fam)
    vb = w.P.add_node("New problem B", "", parent=w.fam)
    for i in range(2):
        w.add(f"tA{i}", f"2020-0{i + 1}-15", "tgt", va, xa)
        w.add(f"tB{i}", f"2023-0{i + 1}-15", "tgt", vb, xb)
    w.xa, w.xb = xa, xb
    return w


def _cross_domain_evidence_left(w_tax, target, reduced_ids, cat) -> int:
    tax = w_tax[1] if target.existing_facet == "method" else w_tax[0]
    return sum(1 for p in tax.extent(target.existing_endpoint) & set(reduced_ids) if cat(p) != target.target_domain)


@pytest.mark.parametrize("nested,mode", [(False, "global"), (True, "global"), (True, "per_target")])
def test_reduction_never_erases_an_existing_endpoint(nested, mode):
    w = _later_target_world(nested)
    cands = enumerate_target_candidates((w.P, w.M), w.corpus, w.cat)
    assert {t.existing_endpoint for t in cands} == {w.xa, w.xb}  # both are admissible on their own
    targets = select_targets((w.P, w.M), w.corpus, w.cat, n_new_problem=2, n_new_method=0, seed=0, reduction=mode)
    assert len(targets) == 1  # the two cannot coexist: one would erase the other's evidence
    reduced, _ = reduce_corpus(w.corpus, targets, mode=mode)
    for t in targets:
        assert t.reduction == mode and t.selection_rank == 0
        assert _cross_domain_evidence_left((w.P, w.M), t, reduced.ids, w.cat.get) >= 2
        assert t.existing_prior_papers >= 2


def test_per_target_reduction_keeps_unrelated_later_targets():
    """Sibling methods: the per-target reduction keeps B's 2021 evidence, so both targets coexist."""
    w = _later_target_world(nested=False)
    targets = select_targets((w.P, w.M), w.corpus, w.cat, n_new_problem=2, n_new_method=0, seed=0,
                             reduction="per_target")
    assert len(targets) == 2
    reduced, _ = reduce_corpus(w.corpus, targets, mode="per_target")
    for t in targets:
        assert _cross_domain_evidence_left((w.P, w.M), t, reduced.ids, w.cat.get) >= 2
        assert t.evidence_before == t.source_date


@pytest.mark.parametrize("mode", ["global", "per_target"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selected_targets_keep_cross_domain_evidence(world, mode, seed):
    targets = select_targets(world.taxonomies, world.corpus, world.cat, n_new_problem=10, n_new_method=10,
                             seed=seed, reduction=mode)
    assert targets
    reduced, _ = reduce_corpus(world.corpus, targets, mode=mode)
    for t in targets:
        assert _cross_domain_evidence_left(world.taxonomies, t, reduced.ids, world.cat) >= 2
        assert t.existing_domain not in (None, t.target_domain)
        ev = supporting_evidence(world.taxonomies, t, reduced.ids, world.cat, world.corpus)
        assert ev["counterpart_cross_domain_papers"] >= t.existing_prior_papers >= 2


def test_existing_endpoint_must_be_of_another_domain():
    def build(n_in_target: int, n_out: int) -> Mini:
        w = Mini()
        x = w.M.add_node("Method X", "", parent=w.mfam)
        for i in range(n_in_target):
            w.add(f"in{i}", f"2016-01-{i % 28 + 1:02d}", "cs.DB", w.oldp, x)
        for i in range(n_out):
            w.add(f"out{i}", f"2017-01-{i % 28 + 1:02d}", "stat.ML", w.oldp, x)
        v = w.P.add_node("New DB problem", "", parent=w.fam)
        for i in range(2):
            w.add(f"t{i}", f"2023-0{i + 1}-15", "cs.DB", v, x)
        return w

    dominated = build(n_in_target=40, n_out=2)  # literature overwhelmingly in the target domain
    assert enumerate_target_candidates((dominated.P, dominated.M), dominated.corpus, dominated.cat) == []
    tied = build(n_in_target=3, n_out=3)  # target domain co-dominant: not "of another domain"
    assert enumerate_target_candidates((tied.P, tied.M), tied.corpus, tied.cat) == []
    foreign = build(n_in_target=3, n_out=5)
    (t,) = enumerate_target_candidates((foreign.P, foreign.M), foreign.corpus, foreign.cat)
    assert (t.target_domain, t.existing_domain) == ("cs.DB", "stat.ML")
    assert (t.existing_prior_papers, t.existing_target_domain_papers) == (5, 3)
    assert existing_endpoint_domain({"b": 2, "a": 2, "tgt": 1}, "tgt") == ("a", 4, 1)  # ties broken by name
    assert existing_endpoint_domain({}, "tgt") == (None, 0, 0)


def test_existing_domain_independent_of_hash_seed():
    script = (
        "from scholarpath.data.corpus import Corpus\n"
        "from scholarpath.data.heldout import enumerate_target_candidates\n"
        "from scholarpath.taxonomy.tree import Taxonomy\n"
        "from scholarpath.types import Paper\n"
        "P, M = Taxonomy('problem', 'Root', ''), Taxonomy('method', 'Root', '')\n"
        "fam = P.add_node('Fam', ''); old = P.add_node('Old', '', parent=fam)\n"
        "x = M.add_node('X', '', parent=M.add_node('MF', ''))\n"
        "papers, cat = [], {}\n"
        "for i, c in enumerate(['math.OC', 'eess.SP', 'stat.ML', 'cs.DB']):\n"
        "    pid = 'prior-' + c; papers.append(Paper(pid, pid, '', date=f'2016-0{i + 1}-01')); cat[pid] = c\n"
        "    P.assign(pid, old); M.assign(pid, x)\n"
        "v = P.add_node('New', '', parent=fam)\n"
        "for i in range(2):\n"
        "    pid = f't{i}'; papers.append(Paper(pid, pid, '', date=f'2023-0{i + 1}-01')); cat[pid] = 'q-bio.PE'\n"
        "    P.assign(pid, v); M.assign(pid, x)\n"
        "print([t.existing_domain for t in enumerate_target_candidates((P, M), Corpus(papers), cat)])\n"
    )
    import os
    import subprocess
    import sys

    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    outs = set()
    for seed in ("0", "1", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=src)
        outs.add(subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                                check=True).stdout.strip())
    assert outs == {"['cs.DB']"}


def test_leakage_paraphrases_are_flagged():
    abstract = "We adapt blind source separation (BSS) to cascading point processes of infections."
    names = ["Blind Source Separation"]
    for text in ("Which technique separating blind sources could recover hidden infection chains?",  # 2 of 3 terms
                 "Would bss-style unmixing help recover hidden infection chains?",  # lower-case acronym
                 "Could ICA or BSS recover them?"):
        assert check_leakage(text, names, abstract).leaked, text
    long_name = ["Blind Source Separation for Cascading Point Processes"]
    assert check_leakage("Could blind source separation recover hidden chains?", long_name).leaked
    # terms of the new endpoint's own description are not leaks
    allowed = "Source attribution of outbreaks: tracing the source of an outbreak."
    assert not check_leakage("How can we trace the source of an outbreak from partial logs?", names,
                             allowed_text=allowed).leaked
    # stopword-like initials are not matched in lower case ("And" from "Adaptive Neural Decomposition")
    assert not check_leakage("Model infections and recoveries jointly?", ["Adaptive Neural Decomposition"]).leaked
    assert check_leakage("Would AND help?", ["Adaptive Neural Decomposition"]).leaked


def test_oversampling_and_quota(world, rt, tmp_path):
    bm = build_heldout_benchmark(world.corpus, world.taxonomies, rt.llm, primary_category_of=world.cat,
                                 n_new_problem=1, n_new_method=1, oversample=2.0, seed=0)
    assert bm.stats["requested"] == {NEW_PROBLEM: 1, NEW_METHOD: 1}
    assert bm.stats["sampled"] == {NEW_PROBLEM: 2, NEW_METHOD: 2}
    assert bm.stats["unsupported_targets"] == []
    assert sum(bm.stats["targets_per_setting"].values()) == len(bm.queries) >= 3
    ranks = sorted(q.metadata["selection_rank"] for q in bm.queries)
    assert ranks == sorted(set(ranks)) and ranks[0] == 0
    sheet = export_annotation_sheet(bm.queries, tmp_path / "sheet.jsonl")
    recs = [json.loads(line) for line in sheet.read_text(encoding="utf-8").splitlines()]
    for rec in recs:
        rec.update({f"{a}_{c}": 1 for a in ("a1", "a2") for c in CONDITIONS})
    # the first-ranked new-problem query fails validation: the next one in seeded order replaces it
    first_np = min((q for q in bm.queries if q.metadata["setting"] == NEW_PROBLEM),
                   key=lambda q: q.metadata["selection_rank"])
    for rec in recs:
        if rec["query_id"] == first_np.query_id:
            rec["a1_predictable"] = rec["a2_predictable"] = 0
    sheet.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    kept = apply_annotations(bm.queries, sheet, quota=bm.stats["requested"])
    assert sorted(q.metadata["setting"] for q in kept) == sorted([NEW_PROBLEM, NEW_METHOD])
    assert first_np.query_id not in {q.query_id for q in kept}
    for q in kept:
        same = [r for r in bm.queries if r.metadata["setting"] == q.metadata["setting"]
                and r.query_id != first_np.query_id]
        assert q.metadata["selection_rank"] == min(r.metadata["selection_rank"] for r in same)
    with pytest.raises(ValueError):
        build_heldout_benchmark(world.corpus, world.taxonomies, rt.llm, oversample=0.5)


def test_build_drops_queries_whose_counterpart_vanished(monkeypatch, rt):
    """Safety net: if selection and reduction disagree, unsupported targets are dropped, not kept."""
    import scholarpath.data.heldout as heldout

    w = _later_target_world(nested=False)
    original = heldout.select_targets

    def mismatched(*args, **kwargs):  # select for the per-target reduction, then reduce globally
        kwargs["reduction"] = "per_target"
        return original(*args, **kwargs)

    monkeypatch.setattr(heldout, "select_targets", mismatched)
    bm = heldout.build_heldout_benchmark(w.corpus, (w.P, w.M), rt.llm, primary_category_of=w.cat,
                                         n_new_problem=2, n_new_method=0, reduction="global")
    assert len(bm.targets) == 2 and len(bm.stats["unsupported_targets"]) == 1
    assert len(bm.queries) == 1
    (q,) = bm.queries
    assert q.metadata["evidence"]["counterpart_papers"] > 0
    assert q.metadata["target_id"] not in bm.stats["unsupported_targets"]
