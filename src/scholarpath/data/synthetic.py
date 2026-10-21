"""Deterministic synthetic scholarly repository for offline runs.

Generates papers over a two-level problem hierarchy (domain -> sub-problem)
and a two-level method hierarchy (family -> method). Each paper studies one
sub-problem with one method; titles, abstracts and section chunks are
templated from facet-specific keywords so that both lexical and embedding
heuristics (and the offline :class:`~scholarpath.llm.MockLLM`) recover the
structure. The generator also returns

* gold problem / method taxonomies with paper assignments;
* evidence queries ("How has <method> been used for <sub-problem>?") whose
  gold papers are J(u, m);
* held-out hypothesis queries: (problem, method) pairs first realized after
  ``heldout_year``; their supporting papers are removed from the returned
  ``corpus`` (they are kept in ``heldout_papers``), mirroring Appendix C.1.

The data is entirely artificial and only meant to exercise the code paths.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from scholarpath.data.corpus import Corpus
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.types import EVIDENCE, HYPOTHESIS_DISCOVERY, Chunk, Paper, Query

PROBLEMS: Dict[str, Dict[str, List[str]]] = {
    "Drug Discovery": {
        "Drug-Target Interaction": ["binding", "affinity", "ligand", "receptor", "docking"],
        "Molecular Property Prediction": ["toxicity", "solubility", "molecule", "descriptor", "admet"],
    },
    "Environmental Forecasting": {
        "Air Quality Forecasting": ["pollution", "pm25", "ozone", "monitoring", "emission"],
        "Flood Forecasting": ["rainfall", "river", "flood", "hydrology", "discharge"],
    },
    "Database Systems": {
        "Cardinality Estimation": ["cardinality", "selectivity", "join", "histogram", "optimizer"],
        "Index Tuning": ["index", "workload", "tuning", "physical", "advisor"],
    },
    "Epidemiology": {
        "Outbreak Detection": ["outbreak", "surveillance", "incidence", "syndromic", "alert"],
        "Contact Tracing": ["contact", "tracing", "transmission", "superspreader", "exposure"],
    },
}

METHODS: Dict[str, Dict[str, List[str]]] = {
    "Graph Learning": {
        "Graph Neural Networks": ["gnn", "message", "passing", "aggregation", "node"],
        "Graph Contrastive Learning": ["contrastive", "augmentation", "views", "infonce", "pretraining"],
    },
    "Probabilistic Modeling": {
        "Gaussian Processes": ["kernel", "gaussian", "posterior", "covariance", "uncertainty"],
        "Bayesian Optimization": ["acquisition", "surrogate", "bayesian", "exploration", "expected"],
    },
    "Signal Processing": {
        "Blind Source Separation": ["separation", "mixture", "independent", "sources", "unmixing"],
        "Compressed Sensing": ["sparse", "sensing", "recovery", "measurements", "incoherence"],
    },
    "Optimization Methods": {
        "Optimal Transport": ["transport", "wasserstein", "coupling", "sinkhorn", "marginals"],
        "Evolutionary Search": ["evolutionary", "mutation", "crossover", "population", "fitness"],
    },
}

_VARIANTS = ["A Scalable", "An Efficient", "A Robust", "A Unified", "Towards", "Revisiting", "Learning", "Fast"]
_DATASETS = ["benchmark", "real-world", "large-scale", "public", "synthetic", "clinical", "industrial"]


@dataclass
class SyntheticBenchmark:
    corpus: Corpus
    problem_gold: Taxonomy
    method_gold: Taxonomy
    evidence_queries: List[Query] = field(default_factory=list)
    hypothesis_queries: List[Query] = field(default_factory=list)
    heldout_papers: List[Paper] = field(default_factory=list)
    heldout_pairs: List[Tuple[str, str]] = field(default_factory=list)  # (sub-problem, method) names
    pair_of: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # paper -> (sub-problem, method)


def _sentence(rng: random.Random, words: Sequence[str], k: int = 3) -> str:
    ws = rng.sample(list(words), min(k, len(words)))
    return " ".join(ws)


def _make_paper(
    rng: random.Random, pid: str, sub: str, sub_kw: List[str], meth: str, meth_kw: List[str], year: int
) -> Paper:
    variant = rng.choice(_VARIANTS)
    title = f"{variant} {meth} Approach for {sub} with {sub_kw[0]} and {meth_kw[0]}"
    abstract = (
        f"We study {sub.lower()}, where {_sentence(rng, sub_kw)} remain challenging. "
        f"We propose a {meth.lower()} model based on {_sentence(rng, meth_kw)} "
        f"to capture {_sentence(rng, sub_kw, 2)}. "
        f"Experiments on {rng.choice(_DATASETS)} data show that {meth_kw[1]} {meth_kw[2]} improves "
        f"{sub_kw[1]} estimation over prior {sub_kw[2]} baselines."
    )
    sections = [
        ("Introduction", f"The problem of {sub.lower()} concerns {', '.join(sub_kw)}. "
                         f"Prior work on {sub_kw[0]} {sub_kw[3]} relies on hand-crafted rules. "
                         f"We address {sub_kw[1]} and {sub_kw[4]} jointly."),
        ("Method", f"Our {meth.lower()} formulation uses {', '.join(meth_kw)}. "
                   f"Specifically, {meth_kw[0]} {meth_kw[1]} is applied to {sub_kw[0]} signals, "
                   f"and we assume {meth_kw[3]} {meth_kw[4]} holds for {sub_kw[2]} data. "
                   f"The {meth_kw[2]} step is repeated until convergence."),
        ("Experiments", f"We evaluate on {rng.choice(_DATASETS)} {sub_kw[0]} datasets measuring "
                        f"{sub_kw[1]} and {sub_kw[3]}. The {meth_kw[0]} variant reduces error by "
                        f"{rng.randint(5, 40)} percent while {meth_kw[4]} remains stable."),
        ("Conclusion", f"{meth} is effective for {sub.lower()}; future work will study {sub_kw[4]}."),
    ]
    chunks = [Chunk(f"{pid}#{i}", pid, h, t, i) for i, (h, t) in enumerate(sections)]
    month = rng.randint(1, 12)
    return Paper(
        paper_id=pid,
        title=title,
        abstract=abstract,
        year=year,
        date=f"{year}-{month:02d}-{rng.randint(1, 28):02d}",
        categories=["synthetic"],
        primary_category="synthetic",
        arxiv_id=None,
        chunks=chunks,
        metadata={"subproblem": sub, "method": meth},
    )


def make_synthetic_benchmark(
    n_papers: int = 240,
    density: float = 0.45,
    n_heldout: int = 2,
    heldout_year: int = 2024,
    seed: int = 0,
) -> SyntheticBenchmark:
    """Generate a synthetic repository with gold taxonomies and queries."""
    rng = random.Random(seed)
    subs = [(d, s, kw) for d, ss in PROBLEMS.items() for s, kw in ss.items()]
    meths = [(f, m, kw) for f, ms in METHODS.items() for m, kw in ms.items()]
    pairs = [(s, m) for _, s, _ in subs for _, m, _ in meths]
    rng.shuffle(pairs)
    n_applied = max(len(subs), int(density * len(pairs)))
    applied = pairs[:n_applied]
    # Held-out pairs: unexplored before heldout_year (new problem-method links).
    heldout = pairs[n_applied : n_applied + n_heldout]
    sub_kw = {s: kw for _, s, kw in subs}
    meth_kw = {m: kw for _, m, kw in meths}

    papers: List[Paper] = []
    pair_of: Dict[str, Tuple[str, str]] = {}
    for i in range(n_papers):
        s, m = applied[i % len(applied)] if i < len(applied) else rng.choice(applied)
        pid = f"syn{i:04d}"
        year = rng.randint(2015, heldout_year - 1)
        papers.append(_make_paper(rng, pid, s, sub_kw[s], m, meth_kw[m], year))
        pair_of[pid] = (s, m)
    heldout_papers: List[Paper] = []
    for j, (s, m) in enumerate(heldout):
        for r in range(2):
            pid = f"held{j:02d}_{r}"
            heldout_papers.append(_make_paper(rng, pid, s, sub_kw[s], m, meth_kw[m], heldout_year))
            pair_of[pid] = (s, m)

    corpus = Corpus(papers, name="synthetic")

    def gold(hier: Dict[str, Dict[str, List[str]]], facet: str, idx: int) -> Taxonomy:
        tax = Taxonomy(facet=facet, root_name="Synthetic Repository",
                       root_definition=f"All {facet}s studied in the synthetic repository.")
        leaf_ids: Dict[str, str] = {}
        for top, children in hier.items():
            tid = tax.add_node(top, f"{facet.title()}s in {top.lower()}.")
            for child, kw in children.items():
                leaf_ids[child] = tax.add_node(child, f"{child}: work on {', '.join(kw[:3])}.", parent=tid)
        for p in papers:
            tax.assign(p.paper_id, leaf_ids[pair_of[p.paper_id][idx]])
        return tax

    problem_gold = gold(PROBLEMS, "problem", 0)
    method_gold = gold(METHODS, "method", 1)

    evidence: List[Query] = []
    for k, (s, m) in enumerate(sorted(set(pair_of[p.paper_id] for p in papers))):
        gold_ids = [p.paper_id for p in papers if pair_of[p.paper_id] == (s, m)]
        evidence.append(Query(
            query_id=f"syn-ev-{k:03d}",
            text=f"How has {m.lower()} been used for {s.lower()}?",
            kind=EVIDENCE,
            benchmark="synthetic",
            gold_papers=gold_ids,
            rubric=[],
            metadata={"problem": s, "method": m},
        ))
    hypotheses: List[Query] = []
    for k, (s, m) in enumerate(heldout):
        hypotheses.append(Query(
            query_id=f"syn-hyp-{k:03d}",
            text=f"Which other methods could be applied to {s.lower()} problems involving "
                 f"{', '.join(sub_kw[s][:2])}?",
            kind=HYPOTHESIS_DISCOVERY,
            benchmark="synthetic-heldout",
            gold_papers=[],
            reference_answer=m,
            metadata={"setting": "new_problem_existing_method", "known_endpoint": s,
                      "heldout_counterpart": m, "rationale": f"{m} can model {sub_kw[s][0]} structure."},
        ))
    return SyntheticBenchmark(
        corpus=corpus,
        problem_gold=problem_gold,
        method_gold=method_gold,
        evidence_queries=evidence,
        hypothesis_queries=hypotheses,
        heldout_papers=heldout_papers,
        heldout_pairs=heldout,
        pair_of=pair_of,
    )
