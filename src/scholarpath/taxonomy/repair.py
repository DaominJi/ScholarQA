"""Audit-guided repair (Section 3.3, Appendix A.2 and B.2).

Repair removes the errors that the audit and the router signals have
exposed, so that the next certification attempt succeeds without reading
many more papers. Three LLM operations are used, each billed to its own cost
component (Equation 15):

* **DISCOVER** (``discovery``) targets missing concepts. Its pool is the
  residual leaf bottom_v (papers the router cannot place) together with the
  papers an earlier audit flagged as having a missing concept. When the pool
  is non-empty the builder draws s^disc_v = min{ceil(eta_A |pool|), s_max}
  papers by proportional stratified sampling across regions
  (s_g = ceil(s^disc n_g / |pool|); rounding up adds at most G_v - 1 papers),
  so a missing concept held by a fraction rho of the pool is absent from the
  draw with probability at most (1 - rho)^{s^disc} <= exp(-rho s^disc)
  (Proposition 2). The LLM assigns each drawn paper to an existing child,
  refines a child definition, or adds a new child.
* **CORRECT** (``correction``) targets routing errors. For each validation or
  audited paper whose router assignment disagrees with its LLM label, the
  LLM label is restored and the LLM labels up to kappa nearest *unread*
  neighbours in the same region (s^corr <= kappa n^dis). Such disagreements
  lie near the boundaries between sibling children, so these labels sharpen
  the router where it errs.
* **Schema-dependent relabeling** (``relabeling``, Appendix B.2) reconciles
  previously verified labels affected by a schema revision: verified papers
  with a missing concept, papers labelled with a refined child, and papers
  whose embedding is at least as close to a new child as to their own
  children.

The functions here are stateless; :class:`~scholarpath.taxonomy.builder.NodeWiseBuilder`
decides when to call them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from scholarpath.taxonomy.llm_ops import LabelBatch, PaperLabel, TaxonomyLLM, resolve_new_children
from scholarpath.taxonomy.tree import Taxonomy

PoolByRegion = Union[Mapping[int, Sequence[str]], Sequence[Sequence[str]]]


# ---------------------------------------------------------------------------
# DISCOVER: proportional stratified sample of the discovery pool (App. A.2)
# ---------------------------------------------------------------------------


def discovery_sample_size(pool_size: int, eta_A: float, s_max: int) -> int:
    """s^disc_v = min{ceil(eta_A |pool|), s_max} (0 for an empty pool)."""
    if pool_size <= 0:
        return 0
    return int(max(1, min(math.ceil(float(eta_A) * pool_size), int(s_max))))


def stratified_allocation(sizes: Sequence[int], s: int) -> List[int]:
    """Proportional allocation s_g = min{n_g, ceil(s n_g / sum_g n_g)} (App. A.2)."""
    total = int(sum(sizes))
    if total <= 0 or s <= 0:
        return [0 for _ in sizes]
    return [int(min(n, math.ceil(s * n / total))) for n in sizes]


def discovery_draws(
    pool_by_region: PoolByRegion,
    eta_A: float,
    s_max: int,
    rng: np.random.Generator,
    max_draws: Optional[int] = None,
) -> List[str]:
    """Draw the DISCOVER sample from the pool by proportional stratified sampling.

    ``pool_by_region`` maps each region to its part of the discovery pool
    (residual leaf plus audit-flagged missing-concept papers). Within every
    region the s_g papers are drawn uniformly without replacement, so the
    draw contains at least s^disc papers (at most s^disc + G - 1) and
    satisfies the bound of Proposition 2. ``max_draws`` optionally truncates
    the sample (budget cap); truncation keeps the per-region uniformity.
    """
    if isinstance(pool_by_region, Mapping):
        keys = sorted(pool_by_region)
        parts = [sorted(set(pool_by_region[k])) for k in keys]
    else:
        parts = [sorted(set(p)) for p in pool_by_region]
    sizes = [len(p) for p in parts]
    s_disc = discovery_sample_size(sum(sizes), eta_A, s_max)
    alloc = stratified_allocation(sizes, s_disc)
    draws: List[str] = []
    for part, s_g in zip(parts, alloc):
        if s_g <= 0:
            continue
        idx = rng.choice(len(part), size=s_g, replace=False)
        draws.extend(part[i] for i in sorted(idx.tolist()))
    if max_draws is not None and len(draws) > max_draws:
        keep = rng.choice(len(draws), size=max(0, int(max_draws)), replace=False)
        draws = [draws[i] for i in sorted(keep.tolist())]
    return draws


@dataclass
class DiscoverResult:
    """Outcome of one DISCOVER invocation (schema changes already applied)."""

    labels: Dict[str, PaperLabel] = field(default_factory=dict)
    new_children: List[str] = field(default_factory=list)  # node ids added
    refined: List[str] = field(default_factory=list)  # node ids whose definition changed

    @property
    def schema_changed(self) -> bool:
        return bool(self.new_children or self.refined)


def discover(
    ops: TaxonomyLLM,
    tax: Taxonomy,
    node_id: str,
    paper_ids: Sequence[str],
    max_children: Optional[int] = None,
    component: str = "discovery",
) -> DiscoverResult:
    """DISCOVER(v): the LLM assigns / refines / adds children for the drawn papers.

    New children are added to ``tax`` under ``node_id`` (merged by name with
    existing ones, capped at ``max_children``) and refined definitions are
    written back; the returned labels refer to node ids.
    """
    out = DiscoverResult()
    if not paper_ids:
        return out
    before = set(tax.children(node_id))
    batch: LabelBatch = ops.discover(tax, node_id, tax.children(node_id), list(paper_ids), component=component)
    for cid, definition in batch.refined.items():
        if cid in tax.nodes and definition and definition != tax.nodes[cid].definition:
            tax.nodes[cid].definition = definition
            out.refined.append(cid)
    resolve_new_children(tax, node_id, batch.new_children, batch.labels, max_children)
    out.new_children = [c for c in tax.children(node_id) if c not in before]
    out.labels = batch.labels
    return out


# ---------------------------------------------------------------------------
# CORRECT: kappa nearest unread neighbours of disagreeing papers
# ---------------------------------------------------------------------------


def correction_targets(
    disagreeing: Iterable[str],
    region_of: Mapping[str, int],
    vectors: Mapping[str, np.ndarray],
    unread: Iterable[str],
    kappa: int,
    max_targets: Optional[int] = None,
) -> List[str]:
    """Up to ``kappa`` nearest unread neighbours (same region) of each disagreeing paper.

    Neighbours are ranked by cosine similarity (vectors are L2-normalised).
    A neighbour already selected for an earlier disagreeing paper is not
    selected twice, so the call labels at most kappa * n^dis papers.
    """
    if kappa <= 0:
        return []
    by_region: Dict[int, List[str]] = {}
    for p in sorted(set(unread)):
        if p in region_of:
            by_region.setdefault(region_of[p], []).append(p)
    mats = {g: np.stack([vectors[p] for p in ids]) for g, ids in by_region.items() if ids}
    chosen: List[str] = []
    taken: Set[str] = set()
    for p in sorted(set(disagreeing)):
        g = region_of.get(p)
        if g is None or g not in mats:
            continue
        ids = by_region[g]
        sims = mats[g] @ np.asarray(vectors[p], dtype=np.float32)
        picked = 0
        for i in np.argsort(-sims, kind="stable"):
            q = ids[int(i)]
            if q == p or q in taken:
                continue
            chosen.append(q)
            taken.add(q)
            picked += 1
            if picked >= kappa:
                break
        if max_targets is not None and len(chosen) >= max_targets:
            return chosen[: int(max_targets)]
    return chosen


def correct(
    ops: TaxonomyLLM,
    tax: Taxonomy,
    node_id: str,
    targets: Sequence[str],
    component: str = "correction",
) -> Dict[str, PaperLabel]:
    """CORRECT(v, kappa): LLM labels of the boundary neighbours (existing schema only)."""
    if not targets:
        return {}
    batch = ops.label(tax, node_id, tax.children(node_id), list(targets), component=component, allow_new=False)
    return batch.labels


# ---------------------------------------------------------------------------
# Schema-dependent relabeling (Appendix B.2)
# ---------------------------------------------------------------------------


def relabel_candidates(
    verified: Mapping[str, Tuple[Sequence[str], bool]],
    new_children: Sequence[str],
    refined_children: Sequence[str],
    paper_vectors: Mapping[str, np.ndarray],
    child_vectors: Mapping[str, np.ndarray],
    exclude: Iterable[str] = (),
    margin: float = 0.0,
) -> List[str]:
    """Verified papers whose labels a schema revision may have invalidated.

    ``verified`` maps a paper to (LLM children, missing flag). Selected are

    * papers with a missing concept (a new or refined child may now fit);
    * papers labelled with a refined child (the definition changed);
    * papers whose cosine similarity to some new child is at least their
      best similarity to their own children minus ``margin`` (embedding
      filter for papers likely to belong to a new child).

    Papers in ``exclude`` (e.g. just labelled under the revised schema) are
    skipped.
    """
    skip = set(exclude)
    refined = set(refined_children)
    new_vecs = [np.asarray(child_vectors[c]) for c in new_children if c in child_vectors]
    new_mat = np.stack(new_vecs) if new_vecs else None
    out: List[str] = []
    for p in sorted(verified):
        if p in skip:
            continue
        kids, missing = verified[p]
        if missing or not kids:
            out.append(p)
            continue
        if refined.intersection(kids):
            out.append(p)
            continue
        if new_mat is None or p not in paper_vectors:
            continue
        x = np.asarray(paper_vectors[p])
        own = [float(np.asarray(child_vectors[c]) @ x) for c in kids if c in child_vectors]
        best_new = float((new_mat @ x).max())
        if not own or best_new >= max(own) - margin:
            out.append(p)
    return out


def relabel(
    ops: TaxonomyLLM,
    tax: Taxonomy,
    node_id: str,
    candidates: Sequence[str],
    component: str = "relabeling",
) -> Dict[str, PaperLabel]:
    """Re-read the candidates against the revised schema (no new children)."""
    if not candidates:
        return {}
    batch = ops.label(tax, node_id, tax.children(node_id), list(candidates), component=component, allow_new=False)
    return batch.labels
