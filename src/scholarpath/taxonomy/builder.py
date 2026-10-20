"""Node-wise confidence-bounded taxonomy builder (Algorithm 1, Section 3).

Definition 1 asks for a taxonomy that reads as few papers with the LLM as
possible while certifying that the node-wise organization loss

    L_v = m_v + (1 - m_v) e_v                                  (Section 3.1)

(m_v: fraction of papers of v needing a child the schema lacks -- *missing
concepts*; e_v: fraction of routing errors among the others) is at most eps
at every node with confidence 1 - delta_v, with sum_v delta_v <= delta.

:class:`NodeWiseBuilder` processes the queue of nodes top-down; the
processing of one node v is an *episode* (Algorithm 1, lines 3-21):

* **InitNode** (line 3): allocate delta_v, group P_v into
  G_v = min{8, floor(n_v / 24)} k-means regions, let the LLM read a seed
  batch from every region (initial schema + labels), train the router
  f_theta_v on the training split of the LLM-labelled papers D_v and route
  P_v \\ D_v.
* **Stage 1 -- quality-targeted Bayesian acquisition** (lines 5-10,
  Section 3.2): if the loss estimate F_{v,t}(0) = L^_{v,t} exceeds eps (or
  the last audit found errors), Thompson-sample the regional response
  curves, plan the batch (Eq. 3 / Eq. 10), let the LLM label the drawn
  papers (updating schema and labels), retrain and reroute, and update the
  slope posteriors.
* **Stage 2 -- audit-guided repair** (lines 11-13, Section 3.3): DISCOVER
  for missing concepts and CORRECT for routing errors when their pools reach
  the repair threshold, then retrain and reroute.
* **Stage 3 -- audit-based certification** (lines 14-20, Section 3.3):
  freeze snapshot j, audit uniform batches of unverified papers and certify
  once U_{v,j,a} <= eps (Eq. 4); otherwise j <- j + 1 and the audited
  failures feed the next cycle.

Rounds of Stage 1 + Stage 2 repeat until the forecast is within eps ("the
forecast F_{v,t}(z) <= eps only sizes batches and triggers the audit",
Section 3.2), nothing is left to read, or ``max_rounds`` rounds have run;
then Stage 3 audits the snapshot left by the last repair.

After the episode the final assignments are written to the taxonomy, the
episode is recorded in the history H (which updates the transferred prior,
Eq. 14) and the children above the size limit are enqueued (line 21).

The proxies used by Stage 1 (Section 3.2) are

* m^_{vg,t}: share of region g in the residual leaf bottom_v (the router
  abstains, or the LLM found a missing concept), i.e. the abstention rate
  a_{v,t} = |bottom_v| / n_v restricted to the region;
* e^_{vg,t}: router/LLM disagreement on held-out verified papers (the
  validation split, refined by the audited papers of earlier snapshots),
  shrunk toward the node-level rate, times the share of placed papers of
  the region that are still unverified (verified papers keep their LLM
  labels and cannot be misrouted);
* L^_{v,t} = sum_g pi_{vg} [m^_{vg,t} + (1 - m^_{vg,t}) e^_{vg,t}].

They only decide what to read and repair; only the audit certifies a node.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from scholarpath.config import BuilderConfig
from scholarpath.data.corpus import Corpus
from scholarpath.embeddings import l2_normalize
from scholarpath.llm.base import LLMClient
from scholarpath.llm.mock import stable_hash
from scholarpath.taxonomy.allocation import expected_token_cost, plan_batch
from scholarpath.taxonomy.certification import BUDGET, AuditOutcome, DeltaAllocator, audit_snapshot
from scholarpath.taxonomy.llm_ops import PaperLabel, TaxonomyLLM, resolve_new_children
from scholarpath.taxonomy.regions import Regions
from scholarpath.taxonomy.repair import (
    correct,
    correction_targets,
    discover,
    discovery_draws,
    relabel,
    relabel_candidates,
)
from scholarpath.taxonomy.response_model import (
    RegionDescriptor,
    RegionResponse,
    TransferredPrior,
    logit_measurement_var,
    region_dispersion,
)
from scholarpath.taxonomy.result import BuildResult, EpisodeRecord, NodeRouter
from scholarpath.taxonomy.router import MLPRouter, stratified_split
from scholarpath.taxonomy.tree import CERTIFIED, LEAF, OPEN, UNRESOLVED, Taxonomy, TaxonomyNode
from scholarpath.text import count_tokens, truncate_tokens
from scholarpath.types import Paper

log = logging.getLogger(__name__)

#: Key of the per-node builder state persisted in ``TaxonomyNode.metadata``
#: (verified labels, audited papers, snapshot index, certificate), used by
#: :mod:`scholarpath.taxonomy.dynamic`.
STATE_KEY = "builder"

#: Pseudo-count shrinking a region's disagreement rate toward the node rate.
DISAGREEMENT_SHRINKAGE = 2.0

# Episode termination reasons.
TERM_CERTIFIED = "certified"
TERM_MAX_CYCLES = "max_cycles"
TERM_BUDGET = "budget"
TERM_FIXED_RATIO = "fixed_ratio"
TERM_NO_SCHEMA = "no_schema"

Label = Tuple[Tuple[str, ...], bool]  # (LLM children, missing concept)
PapersLike = Union[Mapping[str, Paper], Corpus]


# ---------------------------------------------------------------------------
# Per-episode state
# ---------------------------------------------------------------------------


@dataclass
class LossEstimate:
    """Router-signal proxies of round t (Section 3.2)."""

    m: np.ndarray  # m^_{vg,t} per region
    e: np.ndarray  # e^_{vg,t} per region
    e_count: np.ndarray  # effective number of held-out papers behind e^_{vg,t}
    shares: np.ndarray  # pi_{vg}
    abstention: float  # a_{v,t} = |bottom_v| / n_v
    disagreement: float  # d_{v,t} (validation split)

    @property
    def regional_loss(self) -> np.ndarray:
        return self.m + (1.0 - self.m) * self.e

    @property
    def loss(self) -> float:
        """L^_{v,t} = sum_g pi_g [m^_g + (1 - m^_g) e^_g] = F_{v,t}(0)."""
        return float(np.dot(self.shares, self.regional_loss))


@dataclass
class NodeState:
    """Everything the builder knows about node v during (and after) its episode."""

    node_id: str
    depth: int
    paper_ids: List[str]  # P_v (sorted)
    delta_v: float
    regions: Regions
    rng: np.random.Generator
    region_of: Dict[str, int] = field(default_factory=dict)
    verified: Dict[str, Label] = field(default_factory=dict)  # D_v: LLM labels
    audited: Set[str] = field(default_factory=set)  # held out of router training
    train_ids: List[str] = field(default_factory=list)
    val_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    child_vecs: Optional[np.ndarray] = None
    router: Optional[MLPRouter] = None
    train_signature: Optional[Tuple[Any, ...]] = field(default=None, repr=False)
    routed: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    responses: List[RegionResponse] = field(default_factory=list)
    descriptors: List[RegionDescriptor] = field(default_factory=list)
    token_cost: Optional[Callable[[int], float]] = None
    next_snapshot: int = 1  # j of the next snapshot
    rounds: int = 0
    cycles: int = 0
    reads_start: Dict[str, int] = field(default_factory=dict)
    certificate: Optional[AuditOutcome] = None
    last_audit: Optional[AuditOutcome] = None
    audits: List[Dict[str, Any]] = field(default_factory=list)
    round_log: List[Dict[str, Any]] = field(default_factory=list)
    status: str = OPEN
    termination: str = ""

    @property
    def n(self) -> int:
        return len(self.paper_ids)

    def assignment(self, pid: str) -> Tuple[str, ...]:
        """Current assignment: LLM label if verified, router output otherwise."""
        lab = self.verified.get(pid)
        if lab is not None:
            return lab[0]
        return self.routed.get(pid, ())

    def is_residual(self, pid: str) -> bool:
        return not self.assignment(pid)

    def unread(self) -> List[str]:
        return [p for p in self.paper_ids if p not in self.verified]

    def unread_by_region(self) -> List[List[str]]:
        out: List[List[str]] = [[] for _ in range(self.regions.G)]
        for p in self.paper_ids:
            if p not in self.verified:
                out[self.region_of[p]].append(p)
        return out

    @property
    def known_errors(self) -> int:
        """E_D: verified papers with a missing concept (assignments of verified papers are their LLM labels)."""
        return sum(1 for _, missing in self.verified.values() if missing)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class NodeWiseBuilder:
    """Algorithm 1: node-wise confidence-bounded taxonomy construction.

    Args:
        llm: LLM client (its cost meter attributes every call to a component
            of Eq. 15: acquisition, auditing, discovery, correction, relabeling).
        embedder: paper / concept embedder (SPECTER2 in the paper).
        config: :class:`~scholarpath.config.BuilderConfig`.
        papers: the repository P (dict ``paper_id -> Paper`` or a Corpus).
        paper_vectors: optional precomputed paper embeddings.
        prior: optional :class:`TransferredPrior` (e.g. shared across runs);
            by default a fresh one is created from the config.
        min_seed: the seed batch reads at least this many papers in total
            (default: one LLM call, ``config.llm_papers_per_call``), with at
            least ``config.seed_per_region`` papers from every region.
    """

    method_name = "scholarpath"

    def __init__(
        self,
        llm: LLMClient,
        embedder: Any,
        config: Optional[BuilderConfig] = None,
        papers: Optional[PapersLike] = None,
        paper_vectors: Optional[Mapping[str, np.ndarray]] = None,
        prior: Optional[TransferredPrior] = None,
        min_seed: Optional[int] = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.config = config or BuilderConfig()
        if papers is None:
            raise ValueError("NodeWiseBuilder needs the paper repository")
        # A private (shallow) copy: dynamic updates add papers to it without
        # mutating the caller's dict or Corpus.
        self.papers: Dict[str, Paper] = dict(papers.papers) if isinstance(papers, Corpus) else dict(papers)
        self._vectors: Dict[str, np.ndarray] = {}
        if paper_vectors:
            for pid, v in paper_vectors.items():
                self._vectors[str(pid)] = l2_normalize(np.asarray(v, dtype=np.float32).reshape(1, -1))[0]
        self.prior = prior if prior is not None else TransferredPrior.from_config(self.config)
        self.min_seed = int(min_seed) if min_seed is not None else int(self.config.llm_papers_per_call)
        self._concept_vecs: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ vectors
    @property
    def paper_vectors(self) -> Dict[str, np.ndarray]:
        return self._vectors

    def vectors(self, paper_ids: Sequence[str]) -> np.ndarray:
        """Embeddings of ``paper_ids`` (computed once, L2-normalised)."""
        missing = [p for p in dict.fromkeys(paper_ids) if p not in self._vectors]
        for start in range(0, len(missing), 1024):
            chunk = missing[start : start + 1024]
            vecs = l2_normalize(np.asarray(self.embedder.encode([self.papers[p].text for p in chunk]), dtype=np.float32))
            for p, v in zip(chunk, vecs):
                self._vectors[p] = v
        if not paper_ids:
            return np.zeros((0, int(getattr(self.embedder, "dim", 1))), dtype=np.float32)
        return np.stack([self._vectors[p] for p in paper_ids])

    def concept_vector(self, text: str) -> np.ndarray:
        v = self._concept_vecs.get(text)
        if v is None:
            v = l2_normalize(np.asarray(self.embedder.encode([text]), dtype=np.float32))[0]
            self._concept_vecs[text] = v
        return v

    def child_matrix(self, tax: Taxonomy, child_ids: Sequence[str]) -> np.ndarray:
        if not child_ids:
            return np.zeros((0, int(getattr(self.embedder, "dim", 1))), dtype=np.float32)
        return np.stack([self.concept_vector(tax.nodes[c].text) for c in child_ids])

    # --------------------------------------------------------------- misc
    def make_ops(self, facet: str) -> TaxonomyLLM:
        cfg = self.config
        return TaxonomyLLM(
            self.llm,
            self.papers,
            facet,
            papers_per_call=cfg.llm_papers_per_call,
            min_children=cfg.min_children,
            max_children=cfg.max_children,
        )

    def splittable(self, node: TaxonomyNode) -> bool:
        """Line 21: a node enters the queue if it is above the size limit and not too deep."""
        return node.size >= self.config.min_node_size and node.depth < self.config.max_depth

    def _rng(self, facet: str, node_id: str, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng([int(self.config.seed), stable_hash(facet, node_id, salt) % (2 ** 32)])

    @staticmethod
    def _reads(ops: TaxonomyLLM) -> Dict[str, int]:
        return dict(ops.papers_read)

    def _reads_since(self, state: NodeState, ops: TaxonomyLLM) -> Dict[str, int]:
        now = ops.papers_read
        out = {k: v - state.reads_start.get(k, 0) for k, v in now.items()}
        return {k: v for k, v in out.items() if v > 0}

    def _budget_left(self, state: NodeState, ops: TaxonomyLLM) -> int:
        cap = self.config.max_llm_papers_per_node
        if cap is None:
            return 1 << 60
        return max(0, int(cap) - sum(self._reads_since(state, ops).values()))

    # ============================================================== build
    def build(
        self,
        root_name: str = "Root",
        root_definition: str = "",
        paper_ids: Optional[Iterable[str]] = None,
        facet: Optional[str] = None,
    ) -> BuildResult:
        """Run Algorithm 1 over ``paper_ids`` (default: every paper) for one facet."""
        cfg = self.config
        facet = facet or cfg.facet
        t0 = time.perf_counter()
        usage_before = self.llm.meter.snapshot()
        ids = sorted(set(paper_ids) if paper_ids is not None else set(self.papers))
        unknown = [p for p in ids if p not in self.papers]
        if unknown:
            raise KeyError(f"{len(unknown)} paper ids are not in the repository, e.g. {unknown[:3]}")
        self.vectors(ids)
        tax = Taxonomy(facet=facet, root_name=root_name, root_definition=root_definition, root_papers=ids)
        ops = self.make_ops(facet)
        allocator = DeltaAllocator(cfg.delta, cfg.delta_allocation, len(ids), cfg.max_depth, cfg.max_nodes)
        result = BuildResult(taxonomy=tax, method=self.method_name)
        self.process_queue(tax, deque([tax.root_id]), ops, allocator, result)
        result.usage = self.llm.meter.diff(usage_before)
        result.papers_read = dict(ops.papers_read)
        result.runtime_s = time.perf_counter() - t0
        result.metadata.update(
            {
                "facet": facet,
                "epsilon": cfg.epsilon,
                "delta": cfg.delta,
                "delta_spent": allocator.spent,
                "allocator": allocator.state_dict(),
                "n_papers": len(ids),
                "builder_config": dataclasses.asdict(cfg),
                "prior_observations": self.prior.n_observations,
            }
        )
        tax.metadata.update({"method": self.method_name, "epsilon": cfg.epsilon, "delta": cfg.delta})
        return result

    def process_queue(
        self,
        tax: Taxonomy,
        queue: Deque[str],
        ops: TaxonomyLLM,
        allocator: DeltaAllocator,
        result: BuildResult,
    ) -> None:
        """Algorithm 1, lines 2-21: run one episode per queued node (BFS)."""
        cfg = self.config
        while queue:
            nid = queue.popleft()
            node = tax.nodes[nid]
            if not self.splittable(node) or allocator.count >= cfg.max_nodes:
                node.status = LEAF
                continue
            state = self.init_node(tax, nid, ops, allocator)
            if state.termination == TERM_NO_SCHEMA:
                result.episodes.append(self._unsplit_episode(state, tax, ops))
                continue
            self.run_cycles(state, tax, ops)
            _, _, episode = self.finalize(state, tax, ops, result)
            log.info(
                "episode %s (n=%d, depth=%d): %s, bound=%s, read %d papers (%d verified), %d rounds, %d cycles",
                nid, episode.n_v, episode.depth, episode.metadata["termination"],
                None if episode.bound is None else round(episode.bound, 4),
                sum(episode.papers_read.values()), episode.verified, episode.rounds, episode.cycles,
            )
            for c in tax.children(nid):
                if self.splittable(tax.nodes[c]):
                    queue.append(c)
                else:
                    tax.nodes[c].status = LEAF

    # =========================================================== InitNode
    def init_node(
        self,
        tax: Taxonomy,
        nid: str,
        ops: TaxonomyLLM,
        allocator: DeltaAllocator,
    ) -> NodeState:
        """InitNode(v) (Algorithm 1, line 3): delta_v, regions, seed, schema, router.

        If fewer than two of the proposed children fit any seed paper, the LLM
        cannot organize v one level further: the children are discarded and
        the node becomes a leaf (``state.termination == "no_schema"``).
        """
        cfg = self.config
        node = tax.nodes[nid]
        state = self._new_state(tax, nid, ops, allocator.allocate(node.size))
        seed = self._seed_batch(state)[: self._budget_left(state, ops)]
        specs, labels = ops.propose_children(tax, nid, seed, existing=tax.children(nid), component="acquisition")
        resolve_new_children(tax, nid, specs, labels, cfg.max_children)
        used = {c for lab in labels.values() for c in lab.children if c in tax.nodes}
        if len(used) < 2:
            tax.clear_children(nid)
            node.status = state.status = LEAF
            state.termination = TERM_NO_SCHEMA
            node.delta = state.delta_v
            node.metadata[STATE_KEY] = {"termination": TERM_NO_SCHEMA, "delta_v": state.delta_v, "n": state.n}
            return state
        self.record_labels(state, tax, labels)
        self.retrain(state, tax)
        self._init_responses(state, tax)
        return state

    def _unsplit_episode(self, state: NodeState, tax: Taxonomy, ops: TaxonomyLLM) -> EpisodeRecord:
        """Episode record of a node whose seed batch yielded no usable schema."""
        reads = self._reads_since(state, ops)
        return EpisodeRecord(
            node_id=state.node_id,
            depth=state.depth,
            n_v=state.n,
            status=LEAF,
            delta_v=state.delta_v,
            papers_read=reads,
            verified=sum(reads.values()),
            metadata={"termination": TERM_NO_SCHEMA, "regions": state.regions.G, "audits": [], "rounds_log": []},
        )

    def _new_state(self, tax: Taxonomy, nid: str, ops: TaxonomyLLM, delta_v: float) -> NodeState:
        cfg = self.config
        node = tax.nodes[nid]
        pids = sorted(node.papers)
        V = self.vectors(pids)
        regions = Regions.build(pids, V, cfg.max_regions, cfg.papers_per_region, seed=cfg.seed)
        state = NodeState(
            node_id=nid,
            depth=node.depth,
            paper_ids=pids,
            delta_v=float(delta_v),
            regions=regions,
            rng=self._rng(tax.facet, nid, int(node.metadata.get(STATE_KEY, {}).get("resumes", 0))),
            region_of=regions.region_of(),
            reads_start=self._reads(ops),
        )
        state.token_cost = self._token_cost(state, tax, ops)
        return state

    def _seed_batch(self, state: NodeState) -> List[str]:
        """Seed batch: >= seed_per_region papers from every region, >= min_seed in total."""
        cfg = self.config
        out: List[str] = []
        for members, share in zip(state.regions.members, state.regions.shares):
            k = max(cfg.seed_per_region, int(math.ceil(self.min_seed * share)))
            k = min(len(members), k)
            if k <= 0:
                continue
            idx = state.rng.choice(len(members), size=k, replace=False)
            out.extend(members[i] for i in sorted(idx.tolist()))
        return out

    def _token_cost(self, state: NodeState, tax: Taxonomy, ops: TaxonomyLLM) -> Callable[[int], float]:
        """Expected token cost C_{v,t}(z) of reading z papers (Eq. 3)."""
        sample = state.paper_ids
        if len(sample) > 200:
            idx = np.random.default_rng(0).choice(len(sample), 200, replace=False)
            sample = [sample[i] for i in sorted(idx.tolist())]
        per_paper = 0.0
        for p in sample:
            paper = self.papers[p]
            per_paper += count_tokens(paper.title) + count_tokens(truncate_tokens(paper.abstract or "", ops.abstract_tokens))
        per_paper = per_paper / max(1, len(sample)) + 24.0  # id, formatting and the label in the output
        node = tax.nodes[state.node_id]
        per_call = 300.0 + count_tokens(node.text) + 40.0 * self.config.max_children
        return expected_token_cost(per_paper, per_call, self.config.llm_papers_per_call)

    def _init_responses(self, state: NodeState, tax: Taxonomy) -> None:
        """Response models of every region with the transferred prior (Eq. 14)."""
        cfg = self.config
        est = self.estimate_loss(state)
        state.descriptors = []
        state.responses = []
        read_by_region = np.zeros(state.regions.G)
        for p in state.verified:
            read_by_region[state.region_of[p]] += 1
        for g, members in enumerate(state.regions.members):
            h = RegionDescriptor(
                depth=state.depth,
                share=float(state.regions.shares[g]),
                dispersion=region_dispersion(self.vectors(members)),
                child_count=len(state.child_ids),
                papers_read=float(read_by_region[g]),
                abstention=float(est.m[g]),
                disagreement=float(est.e[g]),
            )
            if cfg.use_transferred_prior:
                mean, cov = self.prior.predict(h)
            else:
                mean, cov = np.full(2, cfg.prior_slope_mean), cfg.prior_slope_var * np.eye(2)
            resp = RegionResponse(mean, cov, cfg.obs_noise_var, float(est.m[g]), float(est.e[g]))
            state.descriptors.append(h)
            state.responses.append(resp)

    # ============================================================ labels
    def record_labels(self, state: NodeState, tax: Taxonomy, labels: Mapping[str, PaperLabel]) -> List[str]:
        """Add LLM labels to the verified set D_v (labels refer to current children)."""
        valid = set(tax.children(state.node_id))
        members = state.region_of
        added: List[str] = []
        for pid, lab in labels.items():
            if pid not in members:
                continue
            kids = tuple(sorted(c for c in lab.children if c in valid))
            missing = bool(lab.missing) or not kids
            state.verified[pid] = (kids, missing)
            added.append(pid)
        return added

    def schema_revision(
        self,
        state: NodeState,
        tax: Taxonomy,
        ops: TaxonomyLLM,
        new_children: Sequence[str],
        refined: Sequence[str] = (),
        exclude: Iterable[str] = (),
    ) -> List[str]:
        """Schema-dependent relabeling (App. B.2) after children were added or refined."""
        if not new_children and not refined:
            return []
        children = tax.children(state.node_id)
        child_vecs = {c: self.concept_vector(tax.nodes[c].text) for c in children}
        candidates = relabel_candidates(
            state.verified, [c for c in new_children if c in child_vecs], [c for c in refined if c in child_vecs],
            self._vectors, child_vecs, exclude=exclude,
        )
        candidates = candidates[: self._budget_left(state, ops)]
        labels = relabel(ops, tax, state.node_id, candidates)
        return self.record_labels(state, tax, labels)

    # ============================================================ router
    def retrain(self, state: NodeState, tax: Taxonomy) -> None:
        """Retrain f_theta_v on the training split of D_v and route all of P_v (Section 3.1).

        Verified papers are split 80/20 (stratified) into training and
        validation; audited papers are held out of training so that their
        disagreements remain an unbiased refinement of the validation rate.
        """
        child_ids = tax.children(state.node_id)
        signature = (
            tuple((c, tax.nodes[c].text) for c in child_ids),
            tuple(sorted(state.verified.items())),
            tuple(sorted(state.audited)),
        )
        if state.router is not None and signature == state.train_signature:
            return  # nothing the router depends on has changed since the last fit
        state.train_signature = signature
        state.child_ids = child_ids
        state.child_vecs = self.child_matrix(tax, child_ids)
        trainable = {p: lab[0] for p, lab in state.verified.items() if p not in state.audited}
        router, train, val = self._fit_router(child_ids, state.child_vecs, trainable)
        state.router, state.train_ids, state.val_ids = router, train, val
        if child_ids:
            probs = router.predict_proba(self.vectors(state.paper_ids), state.child_vecs)
            sets = router.assign(probs)
        else:
            sets = [set() for _ in state.paper_ids]
        state.routed = {p: tuple(child_ids[i] for i in sorted(s)) for p, s in zip(state.paper_ids, sets)}

    def _fit_router(
        self, child_ids: Sequence[str], child_vecs: np.ndarray, trainable: Mapping[str, Sequence[str]]
    ) -> Tuple[MLPRouter, List[str], List[str]]:
        """Fit f_theta_v on the 80% stratified training split of ``trainable`` (paper -> LLM children)."""
        cfg = self.config
        train, val = stratified_split(dict(trainable), cfg.router.validation_fraction, seed=cfg.router.seed)
        cidx = {c: i for i, c in enumerate(child_ids)}
        router = MLPRouter(cfg.router)
        if train and len(child_ids):
            targets = [{cidx[c] for c in trainable[p] if c in cidx} for p in train]
            router.fit(self.vectors(train), child_vecs, targets)
        return router, train, val

    def rebuild_router(self, tax: Taxonomy, nid: str) -> NodeRouter:
        """Retrain the frozen router of a finished node from its persisted verified labels.

        Zero LLM cost. :meth:`BuildResult.save` does not store routers, so a
        reloaded taxonomy needs this before :class:`DynamicUpdater` can
        route arrivals. The router is fit exactly as :meth:`retrain` fits it:
        the verified, non-audited papers in ``node.metadata[STATE_KEY]``,
        split 80/20, on the node's current children. The result can differ
        slightly from the original router, for example when children that
        ended up empty were dropped. Certificates stay valid anyway. A
        dynamic update never re-routes existing papers; it uses the router
        only for arrivals, and Eq. (16) counts every unaudited arrival as an
        error whatever the router does.

        Raises:
            ValueError: if the node has no children or no persisted builder
                state (e.g. a taxonomy built by a baseline).
            KeyError: if a verified paper is not in the builder's repository.
        """
        node = tax.nodes[nid]
        child_ids = tax.children(nid)
        meta = node.metadata.get(STATE_KEY) or {}
        if not child_ids:
            raise ValueError(f"node {nid!r} has no children; there is no router to rebuild")
        if "verified" not in meta:
            raise ValueError(
                f"node {nid!r} has no persisted builder state (metadata[{STATE_KEY!r}]['verified']); "
                "its router cannot be rebuilt"
            )
        valid = set(child_ids)
        audited = set(meta.get("audited") or [])
        trainable: Dict[str, Tuple[str, ...]] = {}
        for pid, kids in sorted((meta.get("verified") or {}).items()):
            if pid in audited or pid not in node.papers:
                continue
            trainable[pid] = tuple(sorted(c for c in kids if c in valid))
        unknown = [p for p in trainable if p not in self.papers and p not in self._vectors]
        if unknown:
            raise KeyError(
                f"node {nid!r}: {len(unknown)} verified papers are not in the builder's repository "
                f"(e.g. {unknown[:3]}); pass the papers the taxonomy was built from"
            )
        child_vecs = self.child_matrix(tax, child_ids)
        router, _, _ = self._fit_router(child_ids, child_vecs, trainable)
        return NodeRouter(node_id=nid, child_ids=list(child_ids), child_vecs=child_vecs, router=router,
                          threshold=self.config.router.threshold)

    # ========================================================= proxies
    def estimate_loss(self, state: NodeState) -> LossEstimate:
        """Router-signal proxies m^_{vg,t}, e^_{vg,t}, a_{v,t}, d_{v,t} (Section 3.2)."""
        G = state.regions.G
        n_g = np.array([len(m) for m in state.regions.members], dtype=float)
        miss_g = np.zeros(G)
        placed_g = np.zeros(G)
        unverified_placed_g = np.zeros(G)
        residual = 0
        for p in state.paper_ids:
            g = state.region_of[p]
            lab = state.verified.get(p)
            assigned = state.assignment(p)
            if not assigned:
                residual += 1
            if not assigned or (lab is not None and lab[1]):
                miss_g[g] += 1
            if assigned:
                placed_g[g] += 1
                if lab is None:
                    unverified_placed_g[g] += 1
        # Disagreements on held-out verified papers (validation split + audited papers).
        held = [p for p in list(state.val_ids) + sorted(state.audited) if p in state.verified and not state.verified[p][1]]
        held = list(dict.fromkeys(held))
        dis_g = np.zeros(G)
        cnt_g = np.zeros(G)
        for p in held:
            g = state.region_of[p]
            cnt_g[g] += 1
            dis_g[g] += float(set(state.routed.get(p, ())) != set(state.verified[p][0]))
        e_node = (dis_g.sum() + 0.5) / (cnt_g.sum() + 1.0)
        e_raw = (dis_g + DISAGREEMENT_SHRINKAGE * e_node) / (cnt_g + DISAGREEMENT_SHRINKAGE)
        frac_unverified = np.divide(unverified_placed_g, placed_g, out=np.zeros(G), where=placed_g > 0)
        val = [p for p in state.val_ids if p in state.verified and not state.verified[p][1]]
        d_val = (
            float(np.mean([set(state.routed.get(p, ())) != set(state.verified[p][0]) for p in val])) if val else 0.0
        )
        return LossEstimate(
            m=miss_g / np.maximum(n_g, 1.0),
            e=e_raw * frac_unverified,
            e_count=cnt_g + DISAGREEMENT_SHRINKAGE,
            shares=state.regions.shares,
            abstention=residual / max(1, state.n),
            disagreement=d_val,
        )

    # ============================================================ cycles
    def run_cycles(
        self,
        state: NodeState,
        tax: Taxonomy,
        ops: TaxonomyLLM,
        force: bool = False,
        max_cycles: Optional[int] = None,
    ) -> None:
        """Algorithm 1, lines 4-20: cycles of acquisition, repair and certification.

        A cycle runs rounds of Stage 1 (one planned batch, read only while
        F_{v,t}(0) > eps or the last audit found errors) each followed by
        Stage 2 (repair when its pools reach the threshold). The audit of
        Stage 3 is triggered once the forecast is within eps -- the loss
        estimate after repair is <= eps, or the planned batch was predicted to
        reach eps (Eq. 3) -- or when nothing is left to read, or after
        ``max_rounds`` rounds. A failed audit starts the next cycle with a
        new snapshot index; the node is UNRESOLVED if ``max_cycles`` cycles
        (default ``config.max_cycles``) or the per-node budget are exhausted
        first.
        """
        cfg = self.config
        target = int(math.ceil(cfg.fixed_ratio * state.n))
        n_cycles = cfg.max_cycles if max_cycles is None else max(0, int(max_cycles))
        for _ in range(n_cycles):
            state.cycles += 1
            for _ in range(cfg.max_rounds):
                n_read, forecast_ok = self.acquisition_round(state, tax, ops, force)
                force = False
                self.stage2_repair(state, tax, ops)
                if not cfg.use_snapshot_certification:
                    if n_read == 0 or len(state.verified) >= target:
                        break
                    continue
                if n_read == 0 or forecast_ok or self.estimate_loss(state).loss <= cfg.epsilon:
                    break
            if not cfg.use_snapshot_certification:
                # Ablation (Table 6): stop at a fixed reading ratio eta_v, no audit.
                if len(state.verified) >= target or not state.unread():
                    state.status, state.termination = OPEN, TERM_FIXED_RATIO
                    return
                if self._budget_left(state, ops) <= 0:
                    state.status, state.termination = UNRESOLVED, TERM_BUDGET
                    return
                continue
            if self._budget_left(state, ops) <= 0 and state.unread():
                state.status, state.termination = UNRESOLVED, TERM_BUDGET
                return
            outcome = self.stage3_certification(state, tax, ops)
            if outcome.certified:
                state.status, state.termination = CERTIFIED, TERM_CERTIFIED
                state.certificate = outcome
                return
            if outcome.stop_reason == BUDGET:
                state.status, state.termination = UNRESOLVED, TERM_BUDGET
                return
            force = outcome.found_errors or state.known_errors > 0
        state.status, state.termination = UNRESOLVED, TERM_MAX_CYCLES

    # ---------------------------------------------------------- Stage 1
    def acquisition_round(
        self, state: NodeState, tax: Taxonomy, ops: TaxonomyLLM, force: bool = False
    ) -> Tuple[int, bool]:
        """One round t of Stage 1: quality-targeted Bayesian acquisition (lines 5-10).

        If F_{v,t}(0) = L^_{v,t} > eps or ``force`` (the last audit found
        remaining errors): Thompson-sample the regional response curves, plan
        (b*, z_plan) by Eq. (3)/(10), draw b*_g unread papers uniformly
        without replacement from every region, let the LLM label them
        (updating schema and labels), retrain and reroute, and update the
        slope posteriors. Returns (papers read, forecast within eps).
        """
        cfg = self.config
        est = self.estimate_loss(state)
        unread = state.unread_by_region()
        n_unread = sum(len(u) for u in unread)
        if n_unread == 0:
            return 0, est.loss <= cfg.epsilon
        if cfg.use_snapshot_certification:
            if not (est.loss > cfg.epsilon or force):
                return 0, True
            beta_t = min(cfg.batch_cap, n_unread)  # beta_{v,t} = min{beta, n_v - |S_{v,t}|}
        else:
            need = int(math.ceil(cfg.fixed_ratio * state.n)) - len(state.verified)
            if need <= 0:
                return 0, False
            beta_t = min(cfg.batch_cap, n_unread, need)
        beta_t = min(beta_t, self._budget_left(state, ops))
        if beta_t <= 0:
            return 0, False
        forecast_ok = False
        if cfg.use_bayesian_acquisition:
            for g, resp in enumerate(state.responses):
                resp.set_rates(float(est.m[g]), float(est.e[g]))
            curves = [
                resp.thompson_curve(min(len(unread[g]), beta_t), state.rng)
                for g, resp in enumerate(state.responses)
            ]
            plan = plan_batch(curves, est.shares, beta_t, cfg.epsilon, state.token_cost)
            if plan.z <= 0:
                return 0, False
            alloc = [int(b) for b in plan.allocation]
            batch: List[str] = []
            for g, b in enumerate(alloc):
                if b > 0:
                    idx = state.rng.choice(len(unread[g]), size=b, replace=False)
                    batch.extend(unread[g][i] for i in sorted(idx.tolist()))
            forecast_ok = plan.feasible
            state.round_log.append({"loss": est.loss, "abstention": est.abstention, "z": plan.z,
                                    "forecast": plan.predicted_loss, "feasible": plan.feasible})
        else:
            # Ablation (Table 6): uniform random batches of size beta.
            pool = [p for u in unread for p in u]
            idx = state.rng.choice(len(pool), size=beta_t, replace=False)
            batch = [pool[i] for i in sorted(idx.tolist())]
            alloc = [0] * state.regions.G
            for p in batch:
                alloc[state.region_of[p]] += 1
            state.round_log.append({"loss": est.loss, "abstention": est.abstention, "z": len(batch)})
        self.acquire(state, tax, ops, batch)
        state.rounds += 1
        if cfg.use_bayesian_acquisition:
            self._observe_responses(state, est, self.estimate_loss(state), alloc)
        return len(batch), forecast_ok

    def acquire(self, state: NodeState, tax: Taxonomy, ops: TaxonomyLLM, batch: Sequence[str]) -> None:
        """Line 9-10: the LLM reads the batch (schema may grow), labels are verified, router retrained."""
        cfg = self.config
        before = set(tax.children(state.node_id))
        lb = ops.label(tax, state.node_id, tax.children(state.node_id), list(batch),
                       component="acquisition", allow_new=True)
        resolve_new_children(tax, state.node_id, lb.new_children, lb.labels, cfg.max_children)
        self.record_labels(state, tax, lb.labels)
        new = [c for c in tax.children(state.node_id) if c not in before]
        if new:
            self.schema_revision(state, tax, ops, new, (), exclude=lb.labels.keys())
        self.retrain(state, tax)

    def _observe_responses(self, state: NodeState, before: LossEstimate, after: LossEstimate,
                           alloc: Sequence[int]) -> None:
        """Posterior update of every region read in the round (App. B.1).

        The measured log-odds use Jeffreys-smoothed rates; the Gaussian
        observation noise is ``obs_noise_var`` plus the delta-method variance
        of the two measured log-odds divided by n^2 (a rate measured over few
        papers carries little information about the slope).
        """
        cfg = self.config
        n_g = np.array([len(m) for m in state.regions.members], dtype=float)
        for g, b in enumerate(alloc):
            if b <= 0:
                continue

            def smooth(r: float, n: float) -> float:
                return (r * n + 0.5) / (n + 1.0)

            mb, ma = smooth(before.m[g], n_g[g]), smooth(after.m[g], n_g[g])
            eb, ea = smooth(before.e[g], before.e_count[g]), smooth(after.e[g], after.e_count[g])
            noise = [
                cfg.obs_noise_var
                + (logit_measurement_var(before.m[g], n_g[g]) + logit_measurement_var(after.m[g], n_g[g])) / b ** 2,
                cfg.obs_noise_var
                + (logit_measurement_var(before.e[g], before.e_count[g])
                   + logit_measurement_var(after.e[g], after.e_count[g])) / b ** 2,
            ]
            state.responses[g].observe(int(b), mb, ma, eb, ea, noise_var=noise)

    # ---------------------------------------------------------- Stage 2
    def repair_threshold(self, n_v: int) -> float:
        cfg = self.config
        return max(float(cfg.repair_pool_floor), cfg.repair_pool_fraction * n_v)

    def discovery_pool(self, state: NodeState) -> List[str]:
        """Residual leaf bottom_v plus papers flagged with a missing concept."""
        return [p for p in state.paper_ids if state.is_residual(p) or state.verified.get(p, ((), False))[1]]

    def correction_pool(self, state: NodeState) -> List[str]:
        """Validation or audited papers whose router assignment disagrees with the LLM label."""
        held = set(state.val_ids) | set(state.audited)
        return sorted(
            p for p in held
            if p in state.verified and not state.verified[p][1]
            and set(state.routed.get(p, ())) != set(state.verified[p][0])
        )

    def stage2_repair(self, state: NodeState, tax: Taxonomy, ops: TaxonomyLLM, force: bool = False) -> bool:
        """Stage 2: audit-guided repair (lines 11-13, Section 3.3). Returns True if anything ran.

        DISCOVER and CORRECT each run when their pool reaches
        :meth:`repair_threshold`. With ``force`` (a dynamic update whose
        cohort audit exposed drift, Section 3.4 / App. B.3) any non-empty
        pool triggers its operation.
        """
        cfg = self.config
        if not cfg.use_targeted_repair:
            return False
        threshold = 0.0 if force else self.repair_threshold(state.n)
        repaired = False
        if cfg.use_discovery:
            pool = self.discovery_pool(state)
            if pool and len(pool) >= threshold:
                by_region: Dict[int, List[str]] = {}
                for p in pool:
                    by_region.setdefault(state.region_of[p], []).append(p)
                draws = discovery_draws(by_region, cfg.eta_A, cfg.s_max, state.rng,
                                        max_draws=self._budget_left(state, ops))
                if draws:
                    res = discover(ops, tax, state.node_id, draws, cfg.max_children)
                    self.record_labels(state, tax, res.labels)
                    if res.schema_changed:
                        self.schema_revision(state, tax, ops, res.new_children, res.refined, exclude=res.labels.keys())
                    self.retrain(state, tax)
                    repaired = True
        if cfg.use_correction:
            pool = self.correction_pool(state)
            if pool and len(pool) >= threshold:
                targets = correction_targets(
                    pool, state.region_of, self._vectors, state.unread(), cfg.kappa,
                    max_targets=self._budget_left(state, ops),
                )
                # The LLM labels of the disagreeing papers are restored by construction:
                # verified papers are always assigned their LLM labels.
                if targets:
                    labels = correct(ops, tax, state.node_id, targets)
                    self.record_labels(state, tax, labels)
                    self.retrain(state, tax)
                    repaired = True
        return repaired

    # ---------------------------------------------------------- Stage 3
    def stage3_certification(self, state: NodeState, tax: Taxonomy, ops: TaxonomyLLM) -> AuditOutcome:
        """Stage 3: freeze snapshot j and audit it (lines 14-20, Section 3.3, Eq. 4)."""
        cfg = self.config
        j = state.next_snapshot
        snapshot = {p: state.assignment(p) for p in state.paper_ids}
        valid = set(tax.children(state.node_id))
        judged: Dict[str, Label] = {}

        def judge(batch: List[str]) -> Dict[str, bool]:
            lb = ops.label(tax, state.node_id, tax.children(state.node_id), batch,
                           component="auditing", allow_new=False)
            out: Dict[str, bool] = {}
            for pid, lab in lb.labels.items():
                kids = tuple(sorted(c for c in lab.children if c in valid))
                missing = bool(lab.missing) or not kids
                judged[pid] = (kids, missing)
                out[pid] = missing or set(kids) != set(snapshot[pid])
            return out

        outcome = audit_snapshot(
            state.unread(), judge, state.n, state.known_errors, state.delta_v, j, cfg.epsilon,
            cfg.audit_batch_size, state.rng, cfg.audit_abort_factor, cfg.audit_min_draws,
            max_draws=self._budget_left(state, ops),
        )
        # Audited papers become verified (held out of router training); their
        # failures feed the repair pools of the next cycle.
        for pid, lab in judged.items():
            state.verified[pid] = lab
            state.audited.add(pid)
        state.next_snapshot = j + 1
        state.last_audit = outcome
        state.audits.append(outcome.summary())
        return outcome

    # ========================================================== finalize
    def apply_assignments(
        self, tax: Taxonomy, nid: str, final: Mapping[str, Sequence[str]]
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """Write paper -> children assignments of node ``nid`` into the taxonomy.

        Only the papers in ``final`` are touched. A paper with no child goes to
        the residual leaf bottom_v. Returns ``(added, departed)``: papers newly
        assigned to each child, and papers removed from each node of the
        subtree below ``nid`` (needed by dynamic updates).
        """
        node = tax.nodes[nid]
        children = tax.children(nid)
        child_set = set(children)
        added: Dict[str, Set[str]] = {c: set() for c in children}
        departed: Dict[str, Set[str]] = {}
        for pid, kids in final.items():
            new = set(kids) & child_set
            old = {c for c in children if pid in tax.nodes[c].papers}
            for c in sorted(new - old):
                tax.assign(pid, c)
                added[c].add(pid)
            for c in sorted(old - new):
                for d in tax.descendants(c, include_self=True):
                    if pid in tax.nodes[d].papers:
                        departed.setdefault(d, set()).add(pid)
                tax.unassign_subtree(pid, c)
            if new:
                node.residual.discard(pid)
            else:
                node.residual.add(pid)
        tax.invalidate()
        return added, departed

    def finalize(
        self,
        state: NodeState,
        tax: Taxonomy,
        ops: TaxonomyLLM,
        result: Optional[BuildResult] = None,
        record: bool = True,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], EpisodeRecord]:
        """Write the episode's outcome into the taxonomy (Algorithm 1, line 21)."""
        nid = state.node_id
        node = tax.nodes[nid]
        if state.child_ids != tax.children(nid):
            self.retrain(state, tax)
        final = {p: state.assignment(p) for p in state.paper_ids}
        added, departed = self.apply_assignments(tax, nid, final)
        for c in list(tax.children(nid)):
            if not tax.nodes[c].papers:
                removed = list(tax.descendants(c, include_self=True))
                tax.remove_subtree(c)
                added.pop(c, None)
                for d in removed:
                    departed.pop(d, None)
                    if result is not None:
                        result.routers.pop(d, None)
        kept = tax.children(nid)
        node.status = state.status
        node.delta = state.delta_v
        node.bound = state.certificate.bound if state.certificate is not None else None
        est = self.estimate_loss(state)
        prev = node.metadata.get(STATE_KEY, {})
        node.metadata[STATE_KEY] = {
            "n": state.n,
            "delta_v": state.delta_v,
            "termination": state.termination,
            "next_snapshot": state.next_snapshot,
            "certificate": state.certificate.summary() if state.certificate is not None else None,
            "last_audit": state.last_audit.summary() if state.last_audit is not None else None,
            "verified": {p: list(lab[0]) for p, lab in sorted(state.verified.items())},
            "missing": sorted(p for p, lab in state.verified.items() if lab[1]),
            "audited": sorted(state.audited),
            "regions": state.regions.G,
            "rounds": state.rounds,
            "cycles": state.cycles,
            "abstention": est.abstention,
            "disagreement": est.disagreement,
            "resumes": int(prev.get("resumes", 0)),
        }
        if prev.get("update_audits"):
            node.metadata[STATE_KEY]["update_audits"] = list(prev["update_audits"])
        if result is not None and state.router is not None:
            idx = {c: i for i, c in enumerate(state.child_ids)}
            sel = [idx[c] for c in kept if c in idx]
            result.routers[nid] = NodeRouter(
                node_id=nid,
                child_ids=[state.child_ids[i] for i in sel],
                child_vecs=state.child_vecs[sel] if state.child_vecs is not None else np.zeros((0, 1)),
                router=state.router,
                threshold=self.config.router.threshold,
            )
        # History H: (descriptor, observed slope) pairs update the transferred prior.
        descs, slopes = [], []
        for h, resp in zip(state.descriptors, state.responses):
            s = resp.mean_observed_slope()
            if s is not None:
                descs.append(h.to_list())
                slopes.append([float(s[0]), float(s[1])])
        episode = EpisodeRecord(
            node_id=nid,
            depth=state.depth,
            n_v=state.n,
            n_children=len(kept),
            status=state.status,
            certified=state.status == CERTIFIED,
            bound=node.bound,
            delta_v=state.delta_v,
            rounds=state.rounds,
            cycles=state.cycles,
            papers_read=self._reads_since(state, ops),
            verified=len(state.verified),
            abstention=est.abstention,
            disagreement=est.disagreement,
            region_descriptors=descs,
            region_slopes=slopes,
            metadata={
                "termination": state.termination,
                "regions": state.regions.G,
                "snapshots": state.next_snapshot - 1,
                "audits": list(state.audits),
                "known_errors": state.known_errors,
                "loss_estimate": est.loss,
                "rounds_log": list(state.round_log),
            },
        )
        if descs:
            self.prior.update(descs, slopes)
        if record and result is not None:
            result.episodes.append(episode)
        return added, departed, episode

    # ======================================================== resumption
    def resume_node(self, tax: Taxonomy, nid: str, ops: TaxonomyLLM) -> NodeState:
        """Rebuild the state of a finished node from its persisted metadata (Section 3.4).

        The verified labels, audited papers, delta_v and the snapshot index
        are restored; regions, router and response models are rebuilt for the
        node's current extent.
        """
        node = tax.nodes[nid]
        meta = dict(node.metadata.get(STATE_KEY, {}))
        meta["resumes"] = int(meta.get("resumes", 0)) + 1
        node.metadata[STATE_KEY] = meta
        delta_v = node.delta if node.delta is not None else float(meta.get("delta_v", 0.0))
        state = self._new_state(tax, nid, ops, float(delta_v))
        valid = set(tax.children(nid))
        members = set(state.paper_ids)
        missing = set(meta.get("missing", []))
        for pid, kids in (meta.get("verified") or {}).items():
            if pid not in members:
                continue
            k = tuple(sorted(c for c in kids if c in valid))
            state.verified[pid] = (k, pid in missing or not k)
        state.audited = {p for p in meta.get("audited", []) if p in state.verified}
        state.next_snapshot = int(meta.get("next_snapshot", 1))
        self.retrain(state, tax)
        self._init_responses(state, tax)
        return state

    def recertify(self, state: NodeState, tax: Taxonomy, ops: TaxonomyLLM) -> None:
        """Repair and re-certify a resumed node after drift (Section 3.4, App. B.3).

        The cohort audit triggers Stage 2 repair (DISCOVER and CORRECT, forced
        on every non-empty pool) and then the audit of a new snapshot
        (Stage 3, continuing j). If that audit fails, the regular cycles of
        Algorithm 1 continue, with ``max_cycles`` counting this first one.
        """
        cfg = self.config
        if not cfg.use_snapshot_certification:  # ablation: no certificates to maintain
            self.run_cycles(state, tax, ops, force=True)
            return
        state.cycles += 1
        self.stage2_repair(state, tax, ops, force=True)
        if self._budget_left(state, ops) <= 0 and state.unread():
            state.status, state.termination = UNRESOLVED, TERM_BUDGET
            return
        outcome = self.stage3_certification(state, tax, ops)
        if outcome.certified:
            state.status, state.termination = CERTIFIED, TERM_CERTIFIED
            state.certificate = outcome
            return
        if outcome.stop_reason == BUDGET:
            state.status, state.termination = UNRESOLVED, TERM_BUDGET
            return
        self.run_cycles(state, tax, ops, force=outcome.found_errors or state.known_errors > 0,
                        max_cycles=cfg.max_cycles - 1)


# ---------------------------------------------------------------------------
# Dual taxonomies (Section 2)
# ---------------------------------------------------------------------------


def build_dual_taxonomies(
    llm: LLMClient,
    embedder: Any,
    config: BuilderConfig,
    papers: PapersLike,
    paper_vectors: Optional[Mapping[str, np.ndarray]] = None,
    paper_ids: Optional[Iterable[str]] = None,
    root_names: Tuple[str, str] = ("Research Problems", "Research Methods"),
    root_definitions: Tuple[str, str] = (
        "All research problems (tasks, domains, phenomena) studied by the papers of the repository.",
        "All research methods (techniques, models, algorithms, tools) used by the papers of the repository.",
    ),
    priors: Optional[Tuple[Optional[TransferredPrior], Optional[TransferredPrior]]] = None,
    **builder_kwargs: Any,
) -> Tuple[BuildResult, BuildResult]:
    """Run Algorithm 1 once for the problem taxonomy T_P and once for the method taxonomy T_M.

    Both runs use facet-specific prompts (Section 2) and share the paper
    embeddings. Returns ``(problem_result, method_result)``.
    """
    ids = list(paper_ids) if paper_ids is not None else None
    vecs: Optional[Mapping[str, np.ndarray]] = paper_vectors
    results: List[BuildResult] = []
    for k, facet in enumerate(("problem", "method")):
        cfg = dataclasses.replace(config, facet=facet)
        prior = priors[k] if priors is not None else None
        builder = NodeWiseBuilder(llm, embedder, cfg, papers, paper_vectors=vecs, prior=prior, **builder_kwargs)
        results.append(builder.build(root_names[k], root_definitions[k], ids, facet))
        vecs = builder.paper_vectors
    return results[0], results[1]
