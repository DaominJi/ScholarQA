"""Selection-based taxonomy construction baselines (Section 5.2; App. B.2, C.5).

Every baseline builds the taxonomy top-down from the root with the two LLM
steps of existing LLM-based construction methods [34, 42] (Section 3.1): at
each internal node v the LLM *proposes the children* of v from papers of
P_v and *labels* papers with the matching children. The baselines differ only
in which papers the LLM reads:

**FULL** (``selector="full"``, η = 1; the unbudgeted reference)
    The LLM reads all n_v papers of P_v to propose the children of v
    (successive batches of ``llm_papers_per_call`` papers; each batch refines
    the schema proposed so far) and then reads them *again* to route every
    paper to the matching children of the final schema (multi-label; a paper
    matching no child goes to the residual leaf ⊥_v). Repeating both steps at
    every level reads every paper twice per level, i.e. O(N L D) tokens for N
    papers of average length L and depth D. Cost component ``"full"``.

**Node-wise selection** (``mode="node_wise"``, η ∈ {0.05, ..., 0.5})
    At each internal node v the selector (:mod:`scholarpath.taxonomy.selectors`:
    random / coverage / kmeans / diversity) picks S_v ⊆ P_v with
    |S_v| = ⌈η n_v⌉. The LLM reads S_v to induce the children of v and reads
    S_v again to label their memberships with the final schema (the same two
    passes as FULL, restricted to S_v, so that C_LLM / C_Full ≈ η). A router
    f_θv (:class:`~scholarpath.taxonomy.router.MLPRouter`) trained on these
    labels routes P_v \\ S_v. Sub-sampling stops (v becomes a leaf) when
    n_v < 1/η. Cost component ``"selection"``.

**One-off selection** (``mode="one_off"``, Appendix C.5)
    The selector picks S ⊆ P with |S| = ⌈η |P|⌉ *once*, at the root. At each
    node v only the sampled papers that fall into P_v (S ∩ P_v) are read by
    the LLM (to propose children and label them); all other papers are routed
    top-down by the routers f_θv. Deeper subtrees therefore induce their
    concepts from sparse and skewed samples. Cost component ``"selection"``.

Shared mechanics (identical for all baselines, and identical to the
LLM operations and router of ScholarPath, so costs are comparable):

* LLM calls go through :class:`~scholarpath.taxonomy.llm_ops.TaxonomyLLM`
  (``propose_children`` then ``label`` with the final schema); new children
  are created by :func:`~scholarpath.taxonomy.llm_ops.resolve_new_children`
  (at most ``max_children``).
* A node v is expanded iff depth(v) < ``max_depth``, n_v ≥ ``min_node_size``
  (and n_v ≥ 2), P_v is a proper subset of its parent's papers, and, for
  selection baselines, n_v ≥ 1/η and at least one paper of P_v is read by
  the LLM. Children of an expanded node are queued breadth-first. A child
  that received every paper of its parent (with multi-label routing, a
  catch-all concept can) stays a leaf: expanding it would re-organize the
  parent's papers and rebuild the parent's subtree under it, level after
  level, duplicating concepts and inflating the node count, the cost and
  C_Full.
* Routers follow Appendix C.3: a two-layer MLP on paper and child-definition
  embeddings, trained on the LLM-labelled papers with a
  ``router.validation_fraction`` stratified validation split; the validation
  disagreement with the LLM labels is reported per episode
  (:attr:`EpisodeRecord.disagreement`). Papers whose router probabilities
  are all below ``router.threshold`` go to the residual leaf ⊥_v. Because
  the label sets S_v are small, every router is trained for at least
  ``min_router_steps`` optimizer steps (see
  :meth:`SelectionBuilder._router_config`).
* Children that receive no paper after labelling and routing are removed
  (an empty concept organizes nothing); disable with
  ``prune_empty_children=False``.
* A node is split only if at least ``min_split_children`` (= 2) children
  receive papers; otherwise its children (and their residual) are discarded
  and it becomes a leaf, like ScholarPath's ``InitNode`` (episode metadata
  ``termination="no_split"``). Leaves therefore never carry a residual.

Cost accounting (Appendix B.2): exhaustive construction reads all n_v papers
at every internal node v ∈ V_int, so C_Full = c̄ Σ_{v∈V_int} n_v where c̄ is
the measured average cost of processing one paper at one node
(:func:`per_paper_node_cost`). When running FULL is prohibitive (Section
5.2.2, the 148.4K-paper corpus), C_Full is estimated from the routed paper
counts of any built taxonomy (:func:`estimate_full_cost`).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from scholarpath.config import BaselineBuilderConfig, RouterConfig
from scholarpath.embeddings import Embedder, l2_normalize
from scholarpath.llm.base import LLMClient
from scholarpath.llm.mock import stable_hash
from scholarpath.taxonomy.llm_ops import TaxonomyLLM, resolve_new_children
from scholarpath.taxonomy.result import BuildResult, EpisodeRecord, NodeRouter
from scholarpath.taxonomy.router import MLPRouter, disagreement, stratified_split
from scholarpath.taxonomy.selectors import SELECTION_METHODS, num_to_select, select
from scholarpath.taxonomy.tree import LEAF, Taxonomy
from scholarpath.types import Paper

if TYPE_CHECKING:  # pragma: no cover
    from scholarpath.data.corpus import Corpus

log = logging.getLogger(__name__)

FULL = "full"
NODE_WISE = "node_wise"
ONE_OFF = "one_off"
MODES = (NODE_WISE, ONE_OFF)
SELECTORS = (FULL,) + tuple(SELECTION_METHODS)

COMPONENT_FULL = "full"
COMPONENT_SELECTION = "selection"

# EpisodeRecord.metadata["termination"]: v was split into >= 2 non-empty
# children, or the proposed children organized too little and v became a leaf.
TERM_SPLIT = "split"
TERM_NO_SPLIT = "no_split"

PapersLike = Union[Mapping[str, Paper], "Corpus", Iterable[Paper]]

_METHOD_RE = re.compile(r"^(?:(?P<mode>node_wise|one_off)/)?(?P<selector>[a-z]+)(?:@(?P<eta>[0-9.eE+-]+))?$")


# ---------------------------------------------------------------------------
# Method names
# ---------------------------------------------------------------------------


def method_name(selector: str, eta: float, mode: str = NODE_WISE) -> str:
    """Canonical method name: ``"full"``, ``"kmeans@0.2"``, ``"one_off/kmeans@0.2"``."""
    if selector == FULL:
        return FULL
    base = f"{selector}@{float(eta):g}"
    return base if mode == NODE_WISE else f"{mode}/{base}"


def parse_method(name: str) -> Tuple[str, float, str]:
    """Inverse of :func:`method_name`: ``"one_off/kmeans@0.2"`` -> ``("kmeans", 0.2, "one_off")``."""
    m = _METHOD_RE.match((name or "").strip().lower())
    if m is None:
        raise ValueError(f"Cannot parse baseline method name {name!r}")
    selector, mode = m.group("selector"), m.group("mode") or NODE_WISE
    if selector == FULL:
        return FULL, 1.0, NODE_WISE
    if m.group("eta") is None:
        raise ValueError(f"Selection baseline {name!r} needs a ratio, e.g. '{selector}@0.2'")
    return selector, float(m.group("eta")), mode


def baseline_config(method: str, base: Optional[BaselineBuilderConfig] = None) -> BaselineBuilderConfig:
    """Copy of ``base`` configured for the baseline named ``method`` (see :func:`method_name`)."""
    selector, eta, mode = parse_method(method)
    return dataclasses.replace(base or BaselineBuilderConfig(), selector=selector, eta=eta, mode=mode)


# ---------------------------------------------------------------------------
# Cost accounting (Appendix B.2, Section 5.2.2)
# ---------------------------------------------------------------------------


def internal_paper_count(taxonomy: Taxonomy) -> int:
    """Σ_{v ∈ V_int} n_v: papers an exhaustive construction reads per pass."""
    return int(sum(len(taxonomy.nodes[v].papers) for v in taxonomy.internal_nodes()))


def _measure(result: BuildResult, measure: str) -> float:
    usage = result.total_usage
    if measure == "cost_usd":
        return float(usage.cost_usd)
    if measure == "tokens":
        return float(usage.total_tokens)
    if measure == "input_tokens":
        return float(usage.input_tokens)
    if measure == "calls":
        return float(usage.calls)
    raise ValueError(f"Unknown cost measure {measure!r}")


def per_paper_node_cost(full_result: BuildResult, measure: str = "cost_usd", nodes: str = "internal") -> float:
    """c̄ = C_Full / Σ_{v ∈ V_int} n_v (Appendix B.2).

    The measured average cost of processing one paper at one node by an
    exhaustive (FULL) construction. ``measure`` selects the cost unit:
    ``"cost_usd"`` (default), ``"tokens"``, ``"input_tokens"`` or ``"calls"``.

    ``nodes`` selects the normalizer:

    * ``"internal"`` (default, Appendix B.2): Σ n_v over the internal nodes of
      the FULL taxonomy, so that ``estimate_full_cost(full_result, c̄)``
      reproduces C_Full exactly;
    * ``"processed"``: Σ n_v over every node the LLM processed (one
      :class:`EpisodeRecord` each), including nodes whose expansion produced
      no usable child and that therefore ended as leaves.

    Returns NaN when the normalizer is zero.
    """
    if nodes == "internal":
        denom = internal_paper_count(full_result.taxonomy)
    elif nodes == "processed":
        denom = int(sum(e.n_v for e in full_result.episodes))
    else:
        raise ValueError(f"nodes must be 'internal' or 'processed', got {nodes!r}")
    return _measure(full_result, measure) / denom if denom > 0 else float("nan")


def estimate_full_cost(result: Union[BuildResult, Taxonomy], per_paper_cost: float) -> float:
    """Estimated C_Full = c̄ Σ_{v ∈ V_int} n_v (Section 5.2.2).

    Uses the routed paper counts n_v of the internal nodes of any built
    taxonomy (e.g. ScholarPath's) when running FULL itself is prohibitive.
    """
    tax = result.taxonomy if isinstance(result, BuildResult) else result
    return float(per_paper_cost) * internal_paper_count(tax)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _paper_dict(papers: PapersLike) -> Dict[str, Paper]:
    inner = getattr(papers, "papers", None)  # Corpus
    if isinstance(inner, Mapping):
        return dict(inner)
    if isinstance(papers, Mapping):
        return dict(papers)
    return {p.paper_id: p for p in papers}


@dataclasses.dataclass
class _Context:
    """Per-build state shared by the node episodes."""

    tax: Taxonomy
    ops: TaxonomyLLM
    component: str
    vectors: Optional[np.ndarray] = None  # unit rows aligned with ``row``
    row: Dict[str, int] = dataclasses.field(default_factory=dict)
    sample: Optional[List[str]] = None  # one-off sample S (selection order)
    read: Set[str] = dataclasses.field(default_factory=set)  # distinct papers read so far
    episodes: List[EpisodeRecord] = dataclasses.field(default_factory=list)
    routers: Dict[str, NodeRouter] = dataclasses.field(default_factory=dict)

    def matrix(self, paper_ids: Sequence[str]) -> np.ndarray:
        assert self.vectors is not None
        return self.vectors[[self.row[p] for p in paper_ids]]


class SelectionBuilder:
    """FULL, node-wise selection, and one-off selection taxonomy builders (Section 5.2).

    Args:
        llm: LLM client (its cost meter records the build's usage).
        embedder: embeds papers (if ``paper_vectors`` lacks them) and child definitions.
        config: :class:`~scholarpath.config.BaselineBuilderConfig`; ``selector``
            in {"full", "random", "coverage", "kmeans", "diversity"},
            ``mode`` in {"node_wise", "one_off"}, ``eta`` in (0, 1].
        papers: ``{paper_id: Paper}`` or a :class:`~scholarpath.data.corpus.Corpus`.
        paper_vectors: optional precomputed paper embeddings ``{paper_id: vector}``
            (e.g. SPECTER2 of title + abstract); missing papers are embedded
            with ``embedder.encode(paper.text)``.
        prune_empty_children: remove children that receive no paper.
        fit_full_routers: also train a router per internal node of FULL (not
            needed to build FULL; useful to route papers added later).
        min_sample: one-off selection expands a node only if at least this
            many sampled papers fall into it.
        min_router_steps: lower bound on the AdamW steps of every router
            f_θv; the number of epochs of ``config.router`` is raised for
            small label sets (0 disables the floor).
        min_split_children: a node keeps its children only if at least this
            many of them receive papers (2 = no single-child chains; 1 keeps
            any non-empty split).
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        config: Optional[BaselineBuilderConfig],
        papers: PapersLike,
        paper_vectors: Optional[Mapping[str, np.ndarray]] = None,
        *,
        prune_empty_children: bool = True,
        fit_full_routers: bool = False,
        min_sample: int = 1,
        min_router_steps: int = 500,
        min_split_children: int = 2,
    ):
        self.llm = llm
        self.embedder = embedder
        self.config = config or BaselineBuilderConfig()
        self.papers = _paper_dict(papers)
        self.paper_vectors: Dict[str, np.ndarray] = dict(paper_vectors or {})
        self.prune_empty_children = bool(prune_empty_children)
        self.fit_full_routers = bool(fit_full_routers)
        self.min_sample = max(1, int(min_sample))
        self.min_router_steps = max(0, int(min_router_steps))
        self.min_split_children = max(1, int(min_split_children))
        self._validate()

    # ------------------------------------------------------------ config
    def _validate(self) -> None:
        # Normalized private copy; the caller's config object is left untouched.
        self.config = cfg = dataclasses.replace(
            self.config,
            selector=(self.config.selector or "").strip().lower(),
            mode=(self.config.mode or NODE_WISE).strip().lower(),
        )
        if cfg.selector not in SELECTORS:
            raise ValueError(f"Unknown selector {cfg.selector!r}; expected one of {SELECTORS}")
        if cfg.mode not in MODES:
            raise ValueError(f"Unknown mode {cfg.mode!r}; expected one of {MODES}")
        if not self.is_full and not (0.0 < float(cfg.eta) <= 1.0):
            raise ValueError(f"Selection ratio eta must be in (0, 1], got {cfg.eta}")

    @property
    def is_full(self) -> bool:
        return self.config.selector == FULL

    @property
    def eta(self) -> float:
        return 1.0 if self.is_full else float(self.config.eta)

    @property
    def method(self) -> str:
        return method_name(self.config.selector, self.eta, self.config.mode)

    # ----------------------------------------------------------- vectors
    def _paper_matrix(self, paper_ids: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int]]:
        """Unit-norm embeddings of ``paper_ids`` (computing the missing ones)."""
        missing = [p for p in paper_ids if p not in self.paper_vectors]
        if missing:
            vecs = self.embedder.encode([self.papers[p].text for p in missing])
            for p, v in zip(missing, np.asarray(vecs, dtype=np.float32)):
                self.paper_vectors[p] = v
        if not paper_ids:
            return np.zeros((0, int(getattr(self.embedder, "dim", 0))), dtype=np.float32), {}
        mat = l2_normalize(np.stack([np.asarray(self.paper_vectors[p], dtype=np.float32) for p in paper_ids]))
        return mat, {p: i for i, p in enumerate(paper_ids)}

    def _child_matrix(self, tax: Taxonomy, child_ids: Sequence[str]) -> np.ndarray:
        """Child-definition embeddings ("name: definition") used by the router."""
        return l2_normalize(np.asarray(self.embedder.encode([tax.nodes[c].text for c in child_ids]), dtype=np.float32))

    def _node_seed(self, node_id: str) -> int:
        return int(stable_hash("selection-baseline", self.config.seed, node_id) % (2 ** 32))

    # ------------------------------------------------------------- build
    def build(
        self,
        root_name: str = "Root",
        root_definition: str = "",
        paper_ids: Optional[Sequence[str]] = None,
        facet: Optional[str] = None,
    ) -> BuildResult:
        """Build one taxonomy over ``paper_ids`` (default: all papers) top-down."""
        cfg = self.config
        facet = facet or cfg.facet
        ids = sorted(dict.fromkeys(str(p) for p in (paper_ids if paper_ids is not None else self.papers)))
        unknown = [p for p in ids if p not in self.papers]
        if unknown:
            raise KeyError(f"{len(unknown)} paper ids are not in the corpus, e.g. {unknown[:3]}")
        t0 = time.perf_counter()
        before = self.llm.meter.snapshot()

        tax = Taxonomy(facet=facet, root_name=root_name, root_definition=root_definition, root_papers=ids)
        tax.metadata.update({"method": self.method, "builder": type(self).__name__})
        ops = TaxonomyLLM(
            self.llm, self.papers, facet,
            papers_per_call=cfg.llm_papers_per_call,
            min_children=cfg.min_children,
            max_children=cfg.max_children,
        )
        ctx = _Context(tax=tax, ops=ops, component=COMPONENT_FULL if self.is_full else COMPONENT_SELECTION)
        if not self.is_full or self.fit_full_routers:
            ctx.vectors, ctx.row = self._paper_matrix(ids)
        if not self.is_full and cfg.mode == ONE_OFF:
            k = num_to_select(len(ids), self.eta)
            rows = select(ctx.matrix(ids), k, cfg.selector, seed=cfg.seed) if k else []
            ctx.sample = [ids[i] for i in rows]

        queue = deque([tax.root_id])
        while queue:
            v = queue.popleft()
            node = tax.nodes[v]
            to_read = self._papers_to_read(ctx, v) if self._can_expand(tax, v) else None
            if not to_read:
                node.status = LEAF
                continue
            self._episode(ctx, v, to_read)
            if tax.children(v):
                queue.extend(tax.children(v))
            else:
                node.status = LEAF
        tax.invalidate()

        usage = self.llm.meter.diff(before)
        metadata: Dict[str, Any] = {
            "selector": cfg.selector,
            "mode": FULL if self.is_full else cfg.mode,
            "eta": self.eta,
            "facet": facet,
            "n_papers": len(ids),
            "distinct_papers_read": len(ctx.read),
            "internal_paper_count": internal_paper_count(tax),
            "config": dataclasses.asdict(cfg),
        }
        if ctx.sample is not None:
            metadata["one_off_sample"] = len(ctx.sample)
        return BuildResult(
            taxonomy=tax,
            method=self.method,
            usage=usage,
            papers_read=dict(ops.papers_read),
            episodes=ctx.episodes,
            runtime_s=time.perf_counter() - t0,
            routers=ctx.routers,
            metadata=metadata,
        )

    # --------------------------------------------------------- episodes
    def _can_expand(self, tax: Taxonomy, v: str) -> bool:
        """Size / depth limits of Sec. 3.1 plus the 'n_v < 1/η' stop rule of Sec. 5.2.

        A node holding every paper of its parent is not expanded (it would
        re-organize the same papers; see the module docstring).
        """
        cfg = self.config
        node = tax.nodes[v]
        n_v = len(node.papers)
        if node.depth >= cfg.max_depth or n_v < max(2, cfg.min_node_size):
            return False
        parent = tax.nodes.get(node.parent) if node.parent is not None else None
        if parent is not None and node.papers == parent.papers:
            return False
        if not self.is_full and self.eta * n_v < 1.0 - 1e-9:
            return False
        return True

    def _papers_to_read(self, ctx: _Context, v: str) -> List[str]:
        """Papers of P_v the LLM reads at v, in reading order (empty -> leaf)."""
        P_v = sorted(ctx.tax.nodes[v].papers)
        seed = self._node_seed(v)
        if self.is_full:  # all of P_v, in a seeded random order
            order = np.random.default_rng(seed).permutation(len(P_v))
            return [P_v[i] for i in order]
        if self.config.mode == ONE_OFF:
            members = ctx.tax.nodes[v].papers
            read = [p for p in (ctx.sample or []) if p in members]
            return read if len(read) >= self.min_sample else []
        k = num_to_select(len(P_v), self.eta)
        rows = select(ctx.matrix(P_v), k, self.config.selector, seed=seed)
        return [P_v[i] for i in rows]

    def _induce_and_label(self, ctx: _Context, v: str, read: Sequence[str]) -> Dict[str, List[str]]:
        """LLM pass 1: propose / refine the schema of v; pass 2: label ``read`` with it.

        Returns the LLM label (child node ids, possibly empty = ⊥_v) of every
        paper read. A paper whose labelling call failed keeps the label it got
        while the schema was being proposed.
        """
        cfg, tax, ops = self.config, ctx.tax, ctx.ops
        specs, proposal = ops.propose_children(tax, v, read, existing=(), component=ctx.component)
        if not specs:
            return {}
        resolve_new_children(tax, v, specs, proposal, max_children=cfg.max_children)
        child_ids = tax.children(v)
        if not child_ids:
            return {}
        children = set(child_ids)
        batch = ops.label(tax, v, child_ids, read, component=ctx.component, allow_new=False)
        labels: Dict[str, List[str]] = {}
        for p in read:
            lab = batch.labels.get(p) or proposal.get(p)
            labels[p] = [c for c in lab.children if c in children] if lab is not None else []
        return labels

    def _router_config(self, n_pairs: int) -> RouterConfig:
        """Router hyper-parameters for a training set of ``n_pairs`` (paper, child) pairs.

        :class:`~scholarpath.taxonomy.router.MLPRouter` runs ``epochs ×
        ⌈n_pairs / batch_size⌉`` AdamW steps. On the small label sets of
        selection baselines (|S_v| = ⌈η n_v⌉ papers, often < 64 pairs at deep
        nodes) the default 60 epochs are only ~60 steps: the class-balanced
        outputs then hover around 0.5 and many (paper, child) pairs cross the
        threshold, so P_v \\ S_v is over-assigned and the tree inflates. The
        number of epochs is therefore raised until at least
        ``min_router_steps`` steps are taken; architecture, optimizer,
        learning rate and batch size are unchanged (App. C.3 does not fix the
        training schedule). Large label sets keep ``router.epochs``.
        """
        rcfg = self.config.router
        if self.min_router_steps <= 0 or n_pairs <= 0:
            return rcfg
        steps_per_epoch = int(math.ceil(n_pairs / max(1, min(rcfg.batch_size, n_pairs))))
        epochs = max(int(rcfg.epochs), int(math.ceil(self.min_router_steps / steps_per_epoch)))
        return rcfg if epochs == rcfg.epochs else dataclasses.replace(rcfg, epochs=epochs)

    def _fit_router(
        self, ctx: _Context, v: str, child_ids: List[str], labels: Mapping[str, List[str]]
    ) -> Tuple[NodeRouter, Dict[str, Any]]:
        """Train f_θv on LLM labels with a stratified validation split (App. C.3)."""
        rcfg = self.config.router
        child_vecs = self._child_matrix(ctx.tax, child_ids)
        col = {c: j for j, c in enumerate(child_ids)}
        labelled = {p: list(labels[p]) for p in labels}
        if rcfg.validation_fraction > 0:
            train, val = stratified_split(labelled, rcfg.validation_fraction, seed=rcfg.seed)
        else:
            train, val = sorted(labelled), []
        rcfg = self._router_config(len(train) * len(child_ids))
        mlp = MLPRouter(rcfg)
        if train:
            mlp.fit(ctx.matrix(train), child_vecs, [{col[c] for c in labelled[p]} for p in train])
        rate = 0.0
        if val:
            pred = mlp.assign(mlp.predict_proba(ctx.matrix(val), child_vecs), rcfg.threshold)
            rate = float(disagreement(pred, [{col[c] for c in labelled[p]} for p in val]).mean())
        router = NodeRouter(node_id=v, child_ids=list(child_ids), child_vecs=child_vecs, router=mlp,
                            threshold=rcfg.threshold)
        return router, {"router_train": len(train), "router_val": len(val), "disagreement": rate}

    def _episode(self, ctx: _Context, v: str, read: List[str]) -> None:
        """Expand node v: induce its children from ``read``, then assign all of P_v.

        v stays internal only if at least ``min_split_children`` children
        receive papers; otherwise it becomes a leaf (``termination="no_split"``).
        """
        tax, node = ctx.tax, ctx.tax.nodes[v]
        P_v = sorted(node.papers)
        reads_before = dict(ctx.ops.papers_read)
        usage_before = self.llm.meter.snapshot()

        labels = self._induce_and_label(ctx, v, read)
        ctx.read.update(read)
        child_ids = tax.children(v)
        router: Optional[NodeRouter] = None
        diag: Dict[str, Any] = {}
        read_set = set(read)
        rest = [p for p in P_v if p not in read_set]
        assignment: Dict[str, List[str]] = {}
        if child_ids:
            assignment.update({p: labels.get(p, []) for p in read})
            if not self.is_full or self.fit_full_routers:
                router, diag = self._fit_router(ctx, v, child_ids, {p: labels.get(p, []) for p in read})
            if rest:  # selection baselines only: FULL reads every paper
                assignment.update(zip(rest, router.route(ctx.matrix(rest))))
        self._apply(tax, v, assignment)
        pruned = self._prune_empty(tax, v) if (child_ids and self.prune_empty_children) else []
        used = [c for c in tax.children(v) if tax.nodes[c].papers]
        termination = TERM_SPLIT
        unsplit: Dict[str, int] = {}
        if len(used) < self.min_split_children:
            # Fewer than min_split_children (2) non-empty children is no split:
            # discard them and the residual they left, and make v a leaf, as
            # ScholarPath's InitNode does. The LLM cost is still recorded.
            termination = TERM_NO_SPLIT
            unsplit = {"unsplit_children": len(used), "unsplit_residual": len(node.residual)}
            tax.clear_children(v)
            router, diag = None, {}
        kept = tax.children(v)
        if router is not None and kept:
            idx = [router.child_ids.index(c) for c in kept]
            ctx.routers[v] = NodeRouter(v, kept, router.child_vecs[idx], router.router, router.threshold)

        n_v = len(P_v)
        spent = self.llm.meter.diff(usage_before)
        log.debug("%s episode %s (depth %d): n_v=%d read=%d children=%d residual=%d (%s)",
                  self.method, v, node.depth, n_v, len(read), len(kept), len(node.residual), termination)
        ctx.episodes.append(EpisodeRecord(
            node_id=v,
            depth=node.depth,
            n_v=n_v,
            n_children=len(kept),
            status=node.status if kept else LEAF,
            papers_read={k: n - reads_before.get(k, 0) for k, n in ctx.ops.papers_read.items()
                         if n - reads_before.get(k, 0) > 0},
            verified=len(read),
            abstention=len(node.residual) / max(1, n_v),
            disagreement=float(diag.get("disagreement", 0.0)),
            metadata={
                "termination": termination,
                "selected": len(read),
                "routed": len(rest) if kept else 0,
                "residual": len(node.residual),
                "pruned_children": len(pruned),
                "router_train": diag.get("router_train", 0),
                "router_val": diag.get("router_val", 0),
                "cost_usd": float(sum(r.cost_usd for r in spent.values())),
                "tokens": int(sum(r.total_tokens for r in spent.values())),
                **unsplit,
            },
        ))

    @staticmethod
    def _apply(tax: Taxonomy, v: str, assignment: Mapping[str, Sequence[str]]) -> None:
        """Assign papers to children of v (multi-label); no child -> residual leaf ⊥_v."""
        node = tax.nodes[v]
        children = set(node.children)
        for p, kids in assignment.items():
            kids = [c for c in kids if c in children]
            if kids:
                for c in kids:
                    tax.assign(p, c)
            else:
                node.residual.add(p)

    @staticmethod
    def _prune_empty(tax: Taxonomy, v: str) -> List[str]:
        empty = [c for c in tax.children(v) if not tax.nodes[c].papers]
        for c in empty:
            tax.remove_subtree(c)
        return empty


def build_dual_baselines(
    llm: LLMClient,
    embedder: Embedder,
    config: Optional[BaselineBuilderConfig],
    papers: PapersLike,
    paper_vectors: Optional[Mapping[str, np.ndarray]] = None,
    root_name: str = "Root",
    paper_ids: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Tuple[BuildResult, BuildResult]:
    """Problem and method taxonomies (T_P, T_M) built by the same baseline (arXiv sets, Sec. 5.2)."""
    builder = SelectionBuilder(llm, embedder, config, papers, paper_vectors, **kwargs)
    problem = builder.build(root_name, "Research problems studied in the repository.", paper_ids, facet="problem")
    method = builder.build(root_name, "Methods used or proposed in the repository.", paper_ids, facet="method")
    return problem, method


__all__ = [
    "COMPONENT_FULL",
    "COMPONENT_SELECTION",
    "FULL",
    "MODES",
    "NODE_WISE",
    "ONE_OFF",
    "SELECTORS",
    "SelectionBuilder",
    "TERM_NO_SPLIT",
    "TERM_SPLIT",
    "baseline_config",
    "build_dual_baselines",
    "estimate_full_cost",
    "internal_paper_count",
    "method_name",
    "parse_method",
    "per_paper_node_cost",
]
