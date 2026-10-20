"""Dynamic update of a certified taxonomy (Section 3.4, Appendix B.3).

When M new papers arrive at a certified node v of size n with bound
U_old <= eps, the frozen router f_theta_v assigns them at zero LLM cost.
Routing the arrivals leaves the assignments of the n existing papers
unchanged, so whenever the old loss is at most U_old, at most n U_old + M of
the n + M papers are misorganized, and the combined population satisfies
the worst-case bound (Equation 16)

    U_new = (n U_old + M) / (n + M),

which counts every unaudited arrival as an error and needs no assumption on
the new papers. As long as U_new <= eps and the residual leaf bottom_v stays
below the discovery threshold, the node remains certified and no LLM call is
made. "Once either condition fails, the builder audits the new cohort without
replacement" (App. B.3). It draws uniformly from the M arrivals and judges at
least min(M, ``audit_min_draws``) of them before it may certify. The old
part still contributes at most n U_old errors, so the cohort audit
certifies (n U_old + K^U) / (n + M) <= eps. New subtopics or routing drift
emerge when the cohort audit fails, or when the residual leaf crossed the
discovery threshold and the audit finds errors among the arrivals. Either
way the builder runs audit-guided repair (Stage 2: DISCOVER and CORRECT)
and then re-certifies a new snapshot as in Section 3.3. Every audit
continues the node's snapshot index, so the step levels alpha_{v,j,a}
still sum to at most delta_v.

Arrivals are processed top-down (BFS): the papers a node routes to a child
are that child's arrivals. A repair can also move existing papers between
children; a certified node that loses r of its n papers keeps the valid
bound n U_old / (n - r) (all errors may remain among the rest).

:class:`BuildResult.save` does not persist the frozen routers. On a reloaded
result :class:`DynamicUpdater` therefore retrains every missing router from
the node's persisted verified labels, at zero LLM cost (see
:meth:`NodeWiseBuilder.rebuild_router`). Existing assignments are never
re-routed, so the certificates stay valid.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from scholarpath.llm.accounting import UsageRecord, sum_usage
from scholarpath.taxonomy.builder import STATE_KEY, NodeWiseBuilder
from scholarpath.taxonomy.certification import AuditOutcome, DeltaAllocator, audit_snapshot
from scholarpath.taxonomy.result import BuildResult, EpisodeRecord
from scholarpath.taxonomy.tree import CERTIFIED, LEAF, Taxonomy
from scholarpath.types import Paper


def updated_bound(n: int, U_old: float, M: int) -> float:
    """Worst-case bound after M unaudited arrivals: (n U_old + M) / (n + M) (Eq. 16)."""
    if n + M <= 0:
        return 0.0
    return (n * float(U_old) + M) / float(n + M)


def absorbable_arrivals(n: int, U_old: float, epsilon: float) -> int:
    """Largest M with (n U_old + M) / (n + M) <= eps, i.e. floor(n (eps - U_old) / (1 - eps)).

    Example (Section 3.4): n = 1,000, U_old = 0.06, eps = 0.10 gives 44.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must lie in (0, 1), got {epsilon}")
    if U_old > epsilon:
        return 0
    M = int(n * (epsilon - float(U_old)) / (1.0 - epsilon) + 1e-9)
    while M > 0 and updated_bound(n, U_old, M) > epsilon:
        M -= 1
    return max(0, M)


@dataclass
class UpdateReport:
    """What one :meth:`DynamicUpdater.add_papers` call did."""

    n_new: int = 0
    skipped: List[str] = field(default_factory=list)  # ids already in the taxonomy
    arrivals: Dict[str, int] = field(default_factory=dict)  # node -> M (papers routed into it)
    kept: List[str] = field(default_factory=list)  # certificate kept at zero LLM cost (Eq. 16)
    reaudited: List[str] = field(default_factory=list)  # new cohort audited
    recertified: List[str] = field(default_factory=list)  # certified again after audit / repair
    repaired: List[str] = field(default_factory=list)  # Stage 2 repair + new snapshot
    failed: List[str] = field(default_factory=list)  # no certificate after repair
    uncertified: List[str] = field(default_factory=list)  # uncertified internal nodes, routed only
    grown_leaves: List[str] = field(default_factory=list)  # leaves now above the size limit
    new_nodes: List[str] = field(default_factory=list)  # children added by DISCOVER
    rebuilt_routers: List[str] = field(default_factory=list)  # routers retrained from persisted labels
    episodes: List[EpisodeRecord] = field(default_factory=list)
    usage: Dict[str, UsageRecord] = field(default_factory=dict)
    papers_read: Dict[str, int] = field(default_factory=dict)
    runtime_s: float = 0.0

    @property
    def llm_calls(self) -> int:
        return sum_usage(self.usage).calls

    @property
    def cost_usd(self) -> float:
        return sum_usage(self.usage).cost_usd

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["usage"] = {k: v.to_dict() for k, v in self.usage.items()}
        d["episodes"] = [e.to_dict() for e in self.episodes]
        d["llm_calls"] = self.llm_calls
        d["cost_usd"] = self.cost_usd
        return d


class DynamicUpdater:
    """Keeps a :class:`BuildResult` certified as new papers arrive (Section 3.4).

    Episodes run during an update are merged into ``result.episodes``, which
    keeps the latest episode of every node in the current taxonomy. These
    are the re-certification of a drifting node and the episodes of the
    children DISCOVER creates. Superseded episodes, and episodes of nodes a
    repair removed, are archived under
    ``result.metadata["superseded_episodes"]``. With that, ``result.summary()``
    matches the updated taxonomy.

    Args:
        builder: a :class:`NodeWiseBuilder` with the same LLM, embedder,
            config and repository as the one that produced ``result``.
        result: the build result. Its taxonomy, node metadata (verified
            labels, snapshot indices, certificates), routers and episodes are
            updated in place.
        rebuild_routers: retrain the routers that ``result`` lacks (e.g.
            after :meth:`BuildResult.save` and a reload) from the persisted
            verified labels, at zero LLM cost. If False, a missing router
            raises ``ValueError``.
    """

    def __init__(self, builder: NodeWiseBuilder, result: BuildResult, rebuild_routers: bool = True):
        self.builder = builder
        self.result = result
        self.tax: Taxonomy = result.taxonomy
        self.ops = builder.make_ops(self.tax.facet)
        self.rebuild_routers = bool(rebuild_routers)
        missing = self._missing_routers()
        if missing and not self.rebuild_routers:
            raise ValueError(
                f"{len(missing)} internal nodes have no frozen router (e.g. {missing[:3]}). "
                "BuildResult.save does not persist routers; pass rebuild_routers=True to retrain them "
                "from the persisted verified labels at zero LLM cost"
            )
        cfg = builder.config
        state = result.metadata.get("allocator")
        if state:
            self.allocator = DeltaAllocator.from_state(state)
        else:
            self.allocator = DeltaAllocator(cfg.delta, cfg.delta_allocation, max(1, self.tax.root.size),
                                            cfg.max_depth, cfg.max_nodes)

    @property
    def config(self):
        return self.builder.config

    # ------------------------------------------------------------------ api
    def add_papers(self, new_papers: Sequence[Paper]) -> UpdateReport:
        """Insert ``new_papers`` top-down and maintain every certificate."""
        t0 = time.perf_counter()
        meter = self.builder.llm.meter
        usage_before = meter.snapshot()
        reads_before = dict(self.ops.papers_read)
        report = UpdateReport()
        # Before touching the taxonomy: every internal node needs its router.
        for nid in self._missing_routers():
            self._rebuild_router(nid)
            report.rebuilt_routers.append(nid)
        known = self.tax.root.papers
        fresh: List[str] = []
        for p in new_papers:
            if p.paper_id in known or p.paper_id in fresh:
                report.skipped.append(p.paper_id)
                continue
            self.builder.papers[p.paper_id] = p
            fresh.append(p.paper_id)
        report.n_new = len(fresh)
        if fresh:
            self.builder.vectors(fresh)
            root = self.tax.root_id
            for pid in fresh:
                self.tax.assign(pid, root)
            arrivals: Dict[str, Set[str]] = {root: set(fresh)}
            departures: Dict[str, Set[str]] = {}
            queue: Deque[str] = deque([root])
            queued = {root}
            while queue:
                nid = queue.popleft()
                queued.discard(nid)
                if nid not in self.tax.nodes:
                    continue
                arr = arrivals.pop(nid, set())
                dep = departures.pop(nid, set())
                if not arr and not dep:
                    continue
                if arr:
                    report.arrivals[nid] = report.arrivals.get(nid, 0) + len(arr)
                added, departed = self._process(nid, arr, dep, report)
                for c, ps in added.items():
                    if ps:
                        arrivals.setdefault(c, set()).update(ps)
                for d, ps in departed.items():
                    if ps:
                        departures.setdefault(d, set()).update(ps)
                for c in self.tax.children(nid):
                    if (c in arrivals or c in departures) and c not in queued:
                        queue.append(c)
                        queued.add(c)
            self.tax.invalidate()
        self._merge_episodes(report.episodes)
        self.result.metadata["allocator"] = self.allocator.state_dict()
        report.usage = meter.diff(usage_before)
        report.papers_read = {
            k: v - reads_before.get(k, 0) for k, v in self.ops.papers_read.items() if v - reads_before.get(k, 0) > 0
        }
        report.runtime_s = time.perf_counter() - t0
        updates = self.result.metadata.setdefault("updates", [])
        updates.append({k: v for k, v in report.to_dict().items() if k not in ("episodes",)})
        return report

    # ------------------------------------------------------------ internals
    def _process(
        self, nid: str, arrivals: Set[str], departures: Set[str], report: UpdateReport
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        b = self.builder
        cfg = self.config
        tax = self.tax
        node = tax.nodes[nid]
        meta = node.metadata.setdefault(STATE_KEY, {})
        if departures:
            self._forget(nid, departures)
        if not tax.children(nid):
            # A leaf that crossed the size limit with these arrivals (a rebuild of
            # its subtree would now run an episode for it).
            crossed = node.size - len(arrivals) < cfg.min_node_size
            if arrivals and crossed and b.splittable(node) and nid not in report.grown_leaves:
                report.grown_leaves.append(nid)
            return {}, {}
        router = self.result.routers.get(nid)
        if router is None:  # e.g. an internal node created by a caller after the build
            router = self._rebuild_router(nid)
            report.rebuilt_routers.append(nid)
        arr = sorted(arrivals)
        routed: Dict[str, Tuple[str, ...]] = {}
        if arr:
            routed = {p: tuple(kids) for p, kids in zip(arr, router.route(b.vectors(arr)))}
        cert = meta.get("certificate") if node.status == CERTIFIED else None
        if cert is None:
            if arr:
                report.uncertified.append(nid)
            added, departed = b.apply_assignments(tax, nid, routed)
            meta["n"] = node.size
            return added, departed

        n_old = int(cert["n_v"])
        U_old = float(cert["bound"])
        M = len(arr)
        U_new = updated_bound(n_old, U_old, M)  # Equation (16)
        new_residual = {p for p in arr if not routed[p]}
        residual_after = len(node.residual | new_residual)
        threshold = b.repair_threshold(n_old + M)
        bound_ok = U_new <= cfg.epsilon
        # |bottom_v| "stays below the discovery threshold": the arrivals neither push the
        # residual leaf to the threshold nor grow it while it is already at or above it.
        residual_ok = not new_residual or residual_after < threshold
        if bound_ok and residual_ok:
            added, departed = b.apply_assignments(tax, nid, routed)
            self._update_certificate(node, n_old + M, U_new)
            report.kept.append(nid)
            return added, departed

        # Either condition failed: audit the new cohort (continuing the snapshot index j).
        outcome, labels = self._audit_cohort(nid, arr, routed, n_old, U_old)
        if outcome.draws > 0:
            report.reaudited.append(nid)
        final = {p: (labels[p][0] if p in labels else routed[p]) for p in arr}
        self._remember(nid, labels)
        # New subtopics or routing drift emerge when the cohort audit fails, or when the
        # residual leaf crossed the discovery threshold and the audited arrivals contain errors.
        drift = not outcome.certified or (not residual_ok and outcome.found_errors)
        if not drift:
            added, departed = b.apply_assignments(tax, nid, final)
            self._update_certificate(node, n_old + M, outcome.bound, outcome)
            report.recertified.append(nid)
            return added, departed

        # Drift: audit-guided repair (Stage 2) and re-certification of a new snapshot.
        before_children = set(tax.children(nid))
        state = b.resume_node(tax, nid, self.ops)
        b.recertify(state, tax, self.ops)
        added, departed, episode = b.finalize(state, tax, self.ops, self.result, record=False)
        episode.metadata.update({
            "dynamic_update": True,
            "trigger": [k for k, failed in (("bound", not bound_ok), ("residual", not residual_ok)) if failed],
            "cohort_audit": outcome.summary(),
        })
        report.episodes.append(episode)
        report.repaired.append(nid)
        if state.status == CERTIFIED:
            report.recertified.append(nid)
        else:
            report.failed.append(nid)
        new_kids = [c for c in tax.children(nid) if c not in before_children]
        if new_kids:
            report.new_nodes.extend(new_kids)
            scratch = BuildResult(taxonomy=tax, method=self.result.method)
            todo: Deque[str] = deque()
            for c in new_kids:
                added.pop(c, None)
                if b.splittable(tax.nodes[c]):
                    todo.append(c)
                else:
                    tax.nodes[c].status = LEAF
            b.process_queue(tax, todo, self.ops, self.allocator, scratch)
            self.result.routers.update(scratch.routers)
            report.episodes.extend(scratch.episodes)
        return added, departed

    def _audit_cohort(
        self, nid: str, cohort: Sequence[str], routed: Mapping[str, Tuple[str, ...]], n_old: int, U_old: float
    ) -> Tuple[AuditOutcome, Dict[str, Tuple[Tuple[str, ...], bool]]]:
        """Audit the M arrivals: U = (n U_old + K^U_{v,j,a}) / (n + M) with a new snapshot j."""
        cfg = self.config
        tax = self.tax
        node = tax.nodes[nid]
        meta = node.metadata[STATE_KEY]
        j = int(meta.get("next_snapshot", 1))
        children = tax.children(nid)
        valid = set(children)
        judged: Dict[str, Tuple[Tuple[str, ...], bool]] = {}

        def judge(batch: List[str]) -> Dict[str, bool]:
            lb = self.ops.label(tax, nid, children, batch, component="auditing", allow_new=False)
            out: Dict[str, bool] = {}
            for pid, lab in lb.labels.items():
                kids = tuple(sorted(c for c in lab.children if c in valid))
                missing = bool(lab.missing) or not kids
                judged[pid] = (kids, missing)
                out[pid] = missing or set(kids) != set(routed.get(pid, ()))
            return out

        delta_v = float(node.delta if node.delta is not None else meta.get("delta_v", 0.0))
        rng = self.builder._rng(tax.facet, nid, salt=1000 + j)
        outcome = audit_snapshot(
            list(cohort), judge, n_old + len(cohort), n_old * U_old, delta_v, j, cfg.epsilon,
            cfg.audit_batch_size, rng, cfg.audit_abort_factor, cfg.audit_min_draws,
            max_draws=cfg.max_llm_papers_per_node,
            # A cohort audit looks at real papers: it may not certify on U_{v,j,0} = U_new alone.
            min_certify_draws=max(1, int(cfg.audit_min_draws)),
        )
        meta["next_snapshot"] = j + 1
        meta.setdefault("update_audits", []).append(outcome.summary())
        return outcome, judged

    def _missing_routers(self) -> List[str]:
        """Internal nodes (with children) that have no frozen router in the result."""
        return sorted(nid for nid in self.tax.nodes if self.tax.children(nid) and nid not in self.result.routers)

    def _rebuild_router(self, nid: str) -> Any:
        if not self.rebuild_routers:
            raise ValueError(f"internal node {nid!r} has no frozen router (rebuild_routers=False)")
        router = self.builder.rebuild_router(self.tax, nid)
        self.result.routers[nid] = router
        self.tax.nodes[nid].metadata.setdefault(STATE_KEY, {})["router_rebuilt"] = True
        return router

    def _merge_episodes(self, episodes: Sequence[EpisodeRecord]) -> None:
        """Keep ``result.episodes`` = latest episode of every node of the current taxonomy."""
        latest = {e.node_id: e for e in episodes if e.node_id in self.tax.nodes}
        merged: List[EpisodeRecord] = []
        archived: List[Dict[str, Any]] = []
        placed: Set[str] = set()
        for e in self.result.episodes:  # a new episode takes the place of the one it supersedes
            new = latest.get(e.node_id)
            keep = e.node_id in self.tax.nodes and e.node_id not in placed
            if new is not None or not keep:
                archived.append(e.to_dict())
            if keep:
                merged.append(e if new is None else new)
                placed.add(e.node_id)
        for e in episodes:
            if latest.get(e.node_id) is e and e.node_id not in placed:
                merged.append(e)
                placed.add(e.node_id)
            elif latest.get(e.node_id) is not e:
                archived.append(e.to_dict())
        self.result.episodes = merged
        if archived:
            self.result.metadata.setdefault("superseded_episodes", []).extend(archived)

    @staticmethod
    def _update_certificate(node: Any, n: int, bound: float, outcome: Optional[AuditOutcome] = None) -> None:
        meta = node.metadata.setdefault(STATE_KEY, {})
        cert = dict(meta.get("certificate") or {})
        if outcome is not None:
            cert.update(outcome.summary())
        cert["n_v"] = int(n)
        cert["bound"] = float(bound)
        cert["updated"] = True
        meta["certificate"] = cert
        meta["n"] = int(n)
        node.bound = float(bound)
        node.status = CERTIFIED

    def _remember(self, nid: str, labels: Mapping[str, Tuple[Tuple[str, ...], bool]]) -> None:
        """Persist audited cohort labels as verified (held-out) labels of the node."""
        if not labels:
            return
        meta = self.tax.nodes[nid].metadata.setdefault(STATE_KEY, {})
        verified = dict(meta.get("verified") or {})
        missing = set(meta.get("missing") or [])
        audited = set(meta.get("audited") or [])
        for pid, (kids, miss) in labels.items():
            verified[pid] = list(kids)
            audited.add(pid)
            if miss:
                missing.add(pid)
            else:
                missing.discard(pid)
        meta["verified"] = dict(sorted(verified.items()))
        meta["missing"] = sorted(missing)
        meta["audited"] = sorted(audited)

    def _forget(self, nid: str, departed: Set[str]) -> None:
        """Papers moved out of the node: drop their state; loosen the certificate to n U / (n - r)."""
        node = self.tax.nodes[nid]
        meta = node.metadata.setdefault(STATE_KEY, {})
        if meta.get("verified"):
            meta["verified"] = {p: k for p, k in meta["verified"].items() if p not in departed}
        for key in ("missing", "audited"):
            if meta.get(key):
                meta[key] = [p for p in meta[key] if p not in departed]
        node.residual -= departed
        cert = meta.get("certificate")
        if node.status == CERTIFIED and cert:
            n = int(cert["n_v"])
            r = min(len(departed), n)
            remaining = n - r
            bound = 0.0 if remaining <= 0 else min(1.0, n * float(cert["bound"]) / remaining)
            self._update_certificate(node, remaining, bound)
            if bound > self.config.epsilon:
                # Still marked certified with the (now too loose) bound so that
                # _process sends it through the cohort audit / repair path.
                meta["certificate"]["loosened"] = True
