"""RAPTOR baseline [61] (Table 2, "RAPTOR"): recursive abstractive tree retrieval.

Offline tree construction (Sarthi et al., ICLR 2024, Section 3):

1. **Leaves** are the passages of the pool (abstracts and section chunks),
   embedded with the configured embedder.
2. **Clustering.** Node embeddings of the current layer are reduced
   (UMAP, ``n_neighbors = sqrt(n - 1)`` globally and 10 locally, to
   ``reduction_dim = 10`` dimensions; PCA when ``umap-learn`` is not
   installed) and clustered with a Gaussian mixture whose number of
   components is chosen by BIC. RAPTOR's two-stage scheme is followed: a
   global clustering, then a local clustering inside every global cluster.
   Clustering is *soft*: a node joins every cluster whose posterior exceeds
   ``threshold = 0.1``. Clusters whose text exceeds ``max_cluster_tokens``
   (3,500) are re-clustered recursively; when re-clustering makes no progress
   (RAPTOR would recurse forever), the cluster is bisected (2-means, or a
   token-balanced cut along its first principal axis when 2-means is
   degenerate) until every part fits the cap or is a single node. A k-means
   variant (``clustering="kmeans"``) is provided as a lightweight fallback.
3. **Summarization.** Each cluster is summarized by the LLM (task
   ``baselines.raptor_summarize``, cost component ``index_build``) and the
   summary is embedded, forming the next layer.
4. Steps 2-3 repeat until a layer has at most ``reduction_dim + 1`` nodes,
   clustering no longer reduces the layer, or ``max_layers`` is reached.

Query time uses RAPTOR's **collapsed-tree** strategy: every node of every
layer is ranked by cosine similarity to the query embedding and the ranking
is packed to the budget B (skip-and-continue, deduplicated).

**Summary citations.** A ~100-token summary cannot carry the evidence of
every paper below it, so crediting the whole subtree would let one summary
"retrieve" hundreds of papers and make Recall@B / NDCG@B (computed over the
distinct source papers of the packed units, Table 2 / Fig. 5) meaningless --
at the top layers it would credit the entire candidate pool regardless of B.
A summary node therefore cites at most ``max_sources_per_summary`` (default
5, the same cap as LightRAG's graph units) papers, chosen among the leaves
whose text actually reached its summarization prompt (the cluster text is
truncated to ``max_cluster_tokens``; for higher layers, the covered leaves of
the covered children), ordered by their contribution to the query match: the
maximum query similarity of the paper's covered leaves under the node, ties
broken by the number of such leaves. The cited ids form the unit's block
header, so they count toward B. ``max_sources_per_summary = None`` restores
the uncapped union (an ablation only; it inflates Recall@B).

**Candidate pool.** Building a tree over all units of 148K papers costs
O(#units / cluster size) LLM summaries per layer, which is prohibitive.
With ``candidate_retriever`` set (e.g. the SPECTER2
:class:`~scholarpath.baselines.dense.DenseRetriever`) and
``candidate_papers = k``, a tree is built per query over the units of the
top-k candidate papers only (trees are cached by candidate set and
summaries hit the LLM response cache). Its summarization calls are billed to
``index_build``, and its wall time is reported as
``RetrievalResult.stats["index_build_s"]`` and excluded from
``RetrievalResult.latency_s``; per-query QA accounting excludes both (Table 5
counts online calls only) but reports them separately (the "Index-build"
columns of run_efficiency.py's ``table5_detail``). Building the tree at query
time is a deviation from the paper's globally indexed baseline, used only on
corpora above ``SystemOptions.graph_global_max_papers`` (arXiv-Subfields).
"""

from __future__ import annotations

import json
import logging
import math
import time
import warnings
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from scholarpath.baselines.common import (
    SUMMARY,
    Candidate,
    Passage,
    PassagePool,
    QueryLike,
    as_query,
    ensure_built,
    top_k_indices,
)
from scholarpath.baselines.packing import BudgetPacker
from scholarpath.embeddings import Embedder, l2_normalize
from scholarpath.llm.base import LLMClient
from scholarpath.prompts.baselines import raptor_cluster_coverage, raptor_summarize_request
from scholarpath.retrieval.result import RetrievalResult
from scholarpath.text import count_tokens

log = logging.getLogger(__name__)


@dataclass
class RaptorConfig:
    """RAPTOR hyper-parameters (defaults from the RAPTOR reference implementation)."""

    clustering: str = "gmm"  # gmm | kmeans
    reduction_dim: int = 10
    use_umap: bool = True  # falls back to PCA when umap-learn is missing
    local_neighbors: int = 10
    max_clusters: int = 50  # BIC search range 1 .. min(max_clusters, n) - 1
    threshold: float = 0.1  # soft-clustering posterior threshold
    soft_clustering: bool = True
    max_cluster_tokens: int = 3500  # max_length_in_cluster
    kmeans_cluster_size: int = 10  # k = ceil(n / size) for clustering="kmeans"
    max_layers: int = 5
    summary_max_tokens: int = 100  # summarization_length
    max_sources_per_summary: Optional[int] = 5  # None = uncapped union of covered leaf papers (ablation)
    candidate_papers: Optional[int] = None  # per-query tree over top-k candidate papers
    tree_cache_size: int = 16
    max_workers: Optional[int] = None
    seed: int = 0


@dataclass
class RaptorNode:
    node_id: int
    layer: int  # 0 = leaf
    text: str
    children: List[int] = field(default_factory=list)
    leaves: List[int] = field(default_factory=list)  # leaf node ids below (leaves: itself)
    passage_key: Optional[str] = None  # leaves only
    paper_counts: Dict[str, int] = field(default_factory=dict)  # source paper -> #covered leaves
    #: Leaves whose text reached this node's summary (None = all ``leaves``).
    covered_leaves: Optional[List[int]] = None

    @property
    def cited_leaves(self) -> List[int]:
        return self.leaves if self.covered_leaves is None else self.covered_leaves


class RaptorTree:
    """A RAPTOR tree: nodes of all layers with their embeddings."""

    def __init__(self, nodes: List[RaptorNode], embeddings: np.ndarray):
        self.nodes = nodes
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    @property
    def num_layers(self) -> int:
        return 1 + max((n.layer for n in self.nodes), default=0)

    def layer_sizes(self) -> List[int]:
        c = Counter(n.layer for n in self.nodes)
        return [c[i] for i in range(self.num_layers)]

    @property
    def summary_nodes(self) -> List[RaptorNode]:
        return [n for n in self.nodes if n.layer > 0]

    def save(self, directory: Union[str, Path]) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "raptor_tree.json").write_text(json.dumps([asdict(n) for n in self.nodes]), encoding="utf-8")
        np.save(d / "raptor_embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "RaptorTree":
        d = Path(directory)
        nodes = [RaptorNode(**n) for n in json.loads((d / "raptor_tree.json").read_text(encoding="utf-8"))]
        return cls(nodes, np.load(d / "raptor_embeddings.npy"))


# ---------------------------------------------------------------------------
# Clustering (RAPTOR "cluster_utils")
# ---------------------------------------------------------------------------


def reduce_dimension(X: np.ndarray, dim: int, n_neighbors: int, use_umap: bool = True, seed: int = 0) -> np.ndarray:
    """UMAP (cosine) reduction as in RAPTOR; PCA when umap-learn is unavailable."""
    n, d = X.shape
    dim = max(1, min(int(dim), n - 2 if n > 2 else 1, d))
    if use_umap:
        try:
            import umap  # type: ignore

            return np.asarray(
                umap.UMAP(n_neighbors=max(2, int(n_neighbors)), n_components=dim, metric="cosine",
                          random_state=seed).fit_transform(X),
                dtype=np.float64,
            )
        except ImportError:
            pass
    from sklearn.decomposition import PCA

    return PCA(n_components=dim, random_state=seed).fit_transform(np.asarray(X, dtype=np.float64))


def optimal_gmm_clusters(X: np.ndarray, max_clusters: int = 50, seed: int = 0) -> int:
    """Number of GMM components minimizing BIC over 1 .. min(max_clusters, n) - 1."""
    from sklearn.mixture import GaussianMixture

    hi = min(int(max_clusters), X.shape[0])
    candidates = list(range(1, max(2, hi)))
    bics = []
    for k in candidates:
        gm = GaussianMixture(n_components=k, random_state=seed).fit(X)
        bics.append(gm.bic(X))
    return candidates[int(np.argmin(bics))]


def gmm_cluster(
    X: np.ndarray, threshold: float, max_clusters: int, soft: bool = True, seed: int = 0
) -> Tuple[List[np.ndarray], int]:
    """(cluster labels per point, #clusters) of a BIC-selected Gaussian mixture.

    A point joins every component whose posterior exceeds ``threshold``; a
    point with no such component joins its argmax component (RAPTOR drops it
    silently, which loses content).
    """
    from sklearn.mixture import GaussianMixture

    k = optimal_gmm_clusters(X, max_clusters, seed)
    gm = GaussianMixture(n_components=k, random_state=seed).fit(X)
    probs = gm.predict_proba(X)
    labels: List[np.ndarray] = []
    for prob in probs:
        lab = np.where(prob > threshold)[0] if soft else np.array([int(np.argmax(prob))])
        if lab.size == 0:
            lab = np.array([int(np.argmax(prob))])
        labels.append(lab)
    return labels, k


def raptor_clustering(X: np.ndarray, config: RaptorConfig) -> List[List[int]]:
    """RAPTOR ``perform_clustering``: global then local GMM clustering.

    Returns clusters as lists of row indices (possibly overlapping).
    """
    with warnings.catch_warnings():
        # Duplicate passages (identical embeddings) make sklearn warn about
        # degenerate k-means initializations and zero explained variance.
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return _raptor_clustering(X, config)


def _raptor_clustering(X: np.ndarray, config: RaptorConfig) -> List[List[int]]:
    n = X.shape[0]
    dim = config.reduction_dim
    if n <= 2:
        return [list(range(n))]
    if config.clustering == "kmeans":
        from scholarpath.taxonomy.regions import kmeans_labels

        k = max(1, math.ceil(n / max(1, config.kmeans_cluster_size)))
        lab = kmeans_labels(np.asarray(X, dtype=np.float64), k, seed=config.seed)
        groups: Dict[int, List[int]] = {}
        for i, g in enumerate(lab.tolist()):
            groups.setdefault(int(g), []).append(i)
        return [groups[g] for g in sorted(groups)]
    reduced = reduce_dimension(X, min(dim, n - 2), int((n - 1) ** 0.5), config.use_umap, config.seed)
    global_labels, n_global = gmm_cluster(reduced, config.threshold, config.max_clusters,
                                          config.soft_clustering, config.seed)
    clusters: List[List[int]] = []
    for g in range(n_global):
        members = [i for i, lab in enumerate(global_labels) if g in lab]
        if not members:
            continue
        if len(members) <= dim + 1:
            clusters.append(members)
            continue
        sub = X[members]
        red = reduce_dimension(sub, dim, config.local_neighbors, config.use_umap, config.seed)
        local_labels, n_local = gmm_cluster(red, config.threshold, config.max_clusters,
                                            config.soft_clustering, config.seed)
        for c in range(n_local):
            local = [members[j] for j, lab in enumerate(local_labels) if c in lab]
            if local:
                clusters.append(local)
    # Drop exact duplicate clusters (soft assignment can repeat them).
    uniq: "OrderedDict[Tuple[int, ...], None]" = OrderedDict()
    for c in clusters:
        uniq.setdefault(tuple(sorted(set(c))), None)
    return [list(c) for c in uniq]


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------


class RaptorBuilder:
    """Builds a :class:`RaptorTree` over a list of passages."""

    def __init__(self, llm: LLMClient, embedder: Embedder, config: Optional[RaptorConfig] = None):
        self.llm = llm
        self.embedder = embedder
        self.config = config or RaptorConfig()

    def _cluster_nodes(
        self, ids: List[int], tokens: List[int], emb: np.ndarray, depth: int = 0
    ) -> List[List[int]]:
        """Cluster the given node ids; re-cluster clusters over the token cap."""
        cfg = self.config
        X = emb[ids]
        out: List[List[int]] = []
        for cl in raptor_clustering(X, cfg):
            members = [ids[j] for j in cl]
            if len(members) == 1 or sum(tokens[m] for m in members) <= cfg.max_cluster_tokens:
                out.append(members)
            elif len(members) == len(ids) or depth >= 8:  # no progress: bisect until it fits
                out.extend(self._split(members, tokens, emb))
            else:
                out.extend(self._cluster_nodes(members, tokens, emb, depth + 1))
        return out

    def _split(self, members: List[int], tokens: List[int], emb: np.ndarray) -> List[List[int]]:
        """Bisect an oversized cluster until every part fits ``max_cluster_tokens``
        or is a single node (iterative, order-preserving)."""
        cap = self.config.max_cluster_tokens
        out: List[List[int]] = []
        stack = [members]
        while stack:
            part = stack.pop()
            if len(part) == 1 or sum(tokens[m] for m in part) <= cap:
                out.append(part)
                continue
            stack.extend(reversed(self._bisect(part, tokens, emb)))
        return out

    def _bisect(self, members: List[int], tokens: List[int], emb: np.ndarray) -> List[List[int]]:
        """2-means split; a token-balanced cut along the first principal axis when
        2-means is lopsided (a part with at most 1/4 of the nodes, e.g. on near-duplicate
        texts, which would otherwise be peeled off one node at a time)."""
        from scholarpath.taxonomy.regions import kmeans_labels

        if len(members) == 2:
            return [[members[0]], [members[1]]]
        X = np.asarray(emb[members], dtype=np.float64)
        lab = kmeans_labels(X, 2, seed=self.config.seed).tolist()
        parts = [[m for m, g in zip(members, lab) if g == c] for c in (0, 1)]
        if 4 * min(len(p) for p in parts) > len(members):  # balanced enough (> 1/4 each)
            return parts
        centered = X - X.mean(axis=0, keepdims=True)
        try:
            axis = np.linalg.svd(centered, full_matrices=False)[2][0]
            proj = centered @ axis
        except np.linalg.LinAlgError:  # pragma: no cover - numerically degenerate
            proj = np.zeros(len(members))
        order = sorted(range(len(members)), key=lambda j: (float(proj[j]), j))
        total = sum(tokens[members[j]] for j in order)
        acc, cut = 0, 1
        for pos, j in enumerate(order[:-1]):
            acc += tokens[members[j]]
            cut = pos + 1
            if acc * 2 >= total:
                break
        first = sorted(order[:cut])
        second = sorted(order[cut:])
        return [[members[j] for j in first], [members[j] for j in second]]

    def build(self, passages: Sequence[Passage], leaf_vectors: Optional[np.ndarray] = None) -> RaptorTree:
        cfg = self.config
        nodes: List[RaptorNode] = []
        for i, p in enumerate(passages):
            counts = {pid: 1 for pid in p.papers[:1]}
            nodes.append(RaptorNode(i, 0, p.search_text, [], [i], p.key, counts))
        tokens = [count_tokens(n.text) for n in nodes]
        if leaf_vectors is None:
            leaf_vectors = self.embedder.encode([n.text for n in nodes]) if nodes else np.zeros((0, self.embedder.dim))
        emb = l2_normalize(np.asarray(leaf_vectors, dtype=np.float32)) if len(nodes) else np.zeros((0, 1), np.float32)
        layer_ids = list(range(len(nodes)))
        for layer in range(1, cfg.max_layers + 1):
            if len(layer_ids) <= cfg.reduction_dim + 1:
                break
            clusters = self._cluster_nodes(layer_ids, tokens, emb)
            if len(clusters) >= len(layer_ids):
                break  # clustering no longer reduces the layer
            t0 = time.perf_counter()
            reqs = [
                raptor_summarize_request([nodes[m].text for m in cl], max_tokens=cfg.summary_max_tokens,
                                         max_input_tokens=cfg.max_cluster_tokens)
                for cl in clusters
            ]
            resps = self.llm.map(self.llm.complete, reqs, max_workers=cfg.max_workers)
            summaries = [r.text.strip() for r in resps]
            vecs = self.embedder.encode(summaries)
            new_ids: List[int] = []
            for cl, text in zip(clusters, summaries):
                leaves = sorted({lf for m in cl for lf in nodes[m].leaves})
                covered_children = [m for m, ok in zip(cl, raptor_cluster_coverage(
                    [nodes[m].text for m in cl], cfg.max_cluster_tokens)) if ok]
                covered = sorted({lf for m in covered_children for lf in nodes[m].cited_leaves})
                counts: Counter = Counter()
                for lf in covered:
                    counts.update(nodes[lf].paper_counts)
                nid = len(nodes)
                nodes.append(RaptorNode(nid, layer, text, list(cl), leaves, None, dict(counts),
                                        None if covered == leaves else covered))
                tokens.append(count_tokens(text))
                new_ids.append(nid)
            emb = np.concatenate([emb, l2_normalize(np.asarray(vecs, dtype=np.float32))], axis=0)
            log.info("RAPTOR layer %d: %d -> %d nodes (%.1fs)", layer, len(layer_ids), len(new_ids),
                     time.perf_counter() - t0)
            layer_ids = new_ids
        return RaptorTree(nodes, emb)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class RaptorRetriever:
    """Collapsed-tree RAPTOR retrieval packed to the budget B."""

    def __init__(
        self,
        pool: PassagePool,
        llm: LLMClient,
        embedder: Embedder,
        config: Optional[RaptorConfig] = None,
        candidate_retriever: Optional[Any] = None,
        tree: Optional[RaptorTree] = None,
        name: str = "RAPTOR",
    ):
        self.pool = pool
        self.llm = llm
        self.embedder = embedder
        self.config = config or RaptorConfig()
        self.candidate_retriever = candidate_retriever
        self.name = name
        self.builder = RaptorBuilder(llm, embedder, self.config)
        self.tree: Optional[RaptorTree] = tree
        self._tree_pool: Optional[List[int]] = list(range(len(pool))) if tree is not None else None
        self._cache: "OrderedDict[Tuple[int, ...], RaptorTree]" = OrderedDict()

    @property
    def per_query(self) -> bool:
        return self.candidate_retriever is not None and bool(self.config.candidate_papers)

    def _leaf_vectors(self, indices: Sequence[int]) -> Optional[np.ndarray]:
        cr = self.candidate_retriever
        if cr is not None and getattr(cr, "embedder", None) is self.embedder and getattr(cr, "pool", None) is self.pool:
            return cr.vectors(indices)
        return None

    def build_tree(self, indices: Sequence[int]) -> RaptorTree:
        """Build (or fetch from the cache) the tree over the given pool units."""
        key = tuple(sorted(int(i) for i in indices))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with self.llm.meter.scope("index_build"):
            tree = self.builder.build([self.pool[i] for i in key], self._leaf_vectors(key))
        self._cache[key] = tree
        while len(self._cache) > max(1, self.config.tree_cache_size):
            self._cache.popitem(last=False)
        return tree

    def build(self) -> "RaptorRetriever":
        """Build the global tree over the whole pool (offline indexing)."""
        idx = list(range(len(self.pool)))
        self.tree = self.build_tree(idx)
        self._tree_pool = idx
        return self

    def ensure_built(self) -> float:
        """Build whatever offline index a query needs now; seconds spent (0 if built).

        Global mode builds the tree; candidate-pool mode builds the candidate
        retriever's index (per-query trees are built in :meth:`retrieve`).
        """
        if self.per_query:
            return ensure_built(self.candidate_retriever)
        if self.tree is not None:
            return 0.0
        t0 = time.perf_counter()
        self.build()
        return time.perf_counter() - t0

    def _tree_for(self, query_text: str) -> Tuple[RaptorTree, List[int], Dict[str, Any], float]:
        """(tree, pool index of each leaf, stats, index-build seconds) for a query."""
        index_s = self.ensure_built()
        if self.per_query:
            k = int(self.config.candidate_papers or 0)
            papers = self.candidate_retriever.top_papers(query_text, k)
            key = sorted(set(self.pool.indices_for_papers(papers)))
            t0 = time.perf_counter()
            tree = self.build_tree(key)
            index_s += time.perf_counter() - t0
            return tree, key, {"candidate_papers": len(papers)}, index_s
        assert self.tree is not None and self._tree_pool is not None
        return self.tree, self._tree_pool, {}, index_s

    def retrieve(self, query: QueryLike, budget: int) -> RetrievalResult:
        t0 = time.perf_counter()
        q = as_query(query)
        tree, leaf_pool_idx, extra, index_s = self._tree_for(q.text)
        qv = np.asarray(self.embedder.encode_queries([q.text])[0], dtype=np.float32)
        sims = tree.embeddings @ qv if len(tree.nodes) else np.zeros(0, dtype=np.float32)
        order = top_k_indices(sims, len(tree.nodes))
        leaf_sim = {n.node_id: float(sims[n.node_id]) for n in tree.nodes if n.layer == 0}
        candidates: List[Candidate] = []
        for nid in order:
            node = tree.nodes[int(nid)]
            if node.layer == 0:
                passage = self.pool[leaf_pool_idx[node.node_id]]
            else:
                passage = self._summary_passage(tree, node, leaf_sim)
            candidates.append(Candidate(passage, float(sims[nid]), f"layer{node.layer}"))
        packer = BudgetPacker(budget)
        packer.extend(candidates)
        stats = {"method": self.name, "tree_nodes": len(tree.nodes), "layers": tree.layer_sizes(),
                 "index_build_s": index_s, **extra}
        return packer.result(q.query_id, titles=self.pool.titles, stats=stats,
                             latency_s=max(0.0, time.perf_counter() - t0 - index_s))

    def _summary_passage(self, tree: RaptorTree, node: RaptorNode, leaf_sim: Dict[int, float]) -> Passage:
        """Summary unit citing its most query-relevant covered leaf papers (module docstring)."""
        best: Dict[str, float] = {}
        for lf in node.cited_leaves:
            s = leaf_sim.get(lf, -1.0)
            for pid in tree.nodes[lf].paper_counts:
                if s > best.get(pid, -2.0):
                    best[pid] = s
        papers = sorted(best, key=lambda p: (-best[p], -node.paper_counts.get(p, 0), p))
        cap = self.config.max_sources_per_summary
        if cap is not None:
            papers = papers[: max(0, int(cap))]
        return Passage(
            key=f"raptor:{node.layer}:{node.node_id}:{len(node.leaves)}",
            kind=SUMMARY,
            papers=papers,
            title=f"Summary (RAPTOR layer {node.layer}, {len(node.leaves)} passages)",
            body=node.text,
            index_text=node.text,
            metadata={"layer": node.layer, "num_leaves": len(node.leaves),
                      "num_cited_leaves": len(node.cited_leaves),
                      "num_leaf_papers": len(node.paper_counts)},
        )
