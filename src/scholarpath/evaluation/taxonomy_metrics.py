"""Soft node F1 and soft edge F1 of a taxonomy against a reference (Section 5.2).

The paper measures structural quality with soft node F1 and soft edge F1
[26, 40, 45]: on the arXiv sets every method is compared against the FULL
taxonomy (exhaustive LLM construction), on TaxoBench against the merged
expert gold taxonomy (only concept nodes are compared; papers are not nodes).

Definitions
-----------
Let V̂ and V* be the concept nodes of the predicted and the reference
taxonomy (roots excluded by default: the root is a placeholder for the whole
repository, not a concept). Each node u is represented by the embedding e(u)
of its label -- the node name (default) or "name: definition" -- and

    s(u, u') = clip(cos(e(u), e(u')), 0, 1).

*Soft node F1.*

    P_node = (1/|V̂|)  Σ_{u ∈ V̂}  max_{u' ∈ V*} s(u, u')
    R_node = (1/|V*|) Σ_{u' ∈ V*} max_{u ∈ V̂}  s(u, u')
    F1     = 2 P R / (P + R)            (0 when P + R = 0)

*Soft edge F1.* Edges are the parent → child edges Ê and E*. A predicted edge
(p, c) is similar to a reference edge (p', c') only if *both* endpoints match:

    s_E((p, c), (p', c')) = min(s(p, p'), s(c, c'))

(the combiner is configurable: ``"min"`` (default), ``"product"``,
``"mean"``, ``"geometric"`` or a callable), and P_edge, R_edge, F1_edge are
defined as above with edges in place of nodes. Edges leaving the root are
included by default; when roots are excluded from the node sets, the two roots
are treated as aligned virtual nodes, s(r̂, r*) = 1 and s(root, u) = 0 for
every concept u, so a top-level edge (r̂, c) matches a top-level reference
edge (r*, c') with similarity s(c, c').

*Fuzzy variant* (threshold τ): a node / edge counts as matched when its best
similarity is at least τ, i.e. P = |{u ∈ V̂ : max_{u'} s(u, u') ≥ τ}| / |V̂|,
and symmetrically for R.

*Matching variant* (``aggregate="matching"``): instead of each element taking
its best counterpart independently, predicted and reference elements are
paired one-to-one by a maximum-weight bipartite matching M (Hungarian
algorithm), and P = Σ_{(x, y) ∈ M} s(x, y) / |predicted|,
R = Σ_{(x, y) ∈ M} s(x, y) / |reference| (the "continuous F1" of [45]).

Conventions: if both sets are empty P = R = F1 = 1; if exactly one is empty
P = R = F1 = 0.

Property: with roots excluded, root edges included and the ``"min"`` (or
``"product"``) combiner, every non-root node has exactly one parent edge and
s_E((p, c), ·) ≤ max_{c'} s(c, c'), hence P_edge ≤ P_node, R_edge ≤ R_node and
edge F1 ≤ node F1 (for the soft, fuzzy and matching variants alike).

Implementation notes: similarities are computed with NumPy on unit-norm
label embeddings (one (|V̂|+1) × (|V*|+1) matrix); edge similarities are
evaluated in row blocks, so a TaxoBench-sized reference (814 nodes) costs a
few milliseconds. Label embeddings are cached by text (pass a shared
``cache`` dict, or reuse one :class:`TaxonomyScorer`, to embed every label
only once across many comparisons).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from scholarpath.embeddings import Embedder, l2_normalize
from scholarpath.taxonomy.tree import Taxonomy, TaxonomyNode

TaxonomyLike = Union[Taxonomy, Mapping[str, Any]]
NodeText = Union[str, Callable[[TaxonomyNode], str]]
Combiner = Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]

TEXT_MODES = ("name", "name_definition")
AGGREGATES = ("max", "matching")
COMBINERS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "min": np.minimum,
    "product": np.multiply,
    "mean": lambda a, b: 0.5 * (a + b),
    "geometric": lambda a, b: np.sqrt(a * b),
}


@dataclass(frozen=True)
class PRF:
    """Precision, recall and F1."""

    precision: float
    recall: float
    f1: float

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        return {f"{prefix}p": self.precision, f"{prefix}r": self.recall, f"{prefix}f1": self.f1}


def f1_score(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _prf(precision: float, recall: float) -> PRF:
    return PRF(float(precision), float(recall), float(f1_score(precision, recall)))


def _empty_prf(n_pred: int, n_gold: int) -> Optional[PRF]:
    if n_pred == 0 and n_gold == 0:
        return PRF(1.0, 1.0, 1.0)
    if n_pred == 0 or n_gold == 0:
        return PRF(0.0, 0.0, 0.0)
    return None


def _as_taxonomy(tax: TaxonomyLike) -> Taxonomy:
    """Accept a :class:`Taxonomy`, its ``to_dict()`` form, or a nested ``{"name", "children"}`` dict."""
    if isinstance(tax, Taxonomy):
        return tax
    if isinstance(tax, Mapping):
        if "nodes" in tax and "root_id" in tax:
            return Taxonomy.from_dict(dict(tax))
        return Taxonomy.from_nested(dict(tax))
    raise TypeError(f"Expected a Taxonomy or a dict, got {type(tax).__name__}")


@dataclass
class SimilarityTables:
    """Node similarities and edge index arrays of one (pred, gold) comparison.

    ``pred_ids[0]`` / ``gold_ids[0]`` are the roots; ``S[i, j]`` = s(pred_ids[i],
    gold_ids[j]) with the root convention applied; edges are (parent, child)
    row indices into ``pred_ids`` / ``gold_ids``.
    """

    pred_ids: List[str]
    gold_ids: List[str]
    S: np.ndarray
    pred_edges: np.ndarray  # (|Ê|, 2) int
    gold_edges: np.ndarray  # (|E*|, 2) int


class TaxonomyScorer:
    """Soft node / edge precision, recall and F1 with cached label embeddings.

    Args:
        embedder: label embedder (``embedder.encode(texts)``; SPECTER2 in the paper).
        text: ``"name"`` (default), ``"name_definition"``, or ``callable(node) -> str``.
        include_root: compare the roots as ordinary nodes (default: excluded).
        root_edges: count the edges leaving the root (default: True).
        combine: edge-similarity combiner of the two endpoint similarities.
        aggregate: ``"max"`` (best counterpart per element) or ``"matching"``
            (one-to-one maximum-weight matching).
        fuzzy_threshold: if set, :meth:`score` also reports the thresholded
            ("fuzzy") variant with this τ.
        cache: optional shared ``{text: unit vector}`` dict.
        block_size: predicted edges processed per block.
    """

    def __init__(
        self,
        embedder: Embedder,
        text: NodeText = "name",
        include_root: bool = False,
        root_edges: bool = True,
        combine: Combiner = "min",
        aggregate: str = "max",
        fuzzy_threshold: Optional[float] = None,
        cache: Optional[MutableMapping[str, np.ndarray]] = None,
        block_size: int = 1024,
    ):
        if isinstance(text, str) and text not in TEXT_MODES:
            raise ValueError(f"text must be one of {TEXT_MODES} or a callable, got {text!r}")
        if isinstance(combine, str) and combine not in COMBINERS:
            raise ValueError(f"combine must be one of {tuple(COMBINERS)} or a callable, got {combine!r}")
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")
        self.embedder = embedder
        self.text = text
        self.include_root = bool(include_root)
        self.root_edges = bool(root_edges)
        self.combine: Callable[[np.ndarray, np.ndarray], np.ndarray] = (
            COMBINERS[combine] if isinstance(combine, str) else combine
        )
        self.aggregate = aggregate
        self.fuzzy_threshold = fuzzy_threshold
        self.cache: MutableMapping[str, np.ndarray] = cache if cache is not None else {}
        self.block_size = max(1, int(block_size))

    # ------------------------------------------------------------ labels
    def node_text(self, node: TaxonomyNode) -> str:
        if callable(self.text):
            return str(self.text(node))
        if self.text == "name_definition" and node.definition:
            return f"{node.name.strip()}: {node.definition.strip()}"
        return node.name.strip()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-norm embeddings of ``texts`` (cached by text)."""
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            vecs = l2_normalize(np.asarray(self.embedder.encode(missing), dtype=np.float32))
            for t, v in zip(missing, vecs):
                self.cache[t] = v
        if not texts:
            return np.zeros((0, int(getattr(self.embedder, "dim", 0))), dtype=np.float32)
        return np.stack([self.cache[t] for t in texts])

    # ------------------------------------------------------ similarities
    @staticmethod
    def _order(tax: Taxonomy) -> Tuple[List[str], np.ndarray]:
        ids = tax.bfs()  # root first
        pos = {n: i for i, n in enumerate(ids)}
        edges = np.array([(pos[p], pos[c]) for p, c in tax.edges() if p in pos and c in pos], dtype=int)
        return ids, edges.reshape(-1, 2)

    def tables(self, pred: TaxonomyLike, gold: TaxonomyLike) -> SimilarityTables:
        """Node similarity matrix (root convention applied) and edge index arrays."""
        pt, gt = _as_taxonomy(pred), _as_taxonomy(gold)
        p_ids, p_edges = self._order(pt)
        g_ids, g_edges = self._order(gt)
        Ep = self.encode([self.node_text(pt.nodes[n]) for n in p_ids])
        Eg = self.encode([self.node_text(gt.nodes[n]) for n in g_ids])
        S = np.clip(Ep.astype(np.float64) @ Eg.T.astype(np.float64), 0.0, 1.0)
        if not self.include_root:  # aligned virtual roots
            S[0, :] = 0.0
            S[:, 0] = 0.0
            S[0, 0] = 1.0
        if not self.root_edges:
            p_edges = p_edges[p_edges[:, 0] != 0]
            g_edges = g_edges[g_edges[:, 0] != 0]
        return SimilarityTables(p_ids, g_ids, S, p_edges, g_edges)

    # -------------------------------------------------------- aggregation
    def _from_maxima(self, row_max: np.ndarray, col_max: np.ndarray, threshold: Optional[float]) -> PRF:
        if threshold is None:
            return _prf(row_max.mean(), col_max.mean())
        return _prf((row_max >= threshold).mean(), (col_max >= threshold).mean())

    @staticmethod
    def _from_matching(M: np.ndarray, threshold: Optional[float]) -> PRF:
        from scipy.optimize import linear_sum_assignment

        W = M.astype(np.float64) if threshold is None else (M >= threshold).astype(np.float64)
        rows, cols = linear_sum_assignment(W, maximize=True)
        total = float(W[rows, cols].sum())
        return _prf(total / M.shape[0], total / M.shape[1])

    def _node_matrix(self, t: SimilarityTables) -> np.ndarray:
        start = 0 if self.include_root else 1
        return t.S[start:, start:]

    def _edge_blocks(self, t: SimilarityTables):
        gp, gc = t.gold_edges[:, 0], t.gold_edges[:, 1]
        for s in range(0, t.pred_edges.shape[0], self.block_size):
            blk = t.pred_edges[s : s + self.block_size]
            yield self.combine(t.S[blk[:, 0]][:, gp], t.S[blk[:, 1]][:, gc])

    def _edge_maxima(self, t: SimilarityTables) -> Tuple[np.ndarray, np.ndarray]:
        row_max = np.empty(t.pred_edges.shape[0], dtype=np.float64)
        col_max = np.zeros(t.gold_edges.shape[0], dtype=np.float64)
        s = 0
        for block in self._edge_blocks(t):
            row_max[s : s + block.shape[0]] = block.max(axis=1)
            np.maximum(col_max, block.max(axis=0), out=col_max)
            s += block.shape[0]
        return row_max, col_max

    def _node_prfs(self, t: SimilarityTables, thresholds: Sequence[Optional[float]]) -> List[PRF]:
        M = self._node_matrix(t)
        empty = _empty_prf(*M.shape)
        if empty is not None:
            return [empty for _ in thresholds]
        if self.aggregate == "matching":
            return [self._from_matching(M, thr) for thr in thresholds]
        row_max, col_max = M.max(axis=1), M.max(axis=0)
        return [self._from_maxima(row_max, col_max, thr) for thr in thresholds]

    def _edge_prfs(self, t: SimilarityTables, thresholds: Sequence[Optional[float]]) -> List[PRF]:
        empty = _empty_prf(t.pred_edges.shape[0], t.gold_edges.shape[0])
        if empty is not None:
            return [empty for _ in thresholds]
        if self.aggregate == "matching":
            M = np.concatenate(list(self._edge_blocks(t)), axis=0)
            return [self._from_matching(M, thr) for thr in thresholds]
        row_max, col_max = self._edge_maxima(t)
        return [self._from_maxima(row_max, col_max, thr) for thr in thresholds]

    # ---------------------------------------------------------- public
    def node_prf(self, pred: TaxonomyLike, gold: TaxonomyLike, threshold: Optional[float] = None) -> PRF:
        """Soft (or, with ``threshold``, fuzzy) node precision / recall / F1."""
        return self._node_prfs(self.tables(pred, gold), [threshold])[0]

    def edge_prf(self, pred: TaxonomyLike, gold: TaxonomyLike, threshold: Optional[float] = None) -> PRF:
        """Soft (or, with ``threshold``, fuzzy) edge precision / recall / F1."""
        return self._edge_prfs(self.tables(pred, gold), [threshold])[0]

    def score(self, pred: TaxonomyLike, gold: TaxonomyLike) -> Dict[str, float]:
        """All node and edge scores of ``pred`` against the reference ``gold``.

        Keys: ``node_p, node_r, node_f1, edge_p, edge_r, edge_f1, n_pred_nodes,
        n_gold_nodes, n_pred_edges, n_gold_edges`` and, when
        ``fuzzy_threshold`` is set, ``node_fuzzy_{p,r,f1}`` and
        ``edge_fuzzy_{p,r,f1}``.
        """
        t = self.tables(pred, gold)
        thresholds: List[Optional[float]] = [None]
        if self.fuzzy_threshold is not None:
            thresholds.append(float(self.fuzzy_threshold))
        nodes = self._node_prfs(t, thresholds)
        edges = self._edge_prfs(t, thresholds)
        n_pred, n_gold = self._node_matrix(t).shape
        out: Dict[str, float] = {}
        out.update(nodes[0].to_dict("node_"))
        out.update(edges[0].to_dict("edge_"))
        if len(thresholds) > 1:
            out.update(nodes[1].to_dict("node_fuzzy_"))
            out.update(edges[1].to_dict("edge_fuzzy_"))
        out.update({
            "n_pred_nodes": int(n_pred),
            "n_gold_nodes": int(n_gold),
            "n_pred_edges": int(t.pred_edges.shape[0]),
            "n_gold_edges": int(t.gold_edges.shape[0]),
        })
        return out

    def alignment(self, pred: TaxonomyLike, gold: TaxonomyLike) -> List[Tuple[str, Optional[str], float]]:
        """Best reference node of every predicted concept: ``(pred_id, gold_id, s)`` (for inspection)."""
        t = self.tables(pred, gold)
        start = 0 if self.include_root else 1
        M = t.S[start:, start:]
        out: List[Tuple[str, Optional[str], float]] = []
        for i, pid in enumerate(t.pred_ids[start:]):
            if M.shape[1] == 0:
                out.append((pid, None, 0.0))
                continue
            j = int(np.argmax(M[i]))
            out.append((pid, t.gold_ids[start + j], float(M[i, j])))
        return out


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def soft_node_prf(
    pred: TaxonomyLike,
    gold: TaxonomyLike,
    embedder: Embedder,
    *,
    text: NodeText = "name",
    include_root: bool = False,
    threshold: Optional[float] = None,
    aggregate: str = "max",
    cache: Optional[MutableMapping[str, np.ndarray]] = None,
) -> PRF:
    """Soft node precision / recall / F1 (fuzzy when ``threshold`` is given)."""
    scorer = TaxonomyScorer(embedder, text=text, include_root=include_root, aggregate=aggregate, cache=cache)
    return scorer.node_prf(pred, gold, threshold)


def soft_edge_prf(
    pred: TaxonomyLike,
    gold: TaxonomyLike,
    embedder: Embedder,
    *,
    text: NodeText = "name",
    include_root: bool = False,
    root_edges: bool = True,
    combine: Combiner = "min",
    threshold: Optional[float] = None,
    aggregate: str = "max",
    cache: Optional[MutableMapping[str, np.ndarray]] = None,
) -> PRF:
    """Soft edge precision / recall / F1 (fuzzy when ``threshold`` is given)."""
    scorer = TaxonomyScorer(embedder, text=text, include_root=include_root, root_edges=root_edges,
                            combine=combine, aggregate=aggregate, cache=cache)
    return scorer.edge_prf(pred, gold, threshold)


def soft_node_f1(pred: TaxonomyLike, gold: TaxonomyLike, embedder: Embedder, **kwargs: Any) -> float:
    """Soft node F1 (keyword arguments as in :func:`soft_node_prf`)."""
    return soft_node_prf(pred, gold, embedder, **kwargs).f1


def soft_edge_f1(pred: TaxonomyLike, gold: TaxonomyLike, embedder: Embedder, **kwargs: Any) -> float:
    """Soft edge F1 (keyword arguments as in :func:`soft_edge_prf`)."""
    return soft_edge_prf(pred, gold, embedder, **kwargs).f1


def taxonomy_scores(
    pred: TaxonomyLike,
    gold: TaxonomyLike,
    embedder: Embedder,
    *,
    text: NodeText = "name",
    include_root: bool = False,
    root_edges: bool = True,
    combine: Combiner = "min",
    aggregate: str = "max",
    fuzzy_threshold: Optional[float] = None,
    cache: Optional[MutableMapping[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Soft node and edge P / R / F1 of ``pred`` against ``gold`` (see :meth:`TaxonomyScorer.score`)."""
    scorer = TaxonomyScorer(embedder, text=text, include_root=include_root, root_edges=root_edges,
                            combine=combine, aggregate=aggregate, fuzzy_threshold=fuzzy_threshold, cache=cache)
    return scorer.score(pred, gold)


__all__ = [
    "AGGREGATES",
    "COMBINERS",
    "PRF",
    "SimilarityTables",
    "TEXT_MODES",
    "TaxonomyScorer",
    "f1_score",
    "soft_edge_f1",
    "soft_edge_prf",
    "soft_node_f1",
    "soft_node_prf",
    "taxonomy_scores",
]
