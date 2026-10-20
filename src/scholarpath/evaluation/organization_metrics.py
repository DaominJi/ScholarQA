"""Organization metrics for TaxoBench (ARI, Sem) and CHIME (Cat, Assn).

Section 5.3 / Table 3 evaluate *organization* queries: given only a topic,
a system retrieves papers (budgeted systems: within B tokens, then a single
composition call, :mod:`scholarpath.prompts.organize`) and returns a topic
hierarchy whose categories cite paper ids. Appendix C.2 describes the two
benchmarks:

* **TaxoBench** (Zhang et al., 2026), *Deep Research mode*: 72 survey
  taxonomies; "predicted papers are aligned by normalized title with the
  benchmark's default threshold and organized into a topic hierarchy".
  Metrics: **ARI** (leaf-level paper-to-category agreement) and **Sem**
  (semantic path similarity of the ancestor chains of aligned papers).
* **CHIME** (Hsu et al., 2024): 100 expert-corrected review topics whose
  studies are pooled into one retrieval corpus, "so that systems must both
  retrieve the studies of a topic and organize them into categories".
  Metrics: **Cat** (category soft F1) and **Assn** (assignment F1).

All scores are fractions in [0, 1] (tables report percentages) computed per
query and macro-averaged over queries (:func:`macro_average`).

TaxoBench metrics re-implement the benchmark's public scorer
(``taxobench-score`` v0.2.0: ``taxobench/metrics/{alignment,clustering,
sem_path,tree}.py``) and the paper-era scripts
(``metric/get_clustering_result.py``, ``get_clustering_metric.py``,
``sem_path.py``); differences are documented on each function. TaxoBench's
Deep Research mode defines two leaf-level views: ARI∩ on the retrieved expert
papers (its view (A)) and an end-to-end ARI in which unretrieved expert papers
share an extra label ⊥ (view (B), Eq. 6; the public CLI's ``leaf_ari``).
Table 3 reports the retrieval-conditioned view, consistent with Sem being
averaged over aligned pairs (Eq. 13): with ⊥, even a *perfect* organization of
the retrieved papers caps the macro ARI below the reported values at the
systems' Recall@B (Table 2). For example, on the 72 TaxoBench gold trees at
recall 0.706 the cap is 41-57, depending on how much recall varies across
surveys (per-survey sd 0-0.3), whereas ScholarPath reports 55.8. Keyword
search (BM25 Recall@B 0.473) is capped at 12-26 but reports 31.4. Both views
are always returned by
:func:`organization_scores` (``ari_intersection``, ``ari_e2e``).

The CHIME repository only ships binary P/R/F1 scripts for its annotation
tasks, so Cat and Assn are defined here (see :func:`category_soft_prf` and
:func:`assignment_prf`). Both are embedding-based and need an embedder or an
explicit similarity callable.

Hierarchies are foundation :class:`~scholarpath.taxonomy.tree.Taxonomy`
objects (facet ``"topic"``), e.g. from
:func:`scholarpath.prompts.organize.parse_hierarchy` (predictions) or
:meth:`Taxonomy.from_nested` (gold; TaxoBench trees use
``children_key="subtopics"``). A paper is *attached* to the most specific
categories that contain it (nodes of its extent none of whose children
contain it); a paper attached to several categories is *multi-placed*.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

from scholarpath.embeddings import l2_normalize
from scholarpath.taxonomy.tree import Taxonomy

__all__ = [
    "ARI_E2E",
    "ARI_INTERSECTION",
    "DEFAULT_CATEGORY_THRESHOLD",
    "MISSING_LABEL",
    "ORGANIZATION_METRICS",
    "PUBLIC_RULE",
    "STRICT_RULE",
    "TAXOBENCH_EMBEDDING_MODEL",
    "TAXOBENCH_THRESHOLD",
    "NodeSimilarity",
    "PRF",
    "PaperAlignment",
    "PaperMatch",
    "adjusted_rand_index",
    "align_categories",
    "align_papers",
    "ancestor_chains",
    "assignment_f1",
    "assignment_prf",
    "benchmark_title_similarity",
    "category_soft_f1",
    "category_soft_prf",
    "chain_alignment_cost",
    "leaf_clustering_scores",
    "load_taxobench_embedder",
    "macro_average",
    "merge_hierarchies",
    "normalize_benchmark_title",
    "organization_scores",
    "paper_attachments",
    "semantic_path_scores",
    "semantic_path_similarity",
    "subtree",
    "to_taxobench_tree",
]

#: TaxoBench public scorer default (``taxobench-score --threshold 0.92``).
TAXOBENCH_THRESHOLD = 0.92
#: Paper-era (TaxoBench App. A.5) rule: align if s = 1, or 0.6 <= s < 1 with strict containment.
STRICT_THRESHOLD = 1.0
STRICT_CONTAINMENT_FLOOR = 0.6
PUBLIC_RULE = "public"
STRICT_RULE = "strict"
#: Embedding model of the TaxoBench paper's Sem-Path / US-TED (its App. A.3.1).
TAXOBENCH_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: Minimum name similarity for a predicted category to be aligned to a gold category (Assn).
DEFAULT_CATEGORY_THRESHOLD = 0.5
#: Cluster label of expert papers the system did not retrieve (TaxoBench Eq. 6, label ⊥).
MISSING_LABEL = "⊥"
#: ARI views of TaxoBench Deep Research mode: retrieved expert papers only (ARI∩, Table 3) / with ⊥.
ARI_INTERSECTION = "intersection"
ARI_E2E = "e2e"
#: Metrics reported per benchmark in Table 3.
ORGANIZATION_METRICS: Dict[str, Tuple[str, ...]] = {"taxobench": ("ari", "sem"), "chime": ("cat", "assn")}

#: ``paper_id -> title`` source: None (ids are titles), a mapping to str / Paper / dict, a Corpus, or a callable.
TitleSource = Any
Similarity = Callable[[str, str], float]


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def _preorder(tax: Taxonomy, start: Optional[str] = None) -> List[str]:
    """Node ids of the subtree rooted at ``start`` in depth-first pre-order (children in stored order)."""
    out: List[str] = []
    stack = [start or tax.root_id]
    while stack:
        nid = stack.pop()
        out.append(nid)
        stack.extend(reversed(tax.nodes[nid].children))
    return out


def paper_attachments(tax: Taxonomy, root: Optional[str] = None) -> Dict[str, List[str]]:
    """Map each paper to the most specific categories it is attached to.

    A paper is attached to node v if v's extent contains it and no child of
    v does (the node that lists it in the JSON hierarchy). Attachment lists
    are in pre-order; the dict is ordered by each paper's first attachment in
    pre-order (papers of one node sorted by id), which fixes the order of the
    greedy title alignment. Restricted to the subtree of ``root`` if given.
    """
    out: Dict[str, List[str]] = {}
    for nid in _preorder(tax, root):
        node = tax.nodes[nid]
        below: Set[str] = set()
        for c in node.children:
            below |= tax.nodes[c].papers
        for p in sorted(node.papers - below):
            out.setdefault(p, []).append(nid)
    return out


def _copy_below(src: Taxonomy, src_nid: str, dst: Taxonomy, dst_nid: str) -> None:
    """Copy the descendants and paper attachments of ``src_nid`` under ``dst_nid``."""
    mapping = {src_nid: dst_nid}
    for nid in _preorder(src, src_nid)[1:]:
        node = src.nodes[nid]
        mapping[nid] = dst.add_node(node.name, node.definition, parent=mapping[node.parent])
    for paper, nodes in paper_attachments(src, src_nid).items():
        for nid in nodes:
            dst.assign(paper, mapping[nid])


def subtree(tax: Taxonomy, node_id: str) -> Taxonomy:
    """Copy of the subtree rooted at ``node_id`` (e.g. one survey of the merged TaxoBench gold, App. C.2)."""
    node = tax.nodes[node_id]
    out = Taxonomy(facet=tax.facet, root_name=node.name, root_definition=node.definition)
    _copy_below(tax, node_id, out, out.root_id)
    return out


def merge_hierarchies(hierarchies: Sequence[Taxonomy], root_name: str = "Root", root_definition: str = "") -> Taxonomy:
    """Merge several hierarchies under a common root (each root becomes a top-level category).

    CHIME stores one hierarchy per top-level aspect of a review (2,174
    hierarchies over 472 topics; 320 expert-corrected ones over the 100 gold
    topics). Merging the aspect hierarchies of a topic reconstructs the full
    category hierarchy of that topic, against which Cat and Assn are computed.
    A new root is always created, so the root of every input hierarchy (an
    aspect) counts as a category, also when a topic has a single aspect.
    """
    facet = hierarchies[0].facet if hierarchies else "topic"
    out = Taxonomy(facet=facet, root_name=root_name, root_definition=root_definition)
    for h in hierarchies:
        cid = out.add_node(h.root.name, h.root.definition)
        _copy_below(h, h.root_id, out, cid)
    return out


def _title_resolver(titles: TitleSource) -> Callable[[str], str]:
    """``paper_id -> title`` from a mapping (to str / Paper / dict), a Corpus, or a callable.

    Unknown ids resolve to the id itself, so hierarchies whose "paper ids" are
    already titles (raw TaxoBench trees) need no mapping.
    """
    if titles is None:
        return lambda pid: pid
    papers = getattr(titles, "papers", None)
    if isinstance(papers, Mapping):  # scholarpath.data.corpus.Corpus
        titles = papers
    if not isinstance(titles, Mapping):
        if not callable(titles):
            raise TypeError(f"titles must be a mapping, a Corpus or a callable, not {type(titles).__name__}")
        fn = titles
        return lambda pid: str(fn(pid) or pid)
    table = titles

    def lookup(pid: str) -> str:
        value = table.get(pid)
        if value is None:
            return pid
        if isinstance(value, str):
            return value or pid
        if isinstance(value, Mapping):
            return str(value.get("title") or pid)
        return str(getattr(value, "title", None) or pid)

    return lookup


def to_taxobench_tree(tax: Taxonomy, titles: TitleSource = None, node_id: Optional[str] = None) -> Dict[str, Any]:
    """Export a hierarchy in TaxoBench's prediction format (``{"name", "subtopics", "papers"}``).

    Paper ids are replaced by titles (via ``titles``) so that the output can
    also be scored with the official ``taxobench-score`` CLI as
    ``{"id": ..., "hierarchy_tree": <this>}``.
    """
    title = _title_resolver(titles)
    attachments = paper_attachments(tax, node_id)
    direct: Dict[str, List[str]] = {}
    for paper, nodes in attachments.items():
        for nid in nodes:
            direct.setdefault(nid, []).append(paper)

    def rec(nid: str) -> Dict[str, Any]:
        node = tax.nodes[nid]
        out: Dict[str, Any] = {"name": node.name}
        if node.children:
            out["subtopics"] = [rec(c) for c in node.children]
        if nid in direct or not node.children:
            out["papers"] = [title(p) for p in direct.get(nid, [])]
        return out

    return rec(node_id or tax.root_id)


# ---------------------------------------------------------------------------
# TaxoBench paper alignment (Deep Research mode; TaxoBench Sec. 3.2 / App. A.5)
# ---------------------------------------------------------------------------


def _check_rule(rule: str) -> str:
    rule = (rule or PUBLIC_RULE).lower()
    if rule not in (PUBLIC_RULE, STRICT_RULE):
        raise ValueError(f"Unknown title-matching rule {rule!r}; use 'public' or 'strict'.")
    return rule


def normalize_benchmark_title(title: str, rule: str = PUBLIC_RULE) -> str:
    """TaxoBench title normalisation.

    Lower-case, drop punctuation (``[^\\w\\s]``), collapse whitespace. The
    public scorer replaces punctuation by a space (``alignment.py``); the
    paper-era script deletes it (``get_clustering_result.py``), so e.g.
    ``"Few-Shot"`` becomes ``"few shot"`` vs. ``"fewshot"``.
    """
    rule = _check_rule(rule)
    t = str(title or "").lower()
    t = re.sub(r"[^\w\s]", " " if rule == PUBLIC_RULE else "", t)
    return re.sub(r"\s+", " ", t).strip()


def _contains(a: str, b: str) -> bool:
    return a in b or b in a


def benchmark_title_similarity(a: str, b: str, rule: str = PUBLIC_RULE) -> float:
    """Title similarity s(p, p̂) of the TaxoBench scorer.

    ``public``: 1 if one normalised title contains the other, else the
    :class:`difflib.SequenceMatcher` ratio. ``strict``: the plain ratio
    (containment is checked separately by the matching rule). Empty titles
    score 0. This is also the public scorer's fallback node similarity for
    Sem-Path.
    """
    rule = _check_rule(rule)
    na, nb = normalize_benchmark_title(a, rule), normalize_benchmark_title(b, rule)
    if not na or not nb:
        return 0.0
    if rule == PUBLIC_RULE and _contains(na, nb):
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


@dataclass
class PaperMatch:
    """Alignment outcome of one expert paper (``pred_id`` None if unretrieved)."""

    gold_id: str
    pred_id: Optional[str]
    score: float  # best similarity found (also reported for unmatched papers)


@dataclass
class PaperAlignment:
    """Greedy one-to-one alignment of expert papers P* to predicted papers P̂."""

    matches: List[PaperMatch]
    n_predicted: int
    rule: str = PUBLIC_RULE
    threshold: float = TAXOBENCH_THRESHOLD

    @property
    def gold_to_pred(self) -> Dict[str, str]:
        return {m.gold_id: m.pred_id for m in self.matches if m.pred_id is not None}

    @property
    def pred_to_gold(self) -> Dict[str, str]:
        return {m.pred_id: m.gold_id for m in self.matches if m.pred_id is not None}

    @property
    def n_reference(self) -> int:
        return len(self.matches)

    @property
    def n_aligned(self) -> int:
        return sum(1 for m in self.matches if m.pred_id is not None)

    @property
    def recall(self) -> float:
        """|P* ∩ P̂| / |P*| (TaxoBench Eq. 1)."""
        return self.n_aligned / self.n_reference if self.n_reference else 0.0

    @property
    def precision(self) -> float:
        """|P* ∩ P̂| / |P̂| (TaxoBench Eq. 1)."""
        return self.n_aligned / self.n_predicted if self.n_predicted else 0.0

    @property
    def f1(self) -> float:
        return _f1(self.precision, self.recall)


def align_papers(
    gold_ids: Sequence[str],
    pred_ids: Sequence[str],
    titles: TitleSource = None,
    gold_titles: TitleSource = None,
    threshold: Optional[float] = None,
    rule: str = PUBLIC_RULE,
) -> PaperAlignment:
    """Align expert papers to predicted papers by normalised title (TaxoBench App. A.5).

    Expert papers are processed in the given order; each takes the unused
    predicted paper of highest similarity (first one on ties) and is aligned
    if

    * ``rule="public"`` (default; ``taxobench-score``): s >= ``threshold``
      (default 0.92), where containment counts as s = 1;
    * ``rule="strict"`` (paper-era scripts): s >= ``threshold`` (default 1.0),
      or 0.6 <= s < ``threshold`` and one normalised title contains the other.

    ``titles`` maps predicted ids to titles (mapping, Corpus or callable);
    ``gold_titles`` does the same for expert ids (defaults to ``titles``).
    Ids without a known title are used as titles themselves.
    """
    rule = _check_rule(rule)
    if threshold is None:
        threshold = TAXOBENCH_THRESHOLD if rule == PUBLIC_RULE else STRICT_THRESHOLD
    thr = float(threshold)
    gold_ids = list(dict.fromkeys(map(str, gold_ids)))
    pred_ids = list(dict.fromkeys(map(str, pred_ids)))
    pred_title = _title_resolver(titles)
    gold_title = _title_resolver(gold_titles if gold_titles is not None else titles)
    pred_norm = [normalize_benchmark_title(pred_title(p), rule) for p in pred_ids]
    used = [False] * len(pred_ids)
    matches: List[PaperMatch] = []
    for gid in gold_ids:
        ng = normalize_benchmark_title(gold_title(gid), rule)
        best, best_idx = 0.0, None
        if ng:
            for j, npj in enumerate(pred_norm):
                if used[j] or not npj:
                    continue
                if rule == PUBLIC_RULE and _contains(ng, npj):
                    s = 1.0
                else:
                    # Exact pruning with SequenceMatcher's upper bounds (real_quick_ratio, quick_ratio):
                    # a candidate replaces the incumbent only if strictly better.
                    if 2.0 * min(len(ng), len(npj)) / (len(ng) + len(npj)) <= best:
                        continue
                    sm = SequenceMatcher(None, ng, npj)
                    if sm.quick_ratio() <= best:
                        continue
                    s = sm.ratio()
                if s > best:
                    best, best_idx = s, j
                    if best >= 1.0:
                        break
        ok = False
        if best_idx is not None:
            if best >= thr:
                ok = True
            elif rule == STRICT_RULE and STRICT_CONTAINMENT_FLOOR <= best < thr:
                ok = _contains(ng, pred_norm[best_idx])
        if ok:
            used[best_idx] = True  # type: ignore[index]
            matches.append(PaperMatch(gid, pred_ids[best_idx], float(best)))  # type: ignore[index]
        else:
            matches.append(PaperMatch(gid, None, float(best)))
    return PaperAlignment(matches=matches, n_predicted=len(pred_ids), rule=rule, threshold=thr)


def _alignment_for(
    pred: Taxonomy,
    gold: Taxonomy,
    titles: TitleSource,
    gold_titles: TitleSource,
    threshold: Optional[float],
    rule: str,
) -> PaperAlignment:
    gold_ids, pred_ids = list(paper_attachments(gold)), list(paper_attachments(pred))
    return align_papers(gold_ids, pred_ids, titles, gold_titles, threshold, rule)


# ---------------------------------------------------------------------------
# ARI (TaxoBench Sec. 3.3.1, Eq. 3 and 6)
# ---------------------------------------------------------------------------


def leaf_clustering_scores(
    pred: Taxonomy,
    gold: Taxonomy,
    titles: TitleSource = None,
    gold_titles: TitleSource = None,
    threshold: Optional[float] = None,
    rule: str = PUBLIC_RULE,
    include_missing: bool = False,
    alignment: Optional[PaperAlignment] = None,
) -> Dict[str, float]:
    """Leaf-level agreement of paper-to-category assignments (ARI, V-measure).

    Each expert paper is labelled by its category in the gold hierarchy
    (U*) and by the category of its aligned predicted paper (Û).

    * ``include_missing=False`` (default): only expert papers aligned to a
      predicted paper are labelled, P = P* ∩ P̂ (TaxoBench view (A), ARI∩).
      This is the "ARI" of Table 3 (see the module docstring).
    * ``include_missing=True``: expert papers the system did not retrieve all
      receive one extra label ⊥ (TaxoBench view (B), Eq. 6). This reproduces
      ``leaf_ari`` of the public ``taxobench-score`` CLI
      (``include_unretrieved=True``).

    A multi-placed paper is labelled by its first most specific category in
    pre-order. The public scorer keeps the last listed leaf instead, but
    TaxoBench gold papers are never multi-placed.

    Returns ``ari``, ``v_measure``, ``homogeneity``, ``completeness`` (all 0
    when fewer than two papers are labelled, as in the public scorer) and
    the counts ``n_labelled``, ``n_aligned``, ``n_reference``, ``n_predicted``.
    """
    gold_att = paper_attachments(gold)
    pred_att = paper_attachments(pred)
    if alignment is None:
        alignment = align_papers(list(gold_att), list(pred_att), titles, gold_titles, threshold, rule)
    y_true: List[str] = []
    y_pred: List[str] = []
    for m in alignment.matches:
        if m.gold_id not in gold_att:
            continue
        if m.pred_id is not None and m.pred_id in pred_att:
            label = pred_att[m.pred_id][0]
        elif include_missing:
            label = MISSING_LABEL
        else:
            continue
        y_true.append(gold_att[m.gold_id][0])
        y_pred.append(label)
    out = {
        "ari": 0.0,
        "v_measure": 0.0,
        "homogeneity": 0.0,
        "completeness": 0.0,
        "n_labelled": float(len(y_true)),
        "n_aligned": float(sum(1 for m in alignment.matches if m.pred_id is not None and m.pred_id in pred_att)),
        "n_reference": float(len(gold_att)),
        "n_predicted": float(len(pred_att)),
    }
    if len(y_true) >= 2:
        h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
        out.update(
            ari=float(adjusted_rand_score(y_true, y_pred)),
            homogeneity=float(h),
            completeness=float(c),
            v_measure=float(v),
        )
    return out


def adjusted_rand_index(
    pred: Taxonomy,
    gold: Taxonomy,
    titles: TitleSource = None,
    gold_titles: TitleSource = None,
    threshold: Optional[float] = None,
    rule: str = PUBLIC_RULE,
    include_missing: bool = False,
    alignment: Optional[PaperAlignment] = None,
) -> float:
    """ARI between expert and predicted leaf-cluster labels (see :func:`leaf_clustering_scores`).

    The default is ARI∩ on the retrieved expert papers, the "ARI" of Table 3.
    Pass ``include_missing=True`` for the end-to-end ARI with the ⊥ label,
    which is the public ``taxobench-score`` ``leaf_ari``.
    """
    return leaf_clustering_scores(pred, gold, titles, gold_titles, threshold, rule, include_missing, alignment)["ari"]


# ---------------------------------------------------------------------------
# Node similarity (TaxoBench Eq. 7)
# ---------------------------------------------------------------------------


def _same_label(a: str, b: str) -> bool:
    return " ".join(a.lower().split()) == " ".join(b.lower().split())


class NodeSimilarity:
    """Sim(x, y) = max(0, cos(e(x), e(y))) between category labels (TaxoBench Eq. 7).

    ``embedder`` is any object with ``encode(texts) -> (n, d) array``
    (``scholarpath.embeddings``); vectors are cached per label. Labels that
    are equal up to case and whitespace have similarity 1 (also for
    embedders that map stop-word-only labels such as "Other" to the zero
    vector). Without an embedder, the public TaxoBench scorer's fallback
    string similarity (:func:`benchmark_title_similarity`) is used, which
    scores containment as 1. This fallback is meant only for reproducing the
    public scorer's Sem-Path, and the CHIME metrics never use it implicitly.
    The TaxoBench paper uses all-MiniLM-L6-v2 (:func:`load_taxobench_embedder`).
    """

    def __init__(self, embedder: Any = None):
        self.embedder = embedder
        self._vecs: Dict[str, np.ndarray] = {}

    def prepare(self, labels: Iterable[str]) -> None:
        """Batch-encode labels not yet cached."""
        if self.embedder is None:
            return
        new = [t for t in dict.fromkeys(map(str, labels)) if t not in self._vecs]
        if new:
            vecs = l2_normalize(np.asarray(self.embedder.encode(new), dtype=np.float32))
            for t, v in zip(new, vecs):
                self._vecs[t] = v

    def __call__(self, a: str, b: str) -> float:
        a, b = str(a), str(b)
        if _same_label(a, b):
            return 1.0
        if self.embedder is None:
            return benchmark_title_similarity(a, b)
        self.prepare((a, b))
        return float(min(1.0, max(0.0, float(self._vecs[a] @ self._vecs[b]))))

    def matrix(self, rows: Sequence[str], cols: Sequence[str]) -> np.ndarray:
        """Pairwise similarity matrix (len(rows), len(cols)) with values in [0, 1]."""
        rows, cols = [str(r) for r in rows], [str(c) for c in cols]
        if not rows or not cols:
            return np.zeros((len(rows), len(cols)), dtype=np.float64)
        if self.embedder is None:
            S = np.array([[self(r, c) for c in cols] for r in rows], dtype=np.float64)
        else:
            self.prepare(list(rows) + list(cols))
            A = np.stack([self._vecs[r] for r in rows])
            B = np.stack([self._vecs[c] for c in cols])
            S = np.clip((A @ B.T).astype(np.float64), 0.0, 1.0)
            key_r = [" ".join(r.lower().split()) for r in rows]
            key_c = {}
            for j, c in enumerate(cols):
                key_c.setdefault(" ".join(c.lower().split()), []).append(j)
            for i, k in enumerate(key_r):
                for j in key_c.get(k, ()):
                    S[i, j] = 1.0
        return S


def load_taxobench_embedder(model_name: str = TAXOBENCH_EMBEDDING_MODEL, device: Optional[str] = None) -> Any:
    """all-MiniLM-L6-v2 sentence embedder used by the TaxoBench paper (needs ``sentence-transformers``)."""
    try:
        from scholarpath.embeddings import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model_name, device=device)
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError(
            "load_taxobench_embedder needs the optional 'sentence-transformers' package "
            "(pip install sentence-transformers); or pass any scholarpath embedder instead."
        ) from err


# ---------------------------------------------------------------------------
# Sem-Path (TaxoBench Sec. 3.3.2, Eq. 12-13; App. A.4, Eq. 30-33)
# ---------------------------------------------------------------------------


def ancestor_chains(
    tax: Taxonomy,
    paper: str,
    attachments: Optional[Mapping[str, Sequence[str]]] = None,
    include_root: bool = True,
) -> List[List[str]]:
    """Ancestor-label sequences S_d of a paper: root-to-category names, paper node excluded.

    One chain per category the paper is attached to; empty names are
    skipped, as in the public scorer's ``iter_leaf_paths``.
    """
    att = attachments if attachments is not None else paper_attachments(tax)
    return [_chain(tax, nid, include_root) for nid in att.get(paper, ())]


def _chain(tax: Taxonomy, nid: str, include_root: bool = True) -> List[str]:
    path = tax.path(nid)[0 if include_root else 1:]
    return [tax.nodes[n].name for n in path if str(tax.nodes[n].name).strip()]


def chain_alignment_cost(
    gold_chain: Sequence[str],
    pred_chain: Sequence[str],
    similarity: Similarity,
    unmatched_penalty: float = 1.0,
) -> float:
    """Order-preserving alignment cost J(S, Ŝ) (TaxoBench Eq. 12 / App. A.4.2, Eq. 30-32).

    The shorter chain A (the gold one on ties) is matched to an ordered
    subsequence of the longer chain B by the DP
    dp[0, j] = 0, dp[i, j] = min(dp[i-1, j-1] + δ(a_i, b_j), dp[i, j-1]),
    with δ = 1 - Sim; then J = dp[p, q] + λ (q - p) with λ =
    ``unmatched_penalty``. An empty chain costs λ per node of the other one.
    """
    if len(gold_chain) <= len(pred_chain):
        short, long = list(gold_chain), list(pred_chain)
    else:
        short, long = list(pred_chain), list(gold_chain)
    p, q = len(short), len(long)
    if p == 0:
        return float(q * unmatched_penalty)
    delta = [[1.0 - max(0.0, float(similarity(a, b))) for b in long] for a in short]
    inf = math.inf
    prev = [0.0] * (q + 1)  # dp[0, j] = 0
    for i in range(1, p + 1):
        cur = [inf] * (q + 1)
        for j in range(i, q + 1):
            cur[j] = min(prev[j - 1] + delta[i - 1][j - 1], cur[j - 1])
        prev = cur
    return float(prev[q] + unmatched_penalty * (q - p))


def _all_names(tax: Taxonomy) -> List[str]:
    return [n.name for n in tax.nodes.values()]


def semantic_path_scores(
    pred: Taxonomy,
    gold: Taxonomy,
    embedder: Any = None,
    similarity: Optional[Similarity] = None,
    titles: TitleSource = None,
    gold_titles: TitleSource = None,
    threshold: Optional[float] = None,
    rule: str = PUBLIC_RULE,
    unmatched_penalty: float = 1.0,
    include_root: bool = True,
    alignment: Optional[PaperAlignment] = None,
) -> Dict[str, float]:
    """Sem-Path of a predicted hierarchy (TaxoBench Eq. 13 / Eq. 33).

    For every aligned pair (d, d̂) ∈ D_a, score 1 / (1 + J_d) where J_d is the
    minimum :func:`chain_alignment_cost` over the pair's candidate ancestor
    chains (multi-placed papers have several). Returns

    * ``sem_path``: mean over aligned pairs (Eq. 13, the public scorer, and
      the old script's "alignment" similarity) -- the Sem of Table 3;
    * ``sem_path_expert``: sum over aligned pairs divided by |P*|, i.e.
      unretrieved expert papers score 0 (the old script's "expert"
      similarity);
    * counts ``n_aligned``, ``n_reference``.

    Node similarity: ``similarity`` if given, else :class:`NodeSimilarity`
    over ``embedder`` (string fallback without one). Roots are part of the
    chains (``include_root``), as in TaxoBench. 0 if nothing is aligned.
    """
    sim = similarity if similarity is not None else NodeSimilarity(embedder)
    if isinstance(sim, NodeSimilarity):
        sim.prepare(_all_names(gold) + _all_names(pred))
    gold_att = paper_attachments(gold)
    pred_att = paper_attachments(pred)
    if alignment is None:
        alignment = align_papers(list(gold_att), list(pred_att), titles, gold_titles, threshold, rule)
    pair_cache: Dict[Tuple[str, str], float] = {}
    scores: List[float] = []
    for m in alignment.matches:
        if m.pred_id is None or m.pred_id not in pred_att or m.gold_id not in gold_att:
            continue
        best = 0.0
        for g in gold_att[m.gold_id]:
            for h in pred_att[m.pred_id]:
                if (g, h) not in pair_cache:
                    cost = chain_alignment_cost(
                        _chain(gold, g, include_root), _chain(pred, h, include_root), sim, unmatched_penalty
                    )
                    pair_cache[(g, h)] = 1.0 / (1.0 + cost)
                best = max(best, pair_cache[(g, h)])
        scores.append(best)
    n_ref = len(gold_att)
    return {
        "sem_path": float(np.mean(scores)) if scores else 0.0,
        "sem_path_expert": float(np.sum(scores) / n_ref) if n_ref else 0.0,
        "n_aligned": float(len(scores)),
        "n_reference": float(n_ref),
    }


def semantic_path_similarity(
    pred: Taxonomy,
    gold: Taxonomy,
    embedder: Any = None,
    similarity: Optional[Similarity] = None,
    titles: TitleSource = None,
    gold_titles: TitleSource = None,
    threshold: Optional[float] = None,
    rule: str = PUBLIC_RULE,
    unmatched_penalty: float = 1.0,
    average: str = "aligned",
    include_root: bool = True,
    alignment: Optional[PaperAlignment] = None,
) -> float:
    """Sem (Table 3): ``average="aligned"`` (Eq. 13) or ``"expert"`` (unretrieved papers score 0)."""
    if average not in ("aligned", "expert"):
        raise ValueError("average must be 'aligned' or 'expert'")
    res = semantic_path_scores(
        pred, gold, embedder, similarity, titles, gold_titles, threshold, rule,
        unmatched_penalty, include_root, alignment,
    )
    return res["sem_path"] if average == "aligned" else res["sem_path_expert"]


# ---------------------------------------------------------------------------
# CHIME: category soft F1 (Cat) and assignment F1 (Assn)
# ---------------------------------------------------------------------------


def _f1(p: float, r: float) -> float:
    return 2.0 * p * r / (p + r) if p + r > 0 else 0.0


@dataclass(frozen=True)
class PRF:
    """Precision / recall / F1 (fractions in [0, 1]) with optional counts."""

    precision: float
    recall: float
    f1: float
    counts: Dict[str, float] = field(default_factory=dict, compare=False)

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        out = {f"{prefix}precision": self.precision, f"{prefix}recall": self.recall, f"{prefix}f1": self.f1}
        out.update({f"{prefix}{k}": v for k, v in self.counts.items()})
        return out


def _sim_matrix(sim: Similarity, rows: Sequence[str], cols: Sequence[str]) -> np.ndarray:
    """Similarity matrix from a :class:`NodeSimilarity` or any ``(a, b) -> float`` callable, clipped to [0, 1]."""
    if isinstance(sim, NodeSimilarity):
        return sim.matrix(rows, cols)
    S = np.array([[float(sim(r, c)) for c in cols] for r in rows], dtype=np.float64).reshape(len(rows), len(cols))
    return np.clip(S, 0.0, 1.0)


def _embedding_similarity(embedder: Any, similarity: Optional[Similarity], metric: str) -> Similarity:
    """Node similarity of the CHIME metrics, which are defined on embeddings and so need an embedder.

    The string fallback of :class:`NodeSimilarity` mirrors the TaxoBench
    scorer for Sem only. It scores any substring containment as 1, so a
    hierarchy with one-letter category names would get Cat = Assn = 1. It is
    therefore never used implicitly here. To use it on purpose, pass
    ``similarity=NodeSimilarity()``.
    """
    if similarity is not None:
        return similarity
    if embedder is None:
        raise ValueError(
            f"{metric} is defined on embedding similarity: pass an embedder (e.g. Runtime.embedder or "
            "load_taxobench_embedder()) or an explicit similarity callable."
        )
    return NodeSimilarity(embedder)


def _categories(tax: Taxonomy, include_root: bool = False) -> List[str]:
    nodes = _preorder(tax)
    return nodes if include_root else nodes[1:]


def _names(tax: Taxonomy, nodes: Sequence[str]) -> List[str]:
    return [str(tax.nodes[n].name).strip() for n in nodes]


def category_soft_prf(
    pred: Taxonomy,
    gold: Taxonomy,
    embedder: Any = None,
    similarity: Optional[Similarity] = None,
    include_root: bool = False,
) -> PRF:
    """Category soft precision / recall / F1 (Cat, Table 3).

    Ĉ (predicted) and C* (gold) are the non-root category names. Sim is the
    clipped cosine max(0, cos) of their embeddings (:class:`NodeSimilarity`)::

        P_soft = 1/|Ĉ|  Σ_{ĉ ∈ Ĉ}  max_{c ∈ C*} Sim(ĉ, c)
        R_soft = 1/|C*| Σ_{c ∈ C*} max_{ĉ ∈ Ĉ}  Sim(ĉ, c)
        Cat    = 2 P_soft R_soft / (P_soft + R_soft)

    This is the best-match soft node F1 of Section 5.2
    (:func:`scholarpath.evaluation.taxonomy_metrics.soft_node_prf` with node
    names and roots excluded), applied to category names. Roots (the topic
    itself) are excluded unless ``include_root``. The empty-set conventions
    are the same as there: if both sides are empty, P = R = F1 = 1; if one
    side is empty, all three are 0. Needs ``embedder`` or ``similarity``.
    """
    sim = _embedding_similarity(embedder, similarity, "Cat (category soft F1)")
    pc, gc = _categories(pred, include_root), _categories(gold, include_root)
    counts = {"n_pred": float(len(pc)), "n_gold": float(len(gc))}
    if not pc or not gc:
        both_empty = not pc and not gc
        return PRF(float(both_empty), float(both_empty), float(both_empty), counts)
    S = _sim_matrix(sim, _names(pred, pc), _names(gold, gc))
    p = float(S.max(axis=1).mean())
    r = float(S.max(axis=0).mean())
    return PRF(p, r, _f1(p, r), counts)


def category_soft_f1(
    pred: Taxonomy,
    gold: Taxonomy,
    embedder: Any = None,
    similarity: Optional[Similarity] = None,
    include_root: bool = False,
) -> float:
    """Cat of Table 3 as a float (the F1 of :func:`category_soft_prf`)."""
    return category_soft_prf(pred, gold, embedder, similarity, include_root).f1


def _path_label(tax: Taxonomy, nid: str) -> str:
    return " > ".join(tax.nodes[n].name for n in tax.path(nid)[1:])


def align_categories(
    pred: Taxonomy,
    gold: Taxonomy,
    embedder: Any = None,
    similarity: Optional[Similarity] = None,
    threshold: float = DEFAULT_CATEGORY_THRESHOLD,
    include_root: bool = False,
    tie_break: float = 1e-3,
) -> Dict[str, str]:
    """One-to-one alignment π of predicted to gold categories (used by Assn).

    Only pairs with name similarity Sim(ĉ, c) >= ``threshold`` are eligible.
    Among the eligible pairs, π is a maximum-weight bipartite matching
    (Hungarian algorithm, :func:`scipy.optimize.linear_sum_assignment`) with
    weights Sim(ĉ, c) + ``tie_break`` * Sim(path(ĉ), path(c)). Ineligible
    pairs get weight 0 before the matching, so a sub-threshold similarity can
    never displace an eligible pair. The path term breaks ties between equally
    named categories, e.g. "Adverse events" under two aspects, using the
    similarity of their full root-to-node labels. Returns
    ``{pred node id: gold node id}``. Needs ``embedder`` or ``similarity``.
    """
    sim = _embedding_similarity(embedder, similarity, "Category alignment (Assn)")
    pc, gc = _categories(pred, include_root), _categories(gold, include_root)
    if not pc or not gc:
        return {}
    S = _sim_matrix(sim, _names(pred, pc), _names(gold, gc))
    eligible = S >= float(threshold)
    if not eligible.any():
        return {}
    W = S.copy()
    if tie_break > 0:
        T = _sim_matrix(sim, [_path_label(pred, n) for n in pc], [_path_label(gold, n) for n in gc])
        W = W + float(tie_break) * T
    W = np.where(eligible, W, 0.0)
    rows, cols = linear_sum_assignment(W, maximize=True)
    return {pc[i]: gc[j] for i, j in zip(rows, cols) if eligible[i, j]}


def _assignment_pairs(tax: Taxonomy, mode: str, include_root: bool) -> Set[Tuple[str, str]]:
    root = tax.root_id
    if mode == "extent":
        return {(p, nid) for nid, node in tax.nodes.items() if include_root or nid != root for p in node.papers}
    if mode == "direct":
        return {(p, nid) for p, nodes in paper_attachments(tax).items() for nid in nodes if include_root or nid != root}
    raise ValueError("mode must be 'extent' or 'direct'")


def assignment_prf(
    pred: Taxonomy,
    gold: Taxonomy,
    embedder: Any = None,
    similarity: Optional[Similarity] = None,
    threshold: float = DEFAULT_CATEGORY_THRESHOLD,
    mode: str = "extent",
    include_root: bool = False,
    paper_map: Optional[Mapping[str, str]] = None,
    category_alignment: Optional[Mapping[str, str]] = None,
) -> PRF:
    """Assignment precision / recall / F1 (Assn, Table 3) over (study, category) pairs.

    1. Predicted categories are aligned one-to-one to gold categories with
       :func:`align_categories` (Hungarian over the pairs with Sim >=
       ``threshold``), unless ``category_alignment`` is given.
    2. Gold pairs G = {(s, c)}: study s belongs to gold category c. With
       ``mode="extent"`` (default) a study belongs to a category and all of
       its ancestors. This is the path-aggregated membership of CHIME's
       claim-category annotations (``agg_label``) and of its hierarchies,
       where a category lists the claims of its whole subtree. With
       ``mode="direct"`` only the most specific categories count. Roots are
       excluded unless ``include_root``.
    3. Predicted pairs P = {(s, π(ĉ))} for every predicted membership, where
       π maps aligned categories to their gold counterpart; memberships in
       unaligned categories can never match. ``paper_map`` maps predicted
       paper ids to gold study ids when they differ.
    4. precision = |P ∩ G| / |P|, recall = |P ∩ G| / |G|, Assn = F1. If both
       pair sets are empty, P = R = F1 = 1.

    Studies the system did not retrieve lower recall and irrelevant
    retrieved studies lower precision, so Assn reflects both retrieval and
    organization, as intended by the pooled-corpus setting (App. C.2).
    Needs ``embedder`` or ``similarity`` unless ``category_alignment`` is given.
    """
    if category_alignment is None:
        category_alignment = align_categories(pred, gold, embedder, similarity, threshold, include_root)
    pmap = paper_map or {}
    gold_pairs = _assignment_pairs(gold, mode, include_root)
    pred_pairs = {
        (pmap.get(p, p), category_alignment.get(nid, "\x00pred:" + nid))
        for p, nid in _assignment_pairs(pred, mode, include_root)
    }
    tp = len(pred_pairs & gold_pairs)
    counts = {
        "n_pred_pairs": float(len(pred_pairs)),
        "n_gold_pairs": float(len(gold_pairs)),
        "n_correct_pairs": float(tp),
        "n_aligned_categories": float(len(category_alignment)),
    }
    if not pred_pairs and not gold_pairs:
        return PRF(1.0, 1.0, 1.0, counts)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(gold_pairs) if gold_pairs else 0.0
    return PRF(precision, recall, _f1(precision, recall), counts)


def assignment_f1(
    pred: Taxonomy,
    gold: Taxonomy,
    embedder: Any = None,
    similarity: Optional[Similarity] = None,
    threshold: float = DEFAULT_CATEGORY_THRESHOLD,
    mode: str = "extent",
    include_root: bool = False,
    paper_map: Optional[Mapping[str, str]] = None,
    category_alignment: Optional[Mapping[str, str]] = None,
) -> float:
    """Assn of Table 3 as a float (the F1 of :func:`assignment_prf`)."""
    return assignment_prf(
        pred, gold, embedder, similarity, threshold, mode, include_root, paper_map, category_alignment
    ).f1


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def organization_scores(
    pred: Taxonomy,
    gold: Union[Taxonomy, Sequence[Taxonomy]],
    benchmark: str,
    embedder: Any = None,
    *,
    similarity: Optional[Similarity] = None,
    titles: TitleSource = None,
    gold_titles: TitleSource = None,
    threshold: Optional[float] = None,
    rule: str = PUBLIC_RULE,
    ari_view: str = ARI_INTERSECTION,
    category_threshold: float = DEFAULT_CATEGORY_THRESHOLD,
    paper_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, float]:
    """All organization metrics of one query for ``benchmark`` in {"taxobench", "chime"}.

    TaxoBench: titles are aligned once and the alignment is shared by all
    metrics. The keys are:

    * ``ari``, ``v_measure``, ``homogeneity``, ``completeness`` in the view
      given by ``ari_view``: ``"intersection"`` (default, ARI∩, Table 3) or
      ``"e2e"`` (with ⊥, the public CLI);
    * ``ari_intersection``, ``ari_e2e``, ``v_measure_intersection``,
      ``v_measure_e2e`` (both views, always reported);
    * ``sem`` (Eq. 13, mean over aligned pairs; Table 3) and ``sem_expert``
      (unretrieved expert papers score 0);
    * ``paper_recall``, ``paper_precision``, ``paper_f1`` and counts.

    Without an embedder, Sem uses the public scorer's string fallback.

    CHIME: ``gold`` may be a list of aspect hierarchies, which are merged
    with :func:`merge_hierarchies`. The keys are ``cat``, ``cat_precision``,
    ``cat_recall``, ``assn``, ``assn_precision``, ``assn_recall`` and counts.
    Requires ``embedder`` or ``similarity``.
    """
    bench = benchmark.lower().replace("-", "").replace("_", "")
    if bench not in ("taxobench", "chime"):
        raise ValueError(f"Unknown organization benchmark {benchmark!r}; expected 'taxobench' or 'chime'.")
    if not isinstance(gold, Taxonomy):
        gold = merge_hierarchies(list(gold), root_name=pred.root.name)
    if bench == "taxobench":
        if ari_view not in (ARI_INTERSECTION, ARI_E2E):
            raise ValueError(f"ari_view must be {ARI_INTERSECTION!r} or {ARI_E2E!r}, not {ari_view!r}")
        sim = similarity if similarity is not None else NodeSimilarity(embedder)
        alignment = _alignment_for(pred, gold, titles, gold_titles, threshold, rule)
        views = {
            ARI_INTERSECTION: leaf_clustering_scores(pred, gold, alignment=alignment, include_missing=False),
            ARI_E2E: leaf_clustering_scores(pred, gold, alignment=alignment, include_missing=True),
        }
        main = views[ari_view]
        sem = semantic_path_scores(pred, gold, similarity=sim, alignment=alignment)
        return {
            "ari": main["ari"],
            "v_measure": main["v_measure"],
            "homogeneity": main["homogeneity"],
            "completeness": main["completeness"],
            "ari_intersection": views[ARI_INTERSECTION]["ari"],
            "ari_e2e": views[ARI_E2E]["ari"],
            "v_measure_intersection": views[ARI_INTERSECTION]["v_measure"],
            "v_measure_e2e": views[ARI_E2E]["v_measure"],
            "sem": sem["sem_path"],
            "sem_expert": sem["sem_path_expert"],
            "paper_recall": alignment.recall,
            "paper_precision": alignment.precision,
            "paper_f1": alignment.f1,
            "n_aligned": float(alignment.n_aligned),
            "n_reference": float(alignment.n_reference),
            "n_predicted": float(alignment.n_predicted),
        }
    sim = _embedding_similarity(embedder, similarity, "CHIME Cat / Assn")
    cat = category_soft_prf(pred, gold, similarity=sim)
    assn = assignment_prf(pred, gold, similarity=sim, threshold=category_threshold, paper_map=paper_map)
    out = {
        "cat": cat.f1,
        "cat_precision": cat.precision,
        "cat_recall": cat.recall,
        "assn": assn.f1,
        "assn_precision": assn.precision,
        "assn_recall": assn.recall,
        "n_pred_categories": cat.counts["n_pred"],
        "n_gold_categories": cat.counts["n_gold"],
    }
    out.update(assn.counts)
    return out


def macro_average(rows: Iterable[Mapping[str, Any]], keys: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Macro-average per-query scores (each key over the rows that report a finite value).

    ``keys`` defaults to every numeric key in order of first appearance.
    ``n_queries`` is the number of rows.
    """
    rows = list(rows)
    if keys is None:
        seen: Dict[str, None] = {}
        for r in rows:
            for k, v in r.items():
                if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                    seen.setdefault(k, None)
        keys = list(seen)
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None and math.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else 0.0
    out["n_queries"] = float(len(rows))
    return out
