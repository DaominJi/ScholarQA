"""Multi-granularity semantic index I (Section 2.1).

The index integrates three levels of information:

* **concepts** of the problem taxonomy T_P and the method taxonomy T_M;
* **paper summaries** (title + abstract) linked to their problem and method
  concepts (a paper of a concept also belongs to all of its ancestors);
* **paper details**: section chunks of each paper's full text.

Every index node x has a text, an embedding (unit norm), and a token cost
c(x) (Section 4.2). Index nodes are addressed by namespaced *keys*:

* ``"concept:<node_id>"`` -- e.g. ``"concept:P12"`` (problem) / ``"concept:M3"`` (method)
* ``"paper:<paper_id>"``
* ``"chunk:<chunk_id>"``

For the bound-guided search of Section 4.3 the index caches, for every
concept v and every level (concept / paper / chunk), the embedding center
x̄_v and radius ρ_v = max ||x_y - x̄_v||_2 of the index nodes below v, so
that x_a^T x_y <= x_a^T x̄_v + ρ_v (Cauchy-Schwarz) for every y below v.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from scholarpath.config import IndexConfig
from scholarpath.embeddings import Embedder, l2_normalize
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.text import count_tokens
from scholarpath.types import Chunk, Paper

CONCEPT, PAPER, CHUNK = "concept", "paper", "chunk"
LEVELS = (CONCEPT, PAPER, CHUNK)


def concept_key(node_id: str) -> str:
    return f"{CONCEPT}:{node_id}"


def paper_key(paper_id: str) -> str:
    return f"{PAPER}:{paper_id}"


def chunk_key(chunk_id: str) -> str:
    return f"{CHUNK}:{chunk_id}"


def parse_key(key: str) -> Tuple[str, str]:
    """``"paper:2101.0001"`` -> ``("paper", "2101.0001")``."""
    level, _, raw = key.partition(":")
    if level not in LEVELS:
        raise ValueError(f"Bad index key {key!r}")
    return level, raw


@dataclass(repr=False)
class Ball:
    """Embedding center and radius of a set of unit vectors."""

    center: np.ndarray
    radius: float
    count: int

    def __repr__(self) -> str:
        return f"Ball(count={self.count}, radius={self.radius:.4f}, |center|={float(np.linalg.norm(self.center)):.4f})"

    def upper_cos(self, x: np.ndarray) -> float:
        """Upper bound on max_y x^T y over the set (x unit norm)."""
        if self.count == 0:
            return -1.0
        return float(np.dot(x, self.center) + self.radius)


def _ball(mat: np.ndarray) -> Ball:
    if mat.shape[0] == 0:
        return Ball(np.zeros(mat.shape[1], dtype=np.float32), 0.0, 0)
    c = mat.mean(axis=0)
    r = float(np.sqrt(((mat - c) ** 2).sum(axis=1)).max())
    return Ball(c.astype(np.float32), r, int(mat.shape[0]))


class _Matrix:
    """Row-indexed embedding matrix."""

    def __init__(self, ids: Sequence[str], mat: np.ndarray):
        self.ids = list(ids)
        self.mat = np.asarray(mat, dtype=np.float32)
        self.row = {k: i for i, k in enumerate(self.ids)}

    def get(self, key: str) -> np.ndarray:
        return self.mat[self.row[key]]

    def rows(self, keys: Iterable[str]) -> np.ndarray:
        idx = [self.row[k] for k in keys if k in self.row]
        if not idx:
            return np.zeros((0, self.mat.shape[1] if self.mat.ndim == 2 else 0), dtype=np.float32)
        return self.mat[idx]


class SemanticIndex:
    """Multi-granularity semantic index over a scholarly repository."""

    def __init__(
        self,
        problem: Taxonomy,
        method: Taxonomy,
        papers: Dict[str, Paper],
        chunks: Optional[Dict[str, Chunk]] = None,
    ):
        self.problem = problem
        self.method = method
        self.papers = papers
        self.chunks: Dict[str, Chunk] = chunks or {}
        self.paper_chunks: Dict[str, List[str]] = {}
        for cid, ch in self.chunks.items():
            self.paper_chunks.setdefault(ch.paper_id, []).append(cid)
        for pid in self.paper_chunks:
            self.paper_chunks[pid].sort(key=lambda c: self.chunks[c].order)
        self._emb: Dict[str, _Matrix] = {}
        self.cost: Dict[str, int] = {}
        self.balls: Dict[Tuple[str, str], Ball] = {}  # (concept node id, level) -> Ball
        self.paper_chunk_balls: Dict[str, Ball] = {}
        self.metadata: Dict[str, object] = {}
        # paper -> concepts (incl. ancestors) per facet
        self._paper_concepts: Dict[str, Dict[str, Set[str]]] = {"problem": {}, "method": {}}
        self._refresh_postings()

    # ------------------------------------------------------------ taxonomy
    def taxonomy(self, facet: str) -> Taxonomy:
        if facet == "problem":
            return self.problem
        if facet == "method":
            return self.method
        raise KeyError(facet)

    def facet_of(self, node_id: str) -> str:
        if node_id in self.problem.nodes:
            return "problem"
        if node_id in self.method.nodes:
            return "method"
        raise KeyError(f"Unknown concept {node_id!r}")

    def concept(self, node_id: str):
        return self.taxonomy(self.facet_of(node_id)).nodes[node_id]

    def _refresh_postings(self) -> None:
        for facet, tax in (("problem", self.problem), ("method", self.method)):
            idx: Dict[str, Set[str]] = {}
            for nid, node in tax.nodes.items():
                for p in node.papers:
                    idx.setdefault(p, set()).add(nid)
            self._paper_concepts[facet] = idx

    def extent(self, node_id: str) -> Set[str]:
        """Posting list of a concept: papers in its extent P_v."""
        return self.concept(node_id).papers

    def paper_concepts(self, paper_id: str, facet: str) -> Set[str]:
        return self._paper_concepts[facet].get(paper_id, set())

    def shared_papers(self, u: str, m: str) -> Set[str]:
        """J(u, m): papers that study problem u with method m."""
        a, b = self.extent(u), self.extent(m)
        if len(a) > len(b):
            a, b = b, a
        return {p for p in a if p in b}

    def co_concepts(self, node_id: str) -> Set[str]:
        """Concepts of the opposite taxonomy sharing at least one paper with ``node_id``."""
        facet = self.facet_of(node_id)
        other = "method" if facet == "problem" else "problem"
        out: Set[str] = set()
        for p in self.extent(node_id):
            out |= self.paper_concepts(p, other)
        return out

    # ----------------------------------------------------------- node info
    def text(self, key: str) -> str:
        level, raw = parse_key(key)
        if level == CONCEPT:
            return self.concept(raw).text
        if level == PAPER:
            return self.papers[raw].text
        ch = self.chunks[raw]
        return f"{ch.heading}\n{ch.text}" if ch.heading else ch.text

    def node_cost(self, key: str) -> int:
        c = self.cost.get(key)
        if c is None:
            c = max(1, count_tokens(self.text(key)))
            self.cost[key] = c
        return c

    def vector(self, key: str) -> np.ndarray:
        level, _ = parse_key(key)
        return self._emb[level].get(key)

    def has_vector(self, key: str) -> bool:
        level, _ = parse_key(key)
        m = self._emb.get(level)
        return m is not None and key in m.row

    def vectors(self, keys: Sequence[str]) -> np.ndarray:
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.vector(k) for k in keys])

    @property
    def dim(self) -> int:
        for m in self._emb.values():
            if m.mat.ndim == 2 and m.mat.shape[0]:
                return int(m.mat.shape[1])
        return 0

    def concept_matrix(self, facet: str) -> Tuple[List[str], np.ndarray]:
        """(node ids, embedding matrix) of every concept of a taxonomy."""
        tax = self.taxonomy(facet)
        ids = list(tax.nodes)
        return ids, self._emb[CONCEPT].rows(concept_key(n) for n in ids)

    def paper_matrix(self) -> Tuple[List[str], np.ndarray]:
        m = self._emb[PAPER]
        return [parse_key(k)[1] for k in m.ids], m.mat

    def level_matrix(self, level: str) -> Tuple[List[str], np.ndarray]:
        """(keys, embedding matrix) of every index node of one level (bulk access)."""
        m = self._emb.get(level)
        if m is None:
            return [], np.zeros((0, self.dim), dtype=np.float32)
        return list(m.ids), m.mat

    def chunks_of(self, paper_id: str) -> List[str]:
        return self.paper_chunks.get(paper_id, [])

    def ball(self, node_id: str, level: str) -> Ball:
        return self.balls[(node_id, level)]

    # -------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        problem: Taxonomy,
        method: Taxonomy,
        papers: Union[Dict[str, Paper], Iterable[Paper]],
        embedder: Embedder,
        config: Optional[IndexConfig] = None,
        chunks: Optional[Iterable[Chunk]] = None,
        paper_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> "SemanticIndex":
        """Embed all index nodes, compute token costs and subtree balls.

        ``paper_embeddings`` lets callers reuse embeddings computed during
        taxonomy construction.
        """
        config = config or IndexConfig()
        if not isinstance(papers, dict):
            papers = {p.paper_id: p for p in papers}
        chunk_map: Dict[str, Chunk] = {}
        if chunks is not None:
            chunk_map = {c.chunk_id: c for c in chunks}
        elif config.use_fulltext:
            chunk_map = {c.chunk_id: c for p in papers.values() for c in p.chunks}
        # Keep paper records light: chunks live in the chunk map.
        light = {}
        for pid, p in papers.items():
            q = Paper.from_dict(p.to_dict(include_chunks=False))
            light[pid] = q
        index = cls(problem, method, light, chunk_map)
        index._embed_all(embedder, paper_embeddings)
        index._compute_costs()
        index._compute_balls()
        return index

    def _embed_all(self, embedder: Embedder, paper_embeddings: Optional[Dict[str, np.ndarray]] = None) -> None:
        concept_ids = [concept_key(n) for n in self.problem.nodes] + [concept_key(n) for n in self.method.nodes]
        concept_texts = [self.text(k) for k in concept_ids]
        self._emb[CONCEPT] = _Matrix(concept_ids, embedder.encode(concept_texts))

        pids = list(self.papers)
        pkeys = [paper_key(p) for p in pids]
        if paper_embeddings is not None and all(p in paper_embeddings for p in pids):
            pmat = l2_normalize(np.stack([paper_embeddings[p] for p in pids])) if pids else np.zeros((0, embedder.dim))
        else:
            pmat = embedder.encode([self.papers[p].text for p in pids])
        self._emb[PAPER] = _Matrix(pkeys, pmat)

        ckeys = [chunk_key(c) for c in self.chunks]
        cmat = embedder.encode([self.text(k) for k in ckeys]) if ckeys else np.zeros((0, embedder.dim), dtype=np.float32)
        self._emb[CHUNK] = _Matrix(ckeys, cmat)

    def refresh_concepts(self, embedder: Embedder) -> None:
        """Re-embed concepts and recompute postings/balls after taxonomy edits."""
        concept_ids = [concept_key(n) for n in self.problem.nodes] + [concept_key(n) for n in self.method.nodes]
        self._emb[CONCEPT] = _Matrix(concept_ids, embedder.encode([self.text(k) for k in concept_ids]))
        for k in concept_ids:
            self.cost.pop(k, None)
        self._refresh_postings()
        self._compute_costs()
        self._compute_balls()

    def _compute_costs(self) -> None:
        for m in self._emb.values():
            for k in m.ids:
                self.node_cost(k)

    def _compute_balls(self) -> None:
        self.balls.clear()
        self.paper_chunk_balls.clear()
        cmat = self._emb[CHUNK]
        for pid, cids in self.paper_chunks.items():
            self.paper_chunk_balls[pid] = _ball(cmat.rows(chunk_key(c) for c in cids))
        for tax in (self.problem, self.method):
            for nid in tax.nodes:
                desc = [concept_key(d) for d in tax.descendants(nid, include_self=True)]
                self.balls[(nid, CONCEPT)] = _ball(self._emb[CONCEPT].rows(desc))
                ext = tax.nodes[nid].papers
                self.balls[(nid, PAPER)] = _ball(self._emb[PAPER].rows(paper_key(p) for p in ext))
                ch_keys = [chunk_key(c) for p in ext for c in self.paper_chunks.get(p, [])]
                self.balls[(nid, CHUNK)] = _ball(cmat.rows(ch_keys))

    # ---------------------------------------------------------------- io
    def save(self, directory: Union[str, Path]) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.problem.save(d / "problem_taxonomy.json")
        self.method.save(d / "method_taxonomy.json")
        with open(d / "papers.jsonl", "w", encoding="utf-8") as f:
            for p in self.papers.values():
                f.write(json.dumps(p.to_dict(include_chunks=False), ensure_ascii=False) + "\n")
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks.values():
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        for level, m in self._emb.items():
            np.save(d / f"emb_{level}.npy", m.mat)
            (d / f"emb_{level}.ids.json").write_text(json.dumps(m.ids), encoding="utf-8")
        (d / "costs.json").write_text(json.dumps(self.cost), encoding="utf-8")
        (d / "metadata.json").write_text(json.dumps(self.metadata, indent=1, default=str), encoding="utf-8")

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "SemanticIndex":
        d = Path(directory)
        problem = Taxonomy.load(d / "problem_taxonomy.json")
        method = Taxonomy.load(d / "method_taxonomy.json")
        papers: Dict[str, Paper] = {}
        with open(d / "papers.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = Paper.from_dict(json.loads(line))
                    papers[p.paper_id] = p
        chunks: Dict[str, Chunk] = {}
        if (d / "chunks.jsonl").exists():
            with open(d / "chunks.jsonl", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = Chunk.from_dict(json.loads(line))
                        chunks[c.chunk_id] = c
        index = cls(problem, method, papers, chunks)
        for level in LEVELS:
            mp = d / f"emb_{level}.npy"
            if mp.exists():
                ids = json.loads((d / f"emb_{level}.ids.json").read_text(encoding="utf-8"))
                index._emb[level] = _Matrix(ids, np.load(mp))
        if (d / "costs.json").exists():
            index.cost = {k: int(v) for k, v in json.loads((d / "costs.json").read_text()).items()}
        if (d / "metadata.json").exists():
            index.metadata = json.loads((d / "metadata.json").read_text())
        index._compute_balls()
        return index

    def stats(self) -> Dict[str, object]:
        return {
            "papers": len(self.papers),
            "chunks": len(self.chunks),
            "problem": self.problem.stats(),
            "method": self.method.stats(),
            "tokens": {
                level: int(sum(v for k, v in self.cost.items() if k.startswith(level + ":")))
                for level in LEVELS
            },
        }
