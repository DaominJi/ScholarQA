"""LightRAG baseline [17] (Table 2, Tables 3-5, "LightRAG"), hybrid mode.

Offline graph indexing (Guo et al., 2024, Section 3.1):

* **Extraction.** For every text unit the LLM extracts entities
  ``(name, type, description)`` and relationships
  ``(source, target, keywords, description, strength)`` (task
  ``baselines.lightrag_extract``), followed by ``max_gleaning`` gleaning
  passes ("MANY entities and relationships were missed ..."), all billed to
  ``index_build``. Text units are the passages of the shared pool (abstracts
  and section chunks), so that all baselines read the same units.
* **Deduplication.** Entities are merged by upper-cased name (type = most
  frequent type, descriptions joined), relations by unordered endpoint pair
  (weights summed, keywords and descriptions joined); every record keeps the
  ids of its source text units. A relation endpoint that was never extracted
  as an entity is added with type ``unknown``. When an entity or relation
  accumulates at least ``force_llm_summary_on_merge`` description fragments
  or more than ``description_max_tokens`` tokens, the fragments are
  summarized by the LLM (task ``baselines.lightrag_summarize``).
* **Graph and vector stores.** Entities and relations form an undirected
  graph (dict adjacency; :meth:`KnowledgeGraph.to_networkx` exports it);
  entity vectors embed ``name + description`` and relation vectors embed
  ``keywords + source + target + description``.

Query time ("hybrid" retrieval, Section 3.2), one LLM call:

1. The LLM extracts high-level and low-level keywords from the query (task
   ``baselines.lightrag_keywords``, component ``query_understanding``).
2. *Local*: low-level keywords retrieve the ``top_k`` most similar entities;
   their one-hop relations are ranked by (edge degree, weight) and their
   source text units by (entity rank, number of one-hop neighbours sharing
   the unit).
3. *Global*: high-level keywords retrieve the ``top_k`` most similar
   relations, ranked by (edge degree, weight); their endpoint entities and
   source text units follow in relation order.
4. Local and global lists are merged round-robin with deduplication, and
   packed to the budget B in LightRAG's order -- entities, then relations,
   then text units -- with the entity and relation sections capped at
   ``entity_budget_fraction`` / ``relation_budget_fraction`` of B (LightRAG's
   ``max_entity_tokens`` / ``max_relation_tokens`` relative to its total
   token budget) and the text units filling the rest.

Together with the composition call this gives the 2 LLM calls per query of
Table 5. The candidate-pool option (``candidate_retriever`` +
``LightRAGConfig.candidate_papers``) restricts the graph to the units of the
top-k candidate papers of a query, like RAPTOR; extractions are cached per
text unit so that overlapping candidate sets are extracted once. Graph
construction (extraction, description summaries, entity / relation
embedding) is treated as indexing: its calls are billed to ``index_build`` and
its wall time is reported as ``RetrievalResult.stats["index_build_s"]`` and
excluded from ``RetrievalResult.latency_s``. With a per-query candidate pool
(corpora above ``SystemOptions.graph_global_max_papers``, i.e.
arXiv-Subfields) this construction runs at query time -- a deviation from
the paper's globally indexed LightRAG -- so the "2 calls per query" of
Table 5 cover the online keyword and composition calls only; the per-query
extraction work is recorded with every answer and reported in the
"Index-build" columns of run_efficiency.py's ``table5_detail``.

**Citations of graph units.** An entity / relation description merges
fragments extracted from several text units, so its unit cites (and is
credited with, for Recall@B / NDCG@B) up to ``max_sources_per_unit`` (5)
source papers. They are chosen by query relevance: source units in the order
of this query's merged local/global text-unit ranking first, then the
remaining source units in extraction order. ``ranked_papers`` lists the
papers of packed text units first, then papers known only through graph
provenance. ``max_sources_per_unit = 0`` credits text units only (an
ablation; entity / relation blocks then carry no paper ids).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from scholarpath.baselines.common import (
    ENTITY,
    RELATION,
    Candidate,
    Passage,
    PassagePool,
    QueryLike,
    as_query,
    ensure_built,
    top_k_indices,
)
from scholarpath.baselines.packing import BudgetPacker
from scholarpath.embeddings import Embedder
from scholarpath.llm.base import LLMClient
from scholarpath.llm.parsing import JSONParseError, extract_json
from scholarpath.prompts.baselines import (
    DEFAULT_ENTITY_TYPES,
    lightrag_extract_request,
    lightrag_glean_check_request,
    lightrag_glean_request,
    lightrag_keywords_request,
    lightrag_summarize_request,
    parse_extraction,
    parse_keywords,
)
from scholarpath.retrieval.result import RetrievalResult
from scholarpath.text import count_tokens, truncate_tokens

log = logging.getLogger(__name__)

SEP = "<SEP>"  # LightRAG's GRAPH_FIELD_SEP


@dataclass
class LightRAGConfig:
    """LightRAG hyper-parameters (defaults follow the LightRAG implementation)."""

    entity_types: Tuple[str, ...] = DEFAULT_ENTITY_TYPES
    max_gleaning: int = 1  # entity_extract_max_gleaning
    max_extract_input_tokens: int = 1200  # chunk_token_size
    summarize_descriptions: bool = True
    force_llm_summary_on_merge: int = 6
    description_max_tokens: int = 500  # summary_max_tokens
    top_k: int = 40  # entities (local) / relations (global) retrieved per query
    min_similarity: float = 0.2  # cosine_better_than_threshold
    entity_budget_fraction: float = 0.2  # max_entity_tokens / max_total_tokens = 6000 / 30000
    relation_budget_fraction: float = 0.2667  # max_relation_tokens / max_total_tokens = 8000 / 30000
    max_sources_per_unit: int = 5  # source papers cited by an entity / relation unit (0 = none)
    candidate_papers: Optional[int] = None
    max_workers: Optional[int] = None


@dataclass
class Entity:
    name: str
    type: str = "unknown"
    description: str = ""
    sources: List[str] = field(default_factory=list)  # text-unit keys, in insertion order
    fragments: List[str] = field(default_factory=list)
    types: Dict[str, int] = field(default_factory=dict)


@dataclass
class Relation:
    source: str
    target: str
    keywords: str = ""
    description: str = ""
    weight: float = 0.0
    sources: List[str] = field(default_factory=list)
    fragments: List[str] = field(default_factory=list)

    @property
    def key(self) -> Tuple[str, str]:
        return edge_key(self.source, self.target)


def entity_key(name: str) -> str:
    """LightRAG normalizes entity names to upper case (quotes stripped)."""
    return " ".join(name.strip().strip('"').strip("'").upper().split())


def edge_key(a: str, b: str) -> Tuple[str, str]:
    a, b = entity_key(a), entity_key(b)
    return (a, b) if a <= b else (b, a)


@dataclass
class Extraction:
    """Parsed extraction of one text unit."""

    entities: List[Dict[str, str]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    content_keywords: List[str] = field(default_factory=list)


def _add_unique(seq: List[str], items: Iterable[str]) -> None:
    for x in items:
        if x and x not in seq:
            seq.append(x)


class KnowledgeGraph:
    """Entity-relation graph with source text units (dict-based adjacency)."""

    def __init__(self) -> None:
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[Tuple[str, str], Relation] = {}
        self.adj: Dict[str, Set[str]] = {}

    def __len__(self) -> int:
        return len(self.entities)

    # ------------------------------------------------------------ merge
    def add_extraction(self, unit_key: str, ex: Extraction) -> None:
        for e in ex.entities:
            k = entity_key(e["name"])
            if not k:
                continue
            ent = self.entities.get(k)
            if ent is None:
                ent = self.entities[k] = Entity(name=k)
                self.adj.setdefault(k, set())
            t = e.get("type") or "unknown"
            ent.types[t] = ent.types.get(t, 0) + 1
            _add_unique(ent.fragments, [e.get("description", "").strip()])
            _add_unique(ent.sources, [unit_key])
        for r in ex.relations:
            a, b = entity_key(r["source"]), entity_key(r["target"])
            if not a or not b or a == b:
                continue
            for endpoint in (a, b):
                if endpoint not in self.entities:
                    ent = self.entities[endpoint] = Entity(name=endpoint)
                    ent.types["unknown"] = 1
                    _add_unique(ent.fragments, [r.get("description", "").strip()])
                    _add_unique(ent.sources, [unit_key])
                    self.adj.setdefault(endpoint, set())
            k = edge_key(a, b)
            rel = self.relations.get(k)
            if rel is None:
                rel = self.relations[k] = Relation(source=k[0], target=k[1])
            rel.weight += float(r.get("weight", 1.0))
            kws = [w.strip() for w in str(r.get("keywords", "")).split(",") if w.strip()]
            existing = [w.strip() for w in rel.keywords.split(",") if w.strip()]
            _add_unique(existing, kws)
            rel.keywords = ", ".join(existing)
            _add_unique(rel.fragments, [r.get("description", "").strip()])
            _add_unique(rel.sources, [unit_key])
            self.adj[k[0]].add(k[1])
            self.adj[k[1]].add(k[0])

    def finalize(self, llm: Optional[LLMClient], config: LightRAGConfig) -> None:
        """Set merged types and descriptions (LLM summary of long merges)."""
        jobs: List[Tuple[Union[Entity, Relation], str]] = []
        for ent in self.entities.values():
            ent.type = max(sorted(ent.types), key=lambda t: ent.types[t]) if ent.types else "unknown"
            ent.description = SEP.join(sorted(ent.fragments))
            if self._needs_summary(ent.fragments, config):
                jobs.append((ent, ent.name))
        for rel in self.relations.values():
            rel.description = SEP.join(sorted(rel.fragments))
            if self._needs_summary(rel.fragments, config):
                jobs.append((rel, f"{rel.source}, {rel.target}"))
        if not jobs:
            return
        if llm is None or not config.summarize_descriptions:
            for obj, _ in jobs:
                obj.description = truncate_tokens(obj.description, config.description_max_tokens)
            return
        reqs = [lightrag_summarize_request(name, sorted(obj.fragments), config.description_max_tokens)
                for obj, name in jobs]
        resps = llm.map(llm.complete, reqs, max_workers=config.max_workers)
        for (obj, _), resp in zip(jobs, resps):
            obj.description = resp.text.strip() or obj.description

    @staticmethod
    def _needs_summary(fragments: Sequence[str], config: LightRAGConfig) -> bool:
        if len(fragments) >= config.force_llm_summary_on_merge:
            return True
        return count_tokens(SEP.join(fragments)) > config.description_max_tokens

    # -------------------------------------------------------- structure
    def degree(self, name: str) -> int:
        return len(self.adj.get(name, ()))

    def edge_degree(self, key: Tuple[str, str]) -> int:
        return self.degree(key[0]) + self.degree(key[1])

    def to_networkx(self) -> Any:  # pragma: no cover - optional dependency
        try:
            import networkx as nx
        except ImportError as err:
            raise ImportError("KnowledgeGraph.to_networkx needs networkx: pip install networkx") from err
        g = nx.Graph()
        for k, e in self.entities.items():
            g.add_node(k, entity_type=e.type, description=e.description, source_id=SEP.join(e.sources))
        for (a, b), r in self.relations.items():
            g.add_edge(a, b, weight=r.weight, keywords=r.keywords, description=r.description,
                       source_id=SEP.join(r.sources))
        return g

    # --------------------------------------------------------------- io
    def to_dict(self) -> Dict[str, Any]:
        return {"entities": [asdict(e) for e in self.entities.values()],
                "relations": [asdict(r) for r in self.relations.values()]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeGraph":
        g = cls()
        for e in d.get("entities", []):
            ent = Entity(**e)
            g.entities[ent.name] = ent
            g.adj.setdefault(ent.name, set())
        for r in d.get("relations", []):
            rel = Relation(**r)
            g.relations[rel.key] = rel
            g.adj.setdefault(rel.key[0], set()).add(rel.key[1])
            g.adj.setdefault(rel.key[1], set()).add(rel.key[0])
        return g


def _round_robin(a: Sequence[Any], b: Sequence[Any]) -> List[Any]:
    """Interleave two rankings (``a`` first) and drop repeated items."""
    out: List[Any] = []
    seen: Set[Any] = set()
    for i in range(max(len(a), len(b))):
        for seq in (a, b):
            if i < len(seq) and seq[i] not in seen:
                seen.add(seq[i])
                out.append(seq[i])
    return out


@dataclass
class _GraphView:
    """A knowledge graph with its entity / relation vector stores."""

    graph: KnowledgeGraph
    entity_names: List[str]
    entity_vecs: np.ndarray
    relation_keys: List[Tuple[str, str]]
    relation_vecs: np.ndarray


class LightRAGRetriever:
    """LightRAG hybrid retrieval over the passage pool, packed to B."""

    def __init__(
        self,
        pool: PassagePool,
        llm: LLMClient,
        embedder: Embedder,
        config: Optional[LightRAGConfig] = None,
        candidate_retriever: Optional[Any] = None,
        name: str = "LightRAG",
    ):
        self.pool = pool
        self.llm = llm
        self.embedder = embedder
        self.config = config or LightRAGConfig()
        self.candidate_retriever = candidate_retriever
        self.name = name
        self.extractions: Dict[str, Extraction] = {}
        self.view: Optional[_GraphView] = None

    @property
    def per_query(self) -> bool:
        return self.candidate_retriever is not None and bool(self.config.candidate_papers)

    @property
    def graph(self) -> Optional[KnowledgeGraph]:
        return None if self.view is None else self.view.graph

    # ------------------------------------------------------- extraction
    def _extract_one(self, passage: Passage) -> Extraction:
        cfg = self.config
        text = passage.search_text
        types = list(cfg.entity_types)
        req = lightrag_extract_request(text, types, cfg.max_extract_input_tokens)
        resp = self.llm.complete(req)
        ex = Extraction()
        history = resp.text

        def absorb(raw: str) -> None:
            try:
                obj = extract_json(raw, expect=dict)
            except JSONParseError:
                log.warning("LightRAG: unparsable extraction for %s", passage.key)
                return
            ents, rels, kws = parse_extraction(obj)
            ex.entities.extend(ents)
            ex.relations.extend(rels)
            _add_unique(ex.content_keywords, kws)

        absorb(resp.text)
        for g in range(cfg.max_gleaning):
            glean = self.llm.complete(lightrag_glean_request(text, history, types, cfg.max_extract_input_tokens))
            absorb(glean.text)
            history = history + "\n" + glean.text
            if g == cfg.max_gleaning - 1:
                break
            check = self.llm.complete(lightrag_glean_check_request(text, history, types, cfg.max_extract_input_tokens))
            if check.text.strip().strip('"').strip("'").upper() != "YES":
                break
        return ex

    def extract(self, indices: Sequence[int]) -> int:
        """Extract (and cache) the graph records of the given units; returns #new."""
        todo = [i for i in dict.fromkeys(int(i) for i in indices) if self.pool[i].key not in self.extractions]
        if not todo:
            return 0
        with self.llm.meter.scope("index_build"):
            results = self.llm.map(lambda i: self._extract_one(self.pool[i]), todo,
                                   max_workers=self.config.max_workers)
        for i, ex in zip(todo, results):
            self.extractions[self.pool[i].key] = ex
        return len(todo)

    def _build_view(self, indices: Sequence[int]) -> _GraphView:
        self.extract(indices)
        g = KnowledgeGraph()
        for i in indices:
            key = self.pool[i].key
            g.add_extraction(key, self.extractions[key])
        with self.llm.meter.scope("index_build"):
            g.finalize(self.llm, self.config)
        names = list(g.entities)
        rkeys = list(g.relations)
        ent_texts = [f"{n}\n{g.entities[n].description}" for n in names]
        rel_texts = [f"{g.relations[k].keywords}\t{k[0]}\n{k[1]}\n{g.relations[k].description}" for k in rkeys]
        dim = int(getattr(self.embedder, "dim", 0))
        ev = np.asarray(self.embedder.encode(ent_texts), np.float32) if names else np.zeros((0, dim), np.float32)
        rv = np.asarray(self.embedder.encode(rel_texts), np.float32) if rkeys else np.zeros((0, dim), np.float32)
        return _GraphView(g, names, ev, rkeys, rv)

    def build(self, indices: Optional[Sequence[int]] = None) -> "LightRAGRetriever":
        """Offline indexing over the whole pool (or the given units)."""
        idx = list(range(len(self.pool))) if indices is None else list(indices)
        self.view = self._build_view(idx)
        log.info("LightRAG graph: %d entities, %d relations from %d units",
                 len(self.view.entity_names), len(self.view.relation_keys), len(idx))
        return self

    def ensure_built(self) -> float:
        """Build the global graph (or the candidate retriever's index in
        candidate-pool mode) if needed; seconds spent (0 if built)."""
        if self.per_query:
            return ensure_built(self.candidate_retriever)
        if self.view is not None:
            return 0.0
        t0 = time.perf_counter()
        self.build()
        return time.perf_counter() - t0

    # ------------------------------------------------------------ query
    def query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """One LLM call: (high-level, low-level) keywords."""
        try:
            obj, _ = self.llm.complete_json(lightrag_keywords_request(query), expect=dict)
        except JSONParseError:
            obj = {}
        hl, ll = parse_keywords(obj)
        if not hl and not ll:  # LightRAG would fail the query; fall back to the raw query text
            hl, ll = [query], [query]
        return hl, ll

    def _similar(self, vecs: np.ndarray, keywords: Sequence[str]) -> List[int]:
        if not keywords or vecs.shape[0] == 0:
            return []
        qv = np.asarray(self.embedder.encode_queries([", ".join(keywords)])[0], np.float32)
        sims = vecs @ qv
        return [int(i) for i in top_k_indices(sims, self.config.top_k, min_score=self.config.min_similarity)]

    def _local(self, view: _GraphView, ll: Sequence[str]) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
        g = view.graph
        entities = [view.entity_names[i] for i in self._similar(view.entity_vecs, ll)]
        # One-hop relations of the retrieved entities, by (edge degree, weight).
        edges: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for e in entities:
            for nb in sorted(g.adj.get(e, ())):
                k = edge_key(e, nb)
                if k in g.relations and k not in seen:
                    seen.add(k)
                    edges.append(k)
        edges.sort(key=lambda k: (-g.edge_degree(k), -g.relations[k].weight))
        # Text units of the entities, by (entity order, #one-hop neighbours sharing the unit).
        units: Dict[str, Tuple[int, int]] = {}
        for order, e in enumerate(entities):
            neighbour_sources = [set(g.entities[nb].sources) for nb in g.adj.get(e, ()) if nb in g.entities]
            for src in g.entities[e].sources:
                if src in units:
                    continue
                units[src] = (order, sum(1 for s in neighbour_sources if src in s))
        chunks = sorted(units, key=lambda s: (units[s][0], -units[s][1]))
        return entities, edges, chunks

    def _global(self, view: _GraphView, hl: Sequence[str]) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
        g = view.graph
        edges = [view.relation_keys[i] for i in self._similar(view.relation_vecs, hl)]
        edges.sort(key=lambda k: (-g.edge_degree(k), -g.relations[k].weight))
        entities: List[str] = []
        chunks: List[str] = []
        for k in edges:
            _add_unique(entities, [k[0], k[1]])
            _add_unique(chunks, g.relations[k].sources)
        return entities, edges, chunks

    def _papers_of(self, unit_keys: Sequence[str], unit_rank: Dict[str, int]) -> List[str]:
        """Up to ``max_sources_per_unit`` source papers, most query-relevant first.

        Source units are ordered by their rank in this query's text-unit
        ranking (``unit_rank``); unranked units follow in extraction order.
        """
        cap = int(self.config.max_sources_per_unit)
        if cap <= 0:
            return []
        n = len(unit_rank)
        order = sorted(range(len(unit_keys)), key=lambda j: (unit_rank.get(unit_keys[j], n), j))
        out: List[str] = []
        for j in order:
            p = self.pool.get(unit_keys[j])
            if p is not None:
                _add_unique(out, p.papers)
            if len(out) >= cap:
                break
        return out[:cap]

    def _entity_passage(self, g: KnowledgeGraph, name: str, unit_rank: Dict[str, int]) -> Passage:
        e = g.entities[name]
        return Passage(key=f"entity:{name}", kind=ENTITY, papers=self._papers_of(e.sources, unit_rank),
                       title=f"Entity: {name} ({e.type})", body=e.description.replace(SEP, " | "),
                       metadata={"degree": g.degree(name), "num_sources": len(e.sources)})

    def _relation_passage(self, g: KnowledgeGraph, k: Tuple[str, str], unit_rank: Dict[str, int]) -> Passage:
        r = g.relations[k]
        return Passage(key=f"relation:{k[0]}|{k[1]}", kind=RELATION, papers=self._papers_of(r.sources, unit_rank),
                       title=f"Relation: {k[0]} -- {k[1]} ({r.keywords})", body=r.description.replace(SEP, " | "),
                       metadata={"weight": r.weight, "edge_degree": g.edge_degree(k), "num_sources": len(r.sources)})

    def retrieve(self, query: QueryLike, budget: int) -> RetrievalResult:
        t0 = time.perf_counter()
        q = as_query(query)
        cfg = self.config
        extra: Dict[str, Any] = {}
        index_s = self.ensure_built()
        if self.per_query:
            papers = self.candidate_retriever.top_papers(q.text, int(cfg.candidate_papers or 0))
            tb = time.perf_counter()
            view = self._build_view(self.pool.indices_for_papers(papers))
            index_s += time.perf_counter() - tb
            extra["candidate_papers"] = len(papers)
        else:
            view = self.view
        assert view is not None
        hl, ll = self.query_keywords(q.text)
        l_ents, l_edges, l_chunks = self._local(view, ll)
        g_ents, g_edges, g_chunks = self._global(view, hl)
        entities = _round_robin(l_ents, g_ents)
        edges = _round_robin(l_edges, g_edges)
        chunk_keys = [k for k in _round_robin(l_chunks, g_chunks) if self.pool.get(k) is not None]
        unit_rank = {k: r for r, k in enumerate(chunk_keys)}
        g = view.graph
        packer = BudgetPacker(budget)
        packer.extend((Candidate(self._entity_passage(g, n, unit_rank), 0.0, "entity") for n in entities),
                      limit=int(cfg.entity_budget_fraction * budget))
        packer.extend((Candidate(self._relation_passage(g, k, unit_rank), 0.0, "relation") for k in edges),
                      limit=int(cfg.relation_budget_fraction * budget))
        packer.extend(Candidate(self.pool.get(k), 0.0, "text_unit") for k in chunk_keys)
        # Text units are LightRAG's citable sources: rank their papers first,
        # then papers known only through entity / relation descriptions.
        ranked: List[str] = []
        for u in packer.units:
            if u.kind not in (ENTITY, RELATION):
                _add_unique(ranked, u.papers)
        for u in packer.units:
            _add_unique(ranked, u.papers)
        stats = {
            "method": self.name,
            "high_level_keywords": hl,
            "low_level_keywords": ll,
            "entities": len(entities),
            "relations": len(edges),
            "text_units": len(chunk_keys),
            "graph_entities": len(view.entity_names),
            "graph_relations": len(view.relation_keys),
            "index_build_s": index_s,
            **extra,
        }
        return packer.result(q.query_id, titles=self.pool.titles, ranked_papers=ranked, stats=stats,
                             latency_s=max(0.0, time.perf_counter() - t0 - index_s))

    # --------------------------------------------------------------- io
    def save(self, directory: Union[str, Path]) -> None:
        """Persist the extraction cache and the built graph."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        ex = {k: asdict(v) for k, v in self.extractions.items()}
        (d / "lightrag_extractions.json").write_text(json.dumps(ex), encoding="utf-8")
        if self.view is not None:
            (d / "lightrag_graph.json").write_text(json.dumps(self.view.graph.to_dict()), encoding="utf-8")

    def load_extractions(self, directory: Union[str, Path]) -> "LightRAGRetriever":
        """Reload cached extractions (the graph is rebuilt by :meth:`build`)."""
        p = Path(directory) / "lightrag_extractions.json"
        for k, v in json.loads(p.read_text(encoding="utf-8")).items():
            self.extractions[k] = Extraction(**v)
        return self


__all__ = ["Entity", "Extraction", "KnowledgeGraph", "LightRAGConfig", "LightRAGRetriever", "Relation"]
