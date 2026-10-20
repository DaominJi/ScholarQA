"""Taxonomy data structure T = (V, E) over a paper repository.

Each node v is a research concept with a concise ``name`` and ``definition``;
each edge connects a broader concept to a more specific child (Section 2.1).
A paper that belongs to a concept also belongs to all of its ancestors, so
``node.papers`` stores the node's *extent* P_v (n_v = |P_v|), and
``node.residual`` stores the residual leaf (bottom_v): papers of v that match
none of v's children.

Node ids are short strings prefixed by the facet (``P`` problem, ``M``
method, ``T`` topic), e.g. ``"P0"`` for the root of a problem taxonomy.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

FACET_PREFIX = {"problem": "P", "method": "M", "topic": "T"}

# Node status values.
OPEN = "open"
CERTIFIED = "certified"  # loss bound U_v <= epsilon established by audit
UNRESOLVED = "unresolved"  # budget cap reached before certification
LEAF = "leaf"  # too small / too deep to split further


@dataclass
class TaxonomyNode:
    node_id: str
    name: str
    definition: str = ""
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    papers: Set[str] = field(default_factory=set)
    residual: Set[str] = field(default_factory=set)
    depth: int = 0
    status: str = OPEN
    bound: Optional[float] = None  # certified upper bound U_{v,j,a} on L_v
    delta: Optional[float] = None  # failure probability delta_v spent on v
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Concept description used for embeddings and contexts."""
        if self.definition:
            return f"{self.name}: {self.definition}"
        return self.name

    @property
    def size(self) -> int:
        return len(self.papers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "definition": self.definition,
            "parent": self.parent,
            "children": list(self.children),
            "papers": sorted(self.papers),
            "residual": sorted(self.residual),
            "depth": self.depth,
            "status": self.status,
            "bound": self.bound,
            "delta": self.delta,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaxonomyNode":
        return cls(
            node_id=str(d["node_id"]),
            name=str(d.get("name", "")),
            definition=str(d.get("definition", "") or ""),
            parent=d.get("parent"),
            children=[str(c) for c in d.get("children", [])],
            papers=set(map(str, d.get("papers", []))),
            residual=set(map(str, d.get("residual", []))),
            depth=int(d.get("depth", 0)),
            status=str(d.get("status", OPEN)),
            bound=d.get("bound"),
            delta=d.get("delta"),
            metadata=dict(d.get("metadata", {}) or {}),
        )


class Taxonomy:
    """A rooted concept tree with paper extents."""

    def __init__(
        self,
        facet: str = "problem",
        root_name: str = "Root",
        root_definition: str = "",
        root_papers: Optional[Iterable[str]] = None,
    ):
        self.facet = facet
        self.prefix = FACET_PREFIX.get(facet, facet[:1].upper() or "C")
        self._counter = 0
        self.nodes: Dict[str, TaxonomyNode] = {}
        self.metadata: Dict[str, Any] = {}
        self.root_id = self._new_id()
        self.nodes[self.root_id] = TaxonomyNode(
            node_id=self.root_id,
            name=root_name,
            definition=root_definition,
            papers=set(root_papers or []),
            depth=0,
        )
        self._paper_index: Optional[Dict[str, Set[str]]] = None

    # -------------------------------------------------------------- basics
    def _new_id(self) -> str:
        nid = f"{self.prefix}{self._counter}"
        self._counter += 1
        while nid in getattr(self, "nodes", {}):
            nid = f"{self.prefix}{self._counter}"
            self._counter += 1
        return nid

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes

    def __getitem__(self, node_id: str) -> TaxonomyNode:
        return self.nodes[node_id]

    @property
    def root(self) -> TaxonomyNode:
        return self.nodes[self.root_id]

    def invalidate(self) -> None:
        """Drop cached paper -> node index (call after bulk edits of extents)."""
        self._paper_index = None

    # ---------------------------------------------------------- structure
    def add_node(
        self,
        name: str,
        definition: str = "",
        parent: Optional[str] = None,
        papers: Optional[Iterable[str]] = None,
        node_id: Optional[str] = None,
        **metadata: Any,
    ) -> str:
        parent = parent or self.root_id
        if parent not in self.nodes:
            raise KeyError(f"Unknown parent {parent!r}")
        nid = node_id or self._new_id()
        if nid in self.nodes:
            raise KeyError(f"Duplicate node id {nid!r}")
        node = TaxonomyNode(
            node_id=nid,
            name=name,
            definition=definition,
            parent=parent,
            depth=self.nodes[parent].depth + 1,
            metadata=dict(metadata),
        )
        self.nodes[nid] = node
        self.nodes[parent].children.append(nid)
        if papers:
            for p in papers:
                self.assign(p, nid)
        self._paper_index = None
        return nid

    def remove_subtree(self, node_id: str) -> None:
        if node_id == self.root_id:
            raise ValueError("Cannot remove the root")
        parent = self.parent(node_id)
        for nid in list(self.descendants(node_id, include_self=True)):
            self.nodes.pop(nid, None)
        if parent is not None and parent in self.nodes:
            self.nodes[parent].children = [c for c in self.nodes[parent].children if c != node_id]
        self._paper_index = None

    def clear_children(self, node_id: str) -> None:
        for c in list(self.nodes[node_id].children):
            self.remove_subtree(c)
        self.nodes[node_id].children = []
        self.nodes[node_id].residual = set()

    def parent(self, node_id: str) -> Optional[str]:
        return self.nodes[node_id].parent

    def children(self, node_id: str) -> List[str]:
        return list(self.nodes[node_id].children)

    def is_leaf(self, node_id: str) -> bool:
        return not self.nodes[node_id].children

    def ancestors(self, node_id: str, include_self: bool = False) -> List[str]:
        """Ancestors from the parent up to the root (or from self if requested)."""
        out = [node_id] if include_self else []
        cur = self.nodes[node_id].parent
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent
        return out

    def path(self, node_id: str) -> List[str]:
        """Node ids from the root down to ``node_id``."""
        return list(reversed(self.ancestors(node_id, include_self=True)))

    def path_names(self, node_id: str) -> List[str]:
        return [self.nodes[n].name for n in self.path(node_id)]

    def descendants(self, node_id: str, include_self: bool = False) -> Iterator[str]:
        """Breadth-first iteration over the subtree rooted at ``node_id``."""
        queue = deque([node_id])
        first = True
        while queue:
            nid = queue.popleft()
            if include_self or not first:
                yield nid
            first = False
            queue.extend(self.nodes[nid].children)

    def depth(self, node_id: str) -> int:
        return self.nodes[node_id].depth

    @property
    def max_depth(self) -> int:
        return max((n.depth for n in self.nodes.values()), default=0)

    def leaves(self) -> List[str]:
        return [nid for nid, n in self.nodes.items() if not n.children]

    def internal_nodes(self) -> List[str]:
        return [nid for nid, n in self.nodes.items() if n.children]

    def edges(self) -> List[Tuple[str, str]]:
        """Parent -> child edges."""
        return [(n.parent, nid) for nid, n in self.nodes.items() if n.parent is not None]

    def bfs(self) -> List[str]:
        return list(self.descendants(self.root_id, include_self=True))

    def preorder(self, node_id: Optional[str] = None) -> List[str]:
        """Depth-first pre-order traversal (children in insertion order)."""
        out: List[str] = []
        stack = [node_id or self.root_id]
        while stack:
            nid = stack.pop()
            out.append(nid)
            stack.extend(reversed(self.nodes[nid].children))
        return out

    def direct_papers(self, node_id: str) -> Set[str]:
        """Papers attached directly at ``node_id`` (in its extent but in no child)."""
        n = self.nodes[node_id]
        if not n.children:
            return set(n.papers)
        below: Set[str] = set()
        for c in n.children:
            below |= self.nodes[c].papers
        return n.papers - below

    def attachments(self) -> Dict[str, List[str]]:
        """paper -> nodes where it is directly attached (pre-order)."""
        out: Dict[str, List[str]] = {}
        for nid in self.preorder():
            for p in sorted(self.direct_papers(nid)):
                out.setdefault(p, []).append(nid)
        return out

    def distance(self, a: str, b: str) -> int:
        """Number of edges on the tree path between ``a`` and ``b``."""
        pa = self.ancestors(a, include_self=True)
        pb = self.ancestors(b, include_self=True)
        depth_a = {n: i for i, n in enumerate(pa)}
        for j, n in enumerate(pb):
            if n in depth_a:
                return depth_a[n] + j
        return len(pa) + len(pb)

    def lowest_common_ancestor(self, a: str, b: str) -> str:
        pa = set(self.ancestors(a, include_self=True))
        for n in self.ancestors(b, include_self=True):
            if n in pa:
                return n
        return self.root_id

    # ------------------------------------------------------------ papers
    def assign(self, paper_id: str, node_id: str) -> None:
        """Add ``paper_id`` to ``node_id`` and all of its ancestors."""
        for nid in self.ancestors(node_id, include_self=True):
            self.nodes[nid].papers.add(paper_id)
        if self._paper_index is not None:
            for nid in self.ancestors(node_id, include_self=True):
                self._paper_index.setdefault(paper_id, set()).add(nid)

    def unassign_subtree(self, paper_id: str, node_id: str) -> None:
        """Remove ``paper_id`` from ``node_id`` and its descendants."""
        for nid in self.descendants(node_id, include_self=True):
            self.nodes[nid].papers.discard(paper_id)
            self.nodes[nid].residual.discard(paper_id)
        self._paper_index = None

    def extent(self, node_id: str) -> Set[str]:
        return self.nodes[node_id].papers

    def all_papers(self) -> Set[str]:
        return set(self.root.papers)

    def _build_paper_index(self) -> Dict[str, Set[str]]:
        idx: Dict[str, Set[str]] = {}
        for nid, node in self.nodes.items():
            for p in node.papers:
                idx.setdefault(p, set()).add(nid)
        self._paper_index = idx
        return idx

    def paper_nodes(self, paper_id: str) -> Set[str]:
        """All concepts whose extent contains ``paper_id`` (incl. ancestors)."""
        idx = self._paper_index if self._paper_index is not None else self._build_paper_index()
        return set(idx.get(paper_id, set()))

    def deepest_nodes(self, paper_id: str) -> List[str]:
        """Most specific concepts of a paper (no child also contains it)."""
        nodes = self.paper_nodes(paper_id)
        out = []
        for nid in nodes:
            if not any(c in nodes for c in self.nodes[nid].children):
                out.append(nid)
        return sorted(out, key=lambda n: (-self.nodes[n].depth, n))

    def leaf_assignment(self) -> Dict[str, List[str]]:
        """paper -> deepest concepts (used by clustering metrics)."""
        idx = self._paper_index if self._paper_index is not None else self._build_paper_index()
        return {p: self.deepest_nodes(p) for p in idx}

    # ----------------------------------------------------------- display
    def pretty(self, max_depth: Optional[int] = None, show_sizes: bool = True) -> str:
        lines: List[str] = []

        def rec(nid: str, indent: int) -> None:
            n = self.nodes[nid]
            if max_depth is not None and n.depth > max_depth:
                return
            size = f" ({n.size})" if show_sizes else ""
            flag = "" if n.status == OPEN else f" [{n.status}]"
            lines.append("  " * indent + f"- {n.name}{size}{flag}")
            for c in n.children:
                rec(c, indent + 1)

        rec(self.root_id, 0)
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        internal = self.internal_nodes()
        return {
            "facet": self.facet,
            "num_nodes": len(self.nodes),
            "num_internal": len(internal),
            "num_leaves": len(self.nodes) - len(internal),
            "max_depth": self.max_depth,
            "num_papers": len(self.root.papers),
            "num_certified": sum(1 for n in self.nodes.values() if n.status == CERTIFIED),
            "num_unresolved": sum(1 for n in self.nodes.values() if n.status == UNRESOLVED),
        }

    # ---------------------------------------------------------------- io
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facet": self.facet,
            "prefix": self.prefix,
            "root_id": self.root_id,
            "counter": self._counter,
            "metadata": self.metadata,
            "nodes": [self.nodes[nid].to_dict() for nid in self.bfs()],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Taxonomy":
        tax = cls.__new__(cls)
        tax.facet = d.get("facet", "problem")
        tax.prefix = d.get("prefix", FACET_PREFIX.get(tax.facet, "C"))
        tax.root_id = d["root_id"]
        tax._counter = int(d.get("counter", len(d["nodes"])))
        tax.metadata = dict(d.get("metadata", {}) or {})
        tax.nodes = {}
        for nd in d["nodes"]:
            node = TaxonomyNode.from_dict(nd)
            tax.nodes[node.node_id] = node
        tax._paper_index = None
        return tax

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Taxonomy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_nested(
        cls,
        tree: Dict[str, Any],
        facet: str = "topic",
        name_key: str = "name",
        children_key: str = "children",
        papers_key: str = "papers",
        definition_key: str = "definition",
    ) -> "Taxonomy":
        """Build from a nested ``{"name", "children": [...], "papers": [...]}`` dict.

        Papers listed at a node are assigned to that node (and its ancestors).
        Used to load gold taxonomies (TaxoBench, CHIME).
        """
        tax = cls(facet=facet, root_name=str(tree.get(name_key, "Root")),
                  root_definition=str(tree.get(definition_key, "") or ""))

        def rec(sub: Dict[str, Any], nid: str) -> None:
            for p in sub.get(papers_key, []) or []:
                tax.assign(str(p), nid)
            for ch in sub.get(children_key, []) or []:
                cid = tax.add_node(str(ch.get(name_key, "")), str(ch.get(definition_key, "") or ""), parent=nid)
                rec(ch, cid)

        rec(tree, tax.root_id)
        return tax

    def to_nested(self, node_id: Optional[str] = None, include_papers: bool = True) -> Dict[str, Any]:
        nid = node_id or self.root_id
        n = self.nodes[nid]
        out: Dict[str, Any] = {"id": nid, "name": n.name, "definition": n.definition}
        if include_papers:
            out["papers"] = sorted(n.papers - set().union(*[self.nodes[c].papers for c in n.children]) if n.children else n.papers)
        out["children"] = [self.to_nested(c, include_papers) for c in n.children]
        return out
