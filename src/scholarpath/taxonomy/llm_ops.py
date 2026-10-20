"""LLM operations on one taxonomy node, shared by Algorithm 1 and the baselines.

:class:`TaxonomyLLM` renders the prompts of :mod:`scholarpath.prompts.taxonomy`,
splits paper batches into calls of at most ``papers_per_call`` papers (run
concurrently), parses the JSON answers, and maps prompt-local child keys
(``C1``/``N1``) back to taxonomy node ids. It never mutates the taxonomy:
callers decide how to apply proposed children, refinements and labels.

Every call is billed to the cost component given by the caller
(``acquisition``, ``auditing``, ``discovery``, ``correction``, ``relabeling``
for ScholarPath; ``full`` / ``selection`` for the baselines).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from scholarpath.llm.base import LLMClient
from scholarpath.llm.parsing import JSONParseError
from scholarpath.prompts import taxonomy as P
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.text import truncate_tokens
from scholarpath.types import Paper

log = logging.getLogger(__name__)


@dataclass
class ChildSpec:
    """A child concept proposed by the LLM. ``node_id`` is set for existing children."""

    name: str
    definition: str = ""
    node_id: Optional[str] = None
    key: str = ""


@dataclass
class PaperLabel:
    """LLM label of a paper at node v.

    ``children`` holds node ids of existing children and ``"new:<k>"``
    placeholders for children proposed in the same call (resolved by the
    caller through :attr:`LabelBatch.new_children`). ``missing`` is True when
    the paper belongs to v but fits no child of the current schema (a
    *missing concept*, Section 3.1).
    """

    paper_id: str
    children: List[str] = field(default_factory=list)
    missing: bool = False

    @property
    def is_missing(self) -> bool:
        return self.missing or not self.children


@dataclass
class LabelBatch:
    labels: Dict[str, PaperLabel] = field(default_factory=dict)
    new_children: List[ChildSpec] = field(default_factory=list)  # index k <-> "new:k"
    refined: Dict[str, str] = field(default_factory=dict)  # node id -> new definition


class TaxonomyLLM:
    def __init__(
        self,
        llm: LLMClient,
        papers: Dict[str, Paper],
        facet: str = "problem",
        papers_per_call: int = 16,
        abstract_tokens: int = 320,
        min_children: int = 2,
        max_children: int = 8,
    ):
        self.llm = llm
        self.papers = papers
        self.facet = facet
        self.papers_per_call = max(1, int(papers_per_call))
        self.abstract_tokens = int(abstract_tokens)
        self.min_children = int(min_children)
        self.max_children = int(max_children)
        self.papers_read: Dict[str, int] = {}  # component -> #paper reads

    # ------------------------------------------------------------ helpers
    def node_context(self, tax: Taxonomy, node_id: str) -> Dict[str, object]:
        n = tax.nodes[node_id]
        return {"id": node_id, "name": n.name, "definition": n.definition, "path": tax.path_names(node_id)}

    def paper_brief(self, pid: str) -> Dict[str, str]:
        p = self.papers[pid]
        return {"id": pid, "title": p.title, "abstract": truncate_tokens(p.abstract or "", self.abstract_tokens)}

    def children_context(self, tax: Taxonomy, child_ids: Sequence[str]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        ctx, key_to_id = [], {}
        for i, cid in enumerate(child_ids, 1):
            key = f"C{i}"
            n = tax.nodes[cid]
            ctx.append({"key": key, "name": n.name, "definition": n.definition})
            key_to_id[key] = cid
        return ctx, key_to_id

    def _batches(self, paper_ids: Sequence[str]) -> List[List[str]]:
        ids = list(paper_ids)
        return [ids[i : i + self.papers_per_call] for i in range(0, len(ids), self.papers_per_call)]

    def _count(self, component: Optional[str], n: int) -> None:
        key = component or "default"
        self.papers_read[key] = self.papers_read.get(key, 0) + n

    @staticmethod
    def _parse_labels(
        raw: object, key_to_id: Dict[str, str], new_offset: int, valid: Set[str]
    ) -> Dict[str, PaperLabel]:
        out: Dict[str, PaperLabel] = {}
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("paper_id", item.get("id", "")))
            if pid not in valid:
                continue
            keys = item.get("children") or []
            if isinstance(keys, str):
                keys = [keys]
            kids: List[str] = []
            for k in keys:
                k = str(k).strip()
                if k in key_to_id:
                    kids.append(key_to_id[k])
                elif k.upper().startswith("N") and k[1:].isdigit():
                    kids.append(f"new:{new_offset + int(k[1:]) - 1}")
            kids = list(dict.fromkeys(kids))
            missing = bool(item.get("missing_concept", False)) or not kids
            out[pid] = PaperLabel(pid, kids, missing)
        return out

    @staticmethod
    def _parse_new(raw: object) -> List[Tuple[str, ChildSpec]]:
        out = []
        if not isinstance(raw, list):
            return out
        for c in raw:
            if isinstance(c, dict) and c.get("name"):
                out.append((str(c.get("key", "")), ChildSpec(str(c["name"]).strip(), str(c.get("definition", "")).strip())))
        return out

    # --------------------------------------------------------------- ops
    def propose_children(
        self,
        tax: Taxonomy,
        node_id: str,
        paper_ids: Sequence[str],
        existing: Sequence[str] = (),
        component: Optional[str] = None,
    ) -> Tuple[List[ChildSpec], Dict[str, PaperLabel]]:
        """Propose (or refine) the schema of ``node_id`` from the given papers.

        Papers beyond ``papers_per_call`` are processed in successive calls:
        the first call proposes the schema, later calls refine it with the
        schema proposed so far as ``existing`` children. Returns the final
        children (existing ones carry ``node_id``) and labels of every paper
        read (child placeholders ``"new:k"`` index the returned list).
        """
        node = self.node_context(tax, node_id)
        schema: List[ChildSpec] = [
            ChildSpec(tax.nodes[c].name, tax.nodes[c].definition, node_id=c) for c in existing
        ]
        labels: Dict[str, PaperLabel] = {}
        for batch in self._batches(paper_ids):
            ctx = [{"key": f"C{i}", "name": c.name, "definition": c.definition} for i, c in enumerate(schema, 1)]
            req = P.propose_children_request(
                self.facet, node, [self.paper_brief(p) for p in batch], ctx,
                self.min_children, self.max_children, component,
            )
            try:
                obj, _ = self.llm.complete_json(req)
            except JSONParseError as err:
                log.warning("propose_children failed at %s: %s", node_id, err)
                continue
            self._count(component, len(batch))
            key_to_idx = {f"C{i}": i - 1 for i in range(1, len(schema) + 1)}
            new_schema: List[ChildSpec] = []
            key_map: Dict[str, int] = {}
            for c in obj.get("children", []) or []:
                if not isinstance(c, dict) or not c.get("name"):
                    continue
                key = str(c.get("key", "")).strip()
                name = str(c["name"]).strip()
                definition = str(c.get("definition", "")).strip()
                if key in key_to_idx:
                    old = schema[key_to_idx[key]]
                    spec = ChildSpec(old.name, definition or old.definition, old.node_id, key)
                else:
                    spec = ChildSpec(name, definition, None, key)
                key_map[key] = len(new_schema)
                new_schema.append(spec)
            # Children that already exist in the taxonomy are never dropped.
            for i, old in enumerate(schema):
                if old.node_id is not None and f"C{i + 1}" not in key_map:
                    key_map[f"C{i + 1}"] = len(new_schema)
                    new_schema.append(old)
            if not new_schema:
                continue
            # Re-key earlier labels to indices of the new schema; labels of
            # children the LLM dropped are removed (never aliased).
            remap = {f"new:{key_to_idx[k]}": f"new:{v}" for k, v in key_map.items() if k in key_to_idx}
            for lab in labels.values():
                lab.children = [remap[c] for c in lab.children if c in remap]
                lab.missing = not lab.children
            schema = new_schema
            ktoi = {k: f"new:{v}" for k, v in key_map.items()}
            for item in obj.get("labels", []) or []:
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("paper_id", ""))
                if pid not in batch:
                    continue
                kids = [ktoi[str(k)] for k in (item.get("children") or []) if str(k) in ktoi]
                labels[pid] = PaperLabel(pid, list(dict.fromkeys(kids)), not kids)
        # Express labels of existing children by node id.
        for lab in labels.values():
            lab.children = [
                schema[int(c[4:])].node_id if schema[int(c[4:])].node_id else c for c in lab.children
            ]
        return schema, labels

    def label(
        self,
        tax: Taxonomy,
        node_id: str,
        child_ids: Sequence[str],
        paper_ids: Sequence[str],
        component: Optional[str] = None,
        allow_new: bool = False,
    ) -> LabelBatch:
        """Label papers with the existing children of ``node_id`` (multi-label)."""
        node = self.node_context(tax, node_id)
        ctx, key_to_id = self.children_context(tax, child_ids)
        batches = self._batches(paper_ids)

        def run(batch: List[str]):
            req = P.label_papers_request(
                self.facet, node, ctx, [self.paper_brief(p) for p in batch], allow_new, component
            )
            try:
                obj, _ = self.llm.complete_json(req)
            except JSONParseError as err:
                log.warning("label_papers failed at %s: %s", node_id, err)
                return batch, None
            return batch, obj

        out = LabelBatch()
        for batch, obj in self.llm.map(run, batches):
            if obj is None:
                continue
            self._count(component, len(batch))
            offset = len(out.new_children)
            new = self._parse_new(obj.get("new_children")) if allow_new else []
            out.new_children.extend(spec for _, spec in new)
            out.labels.update(self._parse_labels(obj.get("labels"), key_to_id, offset, set(batch)))
        return out

    def discover(
        self,
        tax: Taxonomy,
        node_id: str,
        child_ids: Sequence[str],
        paper_ids: Sequence[str],
        component: Optional[str] = "discovery",
    ) -> LabelBatch:
        """DISCOVER: assign / refine / add children for residual or flagged papers."""
        node = self.node_context(tax, node_id)
        ctx, key_to_id = self.children_context(tax, child_ids)
        out = LabelBatch()
        for batch in self._batches(paper_ids):
            # Children discovered by earlier batches are offered as existing ones.
            ctx_b = list(ctx) + [
                {"key": f"C{len(ctx) + i + 1}", "name": c.name, "definition": c.definition}
                for i, c in enumerate(out.new_children)
            ]
            k2i = dict(key_to_id)
            for i in range(len(out.new_children)):
                k2i[f"C{len(ctx) + i + 1}"] = f"new:{i}"
            req = P.discover_request(self.facet, node, ctx_b, [self.paper_brief(p) for p in batch], component)
            try:
                obj, _ = self.llm.complete_json(req)
            except JSONParseError as err:
                log.warning("discover failed at %s: %s", node_id, err)
                continue
            self._count(component, len(batch))
            offset = len(out.new_children)
            out.new_children.extend(spec for _, spec in self._parse_new(obj.get("new_children")))
            for r in obj.get("refined", []) or []:
                if isinstance(r, dict) and str(r.get("key")) in key_to_id and r.get("definition"):
                    out.refined[key_to_id[str(r["key"])]] = str(r["definition"]).strip()
            out.labels.update(self._parse_labels(obj.get("labels"), k2i, offset, set(batch)))
        return out


def resolve_new_children(
    tax: Taxonomy,
    parent_id: str,
    specs: Sequence[ChildSpec],
    labels: Dict[str, PaperLabel],
    max_children: Optional[int] = None,
) -> Dict[str, str]:
    """Add proposed children under ``parent_id`` and rewrite ``"new:k"`` labels.

    A proposed child whose normalized name equals an existing child's is
    merged into it. Returns the mapping placeholder -> node id. Children
    beyond ``max_children`` are not created (their labels are dropped).
    """
    existing = {tax.nodes[c].name.strip().lower(): c for c in tax.children(parent_id)}
    mapping: Dict[str, str] = {}
    for k, spec in enumerate(specs):
        if spec.node_id is not None:
            mapping[f"new:{k}"] = spec.node_id
            if spec.definition and spec.definition != tax.nodes[spec.node_id].definition:
                tax.nodes[spec.node_id].definition = spec.definition
            continue
        norm = spec.name.strip().lower()
        if norm in existing:
            mapping[f"new:{k}"] = existing[norm]
            continue
        if max_children is not None and len(tax.children(parent_id)) >= max_children:
            continue
        nid = tax.add_node(spec.name, spec.definition, parent=parent_id)
        existing[norm] = nid
        mapping[f"new:{k}"] = nid
    for lab in labels.values():
        kids = []
        for c in lab.children:
            if c.startswith("new:"):
                if c in mapping:
                    kids.append(mapping[c])
            else:
                kids.append(c)
        lab.children = list(dict.fromkeys(kids))
        lab.missing = lab.missing and not lab.children
    return mapping
