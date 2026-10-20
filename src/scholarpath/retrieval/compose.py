"""Step 6 of Algorithm 2: context assembly and answer synthesis (Section 4.1, line 25).

Once the budget is exhausted or no entry offers a positive gain, the distinct
nodes V(S) of the selected units are merged into ONE hierarchical context in
which shared concepts and papers appear only once, and a single LLM call
(``prompts.answer``, component ``composition``) composes the cited answer.

Layout of the context::

    # Established findings (direct support paths)
    ### [P] Drug discovery: ...                <- concept headings, level = taxonomy depth
    #### [P] Drug-target interaction: ...
    - [p2] Title. Abstract                     <- paper summaries
      - [p2 §Method] section chunk             <- paper details nested under their paper
    # Hypotheses (complementary path sets)
    ## Hypothesis: <candidate> for <known endpoint>
    - Support set: bridge <f> -> <candidate>; problem-side [p4]; method-side [p1 §Method]
    ...nodes of e_C, e_P and e_M not shown before...

Every rendered line is a pure function of its node, so its token count is
used as the node cost c(x) during the search (:class:`ContextRenderer`); the
assembled context therefore never exceeds the budget B. The costs of index
nodes, and their subtree minima used by the marginal-cost lower bounds
c_low(H | S) of Section 4.3, do not depend on the query and are kept in a
:class:`NodeCostTable` shared by all queries over the same index.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from scholarpath.config import RetrievalConfig
from scholarpath.index.semantic_index import SemanticIndex, concept_key, paper_key
from scholarpath.llm import JSONParseError, LLMClient, sum_usage
from scholarpath.prompts.answer import compose_answer_request, parse_answer
from scholarpath.retrieval.result import ESTABLISHED, HYPOTHESIS, RetrievalResult
from scholarpath.retrieval.units import (
    HYP_PREFIX,
    SET_PREFIX,
    ComplementaryPathSet,
    DirectPath,
    is_anchor,
    is_virtual,
    key_id,
    key_level,
)
from scholarpath.text import count_tokens
from scholarpath.types import EVIDENCE, Answer, Query

HEADER_EVIDENCE = "# Established findings (direct support paths)"
HEADER_HYPOTHESES = "# Hypotheses (complementary path sets: bridge concept path, problem-side and method-side premises)"


def _one_line(text: str) -> str:
    return " ".join((text or "").split())


def _index_line(index: SemanticIndex, key: str) -> str:
    """Context line of an index node (a pure function of the node)."""
    level, raw = key.split(":", 1)
    if level == "concept":
        node = index.concept(raw)
        tag = "P" if index.facet_of(raw) == "problem" else "M"
        body = f"{node.name}: {_one_line(node.definition)}" if node.definition else node.name
        return f"{'#' * min(6, node.depth + 2)} [{tag}] {body}"
    if level == "paper":
        p = index.papers[raw]
        body = f"{_one_line(p.title)}. {_one_line(p.abstract)}" if p.abstract else _one_line(p.title)
        return f"- [{raw}] {body}"
    ch = index.chunks[raw]
    return f"  - [{ch.paper_id} §{_one_line(ch.heading)}] {_one_line(ch.text)}"


class NodeCostTable:
    """Query-independent token costs c(x) of index nodes and their subtree minima.

    ``cost(key)`` is the token count of the node's context line plus the line
    break. ``min_paper(facet, v)`` / ``min_concept(facet, v)`` are the smallest
    abstract / concept costs in the subtree of v: the "smallest terminal-node
    token cost in H" that Section 4.3 uses for the marginal-cost lower bound
    c_low(H | S). All values are memoized for the lifetime of the table, so a
    retriever that keeps one table per index prices every node at most once
    across queries.
    """

    def __init__(self, index: SemanticIndex):
        self.index = index
        self._cost: Dict[str, int] = {}
        self._min_paper: Dict[Tuple[str, str], int] = {}
        self._min_concept: Dict[Tuple[str, str], int] = {}

    def cost(self, key: str) -> int:
        c = self._cost.get(key)
        if c is None:
            c = max(1, count_tokens(_index_line(self.index, key) + "\n"))
            self._cost[key] = c
        return c

    def priced(self, level: Optional[str] = None) -> int:
        """Number of index nodes priced so far (of one level: concept | paper | chunk)."""
        if level is None:
            return len(self._cost)
        return sum(1 for k in self._cost if key_level(k) == level)

    def min_paper(self, facet: str, node: str) -> int:
        """Smallest abstract cost among the papers of extent(v) (0 for an empty extent)."""
        k = (facet, node)
        hit = self._min_paper.get(k)
        if hit is None:
            hit = min((self.cost(paper_key(p)) for p in self.index.taxonomy(facet).nodes[node].papers), default=0)
            self._min_paper[k] = hit
        return hit

    def min_concept(self, facet: str, node: str) -> int:
        """Smallest concept-line cost in the subtree of v (bottom-up, memoized per node)."""
        k = (facet, node)
        hit = self._min_concept.get(k)
        if hit is None:
            tax = self.index.taxonomy(facet)
            order = [v for v in tax.descendants(node, include_self=True) if (facet, v) not in self._min_concept]
            for v in reversed(order):  # children before parents
                kids = [self._min_concept[(facet, c)] for c in tax.children(v)]
                self._min_concept[(facet, v)] = min([self.cost(concept_key(v))] + kids)
            hit = self._min_concept[k]
        return hit


class ContextRenderer:
    """Renders nodes as context lines and prices them: c(x) = tokens of the line of x.

    Index nodes are priced through a (shareable) :class:`NodeCostTable`;
    virtual nodes (hypothesis headers and support-set lines), the display
    labels of query-local anchors and the context notes belong to one query.
    """

    def __init__(self, index: SemanticIndex, costs: Optional[NodeCostTable] = None):
        self.index = index
        self.costs = costs if costs is not None and costs.index is index else NodeCostTable(index)
        self._virtual: Dict[str, str] = {}
        self._vcost: Dict[str, int] = {}
        self._labels: Dict[str, str] = {}
        self.notes: List[str] = []

    # ------------------------------------------------------------- lines
    def register_label(self, node_id: str, label: str) -> None:
        """Display name of a query-local anchor (it has no taxonomy node)."""
        self._labels[node_id] = label

    def concept_name(self, node_id: str) -> str:
        label = self._labels.get(node_id)
        return label if label is not None else self.index.concept(node_id).name

    def add_note(self, text: str) -> None:
        """A note line printed above the context (its tokens are reserved by :meth:`header_overhead`)."""
        line = f"Note: {_one_line(text)}"
        if line not in self.notes:
            self.notes.append(line)

    def register(self, key: str, line: str) -> None:
        self._virtual[key] = line

    def register_hypothesis(self, candidate: str, known_label: str) -> str:
        """Register the header line of a hypothesis (virtual node ``hyp:<candidate>``)."""
        key = f"{HYP_PREFIX}{candidate}"
        if key not in self._virtual:
            self._virtual[key] = f"## Hypothesis: {self.concept_name(candidate)} for {known_label}"
        return key

    def register_set(self, b: ComplementaryPathSet) -> None:
        """Register the two context-label lines of a path set (its virtual nodes)."""
        self.register_hypothesis(b.candidate, b.known_label)
        set_key = b.virtual_nodes[1]
        if set_key not in self._virtual:
            chain = " -> ".join(self.concept_name(key_id(k)) for k in b.e_C.nodes)
            if is_anchor(b.candidate):  # e_C ends at the neighbour the query concept is attached to
                chain += f" ~ {self.concept_name(b.candidate)} (query concept)"
            self._virtual[set_key] = (
                f"- Support set for {self.concept_name(b.candidate)}: bridge {chain}; "
                f"problem-side {self.ref(b.e_P.end)}; method-side {self.ref(b.e_M.end)}"
            )

    def ref(self, key: str) -> str:
        level = key_level(key)
        if level == "paper":
            return f"[{key_id(key)}]"
        if level == "chunk":
            ch = self.index.chunks[key_id(key)]
            return f"[{ch.paper_id} §{_one_line(ch.heading)}]"
        return self.concept_name(key_id(key))

    def line(self, key: str) -> str:
        if is_virtual(key):
            line = self._virtual.get(key)
            if line is None:
                raise KeyError(f"Context line of virtual node {key!r} was not registered")
            return line
        return _index_line(self.index, key)

    def cost(self, key: str) -> int:
        """Token cost c(x) of a node: its rendered line plus the line break."""
        if not is_virtual(key):
            return self.costs.cost(key)
        c = self._vcost.get(key)
        if c is None:
            c = max(1, count_tokens(self.line(key) + "\n"))
            self._vcost[key] = c
        return c

    def header_overhead(self) -> int:
        """Tokens of the section headers and notes (reserved from the budget before the search)."""
        lines = [HEADER_EVIDENCE, HEADER_HYPOTHESES] + list(self.notes)
        return sum(count_tokens(line + "\n") for line in lines)


def _render_forest(
    paths: Iterable[Sequence[str]], placed: Dict[str, None], renderer: ContextRenderer
) -> List[str]:
    """Render the not-yet-placed nodes of downward paths as one forest (each node once).

    A concept nests under its nearest taxonomy ancestor shown in the same
    forest, a paper under the concept preceding it on its first path, and a
    chunk under its paper; siblings keep their order of first appearance.
    """
    order: Dict[str, None] = {}
    pred: Dict[str, Optional[str]] = {}
    for nodes in paths:
        prev: Optional[str] = None
        for key in nodes:
            if is_virtual(key):
                continue
            if key not in placed and key not in order:
                order[key] = None
                pred[key] = prev
            prev = key
    index = renderer.index
    parent: Dict[str, Optional[str]] = {}
    for key in order:
        level = key_level(key)
        par: Optional[str] = None
        if level == "concept":
            nid = key_id(key)
            tax = index.taxonomy(index.facet_of(nid))
            for anc in tax.ancestors(nid):
                if f"concept:{anc}" in order:
                    par = f"concept:{anc}"
                    break
        elif level == "chunk":
            par = f"paper:{index.chunks[key_id(key)].paper_id}"
        else:
            par = pred.get(key)
        parent[key] = par if par in order else None
    children: Dict[Optional[str], List[str]] = {}
    for key in order:
        children.setdefault(parent[key], []).append(key)
    out: List[str] = []
    stack = list(reversed(children.get(None, [])))
    while stack:
        key = stack.pop()
        placed[key] = None
        out.append(renderer.line(key))
        stack.extend(reversed(children.get(key, [])))
    return out


def assemble_context(
    index: SemanticIndex,
    units: Sequence[object],
    renderer: Optional[ContextRenderer] = None,
    candidate_order: Optional[Sequence[str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Merge V(S) of the selected units into one hierarchical context (Algorithm 2, line 25).

    Returns ``(context, sources)`` where ``sources`` lists the papers of the
    context as ``{"paper_id", "title", "status", "candidate"}`` records for
    :func:`scholarpath.prompts.answer.compose_answer_request`: one record per
    (premise paper, candidate) pair of the hypothesis sets, in the order of
    ``candidate_order``, so that a paper supporting several candidates is
    attributed to each of them, followed by one ``established`` record per
    paper of the direct paths. The renderer's notes (e.g. a query concept that
    matches nothing in I) open the context when any unit was selected.
    """
    renderer = renderer or ContextRenderer(index)
    direct = [u for u in units if isinstance(u, DirectPath)]
    sets = [u for u in units if isinstance(u, ComplementaryPathSet)]
    for b in sets:
        renderer.register_set(b)
    placed: Dict[str, None] = {}
    lines: List[str] = list(renderer.notes) if units else []
    if direct:
        lines.append(HEADER_EVIDENCE)
        lines.extend(_render_forest((u.nodes for u in direct), placed, renderer))
    if sets:
        lines.append(HEADER_HYPOTHESES)
        order: List[str] = []
        for c in list(candidate_order or []) + [b.candidate for b in sets]:
            if c not in order and any(b.candidate == c for b in sets):
                order.append(c)
        for c in order:
            group = [b for b in sets if b.candidate == c]
            lines.append(renderer.line(f"{HYP_PREFIX}{c}"))
            for b in group:
                lines.append(renderer.line(f"{SET_PREFIX}{b.uid}"))
                paths = (b.known_path.nodes, b.e_C.nodes, b.candidate_path.nodes)
                lines.extend(_render_forest(paths, placed, renderer))
    context = "\n".join(lines)

    sources: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}

    def add(pid: Optional[str], status: str, candidate: Optional[str]) -> None:
        if pid is not None and (pid, candidate) not in sources:
            sources[(pid, candidate)] = {"paper_id": pid, "title": index.papers[pid].title, "status": status,
                                         "candidate": candidate}

    if sets:
        rank = {c: i for i, c in enumerate(candidate_order or [])}
        for b in sorted(sets, key=lambda s: rank.get(s.candidate, len(rank))):  # stable: selection order
            name = renderer.concept_name(b.candidate)
            for pid in b.papers:
                add(pid, HYPOTHESIS, name)
    for u in direct:
        add(u.paper_id, ESTABLISHED, None)
    return context, list(sources.values())


def compose_answer(
    query: Query,
    retrieval: RetrievalResult,
    llm: LLMClient,
    config: Optional[RetrievalConfig] = None,
) -> Answer:
    """Single composition call over the assembled context (Algorithm 2, line 25)."""
    config = config or RetrievalConfig()
    kind = retrieval.stats.get("kind") or query.kind or EVIDENCE
    request = compose_answer_request(
        query.text, retrieval.context, kind, retrieval.sources, max_tokens=config.answer_max_tokens
    )
    before = llm.meter.snapshot()
    t0 = time.perf_counter()
    try:
        obj, _ = llm.complete_json(request, expect=dict)
    except JSONParseError:  # pragma: no cover - depends on the backend
        obj = {"answer": "", "items": []}
    elapsed = time.perf_counter() - t0
    text, items = parse_answer(obj, valid_ids=[s["paper_id"] for s in retrieval.sources])
    scores = retrieval.stats.get("candidate_scores", {}) or {}
    for it in items:
        if it.candidate and it.candidate in scores:
            it.score = float(scores[it.candidate])
    usage = sum_usage(llm.meter.diff(before))
    return Answer(
        query_id=query.query_id,
        text=text,
        items=items,
        retrieved_papers=list(retrieval.ranked_papers),
        context_tokens=retrieval.context_tokens,
        llm_calls=usage.calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
        latency_s=retrieval.latency_s + elapsed,
        metadata={"kind": kind, "candidates": list(retrieval.candidates), "system": "scholarpath"},
    )
