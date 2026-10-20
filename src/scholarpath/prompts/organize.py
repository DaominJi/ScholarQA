"""Hierarchy composition prompt for the organization benchmarks (Section 5.3).

On TaxoBench (Deep Research mode) and CHIME, a system receives only a topic,
retrieves papers, and must organize them into a hierarchy. Every *budgeted*
method (ScholarPath, keyword search, dense retrieval, LightRAG, ...) packs at
most B context tokens (Definition 2) and then invokes the backbone LLM in a
*single* composition call built by :func:`organize_request`, exactly like the
answer composition of Algorithm 2, Step 6 (:mod:`scholarpath.prompts.answer`);
only the requested output differs. The output is a JSON topic hierarchy::

    {"name": "<root topic>",
     "children": [{"name": "<category>", "children": [...], "papers": ["<paper_id>", ...]}],
     "papers": []}

whose categories cite paper ids from the context. :func:`parse_hierarchy`
turns that output into a foundation :class:`~scholarpath.taxonomy.tree.Taxonomy`
(facet ``"topic"``) that the metrics in
:mod:`scholarpath.evaluation.organization_metrics` score against the expert
hierarchy (TaxoBench: ARI, Sem; CHIME: Cat, Assn).

``MOCK_HANDLERS`` answers ``organize.hierarchy`` offline by grouping the
provided sources on shared title keywords (deterministic, no network).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from scholarpath.llm.base import LLMRequest, chat
from scholarpath.text import tokenize

if TYPE_CHECKING:  # pragma: no cover - imported lazily at runtime
    from scholarpath.llm.base import LLMClient
    from scholarpath.taxonomy.tree import Taxonomy

TASK = "organize.hierarchy"

SYSTEM = (
    "You are an expert research librarian who organizes the literature on a topic into a hierarchical "
    "taxonomy, the way a survey or a systematic review structures the works it cites. You use ONLY the "
    "papers that appear in the provided context, cite them by their ids, and answer with a single JSON "
    "object and nothing else."
)

#: Benchmark-specific framing of the hierarchy (TaxoBench surveys vs. CHIME reviews).
STYLES: Dict[str, str] = {
    "survey": (
        "Organize the papers into a TOPIC HIERARCHY like the taxonomy section of a survey on this topic: "
        "top-level categories are the main research themes or families of approaches, and subcategories "
        "refine them into specific problems, techniques, or settings."
    ),
    "review": (
        "Organize the studies into a CATEGORY HIERARCHY like a systematic review on this topic: top-level "
        "categories are the main aspects the studies address (e.g. interventions, populations, outcomes, or "
        "study designs), and subcategories refine them into more specific groups of studies."
    ),
}

_RULES = (
    "Rules:\n"
    "1. Use only papers from the context and cite each one by its exact id as written in the context.\n"
    "2. Include every paper of the context that is relevant to the topic; leave out unrelated papers.\n"
    "3. Place each paper in the single MOST SPECIFIC category that fits it. List a paper under more than one "
    "category only if it clearly and substantially belongs to several of them.\n"
    "4. A paper listed in a category implicitly belongs to all ancestors of that category: do not repeat it "
    "at the ancestors.\n"
    "5. Give every category a concise, descriptive name (2-8 words), without numbering and without using a "
    "paper title as a name; avoid catch-all categories such as \"Other\" unless unavoidable.\n"
    "6. Prefer 2-8 children per category and 2-4 levels below the root; every category must contain at "
    "least one paper in its subtree."
)

_FORMAT = (
    "Answer JSON:\n"
    '{"name": "<root topic>",\n'
    ' "children": [{"name": "<category>",\n'
    '               "children": [{"name": "<subcategory>", "children": [], "papers": ["<paper_id>", ...]}],\n'
    '               "papers": []}],\n'
    ' "papers": []}'
)


def organize_request(
    query: str,
    context: str,
    sources: Sequence[Dict[str, Any]] = (),
    max_tokens: int = 2048,
    component: Optional[str] = "composition",
    style: str = "survey",
    list_sources: bool = False,
) -> LLMRequest:
    """Build the single hierarchy-composition call (Section 5.3, organization benchmarks).

    ``context`` is the (at most B-token) context assembled by the retriever;
    ``sources`` lists the citable papers present in it
    (``{"paper_id", "title", ...}``) and is passed in the payload for the
    offline mock. With ``list_sources`` the prompt additionally enumerates
    ``[paper_id] title`` for every source (off by default so that the context
    remains the only evidence, as for :func:`~scholarpath.prompts.answer.compose_answer_request`).
    ``style`` is ``"survey"`` (TaxoBench) or ``"review"`` (CHIME).
    """
    if style not in STYLES:
        raise ValueError(f"Unknown organization style {style!r}; expected one of {sorted(STYLES)}.")
    guide = STYLES[style]
    ctx = context.strip() or "(no retrieved context)"
    listing = ""
    if list_sources and sources:
        rows = [f"[{s['paper_id']}] {s.get('title', '')}".rstrip() for s in sources if s.get("paper_id")]
        listing = "\n\nPapers available for placement:\n" + "\n".join(rows)
    user = f"Topic: {query}\n\n{guide}\n\n{_RULES}\n\nContext:\n{ctx}{listing}\n\n{_FORMAT}"
    payload = {
        "query": query,
        "style": style,
        "sources": [
            {"paper_id": str(s["paper_id"]), "title": str(s.get("title", "") or "")}
            for s in sources
            if s.get("paper_id") is not None
        ],
        "has_context": bool(context.strip()),
    }
    return LLMRequest(
        task=TASK,
        messages=chat(SYSTEM, user),
        payload=payload,
        component=component,
        max_tokens=max_tokens,
        json_mode=True,
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_WRAPPER_KEYS = ("hierarchy", "hierarchy_tree", "taxonomy", "tree", "root", "result", "output")
_NAME_KEYS = ("name", "title", "category", "label", "topic", "heading")
#: Name keys that denote a category (a dict with only a "title" and a paper id is a paper reference).
_CATEGORY_NAME_KEYS = ("name", "category", "label", "topic", "heading")
_CHILD_KEYS = ("children", "subtopics", "subcategories", "categories", "subtrees", "nodes", "topics")
_PAPER_KEYS = ("papers", "paper_ids", "paper_id", "citations", "ids", "studies", "references", "sources")
_PAPER_LIST_KEYS = tuple(k for k in _PAPER_KEYS if k != "paper_id")
_PAPER_ID_KEYS = ("paper_id", "id", "pid", "paperId")
_PAPER_REF_KEYS = ("paper_id", "paperId", "pid")
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_BULLET_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>[-*+•]|\d+(?:\.\d+)*[.)]?)\s+(?P<text>.*\S)\s*$")
_HEADING_RE = re.compile(r"^\s*(?P<hashes>#{1,6})\s+(?P<text>.*\S)\s*$")
_ID_ONLY_PREFIX_RE = re.compile(r"^\s*\[[^\[\]]+\]")
#: Fenced code blocks, also an unclosed one at the end of a truncated reply.
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n?(.*?)(?:```|\Z)", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_OPENER_RE = re.compile(r"[\[{]")
#: Candidate paper-id tokens inside a free-text citation ("tb-1, tb-2", "tb-1: Some title", "(arxiv:2101.00001)").
_ID_TOKEN_RE = re.compile(r"[A-Za-z0-9_][\w.:/#-]*")
_ID_EDGE_CHARS = "[](){}<>'\"`*.,;:/#- \t"
#: Bounds of the JSON recovery search (replies are at most a few thousand tokens).
_MAX_OPENERS = 256
_MAX_REPAIRS = 64
_MAX_BRACKET_FIXES = 8


def _get(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    lower = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        v = lower.get(k.lower())
        if v not in (None, ""):
            return v
    return None


def _nonempty(value: Any) -> bool:
    return value is not None and not (isinstance(value, (list, tuple, dict, str)) and not value)


def _bracket_ids(text: str) -> List[str]:
    """Ids cited in square brackets: ``"see [p1, p2] and [p3]"`` -> ``["p1", "p2", "p3"]``."""
    return [t.strip() for b in _BRACKET_RE.findall(text or "") for t in b.split(",") if t.strip()]


def _paper_tokens(value: Any) -> List[str]:
    """Paper ids cited by one ``papers`` entry (string, list, dict, number), in order.

    Iterative, so arbitrarily nested model output cannot exhaust the stack.
    """
    out: List[str] = []
    stack: List[Any] = [value]
    while stack:
        v = stack.pop()
        if v is None or isinstance(v, bool):
            continue
        if isinstance(v, (list, tuple, set)):
            stack.extend(reversed(list(v)))
        elif isinstance(v, dict):
            stack.append(_get(v, _PAPER_ID_KEYS))
        else:
            text = str(v).strip()
            if text:
                out.extend(_bracket_ids(text) or [text])
    return out


def _is_chime_node(x: Any) -> bool:
    """CHIME-style nested list node ``[name, [ids], [children]]``."""
    return (
        isinstance(x, (list, tuple)) and bool(x) and isinstance(x[0], str) and len(x) <= 3
        and all(isinstance(y, (list, tuple)) for y in x[1:])
    )


def _flatten(items: Iterable[Any]) -> Iterator[Any]:
    """Items of a list of nodes with bare nested lists flattened (iteratively, like :func:`_paper_tokens`)."""
    stack = [iter(items)]
    while stack:
        for x in stack[-1]:
            if isinstance(x, (list, tuple)) and not _is_chime_node(x):
                stack.append(iter(x))
                break
            yield x
        else:
            stack.pop()


# --- JSON recovery from raw text --------------------------------------------


def _loads(s: str) -> Any:
    """``json.loads`` tolerant of trailing commas; None if ``s`` is not JSON."""
    s = s.strip()
    if not s:
        return None
    for cand in (s, _TRAILING_COMMA_RE.sub(r"\1", s)):
        try:
            return json.loads(cand)
        except (ValueError, RecursionError):
            continue
    return None


def _loads_inserting_commas(s: str, max_fixes: int = 16) -> Any:
    """Parse ``s`` after inserting the commas LLMs often omit between values / members; None on failure."""
    s = _TRAILING_COMMA_RE.sub(r"\1", s.strip())
    for _ in range(max_fixes):
        try:
            return json.loads(s)
        except json.JSONDecodeError as err:
            if not err.msg.startswith("Expecting ',' delimiter"):
                return None
            s = s[: err.pos] + "," + s[err.pos:]
        except RecursionError:
            return None
    return None


def _scan_value(text: str, start: int) -> Tuple[str, int, List[Tuple[int, str]], str, bool]:
    """String-aware bracket scan of the JSON value opening at ``text[start]``.

    Returns ``(status, end, cuts, closers, in_string)``:

    * ``status`` is ``"complete"`` (``end`` is one past the matching closer),
      ``"truncated"`` (the value is never closed, e.g. a reply cut at
      ``max_tokens``; ``end = len(text)``) or ``"mismatch"`` (``end`` is the
      index of a closer that does not match the innermost open bracket);
    * ``cuts`` are ``(index, closers)`` pairs such that
      ``text[start:index] + closers`` closes every open bracket: right after
      each opener, right after each nested closer, and at each comma (the
      comma excluded);
    * ``closers`` / ``in_string`` describe the brackets and string still open
      at ``end``.
    """
    stack: List[str] = []
    cuts: List[Tuple[int, str]] = []
    in_str = esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
            cuts.append((i + 1, "".join(reversed(stack))))
        elif c in "}]":
            if not stack or stack[-1] != c:
                return "mismatch", i, cuts, "".join(reversed(stack)), False
            stack.pop()
            if not stack:
                return "complete", i + 1, cuts, "", False
            cuts.append((i + 1, "".join(reversed(stack))))
        elif c == ",":
            cuts.append((i, "".join(reversed(stack))))
    return "truncated", len(text), cuts, "".join(reversed(stack)), in_str


def _parse_value(text: str, start: int) -> Tuple[Any, int]:
    """Parse the JSON value opening at ``text[start]``, repairing it if needed.

    A complete value is parsed as is, and missing commas between values are
    inserted. A closer that skips open brackets (``["p1"}``) first gets the
    missing closers inserted, and a stray closer is dropped. A truncated
    value (e.g. a reply cut at ``max_tokens``) has its open string and
    brackets closed at the point where it breaks off. If that still fails, the parser keeps the
    longest prefix that ends at a value boundary before the first syntax
    error and closes its open brackets. Either way the categories already
    written keep their place in the tree. Returns ``(value or None, end)``
    with ``end`` the end of the scanned span.
    """
    shift = 0  # characters inserted (+) / removed (-) by the bracket fixes, to report ``end`` in ``text``
    for _ in range(_MAX_BRACKET_FIXES):
        status, end, cuts, closers, in_str = _scan_value(text, start)
        if status != "mismatch":
            break
        k = closers.find(text[end])
        if k > 0:  # brackets left open inside the one this closer ends ('["p1"}'): close them first
            text = text[:end] + closers[:k] + text[end:]
            shift += k
        else:  # a stray closer: drop it
            text = text[:end] + text[end + 1:]
            shift -= 1
    whole = text[start:end] + ('"' if in_str else "") + closers
    val = _loads(whole)
    if val is None:
        val = _loads_inserting_commas(whole)
    if val is not None:
        return val, end - shift
    try:
        json.loads(whole)
        limit = end
    except json.JSONDecodeError as err:
        limit = start + err.pos
    except (ValueError, RecursionError):
        limit = end
    tried = 0
    for idx, close in reversed(cuts):
        if idx > limit:
            continue
        val = _loads(text[start:idx] + close)
        if isinstance(val, (dict, list)):
            return val, end - shift
        tried += 1
        if tried >= _MAX_REPAIRS:
            break
    return None, end - shift


def _is_structured(val: Any) -> bool:
    """Whether a JSON value can describe a hierarchy (not an empty object or a bare list of ids)."""
    if isinstance(val, dict):
        return bool(val)
    if isinstance(val, list):
        return any(isinstance(x, (dict, list)) for x in val)
    return False


def _text_candidates(text: str) -> Iterator[Any]:
    """Readings of a raw reply, most trusted first (consumed lazily by :func:`parse_hierarchy`).

    1. The whole reply, then each fenced block, parsed as JSON (any object or
       list, taken as is).
    2. The JSON objects / lists of the reply in reading order that can
       describe a hierarchy, each repaired if truncated or broken
       (:func:`_parse_value`). Scanning resumes after such a value, so an
       outer value always wins over the categories nested in it (and after
       the opener of anything else).
    3. A markdown outline (:func:`_outline_to_tree`).
    4. The first other JSON list, e.g. a bare list of ids.
    """
    for body in [text] + [m.group(1) for m in _FENCE_RE.finditer(text)]:
        val = _loads(body)
        if isinstance(val, (dict, list)):
            yield val
    other: Any = None
    pos = 0
    for _ in range(_MAX_OPENERS):
        m = _OPENER_RE.search(text, pos)
        if m is None:
            break
        val, end = _parse_value(text, m.start())
        if _is_structured(val):
            yield val
            pos = end  # skip the categories nested in this value
            continue
        if other is None and isinstance(val, list) and val:
            other = val
        pos = m.start() + 1  # e.g. an unclosed "[" in prose: the JSON may start inside the span
    outline = _outline_to_tree(text)
    if outline is not None:
        yield outline
    if other is not None:
        yield other


# --- Tree building ------------------------------------------------------------


class _Builder:
    """Incrementally builds a topic Taxonomy, validating paper ids."""

    def __init__(self, root_name: str, valid_ids: Optional[Sequence[str]], max_depth: int):
        from scholarpath.taxonomy.tree import Taxonomy

        self.tax = Taxonomy(facet="topic", root_name=root_name or "Root")
        self.valid: Optional[Set[str]] = set(valid_ids) if valid_ids is not None else None
        folded: Dict[str, Optional[str]] = {}
        for pid in self.valid or ():
            key = pid.lower()
            folded[key] = pid if key not in folded else None  # ambiguous up to case: exact match only
        self.folded = {k: v for k, v in folded.items() if v is not None}
        self.max_depth = int(max_depth)
        self.dropped: List[str] = []

    def lookup(self, token: str) -> Optional[str]:
        """The valid id equal to ``token`` (exactly, without brackets / quotes, or up to case), if any."""
        if self.valid is None:
            return None
        for t in (token, token.strip(_ID_EDGE_CHARS)):
            if t in self.valid:
                return t
            hit = self.folded.get(t.lower())
            if hit is not None:
                return hit
        return None

    def resolve(self, token: str) -> List[str]:
        """Valid paper ids cited by one token.

        Without ``valid_ids`` every token is taken as an id. Otherwise, in
        order: the token itself, without surrounding brackets / quotes, or up
        to case; else every valid id it contains as a word-like token
        (``"tb-1, tb-2"``, ``"tb-1: Some title"``). Unresolved tokens are
        recorded in ``dropped``.
        """
        token = token.strip()
        if not token:
            return []
        if self.valid is None:
            return [token]
        hit = self.lookup(token)
        if hit is not None:
            return [hit]
        found = [self.lookup(t) for t in _ID_TOKEN_RE.findall(token)]
        ids = list(dict.fromkeys(p for p in found if p is not None))
        if not ids:
            self.dropped.append(token)
        return ids

    def is_paper(self, text: str) -> bool:
        """Whether a category-like string is really a paper reference (``"p1"``, ``"[p1] Title"``, ``"p1: Title"``)."""
        if self.valid is None:
            return False
        s = str(text).strip()
        if self.lookup(s) is not None:
            return True
        head = s.lstrip(_ID_EDGE_CHARS).split(None, 1)
        return bool(head) and self.lookup(head[0]) is not None

    def add_papers(self, nid: str, tokens: Iterable[str]) -> None:
        for t in tokens:
            for pid in self.resolve(t):
                self.tax.assign(pid, nid)

    def add_dict(self, node: Dict[str, Any], nid: str, depth: int) -> None:
        node = _as_node(node)
        self.add_papers(nid, _paper_tokens(_get(node, _PAPER_KEYS)))
        children = _get(node, _CHILD_KEYS)
        if children is None:
            return
        if isinstance(children, dict):
            # {"Category A": {...} | [...], ...} -- category names as keys.
            children = [_keyed_node(k, v) for k, v in children.items()]
        if not isinstance(children, (list, tuple)):
            children = [children]
        for ch in children:
            self.add_child(ch, nid, depth + 1)

    def paper_ref(self, ch: Dict[str, Any]) -> Optional[List[str]]:
        """Paper tokens if a child dict is a paper (``{"paper_id": "p1", "title": ...}``), else None."""
        if _nonempty(_get(ch, _CHILD_KEYS)) or _nonempty(_get(ch, _PAPER_LIST_KEYS)):
            return None
        if _get(ch, _CATEGORY_NAME_KEYS) is not None:
            name = _get(ch, _CATEGORY_NAME_KEYS)
            # A leaf named by a paper id ({"name": "p1"}, outline line "- p1: Title") is a paper.
            if _get(ch, _PAPER_REF_KEYS) is None and isinstance(name, str) and self.is_paper(name):
                return [name.strip()]
            return None
        ref = _get(ch, _PAPER_REF_KEYS)
        if ref is None and self.valid is not None:
            cand = _get(ch, ("id",))
            if cand is not None and not isinstance(cand, (dict, list)) and self.lookup(str(cand)) is not None:
                ref = cand
        return None if ref is None else _paper_tokens(ref)

    def add_child(self, ch: Any, parent: str, depth: int) -> None:
        if depth > self.max_depth:
            # Too deep: attach the whole subtree's papers to the deepest allowed node.
            self.add_papers(parent, _all_paper_tokens(ch))
            return
        if isinstance(ch, dict):
            refs = self.paper_ref(ch)
            if refs is not None:
                self.add_papers(parent, refs)
                return
            cid = self.tax.add_node(_clean_name(_get(ch, _NAME_KEYS)), parent=parent)
            self.add_dict(ch, cid, depth)
        elif isinstance(ch, (list, tuple)):
            if _is_chime_node(ch):  # CHIME-style nested list [name, [ids], [children]]
                cid = self.tax.add_node(_clean_name(ch[0]), parent=parent)
                if len(ch) > 1:
                    self.add_papers(cid, _paper_tokens(list(ch[1])))
                for sub in _flatten(ch[2] if len(ch) > 2 else []):
                    self.add_child(sub, cid, depth + 1)
            else:  # a bare list of nodes
                for sub in _flatten(ch):
                    self.add_child(sub, parent, depth)
        elif isinstance(ch, (str, int)) and not isinstance(ch, bool) and str(ch).strip():
            text = str(ch)
            label = _BRACKET_RE.sub(" ", text).strip()
            if not label or self.is_paper(text):
                # A paper listed among the children ("p1", "[p1]", "p1: Title").
                self.add_papers(parent, _bracket_ids(text) if not label else [text])
                return
            cid = self.tax.add_node(_clean_name(label), parent=parent)
            self.add_papers(cid, _bracket_ids(text))


_META_KEYS = frozenset(k.lower() for k in _NAME_KEYS + ("definition", "description", "id", "summary", "rationale"))
#: Keys with a fixed meaning, never category names of a name-keyed mapping.
_RESERVED_KEYS = _META_KEYS | frozenset(k.lower() for k in _CHILD_KEYS + _PAPER_KEYS + _WRAPPER_KEYS)


def _as_node(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a node given as a name-keyed mapping (``{"GNNs": [...], "Kernels": {...}}``)."""
    if _get(d, _CHILD_KEYS) is not None or _get(d, _PAPER_KEYS) is not None:
        return d
    extra = {k: v for k, v in d.items() if str(k).lower() not in _META_KEYS}
    if extra and all(isinstance(v, (dict, list, tuple)) for v in extra.values()):
        return {"name": _get(d, _NAME_KEYS), "children": extra}
    return d


def _keyed_node(name: Any, value: Any) -> Dict[str, Any]:
    """Node for one entry of a name-keyed mapping: a sub-mapping, a list of nodes, or a list of papers."""
    if isinstance(value, dict):
        node = _as_node(value)
        return {**node, "name": _get(node, _NAME_KEYS) or name}
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, (dict, list, tuple)) for x in value):
        return {"name": name, "children": list(value)}
    return {"name": name, "papers": value}


def _all_paper_tokens(obj: Any) -> List[str]:
    """All paper tokens cited in a subtree (used when nesting exceeds ``max_depth``), in pre-order."""
    out: List[str] = []
    stack: List[Any] = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            x = _as_node(x)
            out.extend(_paper_tokens(_get(x, _PAPER_KEYS)))
            ch = _get(x, _CHILD_KEYS)
            if isinstance(ch, dict):
                ch = list(ch.values())
            stack.extend(reversed(ch if isinstance(ch, (list, tuple)) else ([ch] if ch is not None else [])))
        elif _is_chime_node(x):
            out.extend(_paper_tokens(list(x[1])) if len(x) > 1 else [])
            stack.extend(reversed(list(x[2]) if len(x) > 2 else []))
        elif isinstance(x, (list, tuple)):
            stack.extend(reversed(list(x)))
        elif isinstance(x, str):
            out.extend(_bracket_ids(x))
    return out


def _clean_name(name: Any) -> str:
    s = re.sub(r"\s+", " ", str(name or "")).strip()
    s = re.sub(r"^(?:\d+(?:\.\d+)+[.)]?|\d+[.)]|[-*+•])\s+", "", s)  # leading numbering / bullets
    s = s.strip(" :;-")
    return s or "Untitled"


def _unwrap(obj: Any) -> Any:
    """Strip wrapper objects and normalise the root node.

    A wrapper is an object without children or papers of its own whose
    wrapper key (``hierarchy``, ``taxonomy``, ``tree``, ...) holds the
    hierarchy, e.g. ``{"hierarchy": {...}}`` or ``{"topic": "X", "hierarchy":
    {...}}``. Its name-like key (``topic``, ``title``) names the root only if
    the wrapped hierarchy has no name of its own. A root given as a
    single-entry name-keyed mapping (``{"Topic": {...}}``) becomes a node
    named after its key.
    """
    outer_name: Any = None
    for _ in range(4):
        if not isinstance(obj, dict):
            break
        if _get(obj, _CHILD_KEYS) is not None or _get(obj, _PAPER_KEYS) is not None:
            break
        inner = _get(obj, _WRAPPER_KEYS)
        if not isinstance(inner, (dict, list)):
            break
        outer_name = _get(obj, _NAME_KEYS) or outer_name
        obj = inner
    if isinstance(obj, dict) and len(obj) == 1:
        key, value = next(iter(obj.items()))
        if isinstance(value, (dict, list)) and str(key).lower() not in _RESERVED_KEYS:
            obj = _keyed_node(key, value)
    if outer_name is not None and not isinstance(outer_name, (dict, list)):
        if isinstance(obj, list):
            obj = {"name": outer_name, "children": obj}
        elif isinstance(obj, dict) and _get(obj, _NAME_KEYS) is None:
            obj = {**obj, "name": outer_name}
    return obj


def _outline_to_tree(text: str) -> Optional[Dict[str, Any]]:
    """Fallback: read a markdown outline (headings / indented or numbered bullets).

    Category lines may cite papers in brackets (``- Graph methods [p1, p2]``);
    lines starting with a bracketed id (``- [p1] Some title``) or bare lines
    with brackets place papers under the enclosing category. Returns a nested
    ``{"name", "children", "papers"}`` tree, or None if no category was found.
    A single top-level category becomes the root.
    """
    root: Dict[str, Any] = {"name": "", "children": [], "papers": []}
    stack: List[Tuple[float, Dict[str, Any]]] = [(-float("inf"), root)]
    found = False
    for raw in (text or "").splitlines():
        raw = raw.replace("\t", "    ")
        m_head = _HEADING_RE.match(raw)
        m_bullet = _BULLET_RE.match(raw) if m_head is None else None
        if m_head is not None:
            indent = -1000.0 + len(m_head.group("hashes"))
            body = m_head.group("text")
        elif m_bullet is not None:
            marker = m_bullet.group("marker").rstrip(".)")
            levels = marker.count(".") if marker[:1].isdigit() else 0
            indent = float(len(m_bullet.group("indent")) + 4 * levels)
            body = m_bullet.group("text")
        else:
            stack[-1][1]["papers"].extend(_bracket_ids(raw))
            continue
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        ids = _bracket_ids(body)
        lead = _ID_ONLY_PREFIX_RE.match(body)
        if lead is not None:
            stack[-1][1]["papers"].extend(_bracket_ids(lead.group(0)))
            continue
        name = re.sub(r"\((?:papers?|ids?|citations?)\s*:?\s*\)", "", _BRACKET_RE.sub(" ", body), flags=re.I)
        node: Dict[str, Any] = {"name": _clean_name(name), "children": [], "papers": list(ids)}
        stack[-1][1]["children"].append(node)
        stack.append((indent, node))
        found = True
    if not found:
        return None
    if len(root["children"]) == 1 and not root["papers"]:
        return root["children"][0]
    return root


def _prune_empty(tax: "Taxonomy") -> None:
    for nid in list(tax.bfs()):
        if nid in tax.nodes and nid != tax.root_id and not tax.nodes[nid].papers:
            tax.remove_subtree(nid)


def _build(
    obj: Any,
    valid_ids: Optional[Sequence[str]],
    root_name: Optional[str],
    prune_empty: bool,
    max_depth: int,
) -> "Taxonomy":
    """Build a topic Taxonomy from one parsed reading of the reply."""
    obj = _unwrap(obj)
    name = root_name
    if isinstance(obj, dict) and _get(obj, _NAME_KEYS) is not None:
        name = _clean_name(_get(obj, _NAME_KEYS))
    builder = _Builder(name or root_name or "Root", valid_ids, max_depth)
    tax = builder.tax
    if isinstance(obj, dict):
        builder.add_dict(obj, tax.root_id, 0)
    elif isinstance(obj, list):
        for ch in obj:
            builder.add_child(ch, tax.root_id, 1)
    if prune_empty:
        _prune_empty(tax)
    tax.invalidate()
    tax.metadata["dropped_ids"] = sorted(set(builder.dropped))
    return tax


def parse_hierarchy(
    obj: Any,
    valid_ids: Optional[Iterable[str]] = None,
    root_name: Optional[str] = None,
    prune_empty: bool = True,
    max_depth: int = 8,
) -> "Taxonomy":
    """Convert the composer's output into a topic :class:`~scholarpath.taxonomy.tree.Taxonomy`.

    ``obj`` is the parsed JSON or the raw response text, which is what
    :func:`organize` passes. The parser tolerates malformed output:

    * raw text: the whole reply or a fenced block as JSON, including a
      top-level list of categories; else JSON embedded in prose. Broken JSON
      is repaired: trailing commas are accepted, missing commas and closers
      are inserted, and truncated JSON (a reply cut at ``max_tokens``) or
      JSON with another syntax error is reduced to its longest valid prefix
      with the brackets closed, so the root and the categories already
      written are kept. If no reading yields a paper, a markdown outline is
      tried (headings / indented bullets with ``[id]`` citations). Readings
      are tried in this order, and the first one that places at least one
      valid paper wins;
    * wrapper objects (``{"hierarchy": {...}}``, ``{"topic": "X",
      "taxonomy": {...}}``), alternative keys (``subtopics`` / ``categories``
      for children, ``paper_ids`` / ``citations`` for papers, ``title`` /
      ``category`` for names), name-keyed mappings, categories given as plain
      strings (``"Kernels [p3, p4]"``) or as CHIME-style nested lists
      ``[name, [ids], [children]]``;
    * paper entries given as ``"p1"``, ``"[p1]"``, ``"[p1, p2]"``,
      ``"p1, p2"``, ``"p1: Title"``, ``"P1"`` (case variant),
      ``{"paper_id": "p1"}``, or listed among the children.

    Paper ids not in ``valid_ids`` (when given) are dropped, so systems are
    only credited for papers actually present in their context. With
    ``prune_empty``, categories whose subtree cites no valid paper are
    removed. Nesting deeper than ``max_depth`` levels is collapsed into the
    deepest allowed category. Papers are assigned with
    :meth:`~scholarpath.taxonomy.tree.Taxonomy.assign`, i.e. they belong to the
    extent of every ancestor of the category that lists them.
    ``tax.metadata["dropped_ids"]`` records rejected citations.
    """
    valid = None if valid_ids is None else [str(v) for v in valid_ids]
    readings: Iterable[Any] = _text_candidates(obj) if isinstance(obj, str) else (obj,)
    first: Optional["Taxonomy"] = None
    for reading in readings:
        tax = _build(reading, valid, root_name, prune_empty, max_depth)
        if tax.all_papers():
            return tax
        if first is None:
            first = tax
    return first if first is not None else _build(None, valid, root_name, prune_empty, max_depth)


def organize(
    llm: "LLMClient",
    query: str,
    context: str,
    sources: Sequence[Dict[str, Any]] = (),
    max_tokens: int = 2048,
    style: str = "survey",
    component: Optional[str] = "composition",
    valid_ids: Optional[Iterable[str]] = None,
    list_sources: bool = False,
) -> "Taxonomy":
    """Run the single composition call and parse its hierarchy.

    Exactly one LLM call is issued (no JSON-parse retries, so that budgeted
    systems make ONE composition call); malformed output is handled by
    :func:`parse_hierarchy`. Valid ids default to the paper ids of ``sources``.
    Call statistics are stored in ``tax.metadata``.
    """
    request = organize_request(query, context, sources, max_tokens, component, style, list_sources)
    response = llm.complete(request)
    if valid_ids is None:
        valid_ids = [str(s["paper_id"]) for s in sources if s.get("paper_id") is not None]
    tax = parse_hierarchy(response.text, valid_ids=valid_ids, root_name=query)
    tax.metadata.update(
        {
            "llm_calls": 1,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "latency_s": response.latency_s,
        }
    )
    return tax


# ---------------------------------------------------------------------------
# Offline mock
# ---------------------------------------------------------------------------

# Title words that describe a paper's framing rather than its topic.
_GENERIC = frozenset(
    """
    towards toward revisiting scalable efficient robust unified fast novel approach approaches framework
    frameworks analysis study studies survey review learning model models improved improving effective
    general simple practical rethinking understanding exploring large scale deep randomized trial trials
    """.split()
)


def _title_tokens(title: str) -> List[str]:
    return [t for t in tokenize(title, remove_stopwords=True) if len(t) > 2]


def _group(
    pids: List[str],
    toks: Dict[str, List[str]],
    exclude: Set[str],
    max_groups: int,
    min_size: int = 2,
) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    """Greedy keyword cover: repeatedly take the keyword shared by most remaining papers."""
    n = len(pids)
    keys = {p: set(toks[p]) - exclude - _GENERIC for p in pids}
    remaining = list(pids)
    groups: List[Tuple[str, List[str]]] = []
    while remaining and len(groups) < max_groups:
        counts = Counter(k for p in remaining for k in keys[p])
        cands = [(c, k) for k, c in counts.items() if min_size <= c < n]
        if not cands:
            break
        _, best = min(cands, key=lambda ck: (-ck[0], ck[1]))
        members = [p for p in remaining if best in keys[p]]
        groups.append((best, members))
        remaining = [p for p in remaining if best not in keys[p]]
    return groups, remaining


def _group_name(members: List[str], toks: Dict[str, List[str]], exclude: Set[str], key: str) -> Tuple[str, Set[str]]:
    shared = set(toks[members[0]])
    for p in members[1:]:
        shared &= set(toks[p])
    shared -= exclude
    words = [t for t in dict.fromkeys(toks[members[0]]) if t in shared and t not in _GENERIC][:4]
    if key not in words:
        words = [key] + words[:3]
    return " ".join(w.capitalize() for w in words), set(words)


def _mock_organize(payload: Dict[str, Any], request: LLMRequest) -> Dict[str, Any]:
    sources = [s for s in payload.get("sources", []) or [] if s.get("paper_id")]
    root = str(payload.get("query") or "Topic")[:120]
    pids = list(dict.fromkeys(str(s["paper_id"]) for s in sources))
    toks = {str(s["paper_id"]): _title_tokens(str(s.get("title", ""))) for s in sources}
    universal = set.intersection(*(set(toks[p]) for p in pids)) if pids else set()
    groups, rest = _group(pids, toks, universal, max_groups=6)
    children: List[Dict[str, Any]] = []
    for key, members in groups:
        name, used = _group_name(members, toks, universal, key)
        node: Dict[str, Any] = {"name": name, "children": [], "papers": []}
        if len(members) > 4:
            sub_groups, sub_rest = _group(members, toks, universal | used, max_groups=4)
            for skey, smembers in sub_groups:
                sname, _ = _group_name(smembers, toks, universal | used, skey)
                node["children"].append({"name": sname, "children": [], "papers": smembers})
            node["papers"] = sub_rest
        else:
            node["papers"] = members
        children.append(node)
    if rest:
        children.append({"name": "Other Topics", "children": [], "papers": rest})
    return {"name": root, "children": children, "papers": []}


MOCK_HANDLERS = {TASK: _mock_organize}
