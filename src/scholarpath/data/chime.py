"""CHIME: biomedical review topics organized into human-corrected hierarchies (Sec. 5.3, App. C.2).

Raw data (github.com/allenai/chime, ``resources/``):

* ``raw_source_data.csv`` -- one row per (Cochrane review, included study):
  ``review_id, review_pmid, review_title, review_abstract, study_pmid,
  study_title, study_abstract, claim, claim_id`` (``claim_id`` = 0..n-1
  within a review; ``claim`` is an LLM-extracted finding of the study);
* ``raw_generated_hierarchy.csv`` -- ``review_id, claude_claim_hierarchy``:
  Claude-2's raw output whose ``[Response 2]`` lists, for each top-level
  *aspect*, a hierarchy of categories annotated with the claims they hold,
  e.g. ``- 1.1: Impact on smoking onset (Claim 0, 9, 10)``;
* ``data_split.csv`` -- ``review_id, review_pmid, split``: the 100
  expert-annotated topics.

Expert annotations (Hugging Face, ``joe32140/chime-*``; CSV / parquet /
JSONL exports accepted):

* ``chime-parent-child-relation``: is a child category a valid
  sub-category of its parent (aggregated label ``agg_label``);
* ``chime-sibling-coherence``: ``sibling_group_id, parent_category,
  child_category, agg_label, coherence_rate``;
* ``chime-claim-category``: ``review_pmid, hierarchy_id, claim, category,
  node_id, node_path, human_label, agg_label`` -- ``agg_label`` = 1 iff the
  claim belongs to the node *and* all its ancestors.

App. C.2: *"CHIME provides 472 biomedical review topics with 2,174
hierarchies; we evaluate on its gold-standard subset of 100 expert-curated
topics (320 human-corrected hierarchies over 2,146 distinct studies),
pooling all studies into a unified retrieval corpus so that systems must
both retrieve the studies of a topic and organize them into categories."*

Reconstruction (no corrected-hierarchy file is released; see
:func:`build_chime`):

1. each review's ``[Response 2]`` is parsed into aspect hierarchies
   (:func:`parse_generated_hierarchy`, a re-implementation of CHIME's
   ``utils.get_hierarchy`` that handles every list style in the file);
   ``hierarchy_id`` is the aspect's position, as in CHIME's
   ``parse_hierachy.py`` (claims never categorised form a trailing
   ``"Uncategorized studies"`` hierarchy there and here);
2. the annotated hierarchies are those ``(review_pmid, hierarchy_id)``
   pairs present in the expert annotation tables;
3. parent-child edges with ``agg_label == 0`` are removed together with the
   child's subtree (``edge_policy="drop"``); sibling-coherence labels do not
   identify the offending sibling and are only recorded;
4. a claim stays assigned to a category iff its claim-category
   ``agg_label == 1`` (claim-category pairs that were not annotated keep the
   generated assignment);
5. claims map to studies through ``claim_id`` (or the claim text, after
   CHIME's ``clean_claim``), and categories hold the studies' pmids;
6. the gold studies of a topic are the studies appearing in its annotated
   hierarchies (``gold_studies="annotated"``), and the pooled corpus is the
   union over topics (``"assigned"`` keeps only studies with at least one
   accepted assignment; ``"all"`` every study of the review).

Without annotation files the generated hierarchies of the 100 topics are
used uncorrected (``corrected = False`` in the outputs) -- the raw join gives
2,439 study rows / 2,438 distinct studies for these topics rather than the
paper's 2,146, which counts the studies of the annotated hierarchies only.
"""

from __future__ import annotations

import csv
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

from scholarpath.data.corpus import Corpus, save_queries, write_json
from scholarpath.data.download import read_json_records
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.types import EVIDENCE, Paper, Query

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

BENCHMARK = "chime"
UNCATEGORIZED = "Uncategorized studies"
ANNOTATION_TABLES = ("chime-parent-child-relation", "chime-sibling-coherence", "chime-claim-category")
SPLITS = ("train", "val", "test_ID", "test_OOD")
PAPER_STATS = {"topics": 472, "hierarchies": 2174, "gold_topics": 100, "gold_hierarchies": 320,
               "gold_studies": 2146}
INSTRUCTION = ("Retrieve the studies relevant to the systematic review \"{topic}\" and organize them into a "
               "hierarchy of categories, as a systematic review would.")


# ---------------------------------------------------------------------------
# Hierarchy text parsing
# ---------------------------------------------------------------------------


@dataclass
class ChimeNode:
    """A category of a generated hierarchy: ``name``, the claims it lists, and its children."""

    name: str
    claims: List[int] = field(default_factory=list)
    children: List["ChimeNode"] = field(default_factory=list)

    def walk(self) -> Iterator["ChimeNode"]:
        """Pre-order traversal (CHIME's node-id order)."""
        yield self
        for c in self.children:
            yield from c.walk()

    def all_claims(self) -> Set[int]:
        out = set(self.claims)
        for c in self.children:
            out |= c.all_claims()
        return out

    def to_nested(self) -> Dict[str, Any]:
        return {"name": self.name, "claims": list(self.claims), "children": [c.to_nested() for c in self.children]}


_CLAIMS_RE = re.compile(r"\(\s*(?:Claims?|Papers?|Studies|Study)\s*:?\s*([0-9][0-9,\s\-–andAND&]*)\)", re.IGNORECASE)
_ASPECT_RE = re.compile(r"^\s*(?:[-*•]\s*)?Aspect\s*\d+\s*[:.\-–]?\s*", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^\s*(?:[-*•]\s*)?(\d+(?:\.\d+)*)\.?\s*[:.)\-–]?\s+")
_BULLET_RE = re.compile(r"^(\s*)[-*•]\s+")


def extract_response(text: str) -> str:
    """The ``[Response 2]`` part of a raw Claude output (CHIME ``utils.extract_response``)."""
    start = text.find("[Response 2]")
    if start < 0:
        return text.strip()
    end = text.find("Review:", start)
    if end == -1:
        end = text.find("Human:", start)
        if end == -1:
            end = len(text)
    body = text[start + len("[Response 2]"):end]
    return body.lstrip(":").strip()


def parse_claim_ids(line: str) -> Tuple[str, List[int]]:
    """Split ``"Title (Claim 0, 3-5)"`` into ``("Title", [0, 3, 4, 5])``."""
    claims: List[int] = []
    for m in _CLAIMS_RE.finditer(line):
        for part in re.split(r"[,&]|\band\b", m.group(1), flags=re.IGNORECASE):
            part = part.strip()
            rng = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
            if rng:
                a, b = int(rng.group(1)), int(rng.group(2))
                if 0 <= b - a <= 500:
                    claims.extend(range(a, b + 1))
            elif part.isdigit():
                claims.append(int(part))
    text = _CLAIMS_RE.sub("", line)
    return text.rstrip(), list(dict.fromkeys(claims))


def _clean_name(text: str, aspect: bool) -> str:
    text = _ASPECT_RE.sub("", text) if aspect else text
    text = _NUMBER_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    if aspect:  # CHIME's clean_aspect drops parentheticals (keyword lists) from aspect names
        text = re.sub(r"\([^()]*\)", "", text)
    return " ".join(text.strip().strip(":").split())


def _indent(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip(" "))


def _aspect_mode(lines: Sequence[str]) -> str:
    """How top-level aspects are marked: ``aspect`` ("Aspect k"), ``dash`` (column-0 bullets),
    ``number`` (column-0 integers) or ``plain`` (column-0 unbulleted lines)."""
    first = lines[0].strip()
    if _ASPECT_RE.match(first):
        return "aspect"
    if re.match(r"^[-*•]\s", lines[0]):
        return "dash"
    if re.match(r"^\d+\.?\s", lines[0]) and not re.match(r"^\d+\.\d", lines[0]):
        return "number"
    return "plain"


def _is_aspect_line(line: str, mode: str) -> bool:
    if mode == "aspect":
        return bool(_ASPECT_RE.match(line))
    if _indent(line) > 0:
        return False
    if mode == "dash":
        return bool(re.match(r"^[-*•]\s", line))
    if mode == "number":
        return bool(re.match(r"^\d+\.?\s", line)) and not re.match(r"^\d+\.\d", line)
    return not re.match(r"^[-*•]\s", line) and not re.match(r"^\d+(?:\.\d+)*\.?[\s:]", line)


def _number_depth(line: str) -> Optional[int]:
    m = _NUMBER_RE.match(line)
    return len(m.group(1).split(".")) if m else None


def _build_block(lines: Sequence[str]) -> List[ChimeNode]:
    """Nest the category lines of one aspect by numbering depth (if every line is numbered) or indentation."""
    if not lines:
        return []
    depths = [_number_depth(l) for l in lines]
    use_numbers = all(d is not None for d in depths)
    if use_numbers:
        base = min(d for d in depths if d is not None)
        levels = [int(d) - base for d in depths]  # type: ignore[arg-type]
    else:
        levels = []
        stack: List[int] = []
        for l in lines:
            ind = _indent(l)
            while stack and stack[-1] >= ind:
                stack.pop()
            levels.append(len(stack))
            stack.append(ind)
    roots: List[ChimeNode] = []
    path: List[ChimeNode] = []
    for line, level in zip(lines, levels):
        text, claims = parse_claim_ids(line)
        node = ChimeNode(_clean_name(text, aspect=False), claims)
        level = min(level, len(path))
        del path[level:]
        (path[-1].children if path else roots).append(node)
        path.append(node)
    return roots


def parse_generated_hierarchy(text: str) -> List[ChimeNode]:
    """Parse a raw CHIME hierarchy into its aspect hierarchies (one :class:`ChimeNode` per aspect).

    Handles the list styles of ``raw_generated_hierarchy.csv``: ``Aspect k:``
    headers, column-0 bullets, numbered or unbulleted top-level lines, with
    sub-categories nested by dotted numbering (``1.1``, ``1.1.1``) or by
    indentation. Claim lists ``(Claim 0, 3)`` / ``(Claims: 0, 3)`` / ``(Paper 1)``
    are attached to their category.
    """
    response = extract_response(text)
    lines = [l.rstrip() for l in response.splitlines() if l.strip()]
    lines = [l for l in lines if not re.match(r"^\s*\[Response \d\]", l)]
    if not lines:
        return []
    mode = _aspect_mode(lines)
    aspects: List[ChimeNode] = []
    block: List[str] = []
    for line in lines:
        if _is_aspect_line(line, mode):
            if aspects:
                aspects[-1].children = _build_block(block)
            body, claims = parse_claim_ids(line)
            aspects.append(ChimeNode(_clean_name(body, aspect=True), claims))
            block = []
        elif aspects:
            block.append(line)
    if aspects:
        aspects[-1].children = _build_block(block)
    return [a for a in aspects if a.name]


def add_uncategorized(aspects: List[ChimeNode], n_claims: int) -> List[ChimeNode]:
    """Append CHIME's ``"Uncategorized studies"`` hierarchy for claims no category lists."""
    used: Set[int] = set()
    for a in aspects:
        used |= a.all_claims()
    rest = sorted(set(range(n_claims)) - used)
    if rest:
        aspects = list(aspects) + [ChimeNode(UNCATEGORIZED, rest)]
    return aspects


# ---------------------------------------------------------------------------
# Raw tables
# ---------------------------------------------------------------------------


def read_table(path: PathLike) -> List[Dict[str, Any]]:
    """Read a CSV / TSV / JSON / JSONL / parquet table, or a directory of split files, as a list of dicts.

    Hugging Face ``save_to_disk`` directories and parquet need ``pandas`` (+
    ``pyarrow``) / ``datasets``, imported lazily.
    """
    path = Path(path)
    if path.is_dir():
        if (path / "dataset_dict.json").exists() or (path / "dataset_info.json").exists():
            try:
                from datasets import load_from_disk
            except ImportError as err:
                raise ImportError(f"Reading {path} (datasets.save_to_disk) needs `pip install datasets`") from err
            ds = load_from_disk(str(path))
            parts = [ds[s] for s in ds] if hasattr(ds, "keys") else [ds]
            return [dict(r) for part in parts for r in part]
        rows: List[Dict[str, Any]] = []
        meta = {"dataset_info.json", "dataset_infos.json", "dataset_dict.json", "state.json", "config.json"}
        files = sorted(p for p in path.rglob("*") if p.is_file()
                       and p.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".parquet"}
                       and p.name not in meta
                       and not any(part.startswith(".") for part in p.relative_to(path).parts))
        for f in files:
            part = read_table(f)
            split = _split_from_name(f)
            for r in part:
                if split and not r.get("split"):
                    r["split"] = split
            rows.extend(part)
        return rows
    suffix = "".join(path.suffixes[-2:]).lower() if path.suffix == ".gz" else path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
        with open(path, newline="", encoding="utf-8") as f:
            return [dict(r) for r in csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ",")]
    if suffix in (".json", ".jsonl", ".json.gz", ".jsonl.gz"):
        # JSON array or JSON Lines whatever the extension (`Dataset.to_json` writes JSON Lines to *.json).
        return [dict(r) for r in read_json_records(path)]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError(f"Reading {path} needs pandas and pyarrow: pip install pandas pyarrow") from err
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported table format: {path}")


def _split_from_name(path: Path) -> Optional[str]:
    name = path.stem.lower()
    for s in SPLITS:
        if re.search(r"(?:^|[-_/])" + s.lower() + r"(?:$|[-_.])", name):
            return s
    return None


def find_annotation_tables(directory: PathLike) -> Dict[str, Path]:
    """Locate the three annotation tables (``chime-<task>`` files or directories) under ``directory``."""
    directory = Path(directory)
    found: Dict[str, Path] = {}
    for name in ANNOTATION_TABLES:
        short = name.replace("chime-", "")
        for cand in sorted(directory.iterdir()) if directory.exists() else []:
            stem = cand.name.lower()
            if stem.startswith(name) or stem.startswith(short) or stem.startswith(short.replace("-", "_")):
                found[name] = cand
                break
    return found


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def clean_claim(claim: str) -> str:
    """CHIME ``utils.clean_claim``: drop "New/Null finding(s):" prefixes; whitespace-normalised, lower-cased key."""
    claim = re.sub(r"^\s*(?:new|null)\s+findings?\s*:\s*", "", str(claim or ""), flags=re.IGNORECASE)
    return " ".join(claim.split()).lower()


def _norm_name(name: Any) -> str:
    """Category-name key for matching annotation rows: numbering prefixes (``"1.1.1: "``, which
    CHIME's own parser keeps for some list styles), punctuation and case are ignored."""
    text = re.sub(r"^\s*(?:[-*•]\s*)?(?:aspect\s*)?\d+(?:\.\d+)*\.?\s*[:.)\-–]?\s*", "", str(name or ""),
                  flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _label(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[int]:
    for k in keys:
        v = row.get(k)
        if v is None or v == "":
            continue
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            if str(v).strip().lower() in ("true", "yes"):
                return 1
            if str(v).strip().lower() in ("false", "no"):
                return 0
    return None


def _first(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


@dataclass
class ChimeTopic:
    """One review topic with its studies and parsed hierarchies."""

    review_id: str
    review_pmid: str
    title: str
    abstract: str
    split: Optional[str]
    studies: List[Dict[str, str]]  # rows ordered by claim_id
    hierarchies: List[ChimeNode]

    def claim_to_pmid(self, claim_id: int) -> Optional[str]:
        if 0 <= claim_id < len(self.studies):
            return self.studies[claim_id]["study_pmid"]
        return None


def load_topics(resources_dir: PathLike, review_ids: Optional[Iterable[str]] = None) -> Dict[str, ChimeTopic]:
    """Join ``raw_source_data.csv`` and ``raw_generated_hierarchy.csv`` (optionally for some reviews only)."""
    res = Path(resources_dir)
    if (res / "resources").is_dir():
        res = res / "resources"
    src_path, hier_path = res / "raw_source_data.csv", res / "raw_generated_hierarchy.csv"
    for p in (src_path, hier_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run `python scripts/download_data.py --only chime`")
    wanted = set(review_ids) if review_ids is not None else None
    split_map = {r["review_id"]: r.get("split") for r in read_table(res / "data_split.csv")} \
        if (res / "data_split.csv").exists() else {}
    rows_by_review: Dict[str, List[Dict[str, str]]] = {}
    for row in read_table(src_path):
        rid = row["review_id"]
        if wanted is None or rid in wanted:
            rows_by_review.setdefault(rid, []).append(row)
    hier_text = {r["review_id"]: r.get("claude_claim_hierarchy") or "" for r in read_table(hier_path)}
    topics: Dict[str, ChimeTopic] = {}
    for rid, rows in rows_by_review.items():
        rows.sort(key=lambda r: int(r.get("claim_id") or 0))
        try:
            aspects = parse_generated_hierarchy(hier_text.get(rid, ""))
        except Exception as err:  # pragma: no cover - parser is total on the released file
            log.warning("Could not parse the hierarchy of %s: %s", rid, err)
            aspects = []
        aspects = add_uncategorized(aspects, len(rows))
        first = rows[0]
        topics[rid] = ChimeTopic(rid, str(first["review_pmid"]), " ".join(first["review_title"].split()),
                                 " ".join((first.get("review_abstract") or "").split()), split_map.get(rid),
                                 rows, aspects)
    return topics


@dataclass
class _Annotations:
    claim_category: List[Dict[str, Any]] = field(default_factory=list)
    parent_child: List[Dict[str, Any]] = field(default_factory=list)
    sibling: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.claim_category or self.parent_child)


def load_annotations(annotations_dir: Optional[PathLike]) -> _Annotations:
    ann = _Annotations()
    if annotations_dir is None or not Path(annotations_dir).exists():
        return ann
    tables = find_annotation_tables(annotations_dir)
    if "chime-claim-category" in tables:
        ann.claim_category = read_table(tables["chime-claim-category"])
    if "chime-parent-child-relation" in tables:
        ann.parent_child = read_table(tables["chime-parent-child-relation"])
    if "chime-sibling-coherence" in tables:
        ann.sibling = read_table(tables["chime-sibling-coherence"])
    return ann


def _find_node(aspect: ChimeNode, node_id: Any, category: Any) -> Optional[ChimeNode]:
    """Locate a node by CHIME's pre-order ``node_id`` (validated by name) or else by name."""
    nodes = list(aspect.walk())
    want = _norm_name(category)
    try:
        idx = int(float(node_id)) if node_id not in (None, "") else None
    except (TypeError, ValueError):
        idx = None
    if idx is not None and 0 <= idx < len(nodes) and (not want or _norm_name(nodes[idx].name) == want):
        return nodes[idx]
    if want:
        for n in nodes:
            if _norm_name(n.name) == want:
                return n
    return None


@dataclass
class ChimeData:
    corpus: Corpus
    queries: List[Query]
    gold: Dict[str, Taxonomy]  # review_id -> gold hierarchy
    stats: Dict[str, Any]

    def save(self, out_dir: PathLike) -> Dict[str, str]:
        """Write ``papers.jsonl.gz``, ``gold/<review_id>.json``, ``queries.jsonl``, ``stats.json``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.corpus.save(out / "papers.jsonl.gz", include_chunks=False)
        for rid, tax in self.gold.items():
            tax.save(out / "gold" / f"{rid}.json")
        save_queries(out / "queries.jsonl", self.queries)
        write_json(out / "stats.json", self.stats)
        return {"papers": str(out / "papers.jsonl.gz"), "gold": str(out / "gold"), "queries": str(out / "queries.jsonl")}


def _add_aspect(tax: Taxonomy, aspect: ChimeNode, hid: int, topic: ChimeTopic,
                accepted: Mapping[int, Set[int]], drop: Set[int]) -> Set[str]:
    """Add one corrected aspect hierarchy under the root of ``tax``; returns the pmids it assigns.

    Nodes in ``drop`` (rejected parent-child edges) are skipped with their
    subtrees; each node receives the studies of its ``accepted`` claims.
    Node metadata keep the CHIME pre-order node id and the generated claim list.
    """
    preorder = {id(n): i for i, n in enumerate(aspect.walk())}
    assigned: Set[str] = set()
    stack = [(aspect, tax.root_id)]
    while stack:
        node, parent = stack.pop()
        if id(node) in drop:
            continue
        nid = tax.add_node(node.name, parent=parent, hierarchy_id=hid, chime_node_id=preorder[id(node)],
                           generated_claims=list(node.claims))
        for cidx in sorted(accepted[id(node)]):
            pmid = topic.claim_to_pmid(cidx)
            if pmid is not None:
                tax.assign(study_paper_id(pmid), nid)
                assigned.add(pmid)
        stack.extend((c, nid) for c in reversed(node.children))
    return assigned


def study_paper_id(pmid: Any) -> str:
    """Corpus id of a CHIME study (``pmid:<PubMed id>``)."""
    return f"pmid:{str(pmid).strip()}"


def build_chime(
    resources_dir: PathLike,
    annotations_dir: Optional[PathLike] = None,
    topics: str = "gold",
    gold_studies: str = "annotated",
    edge_policy: str = "drop",
    include_uncategorized: bool = False,
    require_annotations: bool = False,
    limit: Optional[int] = None,
) -> ChimeData:
    """Reconstruct the CHIME organization benchmark (see the module docstring).

    Parameters
    ----------
    topics:
        ``"gold"`` -- the 100 topics of ``data_split.csv``; ``"all"`` -- every review.
    gold_studies:
        ``"annotated"`` (default), ``"assigned"`` or ``"all"`` (module docstring, step 6).
    edge_policy:
        ``"drop"`` removes rejected parent-child edges with the child's
        subtree; ``"ignore"`` keeps every generated edge.
    include_uncategorized:
        Keep CHIME's ``"Uncategorized studies"`` pseudo-hierarchy in the gold trees.
    require_annotations:
        Raise if the expert annotation tables are missing instead of falling
        back to the uncorrected generated hierarchies.
    """
    if gold_studies not in ("annotated", "assigned", "all"):
        raise ValueError("gold_studies must be 'annotated', 'assigned' or 'all'")
    if edge_policy not in ("drop", "ignore"):
        raise ValueError("edge_policy must be 'drop' or 'ignore'")
    res = Path(resources_dir)
    res = res / "resources" if (res / "resources").is_dir() else res
    review_ids: Optional[List[str]] = None
    if topics == "gold":
        split_rows = read_table(res / "data_split.csv")
        review_ids = [r["review_id"] for r in split_rows]
    elif topics != "all":
        raise ValueError("topics must be 'gold' or 'all'")
    loaded = load_topics(res, review_ids)
    order = review_ids if review_ids is not None else sorted(loaded)
    order = [r for r in order if r in loaded]
    if limit is not None:
        order = order[:limit]
    ann = load_annotations(annotations_dir)
    if not ann.available:
        msg = ("CHIME expert annotations not found%s; using the uncorrected generated hierarchies "
               "(run `python scripts/download_data.py --only chime_annotations`)."
               % (f" in {annotations_dir}" if annotations_dir else ""))
        if require_annotations:
            raise FileNotFoundError(msg)
        log.warning(msg)

    # Index annotations by review pmid.
    cc_by_review: Dict[str, List[Dict[str, Any]]] = {}
    for r in ann.claim_category:
        cc_by_review.setdefault(str(_first(r, ("review_pmid",))).split(".")[0], []).append(r)
    pc_by_review: Dict[str, List[Dict[str, Any]]] = {}
    for r in ann.parent_child:
        pc_by_review.setdefault(str(_first(r, ("review_pmid",)) or "").split(".")[0], []).append(r)

    corpus = Corpus(name="chime")
    gold_trees: Dict[str, Taxonomy] = {}
    queries: List[Query] = []
    totals = {"hierarchies": 0, "annotated_hierarchies": 0, "dropped_edges": 0, "rejected_assignments": 0,
              "unmatched_annotations": 0, "study_rows": 0, "topics_without_annotations": 0}
    for rid in order:
        topic = loaded[rid]
        totals["study_rows"] += len(topic.studies)
        claim_by_text = {clean_claim(s.get("claim", "")): i for i, s in enumerate(topic.studies)}
        aspects = topic.hierarchies
        totals["hierarchies"] += sum(1 for a in aspects if a.name != UNCATEGORIZED)
        cc_rows = cc_by_review.get(topic.review_pmid, [])
        pc_rows = pc_by_review.get(topic.review_pmid, [])
        annotated_ids: Optional[Set[int]] = None
        if ann.available and not (cc_rows or pc_rows):
            # No annotation row joins this topic (e.g. a review_pmid mismatch): keep its generated hierarchies.
            totals["topics_without_annotations"] += 1
        elif ann.available:
            annotated_ids = set()
            for r in cc_rows + pc_rows:
                hid = _first(r, ("hierarchy_id",))
                if hid not in (None, ""):
                    annotated_ids.add(int(float(hid)))
        # Assignments: (aspect index, node) -> claim ids; rejected pairs removed.
        accepted: Dict[int, Dict[int, Set[int]]] = {}
        rejected: Dict[int, Set[Tuple[int, int]]] = {}
        for hid, aspect in enumerate(aspects):
            accepted[hid] = {id(n): set(n.claims) for n in aspect.walk()}
            rejected[hid] = set()
        for r in cc_rows:
            hid = _first(r, ("hierarchy_id",))
            if hid in (None, "") or int(float(hid)) >= len(aspects):
                totals["unmatched_annotations"] += 1
                continue
            hid = int(float(hid))
            node = _find_node(aspects[hid], _first(r, ("node_id",)), _first(r, ("category", "node_name")))
            claim_idx = _first(r, ("claim_id",))
            if claim_idx not in (None, ""):
                cidx: Optional[int] = int(float(claim_idx))
            else:
                cidx = claim_by_text.get(clean_claim(_first(r, ("claim",)) or ""))
            label = _label(r, ("agg_label", "human_label", "label"))
            if node is None or cidx is None or label is None:
                totals["unmatched_annotations"] += 1
                continue
            if label == 1:
                accepted[hid][id(node)].add(cidx)
            else:
                accepted[hid][id(node)].discard(cidx)
                rejected[hid].add((id(node), cidx))
                totals["rejected_assignments"] += 1
        # Parent-child edges rejected by the experts.
        drop: Dict[int, Set[int]] = {hid: set() for hid in range(len(aspects))}
        if edge_policy == "drop":
            for r in pc_rows:
                label = _label(r, ("agg_label", "human_label", "label"))
                if label != 0:
                    continue
                hid_raw = _first(r, ("hierarchy_id",))
                hids = [int(float(hid_raw))] if hid_raw not in (None, "") else list(range(len(aspects)))
                parent = _first(r, ("parent_category", "parent", "parent_name"))
                child = _first(r, ("child_category", "child", "child_name"))
                hit = False
                for hid in hids:
                    if hid >= len(aspects):
                        continue
                    for n in aspects[hid].walk():
                        if _norm_name(n.name) != _norm_name(parent):
                            continue
                        for c in n.children:
                            if _norm_name(c.name) == _norm_name(child):
                                drop[hid].add(id(c))
                                hit = True
                if hit:
                    totals["dropped_edges"] += 1
                else:
                    totals["unmatched_annotations"] += 1
        # Build the gold taxonomy of the topic.
        tax = Taxonomy(facet="topic", root_name=topic.title, root_definition=topic.abstract[:500])
        corrected = ann.available and annotated_ids is not None
        tax.metadata.update({"benchmark": BENCHMARK, "review_id": rid, "review_pmid": topic.review_pmid,
                             "corrected": corrected, "split": topic.split})
        studies_in_hier: Set[str] = set()
        studies_assigned: Set[str] = set()
        kept_hids = []
        for hid, aspect in enumerate(aspects):
            if aspect.name == UNCATEGORIZED and not include_uncategorized:
                continue
            if annotated_ids is not None and hid not in annotated_ids:
                continue
            totals["annotated_hierarchies"] += 1
            kept_hids.append(hid)

            studies_assigned |= _add_aspect(tax, aspect, hid, topic, accepted[hid], drop[hid])
            for n in aspect.walk():
                for cidx in n.claims:
                    pmid = topic.claim_to_pmid(cidx)
                    if pmid is not None:
                        studies_in_hier.add(pmid)
            for _, cidx in rejected[hid]:
                pmid = topic.claim_to_pmid(cidx)
                if pmid is not None:
                    studies_in_hier.add(pmid)
        if gold_studies == "annotated":
            gold_pmids = studies_in_hier
        elif gold_studies == "assigned":
            gold_pmids = studies_assigned
        else:
            gold_pmids = {s["study_pmid"] for s in topic.studies}
        rows_by_pmid = {s["study_pmid"]: s for s in topic.studies}
        ordered = [s["study_pmid"] for s in topic.studies if s["study_pmid"] in gold_pmids]
        for pmid in ordered:
            pid = study_paper_id(pmid)
            row = rows_by_pmid[pmid]
            paper = corpus.get(pid)
            if paper is None:
                paper = Paper(paper_id=pid, title=" ".join((row.get("study_title") or "").split()),
                              abstract=" ".join((row.get("study_abstract") or "").split()),
                              metadata={"pmid": pmid, "reviews": [], "claims": {}})
                corpus.add(paper)
            if rid not in paper.metadata["reviews"]:
                paper.metadata["reviews"].append(rid)
            paper.metadata["claims"][rid] = row.get("claim", "")
        tax.metadata["hierarchy_ids"] = kept_hids
        gold_trees[rid] = tax
        queries.append(Query(
            query_id=f"chime-{rid}",
            text=topic.title,
            kind=EVIDENCE,
            benchmark=BENCHMARK,
            gold_papers=[study_paper_id(p) for p in ordered],
            metadata={"task": "organization", "style": "review", "review_id": rid,
                      "review_pmid": topic.review_pmid, "topic": topic.title, "split": topic.split,
                      "instruction": INSTRUCTION.format(topic=topic.title), "hierarchy_ids": kept_hids,
                      "corrected": corrected, "gold_taxonomy": f"gold/{rid}.json"},
        ))
    if totals["topics_without_annotations"]:
        log.warning("%d topic(s) have no expert annotation rows; their generated hierarchies are used uncorrected",
                    totals["topics_without_annotations"])
    n_gold = sum(len(q.gold_papers) for q in queries)
    stats = {
        "topics": len(queries),
        "corrected": ann.available,
        "hierarchies": totals["hierarchies"],
        "gold_hierarchies": totals["annotated_hierarchies"],
        "topics_without_annotations": totals["topics_without_annotations"],
        "study_rows": totals["study_rows"],
        "distinct_studies": len(corpus),
        "gold_per_query": round(n_gold / max(1, len(queries)), 2),
        "dropped_edges": totals["dropped_edges"],
        "rejected_assignments": totals["rejected_assignments"],
        "unmatched_annotations": totals["unmatched_annotations"],
        "incoherent_sibling_groups": sum(1 for r in ann.sibling if _label(r, ("agg_label",)) == 0),
        "gold_studies": gold_studies,
        "edge_policy": edge_policy,
        "coverage": {"total": n_gold, "aligned": n_gold, "coverage": 1.0},
        "paper_reported": PAPER_STATS,
    }
    return ChimeData(corpus, queries, gold_trees, stats)
