"""TaxoBench: pooled corpus, gold taxonomies and organization queries (Sec. 5.2-5.3, App. C.2).

Raw data: ``dataset/data.jsonl`` of github.com/KongLongGeFDU/TaxoBench, one
survey per line::

    {"id": 0, "survey": "...", "survey_topic": "...", "gt_paper_count": 57,
     "pdfs": [{"title": "...", "abs": "..."}, ...],
     "gt": {"name": "...", "subtopics": [{"name": "...", "subtopics": [...]},
                                          {"name": "<leaf>", "papers": ["<title>", ...]}]}}

Papers are identified only by title (no ids, authors or years). The paper
uses TaxoBench in two ways:

* **Taxonomy construction** (Sec. 5.2): *"TaxoBench provides 72
  expert-curated computer-science survey taxonomies over 3,044 distinct
  papers (3,815 citations), which we pool into a single corpus and merge
  under a common root."* App. C.2: *"the papers of all surveys are
  de-duplicated by normalized title, and a paper cited by several surveys
  keeps all its memberships. Paper entries are removed from the expert
  trees, so that only concept nodes are compared, and the root of each
  tree, labeled by its survey topic, becomes a child of the common root."*
  -> :func:`build_pooled_corpus`, :func:`merge_gold_taxonomies`.
* **Organization queries** (Sec. 5.3, Deep Research mode): *"organizing
  survey-cited papers into topic hierarchies"*; the pooled corpus (3,044
  papers) is the retrieval corpus, and each survey yields one query whose
  gold papers are the survey's papers and whose gold hierarchy is the
  survey's expert tree -> :func:`build_queries`, :func:`survey_taxonomy`.

Paper ids are derived from the normalized title (``tb-<sha1[:12]>``), so
they are stable across runs and data versions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from scholarpath.data.alignment import PaperAligner, Reference, title_key
from scholarpath.data.corpus import Corpus, read_jsonl, save_queries, write_json
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.types import EVIDENCE, Paper, Query

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

BENCHMARK = "taxobench"
MERGED_ROOT_NAME = "TaxoBench Surveys"
#: Statistics reported in the paper (Sec. 5.2, App. C.2).
PAPER_STATS = {"surveys": 72, "citations": 3815, "distinct_papers": 3044, "merged_concept_nodes": 814}

#: Deep Research instruction attached to each organization query.
INSTRUCTION = ("Retrieve the key papers on the topic \"{topic}\" and organize them into a topic hierarchy, "
               "as in the taxonomy of a survey on this topic.")


def taxobench_paper_id(title: str) -> str:
    """Stable id of a TaxoBench paper: ``tb-`` + SHA-1 of the normalized title (first 12 hex digits)."""
    return "tb-" + hashlib.sha1(title_key(title).encode("utf-8")).hexdigest()[:12]


def read_taxobench(path: PathLike) -> List[Dict[str, Any]]:
    """Read the survey records from ``data.jsonl`` (also accepts the repo root or ``dataset/`` dir, or a JSON list)."""
    path = Path(path)
    if path.is_dir():
        for cand in (path / "dataset" / "data.jsonl", path / "data.jsonl"):
            if cand.exists():
                path = cand
                break
        else:
            raise FileNotFoundError(f"No dataset/data.jsonl under {path}; run "
                                    "`python scripts/download_data.py --only taxobench`")
    if path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
    else:
        records = list(read_jsonl(path))
    for r in records:
        for key in ("id", "pdfs", "gt"):
            if key not in r:
                raise ValueError(f"TaxoBench record without {key!r}: {str(r)[:200]}")
    return records


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def build_pooled_corpus(records: Sequence[Mapping[str, Any]]) -> Tuple[Corpus, Dict[str, str]]:
    """Pool the papers of all surveys, de-duplicated by normalized title.

    Returns the corpus and a map normalized-title -> paper id. The first
    non-empty abstract (in survey order) is kept; ``metadata`` records the
    surveys citing the paper (``"surveys"``) and the raw title variants.
    """
    papers: Dict[str, Paper] = {}
    key_to_pid: Dict[str, str] = {}
    for rec in records:
        sid = rec["id"]
        for pdf in rec.get("pdfs") or []:
            title = " ".join(str(pdf.get("title") or "").split())
            key = title_key(title)
            if not key:
                continue
            pid = key_to_pid.get(key)
            if pid is None:
                pid = taxobench_paper_id(title)
                key_to_pid[key] = pid
                papers[pid] = Paper(paper_id=pid, title=title,
                                    abstract=" ".join(str(pdf.get("abs") or "").split()),
                                    metadata={"surveys": [], "title_variants": []})
            p = papers[pid]
            if sid not in p.metadata["surveys"]:
                p.metadata["surveys"].append(sid)
            if title != p.title and title not in p.metadata["title_variants"]:
                p.metadata["title_variants"].append(title)
            if not p.abstract and pdf.get("abs"):
                p.abstract = " ".join(str(pdf["abs"]).split())
    for p in papers.values():
        if not p.metadata["title_variants"]:
            del p.metadata["title_variants"]
    return Corpus(papers.values(), name="taxobench"), key_to_pid


# ---------------------------------------------------------------------------
# Gold taxonomies
# ---------------------------------------------------------------------------


@dataclass
class _Resolver:
    """Maps leaf paper titles to pooled-corpus ids (exact normalized title, then the aligner)."""

    key_to_pid: Mapping[str, str]
    aligner: Optional[PaperAligner] = None
    unmatched: List[str] = field(default_factory=list)

    def __call__(self, title: str) -> Optional[str]:
        pid = self.key_to_pid.get(title_key(title))
        if pid is None and self.aligner is not None:
            pid = self.aligner.align(Reference(title=title)).paper_id
        if pid is None:
            self.unmatched.append(title)
        return pid


def _add_subtree(tax: Taxonomy, node: Mapping[str, Any], parent: str, resolve: Optional[_Resolver],
                 survey_id: Any, depth: int, max_depth: Optional[int]) -> None:
    """Add ``node``'s children under ``parent`` (recursively), assigning leaf papers."""
    if resolve is not None:
        for title in node.get("papers") or []:
            pid = resolve(str(title))
            if pid is not None:
                tax.assign(pid, parent)
    for child in node.get("subtopics") or []:
        if max_depth is not None and depth + 1 > max_depth:
            # Truncated: papers of the subtree are attached to the deepest kept ancestor.
            if resolve is not None:
                for title in _subtree_papers(child):
                    pid = resolve(title)
                    if pid is not None:
                        tax.assign(pid, parent)
            continue
        cid = tax.add_node(" ".join(str(child.get("name", "")).split()), parent=parent, survey_id=survey_id)
        _add_subtree(tax, child, cid, resolve, survey_id, depth + 1, max_depth)


def _subtree_papers(node: Mapping[str, Any]) -> List[str]:
    out = [str(t) for t in node.get("papers") or []]
    for c in node.get("subtopics") or []:
        out.extend(_subtree_papers(c))
    return out


def survey_taxonomy(record: Mapping[str, Any], key_to_pid: Mapping[str, str], include_papers: bool = True,
                    aligner: Optional[PaperAligner] = None) -> Taxonomy:
    """Gold taxonomy of one survey (facet ``"topic"``); the root is labelled by the survey topic.

    Leaf ``papers`` entries become paper *assignments* (extents) of the
    leaf concept, not nodes; ``include_papers=False`` keeps concept nodes only.
    """
    topic = str(record.get("survey_topic") or record["gt"].get("name") or record.get("survey") or "")
    tax = Taxonomy(facet="topic", root_name=" ".join(topic.split()))
    tax.root.metadata.update({"survey_id": record["id"], "original_root_name": record["gt"].get("name")})
    tax.metadata.update({"benchmark": BENCHMARK, "survey_id": record["id"], "topic": topic})
    resolver = _Resolver(key_to_pid, aligner) if include_papers else None
    _add_subtree(tax, record["gt"], tax.root_id, resolver, record["id"], 0, None)
    if resolver is not None and resolver.unmatched:
        tax.metadata["unmatched_leaf_titles"] = resolver.unmatched
    return tax


def merge_gold_taxonomies(
    records: Sequence[Mapping[str, Any]],
    key_to_pid: Optional[Mapping[str, str]] = None,
    root_name: str = MERGED_ROOT_NAME,
    include_papers: bool = True,
    max_depth: Optional[int] = None,
) -> Taxonomy:
    """Merged expert gold taxonomy for construction evaluation (App. C.2).

    Each survey tree's root, relabelled by its ``survey_topic``, becomes a
    child of a common root; paper entries are not nodes (with
    ``include_papers`` they are kept as extents of the leaf concepts, so a
    paper cited by several surveys belongs to all of them). ``max_depth``
    optionally truncates the merged tree to that many levels below the
    common root (papers of cut subtrees move to the deepest kept ancestor).
    """
    tax = Taxonomy(facet="topic", root_name=root_name,
                   root_definition="Common root of the TaxoBench survey taxonomies.")
    tax.metadata.update({"benchmark": BENCHMARK, "num_surveys": len(records), "max_depth": max_depth})
    resolver = _Resolver(key_to_pid) if (include_papers and key_to_pid is not None) else None
    for rec in records:
        topic = str(rec.get("survey_topic") or rec["gt"].get("name") or "")
        if max_depth is not None and max_depth < 1:
            break
        sid = tax.add_node(" ".join(topic.split()), parent=tax.root_id, survey_id=rec["id"],
                           original_root_name=rec["gt"].get("name"))
        _add_subtree(tax, rec["gt"], sid, resolver, rec["id"], 1, max_depth)
    if resolver is not None and resolver.unmatched:
        tax.metadata["unmatched_leaf_titles"] = resolver.unmatched
    return tax


def taxonomy_stats(tax: Taxonomy) -> Dict[str, Any]:
    """Concept-node statistics (root excluded) as reported in App. C.2."""
    depths: Dict[int, int] = {}
    for nid, n in tax.nodes.items():
        if nid != tax.root_id:
            depths[n.depth] = depths.get(n.depth, 0) + 1
    return {"concept_nodes": len(tax.nodes) - 1, "levels": len(depths), "nodes_per_level": dict(sorted(depths.items())),
            "papers": len(tax.root.papers)}


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def build_queries(records: Sequence[Mapping[str, Any]], key_to_pid: Mapping[str, str]) -> List[Query]:
    """One organization query per survey (Deep Research mode).

    ``text`` is the survey topic (what the agent receives);
    ``metadata = {"task": "organization", "style": "survey", "survey_id", "topic",
    "instruction", "gt_paper_count", "gold_taxonomy"}``; the gold papers are
    the survey's papers in the pooled corpus.
    """
    queries = []
    for rec in records:
        topic = " ".join(str(rec.get("survey_topic") or rec["gt"].get("name") or "").split())
        gold: Dict[str, None] = {}
        for pdf in rec.get("pdfs") or []:
            pid = key_to_pid.get(title_key(str(pdf.get("title") or "")))
            if pid is not None:
                gold[pid] = None
        queries.append(Query(
            query_id=f"taxobench-{rec['id']}",
            text=topic,
            kind=EVIDENCE,
            benchmark=BENCHMARK,
            gold_papers=list(gold),
            metadata={
                "task": "organization",
                "style": "survey",
                "mode": "deep_research",
                "survey_id": rec["id"],
                "topic": topic,
                "survey": rec.get("survey"),
                "instruction": INSTRUCTION.format(topic=topic),
                "gt_paper_count": rec.get("gt_paper_count"),
                "gold_taxonomy": f"gold/{rec['id']}.json",
            },
        ))
    return queries


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


@dataclass
class TaxoBenchData:
    corpus: Corpus
    queries: List[Query]
    survey_taxonomies: Dict[Any, Taxonomy]
    merged: Taxonomy
    stats: Dict[str, Any]

    def save(self, out_dir: PathLike) -> Dict[str, str]:
        """Write ``papers.jsonl.gz``, ``gold_taxonomy.json``, ``gold/<survey_id>.json``, ``queries.jsonl``, ``stats.json``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.corpus.save(out / "papers.jsonl.gz", include_chunks=False)
        self.merged.save(out / "gold_taxonomy.json")
        for sid, tax in self.survey_taxonomies.items():
            tax.save(out / "gold" / f"{sid}.json")
        save_queries(out / "queries.jsonl", self.queries)
        write_json(out / "stats.json", self.stats)
        return {"papers": str(out / "papers.jsonl.gz"), "gold_taxonomy": str(out / "gold_taxonomy.json"),
                "queries": str(out / "queries.jsonl")}


def load_taxobench(path: PathLike, merged_max_depth: Optional[int] = None, limit: Optional[int] = None) -> TaxoBenchData:
    """Build every TaxoBench artefact from the raw ``data.jsonl`` (``limit`` keeps the first surveys)."""
    records = read_taxobench(path)
    if limit is not None:
        records = records[:limit]
    corpus, key_to_pid = build_pooled_corpus(records)
    surveys = {rec["id"]: survey_taxonomy(rec, key_to_pid) for rec in records}
    merged = merge_gold_taxonomies(records, key_to_pid, max_depth=merged_max_depth)
    queries = build_queries(records, key_to_pid)
    n_cit = sum(len(r.get("pdfs") or []) for r in records)
    multi = sum(1 for p in corpus if len(p.metadata.get("surveys", [])) > 1)
    unmatched = sum(len(t.metadata.get("unmatched_leaf_titles", [])) for t in surveys.values())
    gold_total = sum(q.num_gold for q in queries)
    stats = {
        "surveys": len(records),
        "citations": n_cit,
        "distinct_papers": len(corpus),
        "papers_in_several_surveys": multi,
        "unmatched_leaf_titles": unmatched,
        "gold_per_query": round(gold_total / max(1, len(queries)), 2),
        "merged_gold": taxonomy_stats(merged),
        "coverage": {"total": gold_total, "aligned": gold_total, "coverage": 1.0},
        "paper_reported": PAPER_STATS,
    }
    return TaxoBenchData(corpus, queries, surveys, merged, stats)
