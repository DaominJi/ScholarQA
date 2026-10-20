"""ResearchQA ``test_mini`` (Sec. 5.3, App. C.2).

App. C.2: *"ResearchQA provides survey-mined research questions paired with
query-specific rubric items (covering technical explanations, cross-method
comparisons, limitations, and paper citations) and the gold set of papers
cited in the corresponding survey section [91]; we evaluate on its official
test_mini split (776 queries, 5,824 rubric items)."* Gold papers are aligned
to arXiv-Subfields (10,830 of 11,328 survey-cited papers, 95.6%).

Record schema (Hugging Face ``realliyifei/ResearchQA``; the JSON files and
the parquet splits share it)::

    {"id": "174539437936812632-s1", "general_domain": "...", "subdomain": "...", "field": "...",
     "query": "...", "date": "2020-11-11",
     "rubric": [{"rubric_item": "Does the response ...?", "type": ["Citation"],
                 "citation_metadata": {"first_author": "...", "title": "...", "url": "...", "year": 2007}}, ...]}

Conversion to :class:`~scholarpath.types.Query` (benchmark ``"researchqa"``,
kind *evidence*):

* ``rubric``: one :class:`~scholarpath.types.RubricItem` per item, weight 1
  (the official coverage score averages items uniformly) and ``kind`` = the
  item's type list, lower-cased and comma-joined (e.g. ``"citation"`` or
  ``"comparison,example"``); the original type lists are kept in
  ``metadata["rubric_types"]``;
* gold papers: the cited papers of the ``Citation``-type items
  (``citation_metadata``; if a record lacks it -- e.g. third-party
  re-uploads -- the title written into the item, ``"(title: ...)"`` or a
  quoted title, is used), aligned with :class:`PaperAligner`; unaligned
  citations stay in ``unaligned_gold``. An optional ``extra_gold`` mapping
  (query id -> references) adds externally obtained section citations;
* ``metadata``: ``id``, ``domain`` (``general_domain``), ``subdomain``,
  ``field``, ``date`` (zero-padded ISO; the raw value is kept as
  ``raw_date``). The survey date is informational only: as in the paper,
  every ResearchQA query searches the whole arXiv-Subfields corpus, with no
  per-query date cutoff.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from scholarpath.data.alignment import CoverageStats, PaperAligner, Reference, resolve_references
from scholarpath.data.download import read_json_records
from scholarpath.types import EVIDENCE, Query, RubricItem

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

BENCHMARK = "researchqa"
HF_REPO = "realliyifei/ResearchQA"
TEST_MINI_PARQUET = "data/test_mini-00000-of-00001.parquet"
PAPER_STATS = {"queries": 776, "rubric_items": 5824, "cited_papers": 11328, "aligned": 10830, "coverage": 0.956}
RUBRIC_TYPES = ("Citation", "Comparison", "Example", "Impact", "Limitation", "Other")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _to_plain(value: Any) -> Any:
    """numpy / pyarrow containers from parquet rows -> plain Python lists and dicts."""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def read_researchqa(path: PathLike) -> List[Dict[str, Any]]:
    """Read ResearchQA records from a parquet file, a JSON array / JSON Lines file, or a directory holding one.

    A directory is searched for ``test_mini`` files (``data/test_mini-*.parquet``,
    ``test_mini.json[l]``). JSON files may hold JSON Lines (the default output
    of ``datasets.Dataset.to_json``) -- see :func:`~scholarpath.data.download.read_json_records`.
    """
    path = Path(path)
    if path.is_dir():
        cands = sorted(path.rglob("test_mini*.parquet")) + sorted(path.rglob("test_mini*.json*"))
        if not cands:
            raise FileNotFoundError(f"No test_mini parquet/JSON under {path}; run "
                                    "`python scripts/download_data.py --only researchqa`")
        path = cands[0]
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError("Reading the ResearchQA parquet needs pandas + pyarrow (pip install pandas pyarrow), "
                              "or export the split to JSON") from err
        records = [_to_plain(r) for r in pd.read_parquet(path).to_dict(orient="records")]
    else:
        # JSON array or JSON Lines, whatever the extension: `Dataset.to_json` writes JSON Lines to *.json.
        records = [_to_plain(r) for r in read_json_records(path)]
    for r in records:
        if "query" not in r or "rubric" not in r:
            raise ValueError(f"Not a ResearchQA record: {str(r)[:200]}")
    return records


def load_researchqa_hf(split: str = "test_mini", cache_dir: Optional[PathLike] = None) -> List[Dict[str, Any]]:
    """Load a split with ``datasets.load_dataset`` (needs network access to Hugging Face)."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets (or download the parquet with scripts/download_data.py)") from err
    ds = load_dataset(HF_REPO, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    return [_to_plain(dict(r)) for r in ds]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def normalize_date(date: Optional[str]) -> Optional[str]:
    """``"2021-1-12"`` -> ``"2021-01-12"`` (also ``"2021-1"`` -> ``"2021-01-01"``, ``"2021"`` -> ``"2021-01-01"``)."""
    if not date:
        return None
    m = re.match(r"^\s*(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", str(date))
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1)
    return f"{y:04d}-{mo:02d}-{d:02d}"


def item_types(item: Mapping[str, Any]) -> List[str]:
    """The item's type list (the official loader annotates ``str``; the data holds lists)."""
    t = item.get("type")
    if t is None:
        return []
    if isinstance(t, str):
        return [x.strip() for x in re.split(r"[,|;]", t) if x.strip()]
    return [str(x) for x in t]


def _balanced_after(text: str, start: int) -> str:
    """Text from ``start`` up to the ``)`` closing the parenthesis opened just before ``start``."""
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return text[start:]


def titles_from_rubric_text(text: str) -> List[str]:
    """Paper titles written into a citation rubric item, in textual order.

    Handles ``(title: ...)`` with balanced parentheses, title lists
    ``(titles: A (title: X) and "Y")`` and, if no such marker exists, quoted titles.
    """
    quoted = re.compile(r"[\"“]([^\"”]{12,})[\"”]")
    found: List[Tuple[int, str]] = []
    for m in re.finditer(r"\(\s*titles?\s*:\s*", text, flags=re.IGNORECASE):
        inner = _balanced_after(text, m.end())
        if re.search(r"\(\s*titles?\s*:", inner, flags=re.IGNORECASE):
            # A list of titles: nested "(title: ...)" are matched on their own; keep the quoted ones.
            found.extend((m.end() + q.start(1), q.group(1)) for q in quoted.finditer(inner))
        else:
            found.append((m.end(), inner.strip().strip("\"“”'")))
    if not found:
        found = [(q.start(1), q.group(1)) for q in quoted.finditer(text)]
    out = [" ".join(t.split()).rstrip(".") for _, t in sorted(found)]
    return list(dict.fromkeys(t for t in out if len(t) >= 8))


def citation_references(record: Mapping[str, Any], parse_text_fallback: bool = True) -> List[Reference]:
    """References of the ``Citation``-type rubric items of a record."""
    refs: List[Reference] = []
    for item in record.get("rubric") or []:
        if "Citation" not in item_types(item):
            continue
        meta = item.get("citation_metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = None
        if isinstance(meta, Mapping) and any(meta.get(k) for k in ("title", "url")):
            refs.append(Reference(title=meta.get("title"), url=meta.get("url"), year=meta.get("year"),
                                  first_author=meta.get("first_author"), source="citation_metadata"))
        elif parse_text_fallback:
            for t in titles_from_rubric_text(str(item.get("rubric_item", ""))):
                refs.append(Reference(title=t, source="rubric_text"))
    return refs


def record_to_query(record: Mapping[str, Any], aligner: Optional[PaperAligner] = None,
                    extra_refs: Sequence[Reference] = ()) -> Tuple[Query, CoverageStats]:
    """Convert one ResearchQA record (see module docstring)."""
    rubric: List[RubricItem] = []
    types: List[List[str]] = []
    for item in record.get("rubric") or []:
        ts = item_types(item)
        types.append(ts)
        rubric.append(RubricItem(text=str(item.get("rubric_item", "")), weight=1.0,
                                 kind=",".join(t.lower() for t in ts) or "other"))
    refs = citation_references(record) + list(extra_refs)
    res = resolve_references(refs, aligner)
    date = normalize_date(record.get("date"))
    query = Query(
        query_id=str(record["id"]),
        text=str(record["query"]).strip(),
        kind=EVIDENCE,
        benchmark=BENCHMARK,
        gold_papers=res.gold_papers,
        unaligned_gold=res.unaligned_gold,
        rubric=rubric,
        metadata={
            "id": str(record["id"]),
            "domain": record.get("general_domain"),
            "subdomain": record.get("subdomain"),
            "field": record.get("field"),
            "date": date,
            "raw_date": record.get("date"),
            "rubric_types": types,
            "citations": [m.to_dict() for m in res.matches],
        },
    )
    return query, res.stats


def researchqa_to_queries(
    records: Iterable[Mapping[str, Any]],
    aligner: Optional[PaperAligner] = None,
    extra_gold: Optional[Mapping[str, Sequence[Union[Reference, Mapping[str, Any]]]]] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Query], CoverageStats]:
    """Convert records to queries and accumulate alignment coverage (App. C.2)."""
    queries: List[Query] = []
    stats = CoverageStats()
    for i, rec in enumerate(records):
        if limit is not None and i >= limit:
            break
        extra = [r if isinstance(r, Reference) else Reference.from_dict(r)
                 for r in (extra_gold or {}).get(str(rec["id"]), [])]
        q, st = record_to_query(rec, aligner, extra)
        queries.append(q)
        stats.merge(st)
    return queries, stats


def load_researchqa(path: PathLike, aligner: Optional[PaperAligner] = None, limit: Optional[int] = None,
                    extra_gold_path: Optional[PathLike] = None) -> Tuple[List[Query], CoverageStats]:
    """Read ``path`` and convert it; ``extra_gold_path`` is an optional JSON/JSONL of
    ``{"id": ..., "references": [{title, url, arxiv_id, doi, ...}, ...]}`` section citations."""
    extra: Dict[str, List[Dict[str, Any]]] = {}
    if extra_gold_path is not None:
        for r in read_json_records(extra_gold_path):
            extra.setdefault(str(r["id"]), []).extend(r.get("references") or [])
    return researchqa_to_queries(read_researchqa(path), aligner, extra or None, limit)


def rubric_item_count(queries: Sequence[Query]) -> int:
    return sum(len(q.rubric) for q in queries)
