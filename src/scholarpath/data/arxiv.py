"""arXiv metadata: the *arXiv sets* and the *arXiv-Subfields* corpus (Sec. 5.2, App. C.2).

Source: the Kaggle ``Cornell-University/arxiv`` metadata snapshot
``arxiv-metadata-oai-snapshot.json`` -- JSON lines, one object per paper::

    {"id": "0704.0001", "title": "...", "abstract": "...", "categories": "hep-ph",
     "comments": "...", "doi": "...", "versions": [{"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"}, ...],
     "update_date": "2008-11-26", "authors": "...", "authors_parsed": [["Balázs", "C.", ""], ...], ...}

App. C.2: *"The arXiv sets are drawn from the metadata snapshot of [11]. The
primary category of a paper is the first entry of its category list,
withdrawn papers are excluded, and 2,000 papers are sampled uniformly with a
fixed seed from each of the eight categories for the construction
evaluation against FULL, while all 148,400 papers of the eight categories
form the arXiv-Subfields corpus for ResearchQA, ScholarQABench, and the
held-out hypothesis queries."* The eight primary categories (Sec. 5.2) span
seven top-level arXiv groups (astro-ph and physics both belong to Physics).

Conversion to :class:`~scholarpath.types.Paper`:

* ``paper_id`` = ``arxiv_id`` = the identifier without version
  (old-style ids keep their archive prefix, e.g. ``math/0601001``);
* title and abstract are whitespace-normalised (the snapshot keeps the hard
  line breaks of the submission);
* ``categories`` = the space-separated category list, ``primary_category`` =
  its first entry;
* ``authors`` from ``authors_parsed`` ("First Last Suffix"), falling back to
  the raw ``authors`` string;
* ``date`` = creation date of the first version (RFC 2822 -> ISO), ``year``
  its year; ``doi`` the first DOI listed.

**Withdrawn papers** (:func:`is_withdrawn`): arXiv replaces a withdrawn
paper by a new version whose *comments* state the withdrawal (e.g. "This
paper has been withdrawn by the author", "Withdrawn due to an error") and
whose *abstract* is often replaced by the same notice. A record is
withdrawn when its comments match :data:`WITHDRAWN_COMMENT_RE` or its
abstract *starts with* a withdrawal notice (:data:`WITHDRAWN_ABSTRACT_RE`).
Abstracts merely mentioning the word (e.g. "drugs withdrawn from the
market") are not affected.

All builders stream the snapshot once and keep only what they return in
memory; :func:`write_arxiv_corpora` streams the arXiv-Subfields corpus
straight to disk.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import io
import json
import logging
import re
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import IO, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from scholarpath.data.alignment import parse_arxiv_id, parse_doi, split_arxiv_version
from scholarpath.data.corpus import Corpus
from scholarpath.types import Paper

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: Primary categories of the eight arXiv sets (Sec. 5.2).
ARXIV_SET_CATEGORIES: Tuple[str, ...] = (
    "cs.DB", "math.OC", "astro-ph.GA", "physics.med-ph", "q-bio.PE", "stat.AP", "eess.SP", "econ.EM",
)

#: arXiv archive -> top-level group (https://arxiv.org/category_taxonomy).
ARXIV_GROUPS: Dict[str, str] = {
    "cs": "Computer Science", "econ": "Economics", "eess": "Electrical Engineering and Systems Science",
    "math": "Mathematics", "q-bio": "Quantitative Biology", "q-fin": "Quantitative Finance",
    "stat": "Statistics", "astro-ph": "Physics", "cond-mat": "Physics", "gr-qc": "Physics",
    "hep-ex": "Physics", "hep-lat": "Physics", "hep-ph": "Physics", "hep-th": "Physics",
    "math-ph": "Physics", "nlin": "Physics", "nucl-ex": "Physics", "nucl-th": "Physics",
    "physics": "Physics", "quant-ph": "Physics",
}

#: Size of the arXiv-Subfields corpus in the paper's snapshot (App. C.2).
PAPER_SUBFIELDS_SIZE = 148_400

KAGGLE_DATASET = "Cornell-University/arxiv"
SNAPSHOT_NAME = "arxiv-metadata-oai-snapshot.json"

WITHDRAWN_COMMENT_RE = re.compile(
    r"(?:\b(?:has|have)\s+been\s+withdrawn\b|\b(?:is|was|being)\s+withdrawn\b|\bwithdrawn\s+by\b"
    r"|^\W*withdrawn\b|\b(?:paper|article|submission|manuscript|preprint)\s+(?:is\s+)?withdrawn\b"
    r"|\bwithdrawn\s+(?:due|because|for|as|since|pending)\b|\bwithdrawal\s+(?:of\s+(?:this|the)\s+)?"
    r"(?:paper|article|submission|manuscript|preprint)\b|\bwithdraw(?:n|s)?\s+this\s+(?:paper|article|submission))",
    re.IGNORECASE,
)
WITHDRAWN_ABSTRACT_RE = re.compile(
    r"^\W*(?:withdrawn\b|this\s+(?:paper|article|manuscript|submission|preprint|work|version)\s+"
    r"(?:has\s+been|have\s+been|is|was|is\s+being)\s+(?:permanently\s+)?withdrawn\b|"
    r"(?:paper|article|submission)\s+(?:has\s+been\s+)?withdrawn\b|the\s+authors?\s+(?:has|have)\s+withdrawn\b)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Reading the snapshot
# ---------------------------------------------------------------------------


@contextmanager
def open_snapshot(path: PathLike) -> Iterator[IO[str]]:
    """Open the snapshot as text, transparently handling ``.gz`` and ``.zip`` (Kaggle's ``arxiv.zip``)."""
    path = Path(path)
    if path.is_dir():
        candidates = [path / SNAPSHOT_NAME, path / (SNAPSHOT_NAME + ".gz"), path / "arxiv.zip"]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise FileNotFoundError(f"No {SNAPSHOT_NAME}[.gz] or arxiv.zip in {path}")
        path = found
    if not path.exists():
        raise FileNotFoundError(
            f"arXiv snapshot not found: {path}. Download it with "
            f"`python scripts/download_data.py --only arxiv` (Kaggle dataset {KAGGLE_DATASET})."
        )
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield f
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.endswith((".json", ".jsonl"))]
            if not members:
                raise ValueError(f"{path} contains no .json member")
            member = SNAPSHOT_NAME if SNAPSHOT_NAME in members else members[0]
            with zf.open(member) as raw:
                yield io.TextIOWrapper(raw, encoding="utf-8")
    else:
        with open(path, "r", encoding="utf-8") as f:
            yield f


def iter_snapshot_records(path: PathLike, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Stream raw snapshot records (one JSON object per line); malformed lines are skipped with a warning."""
    with open_snapshot(path) as f:
        n = 0
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed snapshot line %d", lineno)
                continue
            yield rec
            n += 1
            if limit is not None and n >= limit:
                return


# ---------------------------------------------------------------------------
# Record conversion
# ---------------------------------------------------------------------------


def normalize_whitespace(text: Optional[str]) -> str:
    return " ".join((text or "").split())


def is_withdrawn(record: Mapping[str, Any]) -> bool:
    """Whether a snapshot record is a withdrawn paper (see module docstring for the rule)."""
    comments = normalize_whitespace(record.get("comments"))
    if comments and WITHDRAWN_COMMENT_RE.search(comments):
        return True
    abstract = normalize_whitespace(record.get("abstract"))
    return bool(abstract and WITHDRAWN_ABSTRACT_RE.match(abstract))


def first_version_date(record: Mapping[str, Any]) -> Optional[str]:
    """ISO date of the first version (``versions[0].created``, RFC 2822).

    Falls back to the month encoded in a new-style id (``YYMM`` -> first of
    the month) and finally to ``update_date``.
    """
    versions = record.get("versions") or []
    dated = []
    for v in versions:
        created = v.get("created") if isinstance(v, Mapping) else None
        if not created:
            continue
        try:
            dt = parsedate_to_datetime(created)
        except (TypeError, ValueError, IndexError):
            continue
        m = re.match(r"^v?(\d+)$", str(v.get("version", "")).strip())
        dated.append((int(m.group(1)) if m else 10 ** 6, dt))
    if dated:
        dated.sort(key=lambda x: (x[0], x[1]))
        return dated[0][1].date().isoformat()
    aid = str(record.get("id", ""))
    m = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", aid) or re.match(r"^[a-z\-]+(?:\.[A-Za-z]{2})?/(\d{2})(\d{2})\d{3}$", aid)
    if m:
        yy, mm = int(m.group(1)), int(m.group(2))
        year = 1900 + yy if yy >= 91 else 2000 + yy
        return f"{year:04d}-{mm:02d}-01"
    upd = record.get("update_date")
    return str(upd) if upd else None


def parse_authors(record: Mapping[str, Any]) -> List[str]:
    """Author names "First Last [Suffix]" from ``authors_parsed`` (fallback: the ``authors`` string)."""
    parsed = record.get("authors_parsed") or []
    out: List[str] = []
    for entry in parsed:
        if not entry:
            continue
        parts = [normalize_whitespace(x) for x in list(entry) + ["", "", ""]][:3]
        last, first, suffix = parts
        name = " ".join(x for x in (first, last, suffix) if x)
        if name:
            out.append(name)
    if out:
        return out
    raw = normalize_whitespace(record.get("authors"))
    if not raw:
        return []
    return [a.strip() for a in re.split(r",\s*|\s+and\s+", raw) if a.strip()]


def record_to_paper(record: Mapping[str, Any], keep_metadata: bool = True) -> Paper:
    """Convert one snapshot record to a :class:`Paper` (see module docstring)."""
    raw_id = str(record.get("id", "")).strip()
    base, _ = split_arxiv_version(raw_id)
    arxiv_id = parse_arxiv_id(base) or base
    categories = normalize_whitespace(record.get("categories")).split()
    date = first_version_date(record)
    year = int(date[:4]) if date else None
    raw_dois = normalize_whitespace(record.get("doi")).split()
    doi = raw_dois[0] if raw_dois and parse_doi(raw_dois[0]) else None
    md: Dict[str, Any] = {}
    if keep_metadata:
        versions = record.get("versions") or []
        md = {
            "num_versions": len(versions),
            "latest_version": (versions[-1].get("version") if versions and isinstance(versions[-1], Mapping) else None),
            "update_date": record.get("update_date"),
        }
        if record.get("journal-ref"):
            md["journal_ref"] = normalize_whitespace(record.get("journal-ref"))
        if record.get("license"):
            md["license"] = record.get("license")
        md = {k: v for k, v in md.items() if v is not None}
    return Paper(
        paper_id=arxiv_id,
        title=normalize_whitespace(record.get("title")),
        abstract=normalize_whitespace(record.get("abstract")),
        year=year,
        date=date,
        categories=categories,
        primary_category=categories[0] if categories else None,
        authors=parse_authors(record),
        arxiv_id=arxiv_id,
        doi=doi,
        metadata=md,
    )


def primary_category(record: Mapping[str, Any]) -> Optional[str]:
    cats = normalize_whitespace(record.get("categories")).split()
    return cats[0] if cats else None


def top_level_group(category: str) -> str:
    """Top-level arXiv group of a category (``"astro-ph.GA"`` -> ``"Physics"``)."""
    archive = category.split(".", 1)[0]
    return ARXIV_GROUPS.get(archive, archive)


# ---------------------------------------------------------------------------
# Streaming filters
# ---------------------------------------------------------------------------


@dataclass
class ArxivScanStats:
    """Counts collected while streaming the snapshot."""

    records: int = 0
    withdrawn: int = 0
    after_cutoff: int = 0
    selected: int = 0
    per_category: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"records": self.records, "withdrawn_in_categories": self.withdrawn,
                "after_cutoff": self.after_cutoff, "selected": self.selected,
                "per_category": dict(sorted(self.per_category.items()))}


def _category_of(record: Mapping[str, Any], wanted: Optional[frozenset], match: str) -> Optional[str]:
    cats = normalize_whitespace(record.get("categories")).split()
    if not cats:
        return None
    if wanted is None:
        return cats[0]
    if match == "primary":
        return cats[0] if cats[0] in wanted else None
    if match == "any":
        return next((c for c in cats if c in wanted), None)
    raise ValueError(f"match must be 'primary' or 'any', got {match!r}")


def iter_arxiv_papers(
    snapshot: PathLike,
    categories: Optional[Sequence[str]] = ARXIV_SET_CATEGORIES,
    match: str = "primary",
    until: Optional[str] = None,
    include_withdrawn: bool = False,
    limit: Optional[int] = None,
    stats: Optional[ArxivScanStats] = None,
) -> Iterator[Tuple[str, Paper]]:
    """Stream ``(category, Paper)`` for the selected papers of the snapshot.

    ``categories=None`` keeps every paper (category = primary category).
    ``match="primary"`` (the paper's definition) selects papers whose first
    category is listed; ``"any"`` papers listing any of them (the reported
    category is then the first listed one that matches). ``until`` is an
    inclusive ISO cutoff on the first-version date (``None`` = whole snapshot).
    ``limit`` bounds the number of *raw records* read (for quick runs).
    """
    wanted = frozenset(categories) if categories is not None else None
    stats = stats if stats is not None else ArxivScanStats()
    for rec in iter_snapshot_records(snapshot, limit=limit):
        stats.records += 1
        cat = _category_of(rec, wanted, match)
        if cat is None:
            continue
        if not include_withdrawn and is_withdrawn(rec):
            stats.withdrawn += 1
            continue
        paper = record_to_paper(rec)
        if until is not None and (paper.date is None or paper.date > until):
            stats.after_cutoff += 1
            continue
        stats.selected += 1
        stats.per_category[cat] = stats.per_category.get(cat, 0) + 1
        yield cat, paper


# ---------------------------------------------------------------------------
# arXiv sets (uniform fixed-seed samples) and arXiv-Subfields
# ---------------------------------------------------------------------------


def sample_key(paper_id: str, seed: int) -> int:
    """Seeded pseudo-random priority of a paper (64-bit BLAKE2b of ``"<seed>:<paper_id>"``)."""
    h = hashlib.blake2b(f"{seed}:{paper_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big")


class UniformSampler:
    """Fixed-seed uniform sample of ``n`` items from a stream (bottom-k by seeded hash).

    Every paper gets an independent pseudo-random priority
    :func:`sample_key`; the ``n`` papers with the smallest priorities form a
    uniformly random subset of size ``n`` (without replacement). Unlike
    reservoir sampling, the result depends only on the seed and the *set* of
    candidate ids, not on the order of the snapshot, so it is reproducible
    across snapshot re-orderings. Memory is O(n).
    """

    def __init__(self, n: int, seed: int = 0):
        self.n = int(n)
        self.seed = int(seed)
        self._heap: List[Tuple[int, str, Paper]] = []  # max-heap via negated keys

    def offer(self, paper: Paper) -> None:
        key = sample_key(paper.paper_id, self.seed)
        item = (-key, paper.paper_id, paper)
        if len(self._heap) < self.n:
            heapq.heappush(self._heap, item)
        elif -self._heap[0][0] > key:
            heapq.heapreplace(self._heap, item)

    def __len__(self) -> int:
        return len(self._heap)

    def result(self) -> List[Paper]:
        """The sample, sorted by paper id."""
        return sorted((p for _, _, p in self._heap), key=lambda p: p.paper_id)


def build_arxiv_sets(
    snapshot: PathLike,
    categories: Sequence[str] = ARXIV_SET_CATEGORIES,
    n_per_set: int = 2000,
    seed: int = 0,
    until: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Corpus]:
    """The arXiv sets: ``n_per_set`` papers sampled uniformly (fixed seed) per primary category."""
    sets, _ = _scan(snapshot, categories, n_per_set, seed, until, limit, subfields_sink=None)
    return sets


def build_subfields_corpus(
    snapshot: PathLike,
    categories: Sequence[str] = ARXIV_SET_CATEGORIES,
    until: Optional[str] = None,
    limit: Optional[int] = None,
) -> Corpus:
    """arXiv-Subfields: all non-withdrawn papers whose primary category is one of ``categories``.

    Holds the corpus in memory (~148K papers for the paper's snapshot); use
    :func:`write_arxiv_corpora` to stream it to disk instead.
    """
    corpus = Corpus(name="arxiv_subfields")
    for _, paper in iter_arxiv_papers(snapshot, categories, until=until, limit=limit):
        corpus.add(paper)
    return corpus


@dataclass
class ArxivBuildReport:
    """Summary of :func:`write_arxiv_corpora`."""

    sets: Dict[str, int]
    subfields_size: int
    scan: ArxivScanStats
    seed: int
    until: Optional[str]
    paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        groups = sorted({top_level_group(c) for c in self.sets})
        return {"sets": self.sets, "subfields_size": self.subfields_size,
                "paper_subfields_size": PAPER_SUBFIELDS_SIZE, "top_level_groups": groups,
                "seed": self.seed, "until": self.until, "scan": self.scan.to_dict(), "paths": self.paths}


def _scan(
    snapshot: PathLike,
    categories: Sequence[str],
    n_per_set: int,
    seed: int,
    until: Optional[str],
    limit: Optional[int],
    subfields_sink: Optional[Any],
) -> Tuple[Dict[str, Corpus], ArxivScanStats]:
    samplers = {c: UniformSampler(n_per_set, seed) for c in categories}
    stats = ArxivScanStats()
    for cat, paper in iter_arxiv_papers(snapshot, categories, until=until, limit=limit, stats=stats):
        samplers[cat].offer(paper)
        if subfields_sink is not None:
            subfields_sink(paper)
    sets = {}
    for cat, sampler in samplers.items():
        if len(sampler) < n_per_set:
            log.warning("Category %s has only %d eligible papers (< %d)", cat, len(sampler), n_per_set)
        sets[cat] = Corpus(sampler.result(), name=f"arxiv_set_{cat}")
    return sets, stats


def arxiv_set_filename(category: str) -> str:
    """File name of an arXiv set (``cs.DB`` -> ``cs.DB.jsonl.gz``)."""
    return f"{category}.jsonl.gz"


def write_arxiv_corpora(
    snapshot: PathLike,
    out_dir: PathLike,
    categories: Sequence[str] = ARXIV_SET_CATEGORIES,
    n_per_set: int = 2000,
    seed: int = 0,
    until: Optional[str] = None,
    limit: Optional[int] = None,
    write_subfields: bool = True,
) -> ArxivBuildReport:
    """Build both corpora in a single streaming pass and write them under ``out_dir``.

    Layout (docs/ARCHITECTURE.md, Sec. 5)::

        out_dir/arxiv_sets/<category>.jsonl.gz      # n_per_set papers each
        out_dir/arxiv_subfields/papers.jsonl.gz     # all papers of the categories
        out_dir/arxiv_subfields/stats.json

    The subfields corpus is written while streaming; only the samples are kept in memory.
    """
    out_dir = Path(out_dir)
    sub_path = out_dir / "arxiv_subfields" / "papers.jsonl.gz"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = sub_path.with_name(sub_path.name + ".tmp")
    count = [0]
    handle = gzip.open(tmp_path, "wt", encoding="utf-8") if write_subfields else None

    def sink(paper: Paper) -> None:
        assert handle is not None
        handle.write(json.dumps(paper.to_dict(include_chunks=False), ensure_ascii=False) + "\n")
        count[0] += 1

    try:
        sets, stats = _scan(snapshot, categories, n_per_set, seed, until, limit,
                            subfields_sink=sink if write_subfields else None)
    finally:
        if handle is not None:
            handle.close()
    paths: Dict[str, str] = {}
    if write_subfields:
        tmp_path.replace(sub_path)
        paths["arxiv_subfields"] = str(sub_path)
    for cat, corpus in sets.items():
        p = out_dir / "arxiv_sets" / arxiv_set_filename(cat)
        corpus.save(p, include_chunks=False)
        paths[f"arxiv_sets/{cat}"] = str(p)
    report = ArxivBuildReport(
        sets={c: len(s) for c, s in sets.items()},
        subfields_size=count[0] if write_subfields else stats.selected,
        scan=stats, seed=seed, until=until, paths=paths,
    )
    stats_path = out_dir / "arxiv_subfields" / "stats.json"
    stats_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def load_arxiv_sets(directory: PathLike, categories: Sequence[str] = ARXIV_SET_CATEGORIES) -> Dict[str, Corpus]:
    """Load the processed arXiv sets written by :func:`write_arxiv_corpora`."""
    directory = Path(directory)
    out = {}
    for cat in categories:
        path = directory / arxiv_set_filename(cat)
        if path.exists():
            out[cat] = Corpus.load(path, name=f"arxiv_set_{cat}")
    return out
