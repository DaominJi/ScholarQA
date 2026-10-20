"""Aligning external references to corpus papers (Appendix C.2, "Coverage").

Every benchmark cites its gold papers in its own way: ResearchQA through the
``citation_metadata`` of citation rubric items (title, url, year, first
author), ScholarQA-Multi through Semantic Scholar URLs and titles, TaxoBench
through titles only. Appendix C.2: *"Gold papers are aligned to the target
corpora by arXiv ID, Semantic Scholar ID, or normalized title ... unaligned
citations are retained in the denominator so that reported metrics reflect
end-to-end corpus coverage."*

:class:`PaperAligner` implements that cascade over a :class:`Corpus`:

1. **arXiv id** (parsed from ids, ``arXiv:`` strings, abs/pdf URLs and
   ``10.48550/arXiv.*`` DOIs; the version suffix is stripped);
2. **Semantic Scholar id** -- the 40-hex ``paperId`` (``semanticscholar.org/paper/...``
   URLs) or the numeric ``CorpusId``; corpus papers carry them in
   ``Paper.s2_id`` / ``Paper.metadata["s2_corpus_id"]`` (see :func:`attach_s2_ids`);
3. **DOI** and **PMID** (exact identifiers, used when both sides carry them);
4. **normalized title**, exact (:func:`scholarpath.text.normalize_title`; LaTeX
   math is additionally unwrapped so that ``$k$-means`` equals ``k-means``).

These are the paper's rules and the default. An optional, **opt-in** fifth
stage (``fuzzy=True``) matches near-identical titles (typos, punctuation,
British/American spelling): ``difflib`` ratio ``>= fuzzy_threshold`` (0.92),
*and* identical version tokens (numbers, roman numerals: "Qwen2" never
matches "Qwen", nor "... II" its first part), *and* every differing word a
spelling variant of a word of the other title ("Non-Autoregressive" never
matches "Semi-Autoregressive"), *and* -- when both sides know them -- years
within ``fuzzy_year_tolerance`` and a shared first-author surname. There is deliberately no containment rule: a citation of a paper
missing from the corpus must stay unaligned rather than match a longer
title that contains it ("Attention is all you need" vs. "Attention Is All
You Need in Speech Separation"). Fuzzy matches are counted separately
(``by_method["fuzzy_title"]``). Candidates are found by *blocking* on the
rarest title tokens through an inverted index, so the cost per reference is
independent of the corpus size (148,400 papers for arXiv-Subfields).

:meth:`PaperAligner.resolve` turns the references of one query into
``gold_papers`` (aligned, de-duplicated corpus ids) and ``unaligned_gold``
(stable keys of the references that were not found, which stay in the
denominator of Recall@B / NDCG@B through :attr:`Query.num_gold`), and
:class:`CoverageStats` accumulates the coverage numbers reported in App. C.2.

:class:`QuoteAttributor` attributes verbatim evidence *quotes* (ScholarQA-CS
rubric evidence) to the texts that contain them -- corpus abstracts and
full-text chunks, or system answers -- using word 5-gram shingles.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union
from urllib.parse import unquote

from scholarpath.data.corpus import Corpus, read_jsonl
from scholarpath.text import STOPWORDS, normalize_title
from scholarpath.types import Paper

# ---------------------------------------------------------------------------
# Identifier parsing
# ---------------------------------------------------------------------------

# New-style arXiv ids (April 2007 on): YYMM.NNNN (to 2014) / YYMM.NNNNN (2015 on).
_ARXIV_NEW_RE = re.compile(r"(?<![\w])(?<!\d\.)(\d{2}(?:0[1-9]|1[0-2])\.\d{4,5})(?:v(\d+))?(?![\d])")
# Old-style ids: archive[.SUBJ]/YYMMNNN, e.g. hep-th/9901001, math.GT/0309136.
_ARXIV_OLD_RE = re.compile(
    r"(?<![\w-])([a-z]+(?:-[a-z]+)?)(?:\.[A-Za-z]{2})?/(\d{2}(?:0[1-9]|1[0-2])\d{3})(?:v(\d+))?(?!\d)"
)
_ARXIV_CONTEXT_RE = re.compile(r"arxiv|^\s*\d{4}\.\d{4,5}(v\d+)?\s*$|^\s*[a-z-]+(\.[A-Za-z]{2})?/\d{7}", re.I)
_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"'<>]+)", re.I)
_URL_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|www\.|doi\.org/|dx\.doi\.org/)", re.I)
# Publisher landing-page / file suffixes appended to DOIs in URLs (".pdf", "/full", "v1.full", ...).
_DOI_SUFFIX_RE = re.compile(
    r"(?:[./](?:full|abstract|pdf|epdf|fulltext|full-text|summary|meta|short|html|xml|supplementary))+$", re.I)
_PREPRINT_DOI_VERSION_RE = re.compile(r"^(10\.1101/\d{4}\.\d{2}\.\d{2}\.\d+)v\d+$")  # bioRxiv / medRxiv
_S2_URL_RE = re.compile(r"semanticscholar\.org/paper/(?:[^/?#]+/)?([0-9a-f]{40})", re.I)
_S2_CORPUS_RE = re.compile(r"(?:corpus_?id[:=/\s]*|semanticscholar\.org/p/)(\d{3,12})", re.I)
_PMID_RE = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|pmid[:\s]*)(\d{4,9})", re.I)

#: Archives of old-style arXiv identifiers (used to reject false positives such as "and/0301001").
OLD_STYLE_ARCHIVES = frozenset(
    """acc-phys adap-org alg-geom ao-sci astro-ph atom-ph bayes-an chao-dyn chem-ph cmp-lg comp-gas cond-mat cs
    dg-ga funct-an gr-qc hep-ex hep-lat hep-ph hep-th math math-ph mtrl-th nlin nucl-ex nucl-th patt-sol physics
    plasm-ph q-alg q-bio quant-ph solv-int supr-con""".split()
)


def split_arxiv_version(arxiv_id: str) -> Tuple[str, Optional[int]]:
    """``"2101.00001v3"`` -> ``("2101.00001", 3)``; ids without a version give ``None``."""
    m = re.match(r"^(.*?\d)v(\d+)$", arxiv_id.strip())
    if m:
        return m.group(1), int(m.group(2))
    return arxiv_id.strip(), None


def parse_arxiv_id(text: Optional[str], strict: bool = False) -> Optional[str]:
    """Extract a version-less arXiv identifier from an id, URL or free text.

    Handles ``2101.00001``, ``2101.00001v2``, ``arXiv:2101.00001``,
    ``https://arxiv.org/abs/2101.00001v1``, ``.../pdf/2101.00001v1.pdf``,
    ``10.48550/arXiv.2101.00001`` and old-style ids (``hep-th/9901001v2``,
    ``math.GT/0309136`` -> ``math/0309136``; the subject class is not part of
    the canonical identifier). With ``strict`` the text must look like an
    arXiv reference (contain "arxiv" or be a bare identifier) -- this avoids
    reading arbitrary ``dddd.dddd`` numbers in generic URLs as arXiv ids.
    """
    if not text:
        return None
    s = str(text).strip()
    if strict and not _ARXIV_CONTEXT_RE.search(s):
        return None
    m = _ARXIV_NEW_RE.search(s)
    if m:
        return m.group(1)
    for m in _ARXIV_OLD_RE.finditer(s):
        if m.group(1).lower() in OLD_STYLE_ARCHIVES:
            return f"{m.group(1).lower()}/{m.group(2)}"
    return None


def parse_doi(text: Optional[str]) -> Optional[str]:
    """Extract a lower-cased DOI (``10.xxxx/...``) from a DOI string or URL.

    From URLs, the query string / fragment is cut, percent-escapes are
    decoded, and publisher view or file suffixes (``.pdf``, ``/full``,
    ``/abstract``, ``.full``, ...) as well as bioRxiv/medRxiv version
    suffixes (``v1``) are removed, so that the same paper always gets the
    same DOI key::

        https://link.springer.com/content/pdf/10.1007/s10994-020-05000-1.pdf -> 10.1007/s10994-020-05000-1
        https://www.biorxiv.org/content/10.1101/2020.01.01.123456v1.full    -> 10.1101/2020.01.01.123456
    """
    if not text:
        return None
    s = str(text).strip()
    m = _DOI_RE.search(s)
    if not m and "%2f" in s.lower():  # percent-encoded slash, e.g. https://doi.org/10.1000%2Fabc
        s = unquote(s)
        m = _DOI_RE.search(s)
    if not m:
        return None
    doi = m.group(1)
    if _URL_RE.match(s):
        doi = unquote(re.split(r"[?#&]", doi, maxsplit=1)[0])
    doi = doi.rstrip(".,;:)]}")
    doi = _DOI_SUFFIX_RE.sub("", doi).rstrip(".,;:)]}/").lower()
    doi = _PREPRINT_DOI_VERSION_RE.sub(r"\1", doi)
    return doi if "/" in doi and not doi.endswith("/") else None


def parse_s2_id(text: Optional[str]) -> Optional[str]:
    """Extract a 40-hex Semantic Scholar ``paperId`` from a URL or bare id."""
    if not text:
        return None
    s = str(text).strip()
    m = _S2_URL_RE.search(s)
    if m:
        return m.group(1).lower()
    if re.fullmatch(r"[0-9a-fA-F]{40}", s):
        return s.lower()
    return None


def parse_s2_corpus_id(text: Optional[Union[str, int]]) -> Optional[str]:
    """Extract a numeric Semantic Scholar ``CorpusId`` (``CorpusId:123``, ``/p/123`` or a bare integer)."""
    if text is None or text == "":
        return None
    if isinstance(text, float):  # e.g. parquet / pandas columns
        return None if math.isnan(text) else str(int(text))
    if isinstance(text, int):
        return str(text)
    s = str(text).strip()
    if s.isdigit():
        return s
    m = _S2_CORPUS_RE.search(s)
    return m.group(1) if m else None


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip()
    return s or None


# ---------------------------------------------------------------------------
# Title keys
# ---------------------------------------------------------------------------

_MATH_RE = re.compile(r"\$+([^$]*)\$+")
_TEX_CMD_RE = re.compile(r"\\([A-Za-z]+)\s*")


def _unwrap_math(title: str) -> str:
    """Keep the content of inline math (``$k$-means`` -> ``k-means``, ``$\\alpha$`` -> ``alpha``)."""

    def repl(m: "re.Match[str]") -> str:
        inner = _TEX_CMD_RE.sub(lambda c: "" if c.group(1) in {"mathcal", "mathbf", "mathrm", "mathbb",
                                                                 "boldsymbol", "text", "textrm", "rm", "bf",
                                                                 "it", "cal", "left", "right"} else c.group(1),
                                m.group(1))
        return re.sub(r"[{}^_]", "", inner)

    return _MATH_RE.sub(repl, title)


def title_keys(title: Optional[str]) -> List[str]:
    """Normalized-title keys of a title (the foundation normal form, plus a math-unwrapped variant)."""
    if not title:
        return []
    keys = [normalize_title(title)]
    if "$" in title:
        keys.append(normalize_title(_unwrap_math(title)))
    return [k for i, k in enumerate(keys) if k and k not in keys[:i]]


def title_key(title: Optional[str]) -> str:
    """The primary normalized-title key (empty string for missing titles)."""
    keys = title_keys(title)
    return keys[-1] if keys else ""


#: Roman numerals that number parts or versions of a work ("... II"); single letters are too ambiguous.
_ROMAN_NUMERALS = frozenset("ii iii iv vi vii viii ix xi xii xiii xiv xv".split())


def version_tokens(normalized_title: str) -> FrozenSet[str]:
    """Tokens of a normalized title that distinguish versions or parts of a work.

    Tokens containing a digit (``qwen2``, ``3``, ``v2``) and multi-letter roman
    numerals (``ii``, ``iv``). Two titles whose version tokens differ name
    different works even when their strings are almost identical ("Qwen2
    Technical Report" vs. "Qwen Technical Report").
    """
    return frozenset(t for t in normalized_title.split() if t in _ROMAN_NUMERALS or any(c.isdigit() for c in t))


def titles_compatible(a: str, b: str, min_token_ratio: float = 0.75) -> bool:
    """Whether two *normalized* titles can be spelling variants of one title.

    Requires identical :func:`version_tokens` and that every content word
    occurring in only one title has a spelling variant (``difflib`` ratio
    ``>= min_token_ratio``) among the words occurring only in the other one:
    "estimaton"/"estimation" and "optimisation"/"optimization" qualify,
    "Non-Autoregressive"/"Semi-Autoregressive" do not. Titles that are equal
    after removing spaces ("multi task" / "multitask") are compatible.
    """
    if a.replace(" ", "") == b.replace(" ", ""):
        return True
    if version_tokens(a) != version_tokens(b):
        return False
    ta, tb = set(a.split()), set(b.split())
    only_a = {t for t in ta - tb if t not in STOPWORDS}
    only_b = {t for t in tb - ta if t not in STOPWORDS}
    for x, others in [(x, only_b) for x in only_a] + [(y, only_a) for y in only_b]:
        if not any(SequenceMatcher(None, x, y, autojunk=False).ratio() >= min_token_ratio for y in others):
            return False
    return True


def fuzzy_title_score(a: str, b: str) -> float:
    """``difflib`` similarity ratio of two *normalized* titles (1.0 for identical titles).

    Used by the opt-in fuzzy stage of :class:`PaperAligner` together with
    :func:`titles_compatible`; there is no containment shortcut (module docstring).
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def surname(name: Optional[str]) -> Optional[str]:
    """Lower-cased ASCII surname of an author name ("Smith, J." / "John Smith" / "J. Smith" -> "smith")."""
    if not name:
        return None
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    inverted = "," in name  # "Last, First"
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", name.split(",")[0] if inverted else name)
    tokens = [t for t in tokens if len(t) > 1] or tokens  # skip initials
    if not tokens:
        return None
    return re.sub(r"[^a-z]", "", tokens[0 if inverted else -1].lower()) or None


# ---------------------------------------------------------------------------
# References and matches
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """An external citation to be aligned to the corpus.

    Identifiers missing from the explicit fields are parsed from ``url``
    (arXiv abs/pdf links, DOIs, Semantic Scholar paper links, PubMed links).
    """

    title: Optional[str] = None
    arxiv_id: Optional[str] = None
    s2_id: Optional[str] = None  # 40-hex Semantic Scholar paperId
    s2_corpus_id: Optional[str] = None  # numeric Semantic Scholar CorpusId
    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None
    year: Optional[int] = None
    first_author: Optional[str] = None
    source: str = ""  # provenance, e.g. "rubric", "ctxs"

    def __post_init__(self) -> None:
        self.title = _clean(self.title)
        self.url = _clean(self.url)
        self.arxiv_id = parse_arxiv_id(_clean(self.arxiv_id)) or parse_arxiv_id(self.url, strict=True)
        self.s2_id = parse_s2_id(_clean(self.s2_id)) or parse_s2_id(self.url)
        self.s2_corpus_id = parse_s2_corpus_id(_clean(self.s2_corpus_id)) or (
            parse_s2_corpus_id(self.url) if self.url and "semanticscholar" in self.url.lower() else None)
        self.doi = parse_doi(_clean(self.doi)) or parse_doi(self.url)
        if self.doi and self.doi.startswith("10.48550/arxiv.") and not self.arxiv_id:
            self.arxiv_id = parse_arxiv_id(self.doi)
        pmid = _clean(self.pmid)
        if pmid is None and self.url:
            m = _PMID_RE.search(self.url)
            pmid = m.group(1) if m else None
        self.pmid = pmid
        if self.year is not None:
            try:
                self.year = int(str(self.year)[:4])
            except (TypeError, ValueError):
                self.year = None
        self.first_author = _clean(self.first_author)
        # Some records carry a bare CorpusId in the title field (e.g. ScholarQA-Multi).
        if self.title and self.title.isdigit() and self.s2_corpus_id is None:
            self.s2_corpus_id, self.title = self.title, None

    @property
    def key(self) -> str:
        """Stable identity used for de-duplication and as ``Query.unaligned_gold`` entry."""
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.s2_id:
            return f"s2:{self.s2_id}"
        if self.s2_corpus_id:
            return f"s2corpus:{self.s2_corpus_id}"
        if self.doi:
            return f"doi:{self.doi}"
        if self.pmid:
            return f"pmid:{self.pmid}"
        tk = title_key(self.title)
        if tk:
            return f"title:{tk}"
        if self.url:
            return f"url:{self.url}"
        return "unknown:" + hashlib.sha1(repr(asdict(self)).encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Reference":
        names = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class AlignmentMatch:
    """Result of aligning one reference."""

    reference: Reference
    paper_id: Optional[str] = None
    method: Optional[str] = None  # arxiv | s2 | s2_corpus | doi | pmid | title | fuzzy_title
    score: float = 0.0
    ambiguous: bool = False

    @property
    def aligned(self) -> bool:
        return self.paper_id is not None

    def to_dict(self) -> Dict[str, Any]:
        return {"reference": self.reference.to_dict(), "key": self.reference.key, "paper_id": self.paper_id,
                "method": self.method, "score": round(self.score, 4), "ambiguous": self.ambiguous}


@dataclass
class CoverageStats:
    """Alignment coverage: aligned / total references, by matching method."""

    total: int = 0
    aligned: int = 0
    by_method: Dict[str, int] = field(default_factory=dict)

    def add(self, match: AlignmentMatch) -> None:
        self.total += 1
        if match.aligned:
            self.aligned += 1
            self.by_method[match.method or "?"] = self.by_method.get(match.method or "?", 0) + 1

    def merge(self, other: "CoverageStats") -> "CoverageStats":
        self.total += other.total
        self.aligned += other.aligned
        for k, v in other.by_method.items():
            self.by_method[k] = self.by_method.get(k, 0) + v
        return self

    @property
    def unaligned(self) -> int:
        return self.total - self.aligned

    @property
    def coverage(self) -> float:
        return self.aligned / self.total if self.total else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"total": self.total, "aligned": self.aligned, "unaligned": self.unaligned,
                "coverage": round(self.coverage, 6), "by_method": dict(sorted(self.by_method.items()))}

    def __str__(self) -> str:
        return f"{self.aligned:,}/{self.total:,} aligned ({100 * self.coverage:.1f}%)"


@dataclass
class GoldResolution:
    """Gold-paper resolution of one query (see :meth:`PaperAligner.resolve`)."""

    gold_papers: List[str]
    unaligned_gold: List[str]
    matches: List[AlignmentMatch]
    stats: CoverageStats


# ---------------------------------------------------------------------------
# Aligner
# ---------------------------------------------------------------------------


class PaperAligner:
    """Align :class:`Reference` objects to the papers of a corpus (App. C.2).

    The default cascade is the paper's: arXiv id, Semantic Scholar id
    (paperId / CorpusId), DOI / PMID, exact normalized title. See the
    module docstring for the opt-in fuzzy stage.

    Parameters
    ----------
    papers:
        A :class:`Corpus` or iterable of :class:`Paper`.
    fuzzy:
        Enable the fuzzy-title stage (default ``False``: App. C.2 aligns by
        identifiers and normalized title only).
    fuzzy_threshold:
        Minimum ``difflib`` ratio for a fuzzy title match.
    fuzzy_year_tolerance:
        Fuzzy matches whose years (both known) differ by more than this are
        rejected (``None`` disables the check).
    year_tolerance:
        If set and both years are known, *exact* title matches whose years
        differ by more than this are rejected (arXiv first-version years may
        precede venue years, so the default ``None`` disables the check).
    max_candidates, rare_tokens, max_df_fraction:
        Fuzzy blocking: at most ``max_candidates`` titles are scored with
        ``difflib`` per reference, retrieved through the ``rare_tokens``
        rarest title tokens that occur in at most ``max_df_fraction`` of the titles.
    """

    def __init__(
        self,
        papers: Union[Corpus, Iterable[Paper]],
        fuzzy: bool = False,
        fuzzy_threshold: float = 0.92,
        fuzzy_year_tolerance: Optional[int] = 2,
        year_tolerance: Optional[int] = None,
        max_candidates: int = 50,
        rare_tokens: int = 6,
        max_df_fraction: float = 0.02,
    ):
        self.fuzzy = bool(fuzzy)
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.fuzzy_year_tolerance = fuzzy_year_tolerance
        self.year_tolerance = year_tolerance
        self.max_candidates = int(max_candidates)
        self.rare_tokens = int(rare_tokens)
        self.max_df_fraction = float(max_df_fraction)

        self.years: Dict[str, Optional[int]] = {}
        self.surnames: Dict[str, FrozenSet[str]] = {}
        self.by_arxiv: Dict[str, str] = {}
        self.by_s2: Dict[str, str] = {}
        self.by_corpus_id: Dict[str, str] = {}
        self.by_doi: Dict[str, str] = {}
        self.by_pmid: Dict[str, str] = {}
        self.by_title: Dict[str, List[str]] = defaultdict(list)
        # Fuzzy-matching structures (built lazily).
        self._titles: List[str] = []
        self._title_pids: List[List[str]] = []
        self._postings: Optional[Dict[str, List[int]]] = None

        source = papers.papers.values() if isinstance(papers, Corpus) else papers
        for p in source:
            self.add(p)

    def describe(self) -> Dict[str, Any]:
        """The alignment rules in use (recorded in ``coverage_report.json``)."""
        methods = ["arxiv", "s2", "s2_corpus", "doi", "pmid", "title"] + (["fuzzy_title"] if self.fuzzy else [])
        out: Dict[str, Any] = {"methods": methods, "papers": len(self)}
        if self.fuzzy:
            out.update(fuzzy_threshold=self.fuzzy_threshold, fuzzy_year_tolerance=self.fuzzy_year_tolerance)
        return out

    # ------------------------------------------------------------ indexing
    def add(self, paper: Paper) -> None:
        pid = paper.paper_id
        self.years[pid] = paper.year
        names = frozenset(x for x in (surname(a) for a in paper.authors or []) if x)
        if names:
            self.surnames[pid] = names
        aid = parse_arxiv_id(paper.arxiv_id) or (pid if parse_arxiv_id(pid) == pid else None)
        if aid:
            self.by_arxiv.setdefault(aid, pid)
        md = paper.metadata or {}
        s2 = parse_s2_id(paper.s2_id) if paper.s2_id else None
        if s2:
            self.by_s2.setdefault(s2, pid)
        cid = parse_s2_corpus_id(md.get("s2_corpus_id"))
        if cid is None and paper.s2_id and str(paper.s2_id).isdigit():
            cid = str(paper.s2_id)
        if cid:
            self.by_corpus_id.setdefault(cid, pid)
        doi = parse_doi(paper.doi)
        if doi:
            self.by_doi.setdefault(doi, pid)
        pmid = _clean(md.get("pmid"))
        if pmid:
            self.by_pmid.setdefault(pmid, pid)
        for key in title_keys(paper.title):
            bucket = self.by_title[key]
            if pid not in bucket:
                bucket.append(pid)
        self._postings = None  # invalidate fuzzy index

    def __len__(self) -> int:
        return len(self.years)

    def _build_fuzzy_index(self) -> None:
        self._titles = list(self.by_title)
        self._title_pids = [self.by_title[t] for t in self._titles]
        postings: Dict[str, List[int]] = defaultdict(list)
        for i, t in enumerate(self._titles):
            for tok in set(t.split()):
                postings[tok].append(i)
        self._postings = dict(postings)

    # ------------------------------------------------------------- lookups
    def _pick(self, pids: Sequence[str], ref: Reference) -> Tuple[Optional[str], bool]:
        """Choose among papers sharing a title: closest year, then smallest id."""
        if not pids:
            return None, False
        if len(pids) == 1:
            pid = pids[0]
            if not self._year_ok(pid, ref):
                return None, False
            return pid, False
        ok = [p for p in pids if self._year_ok(p, ref)]
        if not ok:
            return None, False
        if ref.year is not None:
            ok.sort(key=lambda p: (abs((self.years.get(p) or 10_000) - ref.year), p))
        else:
            ok.sort()
        return ok[0], True

    def _year_ok(self, pid: str, ref: Reference) -> bool:
        if self.year_tolerance is None or ref.year is None:
            return True
        y = self.years.get(pid)
        return y is None or abs(y - ref.year) <= self.year_tolerance

    def _fuzzy_candidates(self, key: str) -> List[int]:
        """Blocking: titles sharing the rarest tokens of ``key``, ranked by idf-weighted overlap.

        Only tokens occurring in at most ``max_df_fraction`` of the titles
        are used for blocking; a title made only of frequent tokens falls back
        to the intersection of the postings of its rarest tokens.
        """
        if self._postings is None:
            self._build_fuzzy_index()
        postings = self._postings
        assert postings is not None
        present = sorted({t for t in key.split() if t in postings}, key=lambda t: (len(postings[t]), t))
        if not present:
            return []
        content = [t for t in present if t not in STOPWORDS and len(t) > 1] or present
        max_df = max(1000, int(self.max_df_fraction * len(self._titles)))
        rare = [t for t in content if len(postings[t]) <= max_df][: self.rare_tokens]
        n = len(key)
        if not rare:
            pool = set(postings[content[0]])
            for t in content[1:3]:
                pool &= set(postings[t])
            ranked = sorted(pool, key=lambda i: (abs(len(self._titles[i]) - n), i))
            return ranked[: self.max_candidates]
        weight: Dict[int, float] = {}
        hits: Dict[int, int] = {}
        for t in rare:
            w = 1.0 / math.log(2.0 + len(postings[t]))
            for i in postings[t]:
                weight[i] = weight.get(i, 0.0) + w
                hits[i] = hits.get(i, 0) + 1
        need = 1 if len(rare) <= 2 else 2
        scored = [(-w, abs(len(self._titles[i]) - n), i) for i, w in weight.items() if hits[i] >= need]
        if not scored:  # e.g. a typo in all but one rare token
            scored = [(-w, abs(len(self._titles[i]) - n), i) for i, w in weight.items()]
        if len(rare) < 3 and len(scored) > self.max_candidates:
            # Few discriminative tokens: re-rank the pool by idf-weighted overlap with *all* tokens of the key.
            keyset = set(key.split())
            idf = {t: 1.0 / math.log(2.0 + len(postings[t])) for t in keyset if t in postings}
            scored = [(-sum(idf.get(t, 0.0) for t in keyset.intersection(self._titles[i].split())), dl, i)
                      for _, dl, i in scored]
        return [i for _, _, i in heapq.nsmallest(self.max_candidates, scored)]

    def _fuzzy_ok(self, pid: str, ref: Reference) -> bool:
        """Year and first-author agreement of a fuzzy candidate (checked only when both sides are known)."""
        if self.fuzzy_year_tolerance is not None and ref.year is not None:
            y = self.years.get(pid)
            if y is not None and abs(y - ref.year) > self.fuzzy_year_tolerance:
                return False
        first = surname(ref.first_author)
        names = self.surnames.get(pid)
        return not (first and names and first not in names)

    def _fuzzy(self, ref: Reference) -> Tuple[Optional[str], float, bool]:
        """Best fuzzy title match among the blocked candidates (``quick_ratio`` bounds prune ``difflib``).

        A candidate qualifies if its ratio reaches ``fuzzy_threshold``, it is
        :func:`titles_compatible` with the reference (same version tokens,
        differing words are spelling variants), and at least one paper with
        that title passes :meth:`_fuzzy_ok`.
        """
        best: Tuple[float, Optional[str], bool] = (0.0, None, False)
        for key in title_keys(ref.title):
            for i in self._fuzzy_candidates(key):
                cand = self._titles[i]
                floor = max(self.fuzzy_threshold, best[0])
                sm = SequenceMatcher(None, key, cand, autojunk=False)
                if sm.real_quick_ratio() < floor or sm.quick_ratio() < floor:
                    continue
                if not titles_compatible(key, cand):
                    continue
                ratio = sm.ratio()
                if ratio < floor or (ratio == best[0] and best[1] is not None):
                    continue
                pids = [p for p in self._title_pids[i] if self._fuzzy_ok(p, ref)]
                pid, ambiguous = self._pick(pids, ref)
                if pid is not None:
                    best = (ratio, pid, ambiguous)
        score, pid, ambiguous = best
        return pid, score, ambiguous

    def align(self, ref: Union[Reference, Mapping[str, Any]]) -> AlignmentMatch:
        """Align one reference through the id -> normalized-title (-> opt-in fuzzy-title) cascade."""
        if not isinstance(ref, Reference):
            ref = Reference.from_dict(ref)
        for method, value, table in (
            ("arxiv", ref.arxiv_id, self.by_arxiv),
            ("s2", ref.s2_id, self.by_s2),
            ("s2_corpus", ref.s2_corpus_id, self.by_corpus_id),
            ("doi", ref.doi, self.by_doi),
            ("pmid", ref.pmid, self.by_pmid),
        ):
            if value and value in table:
                return AlignmentMatch(ref, table[value], method, 1.0)
        for key in title_keys(ref.title):
            pid, ambiguous = self._pick(self.by_title.get(key, []), ref)
            if pid is not None:
                return AlignmentMatch(ref, pid, "title", 1.0, ambiguous)
        if self.fuzzy and ref.title:
            pid, score, ambiguous = self._fuzzy(ref)
            if pid is not None:
                return AlignmentMatch(ref, pid, "fuzzy_title", score, ambiguous)
        return AlignmentMatch(ref, None, None, 0.0)

    def align_many(self, refs: Iterable[Union[Reference, Mapping[str, Any]]]) -> Tuple[List[AlignmentMatch], CoverageStats]:
        stats = CoverageStats()
        matches = []
        for r in refs:
            m = self.align(r)
            stats.add(m)
            matches.append(m)
        return matches, stats

    def resolve(self, refs: Iterable[Union[Reference, Mapping[str, Any]]]) -> GoldResolution:
        """Gold papers of one query.

        References are de-duplicated by :attr:`Reference.key`; aligned
        references yield corpus ids (several references aligning to the same
        paper count once), unaligned ones their key. Coverage is counted over
        the distinct references, so ``len(gold_papers) + len(unaligned_gold)``
        equals the query's gold count.
        """
        return resolve_references(refs, self)


def resolve_references(refs: Iterable[Union[Reference, Mapping[str, Any]]],
                       aligner: Optional[PaperAligner] = None) -> GoldResolution:
    """Resolve references with an optional aligner (without one, every reference stays unaligned)."""
    seen: Dict[str, Reference] = {}
    for r in refs:
        ref = r if isinstance(r, Reference) else Reference.from_dict(r)
        seen.setdefault(ref.key, ref)
    gold: Dict[str, None] = {}
    unaligned: Dict[str, None] = {}
    matches: List[AlignmentMatch] = []
    stats = CoverageStats()
    for key, ref in seen.items():
        m = aligner.align(ref) if aligner is not None else AlignmentMatch(ref)
        matches.append(m)
        if m.aligned:
            if m.paper_id in gold:
                # Two distinct references resolved to one paper: count the paper once.
                continue
            gold[m.paper_id] = None  # type: ignore[index]
        else:
            unaligned[key] = None
        stats.add(m)
    return GoldResolution(list(gold), list(unaligned), matches, stats)


# ---------------------------------------------------------------------------
# Semantic Scholar identifiers for arXiv papers
# ---------------------------------------------------------------------------


def attach_s2_ids(corpus: Corpus, s2_papers: Union[str, Path, Sequence[Union[str, Path]]],
                  overwrite: bool = False) -> int:
    """Attach Semantic Scholar ids to arXiv papers from S2AG ``papers`` shards.

    Each shard record (``papers`` dataset of the Semantic Scholar Datasets
    API) has ``corpusid``, ``url`` (``.../paper/<40-hex paperId>``) and
    ``externalids`` (``ArXiv``, ``DOI``, ...). Papers of ``corpus`` whose
    arXiv id appears there get ``s2_id`` (paperId),
    ``metadata["s2_corpus_id"]`` and, if missing, ``doi``. Accepts files or
    directories of ``*.jsonl[.gz]`` shards; returns the number of papers updated.
    """
    by_arxiv = {p.arxiv_id or p.paper_id: p for p in corpus}
    paths: List[Path] = []
    items = [s2_papers] if isinstance(s2_papers, (str, Path)) else list(s2_papers)
    for item in items:
        item = Path(item)
        if item.is_dir():
            paths.extend(sorted(x for x in item.iterdir() if x.name.endswith((".jsonl", ".jsonl.gz", ".gz", ".json"))))
        else:
            paths.append(item)
    updated = 0
    for path in paths:
        for rec in read_jsonl(path):
            ext = rec.get("externalids") or rec.get("externalIds") or {}
            aid = parse_arxiv_id(ext.get("ArXiv"))
            paper = by_arxiv.get(aid) if aid else None
            if paper is None:
                continue
            sha = parse_s2_id(rec.get("url")) or parse_s2_id(rec.get("paperId"))
            cid = parse_s2_corpus_id(rec.get("corpusid") or ext.get("CorpusId"))
            changed = False
            if sha and (overwrite or not paper.s2_id):
                paper.s2_id = sha
                changed = True
            if cid and (overwrite or "s2_corpus_id" not in paper.metadata):
                paper.metadata["s2_corpus_id"] = cid
                changed = True
            doi = parse_doi(ext.get("DOI"))
            if doi and not paper.doi:
                paper.doi = doi
                changed = True
            updated += int(changed)
    return updated


# ---------------------------------------------------------------------------
# Evidence-quote attribution (ScholarQA-CS)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")
# A lower-case letter directly followed by a capitalised word: a lost line break between a markdown
# heading and its paragraph ("### DescriptionRandom k-SAT ..." copied from a rendered answer).
_GLUE_RE = re.compile(r"(?<=[a-z])(?=[A-Z][a-z])")


def words(text: Optional[str]) -> List[str]:
    """Lower-cased ASCII word tokens used for quote matching (accents folded, markup ignored).

    Words glued by a lost line break (``"ConceptsOne"``) are split at the
    case change; applied identically to quotes and to the texts they are
    searched in. Only C-level regex/str operations, for fast scanning.
    """
    folded = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    return _WORD_RE.findall(_GLUE_RE.sub(" ", folded).lower())


_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")


def word_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """:func:`words` of ``text`` together with the character span of each token in ``text``."""
    folded: List[str] = []
    origin: List[int] = []
    for i, ch in enumerate(text):
        f = ch if ch.isascii() else unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
        folded.extend(f)
        origin.extend([i] * len(f))
    flat = "".join(folded)
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for m in _ALNUM_RE.finditer(flat):
        cuts = [m.start()] + [m.start() + g.start() for g in _GLUE_RE.finditer(m.group())] + [m.end()]
        for a, b in zip(cuts, cuts[1:]):
            tokens.append(flat[a:b].lower())
            spans.append((origin[a], origin[b - 1] + 1))
    return tokens, spans


def shingles(tokens: Sequence[str], k: int) -> Set[Tuple[str, ...]]:
    """The set of word ``k``-grams of a token sequence."""
    return set(zip(*[tokens[i:] for i in range(k)]))


def quote_key(quote: str) -> str:
    """Stable key of an evidence quote (``quote:<sha1 of the normalized text>``)."""
    return "quote:" + hashlib.sha1(" ".join(words(quote)).encode("utf-8")).hexdigest()[:16]


class QuoteAttributor:
    """Attribute verbatim quotes to the corpus texts that contain them.

    Each quote is represented by its word ``shingle``-grams (normalized text).
    :meth:`scan` streams ``(paper_id, text)`` pairs -- abstracts and section
    chunks -- and, for every quote, records which of its shingles occur in
    each paper (set intersection in C, so scanning millions of chunks is
    fast). A quote is attributed to the paper covering the largest fraction
    of its shingles if that fraction is at least ``min_fraction``; quotes
    with fewer than ``min_words`` words are too ambiguous and never attributed.
    """

    def __init__(self, quotes: Iterable[str], shingle: int = 5, min_fraction: float = 0.6, min_words: int = 8):
        self.shingle = int(shingle)
        self.min_fraction = float(min_fraction)
        self.min_words = int(min_words)
        self.quotes: Dict[str, str] = {}
        self._size: Dict[str, int] = {}
        self._index: Dict[Tuple[str, ...], List[Tuple[str, int]]] = defaultdict(list)
        self._hits: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
        for q in quotes:
            self.add(q)

    def add(self, quote: str) -> str:
        key = quote_key(quote)
        if key in self.quotes:
            return key
        self.quotes[key] = quote
        tokens = words(quote)
        if len(tokens) < self.min_words:
            self._size[key] = 0
            return key
        grams = list(zip(*[tokens[i:] for i in range(self.shingle)]))
        uniq = list(dict.fromkeys(grams))
        self._size[key] = len(uniq)
        for pos, g in enumerate(uniq):
            self._index[g].append((key, pos))
        return key

    def scan(self, items: Iterable[Tuple[str, str]]) -> int:
        """Scan ``(paper_id, text)`` pairs; returns the number of texts with at least one hit."""
        index = self._index
        keys = index.keys()
        k = self.shingle
        n_hit = 0
        for pid, text in items:
            tokens = words(text)
            if len(tokens) < k:
                continue
            common = keys & shingles(tokens, k)
            if not common:
                continue
            n_hit += 1
            for g in common:
                for qkey, pos in index[g]:
                    self._hits[qkey][pid].add(pos)
        return n_hit

    def scan_corpus(self, corpus: Corpus, chunks_path: Optional[Union[str, Path]] = None) -> int:
        """Scan abstracts (and attached chunks) of ``corpus`` and optionally a ``chunks.jsonl(.gz)`` file."""
        n = self.scan((p.paper_id, f"{p.title}. {p.abstract}") for p in corpus)
        n += self.scan((c.paper_id, c.text) for p in corpus for c in p.chunks)
        if chunks_path is not None and Path(chunks_path).exists():
            n += self.scan((str(d["paper_id"]), str(d.get("text", ""))) for d in read_jsonl(chunks_path))
        return n

    def attribution(self, quote: str) -> Tuple[Optional[str], float]:
        """``(paper_id, fraction)`` of the best-covering paper (``(None, best fraction)`` if below threshold)."""
        key = quote_key(quote)
        size = self._size.get(key, 0)
        hits = self._hits.get(key)
        if not size or not hits:
            return None, 0.0
        best = max(len(s) for s in hits.values())
        frac = best / size
        if frac < self.min_fraction:
            return None, frac
        # Deterministic tie-break: smallest paper id among the best-covering papers.
        return min(p for p, s in hits.items() if len(s) == best), frac
