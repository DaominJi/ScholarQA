"""Text utilities: token counting, title normalization, section splitting.

Token costs c(x) of index nodes (Section 4.2) and the context budget B are
measured with :func:`count_tokens`. When ``tiktoken`` is installed we use the
``cl100k_base`` encoding; otherwise a deterministic ~4-characters-per-token
approximation is used so that budgets are reproducible without extra
dependencies.
"""

from __future__ import annotations

import math
import re
import unicodedata
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*|\d+(?:\.\d+)?")

# Common English stop words (kept small and dependency free).
STOPWORDS = frozenset(
    """
    a an the and or of for to in on at by with from as is are was were be been being this that these
    those it its into via using use used we our us their they them than then there here which who whom
    what when where why how can could may might should would will shall do does did done not no nor
    such also both each other more most some any all only very over under between among about against
    within without based new paper study approach method methods results show propose proposed present
    """.split()
)


@lru_cache(maxsize=1)
def _tiktoken_encoder():  # pragma: no cover - optional dependency
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: Optional[str]) -> int:
    """Number of LLM tokens in ``text`` (tiktoken if available, else chars/4)."""
    if not text:
        return 0
    enc = _tiktoken_encoder()
    if enc is not None:  # pragma: no cover - optional dependency
        return len(enc.encode(text, disallowed_special=()))
    return max(1, int(math.ceil(len(text) / 4.0)))


def truncate_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    enc = _tiktoken_encoder()
    if enc is not None:  # pragma: no cover - optional dependency
        return enc.decode(enc.encode(text, disallowed_special=())[:max_tokens])
    return text[: max_tokens * 4]


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Lower-cased word tokens (used by BM25, keyword search, mocks)."""
    toks = [t.lower() for t in _WORD_RE.findall(text or "")]
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    return toks


def normalize_title(title: str) -> str:
    """Normalized title used to align papers across sources (Appendix C.2).

    Lower-cases, strips accents, LaTeX markup and punctuation, and collapses
    whitespace.
    """
    if not title:
        return ""
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"\$([^$]*)\$", r" \1 ", t)  # keep inline-math content ("Fast $k$-means" -> "fast k means")
    t = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", t)  # \emph{x} -> x
    t = re.sub(r"[{}\\]", " ", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def title_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity of two normalized titles."""
    ta, tb = set(normalize_title(a).split()), set(normalize_title(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def keywords(text: str, k: int = 8) -> List[str]:
    """Most frequent non-stopword tokens (ties broken by first occurrence)."""
    counts: dict = {}
    order: dict = {}
    for i, tok in enumerate(tokenize(text, remove_stopwords=True)):
        counts[tok] = counts.get(tok, 0) + 1
        order.setdefault(tok, i)
    ranked = sorted(counts, key=lambda t: (-counts[t], order[t]))
    return ranked[:k]


# ---------------------------------------------------------------------------
# Section splitting (Section 2.1: chunks are obtained "by matching section
# headings with regular expressions")
# ---------------------------------------------------------------------------

_SECTION_PATTERNS: Sequence[re.Pattern] = (
    # Markdown headings: "# 1 Introduction", "## 3.2 Method"
    re.compile(r"^\s{0,3}#{1,4}\s+(?P<title>.+?)\s*#*\s*$"),
    # LaTeX: \section{Introduction}, \subsection*{...}
    re.compile(r"^\s*\\(?:sub)*section\*?\{(?P<title>[^}]*)\}"),
    # Numbered plain-text headings: "1 Introduction", "3.2. Experimental Setup", "IV. RESULTS"
    re.compile(
        r"^\s*(?P<num>(?:\d{1,2}(?:\.\d{1,2}){0,2}\.?)|(?:[IVX]{1,5}\.))\s+(?P<title>[A-Z][A-Za-z0-9 ,:;&/()'\-]{2,80})\s*$"
    ),
    # Common unnumbered headings in ALL CAPS or Title Case on their own line.
    re.compile(
        r"^\s*(?P<title>(?:abstract|introduction|related work|background|preliminaries|method(?:s|ology)?|"
        r"approach|model|experiments?|experimental setup|evaluation|results(?: and discussion)?|discussion|"
        r"analysis|conclusions?|limitations|future work|acknowledg(?:e)?ments?|references|appendix)"
        r")\s*:?\s*$",
        re.IGNORECASE,
    ),
)

_STOP_SECTIONS = re.compile(r"^(references|bibliography|acknowledg(e)?ments?)$", re.IGNORECASE)


def _match_heading(line: str) -> Optional[str]:
    if len(line) > 120:
        return None
    for pat in _SECTION_PATTERNS:
        m = pat.match(line)
        if m:
            title = m.group("title").strip().strip(".:")
            if title:
                return title
    return None


def split_sections(text: str, drop_references: bool = True) -> List[Tuple[str, str]]:
    """Split a full text into ``(heading, body)`` pairs by regex-matched headings.

    Text before the first heading is returned under the heading ``"Preamble"``.
    Sections named References/Bibliography/Acknowledgements (and everything
    after References) are dropped when ``drop_references`` is set.
    """
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Preamble"
    current: List[str] = []
    for raw in (text or "").splitlines():
        heading = _match_heading(raw)
        if heading is not None:
            if current and any(s.strip() for s in current):
                sections.append((current_title, current))
            current_title, current = heading, []
            continue
        current.append(raw)
    if current and any(s.strip() for s in current):
        sections.append((current_title, current))

    out: List[Tuple[str, str]] = []
    for title, lines in sections:
        if drop_references and _STOP_SECTIONS.match(title.strip()):
            if title.strip().lower().startswith(("reference", "bibliography")):
                break
            continue
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if body:
            out.append((title, body))
    return out


def chunk_sections(
    sections: Iterable[Tuple[str, str]],
    max_tokens: int = 512,
    min_tokens: int = 32,
) -> List[Tuple[str, str]]:
    """Split long sections into paragraph-aligned pieces of at most ``max_tokens``.

    Sections shorter than ``min_tokens`` are merged into the following section.
    """
    out: List[Tuple[str, str]] = []
    carry_title, carry_text = None, ""
    for title, body in sections:
        if carry_text:
            body = carry_text + "\n\n" + body
            title = carry_title or title
            carry_title, carry_text = None, ""
        if count_tokens(body) < min_tokens:
            carry_title, carry_text = title, body
            continue
        if count_tokens(body) <= max_tokens:
            out.append((title, body))
            continue
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buf: List[str] = []
        part = 1
        for para in paras:
            while count_tokens(para) > max_tokens:
                head = truncate_tokens(para, max_tokens)
                if buf:
                    out.append((f"{title} (part {part})", "\n\n".join(buf)))
                    part += 1
                    buf = []
                out.append((f"{title} (part {part})", head))
                part += 1
                para = para[len(head):].strip()
            if buf and count_tokens("\n\n".join(buf + [para])) > max_tokens:
                out.append((f"{title} (part {part})", "\n\n".join(buf)))
                part += 1
                buf = []
            if para:
                buf.append(para)
        if buf:
            out.append((f"{title} (part {part})" if part > 1 else title, "\n\n".join(buf)))
    if carry_text:
        if out:
            t, b = out[-1]
            out[-1] = (t, b + "\n\n" + carry_text)
        else:
            out.append((carry_title or "Body", carry_text))
    return out
