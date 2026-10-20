"""Full text and section chunks: the paper-detail level of the index (Section 2.1).

Section 2.1: chunks are obtained from each paper's full text *"by matching
section headings with regular expressions"*. This module obtains the full
text of corpus papers from configurable sources, splits it into sections,
chunks long sections with :func:`scholarpath.text.chunk_sections`
(``IndexConfig.chunk_max_tokens`` / ``chunk_min_tokens``) and writes
:class:`~scholarpath.types.Chunk` records (``chunk_id = "<paper_id>#<i>"``)
to ``chunks.jsonl.gz``. Papers for which no source yields a full text keep
only their abstract (paper-summary level) and get no chunks.

Sources, tried in a configurable order (:func:`make_sources`):

``local``   a directory of pre-downloaded files per paper: ``<id>.pdf``,
            ``<id>.tex``, ``<id>.txt``/``.md``, an e-print archive
            ``<id>.tar.gz``/``<id>.tar``/``<id>.gz`` or a LaTeX source
            directory ``<id>/`` (old-style ids use ``_`` for ``/``; version
            suffixes ``<id>v2.pdf`` are accepted);
``s2orc``   Semantic Scholar S2ORC v2 shards (GROBID parses; ``body.text`` with
            ``section_header`` spans), joined on ``openaccessinfo.externalids.ArXiv``;
``gcs``     the public Google Cloud Storage mirror of arXiv PDFs
            ``https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf/YYMM/<id>vN.pdf``
            (old-style: ``.../arxiv/<archive>/pdf/YYMM/<num>vN.pdf``);
``arxiv``   arxiv.org itself -- the LaTeX e-print (``/e-print/<id>``, parsed by
            ``\\section``) with the PDF (``/pdf/<id>``) as fallback -- with a
            polite rate limit (one request per 3 s by default) and a User-Agent.

Section splitting per format: PDFs are converted with PyMuPDF (lazy import)
and split by :func:`scholarpath.text.split_sections` (regex headings);
LaTeX by ``\\section``/``\\subsection``/``\\subsubsection`` commands
(:func:`latex_to_sections`); S2ORC by its section-header spans
(:func:`s2orc_sections`). Reference lists and acknowledgements are dropped,
and so is front matter (title block, abstract) when the paper has other
sections, since the abstract already forms the paper-summary level.

:class:`FulltextExtractor` is resumable: a JSONL state file records every
processed paper, finished papers are skipped on the next run, and new
chunks are appended to the gzip output as a new gzip member.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import re
import tarfile
import threading
import time
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

from scholarpath.data.alignment import parse_arxiv_id, parse_s2_corpus_id, split_arxiv_version
from scholarpath.data.corpus import read_jsonl
from scholarpath.data.download import USER_AGENT, http_get
from scholarpath.text import chunk_sections, split_sections
from scholarpath.types import Chunk, Paper

log = logging.getLogger(__name__)

PathLike = Union[str, Path]
Section = Tuple[str, str]

GCS_BASE = "https://storage.googleapis.com/arxiv-dataset/arxiv"
ARXIV_BASE = "https://arxiv.org"
DEFAULT_SOURCE_ORDER: Tuple[str, ...] = ("local", "s2orc", "gcs", "arxiv")

_STOP_HEADINGS = re.compile(
    r"^\W*(?:\d+(?:\.\d+)*\.?\s*)?(?:references?|bibliography|literature\s+cited|acknowledge?ments?|"
    r"acknowledgments?\s+and\s+disclosure\s+of\s+funding)\W*$",
    re.IGNORECASE,
)
_FRONT_HEADINGS = re.compile(r"^\W*(?:preamble|abstract|title|front\s*matter|keywords?)\W*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# URLs and file names
# ---------------------------------------------------------------------------


def safe_paper_id(paper_id: str) -> str:
    """File-system safe form of a paper id (``math/0601001`` -> ``math_0601001``)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)


def _split_old_style(arxiv_id: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"^([a-z\-]+)(?:\.[A-Za-z]{2})?/(\d{7})$", arxiv_id)
    return (m.group(1), m.group(2)) if m else None


def gcs_pdf_url(arxiv_id: str, version: int = 1) -> str:
    """URL of a PDF in the public arXiv GCS bucket.

    ``gcs_pdf_url("2601.12369", 1)`` -> ``.../arxiv/arxiv/pdf/2601/2601.12369v1.pdf``;
    ``gcs_pdf_url("math/0601001", 2)`` -> ``.../arxiv/math/pdf/0601/0601001v2.pdf``.
    """
    base, v = split_arxiv_version(arxiv_id)
    version = v or version
    old = _split_old_style(base)
    if old:
        archive, num = old
        return f"{GCS_BASE}/{archive}/pdf/{num[:4]}/{num}v{version}.pdf"
    yymm = base.split(".", 1)[0]
    return f"{GCS_BASE}/arxiv/pdf/{yymm}/{base}v{version}.pdf"


def arxiv_pdf_url(arxiv_id: str, version: Optional[int] = None, base: str = ARXIV_BASE) -> str:
    return f"{base}/pdf/{arxiv_id}" + (f"v{version}" if version else "")


def arxiv_eprint_url(arxiv_id: str, base: str = ARXIV_BASE) -> str:
    return f"{base}/e-print/{arxiv_id}"


def latest_version(paper: Paper) -> Optional[int]:
    """Latest arXiv version recorded in the paper metadata (``latest_version`` / ``num_versions``)."""
    md = paper.metadata or {}
    lv = md.get("latest_version")
    if lv:
        m = re.match(r"^v?(\d+)$", str(lv))
        if m:
            return int(m.group(1))
    nv = md.get("num_versions")
    return int(nv) if isinstance(nv, int) and nv > 0 else None


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def pdf_to_text(data: Union[bytes, PathLike]) -> str:
    """Plain text of a PDF (PyMuPDF, page order, de-hyphenated line breaks)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:  # pragma: no cover - depends on the environment
        try:
            import pymupdf as fitz  # type: ignore
        except ImportError as err:
            raise ImportError("PDF text extraction needs PyMuPDF: pip install pymupdf") from err
    if isinstance(data, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(data), filetype="pdf")
    else:
        doc = fitz.open(str(data))
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return clean_pdf_text("\n".join(pages))


def clean_pdf_text(text: str) -> str:
    """Join words hyphenated across line breaks and drop form feeds / runs of blank lines."""
    text = text.replace("\r", "").replace("\x0c", "\n")
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def text_to_sections(text: str) -> List[Section]:
    """Regex section splitting of plain text (Section 2.1; :func:`scholarpath.text.split_sections`)."""
    return split_sections(text, drop_references=True)


def pdf_to_sections(data: Union[bytes, PathLike]) -> List[Section]:
    return text_to_sections(pdf_to_text(data))


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

_LATEX_HEADING_RE = re.compile(r"\\(chapter|section|subsection|subsubsection)\*?\s*(?:\[[^\]]*\])?\s*\{")
_LATEX_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}")
_DROP_ENVS = ("thebibliography", "tikzpicture", "comment", "filecontents", "lstlisting", "verbatim",
              "algorithmic", "pgfpicture")
_FLOAT_ENVS = ("figure", "figure*", "table", "table*", "wrapfigure", "wraptable", "algorithm", "algorithm*")
_FRONT_MATTER_CMDS = ("title", "author", "date", "thanks", "affiliation", "affil", "institute", "address", "email",
                      "keywords", "ccsdesc", "acmConference", "acmYear", "acmDOI", "acmISBN", "copyrightyear",
                      "setcopyright", "orcid", "icmltitle", "icmlauthor", "icmlaffiliation", "icmlcorrespondingauthor",
                      "icmlkeywords")
_UNWRAP_CMDS = ("emph", "textbf", "textit", "texttt", "textsc", "textsf", "textrm", "underline", "mbox",
                "textnormal", "textmd", "textup", "uline", "hl", "text")


def _matching_brace(s: str, open_idx: int) -> int:
    """Index of the ``}`` matching the ``{`` at ``open_idx`` (or ``len(s)`` if unbalanced)."""
    depth = 0
    i = open_idx
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(s)


def strip_latex_comments(tex: str) -> str:
    """Remove ``%`` comments (escaped ``\\%`` is kept)."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in tex.splitlines())


def resolve_latex_inputs(tex: str, files: Mapping[str, str], depth: int = 0) -> str:
    """Inline ``\\input``/``\\include`` using ``files`` (relative path -> content), recursively."""
    if depth > 10:
        return tex

    def repl(m: "re.Match[str]") -> str:
        name = m.group(1).strip()
        for cand in (name, name + ".tex", name.lstrip("./"), name.lstrip("./") + ".tex"):
            if cand in files:
                return resolve_latex_inputs(strip_latex_comments(files[cand]), files, depth + 1)
        return ""

    return _LATEX_INPUT_RE.sub(repl, tex)


def _remove_env(tex: str, env: str, keep_captions: bool = False) -> str:
    pattern = re.compile(r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{" + re.escape(env) + r"\}", re.S)

    def repl(m: "re.Match[str]") -> str:
        if not keep_captions:
            return "\n"
        caps = []
        body = m.group(1)
        for cm in re.finditer(r"\\caption\s*(?:\[[^\]]*\])?\s*\{", body):
            start = cm.end() - 1
            caps.append(body[start + 1:_matching_brace(body, start)])
        return "\n" + "\n".join(f"[{env.rstrip('*').capitalize()}] {c.strip()}" for c in caps) + "\n"

    return pattern.sub(repl, tex)


def _unwrap_command(tex: str, cmd: str, fmt: Union[str, Callable[[str], str]] = "{}") -> str:
    """Replace ``\\cmd[opt]{arg}`` by ``fmt(arg)`` / ``fmt.format(arg)`` (balanced braces)."""
    pat = re.compile(r"\\" + re.escape(cmd) + r"\*?\s*(?:\[[^\]]*\])?\s*\{")
    render = fmt if callable(fmt) else fmt.format
    out = []
    pos = 0
    while True:
        m = pat.search(tex, pos)
        if not m:
            out.append(tex[pos:])
            break
        start = m.end() - 1
        end = _matching_brace(tex, start)
        out.append(tex[pos:m.start()])
        out.append(render(tex[start + 1:end]))
        pos = end + 1
    return "".join(out)


def _run_in_heading(title: str) -> str:
    """``\\paragraph{Details.}`` -> ``"Details. "`` (a period is added only if the title has no final punctuation)."""
    title = title.strip()
    return "\n" + title + ("" if not title or title[-1] in ".!?:;" else ".") + " "


_MACRO_DEF_RE = re.compile(r"\\(?:(?:re)?newcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z]+)\s*\}?\s*(\[\d\])?\s*(?:\[[^\]]*\]\s*)?\{"
                           r"|\\def\s*\\([A-Za-z]+)\s*(#\d)?\s*\{")


def expand_simple_macros(tex: str, max_rounds: int = 3) -> str:
    """Expand argument-less user macros (``\\newcommand{\\method}{SysName\\xspace}``, ``\\def\\method{...}``).

    Definitions are removed from ``tex``; every use ``\\method``, ``\\method{}``
    is replaced by the macro body (``\\xspace`` dropped), repeated up to
    ``max_rounds`` times for macros defined through other macros. Macros
    with arguments are removed but not expanded.
    """
    macros: Dict[str, str] = {}
    out: List[str] = []
    pos = 0
    for m in _MACRO_DEF_RE.finditer(tex):
        if m.start() < pos:  # inside a previous definition's body
            continue
        start = m.end() - 1
        end = _matching_brace(tex, start)
        name = m.group(1) or m.group(3)
        has_args = bool(m.group(2) or m.group(4))
        if not has_args:
            macros[name] = re.sub(r"\\xspace\b\s*", "", tex[start + 1:end]).strip()
        out.append(tex[pos:m.start()])
        pos = end + 1
    out.append(tex[pos:])
    tex = "".join(out)
    if not macros:
        return tex
    use_re = re.compile(r"\\(" + "|".join(sorted(map(re.escape, macros), key=len, reverse=True)) + r")(?![A-Za-z])(?:\s*\{\})?")
    for _ in range(max_rounds):
        new = use_re.sub(lambda u: macros[u.group(1)], tex)
        if new == tex:
            break
        tex = new
    return tex


_DISPLAY_MATH_ENVS = r"(?:equation|align|alignat|gather|multline|eqnarray|flalign|displaymath|math|dmath)\*?"
_MATH_RE = re.compile(
    r"\\begin\{(" + _DISPLAY_MATH_ENVS + r")\}.*?\\end\{\1\}"  # display environments
    r"|\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)"  # $$...$$, \[...\], \(...\)
    r"|(?<!\\)\$(?:[^$\\]|\\.)+?\$",  # $...$ (escaped \$ is not a delimiter)
    re.S,
)


def clean_latex_text(tex: str) -> str:
    """Light LaTeX -> text conversion for section bodies.

    Floats are replaced by their captions, citations / references by
    ``[cite]`` / ``[ref]``, formatting commands are unwrapped and remaining
    argument-less commands dropped. Math -- inline (``$...$``, ``\\(...\\)``)
    and display (``\\[...\\]``, ``$$...$$``, ``equation``/``align``/... environments)
    -- is protected from this clean-up and kept verbatim (``\\label`` removed).
    """
    for env in _DROP_ENVS:
        tex = _remove_env(tex, env)
    for env in _FLOAT_ENVS:
        tex = _remove_env(tex, env, keep_captions=True)
    tex = re.sub(r"\\(?:label|bibliographystyle|bibliography|vspace|hspace|includegraphics)\*?\s*(?:\[[^\]]*\])?\s*\{[^}]*\}", "", tex)
    tex = re.sub(r"\\(?:nonumber|notag)\b\s*", "", tex)
    # Protect math: placeholders survive the clean-up below and are restored at the end.
    math: List[str] = []

    def protect(m: "re.Match[str]") -> str:
        math.append(m.group(0).strip())
        block = m.group(0).startswith(("\\begin", "$$", "\\["))
        return ("\n" if block else "") + f"\x00{len(math) - 1}\x00" + ("\n" if block else "")

    tex = _MATH_RE.sub(protect, tex)
    tex = re.sub(r"\\(?:cite[a-zA-Z]*|nocite)\*?\s*(?:\[[^\]]*\]\s*)*\{[^}]*\}", "[cite]", tex)
    tex = re.sub(r"\\(?:eqref|ref|autoref|cref|Cref|pageref|vref)\s*\{[^}]*\}", "[ref]", tex)
    for cmd in _FRONT_MATTER_CMDS:
        tex = _unwrap_command(tex, cmd, "")
    tex = re.sub(r"\\href\s*\{[^}]*\}\s*(?=\{)", "", tex)  # \href{url}{text} -> text
    tex = re.sub(r"\\(?:textcolor|colorbox)\s*\{[^}]*\}\s*(?=\{)|\\color\s*\{[^}]*\}", "", tex)
    tex = _unwrap_command(tex, "footnote", " ({})")
    for cmd in _UNWRAP_CMDS:
        tex = _unwrap_command(tex, cmd)
    tex = _unwrap_command(tex, "paragraph", _run_in_heading)
    tex = re.sub(r"\\item\s*(?:\[([^\]]*)\])?", lambda m: "\n- " + (m.group(1) + ": " if m.group(1) else ""), tex)
    tex = re.sub(r"\\(?:begin|end)\{[^}]*\}(?:\[[^\]]*\])?", "\n", tex)
    tex = re.sub(r"\\\\(?:\[[^\]]*\])?", "\n", tex)
    tex = tex.replace("~", " ").replace("\\&", "&").replace("\\%", "%").replace("\\_", "_").replace("\\#", "#")
    tex = tex.replace("\\$", "$")
    # Drop remaining argument-less commands (\noindent, \centering, \maketitle, ...) and grouping braces.
    tex = re.sub(r"\\[A-Za-z]+\*?(?![A-Za-z{])\s*", " ", tex)
    # Other commands with an argument keep the argument (\url{...}); an undefined macro \method{} vanishes.
    tex = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\s*(?=\{)", " ", tex)
    tex = tex.replace("{", "").replace("}", "")  # grouping braces, e.g. "{\bf x}"
    tex = re.sub(r"[ \t]+", " ", tex)
    tex = re.sub(r" *\n *", "\n", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex).strip()
    return re.sub(r"\x00(\d+)\x00", lambda m: math[int(m.group(1))], tex)


def latex_to_sections(tex: str, files: Optional[Mapping[str, str]] = None) -> List[Section]:
    """Split a LaTeX document into ``(heading, body)`` sections by sectioning commands.

    ``\\input``/``\\include`` are inlined from ``files`` when given. The
    abstract environment becomes an ``"Abstract"`` section; subsections are
    returned as separate sections titled ``"<section> > <subsection>"``;
    the bibliography and acknowledgement sections are dropped.
    """
    tex = strip_latex_comments(tex)
    if files:
        tex = resolve_latex_inputs(tex, files)
    tex = expand_simple_macros(tex)
    m = re.search(r"\\begin\{document\}", tex)
    if m:
        tex = tex[m.end():]
    m = re.search(r"\\end\{document\}", tex)
    if m:
        tex = tex[:m.start()]
    sections: List[Section] = []
    abs_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if abs_m:
        sections.append(("Abstract", clean_latex_text(abs_m.group(1))))
        tex = tex[:abs_m.start()] + tex[abs_m.end():]
    # Bibliographies are removed by clean_latex_text (environment / commands), so appendices after them are kept.
    level_of = {"chapter": 0, "section": 1, "subsection": 2, "subsubsection": 3}
    heads = []
    for hm in _LATEX_HEADING_RE.finditer(tex):
        start = hm.end() - 1
        end = _matching_brace(tex, start)
        title = clean_latex_text(tex[start + 1:end]).replace("\n", " ").strip()
        heads.append((hm.start(), end + 1, level_of[hm.group(1)], title))
    pre = clean_latex_text(tex[: heads[0][0]] if heads else tex)
    if pre:
        sections.insert(0, ("Preamble", pre))
    stack: List[Tuple[int, str]] = []
    for i, (hstart, hend, level, title) in enumerate(heads):
        body_end = heads[i + 1][0] if i + 1 < len(heads) else len(tex)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading = " > ".join(t for _, t in stack if t)
        body = clean_latex_text(tex[hend:body_end])
        if _STOP_HEADINGS.match(title or ""):
            continue
        if body:
            sections.append((heading or "Section", body))
    return sections


def find_main_tex(files: Mapping[str, str]) -> Optional[str]:
    """Pick the main ``.tex`` file of a source tree (has ``\\documentclass`` and ``\\begin{document}``)."""
    tex = {n: c for n, c in files.items() if n.lower().endswith(".tex")}
    if not tex:
        return None
    full = [n for n, c in tex.items() if re.search(r"\\documentclass", c) and re.search(r"\\begin\{document\}", c)]
    pool = full or [n for n, c in tex.items() if re.search(r"\\documentclass", c)] or list(tex)
    preferred = [n for n in pool if Path(n).stem.lower() in {"main", "ms", "paper", "article", "manuscript"}]
    pool = preferred or pool
    return max(pool, key=lambda n: (len(tex[n]), n))


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")  # pragma: no cover


def read_latex_source(data: Union[bytes, PathLike]) -> Optional[Tuple[str, Dict[str, str]]]:
    """Read an e-print / LaTeX source: a directory, a (gzipped) tar, a gzipped ``.tex`` or a ``.tex`` file.

    Returns ``(main_tex, files)`` or ``None`` when the input is not LaTeX
    (e.g. arXiv serves a PDF for papers without source).
    """
    files: Dict[str, str] = {}
    if not isinstance(data, (bytes, bytearray)):
        path = Path(data)
        if path.is_dir():
            for f in path.rglob("*"):
                if f.is_file() and f.suffix.lower() in {".tex", ".ltx", ".bbl", ".sty"}:
                    files[str(f.relative_to(path))] = _decode(f.read_bytes())
            main = find_main_tex(files)
            return (files[main], files) if main else None
        data = path.read_bytes()
    raw = bytes(data)
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except OSError:
            return None
    if raw[:4] == b"%PDF":
        return None
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith((".tex", ".ltx", ".bbl")):
                    fh = tar.extractfile(member)
                    if fh is not None:
                        files[member.name.lstrip("./")] = _decode(fh.read())
        main = find_main_tex(files)
        return (files[main], files) if main else None
    except tarfile.TarError:
        pass
    text = _decode(raw)
    if "\\begin{document}" in text or "\\section" in text or "\\documentclass" in text:
        return text, {"main.tex": text}
    return None


# ---------------------------------------------------------------------------
# S2ORC v2
# ---------------------------------------------------------------------------


def _annotation_spans(value: Any) -> List[Dict[str, Any]]:
    """Decode an S2ORC annotation value (a JSON-encoded string, list or null) into spans with int offsets."""
    if value in (None, "", "null"):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    spans = []
    for sp in value or []:
        try:
            start, end = int(sp["start"]), int(sp["end"])
        except (KeyError, TypeError, ValueError):
            continue
        spans.append({"start": start, "end": end, "attributes": sp.get("attributes") or {}})
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return spans


def s2orc_arxiv_id(record: Mapping[str, Any]) -> Optional[str]:
    oa = record.get("openaccessinfo") or {}
    ext = oa.get("externalids") or oa.get("externalIds") or record.get("externalids") or {}
    return parse_arxiv_id(ext.get("ArXiv")) if ext else None


def s2orc_sections(record: Mapping[str, Any]) -> List[Section]:
    """Sections of an S2ORC v2 record from ``body.text`` and its ``section_header`` spans.

    Also accepts the legacy ``s2orc`` layout (``content.text`` /
    ``content.annotations.sectionheader``). Text before the first header is
    returned as ``"Preamble"``; reference / acknowledgement sections are dropped.
    """
    body = record.get("body") or {}
    if not body and record.get("content"):
        content = record["content"] or {}
        text = content.get("text") or ""
        ann = content.get("annotations") or {}
        headers = _annotation_spans(ann.get("sectionheader"))
    else:
        text = body.get("text") or ""
        ann = body.get("annotations") or {}
        headers = _annotation_spans(ann.get("section_header"))
    if not text.strip():
        return []
    sections: List[Section] = []
    if not headers:
        return [("Body", text.strip())]
    pre = text[: headers[0]["start"]].strip()
    if pre:
        sections.append(("Preamble", pre))
    for i, h in enumerate(headers):
        title = " ".join(text[h["start"]:h["end"]].split())
        num = str((h.get("attributes") or {}).get("n") or "").strip()
        end = headers[i + 1]["start"] if i + 1 < len(headers) else len(text)
        body_text = text[h["end"]:end].strip()
        if _STOP_HEADINGS.match(title):
            continue
        if body_text:
            sections.append((f"{num} {title}".strip() if num else title, body_text))
    return sections


def iter_s2orc_records(paths: Union[PathLike, Sequence[PathLike]]) -> Iterator[Dict[str, Any]]:
    """Stream S2ORC (v2 or legacy) records from shard files or directories of shards."""
    items = [paths] if isinstance(paths, (str, Path)) else list(paths)
    for item in items:
        item = Path(item)
        shards = sorted(p for p in item.rglob("*") if p.is_file() and p.name.endswith((".jsonl", ".jsonl.gz", ".gz", ".json"))) \
            if item.is_dir() else [item]
        for shard in shards:
            yield from read_jsonl(shard)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def clean_sections(sections: Sequence[Section], drop_front_matter: bool = True) -> List[Section]:
    """Drop empty, reference and (if other sections exist) front-matter sections."""
    out = [(h.strip() or "Body", b.strip()) for h, b in sections if b and b.strip() and not _STOP_HEADINGS.match(h)]
    if drop_front_matter:
        rest = [(h, b) for h, b in out if not _FRONT_HEADINGS.match(h)]
        if rest:
            out = rest
    return out


def sections_to_chunks(paper_id: str, sections: Sequence[Section], max_tokens: int = 512, min_tokens: int = 32,
                       drop_front_matter: bool = True) -> List[Chunk]:
    """Chunk sections (:func:`scholarpath.text.chunk_sections`) into ``Chunk("<paper_id>#<i>", ...)``."""
    pieces = chunk_sections(clean_sections(sections, drop_front_matter), max_tokens=max_tokens, min_tokens=min_tokens)
    return [Chunk(chunk_id=f"{paper_id}#{i}", paper_id=paper_id, heading=h, text=t, order=i)
            for i, (h, t) in enumerate(pieces)]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@dataclass
class FulltextDocument:
    """Sections of one paper's full text and where they came from."""

    paper_id: str
    source: str  # source name (local | s2orc | gcs | arxiv)
    kind: str  # pdf | latex | text | s2orc
    sections: List[Section]


class FulltextSource:
    """Base class: :meth:`fetch` one paper, or :meth:`fetch_many` for bulk sources."""

    name = "base"

    def fetch(self, paper: Paper) -> Optional[FulltextDocument]:  # pragma: no cover - abstract
        raise NotImplementedError

    def fetch_many(self, papers: Sequence[Paper]) -> Iterator[Tuple[Paper, Optional[FulltextDocument]]]:
        for p in papers:
            try:
                doc = self.fetch(p)
            except ImportError:
                raise
            except Exception as err:  # one bad file / response must not stop the run
                log.debug("%s failed for %s: %s", self.name, p.paper_id, err)
                doc = None
            yield p, doc


class LocalSource(FulltextSource):
    """Pre-downloaded files under ``root`` (see module docstring for the accepted names).

    The directory is listed once; per paper, LaTeX sources are preferred over
    PDFs over plain text, and higher versions over lower ones.
    """

    name = "local"
    _EXTS = (".tar.gz", ".tgz", ".tar", ".gz", ".tex", ".pdf", ".txt", ".md")  # longest suffix first
    _PRIORITY = {"": 0, ".tex": 1, ".tar.gz": 2, ".tgz": 2, ".tar": 2, ".gz": 3, ".pdf": 4, ".txt": 5, ".md": 5}

    def __init__(self, root: PathLike):
        self.root = Path(root)
        self._files: Optional[Dict[str, List[Tuple[int, int, Path]]]] = None

    def _index(self) -> Dict[str, List[Tuple[int, int, Path]]]:
        if self._files is None:
            files: Dict[str, List[Tuple[int, int, Path]]] = {}
            for entry in sorted(self.root.iterdir()) if self.root.is_dir() else []:
                name = entry.name
                ext = "" if entry.is_dir() else next((e for e in self._EXTS if name.lower().endswith(e)), None)
                if ext is None:
                    continue
                stem = name[: len(name) - len(ext)] if ext else name
                m = re.match(r"^(.*\d)v(\d+)$", stem)
                base, version = (m.group(1), int(m.group(2))) if m else (stem, 0)
                files.setdefault(base, []).append((self._PRIORITY[ext], -version, entry))
            for items in files.values():
                items.sort()
            self._files = files
        return self._files

    def _candidates(self, paper: Paper) -> List[Path]:
        ids = [paper.paper_id] + ([paper.arxiv_id] if paper.arxiv_id and paper.arxiv_id != paper.paper_id else [])
        index = self._index()
        out: List[Path] = []
        for pid in ids:
            for stem in dict.fromkeys([safe_paper_id(pid), pid.replace("/", "")]):
                out.extend(path for _, _, path in index.get(stem, []))
        return list(dict.fromkeys(out))

    def fetch(self, paper: Paper) -> Optional[FulltextDocument]:
        for path in self._candidates(paper):
            name = path.name.lower()
            if path.is_dir() or name.endswith((".tex", ".tar.gz", ".tgz", ".tar", ".gz")):
                src = read_latex_source(path)
                if src is not None:
                    secs = latex_to_sections(src[0], src[1])
                    if secs:
                        return FulltextDocument(paper.paper_id, self.name, "latex", secs)
                continue
            if name.endswith(".pdf"):
                secs = pdf_to_sections(path)
                kind = "pdf"
            else:
                secs = text_to_sections(path.read_text(encoding="utf-8", errors="replace"))
                kind = "text"
            if secs:
                return FulltextDocument(paper.paper_id, self.name, kind, secs)
        return None


class _RateLimiter:
    def __init__(self, min_interval_s: float):
        self.min_interval_s = float(min_interval_s)
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class GCSPdfSource(FulltextSource):
    """PDFs from the public arXiv GCS bucket (no authentication).

    The latest version recorded in the paper metadata is tried first, then v1.
    Downloaded PDFs are cached under ``cache_dir`` when given.
    """

    name = "gcs"

    def __init__(self, cache_dir: Optional[PathLike] = None, timeout: float = 60.0, min_interval_s: float = 0.0):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self._limiter = _RateLimiter(min_interval_s)

    def _versions(self, paper: Paper) -> List[int]:
        lv = latest_version(paper)
        return list(dict.fromkeys([v for v in (lv, 1) if v]))

    def fetch(self, paper: Paper) -> Optional[FulltextDocument]:
        aid = parse_arxiv_id(paper.arxiv_id or paper.paper_id)
        if not aid:
            return None
        for v in self._versions(paper):
            cached = self.cache_dir / "pdf" / f"{safe_paper_id(aid)}v{v}.pdf" if self.cache_dir else None
            if cached is not None and cached.exists():
                data = cached.read_bytes()
            else:
                self._limiter.wait()
                try:
                    data = http_get(gcs_pdf_url(aid, v), timeout=self.timeout)
                except urllib.error.HTTPError as err:
                    if err.code in (403, 404):
                        continue
                    raise
                if cached is not None:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(data)
            secs = pdf_to_sections(data)
            if secs:
                return FulltextDocument(paper.paper_id, self.name, "pdf", secs)
        return None


class ArxivSource(FulltextSource):
    """arxiv.org e-print (LaTeX) with PDF fallback, rate-limited (default 1 request / 3 s)."""

    name = "arxiv"

    def __init__(self, cache_dir: Optional[PathLike] = None, min_interval_s: float = 3.0,
                 user_agent: str = USER_AGENT, base_url: str = ARXIV_BASE, prefer: str = "latex",
                 timeout: float = 60.0):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.user_agent = user_agent
        self.base_url = base_url.rstrip("/")
        self.prefer = prefer
        self.timeout = timeout
        self._limiter = _RateLimiter(min_interval_s)

    def _get(self, url: str, cache_name: Optional[str]) -> Optional[bytes]:
        cached = self.cache_dir / cache_name if (self.cache_dir and cache_name) else None
        if cached is not None and cached.exists():
            return cached.read_bytes()
        self._limiter.wait()
        try:
            data = http_get(url, headers={"User-Agent": self.user_agent}, timeout=self.timeout)
        except urllib.error.HTTPError as err:
            if err.code in (403, 404):
                return None
            raise
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
        return data

    def fetch(self, paper: Paper) -> Optional[FulltextDocument]:
        aid = parse_arxiv_id(paper.arxiv_id or paper.paper_id)
        if not aid:
            return None
        order = ("latex", "pdf") if self.prefer == "latex" else ("pdf", "latex")
        for kind in order:
            if kind == "latex":
                data = self._get(arxiv_eprint_url(aid, self.base_url), f"eprint/{safe_paper_id(aid)}")
                if data is None:
                    continue
                if data[:4] == b"%PDF":
                    secs = pdf_to_sections(data)
                    if secs:
                        return FulltextDocument(paper.paper_id, self.name, "pdf", secs)
                    continue
                src = read_latex_source(data)
                if src is not None:
                    secs = latex_to_sections(src[0], src[1])
                    if secs:
                        return FulltextDocument(paper.paper_id, self.name, "latex", secs)
            else:
                data = self._get(arxiv_pdf_url(aid, base=self.base_url), f"pdf/{safe_paper_id(aid)}.pdf")
                if data is not None and data[:4] == b"%PDF":
                    secs = pdf_to_sections(data)
                    if secs:
                        return FulltextDocument(paper.paper_id, self.name, "pdf", secs)
        return None


class S2ORCSource(FulltextSource):
    """S2ORC v2 shards (downloaded with ``scripts/download_data.py --only s2orc_v2``).

    Bulk source: :meth:`fetch_many` streams every shard once and yields the
    records whose arXiv id (``openaccessinfo.externalids.ArXiv``) or
    ``corpusid`` (matched against ``Paper.metadata["s2_corpus_id"]``)
    belongs to a requested paper.
    """

    name = "s2orc"

    def __init__(self, paths: Union[PathLike, Sequence[PathLike]]):
        self.paths = paths

    def fetch(self, paper: Paper) -> Optional[FulltextDocument]:
        for _, doc in self.fetch_many([paper]):
            return doc
        return None

    def fetch_many(self, papers: Sequence[Paper]) -> Iterator[Tuple[Paper, Optional[FulltextDocument]]]:
        by_arxiv: Dict[str, Paper] = {}
        by_corpus: Dict[str, Paper] = {}
        for p in papers:
            aid = parse_arxiv_id(p.arxiv_id or p.paper_id)
            if aid:
                by_arxiv[aid] = p
            cid = parse_s2_corpus_id((p.metadata or {}).get("s2_corpus_id"))
            if cid:
                by_corpus[cid] = p
        found: Set[str] = set()
        for rec in iter_s2orc_records(self.paths):
            paper = None
            aid = s2orc_arxiv_id(rec)
            if aid and aid in by_arxiv:
                paper = by_arxiv[aid]
            elif str(rec.get("corpusid", "")) in by_corpus:
                paper = by_corpus[str(rec["corpusid"])]
            if paper is None or paper.paper_id in found:
                continue
            secs = s2orc_sections(rec)
            if secs:
                found.add(paper.paper_id)
                yield paper, FulltextDocument(paper.paper_id, self.name, "s2orc", secs)
                if len(found) == len(papers):
                    break
        for p in papers:
            if p.paper_id not in found:
                yield p, None


def make_sources(
    order: Sequence[str] = DEFAULT_SOURCE_ORDER,
    local_dir: Optional[PathLike] = None,
    s2orc_paths: Optional[Union[PathLike, Sequence[PathLike]]] = None,
    cache_dir: Optional[PathLike] = None,
    user_agent: str = USER_AGENT,
    arxiv_interval_s: float = 3.0,
) -> List[FulltextSource]:
    """Instantiate sources in ``order``; sources lacking their configuration are skipped with a warning."""
    sources: List[FulltextSource] = []
    for name in order:
        if name == "local":
            if local_dir is None or not Path(local_dir).exists():
                log.warning("Full-text source 'local' skipped: no directory given / found (%s)", local_dir)
                continue
            sources.append(LocalSource(local_dir))
        elif name == "s2orc":
            if not s2orc_paths:
                log.warning("Full-text source 's2orc' skipped: no S2ORC shards given")
                continue
            sources.append(S2ORCSource(s2orc_paths))
        elif name == "gcs":
            sources.append(GCSPdfSource(cache_dir=cache_dir))
        elif name == "arxiv":
            sources.append(ArxivSource(cache_dir=cache_dir, min_interval_s=arxiv_interval_s, user_agent=user_agent))
        else:
            raise ValueError(f"Unknown full-text source {name!r}; choose from {DEFAULT_SOURCE_ORDER}")
    return sources


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------


@dataclass
class FulltextStats:
    """Per-source success statistics of a :class:`FulltextExtractor` run."""

    papers: int = 0
    resumed: int = 0  # already finished in a previous run
    with_fulltext: int = 0  # cumulative (incl. resumed)
    abstract_only: int = 0
    chunks: int = 0  # written in this run
    attempted: Dict[str, int] = field(default_factory=dict)
    succeeded: Dict[str, int] = field(default_factory=dict)
    kinds: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        rates = {s: round(self.succeeded.get(s, 0) / n, 4) for s, n in self.attempted.items() if n}
        return {"papers": self.papers, "resumed": self.resumed, "with_fulltext": self.with_fulltext,
                "abstract_only": self.abstract_only, "chunks_written": self.chunks,
                "attempted": self.attempted, "succeeded": self.succeeded, "success_rate": rates, "kinds": self.kinds}


class FulltextExtractor:
    """Fetch full texts from ``sources`` (in order) and write section chunks.

    Parameters
    ----------
    sources:
        Ordered :class:`FulltextSource` objects (:func:`make_sources`).
    max_tokens, min_tokens:
        Chunk size bounds (``IndexConfig.chunk_max_tokens`` / ``chunk_min_tokens``).
    drop_front_matter:
        Drop title block / abstract sections when other sections exist.
    """

    def __init__(self, sources: Sequence[FulltextSource], max_tokens: int = 512, min_tokens: int = 32,
                 drop_front_matter: bool = True):
        self.sources = list(sources)
        self.max_tokens = int(max_tokens)
        self.min_tokens = int(min_tokens)
        self.drop_front_matter = bool(drop_front_matter)

    @classmethod
    def from_index_config(cls, sources: Sequence[FulltextSource], config: Any) -> "FulltextExtractor":
        return cls(sources, max_tokens=config.chunk_max_tokens, min_tokens=config.chunk_min_tokens)

    def chunks_for(self, doc: FulltextDocument) -> List[Chunk]:
        return sections_to_chunks(doc.paper_id, doc.sections, self.max_tokens, self.min_tokens, self.drop_front_matter)

    @staticmethod
    def load_state(state_path: PathLike) -> Dict[str, Dict[str, Any]]:
        state: Dict[str, Dict[str, Any]] = {}
        path = Path(state_path)
        if path.exists():
            for rec in read_jsonl(path):
                state[str(rec["paper_id"])] = rec
        return state

    def _reconcile(self, out_path: Path, state_path: Path, state: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Make the chunk file and the state file consistent after an interrupted run.

        A paper counts as finished only if the state file says ``ok`` *and*
        the chunk file holds exactly ``n_chunks`` of its chunks. Chunks of
        unfinished papers (and a truncated gzip tail) are removed, so the
        papers are processed again.
        """
        counts: Dict[str, int] = {}
        truncated = False
        if out_path.exists():
            try:
                for d in read_jsonl(out_path):
                    pid = str(d.get("paper_id"))
                    counts[pid] = counts.get(pid, 0) + 1
            except (EOFError, OSError, ValueError):
                truncated = True
        bad = {pid for pid, n in counts.items()
               if state.get(pid, {}).get("status") != "ok" or state[pid].get("n_chunks") != n}
        lost = {pid for pid, rec in state.items() if rec.get("status") == "ok" and counts.get(pid, 0) != rec.get("n_chunks")}
        if truncated or bad:
            log.warning("Repairing %s: dropping chunks of %d unfinished paper(s)%s", out_path, len(bad),
                        " and a truncated tail" if truncated else "")
            tmp = out_path.with_name(out_path.name + ".repair")
            opener = gzip.open if out_path.suffix == ".gz" else open
            with opener(tmp, "wt", encoding="utf-8") as dst:
                try:
                    for d in read_jsonl(out_path):
                        if str(d.get("paper_id")) not in bad:
                            dst.write(json.dumps(d, ensure_ascii=False) + "\n")
                except (EOFError, OSError, ValueError):
                    pass
            tmp.replace(out_path)
        if bad or lost:
            for pid in bad | lost:
                state.pop(pid, None)
            tmp = state_path.with_name(state_path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for rec in state.values():
                    f.write(json.dumps(rec) + "\n")
            tmp.replace(state_path)
        return state

    def run(
        self,
        papers: Iterable[Paper],
        out_path: PathLike,
        state_path: Optional[PathLike] = None,
        limit: Optional[int] = None,
        retry_missing: bool = False,
        progress_every: int = 1000,
    ) -> FulltextStats:
        """Process ``papers`` and append chunks to ``out_path`` (``.jsonl.gz``); resumable via ``state_path``.

        The state file (default ``<out stem>.state.jsonl`` next to the
        output) has one line per processed paper:
        ``{"paper_id", "status": "ok"|"missing", "source", "kind", "n_chunks"}``.
        Finished papers are skipped; papers without full text are retried only
        with ``retry_missing``.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        state_path = Path(state_path) if state_path else out_path.with_name(out_path.name.split(".")[0] + ".state.jsonl")
        state = self._reconcile(out_path, state_path, self.load_state(state_path))
        stats = FulltextStats()
        todo: List[Paper] = []
        for i, p in enumerate(papers):
            if limit is not None and i >= limit:
                break
            stats.papers += 1
            prev = state.get(p.paper_id)
            if prev is not None and (prev.get("status") == "ok" or not retry_missing):
                stats.resumed += 1
                if prev.get("status") == "ok":
                    stats.with_fulltext += 1
                else:
                    stats.abstract_only += 1
                continue
            todo.append(p)
        remaining: Dict[str, Paper] = {p.paper_id: p for p in todo}
        opener = gzip.open if out_path.suffix == ".gz" else open
        with opener(out_path, "at", encoding="utf-8") as out, open(state_path, "a", encoding="utf-8") as st:
            for source in self.sources:
                if not remaining:
                    break
                batch = list(remaining.values())
                stats.attempted[source.name] = stats.attempted.get(source.name, 0) + len(batch)
                for paper, doc in source.fetch_many(batch):
                    if doc is None or paper.paper_id not in remaining:
                        continue
                    chunks = self.chunks_for(doc)
                    if not chunks:
                        continue
                    for c in chunks:
                        out.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
                    out.flush()  # chunks first, then the state line (see _reconcile)
                    st.write(json.dumps({"paper_id": paper.paper_id, "status": "ok", "source": source.name,
                                         "kind": doc.kind, "n_chunks": len(chunks)}) + "\n")
                    st.flush()
                    del remaining[paper.paper_id]
                    stats.chunks += len(chunks)
                    stats.with_fulltext += 1
                    stats.succeeded[source.name] = stats.succeeded.get(source.name, 0) + 1
                    stats.kinds[doc.kind] = stats.kinds.get(doc.kind, 0) + 1
                    if progress_every and stats.with_fulltext % progress_every == 0:
                        log.info("full text for %d papers (%d chunks this run)", stats.with_fulltext, stats.chunks)
            for pid in remaining:
                st.write(json.dumps({"paper_id": pid, "status": "missing",
                                     "tried": [s.name for s in self.sources]}) + "\n")
            stats.abstract_only += len(remaining)
        return stats
