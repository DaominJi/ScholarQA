"""Robust extraction of JSON objects from LLM output."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class JSONParseError(ValueError):
    """Raised when no JSON value can be parsed; ``raw_text`` holds the LLM output."""

    def __init__(self, message: str = "", raw_text: Optional[str] = None):
        super().__init__(message)
        self.raw_text = raw_text


def _balanced_spans(text: str, open_ch: str, close_ch: str):
    """Yield balanced ``open_ch ... close_ch`` substrings (string-aware)."""
    for start, ch in enumerate(text):
        if ch != open_ch:
            continue
        depth = 0
        in_str = False
        esc = False
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
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    break


def _try_load(s: str) -> Optional[Any]:
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    # Tolerate trailing commas and single-quoted keys produced by some models.
    t = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(t)
    except Exception:
        return None


def extract_json(text: str, expect: Optional[type] = None) -> Any:
    """Return the first JSON value found in ``text``.

    Tries, in order: the whole text, fenced code blocks, then the largest
    balanced ``{...}`` / ``[...]`` spans. ``expect`` (``dict`` or ``list``)
    restricts the accepted top-level type.
    """
    candidates = [text]
    candidates += [m.group(1) for m in _FENCE_RE.finditer(text or "")]
    for cand in candidates:
        val = _try_load(cand)
        if val is not None and (expect is None or isinstance(val, expect)):
            return val
    pairs = [("{", "}"), ("[", "]")]
    if expect is list:
        pairs = [("[", "]")]
    elif expect is dict:
        pairs = [("{", "}")]
    spans = []
    for o, c in pairs:
        spans.extend(_balanced_spans(text or "", o, c))
    spans.sort(key=len, reverse=True)
    for span in spans:
        val = _try_load(span)
        if val is not None and (expect is None or isinstance(val, expect)):
            return val
    raise JSONParseError(f"No JSON value found in LLM output: {text[:200]!r}", raw_text=text)
