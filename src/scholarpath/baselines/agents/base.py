"""Common machinery of the unbudgeted agent baselines (Section 5.3, App. C.3).

PaperQA2 [72], OpenScholar [3], AgenticScholar [42] and Codex [51] run over
the same local corpus with the same LLM backend (DeepSeek-V4.0-Flash) and no
external web access. Their token budgets cannot be controlled externally,
so they are compared as *unbudgeted* methods. Every system receives the
identical JSON-schema task prompt of
:func:`scholarpath.prompts.baselines.render_agent_task` and must return
ranked candidate records with cited paper IDs and an explicit status tag
(``established`` vs. ``hypothesis``); :func:`parse_agent_output` turns that
output -- robustly, since agents wrap JSON in prose, code fences, or forget
it altogether -- into the foundation :class:`~scholarpath.types.Answer`.

Efficiency (Table 5: calls, tokens, cost, latency per query) is measured
uniformly by routing every agent's LLM traffic through
:class:`MeteringProxy`, a local OpenAI-compatible forwarding proxy that
injects the real API key, forwards to the configured backend, and records
the ``usage`` block of every successful generation request (streamed or
not; failed / retried requests are reported separately and not billed).
The same proxy is the only route to the outside for network-isolated agents
(Codex).
"""

from __future__ import annotations

import abc
import http.server
import json
import logging
import os
import re
import socket
import socketserver
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from scholarpath.baselines.common import QueryLike, as_query
from scholarpath.config import LLMConfig
from scholarpath.llm.parsing import JSONParseError, extract_json
from scholarpath.prompts.baselines import render_agent_task
from scholarpath.text import count_tokens
from scholarpath.types import EVIDENCE, Answer, AnswerItem, Query

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class AgentUnavailableError(RuntimeError):
    """Raised when an external agent system (or its dependency) is not installed / configured."""


@dataclass
class AgentConfig:
    """Backend and run settings shared by every agent adapter.

    ``model`` / ``base_url`` / ``api_key_env`` describe the OpenAI-compatible
    LLM backend (DeepSeek-V4.0-Flash in the paper). With ``meter_with_proxy``
    the agent talks to a local :class:`MeteringProxy` instead, which forwards
    to ``base_url`` and records usage.
    """

    model: str = "deepseek-v4-flash"
    base_url: Optional[str] = None  # default: $SCHOLARPATH_LLM_BASE_URL or DeepSeek
    api_key_env: str = "SCHOLARPATH_LLM_API_KEY"
    price_input_per_m: float = 0.14
    price_output_per_m: float = 0.28
    timeout_s: float = 3600.0
    max_records: int = 10
    workdir: Optional[str] = None  # scratch space for documents, configs, logs
    corpus_dir: Optional[str] = None  # exported JSONL shards (corpus_export.export_corpus_shards)
    meter_with_proxy: bool = True
    proxy_host: str = "127.0.0.1"
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_llm_config(cls, llm: LLMConfig, **kwargs: Any) -> "AgentConfig":
        """Use the same backbone as the rest of the pipeline (``config.llm``)."""
        return cls(
            model=llm.model,
            base_url=llm.base_url,
            api_key_env=llm.api_key_env,
            price_input_per_m=llm.price_input_per_m,
            price_output_per_m=llm.price_output_per_m,
            **kwargs,
        )

    def resolved_base_url(self) -> str:
        return (
            self.base_url
            or os.environ.get("SCHOLARPATH_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")

    def api_key(self, required: bool = True) -> Optional[str]:
        key = os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY")
        if required and not key:
            raise AgentUnavailableError(
                f"No API key for the agent's LLM backend: set ${self.api_key_env} (or $OPENAI_API_KEY)."
            )
        return key

    def price(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_input_per_m + output_tokens * self.price_output_per_m) / 1e6


@dataclass
class LLMEndpoint:
    """What an agent process should use to reach the LLM backend."""

    base_url: str
    api_key: str
    model: str


@dataclass
class AgentRun:
    """Raw outcome of one agent run.

    Usage fields are ``None`` when the system does not report them (the
    metering proxy then supplies them).
    """

    output_text: str
    llm_calls: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    cited_hint: List[str] = field(default_factory=list)  # paper ids the system itself reported as cited
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_RECORD_KEYS = ("records", "candidates", "items", "results", "hypotheses", "findings", "answers")
_TEXT_KEYS = ("claim", "text", "statement", "finding", "hypothesis", "description", "title", "candidate")
_CITE_KEYS = ("paper_ids", "citations", "cited_papers", "cited_paper_ids", "papers", "sources", "references",
              "evidence", "paper_id")
_HYP_WORDS = ("hyp", "propos", "specul", "unstudied", "untested", "novel", "candidate", "conjecture")
_EST_WORDS = ("establ", "known", "evidence", "finding", "verified", "supported", "existing")
_ARXIV_PREFIX = re.compile(r"^(?:arxiv\s*:\s*|https?://arxiv\.org/(?:abs|pdf)/)", re.IGNORECASE)
_VERSION = re.compile(r"v\d+$")
_BRACKET = re.compile(r"\[([^\[\]]{1,200})\]")
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d{1,3}[.)])\s+")


def normalize_status(value: Any, default: str = "established") -> str:
    s = str(value or "").strip().lower()
    if not s:
        return default
    if any(w in s for w in _HYP_WORDS):
        return "hypothesis"
    if any(w in s for w in _EST_WORDS):
        return "established"
    return default


def normalize_paper_id(raw: Any, valid: Optional[Iterable[str]] = None) -> Optional[str]:
    """Clean a cited id (brackets, ``arXiv:`` prefixes, version suffixes).

    With ``valid`` given, returns the matching corpus id or ``None``.
    """
    if isinstance(raw, Mapping):
        raw = raw.get("paper_id") or raw.get("id") or raw.get("paperId") or ""
    s = str(raw or "").strip().strip("[]()<>{}\"'`.,;").strip()
    s = _ARXIV_PREFIX.sub("", s).strip()
    if not s:
        return None
    if valid is None:
        return s
    valid_set = valid if isinstance(valid, (set, frozenset, dict)) else set(valid)
    for cand in (s, _VERSION.sub("", s), s.lower(), s.upper()):
        if cand in valid_set:
            return cand
    return None


def _split_ids(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for v in value:
            out.extend(_split_ids(v) if isinstance(v, (list, tuple)) or (isinstance(v, str) and "," in v) else [v])
        return out
    if isinstance(value, str):
        inner = _BRACKET.findall(value)
        parts = inner if inner else [value]
        return [p.strip() for part in parts for p in re.split(r"[,;]", part) if p.strip()]
    return [value]


def _clean_ids(values: Any, valid: Optional[set]) -> List[str]:
    out: List[str] = []
    for v in _split_ids(values):
        pid = normalize_paper_id(v, valid)
        if pid and pid not in out:
            out.append(pid)
    return out


def _find_records(obj: Any) -> Optional[List[Any]]:
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict):
        return None
    for k in _RECORD_KEYS:
        if isinstance(obj.get(k), list):
            return obj[k]
    for v in obj.values():  # nested, e.g. {"result": {"records": [...]}}
        if isinstance(v, dict):
            found = _find_records(v)
            if found is not None:
                return found
    return None


def _answer_text(obj: Any) -> str:
    if isinstance(obj, dict):
        for k in ("answer", "summary", "final_answer", "response", "output"):
            if isinstance(obj.get(k), str):
                return obj[k]
    return ""


def _parse_json_payload(text: str) -> Optional[Any]:
    """The agent's JSON object or list of records (``None`` if there is none)."""
    try:
        obj = extract_json(text or "")
    except JSONParseError:
        return None
    return obj if isinstance(obj, (dict, list)) else None


def parse_agent_output(
    text: str,
    valid_ids: Optional[Iterable[str]] = None,
    default_status: str = "established",
) -> Tuple[str, List[AnswerItem]]:
    """Parse an agent's final output into ``(answer text, ranked items)``.

    Accepts the schema of :data:`~scholarpath.prompts.baselines.AGENT_OUTPUT_SCHEMA`
    and common deviations: JSON wrapped in prose or code fences, trailing
    commas, a bare list of records, alternative field names (``candidates``,
    ``citations``, ``type`` ...), ids given as ``"[a, b]"`` strings, with
    ``arXiv:`` prefixes or version suffixes, or as ``{"paper_id": ...}``
    objects. Records are ordered by their ``rank`` field when present.
    Citations outside ``valid_ids`` (the corpus) are dropped. Without
    parseable JSON, every line/bullet that cites ``[ids]`` becomes one item.
    """
    valid = set(valid_ids) if valid_ids is not None else None
    obj = _parse_json_payload(text)
    records = _find_records(obj) if obj is not None else None
    answer = _answer_text(obj) if obj is not None else ""
    items: List[AnswerItem] = []
    if records:
        ranked: List[Tuple[float, int, AnswerItem]] = []
        for pos, r in enumerate(records):
            if isinstance(r, str):
                ranked.append((float(pos + 1), pos, AnswerItem(text=r, status=default_status,
                                                           citations=_clean_ids(r, valid))))
                continue
            if not isinstance(r, dict):
                continue
            body = next((str(r[k]) for k in _TEXT_KEYS if r.get(k) not in (None, "")), "")
            cites: List[str] = []
            for k in _CITE_KEYS:
                if k in r:
                    cites.extend(c for c in _clean_ids(r[k], valid) if c not in cites)
            candidate = r.get("candidate")
            candidate = str(candidate) if candidate not in (None, "", "null", "None") else None
            status_raw = r.get("status") or r.get("type") or r.get("label") or r.get("tag")
            status = normalize_status(status_raw, "hypothesis" if (candidate and not status_raw) else default_status)
            try:
                rank = float(r.get("rank"))
            except (TypeError, ValueError):
                rank = float(pos + 1)  # ranks are 1-based in the schema
            score = r.get("score")
            ranked.append((rank, pos, AnswerItem(
                text=body,
                status=status,
                citations=cites,
                candidate=candidate,
                rationale=str(r["rationale"]) if r.get("rationale") else None,
                score=float(score) if isinstance(score, (int, float)) else None,
            )))
        ranked.sort(key=lambda t: (t[0], t[1]))
        items = [it for _, _, it in ranked]
    structured = bool(items) or bool(answer)
    if not items:
        source = answer or (text or "")
        for line in source.splitlines():
            line = _LIST_MARKER.sub("", line).strip()
            if not line:
                continue
            cites = _clean_ids(_BRACKET.findall(line), valid)
            if not cites:
                continue
            status = "hypothesis" if re.search(r"\bhypothes", line, re.IGNORECASE) else default_status
            items.append(AnswerItem(text=line, status=status, citations=cites))
    if not answer:
        answer = "\n".join(it.text for it in items) if structured else (text or "").strip()
    return answer, items


# ---------------------------------------------------------------------------
# Metering proxy
# ---------------------------------------------------------------------------


#: Path suffixes of OpenAI-compatible generation endpoints; only requests to
#: these count as LLM calls (others, e.g. ``/embeddings``, are reported apart).
GENERATION_PATHS: Tuple[str, ...] = ("/chat/completions", "/completions", "/responses", "/messages")

#: Endpoints whose streaming responses carry usage only when the client sets
#: ``stream_options.include_usage`` (OpenAI chat / legacy completions wire).
_STREAM_OPTIONS_PATHS: Tuple[str, ...] = ("/chat/completions", "/completions")


def _url_path(path: str) -> str:
    return urllib.parse.urlsplit(path).path.rstrip("/")


def is_generation_path(path: str) -> bool:
    """Whether a request path is an LLM generation call (see :data:`GENERATION_PATHS`)."""
    return _url_path(path).endswith(GENERATION_PATHS)


@dataclass
class ProxyCall:
    path: str
    status: int  # HTTP status returned to the agent (502: upstream unreachable)
    input_tokens: int
    output_tokens: int
    latency_s: float
    estimated: bool = False  # usage not reported by the backend: estimated from the message text
    kind: str = "llm"  # "llm" (generation endpoint) | "other"
    client_aborted: bool = False  # the agent closed the connection before the response ended


def _usage_from(obj: Any) -> Optional[Tuple[int, int]]:
    """Find an OpenAI chat (prompt/completion) or responses (input/output) usage block."""
    if not isinstance(obj, dict):
        return None
    for holder in (obj, obj.get("response") if isinstance(obj.get("response"), dict) else None):
        if not isinstance(holder, dict):
            continue
        u = holder.get("usage")
        if isinstance(u, dict):
            inp = u.get("prompt_tokens", u.get("input_tokens"))
            out = u.get("completion_tokens", u.get("output_tokens"))
            if inp is not None or out is not None:
                return int(inp or 0), int(out or 0)
    return None


def _is_sse(text: str, content_type: str) -> bool:
    return "text/event-stream" in content_type or text.lstrip().startswith(("data:", "event:"))


def _sse_data(text: str) -> Iterable[Any]:
    """Decoded JSON payloads of the ``data:`` lines of an SSE body."""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except ValueError:
            continue


def parse_usage(body: bytes, content_type: str = "") -> Optional[Tuple[int, int]]:
    """Usage of a (possibly SSE-streamed) chat-completions / responses body."""
    text = body.decode("utf-8", errors="replace")
    if _is_sse(text, content_type):
        found = None
        for obj in _sse_data(text):
            u = _usage_from(obj)
            if u is not None:
                found = u  # the last usage block is the final total
        return found
    try:
        return _usage_from(json.loads(text))
    except ValueError:
        return None


def _strings(obj: Any, keys: Tuple[str, ...]) -> Iterable[str]:
    """String values stored under ``keys`` anywhere in a JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, str):
                yield v
            else:
                yield from _strings(v, keys)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v, keys)


_REQUEST_TEXT_KEYS = ("content", "text", "arguments", "instructions", "input", "system", "prompt")
_RESPONSE_TEXT_KEYS = ("content", "reasoning_content", "text", "arguments", "refusal")


def _request_text(req: Any) -> str:
    """Prompt text of a generation request (messages / input / tool schemas)."""
    if not isinstance(req, dict):
        return ""
    parts = list(_strings({k: req.get(k) for k in ("messages", "input", "instructions", "system", "prompt")},
                          _REQUEST_TEXT_KEYS))
    if req.get("tools"):
        parts.append(json.dumps(req["tools"], ensure_ascii=False))  # tool schemas are part of the prompt
    return "\n".join(parts)


def _response_text(body: bytes, content_type: str = "") -> str:
    """Generated text of a (possibly streamed) response, without the JSON / SSE envelopes."""
    text = body.decode("utf-8", errors="replace")
    parts: List[str] = []
    if _is_sse(text, content_type):
        for obj in _sse_data(text):
            if not isinstance(obj, dict):
                continue
            if isinstance(obj.get("choices"), list):  # chat / completions chunks
                for ch in obj["choices"]:
                    if isinstance(ch, dict):
                        parts.extend(_strings([ch.get("delta"), {"text": ch.get("text")}], _RESPONSE_TEXT_KEYS))
            elif str(obj.get("type", "")).endswith(".delta") and isinstance(obj.get("delta"), str):
                parts.append(obj["delta"])  # responses API streaming deltas
            elif isinstance(obj.get("delta"), dict):  # messages API streaming deltas
                parts.extend(_strings(obj["delta"], _RESPONSE_TEXT_KEYS + ("partial_json",)))
        return "".join(parts)
    try:
        obj = json.loads(text)
    except ValueError:
        return text
    if isinstance(obj, dict):
        if isinstance(obj.get("choices"), list):
            parts.extend(_strings([[c.get("message"), {"text": c.get("text")}]
                                   for c in obj["choices"] if isinstance(c, dict)], _RESPONSE_TEXT_KEYS))
        elif "output" in obj:  # responses API
            parts.extend(_strings(obj["output"], _RESPONSE_TEXT_KEYS))
        elif "content" in obj:  # messages API
            parts.extend(_strings(obj["content"], _RESPONSE_TEXT_KEYS))
    return "".join(parts)


def estimate_usage(request: Any, response: bytes, content_type: str = "") -> Tuple[int, int]:
    """Token estimate for a call whose response carried no usage block.

    Counts the message text of the request (plus tool schemas) and the
    generated text of the response -- never the JSON or SSE envelopes, which
    would overestimate streamed outputs by more than an order of magnitude.
    """
    return count_tokens(_request_text(request)), count_tokens(_response_text(response, content_type))


def prepare_forward_body(
    body: Optional[bytes], path: str, inject_stream_usage: bool = True
) -> Tuple[Optional[bytes], Any, bool]:
    """(body to forward, parsed JSON request, whether usage reporting was injected).

    For streamed chat / completions requests that do not ask for it, sets
    ``stream_options.include_usage = true`` so that the backend reports the
    exact usage of the stream in a final chunk (which the proxy then hides
    from the agent, see :class:`_SSEUsageFilter`).
    """
    if not body:
        return body, None, False
    try:
        req = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return body, None, False
    if not (inject_stream_usage and isinstance(req, dict) and req.get("stream") is True
            and _url_path(path).endswith(_STREAM_OPTIONS_PATHS)):
        return body, req, False
    opts = req.get("stream_options")
    if isinstance(opts, dict) and opts.get("include_usage"):
        return body, req, False
    req = dict(req)
    req["stream_options"] = {**(opts if isinstance(opts, dict) else {}), "include_usage": True}
    return json.dumps(req).encode("utf-8"), req, True


_SSE_EVENT_END = re.compile(rb"\r?\n\r?\n")


class _SSEUsageFilter:
    """Drops usage-only SSE chunks (``choices: []`` + ``usage``) the agent did not request.

    All other events are re-emitted unchanged, so the agent sees the stream it
    would have received without the proxy's ``include_usage`` injection.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    @staticmethod
    def _usage_only(event: bytes) -> bool:
        datas = [ln.strip()[5:].strip() for ln in event.decode("utf-8", "replace").splitlines()
                 if ln.strip().startswith("data:")]
        if not datas:
            return False
        for d in datas:
            try:
                obj = json.loads(d)
            except ValueError:
                return False
            if not (isinstance(obj, dict) and obj.get("usage") and obj.get("choices") == []):
                return False
        return True

    def feed(self, chunk: bytes) -> bytes:
        self._buf.extend(chunk)
        out = bytearray()
        while True:
            m = _SSE_EVENT_END.search(self._buf)
            if m is None:
                break
            event = bytes(self._buf[: m.end()])
            del self._buf[: m.end()]
            if not self._usage_only(event):
                out.extend(event)
        return bytes(out)

    def flush(self) -> bytes:
        rest = bytes(self._buf)
        self._buf.clear()
        return b"" if rest and self._usage_only(rest) else rest


def _make_handler(proxy: "MeteringProxy") -> type:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # silence default stderr logging
            log.debug("proxy: " + fmt, *args)

        def _send_chunk(self, data: bytes) -> bool:
            """Write one HTTP chunk; False when the agent has gone away."""
            if not data:
                return True
            try:
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False

        def _forward(self) -> None:
            t0 = time.perf_counter()
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            path = self.path
            if path.startswith(proxy.path_prefix):
                path = path[len(proxy.path_prefix):]
            kind = "llm" if is_generation_path(path) else "other"
            injected = False
            req_obj: Any = None
            if self.command == "POST":
                body, req_obj, injected = prepare_forward_body(body, path, proxy.inject_stream_usage)
            url = proxy.upstream.rstrip("/") + (path if path.startswith("/") else "/" + path)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "authorization", "content-length", "connection",
                                            "accept-encoding", "transfer-encoding")}
            headers["Authorization"] = f"Bearer {proxy.api_key}"
            req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
            try:
                resp = proxy.opener.open(req, timeout=proxy.timeout_s)
                status = resp.status
            except urllib.error.HTTPError as err:
                resp, status = err, err.code
            except Exception as err:  # upstream unreachable
                msg = json.dumps({"error": {"message": f"metering proxy: {err}"}}).encode()
                if self.command == "POST":
                    proxy._record(ProxyCall(path, 502, 0, 0, time.perf_counter() - t0, False, kind))
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return
            ctype = resp.headers.get("Content-Type", "")
            self.send_response(status)
            for k, v in resp.headers.items():
                if k.lower() not in ("content-length", "transfer-encoding", "connection", "content-encoding"):
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            sse_filter = _SSEUsageFilter() if injected and "text/event-stream" in ctype else None
            captured = bytearray()
            alive = True
            read = getattr(resp, "read1", resp.read)
            while alive:
                chunk = read(65536)
                if not chunk:
                    break
                captured.extend(chunk)
                alive = self._send_chunk(sse_filter.feed(chunk) if sse_filter is not None else chunk)
            if alive and sse_filter is not None:
                alive = self._send_chunk(sse_filter.flush())
            if self.command == "POST":  # recorded before the client sees the end of the response
                usage = parse_usage(bytes(captured), ctype) if status < 400 else (0, 0)
                estimated = usage is None
                if usage is None:
                    usage = estimate_usage(req_obj, bytes(captured), ctype)
                proxy._record(ProxyCall(path, status, usage[0], usage[1], time.perf_counter() - t0,
                                        estimated, kind, client_aborted=not alive))
            if not alive:
                self.close_connection = True
                return
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        do_POST = _forward
        do_GET = _forward

    return Handler


class _ThreadingTCPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def get_request(self):  # type: ignore[override]
        conn, _ = super().get_request()
        return conn, ("unix", 0)  # BaseHTTPRequestHandler expects a (host, port) address


class MeteringProxy:
    """Local OpenAI-compatible forwarding proxy that meters LLM usage.

    Listens on ``host:port`` (TCP; ``port=0`` picks a free port) or on a
    Unix-domain socket (``unix_socket``; used to reach the host from inside
    a network namespace). Requests to ``<path_prefix>/...`` are forwarded to
    ``upstream + /...`` with the real ``Authorization`` header, so agents only
    ever see a dummy key. Every POST is recorded as a :class:`ProxyCall`.

    Metering rules (:meth:`usage_since`): only successful (HTTP < 400)
    requests to generation endpoints (:data:`GENERATION_PATHS`) count as LLM
    calls and contribute tokens; failed requests (rate limits, server errors,
    an unreachable upstream -- typically retried by the agent) are reported
    as ``failed_calls`` and other endpoints as ``other_calls``. Usage is read
    from the backend's ``usage`` block; for streamed chat requests the proxy
    asks the backend for it (``stream_options.include_usage``, when
    ``inject_stream_usage``) and hides the extra usage chunk from the agent.
    Only when no usage is reported are tokens estimated from the message
    text (``estimated_calls``).
    """

    def __init__(
        self,
        upstream: str,
        api_key: str,
        host: str = "127.0.0.1",
        port: int = 0,
        unix_socket: Optional[str] = None,
        path_prefix: str = "/v1",
        timeout_s: float = 600.0,
        inject_stream_usage: bool = True,
    ):
        self.upstream = upstream.rstrip("/")
        self.inject_stream_usage = bool(inject_stream_usage)
        self.api_key = api_key
        self.host = host
        self.port = int(port)
        self.unix_socket = unix_socket
        self.path_prefix = path_prefix.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.calls: List[ProxyCall] = []
        self._lock = threading.Lock()
        host = urllib.parse.urlsplit(self.upstream).hostname or ""
        # A loopback upstream (local vLLM, tests) must bypass any HTTP(S)_PROXY.
        self.opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
                       if host in ("127.0.0.1", "localhost", "::1") else urllib.request.build_opener())
        self._server: Optional[socketserver.BaseServer] = None
        self._thread: Optional[threading.Thread] = None

    def _record(self, call: ProxyCall) -> None:
        with self._lock:
            self.calls.append(call)

    def start(self) -> "MeteringProxy":
        if self._server is not None:
            return self
        handler = _make_handler(self)
        if self.unix_socket:
            if os.path.exists(self.unix_socket):
                os.unlink(self.unix_socket)
            self._server = _ThreadingUnixServer(self.unix_socket, handler)
        else:
            self._server = _ThreadingTCPServer((self.host, self.port), handler)
            self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="metering-proxy", daemon=True)
        self._thread.start()
        return self

    @property
    def base_url(self) -> str:
        """URL agents use (TCP mode; for a Unix socket see the Codex bridge)."""
        return f"http://{self.host}:{self.port}{self.path_prefix}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self.unix_socket and os.path.exists(self.unix_socket):
            os.unlink(self.unix_socket)

    def __enter__(self) -> "MeteringProxy":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def mark(self) -> int:
        with self._lock:
            return len(self.calls)

    def usage_since(self, mark: int = 0) -> Dict[str, Any]:
        """Usage of the requests recorded after ``mark`` (see the class docstring)."""
        with self._lock:
            calls = self.calls[mark:]
        ok = [c for c in calls if c.kind == "llm" and c.status < 400]
        return {
            "llm_calls": len(ok),
            "input_tokens": sum(c.input_tokens for c in ok),
            "output_tokens": sum(c.output_tokens for c in ok),
            "llm_latency_s": sum(c.latency_s for c in ok),
            "estimated_calls": sum(1 for c in ok if c.estimated),
            "aborted_calls": sum(1 for c in ok if c.client_aborted),
            "failed_calls": sum(1 for c in calls if c.status >= 400),
            "other_calls": sum(1 for c in calls if c.kind != "llm" and c.status < 400),
            "requests": len(calls),
        }


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------


class AgentAdapter(abc.ABC):
    """Base class of the agent adapters: ``answer(query) -> Answer``.

    Subclasses implement :meth:`run` (one system run on the uniform prompt)
    and :meth:`check_available` (raise :class:`AgentUnavailableError` with
    setup instructions when the system cannot run). Adapters for external
    processes set ``uses_endpoint = True`` to receive an :class:`LLMEndpoint`
    (the metering proxy when ``config.meter_with_proxy``).
    """

    name: str = "agent"
    uses_endpoint: bool = True

    def __init__(self, config: Optional[AgentConfig] = None, valid_ids: Optional[Iterable[str]] = None):
        self.config = config or AgentConfig()
        self.valid_ids: Optional[set] = set(valid_ids) if valid_ids is not None else None
        self._proxy: Optional[MeteringProxy] = None

    # ------------------------------------------------------------ hooks
    def check_available(self) -> None:  # noqa: B027 - optional hook
        """Raise :class:`AgentUnavailableError` when the system cannot run."""

    @abc.abstractmethod
    def run(self, query: Query, prompt: str, endpoint: Optional[LLMEndpoint]) -> AgentRun:
        """Run the system on the uniform task ``prompt``; return its final output."""

    def task_prompt(self, query: Query) -> str:
        return render_agent_task(query.text, query.kind, self.config.max_records)

    # ---------------------------------------------------------- endpoint
    def endpoint(self) -> LLMEndpoint:
        cfg = self.config
        if cfg.meter_with_proxy:
            if self._proxy is None:
                key = cfg.api_key(required=True) or ""
                self._proxy = MeteringProxy(cfg.resolved_base_url(), key, host=cfg.proxy_host,
                                            timeout_s=cfg.timeout_s).start()
            return LLMEndpoint(self._proxy.base_url, "sk-metered-by-scholarpath", cfg.model)
        return LLMEndpoint(cfg.resolved_base_url(), cfg.api_key(required=True) or "", cfg.model)

    def close(self) -> None:
        if self._proxy is not None:
            self._proxy.stop()
            self._proxy = None

    def __enter__(self) -> "AgentAdapter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------ answer
    def answer(self, query: QueryLike) -> Answer:
        q = as_query(query)
        self.check_available()
        prompt = self.task_prompt(q)
        endpoint = self.endpoint() if self.uses_endpoint else None
        mark = self._proxy.mark() if self._proxy is not None else 0
        t0 = time.perf_counter()
        run = self.run(q, prompt, endpoint)
        wall = time.perf_counter() - t0
        proxy_usage = self._proxy.usage_since(mark) if self._proxy is not None else None
        if proxy_usage is not None and proxy_usage["requests"] > 0:
            calls, inp, out = proxy_usage["llm_calls"], proxy_usage["input_tokens"], proxy_usage["output_tokens"]
            source = "proxy"
        else:
            calls, inp, out = run.llm_calls or 0, run.input_tokens or 0, run.output_tokens or 0
            source = "agent" if run.llm_calls is not None or run.input_tokens is not None else "unknown"
        cost = self.config.price(inp, out) if source == "proxy" or run.cost_usd is None else float(run.cost_usd)
        default_status = "established" if q.kind == EVIDENCE else "hypothesis"
        text, items = parse_agent_output(run.output_text, self.valid_ids, default_status)
        cited: Dict[str, None] = {}
        for it in items:
            for c in it.citations:
                cited.setdefault(c, None)
        for c in run.cited_hint:
            pid = normalize_paper_id(c, self.valid_ids)
            if pid:
                cited.setdefault(pid, None)
        latency = float(run.metadata.get("latency_s", wall))
        meta = {"system": self.name, "usage_source": source, "wall_clock_s": wall,
                "raw_output": run.output_text, **{k: v for k, v in run.metadata.items() if k != "latency_s"}}
        if proxy_usage is not None:
            meta["proxy_usage"] = proxy_usage
        return Answer(
            query_id=q.query_id,
            text=text,
            items=items,
            retrieved_papers=list(cited),
            context_tokens=0,
            llm_calls=int(calls),
            input_tokens=int(inp),
            output_tokens=int(out),
            cost_usd=float(cost),
            latency_s=latency,
            metadata=meta,
        )


def corpus_ids(source: Any) -> List[str]:
    """Paper ids of a Corpus / mapping / iterable of papers / PassagePool (for ``valid_ids``)."""
    if hasattr(source, "paper_ids") and not callable(source.paper_ids):
        return list(source.paper_ids)
    if hasattr(source, "ids"):
        return list(source.ids)
    if isinstance(source, Mapping):
        return list(source)
    return [p.paper_id if hasattr(p, "paper_id") else str(p) for p in source]
