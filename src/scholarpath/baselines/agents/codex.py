"""Codex CLI adapter [51] (Tables 3-5, "Codex").

Appendix C.3: "Codex (v0.42), configured with its LLM backbone set to
DeepSeek-V4.0-Flash (matching ScholarPath), is executed inside an isolated
container (``unshare -n``) in which only the LLM backend is reachable; the
local corpus is mounted as JSONL shard files indexed by paper ID and primary
category". This adapter reproduces that setup:

* **Backbone.** A private ``CODEX_HOME/config.toml`` declares a
  ``[model_providers.scholarpath]`` entry whose ``base_url`` is the
  OpenAI-compatible backend (through the metering proxy), ``wire_api =
  "chat"``, ``model = <backbone>``, ``approval_policy = "never"`` and the
  web-search tool disabled.
* **Isolation** (``isolation="unshare"``, Linux). Codex runs in a fresh user +
  network namespace (``unshare --user --map-root-user --net``), whose only
  interface is loopback. The host-side :class:`MeteringProxy` listens on a
  Unix-domain socket; inside the namespace this module's ``sandbox-run``
  entry point brings ``lo`` up and bridges ``127.0.0.1:<port>`` to that
  socket, so the LLM backend (and nothing else) is reachable. The proxy
  injects the real API key; Codex only sees a dummy key.
* **Corpus.** Codex's working directory is the exported shard directory
  (:func:`~scholarpath.baselines.agents.corpus_export.export_corpus_shards`),
  bind-mounted read-only inside the namespace (``readonly_corpus``) and
  protected by Codex's own ``read-only`` sandbox.
* **Run.** ``codex exec --json --skip-git-repo-check --sandbox read-only
  --output-last-message <file> <prompt>`` runs non-interactively; the prompt
  is the uniform task prompt preceded by a one-paragraph note on where the
  corpus lives. The final message is read from ``--output-last-message``
  (falling back to the last agent message in the JSON event stream); calls and
  tokens come from the proxy, or from the event log's usage records.

Setup: ``npm install -g @openai/codex@0.42.0`` (or put the ``codex`` binary on
``PATH``); ``isolation="unshare"`` needs unprivileged user namespaces
(``sysctl kernel.unprivileged_userns_clone=1`` on some distributions).

**Version.** Pin the paper's v0.42: later Codex releases removed the
chat-completions wire API and reject ``wire_api = "chat"`` when they parse
``config.toml`` (the backbone endpoint of the paper is chat-completions
only). The adapter reads ``codex --version`` and warns when it is not 0.42;
if Codex rejects the chat wire API, the run aborts with
:class:`AgentUnavailableError` naming the fix (install
``@openai/codex@0.42.0``, or ``wire_api="responses"`` with an endpoint that
implements the Responses API) instead of failing every query.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from scholarpath.baselines.agents.base import (
    AgentAdapter,
    AgentConfig,
    AgentRun,
    AgentUnavailableError,
    LLMEndpoint,
    MeteringProxy,
    free_port,
)
from scholarpath.types import Query

log = logging.getLogger(__name__)

PROVIDER_ID = "scholarpath"
API_KEY_VAR = "SCHOLARPATH_AGENT_API_KEY"
#: Codex version of the paper (App. C.3: "Codex (v0.42)"); later releases removed ``wire_api = "chat"``.
PAPER_CODEX_VERSION = (0, 42)
INSTALL_HINT = "npm install -g @openai/codex@0.42.0"

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def parse_codex_version(text: str) -> Optional[Tuple[int, int, int]]:
    """``"codex-cli 0.42.0"`` -> ``(0, 42, 0)`` (``None`` if no version number is found)."""
    m = _VERSION_RE.search(text or "")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def codex_version(binary: str, timeout_s: float = 30.0) -> Optional[Tuple[int, int, int]]:
    """Version reported by ``<binary> --version`` (``None`` when it cannot be determined)."""
    try:
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as err:
        log.debug("codex --version failed: %s", err)
        return None
    if proc.returncode != 0:
        return None
    return parse_codex_version((proc.stdout or "") + " " + (proc.stderr or ""))


def chat_wire_api_rejected(stderr: str) -> bool:
    """Whether Codex refused ``wire_api = "chat"`` (removed after v0.42; codex-rs ``CHAT_WIRE_API_REMOVED_ERROR``)."""
    text = (stderr or "").lower()
    return "wire_api" in text and "chat" in text and ("no longer supported" in text or "unknown variant" in text)

ENVIRONMENT_NOTE = (
    "Environment: you are running non-interactively without internet access. The local paper corpus is in the "
    "current working directory (read-only): see README.md, manifest.json and paper_index.tsv; papers are JSON "
    "lines under shards/<primary category>/. Use shell tools (rg, grep, jq, python) to search it.\n\n"
)


def _toml_str(value: str) -> str:
    return json.dumps(value)  # JSON string escaping is valid TOML basic-string escaping


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else _toml_str(key)


def _toml_value(value: Any) -> str:
    """A TOML value: strings, booleans, numbers, arrays and (inline) tables."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"cannot write non-finite float {value!r} to TOML")
        return repr(value)
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_toml_key(str(k))} = {_toml_value(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value of type {type(value).__name__}: {value!r}")


def _merge(base: Dict[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge ``extra`` into ``base``; dotted keys (``"a.b"``) address nested tables."""
    for key, value in extra.items():
        parts = str(key).split(".")
        node = base
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = node[part] = {}
            node = child
        last = parts[-1]
        if isinstance(value, Mapping) and isinstance(node.get(last), dict):
            _merge(node[last], value)
        else:
            node[last] = dict(value) if isinstance(value, Mapping) else value
    return base


def dumps_toml(data: Mapping[str, Any]) -> str:
    """Serialize a nested mapping as TOML (scalars first, then ``[tables]``)."""
    lines: List[str] = []

    def emit(table: Mapping[str, Any], prefix: Tuple[str, ...]) -> None:
        scalars = [(k, v) for k, v in table.items() if not isinstance(v, Mapping)]
        tables = [(k, v) for k, v in table.items() if isinstance(v, Mapping)]
        if prefix and (scalars or not tables):
            if lines:
                lines.append("")
            lines.append("[" + ".".join(_toml_key(p) for p in prefix) + "]")
        for k, v in scalars:
            lines.append(f"{_toml_key(str(k))} = {_toml_value(v)}")
        for k, v in tables:
            emit(v, prefix + (str(k),))

    emit(data, ())
    return "\n".join(lines) + "\n"


def codex_config_toml(
    model: str,
    base_url: str,
    sandbox: str = "read-only",
    wire_api: str = "chat",
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """``config.toml`` for a Codex home dedicated to one evaluation run.

    ``extra`` is deep-merged into the generated configuration (nested
    mappings become TOML tables, dotted keys address nested tables), e.g.
    ``{"shell_environment_policy": {"inherit": "core"}}`` or
    ``{"model_reasoning_effort": "high"}``.
    """
    config: Dict[str, Any] = {
        "model": model,
        "model_provider": PROVIDER_ID,
        "approval_policy": "never",
        "sandbox_mode": sandbox,
        "model_providers": {
            PROVIDER_ID: {
                "name": "ScholarPath LLM backbone",
                "base_url": base_url,
                "env_key": API_KEY_VAR,
                "wire_api": wire_api,
            }
        },
        "tools": {"web_search": False},
        "history": {"persistence": "none"},
    }
    return dumps_toml(_merge(config, extra or {}))


def parse_codex_events(lines: Iterable[str]) -> Dict[str, Any]:
    """Final message and usage from ``codex exec --json`` output.

    Handles the item/turn event schema of recent Codex versions
    (``item.completed`` with an ``agent_message`` item, ``turn.completed`` with
    ``usage``) and the older ``{"msg": {"type": ...}}`` schema
    (``agent_message``, ``task_complete``, ``token_count``).
    """
    final: Optional[str] = None
    inp = out = 0
    have_usage = False
    turns = 0
    last_total: Optional[Dict[str, Any]] = None
    events = 0
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        events += 1
        etype = ev.get("type")
        if etype == "item.completed":
            item = ev.get("item") or {}
            if item.get("type") in ("agent_message", "assistant_message") or item.get("item_type") in (
                "agent_message", "assistant_message"):
                final = item.get("text") or final
        elif etype == "turn.completed":
            turns += 1
            u = ev.get("usage") or {}
            inp += int(u.get("input_tokens") or 0)
            out += int(u.get("output_tokens") or 0)
            have_usage = True
        msg = ev.get("msg") if isinstance(ev.get("msg"), dict) else None
        if msg:
            mtype = msg.get("type")
            if mtype == "agent_message":
                final = msg.get("message") or final
            elif mtype == "task_complete":
                final = msg.get("last_agent_message") or final
            elif mtype == "token_count":
                info = msg.get("info") or {}
                last_total = info.get("total_token_usage") or (msg if "input_tokens" in msg else last_total)
    if not have_usage and last_total:
        inp = int(last_total.get("input_tokens") or 0)
        out = int(last_total.get("output_tokens") or 0)
        have_usage = True
    return {"final_message": final, "input_tokens": inp if have_usage else None,
            "output_tokens": out if have_usage else None, "turns": turns, "events": events}


class CodexAgent(AgentAdapter):
    """Codex CLI run non-interactively over the exported corpus."""

    name = "Codex"

    def __init__(
        self,
        corpus_dir: Union[str, Path],
        config: Optional[AgentConfig] = None,
        binary: str = "codex",
        isolation: str = "unshare",
        sandbox: str = "read-only",
        readonly_corpus: bool = True,
        wire_api: str = "chat",
        extra_args: Sequence[str] = (),
        extra_config: Optional[Dict[str, Any]] = None,
        valid_ids: Optional[Iterable[str]] = None,
        bridge_port: Optional[int] = None,
    ):
        if isolation not in ("unshare", "none"):
            raise ValueError("isolation must be 'unshare' or 'none'")
        super().__init__(config, valid_ids=valid_ids if valid_ids is not None else _ids_from_export(corpus_dir))
        self.corpus_dir = Path(corpus_dir)
        self.binary = binary
        self.isolation = isolation
        self.sandbox = sandbox
        self.readonly_corpus = bool(readonly_corpus)
        self.wire_api = wire_api
        self.extra_args = list(extra_args)
        self.extra_config = dict(extra_config or {})
        self.bridge_port = int(bridge_port) if bridge_port else free_port()
        self._workdir: Optional[Path] = None
        self._version_checked = False
        self.version: Optional[Tuple[int, int, int]] = None

    # ------------------------------------------------------------ checks
    def check_available(self) -> None:
        if shutil.which(self.binary) is None:
            raise AgentUnavailableError(
                f"Codex CLI {self.binary!r} not found on PATH: {INSTALL_HINT} (the paper uses v0.42)."
            )
        if not self._version_checked:
            self._version_checked = True
            self.version = codex_version(self.binary)
            if self.version is not None and self.version[:2] != PAPER_CODEX_VERSION and self.wire_api == "chat":
                log.warning("Codex %s found; the paper uses v0.42 (%s). Releases after 0.42 removed "
                            "wire_api='chat', which the chat-completions backbone endpoint needs; the run aborts "
                            "if this version rejects it.", ".".join(map(str, self.version)), INSTALL_HINT)
        if not (self.corpus_dir / "manifest.json").exists():
            raise AgentUnavailableError(
                f"{self.corpus_dir} is not a corpus export: run "
                "scholarpath.baselines.agents.corpus_export.export_corpus_shards first."
            )
        if self.isolation == "unshare":
            if sys.platform != "linux" or shutil.which("unshare") is None:
                raise AgentUnavailableError("isolation='unshare' needs Linux and util-linux 'unshare'.")
            if not self.config.meter_with_proxy:
                raise AgentUnavailableError("isolation='unshare' needs the metering proxy (the only route to the LLM).")

    # --------------------------------------------------------- workspace
    @property
    def workdir(self) -> Path:
        if self._workdir is None:
            base = self.config.workdir
            self._workdir = Path(base) / "codex" if base else Path(tempfile.mkdtemp(prefix="codex-"))
            self._workdir.mkdir(parents=True, exist_ok=True)
        return self._workdir

    @property
    def socket_path(self) -> str:
        return str(self.workdir / "llm.sock")

    def endpoint(self) -> LLMEndpoint:
        cfg = self.config
        if self.isolation == "unshare":
            if self._proxy is None:
                key = cfg.api_key(required=True) or ""
                self._proxy = MeteringProxy(cfg.resolved_base_url(), key, unix_socket=self.socket_path,
                                            timeout_s=cfg.timeout_s).start()
            return LLMEndpoint(f"http://127.0.0.1:{self.bridge_port}/v1", "sk-metered-by-scholarpath", cfg.model)
        return super().endpoint()

    def write_codex_home(self, endpoint: LLMEndpoint) -> Path:
        home = self.workdir / "codex_home"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.toml").write_text(
            codex_config_toml(endpoint.model, endpoint.base_url, self.sandbox, self.wire_api, self.extra_config),
            encoding="utf-8",
        )
        return home

    def build_command(self, prompt: str, last_message: Path, model: str) -> List[str]:
        codex = [
            self.binary, "exec", "--json", "--skip-git-repo-check",
            "--cd", str(self.corpus_dir),
            "--sandbox", self.sandbox,
            "--model", model,
            "--output-last-message", str(last_message),
            *self.extra_args,
            prompt,
        ]
        if self.isolation == "none":
            return codex
        ns = ["unshare", "--user", "--map-root-user", "--net"]
        inner = [sys.executable, "-m", "scholarpath.baselines.agents.codex", "sandbox-run",
                 "--unix", self.socket_path, "--port", str(self.bridge_port)]
        if self.readonly_corpus:
            ns.append("--mount")
            inner += ["--ro-bind", str(self.corpus_dir)]
        return ns + ["--"] + inner + ["--"] + codex

    def build_env(self, endpoint: LLMEndpoint, codex_home: Path) -> Dict[str, str]:
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)
        env.pop(self.config.api_key_env, None)  # the real key stays with the metering proxy
        if endpoint.base_url.startswith(("http://127.0.0.1", "http://localhost")):
            for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
                env.pop(var, None)
            env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
        src_root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = src_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["CODEX_HOME"] = str(codex_home)
        env[API_KEY_VAR] = endpoint.api_key
        return env

    def task_prompt(self, query: Query) -> str:
        return ENVIRONMENT_NOTE + super().task_prompt(query)

    # --------------------------------------------------------------- run
    def run(self, query: Query, prompt: str, endpoint: Optional[LLMEndpoint]) -> AgentRun:
        assert endpoint is not None
        home = self.write_codex_home(endpoint)
        qdir = Path(tempfile.mkdtemp(prefix="q-", dir=self.workdir))
        last = qdir / "last_message.txt"
        cmd = self.build_command(prompt, last, endpoint.model)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=self.build_env(endpoint, home), capture_output=True, text=True,
                              timeout=self.config.timeout_s, cwd=str(self.corpus_dir))
        latency = time.perf_counter() - t0
        (qdir / "events.jsonl").write_text(proc.stdout or "", encoding="utf-8")
        (qdir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        parsed = parse_codex_events((proc.stdout or "").splitlines())
        final = last.read_text(encoding="utf-8") if last.exists() else (parsed["final_message"] or "")
        if proc.returncode != 0 and not final and self.wire_api == "chat" and chat_wire_api_rejected(proc.stderr):
            version = ".".join(map(str, self.version)) if self.version else "this version"
            raise AgentUnavailableError(
                f"Codex {version} rejects wire_api = \"chat\" (removed after v0.42), which the chat-completions "
                f"backbone endpoint requires. Install the paper's version ({INSTALL_HINT}), or pass "
                "wire_api='responses' with an endpoint that implements the Responses API. See "
                f"{qdir / 'stderr.log'}.")
        if proc.returncode != 0 and not final:
            tail = (proc.stderr or "")[-2000:]
            raise RuntimeError(f"codex exec failed (exit {proc.returncode}); see {qdir}:\n{tail}")
        return AgentRun(
            output_text=final,
            input_tokens=parsed["input_tokens"],
            output_tokens=parsed["output_tokens"],
            metadata={"latency_s": latency, "run_dir": str(qdir), "turns": parsed["turns"],
                      "exit_code": proc.returncode},
        )


def _ids_from_export(corpus_dir: Union[str, Path]) -> Optional[List[str]]:
    index = Path(corpus_dir) / "paper_index.tsv"
    if not index.exists():
        return None
    with open(index, encoding="utf-8") as f:
        next(f, None)
        return [line.split("\t", 1)[0] for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Inside the network namespace: loopback + TCP -> Unix-socket bridge
# ---------------------------------------------------------------------------

_SIOCGIFFLAGS = 0x8913
_SIOCSIFFLAGS = 0x8914
_IFF_UP = 0x1


def bring_up_loopback() -> None:
    """``ip link set lo up`` without iproute2 (needs CAP_NET_ADMIN in the namespace)."""
    import fcntl  # POSIX only; imported lazily so the adapter module imports everywhere

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        ifreq = struct.pack("16sh", b"lo", 0)
        flags = struct.unpack("16sh", fcntl.ioctl(s.fileno(), _SIOCGIFFLAGS, ifreq))[1]
        fcntl.ioctl(s.fileno(), _SIOCSIFFLAGS, struct.pack("16sh", b"lo", flags | _IFF_UP))


class TcpToUnixBridge:
    """Accepts TCP connections on ``host:port`` and pipes them to a Unix socket."""

    def __init__(self, host: str, port: int, unix_path: str):
        self.host, self.port, self.unix_path = host, int(port), unix_path
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()

    def start(self) -> "TcpToUnixBridge":
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(64)
        self._sock = s
        threading.Thread(target=self._accept, daemon=True).start()
        return self

    def _accept(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                client, _ = self._sock.accept()
            except OSError:
                break
            try:
                upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                upstream.connect(self.unix_path)
            except OSError:
                client.close()
                continue
            threading.Thread(target=self._pipe, args=(client, upstream), daemon=True).start()
            threading.Thread(target=self._pipe, args=(upstream, client), daemon=True).start()

    @staticmethod
    def _pipe(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            self._sock.close()


def _sandbox_run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="codex sandbox-run")
    parser.add_argument("--unix", required=True, help="host-side metering proxy socket")
    parser.add_argument("--port", type=int, required=True, help="loopback port exposed to the agent")
    parser.add_argument("--ro-bind", action="append", default=[], help="directory to remount read-only")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    bring_up_loopback()
    for d in args.ro_bind:
        subprocess.run(["mount", "--bind", d, d], check=True)
        subprocess.run(["mount", "-o", "remount,bind,ro", d], check=True)
    bridge = TcpToUnixBridge("127.0.0.1", args.port, args.unix).start()
    try:
        return subprocess.call(command)
    finally:
        bridge.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "sandbox-run":
        print("usage: python -m scholarpath.baselines.agents.codex sandbox-run --unix SOCK --port N "
              "[--ro-bind DIR] -- COMMAND ...", file=sys.stderr)
        return 2
    return _sandbox_run(argv[1:])


if __name__ == "__main__":  # pragma: no cover - executed inside the namespace
    sys.exit(main())
