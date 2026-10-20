"""Tests for the unbudgeted agent adapters (Section 5.3, App. C.3), fully offline.

External systems are replaced by local fakes (an OpenAI-compatible upstream
on 127.0.0.1, a fake OpenScholar checkout, a fake ``codex`` binary), so the
adapters' real code paths -- prompt delivery, metering proxy, subprocess
handling, output parsing -- are exercised without network access.
"""

from __future__ import annotations

import http.server
import json
import shutil
import socket
import socketserver
import subprocess
import sys
import textwrap
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scholarpath.baselines import DenseRetriever, PassagePool
from scholarpath.baselines.agents import (
    AgentConfig,
    AgentUnavailableError,
    AgenticScholarAgent,
    CodexAgent,
    MeteringProxy,
    OpenScholarAgent,
    PaperQA2Agent,
    SimulatedAgent,
    codex_config_toml,
    export_corpus_shards,
    load_corpus_shards,
    normalize_paper_id,
    parse_agent_output,
    parse_codex_events,
    read_paper,
)
from scholarpath.baselines.agents.base import parse_usage
from scholarpath.baselines.agents.codex import TcpToUnixBridge
from scholarpath.baselines.agents.openscholar import map_numeric_citations
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.index import SemanticIndex
from scholarpath.prompts.baselines import AGENT_OUTPUT_SCHEMA, render_agent_task
from scholarpath.runtime import Runtime
from scholarpath.types import EVIDENCE, HYPOTHESIS_DISCOVERY

UPSTREAM_USAGE = {"prompt_tokens": 11, "completion_tokens": 7}


@pytest.fixture(scope="module")
def bench():
    return make_synthetic_benchmark(n_papers=30, seed=5)


@pytest.fixture(scope="module")
def pool(bench):
    return PassagePool.from_corpus(bench.corpus)


@pytest.fixture()
def rt():
    return Runtime.from_config()


# ------------------------------------------------------------------ fakes


class _FakeLLM(http.server.BaseHTTPRequestHandler):
    """OpenAI-compatible chat endpoint: JSON answers with a usage block; SSE streams
    that report usage only when asked (``stream_options.include_usage``), like the
    OpenAI / DeepSeek APIs. ``{"fail": true}`` requests get HTTP 429;
    ``{"never_usage": true}`` streams never report usage (backends without
    ``stream_options`` support)."""

    protocol_version = "HTTP/1.1"
    seen_auth: list = []
    seen_bodies: list = []

    def log_message(self, *args):  # quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeLLM.seen_auth.append(self.headers.get("Authorization"))
        _FakeLLM.seen_bodies.append(body)
        status = 200
        if body.get("fail"):
            status, ctype = 429, "application/json"
            data = json.dumps({"error": {"message": "rate limited", "type": "rate_limit"}})
        elif body.get("stream"):
            chunks = [
                {"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Hel"}}]},
                {"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "lo"}}]},
            ]
            if (body.get("stream_options") or {}).get("include_usage") and not body.get("never_usage"):
                chunks.append({"id": "c1", "object": "chat.completion.chunk", "choices": [], "usage": UPSTREAM_USAGE})
            data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            ctype = "text/event-stream"
        else:
            data = json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": UPSTREAM_USAGE})
            ctype = "application/json"
        raw = data.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


@pytest.fixture()
def upstream():
    server = _Server(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _FakeLLM.seen_auth = []
    _FakeLLM.seen_bodies = []
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def _post(url: str, payload: dict) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=10) as r:
        return r.read()


def _post_status(url: str, payload: dict) -> int:
    try:
        _post(url, payload)
        return 200
    except urllib.error.HTTPError as err:
        err.read()
        return err.code


# ---------------------------------------------------------- corpus export


def test_export_corpus_shards_roundtrip(tmp_path, bench):
    corpus = bench.corpus
    for i, p in enumerate(corpus):  # two primary categories
        p.primary_category = "cs.DB" if i % 2 else "q-bio.PE"
    exp = export_corpus_shards(corpus, tmp_path / "corpus", shard_size=4)
    manifest = json.loads(exp.manifest_path.read_text())
    assert manifest["num_papers"] == len(corpus) == exp.num_papers
    assert set(manifest["categories"]) == {"cs.DB", "q-bio.PE"}
    assert all(len(open(tmp_path / "corpus" / s).readlines()) <= 4 for s in exp.shards)
    back = load_corpus_shards(tmp_path / "corpus")
    assert sorted(back.ids) == sorted(corpus.ids)
    for p in corpus:
        assert back[p.paper_id].to_dict() == p.to_dict()
    pid = sorted(corpus.ids)[7]
    assert read_paper(tmp_path / "corpus", pid).to_dict() == corpus[pid].to_dict()
    rows = (tmp_path / "corpus" / "paper_index.tsv").read_text().splitlines()[1:]
    assert [r.split("\t")[0] for r in rows] == sorted(corpus.ids)
    assert "paper_index.tsv" in (tmp_path / "corpus" / "README.md").read_text()


# ---------------------------------------------------------------- parsing


def test_render_agent_task_is_uniform():
    a = render_agent_task("Which methods suit flood forecasting?", HYPOTHESIS_DISCOVERY)
    b = render_agent_task("Which methods suit flood forecasting?", HYPOTHESIS_DISCOVERY)
    assert a == b and json.dumps(AGENT_OUTPUT_SCHEMA, indent=1) in a
    assert '"established"' in a and '"hypothesis"' in a and "paper_ids" in a
    assert render_agent_task("q", EVIDENCE) != a


def test_parse_agent_output_noisy_text():
    valid = {"2101.00001", "2101.00002", "syn0003"}
    noisy = textwrap.dedent(
        """
        Sure! I searched the corpus thoroughly. Here is my answer:
        ```json
        {"answer": "GNNs help [2101.00001].",
         "records": [
           {"rank": 2, "claim": "Optimal transport could model drift", "candidate": "Optimal Transport",
            "status": "Hypothesis (unstudied)", "paper_ids": "[arXiv:2101.00002v3, fake-id]"},
           {"rank": 1, "claim": "GNNs predict binding", "status": "established",
            "paper_ids": ["2101.00001", {"paper_id": "syn0003"}, "2101.00001"],},
           {"claim": "no citations", "status": "established", "citations": []},
         ],}
        ```
        Let me know if you need more.
        """
    )
    text, items = parse_agent_output(noisy, valid_ids=valid)
    assert text == "GNNs help [2101.00001]."
    assert [it.text for it in items][:2] == ["GNNs predict binding", "Optimal transport could model drift"]
    assert items[0].status == "established" and items[0].citations == ["2101.00001", "syn0003"]
    assert items[1].status == "hypothesis" and items[1].candidate == "Optimal Transport"
    assert items[1].citations == ["2101.00002"]
    assert items[2].citations == []
    # Bare list of records with alternative field names.
    _, items = parse_agent_output('[{"text": "x", "type": "proposed idea", "citations": "syn0003"}]', valid)
    assert items[0].status == "hypothesis" and items[0].citations == ["syn0003"]
    # No JSON at all: cited lines become items.
    plain = "1. GNNs work well [2101.00001]\n- A hypothesis: OT may help [2101.00002, bogus]\nNo cite here."
    _, items = parse_agent_output(plain, valid)
    assert [it.citations for it in items] == [["2101.00001"], ["2101.00002"]]
    assert items[1].status == "hypothesis"
    assert normalize_paper_id("https://arxiv.org/abs/2101.00001v2", valid) == "2101.00001"
    assert parse_agent_output("", valid) == ("", [])


def test_parse_usage_json_and_sse():
    assert parse_usage(json.dumps({"usage": UPSTREAM_USAGE}).encode(), "application/json") == (11, 7)
    sse = b'data: {"choices": []}\n\ndata: {"type": "response.completed", "response": {"usage": ' \
          b'{"input_tokens": 5, "output_tokens": 3}}}\n\ndata: [DONE]\n\n'
    assert parse_usage(sse, "text/event-stream") == (5, 3)
    assert parse_usage(b"not json", "") is None


# ---------------------------------------------------------- metering proxy


def test_metering_proxy_forwards_and_meters(upstream):
    with MeteringProxy(upstream, api_key="REAL-KEY") as proxy:
        mark = proxy.mark()
        out = json.loads(_post(proxy.base_url + "/chat/completions", {"model": "m", "messages": []}))
        assert out["choices"][0]["message"]["content"] == "ok"
        streamed = _post(proxy.base_url + "/chat/completions", {"model": "m", "messages": [], "stream": True})
        assert b"[DONE]" in streamed
        usage = proxy.usage_since(mark)
    assert usage["llm_calls"] == 2 and usage["estimated_calls"] == 0
    assert usage["input_tokens"] == 22 and usage["output_tokens"] == 14
    assert _FakeLLM.seen_auth == ["Bearer REAL-KEY"] * 2  # the agent's dummy key never reaches the backend
    # The proxy asked the backend for the stream's usage and hid the extra chunk from the agent.
    assert _FakeLLM.seen_bodies[1]["stream_options"] == {"include_usage": True}
    assert b'"usage"' not in streamed and streamed.count(b"data:") == 3


def test_metering_proxy_does_not_bill_failed_calls(upstream):
    """Regression: rate-limited (429) requests, typically retried by agents, were
    counted as LLM calls with estimated input tokens."""
    msgs = [{"role": "user", "content": "x " * 400}]
    with MeteringProxy(upstream, api_key="K") as proxy:
        url = proxy.base_url + "/chat/completions"
        codes = [_post_status(url, {"messages": msgs, "fail": True}) for _ in range(2)]
        codes.append(_post_status(url, {"messages": msgs}))
        usage = proxy.usage_since(0)
    assert codes == [429, 429, 200]
    assert usage["llm_calls"] == 1 and usage["failed_calls"] == 2 and usage["requests"] == 3
    assert (usage["input_tokens"], usage["output_tokens"]) == (11, 7)


def test_metering_proxy_unreachable_upstream_is_a_failed_call():
    with MeteringProxy(f"http://127.0.0.1:{_free_port()}/v1", api_key="K") as proxy:
        assert _post_status(proxy.base_url + "/chat/completions", {"messages": []}) == 502
        usage = proxy.usage_since(0)
    assert usage["llm_calls"] == 0 and usage["failed_calls"] == 1 and usage["input_tokens"] == 0


def test_metering_proxy_estimates_streams_without_usage_from_content(upstream):
    """Regression: without a usage block the raw SSE bytes (JSON envelopes included)
    were counted as output tokens, overestimating streamed outputs by > 40x."""
    from scholarpath.text import count_tokens

    msgs = [{"role": "user", "content": "What is the answer?"}]
    with MeteringProxy(upstream, api_key="K") as proxy:
        out = _post(proxy.base_url + "/chat/completions", {"messages": msgs, "stream": True, "never_usage": True})
        usage = proxy.usage_since(0)
    assert b"Hel" in out and usage["llm_calls"] == 1 and usage["estimated_calls"] == 1
    assert usage["output_tokens"] == count_tokens("Hello")
    assert usage["input_tokens"] == count_tokens("What is the answer?")
    # With injection disabled the backend reports nothing either: same content-based estimate.
    with MeteringProxy(upstream, api_key="K", inject_stream_usage=False) as proxy:
        _post(proxy.base_url + "/chat/completions", {"messages": msgs, "stream": True})
        usage = proxy.usage_since(0)
    assert "stream_options" not in _FakeLLM.seen_bodies[-1]
    assert usage["estimated_calls"] == 1 and usage["output_tokens"] == count_tokens("Hello")
    # A client that asked for usage itself receives the usage chunk untouched.
    with MeteringProxy(upstream, api_key="K") as proxy:
        out = _post(proxy.base_url + "/chat/completions",
                    {"messages": msgs, "stream": True, "stream_options": {"include_usage": True}})
        assert b'"usage"' in out and proxy.usage_since(0)["output_tokens"] == 7


def test_unix_socket_proxy_with_tcp_bridge(upstream, tmp_path):
    sock = str(tmp_path / "llm.sock")
    with MeteringProxy(upstream, api_key="K", unix_socket=sock) as proxy:
        bridge = TcpToUnixBridge("127.0.0.1", 0, sock)
        bridge.port = _free_port()
        bridge.start()
        try:
            out = json.loads(_post(f"http://127.0.0.1:{bridge.port}/v1/chat/completions", {"messages": []}))
        finally:
            bridge.stop()
        assert out["usage"] == UPSTREAM_USAGE and proxy.usage_since(0)["llm_calls"] == 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------- missing dependencies


def test_external_adapters_report_missing_dependencies(bench, pool, tmp_path):
    q = bench.evidence_queries[0]
    try:
        import paperqa  # noqa: F401
    except ImportError:
        with pytest.raises(AgentUnavailableError, match="paper-qa"):
            PaperQA2Agent(bench.corpus).answer(q)
    with pytest.raises(AgentUnavailableError, match="Codex CLI"):
        CodexAgent(tmp_path, binary="definitely-not-codex-xyz").answer(q)
    with pytest.raises(AgentUnavailableError, match="OpenScholar checkout"):
        OpenScholarAgent(pool, checkout_dir=tmp_path / "missing").answer(q)
    with pytest.raises(AgentUnavailableError, match="AgenticScholar is not bundled"):
        AgenticScholarAgent(bench.problem_gold, bench.method_gold).answer(q)
    with pytest.raises(AgentUnavailableError, match="taxonomies"):
        AgenticScholarAgent(fn=lambda r: "").answer(q)


# --------------------------------------------------------- AgenticScholar


def test_agenticscholar_gets_scholarpath_taxonomies(bench, rt, tmp_path):
    index = SemanticIndex.build(bench.problem_gold, bench.method_gold, bench.corpus.papers, rt.embedder)
    seen = {}
    pid = sorted(bench.corpus.ids)[0]

    def fake(request):
        seen.update(request)
        out = {"answer": "x", "records": [{"rank": 1, "claim": "c", "status": "hypothesis",
                                            "candidate": "Gaussian Processes", "paper_ids": [pid, "unknown"]}]}
        return {"output": out, "usage": {"llm_calls": 24, "input_tokens": 1000, "output_tokens": 200}}

    agent = AgenticScholarAgent(index=index, fn=fake, config=AgentConfig(workdir=str(tmp_path)))
    ans = agent.answer(bench.hypothesis_queries[0])
    for facet, tax in (("problem", index.problem), ("method", index.method)):
        assert json.loads(Path(seen["taxonomies"][facet]).read_text()) == tax.to_dict()
    assert seen["prompt"] == render_agent_task(bench.hypothesis_queries[0].text, HYPOTHESIS_DISCOVERY)
    assert ans.llm_calls == 24 and ans.input_tokens == 1000
    assert ans.cost_usd == pytest.approx((1000 * 0.14 + 200 * 0.28) / 1e6)
    assert ans.items[0].candidate == "Gaussian Processes" and ans.cited_papers == [pid]


# ------------------------------------------------------------ OpenScholar

FAKE_RUN_PY = r'''
import argparse, json, os, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--input_file"); ap.add_argument("--output_file"); ap.add_argument("--model_name")
args, _ = ap.parse_known_args()
rec = json.loads(open(args.input_file).readline())
req = urllib.request.Request(os.environ["OPENAI_BASE_URL"] + "/chat/completions", method="POST",
    data=json.dumps({"model": args.model_name, "messages": [{"role": "user", "content": rec["input"]}]}).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
opener.open(req).read()
ctxs = list(reversed(rec["ctxs"][:3]))  # "reranked"
json.dump([{"input": rec["input"], "ctxs": ctxs, "output": "Finding A [0]. Finding B [1, 2]."}],
          open(args.output_file, "w"))
'''


def test_openscholar_adapter_with_fake_checkout(tmp_path, pool, rt, bench, upstream, monkeypatch):
    monkeypatch.setenv("SCHOLARPATH_LLM_API_KEY", "REAL-KEY")
    checkout = tmp_path / "OpenScholar"
    checkout.mkdir()
    (checkout / "run.py").write_text(FAKE_RUN_PY)
    dense = DenseRetriever(pool, rt.embedder)
    cfg = AgentConfig(base_url=upstream, workdir=str(tmp_path / "work"))
    with OpenScholarAgent(pool, checkout, cfg, initial_retriever=dense, initial_k=5) as agent:
        q = bench.evidence_queries[0]
        ans = agent.answer(q)
        ctxs = list(reversed(agent._ctxs(q)[:3]))
    assert ans.llm_calls == 1 and ans.input_tokens == 11 and ans.metadata["usage_source"] == "proxy"
    expected = map_numeric_citations("[0] [1, 2]", ctxs)
    assert ans.retrieved_papers == expected and len(expected) >= 1
    assert (tmp_path / "work" / "openscholar" / "datastore" / "passages.jsonl").exists()
    assert _FakeLLM.seen_auth == ["Bearer REAL-KEY"]


# ------------------------------------------------------------------- Codex

FAKE_CODEX = r'''#!{python}
import json, os, re, socket, sys, urllib.request
argv = sys.argv[1:]
assert argv[0] == "exec" and "--json" in argv
last = argv[argv.index("--output-last-message") + 1]
prompt = argv[-1]
toml = open(os.path.join(os.environ["CODEX_HOME"], "config.toml")).read()
base = re.search(r'base_url = "([^"]+)"', toml).group(1)
assert "web_search = false" in toml
req = urllib.request.Request(base + "/chat/completions", method="POST",
    data=json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}]}).encode(),
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer " + os.environ["SCHOLARPATH_AGENT_API_KEY"]})
urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=10).read()
direct = "reachable"
try:
    socket.create_connection(("127.0.0.1", int(os.environ["FAKE_UPSTREAM_PORT"])), timeout=2).close()
except OSError:
    direct = "blocked"
writable = os.access(os.getcwd(), os.W_OK)
pid = open("paper_index.tsv").read().splitlines()[1].split("\t")[0]
final = json.dumps({"answer": "a", "direct_upstream": direct, "corpus_writable": writable,
                    "records": [{"rank": 1, "claim": "c", "status": "established", "paper_ids": [pid]}]})
open(last, "w").write(final)
print(json.dumps({"type": "thread.started"}))
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": final}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 99, "output_tokens": 9}}))
'''


def _fake_codex(tmp_path: Path) -> str:
    path = tmp_path / "bin" / "codex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE_CODEX.replace("{python}", sys.executable))
    path.chmod(0o755)
    return str(path)


def test_codex_config_toml_is_valid_with_nested_extra():
    """Regression: nested ``extra`` settings were written with json.dumps (invalid TOML)."""
    tomllib = pytest.importorskip("tomllib")
    extra = {
        "sandbox_workspace_write": {"network_access": False, "writable_roots": ["/tmp/x"]},
        "shell_environment_policy": {"inherit": "core", "set": {"LANG": "C.UTF-8"}},
        "model_reasoning_effort": "high",
        "tools.view_image": False,  # dotted keys address nested tables
        "model_providers": {"scholarpath": {"request_max_retries": 2}},
    }
    cfg = tomllib.loads(codex_config_toml("deepseek-v4-flash", "http://127.0.0.1:8765/v1", extra=extra))
    assert cfg["model"] == "deepseek-v4-flash" and cfg["approval_policy"] == "never"
    assert cfg["sandbox_workspace_write"] == {"network_access": False, "writable_roots": ["/tmp/x"]}
    assert cfg["shell_environment_policy"] == {"inherit": "core", "set": {"LANG": "C.UTF-8"}}
    assert cfg["tools"] == {"web_search": False, "view_image": False}
    provider = cfg["model_providers"]["scholarpath"]
    assert provider["base_url"] == "http://127.0.0.1:8765/v1" and provider["request_max_retries"] == 2
    assert provider["wire_api"] == "chat" and cfg["history"] == {"persistence": "none"}
    with pytest.raises(TypeError):
        codex_config_toml("m", "http://x", extra={"bad": object()})


def test_codex_config_command_and_events(tmp_path):
    toml = codex_config_toml("deepseek-v4-flash", "http://127.0.0.1:8765/v1")
    assert 'model = "deepseek-v4-flash"' in toml and "[model_providers.scholarpath]" in toml
    assert 'base_url = "http://127.0.0.1:8765/v1"' in toml and 'wire_api = "chat"' in toml
    assert "web_search = false" in toml and 'approval_policy = "never"' in toml
    agent = CodexAgent(tmp_path, isolation="unshare", bridge_port=8765)
    cmd = agent.build_command("PROMPT", tmp_path / "last.txt", "deepseek-v4-flash")
    assert cmd[:4] == ["unshare", "--user", "--map-root-user", "--net"] and "--mount" in cmd
    assert "sandbox-run" in cmd and cmd[-1] == "PROMPT" and "--ro-bind" in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    plain = CodexAgent(tmp_path, isolation="none").build_command("P", tmp_path / "l", "m")
    assert plain[:3] == ["codex", "exec", "--json"]
    new = ['{"type":"item.completed","item":{"type":"agent_message","text":"A"}}',
           '{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":2}}',
           '{"type":"turn.completed","usage":{"input_tokens":7,"output_tokens":1}}', "garbage"]
    assert parse_codex_events(new)["final_message"] == "A"
    assert (parse_codex_events(new)["input_tokens"], parse_codex_events(new)["output_tokens"]) == (12, 3)
    old = ['{"id":"0","msg":{"type":"token_count","info":{"total_token_usage":{"input_tokens":4,"output_tokens":6}}}}',
           '{"id":"0","msg":{"type":"task_complete","last_agent_message":"B"}}']
    ev = parse_codex_events(old)
    assert ev["final_message"] == "B" and (ev["input_tokens"], ev["output_tokens"]) == (4, 6)


def _unshare_works() -> bool:
    if sys.platform != "linux" or shutil.which("unshare") is None:
        return False
    try:
        return subprocess.run(["unshare", "--user", "--map-root-user", "--net", "--mount", "true"],
                              capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


@pytest.mark.parametrize("isolation", ["none", "unshare"])
def test_codex_adapter_end_to_end(isolation, tmp_path, bench, upstream, monkeypatch):
    if isolation == "unshare" and not _unshare_works():
        pytest.skip("unprivileged user/network namespaces are not available")
    monkeypatch.setenv("SCHOLARPATH_LLM_API_KEY", "REAL-KEY")
    monkeypatch.setenv("FAKE_UPSTREAM_PORT", upstream.split(":")[2].split("/")[0])
    exp = export_corpus_shards(bench.corpus, tmp_path / "corpus", shard_size=8)
    cfg = AgentConfig(base_url=upstream, workdir=str(tmp_path / "work"), timeout_s=60)
    with CodexAgent(exp.root, cfg, binary=_fake_codex(tmp_path), isolation=isolation) as agent:
        ans = agent.answer(bench.evidence_queries[0])
    raw = json.loads(ans.metadata["raw_output"])
    assert ans.llm_calls == 1 and ans.input_tokens == 11 and ans.metadata["usage_source"] == "proxy"
    assert ans.cited_papers == [sorted(bench.corpus.ids)[0]]
    assert _FakeLLM.seen_auth == ["Bearer REAL-KEY"]
    if isolation == "unshare":
        assert raw["direct_upstream"] == "blocked"  # only the LLM backend (via the proxy) is reachable
        assert raw["corpus_writable"] is False


# --------------------------------------------------------- SimulatedAgent


def test_simulated_agent_multi_call_answer(pool, rt, bench):
    agent = SimulatedAgent(pool, rt.llm, rt.embedder, max_steps=3)
    q = bench.evidence_queries[0]
    ans = agent.answer(q)
    assert ans.llm_calls >= 3 and rt.meter.component("agent").calls == ans.llm_calls
    assert ans.input_tokens > 0 and ans.cost_usd > 0
    assert ans.items and set(ans.cited_papers) <= set(pool.paper_ids)
    assert set(q.gold_papers) & set(ans.cited_papers)
    assert len(ans.metadata["search_queries"]) >= 1
    hyp = agent.answer(bench.hypothesis_queries[0])
    assert hyp.items and all(it.status == "hypothesis" for it in hyp.items)
