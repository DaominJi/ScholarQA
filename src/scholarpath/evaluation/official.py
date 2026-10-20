"""Export predictions to the official benchmark formats and run the official scripts.

Appendix C.3: "ResearchQA rubric coverage and ScholarQA-CS correctness and
citation F1 use the official evaluation scripts of their benchmarks", and
ScholarQA-Multi is scored with Prometheus. :mod:`.qa_metrics` re-implements
those protocols in-process; this module produces the official input files
and, given a local checkout of a benchmark repository, invokes its scripts
in a subprocess and parses their outputs.

Official formats (see ``data/README.md`` for download notes):

* ResearchQA ``compute_coverage.py`` (github.com/realliyifei/ResearchQA):
  ``responses.json`` = ``{"<id>": {"answer": "..."}}``; the data file is the
  official split (``test.json`` / the ``test_mini`` split dumped to JSON).
  Caveat: the official script calls the pre-1.0 ``openai.ChatCompletion``
  API while its requirements ask for ``openai>=1.0``; point
  ``coverage_script`` at the v1 port of rlresearch/dr-tulu
  (``agent/evaluation/research_qa_eval/compute_coverage.py``) or patch it.
  The script only logs the average, so :func:`run_compute_coverage` recomputes
  the per-answer coverage from the written ``{id: [{rubric, score}]}`` file.
* ScholarQA-CS ``scripts/rubric_eval.py`` (github.com/AkariAsai/ScholarQABench):
  a directory of ``<system>.jsonl`` files with ``{"case_id", "answer_text"}``
  rows; run from the repository root (its litellm cache lives in
  ``./data/litellm_cache``).
* ``scripts/citation_correctness_eval.py --citations``: a JSON list of
  ``{"input", "output", "ctxs": [{"title", "text"}]}`` whose ``output`` cites
  passages with **0-based** ``[i]`` markers; results go to
  ``<file>.score_post_fix``. The script reports citation precision and recall;
  citation F1 = 2PR/(P+R) is computed here.
* ``scripts/prometheus_eval.py``: ``-b`` a JSON list of ``{"input", "output"}``
  predictions; the gold answers of ``-f`` are matched **by position**
  (``assert len(gold) == len(preds)``), so :func:`export_prometheus` writes a
  gold file in exactly the prediction order. References are never appended to
  the graded output (``response["docs"] is True`` is always false).

:func:`run_official_evaluation` chains the exporters and runners for one
benchmark's answers; ``scripts/run_official_eval.py`` applies it to the saved
answers of ``run_qa_eval.py`` / ``run_agents.py``. The checkouts come from
``evaluation.researchqa_dir`` / ``evaluation.scholarqabench_dir``
(:meth:`OfficialPaths.from_config`), else the environment variables below.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from scholarpath.evaluation.qa_metrics import (
    PROMETHEUS_ASPECTS,
    PassageLookup,
    answer_passages,
    answer_text,
    f1_score,
    has_passage_source,
    normalize_5scale,
    number_citations,
    prometheus_to_100,
    render_answer,
    render_cs_answer,
)
from scholarpath.prompts.evaluation import COVERAGE_SCORES, PROMETHEUS_INSTRUCTION, normalize_coverage_label
from scholarpath.types import Answer, Query

log = logging.getLogger(__name__)
PathLike = Union[str, Path]


class OfficialEvalError(RuntimeError):
    """An official evaluation script failed or produced no output."""


@dataclass
class OfficialPaths:
    """Local checkouts of the benchmark repositories and the interpreter to run them with.

    Defaults come from ``$SCHOLARPATH_RESEARCHQA_DIR``,
    ``$SCHOLARPATH_SCHOLARQABENCH_DIR`` and ``$SCHOLARPATH_EVAL_PYTHON``
    (an environment with the scripts' requirements: openai, litellm, torch,
    transformers, nltk, rouge_score, vllm, fastchat).
    """

    researchqa_dir: Optional[str] = field(default_factory=lambda: os.environ.get("SCHOLARPATH_RESEARCHQA_DIR"))
    scholarqabench_dir: Optional[str] = field(default_factory=lambda: os.environ.get("SCHOLARPATH_SCHOLARQABENCH_DIR"))
    python: str = field(default_factory=lambda: os.environ.get("SCHOLARPATH_EVAL_PYTHON", sys.executable))

    @classmethod
    def from_config(cls, config: Any = None, **overrides: Optional[str]) -> "OfficialPaths":
        """Paths from ``overrides``, then ``config`` (an ``EvaluationConfig``), then the environment."""
        paths = cls()
        for name in ("researchqa_dir", "scholarqabench_dir"):
            value = overrides.get(name) or (getattr(config, name, None) if config is not None else None)
            if value:
                setattr(paths, name, str(value))
        if overrides.get("python"):
            paths.python = str(overrides["python"])
        return paths

    def researchqa(self) -> Path:
        return _checkout(self.researchqa_dir, "ResearchQA", "SCHOLARPATH_RESEARCHQA_DIR")

    def scholarqabench(self) -> Path:
        return _checkout(self.scholarqabench_dir, "ScholarQABench", "SCHOLARPATH_SCHOLARQABENCH_DIR")


def _checkout(path: Optional[str], name: str, env: str) -> Path:
    if not path:
        raise OfficialEvalError(f"No {name} checkout configured (set ${env} or pass the path).")
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise OfficialEvalError(f"{name} checkout not found: {p}")
    return p


def official_id(query: Query, keys: Sequence[str] = ("official_id", "source_id", "case_id", "id")) -> str:
    """Identifier of a query in its benchmark's own files (metadata, else ``query_id``)."""
    for k in keys:
        v = (query.metadata or {}).get(k)
        if v:
            return str(v)
    return query.query_id


def _write_json(path: PathLike, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _ordered(answers: Mapping[str, Answer], queries: Optional[Sequence[Query]]) -> List[tuple]:
    if queries is None:
        return [(qid, None, a) for qid, a in answers.items()]
    return [(q.query_id, q, answers[q.query_id]) for q in queries if q.query_id in answers]


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


def export_researchqa_responses(
    answers: Mapping[str, Answer],
    path: PathLike,
    queries: Optional[Sequence[Query]] = None,
    papers: Optional[Any] = None,
    references: bool = True,
) -> Dict[str, Dict[str, str]]:
    """``responses.json`` for ``compute_coverage.py``: ``{official id: {"answer": text}}``.

    With ``papers``, each answer carries a reference list of its cited papers
    (:func:`~scholarpath.evaluation.qa_metrics.render_answer`), so that the
    judge can verify rubric items asking whether a given paper is cited.
    """
    out: Dict[str, Dict[str, str]] = {}
    for qid, q, ans in _ordered(answers, queries):
        key = official_id(q) if q is not None else qid
        out[key] = {"answer": render_answer(ans, papers, references)}
    _write_json(path, out)
    return out


def export_scholarqa_cs(
    answers: Mapping[str, Answer],
    path: PathLike,
    queries: Optional[Sequence[Query]] = None,
    papers: Optional[Any] = None,
    passage_fn: Optional[Callable[[Answer], PassageLookup]] = None,
    references: bool = True,
) -> List[Dict[str, str]]:
    """``<qa-dir>/<system>.jsonl`` rows ``{"case_id", "answer_text"}`` for ``rubric_eval.py``.

    ``answer_text`` is what ScholarQABench ``scripts/convert_answer_scholar_cs.py``
    produces from a system output: numbered citations followed by a
    ``References:`` block with the title and text of every cited passage
    (:func:`~scholarpath.evaluation.qa_metrics.render_cs_answer`; passages from
    ``passage_fn(answer)``, ``answer.metadata["passages"]`` or ``papers``).
    Answers without any passage source, or ``references=False``, are written
    as plain text, exactly as the in-process
    :func:`~scholarpath.evaluation.qa_metrics.correctness` scores them.
    """
    rows = []
    for qid, q, ans in _ordered(answers, queries):
        text = answer_text(ans)
        if references and (passage_fn is not None or has_passage_source(ans, papers, None)):
            text = render_cs_answer(ans, passage_fn(ans) if passage_fn is not None else answer_passages(ans, papers))
        rows.append({"case_id": official_id(q, ("case_id", "official_id", "id")) if q is not None else qid,
                     "answer_text": text})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def export_citation_eval(
    answers: Mapping[str, Answer],
    queries: Sequence[Query],
    path: PathLike,
    papers: Optional[Any] = None,
    passage_fn: Optional[Callable[[Answer], PassageLookup]] = None,
) -> List[Dict[str, Any]]:
    """Input of ``citation_correctness_eval.py --citations``.

    Paper-id citations are converted to 0-based ``[i]`` markers into ``ctxs``
    (:func:`~scholarpath.evaluation.qa_metrics.number_citations`); passages are
    the context text the system saw (``answer.metadata["passages"]``) or the
    corpus abstracts. Cited papers without a passage are written as
    out-of-range indices, which the script scores as unsupported. Ctxs are
    written exactly as the in-process scorer formats them (an untitled
    passage has no ``title`` key, so ``_format_document`` gives its bare text).
    """
    items = []
    for qid, q, ans in _ordered(answers, queries):
        lookup = passage_fn(ans) if passage_fn is not None else answer_passages(ans, papers)
        output, ctxs, ids = number_citations(ans, lookup)
        items.append({"id": official_id(q), "input": q.text, "output": output,
                      "ctxs": [dict(c) for c in ctxs], "paper_ids": ids})
    _write_json(path, items)
    return items


def export_prometheus(
    answers: Mapping[str, Answer],
    queries: Sequence[Query],
    path: PathLike,
    gold_path: Optional[PathLike] = None,
) -> List[Dict[str, Any]]:
    """Predictions for ``prometheus_eval.py -b`` and, optionally, a position-aligned gold file for ``-f``.

    The gold file holds ``{"id", "output": reference answer}`` rows in
    exactly the prediction order, which the script requires
    (``assert len(gold_answers) == len(responses)``, matched by position).
    """
    preds, gold = [], []
    for qid, q, ans in _ordered(answers, queries):
        preds.append({"id": official_id(q), "input": q.text, "output": answer_text(ans)})
        if gold_path is not None:
            if q.reference_answer is None:
                raise ValueError(f"Query {qid} has no reference answer for the Prometheus gold file")
            gold.append({"id": official_id(q), "output": q.reference_answer})
    _write_json(path, preds)
    if gold_path is not None:
        _write_json(gold_path, gold)
    return preds


def export_researchqa_data(queries: Sequence[Query], path: PathLike) -> List[Dict[str, Any]]:
    """The official data file of ``compute_coverage.py`` (``researchqa_loader.load_researchqa_data``) for ``queries``.

    One record per query: ``id`` (:func:`official_id`), ``general_domain``,
    ``subdomain``, ``field``, ``query``, ``date`` and the ``rubric`` items
    (``rubric_item``, ``type``; types from ``metadata["rubric_types"]``).
    """
    records = []
    for q in queries:
        meta = q.metadata or {}
        types = meta.get("rubric_types") or []
        rubric = []
        for i, item in enumerate(q.rubric):
            kind = types[i] if i < len(types) else [t for t in (item.kind or "").split(",") if t]
            rubric.append({"rubric_item": item.text, "type": kind, "citation_metadata": None})
        records.append({"id": official_id(q), "general_domain": str(meta.get("domain", "")),
                        "subdomain": str(meta.get("subdomain", "")), "field": str(meta.get("field", "")),
                        "query": q.text, "date": str(meta.get("raw_date") or meta.get("date") or ""),
                        "rubric": rubric})
    _write_json(path, records)
    return records


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str], cwd: Path, env: Optional[Mapping[str, str]] = None, timeout: Optional[float] = None) -> str:
    log.info("Running %s (cwd=%s)", " ".join(map(str, cmd)), cwd)
    full_env = dict(os.environ)
    full_env.update(env or {})
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=full_env, capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-25:])
        raise OfficialEvalError(f"{Path(str(cmd[1])).name} exited with {proc.returncode}:\n{tail}")
    return proc.stdout


_COVERAGE_DRIVER = textwrap.dedent(
    """
    import inspect, json, sys
    sys.path.insert(0, sys.argv[1])
    from compute_coverage import compute_coverage
    data, responses, output, batch_size, model = sys.argv[2:7]
    params = list(inspect.signature(compute_coverage).parameters)
    # The official script takes a path; the dr-tulu port takes the loaded response map.
    resp = json.load(open(responses, encoding="utf-8")) if params[1] == "response_map" else responses
    compute_coverage(data, resp, output, batch_size=int(batch_size), model=model)
    """
)


@dataclass
class OfficialCoverage:
    """Parsed output of ``compute_coverage.py``."""

    per_answer: Dict[str, float]  # id -> coverage in [0, 1]
    raw: Dict[str, List[Dict[str, str]]]
    rub: Optional[float]  # mean x 100 over scored answers
    unparsed_labels: int = 0


def coverage_from_judgements(raw: Mapping[str, Sequence[Mapping[str, str]]]) -> OfficialCoverage:
    """Per-answer coverage from the ``{id: [{"rubric", "score": label}]}`` output file."""
    per: Dict[str, float] = {}
    bad = 0
    for qid, judges in raw.items():
        scores = []
        for j in judges:
            label = normalize_coverage_label(str(j.get("score", "")))
            if label is None:
                bad += 1
                continue
            scores.append(normalize_5scale(COVERAGE_SCORES[label]))
        if scores:
            per[qid] = sum(scores) / len(scores)
    rub = 100.0 * sum(per.values()) / len(per) if per else None
    return OfficialCoverage(per_answer=per, raw={k: list(v) for k, v in raw.items()}, rub=rub, unparsed_labels=bad)


def run_compute_coverage(
    data_path: PathLike,
    responses_path: PathLike,
    output_path: PathLike,
    paths: Optional[OfficialPaths] = None,
    coverage_script: Optional[PathLike] = None,
    model: str = "gpt-4.1-mini",
    batch_size: int = 8,
    timeout: Optional[float] = None,
) -> OfficialCoverage:
    """Run ResearchQA ``compute_coverage`` and parse its output.

    ``coverage_script`` defaults to ``<ResearchQA checkout>/compute_coverage.py``;
    pass the dr-tulu v1 port to avoid the pre-1.0 openai API call. Requires
    ``$OPENAI_API_KEY`` in the environment.
    """
    paths = paths or OfficialPaths()
    script = Path(coverage_script) if coverage_script else paths.researchqa() / "compute_coverage.py"
    if not script.is_file():
        raise OfficialEvalError(f"compute_coverage.py not found: {script}")
    output_path = Path(output_path).resolve()
    _run([paths.python, "-c", _COVERAGE_DRIVER, str(script.parent.resolve()), str(Path(data_path).resolve()),
          str(Path(responses_path).resolve()), str(output_path), str(batch_size), model],
         cwd=script.parent, timeout=timeout)
    if not output_path.exists():
        raise OfficialEvalError(f"compute_coverage wrote no output at {output_path}")
    return coverage_from_judgements(json.loads(output_path.read_text(encoding="utf-8")))


def run_rubric_eval(
    qa_dir: PathLike,
    output_path: PathLike,
    paths: Optional[OfficialPaths] = None,
    test_config: Optional[PathLike] = None,
    src_names: Optional[Sequence[str]] = None,
    rubrics: bool = True,
    snippets: bool = True,
    agreement: bool = False,
    timeout: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run ScholarQA-CS ``scripts/rubric_eval.py``; returns ``{source: {"Corr", "ann_score", "n", "rows"}}``.

    ``qa_dir`` holds one ``<system>.jsonl`` per system (:func:`export_scholarqa_cs`).
    The judge (``gpt-4-turbo``) runs through litellm; set its API key.
    """
    paths = paths or OfficialPaths()
    root = paths.scholarqabench()
    cfg = Path(test_config) if test_config else root / "data" / "scholarqa_cs" / "test_configs_snippets.json"
    output_path = Path(output_path).resolve()
    cmd = [paths.python, str(root / "scripts" / "rubric_eval.py"), "--qa-dir", str(Path(qa_dir).resolve()),
           "--test-config", str(cfg.resolve()), "--output", str(output_path)]
    if rubrics:
        cmd.append("--rubrics")
    if snippets:
        cmd.append("--snippets")
    if agreement:
        cmd.append("--agreement")
    if src_names:
        cmd += ["--src-names", ",".join(src_names)]
    _run(cmd, cwd=root, timeout=timeout)
    raw = json.loads(output_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for src, results in raw.items():
        scores = [r["scores"]["score"] for r in results]
        anns = [r["scores"]["ann_score"] for r in results]
        out[Path(src).stem] = {
            "Corr": 100.0 * sum(scores) / len(scores) if scores else None,
            "ann_score": 100.0 * sum(anns) / len(anns) if anns else None,
            "n": float(len(results)),
            "rows": results,
        }
    return out


def run_citation_eval(
    pred_path: PathLike,
    paths: Optional[OfficialPaths] = None,
    at_most_citations: int = 3,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Run ``citation_correctness_eval.py --citations``; adds ``Cite`` = 2PR/(P+R) to its result."""
    paths = paths or OfficialPaths()
    root = paths.scholarqabench()
    pred_path = Path(pred_path).resolve()
    _run([paths.python, str(root / "scripts" / "citation_correctness_eval.py"), "--f", str(pred_path),
          "--citations", "--at_most_citations", str(at_most_citations), "--no_rouge"], cwd=root, timeout=timeout)
    result_path = Path(str(pred_path) + ".score_post_fix")
    if not result_path.exists():
        raise OfficialEvalError(f"citation_correctness_eval.py wrote no result at {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    p, r = result.get("citation_prec"), result.get("citation_rec")
    if p is not None and r is not None:
        result["Cite"] = 100.0 * f1_score(float(p) / 100.0, float(r) / 100.0)
    return result


def run_prometheus_eval(
    pred_path: PathLike,
    output_dir: PathLike,
    gold_path: Optional[PathLike] = None,
    paths: Optional[OfficialPaths] = None,
    model: str = "prometheus-eval/prometheus-7b-v2.0",
    aspects: Sequence[str] = PROMETHEUS_ASPECTS,
    instruction: str = PROMETHEUS_INSTRUCTION,
    rubric_path: Optional[PathLike] = None,
    load_vllm: bool = True,
    top_n: int = 10,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Run ``scripts/prometheus_eval.py`` and parse ``results.json``.

    Returns the script's ``summary`` (per-criterion means on 1-5 and their
    ``average``) plus ``Corr`` = average mapped to 0-100. Decoding is greedy
    (``temperature=0``, ``top_p=1``) as in the paper ("Prometheus-v2-7B
    (temperature 0)", App. C.3); the script's own defaults are 0.01 / 0.95.
    Predictions and ``gold_path`` must be position-aligned
    (:func:`export_prometheus`).
    """
    paths = paths or OfficialPaths()
    root = paths.scholarqabench()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rub = Path(rubric_path) if rubric_path else root / "rubrics" / "prometheus_rubrics_v8.json"
    cmd = [paths.python, str(root / "scripts" / "prometheus_eval.py"), "-b", str(Path(pred_path).resolve()),
           "-o", str(out_dir), "--rubric_path", str(rub.resolve()), "-i", instruction, "--model", model,
           "--top_n", str(top_n), "--aspects", *aspects]
    if gold_path is not None:
        cmd += ["-f", str(Path(gold_path).resolve())]
    else:
        cmd.append("--no_reference")
    if load_vllm:
        cmd.append("--load_vllm")
    cmd += ["--temperature", str(float(temperature)), "--top_p", str(float(top_p))]
    _run(cmd, cwd=root, timeout=timeout)
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    summary = dict(results.get("summary", {}))
    avg = summary.get("average")
    summary["Corr"] = prometheus_to_100(float(avg)) if avg is not None else None
    return summary


# ---------------------------------------------------------------------------
# One benchmark end to end
# ---------------------------------------------------------------------------

OFFICIAL_BENCHMARKS = ("researchqa", "scholarqa_cs", "scholarqa_multi")


def _zero_filled(value: Optional[float], n_scored: float, n_missing: float) -> Optional[float]:
    """A mean over ``n_scored`` answers re-averaged with ``n_missing`` zeros (failed queries)."""
    if value is None or n_missing <= 0:
        return value
    total = n_scored + n_missing
    return float(value) * n_scored / total if total > 0 else value


def run_official_evaluation(
    benchmark: str,
    answers: Mapping[str, Answer],
    queries: Sequence[Query],
    out_dir: PathLike,
    paths: Optional[OfficialPaths] = None,
    *,
    papers: Optional[Any] = None,
    system: str = "system",
    researchqa_model: str = "gpt-4.1-mini",
    coverage_script: Optional[PathLike] = None,
    prometheus_model: str = "prometheus-eval/prometheus-7b-v2.0",
    load_vllm: bool = True,
    citations: bool = True,
    missing: str = "zero",
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Score one benchmark's answers with the official scripts (App. C.3) and return the Table 3 cells.

    * ``researchqa``: ``compute_coverage.py`` (Rub). The official repository
      has no citation metric; ``Cite`` is the citation F1 against the gold
      papers of :func:`~scholarpath.evaluation.qa_metrics.citation_f1`.
    * ``scholarqa_cs``: ``rubric_eval.py --rubrics --snippets`` (Corr) and,
      with ``citations``, ``citation_correctness_eval.py`` (Cite).
    * ``scholarqa_multi``: ``prometheus_eval.py`` with the expert answers as
      references (Corr) and ``citation_correctness_eval.py`` (Cite).

    The official scripts only score the answers they are given. With
    ``missing="zero"`` (default, as :mod:`.qa_metrics`), queries without an
    answer count as 0, i.e. every mean is re-averaged over all queries;
    ``missing="skip"`` reports the official numbers as they are. Inputs and
    raw outputs are written under ``out_dir``.
    """
    from scholarpath.evaluation.qa_metrics import MISSING_POLICIES, aggregate_prf, citation_f1, citation_prf

    if benchmark not in OFFICIAL_BENCHMARKS:
        raise ValueError(f"No official script for {benchmark!r}; expected one of {OFFICIAL_BENCHMARKS}")
    if missing not in MISSING_POLICIES:
        raise ValueError(f"missing must be one of {MISSING_POLICIES}, got {missing!r}")
    paths = paths or OfficialPaths()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    answered = [q for q in queries if q.query_id in answers]
    absent = [q for q in queries if q.query_id not in answers]
    n_missing = float(len(absent)) if missing == "zero" else 0.0
    summary: Dict[str, Any] = {"benchmark": benchmark, "system": system, "n_queries": float(len(queries)),
                               "n_missing": float(len(absent))}
    if not answered:
        raise OfficialEvalError(f"No answers to score for {benchmark} ({system})")

    def add_citations(prefix: str) -> None:
        items = export_citation_eval(answers, answered, out / f"{prefix}_citations.json", papers=papers)
        if not items:
            return
        cite = run_citation_eval(out / f"{prefix}_citations.json", paths=paths, timeout=timeout)
        n = float(len(items))
        p = _zero_filled(cite.get("citation_prec"), n, n_missing)
        r = _zero_filled(cite.get("citation_rec"), n, n_missing)
        summary.update(citation_prec=p, citation_rec=r,
                       Cite=(100.0 * f1_score(p / 100.0, r / 100.0)) if p is not None and r is not None else None)

    if benchmark == "researchqa":
        export_researchqa_data(queries, out / "researchqa_data.json")
        export_researchqa_responses(answers, out / "responses.json", queries=answered, papers=papers)
        cov = run_compute_coverage(out / "researchqa_data.json", out / "responses.json", out / "coverage.json",
                                   paths=paths, coverage_script=coverage_script, model=researchqa_model,
                                   timeout=timeout)
        scored = float(len(cov.per_answer))
        summary.update(Rub=_zero_filled(cov.rub, scored, n_missing), n=scored + n_missing,
                       n_failed=float(len(answered)) - scored, unparsed_labels=float(cov.unparsed_labels))
        prfs = [citation_f1(answers[q.query_id], q, valid_ids=papers) for q in answered]
        if missing == "zero":
            prfs += [citation_prf([], q.gold_papers, q.num_gold) for q in absent]
        summary.update({k: v for k, v in aggregate_prf(prfs).items() if k != "n"})
    elif benchmark == "scholarqa_cs":
        qa_dir = out / "qa"
        export_scholarqa_cs(answers, qa_dir / f"{system}.jsonl", queries=answered, papers=papers)
        result = run_rubric_eval(qa_dir, out / "rubric_eval.json", paths=paths, src_names=[system],
                                 timeout=timeout).get(system, {})
        scored = float(result.get("n") or 0.0)
        summary.update(Corr=_zero_filled(result.get("Corr"), scored, n_missing),
                       ann_score=_zero_filled(result.get("ann_score"), scored, n_missing), n=scored + n_missing)
        if citations:
            add_citations("cs")
    else:
        export_prometheus(answers, answered, out / "prometheus_preds.json", gold_path=out / "prometheus_gold.json")
        prom = run_prometheus_eval(out / "prometheus_preds.json", out / "prometheus", out / "prometheus_gold.json",
                                   paths=paths, model=prometheus_model, load_vllm=load_vllm, timeout=timeout)
        scored = float(len(answered))
        summary.update(Corr=_zero_filled(prom.get("Corr"), scored, n_missing), n=scored + n_missing,
                       prometheus=prom)
        if citations:
            add_citations("multi")
    _write_json(out / "official_summary.json", summary)
    return summary


__all__ = [
    "OFFICIAL_BENCHMARKS",
    "OfficialCoverage",
    "OfficialEvalError",
    "OfficialPaths",
    "coverage_from_judgements",
    "export_citation_eval",
    "export_prometheus",
    "export_researchqa_data",
    "export_researchqa_responses",
    "export_scholarqa_cs",
    "official_id",
    "run_citation_eval",
    "run_compute_coverage",
    "run_official_evaluation",
    "run_prometheus_eval",
    "run_rubric_eval",
]
