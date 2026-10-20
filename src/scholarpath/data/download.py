"""Download the raw datasets of Section 5 into ``data/raw/<name>/``.

One entry of :data:`DATASETS` per source (App. C.2):

==================  =======  ==========================================================
name                kind     source
==================  =======  ==========================================================
taxobench           git      github.com/KongLongGeFDU/TaxoBench (``dataset/data.jsonl``)
scholarqabench      git      github.com/AkariAsai/ScholarQABench (``data/scholarqa_{cs,multi}``)
chime               git      github.com/allenai/chime (``resources/*.csv``)
chime_annotations   hf       joe32140/chime-{parent-child-relation,sibling-coherence,claim-category}
researchqa          hf       realliyifei/ResearchQA (``data/test_mini-00000-of-00001.parquet``)
researchqa_repo     git      github.com/realliyifei/ResearchQA (official coverage scorer)
arxiv               kaggle   Cornell-University/arxiv (``arxiv-metadata-oai-snapshot.json``)
s2orc_v2            s2ag     Semantic Scholar Datasets API, ``s2orc_v2`` shards (API key)
s2_papers           s2ag     Semantic Scholar Datasets API, ``papers`` shards (API key)
==================  =======  ==========================================================

Downloads are idempotent: a finished download leaves a marker file
``.scholarpath_download.json`` in its target directory and is skipped on the
next run unless ``force=True``. Every downloader only needs the standard
library (git, the kaggle CLI or ``kagglehub``, and ``huggingface_hub`` are
used when available); on failure a :class:`DownloadError` explains how to
fetch the files manually -- every loader in :mod:`scholarpath.data` also
accepts pre-downloaded local files (:func:`read_json_records` reads the JSON
or JSON Lines exports written by the manual instructions).
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

DEFAULT_RAW_ROOT = Path("data/raw")
MARKER = ".scholarpath_download.json"
USER_AGENT = "ScholarPath-data/1.0 (+https://github.com; research data preparation)"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
S2_API = "https://api.semanticscholar.org/datasets/v1"


class DownloadError(RuntimeError):
    """A download failed; the message says how to obtain the data manually."""


def read_json_records(path: PathLike) -> List[Any]:
    """Records of a JSON document or a JSON Lines file, whatever its extension (``.gz`` accepted).

    ``datasets.Dataset.to_json`` -- the manual fallback for Hugging Face
    downloads -- writes JSON Lines by default, even to ``*.json`` files, so
    the format is sniffed rather than taken from the extension. A JSON array
    is returned as is; an object holding the records under ``"data"`` or
    ``"rows"`` is unwrapped, an object of objects (records keyed by id)
    yields its values, and any other object is a single record.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        if "Extra data" not in err.msg:
            raise ValueError(f"{path}: invalid JSON ({err})") from err
        records = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as line_err:
                    raise ValueError(f"{path}:{lineno}: invalid JSON Lines record ({line_err})") from line_err
        return records
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "rows"):
            if isinstance(data.get(key), list):
                return data[key]
        if data and all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
    return [data]


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only; honour HTTPS_PROXY and custom CA bundles)
# ---------------------------------------------------------------------------


def _ssl_context() -> ssl.SSLContext:
    for var in ("SCHOLARPATH_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        cafile = os.environ.get(var)
        if cafile and Path(cafile).exists():
            return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def http_open(url: str, headers: Optional[Mapping[str, str]] = None, timeout: float = 60.0,
              retries: int = 3, backoff: float = 2.0, method: str = "GET"):
    """Open ``url`` with retries on 429 / 5xx / network errors; returns the response object.

    Raises :class:`urllib.error.HTTPError` for other HTTP errors (e.g. 404)
    and after the last retry.
    """
    hdrs = {"User-Agent": USER_AGENT}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs, method=method)
    ctx = _ssl_context()
    attempt = 0
    while True:
        try:
            return urllib.request.urlopen(req, timeout=timeout, context=ctx)
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504) or attempt >= retries:
                raise
            wait = backoff * (2 ** attempt)
            retry_after = err.headers.get("Retry-After") if err.headers else None
            if retry_after and retry_after.isdigit():
                wait = max(wait, float(retry_after))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            if attempt >= retries:
                raise
            wait = backoff * (2 ** attempt)
            log.debug("Retrying %s after %s", url, err)
        attempt += 1
        time.sleep(wait)


def http_get(url: str, headers: Optional[Mapping[str, str]] = None, timeout: float = 60.0,
             retries: int = 3) -> bytes:
    """GET ``url`` and return the body (see :func:`http_open`)."""
    with http_open(url, headers, timeout, retries) as resp:
        return resp.read()


def http_get_json(url: str, headers: Optional[Mapping[str, str]] = None, timeout: float = 60.0) -> Any:
    return json.loads(http_get(url, headers, timeout).decode("utf-8"))


def http_download(url: str, dest: PathLike, headers: Optional[Mapping[str, str]] = None,
                  timeout: float = 300.0, retries: int = 3, chunk_size: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest`` atomically (``dest.part`` then rename)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with http_open(url, headers, timeout, retries) as resp, open(part, "wb") as f:
        shutil.copyfileobj(resp, f, chunk_size)
    os.replace(part, dest)
    return dest


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str  # git | hf | kaggle | s2ag
    description: str
    target: str  # sub-directory of the raw root
    url: Optional[str] = None  # git remote
    revision: Optional[str] = None  # git commit (pinned for reproducibility)
    hf_repos: Tuple[str, ...] = ()
    hf_files: Tuple[str, ...] = ()  # specific files; empty = all data files of the repo
    kaggle_dataset: Optional[str] = None
    s2_dataset: Optional[str] = None
    manual: str = ""  # manual download instructions
    default: bool = True  # part of the default download set

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DATASETS: Dict[str, DatasetSpec] = {
    "taxobench": DatasetSpec(
        "taxobench", "git", "TaxoBench: 72 expert survey taxonomies (dataset/data.jsonl)", "taxobench",
        url="https://github.com/KongLongGeFDU/TaxoBench", revision="49a1869",
        manual="git clone https://github.com/KongLongGeFDU/TaxoBench data/raw/taxobench",
    ),
    "scholarqabench": DatasetSpec(
        "scholarqabench", "git", "ScholarQABench: ScholarQA-CS rubrics and ScholarQA-Multi expert answers",
        "scholarqabench", url="https://github.com/AkariAsai/ScholarQABench", revision="95e6fc5",
        manual="git clone https://github.com/AkariAsai/ScholarQABench data/raw/scholarqabench",
    ),
    "chime": DatasetSpec(
        "chime", "git", "CHIME: Cochrane reviews, studies, claims and Claude-2 hierarchies (resources/*.csv)",
        "chime", url="https://github.com/allenai/chime", revision="35d852a",
        manual="git clone https://github.com/allenai/chime data/raw/chime",
    ),
    "chime_annotations": DatasetSpec(
        "chime_annotations", "hf", "CHIME expert annotations (parent-child, sibling coherence, claim-category)",
        "chime_annotations",
        hf_repos=("joe32140/chime-parent-child-relation", "joe32140/chime-sibling-coherence",
                  "joe32140/chime-claim-category"),
        manual=("python -c \"from datasets import load_dataset; [load_dataset('joe32140/chime-'+n).save_to_disk("
                "'data/raw/chime_annotations/chime-'+n) for n in ['parent-child-relation','sibling-coherence',"
                "'claim-category']]\"  -- or download the parquet files from "
                "https://huggingface.co/datasets/joe32140/chime-<name> into data/raw/chime_annotations/chime-<name>/ "
                "(CSV exports with the same columns are accepted too)"),
    ),
    "researchqa": DatasetSpec(
        "researchqa", "hf", "ResearchQA test_mini split (776 queries, 5,824 rubric items)", "researchqa",
        hf_repos=("realliyifei/ResearchQA",), hf_files=("data/test_mini-00000-of-00001.parquet",),
        manual=("python -c \"from datasets import load_dataset; load_dataset('realliyifei/ResearchQA', "
                "split='test_mini').to_json('data/raw/researchqa/test_mini.jsonl', force_ascii=False)\"  -- or "
                "download data/test_mini-00000-of-00001.parquet from https://huggingface.co/datasets/realliyifei/ResearchQA"),
    ),
    "researchqa_repo": DatasetSpec(
        "researchqa_repo", "git", "ResearchQA code (official rubric-coverage scorer)", "researchqa_repo",
        url="https://github.com/realliyifei/ResearchQA", revision="747a9a1",
        manual="git clone https://github.com/realliyifei/ResearchQA data/raw/researchqa_repo", default=False,
    ),
    "arxiv": DatasetSpec(
        "arxiv", "kaggle", "arXiv metadata snapshot (arxiv-metadata-oai-snapshot.json, ~5 GB)", "arxiv",
        kaggle_dataset="Cornell-University/arxiv",
        manual=("kaggle datasets download -d Cornell-University/arxiv -p data/raw/arxiv --unzip   "
                "(needs Kaggle credentials: KAGGLE_API_TOKEN, ~/.kaggle/access_token or ~/.kaggle/kaggle.json); "
                "or download https://www.kaggle.com/datasets/Cornell-University/arxiv in a browser and place "
                "arxiv-metadata-oai-snapshot.json (or arxiv.zip) in data/raw/arxiv/"),
    ),
    "s2orc_v2": DatasetSpec(
        "s2orc_v2", "s2ag", "Semantic Scholar S2ORC v2 full-text shards (needs S2_API_KEY; very large)",
        "s2orc_v2", s2_dataset="s2orc_v2", default=False,
        manual=("export S2_API_KEY=...; python scripts/download_data.py --only s2orc_v2 [--max-shards N]; "
                "see https://api.semanticscholar.org/api-docs/datasets"),
    ),
    "s2_papers": DatasetSpec(
        "s2_papers", "s2ag", "Semantic Scholar papers shards (arXiv <-> S2 id mapping; needs S2_API_KEY)",
        "s2_papers", s2_dataset="papers", default=False,
        manual="export S2_API_KEY=...; python scripts/download_data.py --only s2_papers",
    ),
}

DEFAULT_DATASETS: Tuple[str, ...] = tuple(n for n, s in DATASETS.items() if s.default)


@dataclass
class DownloadResult:
    name: str
    status: str  # done | skipped | planned | failed
    target: str
    commands: List[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def plan_download(name: str, root: PathLike = DEFAULT_RAW_ROOT, max_shards: Optional[int] = None) -> List[str]:
    """Human-readable commands that :func:`download_dataset` would run."""
    spec = _spec(name)
    target = Path(root) / spec.target
    if spec.kind == "git":
        cmds = [f"git clone {spec.url} {target}"]
        if spec.revision:
            cmds.append(f"git -C {target} checkout {spec.revision}")
        return cmds
    if spec.kind == "hf":
        cmds = []
        for repo in spec.hf_repos:
            sub = target if len(spec.hf_repos) == 1 else target / repo.split("/")[-1]
            if spec.hf_files:
                cmds += [f"GET {HF_ENDPOINT}/datasets/{repo}/resolve/main/{f} -> {sub / f}" for f in spec.hf_files]
            else:
                cmds.append(f"snapshot of {HF_ENDPOINT}/datasets/{repo} (data files) -> {sub}")
        return cmds
    if spec.kind == "kaggle":
        return [f"kaggle datasets download -d {spec.kaggle_dataset} -p {target} --unzip"]
    if spec.kind == "s2ag":
        shards = "all" if max_shards is None else str(max_shards)
        return [f"GET {S2_API}/release/latest  (x-api-key: $S2_API_KEY)",
                f"GET {S2_API}/release/<release_id>/dataset/{spec.s2_dataset}",
                f"download {shards} shard(s) -> {target}/<release_id>/"]
    raise ValueError(f"Unknown dataset kind {spec.kind!r}")


def _spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
    return DATASETS[name]


def _write_marker(target: Path, spec: DatasetSpec, extra: Optional[Dict[str, Any]] = None) -> None:
    info = {"dataset": spec.name, "kind": spec.kind, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "revision": spec.revision}
    info.update(extra or {})
    (target / MARKER).write_text(json.dumps(info, indent=1), encoding="utf-8")


def download_dataset(
    name: str,
    root: PathLike = DEFAULT_RAW_ROOT,
    dry_run: bool = False,
    force: bool = False,
    max_shards: Optional[int] = None,
    api_key: Optional[str] = None,
) -> DownloadResult:
    """Download one dataset into ``root/<target>`` (idempotent; see module docstring)."""
    spec = _spec(name)
    target = Path(root) / spec.target
    cmds = plan_download(name, root, max_shards)
    if dry_run:
        return DownloadResult(name, "planned", str(target), cmds, spec.description)
    # S2AG shards are resumable file by file, so their marker never short-circuits a run
    # (a later run may ask for more shards than an earlier --max-shards run fetched).
    if (target / MARKER).exists() and not force and spec.kind != "s2ag":
        return DownloadResult(name, "skipped", str(target), cmds, "already downloaded (use --force to redo)")
    try:
        if spec.kind == "git":
            extra = _download_git(spec, target, force)
        elif spec.kind == "hf":
            extra = _download_hf(spec, target)
        elif spec.kind == "kaggle":
            extra = _download_kaggle(spec, target)
        elif spec.kind == "s2ag":
            extra = _download_s2ag(spec, target, max_shards, api_key)
        else:  # pragma: no cover - registry is static
            raise ValueError(spec.kind)
    except DownloadError as err:
        return DownloadResult(name, "failed", str(target), cmds, str(err))
    except Exception as err:  # network / IO errors: report with instructions
        return DownloadResult(name, "failed", str(target), cmds,
                              f"{type(err).__name__}: {err}. Manual download: {spec.manual}")
    _write_marker(target, spec, extra)
    return DownloadResult(name, "done", str(target), cmds, spec.description)


# ---------------------------------------------------------------------------
# Back-ends
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    try:
        proc = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError as err:
        raise DownloadError(f"`{cmd[0]}` is not installed ({err})") from err
    if proc.returncode != 0:
        raise DownloadError(f"`{' '.join(cmd)}` failed: {proc.stderr.strip()[-800:]}")


def _download_git(spec: DatasetSpec, target: Path, force: bool) -> Dict[str, Any]:
    if target.exists() and (target / ".git").exists() and not force:
        log.info("%s: existing clone at %s", spec.name, target)
    else:
        if target.exists() and force:
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            _run(["git", "clone", "--quiet", str(spec.url), str(target)])
        except DownloadError as err:
            raise DownloadError(f"{err}. Manual download: {spec.manual}") from err
    if spec.revision:
        try:
            _run(["git", "-C", str(target), "checkout", "--quiet", spec.revision])
        except DownloadError as err:
            log.warning("%s: could not check out pinned revision %s (%s); keeping the default branch",
                        spec.name, spec.revision, err)
            return {"revision_checked_out": False}
    return {"revision_checked_out": bool(spec.revision)}


_HF_DATA_SUFFIXES = (".parquet", ".json", ".jsonl", ".csv", ".arrow", ".md")


def _hf_list_files(repo: str) -> List[str]:
    headers = {"Authorization": f"Bearer {_hf_token()}"} if _hf_token() else {}
    url = f"{HF_ENDPOINT}/api/datasets/{repo}/tree/main?recursive=true"
    try:
        items = http_get_json(url, headers)
        return [it["path"] for it in items if it.get("type") == "file"]
    except urllib.error.HTTPError:
        info = http_get_json(f"{HF_ENDPOINT}/api/datasets/{repo}", headers)
        return [s["rfilename"] for s in info.get("siblings", [])]


def _download_hf(spec: DatasetSpec, target: Path) -> Dict[str, Any]:
    files_done: Dict[str, List[str]] = {}
    for repo in spec.hf_repos:
        sub = target if len(spec.hf_repos) == 1 else target / repo.split("/")[-1]
        sub.mkdir(parents=True, exist_ok=True)
        try:
            files_done[repo] = _hf_fetch(repo, spec.hf_files, sub)
        except Exception as err:
            raise DownloadError(
                f"Hugging Face download of {repo} failed ({type(err).__name__}: {err}). "
                f"Manual download: {spec.manual}") from err
    return {"files": files_done}


def _hf_fetch(repo: str, files: Sequence[str], dest: Path) -> List[str]:
    """Fetch files of a HF dataset repo with ``huggingface_hub`` if installed, else over plain HTTPS."""
    import importlib.util

    have_hub = importlib.util.find_spec("huggingface_hub") is not None
    if have_hub:
        from huggingface_hub import hf_hub_download, snapshot_download

        if files:
            for f in files:
                hf_hub_download(repo_id=repo, filename=f, repo_type="dataset", local_dir=str(dest))
            return list(files)
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest),
                          allow_patterns=[f"*{s}" for s in _HF_DATA_SUFFIXES])
        return sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    wanted = list(files) or [f for f in _hf_list_files(repo) if f.endswith(_HF_DATA_SUFFIXES)]
    headers = {"Authorization": f"Bearer {_hf_token()}"} if _hf_token() else {}
    for f in wanted:
        url = f"{HF_ENDPOINT}/datasets/{repo}/resolve/main/{urllib.parse.quote(f)}"
        http_download(url, dest / f, headers)
    return wanted


def _download_kaggle(spec: DatasetSpec, target: Path) -> Dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    errors = []
    if shutil.which("kaggle"):
        try:
            _run(["kaggle", "datasets", "download", "-d", str(spec.kaggle_dataset), "-p", str(target), "--unzip"])
            return {"method": "kaggle-cli"}
        except DownloadError as err:
            errors.append(str(err))
    else:
        errors.append("kaggle CLI not found (pip install kaggle)")
    try:
        import kagglehub  # type: ignore

        path = Path(kagglehub.dataset_download(str(spec.kaggle_dataset)))
        for item in path.iterdir():
            dest = target / item.name
            if not dest.exists():
                try:
                    dest.symlink_to(item)
                except OSError:
                    shutil.copy2(item, dest)
        return {"method": "kagglehub", "cache": str(path)}
    except ImportError:
        errors.append("kagglehub not installed (pip install kagglehub)")
    except Exception as err:
        errors.append(f"kagglehub failed: {err}")
    raise DownloadError("; ".join(errors) + f". Manual download: {spec.manual}")


def s2_release_files(dataset: str, api_key: str, release: str = "latest") -> Tuple[str, List[str]]:
    """``(release_id, presigned shard URLs)`` of a Semantic Scholar dataset."""
    headers = {"x-api-key": api_key}
    if release == "latest":
        release = str(http_get_json(f"{S2_API}/release/latest", headers)["release_id"])
    info = http_get_json(f"{S2_API}/release/{release}/dataset/{dataset}", headers)
    return release, list(info.get("files", []))


def _download_s2ag(spec: DatasetSpec, target: Path, max_shards: Optional[int], api_key: Optional[str]) -> Dict[str, Any]:
    key = api_key or os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not key:
        raise DownloadError(f"No Semantic Scholar API key (set S2_API_KEY). {spec.manual}")
    release, urls = s2_release_files(str(spec.s2_dataset), key)
    if max_shards is not None:
        urls = urls[: max(0, int(max_shards))]
    out = target / release
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for i, url in enumerate(urls):
        dest = out / f"{spec.s2_dataset}-{i:04d}.jsonl.gz"
        if dest.exists() and dest.stat().st_size > 0:
            done.append(dest.name)
            continue
        http_download(url, dest, timeout=3600)
        done.append(dest.name)
    return {"release": release, "shards": done}
