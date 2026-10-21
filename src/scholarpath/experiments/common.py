"""Shared infrastructure of the experiment drivers.

* **Build cache** (:class:`BuildCache`). Taxonomy builds are the expensive
  part of every construction experiment: the FULL reference alone reads every
  paper twice per level. A build is identified by a :class:`BuildKey` --
  (dataset, facet, method, seed) plus a digest of everything else that
  determines it (the id, title and abstract of every paper, the effective
  builder configuration, the root concept, the LLM and embedding backends). Within one process a key is
  built at most once (results stay in a small LRU and are reloaded from disk
  after eviction); with ``reuse=True`` (the scripts' ``--resume``) builds
  stored on disk by earlier runs, including runs of *other* scripts that share
  the cache directory, are reused as well. FULL is therefore built once per
  (dataset, facet) and reused by every comparison against it.
* **Per-build cost accounting** (:func:`fresh_meter`). Every build runs
  against a fresh :class:`~scholarpath.llm.accounting.CostMeter`, so its
  usage (the C_LLM of Eq. 15, Appendix B.2) never mixes with other builds;
  the usage is merged into the runtime meter afterwards.
* **Result I/O** (:func:`save_build_result`, :func:`load_build_result`):
  :meth:`BuildResult.save` writes ``taxonomy.json``, ``summary.json``,
  ``usage.json``, ``episodes.jsonl`` and the frozen routers;
  :meth:`BuildResult.load` reads them back.
* **Corpora** (:func:`load_corpus`, :func:`embed_corpus`, :func:`corpus_fingerprint`).
* **Aggregation** (:func:`mean_of`, :func:`aggregate_rows`).
* **Writers** (:func:`write_rows`, :func:`write_markdown`, :func:`plot_lines`,
  :func:`plot_grouped_bars`). Tables are rendered with
  :mod:`scholarpath.evaluation.reporting`; plots need matplotlib (optional,
  imported lazily).
* **CLI helpers** (:func:`csv_list`, :func:`add_experiment_args`,
  :func:`cache_from_args`, :func:`seeds_from_args`, :func:`seed_from_args`).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, TypeVar, Union,
)

import numpy as np

from scholarpath.config import ScholarPathConfig
from scholarpath.data.corpus import Corpus
from scholarpath.embeddings import l2_normalize
from scholarpath.evaluation.reporting import write_csv, write_jsonl
from scholarpath.llm.accounting import CostMeter
from scholarpath.llm.base import LLMClient
from scholarpath.taxonomy.result import BuildResult
from scholarpath.taxonomy.tree import Taxonomy
from scholarpath.types import Paper

log = logging.getLogger(__name__)

PathLike = Union[str, Path]
T = TypeVar("T")

#: Bump when the cached artefacts or the key payload change incompatibly.
#: 2: corpus fingerprints hash paper content, FULL keys ignore router settings,
#: the LLM fingerprint includes ``bill_cached_calls`` and ``json_retries``.
CACHE_VERSION = 2

#: Default location of the build cache shared by all taxonomy scripts.
DEFAULT_CACHE_DIR = Path("results") / "taxonomy_cache"

#: Default processed-data directory (``data/README.md``).
DEFAULT_DATA_DIR = Path("data") / "processed"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """JSON fallback of the experiment caches and run records (NumPy scalars / arrays, sets, paths, dataclasses)."""
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return str(value)


def stable_digest(obj: Any, length: int = 16) -> str:
    """Order-independent SHA-256 digest of a JSON-serializable object (hex prefix)."""
    blob = json.dumps(obj, sort_keys=True, default=_json_default, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[: int(length)]


def corpus_fingerprint(papers: Union[Corpus, Mapping[str, Paper], Iterable[Paper]]) -> str:
    """Digest of a corpus: the id, title and abstract of every paper (order-independent).

    These are the only paper fields a taxonomy build reads (the LLM prompts
    and the paper embeddings use title + abstract), so two corpora share a
    fingerprint only if every build on them sees the same input: subsamples
    of one corpus never collide, and neither does a corpus re-prepared with
    the same ids but revised text (e.g. a newer metadata snapshot).
    """
    if isinstance(papers, Corpus):
        items: Iterable[Paper] = papers.papers.values()
    elif isinstance(papers, Mapping):
        items = papers.values()
    else:
        items = papers
    by_id = {str(p.paper_id): p for p in items}
    h = hashlib.sha256()
    for pid in sorted(by_id):
        p = by_id[pid]
        for part in (pid, p.title, p.abstract):
            blob = str(part or "").encode("utf-8")
            h.update(len(blob).to_bytes(8, "little"))  # length prefix: field boundaries are unambiguous
            h.update(blob)
    return h.hexdigest()[:16]


def runtime_fingerprint(config: ScholarPathConfig) -> Dict[str, Any]:
    """The parts of the LLM and embedding configuration that change a build's output or cost.

    Operational fields (cache directories, timeouts, transport retries,
    API-key variables, batch sizes, devices) are left out on purpose.
    ``json_retries`` (re-asking after an unparsable answer) changes outputs
    and ``bill_cached_calls`` changes the billed cost, so both are kept.
    """
    llm, emb = config.llm, config.embedding
    return {
        "llm": {
            "provider": llm.provider,
            "model": llm.model,
            "base_url": llm.base_url,
            "temperature": llm.temperature,
            "max_output_tokens": llm.max_output_tokens,
            "price_input_per_m": llm.price_input_per_m,
            "price_output_per_m": llm.price_output_per_m,
            "bill_cached_calls": llm.bill_cached_calls,
            "json_retries": llm.json_retries,
            "seed": llm.seed,
        },
        "embedding": {
            "backend": emb.backend,
            "model_name": emb.model_name,
            "adapter": emb.adapter,
            "dim": emb.dim,
            "max_length": emb.max_length,
        },
    }


# ---------------------------------------------------------------------------
# Per-build cost accounting
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def fresh_meter(llm: LLMClient) -> Iterator[CostMeter]:
    """Run the enclosed code against a fresh cost meter (same prices), then merge it back.

    Every LLM call issued through ``llm`` inside the block is recorded only on
    the fresh meter, so a :class:`BuildResult` computed inside it carries the
    usage of that build alone. On exit the fresh meter is merged into the
    original one (the runtime's grand total stays complete). Builds must not
    run concurrently on the same client while the meter is swapped.
    """
    outer = llm.meter
    inner = CostMeter(outer.price_input_per_m, outer.price_output_per_m)
    llm.meter = inner
    try:
        yield inner
    finally:
        llm.meter = outer
        outer.merge(inner)


# ---------------------------------------------------------------------------
# BuildResult I/O
# ---------------------------------------------------------------------------


def save_build_result(result: BuildResult, directory: PathLike) -> Path:
    """Write a build with :meth:`BuildResult.save` (taxonomy, summary, usage, episodes, routers)."""
    directory = Path(directory)
    result.save(directory)
    return directory


def load_build_result(directory: PathLike, router_config: Optional[Any] = None) -> BuildResult:
    """Inverse of :func:`save_build_result` (:meth:`BuildResult.load`).

    Restores the taxonomy (including the per-node builder state kept in node
    metadata), the per-component usage, the paper-read counts, the episodes,
    the runtime, the metadata and the frozen routers.
    """
    return BuildResult.load(Path(directory), router_config=router_config)


# ---------------------------------------------------------------------------
# Build cache
# ---------------------------------------------------------------------------

_UNSAFE = re.compile(r"[^A-Za-z0-9._@=+-]+")


def _safe(part: str) -> str:
    return _UNSAFE.sub("_", str(part)).strip("._") or "_"


@dataclass(frozen=True)
class BuildKey:
    """Identity of one taxonomy build.

    ``digest`` covers everything not spelled out in the other fields: the
    paper contents (:func:`corpus_fingerprint`), the effective builder
    configuration, the root concept, the LLM / embedding backends and
    :data:`CACHE_VERSION`.
    """

    dataset: str
    facet: str
    method: str
    seed: int
    digest: str

    @property
    def relpath(self) -> Path:
        leaf = f"seed{int(self.seed)}-{self.digest}"
        return Path(_safe(self.dataset)) / _safe(self.facet) / _safe(self.method) / leaf

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def make_build_key(dataset: str, facet: str, method: str, seed: int, payload: Mapping[str, Any]) -> BuildKey:
    """Key of a build; ``payload`` holds every further input that determines the result."""
    digest = stable_digest({"version": CACHE_VERSION, "dataset": dataset, "facet": facet, "method": method,
                            "seed": int(seed), "payload": dict(payload)})
    return BuildKey(str(dataset), str(facet), str(method), int(seed), digest)


class BuildCache:
    """In-memory LRU plus on-disk store of :class:`BuildResult` objects.

    Args:
        root: cache directory (``None`` keeps every result in memory only).
        reuse: also load builds stored on disk by earlier runs (``--resume``).
            Without it a key is rebuilt (and its stored copy overwritten) the
            first time it is requested in this process; later requests in the
            same process are served from memory or reloaded from that copy.
        max_memory: results kept in memory when ``root`` is set (least recently
            used first out). Builds carry one MLP router per internal node, so
            a paper-scale run cannot keep all of them; evicted results are
            reloaded from disk on demand.

    :meth:`get_or_build` returns a shared object: treat it as read-only and
    ``copy.deepcopy`` it before mutating it (e.g. for a dynamic update).
    ``stats`` counts ``built`` (new builds), ``memory_hits``, ``disk_hits``
    (builds of earlier runs loaded with ``reuse``) and ``reloaded`` (builds of
    this process reloaded after eviction).
    """

    MARKER = "cache.json"

    def __init__(self, root: Optional[PathLike] = None, reuse: bool = False, max_memory: int = 8):
        self.root = Path(root) if root is not None else None
        self.reuse = bool(reuse)
        self.max_memory: Optional[int] = None if self.root is None else max(1, int(max_memory))
        self._memo: "OrderedDict[BuildKey, BuildResult]" = OrderedDict()
        self._stored: Set[BuildKey] = set()
        self.stats: Dict[str, int] = {"built": 0, "memory_hits": 0, "disk_hits": 0, "reloaded": 0}
        self.events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------- paths
    def path(self, key: BuildKey) -> Optional[Path]:
        return None if self.root is None else self.root / key.relpath

    def contains(self, key: BuildKey) -> bool:
        """True if a complete copy of ``key`` is stored on disk."""
        p = self.path(key)
        return p is not None and (p / self.MARKER).exists()

    # --------------------------------------------------------------- i/o
    def load(self, key: BuildKey, router_config: Optional[Any] = None) -> BuildResult:
        p = self.path(key)
        if p is None or not self.contains(key):
            raise KeyError(f"build {key.relpath} is not in the cache")
        return load_build_result(p, router_config)

    def store(self, key: BuildKey, result: BuildResult, payload: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
        """Write ``result`` atomically (temporary directory, then rename); the marker file is written last."""
        p = self.path(key)
        if p is None:
            return None
        tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        save_build_result(result, tmp)
        marker = {"key": key.to_dict(), "payload": dict(payload or {}), "stored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "cost_usd": result.cost_usd, "version": CACHE_VERSION}
        (tmp / self.MARKER).write_text(json.dumps(marker, indent=1, default=_json_default), encoding="utf-8")
        if p.exists():
            shutil.rmtree(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, p)
        return p

    # --------------------------------------------------------------- api
    def get_or_build(
        self,
        key: BuildKey,
        build: Callable[[], BuildResult],
        payload: Optional[Mapping[str, Any]] = None,
        router_config: Optional[Any] = None,
    ) -> BuildResult:
        """The build of ``key``: from memory, from disk (if ``reuse``), or by calling ``build()``.

        ``router_config`` (a :class:`~scholarpath.config.RouterConfig`) is
        attached to the routers restored from disk.
        """
        if key in self._memo:
            self._memo.move_to_end(key)
            self.stats["memory_hits"] += 1
            self.events.append({"event": "memory_hit", **key.to_dict()})
            return self._memo[key]
        if key in self._stored and self.contains(key):
            result = self.load(key, router_config)
            self.stats["reloaded"] += 1
            self.events.append({"event": "reloaded", **key.to_dict()})
        elif self.reuse and self.contains(key):
            result = self.load(key, router_config)
            self.stats["disk_hits"] += 1
            self.events.append({"event": "disk_hit", **key.to_dict()})
            log.info("reusing cached build %s", key.relpath)
        else:
            result = build()
            if self.store(key, result, payload) is not None:
                self._stored.add(key)
            self.stats["built"] += 1
            self.events.append({"event": "built", **key.to_dict(), "cost_usd": result.cost_usd,
                                "runtime_s": result.runtime_s})
        self._memo[key] = result
        while self.max_memory is not None and len(self._memo) > self.max_memory:
            self._memo.popitem(last=False)
        return result

    def built_keys(self, method: Optional[str] = None) -> List[BuildKey]:
        """Keys built (not loaded) in this process, optionally filtered by method."""
        return [BuildKey(e["dataset"], e["facet"], e["method"], e["seed"], e["digest"]) for e in self.events
                if e["event"] == "built" and (method is None or e["method"] == method)]

    def summary(self) -> Dict[str, Any]:
        return {"root": str(self.root) if self.root is not None else None, "reuse": self.reuse, **self.stats}


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------


def load_corpus(path: PathLike, limit_papers: Optional[int] = None, sample_seed: int = 0,
                name: Optional[str] = None) -> Corpus:
    """Load a processed corpus; ``limit_papers`` keeps a uniform seeded sample (quick runs).

    The processed files are sorted by paper id (for arXiv: by submission
    date), so a seeded uniform sample is used instead of the first lines.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus file {path} does not exist (see data/README.md)")
    corpus = Corpus.load(path, name=name)
    if limit_papers is not None and 0 < int(limit_papers) < len(corpus):
        corpus = corpus.sample(int(limit_papers), seed=int(sample_seed), name=corpus.name)
    return corpus


def embed_corpus(embedder: Any, papers: Union[Corpus, Mapping[str, Paper], Sequence[Paper]],
                 batch_size: int = 1024) -> Dict[str, np.ndarray]:
    """Unit-norm float32 embeddings of the paper summaries (title + abstract), keyed by paper id."""
    if isinstance(papers, Corpus):
        items = list(papers.papers.values())
    elif isinstance(papers, Mapping):
        items = list(papers.values())
    else:
        items = list(papers)
    out: Dict[str, np.ndarray] = {}
    for start in range(0, len(items), max(1, int(batch_size))):
        chunk = items[start : start + batch_size]
        vecs = l2_normalize(np.asarray(embedder.encode([p.text for p in chunk]), dtype=np.float32))
        for p, v in zip(chunk, vecs):
            out[p.paper_id] = v
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def finite(value: Any) -> Optional[float]:
    """``float(value)`` if it is a finite number, else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def mean_of(values: Iterable[Any]) -> Optional[float]:
    """Mean of the finite numbers among ``values`` (``None`` if there are none)."""
    nums = [x for x in (finite(v) for v in values) if x is not None]
    return float(np.mean(nums)) if nums else None


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    by: Sequence[str],
    metrics: Sequence[str],
    count_key: str = "n_runs",
    overrides: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Group ``rows`` by the ``by`` fields (first-seen order) and average ``metrics``.

    Missing / non-finite values are ignored; a metric with no finite value is
    ``None``. ``overrides`` replaces grouping fields before grouping (e.g.
    ``{"facet": "avg"}`` pools facets).
    """
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        r = {**row, **(overrides or {})}
        groups.setdefault(tuple(r.get(k) for k in by), []).append(row)
    out: List[Dict[str, Any]] = []
    for key, members in groups.items():
        agg: Dict[str, Any] = dict(zip(by, key))
        agg[count_key] = len(members)
        for m in metrics:
            agg[m] = mean_of(r.get(m) for r in members)
        out.append(agg)
    return out


def scale(value: Any, factor: float = 100.0) -> Optional[float]:
    """``factor * value`` for finite numbers (fractions -> percent), else ``None``."""
    x = finite(value)
    return None if x is None else factor * x


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_rows(out_dir: PathLike, name: str, rows: Sequence[Mapping[str, Any]],
               fieldnames: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """``<name>.csv`` and ``<name>.jsonl`` under ``out_dir``."""
    out = Path(out_dir)
    return {"csv": write_csv(out / f"{name}.csv", rows, fieldnames=fieldnames),
            "jsonl": write_jsonl(out / f"{name}.jsonl", rows)}


def write_markdown(path: PathLike, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path


# Categorical palette in fixed order (same slots as scholarpath.evaluation.reporting),
# recessive ink / grid colours, and one marker per series so identity is never colour-only.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def matplotlib_available() -> bool:
    """Whether the optional plotting dependency is installed (every figure is also written as CSV)."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError("plotting needs matplotlib (pip install matplotlib)") from err
    return plt


def _style_axes(ax: Any, xlabel: str, ylabel: str, title: Optional[str]) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    if title:
        ax.set_title(title, color=INK, fontsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)


def _slots(names: Sequence[str], order: Optional[Sequence[str]]) -> Dict[str, int]:
    """Colour slot of every entity: its position in ``order`` (stable across figures), then first-seen."""
    slots: Dict[str, int] = {}
    for n in list(order or []) + list(names):
        if n not in slots:
            slots[n] = len(slots)
    used = [slots[n] for n in names]
    if used and max(used) >= len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} series per chart; facet the rest into another figure")
    return slots


def plot_lines(
    series: Mapping[str, Sequence[Tuple[float, Optional[float]]]],
    path: PathLike,
    *,
    xlabel: str,
    ylabel: str,
    title: Optional[str] = None,
    references: Optional[Mapping[str, Optional[float]]] = None,
    order: Optional[Sequence[str]] = None,
    xticks: Optional[Sequence[float]] = None,
    xticklabels: Optional[Sequence[str]] = None,
    yscale: str = "linear",
) -> Path:
    """Line chart: one marked line per series plus dashed horizontal reference lines.

    ``series`` maps a name to ``[(x, y), ...]`` (``None`` values are
    skipped); ``references`` maps a name to a constant (e.g. ScholarPath or
    FULL in Fig. 2). ``order`` fixes the colour slot of every entity so the
    same system keeps its colour across figures.
    """
    plt = _pyplot()
    refs = {k: v for k, v in (references or {}).items() if finite(v) is not None}
    names = list(refs) + [k for k in series if k not in refs]
    slots = _slots(names, order)
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    xs_all = sorted({float(x) for pts in series.values() for x, y in pts if finite(y) is not None} | set(xticks or []))
    for name, value in refs.items():
        i = slots[name]
        ax.axhline(float(value), color=SERIES_COLORS[i], linewidth=2, linestyle=(0, (4, 2)), label=name, zorder=2)
    for name, pts in series.items():
        if name in refs:
            continue
        pts = [(float(x), float(y)) for x, y in pts if finite(y) is not None]
        if not pts:
            continue
        i = slots[name]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=SERIES_COLORS[i], linewidth=2, marker=SERIES_MARKERS[i], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, solid_joinstyle="round", solid_capstyle="round",
                label=name, zorder=3)
    if xticks is not None:
        ax.set_xticks(list(xticks))
        if xticklabels is not None:
            ax.set_xticklabels(list(xticklabels))
    elif xs_all:
        ax.set_xticks(xs_all)
    if yscale != "linear":
        ax.set_yscale(yscale)
    _style_axes(ax, xlabel, ylabel, title)
    if len(names) >= 2:
        ax.legend(fontsize=6, frameon=False, labelcolor=INK, loc="best")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_grouped_bars(
    categories: Sequence[str],
    groups: Mapping[str, Sequence[Optional[float]]],
    path: PathLike,
    *,
    ylabel: str,
    xlabel: str = "",
    title: Optional[str] = None,
    order: Optional[Sequence[str]] = None,
    value_format: str = "{:.1f}",
) -> Path:
    """Grouped bar chart (one bar per group within each category), values printed above the bars."""
    plt = _pyplot()
    names = list(groups)
    slots = _slots(names, order)
    n = max(1, len(names))
    width = min(0.8 / n, 0.18)  # thin bars; the slot's leftover stays empty
    gap = 0.02  # surface gap between adjacent bars
    fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(categories), dtype=float)
    for k, name in enumerate(names):
        offs = (k - (n - 1) / 2.0) * width
        vals = [finite(v) for v in groups[name]]
        heights = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(x + offs, heights, width=width - gap, color=SERIES_COLORS[slots[name]], label=name, zorder=3)
        for b, v in zip(bars, vals):
            if v is not None:
                ax.annotate(value_format.format(v), (b.get_x() + b.get_width() / 2.0, b.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                            fontsize=5.5, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(list(categories))
    _style_axes(ax, xlabel, ylabel, title)
    if n >= 2:
        ax.legend(fontsize=6, frameon=False, labelcolor=INK, loc="best")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def split_csv(text: Optional[str]) -> List[str]:
    """``"a, b,,c"`` -> ``["a", "b", "c"]`` (comma-separated CLI values; :func:`csv_list` is the argparse type)."""
    return [t.strip() for t in str(text or "").split(",") if t.strip()]


def restrict_taxonomy(tax: Taxonomy, paper_ids: Iterable[str], drop_empty: bool = True) -> Taxonomy:
    """Copy of ``tax`` whose extents and residual leaves only keep ``paper_ids``.

    Used for prebuilt taxonomies that must not reference papers absent from
    the indexed corpus (e.g. built over the complete corpus) and for expert
    references restricted to a ``--limit-papers`` sample. With ``drop_empty``
    (default), every non-root concept whose extent becomes empty is removed
    together with its subtree (extents contain their descendants' papers,
    so the whole subtree is empty): a concept whose papers were all removed
    -- e.g. the new endpoint of a held-out idea (App. C.1) -- must not stay
    in the index with its name and definition.
    ``out.metadata["restriction"]`` records the removed papers and concepts.
    """
    keep = set(paper_ids)
    out = Taxonomy.from_dict(tax.to_dict())
    before = out.all_papers()
    for node in out.nodes.values():
        node.papers &= keep
        node.residual &= keep
    out.invalidate()
    removed: List[str] = []
    if drop_empty:
        for nid in out.preorder():
            if nid != out.root_id and nid in out.nodes and not out.nodes[nid].papers:
                sub = list(out.descendants(nid, include_self=True))
                removed.extend(sub)
                out.remove_subtree(nid)
    out.invalidate()
    out.metadata["restriction"] = {"papers_removed": len(before - keep), "concepts_removed": len(removed),
                                   "removed_concepts": [str(n) for n in removed[:1000]]}
    return out


def csv_list(cast: Callable[[str], T] = str) -> Callable[[str], List[T]]:
    """argparse ``type=`` for comma-separated lists (``"0.1,0.2"`` -> ``[0.1, 0.2]``)."""

    def parse(text: str) -> List[T]:
        items = [s.strip() for s in str(text).split(",") if s.strip()]
        try:
            return [cast(s) for s in items]
        except ValueError as err:
            raise argparse.ArgumentTypeError(f"invalid list {text!r}: {err}") from err

    parse.__name__ = f"list[{getattr(cast, '__name__', 'value')}]"
    return parse


def add_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    data_dir: bool = True,
    seeds: bool = True,
    plots: bool = True,
) -> argparse._ArgumentGroup:
    """Flags shared by the taxonomy experiment scripts (data, cache, quick-run limits, seeds, plots).

    Scripts that build with a single seed get ``--build-seed`` instead of ``--seeds``.
    """
    g = parser.add_argument_group("experiment")
    if data_dir:
        g.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                       help="processed data root (data/README.md layout; default: data/processed)")
    g.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                   help="build cache shared by the taxonomy scripts (default: results/taxonomy_cache)")
    g.add_argument("--resume", action="store_true",
                   help="reuse builds stored in --cache-dir by earlier runs (FULL is then never rebuilt)")
    g.add_argument("--limit-papers", type=int, default=None,
                   help="quick runs: keep a uniform seeded sample of this many papers per corpus")
    g.add_argument("--sample-seed", type=int, default=0, help="seed of the --limit-papers sample")
    if seeds:
        g.add_argument("--seeds", type=csv_list(int), default=None,
                       help="comma-separated build seeds (default: --seed if given, else builder.seed for "
                            "ScholarPath and baseline.seed for the selection baselines); "
                            "FULL always uses baseline.seed")
    else:
        g.add_argument("--build-seed", type=int, default=None,
                       help="build seed (default: --seed if given, else builder.seed for ScholarPath and "
                            "baseline.seed for the selection baselines); FULL always uses baseline.seed")
    g.add_argument("--min-router-steps", type=int, default=None,
                   help="lower bound on the AdamW steps of the baseline routers "
                        "(SelectionBuilder default 500; lower it only for quick offline runs)")
    if plots:
        g.add_argument("--no-plots", action="store_true",
                       help="skip the matplotlib figures (their CSV data is always written)")
    return g


def cache_from_args(args: argparse.Namespace) -> BuildCache:
    return BuildCache(getattr(args, "cache_dir", None), reuse=bool(getattr(args, "resume", False)))


def seeds_from_args(args: argparse.Namespace) -> Optional[List[int]]:
    """Build seeds of a multi-seed script: ``--seeds``, else the global ``--seed``, else ``None``.

    The global ``--seed`` (``config.seed``) only seeds Python's and NumPy's
    global generators, which no taxonomy build draws from (builds are seeded
    by ``builder.seed`` / ``baseline.seed``). The scripts therefore read it as
    shorthand for ``--seeds <seed>``. ``None`` means each method's configured
    default seed (:meth:`~scholarpath.experiments.taxonomy.ConstructionRunner.resolve_seed`).
    """
    seeds = getattr(args, "seeds", None)
    if seeds:
        return [int(s) for s in seeds]
    seed = getattr(args, "seed", None)
    return None if seed is None else [int(seed)]


def seed_from_args(args: argparse.Namespace) -> Optional[int]:
    """Build seed of a single-seed script: ``--build-seed``, else the global ``--seed``, else ``None``."""
    seed = getattr(args, "build_seed", None)
    if seed is None:
        seed = getattr(args, "seed", None)
    return None if seed is None else int(seed)


__all__ = [
    "BuildCache",
    "BuildKey",
    "CACHE_VERSION",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATA_DIR",
    "add_experiment_args",
    "aggregate_rows",
    "cache_from_args",
    "corpus_fingerprint",
    "csv_list",
    "split_csv",
    "embed_corpus",
    "finite",
    "fresh_meter",
    "load_build_result",
    "load_corpus",
    "make_build_key",
    "matplotlib_available",
    "restrict_taxonomy",
    "mean_of",
    "plot_grouped_bars",
    "plot_lines",
    "runtime_fingerprint",
    "save_build_result",
    "scale",
    "seed_from_args",
    "seeds_from_args",
    "stable_digest",
    "write_markdown",
    "write_rows",
]
