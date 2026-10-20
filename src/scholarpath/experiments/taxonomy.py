"""Taxonomy-construction experiments (Section 5.2, Appendix C.4-C.5, Section 3.4 / App. B.3).

This module holds the reusable logic of the taxonomy scripts:

========================================  ==========================================================
Paper artefact                            Entry points
========================================  ==========================================================
Fig. 2 (soft node / edge F1 vs. eta)      :func:`run_construction_comparison`, :func:`figure2_rows`,
                                          :func:`plot_figure2`
Fig. 3 (relative LLM cost)                :func:`figure3_rows`, :func:`plot_figure3`
Tables 8-9 (one-off vs. node-wise)        :func:`table8`, :func:`table9`
Fig. 4 (scalability, Sec. 5.2.2)          :func:`nested_subsets`, :func:`run_scalability`,
                                          :func:`scalability_summary`, :func:`plot_figure4`
Table 6 (ablation, App. C.4)              :data:`ABLATIONS`, :func:`table6`
Dynamic update (Sec. 3.4, App. B.3)       :func:`split_stream`, :func:`run_dynamic_update`,
                                          :func:`dynamic_summary`
========================================  ==========================================================

Experimental protocol (Section 5.2)
-----------------------------------
* **Methods.** ScholarPath (Algorithm 1, :class:`~scholarpath.taxonomy.builder.NodeWiseBuilder`,
  eps = 0.10, 1 - delta = 0.95 by default), the exhaustive reference FULL
  (eta = 1) and the node-wise selection baselines Random / Coverage / K-means /
  Diversity at eta in {0.05, 0.1, 0.2, 0.3, 0.4, 0.5}, plus one-off selection
  at eta = 0.2 (App. C.5), all from
  :class:`~scholarpath.taxonomy.baselines.SelectionBuilder`. Every method uses
  the same LLM operations, router, root concept and paper embeddings.
* **arXiv sets.** On each of the eight 2,000-paper sets every method builds
  *both* the problem and the method taxonomy; each is evaluated against FULL
  of the same set and facet (no human taxonomy exists). Results are reported
  per facet and averaged over the 8 sets x 2 facets (facet ``"avg"``).
* **TaxoBench.** Every method, FULL included, induces a single
  topic-and-approach taxonomy (facet ``"topic"``) over the pooled papers and
  is evaluated against the merged expert gold taxonomy (paper entries are
  extents, not nodes).
* **Metrics.** Soft node F1 and soft edge F1
  (:class:`~scholarpath.evaluation.taxonomy_metrics.TaxonomyScorer`, roots
  excluded) and the relative LLM cost C_LLM / C_Full, where C_LLM is the
  total billed cost of a build (Eq. 15) and C_Full the cost of FULL on the
  same dataset and facet. The node-wise bars of Fig. 3 are those of the
  strongest baseline, Coverage (:data:`FIG3_SELECTOR`), as in the paper.

Every build goes through :class:`ConstructionRunner`, which runs it against a
fresh cost meter and stores it in a :class:`~scholarpath.experiments.common.BuildCache`
keyed by (dataset, facet, method, seed, config hash): FULL is built once per
(dataset, facet) and reused by every comparison, by every script sharing the
cache directory, and (with ``--resume``) by later runs. FULL trains no router,
so its key ignores the router settings and the baselines' ``min_router_steps``.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from scholarpath.config import BaselineBuilderConfig, BuilderConfig, ScholarPathConfig
from scholarpath.data.corpus import Corpus
from scholarpath.evaluation.reporting import Column, TableSpec, format_value, markdown_table, table_rows, write_csv
from scholarpath.evaluation.taxonomy_metrics import TaxonomyScorer
from scholarpath.experiments.common import (
    BuildCache,
    BuildKey,
    aggregate_rows,
    corpus_fingerprint,
    embed_corpus,
    finite,
    fresh_meter,
    load_corpus,
    make_build_key,
    matplotlib_available,
    mean_of,
    plot_grouped_bars,
    plot_lines,
    restrict_taxonomy,
    runtime_fingerprint,
    scale,
    write_markdown,
    write_rows,
)
from scholarpath.llm.accounting import BUILD_COMPONENTS
from scholarpath.llm.base import LLMClient
from scholarpath.taxonomy.baselines import (
    COMPONENT_FULL,
    COMPONENT_SELECTION,
    FULL,
    NODE_WISE,
    ONE_OFF,
    TERM_NO_SPLIT,
    SelectionBuilder,
    internal_paper_count,
    method_name,
    parse_method,
    per_paper_node_cost,
)
from scholarpath.taxonomy.builder import TERM_NO_SCHEMA, NodeWiseBuilder
from scholarpath.taxonomy.dynamic import DynamicUpdater, UpdateReport
from scholarpath.taxonomy.response_model import TransferredPrior
from scholarpath.taxonomy.result import BuildResult, EpisodeRecord
from scholarpath.taxonomy.selectors import SELECTION_METHODS
from scholarpath.taxonomy.tree import CERTIFIED, Taxonomy
from scholarpath.types import Paper

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper settings
# ---------------------------------------------------------------------------

#: Selection ratios of Fig. 2 (Section 5.2).
FIG2_ETAS: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
#: Node-wise selectors of Fig. 2, in the paper's legend order.
FIG2_SELECTORS: Tuple[str, ...] = ("coverage", "kmeans", "diversity", "random")
#: Ratios of the node-wise bars of Fig. 3.
FIG3_ETAS: Tuple[float, ...] = (0.5, 0.4, 0.2)
#: Selector of the node-wise bars of Fig. 3: the strongest baseline, Coverage
#: (its eta = 0.2 bars, 20.4% / 24.6%, are the Coverage costs of Tables 8-9 and
#: Sec. 5.2.1 quotes them for Coverage). ``"mean"`` averages the selectors run.
FIG3_SELECTOR = "coverage"
FIG3_MEAN = "mean"
#: Ratio of the one-off vs. node-wise comparison (Tables 8-9, App. C.5).
TABLE89_ETA = 0.2
#: Row order of the selectors in Tables 8-9.
TABLE_SELECTOR_ORDER: Tuple[str, ...] = ("random", "diversity", "kmeans", "coverage")
#: Corpus sizes |P| of Fig. 4 (nested samples of arXiv-Subfields; 148.4K = the whole corpus).
SCALABILITY_SIZES: Tuple[int, ...] = (2000, 4000, 6000, 8000, 10000, 50000, 148400)
#: Fixed ratios of the node-wise baselines of Fig. 4.
SCALABILITY_ETAS: Tuple[float, ...] = (0.2, 0.5)

DUAL_FACETS: Tuple[str, ...] = ("problem", "method")
TOPIC_FACET = "topic"
#: Pseudo-facet of the rows averaged over facets (and datasets).
AVG_FACET = "avg"

ARXIV = "arxiv"
TAXOBENCH = "taxobench"
SUBFIELDS = "subfields"
BENCHMARK_LABELS = {ARXIV: "arXiv", TAXOBENCH: "TaxoBench", SUBFIELDS: "arXiv-Subfields"}

SCHOLARPATH = "scholarpath"
SELECTOR_LABELS = {"random": "Random", "coverage": "Coverage", "kmeans": "K-means", "diversity": "Diversity"}
SCHOLARPATH_LABEL = "ScholarPath"
FULL_LABEL = "FULL (η = 1.0)"

#: Root concept of every facet. All methods of a comparison use the same root
#: (the root is excluded from the soft F1 metrics). The problem / method roots
#: are those of :func:`scholarpath.taxonomy.builder.build_dual_taxonomies`.
FACET_ROOTS: Dict[str, Tuple[str, str]] = {
    "problem": ("Research Problems",
                "All research problems (tasks, domains, phenomena) studied by the papers of the repository."),
    "method": ("Research Methods",
               "All research methods (techniques, models, algorithms, tools) used by the papers of the repository."),
    "topic": ("Research Topics",
              "All research topics and approaches of the papers of the repository, as a survey would organize them."),
}

#: Soft-F1 keys reported for every run.
METRICS: Tuple[str, ...] = ("node_p", "node_r", "node_f1", "edge_p", "edge_r", "edge_f1")
#: Cost components reported per run (Eq. 15 plus the baselines' buckets).
COST_COMPONENTS: Tuple[str, ...] = tuple(BUILD_COMPONENTS) + (COMPONENT_FULL, COMPONENT_SELECTION)
#: Columns averaged by :func:`summarize_runs`.
SUMMARY_METRICS: Tuple[str, ...] = METRICS + (
    "relative_cost", "cost_usd", "full_cost_usd", "tokens", "calls", "papers_read", "runtime_s",
    "abstention", "disagreement", "n_nodes", "depth", "certified_nodes", "mean_ratio", "routed_episodes",
)


#: A paper table: ``(rows by system, sections, spec)`` for :func:`render_table`.
Table = Tuple[Mapping[str, Mapping[str, Any]], Sequence[Tuple[str, Sequence[str]]], TableSpec]


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ablation:
    """A ScholarPath variant of Table 6 (App. C.4): builder-config overrides."""

    name: str
    label: str
    overrides: Tuple[Tuple[str, Any], ...]

    def apply(self, config: BuilderConfig) -> BuilderConfig:
        return dataclasses.replace(config, **dict(self.overrides))

    def display(self, config: Optional[BuilderConfig] = None) -> str:
        ratio = (config or BuilderConfig()).fixed_ratio
        return self.label.format(fixed_ratio=f"{ratio:g}")


#: The six variants of Table 6, in the paper's order.
ABLATIONS: "OrderedDict[str, Ablation]" = OrderedDict(
    (a.name, a)
    for a in (
        # Stage 1 batches drawn uniformly at random instead of Eq. (3) / (10).
        Ablation("no_bayesian_acquisition", "w/o Bayesian acquisition", (("use_bayesian_acquisition", False),)),
        # Stage 3 replaced by a fixed reading ratio eta_v (BuilderConfig.fixed_ratio, 0.08).
        Ablation("no_snapshot_certification", "w/o snapshot certification (η_v = {fixed_ratio})",
                 (("use_snapshot_certification", False),)),
        # Fixed prior N(prior_slope_mean, prior_slope_var I) instead of Eq. (14).
        Ablation("no_transferred_prior", "w/o transferred prior", (("use_transferred_prior", False),)),
        Ablation("no_discovery", "w/o discovery (DISCOVER)", (("use_discovery", False),)),
        Ablation("no_correction", "w/o correction (CORRECT)", (("use_correction", False),)),
        # No Stage 2 at all.
        Ablation("no_targeted_repair", "w/o targeted repair", (("use_targeted_repair", False),)),
    )
)


@dataclass(frozen=True)
class MethodSpec:
    """One construction method of the comparison.

    ``kind`` is ``"scholarpath"`` (Algorithm 1, optionally an :data:`ABLATIONS`
    variant), ``"full"`` (eta = 1) or ``"selection"`` (a selector at ratio
    ``eta``, node-wise or one-off). Use :meth:`parse` / :attr:`key` for the
    canonical string form: ``"scholarpath"``, ``"scholarpath/no_discovery"``,
    ``"full"``, ``"coverage@0.2"``, ``"one_off/kmeans@0.2"``.
    """

    kind: str
    selector: str = ""
    eta: float = 1.0
    mode: str = NODE_WISE
    ablation: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in (SCHOLARPATH, FULL, "selection"):
            raise ValueError(f"unknown method kind {self.kind!r}")
        if self.kind == "selection":
            if self.selector not in SELECTION_METHODS:
                raise ValueError(f"unknown selector {self.selector!r}; expected one of {SELECTION_METHODS}")
            if not 0.0 < float(self.eta) <= 1.0:
                raise ValueError(f"selection ratio must lie in (0, 1], got {self.eta}")
            if self.mode not in (NODE_WISE, ONE_OFF):
                raise ValueError(f"unknown selection mode {self.mode!r}")
        if self.ablation is not None and (self.kind != SCHOLARPATH or self.ablation not in ABLATIONS):
            raise ValueError(f"unknown ScholarPath ablation {self.ablation!r}; expected one of {list(ABLATIONS)}")

    # ------------------------------------------------------- constructors
    @classmethod
    def scholarpath(cls, ablation: Optional[str] = None) -> "MethodSpec":
        return cls(SCHOLARPATH, ablation=ablation)

    @classmethod
    def full(cls) -> "MethodSpec":
        return cls(FULL)

    @classmethod
    def selection(cls, selector: str, eta: float, mode: str = NODE_WISE) -> "MethodSpec":
        return cls("selection", selector=selector, eta=float(eta), mode=mode)

    @classmethod
    def parse(cls, name: str) -> "MethodSpec":
        text = (name or "").strip().lower()
        if text == SCHOLARPATH:
            return cls.scholarpath()
        if text.startswith(SCHOLARPATH + "/"):
            return cls.scholarpath(text.split("/", 1)[1])
        selector, eta, mode = parse_method(text)
        return cls.full() if selector == FULL else cls.selection(selector, eta, mode)

    # --------------------------------------------------------- properties
    @property
    def key(self) -> str:
        if self.kind == SCHOLARPATH:
            return SCHOLARPATH if self.ablation is None else f"{SCHOLARPATH}/{self.ablation}"
        if self.kind == FULL:
            return FULL
        return method_name(self.selector, self.eta, self.mode)

    @property
    def is_full(self) -> bool:
        return self.kind == FULL

    @property
    def is_scholarpath(self) -> bool:
        return self.kind == SCHOLARPATH

    @property
    def family(self) -> str:
        """``scholarpath`` | ``ablation`` | ``full`` | ``node_wise`` | ``one_off``."""
        if self.kind == SCHOLARPATH:
            return SCHOLARPATH if self.ablation is None else "ablation"
        if self.kind == FULL:
            return FULL
        return self.mode

    def label(self, config: Optional[BuilderConfig] = None) -> str:
        """Display name as in the paper's figures and tables."""
        if self.kind == SCHOLARPATH:
            return SCHOLARPATH_LABEL if self.ablation is None else ABLATIONS[self.ablation].display(config)
        if self.kind == FULL:
            return FULL_LABEL
        return SELECTOR_LABELS.get(self.selector, self.selector)


SCHOLARPATH_METHOD = MethodSpec.scholarpath()
FULL_METHOD = MethodSpec.full()


def construction_methods(
    selectors: Sequence[str] = FIG2_SELECTORS,
    etas: Sequence[float] = FIG2_ETAS,
    one_off_etas: Sequence[float] = (TABLE89_ETA,),
    include_scholarpath: bool = True,
    *,
    fig3_selector: Optional[str] = None,
    fig3_etas: Sequence[float] = (),
) -> List[MethodSpec]:
    """Methods of Figs. 2-3 and Tables 8-9 (FULL is always added by the runner).

    ``fig3_selector`` / ``fig3_etas`` add the node-wise builds of the Fig. 3
    bars (Coverage at eta = 0.5, 0.4, 0.2 in the paper) when the selector is
    not among ``selectors`` (``None`` or :data:`FIG3_MEAN` adds nothing).
    """
    out: List[MethodSpec] = [SCHOLARPATH_METHOD] if include_scholarpath else []
    out += [MethodSpec.selection(s, e, NODE_WISE) for s in selectors for e in etas]
    out += [MethodSpec.selection(s, e, ONE_OFF) for s in selectors for e in one_off_etas]
    if fig3_selector and fig3_selector != FIG3_MEAN:
        out += [MethodSpec.selection(fig3_selector, e, NODE_WISE) for e in fig3_etas]
    return list(OrderedDict((m.key, m) for m in out).values())


def ablation_methods(variants: Optional[Sequence[str]] = None) -> List[MethodSpec]:
    """ScholarPath and its Table 6 variants (all of :data:`ABLATIONS` by default)."""
    names = list(variants) if variants is not None else list(ABLATIONS)
    return [SCHOLARPATH_METHOD] + [MethodSpec.scholarpath(v) for v in names]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass
class ConstructionDataset:
    """A corpus to organize, its facets and (optionally) expert references per facet."""

    name: str
    benchmark: str
    corpus: Corpus
    facets: Tuple[str, ...] = DUAL_FACETS
    gold: Dict[str, Taxonomy] = field(default_factory=dict)
    vectors: Optional[Dict[str, np.ndarray]] = None
    _fingerprint: Optional[str] = field(default=None, repr=False)

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = corpus_fingerprint(self.corpus)
        return self._fingerprint

    def reference_kind(self, facet: str) -> str:
        """``"gold"`` if an expert taxonomy exists for the facet, else ``"full"`` (arXiv sets)."""
        return "gold" if facet in self.gold else FULL


def arxiv_set_paths(data_dir: Union[str, Path], categories: Optional[Sequence[str]] = None) -> List[Tuple[str, Path]]:
    """``(category, path)`` of the processed arXiv sets under ``<data_dir>/arxiv_sets``.

    Without ``categories`` every set present is returned, the paper's eight
    categories first (Sec. 5.2) and any other file (e.g. the synthetic set)
    after them.
    """
    from scholarpath.data.arxiv import ARXIV_SET_CATEGORIES, arxiv_set_filename

    directory = Path(data_dir) / "arxiv_sets"
    if categories:
        out = [(c, directory / arxiv_set_filename(c)) for c in categories]
        missing = [str(p) for _, p in out if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing arXiv sets {missing}; run scripts/prepare_arxiv.py (data/README.md)")
        return out
    present = {p.name[: -len(".jsonl.gz")]: p for p in sorted(directory.glob("*.jsonl.gz"))}
    if not present:
        raise FileNotFoundError(f"no arXiv sets under {directory}; run scripts/prepare_arxiv.py (data/README.md)")
    known = [c for c in ARXIV_SET_CATEGORIES if c in present]
    ordered = known + sorted(c for c in present if c not in ARXIV_SET_CATEGORIES)
    return [(c, present[c]) for c in ordered]


def load_arxiv_datasets(
    data_dir: Union[str, Path],
    categories: Optional[Sequence[str]] = None,
    limit_papers: Optional[int] = None,
    sample_seed: int = 0,
    facets: Sequence[str] = DUAL_FACETS,
) -> List[ConstructionDataset]:
    """The arXiv-sets benchmark: one dataset per set, both facets, reference = FULL."""
    return [
        ConstructionDataset(name=cat, benchmark=ARXIV,
                            corpus=load_corpus(path, limit_papers, sample_seed, name=f"arxiv_set_{cat}"),
                            facets=tuple(facets))
        for cat, path in arxiv_set_paths(data_dir, categories)
    ]


def truncate_taxonomy(tax: Taxonomy, max_depth: Optional[int]) -> Taxonomy:
    """Copy of ``tax`` without the nodes deeper than ``max_depth`` (extents already include descendants)."""
    out = Taxonomy.from_dict(tax.to_dict())
    if max_depth is None:
        return out
    for nid in [n for n, node in out.nodes.items() if node.depth == int(max_depth)]:
        for c in out.children(nid):
            out.remove_subtree(c)
    out.metadata["truncated_depth"] = int(max_depth)
    out.invalidate()
    return out


#: Levels of the merged TaxoBench gold below the common root (App. C.2: "814 concept nodes across 4 levels").
TAXOBENCH_GOLD_MAX_DEPTH = 4


def parse_max_depth(text: Union[str, int, None]) -> Optional[int]:
    """``--gold-max-depth`` values: a positive level count, or ``none`` / ``0`` for the untruncated gold."""
    if text is None:
        return None
    t = str(text).strip().lower()
    if t in ("", "none", "off", "full", "0"):
        return None
    value = int(t)
    if value < 0:
        raise ValueError(f"max depth must be >= 0, got {text!r}")
    return value


def load_taxobench_dataset(
    data_dir: Union[str, Path],
    limit_papers: Optional[int] = None,
    sample_seed: int = 0,
    gold_max_depth: Optional[int] = None,
) -> ConstructionDataset:
    """TaxoBench: the pooled papers and the merged expert gold taxonomy (facet ``"topic"``).

    ``gold_max_depth`` cuts the merged gold below that many levels under the
    common root (App. C.2 reports 4 levels, :data:`TAXOBENCH_GOLD_MAX_DEPTH`,
    the default of ``run_taxonomy_experiments.py``; the public release has up
    to 7, see data/README.md; ``None`` keeps every level). With ``limit_papers`` (quick runs only) the gold is
    restricted to the sampled papers and concepts left empty are dropped.
    """
    directory = Path(data_dir) / TAXOBENCH
    papers, gold_path = directory / "papers.jsonl.gz", directory / "gold_taxonomy.json"
    if not papers.exists() or not gold_path.exists():
        raise FileNotFoundError(f"TaxoBench is not prepared under {directory}; run scripts/download_data.py "
                                "--only taxobench && scripts/prepare_benchmarks.py --only taxobench")
    corpus = load_corpus(papers, limit_papers, sample_seed, name=TAXOBENCH)
    gold = truncate_taxonomy(Taxonomy.load(gold_path), gold_max_depth)
    if limit_papers is not None:
        gold = restrict_taxonomy(gold, corpus.ids)
    gold.facet = TOPIC_FACET
    return ConstructionDataset(name=TAXOBENCH, benchmark=TAXOBENCH, corpus=corpus, facets=(TOPIC_FACET,),
                               gold={TOPIC_FACET: gold})


# ---------------------------------------------------------------------------
# Running builds
# ---------------------------------------------------------------------------


class ConstructionRunner:
    """Builds taxonomies with any :class:`MethodSpec` on any dataset, cached and individually costed.

    Args:
        llm, embedder, config: the runtime (``Runtime.llm``, ``Runtime.embedder``,
            ``Runtime.config``); ``config.builder`` configures ScholarPath and
            ``config.baseline`` the selection baselines (selector, ratio and mode
            are set per method).
        cache: build cache (default: in-memory only).
        min_router_steps: forwarded to :class:`SelectionBuilder` (its default,
            500, when ``None``).
        roots: root concept ``(name, definition)`` per facet.

    Seeds: a build with seed ``s`` sets ``builder.seed`` / ``baseline.seed`` to
    ``s`` and offsets the router seed by ``s`` (seed 0 leaves the router
    configuration unchanged). Without an explicit seed, ScholarPath uses
    ``config.builder.seed`` and the selection baselines ``config.baseline.seed``
    (:meth:`resolve_seed`). FULL is always built with the reference seed
    (``config.baseline.seed``) so that every comparison uses one FULL per
    (dataset, facet).
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: Any,
        config: ScholarPathConfig,
        cache: Optional[BuildCache] = None,
        *,
        min_router_steps: Optional[int] = None,
        roots: Optional[Mapping[str, Tuple[str, str]]] = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.config = config
        self.cache = cache if cache is not None else BuildCache(None)
        self.baseline_kwargs: Dict[str, Any] = {}
        if min_router_steps is not None:
            self.baseline_kwargs["min_router_steps"] = int(min_router_steps)
        self.roots: Dict[str, Tuple[str, str]] = dict(FACET_ROOTS)
        self.roots.update(dict(roots or {}))

    @classmethod
    def from_runtime(cls, runtime: Any, cache: Optional[BuildCache] = None, **kwargs: Any) -> "ConstructionRunner":
        return cls(runtime.llm, runtime.embedder, runtime.config, cache, **kwargs)

    # ------------------------------------------------------------ config
    @property
    def reference_seed(self) -> int:
        """Seed of FULL (``config.baseline.seed``), whatever seed a comparison runs with."""
        return int(self.config.baseline.seed)

    def root(self, facet: str) -> Tuple[str, str]:
        return self.roots.get(facet, ("Root", f"All {facet}s of the papers of the repository."))

    def builder_config(self, facet: str, seed: int, ablation: Optional[str] = None) -> BuilderConfig:
        """Effective ScholarPath configuration (Algorithm 1) of one build."""
        base = self.config.builder
        cfg = dataclasses.replace(base, facet=facet, seed=int(seed),
                                  router=dataclasses.replace(base.router, seed=int(base.router.seed) + int(seed)))
        return ABLATIONS[ablation].apply(cfg) if ablation else cfg

    def baseline_config(self, facet: str, seed: int, method: MethodSpec) -> BaselineBuilderConfig:
        """Effective selection-baseline configuration of one build."""
        base = self.config.baseline
        if method.is_full:
            selector, eta, mode = FULL, 1.0, NODE_WISE
        else:
            selector, eta, mode = method.selector, float(method.eta), method.mode
        return dataclasses.replace(base, selector=selector, eta=eta, mode=mode, facet=facet, seed=int(seed),
                                   router=dataclasses.replace(base.router, seed=int(base.router.seed) + int(seed)))

    def resolve_seed(self, method: MethodSpec, seed: Optional[int]) -> int:
        """Effective seed of a build: the reference seed for FULL, else ``seed`` or the method's config seed.

        The default comes from the configuration section the method reads:
        ``builder.seed`` for ScholarPath, ``baseline.seed`` for the selection
        baselines (so ``--set baseline.seed=...`` reaches every baseline).
        """
        if method.is_full:
            return self.reference_seed
        if seed is not None:
            return int(seed)
        return int(self.config.builder.seed if method.is_scholarpath else self.config.baseline.seed)

    def vectors(self, dataset: ConstructionDataset) -> Dict[str, np.ndarray]:
        """Paper embeddings of the dataset (computed once, shared by all of its builds)."""
        if dataset.vectors is None:
            dataset.vectors = embed_corpus(self.embedder, dataset.corpus)
        return dataset.vectors

    def build_key(self, dataset: ConstructionDataset, facet: str, method: MethodSpec,
                  seed: Optional[int] = None) -> Tuple[BuildKey, Dict[str, Any]]:
        """Cache key of a build and the payload it hashes.

        The payload holds exactly the inputs that determine the build. FULL
        reads every paper with the LLM and trains no router
        (``SelectionBuilder(fit_full_routers=False)``), so ``baseline.router``
        and the ``SelectionBuilder`` kwargs (``min_router_steps``) are left out
        of its key: runs and scripts that differ only in router settings share
        one FULL per (dataset, facet).
        """
        seed = self.resolve_seed(method, seed)
        extra: Dict[str, Any] = {}
        if method.is_scholarpath:
            builder = dataclasses.asdict(self.builder_config(facet, seed, method.ablation))
        else:
            builder = dataclasses.asdict(self.baseline_config(facet, seed, method))
            if method.is_full:
                builder.pop("router")
            else:
                extra = dict(self.baseline_kwargs)
        payload = {
            "corpus": dataset.fingerprint,
            "n_papers": len(dataset.corpus),
            "root": list(self.root(facet)),
            "builder": builder,
            "builder_kwargs": extra,
            "runtime": runtime_fingerprint(self.config),
        }
        return make_build_key(dataset.name, facet, method.key, seed, payload), payload

    # ------------------------------------------------------------- build
    def build(self, dataset: ConstructionDataset, facet: str, method: MethodSpec,
              seed: Optional[int] = None) -> BuildResult:
        """The (cached) build of ``method`` on ``dataset`` for ``facet``. Treat the result as read-only."""
        seed = self.resolve_seed(method, seed)
        key, payload = self.build_key(dataset, facet, method, seed)
        router = (self.builder_config(facet, seed, method.ablation) if method.is_scholarpath
                  else self.baseline_config(facet, seed, method)).router
        return self.cache.get_or_build(key, lambda: self._run(dataset, facet, method, seed), payload, router)

    def _run(self, dataset: ConstructionDataset, facet: str, method: MethodSpec, seed: int) -> BuildResult:
        root_name, root_definition = self.root(facet)
        vectors = self.vectors(dataset)
        with fresh_meter(self.llm):
            if method.is_scholarpath:
                cfg = self.builder_config(facet, seed, method.ablation)
                builder = NodeWiseBuilder(self.llm, self.embedder, cfg, dataset.corpus, paper_vectors=vectors)
                result = builder.build(root_name, root_definition, facet=facet)
            else:
                bcfg = self.baseline_config(facet, seed, method)
                sel = SelectionBuilder(self.llm, self.embedder, bcfg, dataset.corpus, vectors, **self.baseline_kwargs)
                result = sel.build(root_name, root_definition, facet=facet)
        result.method = method.key
        result.metadata["experiment"] = {"dataset": dataset.name, "benchmark": dataset.benchmark, "facet": facet,
                                         "method": method.key, "seed": seed, "n_papers": len(dataset.corpus)}
        log.info("[%s/%s] %s (seed %d): %d nodes, $%.4f, %d calls, %.1fs", dataset.name, facet, method.key, seed,
                 len(result.taxonomy.nodes) - 1, result.cost_usd, result.total_usage.calls, result.runtime_s)
        return result


# ---------------------------------------------------------------------------
# Scoring and per-run records
# ---------------------------------------------------------------------------


def make_scorer(embedder: Any, text: str = "name") -> TaxonomyScorer:
    """Soft node / edge F1 scorer (roots excluded, root edges included, ``min`` edge combiner)."""
    return TaxonomyScorer(embedder, text=text)


def score_taxonomy(pred: Taxonomy, reference: Taxonomy, scorer: TaxonomyScorer) -> Dict[str, float]:
    """Soft node / edge precision, recall and F1 of ``pred`` against ``reference`` (Sec. 5.2)."""
    s = scorer.score(pred, reference)
    out = {k: float(s[k]) for k in METRICS}
    out["n_ref_nodes"] = int(s["n_gold_nodes"])
    return out


#: Episode terminations of nodes that became leaves without a router: their seed yielded fewer than two
#: usable children (ScholarPath ``no_schema``, baselines ``no_split``), so a_v and d_v are undefined there.
UNROUTED_TERMINATIONS = frozenset({TERM_NO_SCHEMA, TERM_NO_SPLIT})


def routed_episodes(result: BuildResult) -> List[EpisodeRecord]:
    """Episodes of a build that trained a router f_theta_v (every termination but :data:`UNROUTED_TERMINATIONS`)."""
    return [e for e in result.episodes if (e.metadata or {}).get("termination") not in UNROUTED_TERMINATIONS]


def router_rates(result: BuildResult) -> Dict[str, Any]:
    """Table 6's router abstention a_v and validation disagreement d_v of one build.

    App. C.4 reports the *router* abstention rate a_v = |⊥_v| / n_v and the
    validation disagreement rate d_v (Sec. 3.2); both are only defined at
    nodes that trained a router. Leaves created because the seed batch gave
    fewer than two usable children carry the defaults 0.0, and averaging
    them in would deflate the rates by an amount that depends on how many
    such leaves each variant produces. The means are therefore taken over
    :func:`routed_episodes` (``None`` when no node trained a router).
    """
    eps = routed_episodes(result)
    return {
        "abstention": float(np.mean([e.abstention for e in eps])) if eps else None,
        "disagreement": float(np.mean([e.disagreement for e in eps])) if eps else None,
        "routed_episodes": len(eps),
    }


def run_record(
    dataset: ConstructionDataset,
    facet: str,
    method: MethodSpec,
    seed: int,
    result: BuildResult,
    scores: Optional[Mapping[str, float]] = None,
    full_cost: Optional[float] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """One flat result row: identity, soft F1, cost (absolute and relative to FULL) and build statistics."""
    usage = result.total_usage
    tax = result.taxonomy
    summary = result.summary()
    rates = router_rates(result)
    breakdown = result.cost_breakdown()
    rel = None
    if full_cost is not None and full_cost > 0:
        rel = result.cost_usd / float(full_cost)
    row: Dict[str, Any] = {
        "benchmark": dataset.benchmark,
        "dataset": dataset.name,
        "facet": facet,
        "method": method.key,
        "label": label or method.label(result_builder_config(result)),
        "family": method.family,
        "selector": method.selector if method.kind == "selection" else (FULL if method.is_full else ""),
        "eta": 1.0 if method.is_full else (float(method.eta) if method.kind == "selection" else None),
        "mode": method.mode if method.kind == "selection" else (FULL if method.is_full else ""),
        "ablation": method.ablation or "",
        "seed": int(seed),
        "reference": dataset.reference_kind(facet),
        "n_papers": len(tax.root.papers),
    }
    for k in METRICS:
        row[k] = None if scores is None else scores.get(k)
    row.update({
        "n_nodes": len(tax.nodes) - 1,
        "n_ref_nodes": None if scores is None else scores.get("n_ref_nodes"),
        "depth": tax.max_depth,
        "cost_usd": result.cost_usd,
        "full_cost_usd": full_cost,
        "relative_cost": rel,
        "tokens": usage.total_tokens,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "calls": usage.calls,
        "papers_read": int(sum(result.papers_read.values())),
        "runtime_s": result.runtime_s,
        "episodes": len(result.episodes),
        "certified_nodes": summary["certified_nodes"],
        "routed_episodes": rates["routed_episodes"],
        # a_v / d_v over the nodes that trained a router (router_rates); the realized ratio eta_v of every
        # episode, including unsplit leaves (their seed batch was read).
        "abstention": rates["abstention"],
        "disagreement": rates["disagreement"],
        "mean_ratio": summary["mean_ratio"],
    })
    for comp in COST_COMPONENTS:
        row[f"cost_{comp}"] = breakdown.get(comp, 0.0)
    return row


def result_builder_config(result: BuildResult) -> Optional[BuilderConfig]:
    """The ScholarPath builder configuration stored in a build's metadata (``None`` for baselines)."""
    raw = result.metadata.get("builder_config")
    if not isinstance(raw, Mapping):
        return None
    names = {f.name for f in dataclasses.fields(BuilderConfig)}
    try:
        return dataclasses.replace(BuilderConfig(), **{k: v for k, v in raw.items() if k in names and k != "router"})
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Figs. 2-3, Tables 6, 8, 9: comparison against FULL / gold
# ---------------------------------------------------------------------------


def run_construction_comparison(
    datasets: Sequence[ConstructionDataset],
    methods: Sequence[MethodSpec],
    runner: ConstructionRunner,
    seeds: Optional[Sequence[int]] = None,
    scorer: Optional[TaxonomyScorer] = None,
) -> List[Dict[str, Any]]:
    """Build every method on every dataset / facet and evaluate it (Section 5.2).

    For each (dataset, facet), FULL is built first (once, with the reference
    seed); C_Full is its billed cost. The reference taxonomy is the expert
    gold of the dataset when it has one (TaxoBench) and FULL otherwise
    (arXiv sets). Every other method is then built for every seed and scored
    against that reference (without ``seeds``, each method uses its
    configured seed, :meth:`ConstructionRunner.resolve_seed`). FULL's own row
    is included (relative cost 1.0; on the arXiv sets its soft F1 against
    itself is 1.0 by construction).
    """
    scorer = scorer or make_scorer(runner.embedder)
    run_seeds: List[Optional[int]] = list(seeds) if seeds else [None]
    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        for facet in ds.facets:
            full = runner.build(ds, facet, FULL_METHOD)
            reference = ds.gold.get(facet, full.taxonomy)
            c_full = full.cost_usd
            rows.append(run_record(ds, facet, FULL_METHOD, runner.reference_seed, full,
                                   score_taxonomy(full.taxonomy, reference, scorer), c_full))
            for method in methods:
                if method.is_full:
                    continue
                for seed in run_seeds:
                    s = runner.resolve_seed(method, seed)
                    result = runner.build(ds, facet, method, s)
                    rows.append(run_record(ds, facet, method, s, result,
                                           score_taxonomy(result.taxonomy, reference, scorer), c_full))
    return rows


def summarize_runs(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str] = SUMMARY_METRICS) -> List[Dict[str, Any]]:
    """Mean of every metric per (benchmark, facet, method) over datasets and seeds.

    Adds facet :data:`AVG_FACET` rows that pool all facets (on the arXiv sets:
    the average over 8 sets x 2 facets reported in Section 5.2). Relative
    costs are averaged per run (mean of C_LLM / C_Full ratios).
    """
    by = ("benchmark", "facet", "method", "label", "family", "selector", "eta", "mode", "ablation", "reference")
    per_facet = aggregate_rows(rows, by, metrics)
    pooled = aggregate_rows(rows, by, metrics, overrides={"facet": AVG_FACET})
    return per_facet + pooled


def _index(summary: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, str], Mapping[str, Any]]:
    return {(r["benchmark"], r["facet"], r["method"]): r for r in summary}


def _benchmarks(summary: Sequence[Mapping[str, Any]]) -> List[str]:
    return list(OrderedDict((r["benchmark"], None) for r in summary))


def figure2_rows(
    summary: Sequence[Mapping[str, Any]],
    selectors: Sequence[str] = FIG2_SELECTORS,
    etas: Sequence[float] = FIG2_ETAS,
) -> List[Dict[str, Any]]:
    """Fig. 2 data: soft node / edge F1 (%) vs. eta per node-wise selector, plus reference lines.

    One row per (benchmark, facet, series, eta); ScholarPath (and, on
    benchmarks evaluated against an expert gold, FULL) are reference rows
    with ``eta = None``.
    """
    idx = _index(summary)
    out: List[Dict[str, Any]] = []
    facets_of: Dict[str, List[str]] = OrderedDict()
    for r in summary:
        facets_of.setdefault(r["benchmark"], [])
        if r["facet"] not in facets_of[r["benchmark"]]:
            facets_of[r["benchmark"]].append(r["facet"])
    for bench, facets in facets_of.items():
        for facet in facets:
            for ref in (SCHOLARPATH_METHOD, FULL_METHOD):
                r = idx.get((bench, facet, ref.key))
                if r is None or (ref.is_full and r.get("reference") != "gold"):
                    continue
                out.append({"benchmark": bench, "facet": facet, "series": ref.label(), "selector": ref.key,
                            "eta": None, "reference_line": True, "node_f1": scale(r.get("node_f1")),
                            "edge_f1": scale(r.get("edge_f1")), "relative_cost": r.get("relative_cost"),
                            "n_runs": r.get("n_runs")})
            for sel in selectors:
                for eta in etas:
                    r = idx.get((bench, facet, method_name(sel, eta, NODE_WISE)))
                    if r is None:
                        continue
                    out.append({"benchmark": bench, "facet": facet, "series": SELECTOR_LABELS.get(sel, sel),
                                "selector": sel, "eta": float(eta), "reference_line": False,
                                "node_f1": scale(r.get("node_f1")), "edge_f1": scale(r.get("edge_f1")),
                                "relative_cost": r.get("relative_cost"), "n_runs": r.get("n_runs")})
    return out


#: Fixed colour order of the systems (an entity keeps its colour in every figure).
SERIES_ORDER: Tuple[str, ...] = (SCHOLARPATH_LABEL, "Coverage", "K-means", "Diversity", "Random", FULL_LABEL)


def plot_figure2(fig2: Sequence[Mapping[str, Any]], out_dir: Union[str, Path], facet: str = AVG_FACET) -> List[Path]:
    """Fig. 2 panels: soft node and soft edge F1 vs. eta, one PNG per benchmark and metric."""
    paths: List[Path] = []
    letters = iter("abcdefghijklmnop")
    for bench in OrderedDict((r["benchmark"], None) for r in fig2):
        rows = [r for r in fig2 if r["benchmark"] == bench and r["facet"] == facet]
        if not rows:
            continue
        etas = sorted({float(r["eta"]) for r in rows if r["eta"] is not None})
        vs = "Gold" if any(r["series"] == FULL_LABEL for r in rows) else "FULL"
        for metric, name in (("node_f1", "Soft Node F$_1$ (%)"), ("edge_f1", "Soft Edge F$_1$ (%)")):
            series: Dict[str, List[Tuple[float, Optional[float]]]] = OrderedDict()
            refs: Dict[str, Optional[float]] = OrderedDict()
            for r in rows:
                if r["reference_line"]:
                    refs[r["series"]] = r[metric]
                else:
                    series.setdefault(r["series"], []).append((float(r["eta"]), r[metric]))
            title = f"({next(letters)}) {BENCHMARK_LABELS.get(bench, bench)} {name.split(' (')[0]} (vs. {vs})"
            paths.append(plot_lines(series, Path(out_dir) / f"fig2_{bench}_{metric}.png", xlabel="Selection ratio η",
                                    ylabel=name, title=title, references=refs, order=SERIES_ORDER,
                                    xticks=etas, xticklabels=[f"{e:g}" for e in etas]))
    return paths


def figure3_rows(
    summary: Sequence[Mapping[str, Any]],
    etas: Sequence[float] = FIG3_ETAS,
    selectors: Sequence[str] = FIG2_SELECTORS,
    bar_selector: str = FIG3_SELECTOR,
) -> List[Dict[str, Any]]:
    """Fig. 3 data: relative LLM construction cost C_LLM / C_Full (%) per benchmark.

    Bars: FULL (100%), node-wise selection at each eta, ScholarPath. The eta
    bars are the cost of ``bar_selector``, by default Coverage: the paper's
    eta = 0.2 bars (20.4% on arXiv, 24.6% on TaxoBench) are the node-wise
    Coverage costs of Tables 8-9, and Section 5.2.1 quotes the eta = 0.2 / 0.4 /
    0.5 costs for Coverage. ``bar_selector = "mean"`` averages the selectors
    instead. The cost of every selector that was run is kept in its own
    ``<selector>_pct`` column; ``bar_selector`` records the selector the bar
    shows (``"mean"`` if the requested one was not run, with a warning).
    """
    idx = _index(summary)
    names = list(OrderedDict((s, None) for s in list(selectors) + (
        [bar_selector] if bar_selector != FIG3_MEAN else [])))
    out: List[Dict[str, Any]] = []
    for bench in _benchmarks(summary):
        full = idx.get((bench, AVG_FACET, FULL))
        if full is not None:
            out.append({"benchmark": bench, "bar": "Full", "eta": 1.0,
                        "relative_cost_pct": scale(full.get("relative_cost")), "bar_selector": None,
                        "n_selectors": None})
        for eta in etas:
            per = {sel: idx.get((bench, AVG_FACET, method_name(sel, eta, NODE_WISE))) for sel in names}
            per = {s: r for s, r in per.items() if r is not None}
            if not per:
                continue
            if bar_selector in per:
                shown, value = bar_selector, per[bar_selector].get("relative_cost")
            else:
                if bar_selector != FIG3_MEAN:
                    log.warning("Fig. 3 (%s, η = %g): %s was not run; the bar shows the mean over %s",
                                bench, eta, bar_selector, ", ".join(per))
                shown, value = FIG3_MEAN, mean_of(r.get("relative_cost") for r in per.values())
            row: Dict[str, Any] = {"benchmark": bench, "bar": f"η = {eta:g}", "eta": float(eta),
                                   "relative_cost_pct": scale(value), "bar_selector": shown,
                                   "n_selectors": len(per)}
            for sel, r in per.items():
                row[f"{sel}_pct"] = scale(r.get("relative_cost"))
            out.append(row)
        sp = idx.get((bench, AVG_FACET, SCHOLARPATH))
        if sp is not None:
            out.append({"benchmark": bench, "bar": SCHOLARPATH_LABEL, "eta": None,
                        "relative_cost_pct": scale(sp.get("relative_cost")), "bar_selector": None,
                        "n_selectors": None})
    return out


def plot_figure3(fig3: Sequence[Mapping[str, Any]], path: Union[str, Path]) -> Optional[Path]:
    """Fig. 3: grouped bars of relative cost (%), one group per bar label, one colour per benchmark."""
    if not fig3:
        return None
    bars = list(OrderedDict((r["bar"], None) for r in fig3))
    benches = list(OrderedDict((r["benchmark"], None) for r in fig3))
    val = {(r["benchmark"], r["bar"]): r["relative_cost_pct"] for r in fig3}
    groups = OrderedDict((BENCHMARK_LABELS.get(b, b), [val.get((b, bar)) for bar in bars]) for b in benches)
    return plot_grouped_bars(bars, groups, path, ylabel="Relative Cost (%)",
                             order=[BENCHMARK_LABELS[ARXIV], BENCHMARK_LABELS[TAXOBENCH]])


def _pct(r: Optional[Mapping[str, Any]], key: str) -> Optional[float]:
    return None if r is None else scale(r.get(key))


def table8_spec(eta: float = TABLE89_ETA) -> TableSpec:
    return TableSpec(
        "table8", f"Table 8: One-off vs. node-wise selection on arXiv (η = {eta:g}, vs. FULL).",
        [Column("problem.node", "Node", "Problem Taxo."), Column("problem.edge", "Edge", "Problem Taxo."),
         Column("method.node", "Node", "Method Taxo."), Column("method.edge", "Edge", "Method Taxo."),
         Column("avg.node", "Node", "Average"), Column("avg.edge", "Edge", "Average"),
         Column("cost", "Cost", higher_is_better=False, digits=3)],
        first_header="Method",
    )


def table9_spec(eta: float = TABLE89_ETA) -> TableSpec:
    return TableSpec(
        "table9", f"Table 9: One-off vs. node-wise selection on TaxoBench (η = {eta:g}, vs. Gold).",
        [Column("node", "Node F1"), Column("edge", "Edge F1"),
         Column("cost", "Cost", higher_is_better=False, digits=3)],
        first_header="Method",
    )


def _selection_sections(
    bench: str, eta: float, cells: Any,
) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, List[str]]]]:
    """Rows and sections of Tables 8-9: one-off, then node-wise selection at ``eta``, then ScholarPath.

    ``cells(benchmark, method_key)`` returns the row cells of a method, or
    ``None`` if it was not run.
    """
    rows: Dict[str, Dict[str, Any]] = OrderedDict()
    sections: List[Tuple[str, List[str]]] = []
    modes = ((ONE_OFF, "One-off selection", "one-off"), (NODE_WISE, "Node-wise selection", "node-wise"))
    for mode, title, tag in modes:
        names = []
        for sel in TABLE_SELECTOR_ORDER:
            cell = cells(bench, method_name(sel, eta, mode))
            if cell is None:
                continue
            name = f"{SELECTOR_LABELS[sel]} ({tag})"
            rows[name] = cell
            names.append(name)
        if names:
            sections.append((f"{title} (η = {eta:g})", names))
    sp = cells(bench, SCHOLARPATH)
    if sp is not None:
        rows[SCHOLARPATH_LABEL] = sp
        sections.append(("", [SCHOLARPATH_LABEL]))
    return rows, sections


def table8(summary: Sequence[Mapping[str, Any]], eta: float = TABLE89_ETA,
           benchmark: str = ARXIV) -> Table:
    """Table 8 cells (App. C.5): per facet and averaged soft F1 (%) vs. FULL, relative cost."""
    idx = _index(summary)

    def cells(bench: str, method: str) -> Optional[Dict[str, Any]]:
        avg = idx.get((bench, AVG_FACET, method))
        if avg is None:
            return None
        p, m = idx.get((bench, "problem", method)), idx.get((bench, "method", method))
        return {"problem.node": _pct(p, "node_f1"), "problem.edge": _pct(p, "edge_f1"),
                "method.node": _pct(m, "node_f1"), "method.edge": _pct(m, "edge_f1"),
                "avg.node": _pct(avg, "node_f1"), "avg.edge": _pct(avg, "edge_f1"),
                "cost": avg.get("relative_cost")}

    rows, sections = _selection_sections(benchmark, eta, cells)
    return rows, sections, table8_spec(eta)


def table9(summary: Sequence[Mapping[str, Any]], eta: float = TABLE89_ETA,
           benchmark: str = TAXOBENCH) -> Table:
    """Table 9 cells (App. C.5): soft F1 (%) vs. the gold taxonomy, relative cost; FULL first."""
    idx = _index(summary)

    def cells(bench: str, method: str) -> Optional[Dict[str, Any]]:
        r = idx.get((bench, AVG_FACET, method))
        if r is None:
            return None
        return {"node": _pct(r, "node_f1"), "edge": _pct(r, "edge_f1"), "cost": r.get("relative_cost")}

    rows, sections = _selection_sections(benchmark, eta, cells)
    full = cells(benchmark, FULL)
    if full is not None:
        rows = OrderedDict([(FULL_LABEL, full)] + list(rows.items()))
        sections = [("", [FULL_LABEL])] + sections
    return rows, sections, table9_spec(eta)


def table6_spec() -> TableSpec:
    return TableSpec(
        "table6", "Table 6: Ablation of taxonomy construction on arXiv.",
        [Column("node", "Node"), Column("edge", "Edge"), Column("abstain", "Abstain", higher_is_better=False),
         Column("disagr", "Disagr.", higher_is_better=False), Column("cost", "Cost", higher_is_better=False, digits=3)],
        first_header="Variant",
    )


def table6(summary: Sequence[Mapping[str, Any]], benchmark: str = ARXIV,
           variants: Optional[Sequence[str]] = None,
           config: Optional[BuilderConfig] = None) -> Table:
    """Table 6 cells (App. C.4): soft F1 (%) vs. FULL, abstention a_v (%), disagreement d_v (%), relative cost.

    Averages over the sets and both facets (facet ``"avg"``); a_v and d_v
    are first averaged over the episodes of each build that trained a
    router (:func:`router_rates`; nodes left unsplit have no router).
    """
    idx = _index(summary)
    rows: Dict[str, Dict[str, Any]] = OrderedDict()
    names: List[str] = []
    for m in ablation_methods(variants):
        r = idx.get((benchmark, AVG_FACET, m.key))
        if r is None:
            continue
        name = m.label(config)
        rows[name] = {"node": _pct(r, "node_f1"), "edge": _pct(r, "edge_f1"), "abstain": _pct(r, "abstention"),
                      "disagr": _pct(r, "disagreement"), "cost": r.get("relative_cost")}
        names.append(name)
    return rows, ([("", names)] if names else []), table6_spec()


def render_table(table: Table, bold_best: bool = True, reference_rows: Sequence[str] = (FULL_LABEL,)) -> str:
    """Markdown table (:func:`scholarpath.evaluation.reporting.markdown_table`) with the paper's bolding.

    In every column the best value among the compared (budgeted) methods is
    bold across *all* their rows (``reporting`` bolds per section, which would
    never bold a method alone in its section, such as ScholarPath). Reference
    rows (``reference_rows``: the unbudgeted FULL of Table 9) are left out of
    that comparison and are bold where they are at least as good as its best,
    as in Table 9 (FULL's F1 and ScholarPath's F1 and cost are bold; FULL's
    cost of 1.000 is not).
    """
    rows, sections, spec = table
    refs = set(reference_rows)
    best: Dict[str, float] = {}
    for c in (spec.columns if bold_best else ()):
        nums = [round(x, c.digits) for x in (finite(r.get(c.key)) for n, r in rows.items() if n not in refs)
                if x is not None]
        if nums and len(rows) >= 2:  # a lone row is never bold
            best[c.key] = max(nums) if c.higher_is_better else min(nums)
    display: Dict[str, Dict[str, Any]] = OrderedDict()
    for name, row in rows.items():
        cells = dict(row)
        for c in spec.columns:
            x = finite(row.get(c.key))
            if x is None or c.key not in best:
                continue
            v, b = round(x, c.digits), best[c.key]
            if v == b or (name in refs and (v >= b if c.higher_is_better else v <= b)):
                cells[c.key] = f"**{format_value(x, c.digits)}**"
        display[name] = cells
    return markdown_table(display, spec, sections or None, bold_best=False)


def write_table_files(out_dir: Union[str, Path], name: str, table: Table, bold_best: bool = True) -> Optional[Path]:
    """``<name>.md`` (:func:`render_table`) and ``<name>.csv`` (raw values); skipped if the table is empty."""
    rows, sections, spec = table
    if not rows:
        return None
    path = write_markdown(Path(out_dir) / f"{name}.md", render_table(table, bold_best))
    fields = ["section", spec.first_header] + [c.header for c in spec.columns]
    write_csv(path.with_suffix(".csv"), table_rows(rows, spec, sections or None), fieldnames=fields)
    return path


def write_construction_report(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Union[str, Path],
    *,
    selectors: Sequence[str] = FIG2_SELECTORS,
    etas: Sequence[float] = FIG2_ETAS,
    fig3_etas: Sequence[float] = FIG3_ETAS,
    fig3_selector: str = FIG3_SELECTOR,
    table_eta: float = TABLE89_ETA,
    plots: bool = True,
) -> Dict[str, Any]:
    """Write every output of the construction comparison (Figs. 2-3, Tables 8-9).

    Files: ``runs.{csv,jsonl}`` (one row per build), ``summary.{csv,jsonl}``
    (means per benchmark / facet / method), ``fig2.csv``, ``fig3.csv``,
    ``table8.{md,csv}``, ``table9.{md,csv}`` and, with ``plots`` and
    matplotlib, ``fig2_<benchmark>_{node,edge}_f1.png`` and ``fig3.png``.
    """
    out = Path(out_dir)
    paths: Dict[str, Any] = {}
    summary = summarize_runs(rows)
    paths["runs"] = write_rows(out, "runs", rows)
    paths["summary"] = write_rows(out, "summary", summary)
    fig2 = figure2_rows(summary, selectors, etas)
    fig3 = figure3_rows(summary, fig3_etas, selectors, fig3_selector)
    paths["fig2"] = write_rows(out, "fig2", fig2)["csv"]
    paths["fig3"] = write_rows(out, "fig3", fig3)["csv"]
    benches = _benchmarks(summary)
    if ARXIV in benches:
        paths["table8"] = write_table_files(out, "table8", table8(summary, table_eta))
    if TAXOBENCH in benches:
        paths["table9"] = write_table_files(out, "table9", table9(summary, table_eta))
    if plots:
        if matplotlib_available():
            paths["fig2_plots"] = plot_figure2(fig2, out)
            paths["fig3_plot"] = plot_figure3(fig3, out / "fig3.png")
        else:
            log.warning("matplotlib is not installed; skipping the figures (their CSV data is written)")
    return paths


# ---------------------------------------------------------------------------
# Fig. 4: scalability (Section 5.2.2)
# ---------------------------------------------------------------------------


def nested_subsets(paper_ids: Iterable[str], sizes: Sequence[int], seed: int = 0) -> "OrderedDict[int, List[str]]":
    """Nested uniform samples: subset(n) = the first n papers of one seeded permutation.

    Sizes above the corpus size are clipped to it (the largest subset is then
    the whole corpus, as 148.4K is in the paper); duplicates are dropped.
    """
    ids = sorted(set(map(str, paper_ids)))
    order = [ids[i] for i in np.random.default_rng(int(seed)).permutation(len(ids))]
    out: "OrderedDict[int, List[str]]" = OrderedDict()
    for n in sorted({int(s) for s in sizes if int(s) > 0}):
        k = min(n, len(ids))
        if k < n:
            log.warning("requested |P| = %d exceeds the corpus (%d papers); using the whole corpus", n, len(ids))
        if k not in out:
            out[k] = sorted(order[:k])
    return out


def size_label(n: int) -> str:
    """``2000 -> "2K"``, ``148400 -> "148.4K"``, ``500 -> "500"``."""
    if n >= 1000:
        k = n / 1000.0
        return f"{k:.1f}".rstrip("0").rstrip(".") + "K"
    return str(n)


def run_scalability(
    corpus: Corpus,
    sizes: Sequence[int],
    runner: ConstructionRunner,
    *,
    etas: Sequence[float] = SCALABILITY_ETAS,
    selector: str = "random",
    max_full_size: int = 10000,
    seed: Optional[int] = None,
    sample_seed: int = 0,
    facets: Sequence[str] = DUAL_FACETS,
    name: str = SUBFIELDS,
) -> List[Dict[str, Any]]:
    """Fig. 4: build the dual taxonomy at every corpus size and measure C_LLM and C_Full.

    At each size ScholarPath and node-wise selection at each eta build both
    facets; FULL is run only up to ``max_full_size`` papers. Above it,
    C_Full is estimated (Section 5.2.2, App. B.2) as

        C_Full ≈ c̄ Σ_{v ∈ V_int} n_v,

    with c̄ = C_Full / Σ_{v ∈ V_int} n_v measured (per facet) on FULL at the
    largest size where FULL was run (:func:`~scholarpath.taxonomy.baselines.per_paper_node_cost`) and the
    routed paper counts n_v of ScholarPath's taxonomy at the target size
    (:func:`~scholarpath.taxonomy.baselines.estimate_full_cost`). Where FULL was run, the measured C_Full is
    used and the estimate is reported alongside for validation.

    Returns one row per (size, facet, method), FULL included (``method`` =
    ``"full"`` when measured, ``"full_est"`` when estimated).
    """
    subsets = nested_subsets(corpus.ids, sizes, sample_seed)
    if not subsets:
        raise ValueError("no corpus size to run")
    if min(subsets) > max_full_size:
        raise ValueError(f"FULL must run at least at the smallest size ({min(subsets)}) to measure c̄; "
                         f"raise max_full_size (currently {max_full_size})")
    all_vecs = embed_corpus(runner.embedder, corpus)
    methods = [SCHOLARPATH_METHOD] + [MethodSpec.selection(selector, e, NODE_WISE) for e in etas]
    # Only light statistics are kept per build (results of large corpora are big).
    stats: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    c_bar_at: Dict[Tuple[int, str], float] = {}
    for n, ids in subsets.items():
        sub = corpus.subset(ids, name=f"{name}-{n}")
        ds = ConstructionDataset(name=f"{name}-{n}", benchmark=SUBFIELDS, corpus=sub, facets=tuple(facets),
                                 vectors={p: all_vecs[p] for p in ids})
        for facet in facets:
            if n <= max_full_size:
                full = runner.build(ds, facet, FULL_METHOD)
                stats[(n, facet, FULL)] = _scal_stats(full)
                c_bar_at[(n, facet)] = per_paper_node_cost(full)
            for m in methods:
                stats[(n, facet, m.key)] = _scal_stats(runner.build(ds, facet, m, seed))
    # c̄ per facet from FULL at the largest size where it was run.
    c_bar_size = max(n for n in subsets if n <= max_full_size)
    for facet in facets:
        if finite(c_bar_at[(c_bar_size, facet)]) is None and any(n > max_full_size for n in subsets):
            log.warning("FULL (%s, |P| = %d) has no internal node, so c̄ is undefined and C_Full cannot be "
                        "estimated above |P| = %d; raise --max-full-size", facet, c_bar_size, max_full_size)
    rows: List[Dict[str, Any]] = []
    for n in subsets:
        for facet in facets:
            c_bar = c_bar_at[(c_bar_size, facet)]
            sp = stats[(n, facet, SCHOLARPATH)]
            # estimate_full_cost(ScholarPath's result, c̄) = c̄ Σ_{v ∈ V_int} n_v
            estimate = c_bar * sp["internal_paper_count"]
            measured = stats.get((n, facet, FULL))
            c_full = measured["cost_usd"] if measured is not None else estimate
            base = {"size": n, "size_label": size_label(n), "facet": facet, "c_bar": c_bar,
                    "c_bar_size": c_bar_size, "full_cost_usd": c_full, "full_cost_estimated_usd": estimate,
                    "full_measured": measured is not None}
            if measured is not None:
                rows.append({**base, **measured, "method": FULL, "label": "Full (η = 1.0)", "selected_ratio": 1.0})
            else:
                rows.append({**base, "method": "full_est", "label": "Full (est.)", "cost_usd": estimate,
                             "selected_ratio": 1.0, "internal_paper_count": sp["internal_paper_count"]})
            for m in methods:
                st = stats[(n, facet, m.key)]
                label = SCHOLARPATH_LABEL if m.is_scholarpath else f"Node-wise (η = {m.eta:g})"
                ratio = st["cost_usd"] / c_full if finite(c_full) is not None and c_full > 0 else None
                rows.append({**base, **st, "method": m.key, "label": label, "selected_ratio": ratio})
    return rows


def _scal_stats(result: BuildResult) -> Dict[str, Any]:
    usage = result.total_usage
    return {"cost_usd": result.cost_usd, "tokens": usage.total_tokens, "calls": usage.calls,
            "runtime_s": result.runtime_s,
            "n_nodes": len(result.taxonomy.nodes) - 1, "depth": result.taxonomy.max_depth,
            "internal_paper_count": internal_paper_count(result.taxonomy),
            "papers_read": int(sum(result.papers_read.values()))}


def _total(values: Iterable[Any]) -> Optional[float]:
    """Sum of ``values``, or ``None`` if any of them is missing / non-finite (never a silent 0)."""
    nums = [finite(v) for v in values]
    return None if not nums or any(x is None for x in nums) else float(sum(nums))


def scalability_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Fig. 4 series per (size, method), pooling the facets of the dual taxonomy.

    * (a) ``selected_ratio_pct`` = 100 Σ_f C_LLM(f) / Σ_f C_Full(f), the
      ratio of the dual taxonomy's total costs: the paper's 3.9% at 148.4K is
      ScholarPath's $5.4 over FULL's $138.4 of Fig. 4(b) (Section 5.2.2).
      The mean of the per-facet ratios is kept as ``mean_facet_ratio_pct``.
    * (b) ``total_cost_usd`` = Σ_f C_LLM(f) ($), FULL (measured or estimated)
      included.

    A missing or non-finite cost (e.g. C_Full estimated with an undefined c̄)
    makes the totals ``None`` instead of counting as $0.
    """
    groups: "OrderedDict[Tuple[int, str], List[Mapping[str, Any]]]" = OrderedDict()
    for r in rows:
        method = FULL if r["method"] in (FULL, "full_est") else r["method"]
        groups.setdefault((int(r["size"]), method), []).append(r)
    out = []
    for (n, method), members in groups.items():
        estimated = any(not m["full_measured"] for m in members) if method == FULL else None
        label = members[0]["label"] if method != FULL else ("Full (est.)" if estimated else "Full (η = 1.0)")
        total = _total(m.get("cost_usd") for m in members)
        total_full = _total(m.get("full_cost_usd") for m in members)
        if total is None or total_full is None:
            log.warning("Fig. 4 (|P| = %s, %s): a cost is missing or non-finite; reporting no total", n, method)
        ratio = total / total_full if total is not None and total_full is not None and total_full > 0 else None
        out.append({"size": n, "size_label": size_label(n), "method": method, "label": label,
                     "selected_ratio_pct": scale(ratio),
                     "mean_facet_ratio_pct": scale(mean_of(m.get("selected_ratio") for m in members)),
                     "total_cost_usd": total, "full_cost_usd": total_full,
                     "facets": len(members), "full_estimated": estimated})
    return out


def figure4_table(summary: Sequence[Mapping[str, Any]]) -> Table:
    """Markdown view of Fig. 4: one row per size, selected ratio (%) and total cost ($) per method."""
    methods = list(OrderedDict((r["method"], r["label"] if r["method"] != FULL else "Full") for r in summary).items())
    cols = [Column(f"{m}.ratio", "Ratio (%)", label, higher_is_better=False) for m, label in methods if m != FULL]
    cols += [Column(f"{m}.cost", "Cost ($)", label, higher_is_better=False, digits=3) for m, label in methods]
    rows: Dict[str, Dict[str, Any]] = OrderedDict()
    for r in summary:
        cell = rows.setdefault(r["size_label"], {})
        cell[f"{r['method']}.ratio"] = r["selected_ratio_pct"]
        cell[f"{r['method']}.cost"] = r["total_cost_usd"]
    title = "Figure 4: Scalability of taxonomy construction across corpus sizes on arXiv-Subfields."
    spec = TableSpec("figure4", title, cols, first_header="|P|")
    return rows, [], spec


def plot_figure4(summary: Sequence[Mapping[str, Any]], out_dir: Union[str, Path]) -> List[Path]:
    """Fig. 4(a) average selected ratio and 4(b) total construction cost (log scale) vs. |P|."""
    sizes = sorted({int(r["size"]) for r in summary})
    pos = {n: i for i, n in enumerate(sizes)}
    labels = [size_label(n) for n in sizes]
    full_name = "Full (est.)" if any(r["method"] == FULL and r.get("full_estimated") for r in summary) else "Full"
    order = [SCHOLARPATH_LABEL] + [f"Node-wise (η = {e:g})" for e in (0.5, 0.2)] + [full_name]

    def series(metric: str, include_full: bool) -> "OrderedDict[str, List[Tuple[float, Optional[float]]]]":
        out: "OrderedDict[str, List[Tuple[float, Optional[float]]]]" = OrderedDict()
        for r in summary:
            if r["method"] == FULL and not include_full:
                continue
            name = full_name if r["method"] == FULL else r["label"]
            out.setdefault(name, []).append((pos[int(r["size"])], r[metric]))
        return out

    a = plot_lines(series("selected_ratio_pct", False), Path(out_dir) / "fig4a_selected_ratio.png",
                   xlabel="Corpus size |P|", ylabel="Selected Ratio (%)", title="(a) Average Selected Ratio",
                   order=order, xticks=list(range(len(sizes))), xticklabels=labels)
    b = plot_lines(series("total_cost_usd", True), Path(out_dir) / "fig4b_total_cost.png",
                   xlabel="Corpus size |P|", ylabel="Total Cost ($)", title="(b) Total Construction Cost",
                   order=order, xticks=list(range(len(sizes))), xticklabels=labels, yscale="log")
    return [a, b]


# ---------------------------------------------------------------------------
# Dynamic update (Section 3.4, Appendix B.3)
# ---------------------------------------------------------------------------


@dataclass
class Cohort:
    """Papers arriving together (one month, or one fixed-size batch)."""

    label: str
    papers: List[Paper]


@dataclass
class StreamSplit:
    """A corpus split at ``cutoff`` into the base repository and the later arrival cohorts."""

    cutoff: str
    base: Corpus
    cohorts: List[Cohort]
    undated: int = 0

    @property
    def streamed(self) -> List[Paper]:
        return [p for c in self.cohorts for p in c.papers]


def default_cutoff(papers: Iterable[Paper], base_fraction: float = 0.8) -> str:
    """The first-version date that leaves about ``base_fraction`` of the dated papers before it."""
    dates = sorted(p.date for p in papers if p.date)
    if len(dates) < 2:
        raise ValueError("need at least two dated papers to choose a cutoff")
    k = min(len(dates) - 1, max(1, int(math.ceil(float(base_fraction) * len(dates)))))
    return str(dates[k])


def split_stream(
    corpus: Corpus,
    cutoff: Optional[str] = None,
    base_fraction: float = 0.8,
    cohort: str = "month",
    batch_size: Optional[int] = None,
    max_cohorts: Optional[int] = None,
) -> StreamSplit:
    """Split a corpus at a date into the base repository and a stream of arrival cohorts.

    The base holds the papers first submitted strictly before ``cutoff``
    (default: :func:`default_cutoff`); later papers arrive in date order,
    grouped by calendar month (``cohort="month"``) or in batches of
    ``batch_size`` papers (``cohort="size"``). Undated papers cannot be placed
    on the timeline and are left out (counted in ``undated``).
    """
    dated = [p for p in corpus if p.date]
    undated = len(corpus) - len(dated)
    cutoff = cutoff or default_cutoff(dated, base_fraction)
    base = Corpus((p for p in dated if p.date < cutoff), name=f"{corpus.name}<{cutoff}")
    later = sorted((p for p in dated if p.date >= cutoff), key=lambda p: (p.date, p.paper_id))
    cohorts: List[Cohort] = []
    if cohort == "month":
        by_month: "OrderedDict[str, List[Paper]]" = OrderedDict()
        for p in later:
            by_month.setdefault(str(p.date)[:7], []).append(p)
        cohorts = [Cohort(m, ps) for m, ps in by_month.items()]
    elif cohort == "size":
        if not batch_size or batch_size <= 0:
            raise ValueError("cohort='size' needs a positive batch_size")
        for start in range(0, len(later), int(batch_size)):
            ps = later[start : start + int(batch_size)]
            cohorts.append(Cohort(f"{ps[0].date}..{ps[-1].date}", ps))
    else:
        raise ValueError(f"cohort must be 'month' or 'size', got {cohort!r}")
    if max_cohorts is not None:
        cohorts = cohorts[: max(0, int(max_cohorts))]
    if not len(base):
        raise ValueError(f"no paper precedes the cutoff {cutoff}")
    return StreamSplit(cutoff=cutoff, base=base, cohorts=cohorts, undated=undated)


def replay_prior(config: BuilderConfig, episodes: Sequence[EpisodeRecord]) -> TransferredPrior:
    """The transferred prior after the history H of ``episodes`` (Algorithm 1, line 21; Eq. 14).

    Re-applies, in build order, the (region descriptor, observed slope) pairs
    recorded with every episode, as :meth:`NodeWiseBuilder.finalize` did
    during the build, so that repairs during an update start from the
    build's posterior over (W, Ω) rather than from scratch.
    """
    prior = TransferredPrior.from_config(config)
    for e in episodes:
        if e.region_descriptors and len(e.region_descriptors) == len(e.region_slopes):
            prior.update(e.region_descriptors, e.region_slopes)
    return prior


@dataclass
class DynamicUpdateRun:
    """One facet of the dynamic-update study."""

    facet: str
    base: BuildResult
    updated: BuildResult
    cohorts: List[Dict[str, Any]]
    reports: List[UpdateReport]


def taxonomy_certificates(tax: Taxonomy) -> Dict[str, Any]:
    """Certified internal nodes and their bounds (Section 3.3)."""
    internal = tax.internal_nodes()
    certified = [n for n in tax.nodes.values() if n.status == CERTIFIED]
    bounds = [float(n.bound) for n in certified if n.bound is not None]
    certified_internal = sum(1 for n in internal if tax.nodes[n].status == CERTIFIED)
    return {"internal_nodes": len(internal), "certified_nodes": len(certified),
            "certified_fraction": certified_internal / max(1, len(internal)),
            "max_bound": max(bounds) if bounds else None}


def run_dynamic_update(
    base: ConstructionDataset,
    cohorts: Sequence[Cohort],
    runner: ConstructionRunner,
    facet: str,
    seed: Optional[int] = None,
) -> DynamicUpdateRun:
    """Build ScholarPath on ``base`` and stream ``cohorts`` through :class:`DynamicUpdater` (Sec. 3.4).

    The base build is taken from the cache and deep-copied before updating.
    The updater gets a :class:`NodeWiseBuilder` with the same LLM, embedder,
    configuration and repository as the base build and the transferred prior
    replayed from its episodes (:func:`replay_prior`); a frozen router that
    is missing is retrained from the persisted verified labels at zero LLM
    cost. Each cohort runs against a fresh cost meter.

    Per cohort the record reports the nodes whose certificate survived at
    zero LLM cost (U_new = (n U_old + M) / (n + M) <= eps and the residual
    below the discovery threshold, Eq. 16), the nodes whose new cohort was
    audited, repaired (Stage 2 + new snapshot) or failed, and the LLM cost.
    """
    seed = runner.resolve_seed(SCHOLARPATH_METHOD, seed)
    base_result = runner.build(base, facet, SCHOLARPATH_METHOD, seed)
    result = copy.deepcopy(base_result)
    cfg = runner.builder_config(facet, seed)
    builder = NodeWiseBuilder(runner.llm, runner.embedder, cfg, base.corpus, paper_vectors=runner.vectors(base),
                              prior=replay_prior(cfg, result.episodes))
    updater = DynamicUpdater(builder, result)
    rows: List[Dict[str, Any]] = []
    reports: List[UpdateReport] = []
    cumulative = 0.0
    base_cost = base_result.cost_usd
    for i, cohort in enumerate(cohorts):
        with fresh_meter(runner.llm):
            report = updater.add_papers(cohort.papers)
        reports.append(report)
        usage_tokens = sum(r.total_tokens for r in report.usage.values())
        cumulative += report.cost_usd
        certs = taxonomy_certificates(result.taxonomy)
        rows.append({
            "facet": facet, "cohort": cohort.label, "index": i, "n_new": report.n_new, "skipped": len(report.skipped),
            "nodes_with_arrivals": len(report.arrivals),
            "kept_zero_cost": len(report.kept), "reaudited": len(report.reaudited),
            "recertified": len(report.recertified), "repaired": len(report.repaired), "failed": len(report.failed),
            "uncertified_routed": len(report.uncertified), "new_nodes": len(report.new_nodes),
            "grown_leaves": len(report.grown_leaves), "rebuilt_routers": len(report.rebuilt_routers),
            "zero_llm_cost": report.llm_calls == 0, "calls": report.llm_calls, "tokens": int(usage_tokens),
            "cost_usd": report.cost_usd, "cumulative_cost_usd": cumulative,
            "cumulative_relative_to_base": cumulative / base_cost if base_cost > 0 else None,
            "papers_read": int(sum(report.papers_read.values())), "runtime_s": report.runtime_s,
            "n_papers": len(result.taxonomy.root.papers), "n_nodes": len(result.taxonomy.nodes) - 1, **certs,
        })
        log.info("[%s] cohort %s: %d papers, kept %d, re-audited %d, repaired %d, $%.4f", facet, cohort.label,
                 report.n_new, len(report.kept), len(report.reaudited), len(report.repaired), report.cost_usd)
    return DynamicUpdateRun(facet=facet, base=base_result, updated=result, cohorts=rows, reports=reports)


def dynamic_summary(
    run: DynamicUpdateRun,
    scratch: Optional[BuildResult] = None,
    scorer: Optional[TaxonomyScorer] = None,
) -> Dict[str, Any]:
    """Totals of a dynamic-update run, optionally compared with a from-scratch build on all papers."""
    rows = run.cohorts
    out: Dict[str, Any] = {
        "facet": run.facet,
        "cohorts": len(rows),
        "papers_streamed": int(sum(r["n_new"] for r in rows)),
        "zero_cost_cohorts": int(sum(1 for r in rows if r["zero_llm_cost"])),
        "kept_zero_cost": int(sum(r["kept_zero_cost"] for r in rows)),
        "reaudited": int(sum(r["reaudited"] for r in rows)),
        "repaired": int(sum(r["repaired"] for r in rows)),
        "failed": int(sum(r["failed"] for r in rows)),
        "new_nodes": int(sum(r["new_nodes"] for r in rows)),
        "base_papers": len(run.base.taxonomy.root.papers),
        "base_cost_usd": run.base.cost_usd,
        "update_cost_usd": float(sum(r["cost_usd"] for r in rows)),
        "final_papers": len(run.updated.taxonomy.root.papers),
        "final_nodes": len(run.updated.taxonomy.nodes) - 1,
        **taxonomy_certificates(run.updated.taxonomy),
    }
    if scratch is not None:
        out["scratch_cost_usd"] = scratch.cost_usd
        out["update_vs_scratch_cost"] = out["update_cost_usd"] / scratch.cost_usd if scratch.cost_usd > 0 else None
        total = out["base_cost_usd"] + out["update_cost_usd"]
        out["base_plus_update_vs_scratch_cost"] = total / scratch.cost_usd if scratch.cost_usd > 0 else None
        out["scratch_nodes"] = len(scratch.taxonomy.nodes) - 1
        if scorer is not None:
            s = score_taxonomy(run.updated.taxonomy, scratch.taxonomy, scorer)
            out.update({f"vs_scratch_{k}": s[k] for k in METRICS})
    return out


__all__ = [
    "ABLATIONS",
    "AVG_FACET",
    "TAXOBENCH_GOLD_MAX_DEPTH",
    "UNROUTED_TERMINATIONS",
    "Ablation",
    "Cohort",
    "ConstructionDataset",
    "ConstructionRunner",
    "DUAL_FACETS",
    "DynamicUpdateRun",
    "FACET_ROOTS",
    "FIG2_ETAS",
    "FIG2_SELECTORS",
    "FIG3_ETAS",
    "FIG3_MEAN",
    "FIG3_SELECTOR",
    "FULL_METHOD",
    "METRICS",
    "MethodSpec",
    "SCALABILITY_ETAS",
    "SCALABILITY_SIZES",
    "SCHOLARPATH_METHOD",
    "StreamSplit",
    "TABLE89_ETA",
    "TOPIC_FACET",
    "ablation_methods",
    "arxiv_set_paths",
    "construction_methods",
    "default_cutoff",
    "dynamic_summary",
    "figure2_rows",
    "figure3_rows",
    "figure4_table",
    "load_arxiv_datasets",
    "load_taxobench_dataset",
    "parse_max_depth",
    "make_scorer",
    "nested_subsets",
    "plot_figure2",
    "plot_figure3",
    "plot_figure4",
    "render_table",
    "replay_prior",
    "restrict_taxonomy",
    "run_construction_comparison",
    "run_dynamic_update",
    "routed_episodes",
    "router_rates",
    "run_record",
    "run_scalability",
    "scalability_summary",
    "score_taxonomy",
    "size_label",
    "split_stream",
    "summarize_runs",
    "table6",
    "table8",
    "table9",
    "taxonomy_certificates",
    "truncate_taxonomy",
    "write_construction_report",
    "write_table_files",
]
