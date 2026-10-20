"""Result files and paper-style tables (Tables 2-5, Figs. 5-6).

* :func:`write_jsonl` / :func:`write_csv` -- per-query rows and summaries;
* :func:`markdown_table` / :func:`write_table` -- Markdown + CSV tables shaped
  like Tables 2-5 (rows = systems grouped into sections such as "Budgeted
  methods (B = 8,192)" / "Unbudgeted methods"; columns = benchmark metrics;
  the best value of each column in each section is bold);
* :func:`write_budget_sweep` / :func:`plot_budget_sweep` -- context-budget
  sweeps (Fig. 5: macro Recall@B / NDCG@B; Fig. 6: hypothesis Disc) as long
  and wide CSVs and an optional matplotlib line chart (imported lazily).

Systems' results are flat dicts keyed by the column keys of a
:class:`TableSpec`, e.g. ``{"researchqa.R": 70.3, "researchqa.N": 61.8}`` for
Table 2 (:meth:`RetrievalEvaluation.table_row` produces exactly these keys).
For Table 3, :func:`flatten_benchmarks` turns the per-benchmark QA summaries
(already in percent, keys ``Rub`` / ``Cite`` / ``Corr``) into such keys, and
:func:`organization_table_row` maps the fractional ``ari`` / ``sem`` / ``cat``
/ ``assn`` outputs of :mod:`.organization_metrics` to ``taxobench.ARI``,
``taxobench.Sem``, ``chime.Cat`` and ``chime.Assn`` in percent.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]
Sections = Sequence[Tuple[str, Sequence[str]]]


# ---------------------------------------------------------------------------
# Plain files
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """JSON/CSV-safe scalar (numpy numbers, NaN -> None, dataclass-free)."""
    if hasattr(value, "item") and callable(value.item) and not isinstance(value, (list, tuple, dict)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_jsonl(path: PathLike, rows: Iterable[Mapping[str, Any]]) -> Path:
    """One JSON object per line (``default=str`` for non-JSON values)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: _plain(v) for k, v in row.items()}, ensure_ascii=False, default=_json_default) + "\n")
    return path


def _json_default(value: Any) -> Any:
    plain = _plain(value)
    return plain if plain is not value else str(value)


def write_json(path: PathLike, obj: Any) -> Path:
    """Pretty-printed JSON (numpy scalars converted, other objects stringified)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return path


def write_csv(path: PathLike, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> Path:
    """CSV with the union of row keys (first-seen order) unless ``fieldnames`` is given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: Dict[str, None] = {}
        for row in rows:
            for k in row:
                seen.setdefault(k, None)
        fieldnames = list(seen)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = _plain(row.get(k))
                out[k] = "" if v is None else (json.dumps(v) if isinstance(v, (list, dict)) else v)
            writer.writerow(out)
    return path


def write_summary_csv(path: PathLike, summaries: Mapping[str, Mapping[str, Any]], key: str = "system") -> Path:
    """One row per system: ``{system: {metric: value}}`` -> CSV."""
    rows = [{key: name, **dict(vals)} for name, vals in summaries.items()]
    return write_csv(path, rows)


def write_run(
    output_dir: PathLike,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Dict[str, Path]:
    """Per-query ``<name>.jsonl`` + ``<name>_summary.json`` for one system / benchmark run."""
    out = Path(output_dir)
    return {"rows": write_jsonl(out / f"{name}.jsonl", rows),
            "summary": write_json(out / f"{name}_summary.json", dict(summary))}


# ---------------------------------------------------------------------------
# Paper tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    key: str  # lookup key in a system's result dict
    label: str  # column header (metric)
    group: str = ""  # super-header (benchmark)
    higher_is_better: bool = True
    digits: int = 1

    @property
    def header(self) -> str:
        return f"{self.group} {self.label}".strip()


@dataclass
class TableSpec:
    name: str
    title: str
    columns: List[Column] = field(default_factory=list)
    first_header: str = "System"


_BENCH = (("researchqa", "ResearchQA"), ("scholarqabench", "ScholarQA"), ("taxobench", "TaxoBench"), ("chime", "CHIME"))

TABLE2 = TableSpec(
    "table2", "Table 2: Retrieval accuracy (%) on evidence queries (B = 8,192).",
    [Column(f"{b}.{m}", m, d) for b, d in _BENCH for m in ("R", "N")], first_header="Method",
)
TABLE3 = TableSpec(
    "table3", "Table 3: End-to-end QA results (%) on evidence queries.",
    [Column("researchqa.Rub", "Rub", "ResearchQA"), Column("researchqa.Cite", "Cite", "ResearchQA"),
     Column("scholarqabench.Corr", "Corr", "ScholarQA"), Column("scholarqabench.Cite", "Cite", "ScholarQA"),
     Column("taxobench.ARI", "ARI", "TaxoBench"), Column("taxobench.Sem", "Sem", "TaxoBench"),
     Column("chime.Cat", "Cat", "CHIME"), Column("chime.Assn", "Assn", "CHIME")],
)
TABLE4 = TableSpec(
    "table4", "Table 4: End-to-end QA results on hypothesis queries.",
    [Column("Hit@10", "Hit@10"), Column("Feas.", "Feas.", "LLM Judge"), Column("Nov.", "Nov.", "LLM Judge"),
     Column("Sound.", "Sound.", "LLM Judge"), Column("Avg.", "Avg.", "LLM Judge")],
)
TABLE5 = TableSpec(
    "table5", "Table 5: Per-query efficiency on end-to-end QA.",
    [Column("Calls", "Calls", higher_is_better=False), Column("Tokens (K)", "Tokens (K)", higher_is_better=False),
     Column("Cost ($)", "Cost ($)", higher_is_better=False, digits=3),
     Column("Latency (s)", "Latency (s)", higher_is_better=False)],
)
PAPER_TABLES = {t.name: t for t in (TABLE2, TABLE3, TABLE4, TABLE5)}


def flatten_benchmarks(by_benchmark: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """``{"researchqa": {"Rub": 74.2}}`` -> ``{"researchqa.Rub": 74.2}`` (values copied as they are).

    Organization metrics need :func:`organization_table_row` (key mapping and
    fraction -> percent) instead.
    """
    return {f"{b}.{k}": v for b, vals in by_benchmark.items() for k, v in vals.items()}


#: Table 3 columns of the organization benchmarks: output key of
#: :func:`~scholarpath.evaluation.organization_metrics.organization_scores` -> column label.
ORGANIZATION_COLUMNS: Dict[str, Dict[str, str]] = {
    "taxobench": {"ari": "ARI", "sem": "Sem"},
    "chime": {"cat": "Cat", "assn": "Assn"},
}


def organization_table_row(summary: Mapping[str, Any], benchmark: str, scale: float = 100.0) -> Dict[str, Any]:
    """Table 3 cells of TaxoBench / CHIME from an organization-metrics summary.

    ``summary`` is the (macro-averaged) output of
    :func:`~scholarpath.evaluation.organization_metrics.organization_scores`
    / ``macro_average`` for ``benchmark`` (``"taxobench"`` or ``"chime"``),
    whose ``ari`` / ``sem`` / ``cat`` / ``assn`` values are fractions; they
    are multiplied by ``scale`` (Table 3 is in percent) and keyed
    ``"<benchmark>.<Label>"``. Missing metrics map to ``None``.
    """
    bench = benchmark.lower().replace("-", "").replace("_", "")
    if bench not in ORGANIZATION_COLUMNS:
        raise ValueError(f"Unknown organization benchmark {benchmark!r}; "
                         f"expected one of {sorted(ORGANIZATION_COLUMNS)}")
    out: Dict[str, Any] = {}
    for key, label in ORGANIZATION_COLUMNS[bench].items():
        value = _plain(summary.get(key))
        out[f"{bench}.{label}"] = None if value is None else scale * float(value)
    return out


def format_value(value: Any, digits: int = 1) -> str:
    value = _plain(value)
    if value is None:
        return "–"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _best(values: Sequence[Any], higher: bool, digits: int) -> Optional[float]:
    nums = [round(float(v), digits) for v in (_plain(x) for x in values)
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) < 2:
        return None
    return max(nums) if higher else min(nums)


def _sections(rows: Mapping[str, Mapping[str, Any]], sections: Optional[Sections]) -> List[Tuple[str, List[str]]]:
    if not sections:
        return [("", list(rows))]
    out = [(title, [s for s in systems if s in rows]) for title, systems in sections]
    listed = {s for _, systems in out for s in systems}
    rest = [s for s in rows if s not in listed]
    if rest:
        out.append(("Other", rest))
    return [(t, s) for t, s in out if s]


def markdown_table(
    rows: Mapping[str, Mapping[str, Any]],
    spec: TableSpec,
    sections: Optional[Sections] = None,
    bold_best: bool = True,
    title: bool = True,
) -> str:
    """Render ``{system: {column key: value}}`` as a Markdown table shaped like ``spec``.

    ``sections`` = ``[("Budgeted methods (B = 8,192)", [systems...]), ("Unbudgeted methods", [...])]``
    inserts italic section rows; the best value per column within each
    section is bold (lower is better for efficiency columns).
    """
    header = [spec.first_header] + [c.header for c in spec.columns]
    lines = []
    if title:
        lines += [f"**{spec.title}**", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([":---"] + ["---:"] * len(spec.columns)) + "|")
    for sec_title, systems in _sections(rows, sections):
        if sec_title:
            lines.append("| " + " | ".join([f"*{sec_title}*"] + [""] * len(spec.columns)) + " |")
        best = {c.key: _best([rows[s].get(c.key) for s in systems], c.higher_is_better, c.digits) for c in spec.columns}
        for s in systems:
            cells = [s]
            for c in spec.columns:
                v = _plain(rows[s].get(c.key))
                text = format_value(v, c.digits)
                if (bold_best and best[c.key] is not None and isinstance(v, (int, float))
                        and not isinstance(v, bool) and round(float(v), c.digits) == best[c.key]):
                    text = f"**{text}**"
                cells.append(text)
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def table_rows(
    rows: Mapping[str, Mapping[str, Any]], spec: TableSpec, sections: Optional[Sections] = None
) -> List[Dict[str, Any]]:
    """Flat CSV rows (``section``, system, then one column per metric header)."""
    out = []
    for sec_title, systems in _sections(rows, sections):
        for s in systems:
            row: Dict[str, Any] = {"section": sec_title, spec.first_header: s}
            for c in spec.columns:
                row[c.header] = _plain(rows[s].get(c.key))
            out.append(row)
    return out


def write_table(
    path: PathLike,
    rows: Mapping[str, Mapping[str, Any]],
    spec: TableSpec,
    sections: Optional[Sections] = None,
    csv_copy: bool = True,
) -> Path:
    """Write ``path`` (Markdown) and, with ``csv_copy``, the same table as ``path.csv``."""
    path = Path(path)
    if path.suffix != ".md":
        path = path.with_suffix(".md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(rows, spec, sections), encoding="utf-8")
    if csv_copy:
        fields = ["section", spec.first_header] + [c.header for c in spec.columns]
        write_csv(path.with_suffix(".csv"), table_rows(rows, spec, sections), fieldnames=fields)
    return path


# ---------------------------------------------------------------------------
# Budget sweeps (Figs. 5-6)
# ---------------------------------------------------------------------------

Sweep = Mapping[str, Mapping[int, Mapping[str, Any]]]  # system -> budget -> metric -> value


def normalize_sweep(sweep: Mapping[str, Mapping[Any, Mapping[str, Any]]]) -> Dict[str, Dict[int, Mapping[str, Any]]]:
    """Budgets as ``int`` keys (a sweep read back from JSON has ``"8192"`` string keys)."""
    out: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for system, by_b in sweep.items():
        budgets: Dict[int, Mapping[str, Any]] = {}
        for b, vals in by_b.items():
            budget = int(b)
            if budget in budgets:
                raise ValueError(f"Budget {budget} appears twice for {system!r}")
            budgets[budget] = vals
        out[system] = budgets
    return out


def sweep_long_rows(sweep: Sweep) -> List[Dict[str, Any]]:
    """One row per (system, budget, metric)."""
    sweep = normalize_sweep(sweep)
    return [{"system": s, "budget": b, "metric": m, "value": _plain(v)}
            for s, by_b in sweep.items() for b in sorted(by_b) for m, v in by_b[b].items()]


def sweep_wide_rows(sweep: Sweep, metric: str) -> List[Dict[str, Any]]:
    """One row per system, one column per budget, for ``metric``."""
    sweep = normalize_sweep(sweep)
    budgets = sorted({b for by_b in sweep.values() for b in by_b})
    return [{"system": s, **{str(b): _plain(by_b.get(b, {}).get(metric)) for b in budgets}} for s, by_b in sweep.items()]


def write_budget_sweep(path: PathLike, sweep: Sweep, metrics: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """Long CSV at ``path`` plus one wide CSV per metric (``<stem>_<metric>.csv``)."""
    path = Path(path)
    sweep = normalize_sweep(sweep)
    out = {"long": write_csv(path, sweep_long_rows(sweep), fieldnames=["system", "budget", "metric", "value"])}
    metrics = list(metrics or sorted({m for by_b in sweep.values() for vals in by_b.values() for m in vals}))
    for m in metrics:
        safe = "".join(ch if ch.isalnum() else "_" for ch in m).strip("_")
        out[m] = write_csv(path.with_name(f"{path.stem}_{safe}.csv"), sweep_wide_rows(sweep, m))
    return out


# Reference categorical palette (fixed order, never cycled) and recessive ink / grid colours.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
_SURFACE, _INK, _INK2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def plot_budget_sweep(
    sweep: Sweep,
    metric: str,
    path: PathLike,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    systems: Optional[Sequence[str]] = None,
) -> Path:
    """Line chart of ``metric`` vs. context budget B, one line per system (Figs. 5-6).

    Colours follow each system's position in ``systems`` (fixed order, at
    most eight series); every series also has its own marker so identity is
    never colour-only. Requires matplotlib (imported lazily).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError("plot_budget_sweep needs matplotlib (pip install matplotlib)") from err
    sweep = normalize_sweep(sweep)
    systems = list(systems or sweep)
    if len(systems) > len(_SERIES):
        raise ValueError(f"At most {len(_SERIES)} systems per chart; facet the rest into another figure")
    budgets = sorted({int(b) for s in systems for b in sweep.get(s, {})})
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    for i, s in enumerate(systems):
        pts = [(b, _plain(sweep[s][b].get(metric))) for b in budgets if b in sweep.get(s, {})]
        pts = [(b, v) for b, v in pts if isinstance(v, (int, float))]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=_SERIES[i], linewidth=2, solid_joinstyle="round", solid_capstyle="round",
                marker=_MARKERS[i], markersize=6, markeredgecolor=_SURFACE, markeredgewidth=1.5, label=s, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b // 1024}K" if b % 1024 == 0 else str(b) for b in budgets])
    ax.minorticks_off()
    ax.set_xlabel("Context token budget B", color=_INK2)
    ax.set_ylabel(ylabel or metric, color=_INK2)
    if title:
        ax.set_title(title, color=_INK, fontsize=10)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=8)
    if len(systems) >= 2:
        leg = ax.legend(fontsize=7, frameon=False, labelcolor=_INK, loc="best")
        leg.set_zorder(4)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    return path


__all__ = [
    "Column",
    "ORGANIZATION_COLUMNS",
    "PAPER_TABLES",
    "TABLE2",
    "TABLE3",
    "TABLE4",
    "TABLE5",
    "TableSpec",
    "flatten_benchmarks",
    "format_value",
    "markdown_table",
    "normalize_sweep",
    "organization_table_row",
    "plot_budget_sweep",
    "sweep_long_rows",
    "sweep_wide_rows",
    "table_rows",
    "write_budget_sweep",
    "write_csv",
    "write_json",
    "write_jsonl",
    "write_run",
    "write_summary_csv",
    "write_table",
]
