"""Result records shared by the node-wise builder and the baseline builders."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from scholarpath.llm.accounting import BUILD_COMPONENTS, UsageRecord
from scholarpath.taxonomy.router import MLPRouter
from scholarpath.taxonomy.tree import Taxonomy


@dataclass
class NodeRouter:
    """Frozen router of an internal node (used to route new papers at zero LLM cost)."""

    node_id: str
    child_ids: List[str]
    child_vecs: np.ndarray
    router: MLPRouter
    threshold: float = 0.5

    def route(self, paper_vecs: np.ndarray) -> List[List[str]]:
        """Child ids per paper; an empty list means the residual leaf."""
        if not self.child_ids:
            return [[] for _ in range(paper_vecs.shape[0])]
        probs = self.router.predict_proba(paper_vecs, self.child_vecs)
        return [[self.child_ids[c] for c in sorted(s)] for s in self.router.assign(probs, self.threshold)]


@dataclass
class EpisodeRecord:
    """Summary of one node episode (Algorithm 1, lines 3-21)."""

    node_id: str
    depth: int
    n_v: int
    n_children: int = 0
    status: str = "open"
    certified: bool = False
    bound: Optional[float] = None
    delta_v: Optional[float] = None
    rounds: int = 0
    cycles: int = 0
    papers_read: Dict[str, int] = field(default_factory=dict)  # component -> #papers
    verified: int = 0  # |D_v| at the end
    abstention: float = 0.0  # a_v
    disagreement: float = 0.0  # d_v
    region_descriptors: List[List[float]] = field(default_factory=list)  # h_vg
    region_slopes: List[List[float]] = field(default_factory=list)  # observed (lambda_mis, lambda_err)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        """Realized ratio eta_v = (#distinct papers read) / n_v."""
        return self.verified / max(1, self.n_v)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ratio"] = self.ratio
        return d


@dataclass
class BuildResult:
    """Output of a taxonomy construction run."""

    taxonomy: Taxonomy
    method: str = "scholarpath"
    usage: Dict[str, UsageRecord] = field(default_factory=dict)  # component -> usage
    papers_read: Dict[str, int] = field(default_factory=dict)  # component -> #paper reads
    episodes: List[EpisodeRecord] = field(default_factory=list)
    runtime_s: float = 0.0
    routers: Dict[str, NodeRouter] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_usage(self) -> UsageRecord:
        out = UsageRecord()
        for name, rec in self.usage.items():
            if name != "judge":
                out.add(rec)
        return out

    @property
    def cost_usd(self) -> float:
        return self.total_usage.cost_usd

    @property
    def total_tokens(self) -> int:
        return self.total_usage.total_tokens

    def relative_cost(self, reference: Union["BuildResult", float]) -> float:
        """C_LLM / C_Full (Section 5.2): ratio of billed costs."""
        ref = reference.cost_usd if isinstance(reference, BuildResult) else float(reference)
        return self.cost_usd / ref if ref > 0 else float("nan")

    def cost_breakdown(self) -> Dict[str, float]:
        """Cost per component of Eq. (15) (other components reported as-is)."""
        out = {c: 0.0 for c in BUILD_COMPONENTS}
        for name, rec in self.usage.items():
            out[name] = out.get(name, 0.0) + rec.cost_usd
        return out

    def summary(self) -> Dict[str, Any]:
        eps = self.episodes
        return {
            "method": self.method,
            "taxonomy": self.taxonomy.stats(),
            "cost_usd": self.cost_usd,
            "total_tokens": self.total_tokens,
            "llm_calls": self.total_usage.calls,
            "cost_breakdown": self.cost_breakdown(),
            "papers_read": dict(self.papers_read),
            "runtime_s": self.runtime_s,
            "num_episodes": len(eps),
            "certified_nodes": sum(1 for e in eps if e.certified),
            "mean_ratio": float(np.mean([e.ratio for e in eps])) if eps else 0.0,
            "mean_abstention": float(np.mean([e.abstention for e in eps])) if eps else 0.0,
            "mean_disagreement": float(np.mean([e.disagreement for e in eps])) if eps else 0.0,
            "metadata": self.metadata,
        }

    def save(self, directory: Union[str, Path]) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.taxonomy.save(d / "taxonomy.json")
        (d / "summary.json").write_text(json.dumps(self.summary(), indent=2, default=str), encoding="utf-8")
        (d / "usage.json").write_text(
            json.dumps({k: v.to_dict() for k, v in self.usage.items()}, indent=2), encoding="utf-8"
        )
        with open(d / "episodes.jsonl", "w", encoding="utf-8") as f:
            for e in self.episodes:
                f.write(json.dumps(e.to_dict(), default=str) + "\n")
