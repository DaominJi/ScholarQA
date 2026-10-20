"""Shared runtime setup for scripts: config, LLM, embedder, cost meter, logging.

Every script under ``scripts/`` uses::

    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    ...
    args = parser.parse_args()
    rt = Runtime.from_args(args)
    rt.llm, rt.embedder, rt.config, rt.meter, rt.output_dir

Common flags:

* ``--config configs/default.yaml``           -- base configuration
* ``--set llm.provider=openai --set retrieval.budget=4096`` -- overrides
* ``--llm mock|openai`` / ``--embedder hashing|specter2`` -- shortcuts
* ``--output-dir results/...`` and ``--seed``
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from scholarpath.config import ScholarPathConfig
from scholarpath.embeddings import Embedder, build_embedder
from scholarpath.llm import CostMeter, LLMClient, build_llm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

log = logging.getLogger("scholarpath")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def add_common_args(parser: argparse.ArgumentParser, default_output: Optional[str] = None) -> None:
    g = parser.add_argument_group("runtime")
    g.add_argument("--config", type=Path, default=DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None,
                   help="YAML/JSON config (default: configs/default.yaml).")
    g.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="SECTION.FIELD=VALUE", help="Override a config field (repeatable).")
    g.add_argument("--llm", choices=["mock", "openai"], default=None,
                   help="Shortcut for --set llm.provider=...")
    g.add_argument("--model", default=None, help="Shortcut for --set llm.model=...")
    g.add_argument("--embedder", choices=["hashing", "specter2", "sentence-transformers"], default=None,
                   help="Shortcut for --set embedding.backend=...")
    g.add_argument("--output-dir", type=Path,
                   default=Path(default_output) if default_output else None,
                   help="Where results are written.")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--verbose", "-v", action="store_true")


@dataclass
class Runtime:
    config: ScholarPathConfig
    llm: LLMClient
    embedder: Embedder
    meter: CostMeter
    output_dir: Optional[Path] = None
    started: float = field(default_factory=time.time)

    @classmethod
    def from_config(
        cls,
        config: Optional[ScholarPathConfig] = None,
        output_dir: Optional[Path] = None,
    ) -> "Runtime":
        config = config or ScholarPathConfig()
        meter = CostMeter(config.llm.price_input_per_m, config.llm.price_output_per_m)
        llm = build_llm(config.llm, meter)
        embedder = build_embedder(config.embedding)
        seed_everything(config.seed)
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        return cls(config=config, llm=llm, embedder=embedder, meter=meter, output_dir=output_dir)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Runtime":
        setup_logging(getattr(args, "verbose", False))
        config = ScholarPathConfig.load(getattr(args, "config", None))
        overrides = list(getattr(args, "overrides", []) or [])
        if getattr(args, "llm", None):
            overrides.append(f"llm.provider={args.llm}")
        if getattr(args, "model", None):
            overrides.append(f"llm.model={args.model}")
        if getattr(args, "embedder", None):
            overrides.append(f"embedding.backend={args.embedder}")
        if getattr(args, "seed", None) is not None:
            overrides.append(f"seed={args.seed}")
        config.override(overrides)
        rt = cls.from_config(config, getattr(args, "output_dir", None))
        if rt.output_dir is not None:
            config.save(rt.output_dir / "config.yaml")
        log.info("LLM backend: %s (%s); embedder: %s", config.llm.provider, config.llm.model, config.embedding.backend)
        return rt

    def write_json(self, name: str, obj: Any) -> Path:
        assert self.output_dir is not None, "Runtime has no output_dir"
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return path

    def cost_summary(self) -> Dict[str, Any]:
        return {"components": self.meter.snapshot(), "total": self.meter.total().to_dict()}
