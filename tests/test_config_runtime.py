"""Config validation, runtime CLI errors, paper.yaml judges, missing-dependency hints, strict JSON output."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
from pathlib import Path

import pytest

from scholarpath.config import ConfigError, EmbeddingConfig, ScholarPathConfig, UnknownConfigField
from scholarpath.data.corpus import write_json
from scholarpath.embeddings import MissingDependencyError, build_embedder
from scholarpath.runtime import Runtime, add_common_args

ROOT = Path(__file__).resolve().parents[1]


def _has_embedding_extra() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def test_overrides_are_type_checked():
    cfg = ScholarPathConfig()
    cfg.override(["builder.router.learning_rate=1e-3", "retrieval.budget=8e3", "llm.cache_dir=null",
                  "evaluation.attribution_judge=off", "retrieval.granularity_weights.overview.chunk=0.4",
                  "evaluation.judge_models=a,b", "retrieval.use_link_prediction=no"])
    assert cfg.builder.router.learning_rate == pytest.approx(1e-3)  # YAML reads 1e-3 as a string
    assert cfg.retrieval.budget == 8000 and cfg.llm.cache_dir is None
    assert cfg.evaluation.attribution_judge == "off"  # YAML would turn it into False
    assert cfg.retrieval.granularity_weights["overview"]["chunk"] == 0.4
    assert cfg.evaluation.judge_models == ("a", "b") and cfg.retrieval.use_link_prediction is False
    for bad in ("builder.epsilon=0.1x", "retrieval.budget=abc", "llm.model=null", "builder.router=3",
                "llm.temperature=true", "no_equals_sign"):
        with pytest.raises(ConfigError):
            ScholarPathConfig().override([bad])
    with pytest.raises(UnknownConfigField, match="did you mean 'budget'"):
        ScholarPathConfig().override(["retrieval.budgte=4096"])
    with pytest.raises(UnknownConfigField):
        ScholarPathConfig().override(["retrieval.granularity_weights.overview.chnk=0.4"])
    with pytest.raises(KeyError):  # config files: unknown fields are KeyErrors too
        ScholarPathConfig.from_dict({"llm": {"modle": "x"}})
    with pytest.raises(ConfigError):
        ScholarPathConfig.from_dict({"builder": {"epsilon": "0.1x"}})


def test_config_files_keep_their_values_and_types():
    def types(x):
        if isinstance(x, dict):
            return {k: types(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [types(v) for v in x]
        return type(x).__name__

    loaded = ScholarPathConfig.load(ROOT / "configs" / "default.yaml").to_dict()
    assert loaded == ScholarPathConfig().to_dict() and types(loaded) == types(ScholarPathConfig().to_dict())


def test_paper_config_enables_real_judges_and_the_official_citation_judge():
    paper = ScholarPathConfig.load(ROOT / "configs" / "paper.yaml")
    assert (paper.llm.provider, paper.embedding.backend) == ("openai", "specter2")
    assert paper.evaluation.judge_provider == "openai" and paper.evaluation.attribution_judge == "nli"
    assert paper.mock_judge_warning() is None
    assert ScholarPathConfig().mock_judge_warning() is None  # offline: mock LLM + mock judges
    mixed = ScholarPathConfig.load(ROOT / "configs" / "paper.yaml").override(["evaluation.judge_provider=mock"])
    assert "not comparable" in mixed.mock_judge_warning()


def _args(argv):
    parser = argparse.ArgumentParser(prog="x.py")
    add_common_args(parser)
    return parser.parse_args(argv)


def test_runtime_reports_bad_overrides_as_cli_errors(capsys, caplog):
    with pytest.raises(SystemExit) as exc:
        Runtime.from_args(_args(["--set", "retrieval.budget=abc"]))
    assert exc.value.code == 2 and "retrieval.budget expects an integer" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        Runtime.from_args(_args(["--config", "no/such/config.yaml"]))
    assert exc.value.code == 2 and "config file not found" in capsys.readouterr().err
    if not _has_embedding_extra():  # SPECTER2 without torch -> install hint, not a traceback
        with pytest.raises(SystemExit) as exc:
            Runtime.from_args(_args(["--embedder", "specter2"]))
        assert exc.value.code == 1 and 'pip install -e ".[embeddings]"' in capsys.readouterr().err
    with caplog.at_level(logging.WARNING, logger="scholarpath"):
        rt = Runtime.from_args(_args(["--set", "evaluation.judge_provider=mock", "--set", "llm.provider=mock"]))
    assert rt.config.llm.provider == "mock" and not [r for r in caplog.records if "not comparable" in r.getMessage()]


def test_missing_embedding_dependencies_give_an_install_hint():
    if _has_embedding_extra():  # pragma: no cover - environments with the embeddings extra
        pytest.skip("torch / transformers installed")
    with pytest.raises(MissingDependencyError, match=r'pip install -e "\.\[embeddings\]"'):
        build_embedder(EmbeddingConfig(backend="specter2"))


def test_write_json_never_emits_bare_nan(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"kappa": float("nan"), "rates": [1.0, math.inf], "n": 0})

    def strict(token):
        raise ValueError(f"non-standard JSON constant {token}")

    assert json.loads(path.read_text(), parse_constant=strict) == {"kappa": None, "rates": [1.0, None], "n": 0}


def test_malformed_config_files_are_cli_errors(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        ScholarPathConfig.load(bad)
    with pytest.raises(SystemExit) as exc:
        Runtime.from_args(_args(["--config", str(bad)]))
    assert exc.value.code == 2 and "not valid YAML" in capsys.readouterr().err
