# Third-party notices

The license of ScholarPath's own code is chosen by its authors (see `README.md`). Parts of it
reproduce or re-implement material from the projects below, which keeps its
own license. Datasets are **not** redistributed: `scripts/download_data.py`
fetches them from their sources, and each dataset's terms are listed in
`data/README.md` ("Dataset licenses"). The files under `tests/fixtures/` are
small hand-made samples in the released formats (made-up titles, abstracts and
hierarchies) used only by the tests.

Upstream licenses below were checked against checkouts at the pinned commits
of `scholarpath/data/download.py`.

## ScholarQABench — MIT License

github.com/AkariAsai/ScholarQABench (commit `95e6fc5`).

Reproduced in `src/scholarpath/prompts/evaluation.py` and re-implemented in
`src/scholarpath/evaluation/qa_metrics.py` (see their module docstrings):

- the ScholarQA-CS rubric prompts and scoring of `scripts/scholarqacs_rubric.py`
  and `scripts/rubric_eval.py`;
- the answer format of `scripts/convert_answer_scholar_cs.py`;
- the AttrScore prompt and the citation precision / recall procedure of
  `scripts/citation_correctness_eval.py`;
- the Prometheus absolute-grading template of `scripts/prometheus_eval.py` and
  the three rubrics of `rubrics/prometheus_rubrics_v8.json`.

```
MIT License

Copyright (c) 2024 Akari Asai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## TaxoBench — Apache License 2.0 (code); CC BY-NC 4.0 (annotations)

github.com/KongLongGeFDU/TaxoBench (commit `49a1869`), Copyright 2026
TaxoBench Authors.

`src/scholarpath/evaluation/organization_metrics.py` re-implements the
public scorer (`taxobench-score` v0.2.0, `taxobench/metrics/{alignment,
clustering,sem_path,tree}.py`) and the paper-era scripts
(`metric/get_clustering_result.py`, `get_clustering_metric.py`,
`sem_path.py`). The differences from the original code are documented on each
function. The code is licensed under the Apache License 2.0; the full text is
in `LICENSES/Apache-2.0.txt`.

The TaxoBench **annotations** (expert-tree transcriptions and
paper-to-category mappings) are released for **non-commercial research use
under CC BY-NC 4.0** (`DATASET_LICENSE.md` of the TaxoBench repository). They
are downloaded, not redistributed, and the processed files that
`scripts/prepare_benchmarks.py` derives from them (`data/processed/taxobench/`)
fall under the same terms.

## CHIME — Apache License 2.0

github.com/allenai/chime (commit `35d852a`).

`src/scholarpath/data/chime.py` re-implements CHIME's hierarchy parser
(`utils.get_hierarchy`, `parse_hierachy.py`) in `parse_generated_hierarchy`,
and its `clean_claim` clean-up. The differences are documented in the module
docstring (for example, claim ranges such as "Claim 18-22" are expanded).
License text: `LICENSES/Apache-2.0.txt`.

## ResearchQA — no license file

github.com/realliyifei/ResearchQA (commit `747a9a1`).

The rubric-coverage judge prompt of `compute_coverage.py` (`build_prompt`) is
reproduced in `src/scholarpath/prompts/evaluation.py`, so that Rub scores are
comparable with the official script. allenai/olmo-eval and rlresearch/dr-tulu
also copy this prompt. The repository has no license file at this commit, so
the terms for redistributing the prompt have not been granted explicitly. If
you redistribute ScholarPath, check them with the ResearchQA authors.
`scripts/run_official_eval.py` runs the official script from your own
checkout instead.

## Baseline prompts

`src/scholarpath/prompts/baselines.py` follows the prompt wording of the
original implementations of RankGPT (github.com/sunnweiwei/RankGPT), RAPTOR
(github.com/parthsarthi03/raptor) and LightRAG (github.com/HKUDS/LightRAG). The
LightRAG record format is replaced by JSON. These prompts are adapted, not
copied, and their wording remains under the upstream projects' licenses.

## External tools and models (not included)

PaperQA2 (`paper-qa`), OpenScholar, the Codex CLI, SPECTER2
(`allenai/specter2*`), AttrScore-Flan-T5-XL (`osunlp/attrscore-flan-t5-xl`) and
Prometheus-2 are installed or downloaded by the user and used through
adapters. None of their code or weights is included here.
