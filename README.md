# ScholarPath

Implementation of **ScholarPath**, the indexing and retrieval framework described in

> *Empowering Automated Research Agent At Scale: the Role of Evidence Query and
> Hypothesis Query* (SIGMOD '27 submission).

Automated research agents answer two kinds of questions over a scholarly
repository: **evidence queries** ("How have GNNs been used in drug
discovery?"), which synthesize established findings, and **hypothesis
queries** ("Which other methods could be applied to air-quality
forecasting?"), which propose unstudied directions supported by indirect
evidence. ScholarPath organizes a repository into a **multi-granularity
semantic index**: problem and method taxonomies linked to paper abstracts
and full-text section chunks. It contributes two components:

1. **Node-wise confidence-bounded taxonomy builder** (Section 3, Algorithm 1,
   `scholarpath.taxonomy.builder`). It lets the LLM read only the papers
   predicted to reduce each node's organization loss
   (quality-targeted Bayesian acquisition). A lightweight router places
   the rest. A hypergeometric audit then certifies, per node, that the
   loss is at most ε with confidence 1 − δ, and audit-guided repair
   (DISCOVER / CORRECT) fixes the errors the audit exposes. Certified
   nodes absorb new papers at zero LLM cost (dynamic update, Sec. 3.4).
2. **Budgeted support-path retrieval** (Section 4, Algorithm 2,
   `scholarpath.retrieval.pipeline`). It selects direct support paths
   (evidence) and complementary path sets (hypotheses) under a token
   budget B. Selection maximizes redundancy-aware aspect coverage (Eq. 5–6)
   with granularity-adaptive path expansion and the premise-bottleneck
   score (Eq. 9). Hierarchy-bounded subtree pruning with a cross-taxonomy
   link-prediction bound (Eq. 7, 8, 12) selects exactly the same units as
   an exhaustive greedy scan (Prop. 7). One LLM call composes the cited
   answer.

Everything in the paper's evaluation is included: dataset preparation,
the selection-based taxonomy baselines, the retrieval / RAG / agent
baselines, all metrics and LLM judges, and a script for every table and
figure (Tables 1–9, Figs 2–6). The whole pipeline also runs **offline** with
a deterministic mock LLM and a hashing embedder, so it can be tested without
API keys or downloads.

## Contents

- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start (offline)](#quick-start-offline)
- [Paper setting: LLM backbone and embeddings](#paper-setting-llm-backbone-and-embeddings)
- [Data](#data)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Programmatic use](#programmatic-use)
- [Implementation notes](#implementation-notes)

## Repository structure

```
.
├── configs/
│   ├── default.yaml          # all hyper-parameters (paper defaults), offline backends
│   └── paper.yaml            # DeepSeek-V4.0-Flash + SPECTER2 + real judges (the paper's setting)
├── docs/ARCHITECTURE.md      # paper -> code map, conventions, module contracts
├── data/README.md            # datasets: download, preprocessing, layout, coverage
├── scripts/                  # one script per pipeline stage / table / figure (scripts/README.md)
├── src/scholarpath/
│   ├── config.py, types.py, text.py, embeddings.py, runtime.py
│   ├── llm/                  # OpenAI-compatible client, offline MockLLM, cost meter (Eq. 15)
│   ├── prompts/              # every LLM prompt (taxonomy, retrieval, answer, baselines, judges, benchmark)
│   ├── taxonomy/             # Sec. 3: tree, router, regions, response model, allocation,
│   │                         #   certification, repair, builder (Alg. 1), dynamic update,
│   │                         #   selection baselines (FULL / node-wise / one-off), selectors
│   ├── index/                # Sec. 2.1: multi-granularity semantic index
│   ├── retrieval/            # Sec. 4: query understanding, grounding, units, scoring, objective,
│   │                         #   bound-guided search (Alg. 2), composition, link prediction (+ training)
│   ├── baselines/            # BM25, SPECTER, Dense+reranker, RAPTOR, LightRAG, keyword search,
│   │   └── agents/           #   budgeted QA; PaperQA2 / OpenScholar / AgenticScholar / Codex adapters
│   ├── data/                 # arXiv, full text, alignment, ResearchQA, ScholarQABench, TaxoBench,
│   │                         #   CHIME, held-out hypothesis benchmark (App. C.1), synthetic data
│   ├── evaluation/           # soft node/edge F1, Recall@B/NDCG@B, Rub/Cite/Corr, ARI/Sem, Cat/Assn,
│   │                         #   Disc + LLM judges, efficiency, reporting, official-script wrappers
│   └── experiments/          # experiment runners behind the scripts
```

Generated files (all git-ignored): `data/raw/`, `data/processed/` (datasets),
`artifacts/index/<corpus>/` (semantic indexes, with the trained link
predictor in `link_predictor/`), `results/<script>/` (experiment outputs),
`results/taxonomy_cache/` (taxonomy builds shared by the construction
scripts) and `cache/{llm,judge,embeddings}/` (response and embedding caches).

`docs/ARCHITECTURE.md` maps every section, algorithm line and equation of
the paper to the code.

## Installation

```bash
pip install -e .                      # core: numpy, scipy, scikit-learn, pyyaml, tqdm
pip install -e ".[llm]"               # OpenAI-compatible client + tiktoken token counts
pip install -e ".[embeddings]"        # SPECTER2 (torch, transformers, adapters), AttrScore NLI judge
pip install -e ".[data]"              # HF downloads, parquet reading (pandas, pyarrow), PDF text extraction
pip install -e ".[eval]"              # nltk (statement splitting of the ScholarQABench citation eval)
pip install -e ".[baselines]"         # umap-learn (RAPTOR), networkx (LightRAG graph export)
pip install -e ".[agents]"            # paper-qa (PaperQA2; Python >= 3.11)
```

Python ≥ 3.9 is required. The Codex CLI and an OpenScholar checkout are
external tools; see [Unbudgeted agents](#unbudgeted-agents).

## Quick start (offline)

```bash
bash scripts/smoke_test.sh        # ~3 minutes on a laptop; outputs under results/smoke/
```

The smoke test runs the pipeline on a small synthetic repository with the
mock LLM and the hashing embedder:
1. Synthetic data.
2. Dual taxonomies with Algorithm 1.
3. Construction experiments, ablation, scalability and dynamic update.
4. Index building, link-predictor training, and the held-out benchmark
   construction (`build_heldout_benchmark.py build` / `apply --auto-accept`).
5. Retrieval accuracy, end-to-end QA, the agent path (offline simulated
   agent), efficiency and the retrieval ablation.
6. One interactive question.

It writes Tables 1–9 and the data of Figs 2–6 in the paper's format. It does
not exercise the downloads, the real LLM / SPECTER2 backends, the external
agents or the official evaluation scripts. **The mock backends only exercise
the code paths: their numbers are not the paper's results.**

## Paper setting: LLM backbone and embeddings

The paper uses DeepSeek-V4.0-Flash as the backbone LLM for every component
and baseline ($0.14 / 1M input tokens, $0.28 / 1M output tokens). It embeds
papers, concepts and chunks with SPECTER2 (768-d). `configs/paper.yaml`
selects both, plus the real judges of Section 5.3:

```bash
pip install -e ".[llm,embeddings,eval]"
export SCHOLARPATH_LLM_API_KEY=...                            # your provider's key
export SCHOLARPATH_LLM_BASE_URL=https://api.deepseek.com/v1   # or any OpenAI-compatible endpoint
export SCHOLARPATH_JUDGE_API_KEY=...                          # judge endpoint (evaluation scripts only)
export SCHOLARPATH_JUDGE_BASE_URL=...                         # e.g. https://openrouter.ai/api/v1
python scripts/<script>.py --config configs/paper.yaml ...
```

- **Model name.** Set `llm.model` to your provider's identifier for
  DeepSeek-V4.0-Flash.
- **Response cache.** Every LLM response is cached under `cache/llm`. Cached
  calls are still billed in the cost meter, so reported costs do not depend
  on cache state.
- **Clean timings.** For the latencies of Table 5, run without the cache:
  `--set llm.cache_dir=null`.
- **Judges.** The LLM judges of Section 5.3 (GPT-6.0-Sol, Opus-5.5,
  Grok-4.7; the official ResearchQA / ScholarQA-CS judge models;
  Prometheus-2 7B) are set in the `evaluation:` section and use the
  `SCHOLARPATH_JUDGE_BASE_URL` / `SCHOLARPATH_JUDGE_API_KEY` endpoint.
  `configs/paper.yaml` enables them (`judge_provider: openai`) together with
  the AttrScore-Flan-T5-XL NLI citation judge of the official ScholarQABench
  script (`attribution_judge: nli`). `configs/default.yaml` uses offline mock
  judges. Every script logs a warning when a real backbone LLM is combined
  with mock judges, whose scores are not comparable with the paper. Judge
  cost is metered separately and never counts as system cost.

Overrides of the paper defaults (ε = 0.10, 1 − δ = 0.95, η_A = 0.02, β = 16,
κ = 3, min node size 16, max depth 5, B = 8,192, …) go through
`--set section.field=value`, e.g. `--set retrieval.budget=4096`. Values are
checked against the field types, so a typo (`--set builder.epsilon=0.1x`,
`--set retrieval.budgte=4096`) stops the script with a one-line error.

## Data

See **[data/README.md](data/README.md)**:

```bash
python scripts/download_data.py        # TaxoBench, ScholarQABench, CHIME, ResearchQA, arXiv snapshot (Kaggle)
python scripts/prepare_arxiv.py        # 8 arXiv sets x 2,000 papers + arXiv-Subfields (148,400 papers)
python scripts/prepare_fulltext.py     # section chunks (local PDFs/LaTeX, arXiv GCS bucket, arXiv, S2ORC v2)
python scripts/prepare_benchmarks.py   # benchmarks -> data/processed/, corpus alignment report (App. C.2)
```

For offline runs, `python scripts/make_synthetic_data.py --out data/synthetic`
writes a synthetic repository in the same processed layout. The experiment
scripts that read the processed layout take `--data-dir data/synthetic`.
The others take a corpus file (`build_taxonomy`, `run_scalability`,
`run_dynamic_update`: `--corpus data/synthetic/arxiv_sets/synthetic.jsonl.gz`)
or an index (`train_link_predictor`, `ask`: `--index-dir`, `--corpus`); see
their `--help`. The data-preparation scripts have their own flags
(`data/README.md`).

## Reproducing the experiments

Every experiment script (all of `scripts/` except the five data-preparation
scripts) accepts `--config`, `--set`, `--output-dir` and `--help`. Their
docstrings give the paper-setting command and a quick offline command.
Results go to `--output-dir` (default `results/<script>/`) as CSV/JSONL rows,
Markdown tables shaped like the paper's, and plot data. `build_index.py`
writes indexes to `artifacts/index/<corpus>/` (`--index-dir`), and
`build_heldout_benchmark.py` writes the held-out benchmark to
`data/processed/heldout/` (`--out`). Builds and per-query records are cached
under a configuration fingerprint, so `--resume` reuses only results
produced under the same settings. Without `--resume`, the QA scripts
(`run_retrieval_eval`, `run_qa_eval`, `run_agents`, `run_retrieval_ablation`)
refuse to replace cached records of another configuration unless `--force`
is given.

### Taxonomy construction (Section 5.2)

| Paper | Command |
|---|---|
| Fig. 2 (soft node/edge F1 vs η), Fig. 3 (relative cost), Tables 8–9 (one-off vs node-wise) | `python scripts/run_taxonomy_experiments.py --config configs/paper.yaml --gold-max-depth 4 --resume` |
| Table 6 (construction ablation) | `python scripts/run_taxonomy_ablation.py --config configs/paper.yaml --resume` |
| Fig. 4 (scalability 2K → 148.4K papers) | `python scripts/run_scalability.py --config configs/paper.yaml --resume` |
| Sec. 3.4 / App. B.3 (dynamic update) | `python scripts/run_dynamic_update.py --config configs/paper.yaml --corpus data/processed/arxiv_sets/cs.DB.jsonl.gz --cutoff 2024-01-01 --compare-scratch` |
| One taxonomy with any method | `python scripts/build_taxonomy.py --config configs/paper.yaml --corpus data/processed/arxiv_sets/cs.DB.jsonl.gz --facet dual --method scholarpath` |

On the arXiv sets, every method builds both the problem and the method
taxonomy and is evaluated against FULL (η = 1.0). On TaxoBench, every
method, FULL included, builds one topic taxonomy over the pooled 3,044
papers and is evaluated against the merged expert taxonomy, cut at 4 levels
below the common root as in App. C.2. Three flags control this cut:
`--gold-max-depth` in `run_taxonomy_experiments.py` (default 4), `--reference-max-depth` in
`build_taxonomy.py` (pass 4 for a TaxoBench reference), and
`--taxobench-max-depth` in `prepare_benchmarks.py`, which truncates the stored
gold file (default: keep all levels, so that the experiments can choose).

### Query answering (Section 5.3)

```bash
# 1. Indexes (dual ScholarPath taxonomies + multi-granularity index) and link predictors
python scripts/build_index.py --config configs/paper.yaml --corpus arxiv_subfields   # ResearchQA, ScholarQABench
python scripts/build_index.py --config configs/paper.yaml --corpus taxobench
python scripts/build_index.py --config configs/paper.yaml --corpus chime
python scripts/train_link_predictor.py --config configs/paper.yaml --corpus arxiv_subfields
# 2. Held-out hypothesis benchmark (App. C.1): build on the complete corpus, annotate, apply, index the reduced corpus
python scripts/build_heldout_benchmark.py --config configs/paper.yaml build --oversample 1.5 --min-source-date <LLM cutoff>
#    ... two annotators fill data/processed/heldout/annotation_sheet.csv ...
python scripts/build_heldout_benchmark.py apply
python scripts/build_index.py --config configs/paper.yaml --corpus heldout
python scripts/train_link_predictor.py --config configs/paper.yaml --corpus heldout
```

| Paper | Command |
|---|---|
| Table 2, Fig. 5 (Recall@B / NDCG@B; B = 2K…16K) | `python scripts/run_retrieval_eval.py --config configs/paper.yaml --resume` |
| Table 1 (benchmark statistics), Tables 3–4, Fig. 6 (end-to-end QA, budgeted methods) | `python scripts/run_qa_eval.py --config configs/paper.yaml --resume` |
| Tables 3–5 (unbudgeted agents) | `python scripts/run_agents.py --config configs/paper.yaml --agents paperqa2,openscholar,agenticscholar,codex --openscholar-dir <OpenScholar checkout> --agenticscholar-url <endpoint> --resume`, then `python scripts/run_qa_eval.py --config configs/paper.yaml --answers-dir results/run_agents --eval-only` |
| Table 3 with the official evaluation scripts (App. C.3) | `python scripts/run_official_eval.py --config configs/paper.yaml --answers-dir results/run_qa_eval/answers,results/run_agents/answers --researchqa-dir <ResearchQA checkout> --scholarqabench-dir <ScholarQABench checkout>` |
| Table 5 (per-query efficiency) and Sec. 5.3.3 operator statistics | `python scripts/run_efficiency.py --config configs/paper.yaml --inputs results/run_qa_eval,results/run_agents --operator-stats` |
| Table 7 (retrieval ablation on hypothesis queries) | `python scripts/run_retrieval_ablation.py --config configs/paper.yaml --resume` |
| Ask your own question | `python scripts/ask.py --config configs/paper.yaml --corpus arxiv_subfields "Which other methods could be applied to air-quality forecasting?"` |

Every evaluation command needs `--config configs/paper.yaml`. It selects the
real judges, and `run_efficiency.py --operator-stats` re-runs retrieval with
the configured embedder, which must be the one the indexes were built with.
With `--answers-dir`, `run_qa_eval.py` evaluates the agents' answers next to
the cached budgeted answers and regenerates nothing. A failed or timed-out
query scores 0 on every metric of Tables 2–4. The "Failed" column of Tables
3–4 counts these queries and appears only when a system has failures.

**Table 3 metrics.** `run_qa_eval.py` computes Rub, Corr and Cite with
in-process re-implementations of the official ResearchQA and ScholarQABench
protocols (`scholarpath.evaluation.qa_metrics`: same prompts, batching,
retries and aggregation), so that they run with any judge backend.
`run_official_eval.py` scores the same cached answers with the official
scripts themselves (`compute_coverage.py`, `rubric_eval.py`,
`citation_correctness_eval.py`, `prometheus_eval.py`), as App. C.3
describes. The checkouts come from `evaluation.researchqa_dir` /
`evaluation.scholarqabench_dir`, the flags, or `$SCHOLARPATH_RESEARCHQA_DIR` /
`$SCHOLARPATH_SCHOLARQABENCH_DIR`. The scripts run in an environment with
their requirements (`$SCHOLARPATH_EVAL_PYTHON`); see the script's docstring.

#### Unbudgeted agents

The adapters in `scholarpath/baselines/agents/` export the corpus as JSONL
shards indexed by paper ID and primary category (App. C.3). Every system
receives the same JSON-schema task prompt, which asks for ranked candidate
records with cited paper IDs and an established / hypothesis status.

- **PaperQA2:** `pip install paper-qa` (Python ≥ 3.11). The adapter points
  its LLMs at the same OpenAI-compatible endpoint.
- **OpenScholar:** clone github.com/AkariAsai/OpenScholar and pass its path
  (`--openscholar-dir`). The adapter runs `run.py` with the exported corpus
  as its datastore.
- **AgenticScholar:** provide the implementation as a Python callable
  (`--agenticscholar-fn module:function`) or an HTTP endpoint
  (`--agenticscholar-url`). It receives the same constructed taxonomies as
  ScholarPath.
- **Codex:** `npm install -g @openai/codex@0.42.0`, the version used in the
  paper (App. C.3). The adapter runs `codex exec` with a `model_providers`
  entry for the backbone. Later Codex releases require an endpoint that
  implements the Responses API (see `--codex-wire-api`). External web access is
  disabled with a Linux `unshare -n` network namespace and a local
  forwarding proxy to the LLM endpoint.
- An agent whose checkout, implementation or tool is missing is reported as
  unavailable and skipped. Check the "Unavailable agents" line and
  `summary.json` before building Tables 3–5.
- **Offline smoke tests:** `simulated` is a deterministic multi-step agent.
  It is not a paper baseline.

## Programmatic use

```python
from scholarpath.config import ScholarPathConfig
from scholarpath.runtime import Runtime
from scholarpath.data.synthetic import make_synthetic_benchmark
from scholarpath.taxonomy.builder import build_dual_taxonomies
from scholarpath.index import SemanticIndex
from scholarpath.retrieval.pipeline import ScholarPathRetriever
from scholarpath.types import Query, EVIDENCE, HYPOTHESIS_DISCOVERY

rt = Runtime.from_config(ScholarPathConfig.load("configs/default.yaml"))  # configs/paper.yaml for the paper setting
corpus = make_synthetic_benchmark(n_papers=240).corpus                    # or Corpus.load("...jsonl.gz")

# Section 3 / Algorithm 1: confidence-bounded dual taxonomies (BuildResult: taxonomy, cost, episodes, routers)
problem, method = build_dual_taxonomies(rt.llm, rt.embedder, rt.config.builder, corpus)
print(problem.taxonomy.pretty(max_depth=1), problem.cost_usd)

# Section 2.1: multi-granularity semantic index
index = SemanticIndex.build(problem.taxonomy, method.taxonomy, corpus.papers, rt.embedder, rt.config.index)

# Section 4 / Algorithm 2: budgeted support-path retrieval + a single composition call
retriever = ScholarPathRetriever(index, rt.llm, rt.embedder, rt.config.retrieval)
answer = retriever.answer(Query("q", "Which other methods could be applied to contact tracing?",
                                kind=HYPOTHESIS_DISCOVERY), budget=8192)
for item in answer.items:
    print(item.status, item.candidate, item.citations)        # 'hypothesis' items carry a candidate
print(rt.meter.snapshot())                                    # LLM usage per cost component
```

## Implementation notes

The paper leaves some details open. The main choices, all documented in the
module docstrings:

- **Error budget.** δ is split across nodes on the fly so that Σ_v δ_v ≤ δ
  holds by construction (`builder.delta_allocation`, default `spend`).
- **Certification audit.** Audits draw uniform batches without replacement.
  The bound is checked after every draw with the step levels
  α_{v,j,a} = δ_v / [j(j+1)a(a+1)].
- **Router.** A two-layer MLP over the concatenated paper and
  child-definition embeddings, plus their element-wise product. It gets at
  least `router.min_steps` AdamW updates, so small label sets are fit.
- **Transferred prior** (Eq. 14). Bayesian linear regression from region
  descriptors to the observed response slopes, updated after every
  episode.
- **Aspects in retrieval.** Aspects are instantiated over their targets:
  overview aspects over junction sub-concepts, empirical and mechanism
  aspects over the findings in J(u_q, m_q), and hypothesis aspects over
  candidates. Without this, a max-coverage objective would select a
  single paper.
- **Bridges.** The bridge of a complementary set is the lowest common
  ancestor of the baseline method m′ and the candidate. Its discount is
  φ = decay^(d−1) for taxonomic distance d.
- **Temporal link prediction.** The historical index restricts the final
  taxonomies' extents to papers dated by the cutoff t. Candidate
  application labels in (t, t+Δ] are verified by the LLM.
- **ResearchQA "Cite".** The official repository has no separate citation
  metric. We report citation F1 of the cited papers against the gold
  survey citations, with unaligned citations kept in the denominator. Its
  coverage on Citation-type rubric items is reported as well.
- **Failed queries.** The official scripts only score the answers they
  receive. A query on which a system failed or timed out therefore counts as
  0 in Tables 2–4 (`run_retrieval_eval.py`, `run_qa_eval.py`,
  `run_official_eval.py`), so a system
  cannot gain from failing on hard queries. `qa_metrics.evaluate_*(...,
  missing="zero")` does the same, and the default `"skip"` keeps the
  official behaviour.
- **ResearchQA queries** search the whole arXiv-Subfields corpus. The survey
  date of a query is kept in its metadata but is not used as a retrieval
  cutoff, since the paper describes none.
