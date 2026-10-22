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
- [Paper setting](#paper-setting)
- [Data](#data)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Programmatic use](#programmatic-use)

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

Python ≥ 3.9 is required.

## Paper setting

The paper uses DeepSeek-V4.0-Flash as the LLM for every method and SPECTER2
embeddings. `configs/paper.yaml` selects both, plus the LLM judges used for
evaluation. Set the API endpoints first:

```bash
pip install -e ".[llm,embeddings,eval]"
export SCHOLARPATH_LLM_API_KEY=...                            # LLM provider key
export SCHOLARPATH_LLM_BASE_URL=https://api.deepseek.com/v1   # any OpenAI-compatible endpoint
export SCHOLARPATH_JUDGE_API_KEY=...                          # judge endpoint (evaluation only)
export SCHOLARPATH_JUDGE_BASE_URL=...
```

Set `llm.model` in `configs/paper.yaml` to your provider's name for
DeepSeek-V4.0-Flash. Any hyper-parameter can be changed with
`--set section.field=value`, e.g. `--set retrieval.budget=4096`.

## Data

Download and prepare the datasets (details in [data/README.md](data/README.md)):

```bash
python scripts/download_data.py
python scripts/prepare_arxiv.py
python scripts/prepare_fulltext.py
python scripts/prepare_benchmarks.py
```

For a quick offline try-out, `python scripts/make_synthetic_data.py --out data/synthetic`
writes a small synthetic dataset; each script's `--help` shows the matching
offline command (mock LLM, no API keys).

## Reproducing the experiments

Run the commands below in order. Every script takes `--config
configs/paper.yaml`, writes its tables and figure data to `results/<script>/`,
and accepts `--resume` to reuse finished runs.

**1. Taxonomy construction (Section 5.2)**

```bash
python scripts/run_taxonomy_experiments.py --config configs/paper.yaml   # Figs 2-3, Tables 8-9
python scripts/run_taxonomy_ablation.py    --config configs/paper.yaml   # Table 6
python scripts/run_scalability.py          --config configs/paper.yaml   # Fig. 4
```

**2. Indexes and link predictor**

```bash
for corpus in arxiv_subfields taxobench chime; do
  python scripts/build_index.py --config configs/paper.yaml --corpus $corpus
done
python scripts/train_link_predictor.py --config configs/paper.yaml --corpus arxiv_subfields
```

**3. Held-out hypothesis benchmark (App. C.1)**

```bash
python scripts/build_heldout_benchmark.py --config configs/paper.yaml build --oversample 1.5 \
    --min-source-date <knowledge cutoff of the LLM>
# two annotators fill data/processed/heldout/annotation_sheet.csv, then:
python scripts/build_heldout_benchmark.py apply
python scripts/build_index.py          --config configs/paper.yaml --corpus heldout
python scripts/train_link_predictor.py --config configs/paper.yaml --corpus heldout
```

**4. Query answering (Section 5.3)**

```bash
python scripts/run_retrieval_eval.py     --config configs/paper.yaml   # Table 2, Fig. 5
python scripts/run_qa_eval.py            --config configs/paper.yaml   # Tables 1, 3, 4, Fig. 6
python scripts/run_agents.py             --config configs/paper.yaml \
    --agents paperqa2,openscholar,agenticscholar,codex \
    --openscholar-dir <OpenScholar checkout> --agenticscholar-url <endpoint>
python scripts/run_qa_eval.py            --config configs/paper.yaml \
    --answers-dir results/run_agents --eval-only                       # agent rows of Tables 3-4
python scripts/run_efficiency.py         --config configs/paper.yaml \
    --inputs results/run_qa_eval,results/run_agents                    # Table 5
python scripts/run_retrieval_ablation.py --config configs/paper.yaml   # Table 7
```

The agent baselines need their own tools: `pip install paper-qa` (PaperQA2),
a checkout of github.com/AkariAsai/OpenScholar, an AgenticScholar endpoint,
and the Codex CLI (`npm install -g @openai/codex@0.42.0`).

To ask your own question:

```bash
python scripts/ask.py --config configs/paper.yaml --corpus arxiv_subfields \
    "Which other methods could be applied to air-quality forecasting?"
```

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
