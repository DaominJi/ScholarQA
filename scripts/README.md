# Scripts

Every script adds `src/` to `sys.path`, so it runs without installation. All
share the runtime flags of `scholarpath.runtime.add_common_args`:

| Flag | Meaning |
|---|---|
| `--config configs/paper.yaml` | paper setting (DeepSeek-V4.0-Flash + SPECTER2); default `configs/default.yaml` is offline (mock LLM + hashing embedder) |
| `--set section.field=value` | override any config field (repeatable), e.g. `--set retrieval.budget=4096` |
| `--llm mock\|openai`, `--model`, `--embedder` | shortcuts for the LLM / embedding backends |
| `--output-dir` | where results go (default `results/<script>/`) |
| `--seed`, `--verbose` | seeding and logging |

Each script's module docstring (`--help`) states what it reproduces, the
command for the paper setting, and a quick offline command on the synthetic
layout (`python scripts/make_synthetic_data.py --out data/synthetic`).
`bash scripts/smoke_test.sh` chains the offline commands end to end.

## Data preparation (see `data/README.md`)

| Script | Purpose |
|---|---|
| `download_data.py` | download TaxoBench, ScholarQABench, CHIME (+ expert annotations), ResearchQA, the Kaggle arXiv snapshot, optional S2ORC v2 |
| `prepare_arxiv.py` | arXiv sets (8 primary categories × 2,000 papers) and arXiv-Subfields (148,400 papers) |
| `prepare_fulltext.py` | full text → section chunks (local PDFs/LaTeX, arXiv GCS bucket, arXiv, S2ORC v2) |
| `prepare_benchmarks.py` | ResearchQA, ScholarQA-CS/-Multi, TaxoBench, CHIME → `data/processed/`, corpus alignment report (App. C.2) |
| `make_synthetic_data.py` | synthetic repository in the processed layout (offline runs and tests) |

## Taxonomy construction (Section 3, Section 5.2)

| Script | Reproduces |
|---|---|
| `build_taxonomy.py` | one taxonomy (or the dual problem/method pair) with ScholarPath or a baseline (FULL, Random, Coverage, K-means, Diversity; node-wise or one-off), with an optional evaluation against a reference |
| `run_taxonomy_experiments.py` | Fig. 2 (soft node/edge F1 vs η ∈ {0.05,…,0.5}), Fig. 3 (relative cost C_LLM/C_Full), Tables 8–9 (one-off vs node-wise at η = 0.2) on the arXiv sets (vs FULL) and TaxoBench (vs gold) |
| `run_taxonomy_ablation.py` | Table 6 (w/o Bayesian acquisition, snapshot certification, transferred prior, DISCOVER, CORRECT, targeted repair) |
| `run_scalability.py` | Fig. 4 (selected ratio and total cost over 2K…148.4K papers; C_Full estimated beyond `--max-full-size`) |
| `run_dynamic_update.py` | Sec. 3.4 / App. B.3: streaming arrivals into certified nodes |

## Query answering (Section 4, Section 5.3)

| Script | Reproduces |
|---|---|
| `build_index.py` | dual taxonomies + multi-granularity semantic index for a corpus (`arxiv_subfields`, `taxobench`, `chime`, `heldout`) |
| `train_link_predictor.py` | temporal cross-taxonomy link predictor (Sec. 4.3, App. B.5) for an index |
| `build_heldout_benchmark.py` | held-out hypothesis benchmark (App. C.1): `build` (targets, corpus reduction, query generation, leakage screen, annotation sheet) and `apply` (annotator decisions) |
| `run_retrieval_eval.py` | Table 2 and Fig. 5 (Recall@B / NDCG@B for ScholarPath, BM25, SPECTER, Dense + reranker, RAPTOR, LightRAG; B ∈ {2K, 4K, 6K, 8K, 16K}) |
| `run_qa_eval.py` | Tables 3–4 and Fig. 6 (budgeted end-to-end QA: Answer LLM w/o retrieval, Keyword search, Dense retrieval, LightRAG, ScholarPath; evaluation of saved agent answers with `--answers-dir`) |
| `run_agents.py` | unbudgeted agents (PaperQA2, OpenScholar, AgenticScholar, Codex; `simulated` for offline tests) |
| `run_efficiency.py` | Table 5 (calls, tokens, cost, latency per query) and the Sec. 5.3.3 operator statistics (`--operator-stats`) |
| `run_retrieval_ablation.py` | Table 7 (w/o temporal link prediction, bound-guided search, role-aware granularity, shared-cost reuse update) |
| `ask.py` | answer a question with an index and print the selected support paths |

## Notes

- **Caching.** Builds and per-query records are cached under a fingerprint
  of the configuration and the corpus. `--resume` reuses only results
  produced under the same settings, and FULL is built once per dataset and
  facet.
- **Judges.** They are configured in the `evaluation:` config section and
  are mock judges offline. Their cost is metered separately and is
  excluded from Table 5.
- **Clean latencies.** For Table 5 latencies without the LLM response
  cache, run with `--set llm.cache_dir=null`.
- **RAPTOR and LightRAG on arXiv-Subfields.** They build their trees and
  graphs over a dense-retrieval candidate pool
  (`--graph-candidates` / `--graph-global-max-papers`), because indexing
  148K papers with LLM summaries / extractions is prohibitively expensive.
- **Table 7.** Train the link predictor on the held-out index first. The
  ablation script refuses to run with an untrained prior unless
  `--allow-untrained-prior` is given.
