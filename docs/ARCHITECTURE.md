# ScholarPath — Code Architecture

This document maps the paper *"Empowering Automated Research Agent At Scale: the
Role of Evidence Query and Hypothesis Query"* (SIGMOD '27 submission;
system name **ScholarPath**) onto the code, and fixes the interfaces between
modules.

## 1. Paper → code map

| Paper | Code |
|---|---|
| §2.1 multi-granularity semantic index I (concepts / abstracts / section chunks) | `scholarpath.index.semantic_index.SemanticIndex` |
| §2.1 taxonomy T=(V,E), extents, residual leaf ⊥_v | `scholarpath.taxonomy.tree.Taxonomy` |
| §2.2 Def. 1 confidence-bounded taxonomy construction | `scholarpath.taxonomy.builder.NodeWiseBuilder` |
| Alg. 1 node-wise construction (episodes, stages 1–3) | `taxonomy/builder.py` |
| §3.1 lightweight router f_θv (2-layer MLP, GELU, AdamW) | `taxonomy/router.py` (`MLPRouter`) |
| §3.1 node-wise organization loss L_v = m_v + (1−m_v) e_v | `taxonomy/builder.py` (loss proxies), `taxonomy/certification.py` |
| §3.2 regions as decision units (k-means, G_v = min{8,⌊n_v/24⌋}) | `taxonomy/regions.py` |
| §3.2 / App. B.1 logistic response curves (Eq. 13), transferred hierarchical prior (Eq. 14), Thompson sampling | `taxonomy/response_model.py` |
| §3.2 / App. A.1 batch allocation DP (Eq. 10) and sizing z_plan (Eq. 3) | `taxonomy/allocation.py` |
| §3.3 audit-based certification (hypergeometric bound, Eq. 4, α_{v,j,a}) | `taxonomy/certification.py` |
| §3.3 / App. A.2 audit-guided repair: DISCOVER (stratified sample), CORRECT (κ neighbours), relabeling | `taxonomy/repair.py` |
| §3.4 / App. B.3 dynamic update (U_new = (n U_old + M)/(n+M), Eq. 16) | `taxonomy/dynamic.py` |
| App. B.2 cost accounting (Eq. 15: acquisition, auditing, discovery, correction, relabeling) | `scholarpath.llm.accounting.CostMeter` components |
| §5.2 baselines: FULL (η=1), node-wise Random / Coverage / K-means / Diversity, one-off selection (App. C.5) | `taxonomy/baselines.py`, `taxonomy/selectors.py` |
| Prompts ("All prompts are available in our code repository") | `scholarpath/prompts/*.py` |
| Alg. 2 support-path retrieval (6 steps) | `scholarpath.retrieval.pipeline.ScholarPathRetriever` |
| Step 1 query understanding & aspect extraction | `retrieval/query_understanding.py` |
| Step 2 taxonomy grounding (scopes u_q, m_q, query-local anchors) | `retrieval/grounding.py` |
| §4.2 redundancy-aware aspect coverage F_q (Eq. 5), gain per new token δ (Eq. 6), shared-node costs | `retrieval/objective.py` |
| Direct paths / complementary path sets b = {e_C, e_P, e_M} | `retrieval/units.py` |
| §4.4 granularity-weighted adequacy r̂_{a,e}, premise-bottleneck score (Eq. 9), bridge discount φ | `retrieval/scoring.py` |
| §4.3 hierarchy-bounded subtree pruning (Eq. 7), lazy numerators / eager denominators, exact lazy search (Prop. 7) | `retrieval/search.py` |
| §4.3 / App. A.3 ranked join over grounded scopes (Prop. 3) | `retrieval/search.py` |
| §4.3 cross-taxonomy link prediction s_θ(u,m)=z_P(u)ᵀz_M(m) (Eq. 8), bound U(u,v) (Eq. 12), temporal training (Eq. 17–18) | `retrieval/link_prediction.py` |
| §4.4 selective chunk expansion (App. B.4) | `retrieval/search.py`, `retrieval/scoring.py` |
| Step 6 context assembly & single composition call | `retrieval/compose.py`, `prompts/answer.py` |
| §5.3 retrieval baselines: BM25, SPECTER, Dense+reranker, RAPTOR, LightRAG; keyword search | `scholarpath/baselines/*.py` |
| §5.3 unbudgeted agents: PaperQA2, OpenScholar, AgenticScholar, Codex | `scholarpath/baselines/agents/*.py` |
| App. C.1 held-out hypothesis benchmark construction | `scholarpath/data/heldout.py` |
| App. C.2 datasets: arXiv sets, arXiv-Subfields, TaxoBench, CHIME, ResearchQA, ScholarQABench; corpus alignment | `scholarpath/data/*.py` |
| §5.2 metrics: soft node F1, soft edge F1 | `evaluation/taxonomy_metrics.py` |
| §5.3 metrics: Recall@B, NDCG@B; Rub, Cite, Corr, ARI, Sem, Cat, Assn; Disc (Hit@10), LLM-judge Feas/Nov/Sound | `evaluation/*.py` |
| Tables / figures | `scripts/run_*.py` (see `scripts/README.md`) |

## 2. Conventions

* Python ≥ 3.9, `from __future__ import annotations`, type hints, dataclasses.
  Core library dependencies: numpy, scipy, scikit-learn, pyyaml, tqdm.
  Anything else (openai, tiktoken, torch, transformers, adapters,
  sentence-transformers, umap-learn, networkx, paper-qa, datasets, pandas,
  pyarrow, pymupdf, nltk) is an optional extra (`pyproject.toml`) and must be
  imported lazily inside the function that needs it, with an error message
  that names the install command (e.g. `embeddings.MissingDependencyError`).
* Docstrings cite the paper section / equation / algorithm line implemented.
* **Every LLM call** is built by a prompt function in `scholarpath/prompts/<area>.py`
  that returns an `LLMRequest(task="<area>.<name>", messages=..., payload=..., component=...)`,
  and is executed via `llm.complete(...)` / `llm.complete_json(...)`.
  Each prompt module exports `MOCK_HANDLERS = {task: handler}` so that the
  offline `MockLLM` can answer the task from `payload` (deterministic,
  heuristic; must return the same JSON shape the parser expects).
* **Cost components** (`LLMRequest.component` or `with llm.meter.scope(name)`):
  taxonomy: `acquisition`, `auditing`, `discovery`, `correction`, `relabeling`
  (ScholarPath, Eq. 15), `full`, `selection` (baselines);
  query time: `query_understanding`, `composition`, `rerank`, `index_build`
  (LightRAG/RAPTOR offline indexing), `agent`;
  evaluation: `judge` (never counted as system cost); benchmark construction: `benchmark`.
* Embeddings come from `scholarpath.embeddings.build_embedder(config.embedding)`;
  all vectors are L2-normalised float32, so `x @ y` is cosine similarity.
  Use `embedder.encode_queries` for queries/aspects and `embedder.encode` for documents.
* Token counts / costs c(x): `scholarpath.text.count_tokens`.
* Randomness: always seeded (`config.seed` or explicit `seed=` args); no global state.
* Scripts in `scripts/` add `src/` to `sys.path`, so they run without
  installation. Experiment scripts use `scholarpath.runtime.add_common_args`
  / `Runtime.from_args` (invalid `--set` overrides and missing optional
  dependencies end with a one-line error), accept `--limit`-style flags for
  quick runs, and write outputs under `results/<script>/`; indexes go to
  `artifacts/index/<corpus>/`. The data-preparation scripts (`download_data`,
  `prepare_*`, `make_synthetic_data`) have their own flags (`data/README.md`).
* Config values are type-checked on load and on `override` (`ConfigError`,
  `UnknownConfigField`).

## 3. Foundation

* `scholarpath.config` — `ScholarPathConfig` with sections `llm`, `embedding`,
  `builder` (`BuilderConfig`, incl. `router: RouterConfig`), `baseline`
  (`BaselineBuilderConfig`), `retrieval` (`RetrievalConfig`), `link_prediction`
  (`LinkPredictionConfig`), `index` (`IndexConfig`), `evaluation`
  (`EvaluationConfig`: judges, official checkouts). Load/save YAML,
  `override(["a.b=c"])` (type-checked; nested dict keys such as
  `retrieval.granularity_weights.overview.chunk`), `mock_judge_warning()`.
* `scholarpath.types` — `Paper`, `Chunk`, `Query` (kinds `EVIDENCE`,
  `HYPOTHESIS_DISCOVERY`, `HYPOTHESIS_ASSESSMENT`), `RubricItem`, `Answer`, `AnswerItem`.
* `scholarpath.text` — `count_tokens`, `truncate_tokens`, `tokenize`,
  `normalize_title`, `title_similarity`, `keywords`, `split_sections`, `chunk_sections`.
* `scholarpath.llm` — `LLMClient.complete/complete_json/map`, `LLMRequest`,
  `LLMResponse`, `CostMeter` (`scope`, `snapshot`, `diff`, `total`), `MockLLM`,
  `build_llm`, `extract_json`, `stable_hash`.
* `scholarpath.embeddings` — `HashingEmbedder`, `Specter2Embedder`,
  `SentenceTransformerEmbedder`, `CachedEmbedder`, `build_embedder`, `l2_normalize`, `cosine`,
  `cosine_matrix`, `MissingDependencyError`.
* `scholarpath.runtime` — `Runtime(config, llm, embedder, meter, output_dir)`,
  `add_common_args`, `config_from_args`, `Runtime.from_args`, `Runtime.from_config`.
* `scholarpath.data.corpus` — `Corpus` (dict of papers; `load/save` JSONL(.gz),
  `subset`, `sample`, `before(date)`, `attach_chunks`), `read_jsonl`, `write_jsonl`,
  `read_json`, `write_json` (strict JSON: NaN / inf become `null`), `load_queries`, `save_queries`.
* `scholarpath.data.synthetic` — `make_synthetic_benchmark(n_papers, ...)` →
  `SyntheticBenchmark(corpus, problem_gold, method_gold, evidence_queries,
  hypothesis_queries, heldout_papers, heldout_pairs, pair_of)`.
* `scholarpath.taxonomy.tree` — `Taxonomy` (`add_node`, `assign` (propagates to
  ancestors), `extent`, `children`, `ancestors`, `descendants`, `path`, `edges`,
  `distance`, `paper_nodes`, `deepest_nodes`, `leaf_assignment`, `from_nested`,
  `to_nested`, `save/load`), node statuses `OPEN/CERTIFIED/UNRESOLVED/LEAF`,
  `node.residual` = ⊥_v.
* `scholarpath.taxonomy.llm_ops` — `TaxonomyLLM(llm, papers, facet, papers_per_call, ...)`
  with `propose_children(tax, nid, paper_ids, existing, component) -> (List[ChildSpec], Dict[pid, PaperLabel])`,
  `label(tax, nid, child_ids, paper_ids, component, allow_new) -> LabelBatch`,
  `discover(tax, nid, child_ids, paper_ids, component) -> LabelBatch`;
  `resolve_new_children(tax, parent_id, specs, labels, max_children)` adds proposed
  children and rewrites `"new:k"` placeholders to node ids. `ops.papers_read[component]`
  counts paper reads.
* `scholarpath.taxonomy.router` — `MLPRouter(config).fit(paper_vecs, child_vecs, targets)`,
  `.predict_proba`, `.assign(probs)`; `stratified_split(labels, fraction, seed)`, `disagreement(...)`.
* `scholarpath.taxonomy.regions` — `Regions.build(paper_ids, vectors, max_regions, papers_per_region, seed)`
  (`.members`, `.shares`, `.labels`, `.region_of()`), `num_regions`.
* `scholarpath.taxonomy.result` — `BuildResult(taxonomy, method, usage, papers_read,
  episodes, runtime_s, routers, metadata)` (`cost_usd`, `relative_cost`, `summary`,
  `save` / `load`, which also store and restore the frozen node routers in
  `routers.npz` / `routers.json`), `EpisodeRecord`,
  `NodeRouter(node_id, child_ids, child_vecs, router).route(vecs)`.
* `scholarpath.index.semantic_index` — `SemanticIndex.build(problem, method, papers, embedder, config)`;
  keys `concept:<nid>`, `paper:<pid>`, `chunk:<cid>`; `text/node_cost/vector`, `extent`,
  `shared_papers(u, m)` = J(u,m), `co_concepts`, `paper_concepts`, `chunks_of`,
  `concept_matrix(facet)`, `paper_matrix()`, `ball(nid, level) -> Ball(center, radius, count)`
  with `Ball.upper_cos(x)`, `paper_chunk_balls[pid]`, `save/load`.
* `scholarpath.retrieval.result` — `RetrievedUnit`, `RetrievalResult` (shared by ScholarPath and baselines).
* `scholarpath.prompts.answer` — `compose_answer_request(query, context, kind, sources, max_tokens)`,
  `parse_answer(obj, valid_ids)`: the single composition call used by every budgeted method.
* `scholarpath.prompts.taxonomy` — taxonomy prompts + mocks.

## 4. Module APIs

The main public entry points per package (see each module's docstring for the
full API and the paper references). Names are listed by module; import them
from the module (e.g. `from scholarpath.taxonomy.baselines import
SelectionBuilder`), since the package `__init__` files re-export only part of
them.

### 4.1 Taxonomy builder (`scholarpath.taxonomy`)

```python
# certification.py  (Sec. 3.3, Eq. 4)
hypergeom_upper_bound(R, a, y, alpha) -> int           # K^U
step_alpha(delta_v, j, a) -> float                     # delta_v / [j(j+1)a(a+1)]
certificate_bound(n_v, E_D, R, a, y, delta_v, j) -> float
DeltaAllocator(delta, strategy="spend"|"harmonic"|"uniform", n_total, max_depth, max_nodes).allocate(n_v)
audit_snapshot(unverified, judge, n_v, E_D, delta_v, j, epsilon, batch_size, rng, ...) -> AuditOutcome
# allocation.py  (Eq. 3, Eq. 10, Prop. 1)
plan_batch(curves, shares, beta, epsilon, token_cost) -> BatchPlan(z, allocation, predicted_loss, F)
# response_model.py  (Eq. 13-14)
RegionResponse(...).thompson_curve(n_max, rng); TransferredPrior(...).predict(descriptor) / .update(...)
# repair.py  (App. A.2, B.2)
discovery_draws(...), discover(...), correction_targets(...), correct(...), relabel_candidates(...), relabel(...)
# builder.py  (Algorithm 1)
NodeWiseBuilder(llm, embedder, config, papers, paper_vectors=None, prior=None).build(root_name, root_definition,
                paper_ids=None, facet=None) -> BuildResult
build_dual_taxonomies(llm, embedder, config, papers, ...) -> (BuildResult problem, BuildResult method)
# dynamic.py  (Sec. 3.4, Eq. 16)
DynamicUpdater(builder, result).add_papers(new_papers) -> UpdateReport
# baselines.py / selectors.py  (Sec. 5.2, App. C.5)
select(vectors, k, method, seed) ; SelectionBuilder(llm, embedder, BaselineBuilderConfig, papers, ...).build(...)
per_paper_node_cost(full_result) ; estimate_full_cost(result, per_paper_cost)
# result.py
BuildResult(taxonomy, method, usage, papers_read, episodes, runtime_s, routers, metadata).save(dir) / .load(dir)
```

### 4.2 Retrieval (`scholarpath.retrieval`)

```python
analyze_query(query, llm, embedder, config) -> QueryAnalysis          # Step 1 (llm=None: embedding-only parser)
ground(analysis, index, embedder, config, link_predictor=None, anchor_min_sim=None) -> Grounding   # Step 2
build_search_space(analysis, grounding, index, config, link_predictor) -> SearchSpace          # Steps 3-5
make_search(space) -> BoundGuidedSearch | ExhaustiveGreedy; .run(budget) -> SearchResult       # Prop. 7
ranked_join(index, pairs, relevance, k) -> RankedJoinResult                                    # App. A.3
assemble_context(index, units) -> (context, sources); compose_answer(query, retrieval, llm)    # Step 6
ScholarPathRetriever(index, llm, embedder, config, link_predictor=None).retrieve(query, budget) / .answer(...)
# RetrievalResult.stats: scored_nodes, scored_candidates (Table 7 "Scored"), candidate_pool, link_evaluations,
# postings_total / scanned / pruned, postings_pruned_by_junction, regions_expanded / deferred, search_time_s, ...
# link_prediction.py / link_training.py  (Eq. 8, 12, 17, 18)
LinkPredictor: prepare(index), score(u, m), prob, upper_bound(u, v), upper_bound_problems(m, v), has(nid)
EmbeddingLinkPredictor()                       # untrained prior ("w/o temporal link prediction" in Table 7)
temporal_supervision(index, cutoff, horizon_years, llm=None, dates=None)
train_link_predictor(index, config, llm=None, dates=None, exclude_concepts=(), strict=False) -> LinkPredictor
```

### 4.3 Baselines (`scholarpath.baselines`)

```python
PassagePool.from_corpus(corpus) / .from_index(index); make_retriever(method, pool, llm, embedder, dense=None)
# names: bm25, specter, dense, dense_rerank, raptor, lightrag, keyword; each .retrieve(query, budget) -> RetrievalResult
pack_to_budget(candidates, budget, dedup=True, max_candidates=None)
BudgetedQA(retriever, llm, config, *, budget=None, answer_max_tokens=None).answer(query, budget)   # single composition
NoRetrievalQA(llm).answer(query)
agents: PaperQA2Agent, OpenScholarAgent, AgenticScholarAgent, CodexAgent, SimulatedAgent; export_corpus_shards(...)
```
Retrievers with offline indexing report its time in `RetrievalResult.stats["index_build_s"]`; QA latency excludes it.
Every answer stores the context it was composed from in `Answer.metadata["passages"]`.

### 4.4 Data (`scholarpath.data`)

```python
# re-exported by scholarpath.data
build_arxiv_sets / build_subfields_corpus / write_arxiv_corpora(snapshot, out_dir, ...)   # Sec. 5.2, App. C.2
FulltextExtractor(make_sources(...)).run(papers, out_path)                                # Sec. 2.1 chunks
PaperAligner(corpus).resolve(references) -> GoldResolution                                # App. C.2 coverage
load_researchqa / load_researchqa_hf / load_scholarqa_cs / load_scholarqa_multi / load_taxobench / build_chime
download_dataset(name, root, ...)                                                         # scripts/download_data.py
# scholarpath.data.heldout (App. C.1)
select_targets, reduce_corpus, generate_queries, export_annotation_sheet, apply_annotations,
build_heldout_benchmark(..., oversample=), save_heldout_benchmark
# scholarpath.data.synthetic (offline runs)
make_synthetic_benchmark(...)
```

### 4.5 Evaluation (`scholarpath.evaluation`)

```python
taxonomy_metrics: TaxonomyScorer, soft_node_f1, soft_edge_f1, taxonomy_scores          # Sec. 5.2
organization_metrics: adjusted_rand_index, semantic_path_similarity, category_soft_f1,
                      assignment_f1, organization_scores                                # TaxoBench, CHIME
retrieval_metrics: recall_at_budget, ndcg_at_budget, evaluate_retrieval                 # Table 2
qa_metrics: rubric_coverage, citation_prf, correctness, answer_citation_attribution,
            prometheus_grade, scholarqabench_combined, evaluate_researchqa / _scholarqa_cs / _scholarqa_multi
hypothesis_metrics: discovery_hit, judge_hypotheses, evaluate_hypotheses              # Table 4
judges: JudgePanel; efficiency: EfficiencyTracker, summarize_efficiency                # Table 5
reporting: tables / CSV / plots
official: exporters and runners of the official scripts; OfficialPaths.from_config(config.evaluation),
          run_official_evaluation(benchmark, answers, queries, out_dir, paths)    # scripts/run_official_eval.py
```

Every name above is importable from `scholarpath.evaluation` (resolved
lazily). The `qa_metrics.evaluate_*` drivers skip queries without an answer
as the official scripts do (`missing="skip"`) or score them 0
(`missing="zero"`). The Table 3 runner of `experiments/qa.py` and
`run_official_evaluation` count them as 0.

### 4.6 Experiments (`scholarpath.experiments`)

`common.py` (build cache keyed by configuration and corpus content, cost
isolation per build, table/plot helpers), `taxonomy.py` (construction runner,
Figs 2-4, Tables 6, 8, 9, dynamic update) and `qa.py` (data layout, index
store, systems, record store, evaluation of Tables 2-5 and 7, agents, `ask`).
The scripts in `scripts/` are thin command-line wrappers around them.

## 5. Processed data layout

```
data/
├── raw/                       # downloads (git-ignored)
├── processed/
│   ├── arxiv_sets/<category>.jsonl.gz       # 8 x 2,000 papers (taxonomy construction)
│   ├── arxiv_subfields/papers.jsonl.gz      # 148,400 papers (QA corpus)
│   ├── arxiv_subfields/chunks.jsonl.gz      # section chunks
│   ├── taxobench/{papers.jsonl.gz, gold_taxonomy.json, queries.jsonl}
│   ├── chime/{papers.jsonl.gz, gold/<topic>.json, queries.jsonl}
│   ├── researchqa/queries.jsonl             # aligned to arxiv_subfields
│   ├── scholarqabench/{cs,multi}.jsonl
│   └── heldout/{queries.jsonl, targets.json, stats.json, removed_papers.json,
│                reduced_corpus_ids.json, annotation sheet}
└── README.md
```
