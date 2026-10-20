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
  Anything else (openai, torch, transformers, adapters, rank_bm25, networkx,
  paper-qa, datasets, pymupdf, requests) must be imported lazily inside the
  function that needs it, with a clear error message if missing.
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
* Tests live in `tests/test_<module>.py`, use the `MockLLM` + `HashingEmbedder`
  (`Runtime.from_config()` with default config) and
  `scholarpath.data.synthetic.make_synthetic_benchmark`, never touch the network,
  and run in a few seconds each. Run with `python -m pytest -q`.
* Scripts in `scripts/` add `src/` to `sys.path` (like the other repos), use
  `scholarpath.runtime.add_common_args` / `Runtime.from_args`, accept
  `--limit`-style flags for quick runs, and write outputs under `results/<script>/`.

## 3. Foundation (implemented)

* `scholarpath.config` — `ScholarPathConfig` with sections `llm`, `embedding`,
  `builder` (`BuilderConfig`, incl. `router: RouterConfig`), `baseline`
  (`BaselineBuilderConfig`), `retrieval` (`RetrievalConfig`), `link_prediction`
  (`LinkPredictionConfig`), `index` (`IndexConfig`). Load/save YAML, `override(["a.b=c"])`.
* `scholarpath.types` — `Paper`, `Chunk`, `Query` (kinds `EVIDENCE`,
  `HYPOTHESIS_DISCOVERY`, `HYPOTHESIS_ASSESSMENT`), `RubricItem`, `Answer`, `AnswerItem`.
* `scholarpath.text` — `count_tokens`, `truncate_tokens`, `tokenize`,
  `normalize_title`, `title_similarity`, `keywords`, `split_sections`, `chunk_sections`.
* `scholarpath.llm` — `LLMClient.complete/complete_json/map`, `LLMRequest`,
  `LLMResponse`, `CostMeter` (`scope`, `snapshot`, `diff`, `total`), `MockLLM`,
  `build_llm`, `extract_json`, `stable_hash`.
* `scholarpath.embeddings` — `HashingEmbedder`, `Specter2Embedder`,
  `SentenceTransformerEmbedder`, `CachedEmbedder`, `build_embedder`, `l2_normalize`, `cosine_matrix`.
* `scholarpath.runtime` — `Runtime(config, llm, embedder, meter, output_dir)`,
  `add_common_args`, `Runtime.from_args`, `Runtime.from_config`.
* `scholarpath.data.corpus` — `Corpus` (dict of papers; `load/save` JSONL(.gz),
  `subset`, `sample`, `before(date)`, `attach_chunks`), `read_jsonl`, `write_jsonl`,
  `load_queries`, `save_queries`.
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
  `resolve_new_children(tax, parent, specs, labels, max_children)` adds proposed
  children and rewrites `"new:k"` placeholders to node ids. `ops.papers_read[component]`
  counts paper reads.
* `scholarpath.taxonomy.router` — `MLPRouter(config).fit(paper_vecs, child_vecs, targets)`,
  `.predict_proba`, `.assign(probs)`; `stratified_split(labels, fraction, seed)`, `disagreement(...)`.
* `scholarpath.taxonomy.regions` — `Regions.build(paper_ids, vectors, max_regions, papers_per_region, seed)`
  (`.members`, `.shares`, `.labels`, `.region_of()`), `num_regions`.
* `scholarpath.taxonomy.result` — `BuildResult(taxonomy, method, usage, papers_read,
  episodes, runtime_s, routers, metadata)` (`cost_usd`, `relative_cost`, `summary`, `save`),
  `EpisodeRecord`, `NodeRouter(node_id, child_ids, child_vecs, router).route(vecs)`.
* `scholarpath.index.semantic_index` — `SemanticIndex.build(problem, method, papers, embedder, config)`;
  keys `concept:<nid>`, `paper:<pid>`, `chunk:<cid>`; `text/node_cost/vector`, `extent`,
  `shared_papers(u, m)` = J(u,m), `co_concepts`, `paper_concepts`, `chunks_of`,
  `concept_matrix(facet)`, `paper_matrix()`, `ball(nid, level) -> Ball(center, radius, count)`
  with `Ball.upper_cos(x)`, `paper_chunk_balls[pid]`, `save/load`.
* `scholarpath.retrieval.result` — `RetrievedUnit`, `RetrievalResult` (shared by ScholarPath and baselines).
* `scholarpath.prompts.answer` — `compose_answer_request(query, context, kind, sources, max_tokens)`,
  `parse_answer(obj, valid_ids)`: the single composition call used by every budgeted method.
* `scholarpath.prompts.taxonomy` — taxonomy prompts + mocks.

## 4. Module contracts

### 4.1 Taxonomy builder (`scholarpath.taxonomy`)

```python
# certification.py
def hypergeom_upper_bound(R: int, a: int, y: int, alpha: float) -> int      # K^U (Sec. 3.3)
def step_alpha(delta_v: float, j: int, a: int) -> float                    # δ_v / [j(j+1)a(a+1)]
def certificate_bound(n_v, E_D, R, a, y, delta_v, j) -> float              # U_{v,j,a} (Eq. 4)
class DeltaAllocator:  # splits δ over nodes so Σ_v δ_v ≤ δ
    def __init__(self, delta, strategy="spend", n_total=..., max_depth=..., max_nodes=...)
    def allocate(self, n_v: int) -> float
@dataclass class AuditOutcome: certified, bound, audited: Dict[pid, bool], draws, errors
def audit_snapshot(unverified, judge, n_v, E_D, delta_v, j, epsilon, batch_size, rng, ...) -> AuditOutcome

# allocation.py  (Eq. 10 DP, Eq. 3 sizing; App. A.1)
@dataclass class BatchPlan: z: int; allocation: np.ndarray; predicted_loss: float; F: np.ndarray
def plan_batch(curves: Sequence[np.ndarray], shares, beta, epsilon, token_cost) -> BatchPlan

# response_model.py  (Eq. 13–14, App. B.1)
class TransferredPrior  # λ ~ N(W h, Ω); update(descriptors, observed slopes) after each episode
class RegionResponse     # per-region posterior over (λ_mis, λ_err); observe(); thompson_curve(n_max, rng)

# repair.py  (Sec. 3.3 repair, App. A.2)
def discovery_draws(residual_by_region, eta_A, s_max, rng) -> List[str]   # proportional stratified
def correct(...) / relabel(...)

# builder.py  (Algorithm 1)
class NodeWiseBuilder:
    def __init__(self, llm, embedder, config: BuilderConfig, papers: Dict[str, Paper] | Corpus,
                 paper_vectors: Optional[Dict[str, np.ndarray]] = None, prior: Optional[TransferredPrior] = None)
    def build(self, root_name="Root", root_definition="", paper_ids=None, facet=None) -> BuildResult
def build_dual_taxonomies(llm, embedder, config, papers, **kw) -> Tuple[BuildResult, BuildResult]

# dynamic.py  (Sec. 3.4, App. B.3)
class DynamicUpdater:
    def __init__(self, builder: NodeWiseBuilder, result: BuildResult)
    def add_papers(self, new_papers: Sequence[Paper]) -> UpdateReport   # zero-LLM path when U_new ≤ ε
```

### 4.2 Taxonomy baselines (`taxonomy/baselines.py`, `taxonomy/selectors.py`)

```python
def select(vectors: np.ndarray, k: int, method: str, seed: int = 0) -> List[int]
    # "random" | "coverage" (facility location Σ_p max_s cos) | "kmeans" (medoid nearest each centroid) | "diversity" (farthest point)
class SelectionBuilder:
    def __init__(self, llm, embedder, config: BaselineBuilderConfig, papers, paper_vectors=None)
    def build(self, root_name="Root", root_definition="", paper_ids=None, facet=None) -> BuildResult
    # selector="full" → FULL (η=1): LLM reads all n_v papers to propose children, then again to route each paper.
    # mode="node_wise": select ⌈η n_v⌉ at each internal node, LLM proposes + labels them, router routes the rest; stop when n_v < 1/η.
    # mode="one_off": select ⌈η|P|⌉ once at the root; those papers are the only LLM reads, routed top-down by f_θv.
def estimate_full_cost(result: BuildResult, per_paper_cost: float) -> float   # Sec. 5.2.2: c̄ Σ_{v internal} n_v
```

### 4.3 Retrieval (`scholarpath.retrieval`)

```python
# query_understanding.py (Step 1)
@dataclass class Aspect: text; weight; role ("overview"|"empirical"|"mechanism"); vector
@dataclass class QueryAnalysis: query_id; text; kind; problem_concepts: List[str]; method_concepts: List[str];
                                aspects: List[Aspect]; given_pair: Optional[Tuple[str, str]]
def analyze_query(query: Query | str, llm, embedder, config: RetrievalConfig) -> QueryAnalysis

# grounding.py (Step 2)
@dataclass class Grounding: problem_scopes: List[Tuple[str, float]]; method_scopes: ...; anchors: ...
def ground(analysis, index: SemanticIndex, embedder, config) -> Grounding

# units.py, scoring.py, objective.py, search.py (Steps 3–5, Sec. 4.2–4.4, App. A.3–A.5, B.4)
class BoundGuidedSearch  # Algorithm 2 lines 5–23; exact (selects same units as ExhaustiveGreedy)
class ExhaustiveGreedy    # ablation "w/o bound-guided search"

# compose.py (Step 6)
def assemble_context(index, units) -> Tuple[str, List[dict]]    # hierarchical, shared nodes once
def compose_answer(query, retrieval: RetrievalResult, llm, config) -> Answer

# pipeline.py
class ScholarPathRetriever:
    def __init__(self, index: SemanticIndex, llm, embedder, config: RetrievalConfig, link_predictor=None)
    def retrieve(self, query: Query, budget: Optional[int] = None) -> RetrievalResult
    def answer(self, query: Query, budget: Optional[int] = None) -> Answer

# link_prediction.py (Sec. 4.3 Eq. 8, 12; App. B.5 Eq. 17–18)
class LinkPredictor:
    def fit(self, index: SemanticIndex, positives, ...) -> "LinkPredictor"
    def z_problem(self, node_id) -> np.ndarray; def z_method(self, node_id) -> np.ndarray
    def score(self, u: str, m: str) -> float              # s_θ(u, m)
    def upper_bound(self, u: str, v: str) -> float         # U(u, v) over methods m ⪯ v (Eq. 12)
    def upper_bound_problems(self, m: str, v: str) -> float  # roles exchanged
    def encode_query_concept(self, facet, description_vec, ancestor_vecs) -> np.ndarray
    def save(path); @classmethod load(path)
class EmbeddingLinkPredictor(LinkPredictor)   # untrained fallback: cosine of description embeddings
def temporal_supervision(index, cutoff, horizon_years, llm=None) -> List[Tuple[str, str]]   # Eq. 17
def train_link_predictor(index, config: LinkPredictionConfig, llm=None) -> LinkPredictor
```

### 4.4 Retrieval baselines (`scholarpath.baselines`)

```python
class BudgetedRetriever(Protocol):
    name: str
    def retrieve(self, query: Query, budget: int) -> RetrievalResult
# bm25.BM25Retriever, dense.DenseRetriever (SPECTER / SPECTER2), rerank.DenseRerankRetriever (LLM listwise),
# raptor.RaptorRetriever, lightrag.LightRAGRetriever, keyword.KeywordSearchRetriever
def pack_to_budget(candidates, budget) -> List[RetrievedUnit]
class BudgetedQA:   # pack ≤ B tokens + single composition call (prompts.answer)
    def __init__(self, retriever, llm, config); def answer(self, query, budget) -> Answer
class NoRetrievalQA   # "Answer LLM, no retrieval"
# agents/: PaperQA2Agent, OpenScholarAgent, AgenticScholarAgent, CodexAgent (adapters, .answer(query) -> Answer)
```

### 4.5 Evaluation (`scholarpath.evaluation`)

```python
taxonomy_metrics.soft_node_f1(pred, gold, embedder) / soft_edge_f1(...) / taxonomy_scores(...)
organization_metrics.adjusted_rand_index(...), semantic_path_similarity(...),
                     category_soft_f1(...), assignment_f1(...)
retrieval_metrics.recall_at_budget(ranked, gold, num_gold), ndcg_at_budget(...)
qa_metrics.rubric_coverage(answer, rubric, judge), citation_f1(...), correctness(...)
hypothesis_metrics.discovery_hit(answer, query, judge, k=10), judge_hypotheses(answer, query, judges)
judges.JudgePanel, efficiency.EfficiencyTracker, reporting.write_table(...)
```

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
│   └── heldout/{queries.jsonl, removed_papers.json, reduced_corpus_ids.json}
└── README.md
```
