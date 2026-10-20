#!/usr/bin/env bash
# End-to-end offline smoke test of the whole ScholarPath pipeline.
#
# Runs every stage of the paper's pipeline on the synthetic repository with
# the deterministic mock LLM and the hashing embedder (configs/default.yaml):
# no network, no API keys, no model downloads. It checks that the code paths
# work end to end; the numbers it prints are NOT the paper's results.
#
#   bash scripts/smoke_test.sh            # writes everything under results/smoke/
#   OUT=/tmp/sp_smoke bash scripts/smoke_test.sh
#   RUN_TESTS=1 bash scripts/smoke_test.sh  # also run the unit tests first
#
# Stages (paper section in brackets):
#   1. synthetic data in the processed layout                 [data/README.md]
#   2. taxonomy construction: ScholarPath dual taxonomies     [Sec. 3, Alg. 1]
#   3. construction experiments, ablation, scalability,
#      dynamic update                                         [Sec. 5.2, Figs 2-4, Tables 6, 8, 9, Sec. 3.4]
#   4. multi-granularity index (over the stage-2 taxonomies)
#      + link predictor; held-out benchmark construction      [Sec. 2.1, Sec. 4.3, App. B.5, App. C.1]
#   5. retrieval accuracy, end-to-end QA, agents, efficiency,
#      retrieval ablation; the official benchmark formats
#      (ResearchQA, ScholarQABench, TaxoBench, CHIME) prepared
#      from tests/fixtures                                    [Sec. 5.3, Tables 1-5, 7, Figs 5-6, App. C.2]
#   6. interactive question answering                         [Alg. 2]
#
# Caveat: under the mock LLM the budgeted baselines answer the synthetic
# hypothesis queries with "established" items only, so their Table 4 rows
# (Disc and judge scores) are degenerate (0.0); only the code paths are tested.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$REPO/results/smoke}"
PY="${PYTHON:-python3}"
DATA="$OUT/data"
INDEX="$OUT/index"
CACHE="$OUT/cache"
# Small routers keep the offline run fast; the paper's defaults are in configs/.
SMALL=(--set builder.router.hidden_dim=32 --set baseline.router.hidden_dim=32)

cd "$REPO"
mkdir -p "$OUT"
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [[ "${RUN_TESTS:-0}" == "1" ]]; then
  step "Unit tests"
  "$PY" -m pytest -q tests/
fi

step "1. Synthetic data"
"$PY" scripts/make_synthetic_data.py --out "$DATA" --n-papers 240

step "2. ScholarPath dual taxonomies (Algorithm 1)"
"$PY" scripts/build_taxonomy.py --corpus "$DATA/arxiv_sets/synthetic.jsonl.gz" --facet dual \
  --method scholarpath --reference "$DATA/gold/problem_taxonomy.json" \
  --cache-dir "$CACHE/builds" --output-dir "$OUT/build_taxonomy" "${SMALL[@]}"

step "3a. Construction quality vs FULL and selection baselines (Figs 2-3, Tables 8-9)"
"$PY" scripts/run_taxonomy_experiments.py --data-dir "$DATA" --limit-papers 120 \
  --etas 0.1,0.2 --selectors coverage,random --fig3-etas 0.2 --min-router-steps 100 \
  --cache-dir "$CACHE/builds" --resume --no-plots --output-dir "$OUT/run_taxonomy_experiments" "${SMALL[@]}"

step "3b. Construction ablation (Table 6)"
"$PY" scripts/run_taxonomy_ablation.py --data-dir "$DATA" --limit-papers 120 --facets problem \
  --variants no_discovery,no_targeted_repair --cache-dir "$CACHE/builds" --resume \
  --output-dir "$OUT/run_taxonomy_ablation" "${SMALL[@]}"

step "3c. Scalability (Fig. 4)"
"$PY" scripts/run_scalability.py --corpus "$DATA/arxiv_subfields/papers.jsonl.gz" --sizes 60,120 \
  --max-full-size 60 --min-router-steps 100 --cache-dir "$CACHE/builds" --resume --no-plots \
  --output-dir "$OUT/run_scalability" "${SMALL[@]}"

step "3d. Dynamic update (Sec. 3.4)"
"$PY" scripts/run_dynamic_update.py --corpus "$DATA/arxiv_sets/synthetic.jsonl.gz" --cutoff 2022-01-01 \
  --max-cohorts 2 --facets problem --cache-dir "$CACHE/builds" --output-dir "$OUT/run_dynamic_update" \
  "${SMALL[@]}"

step "4a. Multi-granularity semantic index (arXiv-Subfields-style corpus and TaxoBench-style corpus)"
# Indexed over the ScholarPath taxonomies of stage 2 (same papers), not over the gold taxonomies that
# define the synthetic evidence gold J(u, m).
"$PY" scripts/build_index.py --data-dir "$DATA" --index-dir "$INDEX" --corpus arxiv_subfields \
  --problem-taxonomy "$OUT/build_taxonomy/problem/taxonomy.json" \
  --method-taxonomy "$OUT/build_taxonomy/method/taxonomy.json" --output-dir "$OUT/build_index"
"$PY" scripts/build_index.py --data-dir "$DATA" --index-dir "$INDEX" --corpus taxobench --limit-papers 60 \
  --set builder.max_depth=1 --output-dir "$OUT/build_index_taxobench" "${SMALL[@]}"

step "4b. Temporal link predictor (App. B.5)"
"$PY" scripts/train_link_predictor.py --index-dir "$INDEX" --corpus arxiv_subfields \
  --set 'link_prediction.cutoffs=[2018-01-01,2019-01-01,2020-01-01,2021-01-01]' \
  --set link_prediction.horizon_years=1 --set link_prediction.epochs=5 --output-dir "$OUT/train_link_predictor"

step "4c. Held-out hypothesis benchmark construction (App. C.1) on a copy of the layout"
HDATA="$OUT/data_heldout"
rm -rf "$HDATA"
cp -r "$DATA" "$HDATA"
REF="$HDATA/heldout_reference"
# --force: the synthetic layout ships planted held-out queries, which build refuses to overwrite otherwise.
"$PY" scripts/build_heldout_benchmark.py --data-dir "$HDATA" --output-dir "$OUT/build_heldout_benchmark" \
  build --force --problem-taxonomy "$REF/problem_taxonomy.json" --method-taxonomy "$REF/method_taxonomy.json" \
  --domain-map "$REF/domains.json" --n-new-problem 1 --n-new-method 1
"$PY" scripts/build_heldout_benchmark.py --data-dir "$HDATA" --output-dir "$OUT/build_heldout_benchmark" \
  apply --auto-accept
"$PY" scripts/build_index.py --data-dir "$HDATA" --index-dir "$OUT/index_heldout" --corpus heldout \
  --problem-taxonomy "$DATA/gold/problem_taxonomy.json" --method-taxonomy "$DATA/gold/method_taxonomy.json" \
  --output-dir "$OUT/build_index_heldout"
"$PY" scripts/run_qa_eval.py --data-dir "$HDATA" --index-dir "$OUT/index_heldout" --benchmarks heldout \
  --systems scholarpath --budgets 2048 --sweep-budgets none --no-plots --output-dir "$OUT/run_qa_eval_heldout"

step "5a. Retrieval accuracy (Table 2, Fig. 5)"
"$PY" scripts/run_retrieval_eval.py --data-dir "$DATA" --index-dir "$INDEX" --budgets 1024,2048 --limit 5 \
  --graph-global-max-papers 0 --graph-candidates 8 --raptor-clustering kmeans --raptor-no-umap \
  --output-dir "$OUT/run_retrieval_eval"

step "5b. End-to-end QA, evidence and hypothesis queries (Tables 3-4, Fig. 6)"
"$PY" scripts/run_qa_eval.py --data-dir "$DATA" --index-dir "$INDEX" --budgets 2048 --sweep-budgets 1024,2048 \
  --limit 3 --output-dir "$OUT/run_qa_eval"

step "5c. Unbudgeted agent path (offline simulated agent; not a paper baseline)"
"$PY" scripts/run_agents.py --data-dir "$DATA" --index-dir "$INDEX" --agents simulated --limit 3 \
  --output-dir "$OUT/run_agents"

step "5d. Efficiency (Table 5)"
"$PY" scripts/run_efficiency.py --inputs "$OUT/run_qa_eval,$OUT/run_agents" --budget 2048 \
  --data-dir "$DATA" --index-dir "$INDEX" --operator-stats --exhaustive-check --limit 3 \
  --output-dir "$OUT/run_efficiency"

step "5e. Retrieval ablation on hypothesis queries (Table 7)"
"$PY" scripts/run_retrieval_ablation.py --data-dir "$DATA" --index-dir "$INDEX" --budget 2048 --limit 2 \
  --output-dir "$OUT/run_retrieval_ablation"

step "5f. Official benchmark formats prepared from tests/fixtures (Tables 1-3: Rub, Corr, Cite, ARI, Sem, Cat, Assn)"
FIX="$OUT/data_fixtures"
rm -rf "$FIX"
"$PY" scripts/prepare_arxiv.py --snapshot tests/fixtures/arxiv/arxiv-metadata-sample.json --out "$FIX" --n-per-set 2
"$PY" scripts/prepare_fulltext.py --corpus "$FIX/arxiv_subfields/papers.jsonl.gz" --sources local,s2orc \
  --local-dir tests/fixtures/fulltext/local --s2orc tests/fixtures/fulltext/s2orc_v2-sample.jsonl
"$PY" scripts/prepare_benchmarks.py --raw tests/fixtures --out "$FIX" --corpus "$FIX/arxiv_subfields/papers.jsonl.gz"
for corpus in arxiv_subfields taxobench chime; do
  "$PY" scripts/build_index.py --data-dir "$FIX" --index-dir "$OUT/index_fixtures" --corpus "$corpus" \
    --output-dir "$OUT/build_index_fixtures_$corpus" "${SMALL[@]}"
done
"$PY" scripts/run_retrieval_eval.py --data-dir "$FIX" --index-dir "$OUT/index_fixtures" --budgets 2048 --limit 2 \
  --graph-candidates 8 --raptor-clustering kmeans --raptor-no-umap --no-plots \
  --output-dir "$OUT/run_retrieval_eval_fixtures"
"$PY" scripts/run_qa_eval.py --data-dir "$FIX" --index-dir "$OUT/index_fixtures" --budgets 2048 --sweep-budgets none \
  --limit 2 --no-plots --output-dir "$OUT/run_qa_eval_fixtures"

step "6. Ask a question"
"$PY" scripts/ask.py --index-dir "$INDEX" --corpus arxiv_subfields --output-dir "$OUT/ask" \
  "How has bayesian optimization been used for cardinality estimation?"

step "Done: outputs under $OUT"
