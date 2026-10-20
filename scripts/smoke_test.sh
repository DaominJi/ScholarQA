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
#   4. multi-granularity index + link predictor               [Sec. 2.1, Sec. 4.3, App. B.5]
#   5. retrieval accuracy, end-to-end QA, agents, efficiency,
#      retrieval ablation                                     [Sec. 5.3, Tables 2-5, 7, Figs 5-6]
#   6. interactive question answering                         [Alg. 2]
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
"$PY" scripts/build_index.py --data-dir "$DATA" --index-dir "$INDEX" --corpus arxiv_subfields \
  --problem-taxonomy "$DATA/gold/problem_taxonomy.json" --method-taxonomy "$DATA/gold/method_taxonomy.json" \
  --output-dir "$OUT/build_index"
"$PY" scripts/build_index.py --data-dir "$DATA" --index-dir "$INDEX" --corpus taxobench --limit-papers 60 \
  --set builder.max_depth=1 --output-dir "$OUT/build_index_taxobench" "${SMALL[@]}"

step "4b. Temporal link predictor (App. B.5)"
"$PY" scripts/train_link_predictor.py --index-dir "$INDEX" --corpus arxiv_subfields \
  --set 'link_prediction.cutoffs=[2018-01-01,2019-01-01,2020-01-01,2021-01-01]' \
  --set link_prediction.horizon_years=1 --set link_prediction.epochs=5 --output-dir "$OUT/train_link_predictor"

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

step "6. Ask a question"
"$PY" scripts/ask.py --index-dir "$INDEX" --corpus arxiv_subfields --output-dir "$OUT/ask" \
  "How has bayesian optimization been used for cardinality estimation?"

step "Done: outputs under $OUT"
