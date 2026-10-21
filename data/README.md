# Data

This directory holds the datasets of the paper (Section 5, Appendix C.1–C.2).
Raw downloads go to `data/raw/`, preprocessed files to `data/processed/`;
both are git-ignored. Everything is produced by four commands:

```bash
python scripts/download_data.py                 # raw data -> data/raw/<name>/
python scripts/prepare_arxiv.py                 # arXiv sets + arXiv-Subfields
python scripts/prepare_fulltext.py --sources gcs --cache-dir data/fulltext/cache   # section chunks
python scripts/prepare_benchmarks.py            # benchmarks + corpus alignment report
```

Every script documents its flags in `--help`. `prepare_arxiv.py`,
`prepare_fulltext.py` and `prepare_benchmarks.py` take `--limit N` for quick
runs. `download_data.py` selects datasets with `--only` / `--all` and has
`--dry-run` and `--max-shards`. Unlike the experiment scripts, these
data-preparation scripts do not use `--config` / `--set` / `--output-dir`;
`prepare_fulltext.py` alone reads the chunk sizes from `--config` / `--set`.
Downloads go to `--root` (default `data/raw`). `prepare_arxiv.py` and
`prepare_benchmarks.py` write under `--out` (default `data/processed`), and
`prepare_fulltext.py` writes the chunk file next to its corpus. Every loader
in `scholarpath.data` also accepts pre-downloaded local files.

## 1. Datasets

| Dataset | Used for | Source | Size in the paper |
|---|---|---|---|
| **arXiv sets** | taxonomy construction vs. FULL (Sec. 5.2) | Kaggle `Cornell-University/arxiv` metadata snapshot | 8 primary categories x 2,000 papers |
| **arXiv-Subfields** | QA corpus for ResearchQA, ScholarQABench and the held-out hypothesis queries (Sec. 5.3, App. C.1) | same snapshot | 148,400 papers (September-2026 snapshot) |
| **TaxoBench** | construction (merged expert gold) and organization queries in Deep Research mode | github.com/KongLongGeFDU/TaxoBench | 72 surveys, 3,815 citations, 3,044 distinct papers |
| **CHIME** | organization queries (human-corrected hierarchies) | github.com/allenai/chime + HF `joe32140/chime-*` | 100 topics, 320 hierarchies, 2,146 studies |
| **ResearchQA** | evidence queries (rubric coverage, citations) | HF `realliyifei/ResearchQA`, split `test_mini` | 776 queries, 5,824 rubric items |
| **ScholarQABench** | evidence queries: ScholarQA-CS (rubrics) and ScholarQA-Multi (expert answers) | github.com/AkariAsai/ScholarQABench | 100 + 108 queries |
| S2ORC v2 / S2 papers (optional) | full text, Semantic Scholar ids of arXiv papers | Semantic Scholar Datasets API (API key) | – |

The eight arXiv primary categories are `cs.DB, math.OC, astro-ph.GA,
physics.med-ph, q-bio.PE, stat.AP, eess.SP, econ.EM` (seven top-level arXiv
groups: astro-ph and physics both belong to Physics).

## 2. Download

```bash
python scripts/download_data.py --list                 # what can be downloaded
python scripts/download_data.py --dry-run              # print the commands only
python scripts/download_data.py --only taxobench,chime # a subset
python scripts/download_data.py --all                  # incl. optional S2AG datasets (needs S2_API_KEY)
S2_API_KEY=... python scripts/download_data.py --only s2orc_v2 --max-shards 4
```

| name | how | needs |
|---|---|---|
| `taxobench`, `scholarqabench`, `chime`, `researchqa_repo` | `git clone` (pinned commits `49a1869`, `95e6fc5`, `35d852a`, `747a9a1`) | git |
| `researchqa` | `data/test_mini-00000-of-00001.parquet` from HF | network access to huggingface.co (`HF_TOKEN` optional); reading parquet needs `pandas` + `pyarrow` |
| `chime_annotations` | the three HF datasets `joe32140/chime-{parent-child-relation,sibling-coherence,claim-category}` | huggingface.co; `huggingface_hub` is used if installed, plain HTTPS otherwise |
| `arxiv` | `kaggle datasets download -d Cornell-University/arxiv --unzip` (or `kagglehub`) | Kaggle credentials (`KAGGLE_API_TOKEN`, `~/.kaggle/access_token` or `~/.kaggle/kaggle.json`) |
| `s2orc_v2`, `s2_papers` | Semantic Scholar Datasets API (`/release/latest`, `/release/<id>/dataset/<name>`) | `S2_API_KEY` |

Downloads are idempotent (a finished dataset leaves
`data/raw/<name>/.scholarpath_download.json` and is skipped unless
`--force`; S2AG shards resume file by file). When a host is unreachable
(Hugging Face, Kaggle and arxiv.org are blocked by some proxies) the script
prints manual instructions; place the files as below and continue.

Manual alternatives:

* ResearchQA: `python -c "from datasets import load_dataset; load_dataset('realliyifei/ResearchQA', split='test_mini').to_json('data/raw/researchqa/test_mini.jsonl', force_ascii=False)"` (JSON/JSONL exports are accepted; `scholarpath.data.load_researchqa_hf()` returns the same records in Python).
* CHIME annotations: CSV / JSONL / parquet exports of the three datasets, or `datasets.save_to_disk` directories, named `chime-<task>*` under `data/raw/chime_annotations/`.
* arXiv: download the Kaggle dataset in a browser and put `arxiv-metadata-oai-snapshot.json` (or `arxiv.zip`, or a `.gz`) into `data/raw/arxiv/`. (The public GCS copy `gs://arxiv-dataset/metadata-v5/arxiv-metadata-oai.json` is a stale August-2020 snapshot.)

## 3. Expected raw layout

```
data/raw/
├── arxiv/arxiv-metadata-oai-snapshot.json      # or arxiv.zip / .json.gz
├── taxobench/dataset/data.jsonl
├── scholarqabench/data/scholarqa_cs/{test_configs_snippets.json, output_snippets.jsonl}
├── scholarqabench/data/scholarqa_multi/human_answers.json
├── chime/resources/{raw_source_data.csv, raw_generated_hierarchy.csv, data_split.csv}
├── chime_annotations/chime-{parent-child-relation,sibling-coherence,claim-category}/...
├── researchqa/data/test_mini-00000-of-00001.parquet   # or test_mini.json[l] (JSON array or JSON Lines)
├── s2orc_v2/<release>/s2orc_v2-*.jsonl.gz      # optional
└── s2_papers/<release>/papers-*.jsonl.gz       # optional
```

## 4. Preprocessing

### arXiv sets and arXiv-Subfields (`scripts/prepare_arxiv.py`)

One streaming pass over the snapshot (`scholarpath.data.arxiv`):

* `paper_id` = `arxiv_id` = identifier without version (old-style ids keep their archive: `math/0601001`);
  title / abstract whitespace-normalised; `categories` split on spaces; **primary category = first entry**
  (App. C.2); authors from `authors_parsed`; `date` = creation date of the first version (RFC 2822 -> ISO), `year`.
* **Withdrawn papers are excluded** (`scholarpath.data.arxiv.is_withdrawn`; the paper does not define the
  rule): comments or abstract containing a notice whose subject is the paper itself ("This paper has been
  withdrawn by the author", "withdrawn by arXiv administrators", "We withdraw this paper"), or starting with
  a bare notice ("Withdrawn", "Withdrawn due to ..."). Partial or unrelated withdrawals ("Theorem 3
  withdrawn due to a gap", "Withdrawn from ICML 2020; accepted at NeurIPS") do not exclude a paper.
* **arXiv sets**: 2,000 papers per category sampled uniformly with a fixed seed (`--seed`, default 0).
  Sampling is bottom-k by a seeded hash of the paper id, i.e. a uniform sample without replacement that does
  not depend on the order of the snapshot.
* **arXiv-Subfields**: every non-withdrawn paper whose primary category is one of the eight. Use
  `--until YYYY-MM-DD` (inclusive, first-version date) to freeze a snapshot; the paper's September-2026
  snapshot gives 148,400 papers.

### Full text (`scripts/prepare_fulltext.py`)

Section chunks of the paper-detail level (Sec. 2.1). Sources in configurable order
(`--sources local,s2orc,gcs,arxiv`):

* `local` (`--local-dir`): `<id>.pdf`, `<id>.tex`, `<id>.txt`, e-print archives `<id>.tar.gz`, or LaTeX
  directories `<id>/` (old-style ids use `_` for `/`);
* `s2orc` (`--s2orc`): S2ORC v2 shards, joined on `openaccessinfo.externalids.ArXiv`; sections from the
  `section_header` spans (JSON-encoded annotation strings, possibly with string offsets);
* `gcs`: public PDF mirror `https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf/YYMM/<id>vN.pdf`
  (old-style: `.../arxiv/<archive>/pdf/YYMM/<num>vN.pdf`), no authentication;
* `arxiv`: arxiv.org e-print (LaTeX, split by `\section`) with PDF fallback, one request per 3 s.

PDFs are converted with PyMuPDF (`pip install pymupdf`) and split by regex section headings
(`scholarpath.text.split_sections`); long sections are chunked to `IndexConfig.chunk_max_tokens` (512).
References, acknowledgements and front matter are dropped (the abstract is the paper level). Output:
`chunks.jsonl.gz` with `Chunk` records `{"chunk_id": "<paper_id>#<i>", "paper_id", "heading", "text", "order"}`
plus `chunks.stats.json` (per-source attempts / successes). Runs are resumable (`chunks.state.jsonl`);
papers without full text keep only their abstract.

### Benchmarks (`scripts/prepare_benchmarks.py`)

* **ResearchQA**: one `Query` per record (`kind="evidence"`, `benchmark="researchqa"`); rubric items keep
  their type lists (`RubricItem.kind = "comparison,example"`, `metadata["rubric_types"]`); metadata `id`,
  `domain`, `subdomain`, `field`, `date` (survey date; informational only, since queries search the whole
  corpus with no date cutoff, as in the paper). Gold papers = the cited papers of
  `Citation`-type rubric items (`citation_metadata`; titles parsed from the item text if it is missing),
  aligned to arXiv-Subfields.
* **ScholarQA-CS**: one query per unique question (the first of the 10 double-annotated cases, as the official
  scorer; the other case is kept in `metadata["alternate_cases"]`), full `metric_config` kept for
  ScholarQABench's `scripts/rubric_eval.py`; rubric = `other_properties` (weights, evidence quotes). Gold papers are
  reconstructed from the evidence quotes.
* **ScholarQA-Multi**: one query per expert answer (`reference_answer`), gold = the `ctxs` citations
  (citation markers in the answers are **0-based**).
* **TaxoBench**: pooled corpus de-duplicated by normalized title (ids `tb-<sha1>`), per-survey gold trees
  (`gold/<id>.json`), the merged expert gold (`gold_taxonomy.json`: survey roots relabelled by their topic
  under a common root; papers are extents, not nodes), and one organization query per survey
  (`metadata = {"task": "organization", "survey_id", "topic", ...}`, gold = the survey's papers).
* **CHIME**: study corpus (ids `pmid:<PubMed id>`), human-corrected gold hierarchy per topic
  (`gold/<review_id>.json`; root = review title, children = the annotated aspect hierarchies) and one
  organization query per topic (`metadata = {"task": "organization", "review_pmid", ...}`).

## 5. Processed layout

```
data/processed/
├── arxiv_sets/<category>.jsonl.gz          # 8 x 2,000 papers
├── arxiv_subfields/papers.jsonl.gz         # arXiv-Subfields (+ stats.json)
├── arxiv_subfields/chunks.jsonl.gz         # section chunks (+ chunks.stats.json, chunks.state.jsonl)
├── researchqa/queries.jsonl
├── scholarqabench/{cs,multi}.jsonl
├── taxobench/{papers.jsonl.gz, gold_taxonomy.json, gold/<survey_id>.json, queries.jsonl, stats.json}
├── chime/{papers.jsonl.gz, gold/<review_id>.json, queries.jsonl, stats.json}
├── heldout/...                              # App. C.1 (scripts of the held-out benchmark)
└── coverage_report.json
```

Load them with `scholarpath.data.Corpus.load(...)`, `load_queries(...)` and
`scholarpath.taxonomy.tree.Taxonomy.load(...)`.

## 6. Corpus alignment and coverage (App. C.2)

Gold citations are aligned to the target corpus by `scholarpath.data.alignment.PaperAligner`, following
App. C.2 ("by arXiv ID, Semantic Scholar ID, or normalized title"): arXiv id (from ids / URLs /
`10.48550/arXiv.*` DOIs, version stripped) -> Semantic Scholar paperId or CorpusId (`--s2-papers` attaches
them to arXiv papers from the S2AG `papers` dataset) -> DOI / PMID -> exact normalized title (plus a
LaTeX-math-unwrapped variant, so `$k$-means` equals `k-means`).
**Unaligned citations stay in the query's denominator** (`Query.unaligned_gold`, counted by
`Query.num_gold`), so Recall@B / NDCG@B reflect end-to-end corpus coverage.

`--fuzzy` enables an opt-in near-duplicate title stage for typos and spelling variants (`difflib` ratio >=
`--fuzzy-threshold` 0.92, identical version tokens such as numbers and roman numerals, every differing word a
spelling variant of a word of the other title, years within 2 and a shared first-author surname when known;
no containment rule, so "Attention is all you need" never matches "Attention Is All You Need in Speech
Separation", nor "Qwen2 Technical Report" "Qwen Technical Report"). Its matches are reported separately
(`by_method.fuzzy_title`). It is off by default because App. C.2 aligns by identifiers and normalized title
only, and any fuzzy rule risks aligning a citation of a paper missing from the corpus to a different paper.

ResearchQA and ScholarQABench need the alignment corpus; `prepare_benchmarks.py` stops if it is missing
unless `--allow-unaligned` is given. `coverage_report.json` is updated per benchmark, so benchmarks can be
prepared in separate runs.

| Benchmark | Paper (App. C.2) |
|---|---|
| TaxoBench | 100% (3,044 pooled papers) |
| CHIME | 100% (2,146 pooled studies) |
| ScholarQABench | 96.8% (2,375 of 2,454 expert-cited papers aligned to arXiv-Subfields) |
| ResearchQA | 95.6% (10,830 of 11,328 survey-cited papers aligned to arXiv-Subfields) |

`prepare_benchmarks.py` writes the reproduced numbers to `coverage_report.json`.
