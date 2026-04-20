# IR Challenge — Citation Recommendation Pipeline

**Task:** Academic paper retrieval (citation recommendation)  
**Corpus:** 20,000 papers | **Queries:** 100 val (with qrels) + 100 held-out  
**Primary metric:** NDCG@10 | **Secondary:** Recall@100  
**Best result:** Val NDCG@10 = **0.7597** | Codabench (held-out) = **0.73**

---

## Setup

### Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn rank-bm25 sentence-transformers \
            transformers adapters torch tqdm jupyter
```

### Data layout

```
ir_challenge/
├── data/
│   ├── corpus.parquet          # 20K papers (doc_id, title, abstract, full_text, domain, ta)
│   ├── queries_1.parquet       # 100 val queries
│   ├── queries.parquet         # 100 held-out queries
│   ├── qrels_1.json            # val ground truth  {qid: [doc_id, ...]}
│   └── embeddings/
│       ├── sentence-transformers_all-MiniLM-L6-v2/
│       │   ├── corpus_embeddings.npy
│       │   └── query_embeddings.npy    # val queries
│       └── new_queries_minilm/
│           └── query_embeddings.npy    # held-out queries
├── submissions/
│   ├── bge_large_corpus_emb.npy
│   ├── bge_large_query_emb.npy         # val
│   ├── bge_large_heldout_query_emb.npy
│   └── specter_prox_embed/
│       ├── corpus_embeddings.npy
│       ├── queries_1_embeddings.npy    # val
│       └── queries_embeddings.npy      # held-out
├── domain_confusion_matrix_normalized.xls
└── notebooks/
    └── bm25.ipynb              # ← main pipeline notebook
```

---

## Running the pipeline

Open `notebooks/bm25.ipynb` in Jupyter and run cells top-to-bottom.

### Val mode (default — has ground truth)

Cell 2 defaults are already set for val:

```python
HAS_QRELS    = True
QUERIES_PATH = DATA_DIR / 'queries_1.parquet'
QRELS_PATH   = DATA_DIR / 'qrels_1.json'
```

Run all cells. §12 prints val NDCG@10 at each intermediate step; §9 prints the final score.

### Held-out mode (submission)

In cell 2, change:

```python
HAS_QRELS    = False
QUERIES_PATH = DATA_DIR / 'queries.parquet'
# SPECTER_QUERY_EMB = Path('../specter_prox_embed/queries_embeddings.npy')
```

Re-run from cell 6 downward. §9.4 saves the submission to `submissions/`.

---

## Notebook structure

### §1 — TF-IDF

Builds unigram, bigram, and trigram TF-IDF models on **full text** (corpus) and **TA** (title+abstract for queries). Results cached to `submissions/`.

| Model | NDCG@10 | Recall@100 |
|---|---|---|
| TF-IDF uni (FT) | 0.5617 | **0.8476** |
| TF-IDF bi (FT) | **0.5728** | 0.8375 |

Full-text consistently outperforms title+abstract. TF-IDF-uni FT has the best Recall@100 (most important for pool construction).

### §2 — BM25 Variants

Fits BM25, BM25F, BM25+, and BM25L on full text with grid search over `k1` and `b`. Best variant: **BM25L (k1=2.5, b=0.9, delta=0.25)**.

Best params are in `BEST_PARAMS` dict at the top of §2.7 — update these if you re-run the grid search.

### §3.5 — SPECTER2 Proximity

Loads pre-computed `allenai/specter2_base` + proximity adapter embeddings. Embeddings must be in `specter_prox_embed/`. Standalone NDCG@10 ≈ 0.52; critical ensemble member.

### §3 — Comparison

Side-by-side Recall@100 table across all sparse models. The best per-domain model feeds into §4 routing.

### §4 — Domain Routing

Per-domain best model selection using the confusion matrix. Produces `submission_sparse_routed_top100.json`. Evaluated on val when `HAS_QRELS=True`.

### §5 — Recall@k ceiling

Recall at k=10/50/100/150 for every model and the routed ensemble.

### §6 — Hybrid Domain Routing (Sparse + SPECTER via RRF)

Fuses each sparse model with SPECTER2-Prox using standard RRF (k=60). Produces one hybrid submission per sparse model, then routes per domain.

```python
score(doc) = 1/(k + rank_sparse) + 1/(k + rank_dense)
```

### §7 — Domain-Confusion Reranking

Loads `domain_confusion_matrix_normalized.xls`. After RRF fusion, sorts candidates by `(-confusion_weight, original_rank)` — same-domain docs come first, preserving relative order within each tier. Business domain is skipped (only 2 val queries → unreliable weights).

**+0.003 NDCG@10 vs pure ensemble.**

### §8 — 4-way RRF + Domain Reranking

Fuses four lists (sparse-routed + BGE + SPECTER2 + BM25F-FT) with RRF k=3, then applies domain-confusion reranking.

Requires `submissions/submission_bge_dense_top100.json` (generated from `hybrid_pipeline.ipynb`).

**Val NDCG@10 = 0.727**

### §9 — Nested RRF with Per-Domain Model Exclusion ← best pipeline

The full production pipeline. **Val NDCG@10 = 0.7597 | Codabench ≈ 0.73.**

#### Architecture

```
Sparse inputs (domain-filtered, min_pool=300):
  BM25L-FT          → s_bm25l
  TF-IDF-uni ⊕ SPECTER2 (RRF k=5) → s_tuni   (domain filtered)
  TF-IDF-bi  ⊕ SPECTER2 (RRF k=5) → s_tbi    (domain filtered)

Dense inputs (unfiltered):
  BGE-large-en-v1.5   → scores_bge
  SPECTER2-proximity  → scores_specter_prox
  MiniLM-L6-v2        → scores_minilm

Inner RRF (k=1):   BM25L + BGE + MiniLM
                           ↓
Outer RRF (k=2):   inner + TF-IDF-uni + TF-IDF-bi + SPECTER2
                           ↓
Domain hard-sort (skip Business)
```

**Per-domain model exclusion:** for each query's domain, models whose NDCG@10 < 70% of the domain best are excluded from fusion. This prevents bad models (e.g., BGE/MiniLM/SPECTER2 score 0.00 for Philosophy) from diluting the good ones (TF-IDF-bi = 0.61 for Philosophy).

**MiniLM PRF:** domain-adaptive Rocchio feedback — expand query with top-5 BGE docs. β=0.8 for weak domains (CS, Biology, Medicine, Philosophy, Art, Engineering), β=0.0 for strong domains (Geology, Mathematics, etc.), β=0.6 otherwise.

**BGE PRF:** β=0.06, seeded from the baseline ensemble (before PRF) top-5 docs.

#### Files read by §9

| File | Purpose |
|---|---|
| `submissions/bge_large_corpus_emb.npy` | BGE corpus embeddings (1024-dim) |
| `submissions/bge_large_query_emb.npy` | BGE val query embeddings |
| `submissions/bge_large_heldout_query_emb.npy` | BGE held-out query embeddings |
| `data/embeddings/.../corpus_embeddings.npy` | MiniLM corpus embeddings (384-dim) |
| `data/embeddings/.../query_embeddings.npy` | MiniLM query embeddings |

#### Output

Saved to `submissions/submission_nested_rrf_top{TOP_K}.json`.

---

## Key configuration

| Variable | Default | Description |
|---|---|---|
| `TOP_K` | 100 | Submission cutoff (Codabench expects 100) |
| `CANDIDATE_POOL` | 150 | Internal pool before trimming |
| `HAS_QRELS` | True | Set False for held-out |
| `QUERIES_PATH` | `queries_1.parquet` | Change to `queries.parquet` for held-out |

Caching: results are written to `submissions/` as `.npy` (score matrices) and `.json` (ranked lists). Re-running a cell loads from cache if the file exists — delete the cache file to force recomputation.

---

## Score trajectory

| Stage | Val NDCG@10 |
|---|---|
| BM25L alone | 0.52 |
| + BGE-large dense | 0.55 |
| + SPECTER2 proximity | 0.61 |
| + Nested RRF ensemble | 0.63 |
| + Domain reranking | 0.75 |
| + Sparse domain filter | ~0.75 |
| + Model exclusion (thresh=0.70) | 0.754 |
| + Dense PRF (domain-adaptive β) | **0.7597** |
