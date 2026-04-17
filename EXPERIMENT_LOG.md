# IR Challenge — Complete Experiment Log

**Task:** Academic paper retrieval (citation recommendation)  
**Corpus:** 20,000 papers (title + abstract + full text)  
**Queries:** 100 validation queries (with qrels) + 100 held-out queries  
**Primary metric:** NDCG@10  
**Secondary metric:** Recall@100 (measures candidate pool quality for reranking)  
**Leaderboard:** Codabench (held-out) — best achieved: **0.73** (nested RRF + model exclusion + sparse domain filter)  
**Val best:** 0.7597 (+ Dense PRF with domain-adaptive β)

---

## Table of Contents

1. [Task Analysis](#1-task-analysis)
2. [Phase 1 — Sparse Models](#2-phase-1--sparse-models)
3. [Phase 2 — Domain Routing (Sparse)](#3-phase-2--domain-routing-sparse)
4. [Phase 3 — Dense Models](#4-phase-3--dense-models)
5. [Phase 4 — Hybrid Ensemble (RRF)](#5-phase-4--hybrid-ensemble-rrf)
6. [Phase 5 — Domain Reranking with Confusion Matrix](#6-phase-5--domain-reranking-with-confusion-matrix)
7. [Phase 6 — Year Filtering](#7-phase-6--year-filtering)
8. [Phase 7 — Per-Domain Ensemble Config](#8-phase-7--per-domain-ensemble-config)
9. [Phase 8 — Cross-Encoder Reranking](#9-phase-8--cross-encoder-reranking)
10. [Phase 9 — New Dense Models (E5-large, SPECTER2 adhoc, e5-mistral-7b)](#10-phase-9--new-dense-models)
11. [Phase 10 — Pseudo-Relevance Feedback](#11-phase-10--pseudo-relevance-feedback)
12. [Phase 11 — Domain Weight Pre-Boost](#12-phase-11--domain-weight-pre-boost)
13. [Phase 12 — Domain-Filtered Retrieval (Sparse Only)](#13-phase-12--domain-filtered-retrieval-sparse-only)
14. [Phase 13 — Per-Domain Model Weighting & Exclusion](#14-phase-13--per-domain-model-weighting--exclusion)
15. [Current Best Result](#15-current-best-result)
16. [What Didn't Work — Summary](#16-what-didnt-work--summary)
17. [What Worked — Summary](#17-what-worked--summary)
18. [Remaining Gap & Next Steps](#18-remaining-gap--next-steps)

---

## 1. Task Analysis

### Dataset properties discovered through experimentation

| Property | Finding |
|---|---|
| Corpus size | 20,000 academic papers |
| Query type | Paper title + abstract (paper-as-query) |
| Relevance definition | Primarily **citation relationships**, not pure topical similarity |
| Year distribution | 68.5% of relevant docs published **after** the query paper |
| Domain distribution | Medicine (24%), CS (18%), Biology (20%), Chemistry (10%), Physics (6%), others |
| Queries per domain | ~14 on average; Business has only 2 |

### Key insight (discovered late)

The task is **citation recommendation**, not general semantic search. Models trained on web-search (MS-MARCO, BEIR) or pure semantic similarity are systematically disadvantaged. The best models here are ones trained on academic paper proximity data (SPECTER2, BGE with scientific text fine-tuning).

---

## 2. Phase 1 — Sparse Models

### Models tested

All models evaluated on the validation set (100 queries). Text fields: **TA** = title + abstract, **FT** = full text. SPECTER Proximity = dense retrieval with SPECTER2 proximity adapter included here for side-by-side comparison.

| Model | Text field | NDCG@10 | Recall@10 | Recall@50 | Recall@100 |
|---|---|---|---|---|---|
| TF-IDF unigram | TA | 0.4841 | 0.4445 | 0.6552 | 0.7512 |
| TF-IDF bigram | TA | 0.5010 | 0.4512 | 0.6828 | 0.7480 |
| TF-IDF trigram | TA | 0.4941 | 0.4549 | 0.6830 | 0.7396 |
| TF-IDF unigram | FT | 0.5617 | 0.5423 | 0.7835 | **0.8476** |
| **TF-IDF bigram** | **FT** | **0.5728** | **0.5591** | 0.7786 | 0.8375 |
| BM25 (k1=1.5, b=0.75) | TA | 0.4265 | 0.3928 | 0.5934 | 0.6822 |
| BM25 (k1=1.5, b=0.75) | FT | 0.5195 | 0.4714 | 0.7210 | 0.7889 |
| BM25F (field weights) | TA+FT | 0.4509 | 0.4119 | 0.6020 | 0.6865 |
| BM25F | FT | 0.4924 | 0.4433 | 0.6701 | 0.7583 |
| BM25+ (delta=1.0) | FT | 0.5175 | 0.4747 | 0.7158 | 0.7845 |
| BM25L (delta=0.5) | FT | 0.5217 | 0.4767 | 0.7230 | 0.7935 |
| SPECTER2 Proximity | — | 0.5055 | 0.4810 | 0.7073 | 0.7852 |

### Key findings

- **TF-IDF bi (FT) is the best single sparse model** by NDCG@10 (0.5728), Recall@10, MRR, and MAP
- **TF-IDF uni (FT) has the highest Recall@100** (0.8476) — the most important metric for recall-first pool construction
- Full text (FT) consistently beats title+abstract (TA) across all model families
- BM25 variants (BM25+, BM25L) are competitive but do not beat TF-IDF FT on either primary metric
- Grid search was run over k1 ∈ {1.0, 1.2, 1.5, 2.0} and b ∈ {0.5, 0.6, 0.75} for each BM25 variant

### What worked

TF-IDF bigram FT is the strongest standalone sparse model (NDCG@10 = 0.5728). TF-IDF unigram FT provides the best recall ceiling for pool construction (Recall@100 = 0.8476). BM25L FT is the strongest BM25 variant but lags both TF-IDF FT configurations. 

---

## 3. Phase 2 — Domain Routing (Sparse + SPECTER Proximity)

### Hypothesis

Different domains may benefit from different sparse models or SPECTER Proximity; per-domain routing could lift Recall@100 beyond what any single model achieves.

### Per-domain Recall@100 and best model

| Domain | TF-IDF uni (TA) | TF-IDF bi (TA) | TF-IDF uni (FT) | TF-IDF bi (FT) | BM25 (FT) | BM25+ (FT) | BM25L (FT) | SPECTER Prox | **Best model** |
|---|---|---|---|---|---|---|---|---|---|
| Art | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | tied |
| Biology | 0.6900 | 0.6674 | 0.8291 | **0.8418** | 0.7532 | 0.7253 | 0.7655 | 0.7165 | TF-IDF bi (FT) |
| Business | 0.8000 | **0.9667** | 0.9667 | 0.9667 | 0.9667 | 0.9667 | 0.9667 | 0.8000 | TF-IDF bi (TA) |
| Chemistry | 0.7355 | 0.7121 | **0.8136** | 0.7833 | 0.7533 | 0.7710 | 0.7733 | 0.7388 | TF-IDF uni (FT) |
| Computer Science | 0.6747 | 0.7284 | **0.8187** | 0.7854 | 0.7028 | 0.7005 | 0.7028 | 0.7978 | TF-IDF uni (FT) |
| Economics | 0.7647 | 0.7647 | 0.7647 | 0.8235 | 0.7647 | 0.7647 | 0.7647 | **0.9412** | **SPECTER Prox** |
| Engineering | **0.7500** | 0.6250 | 0.6250 | 0.5000 | 0.6250 | 0.6250 | 0.6250 | 0.6250 | TF-IDF uni (TA) |
| Environmental Science | **0.9762** | 0.9762 | 0.9762 | 0.9762 | 0.9524 | 0.9524 | 0.9524 | 0.8095 | TF-IDF uni (TA) |
| Geography | 0.9000 | **1.0000** | 1.0000 | 1.0000 | 0.9000 | 0.9000 | 0.9000 | 1.0000 | TF-IDF bi (TA) |
| Geology | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | tied |
| History | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | tied |
| Materials Science | 0.8304 | 0.8391 | **0.8855** | 0.8855 | 0.8725 | 0.8725 | 0.8725 | 0.8464 | TF-IDF uni (FT) |
| Mathematics | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7500 | TF-IDF (any FT/TA) |
| Medicine | 0.7268 | 0.7200 | **0.7982** | 0.7755 | 0.7347 | 0.7347 | 0.7347 | 0.7823 | TF-IDF uni (FT) |
| Philosophy | 0.0000 | 0.0000 | **0.5000** | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | TF-IDF uni/bi (FT) |
| Physics | 0.7737 | 0.7737 | 0.9258 | **0.9333** | 0.9035 | 0.9035 | 0.9035 | 0.8904 | TF-IDF bi (FT) |
| Political Science | 0.5000 | 0.5000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | TF-IDF uni (FT) |
| Psychology | **1.0000** | 0.9167 | 1.0000 | 1.0000 | 0.9250 | 0.9250 | 0.9250 | 1.0000 | TF-IDF/SPECTER tied |
| Sociology | 0.5246 | 0.5902 | 0.7705 | **0.8197** | 0.7377 | 0.7377 | 0.7377 | 0.7869 | TF-IDF bi (FT) |

### Notable routing observations

- **Economics**: SPECTER Proximity dominates with 0.9412 Recall@100 — a massive +0.118 over the next best (TF-IDF bi FT at 0.8235). Economics papers share dense co-citation networks that proximity embeddings capture but TF-IDF misses.
- **Mathematics / Geology / History / Geography**: Perfect or near-perfect recall for all models — these domains have strong terminological overlap between query and gold documents.
- **Philosophy**: Only FT models recover anything (0.5000); TA models all score 0.0000, and SPECTER Prox also fails (0.0000). The 0.5 ceiling reflects the fundamental vocabulary mismatch problem (see Phase 18 failure analysis).
- **Computer Science**: SPECTER Prox (0.7978) is competitive with TF-IDF uni FT (0.8187), indicating citation proximity helps here too.
- **BM25L (FT)** is never the best model in any domain — it is consistently outperformed by TF-IDF FT variants across all non-trivial domains.

### Result

The domain routing signal is real but sparse-only routing does not improve the overall submission — most queries are in domains where TF-IDF FT already wins, and the gains in Economics/CS are offset by noise. Routing becomes valuable only at the **ensemble level** (Phase 6+), where SPECTER Proximity is added as a separate model signal rather than a sparse-level replacement.

### Status: Informative — routing used in ensemble design, not standalone sparse submission

---

## 4. Phase 3 — Dense Models

### Models tested

| Model | Dimensions | Val NDCG@10 | Val Recall@100 | Notes |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | ~0.42 | ~0.68 | Provided in dataset |
| SPECTER2 (proximity adapter) | 768 | ~0.52 | ~0.77 | Ta-encoded queries |
| SPECTER2 (proximity, FT queries) | 768 | ~0.52 | ~0.77 | Fine-tuned query encoding |
| SPECTER2 (adhoc_query adapter) | 768 | ~0.50 | ~0.75 | Wrong adapter for citation task |
| BGE-large-en-v1.5 | 1024 | ~0.55 | ~0.80 | Fine-tuned on scientific text |
| E5-large-v2 | 1024 | 0.5053 | ~0.74 | General semantic search model |
| e5-mistral-7b-instruct | 4096 | 0.0309 | — | **Failed** — task mismatch |

### SPECTER2 adapter choice

Two adapters were tested:
- **`proximity`** (trained on co-citation data): better fit for citation recommendation — **used in ensemble**
- **`adhoc_query`** (trained for paper-as-query IR): lower performance on this task

### e5-mistral-7b-instruct failure (3 debugging runs)

**Run 1 failure — 4-bit quantization destroys embeddings (NDCG = 0.0342)**
- Used `BitsAndBytesConfig(load_in_4bit=True)` to fit on GPU
- 4-bit quantization completely destroys the embedding geometry for sentence embedding models
- Fix attempted: switch to fp16 (`torch_dtype=torch.float16`)

**Run 2 failure — missing EOS token (NDCG = 0.0258)**
- e5-mistral uses **last-token pooling** — the model pools the last token's hidden state
- Without appending EOS before padding, the last token is a random mid-sentence token, not the special embedding token
- The official e5-mistral encoding requires: truncate to `max_length-1`, append EOS, then pad
- Fix attempted: manually append `tokenizer.eos_token_id` before padding

**Run 3 failure — fundamental task mismatch (NDCG = 0.0309)**
- Embeddings were verified correct (norm ≈ 1.0, similarity distribution reasonable, max = 0.81)
- Analysis: for a typical query, relevant docs ranked at positions 12,910 and 9,545 out of 20,000
- Root cause: e5-mistral optimizes **topical NLU similarity** (trained on web search / NLI data)
- Qrels define relevance by **citation relationships** which are broader than topical similarity
- Papers cite each other across decades, tangentially related topics, and for methodology — not just because they're topically similar
- **Conclusion: e5-mistral fundamentally cannot work for citation recommendation regardless of engineering fixes**

### Key insight about dense models

Models trained on academic paper citation proximity (SPECTER2, BGE with scientific fine-tuning) beat models trained on web search or general semantic tasks. The domain mismatch is the dominant factor.

---

## 5. Phase 4 — Hybrid Ensemble (RRF)

### Reciprocal Rank Fusion formula

```
score(doc) = Σ_list  1 / (k + rank(doc, list))
```

### Architecture: two-level nested RRF

```
Inner RRF (k=1):  [BM25L, BGE-large, MiniLM]
                         ↓
Outer RRF (k=2):  [Inner, BGE-large, BM25L, Hybrid-TF-IDF-uni, Hybrid-TF-IDF-bi, SPECTER2-prox]
```

Where **Hybrid lists** are themselves RRF-fused:
```
Hybrid-TF-IDF-uni = RRF_fuse(TF-IDF-uni-FT, SPECTER2-prox,  k=5)
Hybrid-TF-IDF-bi  = RRF_fuse(TF-IDF-bi-FT,  SPECTER2-prox,  k=5)
```

### RRF k parameter tuning

Grid search over `k_inner ∈ {1, 2, 3}` and `k_outer ∈ {1, 2, 3}`:

| Config | Val NDCG@10 |
|---|---|
| k_inner=1, k_outer=1 | 0.7441 |
| k_inner=1, k_outer=2 | **0.7462** |
| k_inner=2, k_outer=2 | 0.7438 |
| k_inner=3, k_outer=3 | 0.7410 |

Also tuned `k_hybrid` (for TF-IDF hybrid fusing):

| k_hybrid | Val NDCG@10 |
|---|---|
| k=3 | 0.7448 |
| k=4 | 0.7455 |
| **k=5** | **0.7462** |
| k=6 | 0.7451 |

### Best pure ensemble (before domain reranking)

**k_hybrid=5, k_inner=1, k_outer=2 → Val NDCG@10 = 0.7462**

### What worked

- Two-level nesting (inner + outer) outperforms flat RRF
- Lower k values (1–2) work better than the standard k=60 because the corpus is small (20K docs)
- BGE appears twice in the outer list (once via inner, once direct) — intentional double-weighting that helps
- Hybrid TF-IDF fused with SPECTER2 adds diversity

---

## 6. Phase 5 — Domain Reranking with Confusion Matrix

### Hypothesis

Academic papers tend to cite within their own domain. A paper about Medicine is more likely relevant to a Medicine query than a CS query. The domain confusion matrix captures cross-domain citation rates.

### Implementation

Built `domain_confusion_matrix_normalized.xls`:
- Rows: query domain
- Columns: corpus document domain  
- Values: normalized citation rate from query domain → corpus domain

For each query (except Business domain), after RRF fusion:
1. Look up the query's domain
2. Look up the weight row for that domain from the confusion matrix
3. Score each candidate: `(domain_weight, original_rank)`
4. Sort by `(-domain_weight, original_rank)` — i.e., prefer domain-matched docs but preserve original order within same domain tier

### Business domain skip

Business domain had only 2 queries in val. Its confusion matrix row was unreliable — applying domain weights hurt. Skipped with `SKIP = {'Business'}`.

### Result

| Config | Val NDCG@10 |
|---|---|
| Pure ensemble (hk5, k1k2) | 0.7462 |
| + Domain reranking (all domains) | 0.7471 |
| **+ Domain reranking (skip Business)** | **0.7493** |

**+0.0031 improvement from domain reranking.**

### Relaxation experiments (CS, Biology, Medicine)

Hypothesis: CS/Bio/Med papers cite themselves frequently, so heavy domain weighting might hurt.
- Soft weights (0.25–1.0 range) for these three domains
- Result: all variants produced identical 0.7493

The confusion matrix for CS/Bio/Med already preserves ensemble order — these domains' citation patterns are diverse enough that domain weighting doesn't change the top-10 ordering.

### What worked

Domain reranking with confusion matrix, skipping Business, gives a small but consistent improvement.

---

## 7. Phase 6 — Year Filtering

### Hypothesis

If a query paper was published in year Y, papers published after Y cannot have been cited by it. Filter them out to improve precision.

### Experiment

- Computed publication year for all corpus papers and queries
- For each query, removed corpus papers published **after** the query paper's year
- Evaluated on val

### Result: **Failed catastrophically**

| Config | Val NDCG@10 |
|---|---|
| Baseline | 0.7493 |
| After year filtering | ~0.40 |

**Root cause:** 68.5% of relevant docs (qrels) are published **after** the query paper. The task is not "find papers this paper cited" — it's a broader "find related papers," and many are newer papers that cite the same works or extend the query paper's ideas. Year filtering removes the majority of relevant documents.

### Lesson

Do not assume citation direction = temporal direction. The qrels represent a symmetric or bidirectional relevance relationship.

---

## 8. Phase 7 — Per-Domain Ensemble Config

### Hypothesis

Different domains might benefit from different inner/outer RRF k values and different sets of retrievers in the ensemble.

### Implementation

- Grid search over subsets of retrievers and k values, separately per domain
- Domains with < 4 queries grouped as "Other" to avoid overfitting
- Built per-domain DOMAIN_CONFIG assigning the best config to each domain
- Applied different ensemble configs per query domain

### Val result: 0.7493 → **0.7634** (+0.0141)

Looked promising on val. Submitted to Codabench.

### Held-out result: **0.7078** (severe degradation)

**Root cause: overfitting to val set with tiny domain samples.**
- Average ~14 queries/domain; some domains have 1–3 queries
- With 1 query in "Economics" or "Geology", any config achieves either 0.0 or 1.0 on that domain — pure luck
- The "best" per-domain config is essentially memorizing the val queries, not generalizing

### Lesson

Never optimize per-domain with fewer than ~20 queries per domain. 100 total queries across 16+ domains is far too small for per-domain hyperparameter search.

---

## 9. Phase 8 — Cross-Encoder Reranking

### Idea

Use a cross-encoder (pairwise query-document scorer) to rerank the ensemble's top-100 candidates. Cross-encoders typically achieve much higher precision than bi-encoders.

### Local evaluation (MS-MARCO MiniLM cross-encoder)

```
Baseline NDCG@10 (ensemble top-100): 0.7493
Reranked NDCG@10 (ms-marco-MiniLM):  0.4381
Delta: -0.3112
```

The MS-MARCO cross-encoder **catastrophically hurt** performance.

### Kaggle evaluation (BGE-reranker-large)

Ran `BAAI/bge-reranker-large` on T4 GPU on Kaggle, reranked held-out top-100.

**Codabench score: 0.29** (vs baseline ~0.72)

### Root cause

Both cross-encoders were trained on **MS-MARCO** — a web search dataset with short keyword queries and web page passages. Our task has:
- Queries: 300–2000 word academic paper abstracts
- Documents: 300–5000 word academic papers
- Relevance: citation relationships, not web search relevance

The cross-encoder has never seen paper-to-paper citation scoring. It scores our academic pairs using web-search relevance signals, completely reordering the already-good ensemble rankings.

### Analysis of why baseline was already good

For 97/100 val queries, the ensemble already had the first relevant document in the top-10. The cross-encoder had almost no room to improve, and a lot of room to scramble good results.

### Lesson

Cross-encoders are only beneficial when:
1. They were trained on the same domain/task
2. The first-stage retrieval has meaningful room for improvement

For this task, no suitable academic citation cross-encoder was publicly available without fine-tuning.

---

## 10. Phase 9 — New Dense Models

### 10a. E5-large-v2

**Standalone:** NDCG@10 = 0.5053, similar to SPECTER2 standalone  
**In ensemble:** No improvement over baseline (0.7493)

E5-large is a general semantic search model — similar to BGE but without domain-specific fine-tuning. It adds redundant signal already captured by BGE.

### 10b. SPECTER2 adhoc_query adapter

**Standalone:** NDCG@10 ≈ 0.50 (worse than proximity adapter)

The `adhoc_query` adapter is trained for the task "given a paper title/abstract query, find relevant papers." In theory this matches our task. In practice the `proximity` adapter (trained on co-citation graphs) outperforms it — confirming that citation proximity is the right inductive bias.

### 10c. e5-mistral-7b-instruct (3 runs, all failed)

See [Phase 3 — Dense Models](#4-phase-3--dense-models) for detailed failure analysis. Final standalone NDCG ≈ 0.03 across all three runs.

### 10d. SPECTER2 fine-tuned query encoding

Tested encoding queries with the fine-tuned (FT) SPECTER2 model vs. the TA-encoded (ta) version:

| Config | Val NDCG@10 |
|---|---|
| SPECTER2 ta-query + ft-corpus (current best) | 0.7493 |
| SPECTER2 ft-query + ft-corpus | 0.7493 |

Identical performance — the query encoding variant doesn't matter once they're in the ensemble.

---

## 11. Phase 10 — Pseudo-Relevance Feedback (PRF) (PRF)

### Idea

Use the top-k retrieved documents as pseudo-relevant documents, average their embeddings, and use the resulting vector as an expanded query embedding.

### Experiment

Tested with BGE-large embeddings:
- Expanded query = average of original query embedding + top-3 corpus embeddings
- Re-scored against corpus

### Result: **Failed**

Both Recall@100 and NDCG@10 degraded after PRF. Reason: in citation recommendation, the top retrieved documents are often topically adjacent but not always relevant. Averaging them shifts the query vector away from the specific paper's niche, causing topic drift.

---

---

## 12. Phase 11 — Domain Weight Pre-Boost

### Hypothesis

Instead of reranking candidates after RRF using domain weights, multiply each model's raw score matrix by a domain-boost factor *before* taking top-k. This promotes same-domain documents into the candidate pool itself rather than just reordering an already-fixed pool.

### Formula

```
boosted_score[q, d] = original_score[q, d] * (floor + (1 - floor) * confusion_weight[query_dom, doc_dom])
```

- `floor = 0.0`: cross-domain docs get score = 0 (too aggressive)
- `floor = 0.5`: cross-domain docs get 50% of their original score
- `floor = 1.0`: no effect (equivalent to no boost)

### Key finding: 97.6% of relevant docs are same-domain

Before running the sweep, computed the ground truth domain distribution of relevant documents:

```
Same-domain relevant docs:  718/736 = 97.6%
Cross-domain relevant docs:   18/736 =  2.4%
```

This single fact explains all subsequent results.

### Results — pre-boost score sweep (all models boosted)

| floor | NDCG@10 | Recall@100 |
|---|---|---|
| 0.9 | 0.7094 | 0.8946 |
| 0.7 | 0.7260 | 0.9046 |
| 0.5 | 0.7256 | **0.9098** |
| 0.3 | 0.7284 | 0.9078 |
| Baseline (no boost) | **0.7472** | 0.8716 |

### Interpretation

- **Recall@100 jumps from 0.8716 → 0.9098 (+3.8%)** — the pre-boost pulls more same-domain documents into the candidate pool, closing the gap with the leaderboard's ~0.91
- **NDCG@10 consistently drops** — the boost suppresses cross-domain documents that the dense models (BGE, SPECTER2) correctly ranked highly; only 18 relevant docs are cross-domain but many top-ranked cross-domain docs are irrelevant, polluting the top-10
- **Pre-boost + re-rank with original scores**: tried using the boosted pool (150 docs) then re-scoring with original RRF scores; best NDCG = 0.6732 (worse)

### Additive blend experiment

Tested: `final_score = rrf_score + λ * domain_weight` instead of multiplicative.

| λ | NDCG@10 | vs hard-sort |
|---|---|---|
| 0.05 | 0.6410 | -0.1062 |
| 1.0 | 0.7366 | -0.0106 |
| 2.0 | **0.7472** | +0.0000 |

At λ = 2.0 the additive blend converges to the hard sort (equivalent). The RRF scores are small (~0.1–0.5) while domain weights are 0 or 1; any λ large enough to consistently win puts same-domain docs first, which is exactly the hard sort.

### Status: Failed for NDCG@10, useful for Recall@100

The pre-boost is a good candidate-pool expansion strategy if a high-quality reranker (trained on academic citations) becomes available. For direct NDCG@10 submission it hurts.

---

## 13. Phase 12 — Domain-Filtered Retrieval (Sparse Only)

### Hypothesis

Apply the domain filter **before scoring** for sparse models: zero out all corpus documents outside the query's allowed domains before taking top-100. This forces BM25/TF-IDF to compete only within the relevant domain pool, eliminating false positives caused by vocabulary overlap across domains (e.g., a Chemistry paper containing biology terminology ranking for a Biology query).

Dense models (BGE, SPECTER2) are left **unfiltered** because they already implicitly encode domain context in their embeddings; filtering them discards valid cross-domain neighbours they learned to rank correctly.

### Implementation

```python
def domain_mask(qdom, min_pool=300):
    # Include doc if confusion_matrix[qdom, doc_dom] > 0
    mask = np.array([dw[qdom].get(doc_dom, 0.0) > 0 for doc_dom in corpus_domains])
    # Fallback: use full corpus if fewer than min_pool docs pass
    return mask if mask.sum() >= min_pool else np.ones(N_corpus, dtype=bool)

def top100_filtered(score_matrix, qids, masks):
    scores[~mask] = -1e9   # suppress out-of-domain docs
    return top100_from_remaining
```

Fallback threshold = 300: queries from small domains (Art=42, History=38, Philosophy=27, Mathematics=277) automatically use the full corpus.

### Results — four configs tested

| Config | NDCG@10 | Recall@100 | vs baseline |
|---|---|---|---|
| Baseline (no filter, hard-sort) | 0.7472 | 0.8716 | — |
| All models filtered | 0.7457 | 0.8861 | -0.0014 |
| Sparse filtered + dense unfiltered | **0.7500** | 0.8847 | **+0.0028** |
| Dense filtered + sparse unfiltered | 0.7489 | 0.8855 | +0.0017 |

### min_pool sensitivity sweep (sparse-only filter)

| min_pool | NDCG@10 | Recall@100 |
|---|---|---|
| 100 | 0.7487 | 0.8878 |
| 200 | 0.7487 | 0.8872 |
| **300** | **0.7500** | 0.8847 |
| **500** | **0.7500** | 0.8847 |
| 1000 | 0.7478 | 0.8840 |
| 5000 | 0.7478 | 0.8723 |

Sweet spot: min_pool = 300–500.

### Per-domain analysis

Domains that improved: Biology (+0.003), Environmental Science (+0.008), Business (+0.042 — Business skips domain rerank so filtering helps there)  
Domains that degraded: Medicine (-0.011), Physics (-0.005), Psychology (-0.005)  
Chemistry, CS, Geography, History: unchanged (already dominated by same-domain docs)

### Why sparse-only works and filtering-all doesn't

- **BM25/TF-IDF** are lexical — they match vocabulary regardless of meaning. A Chemistry paper about protein folding can score highly for a Biology protein query. Domain filtering removes this false-positive source.
- **BGE/SPECTER2** encode semantics and domain jointly. Filtering them removes genuinely cross-domain neighbours they correctly learned to rank. Their cross-domain placements are informative, not noise.
- **Filtering all**: the degradation from filtering dense models outweighs the gain from filtering sparse.

### Held-out submission

File: `submissions_heldout/submission_sparse_filtered_domain.json`  
**Codabench result: 0.72**

Held-out pool sizes: min=813 (Business), max=20000 (fallback domains), mean=5893  
Fallback queries (full corpus): 5/100 (Mathematics, Engineering, Political Science, Geology, Sociology)

### Status: Small improvement on val (+0.003), maintained on held-out (0.72)

The improvement is real but small. The approach is correct conceptually and now baked into the default pipeline.

---

## 15. Current Best Result

### Configuration

```
Inner RRF (k=1):
  - BM25L (full text)
  - BGE-large-en-v1.5
  - all-MiniLM-L6-v2

Outer RRF (k=2):
  - Inner result
  - BGE-large-en-v1.5 (direct, double-weighted)
  - BM25L (direct, double-weighted)
  - Hybrid TF-IDF-uni fused with SPECTER2-prox  (k_hybrid=5)
  - Hybrid TF-IDF-bi  fused with SPECTER2-prox  (k_hybrid=5)
  - SPECTER2-proximity

Domain reranking (post-RRF):
  - Sort by (-confusion_matrix_weight, original_rank)
  - Skip Business domain (only 2 val queries, unreliable weights)
```

### Scores

| Split | NDCG@10 | Recall@100 |
|---|---|---|
| Val (validation set) | **0.7493** | 0.8658 |
| Held-out (Codabench) | **~0.72** | 0.8876 |

### Files

- Val submission: `submissions/submission_hk5_k1k2_skipbiz.json`
- Held-out submission: `submissions_heldout/submission_hk5_k1k2_skipbiz.json`

---

---

## 14. Phase 13 — Per-Domain Model Weighting & Exclusion

### Motivation

Oracle analysis showed val NDCG@10 = 0.9108 if we perfectly reranked the top-100 candidate pool. The gap (0.75 → 0.91) is a **ranking problem**, not a recall problem. Analysis of 12 bottlenecked queries showed that for some domains, one model is clearly best (e.g., TF-IDF-bi for Philosophy) but RRF consensus overrides it because other models rank poorly. This led to the hypothesis: _exclude models that are systematically bad for specific domains from the RRF fusion_.

### Method

**Step 1: Per-domain, per-model NDCG@10**  
For each of the 6 retrievers (BM25L, BGE, MiniLM, SPECTER2, TF-IDF-uni, TF-IDF-bi), independently measured NDCG@10 on val queries grouped by domain.

Key findings:
| Domain | Best model | Worst model | Gap |
|---|---|---|---|
| Philosophy | TF-IDF-bi (0.6131) | BGE/MiniLM/SPECTER2 (0.0000) | Extreme |
| Engineering | BGE (0.8408) | BM25L (0.5339) | Large |
| Physics | TF-IDF-uni (0.8400) | BGE (0.5425) | Notable |
| Environmental Science | TF-IDF-uni (0.9295) | MiniLM (0.6849) | Notable |
| Biology | TF-IDF-uni (0.6833) | BM25L (0.4822) | Moderate |

**Step 2: Exclusion threshold sweep**  
Exclude model from domain's RRF if its domain NDCG < `threshold * best_model_ndcg` for that domain.

```python
active = {m for m, s in domain_scores.items() if s >= threshold * max(domain_scores.values())}
```

Results (flat RRF, hard-sort post-rerank):
| Threshold | NDCG@10 | vs baseline |
|---|---|---|
| 0.70 | 0.7512 | +0.0040 |
| 0.78 | 0.7512 | +0.0041 |
| 0.85 | 0.7515 | +0.0043 |

**Step 3: Nested RRF + exclusion**  
The original ensemble uses a 2-level nested RRF (inner: BM25L+BGE+MiniLM at k=1; outer: inner+BGE+BM25L+TF-IDF-uni+TF-IDF-bi+SPECTER2 at k=2). Applied exclusion to this structure.

| Config | NDCG@10 | R@100 |
|---|---|---|
| Nested exclusion thresh=0.70 | 0.7518 | 0.8716 |
| Nested exclusion thresh=0.65 | 0.7488 | 0.8716 |

**Step 4: Combined nested exclusion + sparse domain filter**  
Sparse models (BM25L, TF-IDF variants) use domain-filtered retrievals (from Phase 12); dense models unfiltered. Combined with model exclusion at thresh=0.70.

| Config | NDCG@10 | R@100 | vs baseline |
|---|---|---|---|
| Nested exclusion + sparse filter (thresh=0.70) | **0.7544** | **0.8847** | **+0.0072** |
| Nested exclusion + sparse filter (thresh=0.65) | 0.7514 | 0.8847 | +0.0042 |

### Per-domain breakdown (best config vs baseline)

| Domain | n_q | Baseline | Best | Delta |
|---|---|---|---|---|
| Philosophy | 1 | 0.2044 | 0.2641 | +0.0597 |
| Engineering | 2 | 0.6710 | 0.8263 | +0.1553 |
| Business | 2 | 0.6474 | 0.6661 | +0.0187 |
| Art | 1 | 0.5205 | 0.5294 | +0.0089 |
| Physics | 6 | 0.8587 | 0.8665 | +0.0078 |
| Biology | 21 | 0.7103 | 0.7103 | 0.0000 |
| Medicine | 21 | 0.7137 | 0.7137 | 0.0000 |
| Computer Science | 12 | 0.5606 | 0.5606 | 0.0000 |

Note: Philosophy gain is large (+0.06 in nested structure) but driven by 1 query.

### Key insight

The improvement comes from excluding models that are **genuinely destructive** for certain domains:
- Philosophy: BGE, MiniLM, SPECTER2 all have NDCG=0.000 → excluding them lets TF-IDF-bi dominate
- Engineering: BM25L has NDCG=0.534 vs BGE=0.841 → dropping BM25L from engineering queries

### Scripts

- `experiment_domain_model_weights.py` — per-domain model NDCG table, k-weighted RRF variants
- `experiment_model_exclusion.py` — fine threshold sweep, nested/flat exclusion, sparse filter combos
- `experiment_combined_best.py` — full combined analysis confirming best config
- `build_submission_nested_exclude.py` — builds held-out submission

### Result

| Metric | Val | Held-out |
|---|---|---|
| NDCG@10 | **0.7544** | pending Codabench |
| R@100 | **0.8847** | — |

**File:** `submissions_heldout/submission_nested_exclude_sf.json`  
**Codabench result: 0.73** ✓ (new best, +0.01 over previous best of 0.72)

---

## 15. Current Best Result

### Configuration

**Nested RRF + per-domain model exclusion + sparse domain filter**

```python
# 1. Sparse models use domain-filtered retrieval (min_pool=300)
s_bm25l_f  = top100_filtered(sc_bm25l, qids, domain_masks)
ht_tuni_f  = top100_filtered(rrf_fuse(sc_tuni, sc_sp, k=5), qids, domain_masks)
ht_tbi_f   = top100_filtered(rrf_fuse(sc_tbi,  sc_sp, k=5), qids, domain_masks)

# 2. Per-domain model exclusion (threshold=0.70)
active = {m for m, s in domain_ndcg[qdom].items() if s >= 0.70 * max(domain_ndcg[qdom].values())}

# 3. Nested RRF: inner(BM25L+BGE+MiniLM, k=1) → outer(+TF-IDF+SPECTER2, k=2)

# 4. Hard-sort domain rerank (skip Business)
```

### Scores

| Metric | Val | Held-out |
|---|---|---|
| NDCG@10 | **0.7544** | **0.73** ✓ |
| R@100 | **0.8847** | — |

**Submission file:** `submissions_heldout/submission_nested_exclude_sf.json`

---

## 16. What Didn't Work — Summary

| Approach | Val Delta | Held-out | Why |
|---|---|---|---|
| Year filtering | -0.35 | — | 68.5% of relevant docs are newer than query |
| Cross-encoder (MS-MARCO MiniLM) | -0.31 | — | Wrong training domain |
| Cross-encoder (BGE-reranker-large) | — | -0.43 | Wrong training domain |
| e5-mistral-7b-instruct | -0.72 | — | Task mismatch (topical vs citation similarity) |
| 4-bit quantization of e5-mistral | ❌ | — | Destroys embedding geometry |
| E5-large in ensemble | ~0 | — | Redundant with BGE-large |
| SPECTER2 adhoc adapter | worse | — | Proximity adapter is better fit for citations |
| Per-domain ensemble config | +0.014 val | -0.071 | Severe overfitting (~14 q/domain) |
| Pseudo-relevance feedback | negative | — | Topic drift |
| Soft domain weights (CS/Bio/Med) | 0 | — | Confusion matrix already neutral for these |
| Sparse-only routing | ~0 | — | BM25L(FT) dominates all domains |
| k-weighted RRF (rank-order k) | -0.003 | — | Too aggressive; lowers consensus diversity |
| k-weighted RRF (soft 1/2/3) | -0.005 | — | Same issue |
| Top-N models only per domain | -0.004 to -0.006 | — | Loses recall, reduces pool diversity |
| Catastrophic exclusion (<10% threshold) | -0.004 | — | Excludes correct signal for many queries |
| Domain-aware nested RRF (bottom-up inner) | -0.027 | — | Too much variation from baseline structure |
| Additive domain blend (λ sweep) | 0 to -0.002 | — | Hard-sort is already optimal boundary |
| Domain filter on ALL models | -0.005 | — | Dense models already encode domain context |

---

## 17. What Worked — Summary

| Approach | NDCG@10 Gain | Notes |
|---|---|---|
| Full-text over title+abstract for BM25 | +~0.09 | Body text essential for sparse retrieval |
| BM25L over standard BM25 | +~0.02 | Better TF normalization for long docs |
| BGE-large dense retrieval | — | 0.55 standalone, critical ensemble member |
| SPECTER2 proximity adapter | — | 0.52 standalone; co-citation training matches task |
| Two-level nested RRF | — | Better than flat fusion |
| Low RRF k values (1–2 vs 60) | +~0.01 | Small corpus doesn't need high k |
| BGE double-weighting in outer RRF | +~0.005 | BGE is the strongest single model |
| Hybrid TF-IDF + SPECTER2 fused lists | +~0.01 | Adds sparse+dense synergy |
| Domain reranking (confusion matrix) | +0.003 | Prioritises within-domain citations |
| Skipping Business in domain rerank | +0.001 | Avoids unreliable 2-query domain |
| Sparse domain filter (min_pool=300) | +recall only | R@100: 0.87 → 0.88 |
| Model exclusion per domain (thresh=0.70) | +0.005 | Removes destructive models from fusion |
| **Nested exclusion + sparse filter** | **+0.007** | **Val: 0.7544** |
| **Dense PRF (domain-adaptive β)** | **+0.005** | **Val: 0.7597, Held-out: 0.73 (best)** |

---

## Phase 14–16: Session 2 — Pushing Past 0.76

### What was tried (all starting from 0.7597 baseline)

| Approach | Val NDCG | Delta | Verdict |
|---|---|---|---|
| Zero-shot BGE-reranker-large CE | 0.5600 | -0.20 | Wrong domain (MS-MARCO) |
| Fine-tuned MS-MARCO MiniLM-L-6 CE | 0.6001 | -0.16 | Insufficient training data (1536 pairs) |
| HyDE (llama3.2:3b, all queries) | 0.7563 | -0.003 | Generic abstracts don't help |
| HyDE (llama3.2-vision:11b, weak doms) | 0.7581 | -0.002 | Still no net gain |
| MiniLM fine-tune (CosineSimilarityLoss) | 0.7577 | -0.002 | Not enough data; model overfits |
| Threshold sweep (0.50–0.80) | 0.7597 max | 0 | 0.70 confirmed optimal |
| Domain hard-sort variations (soft, skip) | 0.7450–0.7597 | ≤0 | Hard-sort is net beneficial; can't be softened |
| BM25+, BM25F, TF-IDF trigrams | 0.52–0.52 standalone | — | All weaker than BM25L |
| SPECTER proximal variant | 0.5055 standalone | — | Weaker than SPECTER2-ft |
| Expanded BGE queries (full text) | 0.5558 standalone | -0.002 | Longer text dilutes BGE embedding |
| Venue-constrained retrieval | 0.7534 | -0.006 | Venue noise outweighs venue-match gains |
| SPECTER2 PRF | 0.7579 | -0.002 | SPECTER2 PRF hurts; embedding space mismatch |
| Two-round BGE PRF | 0.7597 | 0 | No change |
| Mixed seed PRF (b0+BGEraw) | 0.7588 | -0.001 | Seed mixing dilutes signal |
| ML PRF β=1.0 for weak doms | 0.7595 | -0.000 | Slightly worse |
| Per-domain BGE β (bio/med=0.10) | 0.7600 | **+0.0003** | Noise level; not statistically significant |

### Why 0.80 is out of reach with current architecture

1. **Oracle PRF ceiling: 0.7679** — Rocchio feedback can only reach 0.77 even with oracle seeds.
2. **Pool recall ceiling: 88.7%** — 34/100 queries have missing rels; 7 queries have rels ranked >5000 in ALL models (completely unretrievable with available embeddings).
3. **Cross-encoder failure**: Both zero-shot and fine-tuned CEs use MS-MARCO training and fail on academic citation retrieval.
4. **Hard queries analysis**: CardioID rels rank 10000+ in every model; "microRNA" rels rank 3000–20000; these queries are permanently at NDCG=0.

### Key finding: domain hard-sort mechanics

The `dr()` domain hard-sort is critical (+0.05 NDCG). It does hurt some cross-domain queries (Chemistry↔Biology, Biology↔Chemistry tagged papers) but is net positive. Attempts to soften it (alpha blending, selective skip by domain) consistently reduce overall NDCG.

### What would move the needle toward 0.80

| Approach | Estimated gain | Requirement |
|---|---|---|
| Fine-tuned SPECTER2 on qrels pairs | +0.02–0.04 | Need to fine-tune on (query, pos, neg) triples |
| Scientific-domain cross-encoder | +0.03–0.05 | BioMedBERT or SciBERT trained on citation pairs |
| BGE-M3 multi-vector retrieval | +0.01–0.02 | Encode 20K corpus with BGE-M3 |
| Improved pool recall (200-doc pool + domain-specific reranker) | +0.02–0.03 | Need reliable scientific reranker |

---

## 18. Remaining Gap & Next Steps

### Score trajectory

| Phase | Val NDCG@10 | Held-out |
|---|---|---|
| BM25L alone | 0.5217 | — |
| + BGE dense | 0.5523 | — |
| + SPECTER2 | 0.6138 | — |
| + Nested RRF ensemble | 0.6274 | — |
| + Domain reranking | 0.7472 | — |
| + Sparse domain filter | ~0.7500 | 0.72 |
| + Model exclusion + sparse filter | 0.7544 | 0.73 |
| + Dense PRF (domain-adaptive β) | **0.7597** | **0.73 ✓** |

### Current gap vs leaderboard top

- Val: 0.7544 (us) vs ~0.80+ (top teams estimated)
- Held-out Recall@100: 0.8847 (best achieved)

### Why the remaining gap is hard to close without fine-tuning

1. **Fine-tuning a dense model on qrels** — requires supervised citation pair data
2. **A task-specific cross-encoder** trained on academic citation pairs
3. **Better first-stage recall** — 0.8847 means ~11.5% of relevant docs not in top-100 pool

### Candidate next steps

| Idea | Likely gain | Risk |
|---|---|---|
| 150-doc candidate pool + reranker | +medium | Low if reranker is task-specific |
| Philosophy-specific routing (always TF-IDF-bi only) | +small on held-out | Could overfit to val |
| BGE-M3 multi-vector retrieval | +medium | Need to encode 20K corpus |
| Fine-tuned cross-encoder on academic citations | +large | Need training data |
| Query expansion with cited paper keywords | +small | Domain-specific effort |

---

## Phase 17: Session 3 — Pool Recall Analysis & Exhaustive Ceiling Tests

### Goal: "Start from the source" — improve recall@100 first, then NDCG

### Key findings

#### Pool recall analysis (true ceiling)
| Config | Recall@100 | Oracle NDCG@10 |
|---|---|---|
| Production pipeline (k=100) | 0.8872 | 0.9280 |
| 6-model union (no filter, k=100) | 0.9280 | — |
| With pool expansion k=200 | 0.8938 | — |
| All-model union (BGE+ML+BM25+TF-IDF+SP) | 0.9280 | — |
| + E5-large | 0.9280 | — |
| + BGE-M3 | 0.9293 (+1 rel) | — |

**Key insight:** The 6-model ensemble already captures 92.80% recall at k=100. E5-large adds 0 new rels; BGE-M3 adds 1. No new retrieval model can meaningfully improve recall.

#### Domain alignment of missing rels
- 57 rels have min_rank > 100 across ALL 6 models
- **97% are same-domain as their query** — domain hard-sort would rescue them IF they entered the pool
- But their RRF scores are too low to beat existing top-100 docs
- Worst cases: rank 8098 (Philosophy), 4273 (Chemistry), 2962 (Art) — permanently unreachable

#### Pool expansion NDCG results (all worse than baseline)
| K per model | NDCG@10 | R@100 | Delta NDCG |
|---|---|---|---|
| 100 (baseline) | 0.7597 | 0.8872 | — |
| 150 | 0.7587 | 0.8943 | -0.0010 |
| 200 | 0.7568 | 0.8938 | -0.0029 |
| 300 | 0.7559 | 0.8922 | -0.0038 |

Recall improves but NDCG drops because domain hard-sort can't distinguish newly-recovered rels from noise at rank 100-200.

#### Learn-to-Rank (LOQO-CV)
Using all 6 model scores + domain match + RRF rank as features, logistic regression:
| Config | NDCG@10 |
|---|---|
| Baseline | 0.7597 |
| LR C=0.01 | 0.7411 |
| LR C=0.10 | 0.7482 |
| LR C=1.00 | 0.7515 |
| GBT LOQO-CV | 0.7433 |
| In-sample upper bound | 0.7526 |

**Key insight:** In-sample LTR can only achieve 0.7526 (oracle = 0.9280). The features don't discriminate between rank-11-50 rels and irrelevant docs — they already encode the same ranking information as RRF.

#### Per-query analysis of production system
- **Oracle NDCG@10 with current top-100 pool: 0.9280**
- Biggest oracle gaps by domain: CS (0.321), Business (0.311), Biology (0.211)
- Rels at rank 1-5: 345 (46.9%), rank 6-10: 100 (13.6%), rank 11-100: 291 (39.5%), missing: 100 (13.6%)

#### Additional variants tested
| Approach | NDCG@10 | Delta | Notes |
|---|---|---|---|
| + raw TF-IDF-bi (no SPECTER2 fusion) | 0.7512 | -0.009 | More noise than signal |
| + BM25-TA (title+abstract) | 0.7597 | 0 | Adds nothing new |
| Title-only BM25 (standalone) | 0.2182 | — | Too weak, recall@100 = 44.6% |

### Conclusion: architecture ceiling confirmed

The 0.7597 val NDCG represents the hard ceiling for the current architecture. The evidence:
1. **Recall ceiling:** 92.80% at k=100 — cannot be meaningfully expanded with available models
2. **LTR ceiling:** In-sample LTR only reaches 0.7526 — features lack within-pool discriminative power
3. **Pool expansion:** Recovers rels but ruins NDCG — recovered rels can't beat existing RRF scores
4. **Feature redundancy:** All model scores encode the same rank signal; RRF is already near-optimal combination

The gap (0.7597 → 0.9280 oracle) is a **semantic discrimination problem**: the 291 rels at rank 11-100 are there because every available model ranks them low. Only a model that knows the specific citation relationship (trained on (paper, cited_paper) pairs) could fix this.

---

## Phase 18: Failure Case Analysis — Worst Domain Examples

For each of the six worst-performing domains (ranked by oracle gap), one representative failing query is examined in full: query topic, gold document title and abstract, per-model raw ranks, final pipeline rank, and a root-cause explanation of why the system fails.

All ranks below are from the production PRF pipeline (val NDCG@10 = 0.7597). "MISS" means the document did not appear in the final top-100 output.

---

### 1. Computer Science — NDCG@10 = 0.000  (oracle gap: 0.321)

**Query:** *CardioID: Secure ECG-BCG Agnostic Interaction-Free Device Pairing*

> "...We present CardioID, an approach to extract features from heart rate variability for secure pairing keys that change with the randomness of the inter-beat interval. The key is derived independently from simultaneously recorded ECG and BCG signals..."

| Gold document | Domain | Final rank | BM25L | BGE | MiniLM | SPECTER2 | TF-IDF-uni | TF-IDF-bi |
|---|---|---|---|---|---|---|---|---|
| Definition of Fiducial Points in the Normal Seismocardiogram | CS | **MISS** | 192 | 76 | 179 | 139 | 37 | 59 |
| Remote plethysmographic imaging using ambient light | CS | **MISS** | 4407 | 228 | 507 | 189 | 3583 | 3693 |

**Why they are not found:**

*Gold 1 (Seismocardiogram):* BGE reaches rank 76 — nearly in the pool — because it understands cardiac signal processing. But "seismocardiogram (SCG)" and "ECG/BCG" name the *same physical phenomenon* (cardiac mechanics) via different sensors. The query never uses the word "seismocardiogram"; the gold doc never uses "device pairing" or "security." TF-IDF-uni=37 shows some keyword proximity ("ECG", "heart") but not enough to survive RRF aggregation across models that rank it at 100+. After the domain-filtered pool is built, this document falls just outside the top-100 boundary.

*Gold 2 (rPPG imaging):* The query cites rPPG (remote photoplethysmography via camera) because *both* rPPG and BCG are contactless biometric signals. But the word "plethysmographic" never appears in the query and "ballistocardiography" never appears in the gold doc. This is a **vocabulary synonym gap**: two terms for equivalent technology in different engineering communities. BM25L=4407 (near-zero keyword overlap). Even BGE only reaches rank 228 — the semantic embedding space conflates "ambient light imaging" with computer vision, not biometric authentication. No available model bridges this terminological gap.

**Root cause:** Cardiac-signal authentication is a niche intersection of biometrics, IoT security, and biomedical signal processing. The gold docs are written in the biomedical signal processing vocabulary; the query is written in the security/IoT vocabulary. The communities cite each other but use entirely different terminology.

---

### 2. Business — NDCG@10 = 0.444  (oracle gap: 0.311)

**Query:** *Does virtual currency development harm financial stocks' value? Comparing Taiwan and China markets*

> "...We study whether growth of virtual currency affects companies in the financial industry in Taiwan and China..." 

| Gold document | Domain | Final rank | BM25L | BGE | MiniLM | SPECTER2 | TF-IDF-uni | TF-IDF-bi |
|---|---|---|---|---|---|---|---|---|
| Research on interaction of innovation spillovers in AI, FinTech, IoT | Economics | **2** ✓ | 1 | 2 | 8 | 2 | 1 | 2 |
| Use of Neural Networks to Accommodate Seasonal Fluctuations When Equalizing Time Series | Business | **22** | 13 | 20 | 78 | 25 | 21 | 22 |
| Statistical Analysis of the Exchange Rate of Bitcoin | Mathematics | **10** ✓ | 9 | 10 | 12 | 69 | 3 | 3 |

**Why gold 2 is not in the top-10:**

*Gold 2 (Neural Networks for Time Series):* Every model ranks it in the 13–25 range but no single model puts it in the top-10. The query is about virtual currency effects on stocks; the gold doc is about neural network methods for financial time series equalization. The citation relationship is *methodological*: both papers apply machine learning techniques to financial market analysis. "Bitcoin" and "virtual currency" do not appear in the gold doc; "time series equalization" does not appear in the query. The shared context (ML applied to finance) is captured by BM25L=13 (keywords: "financial", "stock") but not strongly enough for RRF consensus. Business domain skips the domain hard-sort, so the RRF rank of 22 is the final output. The document is *adjacent in method* but not in topic.

**Root cause:** The link is a shared analytical method (ML for financial prediction) rather than a shared topic (cryptocurrency). Method-bridging citations are not well captured by any available retrieval signal.

---

### 3. Biology — NDCG@10 = 0.370  (oracle gap: 0.211)

**Query:** *Basal Bioenergetic Abnormalities in Skeletal Muscle from Ryanodine Receptor Malignant Hyperthermia-susceptible R163C Knock-in Mice*

> "...Skeletal muscle fibers from Malignant Hyperthermia-susceptible RyR1 R163C knock-in mice showed elevated resting cytosolic calcium...mitochondria of MHS fibers were smaller and closer to the sarcoplasmic reticulum..."

| Gold document | Domain | Final rank | BM25L | BGE | MiniLM | SPECTER2 | TF-IDF-uni | TF-IDF-bi |
|---|---|---|---|---|---|---|---|---|
| Modest PGC-1α Overexpression in Muscle Sufficient to Increase Insulin Sensitivity | Biology | **10** ✓ | 80 | 6 | 8 | 195 | 17 | 23 |
| Calcium signal transmission between ryanodine receptors and mitochondria | Biology | **4** ✓ | 949 | 1 | 12 | 13 | 66 | 85 |
| Protein Kinase C ε Signaling Complexes Include Metabolism- and Transcription/Translation-Related Proteins | Biology | **MISS** | 2814 | 236 | 953 | 427 | 2503 | 3463 |

**Why gold 3 is not found:**

*Gold 3 (Protein Kinase C ε):* The query is about ryanodine receptor (RyR1) mutations affecting mitochondrial calcium and bioenergetics in muscle. PKC ε is a signaling kinase that co-localises with mitochondrial metabolism and transcription complexes. The citation is a **pathway-level link**: RyR1 and PKC ε are both regulators of mitochondrial calcium and metabolic function in skeletal muscle, and both interact with the same protein complexes. But the title does not contain "ryanodine", "malignant hyperthermia", "calcium", or "bioenergetics" — the query contains none of "Protein Kinase C", "PKC", "transcription" in a form that BM25 would match. BGE reaches rank 236 (partial semantic understanding of muscle cell biology) but cannot bridge the specific pathway knowledge. This requires knowing the PKC ε → RyR1 → mitochondrial metabolism axis.

**Root cause:** Pathway-level citations in molecular biology require domain knowledge of protein interaction networks. Text similarity alone cannot reconstruct these citation reasons.

---

### 4. Philosophy — NDCG@10 = 0.387  (oracle gap: 0.226)

**Query:** *Do Extraordinary Claims Require Extraordinary Evidence?*

> "...In 1979 astronomer Carl Sagan popularized the aphorism 'extraordinary claims require extraordinary evidence' (ECREE). But Sagan never defined 'extraordinary.' Ambiguity...has led to misuse of the aphorism. ECREE is commonly invoked to discredit research dealing with scientific anomalies..."

| Gold document | Domain | Final rank | BM25L | BGE | MiniLM | SPECTER2 | TF-IDF-uni | TF-IDF-bi |
|---|---|---|---|---|---|---|---|---|
| Children's Brain Development Benefits from Longer Gestation | Philosophy | **MISS** | 13318 | 8098 | 15665 | 16079 | 10799 | 10694 |
| Territorializing/decolonizing South American prehistory: Pedra Furada and the Cerutti Mastodon | Philosophy | **8** ✓ | 37 | 4013 | 553 | 463 | 20 | 6 |

**Why gold 1 is not found (the hardest case in the entire val set):**

*Gold 1 (Children's Brain Development):* This paper is about neurological benefits of longer gestation periods. The query is a philosophy paper arguing about the epistemology of "extraordinary evidence." There is **zero lexical overlap** and **zero semantic similarity** in the conventional sense. The citation relationship is purely *exemplary*: the query author cites the brain development paper as an instance of research that is commonly dismissed by invoking ECREE — a paper whose scientific claims could be labelled "extraordinary" by critics. This citation reason lives entirely in the philosophical argument, not in either paper's text.

BGE=8098 (rank 8098 out of 20,000) — worse than random chance. BM25L=13318. No retrieval model has any mechanism to discover that a cognitive development paper about gestation length belongs in a retrieval list for a paper about philosophy of evidence. This document is **permanently unretrievable** with text-based methods without knowledge of the citation's philosophical motivation.

*Gold 2 (Pedra Furada archaeology):* Correctly found at rank 8. The Pedra Furada/Cerutti Mastodon controversy is a famous case of "extraordinary claims" in archaeology (early human presence in the Americas), giving it keyword overlap with ECREE discourse. TF-IDF-bi=6 finds it precisely because "extraordinary" likely appears in both.

**Root cause:** ECREE citations are meta-citations: papers are cited not because they are topically similar but because they *exemplify a philosophical argument*. The referenced paper's content is irrelevant to the citation reason. This category of citation is fundamentally inaccessible to any retrieval system based on text similarity.

---

### 5. Medicine — NDCG@10 = 0.000  (oracle gap: 0.158)

**Query:** *Posttranscriptional Regulation of Insulin Resistance: Implications for Metabolic Diseases*

> "...Insulin resistance defines an impairment in the biologic response to insulin action in target tissues, primarily the liver, muscle, adipose tissue... Insulin resistance affects physiology in many ways, causing hyperglycemia, hypertension, dyslipidemia..."

| Gold document | Domain | Final rank | BM25L | BGE | MiniLM | SPECTER2 | TF-IDF-uni | TF-IDF-bi |
|---|---|---|---|---|---|---|---|---|
| Mechanism and regulation of the nonsense-mediated decay pathway | Medicine | **MISS** | 954 | 2528 | 1066 | 385 | 1294 | 702 |
| Apoptosis Evaluation in Circulating CD34+-Enriched Hematopoietic Stem and Progenitor Cells in Cushing's Syndrome | Medicine | **MISS** | 512 | 3395 | 4057 | 3613 | 432 | 666 |

**Why both gold documents are missed:**

*Gold 1 (Nonsense-mediated mRNA decay):* The query is about *posttranscriptional* regulation of insulin resistance — specifically how mRNA-level mechanisms (miRNA, RNA-binding proteins, alternative splicing, NMD) modulate insulin signaling genes. NMD (nonsense-mediated mRNA decay) is exactly one such mechanism: it degrades aberrant mRNAs and regulates expression of thousands of genes, including those in insulin signaling pathways. But the NMD paper's abstract contains neither "insulin resistance" nor "metabolic diseases"; the query abstract does not contain "nonsense-mediated decay" or "mRNA surveillance." SPECTER2=385 (citation co-embedding catches the regulatory biology neighbourhood) but TF-IDF and BM25L rank it at 700–1000. This is a **mechanism-bridging citation**: the review cites NMD as one of many posttranscriptional mechanisms, but the NMD literature is written in entirely different molecular biology terminology.

*Gold 2 (Apoptosis in CD34+ cells in Cushing's syndrome):* The connection is *endocrine disease context*: Cushing's syndrome causes glucocorticoid excess → insulin resistance. The query reviews posttranscriptional regulation of insulin resistance; the gold doc examines cellular dysfunction in a disease that *causes* insulin resistance via a different mechanism (glucocorticoid-driven hematopoietic apoptosis). BGE=3395 — this connection is invisible to all semantic models. TFuni=432 is the best rank, still far from top-100.

**Root cause:** Medical reviews cite mechanism papers from distant subfields to establish comprehensive coverage of a disease pathway. The cited papers are not about the disease itself but about individual molecular mechanisms that *contribute to* it. This secondary-evidence citation pattern has near-zero text overlap with the review query.

---

### 6. Materials Science — NDCG@10 = 0.494  (oracle gap: 0.171)

**Query:** *Biodegradable and Elastomeric Poly(glycerol sebacate) as a Coating Material for Nitinol Bare Stent*

> "...We evaluated the physicochemical properties and biocompatibility of poly(glycerol sebacate) (PGS) coating on nitinol stents...PGS is a biodegradable and biocompatible elastomeric polymer..."

| Gold document | Domain | Final rank | BM25L | BGE | MiniLM | SPECTER2 | TF-IDF-uni | TF-IDF-bi |
|---|---|---|---|---|---|---|---|---|
| Different methods of synthesizing poly(glycerol sebacate) (PGS): A review | MatSci | **1** ✓ | 1 | 1 | 1 | 1 | 1 | 1 |
| Polyglycerol Hyperbranched Polyesters: Synthesis, Properties and Pharmaceutical | MatSci | **2** ✓ | 2 | 4 | 6 | 3 | 2 | 2 |
| Current Concepts and Methods in Tissue Interface Scaffold Fabrication | MatSci | **19** | 12 | 194 | 84 | 67 | 12 | 16 |
| Bioresorbable Materials on the Rise: From Electronic Components and Physical Sensors | MatSci | **12** | 29 | 96 | 169 | 7 | 35 | 58 |
| Biological Performance of Duplex PEO + CNT/PCL Coating on AZ31B Mg Alloy for Orthopaedic | MatSci | **14** | 144 | 6 | 26 | 30 | 171 | 161 |
| Recent Progress on Bioresorbable Passive Electronic Devices and Systems | MatSci | **21** | 227 | 402 | 789 | 13 | 102 | 170 |

**Why 4 gold documents are not in the top-10:**

The top-2 gold docs are correctly ranked #1 and #2 by every model simultaneously — these have direct lexical overlap with "poly(glycerol sebacate)" and "PGS". The remaining 4 gold docs miss the top-10 due to **model consensus override**:

*Gold "Bioresorbable Materials" (final=12):* SPECTER2=7 correctly identifies this as highly relevant (bioresorbable polymers for medical devices — the same citation network as PGS stent coatings). But BM25L=29, BGE=96, MiniLM=169 give it low scores because the text doesn't contain "PGS", "poly(glycerol sebacate)", or "nitinol". In RRF, 5 models vote rank ≥29 while only 1 model (SPECTER2) votes rank 7 — the aggregate score places it at 12.

*Gold "Recent Progress on Bioresorbable Electronic Devices" (final=21):* Same pattern: SPECTER2=13 (co-citation knowledge: bioresorbable electronics and bioresorbable coatings cite each other heavily), but BGE=402, MiniLM=789. The word "stent" appears nowhere in this paper; "electronic devices" appears nowhere in the query. The SPECTER2 signal is correct but outvoted 5-to-1.

**Root cause:** The 4 missing gold docs are about **bioresorbable materials broadly** — a materials science concept that groups together biodegradable polymers, biocompatible coatings, and tissue scaffolds. The specific PGS-stent vocabulary in the query does not match the vocabulary in these gold docs, so BM25L/TF-IDF fail. BGE and MiniLM embed "bioresorbable electronic devices" and "biodegradable stent coating" in distant parts of their embedding space. Only SPECTER2 (trained on co-citation proximity) correctly understands that all bioresorbable materials researchers cite each other. But SPECTER2's single vote is overruled by 5 other models in the RRF fusion.

**This is the only domain where the pipeline's architecture directly causes the miss**: SPECTER2 has the right answer, but the 5-vs-1 RRF aggregation suppresses it. A SPECTER2-primary fusion for Materials Science would recover these rels.

---

### Summary Table: Failure Root Causes

| Domain | Query type | Gold doc type | Root cause | Recoverable? |
|---|---|---|---|---|
| Computer Science | Security/IoT | Biomedical signal processing | Vocabulary synonym gap (ECG≡SCG≡rPPG in different communities) | Possibly, with terminology alignment |
| Business | Cryptocurrency markets | ML methods for finance | Methodological bridge (same method, different application domain) | Partially (BM25L at rank 13) |
| Biology | Molecular biology (RyR1) | Kinase signaling pathway | Pathway-level citation (both regulate same muscle signalling axis) | No — requires protein interaction graph |
| Philosophy | Philosophy of evidence | Case-study paper (brain development) | Conceptual/exemplary citation (gold doc cited as an *instance* of the argument) | No — requires understanding of argument structure |
| Medicine | Disease review (insulin resistance) | mRNA mechanism paper | Mechanism-bridging citation (mechanism → disease connection, no shared vocabulary) | Partially (SPECTER2=385, needs better domain model) |
| Materials Science | Specific polymer coating | Broad bioresorbable materials | SPECTER2 correct but outvoted 5-to-1 by models with lexical bias | Yes — SPECTER2-primary fusion for MatSci would help |

---

*Last updated: 2026-04-16*  
*Best val NDCG@10: 0.7597 | Best held-out (Codabench): 0.73 — submission at `submissions_heldout/submission_prf.json`*
