"""
Domain weight pre-boost experiment.

Current approach (post-RRF reranking):
  build scores → top-100 per model → RRF → sort by (-domain_weight, rank)

New approach (pre-boost):
  build scores → multiply by domain boost → top-100 → RRF → (optional post-rerank)

The confusion matrix is near-binary: diagonal ~1.0, off-diagonal ~0.
We apply:  boosted_score = score * (floor + (1-floor) * domain_weight)

  floor=0.0 : cross-domain docs get score=0 (too aggressive)
  floor=0.5 : cross-domain docs get 50% of their score
  floor=0.8 : cross-domain docs get 80% of their score (gentle nudge)
  floor=1.0 : no effect

We sweep floor and also try combining pre-boost WITH post-rerank.

Baseline: 0.7493 (hk5, k1k2, skip Business domain rerank)
"""
import json
import numpy as np
import pandas as pd

# ── IDs & qrels ────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
qrels      = json.load(open('data/qrels_1.json'))

# ── Domain maps ────────────────────────────────────────────────────────────
dw_df     = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw        = dw_df.to_dict(orient='index')
c_df      = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df     = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_dom_map = c_df['domain'].to_dict()
q_dom_val = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids if qid in q1_df.index}

# Precompute arrays for fast vectorised boost
corpus_domains = [c_dom_map.get(cid, '') for cid in corpus_ids]

def make_boost_matrix(qids, q_dom_map, floor):
    """
    Build (n_queries, n_corpus) boost matrix.
    boost[i,j] = floor + (1-floor) * confusion_weight[query_dom, corpus_dom]
    """
    B = np.full((len(qids), len(corpus_ids)), floor, dtype=np.float32)
    for i, qid in enumerate(qids):
        qdom = q_dom_map.get(qid, '')
        row  = dw.get(qdom, {})
        for j, cdom in enumerate(corpus_domains):
            w = row.get(cdom, 0.0)
            if w > 0:
                B[i, j] = floor + (1 - floor) * w
    return B

# ── Helpers ────────────────────────────────────────────────────────────────
def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q / (nq + 1e-10)) @ (c / (nc + 1e-10)).T).astype(np.float32)

def top100(scores, qids, cids=corpus_ids):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-scores[i])[:100]
        sub[qid] = [cids[j] for j in idx]
    return sub

def rrf_fuse(sa, sb, k=5):
    out = np.zeros_like(sa)
    for i in range(len(sa)):
        ra = np.argsort(np.argsort(-sa[i])) + 1
        rb = np.argsort(np.argsort(-sb[i])) + 1
        out[i] = 1.0 / (k + ra) + 1.0 / (k + rb)
    return out

def rrf(lists, qids, k=3):
    sub = {}
    for qid in qids:
        sc = {}
        for lst in lists:
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (k + rank)
        sub[qid] = sorted(sc, key=sc.get, reverse=True)[:100]
    return sub

def ndcg(sub, k=10):
    sc = []
    for qid, rels in qrels.items():
        if qid not in sub: continue
        ranked  = sub[qid][:k]
        rel_set = set(rels) if isinstance(rels, list) else set()
        dcg  = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
        sc.append(dcg / idcg if idcg > 0 else 0)
    return float(np.mean(sc))

def recall100(sub):
    vals = []
    for qid, rels in qrels.items():
        rel_set = set(rels)
        if not rel_set: continue
        vals.append(sum(1 for d in sub.get(qid, [])[:100] if d in rel_set) / len(rel_set))
    return float(np.mean(vals))

def dr(sub, qd_map, skip=frozenset({'Business'})):
    out = {}
    for qid, cands in sub.items():
        qd = qd_map.get(qid, '')
        if qd in skip:
            out[qid] = cands[:100]; continue
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d, ''), 0.0), rank, d)
                  for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _, _, d in scored[:100]]
    return out

# ── Load scores ────────────────────────────────────────────────────────────
print('Loading scores...')
sc_bm25l = np.load('submissions/scores_bm25l_ft.npy')
sc_tuni  = np.load('submissions/scores_tfidf_uni_ft.npy')
sc_tbi   = np.load('submissions/scores_tfidf_bi_ft.npy')
bge_qv   = np.load('submissions/bge_large_query_emb.npy')
bge_c    = np.load('submissions/bge_large_corpus_emb.npy')
sc_bge   = cosine(bge_qv, bge_c)
ml_qv    = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy')
ml_cv    = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy')
sc_ml    = (ml_qv @ ml_cv.T).astype(np.float32)
sp_q     = np.load('specter_prox_embed/queries_1_embeddings.npy')
sp_ft_c  = np.load('submissions/specter2_corpus_emb_ft.npy')
sc_sp    = cosine(sp_q, sp_ft_c)

# ── Baseline (no pre-boost) ────────────────────────────────────────────────
s_bm25l = top100(sc_bm25l, val_qids)
s_bge   = top100(sc_bge,   val_qids)
s_ml    = top100(sc_ml,    val_qids)
s_sp    = top100(sc_sp,    val_qids)
ht_tuni = top100(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids)
ht_tbi  = top100(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids)
inner   = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)
outer   = rrf([inner, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp], val_qids, k=2)
base_dr = dr(outer, q_dom_val)
print(f'Baseline (post-rerank):  NDCG={ndcg(base_dr):.4f}  Recall@100={recall100(outer):.4f}')

# ── Pre-boost sweep ────────────────────────────────────────────────────────
print('\nBuilding boost matrices...')
results = {}

for floor in [0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.0]:
    B = make_boost_matrix(val_qids, q_dom_val, floor=floor)

    # Normalise scores to [0,1] before boosting so different score scales are comparable
    def norm01(sc):
        mn, mx = sc.min(axis=1, keepdims=True), sc.max(axis=1, keepdims=True)
        return (sc - mn) / (mx - mn + 1e-10)

    # Apply boost to all score matrices
    sc_bm25l_b = norm01(sc_bm25l) * B
    sc_bge_b   = norm01(sc_bge)   * B
    sc_ml_b    = norm01(sc_ml)    * B
    sc_sp_b    = norm01(sc_sp)    * B
    sc_tuni_b  = norm01(sc_tuni)  * B
    sc_tbi_b   = norm01(sc_tbi)   * B

    s_bm25l_b = top100(sc_bm25l_b, val_qids)
    s_bge_b   = top100(sc_bge_b,   val_qids)
    s_ml_b    = top100(sc_ml_b,    val_qids)
    s_sp_b    = top100(sc_sp_b,    val_qids)
    ht_tuni_b = top100(rrf_fuse(sc_tuni_b, sc_sp_b, k=5), val_qids)
    ht_tbi_b  = top100(rrf_fuse(sc_tbi_b,  sc_sp_b, k=5), val_qids)

    inner_b = rrf([s_bm25l_b, s_bge_b, s_ml_b], val_qids, k=1)
    outer_b = rrf([inner_b, s_bge_b, s_bm25l_b, ht_tuni_b, ht_tbi_b, s_sp_b], val_qids, k=2)

    # Config 1: pre-boost only (no post-rerank)
    n1 = ndcg(outer_b)
    r1 = recall100(outer_b)

    # Config 2: pre-boost + post-rerank
    n2 = ndcg(dr(outer_b, q_dom_val))

    results[floor] = (n1, r1, n2)
    print(f'  floor={floor:.1f}:  pre-only NDCG={n1:.4f}  R@100={r1:.4f}  '
          f'pre+post NDCG={n2:.4f}')

# ── Also try boost on dense only (keep sparse unmodified) ─────────────────
print('\nBoost on dense models only (sparse unchanged):')
for floor in [0.8, 0.7, 0.6, 0.5]:
    B = make_boost_matrix(val_qids, q_dom_val, floor=floor)
    sc_bge_b = norm01(sc_bge) * B
    sc_ml_b  = norm01(sc_ml)  * B
    sc_sp_b  = norm01(sc_sp)  * B

    s_bge_b   = top100(sc_bge_b, val_qids)
    s_ml_b    = top100(sc_ml_b,  val_qids)
    s_sp_b    = top100(sc_sp_b,  val_qids)
    ht_tuni_b = top100(rrf_fuse(sc_tuni, sc_sp_b, k=5), val_qids)
    ht_tbi_b  = top100(rrf_fuse(sc_tbi,  sc_sp_b, k=5), val_qids)

    inner_b = rrf([s_bm25l, s_bge_b, s_ml_b], val_qids, k=1)
    outer_b = rrf([inner_b, s_bge_b, s_bm25l, ht_tuni_b, ht_tbi_b, s_sp_b], val_qids, k=2)
    n1 = ndcg(outer_b)
    n2 = ndcg(dr(outer_b, q_dom_val))
    print(f'  floor={floor:.1f}:  dense-only NDCG={n1:.4f}  dense+post NDCG={n2:.4f}')

# ── Summary ────────────────────────────────────────────────────────────────
print('\n' + '=' * 62)
best_floor = max(results, key=lambda f: max(results[f][0], results[f][2]))
best_n = max(results[best_floor][0], results[best_floor][2])
print(f'Best pre-boost floor: {best_floor}  NDCG={best_n:.4f}')
print(f'Baseline post-rerank: NDCG={ndcg(base_dr):.4f}')
delta = best_n - ndcg(base_dr)
print(f'Delta: {delta:+.4f}')
