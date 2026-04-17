"""
Experiment: add E5-large and SPECTER2 ft-query embeddings to the ensemble.
Baseline val NDCG@10 = 0.7493 (hk5, k1k2, skip Business domain rerank)

Tests:
  A) Baseline (reproduce 0.7493)
  B) Add E5-large to inner
  C) Add E5-large to outer
  D) SPECTER2 ft queries (instead of ta) with ft corpus
  E) Best combo
"""
import json
import numpy as np
import pandas as pd

# ── IDs ────────────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
q_df       = pd.read_parquet('data/queries.parquet').set_index('doc_id')
qrels      = json.load(open('data/qrels_1.json'))

print(f'Val: {len(val_qids)}  Corpus: {len(corpus_ids)}')

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

def rrf_fuse(sa, sb, k=3):
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
        if qid not in sub:
            continue
        ranked  = sub[qid][:k]
        rel_set = set(rels) if isinstance(rels, list) else set()
        dcg  = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
        sc.append(dcg / idcg if idcg > 0 else 0)
    return float(np.mean(sc))

# ── Domain reranking (skip Business) ──────────────────────────────────────
dw_df = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw    = dw_df.to_dict(orient='index')

c_df   = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df  = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_dom_map     = c_df['domain'].to_dict()
q_dom_map_val = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids if qid in q1_df.index}

SKIP = {'Business'}
def dr(sub, qd_map):
    out = {}
    for qid, cands in sub.items():
        qd = qd_map.get(qid, '')
        if qd in SKIP:
            out[qid] = cands[:100]
            continue
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d, ''), 0.0), rank, d)
                  for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _, _, d in scored[:100]]
    return out

# ── Load all embeddings ────────────────────────────────────────────────────
print('Loading embeddings...')

# Sparse
sc_bm25l = np.load('submissions/scores_bm25l_ft.npy')
sc_tuni  = np.load('submissions/scores_tfidf_uni_ft.npy')
sc_tbi   = np.load('submissions/scores_tfidf_bi_ft.npy')

# BGE-large
bge_qv = np.load('submissions/bge_large_query_emb.npy')
bge_c  = np.load('submissions/bge_large_corpus_emb.npy')
sc_bge = cosine(bge_qv, bge_c)

# MiniLM
ml_qv = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy')
ml_cv = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy')
sc_ml = (ml_qv @ ml_cv.T).astype(np.float32)

# SPECTER2 prox ta-queries + ft-corpus (current best)
sp_q_ta  = np.load('specter_prox_embed/queries_1_embeddings.npy')
sp_ft_c  = np.load('submissions/specter2_corpus_emb_ft.npy')
sc_sp_ta = cosine(sp_q_ta, sp_ft_c)

# SPECTER2 prox ft-queries + ft-corpus (new)
sp_q_ft  = np.load('specter_prox_embed/specter2_prox_val_queries_ft.npy')
sc_sp_ft = cosine(sp_q_ft, sp_ft_c)

# E5-large (new) — already L2-normalized from encoding script
e5_qv = np.load('e5_emb/e5large_val_queries.npy')
e5_cv = np.load('e5_emb/e5large_corpus.npy')
sc_e5 = cosine(e5_qv, e5_cv)

print('All embeddings loaded.')

# ── Build top-100 lists ────────────────────────────────────────────────────
s_bm25l = top100(sc_bm25l, val_qids)
s_bge   = top100(sc_bge,   val_qids)
s_ml    = top100(sc_ml,    val_qids)
s_sp_ta = top100(sc_sp_ta, val_qids)
s_sp_ft = top100(sc_sp_ft, val_qids)
s_e5    = top100(sc_e5,    val_qids)

print(f'Standalone E5:         {ndcg(s_e5):.4f}')
print(f'Standalone SPECTER ft: {ndcg(s_sp_ft):.4f}')
print(f'Standalone SPECTER ta: {ndcg(s_sp_ta):.4f}')
print(f'Standalone BGE:        {ndcg(s_bge):.4f}')
print()

# Hybrid fused lists (hk=5 = current best)
ht_tuni_ta = top100(rrf_fuse(sc_tuni, sc_sp_ta, k=5), val_qids)
ht_tbi_ta  = top100(rrf_fuse(sc_tbi,  sc_sp_ta, k=5), val_qids)
ht_tuni_ft = top100(rrf_fuse(sc_tuni, sc_sp_ft, k=5), val_qids)
ht_tbi_ft  = top100(rrf_fuse(sc_tbi,  sc_sp_ft, k=5), val_qids)

# ── Experiments ────────────────────────────────────────────────────────────
results = {}

# A — Baseline (reproduce 0.7493): ta-query SPECTER, no E5
inner_A = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)
outer_A = rrf([inner_A, s_bge, s_bm25l, ht_tuni_ta, ht_tbi_ta, s_sp_ta], val_qids, k=2)
results['A_baseline_ta']   = ndcg(outer_A)
results['A_baseline_ta_dr'] = ndcg(dr(outer_A, q_dom_map_val))

# B — ft-query SPECTER instead of ta
inner_B = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)
outer_B = rrf([inner_B, s_bge, s_bm25l, ht_tuni_ft, ht_tbi_ft, s_sp_ft], val_qids, k=2)
results['B_sp_ft_query']    = ndcg(outer_B)
results['B_sp_ft_query_dr'] = ndcg(dr(outer_B, q_dom_map_val))

# C — E5 added to inner (alongside BM25L, BGE, MiniLM)
inner_C = rrf([s_bm25l, s_bge, s_ml, s_e5], val_qids, k=1)
outer_C = rrf([inner_C, s_bge, s_bm25l, ht_tuni_ta, ht_tbi_ta, s_sp_ta], val_qids, k=2)
results['C_e5_inner']    = ndcg(outer_C)
results['C_e5_inner_dr'] = ndcg(dr(outer_C, q_dom_map_val))

# D — E5 added to outer
inner_D = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)
outer_D = rrf([inner_D, s_bge, s_bm25l, ht_tuni_ta, ht_tbi_ta, s_sp_ta, s_e5], val_qids, k=2)
results['D_e5_outer']    = ndcg(outer_D)
results['D_e5_outer_dr'] = ndcg(dr(outer_D, q_dom_map_val))

# E — E5 in both inner+outer, ft-query SPECTER
inner_E = rrf([s_bm25l, s_bge, s_ml, s_e5], val_qids, k=1)
outer_E = rrf([inner_E, s_bge, s_bm25l, ht_tuni_ft, ht_tbi_ft, s_sp_ft, s_e5], val_qids, k=2)
results['E_e5_both_sp_ft']    = ndcg(outer_E)
results['E_e5_both_sp_ft_dr'] = ndcg(dr(outer_E, q_dom_map_val))

# F — E5 outer only + ft-query SPECTER
inner_F = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)
outer_F = rrf([inner_F, s_bge, s_bm25l, ht_tuni_ft, ht_tbi_ft, s_sp_ft, s_e5], val_qids, k=2)
results['F_e5_outer_sp_ft']    = ndcg(outer_F)
results['F_e5_outer_sp_ft_dr'] = ndcg(dr(outer_F, q_dom_map_val))

# G — E5 inner only + ft-query SPECTER
inner_G = rrf([s_bm25l, s_bge, s_ml, s_e5], val_qids, k=1)
outer_G = rrf([inner_G, s_bge, s_bm25l, ht_tuni_ft, ht_tbi_ft, s_sp_ft], val_qids, k=2)
results['G_e5_inner_sp_ft']    = ndcg(outer_G)
results['G_e5_inner_sp_ft_dr'] = ndcg(dr(outer_G, q_dom_map_val))

print('=' * 55)
print(f'{"Experiment":<35} {"NDCG@10":>8}')
print('=' * 55)
for name, score in sorted(results.items(), key=lambda x: -x[1]):
    marker = ' <-- BEST' if score == max(results.values()) else ''
    print(f'{name:<35} {score:.4f}{marker}')
print('=' * 55)
