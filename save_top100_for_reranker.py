"""
Save the current-best val and held-out top-100 lists to JSON.
Upload the two output files to Kaggle as a dataset before running
kaggle_rerank_crossencoder.py.

Run locally:
    python3 save_top100_for_reranker.py
"""
import json
import numpy as np
import pandas as pd

# ── IDs ────────────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
q_df       = pd.read_parquet('data/queries.parquet').set_index('doc_id')
held_qids  = q_df.index.tolist()

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

# ── Domain rerank (skip Business) ─────────────────────────────────────────
dw_df = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw    = dw_df.to_dict(orient='index')
c_df2  = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df  = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_dom_map      = c_df2['domain'].to_dict()
q_dom_val      = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids   if qid in q1_df.index}
q_dom_held     = {qid: q_df.loc[qid,  'domain'] for qid in held_qids  if qid in q_df.index}

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

# ── Load scores ────────────────────────────────────────────────────────────
print('Loading scores...')
sc_bm25l_v = np.load('submissions/scores_bm25l_ft.npy')
sc_tuni_v  = np.load('submissions/scores_tfidf_uni_ft.npy')
sc_tbi_v   = np.load('submissions/scores_tfidf_bi_ft.npy')

bge_qv = np.load('submissions/bge_large_query_emb.npy')
bge_c  = np.load('submissions/bge_large_corpus_emb.npy')
sc_bge_v = cosine(bge_qv, bge_c)

ml_qv = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy')
ml_cv = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy')
sc_ml_v = (ml_qv @ ml_cv.T).astype(np.float32)

sp_q_v  = np.load('specter_prox_embed/queries_1_embeddings.npy')
sp_ft_c = np.load('submissions/specter2_corpus_emb_ft.npy')
sc_sp_v = cosine(sp_q_v, sp_ft_c)

# ── Build val top-100 ──────────────────────────────────────────────────────
print('Building val top-100...')
ht_tuni_v = top100(rrf_fuse(sc_tuni_v, sc_sp_v, k=5), val_qids)
ht_tbi_v  = top100(rrf_fuse(sc_tbi_v,  sc_sp_v, k=5), val_qids)
s_bm25l_v = top100(sc_bm25l_v, val_qids)
s_bge_v   = top100(sc_bge_v,   val_qids)
s_ml_v    = top100(sc_ml_v,    val_qids)
s_sp_v    = top100(sc_sp_v,    val_qids)
inner_v = rrf([s_bm25l_v, s_bge_v, s_ml_v],            val_qids, k=1)
outer_v = rrf([inner_v, s_bge_v, s_bm25l_v, ht_tuni_v, ht_tbi_v, s_sp_v], val_qids, k=2)
val_top100 = dr(outer_v, q_dom_val)

# ── Load held-out scores ───────────────────────────────────────────────────
print('Loading held-out scores...')
sc_bm25l_h = np.load('submissions_heldout/scores_bm25l_ft.npy')
sc_tuni_h  = np.load('submissions_heldout/scores_tfidf_uni_ft.npy')
sc_tbi_h   = np.load('submissions_heldout/scores_tfidf_bi_ft.npy')
sc_bge_h   = np.load('submissions_heldout/scores_bge_dense_corrected.npy')

ml_qh = np.load('data/embeddings/new_queries_minilm/query_embeddings.npy')
sc_ml_h = (ml_qh @ ml_cv.T).astype(np.float32)

sp_q_h  = np.load('specter_prox_embed/queries_embeddings.npy')
sc_sp_h = cosine(sp_q_h, sp_ft_c)

# ── Build held-out top-100 ─────────────────────────────────────────────────
print('Building held-out top-100...')
ht_tuni_h = top100(rrf_fuse(sc_tuni_h, sc_sp_h, k=5), held_qids)
ht_tbi_h  = top100(rrf_fuse(sc_tbi_h,  sc_sp_h, k=5), held_qids)
s_bm25l_h = top100(sc_bm25l_h, held_qids)
s_bge_h   = top100(sc_bge_h,   held_qids)
s_ml_h    = top100(sc_ml_h,    held_qids)
s_sp_h    = top100(sc_sp_h,    held_qids)
inner_h = rrf([s_bm25l_h, s_bge_h, s_ml_h],            held_qids, k=1)
outer_h = rrf([inner_h, s_bge_h, s_bm25l_h, ht_tuni_h, ht_tbi_h, s_sp_h], held_qids, k=2)
held_top100 = dr(outer_h, q_dom_held)

# ── Save ───────────────────────────────────────────────────────────────────
with open('val_top100.json', 'w') as f:
    json.dump(val_top100, f)
with open('held_top100.json', 'w') as f:
    json.dump(held_top100, f)

print(f'Saved val_top100.json  ({len(val_top100)} queries)')
print(f'Saved held_top100.json ({len(held_top100)} queries)')
print()
print('Next: upload both files to Kaggle as a dataset, then run')
print('      kaggle_rerank_crossencoder.py  on a T4 GPU notebook.')
