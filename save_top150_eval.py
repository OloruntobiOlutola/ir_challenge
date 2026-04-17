"""
Build top-100 and top-150 ensemble candidate lists for val and held-out.
The top-150 files give the reranker a larger pool; after reranking and
taking the top-100 the submission is complete.

Outputs:
  val_top100.json        (100 candidates / query) — current baseline
  val_top150.json        (150 candidates / query) — larger reranker pool
  held_top100.json       (100 candidates / query)
  held_top150.json       (150 candidates / query)

Run locally:
    python3 save_top150_eval.py
"""
import json
import numpy as np
import pandas as pd

# ── IDs ────────────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
q_df       = pd.read_parquet('data/queries.parquet').set_index('doc_id')
held_qids  = q_df.index.tolist()
qrels      = json.load(open('data/qrels_1.json'))

# ── Helpers ────────────────────────────────────────────────────────────────
def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q / (nq + 1e-10)) @ (c / (nc + 1e-10)).T).astype(np.float32)

def topN(scores, qids, n, cids=corpus_ids):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-scores[i])[:n]
        sub[qid] = [cids[j] for j in idx]
    return sub

def rrf_fuse(sa, sb, k=5):
    out = np.zeros_like(sa)
    for i in range(len(sa)):
        ra = np.argsort(np.argsort(-sa[i])) + 1
        rb = np.argsort(np.argsort(-sb[i])) + 1
        out[i] = 1.0 / (k + ra) + 1.0 / (k + rb)
    return out

def rrf(lists, qids, k=3, n=150):
    sub = {}
    for qid in qids:
        sc = {}
        for lst in lists:
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (k + rank)
        sub[qid] = sorted(sc, key=sc.get, reverse=True)[:n]
    return sub

def recall_at(sub, k):
    vals = []
    for qid, rels in qrels.items():
        rel_set = set(rels)
        if not rel_set: continue
        vals.append(sum(1 for d in sub.get(qid, [])[:k] if d in rel_set) / len(rel_set))
    return float(np.mean(vals)) if vals else 0.0

# ── Domain rerank ─────────────────────────────────────────────────────────
dw_df     = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw        = dw_df.to_dict(orient='index')
c_df2     = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df     = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_dom_map = c_df2['domain'].to_dict()
q_dom_val = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids  if qid in q1_df.index}
q_dom_held= {qid: q_df.loc[qid,  'domain'] for qid in held_qids if qid in q_df.index}

SKIP = {'Business'}
def dr(sub, qd_map, n):
    out = {}
    for qid, cands in sub.items():
        qd = qd_map.get(qid, '')
        if qd in SKIP:
            out[qid] = cands[:n]; continue
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d, ''), 0.0), rank, d)
                  for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _, _, d in scored[:n]]
    return out

# ── Load val scores ────────────────────────────────────────────────────────
print('Loading val scores...')
sc_bm25l_v = np.load('submissions/scores_bm25l_ft.npy')
sc_tuni_v  = np.load('submissions/scores_tfidf_uni_ft.npy')
sc_tbi_v   = np.load('submissions/scores_tfidf_bi_ft.npy')
bge_qv     = np.load('submissions/bge_large_query_emb.npy')
bge_c      = np.load('submissions/bge_large_corpus_emb.npy')
sc_bge_v   = cosine(bge_qv, bge_c)
ml_qv      = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy')
ml_cv      = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy')
sc_ml_v    = (ml_qv @ ml_cv.T).astype(np.float32)
sp_q_v     = np.load('specter_prox_embed/queries_1_embeddings.npy')
sp_ft_c    = np.load('submissions/specter2_corpus_emb_ft.npy')
sc_sp_v    = cosine(sp_q_v, sp_ft_c)

# ── Build val ensembles at n=150, then slice to 100 ───────────────────────
print('Building val candidates (top-150 pool)...')
ht_tuni_v = topN(rrf_fuse(sc_tuni_v, sc_sp_v, k=5), val_qids, n=150)
ht_tbi_v  = topN(rrf_fuse(sc_tbi_v,  sc_sp_v, k=5), val_qids, n=150)
s_bm25l_v = topN(sc_bm25l_v, val_qids, n=150)
s_bge_v   = topN(sc_bge_v,   val_qids, n=150)
s_ml_v    = topN(sc_ml_v,    val_qids, n=150)
s_sp_v    = topN(sc_sp_v,    val_qids, n=150)

inner_v   = rrf([s_bm25l_v, s_bge_v, s_ml_v],                              val_qids, k=1, n=150)
outer_v   = rrf([inner_v, s_bge_v, s_bm25l_v, ht_tuni_v, ht_tbi_v, s_sp_v], val_qids, k=2, n=150)

val_top150 = dr(outer_v, q_dom_val, n=150)
val_top100 = {qid: cands[:100] for qid, cands in val_top150.items()}

# ── Val recall report ──────────────────────────────────────────────────────
print('\n── Val recall ──────────────────────────────────────')
for k in [50, 100, 150]:
    print(f'  Recall@{k:<3} = {recall_at(val_top150, k):.4f}')
print()
print('  Recall@100 from top-100 list:', f'{recall_at(val_top100, 100):.4f}')
print('  Recall@100 from top-150 list:', f'{recall_at(val_top150, 100):.4f}',
      '  (same — first 100 docs identical)')

# Check if any extra docs in 101-150 are relevant
extra_hits = 0
total_rels  = 0
for qid, rels in qrels.items():
    rel_set = set(rels)
    if not rel_set: continue
    cands = val_top150.get(qid, [])
    top100_set = set(cands[:100])
    extra = [d for d in cands[100:] if d in rel_set and d not in top100_set]
    extra_hits += len(extra)
    total_rels += len(rel_set)

print(f'\n  Relevant docs found only in positions 101-150: {extra_hits}')
print(f'  (out of {total_rels} total relevant docs across all queries)')
print(f'  These are the docs a reranker could rescue by seeing 150 instead of 100.')

# ── Load held-out scores ───────────────────────────────────────────────────
print('\nLoading held-out scores...')
sc_bm25l_h = np.load('submissions_heldout/scores_bm25l_ft.npy')
sc_tuni_h  = np.load('submissions_heldout/scores_tfidf_uni_ft.npy')
sc_tbi_h   = np.load('submissions_heldout/scores_tfidf_bi_ft.npy')
sc_bge_h   = np.load('submissions_heldout/scores_bge_dense_corrected.npy')
ml_qh      = np.load('data/embeddings/new_queries_minilm/query_embeddings.npy')
sc_ml_h    = (ml_qh @ ml_cv.T).astype(np.float32)
sp_q_h     = np.load('specter_prox_embed/queries_embeddings.npy')
sc_sp_h    = cosine(sp_q_h, sp_ft_c)

# ── Build held-out ensembles ───────────────────────────────────────────────
print('Building held-out candidates (top-150 pool)...')
ht_tuni_h = topN(rrf_fuse(sc_tuni_h, sc_sp_h, k=5), held_qids, n=150)
ht_tbi_h  = topN(rrf_fuse(sc_tbi_h,  sc_sp_h, k=5), held_qids, n=150)
s_bm25l_h = topN(sc_bm25l_h, held_qids, n=150)
s_bge_h   = topN(sc_bge_h,   held_qids, n=150)
s_ml_h    = topN(sc_ml_h,    held_qids, n=150)
s_sp_h    = topN(sc_sp_h,    held_qids, n=150)

inner_h    = rrf([s_bm25l_h, s_bge_h, s_ml_h],                               held_qids, k=1, n=150)
outer_h    = rrf([inner_h, s_bge_h, s_bm25l_h, ht_tuni_h, ht_tbi_h, s_sp_h], held_qids, k=2, n=150)

held_top150 = dr(outer_h, q_dom_held, n=150)
held_top100 = {qid: cands[:100] for qid, cands in held_top150.items()}

# ── Save ───────────────────────────────────────────────────────────────────
with open('val_top100.json',  'w') as f: json.dump(val_top100,  f)
with open('val_top150.json',  'w') as f: json.dump(val_top150,  f)
with open('held_top100.json', 'w') as f: json.dump(held_top100, f)
with open('held_top150.json', 'w') as f: json.dump(held_top150, f)

print(f'\nSaved val_top100.json  ({len(val_top100)} queries, 100 candidates each)')
print(f'Saved val_top150.json  ({len(val_top150)} queries, 150 candidates each)')
print(f'Saved held_top100.json ({len(held_top100)} queries, 100 candidates each)')
print(f'Saved held_top150.json ({len(held_top150)} queries, 150 candidates each)')
print()
print('Upload held_top100.json and held_top150.json to Kaggle.')
print('In the reranker, set top_k=100 or top_k=150 to compare both.')
