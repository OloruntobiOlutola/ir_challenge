"""
Build best held-out submission: val=0.7465
Config: inner k=1, outer k=2, hybrid_k=3, ft SPECTER corpus, domain rerank
"""
import json
import numpy as np
import pandas as pd

# ── IDs ────────────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
q_df  = pd.read_parquet('data/queries.parquet').set_index('doc_id')
held_qids = q_df.index.tolist()
qrels = json.load(open('data/qrels_1.json'))

print(f'Val: {len(val_qids)}  Held: {len(held_qids)}  Corpus: {len(corpus_ids)}')

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

# ── Domain reranking setup ─────────────────────────────────────────────────
dw_df = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw = dw_df.to_dict(orient='index')

c_df  = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')

c_dom_map = c_df['domain'].to_dict()
q_dom_map_val  = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids if qid in q1_df.index}
q_dom_map_held = {qid: q_df.loc[qid,  'domain'] for qid in held_qids if qid in q_df.index}

def dr_fn(sub, qd_map):
    out = {}
    for qid, cands in sub.items():
        qd = qd_map.get(qid, '')
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d, ''), 0.0), rank, d)
                  for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _, _, d in scored[:100]]
    return out

# ── Load val scores ────────────────────────────────────────────────────────
print('Loading val scores...')
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
sc_sp_ft_v = cosine(sp_q_v, sp_ft_c)

# ── Val ensemble (verify) ──────────────────────────────────────────────────
print('Building val ensemble...')
ht_tuni_ft_v = rrf_fuse(sc_tuni_v, sc_sp_ft_v, k=3)
ht_tbi_ft_v  = rrf_fuse(sc_tbi_v,  sc_sp_ft_v, k=3)

s_bm25l_v = top100(sc_bm25l_v, val_qids)
s_bge_v   = top100(sc_bge_v,   val_qids)
s_ml_v    = top100(sc_ml_v,    val_qids)
s_sp_ft_v = top100(sc_sp_ft_v, val_qids)
s_ht_tuni_ft_v = top100(ht_tuni_ft_v, val_qids)
s_ht_tbi_ft_v  = top100(ht_tbi_ft_v,  val_qids)

inner_v = rrf([s_bm25l_v, s_bge_v, s_ml_v], val_qids, k=1)
outer_v = rrf([inner_v, s_bge_v, s_bm25l_v, s_ht_tuni_ft_v, s_ht_tbi_ft_v, s_sp_ft_v], val_qids, k=2)
final_v = dr_fn(outer_v, q_dom_map_val)

print(f'Val NDCG@10 (before domain rerank): {ndcg(outer_v):.4f}')
print(f'Val NDCG@10 (after  domain rerank): {ndcg(final_v):.4f}')

# ── Load held-out scores ───────────────────────────────────────────────────
print('\nLoading held-out scores...')
sc_bm25l_h = np.load('submissions_heldout/scores_bm25l_ft.npy')
sc_tuni_h  = np.load('submissions_heldout/scores_tfidf_uni_ft.npy')
sc_tbi_h   = np.load('submissions_heldout/scores_tfidf_bi_ft.npy')
sc_bge_h   = np.load('submissions_heldout/scores_bge_dense_corrected.npy')

ml_qh = np.load('data/embeddings/new_queries_minilm/query_embeddings.npy')
sc_ml_h = (ml_qh @ ml_cv.T).astype(np.float32)

sp_q_h  = np.load('specter_prox_embed/queries_embeddings.npy')
sc_sp_ft_h = cosine(sp_q_h, sp_ft_c)

# ── Held-out ensemble ──────────────────────────────────────────────────────
print('Building held-out ensemble...')
ht_tuni_ft_h = rrf_fuse(sc_tuni_h, sc_sp_ft_h, k=3)
ht_tbi_ft_h  = rrf_fuse(sc_tbi_h,  sc_sp_ft_h, k=3)

s_bm25l_h = top100(sc_bm25l_h, held_qids)
s_bge_h   = top100(sc_bge_h,   held_qids)
s_ml_h    = top100(sc_ml_h,    held_qids)
s_sp_ft_h = top100(sc_sp_ft_h, held_qids)
s_ht_tuni_ft_h = top100(ht_tuni_ft_h, held_qids)
s_ht_tbi_ft_h  = top100(ht_tbi_ft_h,  held_qids)

inner_h = rrf([s_bm25l_h, s_bge_h, s_ml_h], held_qids, k=1)
outer_h = rrf([inner_h, s_bge_h, s_bm25l_h, s_ht_tuni_ft_h, s_ht_tbi_ft_h, s_sp_ft_h], held_qids, k=2)
final_h = dr_fn(outer_h, q_dom_map_held)

out_path = 'submissions_heldout/submission_ft_sp_hk3_k1k2.json'
with open(out_path, 'w') as f:
    json.dump(final_h, f)
print(f'\nSaved: {out_path}')
print(f'Queries: {len(final_h)}  Docs/query: {len(list(final_h.values())[0])}')
print('Done.')
