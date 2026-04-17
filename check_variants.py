"""Check which BM25 variants / unused scores we have and their recall."""
import numpy as np
import json

val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
qrels      = json.load(open('data/qrels_1.json'))
ca = np.array(corpus_ids)

def recall(sub):
    vals = []
    for q, r in qrels.items():
        rs = set(r)
        if not rs: continue
        vals.append(sum(1 for d in sub.get(q, [])[:100] if d in rs) / len(rs))
    return float(np.mean(vals))

def ndcg10(sub):
    sc = []
    for q, rels in qrels.items():
        if q not in sub: continue
        ranked = sub[q][:10]
        rel_set = set(rels)
        dcg  = sum(1.0/np.log2(i+2) for i,d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0/np.log2(i+2) for i in range(min(len(rel_set),10)))
        sc.append(dcg/idcg if idcg > 0 else 0)
    return float(np.mean(sc))

files = [
    'submissions/scores_bm25_ft.npy',
    'submissions/scores_bm25f_ft.npy',
    'submissions/scores_bm25l_ft.npy',
    'submissions/scores_bm25plus_ft.npy',
    'submissions/scores_tfidf_uni_ft.npy',
    'submissions/scores_tfidf_bi_ft.npy',
]

print(f'{"File":<40}  {"NDCG@10":>8}  {"R@100":>8}')
print('-' * 60)
for fname in files:
    sc = np.load(fname)
    sub = {}
    for i, q in enumerate(val_qids):
        idx = np.argsort(-sc[i])[:100]
        sub[q] = ca[idx].tolist()
    print(f'{fname:<40}  {ndcg10(sub):>8.4f}  {recall(sub):>8.4f}')

# Now check if adding bm25+ or bm25f to the existing ensemble improves things
# Load current best ensemble components
from collections import defaultdict
import pandas as pd

dw_df     = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw        = dw_df.to_dict(orient='index')
c_df      = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df     = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_dom_map = c_df['domain'].to_dict()
q_dom_val = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids if qid in q1_df.index}
c_domains = np.array([c_dom_map.get(cid, '') for cid in corpus_ids])

def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q / (nq + 1e-10)) @ (c / (nc + 1e-10)).T).astype(np.float32)

def rrf_fuse(sa, sb, k=5):
    out = np.zeros_like(sa)
    for i in range(len(sa)):
        ra = np.argsort(np.argsort(-sa[i])) + 1
        rb = np.argsort(np.argsort(-sb[i])) + 1
        out[i] = 1.0 / (k + ra) + 1.0 / (k + rb)
    return out

def top_n(scores, qids, n=100):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-scores[i])[:n]
        sub[qid] = ca[idx].tolist()
    return sub

def domain_mask(qdom, min_pool=300):
    row = dw.get(qdom, {})
    mask = np.array([row.get(d, 0.0) > 0.0 for d in c_domains])
    if mask.sum() < min_pool:
        return np.ones(len(corpus_ids), dtype=bool)
    return mask

def top_n_filtered(sc, qids, masks, n=100):
    sub = {}
    for i, qid in enumerate(qids):
        s = sc[i].copy()
        s[~masks[i]] = -1e9
        idx = np.argsort(-s)[:n]
        sub[qid] = ca[idx].tolist()
    return sub

def rrf(lists, qids, k=3, n=100):
    sub = {}
    for qid in qids:
        sc = {}
        for lst in lists:
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (k + rank)
        sub[qid] = sorted(sc, key=sc.get, reverse=True)[:n]
    return sub

def dr(sub, skip=frozenset({'Business'})):
    out = {}
    for qid, cands in sub.items():
        qd = q_dom_val.get(qid, '')
        if qd in skip:
            out[qid] = cands[:100]; continue
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d, ''), 0.0), rank, d)
                  for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _, _, d in scored[:100]]
    return out

sc_bm25l  = np.load('submissions/scores_bm25l_ft.npy')
sc_bm25   = np.load('submissions/scores_bm25_ft.npy')
sc_bm25f  = np.load('submissions/scores_bm25f_ft.npy')
sc_bm25p  = np.load('submissions/scores_bm25plus_ft.npy')
sc_tuni   = np.load('submissions/scores_tfidf_uni_ft.npy')
sc_tbi    = np.load('submissions/scores_tfidf_bi_ft.npy')
bge_qv    = np.load('submissions/bge_large_query_emb.npy')
bge_c     = np.load('submissions/bge_large_corpus_emb.npy')
sc_bge    = cosine(bge_qv, bge_c)
ml_qv     = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy')
ml_cv     = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy')
sc_ml     = (ml_qv @ ml_cv.T).astype(np.float32)
sp_q      = np.load('specter_prox_embed/queries_1_embeddings.npy')
sp_ft_c   = np.load('submissions/specter2_corpus_emb_ft.npy')
sc_sp     = cosine(sp_q, sp_ft_c)

sc_ht_tuni = rrf_fuse(sc_tuni, sc_sp, k=5)
sc_ht_tbi  = rrf_fuse(sc_tbi,  sc_sp, k=5)

masks = [domain_mask(q_dom_val.get(qid, ''), min_pool=300) for qid in val_qids]

# Baseline: bm25l only for sparse
s_bm25l = top_n(sc_bm25l, val_qids)
s_bge   = top_n(sc_bge,   val_qids)
s_ml    = top_n(sc_ml,    val_qids)
s_sp    = top_n(sc_sp,    val_qids)
ht_tuni = top_n(sc_ht_tuni, val_qids)
ht_tbi  = top_n(sc_ht_tbi,  val_qids)
inner   = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)
outer   = rrf([inner, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp], val_qids, k=2)
print(f'\nBaseline: NDCG={ndcg10(dr(outer)):.4f}  R@100={recall(outer):.4f}')

# Add bm25 variants to the inner/outer ensemble
print('\nAdding BM25 variants to ensemble:')
for vname, sc_v in [('bm25', sc_bm25), ('bm25f', sc_bm25f), ('bm25+', sc_bm25p)]:
    s_v = top_n(sc_v, val_qids)
    inner_v = rrf([s_bm25l, s_v, s_bge, s_ml], val_qids, k=1)
    outer_v = rrf([inner_v, s_bge, s_bm25l, s_v, ht_tuni, ht_tbi, s_sp], val_qids, k=2)
    print(f'  +{vname:<8}: NDCG={ndcg10(dr(outer_v)):.4f}  R@100={recall(outer_v):.4f}')

# Add filtered bm25 variants
print('\nAdding filtered BM25 variants:')
for vname, sc_v in [('bm25', sc_bm25), ('bm25f', sc_bm25f), ('bm25+', sc_bm25p)]:
    s_vf = top_n_filtered(sc_v, val_qids, masks)
    s_bm25l_f = top_n_filtered(sc_bm25l, val_qids, masks)
    ht_tuni_f = top_n_filtered(sc_ht_tuni, val_qids, masks)
    ht_tbi_f  = top_n_filtered(sc_ht_tbi,  val_qids, masks)
    inner_vf = rrf([s_bm25l_f, s_vf, s_bge, s_ml], val_qids, k=1)
    outer_vf = rrf([inner_vf, s_bge, s_bm25l_f, s_vf, ht_tuni_f, ht_tbi_f, s_sp], val_qids, k=2)
    print(f'  +{vname}(filtered): NDCG={ndcg10(dr(outer_vf)):.4f}  R@100={recall(outer_vf):.4f}')

# Triple BM25: use all BM25 variants together
print('\nAll 4 BM25 variants together (filtered):')
s_bm25_f  = top_n_filtered(sc_bm25,  val_qids, masks)
s_bm25f_f = top_n_filtered(sc_bm25f, val_qids, masks)
s_bm25p_f = top_n_filtered(sc_bm25p, val_qids, masks)
s_bm25l_f = top_n_filtered(sc_bm25l, val_qids, masks)
ht_tuni_f = top_n_filtered(sc_ht_tuni, val_qids, masks)
ht_tbi_f  = top_n_filtered(sc_ht_tbi,  val_qids, masks)
inner_all = rrf([s_bm25l_f, s_bm25_f, s_bm25f_f, s_bm25p_f, s_bge, s_ml], val_qids, k=1)
outer_all = rrf([inner_all, s_bge, s_bm25l_f, ht_tuni_f, ht_tbi_f, s_sp], val_qids, k=2)
print(f'  All BM25s: NDCG={ndcg10(dr(outer_all)):.4f}  R@100={recall(outer_all):.4f}')

# What's the recall if we take union of all BM25 variants?
union_recall = {}
all_bm25_subs = [top_n(sc, val_qids, 150) for sc in [sc_bm25l, sc_bm25, sc_bm25f, sc_bm25p]]
for q in val_qids:
    seen = set()
    merged = []
    for sub in all_bm25_subs:
        for d in sub.get(q, []):
            if d not in seen:
                seen.add(d)
                merged.append(d)
    union_recall[q] = merged[:150]
u_recall = float(__import__('numpy').mean([
    sum(1 for d in union_recall.get(q, [])[:150] if d in set(qrels.get(q, []))) / len(set(qrels.get(q, [])))
    for q in qrels if set(qrels[q])
]))
print(f'\n  Union of all BM25 variants @ 150: R@150={u_recall:.4f}')
