"""
Build held-out submission with Dense PRF (best config):
  - Baseline: Nested RRF + model exclusion (thresh=0.70) + sparse domain filter
  - ML PRF: domain-adaptive β (weak=0.8, med=0.6, strong=0.0), k=5, seed=BGE top-100
  - BGE PRF: β=0.06, k=5, seed=baseline ensemble
  - Domain hard-sort reranking

Val NDCG@10 = 0.7597  R@100 = 0.8872
"""

POOL_K = 100
import json
import numpy as np
import pandas as pd

# ── IDs ────────────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
qrels      = json.load(open('data/qrels_1.json'))
q_df       = pd.read_parquet('data/queries.parquet').set_index('doc_id')
held_qids  = q_df.index.tolist()
corpus_arr = np.array(corpus_ids)
ci_to_idx  = {cid: i for i, cid in enumerate(corpus_ids)}

# ── Domain maps ────────────────────────────────────────────────────────────
dw_df     = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw        = dw_df.to_dict(orient='index')
c_df      = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df     = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_dom_map = c_df['domain'].to_dict()
q_dom_val  = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids  if qid in q1_df.index}
q_dom_held = {qid: q_df.loc[qid,  'domain'] for qid in held_qids if qid in q_df.index}
c_domains  = np.array([c_dom_map.get(cid, '') for cid in corpus_ids])

# ── Per-domain model NDCG table ────────────────────────────────────────────
DOM_MODEL_NDCG = {
    'Art':                  {'BM25L':0.5294,'BGE':0.5414,'MiniLM':0.5856,'SPECTER2':0.1952,'TF-IDF-uni':0.5294,'TF-IDF-bi':0.5135},
    'Biology':              {'BM25L':0.4822,'BGE':0.6187,'MiniLM':0.6042,'SPECTER2':0.5744,'TF-IDF-uni':0.6833,'TF-IDF-bi':0.6492},
    'Business':             {'BM25L':0.6709,'BGE':0.6792,'MiniLM':0.4708,'SPECTER2':0.6088,'TF-IDF-uni':0.7408,'TF-IDF-bi':0.6537},
    'Chemistry':            {'BM25L':0.7473,'BGE':0.7055,'MiniLM':0.6281,'SPECTER2':0.6423,'TF-IDF-uni':0.7702,'TF-IDF-bi':0.7684},
    'Computer Science':     {'BM25L':0.4600,'BGE':0.4922,'MiniLM':0.4733,'SPECTER2':0.4904,'TF-IDF-uni':0.5291,'TF-IDF-bi':0.5391},
    'Economics':            {'BM25L':1.0000,'BGE':1.0000,'MiniLM':1.0000,'SPECTER2':1.0000,'TF-IDF-uni':1.0000,'TF-IDF-bi':1.0000},
    'Engineering':          {'BM25L':0.5339,'BGE':0.8408,'MiniLM':0.5976,'SPECTER2':0.6011,'TF-IDF-uni':0.5751,'TF-IDF-bi':0.5616},
    'Environmental Science':{'BM25L':0.7740,'BGE':0.7832,'MiniLM':0.6849,'SPECTER2':0.7712,'TF-IDF-uni':0.9295,'TF-IDF-bi':0.9153},
    'Geography':            {'BM25L':0.7464,'BGE':0.8939,'MiniLM':0.8208,'SPECTER2':0.8377,'TF-IDF-uni':0.9159,'TF-IDF-bi':0.9159},
    'Geology':              {'BM25L':0.9260,'BGE':0.7491,'MiniLM':0.6793,'SPECTER2':0.7316,'TF-IDF-uni':0.8994,'TF-IDF-bi':0.8932},
    'History':              {'BM25L':1.0000,'BGE':1.0000,'MiniLM':1.0000,'SPECTER2':1.0000,'TF-IDF-uni':1.0000,'TF-IDF-bi':1.0000},
    'Materials Science':    {'BM25L':0.8746,'BGE':0.7890,'MiniLM':0.8164,'SPECTER2':0.7339,'TF-IDF-uni':0.8044,'TF-IDF-bi':0.7851},
    'Mathematics':          {'BM25L':0.8449,'BGE':0.8922,'MiniLM':0.7681,'SPECTER2':0.7980,'TF-IDF-uni':0.9914,'TF-IDF-bi':0.9634},
    'Medicine':             {'BM25L':0.5583,'BGE':0.6620,'MiniLM':0.6499,'SPECTER2':0.6163,'TF-IDF-uni':0.7122,'TF-IDF-bi':0.6772},
    'Philosophy':           {'BM25L':0.2372,'BGE':0.0000,'MiniLM':0.0000,'SPECTER2':0.0000,'TF-IDF-uni':0.2372,'TF-IDF-bi':0.6131},
    'Physics':              {'BM25L':0.8158,'BGE':0.5425,'MiniLM':0.6646,'SPECTER2':0.6646,'TF-IDF-uni':0.8400,'TF-IDF-bi':0.7557},
    'Political Science':    {'BM25L':0.9197,'BGE':0.6131,'MiniLM':0.7904,'SPECTER2':0.3066,'TF-IDF-uni':1.0000,'TF-IDF-bi':1.0000},
    'Psychology':           {'BM25L':0.7031,'BGE':0.7813,'MiniLM':0.8010,'SPECTER2':0.8043,'TF-IDF-uni':0.9007,'TF-IDF-bi':0.9295},
    'Sociology':            {'BM25L':1.0000,'BGE':1.0000,'MiniLM':1.0000,'SPECTER2':1.0000,'TF-IDF-uni':1.0000,'TF-IDF-bi':1.0000},
}

# PRF domain groups
WEAK_DOMS   = {'Computer Science', 'Biology', 'Medicine', 'Philosophy', 'Art', 'Engineering'}
STRONG_DOMS = {'Geology', 'Mathematics', 'Political Science', 'Economics', 'History', 'Sociology',
               'Chemistry', 'Physics', 'Materials Science'}

# ── Helpers ────────────────────────────────────────────────────────────────
def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q / (nq + 1e-10)) @ (c / (nc + 1e-10)).T).astype(np.float32)

def top100(scores, qids):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-scores[i])[:POOL_K]
        sub[qid] = corpus_arr[idx].tolist()
    return sub

def top100_filtered(score_matrix, qids, masks):
    sub = {}
    for i, qid in enumerate(qids):
        sc = score_matrix[i].copy()
        sc[~masks[i]] = -1e9
        idx = np.argsort(-sc)[:POOL_K]
        sub[qid] = corpus_arr[idx].tolist()
    return sub

def rrf_fuse(sa, sb, k=5):
    out = np.zeros_like(sa)
    for i in range(len(sa)):
        ra = np.argsort(np.argsort(-sa[i])) + 1
        rb = np.argsort(np.argsort(-sb[i])) + 1
        out[i] = 1.0 / (k + ra) + 1.0 / (k + rb)
    return out

def domain_mask(qdom, min_pool=300):
    row = dw.get(qdom, {})
    mask = np.array([row.get(d, 0.0) > 0.0 for d in c_domains])
    if mask.sum() < min_pool:
        return np.ones(len(corpus_ids), dtype=bool)
    return mask

def dr(sub, q_dom_map, skip=frozenset({'Business'})):
    out = {}
    for qid, cands in sub.items():
        qd = q_dom_map.get(qid, '')
        if qd in skip:
            out[qid] = cands[:100]; continue
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d, ''), 0.0), rank, d)
                  for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _, _, d in scored[:100]]
    return out

def ndcg10(sub):
    sc = []
    for q, rels in qrels.items():
        if q not in sub: continue
        ranked = sub[q][:10]
        rel_set = set(rels)
        dcg  = sum(1.0/np.log2(i+2) for i,d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0/np.log2(i+2) for i in range(min(len(rel_set),10)))
        sc.append(dcg/idcg if idcg else 0)
    return float(np.mean(sc))

def recall100(sub):
    vals = []
    for qid, rels in qrels.items():
        rel_set = set(rels)
        if not rel_set: continue
        vals.append(sum(1 for d in sub.get(qid, [])[:100] if d in rel_set) / len(rel_set))
    return float(np.mean(vals))

def rrf_nested_exclude_sf(flat_unf, flat_sf, qids, q_dom_map, dom_ndcg,
                          threshold=0.70):
    sub = {}
    for qid in qids:
        qdom = q_dom_map.get(qid, '')
        dscores = dom_ndcg.get(qdom, {})
        if dscores:
            best_s = max(dscores.values())
            active = {m for m, s in dscores.items() if s >= best_s * threshold}
        else:
            active = set(flat_unf.keys())
        if not active:
            active = set(flat_unf.keys())

        def get_lst(mname):
            return flat_sf.get(mname, flat_unf[mname])

        inner_models = [m for m in ['BM25L', 'BGE', 'MiniLM'] if m in active]
        outer_extra  = [m for m in ['TF-IDF-uni', 'TF-IDF-bi', 'SPECTER2'] if m in active]
        if not inner_models:
            inner_models = ['BM25L', 'BGE', 'MiniLM']

        isc = {}
        for mname in inner_models:
            for rank, doc in enumerate(get_lst(mname).get(qid, []), 1):
                isc[doc] = isc.get(doc, 0.0) + 1.0 / (1 + rank)
        inner_sorted = sorted(isc, key=isc.get, reverse=True)[:POOL_K]

        osc = {}
        for rank, doc in enumerate(inner_sorted, 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)
        for mname in outer_extra + [m for m in ['BGE', 'BM25L'] if m in active]:
            for rank, doc in enumerate(get_lst(mname).get(qid, []), 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)

        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:POOL_K]
    return sub

def apply_ml_prf(ml_qv, ml_cv, seed_sub, qids, q_dom_map, k=5):
    """Domain-adaptive ML PRF: β=0.8 for weak, 0.6 for medium, 0.0 for strong."""
    qe = ml_qv.copy()
    for i, qid in enumerate(qids):
        d = q_dom_map.get(qid, '')
        b = 0.8 if d in WEAK_DOMS else (0.0 if d in STRONG_DOMS else 0.60)
        if b == 0: continue
        docs = seed_sub.get(qid, [])[:k]
        idxs = [ci_to_idx[doc] for doc in docs if doc in ci_to_idx]
        if not idxs: continue
        fb = ml_cv[idxs].mean(axis=0)
        nq = qe[i] + b * fb
        qe[i] = nq / (np.linalg.norm(nq) + 1e-10)
    return qe

def apply_bge_prf(bge_qv, bge_c, seed_sub, qids, k=5, beta=0.06):
    """Standard BGE PRF."""
    qe = bge_qv.copy()
    for i, qid in enumerate(qids):
        docs = seed_sub.get(qid, [])[:k]
        idxs = [ci_to_idx[doc] for doc in docs if doc in ci_to_idx]
        if not idxs: continue
        fb = bge_c[idxs].mean(axis=0)
        nq = qe[i] + beta * fb
        qe[i] = nq / (np.linalg.norm(nq) + 1e-10)
    return qe

# ── Load val scores ────────────────────────────────────────────────────────
print('Loading val scores...')
sc_bm25l_v = np.load('submissions/scores_bm25l_ft.npy')
sc_tuni_v  = np.load('submissions/scores_tfidf_uni_ft.npy')
sc_tbi_v   = np.load('submissions/scores_tfidf_bi_ft.npy')
bge_qv_v   = np.load('submissions/bge_large_query_emb.npy')
bge_c      = np.load('submissions/bge_large_corpus_emb.npy')
sc_bge_v   = cosine(bge_qv_v, bge_c)
ml_qv_v    = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy')
ml_cv      = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy')
sc_ml_v    = (ml_qv_v @ ml_cv.T).astype(np.float32)
sp_q_v     = np.load('specter_prox_embed/queries_1_embeddings.npy')
sp_ft_c    = np.load('submissions/specter2_corpus_emb_ft.npy')
sc_sp_v    = cosine(sp_q_v, sp_ft_c)

# ── Val: build baseline and PRF embeddings ─────────────────────────────────
print('Building val retrieval...')
masks_v = [domain_mask(q_dom_val.get(qid, ''), 300) for qid in val_qids]

s_bm25l_vf = top100_filtered(sc_bm25l_v, val_qids, masks_v)
s_bge_v    = top100(sc_bge_v, val_qids)
s_ml_v     = top100(sc_ml_v,  val_qids)
s_sp_v     = top100(sc_sp_v,  val_qids)
ht_tuni_vf = top100_filtered(rrf_fuse(sc_tuni_v, sc_sp_v, k=5), val_qids, masks_v)
ht_tbi_vf  = top100_filtered(rrf_fuse(sc_tbi_v,  sc_sp_v, k=5), val_qids, masks_v)

flat_v_base = {'BM25L': s_bm25l_vf, 'BGE': s_bge_v, 'MiniLM': s_ml_v,
               'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}
flat_v_sf   = {'BM25L': s_bm25l_vf, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

# Baseline ensemble (for BGE PRF seed)
b0_v    = rrf_nested_exclude_sf(flat_v_base, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
b0_v_dr = dr(b0_v, q_dom_val)

# ML PRF
ml_qv_prf_v = apply_ml_prf(ml_qv_v, ml_cv, s_bge_v, val_qids, q_dom_val, k=5)
sc_ml_prf_v = (ml_qv_prf_v @ ml_cv.T).astype(np.float32)
s_ml_prf_v  = top100(sc_ml_prf_v, val_qids)

# BGE PRF (seed = baseline ensemble)
bge_qv_prf_v = apply_bge_prf(bge_qv_v, bge_c, b0_v_dr, val_qids, k=5, beta=0.06)
sc_bge_prf_v = cosine(bge_qv_prf_v, bge_c)
s_bge_prf_v  = top100(sc_bge_prf_v, val_qids)

flat_v_prf = {'BM25L': s_bm25l_vf, 'BGE': s_bge_prf_v, 'MiniLM': s_ml_prf_v,
              'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

val_sub    = rrf_nested_exclude_sf(flat_v_prf, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
val_sub_dr = dr(val_sub, q_dom_val)

print(f'  Val NDCG@10 = {ndcg10(val_sub_dr):.4f}  (expected ~0.7598)')
print(f'  Val R@100   = {recall100(val_sub_dr):.4f}')

# ── Load held-out scores ───────────────────────────────────────────────────
print('\nLoading held-out scores...')
sc_bm25l_h = np.load('submissions_heldout/scores_bm25l_ft.npy')
sc_tuni_h  = np.load('submissions_heldout/scores_tfidf_uni_ft.npy')
sc_tbi_h   = np.load('submissions_heldout/scores_tfidf_bi_ft.npy')
bge_qv_h   = np.load('submissions/bge_large_heldout_query_emb.npy')
sc_bge_h   = cosine(bge_qv_h, bge_c)
ml_qv_h    = np.load('data/embeddings/new_queries_minilm/query_embeddings.npy')
sc_ml_h    = (ml_qv_h @ ml_cv.T).astype(np.float32)
sp_q_h     = np.load('specter_prox_embed/queries_embeddings.npy')
sc_sp_h    = cosine(sp_q_h, sp_ft_c)

# ── Held-out: build baseline and PRF ──────────────────────────────────────
print('Building held-out retrieval...')
masks_h = [domain_mask(q_dom_held.get(qid, ''), 300) for qid in held_qids]

s_bm25l_hf = top100_filtered(sc_bm25l_h, held_qids, masks_h)
s_bge_h    = top100(sc_bge_h, held_qids)
s_ml_h     = top100(sc_ml_h,  held_qids)
s_sp_h     = top100(sc_sp_h,  held_qids)
ht_tuni_hf = top100_filtered(rrf_fuse(sc_tuni_h, sc_sp_h, k=5), held_qids, masks_h)
ht_tbi_hf  = top100_filtered(rrf_fuse(sc_tbi_h,  sc_sp_h, k=5), held_qids, masks_h)

flat_h_base = {'BM25L': s_bm25l_hf, 'BGE': s_bge_h, 'MiniLM': s_ml_h,
               'SPECTER2': s_sp_h, 'TF-IDF-uni': ht_tuni_hf, 'TF-IDF-bi': ht_tbi_hf}
flat_h_sf   = {'BM25L': s_bm25l_hf, 'TF-IDF-uni': ht_tuni_hf, 'TF-IDF-bi': ht_tbi_hf}

# Baseline ensemble for BGE PRF seed
b0_h    = rrf_nested_exclude_sf(flat_h_base, flat_h_sf, held_qids, q_dom_held, DOM_MODEL_NDCG)
b0_h_dr = dr(b0_h, q_dom_held)

# ML PRF (domain-adaptive)
ml_qv_prf_h = apply_ml_prf(ml_qv_h, ml_cv, s_bge_h, held_qids, q_dom_held, k=5)
sc_ml_prf_h = (ml_qv_prf_h @ ml_cv.T).astype(np.float32)
s_ml_prf_h  = top100(sc_ml_prf_h, held_qids)

# BGE PRF (seed = baseline ensemble)
bge_qv_prf_h = apply_bge_prf(bge_qv_h, bge_c, b0_h_dr, held_qids, k=5, beta=0.06)
sc_bge_prf_h = cosine(bge_qv_prf_h, bge_c)
s_bge_prf_h  = top100(sc_bge_prf_h, held_qids)

flat_h_prf = {'BM25L': s_bm25l_hf, 'BGE': s_bge_prf_h, 'MiniLM': s_ml_prf_h,
              'SPECTER2': s_sp_h, 'TF-IDF-uni': ht_tuni_hf, 'TF-IDF-bi': ht_tbi_hf}

held_sub    = rrf_nested_exclude_sf(flat_h_prf, flat_h_sf, held_qids, q_dom_held, DOM_MODEL_NDCG)
held_sub_dr = dr(held_sub, q_dom_held)

print(f'Held-out queries: {len(held_sub_dr)}')
print(f'Docs per query:   {set(len(v) for v in held_sub_dr.values())}')

# ── Save ───────────────────────────────────────────────────────────────────
import os; os.makedirs('submissions_heldout', exist_ok=True)
out_path = 'submissions_heldout/submission_prf.json'
with open(out_path, 'w') as f:
    json.dump(held_sub_dr, f)

print(f'\nSaved: {out_path}')
print(f'Val NDCG@10 = {ndcg10(val_sub_dr):.4f}  R@100 = {recall100(val_sub_dr):.4f}')
print('Upload to Codabench for evaluation.')
