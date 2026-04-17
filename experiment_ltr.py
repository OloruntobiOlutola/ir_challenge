"""
experiment_ltr.py
=================
Learning-to-Rank (LambdaRank) on top of the PRF pipeline.

Strategy:
  - Features: per-(query, doc) scores + ranks from all 6 models + domain weight
  - Labels : binary relevance from qrels_1.json
  - Train  : 100 val queries (cross-validated in 5 folds for evaluation)
  - Predict: 100 held-out queries using model trained on all 100 val queries

Features per (query, doc) pair  [19 total]:
   1-6  : raw scores  (BM25L, BGE-PRF, MiniLM-PRF, SPECTER2, TF-IDF-uni, TF-IDF-bi)
   7-12 : global rank within query  (1/rank, so higher = better)
  13    : current PRF pipeline rank (1/rank)
  14    : domain confusion weight  dw[q_domain][doc_domain]
  15    : same-domain binary flag
  16-17 : per-query score z-score for BGE-PRF and TF-IDF-uni
  18    : log(1 + BM25L score) -- BM25L scores can be large
  19    : number of models that include this doc in their top-30

Usage:
    conda run -n py_env python3 experiment_ltr.py
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
import lightgbm as lgb

POOL_K    = 100   # candidate pool (current pipeline top-N)
FEAT_TOP  = 30    # "how many models include this doc in top-30" feature
SEED      = 42

# ── Data ─────────────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
qrels      = json.load(open('data/qrels_1.json'))
q_df       = pd.read_parquet('data/queries.parquet').set_index('doc_id')
held_qids  = q_df.index.tolist()
corpus_arr = np.array(corpus_ids)
ci_to_idx  = {cid: i for i, cid in enumerate(corpus_ids)}

dw_df     = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw        = dw_df.to_dict(orient='index')
c_df      = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df     = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_dom_map = c_df['domain'].to_dict()
q_dom_val  = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids  if qid in q1_df.index}
q_dom_held = {qid: q_df.loc[qid,  'domain'] for qid in held_qids if qid in q_df.index}
c_domains  = np.array([c_dom_map.get(cid, '') for cid in corpus_ids])

WEAK_DOMS   = {'Computer Science', 'Biology', 'Medicine', 'Philosophy', 'Art', 'Engineering'}
STRONG_DOMS = {'Geology', 'Mathematics', 'Political Science', 'Economics', 'History', 'Sociology',
               'Chemistry', 'Physics', 'Materials Science'}

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


# ── Helpers ───────────────────────────────────────────────────────────────────
def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q / (nq+1e-10)) @ (c/(nc+1e-10)).T).astype(np.float32)

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

def apply_ml_prf(ml_qv, ml_cv, seed_sub, qids, q_dom_map, k=5):
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
    qe = bge_qv.copy()
    for i, qid in enumerate(qids):
        docs = seed_sub.get(qid, [])[:k]
        idxs = [ci_to_idx[doc] for doc in docs if doc in ci_to_idx]
        if not idxs: continue
        fb = bge_c[idxs].mean(axis=0)
        nq = qe[i] + beta * fb
        qe[i] = nq / (np.linalg.norm(nq) + 1e-10)
    return qe

def rrf_nested(flat_unf, flat_sf, qids, q_dom_map, dom_ndcg, threshold=0.70):
    sub = {}
    for qid in qids:
        qdom   = q_dom_map.get(qid, '')
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

        avail = set(flat_unf.keys()) | set(flat_sf.keys())
        inner_models = [m for m in ['BM25L', 'BGE', 'MiniLM'] if m in active and m in avail]
        outer_extra  = [m for m in ['TF-IDF-uni', 'TF-IDF-bi', 'SPECTER2'] if m in active and m in avail]
        if not inner_models:
            inner_models = [m for m in ['BM25L', 'BGE', 'MiniLM'] if m in avail]

        isc = {}
        for mname in inner_models:
            for rank, doc in enumerate(get_lst(mname).get(qid, []), 1):
                isc[doc] = isc.get(doc, 0.0) + 1.0 / (1 + rank)
        inner_sorted = sorted(isc, key=isc.get, reverse=True)[:POOL_K]

        osc = {}
        for rank, doc in enumerate(inner_sorted, 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)
        for mname in outer_extra + [m for m in ['BGE', 'BM25L'] if m in active and m in avail]:
            for rank, doc in enumerate(get_lst(mname).get(qid, []), 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)

        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:POOL_K]
    return sub

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

def ndcg_at_k(ranked, rel_set, k=10):
    dcg  = sum(1.0/math.log2(i+2) for i, d in enumerate(ranked[:k]) if d in rel_set)
    idcg = sum(1.0/math.log2(i+2) for i in range(min(len(rel_set), k)))
    return dcg/idcg if idcg else 0.0

def ndcg10(sub):
    sc = [ndcg_at_k(sub.get(q, []), set(rels))
          for q, rels in qrels.items() if q in sub]
    return float(np.mean(sc)) if sc else 0.0

def recall100(sub):
    vals = []
    for qid, rels in qrels.items():
        rel_set = set(rels)
        if not rel_set: continue
        vals.append(sum(1 for d in sub.get(qid, [])[:100] if d in rel_set)/len(rel_set))
    return float(np.mean(vals))


# ── Load scores ───────────────────────────────────────────────────────────────
print('Loading all score matrices...')
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
sc_tuni_sp_v = rrf_fuse(sc_tuni_v, sc_sp_v, k=5)
sc_tbi_sp_v  = rrf_fuse(sc_tbi_v,  sc_sp_v, k=5)

sc_bm25l_h = np.load('submissions_heldout/scores_bm25l_ft.npy')
sc_tuni_h  = np.load('submissions_heldout/scores_tfidf_uni_ft.npy')
sc_tbi_h   = np.load('submissions_heldout/scores_tfidf_bi_ft.npy')
bge_qv_h   = np.load('submissions/bge_large_heldout_query_emb.npy')
sc_bge_h   = cosine(bge_qv_h, bge_c)
ml_qv_h    = np.load('data/embeddings/new_queries_minilm/query_embeddings.npy')
sc_ml_h    = (ml_qv_h @ ml_cv.T).astype(np.float32)
sp_q_h     = np.load('specter_prox_embed/queries_embeddings.npy')
sc_sp_h    = cosine(sp_q_h, sp_ft_c)
sc_tuni_sp_h = rrf_fuse(sc_tuni_h, sc_sp_h, k=5)
sc_tbi_sp_h  = rrf_fuse(sc_tbi_h,  sc_sp_h, k=5)

# ── Build PRF baseline submissions (val + held-out) ───────────────────────────
print('Building PRF pipeline submissions...')
masks_v = [domain_mask(q_dom_val.get(qid, ''), 300) for qid in val_qids]
masks_h = [domain_mask(q_dom_held.get(qid, ''), 300) for qid in held_qids]

def _top(sc, i, k=POOL_K): return corpus_arr[np.argsort(-sc[i])[:k]].tolist()
def _topf(sc, i, mask, k=POOL_K):
    s = sc[i].copy(); s[~mask] = -1e9
    return corpus_arr[np.argsort(-s)[:k]].tolist()

# Val retrieval
s_bm25l_vf = {val_qids[i]: _topf(sc_bm25l_v, i, masks_v[i]) for i in range(len(val_qids))}
s_bge_v    = {val_qids[i]: _top(sc_bge_v,    i) for i in range(len(val_qids))}
s_ml_v     = {val_qids[i]: _top(sc_ml_v,     i) for i in range(len(val_qids))}
s_sp_v     = {val_qids[i]: _top(sc_sp_v,     i) for i in range(len(val_qids))}
ht_tuni_vf = {val_qids[i]: _topf(sc_tuni_sp_v, i, masks_v[i]) for i in range(len(val_qids))}
ht_tbi_vf  = {val_qids[i]: _topf(sc_tbi_sp_v,  i, masks_v[i]) for i in range(len(val_qids))}

flat_v_base = {'BM25L': s_bm25l_vf, 'BGE': s_bge_v, 'MiniLM': s_ml_v,
               'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}
flat_v_sf   = {'BM25L': s_bm25l_vf, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

b0_v    = rrf_nested(flat_v_base, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
b0_v_dr = dr(b0_v, q_dom_val)
ml_prf_v   = apply_ml_prf(ml_qv_v, ml_cv, s_bge_v, val_qids, q_dom_val)
sc_ml_prf_v  = (ml_prf_v @ ml_cv.T).astype(np.float32)
bge_prf_v  = apply_bge_prf(bge_qv_v, bge_c, b0_v_dr, val_qids)
sc_bge_prf_v = cosine(bge_prf_v, bge_c)

s_ml_prf_v  = {val_qids[i]: _top(sc_ml_prf_v,  i) for i in range(len(val_qids))}
s_bge_prf_v = {val_qids[i]: _top(sc_bge_prf_v, i) for i in range(len(val_qids))}

flat_v_prf = {'BM25L': s_bm25l_vf, 'BGE': s_bge_prf_v, 'MiniLM': s_ml_prf_v,
              'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}
val_base    = rrf_nested(flat_v_prf, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
val_base_dr = dr(val_base, q_dom_val)
print(f'  Val baseline NDCG@10={ndcg10(val_base_dr):.4f}  R@100={recall100(val_base_dr):.4f}')

# Held-out retrieval
s_bm25l_hf = {held_qids[i]: _topf(sc_bm25l_h, i, masks_h[i]) for i in range(len(held_qids))}
s_bge_h    = {held_qids[i]: _top(sc_bge_h,    i) for i in range(len(held_qids))}
s_ml_h     = {held_qids[i]: _top(sc_ml_h,     i) for i in range(len(held_qids))}
s_sp_h     = {held_qids[i]: _top(sc_sp_h,     i) for i in range(len(held_qids))}
ht_tuni_hf = {held_qids[i]: _topf(sc_tuni_sp_h, i, masks_h[i]) for i in range(len(held_qids))}
ht_tbi_hf  = {held_qids[i]: _topf(sc_tbi_sp_h,  i, masks_h[i]) for i in range(len(held_qids))}

flat_h_base = {'BM25L': s_bm25l_hf, 'BGE': s_bge_h, 'MiniLM': s_ml_h,
               'SPECTER2': s_sp_h, 'TF-IDF-uni': ht_tuni_hf, 'TF-IDF-bi': ht_tbi_hf}
flat_h_sf   = {'BM25L': s_bm25l_hf, 'TF-IDF-uni': ht_tuni_hf, 'TF-IDF-bi': ht_tbi_hf}

b0_h    = rrf_nested(flat_h_base, flat_h_sf, held_qids, q_dom_held, DOM_MODEL_NDCG)
b0_h_dr = dr(b0_h, q_dom_held)
ml_prf_h   = apply_ml_prf(ml_qv_h, ml_cv, s_bge_h, held_qids, q_dom_held)
sc_ml_prf_h  = (ml_prf_h @ ml_cv.T).astype(np.float32)
bge_prf_h  = apply_bge_prf(bge_qv_h, bge_c, b0_h_dr, held_qids)
sc_bge_prf_h = cosine(bge_prf_h, bge_c)

s_ml_prf_h  = {held_qids[i]: _top(sc_ml_prf_h,  i) for i in range(len(held_qids))}
s_bge_prf_h = {held_qids[i]: _top(sc_bge_prf_h, i) for i in range(len(held_qids))}

flat_h_prf = {'BM25L': s_bm25l_hf, 'BGE': s_bge_prf_h, 'MiniLM': s_ml_prf_h,
              'SPECTER2': s_sp_h, 'TF-IDF-uni': ht_tuni_hf, 'TF-IDF-bi': ht_tbi_hf}
held_base    = rrf_nested(flat_h_prf, flat_h_sf, held_qids, q_dom_held, DOM_MODEL_NDCG)
held_base_dr = dr(held_base, q_dom_held)
print(f'  Held-out pipeline built: {len(held_base_dr)} queries')


# ── Feature extraction ────────────────────────────────────────────────────────
FEAT_NAMES = [
    'bm25l_score', 'bge_prf_score', 'ml_prf_score', 'sp_score',
    'tuni_score',  'tbi_score',
    'bm25l_rank',  'bge_prf_rank',  'ml_prf_rank',  'sp_rank',
    'tuni_rank',   'tbi_rank',
    'ensemble_rank',
    'dom_weight',  'same_domain',
    'bge_prf_zscore', 'tuni_zscore',
    'log1p_bm25l',
    'n_top30_models',
]
N_FEATS = len(FEAT_NAMES)
print(f'\nFeatures: {N_FEATS}  ({FEAT_NAMES})')


def extract_features(qids, q_dom_map, q_idx_map,
                     sc_bm25l, sc_bge_prf, sc_ml_prf, sc_sp, sc_tuni, sc_tbi,
                     ensemble_sub_dr, pool_sub):
    """
    Build feature matrix + label vector for a set of queries.

    qids        : list of query IDs to process
    q_idx_map   : dict mapping qid → row index in score matrices
    pool_sub    : dict qid→[doc_id, ...] — candidate pool (top-POOL_K from pipeline)
    ensemble_sub_dr : same as pool_sub but domain-reranked (the final ranked list)

    Returns X (n_pairs, N_FEATS), y (n_pairs,), groups (n_pairs per query list), pair_info
    """
    X_rows, y_rows, groups, pair_info = [], [], [], []

    for qid in qids:
        qi      = q_idx_map[qid]
        qdom    = q_dom_map.get(qid, '')
        rel_set = set(qrels.get(qid, []))
        candidates = pool_sub.get(qid, [])   # top-POOL_K from pipeline (pre-dr)
        if not candidates:
            continue

        # Per-query score vectors (for z-score normalisation)
        scores_bge_prf_q = sc_bge_prf[qi]
        scores_tuni_q    = sc_tuni[qi]
        mu_bge, sd_bge   = scores_bge_prf_q.mean(), scores_bge_prf_q.std() + 1e-9
        mu_tuni, sd_tuni = scores_tuni_q.mean(),    scores_tuni_q.std()    + 1e-9

        # Pre-compute 1/rank for each model over full corpus
        def inv_rank(sc_row):
            # returns array of shape (corpus_size,): 1/rank for each doc
            order = np.argsort(-sc_row)          # sorted doc indices
            r = np.empty_like(order, dtype=np.float32)
            r[order] = 1.0 / (np.arange(1, len(order)+1, dtype=np.float32))
            return r

        ir_bm25l   = inv_rank(sc_bm25l[qi])
        ir_bge_prf = inv_rank(sc_bge_prf[qi])
        ir_ml_prf  = inv_rank(sc_ml_prf[qi])
        ir_sp      = inv_rank(sc_sp[qi])
        ir_tuni    = inv_rank(sc_tuni[qi])
        ir_tbi     = inv_rank(sc_tbi[qi])

        # Pre-compute top-FEAT_TOP sets for each model
        top30 = {
            'bm25l':    set(np.argsort(-sc_bm25l[qi])[:FEAT_TOP]),
            'bge_prf':  set(np.argsort(-sc_bge_prf[qi])[:FEAT_TOP]),
            'ml_prf':   set(np.argsort(-sc_ml_prf[qi])[:FEAT_TOP]),
            'sp':       set(np.argsort(-sc_sp[qi])[:FEAT_TOP]),
            'tuni':     set(np.argsort(-sc_tuni[qi])[:FEAT_TOP]),
            'tbi':      set(np.argsort(-sc_tbi[qi])[:FEAT_TOP]),
        }

        # Ensemble rank from domain-reranked final list
        ens_ranked = ensemble_sub_dr.get(qid, [])
        ens_rank_map = {doc: (i+1) for i, doc in enumerate(ens_ranked)}

        n_added = 0
        for doc in candidates:
            di = ci_to_idx.get(doc)
            if di is None:
                continue
            doc_dom = c_dom_map.get(doc, '')
            dom_w   = dw.get(qdom, {}).get(doc_dom, 0.0)
            same_d  = 1.0 if doc_dom == qdom else 0.0
            ens_r   = ens_rank_map.get(doc, POOL_K + 1)

            n_t30 = sum(1 for s in top30.values() if di in s)

            feats = [
                float(sc_bm25l[qi, di]),
                float(sc_bge_prf[qi, di]),
                float(sc_ml_prf[qi, di]),
                float(sc_sp[qi, di]),
                float(sc_tuni[qi, di]),
                float(sc_tbi[qi, di]),
                float(ir_bm25l[di]),
                float(ir_bge_prf[di]),
                float(ir_ml_prf[di]),
                float(ir_sp[di]),
                float(ir_tuni[di]),
                float(ir_tbi[di]),
                1.0 / ens_r,
                dom_w,
                same_d,
                float((sc_bge_prf[qi, di] - mu_bge) / sd_bge),
                float((sc_tuni[qi, di] - mu_tuni) / sd_tuni),
                float(np.log1p(max(sc_bm25l[qi, di], 0))),
                float(n_t30),
            ]
            X_rows.append(feats)
            y_rows.append(1 if doc in rel_set else 0)
            pair_info.append((qid, doc))
            n_added += 1

        groups.append(n_added)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), groups, pair_info


print('\nBuilding val feature matrix...')
val_q_idx = {qid: i for i, qid in enumerate(val_qids)}
X_val, y_val, groups_val, pairs_val = extract_features(
    val_qids, q_dom_val, val_q_idx,
    sc_bm25l_v, sc_bge_prf_v, sc_ml_prf_v, sc_sp_v, sc_tuni_sp_v, sc_tbi_sp_v,
    val_base_dr, val_base,
)
print(f'  Val pairs: {len(X_val)}  positives: {y_val.sum()}  '
      f'({100*y_val.mean():.2f}% relevant)  features: {X_val.shape[1]}')

print('Building held-out feature matrix...')
held_q_idx = {qid: i for i, qid in enumerate(held_qids)}
X_hld, _, groups_hld, pairs_hld = extract_features(
    held_qids, q_dom_held, held_q_idx,
    sc_bm25l_h, sc_bge_prf_h, sc_ml_prf_h, sc_sp_h, sc_tuni_sp_h, sc_tbi_sp_h,
    held_base_dr, held_base,
)
print(f'  Held-out pairs: {len(X_hld)}  features: {X_hld.shape[1]}')


# ── LightGBM LambdaRank model ─────────────────────────────────────────────────
LGB_PARAMS = {
    'objective':        'lambdarank',
    'metric':           'ndcg',
    'ndcg_eval_at':     [10],
    'learning_rate':    0.05,
    'num_leaves':       31,
    'min_data_in_leaf': 5,
    'n_estimators':     300,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'lambda_l1':        0.1,
    'lambda_l2':        0.1,
    'verbose':          -1,
    'random_state':     SEED,
    'label_gain':       [0, 1],   # binary relevance
}


def scores_to_sub(pairs, scores, pool_k=100):
    """Convert (qid, doc) pairs + LTR scores into a {qid: [doc, ...]} submission."""
    from collections import defaultdict
    qd = defaultdict(list)
    for (qid, doc), sc in zip(pairs, scores):
        qd[qid].append((sc, doc))
    sub = {}
    for qid, lst in qd.items():
        lst.sort(reverse=True)
        sub[qid] = [d for _, d in lst[:pool_k]]
    return sub


# ── 5-fold cross-validation on val queries ───────────────────────────────────
print('\n' + '='*65)
print('5-fold cross-validation (query-level)')
print('='*65)

# Split at query level
query_indices = np.arange(len(val_qids))
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

cv_ndcg, cv_ndcg_base = [], []
fold_results = []

# Build cumulative group sizes for slicing
cum_groups_val = np.concatenate([[0], np.cumsum(groups_val)])

for fold, (train_qi, test_qi) in enumerate(kf.split(query_indices)):
    # Slice feature matrix by query
    tr_slices = [range(cum_groups_val[i], cum_groups_val[i+1]) for i in train_qi]
    te_slices = [range(cum_groups_val[i], cum_groups_val[i+1]) for i in test_qi]

    tr_rows = [idx for sl in tr_slices for idx in sl]
    te_rows = [idx for sl in te_slices for idx in sl]

    X_tr, y_tr = X_val[tr_rows], y_val[tr_rows]
    X_te, y_te = X_val[te_rows], y_val[te_rows]
    g_tr = [groups_val[i] for i in train_qi]
    g_te = [groups_val[i] for i in test_qi]

    # Train
    model = lgb.LGBMRanker(**LGB_PARAMS)
    model.fit(X_tr, y_tr, group=g_tr,
              eval_set=[(X_te, y_te)], eval_group=[g_te],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(period=-1)])

    # Predict on test fold
    scores_te  = model.predict(X_te)
    pairs_te   = [pairs_val[i] for i in te_rows]
    sub_ltr    = scores_to_sub(pairs_te, scores_te)

    # Baseline for this fold
    test_qids_fold = [val_qids[i] for i in test_qi]
    sub_base_fold  = {q: val_base_dr[q] for q in test_qids_fold if q in val_base_dr}

    nd_ltr  = ndcg10(sub_ltr)
    nd_base = ndcg10(sub_base_fold)
    cv_ndcg.append(nd_ltr)
    cv_ndcg_base.append(nd_base)
    fold_results.append((fold+1, nd_base, nd_ltr, nd_ltr - nd_base))
    print(f'  Fold {fold+1}: baseline={nd_base:.4f}  LTR={nd_ltr:.4f}  Δ={nd_ltr-nd_base:+.4f}'
          f'  (best_iter={model.best_iteration_})')

print(f'\n  CV mean NDCG@10 — Baseline: {np.mean(cv_ndcg_base):.4f}  '
      f'LTR: {np.mean(cv_ndcg):.4f}  Δ={np.mean(cv_ndcg)-np.mean(cv_ndcg_base):+.4f}')


# ── Train on ALL val queries and predict held-out ─────────────────────────────
print('\n' + '='*65)
print('Full training on all val queries → held-out prediction')
print('='*65)

full_model = lgb.LGBMRanker(**LGB_PARAMS)
full_model.fit(X_val, y_val, group=groups_val,
               callbacks=[lgb.log_evaluation(period=-1)])
print(f'  Trained on {len(val_qids)} queries  ({len(X_val)} pairs)')

# Val self-evaluation (train=test, upper bound)
scores_val_self = full_model.predict(X_val)
sub_ltr_val     = scores_to_sub(pairs_val, scores_val_self)
nd_val_self     = ndcg10(sub_ltr_val)
rc_val_self     = recall100(sub_ltr_val)
print(f'  Val self-score  : NDCG@10={nd_val_self:.4f}  R@100={rc_val_self:.4f}')
print(f'  Val baseline    : NDCG@10={ndcg10(val_base_dr):.4f}  R@100={recall100(val_base_dr):.4f}')
print(f'  CV estimate     : NDCG@10={np.mean(cv_ndcg):.4f}  (cross-validated, realistic)')

# Predict held-out
scores_hld  = full_model.predict(X_hld)
sub_ltr_hld = scores_to_sub(pairs_hld, scores_hld)
print(f'\n  Held-out queries: {len(sub_ltr_hld)}')
print(f'  Docs per query  : {set(len(v) for v in sub_ltr_hld.values())}')

# Feature importances
fi = sorted(zip(FEAT_NAMES, full_model.feature_importances_), key=lambda x: -x[1])
print('\n  Feature importances (top 10):')
for fname, imp in fi[:10]:
    print(f'    {fname:<22} {imp:6.0f}')


# ── Per-domain CV breakdown ───────────────────────────────────────────────────
print('\n' + '='*65)
print('Per-domain breakdown (full val, train=test self-score)')
print('='*65)
print(f'{"Domain":<25} {"Baseline":>8}  {"LTR":>8}  {"Δ":>7}')
print('-'*55)
all_domains = sorted(set(q_dom_val.values()))
for dom in all_domains:
    qids_dom = [q for q, d in q_dom_val.items() if d == dom and q in qrels]
    if not qids_dom: continue
    sub_base_dom = {q: val_base_dr[q] for q in qids_dom if q in val_base_dr}
    sub_ltr_dom  = {q: sub_ltr_val[q]  for q in qids_dom if q in sub_ltr_val}
    nd_b = ndcg10(sub_base_dom)
    nd_l = ndcg10(sub_ltr_dom)
    print(f'  {dom:<23} {nd_b:8.4f}  {nd_l:8.4f}  {nd_l-nd_b:+7.4f}')


# ── Save held-out submission ──────────────────────────────────────────────────
print('\n' + '='*65)
out_path = Path('submissions_heldout/submission_ltr.json')
with open(out_path, 'w') as f:
    json.dump(sub_ltr_hld, f)
print(f'  Saved: {out_path}')
print(f'  CV NDCG@10 estimate: {np.mean(cv_ndcg):.4f}  (baseline: {np.mean(cv_ndcg_base):.4f})')
print(f'  Upload to Codabench to get true held-out score.')
print('='*65)
