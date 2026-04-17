"""
experiment_weighted_rrf.py
==========================
Test two improvements over the 0.7597 PRF baseline:

  A) Weighted RRF: replace binary model exclusion (threshold=0.70) with
     continuous per-domain NDCG weights.  Each model contributes
       w_m / (k + rank_m(doc))
     where w_m = DOM_MODEL_NDCG[domain][m] (or 0 if absent).
     Benefit: no hard cliff at 0.70 — models near threshold are handled
     smoothly; domain mis-classification causes a soft degradation, not
     a cascade failure.

  B) Ablation: drop MiniLM from the inner ensemble (it's the weakest
     dense model; removing it may clean up the inner RRF signal).

  C) Both together.

Baseline: current nested RRF + binary exclusion  → Val NDCG@10 = 0.7597
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ── Shared config / data (identical to build_prf_submission.py) ──────────────
POOL_K = 100

val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
qrels      = json.load(open('data/qrels_1.json'))
corpus_arr = np.array(corpus_ids)
ci_to_idx  = {cid: i for i, cid in enumerate(corpus_ids)}

dw_df     = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw        = dw_df.to_dict(orient='index')
c_df      = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df     = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_dom_map = c_df['domain'].to_dict()
q_dom_val = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids if qid in q1_df.index}
c_domains = np.array([c_dom_map.get(cid, '') for cid in corpus_ids])

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

WEAK_DOMS   = {'Computer Science', 'Biology', 'Medicine', 'Philosophy', 'Art', 'Engineering'}
STRONG_DOMS = {'Geology', 'Mathematics', 'Political Science', 'Economics', 'History', 'Sociology',
               'Chemistry', 'Physics', 'Materials Science'}


# ── Shared helpers ────────────────────────────────────────────────────────────
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


# ── BASELINE: current nested RRF + binary exclusion ──────────────────────────
def rrf_baseline(flat_unf, flat_sf, qids, q_dom_map, dom_ndcg, threshold=0.70):
    """Existing pipeline: binary exclusion at threshold × best, inner/outer RRF."""
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

        all_available = set(flat_unf.keys()) | set(flat_sf.keys())
        inner_models = [m for m in ['BM25L', 'BGE', 'MiniLM']
                        if m in active and m in all_available]
        outer_extra  = [m for m in ['TF-IDF-uni', 'TF-IDF-bi', 'SPECTER2']
                        if m in active and m in all_available]
        if not inner_models:
            inner_models = [m for m in ['BM25L', 'BGE', 'MiniLM'] if m in all_available]

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


# ── VARIANT A: weighted RRF (continuous domain-NDCG weights) ─────────────────
def rrf_weighted(flat_unf, flat_sf, qids, q_dom_map, dom_ndcg,
                 inner_k=1, outer_k=2, no_minilm=False):
    """
    Same inner/outer structure, but binary exclusion replaced by continuous
    per-domain NDCG weights: each model contributes
        w_m / (k + rank_m(doc))
    where w_m = dom_ndcg[domain][model] (raw val NDCG, not thresholded).
    Models absent from the table get weight = global_mean_ndcg.
    """
    all_models = set(flat_unf.keys())
    inner_names = ['BM25L', 'BGE'] if no_minilm else ['BM25L', 'BGE', 'MiniLM']
    outer_names = ['TF-IDF-uni', 'TF-IDF-bi', 'SPECTER2']
    # outer also re-adds BGE and BM25L directly (like baseline)
    outer_direct = ['BGE', 'BM25L']

    sub = {}
    for qid in qids:
        qdom   = q_dom_map.get(qid, '')
        dscores = dom_ndcg.get(qdom, {})
        # fallback weight = mean of all available scores in this domain
        fallback = float(np.mean(list(dscores.values()))) if dscores else 0.5

        def w(mname):
            return dscores.get(mname, fallback)

        def get_lst(mname):
            return flat_sf.get(mname, flat_unf[mname])

        # Inner weighted RRF (dense models)
        isc = {}
        for mname in [m for m in inner_names if m in all_models]:
            wt = w(mname)
            if wt == 0:
                continue
            for rank, doc in enumerate(get_lst(mname).get(qid, []), 1):
                isc[doc] = isc.get(doc, 0.0) + wt / (inner_k + rank)
        inner_sorted = sorted(isc, key=isc.get, reverse=True)[:POOL_K]

        # Outer weighted RRF
        osc = {}
        # inner result as a virtual "model"
        for rank, doc in enumerate(inner_sorted, 1):
            # weight for inner bundle = average weight of its components
            inner_w = float(np.mean([w(m) for m in inner_names if m in all_models]))
            osc[doc] = osc.get(doc, 0.0) + inner_w / (outer_k + rank)
        # add outer models
        for mname in [m for m in outer_names + outer_direct if m in all_models]:
            wt = w(mname)
            if wt == 0:
                continue
            for rank, doc in enumerate(get_lst(mname).get(qid, []), 1):
                osc[doc] = osc.get(doc, 0.0) + wt / (outer_k + rank)

        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:POOL_K]
    return sub


# ── Load val scores ───────────────────────────────────────────────────────────
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

masks_v    = [domain_mask(q_dom_val.get(qid, ''), 300) for qid in val_qids]
s_bm25l_vf = top100_filtered(sc_bm25l_v, val_qids, masks_v)
s_bge_v    = top100(sc_bge_v, val_qids)
s_ml_v     = top100(sc_ml_v,  val_qids)
s_sp_v     = top100(sc_sp_v,  val_qids)
ht_tuni_vf = top100_filtered(rrf_fuse(sc_tuni_v, sc_sp_v, k=5), val_qids, masks_v)
ht_tbi_vf  = top100_filtered(rrf_fuse(sc_tbi_v,  sc_sp_v, k=5), val_qids, masks_v)

flat_v_base = {'BM25L': s_bm25l_vf, 'BGE': s_bge_v, 'MiniLM': s_ml_v,
               'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}
flat_v_sf   = {'BM25L': s_bm25l_vf, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

# ── PRF helpers (same as baseline) ───────────────────────────────────────────
def build_prf_flats(b0_sub, b0_sub_dr):
    """Given a b0 seed dict, build PRF-enhanced flat maps."""
    ml_qv_prf = apply_ml_prf(ml_qv_v, ml_cv, s_bge_v, val_qids, q_dom_val, k=5)
    sc_ml_prf = (ml_qv_prf @ ml_cv.T).astype(np.float32)
    s_ml_prf  = top100(sc_ml_prf, val_qids)

    bge_qv_prf = apply_bge_prf(bge_qv_v, bge_c, b0_sub_dr, val_qids, k=5, beta=0.06)
    sc_bge_prf = cosine(bge_qv_prf, bge_c)
    s_bge_prf  = top100(sc_bge_prf, val_qids)

    flat_prf = {'BM25L': s_bm25l_vf, 'BGE': s_bge_prf, 'MiniLM': s_ml_prf,
                'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}
    return flat_prf, s_bge_v  # return s_bge_v so MiniLM PRF seed is consistent


# ── Run experiments ──────────────────────────────────────────────────────────
print('\n' + '='*70)
print('EXPERIMENT RESULTS')
print('='*70)
results = {}

# ── Baseline ─────────────────────────────────────────────────────────────────
b0_v    = rrf_baseline(flat_v_base, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
b0_v_dr = dr(b0_v, q_dom_val)

ml_qv_prf_v = apply_ml_prf(ml_qv_v, ml_cv, s_bge_v, val_qids, q_dom_val, k=5)
sc_ml_prf_v = (ml_qv_prf_v @ ml_cv.T).astype(np.float32)
s_ml_prf_v  = top100(sc_ml_prf_v, val_qids)
bge_qv_prf_v = apply_bge_prf(bge_qv_v, bge_c, b0_v_dr, val_qids, k=5, beta=0.06)
sc_bge_prf_v = cosine(bge_qv_prf_v, bge_c)
s_bge_prf_v  = top100(sc_bge_prf_v, val_qids)
flat_v_prf = {'BM25L': s_bm25l_vf, 'BGE': s_bge_prf_v, 'MiniLM': s_ml_prf_v,
              'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

val_base    = rrf_baseline(flat_v_prf, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
val_base_dr = dr(val_base, q_dom_val)
results['Baseline (binary excl.)'] = (ndcg10(val_base_dr), recall100(val_base_dr))
print(f"Baseline              : NDCG@10={results['Baseline (binary excl.)'][0]:.4f}  R@100={results['Baseline (binary excl.)'][1]:.4f}")

# ── Variant A: weighted RRF, PRF seed from weighted b0 ───────────────────────
b0_w    = rrf_weighted(flat_v_base, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
b0_w_dr = dr(b0_w, q_dom_val)

ml_qv_prf_w  = apply_ml_prf(ml_qv_v, ml_cv, s_bge_v, val_qids, q_dom_val, k=5)
sc_ml_prf_w  = (ml_qv_prf_w @ ml_cv.T).astype(np.float32)
s_ml_prf_w   = top100(sc_ml_prf_w, val_qids)
bge_qv_prf_w = apply_bge_prf(bge_qv_v, bge_c, b0_w_dr, val_qids, k=5, beta=0.06)
sc_bge_prf_w = cosine(bge_qv_prf_w, bge_c)
s_bge_prf_w  = top100(sc_bge_prf_w, val_qids)
flat_v_prf_w = {'BM25L': s_bm25l_vf, 'BGE': s_bge_prf_w, 'MiniLM': s_ml_prf_w,
                'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

val_w    = rrf_weighted(flat_v_prf_w, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
val_w_dr = dr(val_w, q_dom_val)
results['Weighted RRF'] = (ndcg10(val_w_dr), recall100(val_w_dr))
print(f"Weighted RRF          : NDCG@10={results['Weighted RRF'][0]:.4f}  R@100={results['Weighted RRF'][1]:.4f}  Δ={results['Weighted RRF'][0]-results['Baseline (binary excl.)'][0]:+.4f}")

# ── Variant B: drop MiniLM from baseline ────────────────────────────────────
b0_nom   = rrf_baseline(flat_v_base, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)  # same seed
b0_nom_dr = dr(b0_nom, q_dom_val)

# PRF with same seed as baseline
bge_qv_prf_nom = apply_bge_prf(bge_qv_v, bge_c, b0_nom_dr, val_qids, k=5, beta=0.06)
sc_bge_prf_nom = cosine(bge_qv_prf_nom, bge_c)
s_bge_prf_nom  = top100(sc_bge_prf_nom, val_qids)

# For MiniLM PRF we still compute it but drop from final flat
ml_qv_prf_nom = apply_ml_prf(ml_qv_v, ml_cv, s_bge_v, val_qids, q_dom_val, k=5)
sc_ml_prf_nom = (ml_qv_prf_nom @ ml_cv.T).astype(np.float32)
s_ml_prf_nom  = top100(sc_ml_prf_nom, val_qids)

# Final flat WITHOUT MiniLM
flat_v_prf_nom = {'BM25L': s_bm25l_vf, 'BGE': s_bge_prf_nom,
                  'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}
flat_v_sf_nom  = {'BM25L': s_bm25l_vf, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

# DOM_MODEL_NDCG without MiniLM for exclusion threshold computation
dom_ndcg_nom = {d: {m: s for m, s in v.items() if m != 'MiniLM'}
                for d, v in DOM_MODEL_NDCG.items()}

val_nom    = rrf_baseline(flat_v_prf_nom, flat_v_sf_nom, val_qids, q_dom_val, dom_ndcg_nom)
val_nom_dr = dr(val_nom, q_dom_val)
results['No MiniLM (binary excl.)'] = (ndcg10(val_nom_dr), recall100(val_nom_dr))
print(f"No MiniLM             : NDCG@10={results['No MiniLM (binary excl.)'][0]:.4f}  R@100={results['No MiniLM (binary excl.)'][1]:.4f}  Δ={results['No MiniLM (binary excl.)'][0]-results['Baseline (binary excl.)'][0]:+.4f}")

# ── Variant C: weighted RRF + no MiniLM ─────────────────────────────────────
b0_cn    = rrf_weighted(flat_v_base, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG,
                        no_minilm=True)
b0_cn_dr = dr(b0_cn, q_dom_val)

bge_qv_prf_cn = apply_bge_prf(bge_qv_v, bge_c, b0_cn_dr, val_qids, k=5, beta=0.06)
sc_bge_prf_cn = cosine(bge_qv_prf_cn, bge_c)
s_bge_prf_cn  = top100(sc_bge_prf_cn, val_qids)
flat_v_prf_cn = {'BM25L': s_bm25l_vf, 'BGE': s_bge_prf_cn,
                 'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

val_cn    = rrf_weighted(flat_v_prf_cn, flat_v_sf_nom, val_qids, q_dom_val,
                         dom_ndcg_nom, no_minilm=True)
val_cn_dr = dr(val_cn, q_dom_val)
results['Weighted RRF + no MiniLM'] = (ndcg10(val_cn_dr), recall100(val_cn_dr))
print(f"Weighted + no MiniLM  : NDCG@10={results['Weighted RRF + no MiniLM'][0]:.4f}  R@100={results['Weighted RRF + no MiniLM'][1]:.4f}  Δ={results['Weighted RRF + no MiniLM'][0]-results['Baseline (binary excl.)'][0]:+.4f}")

# ── Per-domain breakdown for best variant ────────────────────────────────────
print('\n' + '='*70)
print('Per-domain NDCG@10 breakdown')
print('='*70)
best_key = max(results, key=lambda k: results[k][0])
print(f'Best variant: {best_key}  (NDCG@10={results[best_key][0]:.4f})')
print()

# pick the best variant's dr output
best_sub_dr = {
    'Baseline (binary excl.)':      val_base_dr,
    'Weighted RRF':                  val_w_dr,
    'No MiniLM (binary excl.)':      val_nom_dr,
    'Weighted RRF + no MiniLM':      val_cn_dr,
}[best_key]

domains = sorted({q_dom_val.get(q, '') for q in qrels if q in q_dom_val})
print(f"{'Domain':<25} {'Base':>6}  {'Best':>6}  {'Δ':>6}")
print('-'*50)
for dom in domains:
    qids_dom = [q for q, d in q_dom_val.items() if d == dom and q in qrels]
    if not qids_dom:
        continue
    def dom_ndcg(sub_dr):
        sc = []
        for q in qids_dom:
            rels = qrels.get(q, [])
            if not rels: continue
            ranked = sub_dr.get(q, [])[:10]
            rel_set = set(rels)
            dcg  = sum(1/math.log2(i+2) for i,d in enumerate(ranked) if d in rel_set)
            idcg = sum(1/math.log2(i+2) for i in range(min(len(rel_set),10)))
            sc.append(dcg/idcg if idcg else 0)
        return float(np.mean(sc)) if sc else 0.0
    b = dom_ndcg(val_base_dr)
    be = dom_ndcg(best_sub_dr)
    print(f"  {dom:<23} {b:6.4f}  {be:6.4f}  {be-b:+6.4f}")

print()
print('='*70)
print('Summary')
print('='*70)
for name, (nd, rc) in results.items():
    delta = nd - results['Baseline (binary excl.)'][0]
    print(f"  {name:<30} NDCG@10={nd:.4f}  R@100={rc:.4f}  Δ={delta:+.4f}")
