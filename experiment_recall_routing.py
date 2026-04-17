"""
experiment_recall_routing.py
=============================
Test a revised pipeline:
  1. Apply domain-weight mask to ALL models (not just sparse)
  2. Route by per-domain recall@100 (threshold=0.70×best) instead of NDCG@10
  3. Single flat RRF over all selected models (k=1)

Motivation:
  - Domain mask on dense models removes cross-domain noise at retrieval time
  - Recall@100 is the right metric for candidate-generation routing:
    a model that finds relevant docs at rank 80 is still useful for RRF,
    but gets penalised to 0 by NDCG@10
  - Flat RRF avoids the inner/outer coupling artefact

Baseline: 0.7597 NDCG@10  0.8872 R@100
"""

import json
import math
import numpy as np
import pandas as pd

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
    return ((q / (nq + 1e-10)) @ (c / (nc + 1e-10)).T).astype(np.float32)

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

def top_k_filtered(scores, qid, mask, k=POOL_K):
    sc = scores.copy()
    sc[~mask] = -1e9
    idx = np.argsort(-sc)[:k]
    return corpus_arr[idx].tolist()

def top_k(scores, k=POOL_K):
    idx = np.argsort(-scores)[:k]
    return corpus_arr[idx].tolist()

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

# pre-fused TF-IDF hybrid scores (still score matrices, not yet ranked)
sc_tuni_sp_v = rrf_fuse(sc_tuni_v, sc_sp_v, k=5)
sc_tbi_sp_v  = rrf_fuse(sc_tbi_v,  sc_sp_v, k=5)

# per-query masks (domain filter)
masks_v = [domain_mask(q_dom_val.get(qid, ''), 300) for qid in val_qids]

ALL_SCORES = {
    'BM25L':      sc_bm25l_v,
    'BGE':        sc_bge_v,
    'MiniLM':     sc_ml_v,
    'SPECTER2':   sc_sp_v,
    'TF-IDF-uni': sc_tuni_sp_v,
    'TF-IDF-bi':  sc_tbi_sp_v,
}


# ── STEP 1: Compute per-domain recall@100 for each model ─────────────────────
print('\nComputing per-domain recall@100 per model...')

DOM_MODEL_RECALL = {}  # DOM_MODEL_RECALL[domain][model] = recall@100

all_domains = sorted(set(q_dom_val.values()))

for dom in all_domains:
    qids_dom = [q for q, d in q_dom_val.items() if d == dom and q in qrels]
    if not qids_dom:
        continue
    DOM_MODEL_RECALL[dom] = {}
    for mname, sc_mat in ALL_SCORES.items():
        recs = []
        for qid in qids_dom:
            qi = val_qids.index(qid)
            mask = masks_v[val_qids.index(qid)]
            rel_set = set(qrels[qid])
            if not rel_set:
                continue
            # apply domain filter for this model too (same as what we'll do in pipeline)
            docs = top_k_filtered(sc_mat[qi], qid, mask, k=POOL_K)
            recs.append(sum(1 for d in docs if d in rel_set) / len(rel_set))
        DOM_MODEL_RECALL[dom][mname] = float(np.mean(recs)) if recs else 0.0

print(f'\n{"Domain":<25} {"BM25L":>6} {"BGE":>6} {"MiniLM":>6} {"SPEC2":>6} {"TFuni":>6} {"TFbi":>6}')
print('-' * 70)
for dom in all_domains:
    r = DOM_MODEL_RECALL.get(dom, {})
    print(f'  {dom:<23} {r.get("BM25L",0):6.4f} {r.get("BGE",0):6.4f} '
          f'{r.get("MiniLM",0):6.4f} {r.get("SPECTER2",0):6.4f} '
          f'{r.get("TF-IDF-uni",0):6.4f} {r.get("TF-IDF-bi",0):6.4f}')


# ── STEP 2: Build per-query domain-filtered retrievals (all models masked) ────
print('\nBuilding domain-filtered retrievals for all models...')

def build_filtered_subs(qids, q_dom_map, score_dict, masks):
    """For each model, return top-POOL_K docs after domain filter."""
    subs = {m: {} for m in score_dict}
    for i, qid in enumerate(qids):
        mask = masks[i]
        for mname, sc_mat in score_dict.items():
            subs[mname][qid] = top_k_filtered(sc_mat[i], qid, mask, k=POOL_K)
    return subs

subs_filtered = build_filtered_subs(val_qids, q_dom_val, ALL_SCORES, masks_v)


# ── STEP 3: Flat RRF with recall@100 routing ─────────────────────────────────
def rrf_recall_routed(subs, qids, q_dom_map, dom_recall, threshold=0.70, k_rrf=1):
    """
    Flat RRF over models selected by recall@100 per domain.
    If a model's domain recall < threshold * best_domain_recall, exclude it.
    """
    sub = {}
    for qid in qids:
        qdom   = q_dom_map.get(qid, '')
        rscores = dom_recall.get(qdom, {})
        if rscores:
            best_r  = max(rscores.values())
            active  = {m for m, r in rscores.items() if r >= best_r * threshold}
        else:
            active = set(subs.keys())
        if not active:
            active = set(subs.keys())

        osc = {}
        for mname in active:
            for rank, doc in enumerate(subs[mname].get(qid, []), 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (k_rrf + rank)
        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:POOL_K]
    return sub


# ── STEP 4: PRF on top of the recall-routed baseline ─────────────────────────
print('\nRunning recall-routed pipeline + PRF...')

# b0 seed from recall-routed ensemble
b0_rr    = rrf_recall_routed(subs_filtered, val_qids, q_dom_val, DOM_MODEL_RECALL)
b0_rr_dr = dr(b0_rr, q_dom_val)

# ML PRF (seed = BGE filtered top-k)
ml_qv_prf_rr = apply_ml_prf(ml_qv_v, ml_cv, subs_filtered['BGE'], val_qids, q_dom_val, k=5)
sc_ml_prf_rr = (ml_qv_prf_rr @ ml_cv.T).astype(np.float32)

# BGE PRF (seed = b0 ensemble)
bge_qv_prf_rr = apply_bge_prf(bge_qv_v, bge_c, b0_rr_dr, val_qids, k=5, beta=0.06)
sc_bge_prf_rr = cosine(bge_qv_prf_rr, bge_c)

# Rebuild filtered subs with PRF-enhanced BGE and MiniLM
ALL_SCORES_PRF = {
    'BM25L':      sc_bm25l_v,
    'BGE':        sc_bge_prf_rr,
    'MiniLM':     sc_ml_prf_rr,
    'SPECTER2':   sc_sp_v,
    'TF-IDF-uni': sc_tuni_sp_v,
    'TF-IDF-bi':  sc_tbi_sp_v,
}
subs_prf = build_filtered_subs(val_qids, q_dom_val, ALL_SCORES_PRF, masks_v)


# ── STEP 5: Run all configurations and compare ───────────────────────────────
print('\n' + '='*72)
print('RESULTS')
print('='*72)

configs = [
    ('k=1, thresh=0.70', dict(k_rrf=1, threshold=0.70)),
    ('k=1, thresh=0.60', dict(k_rrf=1, threshold=0.60)),
    ('k=1, thresh=0.50', dict(k_rrf=1, threshold=0.50)),
    ('k=2, thresh=0.70', dict(k_rrf=2, threshold=0.70)),
    ('k=2, thresh=0.60', dict(k_rrf=2, threshold=0.60)),
    ('k=5, thresh=0.70', dict(k_rrf=5, threshold=0.70)),
]

# Baseline for reference
from build_prf_submission import (
    rrf_nested_exclude_sf as _baseline_rrf,
    apply_ml_prf as _bml, apply_bge_prf as _bbge,
)

baseline_masks = [domain_mask(q_dom_val.get(qid, ''), 300) for qid in val_qids]
baseline_bm25l = {val_qids[i]: top_k_filtered(sc_bm25l_v[i], val_qids[i], baseline_masks[i])
                  for i in range(len(val_qids))}
baseline_bge   = {val_qids[i]: top_k(sc_bge_v[i]) for i in range(len(val_qids))}
baseline_ml    = {val_qids[i]: top_k(sc_ml_v[i])  for i in range(len(val_qids))}
baseline_sp    = {val_qids[i]: top_k(sc_sp_v[i])  for i in range(len(val_qids))}
baseline_tuni  = {val_qids[i]: top_k_filtered(sc_tuni_sp_v[i], val_qids[i], baseline_masks[i])
                  for i in range(len(val_qids))}
baseline_tbi   = {val_qids[i]: top_k_filtered(sc_tbi_sp_v[i], val_qids[i], baseline_masks[i])
                  for i in range(len(val_qids))}

flat_base = {'BM25L': baseline_bm25l, 'BGE': baseline_bge, 'MiniLM': baseline_ml,
             'SPECTER2': baseline_sp, 'TF-IDF-uni': baseline_tuni, 'TF-IDF-bi': baseline_tbi}
flat_sf   = {'BM25L': baseline_bm25l, 'TF-IDF-uni': baseline_tuni, 'TF-IDF-bi': baseline_tbi}

b0_base    = _baseline_rrf(flat_base, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
b0_base_dr = dr(b0_base, q_dom_val)

ml_qv_prf_b  = _bml(ml_qv_v, ml_cv, baseline_bge, val_qids, q_dom_val, k=5)
sc_ml_prf_b  = (ml_qv_prf_b @ ml_cv.T).astype(np.float32)
s_ml_prf_b   = {val_qids[i]: top_k(sc_ml_prf_b[i]) for i in range(len(val_qids))}

bge_qv_prf_b = _bbge(bge_qv_v, bge_c, b0_base_dr, val_qids, k=5, beta=0.06)
sc_bge_prf_b = cosine(bge_qv_prf_b, bge_c)
s_bge_prf_b  = {val_qids[i]: top_k(sc_bge_prf_b[i]) for i in range(len(val_qids))}

flat_prf_b = {'BM25L': baseline_bm25l, 'BGE': s_bge_prf_b, 'MiniLM': s_ml_prf_b,
              'SPECTER2': baseline_sp, 'TF-IDF-uni': baseline_tuni, 'TF-IDF-bi': baseline_tbi}

val_base    = _baseline_rrf(flat_prf_b, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
val_base_dr = dr(val_base, q_dom_val)
base_nd = ndcg10(val_base_dr)
base_rc = recall100(val_base_dr)
print(f'  {"Baseline (NDCG routing, sparse filter)":<45} NDCG={base_nd:.4f}  R@100={base_rc:.4f}')
print()

best_nd = base_nd
best_cfg = None
best_sub_dr = val_base_dr

for name, cfg in configs:
    sub_rr    = rrf_recall_routed(subs_prf, val_qids, q_dom_val, DOM_MODEL_RECALL, **cfg)
    sub_rr_dr = dr(sub_rr, q_dom_val)
    nd = ndcg10(sub_rr_dr)
    rc = recall100(sub_rr_dr)
    delta = nd - base_nd
    marker = ' ★' if nd > base_nd else ''
    print(f'  Recall-routed {name:<28} NDCG={nd:.4f}  R@100={rc:.4f}  Δ={delta:+.4f}{marker}')
    if nd > best_nd:
        best_nd = nd
        best_cfg = name
        best_sub_dr = sub_rr_dr

print()
if best_cfg:
    print(f'Best: {best_cfg}  NDCG={best_nd:.4f}  (Δ={best_nd-base_nd:+.4f})')
    # Per-domain breakdown for best
    print(f'\n{"Domain":<25} {"Baseline":>8}  {"Best":>8}  {"Δ":>7}')
    print('-'*55)
    for dom in all_domains:
        qids_dom = [q for q, d in q_dom_val.items() if d == dom and q in qrels]
        if not qids_dom: continue
        def dom_ndcg_fn(sub_dr):
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
        b = dom_ndcg_fn(val_base_dr)
        be = dom_ndcg_fn(best_sub_dr)
        print(f'  {dom:<23} {b:8.4f}  {be:8.4f}  {be-b:+7.4f}')
else:
    print('No configuration beat the baseline.')
    print()
    print('Per-domain recall@100 table (for reference):')
    print(f'{"Domain":<25} {"BM25L":>6} {"BGE":>6} {"MiniLM":>6} {"SPEC2":>6} {"TFuni":>6} {"TFbi":>6}  {"Best model"}')
    print('-'*80)
    for dom in all_domains:
        r = DOM_MODEL_RECALL.get(dom, {})
        best_m = max(r, key=r.get) if r else '—'
        print(f'  {dom:<23} {r.get("BM25L",0):6.4f} {r.get("BGE",0):6.4f} '
              f'{r.get("MiniLM",0):6.4f} {r.get("SPECTER2",0):6.4f} '
              f'{r.get("TF-IDF-uni",0):6.4f} {r.get("TF-IDF-bi",0):6.4f}  {best_m}')
