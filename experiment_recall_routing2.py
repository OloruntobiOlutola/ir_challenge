"""
experiment_recall_routing2.py
==============================
Domain-recall routing: for each domain, route to the sparse model
with the highest Recall@100 (from the per-domain recall table in the notebook).

Routing table (best recall model per domain):
  Biology      → TF-IDF bi (FT)       0.8418
  CS           → TF-IDF uni (FT)      0.8187
  Economics    → SPECTER Prox         0.9412
  Engineering  → TF-IDF uni (TA)      0.75
  Physics      → TF-IDF bi (FT)       0.9333
  Sociology    → TF-IDF bi (FT)       0.8197
  Medicine     → TF-IDF uni (FT)      0.7982
  ... (full table below)

The routed sparse model (domain-filtered) is fused via RRF with:
  - BGE-large (PRF-expanded)
  - MiniLM (PRF-expanded)
  - SPECTER Prox (proximity)
  - A secondary sparse model (2nd-best recall, for diversity)
"""

import json, math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

POOL_K = 100
SEED   = 42

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

# ── Recall routing table (from notebook per-domain table) ────────────────────
# Format: domain → [primary_model, secondary_model]
# Keys map to score file keys loaded below
DOMAIN_RECALL_ROUTING = {
    'Art':                  ['tfidf_uni_ta', 'tfidf_bi_ta'],    # all tied 0.5
    'Biology':              ['tfidf_bi_ft',  'tfidf_uni_ft'],   # 0.8418 > 0.8291
    'Business':             ['tfidf_bi_ta',  'tfidf_uni_ft'],   # 0.9667
    'Chemistry':            ['tfidf_uni_ft', 'tfidf_bi_ft'],    # 0.8136 > 0.7833
    'Computer Science':     ['tfidf_uni_ft', 'tfidf_bi_ft'],    # 0.8187 > 0.7854
    'Economics':            ['specter_prox', 'tfidf_bi_ft'],    # 0.9412
    'Engineering':          ['tfidf_uni_ta', 'bm25_ft'],        # 0.75 (TA best)
    'Environmental Science':['tfidf_uni_ta', 'tfidf_uni_ft'],   # tied 0.9762
    'Geography':            ['tfidf_bi_ta',  'tfidf_uni_ft'],   # tied 1.0
    'Geology':              ['tfidf_uni_ft', 'bm25_ft'],        # tied 1.0
    'History':              ['tfidf_uni_ft', 'bm25_ft'],        # tied 1.0
    'Materials Science':    ['tfidf_uni_ft', 'tfidf_bi_ft'],    # tied 0.8855
    'Mathematics':          ['tfidf_uni_ft', 'tfidf_bi_ft'],    # tied 1.0
    'Medicine':             ['tfidf_uni_ft', 'tfidf_bi_ft'],    # 0.7982 > 0.7755
    'Philosophy':           ['tfidf_uni_ft', 'tfidf_bi_ft'],    # tied 0.5
    'Physics':              ['tfidf_bi_ft',  'tfidf_uni_ft'],   # 0.9333 > 0.9258
    'Political Science':    ['tfidf_uni_ft', 'bm25_ft'],        # tied 1.0
    'Psychology':           ['tfidf_uni_ta', 'tfidf_uni_ft'],   # tied 1.0
    'Sociology':            ['tfidf_bi_ft',  'tfidf_uni_ft'],   # 0.8197 > 0.7705
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q/(nq+1e-10)) @ (c/(nc+1e-10)).T).astype(np.float32)

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

def dr(sub, q_dom_map, skip=frozenset({'Business'})):
    out = {}
    for qid, cands in sub.items():
        qd = q_dom_map.get(qid, '')
        if qd in skip:
            out[qid] = cands[:100]; continue
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d,''), 0.0), rank, d) for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _,_,d in scored[:100]]
    return out

def ndcg10(sub):
    sc = []
    for q, rels in qrels.items():
        if q not in sub: continue
        ranked = sub[q][:10]
        rel_set = set(rels)
        dcg  = sum(1.0/math.log2(i+2) for i,d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0/math.log2(i+2) for i in range(min(len(rel_set),10)))
        sc.append(dcg/idcg if idcg else 0)
    return float(np.mean(sc))

def recall100(sub):
    vals = []
    for qid, rels in qrels.items():
        rel_set = set(rels)
        if not rel_set: continue
        vals.append(sum(1 for d in sub.get(qid,[])[:100] if d in rel_set)/len(rel_set))
    return float(np.mean(vals))

def topk_masked(scores_row, mask, k=POOL_K):
    sc = scores_row.copy(); sc[~mask] = -1e9
    return corpus_arr[np.argsort(-sc)[:k]].tolist()

def topk(scores_row, k=POOL_K):
    return corpus_arr[np.argsort(-scores_row)[:k]].tolist()


# ── Load ALL sparse score matrices ────────────────────────────────────────────
print('Loading scores...')
SUB_V = 'submissions'
SUB_H = 'submissions_heldout'

sparse_val = {
    'tfidf_uni_ft':  np.load(f'{SUB_V}/scores_tfidf_uni_ft.npy'),
    'tfidf_bi_ft':   np.load(f'{SUB_V}/scores_tfidf_bi_ft.npy'),
    'tfidf_uni_ta':  np.load(f'{SUB_V}/scores_tfidf_uni_ta.npy'),
    'tfidf_bi_ta':   np.load(f'{SUB_V}/scores_tfidf_bi_ta.npy'),
    'bm25_ft':       np.load(f'{SUB_V}/scores_bm25_ft.npy'),
    'bm25l_ft':      np.load(f'{SUB_V}/scores_bm25l_ft.npy'),
    'bm25plus_ft':   np.load(f'{SUB_V}/scores_bm25plus_ft.npy'),
    'specter_prox':  np.load(f'{SUB_V}/scores_specter_prox.npy'),
}
sparse_hld = {
    'tfidf_uni_ft':  np.load(f'{SUB_H}/scores_tfidf_uni_ft.npy'),
    'tfidf_bi_ft':   np.load(f'{SUB_H}/scores_tfidf_bi_ft.npy'),
    'tfidf_uni_ta':  np.load(f'{SUB_H}/scores_tfidf_uni_ta.npy'),
    'tfidf_bi_ta':   np.load(f'{SUB_H}/scores_tfidf_bi_ta.npy'),
    'bm25_ft':       np.load(f'{SUB_H}/scores_bm25_ft.npy'),
    'bm25l_ft':      np.load(f'{SUB_H}/scores_bm25l_ft.npy'),
    'bm25plus_ft':   np.load(f'{SUB_H}/scores_bm25plus_ft.npy'),
    'specter_prox':  np.load(f'{SUB_H}/scores_specter_prox.npy'),
}
# Dense models
bge_qv_v = np.load(f'{SUB_V}/bge_large_query_emb.npy')
bge_c    = np.load(f'{SUB_V}/bge_large_corpus_emb.npy')
sc_bge_v = cosine(bge_qv_v, bge_c)
ml_qv_v  = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy')
ml_cv    = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy')
sc_ml_v  = (ml_qv_v @ ml_cv.T).astype(np.float32)

bge_qv_h = np.load(f'{SUB_V}/bge_large_heldout_query_emb.npy')
sc_bge_h = cosine(bge_qv_h, bge_c)
ml_qv_h  = np.load('data/embeddings/new_queries_minilm/query_embeddings.npy')
sc_ml_h  = (ml_qv_h @ ml_cv.T).astype(np.float32)

print(f'  Loaded {len(sparse_val)} sparse models  +  BGE + MiniLM')
shapes = {k: v.shape for k,v in sparse_val.items()}
print(f'  Val score shapes: { {k: v for k,v in list(shapes.items())[:2]} } ...')


# ── Build domain-routed pipeline ──────────────────────────────────────────────
def build_routed_pipeline(qids, q_dom_map, sparse_scores, sc_bge, sc_ml,
                           bge_qv, ml_qv, label=''):
    """
    1. Per query: select primary + secondary sparse model by domain recall routing
    2. Apply domain filter to sparse models
    3. RRF(sparse_primary, sparse_secondary, BGE-PRF, MiniLM-PRF, SPECTER_prox)
    4. Domain rerank
    """
    masks = [domain_mask(q_dom_map.get(qid, ''), 300) for qid in
             tqdm(qids, desc=f'  {label} masks', ncols=80)]

    # ── Stage 1: build b0 seed from best sparse + dense (for PRF) ─────────────
    print(f'  [{label}] Building b0 seed...')
    b0 = {}
    for i, qid in enumerate(qids):
        dom  = q_dom_map.get(qid, '')
        mask = masks[i]
        mods = DOMAIN_RECALL_ROUTING.get(dom, ['tfidf_uni_ft', 'tfidf_bi_ft'])
        osc  = {}
        # primary + secondary sparse (domain-filtered)
        for mname in mods[:2]:
            sc_row = sparse_scores[mname][i]
            docs   = topk_masked(sc_row, mask)
            for rank, doc in enumerate(docs, 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (1 + rank)
        # BGE dense (no filter — dense models are domain-aware already)
        for rank, doc in enumerate(topk(sc_bge[i]), 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (1 + rank)
        b0[qid] = sorted(osc, key=osc.get, reverse=True)[:POOL_K]

    b0_dr = dr(b0, q_dom_map)

    # ── Stage 2: PRF-expand BGE and MiniLM ────────────────────────────────────
    print(f'  [{label}] PRF expansion...')
    ml_prf_v  = apply_ml_prf(ml_qv, ml_cv, {val_qids[i] if qids is val_qids else qids[i]: topk(sc_bge[i])
                              for i in range(len(qids))}, qids, q_dom_map)
    # remap for apply_ml_prf — it expects a dict indexed by qid
    bge_seed_sub = {qids[i]: topk(sc_bge[i]) for i in range(len(qids))}
    ml_prf   = apply_ml_prf(ml_qv, ml_cv, bge_seed_sub, qids, q_dom_map)
    bge_prf  = apply_bge_prf(bge_qv, bge_c, b0_dr, qids)
    sc_ml_prf  = (ml_prf  @ ml_cv.T).astype(np.float32)
    sc_bge_prf = cosine(bge_prf, bge_c)

    # ── Stage 3: Final RRF with routed sparse + PRF-dense ─────────────────────
    print(f'  [{label}] Final RRF...')
    sub = {}
    for i, qid in enumerate(qids):
        dom  = q_dom_map.get(qid, '')
        mask = masks[i]
        mods = DOMAIN_RECALL_ROUTING.get(dom, ['tfidf_uni_ft', 'tfidf_bi_ft'])
        osc  = {}
        k_rrf = 1  # sharp RRF

        # Routed sparse models (domain-filtered, primary gets double weight via 2 insertions)
        for j, mname in enumerate(mods[:2]):
            sc_row = sparse_scores[mname][i]
            docs   = topk_masked(sc_row, mask)
            for rank, doc in enumerate(docs, 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (k_rrf + rank)

        # PRF-expanded BGE
        for rank, doc in enumerate(topk(sc_bge_prf[i]), 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (k_rrf + rank)

        # PRF-expanded MiniLM
        for rank, doc in enumerate(topk(sc_ml_prf[i]), 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (k_rrf + rank)

        # SPECTER Prox (as a 5th signal — dense semantic proximity)
        for rank, doc in enumerate(topk(sparse_scores['specter_prox'][i]), 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (k_rrf + rank)

        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:POOL_K]

    return dr(sub, q_dom_map), b0_dr, sc_bge_prf, sc_ml_prf


# ── Run val pipeline ──────────────────────────────────────────────────────────
print('\n' + '='*65)
print('Building VAL pipeline (recall-routed)')
print('='*65)
val_sub_dr, _, _, _ = build_routed_pipeline(
    val_qids, q_dom_val, sparse_val, sc_bge_v, sc_ml_v,
    bge_qv_v, ml_qv_v, label='Val'
)
nd_new  = ndcg10(val_sub_dr)
rc_new  = recall100(val_sub_dr)
print(f'\n  Recall-routed   NDCG@10={nd_new:.4f}  R@100={rc_new:.4f}')
print(f'  Baseline (PRF)  NDCG@10=0.7597  R@100=0.8872')
print(f'  Delta           NDCG@10={nd_new-0.7597:+.4f}  R@100={rc_new-0.8872:+.4f}')

# Per-domain breakdown
print(f'\n  {"Domain":<25} {"Baseline":>8}  {"Routed":>8}  {"Δ NDCG":>8}  {"Δ R@100":>8}')
print('  ' + '-'*68)
BASE_ND = {
    'Art':0.5294,'Biology':0.7133,'Business':0.6889,'Chemistry':0.8121,
    'Computer Science':0.5864,'Economics':1.0000,'Engineering':0.8347,
    'Environmental Science':0.9069,'Geography':0.9107,'Geology':0.8932,
    'History':1.0000,'Materials Science':0.8289,'Mathematics':0.9299,
    'Medicine':0.7152,'Philosophy':0.3869,'Physics':0.8689,
    'Political Science':0.9197,'Psychology':0.8969,'Sociology':1.0000,
}
BASE_RC = {
    'Art':0.5000,'Biology':0.8716,'Business':0.9667,'Chemistry':0.7960,
    'Computer Science':0.8499,'Economics':0.9412,'Engineering':0.8750,
    'Environmental Science':1.0000,'Geography':1.0000,'Geology':1.0000,
    'History':1.0000,'Materials Science':0.9275,'Mathematics':1.0000,
    'Medicine':0.8673,'Philosophy':0.5000,'Physics':0.9591,
    'Political Science':1.0000,'Psychology':1.0000,'Sociology':0.8689,
}
for dom in sorted(set(q_dom_val.values())):
    qids_dom = [q for q,d in q_dom_val.items() if d==dom and q in qrels]
    if not qids_dom: continue
    sub_d = {q: val_sub_dr[q] for q in qids_dom if q in val_sub_dr}
    nd = ndcg10(sub_d)
    rc = sum(
        sum(1 for d in val_sub_dr.get(q,[])[:100] if d in set(qrels[q])) / len(qrels[q])
        for q in qids_dom if qrels.get(q)
    ) / sum(1 for q in qids_dom if qrels.get(q))
    b_nd = BASE_ND.get(dom, 0)
    b_rc = BASE_RC.get(dom, 0)
    flag = ' ★' if nd > b_nd else ''
    print(f'  {dom:<25} {b_nd:8.4f}  {nd:8.4f}  {nd-b_nd:+8.4f}  {rc-b_rc:+8.4f}{flag}')


# ── Run held-out pipeline and save ───────────────────────────────────────────
print('\n' + '='*65)
print('Building HELD-OUT pipeline (recall-routed)')
print('='*65)
held_sub_dr, _, _, _ = build_routed_pipeline(
    held_qids, q_dom_held, sparse_hld, sc_bge_h, sc_ml_h,
    bge_qv_h, ml_qv_h, label='Held'
)
out = Path('submissions_heldout/submission_recall_routed.json')
with open(out, 'w') as f:
    json.dump(held_sub_dr, f)
print(f'\n  Saved: {out}')
print(f'  Queries: {len(held_sub_dr)}  docs/query: {set(len(v) for v in held_sub_dr.values())}')
print(f'  Val NDCG@10 = {nd_new:.4f}  (baseline = 0.7597)')
print('='*65)
