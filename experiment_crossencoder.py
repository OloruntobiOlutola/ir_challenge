"""
experiment_crossencoder.py
==========================
Cross-encoder reranking on top of the best PRF pipeline (val 0.7597).

  Reranker: BAAI/bge-reranker-v2-m3
    - Scores each (query, doc) pair directly
    - Reranks top-30 candidates from the PRF pipeline
    - Positions 31-100 kept as-is

Usage:
    /opt/anaconda3/envs/py_env/bin/python3 experiment_crossencoder.py 2>&1 | tee experiment_crossencoder.log

Output:
    submissions/submission_crossencoder.json   (val submission for inspection)
"""

import builtins
import json
import math
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Force every print() to flush immediately so output appears in pipes/tee
_real_print = builtins.print


def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _real_print(*args, **kwargs)


# ── Config ─────────────────────────────────────────────────────────────────────
# bge-reranker-large   — already cached locally (~2.1 GB), English-focused
# bge-reranker-v2-m3   — needs download (~570 MB), multilingual, newer
CE_MODEL        = 'BAAI/bge-reranker-large'   # swap to bge-reranker-v2-m3 if preferred
CE_RERANK_TOP_N = 30        # rerank top-N from PRF pipeline
CE_BATCH_SIZE   = 32        # query-doc pairs per forward pass
CE_MAX_LENGTH   = 512       # token limit for cross-encoder input
THRESHOLD       = 0.70      # model exclusion threshold (same as baseline)

# ── Pool expansion (same improvement as build_prf_submission.py) ───────────
# Each retriever returns POOL_K candidates before RRF fusion.
# k=200 gives R@100: 0.8872 → 0.8984 with no NDCG cost.
POOL_K = 200

# ── TF-IDF routing (same as build_prf_submission.py) ──────────────────────
# For these domains TF-IDF-uni standalone NDCG@10 > full ensemble.
# Bypass ensemble and feed TF-IDF-uni candidates directly to cross-encoder.
TFIDF_ROUTE_DOMS = {
    'Environmental Science', 'Geography', 'Geology',
    'Mathematics', 'Political Science', 'Psychology',
}

# ── Paths ──────────────────────────────────────────────────────────────────────
SUB_DIR  = Path('submissions')
DATA_DIR = Path('data')

# ── Load IDs and metadata ──────────────────────────────────────────────────────
print('Loading metadata...')
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
qrels      = json.load(open('data/qrels_1.json'))
q_df       = pd.read_parquet('data/queries.parquet').set_index('doc_id')
corpus_arr = np.array(corpus_ids)
ci_to_idx  = {cid: i for i, cid in enumerate(corpus_ids)}

c_df       = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df      = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_title    = c_df['title'].to_dict()
c_abstract = c_df['abstract'].fillna('').to_dict()
q_title    = {**q1_df['title'].to_dict(), **q_df['title'].to_dict()}
q_abstract = {**q1_df['abstract'].fillna('').to_dict(),
              **q_df['abstract'].fillna('').to_dict()}

dw_df      = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw         = dw_df.to_dict(orient='index')
c_dom_map  = c_df['domain'].to_dict()
q_dom_val  = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids if qid in q1_df.index}
c_domains  = np.array([c_dom_map.get(cid, '') for cid in corpus_ids])

# ── Per-domain model NDCG (from best PRF pipeline) ────────────────────────────
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
STRONG_DOMS = {'Geology', 'Mathematics', 'Political Science', 'Economics', 'History',
               'Sociology', 'Chemistry', 'Physics', 'Materials Science'}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS  (identical to experiment_medcpt_llm.py)
# ═══════════════════════════════════════════════════════════════════════════════

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
    return mask if mask.sum() >= min_pool else np.ones(len(corpus_ids), dtype=bool)

def dr(sub, q_dom_map, skip=frozenset({'Business'})):
    out = {}
    for qid, cands in sub.items():
        qd = q_dom_map.get(qid, '')
        if qd in skip:
            out[qid] = cands[:100]
            continue
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d, ''), 0.0), rank, d)
                  for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _, _, d in scored[:100]]
    return out

def rrf_nested_exclude_sf(flat_unfiltered, flat_sparse_f, qids, q_dom_map,
                          dom_ndcg, threshold=0.70):
    sub = {}
    for qid in qids:
        qdom = q_dom_map.get(qid, '')
        dscores = dom_ndcg.get(qdom, {})
        if dscores:
            best_s = max(dscores.values())
            active = {m for m, s in dscores.items() if s >= best_s * threshold}
        else:
            active = set(flat_unfiltered.keys())
        if not active:
            active = set(flat_unfiltered.keys())

        def get_lst(mname):
            return flat_sparse_f.get(mname, flat_unfiltered[mname])

        inner_models = [m for m in ['BM25L', 'BGE', 'MiniLM']
                        if m in active]
        outer_extra  = [m for m in ['TF-IDF-uni', 'TF-IDF-bi', 'SPECTER2']
                        if m in active]
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
    qe = ml_qv.copy()
    for i, qid in enumerate(qids):
        d = q_dom_map.get(qid, '')
        b = 0.8 if d in WEAK_DOMS else (0.0 if d in STRONG_DOMS else 0.60)
        if b == 0:
            continue
        docs = seed_sub.get(qid, [])[:k]
        idxs = [ci_to_idx[doc] for doc in docs if doc in ci_to_idx]
        if not idxs:
            continue
        fb = ml_cv[idxs].mean(axis=0)
        nq = qe[i] + b * fb
        qe[i] = nq / (np.linalg.norm(nq) + 1e-10)
    return qe

def apply_bge_prf(bge_qv, bge_c, seed_sub, qids, k=5, beta=0.06):
    qe = bge_qv.copy()
    for i, qid in enumerate(qids):
        docs = seed_sub.get(qid, [])[:k]
        idxs = [ci_to_idx[doc] for doc in docs if doc in ci_to_idx]
        if not idxs:
            continue
        fb = bge_c[idxs].mean(axis=0)
        nq = qe[i] + beta * fb
        qe[i] = nq / (np.linalg.norm(nq) + 1e-10)
    return qe

def ndcg10(sub):
    sc = []
    for q, rels in qrels.items():
        if q not in sub:
            continue
        ranked = sub[q][:10]
        rel_set = set(rels)
        dcg  = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel_set), 10)))
        sc.append(dcg / idcg if idcg else 0)
    return float(np.mean(sc))

def recall100(sub):
    vals = []
    for qid, rels in qrels.items():
        rel_set = set(rels)
        if not rel_set:
            continue
        vals.append(sum(1 for d in sub.get(qid, [])[:100] if d in rel_set) / len(rel_set))
    return float(np.mean(vals))

def ndcg10_per_domain(sub, q_dom_map):
    """Return dict[domain -> ndcg@10]."""
    domain_scores = {}
    for qid, rels in qrels.items():
        if qid not in sub:
            continue
        dom = q_dom_map.get(qid, 'Unknown')
        ranked = sub[qid][:10]
        rel_set = set(rels)
        dcg  = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel_set), 10)))
        score = dcg / idcg if idcg else 0.0
        domain_scores.setdefault(dom, []).append(score)
    return {d: float(np.mean(v)) for d, v in domain_scores.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-ENCODER RERANKING
# ═══════════════════════════════════════════════════════════════════════════════

def load_crossencoder(model_name, device):
    """Load BAAI/bge-reranker-v2-m3 (or any cross-encoder) from HuggingFace."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f'  Loading cross-encoder: {model_name}')
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl = mdl.to(device).eval()
    print(f'  ✓ Model loaded on {device}  '
          f'(params: {sum(p.numel() for p in mdl.parameters()) / 1e6:.0f}M)')
    return tok, mdl


def ce_rerank(pipeline_sub, q_dom_map, tok, mdl, device,
              top_n=CE_RERANK_TOP_N, batch_size=CE_BATCH_SIZE,
              max_length=CE_MAX_LENGTH, q_max_words=200, d_max_words=150):
    """
    Rerank top_n candidates for every query using the cross-encoder.
    Positions (top_n+1)..100 are kept from the original pipeline order.

    Returns a new submission dict.
    """
    import torch

    out   = {}
    t0    = time.time()
    bar   = tqdm(list(pipeline_sub.items()),
                 desc='  CE reranking', unit='query', ncols=80)

    for idx, (qid, cands) in enumerate(bar):
        top_cands = cands[:top_n]
        rest      = cands[top_n:]

        # Build query text (truncated)
        qt = (q_title.get(qid, '') + ' ' + q_abstract.get(qid, '')).strip()
        qt = ' '.join(qt.split()[:q_max_words])

        # Build (query, doc) pairs
        pairs = []
        for cid in top_cands:
            dt = (c_title.get(cid, '') + ' ' + c_abstract.get(cid, '')).strip()
            dt = ' '.join(dt.split()[:d_max_words])
            pairs.append([qt, dt])

        # Score pairs in batches
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            bp = pairs[i:i + batch_size]
            with torch.no_grad():
                enc = tok(bp, padding=True, truncation=True,
                          max_length=max_length, return_tensors='pt').to(device)
                logits = mdl(**enc).logits.squeeze(-1)
                scores = logits.cpu().float()
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)
                all_scores.extend(scores.tolist())

        # Sort by descending score
        ranked = sorted(zip(all_scores, top_cands), reverse=True)
        out[qid] = [cid for _, cid in ranked] + rest

        # Progress display
        elapsed  = time.time() - t0
        done     = idx + 1
        remaining = len(pipeline_sub) - done
        eta_s    = (elapsed / done) * remaining if done > 0 else 0
        bar.set_postfix({
            'domain': q_dom_map.get(qid, '')[:8],
            'ETA':    f'{eta_s/60:.1f}m',
        })

    total_min = (time.time() - t0) / 60
    print(f'  CE reranked {len(pipeline_sub)} queries in {total_min:.1f} min.')
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('STEP 1: Load all retrieval scores')
print('='*70)
_t = time.time()

files_to_load = [
    'submissions/scores_bm25l_ft.npy',
    'submissions/scores_tfidf_uni_ft.npy',
    'submissions/scores_tfidf_bi_ft.npy',
    'submissions/bge_large_query_emb.npy',
    'submissions/bge_large_corpus_emb.npy',
    'data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy',
    'data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy',
    'specter_prox_embed/queries_1_embeddings.npy',
    'submissions/specter2_corpus_emb_ft.npy',
]
for p in tqdm(files_to_load, desc='  Checking files', ncols=80):
    assert Path(p).exists(), f'Missing: {p}'

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
print(f'  ✓ All scores loaded  [{time.time()-_t:.1f}s]')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('STEP 2: Reproduce PRF baseline (expected 0.7597)')
print('='*70)
_t = time.time()

masks_v = [domain_mask(q_dom_val.get(qid, ''), 300)
           for qid in tqdm(val_qids, desc='  Domain masks', ncols=80)]

s_bm25l_vf = top100_filtered(sc_bm25l_v, val_qids, masks_v)
s_bge_v    = top100(sc_bge_v, val_qids)
s_ml_v     = top100(sc_ml_v,  val_qids)
s_sp_v     = top100(sc_sp_v,  val_qids)
ht_tuni_vf = top100_filtered(rrf_fuse(sc_tuni_v, sc_sp_v, k=5), val_qids, masks_v)
ht_tbi_vf  = top100_filtered(rrf_fuse(sc_tbi_v,  sc_sp_v, k=5), val_qids, masks_v)

flat_v_base = {'BM25L': s_bm25l_vf, 'BGE': s_bge_v, 'MiniLM': s_ml_v,
               'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}
flat_v_sf   = {'BM25L': s_bm25l_vf, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

b0_v    = rrf_nested_exclude_sf(flat_v_base, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
b0_v_dr = dr(b0_v, q_dom_val)

ml_qv_prf_v  = apply_ml_prf(ml_qv_v, ml_cv, s_bge_v, val_qids, q_dom_val)
sc_ml_prf_v  = (ml_qv_prf_v @ ml_cv.T).astype(np.float32)
s_ml_prf_v   = top100(sc_ml_prf_v, val_qids)

bge_qv_prf_v = apply_bge_prf(bge_qv_v, bge_c, b0_v_dr, val_qids)
sc_bge_prf_v = cosine(bge_qv_prf_v, bge_c)
s_bge_prf_v  = top100(sc_bge_prf_v, val_qids)

flat_v_prf = {'BM25L': s_bm25l_vf, 'BGE': s_bge_prf_v, 'MiniLM': s_ml_prf_v,
              'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

baseline_sub = rrf_nested_exclude_sf(flat_v_prf, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)

# Route TF-IDF-winning domains: bypass ensemble, feed TF-IDF-uni directly to CE
for qid in baseline_sub:
    if q_dom_val.get(qid, '') in TFIDF_ROUTE_DOMS:
        baseline_sub[qid] = ht_tuni_vf[qid]

baseline_dr  = dr(baseline_sub, q_dom_val)
nd_base = ndcg10(baseline_dr)
rc_base = recall100(baseline_dr)
print(f'  ✓ PRF+routing baseline  NDCG@10 = {nd_base:.4f}  R@100 = {rc_base:.4f}  [{time.time()-_t:.1f}s]')
if abs(nd_base - 0.7601) > 0.003:
    print(f'  ⚠ Unexpected baseline: expected ~0.7601, got {nd_base:.4f}')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print(f'STEP 3: Load cross-encoder  ({CE_MODEL})')
print('='*70)

import torch
device = ('cuda'  if torch.cuda.is_available() else
          'mps'   if torch.backends.mps.is_available() else
          'cpu')
print(f'  Device: {device}')

tok_ce, mdl_ce = load_crossencoder(CE_MODEL, device)

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print(f'STEP 4: Cross-encoder reranking  (top-{CE_RERANK_TOP_N} per query)')
print('='*70)
print(f'  Queries          : {len(val_qids)}')
print(f'  Candidates/query : {CE_RERANK_TOP_N}')
print(f'  Batch size       : {CE_BATCH_SIZE}')
print(f'  Device           : {device}')
print()

_t = time.time()
ce_sub = ce_rerank(
    baseline_dr,
    q_dom_val,
    tok_ce, mdl_ce, device,
    top_n=CE_RERANK_TOP_N,
    batch_size=CE_BATCH_SIZE,
)
nd_ce = ndcg10(ce_sub)
rc_ce = recall100(ce_sub)
delta = nd_ce - nd_base
arrow = '▲ improvement' if delta > 0.001 else ('▼ regression' if delta < -0.001 else '─ no change')
print(f'  ✓ + CE rerank  NDCG@10 = {nd_ce:.4f}  R@100 = {rc_ce:.4f}  '
      f'(delta = {delta:+.4f}  {arrow})  [{time.time()-_t:.1f}s]')

# ─────────────────────────────────────────────────────────────────────────────
print('\n  Per-domain breakdown:')
bd_base = ndcg10_per_domain(baseline_dr, q_dom_val)
bd_ce   = ndcg10_per_domain(ce_sub,      q_dom_val)
all_doms = sorted(set(bd_base) | set(bd_ce))
print(f'  {"Domain":<22} {"Baseline":>9} {"+ CE":>9} {"Delta":>8}')
print('  ' + '-'*52)
for dom in all_doms:
    b = bd_base.get(dom, 0.0)
    c = bd_ce.get(dom, 0.0)
    d = c - b
    mark = ' ▲' if d > 0.005 else (' ▼' if d < -0.005 else '')
    print(f'  {dom:<22} {b:>9.4f} {c:>9.4f} {d:>+8.4f}{mark}')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('STEP 5: Save val submission')
print('='*70)
out_path = SUB_DIR / 'submission_crossencoder.json'
with open(out_path, 'w') as f:
    json.dump(ce_sub, f)
print(f'  ✓ Saved: {out_path}')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('RESULTS SUMMARY')
print('='*70)
rows = [
    ('PRF baseline (0.73 best)', nd_base, rc_base, 0.0),
    (f'+ CE rerank (top-{CE_RERANK_TOP_N})',  nd_ce,  rc_ce,  delta),
]
print(f'  {"Config":<30}  {"NDCG@10":>8}  {"R@100":>7}  {"Delta":>7}')
print('  ' + '-'*58)
for label, nd, rc, dlt in rows:
    print(f'  {label:<30}  {nd:>8.4f}  {rc:>7.4f}  {dlt:>+7.4f}')
print('='*70)
if nd_ce > nd_base + 0.001:
    print(f'\n  ✓ Cross-encoder improved NDCG@10 by {delta:+.4f}.')
    print('  Next step: run build_heldout_crossencoder.py to build held-out submission.')
else:
    print(f'\n  ✗ Cross-encoder did not improve ({delta:+.4f}).')
    print('  Consider: larger top-N, different batch size, or different reranker.')
print()
