"""
Model exclusion experiment (follow-up to experiment_domain_model_weights.py).

Best lead from previous experiment:
  B. Exclude models <70% of best per domain: NDCG=0.7512 (baseline=0.7472)

This script digs deeper:
1. Re-check which models get excluded at 70% per domain
2. Try exclusion with nested RRF structure (k=1 inner, k=2 outer) instead of flat RRF
3. Try exclusion combined with sparse domain filter (best from earlier experiments)
4. Sweep threshold more finely around 70%
5. Try asymmetric: only exclude models that are CATASTROPHICALLY bad (< 10% of best)

Baseline: 0.7472/0.7500
"""
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# ── IDs & qrels ────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
qrels      = json.load(open('data/qrels_1.json'))

# ── Domain maps ────────────────────────────────────────────────────────────
dw_df     = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw        = dw_df.to_dict(orient='index')
c_df      = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df     = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_dom_map = c_df['domain'].to_dict()
q_dom_val = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids if qid in q1_df.index}
corpus_arr = np.array(corpus_ids)
c_domains  = np.array([c_dom_map.get(cid, '') for cid in corpus_ids])

dom_to_qids = defaultdict(list)
for qid in val_qids:
    dom_to_qids[q_dom_val.get(qid, 'Unknown')].append(qid)

# Per-domain model NDCG from previous experiment
# (hand-transferred from experiment_domain_model_weights output)
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

# ── Helpers ────────────────────────────────────────────────────────────────
def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q / (nq + 1e-10)) @ (c / (nc + 1e-10)).T).astype(np.float32)

def top100(scores, qids):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-scores[i])[:100]
        sub[qid] = corpus_arr[idx].tolist()
    return sub

def top100_filtered(score_matrix, qids, masks):
    sub = {}
    for i, qid in enumerate(qids):
        sc = score_matrix[i].copy()
        sc[~masks[i]] = -1e9
        idx = np.argsort(-sc)[:100]
        sub[qid] = corpus_arr[idx].tolist()
    return sub

def rrf_fuse(sa, sb, k=5):
    out = np.zeros_like(sa)
    for i in range(len(sa)):
        ra = np.argsort(np.argsort(-sa[i])) + 1
        rb = np.argsort(np.argsort(-sb[i])) + 1
        out[i] = 1.0 / (k + ra) + 1.0 / (k + rb)
    return out

def rrf(lists, qids, k=3, n=100):
    sub = {}
    for qid in qids:
        sc = {}
        for lst in lists:
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (k + rank)
        sub[qid] = sorted(sc, key=sc.get, reverse=True)[:n]
    return sub

def ndcg(sub, k=10):
    sc = []
    for qid, rels in qrels.items():
        if qid not in sub: continue
        ranked  = sub[qid][:k]
        rel_set = set(rels) if isinstance(rels, list) else set()
        dcg  = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
        sc.append(dcg / idcg if idcg > 0 else 0)
    return float(np.mean(sc))

def ndcg_subset(sub, qid_list, k=10):
    sc = []
    for qid in qid_list:
        rels = qrels.get(qid)
        if not rels or qid not in sub: continue
        ranked  = sub[qid][:k]
        rel_set = set(rels)
        dcg  = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
        sc.append(dcg / idcg if idcg > 0 else 0)
    return float(np.mean(sc)) if sc else 0.0

def recall100(sub):
    vals = []
    for qid, rels in qrels.items():
        rel_set = set(rels)
        if not rel_set: continue
        vals.append(sum(1 for d in sub.get(qid, [])[:100] if d in rel_set) / len(rel_set))
    return float(np.mean(vals))

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

# Domain mask for sparse filter
from functools import lru_cache
def domain_mask(qdom, min_pool=300):
    row = dw.get(qdom, {})
    mask = np.array([row.get(d, 0.0) > 0.0 for d in c_domains])
    if mask.sum() < min_pool:
        return np.ones(len(corpus_ids), dtype=bool)
    return mask

# ── Load scores ────────────────────────────────────────────────────────────
print('Loading scores...')
sc_bm25l = np.load('submissions/scores_bm25l_ft.npy')
sc_tuni  = np.load('submissions/scores_tfidf_uni_ft.npy')
sc_tbi   = np.load('submissions/scores_tfidf_bi_ft.npy')
bge_qv   = np.load('submissions/bge_large_query_emb.npy')
bge_c    = np.load('submissions/bge_large_corpus_emb.npy')
sc_bge   = cosine(bge_qv, bge_c)
ml_qv    = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy')
ml_cv    = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy')
sc_ml    = (ml_qv @ ml_cv.T).astype(np.float32)
sp_q     = np.load('specter_prox_embed/queries_1_embeddings.npy')
sp_ft_c  = np.load('submissions/specter2_corpus_emb_ft.npy')
sc_sp    = cosine(sp_q, sp_ft_c)

# ── Base retrievals ────────────────────────────────────────────────────────
s_bm25l = top100(sc_bm25l, val_qids)
s_bge   = top100(sc_bge,   val_qids)
s_ml    = top100(sc_ml,    val_qids)
s_sp    = top100(sc_sp,    val_qids)
ht_tuni = top100(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids)
ht_tbi  = top100(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids)

flat_models = {
    'BM25L': s_bm25l, 'BGE': s_bge, 'MiniLM': s_ml,
    'SPECTER2': s_sp, 'TF-IDF-uni': ht_tuni, 'TF-IDF-bi': ht_tbi,
}

# ── Baseline ────────────────────────────────────────────────────────────────
inner_base = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)
outer_base = rrf([inner_base, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp], val_qids, k=2)
base_dr    = dr(outer_base)
print(f'Baseline (hard-sort): NDCG={ndcg(base_dr):.4f}  R@100={recall100(outer_base):.4f}')

# ── Show which models are excluded at various thresholds ──────────────────
print('\n── Model exclusions at threshold=0.70 ───────────────────────────────────')
for domain, scores in sorted(DOM_MODEL_NDCG.items()):
    best = max(scores.values())
    excluded = [m for m, s in scores.items() if s < 0.70 * best]
    kept = [m for m, s in scores.items() if s >= 0.70 * best]
    if excluded:
        print(f'  {domain:<22} exclude={excluded}')
        print(f'    {"":22} keep   ={kept}')

# ── Core exclusion function with NESTED RRF ──────────────────────────────
def rrf_exclude_nested(lists_dict, qids, dom_model_ndcg, threshold, n=100):
    """
    Per-domain exclusion + nested RRF structure.
    Inner (k=1): BM25L, BGE, MiniLM (or filtered subset)
    Outer (k=2): inner + remaining models
    """
    sub = {}
    for qid in qids:
        qdom = q_dom_val.get(qid, '')
        domain_scores = dom_model_ndcg.get(qdom, {})
        if domain_scores:
            best_s = max(domain_scores.values())
            min_s  = best_s * threshold
            active = {m for m, s in domain_scores.items() if s >= min_s}
        else:
            active = set(lists_dict.keys())

        # Nested structure
        inner_models = ['BM25L', 'BGE', 'MiniLM']
        inner_active = [m for m in inner_models if m in active]
        outer_extra  = [m for m in ['TF-IDF-uni', 'TF-IDF-bi', 'SPECTER2'] if m in active]

        if not inner_active:
            inner_active = inner_models  # fallback

        # Inner RRF (k=1)
        inner_sc = {}
        for mname in inner_active:
            lst = lists_dict.get(mname, {})
            for rank, doc in enumerate(lst.get(qid, []), 1):
                inner_sc[doc] = inner_sc.get(doc, 0.0) + 1.0 / (1 + rank)
        inner_sorted = sorted(inner_sc, key=inner_sc.get, reverse=True)[:100]

        # Outer RRF (k=2): inner result + extra models
        outer_sc = {}
        for rank, doc in enumerate(inner_sorted, 1):
            outer_sc[doc] = outer_sc.get(doc, 0.0) + 1.0 / (2 + rank)

        # Also add non-inner models separately (same as original structure)
        outer_extra2 = [m for m in ['BGE', 'BM25L'] if m in active]
        for mname in outer_extra + outer_extra2:
            lst = lists_dict.get(mname, {})
            for rank, doc in enumerate(lst.get(qid, []), 1):
                outer_sc[doc] = outer_sc.get(doc, 0.0) + 1.0 / (2 + rank)

        sub[qid] = sorted(outer_sc, key=outer_sc.get, reverse=True)[:n]
    return sub

# ── Fine sweep of exclusion threshold ────────────────────────────────────
print('\n── Fine threshold sweep (flat RRF + hard-sort) ──────────────────────────')
print(f'  {"threshold":>12}  {"NDCG@10":>8}  {"R@100":>8}  {"vs base":>8}')
print('  ' + '-' * 44)
best_flat_ndcg = ndcg(base_dr)
best_flat_thresh = None
best_flat_sub = None

for thresh in [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.90]:
    # Flat exclusion (same as experiment B in previous script)
    sub_q = {}
    for qid in val_qids:
        qdom  = q_dom_val.get(qid, '')
        dscores = DOM_MODEL_NDCG.get(qdom, {})
        if dscores:
            best_s = max(dscores.values())
            active = {m: lst for m, lst in flat_models.items()
                      if dscores.get(m, 0) >= thresh * best_s}
        else:
            active = flat_models
        if not active:
            active = flat_models
        sc = {}
        for mname, lst in active.items():
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (2 + rank)
        sub_q[qid] = sorted(sc, key=sc.get, reverse=True)[:100]

    sub_dr = dr(sub_q)
    n_val = ndcg(sub_dr)
    r_val = recall100(sub_q)
    delta = n_val - ndcg(base_dr)
    marker = ' <--' if n_val > best_flat_ndcg else ''
    print(f'  {thresh:>12.2f}  {n_val:>8.4f}  {r_val:>8.4f}  {delta:>+8.4f}{marker}')
    if n_val > best_flat_ndcg:
        best_flat_ndcg  = n_val
        best_flat_thresh = thresh
        best_flat_sub   = sub_dr

# ── Nested RRF + exclusion sweep ─────────────────────────────────────────
print('\n── Nested RRF + exclusion threshold sweep ───────────────────────────────')
print(f'  {"threshold":>12}  {"NDCG@10":>8}  {"R@100":>8}  {"vs base":>8}')
print('  ' + '-' * 44)
best_nest_ndcg = ndcg(base_dr)
best_nest_thresh = None
best_nest_sub = None

for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
    outer_ne = rrf_exclude_nested(flat_models, val_qids, DOM_MODEL_NDCG, threshold=thresh)
    outer_ne_dr = dr(outer_ne)
    n_val = ndcg(outer_ne_dr)
    r_val = recall100(outer_ne)
    delta = n_val - ndcg(base_dr)
    marker = ' <--' if n_val > best_nest_ndcg else ''
    print(f'  {thresh:>12.2f}  {n_val:>8.4f}  {r_val:>8.4f}  {delta:>+8.4f}{marker}')
    if n_val > best_nest_ndcg:
        best_nest_ndcg  = n_val
        best_nest_thresh = thresh
        best_nest_sub   = outer_ne_dr

# ── Combine exclusion with sparse domain filter ───────────────────────────
print('\n── Flat exclusion + sparse domain filter (min_pool=300) ─────────────────')
masks = [domain_mask(q_dom_val.get(qid, ''), min_pool=300) for qid in val_qids]
mask_list = {qid: masks[i] for i, qid in enumerate(val_qids)}

# Sparse filtered retrievals
s_bm25l_f = top100_filtered(sc_bm25l, val_qids, masks)
ht_tuni_f = top100_filtered(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids, masks)
ht_tbi_f  = top100_filtered(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids, masks)

flat_models_sf = {
    'BM25L': s_bm25l_f, 'BGE': s_bge, 'MiniLM': s_ml,
    'SPECTER2': s_sp, 'TF-IDF-uni': ht_tuni_f, 'TF-IDF-bi': ht_tbi_f,
}

for thresh in [0.65, 0.70, 0.75, 0.80]:
    sub_q = {}
    for qid in val_qids:
        qdom  = q_dom_val.get(qid, '')
        dscores = DOM_MODEL_NDCG.get(qdom, {})
        if dscores:
            best_s = max(dscores.values())
            active = {m: lst for m, lst in flat_models_sf.items()
                      if dscores.get(m, 0) >= thresh * best_s}
        else:
            active = flat_models_sf
        if not active:
            active = flat_models_sf
        sc = {}
        for mname, lst in active.items():
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (2 + rank)
        sub_q[qid] = sorted(sc, key=sc.get, reverse=True)[:100]

    sub_dr = dr(sub_q)
    n_val = ndcg(sub_dr)
    r_val = recall100(sub_q)
    delta = n_val - ndcg(base_dr)
    marker = ' <--' if n_val > ndcg(base_dr) else ''
    print(f'  thresh={thresh:.2f} + sparse filter: NDCG={n_val:.4f}  R@100={r_val:.4f}  Δ={delta:+.4f}{marker}')

# ── Catastrophic exclusion only (< 10% threshold) ────────────────────────
print('\n── Catastrophic exclusion only (exclude if score < 10% of best) ─────────')
for thresh in [0.05, 0.10, 0.15, 0.20, 0.25]:
    sub_q = {}
    for qid in val_qids:
        qdom  = q_dom_val.get(qid, '')
        dscores = DOM_MODEL_NDCG.get(qdom, {})
        if dscores:
            best_s = max(dscores.values())
            active = {m: lst for m, lst in flat_models.items()
                      if dscores.get(m, 0) >= thresh * best_s}
        else:
            active = flat_models
        if not active:
            active = flat_models
        sc = {}
        for mname, lst in active.items():
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (2 + rank)
        sub_q[qid] = sorted(sc, key=sc.get, reverse=True)[:100]

    sub_dr = dr(sub_q)
    n_val = ndcg(sub_dr)
    print(f'  thresh={thresh:.2f}: NDCG={n_val:.4f}  Δ={n_val-ndcg(base_dr):+.4f}')

# ── Per-domain breakdown for best config ─────────────────────────────────
print(f'\n── Per-domain breakdown (best exclusion vs baseline) ────────────────────')
if best_flat_sub is not None:
    print(f'  Best flat exclusion: threshold={best_flat_thresh}  NDCG={best_flat_ndcg:.4f}')
    print(f'\n  {"Domain":<22}  {"n_q":>3}  {"baseline":>10}  {"excluded":>10}  {"delta":>7}')
    print('  ' + '-' * 60)
    for domain in sorted(dom_to_qids.keys()):
        qids_dom = [q for q in dom_to_qids[domain] if q in qrels]
        if not qids_dom: continue
        n_base = ndcg_subset(base_dr,       qids_dom)
        n_best = ndcg_subset(best_flat_sub, qids_dom)
        print(f'  {domain:<22}  {len(qids_dom):>3}  {n_base:>10.4f}  {n_best:>10.4f}  {n_best-n_base:>+7.4f}')

# ── Final summary ──────────────────────────────────────────────────────────
print(f'\n── Final summary ─────────────────────────────────────────────────────────')
print(f'  Baseline (nested k=1/k=2 + hard-sort):  NDCG={ndcg(base_dr):.4f}')
if best_flat_sub is not None:
    print(f'  Best flat exclusion (thresh={best_flat_thresh}):     NDCG={best_flat_ndcg:.4f}  Δ={best_flat_ndcg-ndcg(base_dr):+.4f}')
if best_nest_sub is not None:
    print(f'  Best nested exclusion (thresh={best_nest_thresh}):   NDCG={best_nest_ndcg:.4f}  Δ={best_nest_ndcg-ndcg(base_dr):+.4f}')
