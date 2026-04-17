"""
Combined best experiment: nested RRF + model exclusion + sparse domain filter.

Building on experiment_model_exclusion.py results:
  - Nested RRF (k=1/k=2) + exclusion thresh=0.70: NDCG=0.7518
  - Sparse filter (min_pool=300): improved recall but similar NDCG
  - Key gains: Philosophy (+0.41), Engineering (+0.17), EnvSci (+0.07)
  - Key losses: Biology (-0.03), Physics (-0.03)

This experiment:
1. Nested exclusion + sparse filter for sparse models
2. Domain-specific threshold: relaxed for large domains (Biology, Physics)
3. Try to recover Biology/Physics with adjusted thresholds
4. Combine additive domain boost with exclusion
5. Build and evaluate best combo for potential submission

Baseline: 0.7472 (nested k=1/k=2 + hard-sort)
Best so far: 0.7518 (nested exclusion thresh=0.70)
"""
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# ── IDs & qrels ────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
qrels      = json.load(open('data/qrels_1.json'))

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

# Per-domain model NDCG (from experiment_domain_model_weights.py)
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

# Number of val queries per domain
DOM_N_QUERIES = {d: len(qs) for d, qs in dom_to_qids.items()}

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
print('Building base retrievals...')
s_bm25l = top100(sc_bm25l, val_qids)
s_bge   = top100(sc_bge,   val_qids)
s_ml    = top100(sc_ml,    val_qids)
s_sp    = top100(sc_sp,    val_qids)
ht_tuni = top100(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids)
ht_tbi  = top100(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids)

# Sparse-filtered versions
print('Building sparse-filtered retrievals...')
masks   = [domain_mask(q_dom_val.get(qid, ''), min_pool=300) for qid in val_qids]
s_bm25l_f = top100_filtered(sc_bm25l, val_qids, masks)
ht_tuni_f = top100_filtered(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids, masks)
ht_tbi_f  = top100_filtered(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids, masks)

inner_base = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)
outer_base = rrf([inner_base, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp], val_qids, k=2)
base_dr    = dr(outer_base)
print(f'Baseline: NDCG={ndcg(base_dr):.4f}  R@100={recall100(outer_base):.4f}')

# ── Nested exclusion function ──────────────────────────────────────────────
def rrf_nested_exclude(flat_m, qids, dom_ndcg, threshold, n=100):
    """
    Nested RRF with per-domain model exclusion.
    threshold: exclude model if domain_ndcg[model] < threshold * best_domain_ndcg
    """
    sub = {}
    for qid in qids:
        qdom = q_dom_val.get(qid, '')
        dscores = dom_ndcg.get(qdom, {})
        if dscores:
            best_s = max(dscores.values())
            min_s  = best_s * threshold
            active = {m for m, s in dscores.items() if s >= min_s}
        else:
            active = set(flat_m.keys())
        if not active:
            active = set(flat_m.keys())

        inner_models = [m for m in ['BM25L', 'BGE', 'MiniLM'] if m in active]
        outer_extra  = [m for m in ['TF-IDF-uni', 'TF-IDF-bi', 'SPECTER2'] if m in active]
        if not inner_models:
            inner_models = ['BM25L', 'BGE', 'MiniLM']

        # Inner RRF (k=1)
        isc = {}
        for mname in inner_models:
            for rank, doc in enumerate(flat_m[mname].get(qid, []), 1):
                isc[doc] = isc.get(doc, 0.0) + 1.0 / (1 + rank)
        inner_sorted = sorted(isc, key=isc.get, reverse=True)[:100]

        # Outer RRF (k=2): inner + extra + bge + bm25l again
        osc = {}
        for rank, doc in enumerate(inner_sorted, 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)
        for mname in outer_extra + [m for m in ['BGE', 'BM25L'] if m in active]:
            for rank, doc in enumerate(flat_m[mname].get(qid, []), 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)

        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:n]
    return sub

# ── Config 1: Nested exclusion + original sparse models ────────────────────
print('\n── Config 1: Nested exclusion (thresh=0.70) + unfiltered ────────────────')
flat_orig = {'BM25L': s_bm25l, 'BGE': s_bge, 'MiniLM': s_ml,
             'SPECTER2': s_sp, 'TF-IDF-uni': ht_tuni, 'TF-IDF-bi': ht_tbi}
o_ne70 = rrf_nested_exclude(flat_orig, val_qids, DOM_MODEL_NDCG, threshold=0.70)
o_ne70_dr = dr(o_ne70)
print(f'  NDCG={ndcg(o_ne70_dr):.4f}  R@100={recall100(o_ne70):.4f}')

# ── Config 2: Nested exclusion + sparse filter for sparse models ───────────
print('\n── Config 2: Nested exclusion + sparse domain filter for sparse ──────────')
flat_sparse_f = {'BM25L': s_bm25l_f, 'BGE': s_bge, 'MiniLM': s_ml,
                 'SPECTER2': s_sp, 'TF-IDF-uni': ht_tuni_f, 'TF-IDF-bi': ht_tbi_f}
for thresh in [0.65, 0.70, 0.75]:
    o_ne_sf = rrf_nested_exclude(flat_sparse_f, val_qids, DOM_MODEL_NDCG, threshold=thresh)
    o_ne_sf_dr = dr(o_ne_sf)
    print(f'  thresh={thresh}: NDCG={ndcg(o_ne_sf_dr):.4f}  R@100={recall100(o_ne_sf):.4f}  Δ={ndcg(o_ne_sf_dr)-ndcg(base_dr):+.4f}')

# ── Config 3: Domain-specific thresholds ──────────────────────────────────
print('\n── Config 3: Domain-specific thresholds (relaxed for large domains) ──────')
# Use higher threshold for small domains (more aggressive exclusion),
# lower threshold for large domains (more conservative)
# Small domains: Philosophy(1), Art(1), History(1), Economics(1), Political Science(1), Sociology(1)
# Medium: Engineering(2), Business(2), Geography(2), Geology(2), Mathematics(2)
# Large: Biology(21), Medicine(21), CS(12)
DOMAIN_THRESH = {
    'Philosophy': 0.70,      # aggressive — we know TF-IDF-bi is clearly best
    'Art': 0.60,
    'Engineering': 0.65,     # BGE clearly best
    'Physics': 0.55,         # relaxed — large enough set to be risky
    'Biology': 0.55,         # relaxed — large, don't want to hurt
    'Medicine': 0.55,        # relaxed
    'Computer Science': 0.55,
    # Others: default 0.65
}

def rrf_nested_exclude_custom(flat_m, qids, dom_ndcg, dom_thresh, default_thresh=0.65, n=100):
    sub = {}
    for qid in qids:
        qdom = q_dom_val.get(qid, '')
        dscores = dom_ndcg.get(qdom, {})
        thresh = dom_thresh.get(qdom, default_thresh)
        if dscores:
            best_s = max(dscores.values())
            min_s  = best_s * thresh
            active = {m for m, s in dscores.items() if s >= min_s}
        else:
            active = set(flat_m.keys())
        if not active:
            active = set(flat_m.keys())

        inner_models = [m for m in ['BM25L', 'BGE', 'MiniLM'] if m in active]
        outer_extra  = [m for m in ['TF-IDF-uni', 'TF-IDF-bi', 'SPECTER2'] if m in active]
        if not inner_models:
            inner_models = ['BM25L', 'BGE', 'MiniLM']

        isc = {}
        for mname in inner_models:
            for rank, doc in enumerate(flat_m[mname].get(qid, []), 1):
                isc[doc] = isc.get(doc, 0.0) + 1.0 / (1 + rank)
        inner_sorted = sorted(isc, key=isc.get, reverse=True)[:100]

        osc = {}
        for rank, doc in enumerate(inner_sorted, 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)
        for mname in outer_extra + [m for m in ['BGE', 'BM25L'] if m in active]:
            for rank, doc in enumerate(flat_m[mname].get(qid, []), 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)

        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:n]
    return sub

for def_thresh in [0.60, 0.65, 0.70]:
    o_cust = rrf_nested_exclude_custom(flat_orig, val_qids, DOM_MODEL_NDCG, DOMAIN_THRESH, default_thresh=def_thresh)
    o_cust_dr = dr(o_cust)
    print(f'  default_thresh={def_thresh}: NDCG={ndcg(o_cust_dr):.4f}  R@100={recall100(o_cust):.4f}  Δ={ndcg(o_cust_dr)-ndcg(base_dr):+.4f}')

# ── Config 4: Surgical exclusion — only exclude absolutely clear losers ────
print('\n── Config 4: Surgical exclusion (only exclude score=0 models) ───────────')
# Only exclude models with literally 0 NDCG for that domain
def rrf_surgical(flat_m, qids, dom_ndcg, n=100):
    sub = {}
    for qid in qids:
        qdom = q_dom_val.get(qid, '')
        dscores = dom_ndcg.get(qdom, {})
        if dscores:
            active = {m for m, s in dscores.items() if s > 0.0}
            if not active:
                active = set(flat_m.keys())
        else:
            active = set(flat_m.keys())

        inner_models = [m for m in ['BM25L', 'BGE', 'MiniLM'] if m in active]
        outer_extra  = [m for m in ['TF-IDF-uni', 'TF-IDF-bi', 'SPECTER2'] if m in active]
        if not inner_models:
            inner_models = list(flat_m.keys())[:3]

        isc = {}
        for mname in inner_models:
            for rank, doc in enumerate(flat_m[mname].get(qid, []), 1):
                isc[doc] = isc.get(doc, 0.0) + 1.0 / (1 + rank)
        inner_sorted = sorted(isc, key=isc.get, reverse=True)[:100]

        osc = {}
        for rank, doc in enumerate(inner_sorted, 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)
        for mname in outer_extra + [m for m in ['BGE', 'BM25L'] if m in active]:
            for rank, doc in enumerate(flat_m[mname].get(qid, []), 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)

        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:n]
    return sub

o_surg = rrf_surgical(flat_orig, val_qids, DOM_MODEL_NDCG)
o_surg_dr = dr(o_surg)
print(f'  Surgical (score>0 only): NDCG={ndcg(o_surg_dr):.4f}  R@100={recall100(o_surg):.4f}  Δ={ndcg(o_surg_dr)-ndcg(base_dr):+.4f}')

# ── Per-domain breakdown for all configs ──────────────────────────────────
print('\n── Per-domain breakdown ──────────────────────────────────────────────────')
configs = {
    'Baseline': base_dr,
    'NestExcl70': o_ne70_dr,
    'Surgical': o_surg_dr,
}
# Add best custom thresh
o_best_cust = rrf_nested_exclude_custom(flat_orig, val_qids, DOM_MODEL_NDCG, DOMAIN_THRESH, default_thresh=0.65)
o_best_cust_dr = dr(o_best_cust)
configs['CustomThresh'] = o_best_cust_dr

print(f'  {"Domain":<22}  {"n_q":>3}  {"Base":>8}  {"NE70":>8}  {"Surg":>8}  {"Cust":>8}')
print('  ' + '-' * 66)
for domain in sorted(dom_to_qids.keys()):
    qids_dom = [q for q in dom_to_qids[domain] if q in qrels]
    if not qids_dom: continue
    scores = {cn: ndcg_subset(cs, qids_dom) for cn, cs in configs.items()}
    row = f'  {domain:<22}  {len(qids_dom):>3}  {scores["Baseline"]:>8.4f}  {scores["NestExcl70"]:>8.4f}  {scores["Surgical"]:>8.4f}  {scores["CustomThresh"]:>8.4f}'
    print(row)

print(f'\n  {"Config":<20}  {"NDCG@10":>8}  {"R@100":>8}')
print('  ' + '-' * 42)
for cname, csub in configs.items():
    r = recall100(csub)
    print(f'  {cname:<20}  {ndcg(csub):>8.4f}  {r:>8.4f}')

# Find best overall
all_configs = [
    ('Baseline',         base_dr,          recall100(outer_base)),
    ('NestExcl70',       o_ne70_dr,         recall100(o_ne70)),
    ('Surgical',         o_surg_dr,         recall100(o_surg)),
    ('CustomThresh0.65', o_best_cust_dr,    recall100(o_best_cust)),
]

# Add sparse filter combos
o_ne_sf70 = rrf_nested_exclude(flat_sparse_f, val_qids, DOM_MODEL_NDCG, threshold=0.70)
o_ne_sf70_dr = dr(o_ne_sf70)
all_configs.append(('NestExcl70+SparseF', o_ne_sf70_dr, recall100(o_ne_sf70)))

print(f'\n── Final ranking ─────────────────────────────────────────────────────────')
all_configs.sort(key=lambda x: ndcg(x[1]), reverse=True)
for i, (cname, csub, r) in enumerate(all_configs):
    n_val = ndcg(csub)
    print(f'  #{i+1}  {cname:<30}  NDCG={n_val:.4f}  R@100={r:.4f}  Δ={n_val-ndcg(base_dr):+.4f}')
