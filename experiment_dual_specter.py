"""
Dual SPECTER2 adapter experiment + improved RRF configurations.

Available SPECTER2 embeddings:
  - proximity queries:  specter_prox_embed/queries_1_embeddings.npy
  - proximity corpus:   submissions/specter2_corpus_emb_ft.npy
  - adhoc queries:      specter_prox_embed/specter2_adhoc_val_queries.npy
  - adhoc corpus:       specter_prox_embed/specter2_adhoc_corpus.npy

Proximity = trained on citation co-occurrence (papers close in citation graph)
Adhoc     = trained on query-document matching (more like general search)

Hypothesis: using BOTH in the ensemble adds complementary signal.

Also tests:
  - Different RRF k configurations for the outer ensemble
  - Adding specter2_adhoc as a 7th model
  - Tuning the threshold more precisely around 0.70

Baseline: 0.7544
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

def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q / (nq + 1e-10)) @ (c / (nc + 1e-10)).T).astype(np.float32)

def top_n(scores, qids, n=100):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-scores[i])[:n]
        sub[qid] = corpus_arr[idx].tolist()
    return sub

def top_n_filtered(sc, qids, masks, n=100):
    sub = {}
    for i, qid in enumerate(qids):
        s = sc[i].copy(); s[~masks[i]] = -1e9
        idx = np.argsort(-s)[:n]
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

def dr(sub, q_dom_map=q_dom_val, skip=frozenset({'Business'})):
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

def rrf_nested_exclude_sf(flat_unf, flat_sf, qids, q_dom_map, dom_ndcg,
                          threshold=0.70, inner_k=1, outer_k=2, n=100):
    sub = {}
    for qid in qids:
        qdom = q_dom_map.get(qid, '')
        dscores = dom_ndcg.get(qdom, {})
        if dscores:
            best_s = max(dscores.values())
            active = {m for m, s in dscores.items() if s >= threshold * best_s}
        else:
            active = set(flat_unf.keys())
        if not active:
            active = set(flat_unf.keys())
        def get(mname): return flat_sf.get(mname, flat_unf[mname])
        inner_m = [m for m in ['BM25L','BGE','MiniLM'] if m in active]
        outer_x = [m for m in ['TF-IDF-uni','TF-IDF-bi','SPECTER2','SP2-adhoc'] if m in active]
        if not inner_m: inner_m = ['BM25L','BGE','MiniLM']
        isc = {}
        for mname in inner_m:
            for rank, doc in enumerate(get(mname).get(qid, []), 1):
                isc[doc] = isc.get(doc, 0.0) + 1.0 / (inner_k + rank)
        inner_s = sorted(isc, key=isc.get, reverse=True)[:100]
        osc = {}
        for rank, doc in enumerate(inner_s, 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (outer_k + rank)
        for mname in outer_x + [m for m in ['BGE','BM25L'] if m in active]:
            for rank, doc in enumerate(get(mname).get(qid, []), 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (outer_k + rank)
        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:n]
    return sub

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
# SPECTER2 proximity (current)
sp_q_prox  = np.load('specter_prox_embed/queries_1_embeddings.npy')
sp_ft_c    = np.load('submissions/specter2_corpus_emb_ft.npy')
sc_sp_prox = cosine(sp_q_prox, sp_ft_c)
# SPECTER2 adhoc
sp_q_adhoc = np.load('specter_prox_embed/specter2_adhoc_val_queries.npy')
sp_ad_c    = np.load('specter_prox_embed/specter2_adhoc_corpus.npy')
sc_sp_adhoc = cosine(sp_q_adhoc, sp_ad_c)

sc_ht_tuni = rrf_fuse(sc_tuni, sc_sp_prox, k=5)
sc_ht_tbi  = rrf_fuse(sc_tbi,  sc_sp_prox, k=5)

masks = [domain_mask(q_dom_val.get(qid, ''), min_pool=300) for qid in val_qids]

# ── Baseline ────────────────────────────────────────────────────────────────
s_bm25l = top_n(sc_bm25l, val_qids);  s_bge = top_n(sc_bge, val_qids)
s_ml    = top_n(sc_ml,    val_qids);  s_sp  = top_n(sc_sp_prox, val_qids)
ht_tuni = top_n(sc_ht_tuni, val_qids);  ht_tbi = top_n(sc_ht_tbi, val_qids)
s_bm25l_f = top_n_filtered(sc_bm25l,   val_qids, masks)
ht_tuni_f = top_n_filtered(sc_ht_tuni, val_qids, masks)
ht_tbi_f  = top_n_filtered(sc_ht_tbi,  val_qids, masks)

flat_unf = {'BM25L': s_bm25l,'BGE': s_bge,'MiniLM': s_ml,'SPECTER2': s_sp,'TF-IDF-uni': ht_tuni,'TF-IDF-bi': ht_tbi}
flat_sf  = {'BM25L': s_bm25l_f,'TF-IDF-uni': ht_tuni_f,'TF-IDF-bi': ht_tbi_f}

best_rrf  = rrf_nested_exclude_sf(flat_unf, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG, threshold=0.70)
best_dr   = dr(best_rrf)
print(f'Baseline: NDCG={ndcg(best_dr):.4f}  R@100={recall100(best_rrf):.4f}')

# ── Measure SPECTER2 adhoc standalone ────────────────────────────────────
s_sp_adhoc = top_n(sc_sp_adhoc, val_qids)
sp_adhoc_dr = dr(s_sp_adhoc)
print(f'SPECTER2 adhoc standalone: NDCG={ndcg(sp_adhoc_dr):.4f}  R@100={recall100(s_sp_adhoc):.4f}')

# Per-domain NDCG of adhoc
print('\nPer-domain NDCG of SP2-adhoc:')
dom_to_qids = defaultdict(list)
for qid in val_qids:
    dom_to_qids[q_dom_val.get(qid,'')].append(qid)
adhoc_dom_ndcg = {}
for domain, qids_d in sorted(dom_to_qids.items()):
    qids_d_in_qrels = [q for q in qids_d if q in qrels]
    if not qids_d_in_qrels: continue
    n = ndcg_subset(sp_adhoc_dr, qids_d_in_qrels)
    adhoc_dom_ndcg[domain] = n
    if n > DOM_MODEL_NDCG.get(domain,{}).get('SPECTER2', 0):
        print(f'  {domain:<22}: adhoc={n:.4f}  prox={DOM_MODEL_NDCG.get(domain,{}).get("SPECTER2",0):.4f}  +{n-DOM_MODEL_NDCG.get(domain,{}).get("SPECTER2",0):+.4f} ← adhoc BETTER')

# ── Add SPECTER2 adhoc to the ensemble ────────────────────────────────────
print('\n── Adding SP2-adhoc as 7th model ────────────────────────────────────────')

# Build combined DOM_MODEL_NDCG with adhoc
DOM_MODEL_NDCG_7 = {d: {**v, 'SP2-adhoc': adhoc_dom_ndcg.get(d, 0.0)}
                     for d, v in DOM_MODEL_NDCG.items()}

flat_unf_7 = {**flat_unf, 'SP2-adhoc': s_sp_adhoc}

for thresh in [0.65, 0.70, 0.75]:
    sub_7 = rrf_nested_exclude_sf(flat_unf_7, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG_7, threshold=thresh)
    sub_7_dr = dr(sub_7)
    print(f'  +SP2-adhoc thresh={thresh}: NDCG={ndcg(sub_7_dr):.4f}  R@100={recall100(sub_7):.4f}  Δ={ndcg(sub_7_dr)-ndcg(best_dr):+.4f}')

# ── Hybrid TF-IDF using adhoc SP2 instead of prox SP2 ───────────────────
print('\n── TF-IDF hybrid with adhoc SP2 ─────────────────────────────────────────')
sc_ht_tuni_ad = rrf_fuse(sc_tuni, sc_sp_adhoc, k=5)
sc_ht_tbi_ad  = rrf_fuse(sc_tbi,  sc_sp_adhoc, k=5)
ht_tuni_ad = top_n(sc_ht_tuni_ad, val_qids)
ht_tbi_ad  = top_n(sc_ht_tbi_ad,  val_qids)
ht_tuni_adf = top_n_filtered(sc_ht_tuni_ad, val_qids, masks)
ht_tbi_adf  = top_n_filtered(sc_ht_tbi_ad,  val_qids, masks)

flat_unf_ad = {**flat_unf, 'TF-IDF-uni': ht_tuni_ad, 'TF-IDF-bi': ht_tbi_ad}
flat_sf_ad  = {'BM25L': s_bm25l_f, 'TF-IDF-uni': ht_tuni_adf, 'TF-IDF-bi': ht_tbi_adf}
sub_ad = rrf_nested_exclude_sf(flat_unf_ad, flat_sf_ad, val_qids, q_dom_val, DOM_MODEL_NDCG, threshold=0.70)
sub_ad_dr = dr(sub_ad)
print(f'  TF-IDF hybrid with adhoc SP2: NDCG={ndcg(sub_ad_dr):.4f}  Δ={ndcg(sub_ad_dr)-ndcg(best_dr):+.4f}')

# Both prox and adhoc TF-IDF hybrids
flat_unf_both = {
    'BM25L': s_bm25l, 'BGE': s_bge, 'MiniLM': s_ml,
    'SPECTER2': s_sp, 'SP2-adhoc': s_sp_adhoc,
    'TF-IDF-uni': ht_tuni, 'TF-IDF-bi': ht_tbi,
    'TF-IDF-uni-ad': ht_tuni_ad, 'TF-IDF-bi-ad': ht_tbi_ad,
}
flat_sf_both = {
    'BM25L': s_bm25l_f, 'TF-IDF-uni': ht_tuni_f, 'TF-IDF-bi': ht_tbi_f,
    'TF-IDF-uni-ad': ht_tuni_adf, 'TF-IDF-bi-ad': ht_tbi_adf,
}
DOM_MODEL_NDCG_BOTH = {d: {**v, 'SP2-adhoc': adhoc_dom_ndcg.get(d, 0.0),
                            'TF-IDF-uni-ad': v.get('TF-IDF-uni', 0),
                            'TF-IDF-bi-ad':  v.get('TF-IDF-bi', 0)}
                        for d, v in DOM_MODEL_NDCG.items()}

sub_both = rrf_nested_exclude_sf(flat_unf_both, flat_sf_both, val_qids, q_dom_val, DOM_MODEL_NDCG_BOTH, threshold=0.70)
sub_both_dr = dr(sub_both)
print(f'  Both adapters + 4 TF-IDF hybrids: NDCG={ndcg(sub_both_dr):.4f}  Δ={ndcg(sub_both_dr)-ndcg(best_dr):+.4f}')

# ── Fine-tune threshold around 0.70 with current best config ─────────────
print('\n── Fine threshold sweep (current best config) ───────────────────────────')
print(f'  {"threshold":>12}  {"NDCG@10":>8}  {"R@100":>8}  {"Δ":>7}')
best_thresh = 0.70
best_thresh_n = ndcg(best_dr)
for thresh in [0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75]:
    sub_t = rrf_nested_exclude_sf(flat_unf, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG, threshold=thresh)
    sub_t_dr = dr(sub_t)
    n_t = ndcg(sub_t_dr)
    marker = ' <--' if n_t > best_thresh_n else ''
    print(f'  {thresh:>12.2f}  {n_t:>8.4f}  {recall100(sub_t):>8.4f}  {n_t-ndcg(best_dr):>+7.4f}{marker}')
    if n_t > best_thresh_n:
        best_thresh_n = n_t
        best_thresh = thresh

print(f'\n  Best threshold: {best_thresh}  NDCG={best_thresh_n:.4f}')

# ── Summary ──────────────────────────────────────────────────────────────
print('\n── Summary ──────────────────────────────────────────────────────────────')
print(f'  Baseline:                    NDCG={ndcg(best_dr):.4f}')
print(f'  SP2-adhoc standalone:        NDCG={ndcg(sp_adhoc_dr):.4f}')
