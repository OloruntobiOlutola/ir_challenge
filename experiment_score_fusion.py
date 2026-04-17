"""
Score-based fusion experiments targeting the ranking bottleneck.

Diagnosis showed:
  - RRF is losing signal when ONE model is strongly correct (e.g., MiniLM rank=4 for CS)
    but others rank it 19-40 → RRF puts it at 26
  - CS oracle gap is largest (3.97 weighted points)
  - Issue: RRF only uses RANKS, losing the magnitude of similarity scores

This script tests:
  A) Softmax-normalized score fusion — preserves confidence magnitude
  B) Min-rank fusion (best of any model) — "OR" logic instead of "AND"
  C) Hybrid: RRF for recall + score-boosted top candidates
  D) Borda count fusion — alternative to RRF
  E) CombSUM — raw score sum after normalization
  F) Title-only BM25 as additional signal for CS/Bio/Med

Baseline: val NDCG@10 = 0.7544
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

# ── Helpers ────────────────────────────────────────────────────────────────
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

def top_n_filtered(score_matrix, qids, masks, n=100):
    sub = {}
    for i, qid in enumerate(qids):
        sc = score_matrix[i].copy()
        sc[~masks[i]] = -1e9
        idx = np.argsort(-sc)[:n]
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

def norm01_matrix(sc):
    mn = sc.min(axis=1, keepdims=True)
    mx = sc.max(axis=1, keepdims=True)
    return (sc - mn) / (mx - mn + 1e-10)

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

# Hybrid scores (sparse+dense fused at score level)
sc_ht_tuni = rrf_fuse(sc_tuni, sc_sp, k=5)
sc_ht_tbi  = rrf_fuse(sc_tbi,  sc_sp, k=5)

# ── Base retrievals ────────────────────────────────────────────────────────
masks = [domain_mask(q_dom_val.get(qid, ''), min_pool=300) for qid in val_qids]
s_bm25l = top_n(sc_bm25l, val_qids);  s_bge = top_n(sc_bge, val_qids)
s_ml    = top_n(sc_ml,    val_qids);  s_sp  = top_n(sc_sp,  val_qids)
ht_tuni = top_n(sc_ht_tuni, val_qids);  ht_tbi = top_n(sc_ht_tbi, val_qids)
s_bm25l_f = top_n_filtered(sc_bm25l, val_qids, masks)
ht_tuni_f = top_n_filtered(sc_ht_tuni, val_qids, masks)
ht_tbi_f  = top_n_filtered(sc_ht_tbi,  val_qids, masks)

flat_unf = {'BM25L': s_bm25l,'BGE': s_bge,'MiniLM': s_ml,'SPECTER2': s_sp,'TF-IDF-uni': ht_tuni,'TF-IDF-bi': ht_tbi}
flat_sf  = {'BM25L': s_bm25l_f, 'TF-IDF-uni': ht_tuni_f, 'TF-IDF-bi': ht_tbi_f}

# Current best baseline
def rrf_nested_exclude_sf(flat_unf, flat_sf, qids, q_dom_map, dom_ndcg, threshold=0.70, n=100):
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
        outer_x = [m for m in ['TF-IDF-uni','TF-IDF-bi','SPECTER2'] if m in active]
        if not inner_m: inner_m = ['BM25L','BGE','MiniLM']
        isc = {}
        for mname in inner_m:
            for rank, doc in enumerate(get(mname).get(qid, []), 1):
                isc[doc] = isc.get(doc, 0.0) + 1.0 / (1 + rank)
        inner_s = sorted(isc, key=isc.get, reverse=True)[:100]
        osc = {}
        for rank, doc in enumerate(inner_s, 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)
        for mname in outer_x + [m for m in ['BGE','BM25L'] if m in active]:
            for rank, doc in enumerate(get(mname).get(qid, []), 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)
        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:n]
    return sub

best_sub = rrf_nested_exclude_sf(flat_unf, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG, threshold=0.70)
best_sub_dr = dr(best_sub)
print(f'Current best (nested excl + sparse): NDCG={ndcg(best_sub_dr):.4f}  R@100={recall100(best_sub):.4f}')

# ── Experiment A: CombSUM — normalized score linear combination ────────────
print('\n── A. CombSUM (weighted norm scores) ────────────────────────────────────')

# Normalize each score matrix to [0,1] per query
n_bge  = norm01_matrix(sc_bge)
n_ml   = norm01_matrix(sc_ml)
n_sp   = norm01_matrix(sc_sp)
n_bm25 = norm01_matrix(sc_bm25l)
n_tuni = norm01_matrix(sc_tuni)
n_tbi  = norm01_matrix(sc_tbi)

# Weighted combsum: use per-domain model NDCG as weights
# Build weight vector for each query
qid_to_idx = {qid: i for i, qid in enumerate(val_qids)}
score_mats = {'BGE': n_bge, 'MiniLM': n_ml, 'SPECTER2': n_sp,
              'BM25L': n_bm25, 'TF-IDF-uni': n_tuni, 'TF-IDF-bi': n_tbi}

def combsum_weighted(score_mats, qids, dom_ndcg, q_dom_map, n=100):
    """Weighted CombSUM: sum of domain-NDCG-weighted normalized scores."""
    sub = {}
    for i, qid in enumerate(qids):
        qdom = q_dom_map.get(qid, '')
        weights = dom_ndcg.get(qdom, {})
        if not weights:
            weights = {m: 1.0 for m in score_mats}
        total_w = sum(weights.values()) + 1e-10
        combined = np.zeros(len(corpus_ids), dtype=np.float32)
        for mname, sc in score_mats.items():
            w = weights.get(mname, 0.0) / total_w
            combined += w * sc[i]
        idx = np.argsort(-combined)[:n]
        sub[qid] = corpus_arr[idx].tolist()
    return sub

for exclude_thresh in [None, 0.70]:
    if exclude_thresh:
        # Only use models above threshold
        def combsum_excl(score_mats, qids, dom_ndcg, q_dom_map, threshold, n=100):
            sub = {}
            for i, qid in enumerate(qids):
                qdom = q_dom_map.get(qid, '')
                weights = dom_ndcg.get(qdom, {})
                if weights:
                    best_s = max(weights.values())
                    weights = {m: w for m, w in weights.items() if w >= threshold * best_s}
                if not weights:
                    weights = dom_ndcg.get(qdom, {m: 1.0 for m in score_mats})
                total_w = sum(weights.values()) + 1e-10
                combined = np.zeros(len(corpus_ids), dtype=np.float32)
                for mname, sc in score_mats.items():
                    if mname in weights:
                        w = weights[mname] / total_w
                        combined += w * sc[i]
                idx = np.argsort(-combined)[:n]
                sub[qid] = corpus_arr[idx].tolist()
            return sub
        cs_sub = combsum_excl(score_mats, val_qids, DOM_MODEL_NDCG, q_dom_val, threshold=exclude_thresh)
    else:
        cs_sub = combsum_weighted(score_mats, val_qids, DOM_MODEL_NDCG, q_dom_val)
    cs_dr = dr(cs_sub)
    tag = f'thresh={exclude_thresh}' if exclude_thresh else 'all models'
    print(f'  CombSUM weighted ({tag}): NDCG={ndcg(cs_dr):.4f}  R@100={recall100(cs_sub):.4f}')

# Equal-weight CombSUM
eq_sub = {}
for i, qid in enumerate(val_qids):
    combined = n_bge[i] + n_ml[i] + n_sp[i] + n_bm25[i] + n_tuni[i] + n_tbi[i]
    idx = np.argsort(-combined)[:100]
    eq_sub[qid] = corpus_arr[idx].tolist()
eq_dr = dr(eq_sub)
print(f'  CombSUM equal weights:          NDCG={ndcg(eq_dr):.4f}  R@100={recall100(eq_sub):.4f}')

# Dense-only CombSUM (BGE + MiniLM + SPECTER2)
dense_sub = {}
for i, qid in enumerate(val_qids):
    combined = n_bge[i] + n_ml[i] + n_sp[i]
    idx = np.argsort(-combined)[:100]
    dense_sub[qid] = corpus_arr[idx].tolist()
dense_dr = dr(dense_sub)
print(f'  CombSUM dense-only (BGE+ML+SP): NDCG={ndcg(dense_dr):.4f}  R@100={recall100(dense_sub):.4f}')

# ── Experiment B: Min-rank fusion ─────────────────────────────────────────
print('\n── B. Min-rank fusion (best of any model) ───────────────────────────────')
def minrank_fusion(lists_dict, qids, n=100):
    """Score a doc by 1/min_rank — trust whichever model ranks it highest."""
    sub = {}
    for qid in qids:
        min_ranks = {}
        for mname, lst in lists_dict.items():
            for rank, doc in enumerate(lst.get(qid, []), 1):
                if doc not in min_ranks or rank < min_ranks[doc]:
                    min_ranks[doc] = rank
        sub[qid] = sorted(min_ranks, key=lambda d: min_ranks[d])[:n]
    return sub

mr_sub = minrank_fusion(flat_unf, val_qids)
mr_dr  = dr(mr_sub)
print(f'  Min-rank (all models):   NDCG={ndcg(mr_dr):.4f}  R@100={recall100(mr_sub):.4f}')

mr_sf = {**flat_unf, **flat_sf}
# For sparse-filtered version
flat_combined = {'BM25L': s_bm25l_f, 'BGE': s_bge, 'MiniLM': s_ml,
                 'SPECTER2': s_sp, 'TF-IDF-uni': ht_tuni_f, 'TF-IDF-bi': ht_tbi_f}
mr_sf_sub = minrank_fusion(flat_combined, val_qids)
mr_sf_dr  = dr(mr_sf_sub)
print(f'  Min-rank + sparse-filter: NDCG={ndcg(mr_sf_dr):.4f}  R@100={recall100(mr_sf_sub):.4f}')

# ── Experiment C: Hybrid RRF + score boost for high-confidence docs ────────
print('\n── C. RRF + high-confidence score boost ─────────────────────────────────')
def rrf_score_boost(rrf_sub, score_mats_list, qids, boost_lambda=0.1, n=100):
    """
    After RRF, re-score with: final = rrf_score + lambda * max_model_score
    where max_model_score is the maximum normalized score across all models.
    This boosts docs that any model is highly confident about.
    """
    qid_to_i = {qid: i for i, qid in enumerate(qids)}
    sub = {}
    for qid in qids:
        qi = qid_to_i[qid]
        cands = rrf_sub.get(qid, [])
        cand_idx = {doc: j for j, doc in enumerate(corpus_ids) if doc in set(cands)}
        # Build RRF score dict
        rrf_sc = {}
        for rank, doc in enumerate(cands, 1):
            rrf_sc[doc] = 1.0 / (2 + rank)  # use k=2 for RRF score approx
        # Max norm score across all models
        combined = {}
        for doc in cands:
            j = next((idx for idx, cid in enumerate(corpus_ids) if cid == doc), None)
            if j is None: continue
            max_sc = max(sc[qi, j] for sc in score_mats_list)
            combined[doc] = rrf_sc[doc] + boost_lambda * max_sc
        sub[qid] = sorted(combined, key=combined.get, reverse=True)[:n]
    return sub

# Build RRF first (nested excl. baseline)
# Use vectorized approach for efficiency
from scipy.sparse import find as sparse_find

# Fast vectorized max-score approach
print('  Building max-score-boosted fusion...')
norm_mats = [n_bge, n_ml, n_sp, n_bm25, n_tuni, n_tbi]

def rrf_nested_plus_maxboost(flat_unf, flat_sf, qids, q_dom_map, dom_ndcg, threshold, boost_lam, n=100):
    """Nested excl RRF + add lambda * max_model_score for each candidate."""
    rrf_res = rrf_nested_exclude_sf(flat_unf, flat_sf, qids, q_dom_map, dom_ndcg, threshold, n=200)
    qid_to_i = {qid: i for i, qid in enumerate(qids)}
    cid_to_j = {cid: j for j, cid in enumerate(corpus_ids)}
    sub = {}
    for qid in qids:
        qi = qid_to_i.get(qid)
        if qi is None: continue
        cands = rrf_res.get(qid, [])
        rrf_scores = {doc: 1.0/(2+rank) for rank, doc in enumerate(cands, 1)}
        final = {}
        for doc in cands:
            j = cid_to_j.get(doc)
            if j is None: continue
            max_s = max(sc[qi, j] for sc in norm_mats)
            final[doc] = rrf_scores[doc] + boost_lam * max_s
        sub[qid] = sorted(final, key=final.get, reverse=True)[:n]
    return sub

for lam in [0.05, 0.10, 0.20, 0.30]:
    bs = rrf_nested_plus_maxboost(flat_unf, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG, 0.70, boost_lam=lam)
    bs_dr = dr(bs)
    print(f'  RRF + max-score boost λ={lam}: NDCG={ndcg(bs_dr):.4f}  R@100={recall100(bs):.4f}')

# ── Experiment D: Borda count ─────────────────────────────────────────────
print('\n── D. Borda count fusion ────────────────────────────────────────────────')
def borda_count(lists_dict, qids, n=100, N=100):
    """Borda count: each doc gets (N - rank) points from each list."""
    sub = {}
    for qid in qids:
        sc = {}
        for mname, lst in lists_dict.items():
            for rank, doc in enumerate(lst.get(qid, [])[:N], 1):
                sc[doc] = sc.get(doc, 0.0) + (N - rank)
        sub[qid] = sorted(sc, key=sc.get, reverse=True)[:n]
    return sub

bc_sub = borda_count(flat_combined, val_qids)
bc_dr  = dr(bc_sub)
print(f'  Borda count + sparse-filter: NDCG={ndcg(bc_dr):.4f}  R@100={recall100(bc_sub):.4f}')

# ── Experiment E: Extend pool to 200 then RRF — check recall improvement ──
print('\n── E. Extended pool (200 candidates) ────────────────────────────────────')
masks200 = [domain_mask(q_dom_val.get(qid, ''), min_pool=300) for qid in val_qids]
s_bm25l_200 = top_n(sc_bm25l, val_qids, n=200)
s_bge_200   = top_n(sc_bge,   val_qids, n=200)
s_ml_200    = top_n(sc_ml,    val_qids, n=200)
s_sp_200    = top_n(sc_sp,    val_qids, n=200)
ht_tuni_200 = top_n(sc_ht_tuni, val_qids, n=200)
ht_tbi_200  = top_n(sc_ht_tbi,  val_qids, n=200)
s_bm25l_200f = top_n_filtered(sc_bm25l,   val_qids, masks200, n=200)
ht_tuni_200f = top_n_filtered(sc_ht_tuni, val_qids, masks200, n=200)
ht_tbi_200f  = top_n_filtered(sc_ht_tbi,  val_qids, masks200, n=200)

flat_unf_200 = {'BM25L': s_bm25l_200,'BGE': s_bge_200,'MiniLM': s_ml_200,
                'SPECTER2': s_sp_200,'TF-IDF-uni': ht_tuni_200,'TF-IDF-bi': ht_tbi_200}
flat_sf_200  = {'BM25L': s_bm25l_200f, 'TF-IDF-uni': ht_tuni_200f, 'TF-IDF-bi': ht_tbi_200f}

sub_200 = rrf_nested_exclude_sf(flat_unf_200, flat_sf_200, val_qids, q_dom_val, DOM_MODEL_NDCG, threshold=0.70, n=200)
sub_200_top100 = {qid: cands[:100] for qid, cands in sub_200.items()}
sub_200_dr = dr(sub_200_top100)
print(f'  Extended pool 200→100 + excl + sf: NDCG={ndcg(sub_200_dr):.4f}  R@100={recall100(sub_200_top100):.4f}')

# Check recall at different cutoffs
for k in [100, 150, 200]:
    hits = sum(1 for qid, rels in qrels.items()
               for d in sub_200.get(qid,[])[:k] if d in set(rels))
    total = sum(len(set(v)) for v in qrels.values())
    print(f'  Recall@{k:<3} with 200-pool = {hits/total:.4f}')

# ── Experiment F: Hybrid score combination then RRF ────────────────────────
print('\n── F. Dense ensemble score combination (CombMNZ-style) ─────────────────')
# CombMNZ: sum of scores * number of models retrieving it
def combmnz(score_mats_list, model_names, qids, n=100):
    """CombMNZ: sum of norm scores, but multiply by number of models that retrieved."""
    sub = {}
    for i, qid in enumerate(qids):
        # Sum of normalized scores
        sc_sum = np.zeros(len(corpus_ids), dtype=np.float32)
        hit_count = np.zeros(len(corpus_ids), dtype=np.int32)
        for sc in score_mats_list:
            sc_sum += sc[i]
            # Count as "retrieved" if in top-200
            top200_idx = np.argsort(-sc[i])[:200]
            hit_count[top200_idx] += 1
        # CombMNZ: multiply sum by hit count
        combined = sc_sum * hit_count
        idx = np.argsort(-combined)[:n]
        sub[qid] = corpus_arr[idx].tolist()
    return sub

mnz_sub = combmnz(norm_mats, list(score_mats.keys()), val_qids)
mnz_dr  = dr(mnz_sub)
print(f'  CombMNZ all models:    NDCG={ndcg(mnz_dr):.4f}  R@100={recall100(mnz_sub):.4f}')

# Dense-only CombMNZ
mnz_dense_sub = combmnz([n_bge, n_ml, n_sp], ['BGE','MiniLM','SPECTER2'], val_qids)
mnz_dense_dr  = dr(mnz_dense_sub)
print(f'  CombMNZ dense-only:    NDCG={ndcg(mnz_dense_dr):.4f}  R@100={recall100(mnz_dense_sub):.4f}')

# ── Summary ────────────────────────────────────────────────────────────────
print('\n── Summary ──────────────────────────────────────────────────────────────')
results = [
    ('Current best (RRF nested excl+sf)',  ndcg(best_sub_dr),  recall100(best_sub)),
    ('A. CombSUM weighted excl=0.70',      ndcg(combsum_excl(score_mats, val_qids, DOM_MODEL_NDCG, q_dom_val, 0.70)), None),
    ('A. CombSUM equal weights',           ndcg(eq_dr),        recall100(eq_sub)),
    ('B. Min-rank + sparse-filter',        ndcg(mr_sf_dr),     recall100(mr_sf_sub)),
    ('C. RRF + max-boost λ=0.10',          ndcg(dr(rrf_nested_plus_maxboost(flat_unf, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG, 0.70, 0.10))), None),
    ('D. Borda + sparse-filter',           ndcg(bc_dr),        recall100(bc_sub)),
    ('E. Pool-200 → top-100',              ndcg(sub_200_dr),   recall100(sub_200_top100)),
    ('F. CombMNZ all models',              ndcg(mnz_dr),       recall100(mnz_sub)),
]
results.sort(key=lambda x: x[1], reverse=True)
print(f'  {"Config":<38}  {"NDCG@10":>8}  {"R@100":>8}  {"Δ":>7}')
print('  ' + '-' * 66)
baseline = ndcg(best_sub_dr)
for name, n_val, r_val in results:
    r_str = f'{r_val:.4f}' if r_val is not None else '  —   '
    print(f'  {name:<38}  {n_val:>8.4f}  {r_str:>8}  {n_val-baseline:>+7.4f}')
