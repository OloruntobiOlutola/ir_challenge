"""
Targeted reranking for hard queries.

CS domain observations:
  - Domain hard-sort does NOTHING for CS queries (all candidates are CS anyway)
  - For CS query 4a45163b94c0: MiniLM rank=4, ensemble=26 → need per-query reranking
  - SPECTER2 is citation-trained and might give better within-domain ranking

Strategy A: For CS (and other domains where all-same-domain queries dominate),
re-sort the top-100 RRF pool by a specific model's score instead of hard domain sort.

Strategy B: Use RRF score as primary sort but break ties with dense model confidence.

Strategy C: Per-query, take the score from the model that was MOST DIFFERENT from
consensus (highest-confidence "outlier" model) and blend with RRF score.

Strategy D: Stochastic top-K: take union of top-K from each model, then re-rank
by weighted sum of scores (not ranks) using domain-model NDCG as weights.

Strategy E: Per-query model selection — for each query, pick the single best model
based on the query's domain and use only that model's ranking.

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

# Best model per domain
DOM_BEST_MODEL = {d: max(scores, key=scores.get) for d, scores in DOM_MODEL_NDCG.items()}

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
sc_ht_tuni = rrf_fuse(sc_tuni, sc_sp, k=5)
sc_ht_tbi  = rrf_fuse(sc_tbi,  sc_sp, k=5)

qid_to_idx = {q: i for i, q in enumerate(val_qids)}
cid_to_idx = {c: j for j, c in enumerate(corpus_ids)}

score_matrices = {
    'BM25L': sc_bm25l, 'BGE': sc_bge, 'MiniLM': sc_ml,
    'SPECTER2': sc_sp, 'TF-IDF-uni': sc_tuni, 'TF-IDF-bi': sc_tbi,
}

# ── Build baseline ─────────────────────────────────────────────────────────
masks = [domain_mask(q_dom_val.get(qid, ''), min_pool=300) for qid in val_qids]
s_bm25l = top_n(sc_bm25l, val_qids);  s_bge = top_n(sc_bge, val_qids)
s_ml    = top_n(sc_ml,    val_qids);  s_sp  = top_n(sc_sp,  val_qids)
ht_tuni = top_n(sc_ht_tuni, val_qids);  ht_tbi = top_n(sc_ht_tbi, val_qids)
s_bm25l_f = top_n_filtered(sc_bm25l,   val_qids, masks)
ht_tuni_f = top_n_filtered(sc_ht_tuni, val_qids, masks)
ht_tbi_f  = top_n_filtered(sc_ht_tbi,  val_qids, masks)

flat_unf = {'BM25L': s_bm25l,'BGE': s_bge,'MiniLM': s_ml,'SPECTER2': s_sp,'TF-IDF-uni': ht_tuni,'TF-IDF-bi': ht_tbi}
flat_sf  = {'BM25L': s_bm25l_f,'TF-IDF-uni': ht_tuni_f,'TF-IDF-bi': ht_tbi_f}

best_rrf = rrf_nested_exclude_sf(flat_unf, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG, threshold=0.70)
best_dr  = dr(best_rrf)
print(f'Baseline: NDCG={ndcg(best_dr):.4f}  R@100={recall100(best_rrf):.4f}')

# ── Strategy A: Per-domain secondary sort using best model score ───────────
print('\n── A. Secondary sort by best-model score within domain ──────────────────')
def dr_model_score(rrf_sub, score_mats, qid_to_i, cid_to_j, dom_best_model,
                   q_dom_map=q_dom_val, skip=frozenset({'Business'}), alpha=0.1):
    """
    Hard-sort domain rerank, but within same-domain docs, use best model's
    cosine score as tiebreaker: sort_key = (-dom_weight, -alpha*model_score, rank)
    """
    out = {}
    for qid, cands in rrf_sub.items():
        qd   = q_dom_map.get(qid, '')
        if qd in skip:
            out[qid] = cands[:100]; continue
        qi   = qid_to_i.get(qid)
        wr   = dw.get(qd, {})
        bm   = dom_best_model.get(qd, 'BGE')
        sc   = score_mats.get(bm)
        scored = []
        for rank, d in enumerate(cands):
            dom_w  = wr.get(c_dom_map.get(d, ''), 0.0)
            cj     = cid_to_j.get(d)
            model_s = float(sc[qi, cj]) if (sc is not None and qi is not None and cj is not None) else 0.0
            scored.append((-dom_w - alpha * model_s, rank, d))
        scored.sort()
        out[qid] = [d for _, _, d in scored[:100]]
    return out

for alpha in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    sub_a = dr_model_score(best_rrf, score_matrices, qid_to_idx, cid_to_idx, DOM_BEST_MODEL, alpha=alpha)
    print(f'  alpha={alpha:.2f}: NDCG={ndcg(sub_a):.4f}  Δ={ndcg(sub_a)-ndcg(best_dr):+.4f}')

# ── Strategy B: Pool-level reranking by best model score (no domain sort) ─
print('\n── B. Rerank top-100 pool by best-model score only ──────────────────────')
def rerank_by_score(rrf_sub, score_mats, qid_to_i, cid_to_j, dom_best_model,
                    q_dom_map=q_dom_val, n=100):
    """For each query, rerank the RRF top-100 using best model's cosine score."""
    out = {}
    for qid, cands in rrf_sub.items():
        qd = q_dom_map.get(qid, '')
        qi = qid_to_i.get(qid)
        bm = dom_best_model.get(qd, 'TF-IDF-uni')
        sc = score_mats.get(bm)
        if sc is None or qi is None:
            out[qid] = cands[:n]; continue
        scored = []
        for rank, d in enumerate(cands):
            cj = cid_to_j.get(d)
            s  = float(sc[qi, cj]) if cj is not None else 0.0
            scored.append((-s, rank, d))
        scored.sort()
        out[qid] = [d for _, _, d in scored[:n]]
    return out

sub_b = rerank_by_score(best_rrf, score_matrices, qid_to_idx, cid_to_idx, DOM_BEST_MODEL)
print(f'  Best-model rescore: NDCG={ndcg(sub_b):.4f}  Δ={ndcg(sub_b)-ndcg(best_dr):+.4f}')

# ── Strategy C: RRF + domain sort THEN re-sort by best-model score for hard domains ─
print('\n── C. Two-stage: domain sort then within-domain sort by model score ──────')

# Hard domains = where oracle gap is large and all models score poorly
HARD_DOMAINS = {'Computer Science', 'Medicine', 'Biology'}
SOFT_DOMAINS = {'Philosophy', 'Engineering', 'Physics'}

def dr_twostage(rrf_sub, score_mats, qid_to_i, cid_to_j, dom_best_model,
                hard_domains, q_dom_map=q_dom_val, skip=frozenset({'Business'}),
                blend=0.3):
    """
    For hard domains: blend RRF rank with best model score.
    For other domains: use standard hard-sort domain rerank.
    """
    out = {}
    for qid, cands in rrf_sub.items():
        qd = q_dom_map.get(qid, '')
        if qd in skip:
            out[qid] = cands[:100]; continue
        wr  = dw.get(qd, {})
        qi  = qid_to_i.get(qid)
        if qd in hard_domains and qi is not None:
            # For hard domains: sort by (-dom_weight, -blend*model_score, rank)
            bm = dom_best_model.get(qd, 'TF-IDF-uni')
            sc = score_mats.get(bm)
            scored = []
            for rank, d in enumerate(cands):
                dom_w  = wr.get(c_dom_map.get(d, ''), 0.0)
                cj     = cid_to_j.get(d)
                ms     = float(sc[qi, cj]) if sc is not None and cj is not None else 0.0
                scored.append((-dom_w - blend * ms, rank, d))
        else:
            # Standard hard-sort
            scored = [(-wr.get(c_dom_map.get(d,''), 0.0), rank, d) for rank, d in enumerate(cands)]
        scored.sort()
        out[qid] = [d for _, _, d in scored[:100]]
    return out

dom_to_qids = defaultdict(list)
for qid in val_qids:
    dom_to_qids[q_dom_val.get(qid,'')].append(qid)

for blend in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]:
    sub_c = dr_twostage(best_rrf, score_matrices, qid_to_idx, cid_to_idx, DOM_BEST_MODEL,
                        HARD_DOMAINS, blend=blend)
    n_c = ndcg(sub_c)
    marker = ' <--' if n_c > ndcg(best_dr) else ''
    print(f'  blend={blend:.1f}: NDCG={n_c:.4f}  Δ={n_c-ndcg(best_dr):+.4f}{marker}')

# ── Strategy D: Per-query top model (oracle-ish, for ablation) ────────────
print('\n── D. Use single best model per query (oracle ablation) ─────────────────')
# For each query, use the model that scores highest for that specific query
# This is essentially "if we knew which model to trust per query"
def oracle_single_model(score_mats, qids, q_dom_map, dom_model_ndcg, n=100):
    """Use domain-best model for each query."""
    sub = {}
    for i, qid in enumerate(qids):
        qdom = q_dom_map.get(qid, '')
        scores = dom_model_ndcg.get(qdom, {})
        if scores:
            bm = max(scores, key=scores.get)
        else:
            bm = 'TF-IDF-uni'
        sc = score_mats.get(bm)
        if sc is None: sc = score_mats['BGE']
        idx = np.argsort(-sc[i])[:n]
        sub[qid] = corpus_arr[idx].tolist()
    return sub

single_best = oracle_single_model(score_matrices, val_qids, q_dom_val, DOM_MODEL_NDCG)
single_dr   = dr(single_best)
print(f'  Domain-best single model: NDCG={ndcg(single_dr):.4f}  R@100={recall100(single_best):.4f}')
print(f'  (This shows the ceiling for per-domain model selection without ensemble)')

# ── Strategy E: Domain-specific secondary score models ───────────────────
print('\n── E. Domain-specific secondary rerank models ────────────────────────────')
# For CS: use MiniLM (best single-query performance for some CS queries)
# For Medicine/Biology: use TF-IDF-uni (best domain score)
DOMAIN_RERANK_MODEL = {
    'Computer Science': 'TF-IDF-bi',  # best CS model
    'Biology': 'TF-IDF-uni',
    'Medicine': 'TF-IDF-uni',
    'Physics': 'TF-IDF-uni',
}

for blend in [0.3, 0.5, 1.0, 2.0]:
    def dr_domain_model(rrf_sub, blend=blend):
        out = {}
        for qid, cands in rrf_sub.items():
            qd = q_dom_val.get(qid, '')
            if qd in {'Business'}:
                out[qid] = cands[:100]; continue
            wr  = dw.get(qd, {})
            qi  = qid_to_idx.get(qid)
            bm  = DOMAIN_RERANK_MODEL.get(qd, None)
            if bm and qi is not None:
                sc = score_matrices.get(bm)
                scored = []
                for rank, d in enumerate(cands):
                    dom_w = wr.get(c_dom_map.get(d,''), 0.0)
                    cj    = cid_to_idx.get(d)
                    ms    = float(sc[qi, cj]) if sc is not None and cj is not None else 0.0
                    scored.append((-dom_w - blend * ms, rank, d))
                scored.sort()
            else:
                scored = [(-wr.get(c_dom_map.get(d,''), 0.0), rank, d) for rank, d in enumerate(cands)]
                scored.sort()
            out[qid] = [d for _, _, d in scored[:100]]
        return out

    sub_e = dr_domain_model(best_rrf)
    n_e = ndcg(sub_e)
    marker = ' <--' if n_e > ndcg(best_dr) else ''
    print(f'  blend={blend:.1f}: NDCG={n_e:.4f}  Δ={n_e-ndcg(best_dr):+.4f}{marker}')

# ── Strategy F: Different k in RRF based on query characteristics ─────────
print('\n── F. Adaptive RRF k based on domain ────────────────────────────────────')
# Hard domains (CS, Medicine, Biology) get higher k (softer fusion → more model diversity)
# Easy domains (Geology, Chemistry, Math) get lower k (harder → trust top-ranked docs)
DOMAIN_K = {
    'Computer Science': {'inner': 3, 'outer': 5},
    'Medicine': {'inner': 2, 'outer': 3},
    'Biology': {'inner': 2, 'outer': 3},
    'Philosophy': {'inner': 1, 'outer': 2},  # only TF-IDF-bi anyway
    'Engineering': {'inner': 1, 'outer': 1},  # only BGE anyway
}
DEFAULT_K = {'inner': 1, 'outer': 2}

def rrf_adaptive_k(flat_unf, flat_sf, qids, q_dom_map, dom_ndcg, threshold,
                   domain_k, default_k, n=100):
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
        k_vals = domain_k.get(qdom, default_k)
        ki = k_vals['inner']; ko = k_vals['outer']
        inner_m = [m for m in ['BM25L','BGE','MiniLM'] if m in active]
        outer_x = [m for m in ['TF-IDF-uni','TF-IDF-bi','SPECTER2'] if m in active]
        if not inner_m: inner_m = ['BM25L','BGE','MiniLM']
        isc = {}
        for mname in inner_m:
            for rank, doc in enumerate(get(mname).get(qid, []), 1):
                isc[doc] = isc.get(doc, 0.0) + 1.0 / (ki + rank)
        inner_s = sorted(isc, key=isc.get, reverse=True)[:100]
        osc = {}
        for rank, doc in enumerate(inner_s, 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (ko + rank)
        for mname in outer_x + [m for m in ['BGE','BM25L'] if m in active]:
            for rank, doc in enumerate(get(mname).get(qid, []), 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (ko + rank)
        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:n]
    return sub

sub_f = rrf_adaptive_k(flat_unf, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG, 0.70,
                       DOMAIN_K, DEFAULT_K)
sub_f_dr = dr(sub_f)
print(f'  Adaptive k (hard domains → higher k): NDCG={ndcg(sub_f_dr):.4f}  Δ={ndcg(sub_f_dr)-ndcg(best_dr):+.4f}')

# ── Per-domain comparison of best new approach vs baseline ────────────────
print('\n── Summary ──────────────────────────────────────────────────────────────')
all_res = [('Baseline', ndcg(best_dr), recall100(best_rrf))]
# Re-compute best from each strategy
for blend_val in [0.3, 0.5, 1.0]:
    for hd in [HARD_DOMAINS, {'Computer Science'}]:
        sub_x = dr_twostage(best_rrf, score_matrices, qid_to_idx, cid_to_idx, DOM_BEST_MODEL, hd, blend=blend_val)
        all_res.append((f'TwoStage hd={len(hd)} bl={blend_val}', ndcg(sub_x), recall100(sub_x)))

all_res.sort(key=lambda x: -x[1])
for name, n_v, r_v in all_res[:5]:
    print(f'  {name:<40}  NDCG={n_v:.4f}  R@100={r_v:.4f}  Δ={n_v-ndcg(best_dr):+.4f}')
