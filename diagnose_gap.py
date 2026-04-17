"""
Deep diagnostic: where are we losing points and what does 0.80 require?

1. Per-query NDCG@10 breakdown — find which queries are dragging us down
2. Oracle analysis per domain — where is the ranking ceiling?
3. Per-query: rank of the first relevant doc — where are the hardest misses?
4. CS-specific analysis (12 queries, 0.56 NDCG — biggest drag)
5. Title-only vs full-text retrieval comparison
6. Model correlation analysis — which pairs of models agree/disagree most?

Baseline (current best): val NDCG@10 = 0.7544
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

def rrf_nested_exclude_sf(flat_unf, flat_sf, qids, q_dom_map, dom_ndcg, threshold=0.70, n=100):
    sub = {}
    for qid in qids:
        qdom = q_dom_map.get(qid, '')
        dscores = dom_ndcg.get(qdom, {})
        if dscores:
            best_s = max(dscores.values())
            min_s  = best_s * threshold
            active = {m for m, s in dscores.items() if s >= min_s}
        else:
            active = set(flat_unf.keys())
        if not active:
            active = set(flat_unf.keys())
        def get(mname):
            return flat_sf.get(mname, flat_unf[mname])
        inner_m = [m for m in ['BM25L', 'BGE', 'MiniLM'] if m in active]
        outer_x = [m for m in ['TF-IDF-uni', 'TF-IDF-bi', 'SPECTER2'] if m in active]
        if not inner_m:
            inner_m = ['BM25L', 'BGE', 'MiniLM']
        isc = {}
        for mname in inner_m:
            for rank, doc in enumerate(get(mname).get(qid, []), 1):
                isc[doc] = isc.get(doc, 0.0) + 1.0 / (1 + rank)
        inner_s = sorted(isc, key=isc.get, reverse=True)[:100]
        osc = {}
        for rank, doc in enumerate(inner_s, 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)
        for mname in outer_x + [m for m in ['BGE', 'BM25L'] if m in active]:
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

# ── Build best submission ──────────────────────────────────────────────────
masks = [domain_mask(q_dom_val.get(qid, ''), min_pool=300) for qid in val_qids]
s_bm25l = top100(sc_bm25l, val_qids);  s_bge = top100(sc_bge, val_qids)
s_ml    = top100(sc_ml,    val_qids);  s_sp  = top100(sc_sp,  val_qids)
ht_tuni = top100(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids)
ht_tbi  = top100(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids)
s_bm25l_f = top100_filtered(sc_bm25l, val_qids, masks)
ht_tuni_f = top100_filtered(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids, masks)
ht_tbi_f  = top100_filtered(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids, masks)

flat_unf = {'BM25L': s_bm25l,'BGE': s_bge,'MiniLM': s_ml,'SPECTER2': s_sp,'TF-IDF-uni': ht_tuni,'TF-IDF-bi': ht_tbi}
flat_sf  = {'BM25L': s_bm25l_f, 'TF-IDF-uni': ht_tuni_f, 'TF-IDF-bi': ht_tbi_f}

best_sub   = rrf_nested_exclude_sf(flat_unf, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG, threshold=0.70)
best_sub_dr = dr(best_sub)

corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
qid_to_idx       = {qid: i for i, qid in enumerate(val_qids)}

# ── 1. Per-query NDCG@10 breakdown ────────────────────────────────────────
print('\n── 1. Per-query NDCG@10 (sorted worst to best, showing 30 worst) ─────────')
q_ndcg = {}
for qid, rels in qrels.items():
    ranked  = best_sub_dr.get(qid, [])[:10]
    rel_set = set(rels) if isinstance(rels, list) else set()
    if not rel_set: continue
    dcg  = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked) if d in rel_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), 10)))
    q_ndcg[qid] = dcg / idcg if idcg > 0 else 0.0

print(f'  Mean NDCG@10 = {np.mean(list(q_ndcg.values())):.4f}')
print(f'  Queries with NDCG=0: {sum(1 for v in q_ndcg.values() if v == 0)}')
print(f'  Queries with NDCG<0.3: {sum(1 for v in q_ndcg.values() if v < 0.3)}')
print(f'  Queries with NDCG<0.5: {sum(1 for v in q_ndcg.values() if v < 0.5)}')
print(f'  Queries with NDCG>=0.9: {sum(1 for v in q_ndcg.values() if v >= 0.9)}')

sorted_q = sorted(q_ndcg.items(), key=lambda x: x[1])
print(f'\n  {"qid[:12]":<14}  {"domain":<22}  {"NDCG":>6}  {"n_rel":>5}  {"in_pool":>7}  {"best_rank":>9}')
print('  ' + '-' * 70)
for qid, score in sorted_q[:30]:
    rels = set(qrels.get(qid, []))
    cands = best_sub_dr.get(qid, [])
    in_pool = sum(1 for d in cands[:100] if d in rels)
    best_rank = next((i+1 for i, d in enumerate(cands) if d in rels), None)
    domain = q_dom_val.get(qid, '?')
    print(f'  {qid[:12]:<14}  {domain:<22}  {score:>6.4f}  {len(rels):>5}  {in_pool:>7}/{len(rels)}  {str(best_rank) if best_rank else "NOT IN":>9}')

# ── 2. Oracle NDCG per domain ──────────────────────────────────────────────
print('\n── 2. Oracle NDCG@10 per domain (ceiling with perfect reranking) ─────────')
print(f'  {"Domain":<22}  {"n_q":>3}  {"current":>9}  {"oracle":>9}  {"gap":>7}  {"gap×n":>7}')
print('  ' + '-' * 62)

dom_to_qids = defaultdict(list)
for qid in val_qids:
    dom_to_qids[q_dom_val.get(qid, 'Unknown')].append(qid)

total_gap = 0
for domain in sorted(dom_to_qids.keys()):
    qids_dom = [q for q in dom_to_qids[domain] if q in qrels]
    if not qids_dom: continue
    curr_scores, oracle_scores = [], []
    for qid in qids_dom:
        rels    = set(qrels[qid])
        cands   = best_sub_dr.get(qid, [])[:100]
        ranked  = cands[:10]
        # Current DCG
        dcg  = sum(1.0/np.log2(i+2) for i,d in enumerate(ranked) if d in rels)
        idcg = sum(1.0/np.log2(i+2) for i in range(min(len(rels),10)))
        curr_scores.append(dcg/idcg if idcg > 0 else 0)
        # Oracle: best possible reranking of top-100
        oracle_ranked = [d for d in cands if d in rels][:10]
        oracle_dcg  = sum(1.0/np.log2(i+2) for i in range(len(oracle_ranked)))
        oracle_scores.append(oracle_dcg/idcg if idcg > 0 else 0)
    c_mean = np.mean(curr_scores)
    o_mean = np.mean(oracle_scores)
    gap = o_mean - c_mean
    total_gap += gap * len(qids_dom)
    print(f'  {domain:<22}  {len(qids_dom):>3}  {c_mean:>9.4f}  {o_mean:>9.4f}  {gap:>+7.4f}  {gap*len(qids_dom):>7.4f}')

print(f'\n  Total weighted gap (sum over queries): {total_gap:.4f}')
print(f'  → Fixing ranking within pool would add {total_gap/len([q for q in qrels if q in q_dom_val]):.4f} to avg NDCG')

# ── 3. Recall gap analysis ─────────────────────────────────────────────────
print('\n── 3. Recall gap analysis ────────────────────────────────────────────────')
recall_miss = {}
for qid, rels in qrels.items():
    rel_set = set(rels)
    if not rel_set: continue
    cands = best_sub_dr.get(qid, [])[:100]
    missing = rel_set - set(cands)
    recall_miss[qid] = len(missing)

total_rels  = sum(len(set(v)) for v in qrels.values())
total_found = sum(len([d for d in best_sub_dr.get(qid,[])[:100] if d in set(rels)]) for qid, rels in qrels.items())
total_miss  = total_rels - total_found

print(f'  Total relevant docs: {total_rels}')
print(f'  Found in top-100:    {total_found} ({100*total_found/total_rels:.1f}%)')
print(f'  Missing from pool:   {total_miss} ({100*total_miss/total_rels:.1f}%)')
print(f'  Queries with 0 missed: {sum(1 for v in recall_miss.values() if v==0)}/{len(recall_miss)}')
print(f'  Queries with ≥1 missed: {sum(1 for v in recall_miss.values() if v>=1)}')

# ── 4. CS-specific analysis ────────────────────────────────────────────────
print('\n── 4. Computer Science deep-dive ─────────────────────────────────────────')
cs_qids = [q for q in dom_to_qids['Computer Science'] if q in qrels]
print(f'  CS queries: {len(cs_qids)}')
print(f'\n  {"qid[:12]":<14}  {"NDCG":>6}  {"n_rel":>5}  {"in_pool":>7}  {"best_rank":>9}  {"oracle_rank10":>13}')
print('  ' + '-' * 66)
for qid in cs_qids:
    rels  = set(qrels.get(qid, []))
    cands = best_sub_dr.get(qid, [])[:100]
    score = q_ndcg.get(qid, 0.0)
    in_pool = sum(1 for d in cands if d in rels)
    best_rank = next((i+1 for i, d in enumerate(cands) if d in rels), None)
    oracle_rank10 = min(in_pool, 10)
    print(f'  {qid[:12]:<14}  {score:>6.4f}  {len(rels):>5}  {in_pool:>7}/{len(rels)}  {str(best_rank) if best_rank else "MISS":>9}  {oracle_rank10:>13}')

# Per-model rank for CS queries
print(f'\n  CS queries — per-model rank of first relevant doc:')
print(f'  {"qid[:12]":<14}  {"BM25L":>7}  {"BGE":>7}  {"MiniLM":>7}  {"SP2":>7}  {"TF-u":>7}  {"TF-b":>7}  {"Ens":>7}')
print('  ' + '-' * 72)
model_subs = {
    'BM25L': s_bm25l, 'BGE': s_bge, 'MiniLM': s_ml,
    'SPECTER2': s_sp, 'TF-IDF-uni': ht_tuni, 'TF-IDF-bi': ht_tbi
}
for qid in cs_qids:
    rels = set(qrels.get(qid, []))
    ens_rank = next((i+1 for i,d in enumerate(best_sub_dr.get(qid,[])[:100]) if d in rels), 999)
    ranks = []
    for mname in ['BM25L','BGE','MiniLM','SPECTER2','TF-IDF-uni','TF-IDF-bi']:
        r = next((i+1 for i,d in enumerate(model_subs[mname].get(qid,[])[:100]) if d in rels), 999)
        ranks.append(r)
    print(f'  {qid[:12]:<14}  {ranks[0]:>7}  {ranks[1]:>7}  {ranks[2]:>7}  {ranks[3]:>7}  {ranks[4]:>7}  {ranks[5]:>7}  {ens_rank:>7}')

# ── 5. What improvement per domain would get us to 0.80 ───────────────────
print('\n── 5. Gap to 0.80 ────────────────────────────────────────────────────────')
current_mean = np.mean(list(q_ndcg.values()))
target = 0.80
needed = target - current_mean
print(f'  Current NDCG@10 = {current_mean:.4f}')
print(f'  Target           = {target:.4f}')
print(f'  Gap to close     = {needed:.4f}')
print(f'  (Across {len(q_ndcg)} queries, need avg +{needed:.4f} per query)')

# If we perfect CS queries (NDCG=0.56 → 0.80):
cs_gain = sum(max(0, 0.80 - q_ndcg[q]) for q in cs_qids if q in q_ndcg)
print(f'\n  If CS queries went from 0.56 to 0.80:  +{cs_gain/len(q_ndcg):.4f} to overall')

# If we perfect Medicine queries (0.71 → 0.85):
med_qids = [q for q in dom_to_qids['Medicine'] if q in q_ndcg]
med_gain = sum(max(0, 0.85 - q_ndcg[q]) for q in med_qids)
print(f'  If Medicine went from 0.71 to 0.85:    +{med_gain/len(q_ndcg):.4f} to overall')

# If we perfect Biology queries (0.71 → 0.85):
bio_qids = [q for q in dom_to_qids['Biology'] if q in q_ndcg]
bio_gain = sum(max(0, 0.85 - q_ndcg[q]) for q in bio_qids)
print(f'  If Biology went from 0.71 to 0.85:     +{bio_gain/len(q_ndcg):.4f} to overall')

print(f'\n  CS+Medicine+Biology together:           +{(cs_gain+med_gain+bio_gain)/len(q_ndcg):.4f}')
print(f'  → Would put us at {current_mean + (cs_gain+med_gain+bio_gain)/len(q_ndcg):.4f}')

# ── 6. Cross-domain recall audit ──────────────────────────────────────────
print('\n── 6. Cross-domain relevant docs (in pool) ───────────────────────────────')
cross_domain = 0
same_domain  = 0
for qid, rels in qrels.items():
    qdom  = q_dom_val.get(qid, '')
    cands = best_sub_dr.get(qid, [])[:100]
    for d in cands:
        if d in set(rels):
            ddom = c_dom_map.get(d, '')
            if ddom == qdom:
                same_domain += 1
            else:
                cross_domain += 1

print(f'  Relevant docs in pool:')
print(f'    Same domain:  {same_domain} ({100*same_domain/(same_domain+cross_domain):.1f}%)')
print(f'    Cross domain: {cross_domain} ({100*cross_domain/(same_domain+cross_domain):.1f}%)')

# ── 7. Where do our zero-NDCG queries land? ──────────────────────────────
zero_ndcg_qids = [q for q, s in q_ndcg.items() if s == 0]
print(f'\n── 7. Zero-NDCG queries analysis ─────────────────────────────────────────')
print(f'  Count: {len(zero_ndcg_qids)}')
for qid in zero_ndcg_qids:
    rels  = set(qrels.get(qid, []))
    cands = best_sub_dr.get(qid, [])[:100]
    in_pool = sum(1 for d in cands if d in rels)
    best_rank = next((i+1 for i, d in enumerate(cands) if d in rels), None)
    domain = q_dom_val.get(qid, '?')
    # Check all 100 positions
    top20  = [d for i, d in enumerate(cands[:20]) if d in rels]
    print(f'  {qid[:12]}  {domain:<22}  n_rel={len(rels)}  in_pool={in_pool}  best_rank={best_rank}  top20_hits={len(top20)}')
