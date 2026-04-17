"""
Venue-based reranking experiment.

Key finding: 21.2% of relevant docs share the same venue as query paper.
This is a signal we haven't exploited at all.

Hypothesis: after RRF retrieval, prioritize docs from the same venue
(on top of the existing domain reranking).

Also tests:
- n_relevant-aware pool sizing (queries with more rels need bigger pool)
- Year proximity reranking (mild signal)
- Combined venue + domain reranking

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
c_venue   = c_df['venue'].fillna('').to_dict()
c_year    = c_df['year'].fillna(0).to_dict()
q_dom_val = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids if qid in q1_df.index}
q_venue_v = {qid: q1_df.loc[qid, 'venue']  if qid in q1_df.index else '' for qid in val_qids}
q_year_v  = {qid: q1_df.loc[qid, 'year']   if qid in q1_df.index else 0  for qid in val_qids}
q_nrel_v  = {qid: int(q1_df.loc[qid, 'n_relevant']) if qid in q1_df.index else 2 for qid in val_qids}
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

# Current best reranking
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

def dr_venue(sub, q_dom_map=q_dom_val, q_venue=q_venue_v,
             skip=frozenset({'Business'}), venue_weight=1.0):
    """
    Hard-sort domain rerank WITH venue bonus.
    Sort key: (-domain_weight, -venue_match, rank)
    venue_match = 1 if same venue, 0 otherwise
    """
    out = {}
    for qid, cands in sub.items():
        qd   = q_dom_map.get(qid, '')
        qv   = q_venue.get(qid, '')
        if qd in skip:
            out[qid] = cands[:100]; continue
        wr = dw.get(qd, {})
        scored = []
        for rank, d in enumerate(cands):
            dom_w = wr.get(c_dom_map.get(d, ''), 0.0)
            venue_m = 1 if (qv and c_venue.get(d, '') == qv) else 0
            scored.append((-dom_w - venue_weight * venue_m, rank, d))
        scored.sort()
        out[qid] = [d for _, _, d in scored[:100]]
    return out

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

# ── Build RRF pool ─────────────────────────────────────────────────────────
masks = [domain_mask(q_dom_val.get(qid, ''), min_pool=300) for qid in val_qids]
s_bm25l = top_n(sc_bm25l, val_qids);  s_bge = top_n(sc_bge, val_qids)
s_ml    = top_n(sc_ml,    val_qids);  s_sp  = top_n(sc_sp,  val_qids)
ht_tuni = top_n(sc_ht_tuni, val_qids);  ht_tbi = top_n(sc_ht_tbi, val_qids)
s_bm25l_f = top_n_filtered(sc_bm25l,   val_qids, masks)
ht_tuni_f = top_n_filtered(sc_ht_tuni, val_qids, masks)
ht_tbi_f  = top_n_filtered(sc_ht_tbi,  val_qids, masks)

flat_unf = {'BM25L': s_bm25l,'BGE': s_bge,'MiniLM': s_ml,'SPECTER2': s_sp,'TF-IDF-uni': ht_tuni,'TF-IDF-bi': ht_tbi}
flat_sf  = {'BM25L': s_bm25l_f,'TF-IDF-uni': ht_tuni_f,'TF-IDF-bi': ht_tbi_f}

best_rrf  = rrf_nested_exclude_sf(flat_unf, flat_sf, val_qids, q_dom_val, DOM_MODEL_NDCG, threshold=0.70)
best_dr   = dr(best_rrf)
print(f'Baseline (nested excl + sf + domain-sort): NDCG={ndcg(best_dr):.4f}  R@100={recall100(best_rrf):.4f}')

# ── Venue analysis ─────────────────────────────────────────────────────────
print('\n── Venue analysis ───────────────────────────────────────────────────────')
# What fraction of relevant docs in pool are same-venue?
same_v_in_pool = 0; same_v_total = 0; diff_v_total = 0
for qid, rels in qrels.items():
    qv    = q_venue_v.get(qid, '')
    cands = best_rrf.get(qid, [])[:100]
    for d in rels:
        dv = c_venue.get(d, '')
        if d in set(cands):
            if dv == qv and qv:
                same_v_in_pool += 1
        if dv == qv and qv:
            same_v_total += 1
        elif d in set(cands):
            diff_v_total += 1

print(f'  Relevant same-venue docs in pool: {same_v_in_pool} / (total same-venue: {same_v_total})')

# Where are same-venue relevant docs ranked in the current pool?
ranks_same = []; ranks_diff = []
for qid, rels in qrels.items():
    qv    = q_venue_v.get(qid, '')
    cands = best_dr.get(qid, [])[:100]
    cand_rank = {d: i+1 for i, d in enumerate(cands)}
    for d in rels:
        dv = c_venue.get(d, '')
        if d in cand_rank:
            if dv == qv and qv:
                ranks_same.append(cand_rank[d])
            else:
                ranks_diff.append(cand_rank[d])

print(f'  Avg rank of same-venue relevant: {np.mean(ranks_same):.1f}  (n={len(ranks_same)})')
print(f'  Avg rank of diff-venue relevant: {np.mean(ranks_diff):.1f}  (n={len(ranks_diff)})')

# ── Venue-boosted reranking sweep ─────────────────────────────────────────
print('\n── Venue-boosted reranking sweep ────────────────────────────────────────')
print(f'  {"venue_weight":>14}  {"NDCG@10":>8}  {"R@100":>8}  {"Δ":>7}')
print('  ' + '-' * 44)

base_n = ndcg(best_dr)
best_venue_n = base_n
best_venue_w = None
best_venue_sub = None

for w in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
    sub_v = dr_venue(best_rrf, venue_weight=w)
    n_v = ndcg(sub_v)
    r_v = recall100(best_rrf)
    delta = n_v - base_n
    marker = ' <--' if n_v > best_venue_n else ''
    print(f'  {w:>14.1f}  {n_v:>8.4f}  {r_v:>8.4f}  {delta:>+7.4f}{marker}')
    if n_v > best_venue_n:
        best_venue_n = n_v
        best_venue_w = w
        best_venue_sub = sub_v

# ── Year-proximity reranking ───────────────────────────────────────────────
print('\n── Year-proximity reranking ─────────────────────────────────────────────')
def dr_year(sub, year_window=5, year_weight=0.05, q_dom_map=q_dom_val,
            skip=frozenset({'Business'})):
    """Soft year proximity boost alongside domain hard-sort."""
    out = {}
    for qid, cands in sub.items():
        qd  = q_dom_map.get(qid, '')
        qy  = q_year_v.get(qid, 0)
        if qd in skip:
            out[qid] = cands[:100]; continue
        wr = dw.get(qd, {})
        scored = []
        for rank, d in enumerate(cands):
            dom_w = wr.get(c_dom_map.get(d, ''), 0.0)
            dy    = c_year.get(d, 0)
            # Year proximity: score = 1 if within window, decays linearly beyond
            year_diff = abs(int(dy) - int(qy)) if dy and qy else 999
            year_sc = max(0, 1.0 - year_diff / year_window) if year_window > 0 else 0
            scored.append((-dom_w - year_weight * year_sc, rank, d))
        scored.sort()
        out[qid] = [d for _, _, d in scored[:100]]
    return out

for yw, ywin in [(0.05, 3), (0.1, 5), (0.2, 5), (0.3, 10)]:
    sub_y = dr_year(best_rrf, year_window=ywin, year_weight=yw)
    n_y = ndcg(sub_y)
    print(f'  year_weight={yw}  window={ywin}: NDCG={n_y:.4f}  Δ={n_y-base_n:+.4f}')

# ── Combined venue + domain + year ────────────────────────────────────────
print('\n── Combined venue + year reranking ──────────────────────────────────────')
def dr_combined(sub, venue_weight=0.5, year_weight=0.1, year_window=5,
                q_dom_map=q_dom_val, skip=frozenset({'Business'})):
    out = {}
    for qid, cands in sub.items():
        qd = q_dom_map.get(qid, '')
        qv = q_venue_v.get(qid, '')
        qy = q_year_v.get(qid, 0)
        if qd in skip:
            out[qid] = cands[:100]; continue
        wr = dw.get(qd, {})
        scored = []
        for rank, d in enumerate(cands):
            dom_w   = wr.get(c_dom_map.get(d, ''), 0.0)
            venue_m = 1 if (qv and c_venue.get(d,'') == qv) else 0
            dy      = c_year.get(d, 0)
            year_diff = abs(int(dy) - int(qy)) if dy and qy else 999
            year_sc = max(0, 1.0 - year_diff / max(year_window, 1))
            sort_key = -(dom_w + venue_weight * venue_m + year_weight * year_sc)
            scored.append((sort_key, rank, d))
        scored.sort()
        out[qid] = [d for _, _, d in scored[:100]]
    return out

best_combo = None
best_combo_n = base_n
for vw in [0.3, 0.5, 1.0]:
    for yw in [0.0, 0.1, 0.2]:
        sub_c = dr_combined(best_rrf, venue_weight=vw, year_weight=yw)
        n_c = ndcg(sub_c)
        if n_c > best_combo_n or (vw == 0.3 and yw == 0.0):
            marker = ' <--' if n_c > best_combo_n else ''
            print(f'  venue={vw}  year={yw}: NDCG={n_c:.4f}  Δ={n_c-base_n:+.4f}{marker}')
            if n_c > best_combo_n:
                best_combo_n = n_c
                best_combo = sub_c

# ── Per-domain breakdown for venue boost ──────────────────────────────────
print('\n── Per-domain breakdown (venue boost vs baseline) ───────────────────────')
if best_venue_sub is not None:
    print(f'  Best venue_weight={best_venue_w}  NDCG={best_venue_n:.4f}')
    dom_to_qids = defaultdict(list)
    for qid in val_qids:
        dom_to_qids[q_dom_val.get(qid,'')].append(qid)
    print(f'  {"Domain":<22}  {"n_q":>3}  {"baseline":>10}  {"venue":>10}  {"delta":>7}')
    print('  ' + '-' * 58)
    for domain in sorted(dom_to_qids.keys()):
        qids_dom = [q for q in dom_to_qids[domain] if q in qrels]
        if not qids_dom: continue
        n_base = ndcg_subset(best_dr, qids_dom)
        n_ven  = ndcg_subset(best_venue_sub, qids_dom)
        # Count same-venue relevant pairs
        n_sv_pairs = sum(1 for q in qids_dom for d in qrels.get(q,[])
                         if c_venue.get(d,'') == q_venue_v.get(q,'') and q_venue_v.get(q,''))
        sv_tag = f'  ({n_sv_pairs} sv-pairs)' if n_sv_pairs > 0 else ''
        print(f'  {domain:<22}  {len(qids_dom):>3}  {n_base:>10.4f}  {n_ven:>10.4f}  {n_ven-n_base:>+7.4f}{sv_tag}')
else:
    print('  No improvement from venue boost.')

# ── n_relevant-aware pool sizing ──────────────────────────────────────────
print('\n── n_relevant-aware pool sizing ─────────────────────────────────────────')
# Idea: for queries with many relevant docs (say ≥10), expand pool to 200
# to ensure we capture more of them
large_nrel_qids = [q for q in val_qids if q_nrel_v.get(q, 0) >= 10]
small_nrel_qids = [q for q in val_qids if q_nrel_v.get(q, 0) < 10]
print(f'  Queries with n_relevant >= 10: {len(large_nrel_qids)}')
print(f'  Queries with n_relevant <  10: {len(small_nrel_qids)}')

# Recall for large-nrel queries with current 100-pool vs 200-pool
s_bm25l_200 = top_n(sc_bm25l, val_qids, n=200)
s_bge_200   = top_n(sc_bge,   val_qids, n=200)
s_ml_200    = top_n(sc_ml,    val_qids, n=200)
s_sp_200    = top_n(sc_sp,    val_qids, n=200)
ht_tuni_200 = top_n(sc_ht_tuni, val_qids, n=200)
ht_tbi_200  = top_n(sc_ht_tbi,  val_qids, n=200)
s_bm25l_200f = top_n_filtered(sc_bm25l,   val_qids, masks, n=200)
ht_tuni_200f = top_n_filtered(sc_ht_tuni, val_qids, masks, n=200)
ht_tbi_200f  = top_n_filtered(sc_ht_tbi,  val_qids, masks, n=200)

flat_unf_200 = {'BM25L': s_bm25l_200,'BGE': s_bge_200,'MiniLM': s_ml_200,
                'SPECTER2': s_sp_200,'TF-IDF-uni': ht_tuni_200,'TF-IDF-bi': ht_tbi_200}
flat_sf_200  = {'BM25L': s_bm25l_200f,'TF-IDF-uni': ht_tuni_200f,'TF-IDF-bi': ht_tbi_200f}

best_rrf_200 = rrf_nested_exclude_sf(flat_unf_200, flat_sf_200, val_qids, q_dom_val, DOM_MODEL_NDCG, threshold=0.70, n=200)

# Adaptive: use 200-pool for large-nrel, 100-pool for small-nrel
adaptive_sub = {}
for qid in val_qids:
    n_rel = q_nrel_v.get(qid, 2)
    if n_rel >= 10:
        cands = best_rrf_200.get(qid, [])[:100]
    else:
        cands = best_rrf.get(qid, [])[:100]
    adaptive_sub[qid] = cands

adaptive_dr = dr(adaptive_sub)
print(f'\n  Adaptive pool (200 for n_rel≥10, 100 otherwise): NDCG={ndcg(adaptive_dr):.4f}  R@100={recall100(adaptive_sub):.4f}')

# ── Final summary ──────────────────────────────────────────────────────────
print('\n── Final summary ─────────────────────────────────────────────────────────')
configs = [
    ('Baseline', ndcg(best_dr), recall100(best_rrf)),
    (f'Venue boost w={best_venue_w}', best_venue_n, recall100(best_rrf)),
    ('Adaptive pool', ndcg(adaptive_dr), recall100(adaptive_sub)),
]
if best_combo is not None:
    configs.append(('Best venue+year combo', best_combo_n, recall100(best_rrf)))
configs.sort(key=lambda x: -x[1])
for name, n_v, r_v in configs:
    print(f'  {name:<35}  NDCG={n_v:.4f}  R@100={r_v:.4f}  Δ={n_v-ndcg(best_dr):+.4f}')
