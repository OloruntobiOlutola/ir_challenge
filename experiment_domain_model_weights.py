"""
Per-domain model weighting in RRF.

Analysis of bottlenecked queries showed that for certain domains:
  - Philosophy: BGE is catastrophic (rank ~4013), TF-IDF is best
  - Computer Science: MiniLM often best but RRF consensus overrides it
  - Medicine: SPECTER2 sometimes 3x better than others

Strategy:
  1. Measure per-domain NDCG@10 for each individual model
  2. Build a domain→model performance table
  3. In the final RRF, weight models per domain:
     - More weight (lower k) to top-performing model for that domain
     - Optionally EXCLUDE models that are systematically bad per domain

We test:
  A) Baseline hard-sort (0.7493/0.7500)
  B) Per-domain k-weighted RRF: k_model = k_base / perf_weight
  C) Domain-model exclusion: drop models below threshold per domain
  D) Best 2 models per domain only (oracle upper bound variant)

Baseline: 0.7500 (hk5, k1k2, hard-sort domain rerank, skip Business)
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

# Group val queries by domain
dom_to_qids = defaultdict(list)
for qid in val_qids:
    dom_to_qids[q_dom_val.get(qid, 'Unknown')].append(qid)

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

def rrf_fuse(sa, sb, k=5):
    out = np.zeros_like(sa)
    for i in range(len(sa)):
        ra = np.argsort(np.argsort(-sa[i])) + 1
        rb = np.argsort(np.argsort(-sb[i])) + 1
        out[i] = 1.0 / (k + ra) + 1.0 / (k + rb)
    return out

def rrf_weighted(lists_ks, qids, n=100):
    """
    RRF with per-list k values.
    lists_ks: list of (ranked_list_dict, k) tuples
    Lower k = more influence for that list.
    """
    sub = {}
    for qid in qids:
        sc = {}
        for lst, k in lists_ks:
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (k + rank)
        sub[qid] = sorted(sc, key=sc.get, reverse=True)[:n]
    return sub

def rrf(lists, qids, k=3, n=100):
    return rrf_weighted([(lst, k) for lst in lists], qids, n=n)

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
    """NDCG restricted to a specific query list."""
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

# ── Build per-model top-100 lists ───────────────────────────────────────────
print('Building individual model retrievals...')
s_bm25l = top100(sc_bm25l, val_qids)
s_bge   = top100(sc_bge,   val_qids)
s_ml    = top100(sc_ml,    val_qids)
s_sp    = top100(sc_sp,    val_qids)
ht_tuni = top100(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids)
ht_tbi  = top100(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids)

# ── Baseline ────────────────────────────────────────────────────────────────
inner_base = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)
outer_base = rrf([inner_base, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp], val_qids, k=2)
base_dr    = dr(outer_base)
print(f'Baseline (hard-sort): NDCG={ndcg(base_dr):.4f}  R@100={recall100(outer_base):.4f}')

# ── Step 1: Per-domain, per-model NDCG@10 ─────────────────────────────────
print('\n── Per-domain NDCG@10 by model (with domain hard-sort rerank) ──────────')
models = {
    'BM25L':    s_bm25l,
    'BGE':      s_bge,
    'MiniLM':   s_ml,
    'SPECTER2': s_sp,
    'TF-IDF-uni': ht_tuni,
    'TF-IDF-bi':  ht_tbi,
}

# Also apply domain rerank to each individual model
def single_model_dr(model_sub):
    return dr(model_sub)

dom_model_ndcg = {}  # dom_model_ndcg[domain][model_name] = ndcg_score

all_domains = sorted(dom_to_qids.keys())
header = f'  {"Domain":<22}  {"n_q":>3} | ' + '  '.join(f'{m[:8]:>8}' for m in models)
print(header)
print('  ' + '-' * len(header))

for domain in all_domains:
    qids_dom = [q for q in dom_to_qids[domain] if q in qrels]
    if not qids_dom:
        continue
    dom_model_ndcg[domain] = {}
    scores_row = {}
    for mname, msub in models.items():
        msub_dr = single_model_dr({qid: msub.get(qid, []) for qid in qids_dom})
        score = ndcg_subset(msub_dr, qids_dom)
        dom_model_ndcg[domain][mname] = score
        scores_row[mname] = score
    score_str = '  '.join(f'{scores_row[m]:>8.4f}' for m in models)
    print(f'  {domain:<22}  {len(qids_dom):>3} | {score_str}')

# ── Step 2: Identify best/worst model per domain ──────────────────────────
print('\n── Best and worst model per domain ──────────────────────────────────────')
dom_rank = {}  # dom_rank[domain] = sorted [(model_name, ndcg)]
for domain in all_domains:
    if domain not in dom_model_ndcg:
        continue
    ranked = sorted(dom_model_ndcg[domain].items(), key=lambda x: -x[1])
    dom_rank[domain] = ranked
    best = ranked[0]
    worst = ranked[-1]
    print(f'  {domain:<22}  best={best[0]:<12} {best[1]:.4f}   worst={worst[0]:<12} {worst[1]:.4f}')

# ── Step 3: Per-domain k-weighted RRF ─────────────────────────────────────
print('\n── Experiment A: Per-domain k-weighted RRF ──────────────────────────────')
print('  Strategy: per-query, use model-specific k in RRF based on domain performance')
print('  (lower k = stronger influence; best model per domain gets k=1, others scaled up)')

def rrf_domain_weighted(lists_dict, qids, dom_weights, default_k=2, n=100):
    """
    lists_dict: {model_name: sub_dict}
    dom_weights: {domain: {model_name: k_value}}
    """
    sub = {}
    for qid in qids:
        qdom = q_dom_val.get(qid, '')
        kvals = dom_weights.get(qdom, {})
        sc = {}
        for mname, lst in lists_dict.items():
            k = kvals.get(mname, default_k)
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (k + rank)
        sub[qid] = sorted(sc, key=sc.get, reverse=True)[:n]
    return sub

# Build k values: best model per domain gets k=1, second gets k=2, rest get k=5
dom_k_weights = {}
for domain, ranked in dom_rank.items():
    dom_k_weights[domain] = {}
    k_vals = [1, 2, 3, 5, 7, 10]
    for i, (mname, _) in enumerate(ranked):
        dom_k_weights[domain][mname] = k_vals[min(i, len(k_vals)-1)]

# Flat model dict for domain-weighted RRF
flat_models = {
    'BM25L': s_bm25l, 'BGE': s_bge, 'MiniLM': s_ml,
    'SPECTER2': s_sp, 'TF-IDF-uni': ht_tuni, 'TF-IDF-bi': ht_tbi,
}

outer_dw = rrf_domain_weighted(flat_models, val_qids, dom_k_weights, default_k=2)
outer_dw_dr = dr(outer_dw)
print(f'  A1. Domain-k weighted (rank order k): NDCG={ndcg(outer_dw_dr):.4f}  R@100={recall100(outer_dw):.4f}')

# Softer version: best=1, second=2, rest=3 (similar to original but domain-aware)
dom_k_soft = {}
for domain, ranked in dom_rank.items():
    dom_k_soft[domain] = {}
    k_vals = [1, 2, 3, 3, 3, 3]
    for i, (mname, _) in enumerate(ranked):
        dom_k_soft[domain][mname] = k_vals[min(i, len(k_vals)-1)]

outer_dw2 = rrf_domain_weighted(flat_models, val_qids, dom_k_soft, default_k=2)
outer_dw2_dr = dr(outer_dw2)
print(f'  A2. Domain-k soft (1/2/3 for top/2nd/rest): NDCG={ndcg(outer_dw2_dr):.4f}  R@100={recall100(outer_dw2):.4f}')

# ── Step 4: Exclude worst model per domain ────────────────────────────────
print('\n── Experiment B: Exclude worst model per domain ─────────────────────────')
print('  Strategy: for each query, drop models whose per-domain NDCG < threshold * best')

EXCLUDE_THRESHOLD = 0.5  # drop model if its domain NDCG < 50% of best for that domain

def rrf_domain_exclude(lists_dict, qids, dom_model_ndcg, threshold=0.5, default_k=2, n=100):
    sub = {}
    for qid in qids:
        qdom = q_dom_val.get(qid, '')
        domain_scores = dom_model_ndcg.get(qdom, {})
        if domain_scores:
            best_score = max(domain_scores.values())
            min_score  = best_score * threshold
            active = {m: lst for m, lst in lists_dict.items()
                      if domain_scores.get(m, 0) >= min_score}
        else:
            active = lists_dict
        if not active:
            active = lists_dict  # fallback: use all
        sc = {}
        for mname, lst in active.items():
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (default_k + rank)
        sub[qid] = sorted(sc, key=sc.get, reverse=True)[:n]
    return sub

for thresh in [0.3, 0.5, 0.7, 0.8]:
    outer_ex = rrf_domain_exclude(flat_models, val_qids, dom_model_ndcg, threshold=thresh)
    outer_ex_dr = dr(outer_ex)
    print(f'  B. Exclude models <{thresh*100:.0f}% of best: NDCG={ndcg(outer_ex_dr):.4f}  R@100={recall100(outer_ex):.4f}')

# ── Step 5: Best-N models only per domain ─────────────────────────────────
print('\n── Experiment C: Top-N models only per domain ───────────────────────────')

def rrf_top_n_models(lists_dict, qids, dom_rank, top_n=3, k=2, n=100):
    sub = {}
    for qid in qids:
        qdom = q_dom_val.get(qid, '')
        ranked = dom_rank.get(qdom, [])
        if ranked:
            top_models = {m for m, _ in ranked[:top_n]}
            active = {m: lst for m, lst in lists_dict.items() if m in top_models}
        else:
            active = lists_dict
        if not active:
            active = lists_dict
        sc = {}
        for mname, lst in active.items():
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (k + rank)
        sub[qid] = sorted(sc, key=sc.get, reverse=True)[:n]
    return sub

for top_n in [2, 3, 4, 5]:
    outer_tn = rrf_top_n_models(flat_models, val_qids, dom_rank, top_n=top_n, k=2)
    outer_tn_dr = dr(outer_tn)
    print(f'  C. Top-{top_n} models per domain: NDCG={ndcg(outer_tn_dr):.4f}  R@100={recall100(outer_tn):.4f}')

# ── Step 6: Nested RRF with domain-aware inner grouping ───────────────────
print('\n── Experiment D: Domain-aware nested RRF ────────────────────────────────')
print('  Strategy: same nested structure but inner/outer k chosen per domain')

def rrf_nested_domain(lists_dict, qids, dom_rank, n=100):
    """
    Nested RRF where:
      - Inner: top-3 models for domain (k=1)
      - Outer: all models but inner pre-fused (k=2)
    """
    sub = {}
    for qid in qids:
        qdom = q_dom_val.get(qid, '')
        ranked = dom_rank.get(qdom, list(lists_dict.items()))
        if ranked:
            top3 = [m for m, _ in ranked[:3]]
            bottom = [m for m, _ in ranked[3:]]
        else:
            top3 = list(lists_dict.keys())[:3]
            bottom = []

        # Inner RRF of top-3 models
        inner_sc = {}
        for mname in top3:
            lst = lists_dict.get(mname, {})
            for rank, doc in enumerate(lst.get(qid, []), 1):
                inner_sc[doc] = inner_sc.get(doc, 0.0) + 1.0 / (1 + rank)
        inner_sorted = sorted(inner_sc, key=inner_sc.get, reverse=True)[:100]

        # Outer RRF: inner + each bottom model separately
        outer_sc = {}
        for rank, doc in enumerate(inner_sorted, 1):
            outer_sc[doc] = outer_sc.get(doc, 0.0) + 1.0 / (2 + rank)
        for mname in bottom:
            lst = lists_dict.get(mname, {})
            for rank, doc in enumerate(lst.get(qid, []), 1):
                outer_sc[doc] = outer_sc.get(doc, 0.0) + 1.0 / (2 + rank)

        sub[qid] = sorted(outer_sc, key=outer_sc.get, reverse=True)[:n]
    return sub

outer_nd = rrf_nested_domain(flat_models, val_qids, dom_rank)
outer_nd_dr = dr(outer_nd)
print(f'  D. Domain-aware nested RRF: NDCG={ndcg(outer_nd_dr):.4f}  R@100={recall100(outer_nd):.4f}')

# ── Step 7: Per-domain breakdown for best approach ─────────────────────────
print('\n── Per-domain breakdown: best approach vs baseline ──────────────────────')

# Find best from A1, A2, B(0.5), C(3), D
approaches = [
    ('Baseline', base_dr),
    ('A1. dom-k rank', outer_dw_dr),
    ('A2. dom-k soft', outer_dw2_dr),
    ('C3. top-3 only', dr(rrf_top_n_models(flat_models, val_qids, dom_rank, top_n=3, k=2))),
    ('D. nested dom',  outer_nd_dr),
]

best_name, best_sub = max(approaches[1:], key=lambda x: ndcg(x[1]))
print(f'  Winner: {best_name} (NDCG={ndcg(best_sub):.4f})')

print(f'\n  {"Domain":<22}  {"n_q":>3}  {"baseline":>10}  {best_name[:12]:>12}  {"delta":>7}')
print('  ' + '-' * 62)
for domain in all_domains:
    qids_dom = [q for q in dom_to_qids[domain] if q in qrels]
    if not qids_dom: continue
    n_base = ndcg_subset(base_dr,  qids_dom)
    n_best = ndcg_subset(best_sub, qids_dom)
    print(f'  {domain:<22}  {len(qids_dom):>3}  {n_base:>10.4f}  {n_best:>12.4f}  {n_best-n_base:>+7.4f}')

print(f'\n── Summary ──────────────────────────────────────────────────────────────')
print(f'  {"Approach":<30}  {"NDCG@10":>8}  {"R@100":>8}')
print('  ' + '-' * 52)
for aname, asub in approaches:
    r = recall100({qid: outer_base.get(qid, []) for qid in val_qids}) if aname == 'Baseline' else recall100(asub)
    print(f'  {aname:<30}  {ndcg(asub):>8.4f}  {r:>8.4f}')
