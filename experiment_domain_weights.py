"""
Per-domain ensemble weights experiment.
Baseline val NDCG@10 = 0.7493 (global best config).

Strategy:
  For each domain, find the best subset of retrievers and RRF k value,
  then build a per-domain submission and compare to the global baseline.

  Domains with <4 queries are grouped as 'Other' during optimization
  but each query still gets the per-domain assignment at inference time.
"""
import json
import numpy as np
import pandas as pd
from itertools import combinations

# ── IDs & qrels ────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
q_df       = pd.read_parquet('data/queries.parquet').set_index('doc_id')
held_qids  = q_df.index.tolist()
qrels      = json.load(open('data/qrels_1.json'))

# ── Helpers ────────────────────────────────────────────────────────────────
def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q / (nq + 1e-10)) @ (c / (nc + 1e-10)).T).astype(np.float32)

def top100(scores, qids, cids=corpus_ids):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-scores[i])[:100]
        sub[qid] = [cids[j] for j in idx]
    return sub

def rrf_fuse(sa, sb, k=5):
    out = np.zeros_like(sa)
    for i in range(len(sa)):
        ra = np.argsort(np.argsort(-sa[i])) + 1
        rb = np.argsort(np.argsort(-sb[i])) + 1
        out[i] = 1.0 / (k + ra) + 1.0 / (k + rb)
    return out

def rrf(lists, qids, k=3):
    sub = {}
    for qid in qids:
        sc = {}
        for lst in lists:
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0 / (k + rank)
        sub[qid] = sorted(sc, key=sc.get, reverse=True)[:100]
    return sub

def ndcg_queries(sub, query_list, k=10):
    """NDCG@10 for a specific list of queries."""
    sc = []
    for qid in query_list:
        rels = qrels.get(qid, [])
        if qid not in sub:
            sc.append(0.0)
            continue
        ranked  = sub[qid][:k]
        rel_set = set(rels) if isinstance(rels, list) else set()
        dcg  = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
        sc.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(sc)) if sc else 0.0

def ndcg(sub, k=10):
    return ndcg_queries(sub, list(qrels.keys()), k)

# ── Domain rerank (skip Business) ─────────────────────────────────────────
dw_df = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw    = dw_df.to_dict(orient='index')
c_df2  = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df  = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_dom_map  = c_df2['domain'].to_dict()
q_dom_val  = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids  if qid in q1_df.index}
q_dom_held = {qid: q_df.loc[qid,  'domain'] for qid in held_qids if qid in q_df.index}

SKIP = {'Business'}
def dr(sub, qd_map):
    out = {}
    for qid, cands in sub.items():
        qd = qd_map.get(qid, '')
        if qd in SKIP:
            out[qid] = cands[:100]
            continue
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d, ''), 0.0), rank, d)
                  for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _, _, d in scored[:100]]
    return out

# ── Load all val scores ────────────────────────────────────────────────────
print('Loading scores...')
sc_bm25l = np.load('submissions/scores_bm25l_ft.npy')
sc_tuni  = np.load('submissions/scores_tfidf_uni_ft.npy')
sc_tbi   = np.load('submissions/scores_tfidf_bi_ft.npy')

bge_qv = np.load('submissions/bge_large_query_emb.npy')
bge_c  = np.load('submissions/bge_large_corpus_emb.npy')
sc_bge = cosine(bge_qv, bge_c)

ml_qv = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy')
ml_cv = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy')
sc_ml = (ml_qv @ ml_cv.T).astype(np.float32)

sp_q_ta = np.load('specter_prox_embed/queries_1_embeddings.npy')
sp_ft_c = np.load('submissions/specter2_corpus_emb_ft.npy')
sc_sp   = cosine(sp_q_ta, sp_ft_c)

# Pre-built top-100 lists (global)
s_bm25l = top100(sc_bm25l, val_qids)
s_bge   = top100(sc_bge,   val_qids)
s_ml    = top100(sc_ml,    val_qids)
s_sp    = top100(sc_sp,    val_qids)

# Hybrid fused lists (hk=5)
ht_tuni = top100(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids)
ht_tbi  = top100(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids)

# Named retrievers (for grid search)
retriever_lists = {
    'bm25l':   s_bm25l,
    'bge':     s_bge,
    'ml':      s_ml,
    'sp':      s_sp,
    'ht_tuni': ht_tuni,
    'ht_tbi':  ht_tbi,
}
retriever_names = list(retriever_lists.keys())

# ── Baseline (global best) ─────────────────────────────────────────────────
inner_base = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)
outer_base = rrf([inner_base, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp], val_qids, k=2)
final_base = dr(outer_base, q_dom_val)
print(f'\nBaseline NDCG@10: {ndcg(final_base):.4f}')

# ── Group queries by domain ────────────────────────────────────────────────
MIN_QUERIES = 4   # domains below this are treated as 'Other' during optimization

domain_groups = {}
for qid in val_qids:
    dom = q_dom_val.get(qid, 'Unknown')
    domain_groups.setdefault(dom, []).append(qid)

# Print per-domain baseline
print('\nPer-domain baseline NDCG@10:')
for dom in sorted(domain_groups, key=lambda d: -len(domain_groups[d])):
    qlist = domain_groups[dom]
    score = ndcg_queries(final_base, qlist)
    print(f'  {dom:<25} n={len(qlist):2d}  NDCG={score:.4f}')

# Collapse small domains
opt_groups = {}
other_qids = []
for dom, qlist in domain_groups.items():
    if len(qlist) >= MIN_QUERIES:
        opt_groups[dom] = qlist
    else:
        other_qids.extend(qlist)
if other_qids:
    opt_groups['Other'] = other_qids

# ── Per-domain retriever selection ─────────────────────────────────────────
# For each domain, try all subsets of ≥3 retrievers as outer list inputs
# plus k ∈ {1, 2, 3}.  Pick the combo with highest domain NDCG.

K_VALUES   = [1, 2, 3]
MIN_SUBSET = 3

print('\n--- Per-domain optimization ---')
best_config = {}   # domain → (k, retriever_subset_names, ndcg)

for dom, qlist in opt_groups.items():
    base_score = ndcg_queries(final_base, qlist)
    best = (None, None, base_score)  # start at baseline

    # inner is always bm25l+bge+ml at k=1 (works well globally)
    inner_d = {qid: inner_base[qid] for qid in qlist}

    # Try subsets of outer retrievers of size 3-6
    outer_candidates = ['inner', 'bge', 'bm25l', 'sp', 'ht_tuni', 'ht_tbi']
    # 'inner' must always be included; vary the rest
    rest = [r for r in outer_candidates if r != 'inner']

    for subset_size in range(2, len(rest) + 1):
        for combo in combinations(rest, subset_size):
            chosen = ['inner'] + list(combo)
            for k in K_VALUES:
                lists_d = []
                for name in chosen:
                    if name == 'inner':
                        lists_d.append(inner_d)
                    else:
                        lists_d.append(retriever_lists[name])
                outer_d = rrf(lists_d, qlist, k=k)
                outer_d_dr = dr(outer_d, {q: q_dom_val[q] for q in qlist})
                score = ndcg_queries(outer_d_dr, qlist)
                if score > best[2]:
                    best = (k, chosen, score)

    best_config[dom] = best
    delta = best[2] - base_score
    marker = '  *** IMPROVED ***' if delta > 0.001 else ''
    print(f'  {dom:<25}  base={base_score:.4f}  best={best[2]:.4f}  '
          f'Δ={delta:+.4f}  k={best[0]}  {best[1]}{marker}')

# ── Build per-domain submission ────────────────────────────────────────────
print('\n--- Building per-domain submission ---')

# Map small-domain queries to 'Other' config
def get_domain_config(qid):
    dom = q_dom_val.get(qid, 'Unknown')
    if dom in best_config:
        return dom, best_config[dom]
    if 'Other' in best_config:
        return 'Other', best_config['Other']
    return None, (2, ['inner', 'bge', 'bm25l', 'sp', 'ht_tuni', 'ht_tbi'], 0)

# Build per-query submission: apply each query's domain-specific config
per_domain_sub = {}
for qid in val_qids:
    dom, (k, chosen, _) = get_domain_config(qid)
    lists_q = []
    for name in chosen:
        if name == 'inner':
            lists_q.append({qid: inner_base[qid]})
        else:
            lists_q.append({qid: retriever_lists[name][qid]})
    merged = rrf(lists_q, [qid], k=k)
    per_domain_sub[qid] = merged[qid]

per_domain_sub_dr = dr(per_domain_sub, q_dom_val)

pd_score = ndcg(per_domain_sub_dr)
print(f'\nBaseline NDCG@10:          {ndcg(final_base):.4f}')
print(f'Per-domain NDCG@10:        {pd_score:.4f}  (Δ={pd_score - ndcg(final_base):+.4f})')

# ── Per-domain breakdown comparison ───────────────────────────────────────
print('\nPer-domain comparison (baseline vs per-domain):')
print(f'{"Domain":<25} {"n":>3}  {"Baseline":>9}  {"PerDomain":>9}  {"Δ":>7}')
print('-' * 60)
for dom in sorted(domain_groups, key=lambda d: -len(domain_groups[d])):
    qlist = domain_groups[dom]
    b = ndcg_queries(final_base, qlist)
    p = ndcg_queries(per_domain_sub_dr, qlist)
    print(f'{dom:<25} {len(qlist):>3}  {b:>9.4f}  {p:>9.4f}  {p-b:>+7.4f}')

# ── Build held-out submission with per-domain configs ─────────────────────
print('\n--- Building held-out submission ---')

sc_bm25l_h = np.load('submissions_heldout/scores_bm25l_ft.npy')
sc_tuni_h  = np.load('submissions_heldout/scores_tfidf_uni_ft.npy')
sc_tbi_h   = np.load('submissions_heldout/scores_tfidf_bi_ft.npy')
sc_bge_h   = np.load('submissions_heldout/scores_bge_dense_corrected.npy')

ml_qh   = np.load('data/embeddings/new_queries_minilm/query_embeddings.npy')
sc_ml_h = (ml_qh @ ml_cv.T).astype(np.float32)

sp_q_h  = np.load('specter_prox_embed/queries_embeddings.npy')
sc_sp_h = cosine(sp_q_h, sp_ft_c)

s_bm25l_h = top100(sc_bm25l_h, held_qids)
s_bge_h   = top100(sc_bge_h,   held_qids)
s_ml_h    = top100(sc_ml_h,    held_qids)
s_sp_h    = top100(sc_sp_h,    held_qids)
ht_tuni_h = top100(rrf_fuse(sc_tuni_h, sc_sp_h, k=5), held_qids)
ht_tbi_h  = top100(rrf_fuse(sc_tbi_h,  sc_sp_h, k=5), held_qids)

inner_h = rrf([s_bm25l_h, s_bge_h, s_ml_h], held_qids, k=1)

retriever_lists_h = {
    'bm25l':   s_bm25l_h,
    'bge':     s_bge_h,
    'ml':      s_ml_h,
    'sp':      s_sp_h,
    'ht_tuni': ht_tuni_h,
    'ht_tbi':  ht_tbi_h,
}

def get_held_config(qid):
    dom = q_dom_held.get(qid, 'Unknown')
    if dom in best_config:
        return best_config[dom]
    if 'Other' in best_config:
        return best_config['Other']
    return (2, ['inner', 'bge', 'bm25l', 'sp', 'ht_tuni', 'ht_tbi'], 0)

per_domain_held = {}
for qid in held_qids:
    k, chosen, _ = get_held_config(qid)
    lists_q = []
    for name in chosen:
        if name == 'inner':
            lists_q.append({qid: inner_h[qid]})
        else:
            lists_q.append({qid: retriever_lists_h[name][qid]})
    merged = rrf(lists_q, [qid], k=k)
    per_domain_held[qid] = merged[qid]

per_domain_held_dr = dr(per_domain_held, q_dom_held)

out_path = 'submissions_heldout/submission_per_domain.json'
with open(out_path, 'w') as f:
    json.dump(per_domain_held_dr, f)
print(f'Saved: {out_path}')
print(f'Val NDCG@10: {pd_score:.4f}  (Δ vs baseline: {pd_score - ndcg(final_base):+.4f})')
