"""
Domain-aware scoring using the full confusion matrix as an additive component.

Current approach (hard sort):
  Sort candidates by (-domain_weight, original_rank)
  → effectively two tiers: same-domain first, then everything else
  → cross-domain weights like Biology→Chemistry=0.016 are treated as 0

New approach (additive blend):
  final_score = rrf_score + lambda * domain_weight
  → same-domain docs get   +lambda * 1.0
  → cross-domain cited docs get +lambda * 0.016 etc.
  → truly unrelated docs get +lambda * 0.0 (no change)
  → re-sort by final_score

We also print the full confusion matrix to see which domains benefit most.

Baseline: 0.7493 (hk5, k1k2, skip Business domain rerank)
"""
import json
import numpy as np
import pandas as pd

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

# Print full confusion matrix cross-domain weights
print('Cross-domain citation weights (non-zero off-diagonal):')
for qdom in sorted(dw):
    row = dw[qdom]
    cross = {cdom: w for cdom, w in row.items() if cdom != qdom and w > 0}
    if cross:
        print(f'  {qdom:<24} cites: {cross}')

# ── Helpers ────────────────────────────────────────────────────────────────
def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q/(nq+1e-10)) @ (c/(nc+1e-10)).T).astype(np.float32)

def top100(scores, qids):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-scores[i])[:100]
        sub[qid] = [corpus_ids[j] for j in idx]
    return sub

def rrf_fuse(sa, sb, k=5):
    out = np.zeros_like(sa)
    for i in range(len(sa)):
        ra = np.argsort(np.argsort(-sa[i])) + 1
        rb = np.argsort(np.argsort(-sb[i])) + 1
        out[i] = 1.0/(k+ra) + 1.0/(k+rb)
    return out

def rrf_with_scores(lists, qids, k=3, n=100):
    """RRF that returns both the ranked list AND the raw RRF scores."""
    sub = {}
    scores = {}
    for qid in qids:
        sc = {}
        for lst in lists:
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0/(k+rank)
        sorted_docs = sorted(sc, key=sc.get, reverse=True)[:n]
        sub[qid]    = sorted_docs
        scores[qid] = {doc: sc[doc] for doc in sorted_docs}
    return sub, scores

def rrf(lists, qids, k=3, n=100):
    sub, _ = rrf_with_scores(lists, qids, k=k, n=n)
    return sub

def ndcg(sub, k=10):
    sc = []
    for qid, rels in qrels.items():
        if qid not in sub: continue
        ranked  = sub[qid][:k]
        rel_set = set(rels) if isinstance(rels, list) else set()
        dcg  = sum(1.0/np.log2(i+2) for i,d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0/np.log2(i+2) for i in range(min(len(rel_set),k)))
        sc.append(dcg/idcg if idcg>0 else 0)
    return float(np.mean(sc))

def recall100(sub):
    vals = []
    for qid, rels in qrels.items():
        rel_set = set(rels)
        if not rel_set: continue
        vals.append(sum(1 for d in sub.get(qid,[])[:100] if d in rel_set)/len(rel_set))
    return float(np.mean(vals))

def domain_additive(rrf_scores_dict, lambda_=0.5, skip=frozenset({'Business'})):
    """
    Re-rank candidates using: final_score = rrf_score + lambda * domain_weight
    Uses the FULL confusion matrix including off-diagonal cross-domain weights.
    """
    out = {}
    for qid, scores in rrf_scores_dict.items():
        qdom = q_dom_val.get(qid, '')
        if qdom in skip:
            out[qid] = sorted(scores, key=scores.get, reverse=True)[:100]
            continue
        wr = dw.get(qdom, {})
        combined = {}
        for doc, rrf_sc in scores.items():
            cdom  = c_dom_map.get(doc, '')
            dw_sc = wr.get(cdom, 0.0)
            combined[doc] = rrf_sc + lambda_ * dw_sc
        out[qid] = sorted(combined, key=combined.get, reverse=True)[:100]
    return out

# ── Load scores ────────────────────────────────────────────────────────────
print('\nLoading scores...')
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

# ── Build ensemble ─────────────────────────────────────────────────────────
s_bm25l = top100(sc_bm25l, val_qids)
s_bge   = top100(sc_bge,   val_qids)
s_ml    = top100(sc_ml,    val_qids)
s_sp    = top100(sc_sp,    val_qids)
ht_tuni = top100(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids)
ht_tbi  = top100(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids)

inner = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)
outer, outer_scores = rrf_with_scores(
    [inner, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp], val_qids, k=2
)

# ── Baselines ──────────────────────────────────────────────────────────────
# Current hard-sort approach
def dr_hard(sub, skip=frozenset({'Business'})):
    out = {}
    for qid, cands in sub.items():
        qd = q_dom_val.get(qid, '')
        if qd in skip:
            out[qid] = cands[:100]; continue
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d,''), 0.0), rank, d)
                  for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _,_,d in scored[:100]]
    return out

base_plain  = outer
base_hard   = dr_hard(outer)
print(f'Baseline (no domain):      NDCG={ndcg(base_plain):.4f}  R@100={recall100(base_plain):.4f}')
print(f'Baseline (hard sort):      NDCG={ndcg(base_hard):.4f}  R@100={recall100(base_hard):.4f}')

# ── Lambda sweep: additive domain blend ───────────────────────────────────
print('\nAdditive domain blend (lambda sweep):')
print(f'{"lambda":<10} {"NDCG@10":>10}  {"R@100":>8}  {"vs hard-sort":>12}')
print('-' * 48)

best_ndcg   = ndcg(base_hard)
best_lambda = None
best_sub    = None

for lam in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]:
    sub = domain_additive(outer_scores, lambda_=lam)
    n   = ndcg(sub)
    r   = recall100(sub)
    delta = n - ndcg(base_hard)
    marker = ' <--' if n > best_ndcg else ''
    print(f'{lam:<10.2f} {n:>10.4f}  {r:>8.4f}  {delta:>+12.4f}{marker}')
    if n > best_ndcg:
        best_ndcg   = n
        best_lambda = lam
        best_sub    = sub

# ── Per-domain breakdown at best lambda ───────────────────────────────────
if best_sub is not None:
    print(f'\nBest lambda={best_lambda}  NDCG={best_ndcg:.4f}')
    print('\nPer-domain NDCG change (additive vs hard-sort):')
    for domain in sorted(set(q_dom_val.values())):
        qids_dom = [q for q in qrels if q_dom_val.get(q) == domain]
        if not qids_dom: continue
        n_hard = ndcg({q: base_hard.get(q,[]) for q in qids_dom})
        n_new  = ndcg({q: best_sub.get(q,[])  for q in qids_dom})
        has_cross = bool({cdom: w for cdom, w in dw.get(domain,{}).items()
                          if cdom != domain and w > 0})
        cross_tag = ' [has cross-domain weights]' if has_cross else ''
        print(f'  {domain:<24} hard={n_hard:.4f}  new={n_new:.4f}  Δ={n_new-n_hard:+.4f}{cross_tag}')
else:
    print('\nNo improvement over hard-sort baseline.')

print(f'\nFinal summary:')
print(f'  Plain ensemble:    {ndcg(base_plain):.4f}')
print(f'  Hard sort (current best): {ndcg(base_hard):.4f}')
print(f'  Additive blend (best):    {best_ndcg:.4f}  (lambda={best_lambda})')
