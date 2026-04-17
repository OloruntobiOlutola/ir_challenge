"""
Domain-filtered retrieval: for each query, restrict the scored corpus to
documents whose domain has a non-zero confusion-matrix weight for that
query's domain — then retrieve top-100 from that filtered pool.

This is applied per-model BEFORE scoring, so the ranking is over a
tighter, domain-relevant candidate set.

Three filter strategies tested:
  strict  : only weight > 0 (same-domain + any cited cross-domain)
  soft    : weight > 0 OR fallback to full corpus if pool < MIN_POOL
  expand  : same-domain + next-best K% of corpus by domain size

Baseline: 0.7493 (hk5, k1k2, hard-sort domain rerank, skip Business)
"""
import json
import numpy as np
import pandas as pd

# ── Load ───────────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
qrels      = json.load(open('data/qrels_1.json'))

dw_df     = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw        = dw_df.to_dict(orient='index')
c_df      = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df     = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_dom_map = c_df['domain'].to_dict()
q_dom_val = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids if qid in q1_df.index}

# Precompute corpus domain array (same order as corpus_ids)
c_domains = np.array([c_dom_map.get(cid, '') for cid in corpus_ids])
corpus_arr = np.array(corpus_ids)

# Per-domain index arrays for fast masking
from collections import defaultdict
dom_to_idx = defaultdict(list)
for j, d in enumerate(c_domains):
    dom_to_idx[d].append(j)
dom_to_idx = {d: np.array(idxs) for d, idxs in dom_to_idx.items()}

MIN_POOL = 300   # fall back to full corpus if filtered pool < this

def domain_mask(qdom, threshold=0.0, min_pool=MIN_POOL):
    """
    Returns boolean mask over corpus_ids.
    Includes any corpus doc whose domain has weight > threshold for qdom.
    Falls back to all-True if fewer than min_pool docs pass.
    """
    row = dw.get(qdom, {})
    mask = np.array([row.get(d, 0.0) > threshold for d in c_domains])
    if mask.sum() < min_pool:
        return np.ones(len(corpus_ids), dtype=bool)   # fallback: full corpus
    return mask

# ── Helpers ────────────────────────────────────────────────────────────────
def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q / (nq + 1e-10)) @ (c / (nc + 1e-10)).T).astype(np.float32)

def top100_filtered(score_matrix, qids, masks):
    """
    For each query i, zero-out (suppress) scores of docs outside the mask,
    then return top-100.  masks[i] is the boolean mask for query i.
    """
    sub = {}
    for i, qid in enumerate(qids):
        sc = score_matrix[i].copy()
        sc[~masks[i]] = -1e9
        idx = np.argsort(-sc)[:100]
        sub[qid] = corpus_arr[idx].tolist()
    return sub

def top100(score_matrix, qids):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-score_matrix[i])[:100]
        sub[qid] = corpus_arr[idx].tolist()
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

# ── Baseline (no filter, with hard-sort domain rerank) ─────────────────────
print('Computing baseline...')
s_bm25l_base = top100(sc_bm25l, val_qids)
s_bge_base   = top100(sc_bge,   val_qids)
s_ml_base    = top100(sc_ml,    val_qids)
s_sp_base    = top100(sc_sp,    val_qids)
ht_tuni_base = top100(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids)
ht_tbi_base  = top100(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids)
inner_base   = rrf([s_bm25l_base, s_bge_base, s_ml_base], val_qids, k=1)
outer_base   = rrf([inner_base, s_bge_base, s_bm25l_base, ht_tuni_base,
                    ht_tbi_base, s_sp_base], val_qids, k=2)
base_dr      = dr(outer_base)
print(f'Baseline (no filter, hard-sort): NDCG={ndcg(base_dr):.4f}  R@100={recall100(outer_base):.4f}')

# ── Build per-query domain masks ───────────────────────────────────────────
print('\nBuilding domain masks...')
masks = {}
pool_sizes = {}
for qid in val_qids:
    qdom = q_dom_val.get(qid, '')
    m = domain_mask(qdom, threshold=0.0, min_pool=MIN_POOL)
    masks[qid]      = m
    pool_sizes[qid] = int(m.sum())

mask_list = [masks[qid] for qid in val_qids]

# Show pool size summary
sizes = list(pool_sizes.values())
print(f'  Pool sizes — min:{min(sizes)}  max:{max(sizes)}  mean:{sum(sizes)/len(sizes):.0f}')
full_corpus = sum(1 for s in sizes if s == len(corpus_ids))
print(f'  Queries using full corpus (fallback): {full_corpus}/{len(val_qids)}')

# ── Domain-filtered retrieval ──────────────────────────────────────────────
print('\nDomain-filtered retrieval (filter applied to each model):')

s_bm25l_f = top100_filtered(sc_bm25l, val_qids, mask_list)
s_bge_f   = top100_filtered(sc_bge,   val_qids, mask_list)
s_ml_f    = top100_filtered(sc_ml,    val_qids, mask_list)
s_sp_f    = top100_filtered(sc_sp,    val_qids, mask_list)

# For hybrid fused (rrf_fuse operates on score matrices — apply mask after fusing)
sc_ht_tuni = rrf_fuse(sc_tuni, sc_sp, k=5)
sc_ht_tbi  = rrf_fuse(sc_tbi,  sc_sp, k=5)
ht_tuni_f  = top100_filtered(sc_ht_tuni, val_qids, mask_list)
ht_tbi_f   = top100_filtered(sc_ht_tbi,  val_qids, mask_list)

inner_f = rrf([s_bm25l_f, s_bge_f, s_ml_f], val_qids, k=1)
outer_f = rrf([inner_f, s_bge_f, s_bm25l_f, ht_tuni_f, ht_tbi_f, s_sp_f], val_qids, k=2)

# Config A: filter only, no post-rerank
print(f'  A. filter only (no post-rerank):          NDCG={ndcg(outer_f):.4f}  R@100={recall100(outer_f):.4f}')

# Config B: filter + post hard-sort rerank
outer_f_dr = dr(outer_f)
print(f'  B. filter + hard-sort rerank:             NDCG={ndcg(outer_f_dr):.4f}  R@100={recall100(outer_f):.4f}')

# Config C: filter only sparse, leave dense unfiltered
s_bm25l_f2 = top100_filtered(sc_bm25l, val_qids, mask_list)
ht_tuni_f2 = top100_filtered(sc_ht_tuni, val_qids, mask_list)
ht_tbi_f2  = top100_filtered(sc_ht_tbi,  val_qids, mask_list)
inner_c    = rrf([s_bm25l_f2, s_bge_base, s_ml_base], val_qids, k=1)
outer_c    = rrf([inner_c, s_bge_base, s_bm25l_f2, ht_tuni_f2, ht_tbi_f2,
                  s_sp_base], val_qids, k=2)
print(f'  C. sparse filtered + dense unfiltered:    NDCG={ndcg(dr(outer_c)):.4f}  R@100={recall100(outer_c):.4f}')

# Config D: filter only dense, leave sparse unfiltered
s_bge_f2  = top100_filtered(sc_bge, val_qids, mask_list)
s_ml_f2   = top100_filtered(sc_ml,  val_qids, mask_list)
s_sp_f2   = top100_filtered(sc_sp,  val_qids, mask_list)
ht_tuni_f3= top100_filtered(sc_ht_tuni, val_qids, mask_list)
ht_tbi_f3 = top100_filtered(sc_ht_tbi,  val_qids, mask_list)
inner_d   = rrf([s_bm25l_base, s_bge_f2, s_ml_f2], val_qids, k=1)
outer_d   = rrf([inner_d, s_bge_f2, s_bm25l_base, ht_tuni_f3, ht_tbi_f3,
                 s_sp_f2], val_qids, k=2)
print(f'  D. dense filtered + sparse unfiltered:    NDCG={ndcg(dr(outer_d)):.4f}  R@100={recall100(outer_d):.4f}')

# ── Per-domain NDCG breakdown ──────────────────────────────────────────────
print('\nPer-domain NDCG@10  (filter+rerank vs baseline):')
print(f'  {"Domain":<24}  {"n_q":>4}  {"pool":>6}  {"baseline":>10}  {"filtered":>10}  {"delta":>7}')
print('  ' + '-' * 70)
for domain in sorted(set(q_dom_val.values())):
    qids_dom  = [q for q in qrels if q_dom_val.get(q) == domain]
    if not qids_dom: continue
    n_base  = ndcg({q: base_dr.get(q,   []) for q in qids_dom})
    n_filt  = ndcg({q: outer_f_dr.get(q,[]) for q in qids_dom})
    avg_pool = np.mean([pool_sizes[q] for q in qids_dom if q in pool_sizes])
    print(f'  {domain:<24}  {len(qids_dom):>4}  {avg_pool:>6.0f}  {n_base:>10.4f}  '
          f'{n_filt:>10.4f}  {n_filt-n_base:>+7.4f}')

# ── Also try: skip filter for Business (too few docs) ─────────────────────
print(f'\n{"="*60}')
print(f'Best (filter + rerank): NDCG={ndcg(outer_f_dr):.4f}')
print(f'Baseline:               NDCG={ndcg(base_dr):.4f}')
print(f'Delta:                  {ndcg(outer_f_dr)-ndcg(base_dr):+.4f}')
