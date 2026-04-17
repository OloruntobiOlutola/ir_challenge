"""
Test e5-mistral-7b-instruct embeddings in the ensemble.
Run after downloading the 3 .npy files from Kaggle into e5_emb/.

Usage:
    python3 experiment_e5mistral.py
"""
import json, numpy as np, pandas as pd

val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
held_qids  = pd.read_parquet('data/queries.parquet')['doc_id'].tolist()
qrels      = json.load(open('data/qrels_1.json'))

def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q/(nq+1e-10)) @ (c/(nc+1e-10)).T).astype(np.float32)

def top100(sc, qids):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-sc[i])[:100]
        sub[qid] = [corpus_ids[j] for j in idx]
    return sub

def rrf_fuse(sa, sb, k=5):
    out = np.zeros_like(sa)
    for i in range(len(sa)):
        ra = np.argsort(np.argsort(-sa[i])) + 1
        rb = np.argsort(np.argsort(-sb[i])) + 1
        out[i] = 1.0/(k+ra) + 1.0/(k+rb)
    return out

def rrf(lists, qids, k=3):
    sub = {}
    for qid in qids:
        sc = {}
        for lst in lists:
            for rank, doc in enumerate(lst.get(qid, []), 1):
                sc[doc] = sc.get(doc, 0.0) + 1.0/(k+rank)
        sub[qid] = sorted(sc, key=sc.get, reverse=True)[:100]
    return sub

def recall100(sub):
    r = []
    for qid, rels in qrels.items():
        rel_set = set(rels)
        r.append(sum(1 for d in sub.get(qid,[])[:100] if d in rel_set)/len(rel_set) if rel_set else 0)
    return float(np.mean(r))

def ndcg(sub, k=10):
    sc = []
    for qid, rels in qrels.items():
        if qid not in sub: continue
        ranked  = sub[qid][:k]
        rel_set = set(rels) if isinstance(rels, list) else set()
        dcg  = sum(1/np.log2(i+2) for i,d in enumerate(ranked) if d in rel_set)
        idcg = sum(1/np.log2(i+2) for i in range(min(len(rel_set), k)))
        sc.append(dcg/idcg if idcg>0 else 0)
    return float(np.mean(sc))

# Domain rerank
dw_df = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw    = dw_df.to_dict(orient='index')
c_df  = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
q_df  = pd.read_parquet('data/queries.parquet').set_index('doc_id')
c_dom_map  = c_df['domain'].to_dict()
q_dom_val  = {qid: q1_df.loc[qid,'domain'] for qid in val_qids  if qid in q1_df.index}
q_dom_held = {qid: q_df.loc[qid, 'domain'] for qid in held_qids if qid in q_df.index}

def dr(sub, qd_map, skip=frozenset({'Business'})):
    out = {}
    for qid, cands in sub.items():
        qd = qd_map.get(qid, '')
        if qd in skip:
            out[qid] = cands[:100]; continue
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d,''), 0.0), rank, d) for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _,_,d in scored[:100]]
    return out

# ── Load existing scores ───────────────────────────────────────────────────
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

# ── Load e5-mistral embeddings ─────────────────────────────────────────────
print('Loading e5-mistral embeddings...')
em_qv = np.load('e5-mistral/e5mistral_val_queries.npy')
em_c  = np.load('e5-mistral/e5mistral_corpus.npy')
sc_em = cosine(em_qv, em_c)
print(f'e5-mistral shapes — query: {em_qv.shape}  corpus: {em_c.shape}')

# ── Baseline lists ─────────────────────────────────────────────────────────
s_bm25l = top100(sc_bm25l, val_qids); s_bge = top100(sc_bge, val_qids)
s_ml    = top100(sc_ml,    val_qids); s_sp  = top100(sc_sp,  val_qids)
s_em    = top100(sc_em,    val_qids)
ht_tuni = top100(rrf_fuse(sc_tuni, sc_sp, k=5), val_qids)
ht_tbi  = top100(rrf_fuse(sc_tbi,  sc_sp, k=5), val_qids)
inner   = rrf([s_bm25l, s_bge, s_ml], val_qids, k=1)

print(f'\nStandalone e5-mistral: NDCG={ndcg(s_em):.4f}  Recall={recall100(s_em):.4f}')
print(f'Standalone BGE-large:  NDCG={ndcg(s_bge):.4f}  Recall={recall100(s_bge):.4f}')

# Baseline (no e5-mistral)
outer_base = rrf([inner, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp], val_qids, k=2)
base       = dr(outer_base, q_dom_val)
print(f'\nBaseline              NDCG={ndcg(base):.4f}  Recall={recall100(outer_base):.4f}')

# e5-mistral in inner
inner_em  = rrf([s_bm25l, s_bge, s_ml, s_em], val_qids, k=1)
outer_em1 = rrf([inner_em, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp], val_qids, k=2)
res_em1   = dr(outer_em1, q_dom_val)
print(f'e5m in inner          NDCG={ndcg(res_em1):.4f}  Recall={recall100(outer_em1):.4f}')

# e5-mistral in outer
outer_em2 = rrf([inner, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp, s_em], val_qids, k=2)
res_em2   = dr(outer_em2, q_dom_val)
print(f'e5m in outer          NDCG={ndcg(res_em2):.4f}  Recall={recall100(outer_em2):.4f}')

# e5-mistral in both
inner_em2 = rrf([s_bm25l, s_bge, s_ml, s_em], val_qids, k=1)
outer_em3 = rrf([inner_em2, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp, s_em], val_qids, k=2)
res_em3   = dr(outer_em3, q_dom_val)
print(f'e5m in inner+outer    NDCG={ndcg(res_em3):.4f}  Recall={recall100(outer_em3):.4f}')

# Also try hybrid fuse e5m with tfidf
ht_tuni_em = top100(rrf_fuse(sc_tuni, sc_em, k=5), val_qids)
ht_tbi_em  = top100(rrf_fuse(sc_tbi,  sc_em, k=5), val_qids)
outer_em4  = rrf([inner, s_bge, s_bm25l, ht_tuni, ht_tbi, s_sp, ht_tuni_em, ht_tbi_em, s_em], val_qids, k=2)
res_em4    = dr(outer_em4, q_dom_val)
print(f'e5m hybrid+outer      NDCG={ndcg(res_em4):.4f}  Recall={recall100(outer_em4):.4f}')

# ── Find best config, build held-out submission ────────────────────────────
configs = {
    'base':        (outer_base, base),
    'em_inner':    (outer_em1, res_em1),
    'em_outer':    (outer_em2, res_em2),
    'em_both':     (outer_em3, res_em3),
    'em_hybrid':   (outer_em4, res_em4),
}
best_name = max(configs, key=lambda k: ndcg(configs[k][1]))
best_ndcg = ndcg(configs[best_name][1])
print(f'\nBest config: {best_name}  NDCG={best_ndcg:.4f}')

if best_ndcg > 0.7493:
    print('IMPROVEMENT over baseline! Building held-out submission...')

    sc_bm25l_h = np.load('submissions_heldout/scores_bm25l_ft.npy')
    sc_tuni_h  = np.load('submissions_heldout/scores_tfidf_uni_ft.npy')
    sc_tbi_h   = np.load('submissions_heldout/scores_tfidf_bi_ft.npy')
    sc_bge_h   = np.load('submissions_heldout/scores_bge_dense_corrected.npy')
    ml_qh      = np.load('data/embeddings/new_queries_minilm/query_embeddings.npy')
    sc_ml_h    = (ml_qh @ ml_cv.T).astype(np.float32)
    sp_q_h     = np.load('specter_prox_embed/queries_embeddings.npy')
    sc_sp_h    = cosine(sp_q_h, sp_ft_c)
    em_qh      = np.load('e5_emb/e5mistral_heldout_queries.npy')
    sc_em_h    = cosine(em_qh, em_c)

    s_bm25l_h  = top100(sc_bm25l_h, held_qids); s_bge_h = top100(sc_bge_h, held_qids)
    s_ml_h     = top100(sc_ml_h,    held_qids); s_sp_h  = top100(sc_sp_h,  held_qids)
    s_em_h     = top100(sc_em_h,    held_qids)
    ht_tuni_h  = top100(rrf_fuse(sc_tuni_h, sc_sp_h, k=5), held_qids)
    ht_tbi_h   = top100(rrf_fuse(sc_tbi_h,  sc_sp_h, k=5), held_qids)

    if best_name == 'em_inner':
        inner_h = rrf([s_bm25l_h, s_bge_h, s_ml_h, s_em_h], held_qids, k=1)
        outer_h = rrf([inner_h, s_bge_h, s_bm25l_h, ht_tuni_h, ht_tbi_h, s_sp_h], held_qids, k=2)
    elif best_name == 'em_outer':
        inner_h = rrf([s_bm25l_h, s_bge_h, s_ml_h], held_qids, k=1)
        outer_h = rrf([inner_h, s_bge_h, s_bm25l_h, ht_tuni_h, ht_tbi_h, s_sp_h, s_em_h], held_qids, k=2)
    elif best_name == 'em_both':
        inner_h = rrf([s_bm25l_h, s_bge_h, s_ml_h, s_em_h], held_qids, k=1)
        outer_h = rrf([inner_h, s_bge_h, s_bm25l_h, ht_tuni_h, ht_tbi_h, s_sp_h, s_em_h], held_qids, k=2)
    elif best_name == 'em_hybrid':
        ht_tuni_em_h = top100(rrf_fuse(sc_tuni_h, sc_em_h, k=5), held_qids)
        ht_tbi_em_h  = top100(rrf_fuse(sc_tbi_h,  sc_em_h, k=5), held_qids)
        inner_h = rrf([s_bm25l_h, s_bge_h, s_ml_h], held_qids, k=1)
        outer_h = rrf([inner_h, s_bge_h, s_bm25l_h, ht_tuni_h, ht_tbi_h, s_sp_h,
                        ht_tuni_em_h, ht_tbi_em_h, s_em_h], held_qids, k=2)

    final_h = dr(outer_h, q_dom_held)
    out_path = f'submissions_heldout/submission_e5mistral_{best_name}.json'
    import json as _json
    with open(out_path, 'w') as f:
        _json.dump(final_h, f)
    print(f'Saved: {out_path}  (val NDCG={best_ndcg:.4f})')
else:
    print(f'No improvement over baseline (0.7493). e5-mistral does not help.')
