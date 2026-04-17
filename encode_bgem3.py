"""
Encode corpus + queries with BGE-M3 and find best fusion config.
"""
import json
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

# ── Ordering ───────────────────────────────────────────────────────────────
val_qids = json.load(
    open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json')
)
corpus_ids = json.load(
    open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json')
)
q_df  = pd.read_parquet('data/queries.parquet').set_index('doc_id')
q1_df = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_df  = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
held_qids = q_df.index.tolist()

print(f'Val: {len(val_qids)}  Held-out: {len(held_qids)}  Corpus: {len(corpus_ids)}')


def get_ta(df, ids):
    texts = []
    for did in ids:
        row = df.loc[did]
        ta = row.get('ta', '')
        if pd.notna(ta) and str(ta).strip():
            texts.append(str(ta))
        else:
            parts = []
            for col in ('title', 'abstract'):
                v = row.get(col, '')
                if pd.notna(v):
                    parts.append(str(v))
            texts.append(' '.join(parts))
    return texts


corpus_texts = get_ta(c_df, corpus_ids)
val_texts    = get_ta(q1_df, val_qids)
held_texts   = get_ta(q_df, held_qids)

# ── Load BGE-M3 ────────────────────────────────────────────────────────────
print('Loading BAAI/bge-m3 ...')
model = SentenceTransformer('BAAI/bge-m3', device=device)
model.max_seq_length = 512

# ── Encode ─────────────────────────────────────────────────────────────────
OUT = 'submissions_heldout'

print('Encoding corpus (20K docs)...')
corpus_emb = model.encode(
    corpus_texts, batch_size=128, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True
)
print(f'Corpus: {corpus_emb.shape}')
np.save(f'{OUT}/bgem3_corpus.npy', corpus_emb)

print('Encoding val queries...')
val_emb = model.encode(
    val_texts, batch_size=64, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True
)
np.save(f'{OUT}/bgem3_val_queries.npy', val_emb)

print('Encoding held-out queries...')
held_emb = model.encode(
    held_texts, batch_size=64, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True
)
np.save(f'{OUT}/bgem3_heldout_queries.npy', held_emb)

# ── Scores ─────────────────────────────────────────────────────────────────
sc_m3_val  = (val_emb  @ corpus_emb.T).astype(np.float32)
sc_m3_held = (held_emb @ corpus_emb.T).astype(np.float32)
np.save(f'{OUT}/bgem3_scores_val.npy',    sc_m3_val)
np.save(f'{OUT}/bgem3_scores_heldout.npy', sc_m3_held)

# ── Helpers ────────────────────────────────────────────────────────────────
qrels = json.load(open('data/qrels_1.json'))


def top100(scores, qids, cids=corpus_ids):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-scores[i])[:100]
        sub[qid] = [cids[j] for j in idx]
    return sub


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
        if qid not in sub:
            continue
        ranked  = sub[qid][:k]
        rel_set = set(rels) if isinstance(rels, list) else set()
        dcg  = sum(1.0/np.log2(i+2) for i, d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0/np.log2(i+2) for i in range(min(len(rel_set), k)))
        sc.append(dcg/idcg if idcg > 0 else 0)
    return float(np.mean(sc))


# ── Val baselines ──────────────────────────────────────────────────────────
sc_bm25l_v = np.load('submissions/scores_bm25l_ft.npy')
sc_tuni_v  = np.load('submissions/scores_tfidf_uni_ft.npy')
sc_tbi_v   = np.load('submissions/scores_tfidf_bi_ft.npy')
bge_qv     = np.load('submissions/bge_large_query_emb.npy')
bge_c      = np.load('submissions/bge_large_corpus_emb.npy')
sc_bge_v   = (bge_qv @ bge_c.T).astype(np.float32)
sp_q1      = np.load('specter_prox_embed/queries_1_embeddings.npy')
sp_c       = np.load('specter_prox_embed/corpus_embeddings.npy')
sc_sp_v    = (sp_q1 @ sp_c.T).astype(np.float32)
ml_qv      = np.load(
    'data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy'
)
ml_cv      = np.load(
    'data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy'
)
sc_ml_v    = (ml_qv @ ml_cv.T).astype(np.float32)


def rrf_fuse(sa, sb, k=60):
    out = np.zeros_like(sa)
    for i in range(len(sa)):
        ra = np.argsort(np.argsort(-sa[i])) + 1
        rb = np.argsort(np.argsort(-sb[i])) + 1
        out[i] = 1.0/(k+ra) + 1.0/(k+rb)
    return out


sc_htbm25l_v = rrf_fuse(sc_bm25l_v, sc_sp_v)
sc_httuni_v  = rrf_fuse(sc_tuni_v,  sc_sp_v)
sc_httbi_v   = rrf_fuse(sc_tbi_v,   sc_sp_v)
sc_htm3_v    = rrf_fuse(sc_bm25l_v, sc_m3_val)  # bm25l + bgem3 hybrid

sub_bm25l_v   = top100(sc_bm25l_v,   val_qids)
sub_bge_v     = top100(sc_bge_v,      val_qids)
sub_sp_v      = top100(sc_sp_v,       val_qids)
sub_ml_v      = top100(sc_ml_v,       val_qids)
sub_m3_v      = top100(sc_m3_val,     val_qids)
sub_htbm25l_v = top100(sc_htbm25l_v,  val_qids)
sub_httuni_v  = top100(sc_httuni_v,   val_qids)
sub_httbi_v   = top100(sc_httbi_v,    val_qids)
sub_htm3_v    = top100(sc_htm3_v,     val_qids)

print('\n=== Val NDCG@10 baselines ===')
print(f'BGE-large:      {ndcg(sub_bge_v):.4f}')
print(f'BGE-M3:         {ndcg(sub_m3_v):.4f}')
print(f'MiniLM:         {ndcg(sub_ml_v):.4f}')
print(f'SPECTER:        {ndcg(sub_sp_v):.4f}')
print(f'BM25L:          {ndcg(sub_bm25l_v):.4f}')
print(f'ht_bm25l:       {ndcg(sub_htbm25l_v):.4f}')
print(f'ht_tuni:        {ndcg(sub_httuni_v):.4f}')
print(f'ht_tbi:         {ndcg(sub_httbi_v):.4f}')
print(f'ht_m3(bm25l+m3):{ndcg(sub_htm3_v):.4f}')

# Replicate D-equivalent
inner_d = rrf([sub_bm25l_v, sub_bge_v, sub_ml_v], val_qids, k=3)
outer_d = rrf(
    [inner_d, sub_bge_v, sub_bm25l_v, sub_httuni_v, sub_httbi_v],
    val_qids, k=3
)
print(f'\nD-equiv val:    {ndcg(outer_d):.4f}')

# M3 replaces BGE-large
inner_m3 = rrf([sub_bm25l_v, sub_m3_v, sub_ml_v], val_qids, k=3)
outer_m3 = rrf(
    [inner_m3, sub_m3_v, sub_bm25l_v, sub_httuni_v, sub_httbi_v],
    val_qids, k=3
)
print(f'M3 replaces BGE:{ndcg(outer_m3):.4f}')

# M3 added to D
outer_m3_add = rrf(
    [inner_d, sub_bge_v, sub_m3_v, sub_bm25l_v, sub_httuni_v, sub_httbi_v],
    val_qids, k=3
)
print(f'M3 added to D:  {ndcg(outer_m3_add):.4f}')

# ht_m3 added to D
outer_htm3_add = rrf(
    [inner_d, sub_bge_v, sub_bm25l_v, sub_httuni_v, sub_httbi_v, sub_htm3_v],
    val_qids, k=3
)
print(f'ht_m3 added to D:{ndcg(outer_htm3_add):.4f}')

# Full M3 ensemble (M3 everywhere + ht_m3)
inner_full = rrf([sub_bm25l_v, sub_m3_v, sub_ml_v], val_qids, k=3)
outer_full = rrf(
    [inner_full, sub_m3_v, sub_bge_v, sub_bm25l_v,
     sub_httuni_v, sub_httbi_v, sub_htm3_v],
    val_qids, k=3
)
print(f'Full M3 7-way:  {ndcg(outer_full):.4f}')

# ── Build best held-out submission ─────────────────────────────────────────
print('\n=== Building held-out submission ===')

sc_bm25l_h  = np.load(f'{OUT}/scores_bm25l_ft.npy')
sc_bge_h    = np.load(f'{OUT}/scores_bge_dense_corrected.npy')
sc_sp_h     = np.load(f'{OUT}/scores_specter_prox.npy')
sc_htbm25l_h = np.load(f'{OUT}/scores_hybrid_bm25l_ft.npy')
sc_httuni_h  = np.load(f'{OUT}/scores_hybrid_tfidf_uni_ft.npy')
sc_httbi_h   = np.load(f'{OUT}/scores_hybrid_tfidf_bi_ft.npy')
sc_htm3_h    = rrf_fuse(sc_bm25l_h, sc_m3_held)

# MiniLM held-out
ml_qh = np.load('data/embeddings/new_queries_minilm/query_embeddings.npy')
sc_ml_h = (ml_qh @ ml_cv.T).astype(np.float32)

sub_bm25l_h   = top100(sc_bm25l_h,   held_qids)
sub_bge_h     = top100(sc_bge_h,      held_qids)
sub_ml_h      = top100(sc_ml_h,       held_qids)
sub_m3_h      = top100(sc_m3_held,    held_qids)
sub_httuni_h  = top100(sc_httuni_h,   held_qids)
sub_httbi_h   = top100(sc_httbi_h,    held_qids)
sub_htm3_h    = top100(sc_htm3_h,     held_qids)

# Build best config (use same structure as best val config above)
inner_best_h = rrf([sub_bm25l_h, sub_m3_h, sub_ml_h], held_qids, k=3)
outer_best_h = rrf(
    [inner_best_h, sub_m3_h, sub_bge_h, sub_bm25l_h,
     sub_httuni_h, sub_httbi_h, sub_htm3_h],
    held_qids, k=3
)

out_path = f'{OUT}/submission_bgem3_7way.json'
with open(out_path, 'w') as f:
    json.dump(outer_best_h, f)
print(f'Saved: {out_path}')
print(f'Queries: {len(outer_best_h)}  Docs/query: {len(list(outer_best_h.values())[0])}')

# Also save D-equivalent with M3 added (simpler config)
inner_d_h = rrf([sub_bm25l_h, sub_bge_h, sub_ml_h], held_qids, k=3)
outer_d_m3_h = rrf(
    [inner_d_h, sub_bge_h, sub_m3_h, sub_bm25l_h, sub_httuni_h, sub_httbi_h],
    held_qids, k=3
)
out_path2 = f'{OUT}/submission_d_plus_m3.json'
with open(out_path2, 'w') as f:
    json.dump(outer_d_m3_h, f)
print(f'Saved: {out_path2}')
print('Done.')
