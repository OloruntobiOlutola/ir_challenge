"""
Evaluate cross-encoder reranking on val locally using qrels.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU-friendly, ~2 min for 100 queries).

Run:
    pip install sentence-transformers   # if not already installed
    python3 local_eval_crossencoder.py

This lets you estimate the quality gain before spending a Codabench submission.
"""
import json
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

# ── Load data ──────────────────────────────────────────────────────────────
val_top100 = json.load(open('val_top100.json'))
qrels      = json.load(open('data/qrels_1.json'))
c_df       = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df      = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')

def get_ta(df, did):
    row = df.loc[did]
    ta  = row.get('ta', '')
    if pd.notna(ta) and str(ta).strip():
        return str(ta)[:512]
    parts = [str(row.get(c, '')) for c in ('title', 'abstract') if pd.notna(row.get(c, ''))]
    return ' '.join(parts)[:512]

def ndcg(sub, k=10):
    sc = []
    for qid, rels in qrels.items():
        if qid not in sub:
            continue
        ranked  = sub[qid][:k]
        rel_set = set(rels) if isinstance(rels, list) else set()
        dcg  = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
        sc.append(dcg / idcg if idcg > 0 else 0)
    return float(np.mean(sc))

# ── Baseline (before reranking) ────────────────────────────────────────────
print(f'Baseline NDCG@10 (ensemble top-100): {ndcg(val_top100):.4f}')

# ── Load cross-encoder ─────────────────────────────────────────────────────
# Swap model name for a stronger one if you have GPU:
#   'cross-encoder/ms-marco-MiniLM-L-12-v2'  — heavier, better
#   'BAAI/bge-reranker-large'                 — best, needs GPU
MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
print(f'Loading {MODEL}...')
reranker = CrossEncoder(MODEL, max_length=512)

# ── Rerank ─────────────────────────────────────────────────────────────────
reranked = {}
qids = list(val_top100.keys())
for i, qid in enumerate(qids):
    q_text  = get_ta(q1_df, qid)
    cands   = val_top100[qid][:100]
    c_texts = [get_ta(c_df, d) for d in cands]
    pairs   = [(q_text, ct) for ct in c_texts]
    scores  = reranker.predict(pairs, batch_size=64, show_progress_bar=False)
    order   = np.argsort(-np.array(scores))
    reranked[qid] = [cands[j] for j in order]
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{len(qids)}')

print(f'\nReranked NDCG@10 ({MODEL}): {ndcg(reranked):.4f}')
print(f'Delta vs baseline: {ndcg(reranked) - ndcg(val_top100):+.4f}')
print()
print('If gain looks good, run kaggle_rerank_crossencoder.py on Kaggle T4')
print('with BAAI/bge-reranker-large for the full-strength version.')
