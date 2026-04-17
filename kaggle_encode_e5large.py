"""
Encode corpus + queries with E5-large-v2 (or intfloat/e5-large-v2).
E5-large uses instruction prefixes:
  - Query: "query: <text>"
  - Passage: "passage: <text>"

Run on Kaggle T4 GPU (~20-30 min for corpus).
Upload:
  - e5large_corpus.npy         (20000, 1024)
  - e5large_val_queries.npy    (100, 1024)
  - e5large_heldout_queries.npy (100, 1024)
"""
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# ── Load data ───────────────────────────────────────────────────────────────
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
q1_df = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
q_df  = pd.read_parquet('data/queries.parquet').set_index('doc_id')
c_df  = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
held_qids = q_df.index.tolist()

print(f'Val: {len(val_qids)}, Held: {len(held_qids)}, Corpus: {len(corpus_ids)}')

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

# E5 needs "query: " and "passage: " prefixes
corpus_texts   = ['passage: ' + t for t in get_ta(c_df,  corpus_ids)]
val_texts      = ['query: '   + t for t in get_ta(q1_df, val_qids)]
held_texts     = ['query: '   + t for t in get_ta(q_df,  held_qids)]

# ── Load E5-large-v2 ────────────────────────────────────────────────────────
print('Loading intfloat/e5-large-v2...')
model = SentenceTransformer('intfloat/e5-large-v2', device=device)
model.max_seq_length = 512

# ── Encode ──────────────────────────────────────────────────────────────────
print('Encoding corpus (20K docs)...')
corpus_emb = model.encode(
    corpus_texts, batch_size=128, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True
)
print(f'Corpus: {corpus_emb.shape}')
np.save('e5large_corpus.npy', corpus_emb)

print('Encoding val queries...')
val_emb = model.encode(
    val_texts, batch_size=64, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True
)
np.save('e5large_val_queries.npy', val_emb)

print('Encoding held-out queries...')
held_emb = model.encode(
    held_texts, batch_size=64, show_progress_bar=True,
    normalize_embeddings=True, convert_to_numpy=True
)
np.save('e5large_heldout_queries.npy', held_emb)

# Quick standalone NDCG check
qrels = json.load(open('data/qrels_1.json'))
sc_val = (val_emb @ corpus_emb.T).astype(np.float32)
def ndcg(sc, qids, k=10):
    scores = []
    for i, qid in enumerate(qids):
        rels = qrels.get(qid, [])
        if not rels: continue
        ranked = [corpus_ids[j] for j in np.argsort(-sc[i])[:k]]
        rel_set = set(rels)
        dcg  = sum(1.0/np.log2(r+2) for r, d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0/np.log2(r+2) for r in range(min(len(rel_set), k)))
        scores.append(dcg/idcg if idcg > 0 else 0)
    return float(np.mean(scores)) if scores else 0

print(f'E5-large standalone NDCG@10 val: {ndcg(sc_val, val_qids):.4f}')
print('Done!')
