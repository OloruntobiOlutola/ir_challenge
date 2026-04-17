"""
Embed held-out queries (queries.parquet) with SPECTER2-Proximity and save
all outputs to the submissions folder.

Outputs written to ../submissions/:
  specter_heldout_query_emb.npy   — (N_q, 768) query embeddings
  specter_heldout_query_ids.json  — ordered doc_id list
  scores_specter_heldout.npy      — (N_q, 20000) cosine similarity matrix
  submission_specter_heldout.json — {query_id: [top-100 corpus_ids]}
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path('../data')
SUB_DIR     = Path('../submissions')
SPECTER_DIR = Path('../specter_prox_embed')
TOP_K       = 100

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading queries and corpus ...')
queries    = pd.read_parquet(DATA_DIR / 'queries.parquet')
corpus     = pd.read_parquet(DATA_DIR / 'corpus.parquet')
query_ids  = queries['doc_id'].tolist()
corpus_ids = corpus['doc_id'].tolist()
print(f'  queries : {len(queries)}  |  corpus : {len(corpus)}')

# ── Load pre-computed SPECTER2-Proximity embeddings ───────────────────────────
print('Loading SPECTER2-Proximity embeddings ...')
corp_emb = np.load(SPECTER_DIR / 'corpus_embeddings.npy').astype(np.float32)
quer_emb = np.load(SPECTER_DIR / 'queries_embeddings.npy').astype(np.float32)

with open(SPECTER_DIR / 'corpus_ids.json') as f: sp_corpus_ids = json.load(f)
with open(SPECTER_DIR / 'queries_ids.json') as f: sp_query_ids  = json.load(f)

print(f'  corpus emb : {corp_emb.shape}  |  query emb : {quer_emb.shape}')

# ── Align to notebook ordering ────────────────────────────────────────────────
sp_c_idx = {cid: i for i, cid in enumerate(sp_corpus_ids)}
sp_q_idx = {qid: i for i, qid in enumerate(sp_query_ids)}

missing_q = [q for q in query_ids if q not in sp_q_idx]
if missing_q:
    raise KeyError(
        f'{len(missing_q)} query ids not found in queries_embeddings.npy.\n'
        f'Re-run specter_embeddings_kaggle.ipynb on queries.parquet first.\n'
        f'First missing: {missing_q[:3]}'
    )

corp_emb = corp_emb[[sp_c_idx[c] for c in corpus_ids]]   # (20000, 768)
quer_emb = quer_emb[[sp_q_idx[q] for q in query_ids]]    # (N_q, 768)

# ── L2-normalise (dot product == cosine similarity) ───────────────────────────
corp_emb /= np.linalg.norm(corp_emb, axis=1, keepdims=True).clip(min=1e-9)
quer_emb /= np.linalg.norm(quer_emb, axis=1, keepdims=True).clip(min=1e-9)

# ── Cosine similarity matrix in batches ───────────────────────────────────────
BATCH = 20
scores = np.zeros((len(query_ids), len(corpus_ids)), dtype=np.float32)
for start in tqdm(range(0, len(query_ids), BATCH), desc='SPECTER cosine'):
    end = min(start + BATCH, len(query_ids))
    scores[start:end] = quer_emb[start:end] @ corp_emb.T

# ── Build top-100 submission ──────────────────────────────────────────────────
c_arr      = np.array(corpus_ids)
submission = {
    query_ids[i]: c_arr[np.argsort(-scores[i])[:TOP_K]].tolist()
    for i in range(len(query_ids))
}

# ── Save all outputs ──────────────────────────────────────────────────────────
SUB_DIR.mkdir(exist_ok=True)

np.save(SUB_DIR / 'specter_heldout_query_emb.npy', quer_emb)
with open(SUB_DIR / 'specter_heldout_query_ids.json', 'w') as f:
    json.dump(query_ids, f)
np.save(SUB_DIR / 'scores_specter_heldout.npy', scores)
with open(SUB_DIR / 'submission_specter_heldout.json', 'w') as f:
    json.dump(submission, f)

print('\n=== Saved to submissions/ ===')
print(f'  specter_heldout_query_emb.npy   shape={quer_emb.shape}')
print(f'  specter_heldout_query_ids.json  n={len(query_ids)}')
print(f'  scores_specter_heldout.npy      shape={scores.shape}')
print(f'  submission_specter_heldout.json queries={len(submission)}')
