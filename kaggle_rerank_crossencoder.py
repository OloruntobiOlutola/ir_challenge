"""
Cross-encoder reranking with BAAI/bge-reranker-large on Kaggle T4 GPU.

Steps:
  1. Run locally first:
       python3 save_top100_for_reranker.py
     → produces held_top100.json  (and val_top100.json if you want local eval)

  2. Upload held_top100.json to Kaggle as a dataset (e.g. "ir-submissions")

  3. Upload this script to Kaggle, set GPU=T4, add the dataset + challenge data,
     then run.  Expected runtime ~10 min.

Outputs (download this):
  - held_reranked_bge.json      (submit to Codabench)

Tip: run local_eval_crossencoder.py first to estimate val NDCG without
     using a Codabench submission.
"""
import subprocess, sys, json, os
import numpy as np
import pandas as pd
import torch

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'sentence-transformers'])

from sentence_transformers import CrossEncoder   # noqa: E402  (after install)

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = '/kaggle/input/ir-challenge/data'   # parquet files

# Auto-locate held_top100.json anywhere under /kaggle/input/
import glob as _glob
_matches = _glob.glob('/kaggle/input/**/held_top100.json', recursive=True)
if not _matches:
    raise FileNotFoundError(
        'held_top100.json not found under /kaggle/input/. '
        'Make sure you added the dataset containing held_top100.json to this notebook.'
    )
SUB_DIR = os.path.dirname(_matches[0])
print(f'Found submissions at: {SUB_DIR}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# ── Load text ──────────────────────────────────────────────────────────────
print('Loading text...')
c_df = pd.read_parquet(f'{DATA_DIR}/corpus.parquet').set_index('doc_id')
q_df = pd.read_parquet(f'{DATA_DIR}/queries.parquet').set_index('doc_id')

def get_ta(df, did):
    row = df.loc[did]
    ta  = row.get('ta', '')
    if pd.notna(ta) and str(ta).strip():
        return str(ta)[:1024]
    parts = [str(row.get(c, '')) for c in ('title', 'abstract') if pd.notna(row.get(c, ''))]
    return ' '.join(parts)[:1024]

# ── Load held-out submission ───────────────────────────────────────────────
held_top100 = json.load(open(f'{SUB_DIR}/held_top100.json'))
print(f'Held queries: {len(held_top100)}')

# ── Load cross-encoder ─────────────────────────────────────────────────────
print('Loading BAAI/bge-reranker-large...')
reranker = CrossEncoder('BAAI/bge-reranker-large', device=device, max_length=512)

def rerank(submission, query_df, out_path, top_k=100, batch_size=128):
    """Rerank top_k candidates per query, checkpointing after every query."""
    # Load existing progress if resuming
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f'  Resuming from checkpoint: {len(out)} queries already done.')
    else:
        out = {}

    qids = list(submission.keys())
    todo = [qid for qid in qids if qid not in out]
    print(f'  {len(out)}/{len(qids)} done, {len(todo)} remaining.')

    for i, qid in enumerate(todo):
        q_text  = get_ta(query_df, qid)
        cands   = submission[qid][:top_k]
        c_texts = [get_ta(c_df, d) for d in cands]
        pairs   = [(q_text, ct) for ct in c_texts]
        scores  = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        order   = np.argsort(-np.array(scores))
        out[qid] = [cands[j] for j in order]

        # Save after every query so any crash can be resumed
        with open(out_path, 'w') as f:
            json.dump(out, f)

        if (i + 1) % 10 == 0:
            print(f'  {len(out)}/{len(qids)}')

    return out

# ── Rerank held-out ────────────────────────────────────────────────────────
print('\nReranking held-out queries...')
held_reranked = rerank(held_top100, q_df, 'held_reranked_bge.json')
print(f'Saved held_reranked_bge.json  ({len(held_reranked)} queries)')

print('\nDone!  Download held_reranked_bge.json and submit to Codabench.')
