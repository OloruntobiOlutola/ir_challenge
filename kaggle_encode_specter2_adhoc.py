"""
Encode corpus + val + held-out queries with SPECTER2 adhoc_query adapter.

The proximity adapter (already used) is trained on co-citation proximity.
The adhoc_query adapter is trained for IR tasks where a paper title/abstract
is the query — a better fit for our task.

Run on Kaggle T4 GPU (~20-30 min for corpus, ~2 min for queries).

Outputs (download all):
  - specter2_adhoc_corpus.npy          (20000, 768)
  - specter2_adhoc_val_queries.npy     (100, 768)
  - specter2_adhoc_heldout_queries.npy (100, 768)
"""
import subprocess, sys, os, json
import numpy as np
import pandas as pd
import torch

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'huggingface_hub==0.22.2'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'transformers', 'accelerate', 'adapters>=1.0.0'])

from adapters import AutoAdapterModel
from transformers import AutoTokenizer

DATA_DIR = '/kaggle/input/ir-challenge/data'
device   = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# ── Load model with adhoc_query adapter ───────────────────────────────────
print('Loading SPECTER2 adhoc_query adapter...')
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model     = AutoAdapterModel.from_pretrained('allenai/specter2_base')
model.load_adapter('allenai/specter2', source='hf',
                   load_as='adhoc_query', set_active=True)
model = model.to(device)
model.eval()

def get_ta(df, did):
    row = df.loc[did]
    ta  = row.get('ta', '')
    if pd.notna(ta) and str(ta).strip():
        return str(ta)
    parts = [str(row.get(c, '')) for c in ('title', 'abstract') if pd.notna(row.get(c, ''))]
    return ' '.join(parts)

def encode(texts, batch_size=32, out_path=None):
    # Resume from checkpoint if interrupted
    if out_path and os.path.exists(out_path):
        existing = np.load(out_path)
        done = existing.shape[0]
        print(f'  Resuming: {done}/{len(texts)} already done')
        remaining = texts[done:]
    else:
        existing = None
        done = 0
        remaining = texts

    all_embs = [] if existing is None else [existing]
    for i in range(0, len(remaining), batch_size):
        batch  = remaining[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           return_tensors='pt', max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs)
        emb = out.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
        # Save checkpoint every batch
        if out_path:
            np.save(out_path, np.vstack(all_embs))
        total_done = done + i + len(batch)
        if total_done % 1000 < batch_size:
            print(f'  {total_done}/{len(texts)}')
    return np.vstack(all_embs)

# ── Load data ──────────────────────────────────────────────────────────────
corpus_ids = json.load(open(
    f'{DATA_DIR}/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
val_qids   = json.load(open(
    f'{DATA_DIR}/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))

c_df  = pd.read_parquet(f'{DATA_DIR}/corpus.parquet').set_index('doc_id')
q1_df = pd.read_parquet(f'{DATA_DIR}/queries_1.parquet').set_index('doc_id')
q_df  = pd.read_parquet(f'{DATA_DIR}/queries.parquet').set_index('doc_id')
held_qids = q_df.index.tolist()

# ── Encode corpus ──────────────────────────────────────────────────────────
print(f'\nEncoding corpus ({len(corpus_ids)} docs)...')
corpus_texts = [get_ta(c_df, d) for d in corpus_ids]
corpus_emb   = encode(corpus_texts, batch_size=32, out_path='specter2_adhoc_corpus.npy')
print(f'Corpus shape: {corpus_emb.shape}')

# ── Encode val queries ─────────────────────────────────────────────────────
print(f'\nEncoding val queries ({len(val_qids)})...')
val_texts = [get_ta(q1_df, q) for q in val_qids]
val_emb   = encode(val_texts, batch_size=32, out_path='specter2_adhoc_val_queries.npy')
np.save('specter2_adhoc_val_queries.npy', val_emb)
print(f'Val shape: {val_emb.shape}')

# ── Encode held-out queries ────────────────────────────────────────────────
print(f'\nEncoding held-out queries ({len(held_qids)})...')
held_texts = [get_ta(q_df, q) for q in held_qids]
held_emb   = encode(held_texts, batch_size=32, out_path='specter2_adhoc_heldout_queries.npy')
np.save('specter2_adhoc_heldout_queries.npy', held_emb)
print(f'Held shape: {held_emb.shape}')

print('\nDone! Download all three .npy files.')
