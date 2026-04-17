"""
Encode val + held-out queries with SPECTER2 proximity adapter using FULL TEXT.

Run on Kaggle T4 GPU (~5-10 min).

Outputs:
  - specter2_prox_val_queries_ft.npy     (100, 768)
  - specter2_prox_heldout_queries_ft.npy (100, 768)

Download both and place in the bge_emb/ folder.
"""
import subprocess, sys

# Pin huggingface_hub first — adapters uses HfFolder which was removed in >=0.23
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'huggingface_hub==0.22.2'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'transformers', 'accelerate', 'adapters>=1.0.0'])

import json
import numpy as np
import pandas as pd
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# ── Load data ───────────────────────────────────────────────────────────────
q1_df = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
q_df  = pd.read_parquet('data/queries.parquet').set_index('doc_id')
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
held_qids  = q_df.index.tolist()

print(f'Val queries: {len(val_qids)}, Held-out: {len(held_qids)}')

# ── Load SPECTER2 with proximity adapter ────────────────────────────────────
print('Loading SPECTER2 proximity adapter...')
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
model.load_adapter('allenai/specter2', source='hf', load_as='proximity', set_active=True)
model = model.to(device)
model.eval()

def get_full_text(df, ids):
    texts = []
    for did in ids:
        row = df.loc[did]
        ft = row.get('full_text', '')
        if pd.notna(ft) and str(ft).strip():
            texts.append(str(ft)[:8192])  # truncate for safety
        else:
            # fallback to ta
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

def encode_batch(texts, batch_size=16):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            return_tensors='pt', max_length=512
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        emb = out.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
        if i % 32 == 0:
            print(f'  {i}/{len(texts)}')
    return np.vstack(all_embs)

# Encode val queries with full_text
print('Encoding val queries with full_text...')
val_texts = get_full_text(q1_df, val_qids)
print(f'Sample val text (first 100 chars): {val_texts[0][:100]}')
val_emb = encode_batch(val_texts)
print(f'Val shape: {val_emb.shape}')
np.save('specter2_prox_val_queries_ft.npy', val_emb)

# Encode held-out queries with full_text
print('Encoding held-out queries with full_text...')
held_texts = get_full_text(q_df, held_qids)
held_emb = encode_batch(held_texts)
print(f'Held shape: {held_emb.shape}')
np.save('specter2_prox_heldout_queries_ft.npy', held_emb)

print('Done!')
