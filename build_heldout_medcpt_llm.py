"""
build_heldout_medcpt_llm.py
============================
Build held-out submission using:
  - PRF pipeline (val 0.7597 baseline)
  - + MedCPT for Medicine & Biology queries
  - + LLM listwise reranking (top-20) for all queries

Run ONLY after experiment_medcpt_llm.py confirms val improvement.

Usage:
    conda run -n py_env python3 build_heldout_medcpt_llm.py

Output:
    submissions_heldout/submission_medcpt_llm.json
"""

import json, math, time, re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Import all helpers from the experiment script ─────────────────────────────
# (re-define here to keep this script standalone)
from experiment_medcpt_llm import (
    corpus_ids, corpus_arr, ci_to_idx, val_qids, held_qids, qrels,
    c_dom_map, q_dom_val, q_dom_held, c_domains, dw,
    c_title, c_abstract, q_title, q_abstract,
    DOM_MODEL_NDCG, WEAK_DOMS, STRONG_DOMS,
    MEDCPT_CORPUS_EMB, MEDCPT_HLD_Q_EMB,
    LLM_MODEL, LLM_RERANK_TOP_N,
    cosine, top100, top100_filtered, rrf_fuse, domain_mask, dr,
    rrf_nested_exclude_sf, apply_ml_prf, apply_bge_prf,
    llm_rerank_submission, ndcg10, recall100, encode_medcpt,
)

SUB_DIR   = Path('submissions')
SUB_H_DIR = Path('submissions_heldout')

# ── Ensure MedCPT embeddings exist ────────────────────────────────────────────
print('\n' + '='*70)
print('STEP 1: MedCPT embeddings')
print('='*70)
encode_medcpt()

# ── Load val scores (to verify pipeline) ─────────────────────────────────────
print('\n' + '='*70)
print('STEP 2: Load scores & reproduce val baseline')
print('='*70)
_t = time.time()
print('  Loading val scores...')
sc_bm25l_v = np.load('submissions/scores_bm25l_ft.npy')
sc_tuni_v  = np.load('submissions/scores_tfidf_uni_ft.npy')
sc_tbi_v   = np.load('submissions/scores_tfidf_bi_ft.npy')
bge_qv_v   = np.load('submissions/bge_large_query_emb.npy')
bge_c      = np.load('submissions/bge_large_corpus_emb.npy')
sc_bge_v   = cosine(bge_qv_v, bge_c)
ml_qv_v    = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy')
ml_cv      = np.load('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy')
sc_ml_v    = (ml_qv_v @ ml_cv.T).astype(np.float32)
sp_q_v     = np.load('specter_prox_embed/queries_1_embeddings.npy')
sp_ft_c    = np.load('submissions/specter2_corpus_emb_ft.npy')
sc_sp_v    = cosine(sp_q_v, sp_ft_c)
medcpt_qv_v = np.load(SUB_DIR / 'medcpt_val_query_emb.npy')
medcpt_cv   = np.load(MEDCPT_CORPUS_EMB)
sc_medcpt_v = cosine(medcpt_qv_v, medcpt_cv)

# ── Val PRF pipeline ──────────────────────────────────────────────────────────
print('  Reproducing val pipeline (sanity check)...')
masks_v = [domain_mask(q_dom_val.get(qid, ''), 300) for qid in
           tqdm(val_qids, desc='  Val domain masks', ncols=80)]
s_bm25l_vf = top100_filtered(sc_bm25l_v, val_qids, masks_v)
s_bge_v    = top100(sc_bge_v, val_qids)
s_ml_v     = top100(sc_ml_v,  val_qids)
s_sp_v     = top100(sc_sp_v,  val_qids)
ht_tuni_vf = top100_filtered(rrf_fuse(sc_tuni_v, sc_sp_v, k=5), val_qids, masks_v)
ht_tbi_vf  = top100_filtered(rrf_fuse(sc_tbi_v,  sc_sp_v, k=5), val_qids, masks_v)
s_medcpt_v = top100(sc_medcpt_v, val_qids)

flat_v_base = {'BM25L': s_bm25l_vf, 'BGE': s_bge_v, 'MiniLM': s_ml_v,
               'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}
flat_v_sf   = {'BM25L': s_bm25l_vf, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

b0_v       = rrf_nested_exclude_sf(flat_v_base, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
b0_v_dr    = dr(b0_v, q_dom_val)

ml_qv_prf_v  = apply_ml_prf(ml_qv_v, ml_cv, s_bge_v, val_qids, q_dom_val)
sc_ml_prf_v  = (ml_qv_prf_v @ ml_cv.T).astype(np.float32)
s_ml_prf_v   = top100(sc_ml_prf_v, val_qids)

bge_qv_prf_v = apply_bge_prf(bge_qv_v, bge_c, b0_v_dr, val_qids)
sc_bge_prf_v = cosine(bge_qv_prf_v, bge_c)
s_bge_prf_v  = top100(sc_bge_prf_v, val_qids)

# MedCPT NDCG on val (needed for exclusion table)
dom_ndcg_with_medcpt = {d: dict(v) for d, v in DOM_MODEL_NDCG.items()}
for dom in ['Medicine', 'Biology']:
    qids_dom = [q for q, d in q_dom_val.items() if d == dom]
    nd = []
    for qid in qids_dom:
        rel = set(qrels.get(qid, []))
        if not rel: continue
        ranked = s_medcpt_v.get(qid, [])[:10]
        dcg  = sum(1/math.log2(r+2) for r, d in enumerate(ranked) if d in rel)
        idcg = sum(1/math.log2(r+2) for r in range(min(len(rel), 10)))
        nd.append(dcg/idcg if idcg else 0)
    dom_ndcg_with_medcpt[dom]['MedCPT'] = float(np.mean(nd)) if nd else 0.0
    print(f'    MedCPT val NDCG@10 [{dom}]: {dom_ndcg_with_medcpt[dom]["MedCPT"]:.4f}')

flat_v_prf_medcpt = {'BM25L': s_bm25l_vf, 'BGE': s_bge_prf_v, 'MiniLM': s_ml_prf_v,
                     'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf,
                     'MedCPT': s_medcpt_v}

val_before_llm    = rrf_nested_exclude_sf(flat_v_prf_medcpt, flat_v_sf,
                                          val_qids, q_dom_val, dom_ndcg_with_medcpt)
val_before_llm_dr = dr(val_before_llm, q_dom_val)
print(f'  ✓ Val NDCG@10 (PRF + MedCPT, before LLM) = {ndcg10(val_before_llm_dr):.4f}  '
      f'[{time.time()-_t:.1f}s]')

# ── Load held-out scores ──────────────────────────────────────────────────────
print('\n' + '='*70)
print('STEP 3: Build held-out pipeline')
print('='*70)
_t = time.time()
print('  Loading held-out scores...')
sc_bm25l_h  = np.load('submissions_heldout/scores_bm25l_ft.npy')
sc_tuni_h   = np.load('submissions_heldout/scores_tfidf_uni_ft.npy')
sc_tbi_h    = np.load('submissions_heldout/scores_tfidf_bi_ft.npy')
bge_qv_h    = np.load('submissions/bge_large_heldout_query_emb.npy')
sc_bge_h    = cosine(bge_qv_h, bge_c)
ml_qv_h     = np.load('data/embeddings/new_queries_minilm/query_embeddings.npy')
sc_ml_h     = (ml_qv_h @ ml_cv.T).astype(np.float32)
sp_q_h      = np.load('specter_prox_embed/queries_embeddings.npy')
sc_sp_h     = cosine(sp_q_h, sp_ft_c)
medcpt_qv_h = np.load(MEDCPT_HLD_Q_EMB)
sc_medcpt_h = cosine(medcpt_qv_h, medcpt_cv)

# ── Held-out pipeline ─────────────────────────────────────────────────────────
print('  Building held-out retrieval...')
masks_h = [domain_mask(q_dom_held.get(qid, ''), 300) for qid in
           tqdm(held_qids, desc='  Held domain masks', ncols=80)]
s_bm25l_hf = top100_filtered(sc_bm25l_h, held_qids, masks_h)
s_bge_h    = top100(sc_bge_h, held_qids)
s_ml_h     = top100(sc_ml_h,  held_qids)
s_sp_h     = top100(sc_sp_h,  held_qids)
ht_tuni_hf = top100_filtered(rrf_fuse(sc_tuni_h, sc_sp_h, k=5), held_qids, masks_h)
ht_tbi_hf  = top100_filtered(rrf_fuse(sc_tbi_h,  sc_sp_h, k=5), held_qids, masks_h)
s_medcpt_h = top100(sc_medcpt_h, held_qids)

flat_h_base = {'BM25L': s_bm25l_hf, 'BGE': s_bge_h, 'MiniLM': s_ml_h,
               'SPECTER2': s_sp_h, 'TF-IDF-uni': ht_tuni_hf, 'TF-IDF-bi': ht_tbi_hf}
flat_h_sf   = {'BM25L': s_bm25l_hf, 'TF-IDF-uni': ht_tuni_hf, 'TF-IDF-bi': ht_tbi_hf}

b0_h       = rrf_nested_exclude_sf(flat_h_base, flat_h_sf, held_qids, q_dom_held, DOM_MODEL_NDCG)
b0_h_dr    = dr(b0_h, q_dom_held)

ml_qv_prf_h  = apply_ml_prf(ml_qv_h, ml_cv, s_bge_h, held_qids, q_dom_held)
sc_ml_prf_h  = (ml_qv_prf_h @ ml_cv.T).astype(np.float32)
s_ml_prf_h   = top100(sc_ml_prf_h, held_qids)

bge_qv_prf_h = apply_bge_prf(bge_qv_h, bge_c, b0_h_dr, held_qids)
sc_bge_prf_h = cosine(bge_qv_prf_h, bge_c)
s_bge_prf_h  = top100(sc_bge_prf_h, held_qids)

flat_h_prf_medcpt = {'BM25L': s_bm25l_hf, 'BGE': s_bge_prf_h, 'MiniLM': s_ml_prf_h,
                     'SPECTER2': s_sp_h, 'TF-IDF-uni': ht_tuni_hf, 'TF-IDF-bi': ht_tbi_hf,
                     'MedCPT': s_medcpt_h}

print('  Running nested RRF + PRF + MedCPT on held-out...')
held_pre_llm    = rrf_nested_exclude_sf(flat_h_prf_medcpt, flat_h_sf,
                                        held_qids, q_dom_held, dom_ndcg_with_medcpt)
held_pre_llm_dr = dr(held_pre_llm, q_dom_held)
print(f'  ✓ Held-out pipeline done  [{time.time()-_t:.1f}s]  '
      f'queries={len(held_pre_llm_dr)}')

# ── LLM reranking (held-out) ──────────────────────────────────────────────────
print('\n' + '='*70)
print('STEP 4: LLM reranking (held-out)')
print('='*70)
n_heldout = len(held_qids)
print(f'  Queries          : {n_heldout}')
print(f'  Candidates/query : {LLM_RERANK_TOP_N}')
print(f'  LLM model        : {LLM_MODEL}')
print(f'  Estimated time   : {n_heldout * 20 / 60:.0f}–{n_heldout * 45 / 60:.0f} min')
print()
t0 = time.time()
held_final = llm_rerank_submission(
    held_pre_llm_dr,
    q_dom_held,
    target_domains=None,
    llm_model=LLM_MODEL,
    top_n=LLM_RERANK_TOP_N,
)
print(f'  ✓ LLM reranking done in {(time.time()-t0)/60:.1f} min')

# ── Save ──────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('STEP 5: Save')
print('='*70)
out = SUB_H_DIR / 'submission_medcpt_llm.json'
with open(out, 'w') as f:
    json.dump(held_final, f)
docs_per_q = set(len(v) for v in held_final.values())
print(f'  ✓ Saved: {out}')
print(f'    Queries    : {len(held_final)}')
print(f'    Docs/query : {docs_per_q}')
print()
print('  Next: upload submissions_heldout/submission_medcpt_llm.json to Codabench.')
print('='*70)
