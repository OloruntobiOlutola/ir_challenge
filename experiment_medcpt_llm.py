"""
experiment_medcpt_llm.py
========================
Two-stage improvement on top of the best PRF pipeline (val 0.7597):

  Stage 1 — MedCPT domain specialist
    ncbi/MedCPT-Query-Encoder  +  ncbi/MedCPT-Article-Encoder
    Added to the RRF ensemble for Medicine and Biology queries only.
    These two domains account for 42/100 val queries.

  Stage 2 — LLM listwise reranking (top-20)
    Reranks the top-20 candidates from the Stage-1 pipeline
    using ollama llama3.2-vision:11b with a citation-specific prompt.
    The top-10 positions are replaced by the LLM ranking;
    positions 11-100 remain as-is from the pipeline.

Usage:
    conda run -n py_env python3 experiment_medcpt_llm.py

Outputs:
    submissions/medcpt_corpus_emb.npy       (cached, one-time)
    submissions/medcpt_val_query_emb.npy    (cached, one-time)
    submissions/submission_medcpt_llm.json  (val submission for inspection)
"""

import builtins
import json
import math
import re
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Force every print() to flush immediately so output appears in pipes/tee
_real_print = builtins.print


def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _real_print(*args, **kwargs)

# ── Config ─────────────────────────────────────────────────────────────────────
MEDCPT_QUERY_MODEL   = 'ncbi/MedCPT-Query-Encoder'
MEDCPT_ARTICLE_MODEL = 'ncbi/MedCPT-Article-Encoder'
LLM_MODEL            = 'llama3.2-vision:11b'   # local ollama model
LLM_RERANK_TOP_N     = 20                       # rerank top-N from pipeline
LLM_DOMAINS          = None                     # None = rerank all queries
MEDCPT_DOMAINS       = {'Medicine', 'Biology'}
THRESHOLD            = 0.70                     # model exclusion threshold

# ── Paths ──────────────────────────────────────────────────────────────────────
SUB_DIR   = Path('submissions')
SUB_H_DIR = Path('submissions_heldout')
DATA_DIR  = Path('data')

MEDCPT_CORPUS_EMB = SUB_DIR / 'medcpt_corpus_emb.npy'
MEDCPT_VAL_Q_EMB  = SUB_DIR / 'medcpt_val_query_emb.npy'
MEDCPT_HLD_Q_EMB  = SUB_DIR / 'medcpt_heldout_query_emb.npy'

# ── Load IDs and metadata ──────────────────────────────────────────────────────
print('Loading metadata...')
val_qids   = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_ids.json'))
corpus_ids = json.load(open('data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_ids.json'))
qrels      = json.load(open('data/qrels_1.json'))
q_df       = pd.read_parquet('data/queries.parquet').set_index('doc_id')
held_qids  = q_df.index.tolist()
corpus_arr = np.array(corpus_ids)
ci_to_idx  = {cid: i for i, cid in enumerate(corpus_ids)}

# Text lookup dicts for LLM prompts
c_df       = pd.read_parquet('data/corpus.parquet').set_index('doc_id')
q1_df      = pd.read_parquet('data/queries_1.parquet').set_index('doc_id')
c_title    = c_df['title'].to_dict()
c_abstract = c_df['abstract'].fillna('').to_dict()
q_title    = {**q1_df['title'].to_dict(), **q_df['title'].to_dict()}
q_abstract = {**q1_df['abstract'].fillna('').to_dict(),
              **q_df['abstract'].fillna('').to_dict()}

# Domain maps
dw_df     = pd.read_csv('domain_confusion_matrix_normalized.xls', index_col=0)
dw        = dw_df.to_dict(orient='index')
c_dom_map = c_df['domain'].to_dict()
q_dom_val  = {qid: q1_df.loc[qid, 'domain'] for qid in val_qids if qid in q1_df.index}
q_dom_held = {qid: q_df.loc[qid, 'domain']  for qid in held_qids if qid in q_df.index}
c_domains  = np.array([c_dom_map.get(cid, '') for cid in corpus_ids])

# ── Per-domain model NDCG (from build_prf_submission.py) ──────────────────────
DOM_MODEL_NDCG = {
    'Art':                  {'BM25L':0.5294,'BGE':0.5414,'MiniLM':0.5856,'SPECTER2':0.1952,'TF-IDF-uni':0.5294,'TF-IDF-bi':0.5135},
    'Biology':              {'BM25L':0.4822,'BGE':0.6187,'MiniLM':0.6042,'SPECTER2':0.5744,'TF-IDF-uni':0.6833,'TF-IDF-bi':0.6492},
    'Business':             {'BM25L':0.6709,'BGE':0.6792,'MiniLM':0.4708,'SPECTER2':0.6088,'TF-IDF-uni':0.7408,'TF-IDF-bi':0.6537},
    'Chemistry':            {'BM25L':0.7473,'BGE':0.7055,'MiniLM':0.6281,'SPECTER2':0.6423,'TF-IDF-uni':0.7702,'TF-IDF-bi':0.7684},
    'Computer Science':     {'BM25L':0.4600,'BGE':0.4922,'MiniLM':0.4733,'SPECTER2':0.4904,'TF-IDF-uni':0.5291,'TF-IDF-bi':0.5391},
    'Economics':            {'BM25L':1.0000,'BGE':1.0000,'MiniLM':1.0000,'SPECTER2':1.0000,'TF-IDF-uni':1.0000,'TF-IDF-bi':1.0000},
    'Engineering':          {'BM25L':0.5339,'BGE':0.8408,'MiniLM':0.5976,'SPECTER2':0.6011,'TF-IDF-uni':0.5751,'TF-IDF-bi':0.5616},
    'Environmental Science':{'BM25L':0.7740,'BGE':0.7832,'MiniLM':0.6849,'SPECTER2':0.7712,'TF-IDF-uni':0.9295,'TF-IDF-bi':0.9153},
    'Geography':            {'BM25L':0.7464,'BGE':0.8939,'MiniLM':0.8208,'SPECTER2':0.8377,'TF-IDF-uni':0.9159,'TF-IDF-bi':0.9159},
    'Geology':              {'BM25L':0.9260,'BGE':0.7491,'MiniLM':0.6793,'SPECTER2':0.7316,'TF-IDF-uni':0.8994,'TF-IDF-bi':0.8932},
    'History':              {'BM25L':1.0000,'BGE':1.0000,'MiniLM':1.0000,'SPECTER2':1.0000,'TF-IDF-uni':1.0000,'TF-IDF-bi':1.0000},
    'Materials Science':    {'BM25L':0.8746,'BGE':0.7890,'MiniLM':0.8164,'SPECTER2':0.7339,'TF-IDF-uni':0.8044,'TF-IDF-bi':0.7851},
    'Mathematics':          {'BM25L':0.8449,'BGE':0.8922,'MiniLM':0.7681,'SPECTER2':0.7980,'TF-IDF-uni':0.9914,'TF-IDF-bi':0.9634},
    'Medicine':             {'BM25L':0.5583,'BGE':0.6620,'MiniLM':0.6499,'SPECTER2':0.6163,'TF-IDF-uni':0.7122,'TF-IDF-bi':0.6772},
    'Philosophy':           {'BM25L':0.2372,'BGE':0.0000,'MiniLM':0.0000,'SPECTER2':0.0000,'TF-IDF-uni':0.2372,'TF-IDF-bi':0.6131},
    'Physics':              {'BM25L':0.8158,'BGE':0.5425,'MiniLM':0.6646,'SPECTER2':0.6646,'TF-IDF-uni':0.8400,'TF-IDF-bi':0.7557},
    'Political Science':    {'BM25L':0.9197,'BGE':0.6131,'MiniLM':0.7904,'SPECTER2':0.3066,'TF-IDF-uni':1.0000,'TF-IDF-bi':1.0000},
    'Psychology':           {'BM25L':0.7031,'BGE':0.7813,'MiniLM':0.8010,'SPECTER2':0.8043,'TF-IDF-uni':0.9007,'TF-IDF-bi':0.9295},
    'Sociology':            {'BM25L':1.0000,'BGE':1.0000,'MiniLM':1.0000,'SPECTER2':1.0000,'TF-IDF-uni':1.0000,'TF-IDF-bi':1.0000},
}

# PRF domain groups (from build_prf_submission.py)
WEAK_DOMS   = {'Computer Science', 'Biology', 'Medicine', 'Philosophy', 'Art', 'Engineering'}
STRONG_DOMS = {'Geology', 'Mathematics', 'Political Science', 'Economics', 'History',
               'Sociology', 'Chemistry', 'Physics', 'Materials Science'}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def cosine(q, c):
    nq = np.linalg.norm(q, axis=1, keepdims=True)
    nc = np.linalg.norm(c, axis=1, keepdims=True)
    return ((q / (nq + 1e-10)) @ (c / (nc + 1e-10)).T).astype(np.float32)

def top100(scores, qids):
    sub = {}
    for i, qid in enumerate(qids):
        idx = np.argsort(-scores[i])[:100]
        sub[qid] = corpus_arr[idx].tolist()
    return sub

def top100_filtered(score_matrix, qids, masks):
    sub = {}
    for i, qid in enumerate(qids):
        sc = score_matrix[i].copy()
        sc[~masks[i]] = -1e9
        idx = np.argsort(-sc)[:100]
        sub[qid] = corpus_arr[idx].tolist()
    return sub

def rrf_fuse(sa, sb, k=5):
    out = np.zeros_like(sa)
    for i in range(len(sa)):
        ra = np.argsort(np.argsort(-sa[i])) + 1
        rb = np.argsort(np.argsort(-sb[i])) + 1
        out[i] = 1.0 / (k + ra) + 1.0 / (k + rb)
    return out

def domain_mask(qdom, min_pool=300):
    row = dw.get(qdom, {})
    mask = np.array([row.get(d, 0.0) > 0.0 for d in c_domains])
    return mask if mask.sum() >= min_pool else np.ones(len(corpus_ids), dtype=bool)

def dr(sub, q_dom_map, skip=frozenset({'Business'})):
    out = {}
    for qid, cands in sub.items():
        qd = q_dom_map.get(qid, '')
        if qd in skip:
            out[qid] = cands[:100]
            continue
        wr = dw.get(qd, {})
        scored = [(wr.get(c_dom_map.get(d, ''), 0.0), rank, d)
                  for rank, d in enumerate(cands)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out[qid] = [d for _, _, d in scored[:100]]
    return out

def rrf_nested_exclude_sf(flat_unfiltered, flat_sparse_f, qids, q_dom_map,
                          dom_ndcg, threshold=0.70):
    sub = {}
    for qid in qids:
        qdom = q_dom_map.get(qid, '')
        dscores = dom_ndcg.get(qdom, {})
        if dscores:
            best_s = max(dscores.values())
            active = {m for m, s in dscores.items() if s >= best_s * threshold}
        else:
            active = set(flat_unfiltered.keys())
        if not active:
            active = set(flat_unfiltered.keys())

        def get_lst(mname):
            return flat_sparse_f.get(mname, flat_unfiltered[mname])

        inner_models = [m for m in ['BM25L', 'BGE', 'MiniLM', 'MedCPT']
                        if m in active]
        outer_extra  = [m for m in ['TF-IDF-uni', 'TF-IDF-bi', 'SPECTER2']
                        if m in active]
        if not inner_models:
            inner_models = ['BM25L', 'BGE', 'MiniLM']

        isc = {}
        for mname in inner_models:
            for rank, doc in enumerate(get_lst(mname).get(qid, []), 1):
                isc[doc] = isc.get(doc, 0.0) + 1.0 / (1 + rank)
        inner_sorted = sorted(isc, key=isc.get, reverse=True)[:100]

        osc = {}
        for rank, doc in enumerate(inner_sorted, 1):
            osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)
        for mname in outer_extra + [m for m in ['BGE', 'BM25L', 'MedCPT']
                                    if m in active]:
            for rank, doc in enumerate(get_lst(mname).get(qid, []), 1):
                osc[doc] = osc.get(doc, 0.0) + 1.0 / (2 + rank)

        sub[qid] = sorted(osc, key=osc.get, reverse=True)[:100]
    return sub

def apply_ml_prf(ml_qv, ml_cv, seed_sub, qids, q_dom_map, k=5):
    qe = ml_qv.copy()
    for i, qid in enumerate(qids):
        d = q_dom_map.get(qid, '')
        b = 0.8 if d in WEAK_DOMS else (0.0 if d in STRONG_DOMS else 0.60)
        if b == 0:
            continue
        docs = seed_sub.get(qid, [])[:k]
        idxs = [ci_to_idx[doc] for doc in docs if doc in ci_to_idx]
        if not idxs:
            continue
        fb = ml_cv[idxs].mean(axis=0)
        nq = qe[i] + b * fb
        qe[i] = nq / (np.linalg.norm(nq) + 1e-10)
    return qe

def apply_bge_prf(bge_qv, bge_c, seed_sub, qids, k=5, beta=0.06):
    qe = bge_qv.copy()
    for i, qid in enumerate(qids):
        docs = seed_sub.get(qid, [])[:k]
        idxs = [ci_to_idx[doc] for doc in docs if doc in ci_to_idx]
        if not idxs:
            continue
        fb = bge_c[idxs].mean(axis=0)
        nq = qe[i] + beta * fb
        qe[i] = nq / (np.linalg.norm(nq) + 1e-10)
    return qe

def ndcg10(sub):
    sc = []
    for q, rels in qrels.items():
        if q not in sub:
            continue
        ranked = sub[q][:10]
        rel_set = set(rels)
        dcg  = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in rel_set)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel_set), 10)))
        sc.append(dcg / idcg if idcg else 0)
    return float(np.mean(sc))

def recall100(sub):
    vals = []
    for qid, rels in qrels.items():
        rel_set = set(rels)
        if not rel_set:
            continue
        vals.append(sum(1 for d in sub.get(qid, [])[:100] if d in rel_set) / len(rel_set))
    return float(np.mean(vals))

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — MedCPT ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_medcpt():
    """Encode corpus and val queries with MedCPT. Cached on disk."""
    from transformers import AutoTokenizer, AutoModel
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'  MedCPT encoding on: {device}')

    def encode_texts(model_name, texts, batch_size=64, max_length=512):
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name).to(device).eval()
        all_emb = []
        n_batches = math.ceil(len(texts) / batch_size)
        bar = tqdm(range(0, len(texts), batch_size), total=n_batches,
                   desc=f'  Encoding [{model_name.split("/")[-1]}]',
                   unit='batch', ncols=80)
        for i in bar:
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                enc = tok(batch, padding=True, truncation=True,
                          max_length=max_length, return_tensors='pt').to(device)
                out = mdl(**enc)
                # MedCPT uses [CLS] token embedding
                emb = out.last_hidden_state[:, 0, :]
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-10)
                all_emb.append(emb.cpu().float().numpy())
            bar.set_postfix({'docs': min(i + batch_size, len(texts))})
        return np.vstack(all_emb)

    if not MEDCPT_CORPUS_EMB.exists():
        print('  Encoding corpus with MedCPT-Article-Encoder...')
        texts = [f"{c_title.get(cid, '')} {c_abstract.get(cid, '')}"
                 for cid in corpus_ids]
        emb = encode_texts(MEDCPT_ARTICLE_MODEL, texts)
        np.save(MEDCPT_CORPUS_EMB, emb)
        print(f'  Saved {MEDCPT_CORPUS_EMB}  shape={emb.shape}')
    else:
        print(f'  Corpus embeddings found: {MEDCPT_CORPUS_EMB}')

    if not MEDCPT_VAL_Q_EMB.exists():
        print('  Encoding val queries with MedCPT-Query-Encoder...')
        texts = [f"{q_title.get(qid, '')} {q_abstract.get(qid, '')}"
                 for qid in val_qids]
        emb = encode_texts(MEDCPT_QUERY_MODEL, texts)
        np.save(MEDCPT_VAL_Q_EMB, emb)
        print(f'  Saved {MEDCPT_VAL_Q_EMB}  shape={emb.shape}')
    else:
        print(f'  Val query embeddings found: {MEDCPT_VAL_Q_EMB}')

    if not MEDCPT_HLD_Q_EMB.exists():
        print('  Encoding held-out queries with MedCPT-Query-Encoder...')
        texts = [f"{q_title.get(qid, '')} {q_abstract.get(qid, '')}"
                 for qid in held_qids]
        emb = encode_texts(MEDCPT_QUERY_MODEL, texts)
        np.save(MEDCPT_HLD_Q_EMB, emb)
        print(f'  Saved {MEDCPT_HLD_Q_EMB}  shape={emb.shape}')
    else:
        print(f'  Held-out query embeddings found: {MEDCPT_HLD_Q_EMB}')


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — LLM LISTWISE RERANKING
# ═══════════════════════════════════════════════════════════════════════════════

RERANK_PROMPT_TMPL = """\
You are an expert in academic citation recommendation.
Your task: given a QUERY paper, rank the CANDIDATE papers by how likely the \
query paper would cite them (or be cited alongside them in the same context).
Focus on shared methodology, shared subject matter, and direct citation relationships \
— NOT just surface topical similarity.

QUERY PAPER
Title: {q_title}
Abstract: {q_abstract}

CANDIDATE PAPERS (numbered 1–{n})
{candidates}

Return ONLY a comma-separated ranking of the candidate numbers, most-likely-cited first.
Example format: 3,1,7,2,5,...
Do NOT include explanations. Numbers only."""


def build_candidate_block(doc_ids, max_abstract_words=80):
    lines = []
    for i, did in enumerate(doc_ids, 1):
        title = c_title.get(did, '(no title)')
        abstr = ' '.join(c_abstract.get(did, '').split()[:max_abstract_words])
        lines.append(f'[{i}] Title: {title}\n    Abstract: {abstr}...')
    return '\n\n'.join(lines)


def parse_ranking(response_text, n):
    """Extract a list of integers 1..n from LLM response. Falls back to 1..n."""
    # Try to find a sequence of comma-separated numbers
    nums = re.findall(r'\d+', response_text)
    seen = set()
    result = []
    for s in nums:
        v = int(s)
        if 1 <= v <= n and v not in seen:
            seen.add(v)
            result.append(v)
    # Append any missing numbers in original order
    for v in range(1, n + 1):
        if v not in seen:
            result.append(v)
    return result  # 1-indexed


def llm_rerank_query(qid, pipeline_top20, llm_model=LLM_MODEL):
    """
    Rerank pipeline_top20 (list of doc_ids) using an LLM.
    Returns a reranked list of doc_ids (same set, different order).
    """
    import ollama

    n = len(pipeline_top20)
    q_abs_short = ' '.join(q_abstract.get(qid, '').split()[:200])
    candidates  = build_candidate_block(pipeline_top20, max_abstract_words=80)
    prompt = RERANK_PROMPT_TMPL.format(
        q_title=q_title.get(qid, ''),
        q_abstract=q_abs_short,
        n=n,
        candidates=candidates,
    )
    try:
        resp = ollama.chat(
            model=llm_model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0, 'num_predict': 128},
        )
        text = resp['message']['content']
        ranking = parse_ranking(text, n)  # 1-indexed
        return [pipeline_top20[r - 1] for r in ranking]
    except Exception as e:
        print(f'    LLM error for {qid}: {e}')
        return pipeline_top20  # fall back to original order


def llm_rerank_submission(pipeline_sub, q_dom_map, target_domains=None,
                          llm_model=LLM_MODEL, top_n=LLM_RERANK_TOP_N):
    """
    Apply LLM reranking to a submission dict.
    target_domains=None means rerank all queries.
    Returns a new submission dict with top_n positions reranked.
    """
    qids_to_rerank = [
        qid for qid, _ in pipeline_sub.items()
        if target_domains is None or q_dom_map.get(qid, '') in target_domains
    ]
    out = dict(pipeline_sub)   # copy; will overwrite reranked entries
    t_start = time.time()
    bar = tqdm(qids_to_rerank, desc='  LLM reranking', unit='query', ncols=80)
    for idx, qid in enumerate(bar):
        cands = pipeline_sub[qid]
        top_n_cands  = cands[:top_n]
        rest         = cands[top_n:]
        t_q = time.time()
        reranked_top = llm_rerank_query(qid, top_n_cands, llm_model=llm_model)
        out[qid] = reranked_top + rest
        elapsed_q  = time.time() - t_q
        elapsed_total = time.time() - t_start
        done = idx + 1
        remaining = len(qids_to_rerank) - done
        eta_s = (elapsed_total / done) * remaining
        bar.set_postfix({
            'domain': q_dom_map.get(qid, '')[:8],
            'q_time': f'{elapsed_q:.1f}s',
            'ETA':    f'{eta_s/60:.1f}m',
        })
    print(f'  LLM reranked {len(qids_to_rerank)} queries in '
          f'{(time.time()-t_start)/60:.1f} min.')
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('STEP 1: MedCPT encoding')
print('='*70)
encode_medcpt()

print('\n' + '='*70)
print('STEP 2: Load all scores')
print('='*70)
_score_files = {
    'BM25L':       'submissions/scores_bm25l_ft.npy',
    'TF-IDF uni':  'submissions/scores_tfidf_uni_ft.npy',
    'TF-IDF bi':   'submissions/scores_tfidf_bi_ft.npy',
    'BGE query':   'submissions/bge_large_query_emb.npy',
    'BGE corpus':  'submissions/bge_large_corpus_emb.npy',
    'MiniLM q':    'data/embeddings/sentence-transformers_all-MiniLM-L6-v2/query_embeddings.npy',
    'MiniLM c':    'data/embeddings/sentence-transformers_all-MiniLM-L6-v2/corpus_embeddings.npy',
    'SPECTER q':   'specter_prox_embed/queries_1_embeddings.npy',
    'SPECTER c':   'submissions/specter2_corpus_emb_ft.npy',
    'MedCPT q':    str(MEDCPT_VAL_Q_EMB),
    'MedCPT c':    str(MEDCPT_CORPUS_EMB),
}
for label, path in tqdm(_score_files.items(), desc='  Loading files', ncols=80):
    pass  # just show progress; actual loads below

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

medcpt_qv   = np.load(MEDCPT_VAL_Q_EMB)
medcpt_cv   = np.load(MEDCPT_CORPUS_EMB)
sc_medcpt_v = cosine(medcpt_qv, medcpt_cv)
print(f'  ✓ All scores loaded. Shapes — BM25L: {sc_bm25l_v.shape}  '
      f'BGE: {sc_bge_v.shape}  MedCPT: {sc_medcpt_v.shape}')

print('\n' + '='*70)
print('STEP 3: Val PRF baseline (reproduce 0.7597)')
print('='*70)
_t = time.time()
masks_v = [domain_mask(q_dom_val.get(qid, ''), 300) for qid in
           tqdm(val_qids, desc='  Building domain masks', ncols=80)]

s_bm25l_vf = top100_filtered(sc_bm25l_v, val_qids, masks_v)
s_bge_v    = top100(sc_bge_v, val_qids)
s_ml_v     = top100(sc_ml_v,  val_qids)
s_sp_v     = top100(sc_sp_v,  val_qids)
ht_tuni_vf = top100_filtered(rrf_fuse(sc_tuni_v, sc_sp_v, k=5), val_qids, masks_v)
ht_tbi_vf  = top100_filtered(rrf_fuse(sc_tbi_v,  sc_sp_v, k=5), val_qids, masks_v)

flat_v_base = {'BM25L': s_bm25l_vf, 'BGE': s_bge_v, 'MiniLM': s_ml_v,
               'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}
flat_v_sf   = {'BM25L': s_bm25l_vf, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

# Baseline for BGE PRF seed
b0_v    = rrf_nested_exclude_sf(flat_v_base, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
b0_v_dr = dr(b0_v, q_dom_val)

# ML PRF + BGE PRF
ml_qv_prf_v  = apply_ml_prf(ml_qv_v, ml_cv, s_bge_v, val_qids, q_dom_val)
sc_ml_prf_v  = (ml_qv_prf_v @ ml_cv.T).astype(np.float32)
s_ml_prf_v   = top100(sc_ml_prf_v, val_qids)

bge_qv_prf_v = apply_bge_prf(bge_qv_v, bge_c, b0_v_dr, val_qids)
sc_bge_prf_v = cosine(bge_qv_prf_v, bge_c)
s_bge_prf_v  = top100(sc_bge_prf_v, val_qids)

flat_v_prf = {'BM25L': s_bm25l_vf, 'BGE': s_bge_prf_v, 'MiniLM': s_ml_prf_v,
              'SPECTER2': s_sp_v, 'TF-IDF-uni': ht_tuni_vf, 'TF-IDF-bi': ht_tbi_vf}

print('  Running nested RRF + PRF ensemble...')
baseline_sub = rrf_nested_exclude_sf(flat_v_prf, flat_v_sf, val_qids, q_dom_val, DOM_MODEL_NDCG)
baseline_dr  = dr(baseline_sub, q_dom_val)
print(f'  ✓ PRF baseline  NDCG@10 = {ndcg10(baseline_dr):.4f}  (expected 0.7597)  '
      f'[{time.time()-_t:.1f}s]')
print(f'               R@100   = {recall100(baseline_dr):.4f}')

print('\n' + '='*70)
print('STEP 4: Evaluate MedCPT standalone on Medicine & Biology')
print('='*70)
_t = time.time()
s_medcpt_v = top100(sc_medcpt_v, val_qids)

# Per-domain NDCG for MedCPT
medcpt_domain_ndcg = {}
for dom in ['Medicine', 'Biology']:
    qids_dom = [q for q, d in q_dom_val.items() if d == dom]
    nd = []
    for qid in qids_dom:
        rel = set(qrels.get(qid, []))
        if not rel:
            continue
        ranked = s_medcpt_v.get(qid, [])[:10]
        dcg  = sum(1/math.log2(r+2) for r, d in enumerate(ranked) if d in rel)
        idcg = sum(1/math.log2(r+2) for r in range(min(len(rel), 10)))
        nd.append(dcg / idcg if idcg else 0)
    score = float(np.mean(nd)) if nd else 0.0
    medcpt_domain_ndcg[dom] = score
    bge_s   = DOM_MODEL_NDCG[dom]['BGE']
    tfidf_s = DOM_MODEL_NDCG[dom]['TF-IDF-uni']
    marker  = '  ✓ NEW BEST' if score > max(bge_s, tfidf_s) else ''
    print(f'  MedCPT  {dom:<16}  NDCG@10 = {score:.4f}  '
          f'(BGE={bge_s:.4f}  TF-IDF-uni={tfidf_s:.4f}){marker}')
print(f'  [{time.time()-_t:.1f}s]')

print('\n' + '='*70)
print('STEP 5: Add MedCPT to ensemble (Medicine + Biology only)')
print('='*70)
_t = time.time()

# Add MedCPT to DOM_MODEL_NDCG for the two target domains
# Use the computed val NDCG as the exclusion metric
dom_ndcg_with_medcpt = {d: dict(v) for d, v in DOM_MODEL_NDCG.items()}
for dom in ['Medicine', 'Biology']:
    dom_ndcg_with_medcpt[dom]['MedCPT'] = medcpt_domain_ndcg.get(dom, 0.0)

# MedCPT-enhanced flat dict (MedCPT available for all queries,
# but domain exclusion will zero it out for non-Med/Bio domains
# since 0.0 < threshold × best for those domains)
flat_v_prf_medcpt = dict(flat_v_prf)
flat_v_prf_medcpt['MedCPT'] = s_medcpt_v

print('  Running ensemble with MedCPT...')
medcpt_sub = rrf_nested_exclude_sf(flat_v_prf_medcpt, flat_v_sf,
                                   val_qids, q_dom_val, dom_ndcg_with_medcpt)
medcpt_dr  = dr(medcpt_sub, q_dom_val)
_nd_med = ndcg10(medcpt_dr)
_nd_base = ndcg10(baseline_dr)
_delta = _nd_med - _nd_base
print(f'  ✓ + MedCPT  NDCG@10 = {_nd_med:.4f}  '
      f'(delta = {_delta:+.4f}  {"▲ improvement" if _delta > 0 else "▼ no improvement"})  '
      f'[{time.time()-_t:.1f}s]')
print(f'              R@100   = {recall100(medcpt_dr):.4f}')

# Per-domain breakdown
print('\n  Per-domain breakdown (Medicine & Biology):')
print(f'  {"Domain":<18} {"Baseline":>10} {"+ MedCPT":>10} {"Delta":>8}')
print('  ' + '-'*50)
for dom in ['Medicine', 'Biology']:
    qids_dom = [q for q, d in q_dom_val.items() if d == dom]
    scores = {}
    for sub_label, sub in [('Baseline', baseline_dr), ('MedCPT', medcpt_dr)]:
        nd = []
        for qid in qids_dom:
            rel = set(qrels.get(qid, []))
            if not rel:
                continue
            ranked = sub.get(qid, [])[:10]
            dcg  = sum(1/math.log2(r+2) for r, d in enumerate(ranked) if d in rel)
            idcg = sum(1/math.log2(r+2) for r in range(min(len(rel), 10)))
            nd.append(dcg / idcg if idcg else 0)
        scores[sub_label] = float(np.mean(nd)) if nd else 0.0
    delta = scores['MedCPT'] - scores['Baseline']
    marker = '▲' if delta > 0 else ('▼' if delta < 0 else '─')
    print(f'  {dom:<18} {scores["Baseline"]:>10.4f} {scores["MedCPT"]:>10.4f} '
          f'{delta:>+8.4f} {marker}')

print('\n' + '='*70)
print(f'STEP 6: LLM reranking (top-{LLM_RERANK_TOP_N}) — model: {LLM_MODEL}')
print('='*70)
n_to_rerank = (len(val_qids) if LLM_DOMAINS is None
               else sum(1 for d in q_dom_val.values() if d in LLM_DOMAINS))
print(f'  Queries to rerank : {n_to_rerank}')
print(f'  Candidates/query  : {LLM_RERANK_TOP_N}')
print(f'  LLM model         : {LLM_MODEL}')
print(f'  Estimated time    : {n_to_rerank * 20 / 60:.0f}–{n_to_rerank * 45 / 60:.0f} min '
      f'(20–45 s/query on 11B model)')
print()

t0 = time.time()
llm_sub = llm_rerank_submission(
    medcpt_dr,
    q_dom_val,
    target_domains=LLM_DOMAINS,
    llm_model=LLM_MODEL,
    top_n=LLM_RERANK_TOP_N,
)

_nd_llm = ndcg10(llm_sub)
print(f'\n  ✓ + MedCPT + LLM  NDCG@10 = {_nd_llm:.4f}  '
      f'(delta vs PRF baseline = {_nd_llm - _nd_base:+.4f}  '
      f'{"▲" if _nd_llm > _nd_base else "▼"})')
print(f'                    R@100   = {recall100(llm_sub):.4f}')

# ── Save val submission ────────────────────────────────────────────────────────
out_val = SUB_DIR / 'submission_medcpt_llm.json'
with open(out_val, 'w') as f:
    json.dump(llm_sub, f)
print(f'\nSaved val submission: {out_val}')

# ── Summary table ──────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('RESULTS SUMMARY')
print('='*70)
rows = [
    ('PRF baseline (0.73 best)',   ndcg10(baseline_dr), recall100(baseline_dr)),
    ('+ MedCPT (Med/Bio)',         ndcg10(medcpt_dr),   recall100(medcpt_dr)),
    ('+ MedCPT + LLM rerank',      ndcg10(llm_sub),     recall100(llm_sub)),
]
print(f'  {"Config":<35}  NDCG@10  R@100   Delta')
print('  ' + '-'*60)
base = rows[0][1]
for label, nd, rc in rows:
    print(f'  {label:<35}  {nd:.4f}  {rc:.4f}  {nd-base:+.4f}')
print('='*70)
print('\nNext step: if + MedCPT + LLM > 0.77, run on held-out with build_heldout_medcpt_llm.py')
