"""
experiment_xgb.py
=================
LTR with XGBoost (rank:ndcg) + logistic regression (pointwise).
Reuses the same 19 features built in experiment_ltr.py.

Models tested:
  A) XGBoost rank:ndcg  (pairwise listwise)
  B) XGBoost rank:ndcg  (heavy regularisation)
  C) Logistic regression (pointwise, sklearn)

All evaluated via 5-fold CV on 100 val queries.
"""

import json, math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')

POOL_K = 100
SEED   = 42

# ── Re-use all data-loading from experiment_ltr ───────────────────────────────
# (import the already-built feature matrices and pair metadata)
print('Loading features from experiment_ltr.py...')
import importlib.util, sys

spec = importlib.util.spec_from_file_location('ltr', 'experiment_ltr.py')
# We only need the data — suppress the heavy output by redirecting briefly
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ltr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ltr)

X_val      = ltr.X_val
y_val      = ltr.y_val
groups_val = ltr.groups_val
pairs_val  = ltr.pairs_val
X_hld      = ltr.X_hld
groups_hld = ltr.groups_hld
pairs_hld  = ltr.pairs_hld
val_qids   = ltr.val_qids
held_qids  = ltr.held_qids
qrels      = ltr.qrels
val_base_dr = ltr.val_base_dr
held_base_dr = ltr.held_base_dr
FEAT_NAMES  = ltr.FEAT_NAMES

print(f'  Val  : {len(X_val)} pairs  positives={y_val.sum()}')
print(f'  Held : {len(X_hld)} pairs')
print(f'  Features ({len(FEAT_NAMES)}): {FEAT_NAMES}')


# ── Shared helpers ────────────────────────────────────────────────────────────
def ndcg_at_k(ranked, rel_set, k=10):
    dcg  = sum(1.0/math.log2(i+2) for i, d in enumerate(ranked[:k]) if d in rel_set)
    idcg = sum(1.0/math.log2(i+2) for i in range(min(len(rel_set), k)))
    return dcg/idcg if idcg else 0.0

def ndcg10(sub):
    sc = [ndcg_at_k(sub.get(q, []), set(rels))
          for q, rels in qrels.items() if q in sub]
    return float(np.mean(sc)) if sc else 0.0

def recall100(sub):
    vals = []
    for qid, rels in qrels.items():
        rel_set = set(rels)
        if not rel_set: continue
        vals.append(sum(1 for d in sub.get(qid, [])[:100] if d in rel_set)/len(rel_set))
    return float(np.mean(vals))

def scores_to_sub(pairs, scores, pool_k=100):
    qd = defaultdict(list)
    for (qid, doc), sc in zip(pairs, scores):
        qd[qid].append((sc, doc))
    return {qid: [d for _, d in sorted(lst, reverse=True)[:pool_k]]
            for qid, lst in qd.items()}

cum_groups_val = np.concatenate([[0], np.cumsum(groups_val)])


def run_cv(predict_fn, name, n_splits=5):
    """5-fold CV at query level. predict_fn(X_tr, y_tr, g_tr, X_te, g_te) → scores."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    cv_nd, cv_base_nd = [], []
    for fold, (tr_qi, te_qi) in enumerate(kf.split(np.arange(len(val_qids)))):
        tr_rows = [i for qi in tr_qi for i in range(cum_groups_val[qi], cum_groups_val[qi+1])]
        te_rows = [i for qi in te_qi for i in range(cum_groups_val[qi], cum_groups_val[qi+1])]
        g_tr = [groups_val[qi] for qi in tr_qi]
        g_te = [groups_val[qi] for qi in te_qi]

        scores_te = predict_fn(
            X_val[tr_rows], y_val[tr_rows], g_tr,
            X_val[te_rows], g_te
        )
        pairs_te   = [pairs_val[i] for i in te_rows]
        sub_ltr    = scores_to_sub(pairs_te, scores_te)
        sub_base   = {val_qids[qi]: val_base_dr[val_qids[qi]]
                      for qi in te_qi if val_qids[qi] in val_base_dr}

        cv_nd.append(ndcg10(sub_ltr))
        cv_base_nd.append(ndcg10(sub_base))

    mean_nd   = float(np.mean(cv_nd))
    mean_base = float(np.mean(cv_base_nd))
    print(f'  {name:<35} CV NDCG@10={mean_nd:.4f}  base={mean_base:.4f}  '
          f'Δ={mean_nd - mean_base:+.4f}')
    return mean_nd, mean_base, cv_nd


def _xgb_train(params, X_tr, y_tr, g_tr, X_te, rounds):
    """Train XGBoost ranker, no early stopping (avoids eval label requirement)."""
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEAT_NAMES)
    dtr.set_group(g_tr)
    dte = xgb.DMatrix(X_te, feature_names=FEAT_NAMES)
    bst = xgb.train(params, dtr, num_boost_round=rounds, verbose_eval=False)
    return bst.predict(dte), bst


# ── Model A: XGBoost rank:ndcg default ────────────────────────────────────────
def xgb_default(X_tr, y_tr, g_tr, X_te, g_te=None):
    params = {
        'objective': 'rank:ndcg', 'eta': 0.05,
        'max_depth': 4, 'min_child_weight': 5,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'lambda': 1.0, 'alpha': 0.1, 'seed': SEED,
    }
    scores, _ = _xgb_train(params, X_tr, y_tr, g_tr, X_te, rounds=150)
    return scores


# ── Model B: XGBoost rank:ndcg heavy regularisation ──────────────────────────
def xgb_heavy(X_tr, y_tr, g_tr, X_te, g_te=None):
    params = {
        'objective': 'rank:ndcg', 'eta': 0.02,
        'max_depth': 3, 'min_child_weight': 20,
        'subsample': 0.7, 'colsample_bytree': 0.6,
        'lambda': 5.0, 'alpha': 1.0, 'gamma': 1.0, 'seed': SEED,
    }
    scores, _ = _xgb_train(params, X_tr, y_tr, g_tr, X_te, rounds=80)
    return scores


# ── Model C: XGBoost rank:pairwise ────────────────────────────────────────────
def xgb_pairwise(X_tr, y_tr, g_tr, X_te, g_te=None):
    params = {
        'objective': 'rank:pairwise', 'eta': 0.03,
        'max_depth': 3, 'min_child_weight': 10,
        'subsample': 0.8, 'colsample_bytree': 0.7,
        'lambda': 3.0, 'alpha': 0.5, 'seed': SEED,
    }
    scores, _ = _xgb_train(params, X_tr, y_tr, g_tr, X_te, rounds=100)
    return scores


# ── Model D: Logistic Regression (pointwise, global normalisation) ────────────
def logreg(X_tr, y_tr, g_tr, X_te, g_te=None):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    clf = LogisticRegression(C=0.1, max_iter=500, random_state=SEED, solver='lbfgs')
    clf.fit((X_tr - mu) / sd, y_tr)
    return clf.predict_proba((X_te - mu) / sd)[:, 1]


# ── Model E: Logistic Regression looser regularisation ───────────────────────
def logreg_loose(X_tr, y_tr, g_tr, X_te, g_te=None):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED, solver='lbfgs')
    clf.fit((X_tr - mu) / sd, y_tr)
    return clf.predict_proba((X_te - mu) / sd)[:, 1]


print('\n' + '='*70)
print('5-fold cross-validation — model comparison')
print('='*70)

results = {}
results['A_xgb_default'],    _, cvA = run_cv(xgb_default,   'XGBoost rank:ndcg (default)')
results['B_xgb_heavy'],      _, cvB = run_cv(xgb_heavy,     'XGBoost rank:ndcg (heavy reg)')
results['C_xgb_pairwise'],   _, cvC = run_cv(xgb_pairwise,  'XGBoost rank:pairwise')
results['D_logreg'],         _, cvD = run_cv(logreg,        'Logistic Regression (C=0.1)')
results['E_logreg_loose'],   _, cvE = run_cv(logreg_loose,  'Logistic Regression (C=1.0)')

baseline_nd = ndcg10(val_base_dr)
print(f'\n  {"Baseline (PRF pipeline)":<35} NDCG@10={baseline_nd:.4f}')

best_name = max(results, key=results.get)
best_nd   = results[best_name]
print(f'\n  Best model: {best_name}  NDCG@10={best_nd:.4f}  '
      f'Δ={best_nd - baseline_nd:+.4f}')


# ── Train best model on all val, predict held-out ─────────────────────────────
print('\n' + '='*70)
print('Train best model on ALL val → held-out submission')
print('='*70)

model_fns = {
    'A_xgb_default':   xgb_default,
    'B_xgb_heavy':     xgb_heavy,
    'C_xgb_pairwise':  xgb_pairwise,
    'D_logreg':        logreg,
    'E_logreg_loose':  logreg_loose,
}

best_fn = model_fns[best_name]

# For XGB models: train on full val, predict held-out
if best_name.startswith('A') or best_name.startswith('B') or best_name.startswith('C'):
    fn_map = {'A_xgb_default': (xgb_default, 150),
              'B_xgb_heavy':   (xgb_heavy,   80),
              'C_xgb_pairwise':(xgb_pairwise,100)}
    fn, rounds = fn_map[best_name]
    # reuse _xgb_train with full val as train, heldout as test
    scores_hld_best, bst_full = _xgb_train(
        {'objective': 'rank:ndcg' if best_name != 'C_xgb_pairwise' else 'rank:pairwise',
         **{k: v for k, v in [
             ('eta', 0.05 if best_name=='A_xgb_default' else (0.02 if best_name=='B_xgb_heavy' else 0.03)),
             ('max_depth', 4 if best_name=='A_xgb_default' else 3),
             ('min_child_weight', 5 if best_name=='A_xgb_default' else (20 if best_name=='B_xgb_heavy' else 10)),
             ('subsample', 0.8 if best_name!='B_xgb_heavy' else 0.7),
             ('colsample_bytree', 0.8 if best_name=='A_xgb_default' else (0.6 if best_name=='B_xgb_heavy' else 0.7)),
             ('lambda', 1.0 if best_name=='A_xgb_default' else (5.0 if best_name=='B_xgb_heavy' else 3.0)),
             ('alpha', 0.1 if best_name=='A_xgb_default' else (1.0 if best_name=='B_xgb_heavy' else 0.5)),
             ('seed', SEED),
         ]}},
        X_val, y_val, groups_val, X_hld, rounds=rounds
    )
    scores_val_self = bst_full.predict(xgb.DMatrix(X_val, feature_names=FEAT_NAMES))

else:  # logistic
    X_sc = (X_val - X_val.mean(0)) / (X_val.std(0) + 1e-9)
    C = 0.1 if best_name == 'D_logreg' else 1.0
    clf_full = LogisticRegression(C=C, max_iter=500, random_state=SEED, solver='lbfgs')
    clf_full.fit(X_sc, y_val)
    X_hld_sc = (X_hld - X_val.mean(0)) / (X_val.std(0) + 1e-9)
    scores_hld_best = clf_full.predict_proba(X_hld_sc)[:, 1]
    scores_val_self = clf_full.predict_proba(X_sc)[:, 1]

sub_hld_best  = scores_to_sub(pairs_hld,  scores_hld_best)
sub_val_self  = scores_to_sub(pairs_val,  scores_val_self)

print(f'  Best model      : {best_name}')
print(f'  Val self-score  : NDCG@10={ndcg10(sub_val_self):.4f}  R@100={recall100(sub_val_self):.4f}')
print(f'  Val baseline    : NDCG@10={baseline_nd:.4f}  R@100={recall100(val_base_dr):.4f}')
print(f'  CV estimate     : NDCG@10={best_nd:.4f}  (realistic, cross-validated)')

# Per-domain breakdown (self-score)
print(f'\n  Per-domain (self-score, train=test):')
print(f'  {"Domain":<25} {"Baseline":>8}  {"LTR":>8}  {"Δ":>7}')
print('  ' + '-'*55)
for dom in sorted(set(ltr.q_dom_val.values())):
    qids_dom = [q for q, d in ltr.q_dom_val.items() if d == dom and q in qrels]
    if not qids_dom: continue
    nd_b = ndcg10({q: val_base_dr[q] for q in qids_dom if q in val_base_dr})
    nd_l = ndcg10({q: sub_val_self[q]  for q in qids_dom if q in sub_val_self})
    print(f'  {dom:<25} {nd_b:8.4f}  {nd_l:8.4f}  {nd_l-nd_b:+7.4f}')

# Save
out_path = Path('submissions_heldout/submission_xgb.json')
with open(out_path, 'w') as f:
    json.dump(sub_hld_best, f)
print(f'\n  Saved: {out_path}')
print(f'  CV NDCG@10 estimate: {best_nd:.4f}  (baseline: {baseline_nd:.4f})')
print('='*70)
