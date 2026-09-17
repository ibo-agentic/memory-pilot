"""Doubly robust (AIPW) estimator: IPS correction on top of a logistic
outcome model fit on task features + the full inclusion-indicator vector.

For memory m: DR(m) = mean over episodes where m is a candidate of
  (Z/p)*(Y-mu(X,Z_-m,1)) - ((1-Z)/(1-p))*(Y-mu(X,Z_-m,0)) + mu(X,Z_-m,1) - mu(X,Z_-m,0)
where mu is the fitted outcome model queried with memory m's own inclusion
indicator forced to 1 and to 0, holding every other memory's observed
inclusion (and task features) fixed.

Cross-fitting (default on): mu for each episode is predicted by a model
trained on the OTHER folds, never the fold containing that episode. This
avoids the outcome model overfitting to the same data the AIPW correction
term is evaluated on. Set cross_fit=False to fit and evaluate on the full
dataset (the simpler, slightly overfit-prone variant), useful only as a
comparison point.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def _build_feature_matrix(
    episodes: list[dict], all_memory_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, int], int]:
    task_types = sorted({ep["task_type"] for ep in episodes})
    tt_index = {t: i for i, t in enumerate(task_types)}
    mem_index = {mem_id: i for i, mem_id in enumerate(all_memory_ids)}
    n_tt = len(task_types)

    n = len(episodes)
    x = np.zeros((n, n_tt + len(all_memory_ids)))
    y = np.zeros(n)
    for i, ep in enumerate(episodes):
        x[i, tt_index[ep["task_type"]]] = 1.0
        for mem_id, z in ep["included"].items():
            x[i, n_tt + mem_index[mem_id]] = z
        y[i] = ep["success"]
    return x, y, mem_index, n_tt


def compute(
    episodes: list[dict],
    all_memory_ids: list[str],
    l2_c: float = 1.0,
    cross_fit: bool = True,
    n_folds: int = 5,
    cv_seed: int = 0,
) -> dict[str, float]:
    x, y, mem_index, n_tt = _build_feature_matrix(episodes, all_memory_ids)
    n = len(episodes)

    fold_of = np.zeros(n, dtype=int)
    if cross_fit:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
        models = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x)):
            fold_of[test_idx] = fold_idx
            model = LogisticRegression(max_iter=2000, C=l2_c)
            model.fit(x[train_idx], y[train_idx])
            models.append(model)
    else:
        model = LogisticRegression(max_iter=2000, C=l2_c)
        model.fit(x, y)
        models = [model]

    candidate_episode_idx: dict[str, list[int]] = {mem_id: [] for mem_id in all_memory_ids}
    for i, ep in enumerate(episodes):
        for mem_id in ep["candidate_ids"]:
            candidate_episode_idx[mem_id].append(i)

    results = {}
    for mem_id, idxs in candidate_episode_idx.items():
        if not idxs:
            continue
        col = n_tt + mem_index[mem_id]
        idxs_arr = np.array(idxs)
        fold_ids = fold_of[idxs_arr]
        mu1 = np.empty(len(idxs_arr))
        mu0 = np.empty(len(idxs_arr))
        for fold_idx in np.unique(fold_ids):
            pos = fold_ids == fold_idx
            rows = x[idxs_arr[pos]]
            x1 = rows.copy()
            x1[:, col] = 1
            x0 = rows.copy()
            x0[:, col] = 0
            model = models[fold_idx]
            mu1[pos] = model.predict_proba(x1)[:, 1]
            mu0[pos] = model.predict_proba(x0)[:, 1]

        terms = np.empty(len(idxs_arr))
        for k, i in enumerate(idxs_arr):
            ep = episodes[i]
            p = ep["propensities"][mem_id]
            z = ep["included"][mem_id]
            yy = ep["success"]
            terms[k] = (z / p) * (yy - mu1[k]) - ((1 - z) / (1 - p)) * (yy - mu0[k]) + mu1[k] - mu0[k]
        results[mem_id] = float(terms.mean())

    return results
