"""Effective sample size (ESS) for the IPS importance weights used by each
memory's estimate. Standard Kish ESS: (sum w)^2 / sum(w^2). Computed
separately for the "included" arm (weight 1/p when Z=1) and the "not
included" arm (weight 1/(1-p) when Z=0), since each feeds a separate term of
the IPS/SNIPS contrast.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def effective_sample_size(episodes: list[dict]) -> dict[str, dict[str, float]]:
    pos_weights: dict[str, list[float]] = defaultdict(list)
    neg_weights: dict[str, list[float]] = defaultdict(list)

    for ep in episodes:
        props = ep["propensities"]
        included = ep["included"]
        for mem_id in ep["candidate_ids"]:
            p = props[mem_id]
            z = included[mem_id]
            if z == 1:
                pos_weights[mem_id].append(1.0 / p)
            else:
                neg_weights[mem_id].append(1.0 / (1.0 - p))

    result = {}
    all_ids = set(pos_weights) | set(neg_weights)
    for mem_id in all_ids:
        pw = np.array(pos_weights.get(mem_id, []))
        nw = np.array(neg_weights.get(mem_id, []))
        ess_pos = float((pw.sum() ** 2) / (pw ** 2).sum()) if pw.size else 0.0
        ess_neg = float((nw.sum() ** 2) / (nw ** 2).sum()) if nw.size else 0.0
        result[mem_id] = {
            "ess_pos": ess_pos,
            "ess_neg": ess_neg,
            "n_pos": int(pw.size),
            "n_neg": int(nw.size),
        }
    return result


def summarize_ess(ess: dict[str, dict[str, float]]) -> dict[str, float]:
    ess_pos = np.array([v["ess_pos"] for v in ess.values()])
    ess_neg = np.array([v["ess_neg"] for v in ess.values()])
    n_pos = np.array([v["n_pos"] for v in ess.values()])
    n_neg = np.array([v["n_neg"] for v in ess.values()])
    return {
        "mean_ess_pos": float(ess_pos.mean()),
        "mean_ess_neg": float(ess_neg.mean()),
        "mean_ess_pos_ratio": float((ess_pos / np.maximum(n_pos, 1)).mean()),
        "mean_ess_neg_ratio": float((ess_neg / np.maximum(n_neg, 1)).mean()),
    }
