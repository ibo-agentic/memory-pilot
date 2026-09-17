"""Inverse propensity scoring (IPS) and self-normalized IPS (SNIPS) for the
per-memory causal contrast E[Y(1)] - E[Y(0)], using the known inclusion
propensities p_i logged for every candidate episode.

IPS(m)   = mean over episodes where m is a candidate of [Z*Y/p - (1-Z)*Y/(1-p)]
SNIPS(m) = (sum Z*Y/p) / (sum Z/p)  -  (sum (1-Z)*Y/(1-p)) / (sum (1-Z)/(1-p))
"""

from __future__ import annotations

from collections import defaultdict


def compute(episodes: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    ips_sum: dict[str, float] = defaultdict(float)
    ips_count: dict[str, int] = defaultdict(int)
    pos_weight_sum: dict[str, float] = defaultdict(float)
    neg_weight_sum: dict[str, float] = defaultdict(float)
    pos_term_sum: dict[str, float] = defaultdict(float)
    neg_term_sum: dict[str, float] = defaultdict(float)

    for ep in episodes:
        y = ep["success"]
        props = ep["propensities"]
        included = ep["included"]
        for mem_id in ep["candidate_ids"]:
            p = props[mem_id]
            z = included[mem_id]
            pos_term = (z * y) / p
            neg_term = ((1 - z) * y) / (1 - p)
            ips_sum[mem_id] += pos_term - neg_term
            ips_count[mem_id] += 1
            pos_weight_sum[mem_id] += z / p
            neg_weight_sum[mem_id] += (1 - z) / (1 - p)
            pos_term_sum[mem_id] += pos_term
            neg_term_sum[mem_id] += neg_term

    ips_result = {mem_id: ips_sum[mem_id] / ips_count[mem_id] for mem_id in ips_count}

    snips_result = {}
    for mem_id in ips_count:
        pos = pos_term_sum[mem_id] / pos_weight_sum[mem_id] if pos_weight_sum[mem_id] > 0 else 0.0
        neg = neg_term_sum[mem_id] / neg_weight_sum[mem_id] if neg_weight_sum[mem_id] > 0 else 0.0
        snips_result[mem_id] = pos - neg

    return ips_result, snips_result
