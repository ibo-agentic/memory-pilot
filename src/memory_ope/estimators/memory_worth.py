"""Memory Worth baseline (Simsek 2026): success rate over episodes where the
memory was actually included, i.e. hits+(m) / (hits+(m) + hits-(m)).
Defaults to 0.5 for a memory with no inclusion data.
"""

from __future__ import annotations

from collections import defaultdict


def compute(episodes: list[dict]) -> dict[str, float]:
    hits_pos: dict[str, int] = defaultdict(int)
    hits_neg: dict[str, int] = defaultdict(int)
    all_ids: set[str] = set()

    for ep in episodes:
        y = ep["success"]
        all_ids.update(ep["candidate_ids"])
        for mem_id, z in ep["included"].items():
            if z == 1:
                if y == 1:
                    hits_pos[mem_id] += 1
                else:
                    hits_neg[mem_id] += 1

    result = {}
    for mem_id in all_ids:
        pos, neg = hits_pos.get(mem_id, 0), hits_neg.get(mem_id, 0)
        result[mem_id] = 0.5 if (pos + neg) == 0 else pos / (pos + neg)
    return result
