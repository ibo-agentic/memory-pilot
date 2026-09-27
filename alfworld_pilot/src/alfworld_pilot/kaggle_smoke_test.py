"""Local, pre-Kaggle smoke test for the Kaggle-replication pieces -- run this
BEFORE ever touching a Kaggle session, per the approved plan's verification
step 1 ("unit-test embedding_retrieval.py's similarity function and
kaggle_memory_store.py's content locally, no GPU needed").

Not part of `pytest tests/` (the main, fast, zero-heavy-deps suite) on
purpose -- this needs sentence-transformers (and, for --full, torch/
transformers/a real local model + a GPU), which this project deliberately
keeps out of the main venv/requirements to avoid disturbing the paid run's
already-verified, already-cited exact package versions (see
results/paper_data.md §1.6 and §7 item 12). Run this from the dedicated
alfworld_pilot/.venv-kaggle venv instead.

Usage:
    .venv-kaggle/bin/python -m alfworld_pilot.kaggle_smoke_test            # embeddings only, no GPU
    .venv-kaggle/bin/python -m alfworld_pilot.kaggle_smoke_test --full     # + one real local-model episode + determinism check (needs a GPU)
"""

from __future__ import annotations

import argparse
import sys


def check_embeddings() -> bool:
    print("=== 1. Embedding retrieval sanity check (no GPU needed) ===")
    from .embedding_retrieval import SentenceEmbedder, embedding_similarity_scores, task_type_candidacy_scores
    from .kaggle_memory_store import CATEGORY_BY_MEMORY_ID, build_kaggle_store

    memories = build_kaggle_store()
    print(f"Built {len(memories)} memories: "
          f"{sum(1 for c in CATEGORY_BY_MEMORY_ID.values() if c == 'correct')} correct, "
          f"{sum(1 for c in CATEGORY_BY_MEMORY_ID.values() if c == 'harmful')} harmful, "
          f"{sum(1 for c in CATEGORY_BY_MEMORY_ID.values() if c == 'partial')} partial, "
          f"{sum(1 for c in CATEGORY_BY_MEMORY_ID.values() if c == 'irrelevant')} irrelevant.")
    lengths = [m.approx_tokens for m in memories]
    print(f"Lengths: mean={sum(lengths)/len(lengths):.1f}, min={min(lengths)}, max={max(lengths)} words "
          f"(should show real variation, not a uniform pad -- contrast with the paid run's natural "
          f"store, which was 100-120 words for every one of 30 memories).")

    embedder = SentenceEmbedder(revision=None, device="cpu")  # revision=None is fine for THIS smoke test; pin before a real run

    all_pass = True
    # Queries deliberately avoid name-dropping another task type's action word
    # or appliance (an earlier version used ALFWorld mock-template goal
    # strings like "cool a tomato and put it in the SINK" -- the literal word
    # "sink" pulled similarity toward the cleaning memories over the cooling
    # ones, which is a real property of short-text embedding similarity, not
    # a bug, but made those specific queries a bad test of task-type
    # separation. See kaggle_smoke_test.py's own history / README for this).
    checks = [
        ("put the pillow on the couch", "pas_correct", "heat_correct"),
        ("heat the mug in the microwave and put it on the counter", "heat_correct", "cool_correct"),
        ("cool the tomato in the fridge and put it on the table", "cool_correct", "heat_correct"),
        ("wash the plate in the sink and put it in the cupboard", "clean_correct", "two_correct"),
        ("put both spoons in the drawer", "two_correct", "heat_correct"),
        ("examine the statue under the desk lamp", "light_correct", "clean_correct"),
    ]
    print("\nPer-query check: does the matching task type's own memory clearly outrank a")
    print("DIFFERENT, unrelated task type's memory? (top-3 is printed for visibility only --")
    print("not gated on, since two semantically related task types, e.g. heat/cool, crowding")
    print("each other out of a top-3 is a real and acceptable property of real embeddings, not")
    print("a failure -- the meaningful check is the pairwise comparison against something")
    print("genuinely unrelated):")
    for query, expected_top, should_rank_below in checks:
        scores = embedding_similarity_scores(memories, query, embedder)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_ids = [mid for mid, _ in ranked[:3]]
        ok = scores[expected_top] > scores[should_rank_below]
        all_pass &= ok
        print(f"  {'OK  ' if ok else 'FAIL'} query={query!r} -> top3={top_ids} "
              f"({expected_top}={scores[expected_top]:.3f} vs {should_rank_below}={scores[should_rank_below]:.3f})")

    print("\nCandidacy-probe (task-type-keyed) scores, same sanity pattern, cheaper query text:")
    for task_type, expected_prefix in [
        ("pick_heat_then_place_in_recep", "heat_"),
        ("pick_two_obj_and_place", "two_"),
    ]:
        scores = task_type_candidacy_scores(memories, task_type, embedder)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_id = ranked[0][0]
        ok = top_id.startswith(expected_prefix)
        all_pass &= ok
        print(f"  {'OK  ' if ok else 'FAIL'} task_type={task_type!r} -> top1={top_id}")

    print(f"\n=> Embedding checks {'ALL PASSED' if all_pass else 'HAD FAILURES -- inspect before proceeding'}")
    return all_pass


def check_full_pipeline() -> bool:
    print("\n=== 2. Full pipeline: real ALFWorld + real local model + real embeddings, a few episodes ===")
    import pathlib

    from . import config as config_mod
    from .cache import LLMCache
    from .cost_tracker import CostTracker
    from .determinism_check import check_determinism
    from .embedding_retrieval import SentenceEmbedder, make_similarity_fn
    from .env_factory import build_task_source
    from .episode_runner import run_logged_episode
    from .kaggle_memory_store import build_kaggle_store
    from .local_model_client import LocalTransformersClient

    cfg_path = pathlib.Path(__file__).resolve().parents[2] / "kaggle_config.yaml"
    cfg = config_mod.load_config(cfg_path)
    config_mod.ensure_dirs(cfg)

    memories = build_kaggle_store()
    task_source = build_task_source(cfg)
    embedder = SentenceEmbedder(model_name=cfg["embedding"]["model_name"], revision=cfg["embedding"].get("revision"), device="cpu")
    similarity_fn = make_similarity_fn(embedder)

    cache = LLMCache(cfg["cache"]["dir"])
    cost_tracker = CostTracker(cfg["cost_control"]["hard_cap_usd"], 0.0, 0.0)  # no state_path -- this is a throwaway smoke test
    llm_cfg = cfg["llm"]
    print(f"Loading {llm_cfg['model_id']}@{llm_cfg['revision']} ({'4-bit' if llm_cfg.get('quantize_4bit', True) else 'fp16'})... this can take a few minutes the first time (downloads ~15GB).")
    client = LocalTransformersClient(
        model_id=llm_cfg["model_id"], revision=llm_cfg["revision"], temperature=llm_cfg["temperature"],
        max_output_tokens=llm_cfg["max_output_tokens"], cache=cache, cost_tracker=cost_tracker,
        device=llm_cfg.get("device", "cuda"), quantize_4bit=llm_cfg.get("quantize_4bit", True),
    )
    print("Model loaded.")

    print("\n--- Determinism check (never run against the paid model -- running it now) ---")
    probe_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with exactly one word: hello"},
    ]
    det = check_determinism(client, probe_messages)
    print(f"determinism_check.check_determinism -> holds={det['holds']}")
    print(f"  response_1={det['response_1']!r}")
    print(f"  response_2={det['response_2']!r}")
    if not det["holds"]:
        print("  NOTE: this is itself a real, reportable finding (per the plan) -- local greedy decoding "
              "is not guaranteed bit-identical across calls on this hardware/stack. Report it, don't hide it.")

    print(f"\n--- Running {3} real episodes end to end ---")
    for task_id in range(3):
        env = task_source.build_env(task_id)
        try:
            import random

            ep = run_logged_episode(
                env, client, memories, cfg["retrieval"]["M"], cfg["retrieval"]["propensity_min"],
                cfg["retrieval"]["propensity_max"], cfg["env"]["max_steps"], task_id=task_id,
                rng=random.Random(task_id), similarity_fn=similarity_fn,
            )
        finally:
            env.close()
        print(
            f"  episode {task_id}: task_type={ep['task_type']} success={ep['success']} "
            f"steps={ep['steps_taken']} n_calls={ep['n_llm_calls']} candidates={ep['candidate_ids']} "
            f"included={[k for k, v in ep['included'].items() if v == 1]}"
        )

    print(f"\nCost tracker summary (should show real token counts, $0 cost): {cost_tracker.summary()}")
    return det["holds"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="also run the full pipeline (real ALFWorld + real local model) -- needs a GPU")
    args = p.parse_args()

    ok = check_embeddings()
    if args.full:
        ok = check_full_pipeline() and ok

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
