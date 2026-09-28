"""Backend parity check (2026-09-29, Task 3): with the SAME task_ids (same
seed -> same retrieval draw, same underlying ALFWorld game instance), does
the optional vLLM backend produce the SAME action sequence as the existing
transformers backend, over N episodes?

Compares the PARSED ACTION at each step, not raw completion text -- two
different inference backends can legitimately produce different tokens at
the margin even under greedy (temperature=0) decoding, due to different
numerical kernels/reduction order (the same caveat local_model_client.py's
own docstring already raises for GPU matmul non-determinism within a single
backend). What actually matters for this project is whether the two
backends drive the environment identically, i.e. whether every parsed
action matches -- not byte-identical model output.

Must run on the real Kaggle T4 session (needs both `transformers` and
`vllm` installed and a GPU) -- NOT run locally as part of this change; see
kaggle/timing_probe.ipynb's parity-check cell, which is the actual
execution of this script.

Usage (on Kaggle, from alfworld_pilot/):
    PYTHONPATH=src python -m alfworld_pilot.vllm_parity_check --n-episodes 5
"""

from __future__ import annotations

import argparse
import pathlib
import random

from . import config as config_mod
from . import react_agent
from .capability_check import _single_type_source, build_local_client
from .embedding_retrieval import SentenceEmbedder, embedding_similarity_scores
from .env_factory import load_real_alfworld_config
from .kaggle_memory_store import build_kaggle_store

ALFWORLD_PILOT_DIR = pathlib.Path(__file__).resolve().parents[2]


def run_one_episode(source, task_id: int, client, memories, embedder, m, prop_min, prop_max, max_steps) -> list[str]:
    from memory_ope.retrieval import retrieve as ope_retrieve

    env = source.build_env(task_id)
    try:
        obs, info = env.reset(task_seed=task_id)
        rng = random.Random(task_id)
        sims = embedding_similarity_scores(memories, obs, embedder)
        candidate_ids, _props, included = ope_retrieve(sims, m, prop_min, prop_max, rng)
        mem_by_id = {mem.mem_id: mem for mem in memories}
        included_texts = [mem_by_id[mid].text for mid in candidate_ids if included[mid] == 1]
        result = react_agent.run_episode(env, client, included_texts, max_steps, obs, info)
    finally:
        env.close()
    return [s.action for s in result.steps]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-type", default="pick_and_place_simple")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--start-seed", type=int, default=2000)  # past capability_check's and timing_probe's own ranges
    args = p.parse_args()

    cfg_path = ALFWORLD_PILOT_DIR / "kaggle_config.yaml"
    cfg = config_mod.load_config(cfg_path)
    real_cfg = load_real_alfworld_config(cfg["env"]["real_alfworld_config_path"])
    split = cfg["env"]["real_split"]
    source = _single_type_source(real_cfg, split, args.task_type)

    memories = build_kaggle_store()
    embedder = SentenceEmbedder(
        model_name=cfg["embedding"]["model_name"], revision=cfg["embedding"].get("revision"), device="cpu"
    )
    m = cfg["retrieval"]["M"]
    prop_min = cfg["retrieval"]["propensity_min"]
    prop_max = cfg["retrieval"]["propensity_max"]
    max_steps = cfg["env"]["max_steps"]

    print("Loading transformers backend...")
    tf_client, _ = build_local_client(cfg)

    print("Loading vLLM backend...")
    from .vllm_client import build_vllm_client

    vllm_client, _ = build_vllm_client(cfg)

    mismatches = []
    for i in range(args.n_episodes):
        task_id = args.start_seed + i
        tf_actions = run_one_episode(source, task_id, tf_client, memories, embedder, m, prop_min, prop_max, max_steps)
        vllm_actions = run_one_episode(source, task_id, vllm_client, memories, embedder, m, prop_min, prop_max, max_steps)
        match = tf_actions == vllm_actions
        print(f"task_id={task_id}: {'MATCH' if match else 'MISMATCH'} (transformers {len(tf_actions)} steps, vllm {len(vllm_actions)} steps)")
        if not match:
            mismatches.append({"task_id": task_id, "transformers_actions": tf_actions, "vllm_actions": vllm_actions})

    n_match = args.n_episodes - len(mismatches)
    print(f"\n{n_match}/{args.n_episodes} episodes matched exactly.")
    if mismatches:
        print("MISMATCHES -- do not treat the two backends as interchangeable for a real run without investigating why:")
        for mm in mismatches:
            print(f"  task_id={mm['task_id']}")
            print(f"    transformers: {mm['transformers_actions']}")
            print(f"    vllm:         {mm['vllm_actions']}")


if __name__ == "__main__":
    main()
