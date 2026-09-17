# Stage 2 — ALFWorld pilot (not yet implemented)

This package is a placeholder. Stage 2 is gated on approval of the Stage 1
synthetic results (see `../results/stage1_report.json` and the root
`README.md`).

When approved, this will hold:
- A ReAct-style agent wired to [ALFWorld](https://github.com/alfworld/alfworld)
  (official install steps only — see that repo, not restated here).
- A ~50-trajectory memory store (successful + failed episodes).
- The same randomized top-M / known-propensity retrieval design as Stage 1,
  logging every episode to `../logs/` as JSONL.
- Forced-in/forced-out ground-truth reruns for 10 chosen memories.
- An LLM call cache (`../cache/`) and a cost-limit / dry-run mode, so no
  rerun costs money twice and so a run's expected LLM-call count is known
  before it starts.

None of this is implemented yet.
