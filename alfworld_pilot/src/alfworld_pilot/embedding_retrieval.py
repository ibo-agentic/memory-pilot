"""Real sentence-embedding retrieval similarity, replacing `memory_store.py`'s
mock topic-match heuristic (base=0.7 if the memory's tagged task_type matches
the episode's, else 0.3, plus noise) for the Kaggle replication. See the
approved plan (design decision #2) for the full rationale, including why this
module intentionally exposes TWO ways to score similarity rather than one:

  - `embedding_similarity_scores(memories, query_text, embedder)`: real cosine
    similarity between an arbitrary query string and each memory's text. Used
    for MAIN LOGGING episodes, where `env.reset()` has already produced the
    real per-episode ALFWorld goal text (e.g. "put a clean mug in
    coffeemachine") for free -- this is the faithful, per-instance case.

  - `task_type_candidacy_scores(memories, task_type, embedder)`: the same
    scorer, but called with a fixed per-task-type description instead of a
    real goal string. Used ONLY for ground-truth candidacy PROBING
    (`ground_truth_runner._is_natural_candidate`), which must stay cheap (no
    env construction) since most probed task_seeds are rejected before a
    natural candidate is found -- this function needs an answer with only
    `task_type` (derivable from a game's file path) in hand, not a real goal
    string, which would require actually resetting an env.

Both return the same `dict[mem_id, float]` shape `memory_store.similarity_scores`
did, so `retrieval.py`'s top-M selection (which only depends on RELATIVE order,
not absolute scale) is unaffected by which one is used.
"""

from __future__ import annotations

import numpy as np

from .memory_store import Memory

# One short natural-language description per official ALFWorld task type, used
# ONLY for cheap ground-truth candidacy probing (see module docstring) -- these
# are deliberately generic, not tied to any specific game instance, since a
# probe has no game-specific text to work with yet.
TASK_TYPE_QUERY_TEXT: dict[str, str] = {
    # Leads with the distinguishing verb/appliance, not shared "pick up an
    # object... place it in a receptacle" boilerplate -- an earlier version
    # shared that phrasing across all 6 and a smoke test caught it: the
    # distinguishing word was getting diluted by the common boilerplate,
    # occasionally losing to an unrelated task type's memory in cosine
    # similarity. See kaggle_smoke_test.py for the check that caught this.
    "pick_and_place_simple": "move an object to a receptacle, no other action needed",
    "look_at_obj_in_light": "examine an object under a lamp or other light source",
    "pick_clean_then_place_in_recep": "wash or clean an object in the sink before placing it",
    "pick_heat_then_place_in_recep": "heat an object in the microwave before placing it",
    "pick_cool_then_place_in_recep": "cool an object in the fridge before placing it",
    "pick_two_obj_and_place": "find two objects of the same type and place both",
}


class SentenceEmbedder:
    """Thin wrapper around a sentence-transformers model, with an in-memory
    cache keyed by exact text -- each distinct memory text and each of the 6
    task-type query strings is embedded at most once per process, regardless
    of how many episodes reuse it.

    Model id/revision are NOT pinned to a specific commit hash here (unlike
    this project's convention for llm.model_id) because HuggingFace's web page
    didn't expose the exact revision SHA to a simple fetch during planning --
    Phase 0 setup on Kaggle (or wherever this actually runs first) must record
    the exact revision it downloaded and pass it explicitly via `revision=`,
    the same "never trust an implicit default" discipline `llm_client.py`
    already enforces for the LLM. Do not ship a real run against `revision=None`.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", revision: str | None = None, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        if revision is None:
            import warnings

            warnings.warn(
                "SentenceEmbedder constructed with revision=None -- pin the exact "
                "downloaded snapshot's commit hash before any real run (see class docstring).",
                stacklevel=2,
            )
        self.model_name = model_name
        self.revision = revision
        self.model = SentenceTransformer(model_name, revision=revision, device=device)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            vec = self.model.encode(text, normalize_embeddings=True)
            self._cache[text] = np.asarray(vec, dtype=np.float64)
        return self._cache[text]


def embedding_similarity_scores(memories: list[Memory], query_text: str, embedder: SentenceEmbedder) -> dict[str, float]:
    """Real cosine similarity (embeddings are pre-normalized by `embed()`, so
    this reduces to a dot product) between `query_text` and each memory's
    text. Returns raw similarity, typically in [0, 1] for on-topic text --
    no rescaling to the old mock's ~[0.1, 0.9] range is needed since
    `retrieval.py`'s top-M selection only depends on relative order."""
    q = embedder.embed(query_text)
    return {mem.mem_id: float(np.dot(q, embedder.embed(mem.text))) for mem in memories}


def task_type_candidacy_scores(memories: list[Memory], task_type: str, embedder: SentenceEmbedder) -> dict[str, float]:
    """Cheap-probe variant: uses a fixed per-task-type description instead of
    a real per-episode goal string. See module docstring for why this exists
    as a separate function rather than always calling
    `embedding_similarity_scores` with real goal text."""
    query_text = TASK_TYPE_QUERY_TEXT.get(task_type)
    if query_text is None:
        raise ValueError(f"No canonical query text for task_type={task_type!r} -- add one to TASK_TYPE_QUERY_TEXT.")
    return embedding_similarity_scores(memories, query_text, embedder)


def make_similarity_fn(embedder: SentenceEmbedder):
    """Adapts embedding_similarity_scores to episode_runner.run_logged_episode's
    and ground_truth_runner.run_ground_truth_pair's `similarity_fn(memories,
    task_type, obs) -> dict` shape, for real per-episode goal text (obs is
    already available for free once env.reset() has happened)."""

    def _fn(memories: list[Memory], task_type: str, obs: str) -> dict[str, float]:
        return embedding_similarity_scores(memories, obs, embedder)

    return _fn


def make_candidacy_similarity_fn(embedder: SentenceEmbedder):
    """Adapts task_type_candidacy_scores to ground_truth_runner's
    `candidacy_similarity_fn(memories, task_type) -> dict` shape, for the
    cheap (no env construction) candidacy-probing path."""

    def _fn(memories: list[Memory], task_type: str) -> dict[str, float]:
        return task_type_candidacy_scores(memories, task_type, embedder)

    return _fn
