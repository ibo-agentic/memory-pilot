"""Re-exports the EXACT Stage 1 retrieval mechanism (top-M selection, rank-
linear propensity assignment, independent Bernoulli inclusion) from the main
memory_ope package, rather than re-implementing it -- "randomized retrieval
exactly as in Stage 1" per the Part B spec. The two projects use separate
venvs (this one's on Python 3.11 for ALFWorld compatibility, the main one's
on 3.14), so this adds the sibling package's src/ to sys.path instead of a
cross-venv install.
"""

from __future__ import annotations

import pathlib
import sys

_MAIN_SRC = pathlib.Path(__file__).resolve().parents[3] / "src"
if str(_MAIN_SRC) not in sys.path:
    sys.path.insert(0, str(_MAIN_SRC))

from memory_ope.retrieval import retrieve  # noqa: E402  (import after sys.path fix, by design)

__all__ = ["retrieve"]
