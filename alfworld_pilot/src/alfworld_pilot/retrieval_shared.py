"""Re-exports the EXACT Stage 1 retrieval mechanism (top-M selection, rank-
linear propensity assignment, independent Bernoulli inclusion) from the main
memory_ope package, rather than re-implementing it -- "randomized retrieval
exactly as in Stage 1" per the Part B spec. `memory_ope` is now an installed
package (`pip install -e .` at the repo root), so this is a plain import --
no cross-venv sys.path shim needed, as long as whichever venv runs this has
also run `pip install -e <repo_root>`.
"""

from __future__ import annotations

from memory_ope.retrieval import retrieve

__all__ = ["retrieve"]
