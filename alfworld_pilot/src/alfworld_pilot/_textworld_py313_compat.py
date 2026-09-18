"""Compatibility shim for textworld==1.7.0 on Python >=3.13.

textworld's PDDL grammar engine (textworld.envs.pddl.textgen.EvalSymbol.derive,
used to render EVERY piece of generated game text, including the very first
`env.reset()` call) does:

    locals().update(context["variables"])
    value = eval(self.expression)

relying on `eval()` with no explicit namespace picking up the CALLING frame's
locals, and on `locals().update()` actually mutating that frame's namespace.
That second part was already undefined behavior in CPython, but historically
"worked" for injecting extra names because `locals()` returned a semi-live
dict tied to the frame. PEP 667 (Python 3.13, "Consistent views of
namespaces") makes `locals()` in a function always return a fresh, discarded
snapshot -- so the update is silently lost, and `eval()`'s own default
`locals()` call gets a snapshot without `context["variables"]` in it,
raising `NameError` for every name that dict was supposed to supply (e.g.
`NameError: name 'r' is not defined` on the very first `env.reset()`).

No fix exists yet in textworld 1.7.0 (the latest release as of 2026-09-18;
confirmed by grepping the installed package -- `locals().update(` appears
exactly once, at this call site, nowhere else). Verified upstream via
PEP 667 itself: the documented migration is to pass an explicit namespace
to `eval()`/`exec()` instead of relying on the old locals()-mutation trick.
This monkeypatch does exactly that, reproducing the pre-3.13 behavior
(module globals + the extra `context["variables"]` names) without touching
the installed package's files, so `pip install -U textworld` cleanly
replaces it if upstream ships a real fix later.

Call `apply()` once, before constructing any AlfredTWEnv (env_interface.py
does this at import time).
"""

from __future__ import annotations

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return

    import textworld.envs.pddl.textgen as textgen

    module_globals = textgen.__dict__

    def _patched_derive(self, context=None):
        context = context or self.context
        value = eval(self.expression, module_globals, dict(context["variables"]))
        return [textgen.TerminalSymbol(value)]

    textgen.EvalSymbol.derive = _patched_derive
    _applied = True
