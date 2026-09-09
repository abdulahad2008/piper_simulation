"""Import ordering guard for the whole test suite.

OSMesa (the headless GL backend gym-hil needs) and triton (pulled in by torch)
each load their own LLVM. If the OSMesa context is created first, importing
torch afterwards segfaults the interpreter during pytest's collection phase,
which looks like a crashed test run rather than a failing test. Importing
torch first is enough to make the order deterministic.
"""

try:
    import torch  # noqa: F401
    import triton  # noqa: F401  -- torch imports it lazily; force it now
except Exception:
    pass
