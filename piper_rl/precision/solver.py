"""Contact-solver overrides -- the H0d kill gate.

The published claim of the precision-floor paper is that the floor is set by
the action interface and not by the simulator. That claim is falsifiable only
if the solver knobs most likely to shape terminal contact can actually be
moved, so they are exposed here in one place.

Everything is applied to an already-loaded ``MjModel``:

    timestep     model.opt.timestep      -- integration step (s)
    iterations   model.opt.iterations    -- Newton solver iterations
    impratio     model.opt.impratio      -- normal/friction impedance ratio
    solref_scale geom_solref[:, 0]       -- contact time constant of the object
                                           and the two finger pads, scaled;
                                           dampratio (column 1) is preserved.

``n_substeps`` in the env is recomputed from the new timestep by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import numpy as np


@dataclass
class SolverOverride:
    """One cell of the solver kill gate. ``None`` means 'leave the MJCF value'."""

    timestep: Optional[float] = None       # s   (MJCF default 0.002)
    iterations: Optional[int] = None       # -   (MJCF default 100)
    impratio: Optional[float] = None       # -   (MJCF default 10)
    solref_scale: float = 1.0              # multiplies solref timeconst

    def is_identity(self) -> bool:
        return (self.timestep is None and self.iterations is None
                and self.impratio is None and self.solref_scale == 1.0)

    def tag(self) -> str:
        if self.is_identity():
            return "nominal"
        parts = []
        if self.timestep is not None:
            parts.append(f"dt{self.timestep * 1000:g}ms")
        if self.iterations is not None:
            parts.append(f"it{self.iterations}")
        if self.impratio is not None:
            parts.append(f"imp{self.impratio:g}")
        if self.solref_scale != 1.0:
            parts.append(f"solref{self.solref_scale:g}x")
        return "_".join(parts)

    def as_dict(self) -> dict:
        return asdict(self)


def apply_solver_overrides(model, spec: SolverOverride,
                           contact_geom_ids: Iterable[int] = ()) -> dict:
    """Mutate ``model`` in place. Returns the resolved values, for the log.

    ``contact_geom_ids`` are the geoms whose ``solref`` timeconst is scaled --
    for this task, the object geom and the four finger-pad geoms.
    """
    resolved = {
        "solver_timestep": float(model.opt.timestep),
        "solver_iterations": int(model.opt.iterations),
        "solver_impratio": float(model.opt.impratio),
        "solver_solref_scale": float(spec.solref_scale),
    }
    if spec.timestep is not None:
        model.opt.timestep = float(spec.timestep)
        resolved["solver_timestep"] = float(spec.timestep)
    if spec.iterations is not None:
        model.opt.iterations = int(spec.iterations)
        resolved["solver_iterations"] = int(spec.iterations)
    if spec.impratio is not None:
        model.opt.impratio = float(spec.impratio)
        resolved["solver_impratio"] = float(spec.impratio)
    if spec.solref_scale != 1.0:
        for gid in contact_geom_ids:
            model.geom_solref[gid, 0] = \
                float(model.geom_solref[gid, 0]) * float(spec.solref_scale)
    return resolved


# The nine-cell eval-only kill gate of the paper's Phase 0.
KILL_GATE_CELLS = [
    SolverOverride(),                                   # nominal
    SolverOverride(timestep=0.001),
    SolverOverride(timestep=0.004),
    SolverOverride(solref_scale=0.5),
    SolverOverride(solref_scale=2.0),
    SolverOverride(iterations=20),
    SolverOverride(impratio=1.0),
    SolverOverride(timestep=0.001, solref_scale=0.5),   # extreme A
    SolverOverride(timestep=0.004, solref_scale=2.0),   # extreme B
]
