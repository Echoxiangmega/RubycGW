"""High-level workflow for finite-size ED of the strong-coupling pseudospin model."""
from __future__ import annotations

from dataclasses import dataclass

from ..models import RubyModel
from ..solvers.effective import EffectiveEDResult, solve_effective_pseudospin_ed


@dataclass(frozen=True)
class EffectiveEDConfig:
    Lx: int = 3
    Ly: int = 3
    nev: int = 4
    tol: float = 1e-10
    maxiter: int | None = None
    degeneracy_tol: float = 1e-7


def run_effective_ed(
    model: RubyModel,
    config: EffectiveEDConfig = EffectiveEDConfig(),
) -> EffectiveEDResult:
    """Project model parameters to ``Jn,Jm,Jz`` and solve the pseudospin ED."""
    jn, jm, jz = model.effective_couplings()
    return solve_effective_pseudospin_ed(
        Lx=int(config.Lx),
        Ly=int(config.Ly),
        Jn=float(jn),
        Jm=float(jm),
        Jz=float(jz),
        nev=int(config.nev),
        tol=float(config.tol),
        maxiter=config.maxiter,
        degeneracy_tol=float(config.degeneracy_tol),
    )


__all__ = ["EffectiveEDConfig", "run_effective_ed"]
