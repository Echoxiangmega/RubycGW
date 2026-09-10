"""Full two-dimensional Fierz-PMS closure at fixed physical V.

Coordinates are

    s = lambda_B + lambda_J,
    a = lambda_B - lambda_J,

with lambda_n=1-s.  The longitudinal helper already solves
[R_GW,dF/ds]=0 for prescribed a.  This module promotes a to an additional
unknown and appends dF/da to obtain the square system

    R_GW = 0,
    dF/ds = 0,
    dF/da = 0.

A root is therefore a genuine interior stationary point of the GW
Luttinger-Ward functional in the full n/B/J Fierz simplex.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fierz_pms_transverse import (
    LongitudinalFierzPMSResidual,
    LongitudinalPMSEvaluation,
    TransversePMSDiagnostics,
)
from .grids import MatsubaraGrid


@dataclass
class FullSimplexPMSEvaluation:
    residual: np.ndarray
    longitudinal: LongitudinalPMSEvaluation
    transverse: TransversePMSDiagnostics
    a: float
    pms_scale: float

    @property
    def weights(self):
        return self.longitudinal.weights

    @property
    def s(self) -> float:
        return float(self.longitudinal.s)

    @property
    def free_energy(self):
        return self.longitudinal.free_energy

    @property
    def physical_residual(self) -> float:
        return float(self.longitudinal.physical_residual)

    @property
    def filling_error(self) -> float:
        return float(self.longitudinal.filling_error)

    @property
    def smin(self) -> float:
        return float(self.longitudinal.smin)


class FullSimplexFierzPMSResidual:
    """Solve [R_GW,dF/ds,dF/da]=0 with (s,a) both self-consistent."""

    def __init__(
        self,
        h0: np.ndarray,
        interaction_pairs,
        grid: MatsubaraGrid,
        target: float,
        *,
        V: float,
        primitive_cells: int,
        fd_h: float = 2e-3,
        simplex_margin: float = 1e-8,
        free_energy_scale_floor: float = 1e-4,
    ):
        self.V = float(V)
        self.free_energy_scale_floor = float(free_energy_scale_floor)
        self.longitudinal = LongitudinalFierzPMSResidual(
            h0,
            interaction_pairs,
            grid,
            target,
            V=self.V,
            primitive_cells=primitive_cells,
            fd_h=fd_h,
            simplex_margin=simplex_margin,
            free_energy_scale_floor=free_energy_scale_floor,
        )
        self.base_codec = self.longitudinal.base_codec
        self.simplex_margin = float(simplex_margin)

    @property
    def size(self) -> int:
        return int(self.longitudinal.codec.size) + 1

    def encode(self, x: np.ndarray, s: float, a: float) -> np.ndarray:
        y = self.longitudinal.encode(x, float(s), float(a))
        return np.concatenate([np.asarray(y, dtype=float), [float(a)]])

    def decode(self, z: np.ndarray):
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.size != self.size:
            raise ValueError(f"full-PMS state size {z.size} != {self.size}")
        a = float(z[-1])
        if not np.isfinite(a) or abs(a) >= 1.0 - self.simplex_margin:
            raise ValueError("full-PMS transverse coordinate left the simplex")
        y = np.asarray(z[:-1], dtype=float)
        sigma_static, sigma_c, mu, s, _, u = self.longitudinal.decode(y, a)
        return sigma_static, sigma_c, mu, float(s), a, float(u), y

    def evaluate(self, z: np.ndarray) -> FullSimplexPMSEvaluation:
        *_, a, _, y = self.decode(z)
        ev = self.longitudinal.evaluate(y, a)
        diag = self.longitudinal.transverse_diagnostics(y, a, evaluation=ev)
        scale = max(self.V * self.V, self.free_energy_scale_floor)
        residual = np.concatenate([
            np.asarray(ev.residual, dtype=float),
            [float(diag.dF_da_per_cell) / scale],
        ])
        return FullSimplexPMSEvaluation(
            residual=np.asarray(residual, dtype=float),
            longitudinal=ev,
            transverse=diag,
            a=float(a),
            pms_scale=float(scale),
        )

    def __call__(self, z: np.ndarray, _dummy_parameter: float = 0.0) -> np.ndarray:
        return self.evaluate(z).residual


__all__ = ["FullSimplexPMSEvaluation", "FullSimplexFierzPMSResidual"]
