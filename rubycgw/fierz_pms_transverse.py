"""Transverse diagnostics around one-dimensional Fierz-PMS roots.

Use coordinates

    s = lambda_B + lambda_J,
    a = lambda_B - lambda_J,

so that

    lambda_n = 1 - s,
    lambda_B = (s+a)/2,
    lambda_J = (s-a)/2.

The existing one-parameter PMS family is the line a=0 and solves dF/ds=0
(equivalently dF/dlambda_n=0).  For a prescribed transverse coordinate ``a``
this module promotes ``s`` to an unknown together with the fermionic GW state
and solves

    R_GW = 0,
    dF/ds = 0.

Scanning ``a`` therefore follows the *longitudinally stationary manifold* rather
than keeping s artificially fixed.  A full two-dimensional Fierz stationary
point is encountered when, in addition, dF/da=0.

All Fierz derivatives are explicit derivatives of the stationary
Luttinger-Ward functional at fixed fermionic G.  The self-consistent GW state is
re-solved at every accepted value of a by the calling scan driver.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .fierz_free_energy import ChannelGWFreeEnergyResult, evaluate_channel_gw_free_energy
from .fierz_mixed import (
    FierzWeights,
    WeightedFierzGWResidual,
    build_weighted_nbj_definition,
    validate_weights,
)
from .grids import MatsubaraGrid


def weights_from_s_a(s: float, a: float, *, atol: float = 2e-12) -> FierzWeights:
    """Convert longitudinal/transverse Fierz coordinates to n/B/J weights."""
    s = float(s)
    a = float(a)
    if not np.isfinite(s) or not np.isfinite(a):
        raise ValueError("s and a must be finite")
    return validate_weights(
        FierzWeights(1.0 - s, 0.5 * (s + a), 0.5 * (s - a)),
        atol=atol,
    )


def s_a_from_weights(weights: FierzWeights) -> tuple[float, float]:
    """Return (s,a) for validated n/B/J Fierz weights."""
    w = validate_weights(weights)
    return float(w.bond + w.current), float(w.bond - w.current)


def _sigmoid(u: float) -> float:
    u = float(u)
    if u >= 0.0:
        e = np.exp(-min(u, 745.0))
        return float(1.0 / (1.0 + e))
    e = np.exp(max(u, -745.0))
    return float(e / (1.0 + e))


@dataclass(frozen=True)
class LongitudinalPMSCodec:
    """Append a bounded longitudinal coordinate s for a prescribed a.

    At fixed ``a`` the interior simplex interval is |a| < s < 1.  A logit
    coordinate keeps Newton-Krylov trial states inside that interval.
    """

    base_codec: object
    margin: float = 1e-8

    @property
    def size(self) -> int:
        return int(self.base_codec.size) + 1

    def bounds(self, a: float) -> tuple[float, float]:
        m = float(self.margin)
        lo = abs(float(a)) + m
        hi = 1.0 - m
        if not lo < hi:
            raise ValueError("|a| is too close to the simplex boundary")
        return float(lo), float(hi)

    def encode(self, x: np.ndarray, s: float, a: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != int(self.base_codec.size):
            raise ValueError("base Fierz state has the wrong size")
        lo, hi = self.bounds(a)
        s = float(s)
        if not lo < s < hi:
            raise ValueError(f"s must lie strictly inside ({lo:g},{hi:g}) at a={float(a):g}")
        q = (s - lo) / (hi - lo)
        u = np.log(q / (1.0 - q))
        return np.concatenate([x, [float(u)]])

    def decode(self, y: np.ndarray, a: float) -> tuple[np.ndarray, float, float]:
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != self.size:
            raise ValueError(f"transverse-PMS state size {y.size} != {self.size}")
        lo, hi = self.bounds(a)
        u = float(y[-1])
        s = lo + (hi - lo) * _sigmoid(u)
        return np.asarray(y[:-1], dtype=float), float(s), u


@dataclass
class LongitudinalPMSEvaluation:
    residual: np.ndarray
    base_evaluation: object
    weights: FierzWeights
    s: float
    a: float
    s_logit: float
    dF_ds_per_cell: float
    d2F_ds2_per_cell: float
    free_energy: ChannelGWFreeEnergyResult
    fd_h_s_used: float
    pms_scaled_residual: float

    @property
    def smin(self) -> float:
        return float(self.base_evaluation.smin)

    @property
    def physical_residual(self) -> float:
        return float(self.base_evaluation.physical_residual)

    @property
    def filling_error(self) -> float:
        return float(self.base_evaluation.filling_error)


@dataclass(frozen=True)
class TransversePMSDiagnostics:
    """Free-energy derivatives at a longitudinally stationary point."""

    dF_ds_per_cell: float
    dF_da_per_cell: float
    dF_dlambda_B_per_cell: float
    dF_dlambda_J_per_cell: float
    d2F_da2_per_cell: float
    gradient_norm_per_cell: float
    fd_h_a_used: float


class LongitudinalFierzPMSResidual:
    """Solve [R_GW, dF/ds]=0 at fixed V and externally prescribed a."""

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
        self.h0 = np.asarray(h0, dtype=complex)
        self.pairs = tuple((int(i), int(j)) for i, j in interaction_pairs)
        self.grid = grid
        self.target = float(target)
        self.V = float(V)
        self.primitive_cells = int(primitive_cells)
        self.fd_h = float(fd_h)
        self.simplex_margin = float(simplex_margin)
        self.free_energy_scale_floor = float(free_energy_scale_floor)
        if self.primitive_cells < 1:
            raise ValueError("primitive_cells must be positive")
        if not np.isfinite(self.fd_h) or self.fd_h <= 0.0:
            raise ValueError("fd_h must be positive")
        if not 0.0 <= self.simplex_margin < 0.1:
            raise ValueError("simplex_margin must lie in [0,0.1)")
        if not np.isfinite(self.free_energy_scale_floor) or self.free_energy_scale_floor <= 0.0:
            raise ValueError("free_energy_scale_floor must be positive")

        probe = WeightedFierzGWResidual(
            self.h0,
            self.pairs,
            FierzWeights(0.5, 0.25, 0.25),
            self.grid,
            self.target,
        )
        self.base_codec = probe.codec
        self.codec = LongitudinalPMSCodec(self.base_codec, margin=self.simplex_margin)
        self.norb = int(self.h0.shape[-1])

    def _problem(self, weights: FierzWeights) -> WeightedFierzGWResidual:
        return WeightedFierzGWResidual(
            self.h0,
            self.pairs,
            validate_weights(weights),
            self.grid,
            self.target,
        )

    def encode(self, x: np.ndarray, s: float, a: float) -> np.ndarray:
        return self.codec.encode(x, s, a)

    def decode(self, y: np.ndarray, a: float):
        x, s, u = self.codec.decode(y, a)
        sigma_static, sigma_c, mu = self.base_codec.decode(x)
        return sigma_static, sigma_c, mu, s, float(a), u

    def _background(self, x: np.ndarray, base_ev) -> SimpleNamespace:
        sigma_static, sigma_c, mu = self.base_codec.decode(x)
        return SimpleNamespace(
            G=np.asarray(base_ev.G),
            P=np.asarray(base_ev.P),
            Sigma_static=np.asarray(sigma_static),
            Sigma_c=np.asarray(sigma_c),
            mu=float(mu),
            rho=np.asarray(base_ev.rho),
            density=np.real(np.diag(np.asarray(base_ev.rho))),
        )

    def _free_energy_at_weights(
        self,
        background,
        weights: FierzWeights,
        reference_definition,
    ) -> ChannelGWFreeEnergyResult:
        definition = build_weighted_nbj_definition(
            self.pairs,
            self.norb,
            self.V,
            validate_weights(weights),
        )
        if definition.labels != reference_definition.labels:
            raise RuntimeError("transverse Fierz stencil changed the active channel basis")
        if not np.array_equal(definition.vertices, reference_definition.vertices):
            raise RuntimeError("transverse Fierz stencil changed channel vertices")
        return evaluate_channel_gw_free_energy(
            background,
            definition,
            self.h0,
            self.grid,
            target_particles=self.target,
            primitive_cells_per_supercell=self.primitive_cells,
        )

    def _h_s(self, s: float, a: float) -> float:
        distance = min(float(s) - abs(float(a)), 1.0 - float(s))
        h = min(self.fd_h, 0.20 * distance)
        if h <= 1e-8:
            raise FloatingPointError("too close to simplex boundary for longitudinal PMS derivative")
        return float(h)

    def _h_a(self, s: float, a: float) -> float:
        distance = float(s) - abs(float(a))
        h = min(self.fd_h, 0.20 * distance)
        if h <= 1e-8:
            raise FloatingPointError("too close to simplex boundary for transverse PMS derivative")
        return float(h)

    def evaluate(self, y: np.ndarray, a: float) -> LongitudinalPMSEvaluation:
        x, s, u = self.codec.decode(y, a)
        weights = weights_from_s_a(s, a)
        problem = self._problem(weights)
        base_ev = problem.evaluate(x, self.V)
        reference = problem.definition(self.V)
        background = self._background(x, base_ev)
        h = self._h_s(s, a)

        def F(ss: float) -> ChannelGWFreeEnergyResult:
            return self._free_energy_at_weights(
                background,
                weights_from_s_a(ss, a),
                reference,
            )

        fm2 = F(s - 2.0 * h)
        fm1 = F(s - h)
        f0 = F(s)
        fp1 = F(s + h)
        fp2 = F(s + 2.0 * h)
        vals = [float(z.free_energy_per_primitive_cell) for z in (fm2, fm1, f0, fp1, fp2)]
        Fm2, Fm1, F0, Fp1, Fp2 = vals
        dF = (Fm2 - 8.0 * Fm1 + 8.0 * Fp1 - Fp2) / (12.0 * h)
        d2F = (-Fp2 + 16.0 * Fp1 - 30.0 * F0 + 16.0 * Fm1 - Fm2) / (12.0 * h * h)
        scale = max(self.V * self.V, self.free_energy_scale_floor)
        pms_scaled = float(dF / scale)
        residual = np.concatenate([
            np.asarray(base_ev.residual, dtype=float),
            [pms_scaled],
        ])
        return LongitudinalPMSEvaluation(
            residual=np.asarray(residual, dtype=float),
            base_evaluation=base_ev,
            weights=weights,
            s=float(s),
            a=float(a),
            s_logit=float(u),
            dF_ds_per_cell=float(dF),
            d2F_ds2_per_cell=float(d2F),
            free_energy=f0,
            fd_h_s_used=float(h),
            pms_scaled_residual=pms_scaled,
        )

    def transverse_diagnostics(
        self,
        y: np.ndarray,
        a: float,
        *,
        evaluation: LongitudinalPMSEvaluation | None = None,
    ) -> TransversePMSDiagnostics:
        x, s, _ = self.codec.decode(y, a)
        ev = self.evaluate(y, a) if evaluation is None else evaluation
        problem = self._problem(ev.weights)
        reference = problem.definition(self.V)
        background = self._background(x, ev.base_evaluation)
        h = self._h_a(s, a)

        def F(aa: float) -> float:
            result = self._free_energy_at_weights(
                background,
                weights_from_s_a(s, aa),
                reference,
            )
            return float(result.free_energy_per_primitive_cell)

        Fm2 = F(a - 2.0 * h)
        Fm1 = F(a - h)
        F0 = float(ev.free_energy.free_energy_per_primitive_cell)
        Fp1 = F(a + h)
        Fp2 = F(a + 2.0 * h)
        dFa = (Fm2 - 8.0 * Fm1 + 8.0 * Fp1 - Fp2) / (12.0 * h)
        d2Fa = (-Fp2 + 16.0 * Fp1 - 30.0 * F0 + 16.0 * Fm1 - Fm2) / (12.0 * h * h)
        dFs = float(ev.dF_ds_per_cell)
        # Since ds changes (b,j) by (+1/2,+1/2) and da by (+1/2,-1/2):
        #   dF/ds=(gB+gJ)/2, dF/da=(gB-gJ)/2.
        gB = dFs + dFa
        gJ = dFs - dFa
        return TransversePMSDiagnostics(
            dF_ds_per_cell=dFs,
            dF_da_per_cell=float(dFa),
            dF_dlambda_B_per_cell=float(gB),
            dF_dlambda_J_per_cell=float(gJ),
            d2F_da2_per_cell=float(d2Fa),
            gradient_norm_per_cell=float(np.hypot(gB, gJ)),
            fd_h_a_used=float(h),
        )

    def __call__(self, y: np.ndarray, a: float) -> np.ndarray:
        return self.evaluate(y, a).residual


__all__ = [
    "weights_from_s_a",
    "s_a_from_weights",
    "LongitudinalPMSCodec",
    "LongitudinalPMSEvaluation",
    "TransversePMSDiagnostics",
    "LongitudinalFierzPMSResidual",
]
