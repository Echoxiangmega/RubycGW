"""Principle-of-minimal-sensitivity closure for weighted n/B/J Fierz GW.

The exact fermionic interaction is independent of the one-parameter Fierz
coordinate

    lambda_n = lambda,
    lambda_B = lambda_J = (1-lambda)/2.

A truncated GW functional is not Fierz invariant.  This module promotes lambda
to an additional unknown and closes the self-consistent equations with the
stationarity condition

    d F_GW / d lambda = 0,

where F_GW is the fixed-filling Luttinger-Ward Helmholtz free energy evaluated
on the same branch.  At a stationary GW solution the total derivative along the
self-consistent branch equals the explicit derivative of the LW functional at
fixed G.  We therefore evaluate the latter with a five-point finite difference
in lambda while keeping the fermionic codec state fixed.

The augmented state uses an unconstrained logit coordinate u for lambda.  This
keeps 0 < lambda < 1 automatically without making a boundary minimum look like
an interior PMS solution.

The module also exposes a *diagnostic* derivative in the full two-dimensional
Fierz simplex,

    lambda_n + lambda_B + lambda_J = 1,

using (lambda_B, lambda_J) as independent coordinates.  This is useful for
checking whether a stationary point on the symmetric B=J line is genuinely
stationary in the full Fierz plane or only stationary along that one-dimensional
cut.
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
    weights_from_lambda,
)
from .grids import MatsubaraGrid


@dataclass(frozen=True)
class PMSLambdaCodec:
    """Append one unconstrained logit coordinate to a Fierz fermionic state."""

    base_codec: object
    margin: float = 1e-8

    @property
    def size(self) -> int:
        return int(self.base_codec.size) + 1

    def _scaled_lambda(self, u: float) -> float:
        # Stable logistic map.
        u = float(u)
        if u >= 0.0:
            e = np.exp(-min(u, 745.0))
            s = 1.0 / (1.0 + e)
        else:
            e = np.exp(max(u, -745.0))
            s = e / (1.0 + e)
        m = float(self.margin)
        return float(m + (1.0 - 2.0 * m) * s)

    def encode(self, x: np.ndarray, lambda_value: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != int(self.base_codec.size):
            raise ValueError("base Fierz state has the wrong size")
        lam = float(lambda_value)
        m = float(self.margin)
        if not np.isfinite(lam) or not (m < lam < 1.0 - m):
            raise ValueError(f"lambda must lie strictly inside ({m:g},{1.0-m:g})")
        s = (lam - m) / (1.0 - 2.0 * m)
        u = np.log(s / (1.0 - s))
        return np.concatenate([x, [float(u)]])

    def decode(self, y: np.ndarray) -> tuple[np.ndarray, float, float]:
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != self.size:
            raise ValueError(f"PMS state size {y.size} != {self.size}")
        u = float(y[-1])
        lam = self._scaled_lambda(u)
        return np.asarray(y[:-1], dtype=float), u, lam


@dataclass
class PMSResidualEvaluation:
    residual: np.ndarray
    base_evaluation: object
    lambda_value: float
    lambda_logit: float
    dF_dlambda_per_cell: float
    d2F_dlambda2_per_cell: float
    free_energy: ChannelGWFreeEnergyResult
    fd_h_used: float
    pms_scaled_residual: float

    @property
    def weights(self):
        return weights_from_lambda(self.lambda_value)

    @property
    def smin(self) -> float:
        return float(self.base_evaluation.smin)

    @property
    def physical_residual(self) -> float:
        return float(self.base_evaluation.physical_residual)

    @property
    def filling_error(self) -> float:
        return float(self.base_evaluation.filling_error)

    @property
    def G(self):
        return self.base_evaluation.G

    @property
    def P(self):
        return self.base_evaluation.P

    @property
    def W(self):
        return self.base_evaluation.W

    @property
    def rho(self):
        return self.base_evaluation.rho

    @property
    def soft_mode(self):
        return self.base_evaluation.soft_mode

    @property
    def soft_m(self) -> int:
        return int(self.base_evaluation.soft_m)

    @property
    def soft_Omega(self) -> float:
        return float(self.base_evaluation.soft_Omega)


@dataclass(frozen=True)
class FierzSimplexGradient:
    """Explicit fixed-G free-energy gradient in the full n/B/J simplex.

    Independent coordinates are b=lambda_B and j=lambda_J, with
    lambda_n=1-b-j.  ``parallel`` is the normalized derivative along the
    symmetric one-parameter PMS line when lambda increases, i.e. direction
    (-1,-1)/sqrt(2) in (b,j).  ``perpendicular`` probes the missing B-J
    antisymmetric direction (1,-1)/sqrt(2).
    """

    dF_dlambda_B_per_cell: float
    dF_dlambda_J_per_cell: float
    gradient_norm_per_cell: float
    parallel_per_cell: float
    perpendicular_per_cell: float
    dF_dlambda_reconstructed_per_cell: float
    line_consistency_error_per_cell: float
    fd_h_used: float


class WeightedFierzPMSResidual:
    """Augmented GW+PMS residual with lambda solved self-consistently.

    The augmented residual is

        [ R_GW(X; V, lambda), (dF/dlambda)/S_F ],

    with lambda represented by a logit coordinate.  ``dF/dlambda`` is the
    explicit derivative of the fixed-filling LW free energy per primitive cell
    at fixed fermionic state X, evaluated with a five-point stencil.  The root
    is unaffected by ``free_energy_scale``; it only balances the Newton-Krylov
    coordinates numerically.
    """

    def __init__(
        self,
        h0: np.ndarray,
        interaction_pairs,
        grid: MatsubaraGrid,
        target: float,
        *,
        primitive_cells: int,
        lambda_fd_h: float = 2e-3,
        lambda_margin: float = 1e-8,
        free_energy_scale_floor: float = 1e-4,
    ):
        self.h0 = np.asarray(h0, dtype=complex)
        self.pairs = tuple((int(i), int(j)) for i, j in interaction_pairs)
        self.grid = grid
        self.target = float(target)
        self.primitive_cells = int(primitive_cells)
        if self.primitive_cells < 1:
            raise ValueError("primitive_cells must be positive")
        self.lambda_fd_h = float(lambda_fd_h)
        if not np.isfinite(self.lambda_fd_h) or self.lambda_fd_h <= 0.0:
            raise ValueError("lambda_fd_h must be positive")
        self.lambda_margin = float(lambda_margin)
        if not 0.0 <= self.lambda_margin < 0.1:
            raise ValueError("lambda_margin must lie in [0,0.1)")
        self.free_energy_scale_floor = float(free_energy_scale_floor)
        if not np.isfinite(self.free_energy_scale_floor) or self.free_energy_scale_floor <= 0.0:
            raise ValueError("free_energy_scale_floor must be positive")

        probe = WeightedFierzGWResidual(
            self.h0,
            self.pairs,
            weights_from_lambda(0.5),
            self.grid,
            self.target,
        )
        self.base_codec = probe.codec
        self.codec = PMSLambdaCodec(self.base_codec, margin=self.lambda_margin)
        self.norb = int(self.h0.shape[-1])

    def _problem(self, lambda_value: float) -> WeightedFierzGWResidual:
        return WeightedFierzGWResidual(
            self.h0,
            self.pairs,
            weights_from_lambda(lambda_value),
            self.grid,
            self.target,
        )

    def encode_base_state(self, x: np.ndarray, lambda_value: float) -> np.ndarray:
        return self.codec.encode(x, lambda_value)

    def encode_result(self, result, lambda_value: float) -> np.ndarray:
        return self.codec.encode(self.base_codec.encode_result(result), lambda_value)

    def decode(self, y: np.ndarray):
        x, u, lam = self.codec.decode(y)
        sigma_static, sigma_c, mu = self.base_codec.decode(x)
        return sigma_static, sigma_c, mu, lam, u

    def _background_from_fixed_state(self, x: np.ndarray, base_ev) -> SimpleNamespace:
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
        V: float,
        weights: FierzWeights,
        reference_definition,
    ) -> ChannelGWFreeEnergyResult:
        weights = validate_weights(weights)
        definition = build_weighted_nbj_definition(
            self.pairs,
            self.norb,
            float(V),
            weights,
        )
        # At an interior simplex point the n/B/J vertex basis is independent of
        # the weights; only g changes.  P=GG is therefore unchanged at fixed G.
        if definition.labels != reference_definition.labels:
            raise RuntimeError("Fierz finite difference changed the active channel basis")
        if not np.array_equal(definition.vertices, reference_definition.vertices):
            raise RuntimeError("Fierz finite difference changed channel vertices")
        return evaluate_channel_gw_free_energy(
            background,
            definition,
            self.h0,
            self.grid,
            target_particles=self.target,
            primitive_cells_per_supercell=self.primitive_cells,
        )

    def _free_energy_at_lambda(
        self,
        background,
        V: float,
        lambda_value: float,
        reference_definition,
    ) -> ChannelGWFreeEnergyResult:
        return self._free_energy_at_weights(
            background,
            V,
            weights_from_lambda(lambda_value),
            reference_definition,
        )

    def _stencil_h(self, lambda_value: float) -> float:
        distance = min(float(lambda_value), 1.0 - float(lambda_value))
        h = min(self.lambda_fd_h, 0.20 * distance)
        if h <= 1e-8:
            raise FloatingPointError(
                "lambda approached a simplex boundary too closely for a stable PMS derivative"
            )
        return float(h)

    def _simplex_stencil_h(self, weights: FierzWeights, h: float | None = None) -> float:
        w = validate_weights(weights)
        distance = min(w.density, w.bond, w.current)
        nominal = self.lambda_fd_h if h is None else float(h)
        if not np.isfinite(nominal) or nominal <= 0.0:
            raise ValueError("simplex finite-difference h must be positive")
        # Five-point stencil reaches +/-2h.  Keep a generous factor-five margin
        # from every simplex boundary so all three channel families stay active.
        used = min(nominal, 0.20 * distance)
        if used <= 1e-8:
            raise FloatingPointError(
                "Fierz weights approached a simplex boundary too closely for a stable 2D gradient"
            )
        return float(used)

    def simplex_gradient(
        self,
        y: np.ndarray,
        V: float,
        *,
        h: float | None = None,
        evaluation: PMSResidualEvaluation | None = None,
    ) -> FierzSimplexGradient:
        """Return the explicit full-simplex gradient at a PMS state.

        The fermionic state is held fixed.  Coordinates are b=lambda_B and
        j=lambda_J with n=1-b-j.  This is the correct diagnostic derivative of
        the stationary LW functional.  No additional self-consistent GW solves
        are performed for the finite-difference stencil.
        """
        x, _, lam = self.codec.decode(y)
        ev = self.evaluate(y, float(V)) if evaluation is None else evaluation
        reference = self._problem(lam).definition(float(V))
        background = self._background_from_fixed_state(x, ev.base_evaluation)
        w0 = weights_from_lambda(lam)
        hh = self._simplex_stencil_h(w0, h)

        def F_of(b: float, j: float) -> float:
            n = 1.0 - float(b) - float(j)
            result = self._free_energy_at_weights(
                background,
                V,
                FierzWeights(n, float(b), float(j)),
                reference,
            )
            return float(result.free_energy_per_primitive_cell)

        b0 = float(w0.bond)
        j0 = float(w0.current)

        Fb_m2 = F_of(b0 - 2.0 * hh, j0)
        Fb_m1 = F_of(b0 - hh, j0)
        Fb_p1 = F_of(b0 + hh, j0)
        Fb_p2 = F_of(b0 + 2.0 * hh, j0)
        gB = (Fb_m2 - 8.0 * Fb_m1 + 8.0 * Fb_p1 - Fb_p2) / (12.0 * hh)

        Fj_m2 = F_of(b0, j0 - 2.0 * hh)
        Fj_m1 = F_of(b0, j0 - hh)
        Fj_p1 = F_of(b0, j0 + hh)
        Fj_p2 = F_of(b0, j0 + 2.0 * hh)
        gJ = (Fj_m2 - 8.0 * Fj_m1 + 8.0 * Fj_p1 - Fj_p2) / (12.0 * hh)

        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        parallel = -(gB + gJ) * inv_sqrt2
        perpendicular = (gB - gJ) * inv_sqrt2
        reconstructed = -0.5 * (gB + gJ)
        line_error = reconstructed - float(ev.dF_dlambda_per_cell)
        grad_norm = float(np.hypot(gB, gJ))

        return FierzSimplexGradient(
            dF_dlambda_B_per_cell=float(gB),
            dF_dlambda_J_per_cell=float(gJ),
            gradient_norm_per_cell=grad_norm,
            parallel_per_cell=float(parallel),
            perpendicular_per_cell=float(perpendicular),
            dF_dlambda_reconstructed_per_cell=float(reconstructed),
            line_consistency_error_per_cell=float(line_error),
            fd_h_used=float(hh),
        )

    def evaluate(self, y: np.ndarray, V: float) -> PMSResidualEvaluation:
        x, u, lam = self.codec.decode(y)
        problem = self._problem(lam)
        base_ev = problem.evaluate(x, float(V))
        definition = problem.definition(float(V))
        background = self._background_from_fixed_state(x, base_ev)

        h = self._stencil_h(lam)
        f0 = self._free_energy_at_lambda(background, V, lam, definition)
        fm1 = self._free_energy_at_lambda(background, V, lam - h, definition)
        fp1 = self._free_energy_at_lambda(background, V, lam + h, definition)
        fm2 = self._free_energy_at_lambda(background, V, lam - 2.0 * h, definition)
        fp2 = self._free_energy_at_lambda(background, V, lam + 2.0 * h, definition)

        # Five-point O(h^4) formulas, using F per primitive cell.
        Fm2 = float(fm2.free_energy_per_primitive_cell)
        Fm1 = float(fm1.free_energy_per_primitive_cell)
        F0 = float(f0.free_energy_per_primitive_cell)
        Fp1 = float(fp1.free_energy_per_primitive_cell)
        Fp2 = float(fp2.free_energy_per_primitive_cell)
        dF = (Fm2 - 8.0 * Fm1 + 8.0 * Fp1 - Fp2) / (12.0 * h)
        d2F = (-Fp2 + 16.0 * Fp1 - 30.0 * F0 + 16.0 * Fm1 - Fm2) / (12.0 * h * h)

        # Fierz dependence starts beyond the exactly matched first-order HF
        # contribution, so V^2 is a natural numerical scale for the PMS row.
        scale = max(float(V) * float(V), self.free_energy_scale_floor)
        pms_scaled = float(dF / scale)
        residual = np.concatenate([
            np.asarray(base_ev.residual, dtype=float),
            [pms_scaled],
        ])
        return PMSResidualEvaluation(
            residual=np.asarray(residual, dtype=float),
            base_evaluation=base_ev,
            lambda_value=float(lam),
            lambda_logit=float(u),
            dF_dlambda_per_cell=float(dF),
            d2F_dlambda2_per_cell=float(d2F),
            free_energy=f0,
            fd_h_used=float(h),
            pms_scaled_residual=pms_scaled,
        )

    def __call__(self, y: np.ndarray, V: float) -> np.ndarray:
        return self.evaluate(y, V).residual


__all__ = [
    "PMSLambdaCodec",
    "PMSResidualEvaluation",
    "FierzSimplexGradient",
    "WeightedFierzPMSResidual",
]
