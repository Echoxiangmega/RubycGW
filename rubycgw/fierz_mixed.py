"""Weighted density/B/J Fierz representation for same-torus Ruby GW.

For every interacting spinless bond (i,j), the normal-ordered quartic term has
three exactly equivalent representations,

    V n_i n_j = -(V/2) :B_ij^2: = -(V/2) :J_ij^2:.

Hence

    lambda_n + lambda_B + lambda_J = 1

leaves the exact fermionic interaction unchanged.  A truncated GW resummation
is not Fierz invariant, however, so the weights define a useful family of
multichannel approximations.  The density, B, and J vertices are placed in one
joint channel space and the *full* polarization/screening matrices are used;
this is not a linear average of separately solved GW theories.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fierz_channel_gw import (
    ChannelDefinition,
    channel_static_self_energy,
    compute_channel_correlation_self_energy,
    compute_channel_polarization,
    compute_channel_screened_interaction,
)
from .fierz_pac import FierzResidualEvaluation, FierzStateCodec, screening_soft_mode
from .grids import MatsubaraGrid
from .supercell_gw import dyson_from_sigma_matrix
from .supercell_gw_split import one_body_density_matrix_tail


@dataclass(frozen=True)
class FierzWeights:
    density: float
    bond: float
    current: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.density), float(self.bond), float(self.current))


def weights_from_lambda(lambda_n: float) -> FierzWeights:
    """One-parameter family lambda_n=lambda, lambda_B=lambda_J=(1-lambda)/2."""
    lam = float(lambda_n)
    if not np.isfinite(lam) or lam < 0.0 or lam > 1.0:
        raise ValueError("lambda must be finite and lie in [0,1]")
    rem = 0.5 * (1.0 - lam)
    return FierzWeights(lam, rem, rem)


def validate_weights(weights: FierzWeights, *, atol: float = 2e-12) -> FierzWeights:
    vals = np.asarray(weights.as_tuple(), dtype=float)
    if np.any(~np.isfinite(vals)):
        raise ValueError("Fierz weights must be finite")
    if np.any(vals < -float(atol)):
        raise ValueError("Fierz weights must be non-negative")
    if abs(float(np.sum(vals)) - 1.0) > float(atol):
        raise ValueError("Fierz weights must satisfy lambda_n+lambda_B+lambda_J=1")
    vals[np.abs(vals) < float(atol)] = 0.0
    vals = vals / float(np.sum(vals))
    return FierzWeights(*[float(x) for x in vals])


def build_weighted_nbj_definition(
    interaction_pairs,
    nsite: int,
    V: float,
    weights: FierzWeights,
    *,
    active_tol: float = 1e-14,
) -> ChannelDefinition:
    """Build one joint density+B+J channel space with prescribed Fierz weights.

    The bosonic coupling is block diagonal at bare level, but P and W are full
    matrices, so density/bond/current sectors mix dynamically through fermion
    bubbles.  Exactly zero-weight sectors are omitted to avoid unnecessary work
    at the pure endpoints lambda=0 or 1.
    """
    weights = validate_weights(weights)
    ln, lb, lj = weights.as_tuple()
    pairs = tuple((int(i), int(j)) for i, j in interaction_pairs)
    nsite = int(nsite)
    V = float(V)

    vertices: list[np.ndarray] = []
    labels: list[str] = []
    blocks: list[tuple[str, int, int]] = []

    if ln > float(active_tol):
        start = len(vertices)
        for i in range(nsite):
            K = np.zeros((nsite, nsite), dtype=complex)
            K[i, i] = 1.0
            vertices.append(K)
            labels.append(f"n{i}")
        blocks.append(("n", start, len(vertices)))

    if lb > float(active_tol):
        start = len(vertices)
        for i, j in pairs:
            K = np.zeros((nsite, nsite), dtype=complex)
            K[i, j] = 1.0
            K[j, i] = 1.0
            vertices.append(K)
            labels.append(f"B({i},{j})")
        blocks.append(("B", start, len(vertices)))

    if lj > float(active_tol):
        start = len(vertices)
        for i, j in pairs:
            K = np.zeros((nsite, nsite), dtype=complex)
            K[i, j] = 1j
            K[j, i] = -1j
            vertices.append(K)
            labels.append(f"J({i},{j})")
        blocks.append(("J", start, len(vertices)))

    if not vertices:
        raise RuntimeError("weighted Fierz definition has no active channels")

    K = np.asarray(vertices, dtype=complex)
    nch = len(vertices)
    g = np.zeros((nch, nch), dtype=complex)
    for name, a, b in blocks:
        if name == "n":
            for i, j in pairs:
                g[a + i, a + j] += ln * V
                g[a + j, a + i] += ln * V
        elif name == "B":
            idx = np.arange(a, b)
            g[idx, idx] = -lb * V
        elif name == "J":
            idx = np.arange(a, b)
            g[idx, idx] = -lj * V

    mode = f"nbj[{ln:.12g},{lb:.12g},{lj:.12g}]"
    return ChannelDefinition(mode, K, g, tuple(labels), pairs)


def soft_mode_sector_fractions(definition: ChannelDefinition, mode: np.ndarray) -> dict[str, float]:
    """Squared-norm fractions of a channel vector in n/B/J sectors."""
    z = np.asarray(mode, dtype=complex).reshape(-1)
    if z.size != len(definition.labels):
        raise ValueError("soft-mode size does not match channel definition")
    den = float(np.vdot(z, z).real)
    out = {"n": 0.0, "B": 0.0, "J": 0.0}
    if den <= 1e-300:
        return out
    for coeff, label in zip(z, definition.labels):
        w = float(abs(coeff) ** 2 / den)
        if label.startswith("n"):
            out["n"] += w
        elif label.startswith("B("):
            out["B"] += w
        elif label.startswith("J("):
            out["J"] += w
    return out


class WeightedFierzGWResidual:
    """GW fixed-point residual at fixed n/B/J Fierz weights and variable V."""

    def __init__(
        self,
        h0: np.ndarray,
        interaction_pairs,
        weights: FierzWeights,
        grid: MatsubaraGrid,
        target: float,
    ):
        self.h0 = np.asarray(h0, dtype=complex)
        self.pairs = tuple((int(i), int(j)) for i, j in interaction_pairs)
        self.weights = validate_weights(weights)
        self.grid = grid
        self.target = float(target)
        self.norb = int(self.h0.shape[-1])
        self.codec = FierzStateCodec(self.norb, grid, target)

    def definition(self, V: float) -> ChannelDefinition:
        return build_weighted_nbj_definition(
            self.pairs, self.norb, float(V), self.weights
        )

    def evaluate(self, x: np.ndarray, V: float) -> FierzResidualEvaluation:
        sigma_static, sigma_c, mu = self.codec.decode(x)
        definition = self.definition(V)
        G = dyson_from_sigma_matrix(self.h0, self.grid, mu, sigma_static, sigma_c)
        rho = one_body_density_matrix_tail(
            G, self.grid, self.h0, mu, sigma_static
        )
        rho0 = np.asarray(rho[0, 0], dtype=complex)
        sigma_static_out, _, _ = channel_static_self_energy(rho0, definition)
        P = compute_channel_polarization(G, definition, self.grid)
        W = compute_channel_screened_interaction(P, definition)
        sigma_c_out = compute_channel_correlation_self_energy(
            G, W, definition, self.grid
        )
        r_static = sigma_static - sigma_static_out
        r_dynamic = sigma_c - sigma_c_out
        filling_error = float(np.trace(rho0).real - self.target)
        residual = self.codec.encode_residual(r_static, r_dynamic, filling_error)
        physical_residual = max(
            float(np.max(np.abs(r_static))),
            float(np.max(np.abs(r_dynamic))),
            abs(filling_error),
        )
        smin, soft_m, soft_Omega, soft_mode = screening_soft_mode(
            P, definition, self.grid
        )
        return FierzResidualEvaluation(
            residual=np.asarray(residual, dtype=float),
            G=np.asarray(G), P=np.asarray(P), W=np.asarray(W),
            sigma_static_out=np.asarray(sigma_static_out),
            sigma_c_out=np.asarray(sigma_c_out), rho=np.asarray(rho0),
            filling_error=filling_error, physical_residual=physical_residual,
            smin=smin, soft_m=soft_m, soft_Omega=soft_Omega,
            soft_mode=np.asarray(soft_mode),
        )

    def __call__(self, x: np.ndarray, V: float) -> np.ndarray:
        return self.evaluate(x, V).residual


__all__ = [
    "FierzWeights",
    "weights_from_lambda",
    "validate_weights",
    "build_weighted_nbj_definition",
    "soft_mode_sector_fractions",
    "WeightedFierzGWResidual",
]
