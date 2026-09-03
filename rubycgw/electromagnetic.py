"""Electromagnetic covariant response for the 18-site Ruby GW checkpoint.

The external field used here is a periodic Peierls-flux perturbation on the
elementary Ruby triangles.  It is a genuine electromagnetic lattice source:
for a directed bond I->J,

    t_IJ -> t_IJ exp(i phi c_IJ),

with c_IJ=-c_JI.  The total algebraic Peierls phase around a selected triangle
is ``phi * weight``.  Since all elementary triangle hoppings are intracell, this
source has q_sc=0 in the 18-site supercell and can be passed directly to the
existing covariant-GW vertex solver.

For the split self-energy used by the production supercell GW solver,

    Sigma = Sigma_H + Sigma_F + Sigma_c,
    Sigma_c = -G * (W-V),

the covariant derivative obeys

    G_phi = G Gamma_phi G,
    Gamma_phi = K_phi - mu_phi I
                + Sigma_H,phi + Sigma_F,phi
                + Sigma_MT,c,phi + Sigma_AL1,phi + Sigma_AL2,phi.

At fixed chemical potential ``mu_phi=0``.  At fixed filling we exploit
linearity: solve once for the Peierls source K_phi and once for the chemical
potential source -I, then choose mu_phi so the total density derivative
vanishes.

This module also provides a centered finite-difference validation in which
fully self-consistent GW calculations are repeated at +delta_phi and
-delta_phi.  Agreement is limited by the GW/vertex tolerances and by the finite
Matsubara box: the production GW density/Fock map uses analytic tail
subtraction, while the cGW response sums the absolutely convergent G Gamma G
response over the stored fermion box.  The mismatch must decrease when the
fermionic cutoff is increased.

Important: this periodic plaquette-flux response is not yet the strict uniform
bulk orbital magnetization.  A uniform B field in a periodic crystal requires
a long-wavelength transverse A(q) construction (or an equivalent magnetic-cell
limit).  The present implementation is the controlled electromagnetic response
needed to validate the covariant derivative and to probe local orbital-current
channels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .checkpoint import GWCheckpointSeed, load_supercell_checkpoint, read_checkpoint_metadata
from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .gw import GWOptions, _check_backend
from .model import RubyParameters
from .orbital_moment import grid_and_params_from_checkpoint_metadata
from .supercell import (
    NSUB,
    NSUP,
    NSECTOR,
    build_supercell_h0,
    build_supercell_interaction,
)
from .supercell_cgw import (
    SupercellVertexOptions,
    SupercellVertexResult,
    solve_vertex_q0,
)
from .supercell_gw import (
    _reverse_fft_spectrum,
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    dyson_from_sigma_matrix,
)
from .supercell_gw_fast import solve_matrix_gw_fast


EM_CHANNELS = ("A", "B", "opposite", "same")

_TRIANGLE_EDGES = {
    "A": ((0, 1), (1, 2), (2, 0)),
    "B": ((3, 4), (4, 5), (5, 3)),
}


@dataclass
class ElectromagneticBackground:
    metadata: dict
    grid: MatsubaraGrid
    params: RubyParameters
    h0: np.ndarray
    Vq: np.ndarray
    G: np.ndarray
    P: np.ndarray
    W: np.ndarray
    Sigma_H: np.ndarray
    Sigma_GW: np.ndarray
    density: np.ndarray
    mu: float


@dataclass
class ElectromagneticResponse:
    channel: str
    flux_weights: tuple[float, float]
    fixed_filling: bool
    K: np.ndarray
    Gamma: np.ndarray
    G_phi: np.ndarray
    P_phi: np.ndarray
    W_phi: np.ndarray
    Sigma_H_phi: np.ndarray
    Sigma_GW_phi: np.ndarray
    density_phi: np.ndarray
    mu_phi: float
    density_constraint_residual: float
    equation_residual: float
    vertex_converged: bool
    vertex_iterations: int
    vertex_final_error: float
    mu_vertex_converged: bool | None = None
    mu_vertex_iterations: int | None = None
    mu_vertex_final_error: float | None = None


@dataclass
class FiniteDifferenceResponse:
    channel: str
    delta_phi: float
    fixed_filling: bool
    G_phi: np.ndarray
    P_phi: np.ndarray
    W_phi: np.ndarray
    Sigma_H_phi: np.ndarray
    Sigma_GW_phi: np.ndarray
    Gamma_phi: np.ndarray
    density_phi: np.ndarray
    mu_phi: float
    plus_converged: bool
    minus_converged: bool
    plus_error: float
    minus_error: float
    plus_iterations: int
    minus_iterations: int


def channel_flux_weights(channel: str) -> tuple[float, float]:
    """Return algebraic triangle-flux weights (A,B).

    ``opposite`` and ``same`` follow the repository current labels:

    * opposite: (A+B)/sqrt(2), conjugate to physical-opposite circulation;
    * same:     (A-B)/sqrt(2), conjugate to physical-same circulation.

    The total Peierls phase around triangle X is ``phi * weight_X``.
    """
    lower = str(channel).strip().lower()
    if lower == "a":
        return 1.0, 0.0
    if lower == "b":
        return 0.0, 1.0
    if lower in {"opposite", "plus", "eta_plus", "eta+"}:
        s = 1.0 / np.sqrt(2.0)
        return s, s
    if lower in {"same", "minus", "eta_minus", "eta-"}:
        s = 1.0 / np.sqrt(2.0)
        return s, -s
    raise ValueError(
        f"unknown electromagnetic channel {channel!r}; expected A, B, opposite, or same"
    )


def _canonical_channel(channel: str) -> str:
    wa, wb = channel_flux_weights(channel)
    refs = {
        "A": (1.0, 0.0),
        "B": (0.0, 1.0),
        "opposite": (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
        "same": (1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)),
    }
    for name, pair in refs.items():
        if np.allclose([wa, wb], pair, rtol=0.0, atol=1e-15):
            return name
    raise RuntimeError("internal electromagnetic channel normalization error")


def build_supercell_h0_peierls(
    kpts: np.ndarray,
    params: RubyParameters,
    phi: float,
    channel: str,
) -> np.ndarray:
    """Return H0 with a periodic elementary-triangle Peierls flux.

    On each selected triangle, the total algebraic loop phase is distributed
    equally over its three directed edges.  Reverse hoppings carry the opposite
    phase, so Hermiticity is exact.
    """
    h0 = build_supercell_h0(kpts, params, source_strength=0.0)
    out = np.array(h0, dtype=complex, copy=True)
    wa, wb = channel_flux_weights(channel)

    for s in range(NSECTOR):
        base = NSUB * s
        for tri, weight in (("A", wa), ("B", wb)):
            if abs(weight) == 0.0:
                continue
            theta = float(phi) * float(weight) / 3.0
            forward = np.exp(1j * theta)
            reverse = np.exp(-1j * theta)
            for a, b in _TRIANGLE_EDGES[tri]:
                I = base + a
                J = base + b
                out[..., I, J] *= forward
                out[..., J, I] *= reverse

    return 0.5 * (out + np.swapaxes(out.conj(), -1, -2))


def peierls_flux_vertex(params: RubyParameters, channel: str) -> np.ndarray:
    """Return K=dH0/dphi at phi=0 for the chosen periodic flux channel."""
    wa, wb = channel_flux_weights(channel)
    K = np.zeros((NSUP, NSUP), dtype=complex)

    for s in range(NSECTOR):
        base = NSUB * s
        for tri, weight in (("A", wa), ("B", wb)):
            coeff = float(weight) / 3.0
            for a, b in _TRIANGLE_EDGES[tri]:
                I = base + a
                J = base + b
                K[I, J] += 1j * coeff * float(params.ti)
                K[J, I] -= 1j * coeff * float(params.ti)

    return 0.5 * (K + K.conj().T)


def load_electromagnetic_background(
    path: str | Path,
    *,
    require_converged: bool = True,
    momentum_backend: str = "fft",
) -> ElectromagneticBackground:
    """Reconstruct G,P,W and self-energies from a zero-source GW checkpoint."""
    backend = _check_backend(momentum_backend)
    path = Path(path)
    meta = read_checkpoint_metadata(path)

    if require_converged and not bool(meta.get("converged", False)):
        raise ValueError(
            f"Checkpoint {path} is marked nonconverged (final_error={meta.get('final_error')!r})."
        )
    if abs(float(meta.get("source", 0.0))) > 1e-14:
        raise ValueError("Electromagnetic response currently requires a zero-source checkpoint.")

    grid, params = grid_and_params_from_checkpoint_metadata(meta)
    seed, checked_meta, density = load_supercell_checkpoint(
        path, params, grid, float(meta["primitive_filling"])
    )
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    G = dyson_from_sigma_matrix(
        h0,
        grid,
        float(seed.mu),
        np.asarray(seed.Sigma_H, dtype=complex),
        np.asarray(seed.Sigma_GW, dtype=complex),
    )
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)

    return ElectromagneticBackground(
        metadata=dict(checked_meta),
        grid=grid,
        params=params,
        h0=h0,
        Vq=Vq,
        G=G,
        P=P,
        W=W,
        Sigma_H=np.asarray(seed.Sigma_H, dtype=complex),
        Sigma_GW=np.asarray(seed.Sigma_GW, dtype=complex),
        density=np.asarray(density, dtype=float),
        mu=float(seed.mu),
    )


def _polarization_tangent_direct(G: np.ndarray, X: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    norb = int(G.shape[-1])
    out = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        Gsrc, Gbase = G[src], G[dst]
        Xsrc, Xbase = X[src], X[dst]
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Gq = roll_spatial(Gsrc, iq1, iq2)
                Xq = roll_spatial(Xsrc, iq1, iq2)
                out[im, iq1, iq2] = pref * (
                    np.einsum("nxyab,nxyba->ab", Xq, Gbase, optimize=True)
                    + np.einsum("nxyab,nxyba->ab", Gq, Xbase, optimize=True)
                )
    return out


def _polarization_tangent_fft(G: np.ndarray, X: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    norb = int(G.shape[-1])
    out = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        Gsrc = G[src]
        GbaseT = np.swapaxes(G[dst], -1, -2)
        Xsrc = X[src]
        XbaseT = np.swapaxes(X[dst], -1, -2)
        Ghat = np.fft.fftn(Gsrc, axes=(1, 2))
        Xhat = np.fft.fftn(Xsrc, axes=(1, 2))
        Gbase_minus = _reverse_fft_spectrum(GbaseT, axes=(1, 2))
        Xbase_minus = _reverse_fft_spectrum(XbaseT, axes=(1, 2))
        product = np.sum(Xhat * Gbase_minus + Ghat * Xbase_minus, axis=0)
        out[im] = pref * np.fft.ifftn(product, axes=(0, 1))
    return out


def compute_polarization_tangent(
    G: np.ndarray,
    X: np.ndarray,
    grid: MatsubaraGrid,
    backend: str = "fft",
) -> np.ndarray:
    """Derivative of the repository polarization P[G] in direction X=dG/dphi."""
    backend = _check_backend(backend)
    if np.asarray(G).shape != np.asarray(X).shape:
        raise ValueError("G and X must have identical shapes")
    if backend == "fft":
        return _polarization_tangent_fft(G, X, grid)
    return _polarization_tangent_direct(G, X, grid)


def density_response_from_G_tangent(X: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    """Return dn_a/dphi from the stored Matsubara response box."""
    diag = np.diagonal(np.asarray(X, dtype=complex), axis1=-2, axis2=-1)
    return (grid.T / grid.nk) * np.sum(diag, axis=(0, 1, 2))


def _x_from_gamma(G: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    return np.einsum("...ab,...bc,...cd->...ad", G, Gamma, G, optimize=True)


def _combine_vertex_fields(
    source: SupercellVertexResult,
    mu_source: SupercellVertexResult | None,
    mu_phi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mu_source is None or float(mu_phi) == 0.0:
        gamma = np.asarray(source.Gamma, dtype=complex)
        sigma_h = np.asarray(source.Gamma_H, dtype=complex)
        sigma_gw = (
            np.asarray(source.Gamma_F, dtype=complex)
            + np.asarray(source.Gamma_MT, dtype=complex)
            + np.asarray(source.Gamma_AL1, dtype=complex)
            + np.asarray(source.Gamma_AL2, dtype=complex)
        )
        return gamma, sigma_h, sigma_gw

    alpha = float(mu_phi)
    gamma = source.Gamma + alpha * mu_source.Gamma
    sigma_h = source.Gamma_H + alpha * mu_source.Gamma_H
    sigma_gw = (
        source.Gamma_F + source.Gamma_MT + source.Gamma_AL1 + source.Gamma_AL2
        + alpha * (
            mu_source.Gamma_F + mu_source.Gamma_MT + mu_source.Gamma_AL1 + mu_source.Gamma_AL2
        )
    )
    return gamma, sigma_h, sigma_gw


def solve_electromagnetic_response(
    background: ElectromagneticBackground,
    channel: str,
    *,
    fixed_filling: bool = True,
    vertex_options: SupercellVertexOptions | None = None,
) -> ElectromagneticResponse:
    """Solve the q_sc=0 covariant response to a periodic Peierls-flux source."""
    name = _canonical_channel(channel)
    grid = background.grid
    G = background.G
    K = peierls_flux_vertex(background.params, name)

    if vertex_options is None:
        vertex_options = SupercellVertexOptions(verbose=False, momentum_backend="fft")

    source = solve_vertex_q0(G, background.W, background.Vq, K, grid, opts=vertex_options)

    mu_source = None
    mu_phi = 0.0
    if fixed_filling:
        K_mu = -np.eye(NSUP, dtype=complex)
        mu_source = solve_vertex_q0(
            G, background.W, background.Vq, K_mu, grid, opts=vertex_options
        )
        X_source = _x_from_gamma(G, source.Gamma)
        X_mu = _x_from_gamma(G, mu_source.Gamma)
        dN_source = float(np.sum(density_response_from_G_tangent(X_source, grid)).real)
        dN_mu = float(np.sum(density_response_from_G_tangent(X_mu, grid)).real)
        if not np.isfinite(dN_mu) or abs(dN_mu) < 1e-12:
            raise RuntimeError(
                "Fixed-filling electromagnetic response is ill-conditioned: "
                f"dN/dmu={dN_mu:.3e}"
            )
        mu_phi = -dN_source / dN_mu

    Gamma, sigma_h_field, sigma_gw_phi = _combine_vertex_fields(source, mu_source, mu_phi)
    X = _x_from_gamma(G, Gamma)
    density_phi_complex = density_response_from_G_tangent(X, grid)
    density_phi = np.asarray(density_phi_complex.real, dtype=float)
    density_constraint = float(np.sum(density_phi))

    P_phi = compute_polarization_tangent(
        G, X, grid, backend=vertex_options.momentum_backend
    )
    W_phi = np.matmul(np.matmul(background.W, P_phi), background.W)
    sigma_h_static = np.asarray(sigma_h_field[0, 0, 0], dtype=complex)

    Kfield = np.broadcast_to(K, G.shape)
    identity_field = np.broadcast_to(np.eye(NSUP, dtype=complex), G.shape)
    rhs = Kfield - float(mu_phi) * identity_field + sigma_h_field + sigma_gw_phi
    equation_residual = float(np.max(np.abs(Gamma - rhs)))

    return ElectromagneticResponse(
        channel=name,
        flux_weights=channel_flux_weights(name),
        fixed_filling=bool(fixed_filling),
        K=K,
        Gamma=np.asarray(Gamma, dtype=complex),
        G_phi=np.asarray(X, dtype=complex),
        P_phi=np.asarray(P_phi, dtype=complex),
        W_phi=np.asarray(W_phi, dtype=complex),
        Sigma_H_phi=sigma_h_static,
        Sigma_GW_phi=np.asarray(sigma_gw_phi, dtype=complex),
        density_phi=density_phi,
        mu_phi=float(mu_phi),
        density_constraint_residual=density_constraint,
        equation_residual=equation_residual,
        vertex_converged=bool(source.converged),
        vertex_iterations=int(source.iterations),
        vertex_final_error=float(source.final_error),
        mu_vertex_converged=None if mu_source is None else bool(mu_source.converged),
        mu_vertex_iterations=None if mu_source is None else int(mu_source.iterations),
        mu_vertex_final_error=None if mu_source is None else float(mu_source.final_error),
    )


def _default_fd_gw_options(
    background: ElectromagneticBackground,
    *,
    fixed_filling: bool,
    momentum_backend: str,
) -> GWOptions:
    target = 3.0 * float(background.metadata["primitive_filling"]) if fixed_filling else None
    return GWOptions(
        mu=float(background.mu),
        target_filling=target,
        max_iter=250,
        tol=1e-9,
        mixing=0.20,
        mixing_method="pulay",
        pulay_history=6,
        pulay_start=3,
        pulay_regularization=1e-10,
        mu_tol=1e-10,
        mu_max_iter=80,
        verbose=False,
        momentum_backend=_check_backend(momentum_backend),
    )


def finite_difference_electromagnetic_response(
    background: ElectromagneticBackground,
    channel: str,
    delta_phi: float,
    *,
    fixed_filling: bool = True,
    gw_options: GWOptions | None = None,
    require_converged: bool = True,
) -> FiniteDifferenceResponse:
    """Centered finite difference using two fully self-consistent GW solves."""
    name = _canonical_channel(channel)
    delta = float(delta_phi)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta_phi must be positive")

    if gw_options is None:
        gw_options = _default_fd_gw_options(
            background, fixed_filling=fixed_filling, momentum_backend="fft"
        )
    else:
        target = 3.0 * float(background.metadata["primitive_filling"]) if fixed_filling else None
        gw_options = replace(gw_options, mu=float(background.mu), target_filling=target)

    initial = GWCheckpointSeed(
        Sigma_H=np.array(background.Sigma_H, copy=True),
        Sigma_GW=np.array(background.Sigma_GW, copy=True),
        mu=float(background.mu),
    )
    hplus = build_supercell_h0_peierls(
        background.grid.kmesh(), background.params, +delta, name
    )
    hminus = build_supercell_h0_peierls(
        background.grid.kmesh(), background.params, -delta, name
    )

    plus = solve_matrix_gw_fast(
        hplus, background.Vq, background.grid, opts=gw_options, initial=initial
    )
    minus = solve_matrix_gw_fast(
        hminus, background.Vq, background.grid, opts=gw_options, initial=initial
    )

    if require_converged and (not plus.converged or not minus.converged):
        raise RuntimeError(
            "Finite-difference GW did not converge: "
            f"+delta converged={plus.converged} err={plus.final_error:.3e}, "
            f"-delta converged={minus.converged} err={minus.final_error:.3e}"
        )

    inv = 1.0 / (2.0 * delta)
    G_phi = (plus.G - minus.G) * inv
    P_phi = (plus.P - minus.P) * inv
    W_phi = (plus.W - minus.W) * inv
    sigma_h_phi = (plus.Sigma_H - minus.Sigma_H) * inv
    sigma_gw_phi = (plus.Sigma_GW - minus.Sigma_GW) * inv
    density_phi = (plus.density - minus.density) * inv
    mu_phi = float((plus.mu - minus.mu) * inv)

    K = peierls_flux_vertex(background.params, name)
    Kfield = np.broadcast_to(K, G_phi.shape)
    identity_field = np.broadcast_to(np.eye(NSUP, dtype=complex), G_phi.shape)
    Gamma_phi = (
        Kfield - mu_phi * identity_field
        + sigma_h_phi[None, None, None, :, :] + sigma_gw_phi
    )

    return FiniteDifferenceResponse(
        channel=name,
        delta_phi=delta,
        fixed_filling=bool(fixed_filling),
        G_phi=G_phi,
        P_phi=P_phi,
        W_phi=W_phi,
        Sigma_H_phi=sigma_h_phi,
        Sigma_GW_phi=sigma_gw_phi,
        Gamma_phi=Gamma_phi,
        density_phi=np.asarray(density_phi, dtype=float),
        mu_phi=mu_phi,
        plus_converged=bool(plus.converged),
        minus_converged=bool(minus.converged),
        plus_error=float(plus.final_error),
        minus_error=float(minus.final_error),
        plus_iterations=int(plus.iterations),
        minus_iterations=int(minus.iterations),
    )


def _array_error_metrics(
    analytic: np.ndarray,
    finite_difference: np.ndarray,
    *,
    floor: float = 1e-14,
) -> dict[str, float]:
    a = np.asarray(analytic)
    b = np.asarray(finite_difference)
    diff = a - b
    abs_max = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), float(floor))
    return {"abs_max": abs_max, "rms": rms, "scale_max": scale, "rel_max": abs_max / scale}


def compare_covariant_to_finite_difference(
    analytic: ElectromagneticResponse,
    finite_difference: FiniteDifferenceResponse,
) -> dict[str, dict[str, float]]:
    """Return max/RMS error metrics for all available derivative arrays."""
    if analytic.channel != finite_difference.channel:
        raise ValueError("analytic and finite-difference channels differ")
    if analytic.fixed_filling != finite_difference.fixed_filling:
        raise ValueError("analytic and finite-difference ensembles differ")

    out = {
        "Gamma": _array_error_metrics(analytic.Gamma, finite_difference.Gamma_phi),
        "G": _array_error_metrics(analytic.G_phi, finite_difference.G_phi),
        "P": _array_error_metrics(analytic.P_phi, finite_difference.P_phi),
        "W": _array_error_metrics(analytic.W_phi, finite_difference.W_phi),
        "Sigma_H": _array_error_metrics(analytic.Sigma_H_phi, finite_difference.Sigma_H_phi),
        "Sigma_GW": _array_error_metrics(analytic.Sigma_GW_phi, finite_difference.Sigma_GW_phi),
        "density": _array_error_metrics(analytic.density_phi, finite_difference.density_phi),
    }
    mu_scale = max(abs(float(analytic.mu_phi)), abs(float(finite_difference.mu_phi)), 1e-14)
    mu_abs = abs(float(analytic.mu_phi) - float(finite_difference.mu_phi))
    out["mu"] = {"abs_max": mu_abs, "rms": mu_abs, "scale_max": mu_scale, "rel_max": mu_abs / mu_scale}
    return out
