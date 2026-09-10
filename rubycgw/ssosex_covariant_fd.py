"""Static covariant response of the self-consistent GW+sSOSEX closure.

This module evaluates the *full static tangent* of the converged no-Gamma
GW+sSOSEX approximation by central finite differences of the self-consistent
fixed point.  The purpose is diagnostic: it provides a reference implementation
of the covariant derivative that automatically contains every derivative path
of the approximate closure, including the screened-line derivative of sSOSEX.

The background approximation is

    P[G] = G G,
    W[G] = (1 - V P[G])^{-1} V,

    Sigma[G] = Sigma_H[G]
             + Sigma_F[G]
             - G (W[G]-V)
             + Sigma_sSOSEX[G,W[G]].

For the default static one-screened-line correction,

    Sigma_sSOSEX = 1/2 { S[V,W0] + S[W0,V] },

with

    S[U,X]_ij(tau)
      = sum_kl U_il X_kj G_ik(tau) G_kl(-tau) G_lj(tau),

and W0(q)=W(q,Omega=0).

Analytically, the covariant kernel contains both

    (d Sigma_sSOSEX / dG)_W

(the three terms obtained by replacing one of the three G lines by the tangent
X=G Gamma G) and

    (d Sigma_sSOSEX / dW0)_G * dW0/dG.

For ordinary GW screening dW = W (dP) W and dP contains the two possible
insertions of X into P=GG.  Hand-coding the complete finite-transfer routing of
this second term is cumbersome and easy to get wrong.  A central finite
difference of the *fully converged* fixed point is, however, exactly the same
static covariant derivative up to O(eps^2), and automatically differentiates
Hartree, Fock, GW/MT+AL, the explicit G dependence of sSOSEX, and its W[G]
dependence.

The source convention is

    h0(phi) = h0 - phi K,

so the returned response is d<O>/dphi.  Perturbed calculations are performed
at fixed chemical potential, as required for a thermodynamic/covariant
susceptibility.  The unperturbed background may still have been obtained at a
fixed target filling.

The current implementation intentionally targets static same-torus benchmarks
(nk1=nk2=1).  This lets a 2x1 Ruby torus be treated as one 12-orbital cell and
makes the full static density response a finite orbital matrix, which can then
be used in the post identity W_post(0)=V-V chi_nn V.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .grids import MatsubaraGrid
from .gw import GWOptions
from .gw_ssosex import GWScreenedSOSEXResult, solve_matrix_gw_ssosex
from .post_gw import one_shot_post_dyson
from .ssosex_static import (
    ScreenedSOSEXOptions,
    compute_static_screened_sosex_self_energy_periodic_fast,
)


@dataclass
class StaticFDObservableResult:
    value: complex
    plus_value: complex
    minus_value: complex
    plus: GWScreenedSOSEXResult
    minus: GWScreenedSOSEXResult
    eps: float


@dataclass
class StaticFDDensityResult:
    chi_raw: np.ndarray
    chi_sym: np.ndarray
    reciprocity_error: float
    plus_residual: np.ndarray
    minus_residual: np.ndarray
    converged: np.ndarray
    eps: float


def _fermi(e_minus_mu: np.ndarray, T: float) -> np.ndarray:
    x = np.asarray(e_minus_mu, dtype=float) / float(T)
    out = np.empty_like(x)
    hi = x > 40.0
    lo = x < -40.0
    mid = ~(hi | lo)
    out[hi] = 0.0
    out[lo] = 1.0
    out[mid] = 1.0 / (np.exp(x[mid]) + 1.0)
    return out


def onebody_expectation_tail_completed(
    result: GWScreenedSOSEXResult,
    K: np.ndarray,
    h0: np.ndarray,
    grid: MatsubaraGrid,
) -> complex:
    """Return <K> with a static H+F reference-tail completion."""
    G = np.asarray(result.G, dtype=complex)
    K = np.asarray(K, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    if G.shape[1:3] != h0.shape[:2] or h0.shape[-2:] != K.shape:
        raise ValueError("G/h0/K shape mismatch")
    h_static = (
        h0
        + np.asarray(result.Sigma_H, dtype=complex)[None, None, :, :]
        + np.asarray(result.Sigma_F, dtype=complex)
    )
    h_static = 0.5 * (h_static + np.swapaxes(h_static.conj(), -1, -2))
    evals, evecs = np.linalg.eigh(h_static)
    occ = _fermi(evals - float(result.mu), float(grid.T))
    rho_ref = np.einsum(
        "xyai,xyi,xybi->xyab", evecs, occ, evecs.conj(), optimize=True
    )
    ref_value = (1.0 / float(grid.nk)) * np.einsum(
        "ab,xyba->", K, rho_ref, optimize=True
    )
    denom = (
        1j * np.asarray(grid.omega)[:, None, None, None]
        + float(result.mu)
        - evals[None, :, :, :]
    )
    Gref = np.einsum(
        "xyai,nxyi,xybi->nxyab",
        evecs,
        1.0 / denom,
        evecs.conj(),
        optimize=True,
    )
    corr = (float(grid.T) / float(grid.nk)) * np.einsum(
        "ab,nxyba->", K, G - Gref, optimize=True
    )
    return complex(ref_value + corr)


def _fixed_mu_options(
    opts: GWOptions,
    mu: float,
    *,
    tol: float | None = None,
    max_iter: int | None = None,
    mixing: float | None = None,
) -> GWOptions:
    """Clone solver options for a static derivative at fixed chemical potential."""
    return replace(
        opts,
        target_filling=None,
        mu=float(mu),
        tol=float(opts.tol if tol is None else tol),
        max_iter=int(opts.max_iter if max_iter is None else max_iter),
        mixing=float(opts.mixing if mixing is None else mixing),
    )


def solve_fixed_mu_perturbation(
    background: GWScreenedSOSEXResult,
    h0: np.ndarray,
    Vq: np.ndarray,
    source: np.ndarray,
    phi: float,
    grid: MatsubaraGrid,
    *,
    gw_opts: GWOptions,
    ssosex_opts: ScreenedSOSEXOptions,
    tol: float | None = None,
    max_iter: int | None = None,
    mixing: float | None = None,
) -> GWScreenedSOSEXResult:
    """Re-solve the full GW+sSOSEX fixed point for h0-phi*source at fixed mu."""
    h0 = np.asarray(h0, dtype=complex)
    source = np.asarray(source, dtype=complex)
    norb = int(h0.shape[-1])
    if source.shape != (norb, norb):
        raise ValueError("source must be an orbital-space Hermitian matrix")
    hp = h0 - float(phi) * source[None, None, :, :]
    hp = 0.5 * (hp + np.swapaxes(hp.conj(), -1, -2))
    opts = _fixed_mu_options(
        gw_opts,
        float(background.mu),
        tol=tol,
        max_iter=max_iter,
        mixing=mixing,
    )
    out = solve_matrix_gw_ssosex(
        hp,
        Vq,
        grid,
        opts=opts,
        ssosex_opts=ssosex_opts,
        initial=background,
    )
    # target_filling=None means mu is a fixed external parameter.  Keep this
    # assertion explicit because accidentally differentiating at fixed density
    # would give the wrong density response for post screening.
    if abs(float(out.mu) - float(background.mu)) > 1.0e-12:
        raise RuntimeError("fixed-mu perturbation unexpectedly changed chemical potential")
    return out


def finite_difference_static_observable(
    background: GWScreenedSOSEXResult,
    h0: np.ndarray,
    Vq: np.ndarray,
    source: np.ndarray,
    observable: np.ndarray,
    grid: MatsubaraGrid,
    *,
    eps: float,
    gw_opts: GWOptions,
    ssosex_opts: ScreenedSOSEXOptions,
    tol: float | None = None,
    max_iter: int | None = None,
    mixing: float | None = None,
) -> StaticFDObservableResult:
    """Central static derivative d<observable>/dphi for source -phi*source."""
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    plus = solve_fixed_mu_perturbation(
        background,
        h0,
        Vq,
        source,
        +eps,
        grid,
        gw_opts=gw_opts,
        ssosex_opts=ssosex_opts,
        tol=tol,
        max_iter=max_iter,
        mixing=mixing,
    )
    minus = solve_fixed_mu_perturbation(
        background,
        h0,
        Vq,
        source,
        -eps,
        grid,
        gw_opts=gw_opts,
        ssosex_opts=ssosex_opts,
        tol=tol,
        max_iter=max_iter,
        mixing=mixing,
    )
    hplus = np.asarray(h0, dtype=complex) - eps * np.asarray(source)[None, None]
    hminus = np.asarray(h0, dtype=complex) + eps * np.asarray(source)[None, None]
    op = np.asarray(observable, dtype=complex)
    vp = onebody_expectation_tail_completed(plus, op, hplus, grid)
    vm = onebody_expectation_tail_completed(minus, op, hminus, grid)
    return StaticFDObservableResult(
        value=(vp - vm) / (2.0 * eps),
        plus_value=vp,
        minus_value=vm,
        plus=plus,
        minus=minus,
        eps=eps,
    )


def finite_difference_density_response_same_torus(
    background: GWScreenedSOSEXResult,
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    *,
    eps: float,
    gw_opts: GWOptions,
    ssosex_opts: ScreenedSOSEXOptions,
    tol: float | None = None,
    max_iter: int | None = None,
    mixing: float | None = None,
    allow_unconverged: bool = False,
) -> StaticFDDensityResult:
    """Full static orbital density matrix chi_nn on an nk=1 same-torus cell.

    chi[a,b] = d<n_a>/dphi_b with H_source=-phi_b n_b.  Each column is obtained
    by re-solving the complete self-consistent GW+sSOSEX closure at +/-eps while
    keeping the background chemical potential fixed.
    """
    if int(grid.nk1) != 1 or int(grid.nk2) != 1:
        raise NotImplementedError(
            "same-torus finite-difference density response currently requires nk=1"
        )
    norb = int(np.asarray(h0).shape[-1])
    chi = np.zeros((norb, norb), dtype=complex)
    rp = np.full(norb, np.nan)
    rm = np.full(norb, np.nan)
    ok = np.zeros(norb, dtype=bool)
    for b in range(norb):
        Kb = np.zeros((norb, norb), dtype=complex)
        Kb[b, b] = 1.0
        plus = solve_fixed_mu_perturbation(
            background,
            h0,
            Vq,
            Kb,
            +float(eps),
            grid,
            gw_opts=gw_opts,
            ssosex_opts=ssosex_opts,
            tol=tol,
            max_iter=max_iter,
            mixing=mixing,
        )
        minus = solve_fixed_mu_perturbation(
            background,
            h0,
            Vq,
            Kb,
            -float(eps),
            grid,
            gw_opts=gw_opts,
            ssosex_opts=ssosex_opts,
            tol=tol,
            max_iter=max_iter,
            mixing=mixing,
        )
        rp[b] = float(plus.final_error)
        rm[b] = float(minus.final_error)
        ok[b] = bool(plus.converged and minus.converged)
        if not ok[b] and not allow_unconverged:
            raise RuntimeError(
                f"density finite difference failed for orbital {b}: "
                f"r+={rp[b]:.3e}, r-={rm[b]:.3e}"
            )
        chi[:, b] = (
            np.asarray(plus.density, dtype=complex)
            - np.asarray(minus.density, dtype=complex)
        ) / (2.0 * float(eps))

    # A static equilibrium density susceptibility is reciprocal/Hermitian.
    # Keep the raw matrix for diagnostics and use the symmetrized matrix for the
    # post screened interaction to suppress O(eps^2) and solver noise.
    anti = chi - chi.conj().T
    denom = max(float(np.linalg.norm(chi.ravel())), 1.0e-300)
    rec_err = float(np.linalg.norm(anti.ravel()) / denom)
    chi_sym = 0.5 * (chi + chi.conj().T)
    return StaticFDDensityResult(
        chi_raw=chi,
        chi_sym=chi_sym,
        reciprocity_error=rec_err,
        plus_residual=rp,
        minus_residual=rm,
        converged=ok,
        eps=float(eps),
    )


def build_static_post_w_same_torus(
    Vq: np.ndarray,
    W_background: np.ndarray,
    chi_nn_static: np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Replace only the Omega=0 same-torus W by V-V chi_nn V."""
    if int(grid.nk1) != 1 or int(grid.nk2) != 1:
        raise NotImplementedError("static same-torus post W currently requires nk=1")
    Vq = np.asarray(Vq, dtype=complex)
    Wbg = np.asarray(W_background, dtype=complex)
    chi = np.asarray(chi_nn_static, dtype=complex)
    norb = int(Vq.shape[-1])
    if Vq.shape != (1, 1, norb, norb):
        raise ValueError("unexpected same-torus Vq shape")
    if Wbg.shape != (grid.nb, 1, 1, norb, norb):
        raise ValueError("unexpected same-torus W shape")
    if chi.shape != (norb, norb):
        raise ValueError("chi_nn_static shape mismatch")
    m0 = np.flatnonzero(np.asarray(grid.m_values, dtype=int) == 0)
    if m0.size != 1:
        raise ValueError("bosonic grid must contain one Omega=0 sector")
    out = np.array(Wbg, copy=True)
    v = Vq[0, 0]
    out[int(m0[0]), 0, 0] = v - v @ chi @ v
    return out


def one_shot_post_ssosex_same_torus(
    background: GWScreenedSOSEXResult,
    h0: np.ndarray,
    Vq: np.ndarray,
    W_post: np.ndarray,
    grid: MatsubaraGrid,
    *,
    target_filling: float | None,
    gw_opts: GWOptions,
    ssosex_opts: ScreenedSOSEXOptions,
) -> dict[str, np.ndarray | float]:
    """One-shot post Dyson step, recomputing both GW and sSOSEX with W_post."""
    h0 = np.asarray(h0, dtype=complex)
    h_x_ref = (
        h0
        + np.asarray(background.Sigma_H, dtype=complex)[None, None, :, :]
        + np.asarray(background.Sigma_F, dtype=complex)
    )
    sigma_ssosex_post = compute_static_screened_sosex_self_energy_periodic_fast(
        np.asarray(background.G, dtype=complex),
        np.asarray(Vq, dtype=complex),
        np.asarray(W_post, dtype=complex),
        h_x_ref,
        float(background.mu),
        grid,
        opts=ssosex_opts,
    )
    step = one_shot_post_dyson(
        np.asarray(background.G, dtype=complex),
        np.asarray(W_post, dtype=complex),
        np.asarray(Vq, dtype=complex),
        h0,
        np.asarray(background.Sigma_H, dtype=complex),
        float(background.mu),
        grid,
        target_filling=target_filling,
        backend=str(gw_opts.momentum_backend),
        mu_tol=float(gw_opts.mu_tol),
        mu_max_iter=int(gw_opts.mu_max_iter),
        sigma_extra=sigma_ssosex_post,
    )
    step["Sigma_sSOSEX_post"] = np.asarray(sigma_ssosex_post)
    return step


__all__ = [
    "StaticFDObservableResult",
    "StaticFDDensityResult",
    "onebody_expectation_tail_completed",
    "solve_fixed_mu_perturbation",
    "finite_difference_static_observable",
    "finite_difference_density_response_same_torus",
    "build_static_post_w_same_torus",
    "one_shot_post_ssosex_same_torus",
]
