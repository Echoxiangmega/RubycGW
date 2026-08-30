"""Performance-oriented 18-site period-three Ruby GW solver.

This module keeps the infinite-frequency GW equations of :mod:`supercell_gw`
but evaluates the self-energy as a static bare-V Fock term plus a finite
Matsubara convolution of the decaying retarded part ``W-V``.  This removes the
artificial bosonic-cutoff dependence caused by truncating the non-decaying bare
interaction inside ``-G W``.

The numerical optimizations are:
1. the Hartree-reference eigensystem used by the Matsubara tail subtraction is
   built once per self-consistency iterate rather than once per trial chemical
   potential;
2. the fixed-filling chemical potential is solved by a warm-started safeguarded
   Newton method, using the analytic derivative of the numerical filling with
   respect to ``mu``;
3. intermediate GW iterates use an inexact inner ``mu`` tolerance and tighten
   automatically near the fixed point.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid
from .gw import (
    GWOptions,
    GWResult,
    _check_backend,
    _check_mixing_method,
    _mixed_self_energies,
    _residual_error,
)
from .model import RubyParameters
from .supercell import NSUP, build_supercell_h0, build_supercell_interaction
from .supercell_gw import (
    _compatible_initial,
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
    screening_soft_modes_matrix,
)
from .supercell_gw_split import compute_sigma_gw_split_matrix


_MU_LOOSE_TOL = 1.0e-4
_MU_RESIDUAL_FACTOR = 1.0e-2


@dataclass(frozen=True)
class _TailCache:
    evals: np.ndarray
    weights: np.ndarray


def _fermi(e_minus_mu: np.ndarray, T: float) -> np.ndarray:
    x = np.asarray(e_minus_mu, dtype=float) / float(T)
    out = np.empty_like(x)
    high = x > 40.0
    low = x < -40.0
    mid = ~(high | low)
    out[high] = 0.0
    out[low] = 1.0
    out[mid] = 1.0 / (np.exp(x[mid]) + 1.0)
    return out


def _build_tail_cache(h0: np.ndarray, sigma_h: np.ndarray) -> _TailCache:
    """Diagonalize the static reference Hamiltonian once for a GW iterate."""
    href = h0 + sigma_h[None, None, :, :]
    href = 0.5 * (href + np.swapaxes(href.conj(), -1, -2))
    evals, evecs = np.linalg.eigh(href)
    weights = np.abs(evecs) ** 2
    return _TailCache(evals=evals, weights=weights)


def density_from_G_cached(
    G: np.ndarray,
    grid: MatsubaraGrid,
    mu: float,
    cache: _TailCache,
) -> np.ndarray:
    """Tail-subtracted orbital density using a precomputed static eigensystem."""
    diag = np.diagonal(G, axis1=-2, axis2=-1)
    evals = cache.evals
    weights = cache.weights

    occ = _fermi(evals - float(mu), grid.T)
    n_ref = np.sum(weights * occ[..., None, :], axis=(0, 1, 3)) / grid.nk

    denom = (
        1j * grid.omega[:, None, None, None]
        + float(mu)
        - evals[None, :, :, :]
    )
    gref_diag = np.einsum(
        "xyaj,nxyj->nxya", weights, 1.0 / denom, optimize=True
    )
    correction = (
        (grid.T / grid.nk)
        * np.sum(diag - gref_diag, axis=(0, 1, 2)).real
    )
    return n_ref + correction


def _total_filling_slope(
    G: np.ndarray,
    grid: MatsubaraGrid,
    mu: float,
    cache: _TailCache,
) -> float:
    """Derivative dN/dmu of the same tail-subtracted numerical filling."""
    evals = cache.evals
    occ = _fermi(evals - float(mu), grid.T)
    d_ref = float(np.sum(occ * (1.0 - occ)) / (grid.T * grid.nk))

    tr_g2 = np.einsum("nxyab,nxyba->nxy", G, G, optimize=True)
    denom = (
        1j * grid.omega[:, None, None, None]
        + float(mu)
        - evals[None, :, :, :]
    )
    tr_gref2 = np.sum(1.0 / (denom * denom), axis=-1)
    d_corr = float(
        (grid.T / grid.nk)
        * np.sum(-tr_g2 + tr_gref2).real
    )
    return d_ref + d_corr


def _effective_mu_tol(
    strict_tol: float,
    outer_residual: float | None,
    loose_tol: float = _MU_LOOSE_TOL,
    residual_factor: float = _MU_RESIDUAL_FACTOR,
) -> float:
    """Tolerance for an inexact fixed-filling inner solve."""
    strict = float(strict_tol)
    loose = max(strict, float(loose_tol))
    if outer_residual is None or not np.isfinite(float(outer_residual)):
        return loose
    dynamic = float(residual_factor) * max(float(outer_residual), 0.0)
    return max(strict, min(loose, dynamic))


def _solve_mu_matrix_fast(
    h0: np.ndarray,
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
    grid: MatsubaraGrid,
    target: float,
    mu0: float,
    tol: float,
    max_iter: int,
) -> tuple[float, np.ndarray, _TailCache, int]:
    """Warm-started safeguarded Newton solve for fixed filling."""
    cache = _build_tail_cache(h0, sigma_h)
    neval = 0

    def evaluate(mu: float) -> tuple[float, np.ndarray, float]:
        nonlocal neval
        value = float(mu)
        G = dyson_from_sigma_matrix(h0, grid, value, sigma_h, sigma_gw)
        density = density_from_G_cached(G, grid, value, cache)
        residual = float(np.sum(density) - target)
        slope = _total_filling_slope(G, grid, value, cache)
        neval += 1
        return residual, G, float(slope)

    x = float(mu0)
    fx, Gx, dfx = evaluate(x)
    if abs(fx) < tol:
        return x, Gx, cache, neval

    best_mu, best_f, best_G = x, fx, Gx
    lo = None
    hi = None
    step_cap = max(0.50, 12.0 * float(grid.T))

    for _ in range(max(int(max_iter), 1)):
        if fx < 0.0:
            lo = (x, fx, Gx)
        elif fx > 0.0:
            hi = (x, fx, Gx)
        else:
            return x, Gx, cache, neval

        use_newton = bool(np.isfinite(dfx) and dfx > 1.0e-10)
        if use_newton:
            trial = x - fx / dfx
        else:
            trial = x + (step_cap if fx < 0.0 else -step_cap)

        if lo is not None and hi is not None:
            xlo, _, _ = lo
            xhi, _, _ = hi
            if not np.isfinite(trial) or trial <= xlo or trial >= xhi:
                trial = 0.5 * (xlo + xhi)
        else:
            if not np.isfinite(trial):
                trial = x + (step_cap if fx < 0.0 else -step_cap)
            delta = float(np.clip(trial - x, -step_cap, step_cap))
            if abs(delta) < 1.0e-12:
                delta = step_cap if fx < 0.0 else -step_cap
            trial = x + delta

        ftrial, Gtrial, dftrial = evaluate(float(trial))
        if abs(ftrial) < abs(best_f):
            best_mu, best_f, best_G = float(trial), ftrial, Gtrial
        if abs(ftrial) < tol:
            return float(trial), Gtrial, cache, neval

        x, fx, Gx, dfx = float(trial), ftrial, Gtrial, dftrial

    return float(best_mu), best_G, cache, neval


def _strict_refine_fixed_filling(
    h0: np.ndarray,
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
    grid: MatsubaraGrid,
    target: float | None,
    mu: float,
    strict_tol: float,
    max_iter: int,
) -> tuple[float, np.ndarray, _TailCache, int]:
    if target is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
        return float(mu), G, _build_tail_cache(h0, sigma_h), 0
    return _solve_mu_matrix_fast(
        h0,
        sigma_h,
        sigma_gw,
        grid,
        float(target),
        float(mu),
        float(strict_tol),
        int(max_iter),
    )


def solve_matrix_gw_fast(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    initial: GWResult | None = None,
) -> GWResult:
    """Solve matrix-valued periodic GW with cached tails and fast fixed filling."""
    backend = _check_backend(opts.momentum_backend)
    method = _check_mixing_method(opts.mixing_method)
    if opts.pulay_history < 2:
        raise ValueError("pulay_history must be at least 2")
    if opts.pulay_start < 1:
        raise ValueError("pulay_start must be at least 1")

    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    expected_h = (grid.nk1, grid.nk2, norb, norb)
    expected_v = (grid.nk1, grid.nk2, norb, norb)
    if h0.shape != expected_h:
        raise ValueError(f"h0 shape {h0.shape} != expected {expected_h}")
    if Vq.shape != expected_v:
        raise ValueError(f"Vq shape {Vq.shape} != expected {expected_v}")
    Vq0 = Vq[0, 0]

    if _compatible_initial(initial, grid, norb):
        sigma_h = np.array(initial.Sigma_H, copy=True)
        sigma_gw = np.array(initial.Sigma_GW, copy=True)
        mu = float(initial.mu)
    else:
        sigma_h = np.zeros((norb, norb), dtype=complex)
        sigma_gw = np.zeros(
            (grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex
        )
        mu = float(opts.mu)

    initial_mu_tol = _effective_mu_tol(opts.mu_tol, None)
    if opts.target_filling is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
        tail_cache = _build_tail_cache(h0, sigma_h)
        mu_neval = 0
    else:
        mu, G, tail_cache, mu_neval = _solve_mu_matrix_fast(
            h0,
            sigma_h,
            sigma_gw,
            grid,
            float(opts.target_filling),
            mu,
            initial_mu_tol,
            opts.mu_max_iter,
        )
    mu_tol_used = initial_mu_tol

    W = np.zeros(
        (grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex
    )
    P = np.zeros_like(W)
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    err = float("inf")
    converged = False
    it = 0
    for it in range(1, opts.max_iter + 1):
        density = density_from_G_cached(G, grid, mu, tail_cache)
        sigma_h_out = hartree_self_energy_matrix(density, Vq0)
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_out = compute_sigma_gw_split_matrix(
            G, W, Vq, grid, h0, mu, sigma_h, backend=backend
        )

        res_h = sigma_h_out - sigma_h
        res_gw = sigma_gw_out - sigma_gw
        err = _residual_error(res_h, res_gw)

        if opts.verbose:
            print(
                f"SC-GW iter {it:4d}: residual={err:.3e}, mu={mu:.10f}, "
                f"n={np.sum(density):.10f}, mu_eval={mu_neval}, "
                f"mu_tol={mu_tol_used:.1e}, method={method}, backend={backend}"
            )
        if err < opts.tol:
            converged = True
            break

        sigma_h_next, sigma_gw_next = _mixed_self_energies(
            sigma_h,
            sigma_gw,
            sigma_h_out,
            sigma_gw_out,
            opts,
            it,
            history,
        )

        mu_tol_next = _effective_mu_tol(opts.mu_tol, err)
        if opts.target_filling is None:
            Gnext = dyson_from_sigma_matrix(
                h0, grid, mu, sigma_h_next, sigma_gw_next
            )
            tail_cache_next = _build_tail_cache(h0, sigma_h_next)
            mu_neval_next = 0
        else:
            mu, Gnext, tail_cache_next, mu_neval_next = _solve_mu_matrix_fast(
                h0,
                sigma_h_next,
                sigma_gw_next,
                grid,
                float(opts.target_filling),
                mu,
                mu_tol_next,
                opts.mu_max_iter,
            )

        sigma_h = sigma_h_next
        sigma_gw = sigma_gw_next
        G = Gnext
        tail_cache = tail_cache_next
        mu_neval = mu_neval_next
        mu_tol_used = mu_tol_next

    mu, G, tail_cache, mu_neval_final = _strict_refine_fixed_filling(
        h0,
        sigma_h,
        sigma_gw,
        grid,
        opts.target_filling,
        mu,
        opts.mu_tol,
        opts.mu_max_iter,
    )
    density = density_from_G_cached(G, grid, mu, tail_cache)
    sigma_h_out = hartree_self_energy_matrix(density, Vq0)
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_split_matrix(
        G, W, Vq, grid, h0, mu, sigma_h, backend=backend
    )
    err = _residual_error(sigma_h_out - sigma_h, sigma_gw_out - sigma_gw)
    converged = bool(err < opts.tol)

    if opts.verbose and opts.target_filling is not None:
        print(
            f"SC-GW strict mu refine: mu={mu:.10f}, "
            f"n={np.sum(density):.10f}, mu_eval={mu_neval_final}, "
            f"mu_tol={opts.mu_tol:.1e}, residual={err:.3e}"
        )

    (
        smin,
        mmin,
        omin,
        q1min,
        q2min,
        screening_mode,
        density_mode,
        density_mode_residual,
    ) = screening_soft_modes_matrix(P, Vq, grid)

    return GWResult(
        G=G,
        W=W,
        P=P,
        Sigma_H=sigma_h,
        Sigma_GW=sigma_gw,
        mu=mu,
        density=density,
        converged=converged,
        iterations=it,
        final_error=float(err),
        mixing_method=method,
        min_screening_singular_value=smin,
        min_screening_m=mmin,
        min_screening_Omega=omin,
        min_screening_q1=q1min,
        min_screening_q2=q2min,
        min_screening_mode=screening_mode,
        min_density_mode=density_mode,
        min_density_mode_residual=density_mode_residual,
    )


def solve_supercell_gw_fast(
    params: RubyParameters,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    source_strength: float = 0.0,
    initial: GWResult | None = None,
) -> GWResult:
    """Solve the 18-site Q=(1/3,1/3)-compatible supercell with fast numerics."""
    h0 = build_supercell_h0(
        grid.kmesh(), params, source_strength=source_strength
    )
    Vq = build_supercell_interaction(grid.qmesh(), params)
    if h0.shape[-1] != NSUP:
        raise RuntimeError("unexpected supercell matrix dimension")
    return solve_matrix_gw_fast(h0, Vq, grid, opts=opts, initial=initial)
