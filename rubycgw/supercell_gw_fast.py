"""Performance-oriented 18-site period-three Ruby GW solver.

This module keeps the physics and fixed-point equations of :mod:`supercell_gw`
unchanged, but removes two expensive pieces of numerical overhead that become
important for the 18-site problem:

1. the Hartree-reference eigensystem used by the Matsubara tail subtraction is
   built once per self-consistency iterate rather than once per trial chemical
   potential;
2. the fixed-filling chemical potential is solved by a warm-started,
   safeguarded secant/bracketing method rather than a fresh wide bisection.

The returned object is the same ``GWResult`` used everywhere else in RubycGW.
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
    compute_sigma_gw_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
    screening_soft_modes_matrix,
)


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
    """Warm-started safeguarded secant solve for the fixed-filling chemical potential.

    The filling is monotone in ``mu`` for the converged Matsubara expression.
    Starting from the previous GW iteration's chemical potential therefore lets
    most self-consistency iterations bracket the root locally.  Secant steps are
    accepted only inside the current bracket; otherwise a bisection step is used.
    """
    cache = _build_tail_cache(h0, sigma_h)
    neval = 0

    def f(mu: float) -> tuple[float, np.ndarray]:
        nonlocal neval
        G = dyson_from_sigma_matrix(h0, grid, float(mu), sigma_h, sigma_gw)
        density = density_from_G_cached(G, grid, float(mu), cache)
        neval += 1
        return float(np.sum(density) - target), G

    x0 = float(mu0)
    f0, G0 = f(x0)
    if abs(f0) < tol:
        return x0, G0, cache, neval

    step = max(0.10, 4.0 * float(grid.T))
    if f0 < 0.0:
        lo, flo, Glo = x0, f0, G0
        hi = x0 + step
        fhi, Ghi = f(hi)
        for _ in range(30):
            if fhi >= 0.0:
                break
            lo, flo, Glo = hi, fhi, Ghi
            step *= 2.0
            hi = x0 + step
            fhi, Ghi = f(hi)
        else:
            raise RuntimeError(
                "Could not bracket supercell chemical potential above warm start; "
                f"target={target}, f(mu0)={f0}, f(hi)={fhi}"
            )
    else:
        hi, fhi, Ghi = x0, f0, G0
        lo = x0 - step
        flo, Glo = f(lo)
        for _ in range(30):
            if flo <= 0.0:
                break
            hi, fhi, Ghi = lo, flo, Glo
            step *= 2.0
            lo = x0 - step
            flo, Glo = f(lo)
        else:
            raise RuntimeError(
                "Could not bracket supercell chemical potential below warm start; "
                f"target={target}, f(mu0)={f0}, f(lo)={flo}"
            )

    if abs(flo) <= abs(fhi):
        best_mu, best_f, best_G = lo, flo, Glo
    else:
        best_mu, best_f, best_G = hi, fhi, Ghi

    for _ in range(max(int(max_iter), 1)):
        width = hi - lo
        if width <= 0.0:
            break

        denom = fhi - flo
        if np.isfinite(denom) and abs(denom) > 1e-16:
            trial = hi - fhi * width / denom
        else:
            trial = 0.5 * (lo + hi)

        margin = 0.08 * width
        if (
            not np.isfinite(trial)
            or trial <= lo + margin
            or trial >= hi - margin
        ):
            trial = 0.5 * (lo + hi)

        ftrial, Gtrial = f(float(trial))
        if abs(ftrial) < abs(best_f):
            best_mu, best_f, best_G = float(trial), ftrial, Gtrial
        if abs(ftrial) < tol:
            return float(trial), Gtrial, cache, neval

        if ftrial > 0.0:
            hi, fhi, Ghi = float(trial), ftrial, Gtrial
        else:
            lo, flo, Glo = float(trial), ftrial, Gtrial

    return float(best_mu), best_G, cache, neval


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
        sigma_gw = np.zeros((grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex)
        mu = float(opts.mu)

    if opts.target_filling is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
        tail_cache = _build_tail_cache(h0, sigma_h)
        mu_neval = 0
    else:
        mu, G, tail_cache, mu_neval = _solve_mu_matrix_fast(
            h0, sigma_h, sigma_gw, grid, float(opts.target_filling),
            mu, opts.mu_tol, opts.mu_max_iter,
        )

    W = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
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
        sigma_gw_out = compute_sigma_gw_matrix(G, W, grid, backend=backend)

        res_h = sigma_h_out - sigma_h
        res_gw = sigma_gw_out - sigma_gw
        err = _residual_error(res_h, res_gw)

        if opts.verbose:
            print(
                f"SC-GW iter {it:4d}: residual={err:.3e}, mu={mu:.10f}, "
                f"n={np.sum(density):.10f}, mu_eval={mu_neval}, "
                f"method={method}, backend={backend}"
            )
        if err < opts.tol:
            converged = True
            break

        sigma_h_next, sigma_gw_next = _mixed_self_energies(
            sigma_h, sigma_gw, sigma_h_out, sigma_gw_out,
            opts, it, history,
        )

        if opts.target_filling is None:
            Gnext = dyson_from_sigma_matrix(h0, grid, mu, sigma_h_next, sigma_gw_next)
            tail_cache_next = _build_tail_cache(h0, sigma_h_next)
            mu_neval_next = 0
        else:
            mu, Gnext, tail_cache_next, mu_neval_next = _solve_mu_matrix_fast(
                h0, sigma_h_next, sigma_gw_next, grid, float(opts.target_filling),
                mu, opts.mu_tol, opts.mu_max_iter,
            )

        sigma_h = sigma_h_next
        sigma_gw = sigma_gw_next
        G = Gnext
        tail_cache = tail_cache_next
        mu_neval = mu_neval_next

    density = density_from_G_cached(G, grid, mu, tail_cache)
    sigma_h_out = hartree_self_energy_matrix(density, Vq0)
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_matrix(G, W, grid, backend=backend)
    err = _residual_error(sigma_h_out - sigma_h, sigma_gw_out - sigma_gw)
    converged = bool(err < opts.tol)

    (
        smin, mmin, omin, q1min, q2min,
        screening_mode, density_mode, density_mode_residual,
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
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=source_strength)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    if h0.shape[-1] != NSUP:
        raise RuntimeError("unexpected supercell matrix dimension")
    return solve_matrix_gw_fast(h0, Vq, grid, opts=opts, initial=initial)
