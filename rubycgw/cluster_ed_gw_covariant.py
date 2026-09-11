"""Numerical covariant response of the cluster-ED+GW fixed point.

The ordinary cGW vertex differentiates the GW self-energy.  For the embedded
cluster method the self-energy is instead

    Sigma_tot = Sigma_GW^lat + Sigma_ED^C - Sigma_GW^C,

and a response that differentiates only the GW part is therefore not the
covariant response of the actual approximation.  This module differentiates
the *whole converged fixed-point map* numerically by solving the source-shifted
problem

    h0 -> h0 - h K

from a warm start and taking a symmetric finite difference.  The source solve
keeps the lattice filling fixed, so the chemical potential, GW background,
finite bath and impurity self-energy all relax together.

Loop-current sources make the Weiss hybridization genuinely complex.  The
production zero-field embedding historically used a real finite bath; here the
source solver uses a complex bath fit (real bath energies, complex couplings)
so the TR-odd response is not artificially projected onto a real bath
manifold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    build_impurity_one_body,
    cluster_gw_self_energy,
    ruby_cluster_interactions,
)
from .grids import MatsubaraGrid
from .gw import GWOptions, _mixed_self_energies
from .impurity_ed import FiniteBathImpurityED
from .model import NSUB, RubyParameters
from .supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    hartree_self_energy_matrix,
)
from .supercell_gw_fast import _solve_mu_matrix_fast
from .supercell_gw_split import (
    compute_sigma_gw_split_matrix,
    one_body_density_matrix_tail,
)


@dataclass
class ClusterCovariantState:
    Sigma_H: np.ndarray
    Sigma_emb: np.ndarray
    Sigma_imp: np.ndarray
    G: np.ndarray
    mu: float
    bath: BathParameters


@dataclass(frozen=True)
class ClusterCovariantOptions:
    max_iter: int = 60
    tol: float = 2.0e-6
    mixing: float = 0.70
    pulay_history: int = 6
    pulay_start: int = 3
    pulay_regularization: float = 1.0e-7
    pulay_step_cap: float = 3.0
    nbath: int = 6
    bath_fit_nfreq: int = 12
    bath_fit_max_nfev: int = 500
    bath_energy_window: float = 4.0
    bath_coupling_bound: float = 4.0
    bath_fit_xtol: float = 1.0e-10
    discard_weight_tol: float = 1.0e-11
    mu_tol: float = 1.0e-11
    mu_max_iter: int = 80
    verbose: bool = True


@dataclass
class ClusterCovariantSourceResult:
    state: ClusterCovariantState
    expectation: float
    converged: bool
    iterations: int
    final_error: float
    impurity_mismatch: float
    bath_fit_error: float
    pulay_fallbacks: int


def _maxabs(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a)), initial=0.0))


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(b).ravel())), 1.0e-300)
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel()) / den)


def _pack_dynamic(sigma_emb: np.ndarray, sigma_imp: np.ndarray, nk: int) -> np.ndarray:
    scale = np.sqrt(float(max(int(nk), 1)))
    return np.concatenate([
        np.asarray(sigma_emb, dtype=complex).ravel(),
        scale * np.asarray(sigma_imp, dtype=complex).ravel(),
    ])


def _unpack_dynamic(
    packed: np.ndarray,
    sigma_emb_shape: tuple[int, ...],
    sigma_imp_shape: tuple[int, ...],
    nk: int,
) -> tuple[np.ndarray, np.ndarray]:
    nemb = int(np.prod(sigma_emb_shape))
    scale = np.sqrt(float(max(int(nk), 1)))
    flat = np.asarray(packed, dtype=complex).reshape(-1)
    return (
        flat[:nemb].reshape(sigma_emb_shape),
        (flat[nemb:] / scale).reshape(sigma_imp_shape),
    )


def _initial_complex_bath_guess(
    target_delta: np.ndarray,
    omega: np.ndarray,
    mu: float,
    nbath: int,
    energy_window: float,
) -> tuple[np.ndarray, np.ndarray]:
    norb = int(target_delta.shape[-1])
    eps = float(mu) + np.linspace(-float(energy_window), float(energy_window), int(nbath))
    w = np.asarray(omega, dtype=float)
    pos = np.flatnonzero(w > 0.0)
    idx = int(pos[np.argmax(w[pos])]) if pos.size else len(w) - 1
    moment = (1j * float(w[idx])) * np.asarray(target_delta[idx], dtype=complex)
    moment = 0.5 * (moment + moment.conj().T)
    evals, evecs = np.linalg.eigh(moment)
    order = np.argsort(evals.real)[::-1]
    evals = np.maximum(np.asarray(evals[order].real), 1.0e-10)
    evecs = np.asarray(evecs[:, order], dtype=complex)
    hyb = np.zeros((norb, int(nbath)), dtype=complex)
    rank = min(norb, int(nbath))
    hyb[:, :rank] = evecs[:, :rank] * np.sqrt(evals[:rank])[None, :]
    if int(nbath) > rank:
        hyb[:, rank:] = 1.0e-4
    return eps, hyb


def fit_finite_bath_complex(
    target_delta: np.ndarray,
    omega: np.ndarray,
    mu: float,
    *,
    nbath: int = 6,
    nfit: int = 12,
    max_nfev: int = 500,
    energy_window: float = 4.0,
    coupling_bound: float = 4.0,
    xtol: float = 1.0e-10,
    initial: BathParameters | None = None,
) -> BathParameters:
    """Fit Delta(iw) with real bath energies and complex bath couplings."""
    delta = np.asarray(target_delta, dtype=complex)
    w = np.asarray(omega, dtype=float).reshape(-1)
    if delta.ndim != 3 or delta.shape[0] != len(w) or delta.shape[1] != delta.shape[2]:
        raise ValueError("target_delta must have shape (nf,norb,norb)")
    norb = int(delta.shape[-1])
    nbath = int(nbath)
    if nbath < 1:
        raise ValueError("nbath must be positive")

    pos = np.flatnonzero(w > 0.0)
    if pos.size == 0:
        raise ValueError("bath fit requires positive Matsubara frequencies")
    pos = pos[np.argsort(w[pos])[: min(int(nfit), len(pos))]]
    wfit = w[pos]
    dfit = delta[pos]
    w0 = max(float(wfit[0]), 1.0e-12)
    weights = 1.0 / np.sqrt(wfit * wfit + w0 * w0)
    weights /= float(np.max(weights))

    if initial is None:
        eps0, hyb0 = _initial_complex_bath_guess(delta, w, mu, nbath, energy_window)
    else:
        eps0 = np.asarray(initial.energies, dtype=float)
        hyb0 = np.asarray(initial.couplings, dtype=complex)
        if eps0.shape != (nbath,) or hyb0.shape != (norb, nbath):
            eps0, hyb0 = _initial_complex_bath_guess(delta, w, mu, nbath, energy_window)

    lo_e = float(mu) - float(energy_window)
    hi_e = float(mu) + float(energy_window)
    eps0 = np.clip(eps0, lo_e + 1.0e-8, hi_e - 1.0e-8)
    bound = float(coupling_bound)
    re0 = np.clip(hyb0.real, -bound + 1.0e-8, bound - 1.0e-8)
    im0 = np.clip(hyb0.imag, -bound + 1.0e-8, bound - 1.0e-8)
    x0 = np.concatenate([eps0, re0.ravel(), im0.ravel()])
    lower = np.concatenate([
        np.full(nbath, lo_e),
        np.full(2 * norb * nbath, -bound),
    ])
    upper = np.concatenate([
        np.full(nbath, hi_e),
        np.full(2 * norb * nbath, bound),
    ])

    nvc = norb * nbath

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eps = x[:nbath]
        re = x[nbath:nbath + nvc].reshape(norb, nbath)
        im = x[nbath + nvc:].reshape(norb, nbath)
        return eps, re + 1j * im

    def residual(x: np.ndarray) -> np.ndarray:
        eps, hyb = unpack(x)
        model = bath_hybridization(wfit, mu, eps, hyb)
        diff = (model - dfit) * weights[:, None, None]
        return np.concatenate([diff.real.ravel(), diff.imag.ravel()])

    opt = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        xtol=float(xtol),
        ftol=float(xtol),
        gtol=float(xtol),
    )
    eps, hyb = unpack(opt.x)
    order = np.argsort(eps)
    eps = np.asarray(eps[order], dtype=float)
    hyb = np.asarray(hyb[:, order], dtype=complex)
    fit = bath_hybridization(wfit, mu, eps, hyb)
    den = max(float(np.linalg.norm(dfit.ravel())), 1.0e-300)
    err = float(np.linalg.norm((fit - dfit).ravel()) / den)
    return BathParameters(eps, hyb, err, int(opt.nfev))


def onebody_expectation_from_lattice_G(
    G: np.ndarray,
    K: np.ndarray,
    h0: np.ndarray,
    mu: float,
    sigma_h: np.ndarray,
    grid: MatsubaraGrid,
) -> float:
    """Per-cell <K> with the same Matsubara-tail completion as the embedding."""
    rho_k = one_body_density_matrix_tail(G, grid, h0, float(mu), sigma_h)
    rho_c = np.mean(rho_k, axis=(0, 1))
    value = np.einsum("ab,ba->", np.asarray(K, dtype=complex), rho_c, optimize=True)
    return float(np.real_if_close(value).real)


def solve_cluster_source_warm(
    h0_source: np.ndarray,
    Vq: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    K: np.ndarray,
    target_filling: float,
    initial: ClusterCovariantState,
    opts: ClusterCovariantOptions = ClusterCovariantOptions(),
) -> ClusterCovariantSourceResult:
    """Relax the complete cluster-ED+GW map in one static source field."""
    h0 = np.asarray(h0_source, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    K = np.asarray(K, dtype=complex)
    if h0.shape != (grid.nk1, grid.nk2, NSUB, NSUB):
        raise ValueError("h0_source shape mismatch")
    if Vq.shape != h0.shape or K.shape != (NSUB, NSUB):
        raise ValueError("Vq/K shape mismatch")

    sigma_h = np.asarray(initial.Sigma_H, dtype=complex).copy()
    sigma_emb = np.asarray(initial.Sigma_emb, dtype=complex).copy()
    sigma_imp = np.asarray(initial.Sigma_imp, dtype=complex).copy()
    G = np.asarray(initial.G, dtype=complex).copy()
    mu = float(initial.mu)
    bath = BathParameters(
        np.asarray(initial.bath.energies, dtype=float).copy(),
        np.asarray(initial.bath.couplings, dtype=complex).copy(),
        float(initial.bath.fit_error),
        int(initial.bath.nfev),
    )

    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
    interactions = ruby_cluster_interactions(params)
    V_cluster = np.asarray(Vq[0, 0], dtype=complex)
    eye = np.eye(NSUB, dtype=complex)

    mix_opts = GWOptions(
        mixing=float(opts.mixing),
        mixing_method="pulay",
        pulay_history=int(opts.pulay_history),
        pulay_start=int(opts.pulay_start),
        pulay_regularization=float(opts.pulay_regularization),
    )
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    fallbacks = 0
    converged = False
    err = float("inf")
    mismatch = float("inf")
    bath_error = float(bath.fit_error)
    it = 0

    for it in range(1, int(opts.max_iter) + 1):
        rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        rho_c = np.mean(rho_k, axis=(0, 1))
        density = np.real(np.diag(rho_c))
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])

        P = compute_polarization_matrix(G, grid, backend="fft")
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_lattice = compute_sigma_gw_split_matrix(
            G, W, Vq, grid, h0, mu, sigma_h, backend="fft"
        )

        Gc = np.mean(G, axis=(1, 2))
        sigma_cgw, _, _ = cluster_gw_self_energy(Gc, rho_c, V_cluster, grid)
        g0_inv = np.linalg.inv(Gc) + sigma_imp
        delta_target = (
            (1j * grid.omega[:, None, None] + float(mu)) * eye[None, :, :]
            - h_cluster[None, :, :]
            - g0_inv
        )
        bath = fit_finite_bath_complex(
            delta_target,
            grid.omega,
            mu,
            nbath=int(opts.nbath),
            nfit=int(opts.bath_fit_nfreq),
            max_nfev=int(opts.bath_fit_max_nfev),
            energy_window=float(opts.bath_energy_window),
            coupling_bound=float(opts.bath_coupling_bound),
            xtol=float(opts.bath_fit_xtol),
            initial=bath,
        )
        bath_error = float(bath.fit_error)

        himp = build_impurity_one_body(h_cluster, bath)
        impurity = FiniteBathImpurityED(
            himp,
            interactions,
            correlated_orbitals=tuple(range(NSUB)),
        )
        impurity.diagonalize()
        Gimp, _ = impurity.green_iomega(
            1j * grid.omega,
            mu,
            grid.T,
            orbitals=tuple(range(NSUB)),
            discard_weight_tol=float(opts.discard_weight_tol),
        )
        delta_fit = bath_hybridization(
            grid.omega, mu, bath.energies, bath.couplings
        )
        g0_fit_inv = (
            (1j * grid.omega[:, None, None] + float(mu)) * eye[None, :, :]
            - h_cluster[None, :, :]
            - delta_fit
        )
        sigma_imp_out = g0_fit_inv - np.linalg.inv(Gimp)
        correction = sigma_imp_out - sigma_cgw
        sigma_emb_out = sigma_gw_lattice + correction[:, None, None, :, :]

        res_h = _maxabs(sigma_h_out - sigma_h)
        res_emb = _maxabs(sigma_emb_out - sigma_emb)
        res_imp = _maxabs(sigma_imp_out - sigma_imp)
        err = max(res_h, res_emb, res_imp)
        mismatch = _relative_error(Gimp, Gc)

        if opts.verbose:
            print(
                f"[cluster-cov] iter {it:02d}: residual={err:.3e} "
                f"(emb={res_emb:.3e}, imp={res_imp:.3e}), "
                f"Gimp/Gc={mismatch:.3e}, bath={bath_error:.3e}, mu={mu:+.9f}",
                flush=True,
            )

        if err < float(opts.tol):
            sigma_h = np.asarray(sigma_h_out)
            sigma_emb = np.asarray(sigma_emb_out)
            sigma_imp = np.asarray(sigma_imp_out)
            converged = True
        else:
            dyn = _pack_dynamic(sigma_emb, sigma_imp, grid.nk)
            dyn_out = _pack_dynamic(sigma_emb_out, sigma_imp_out, grid.nk)
            sigma_h_next, dyn_next = _mixed_self_energies(
                sigma_h,
                dyn,
                sigma_h_out,
                dyn_out,
                mix_opts,
                it,
                history,
            )
            raw_step = max(_maxabs(sigma_h_out - sigma_h), _maxabs(dyn_out - dyn))
            mixed_step = max(_maxabs(sigma_h_next - sigma_h), _maxabs(dyn_next - dyn))
            unsafe = (
                not np.all(np.isfinite(sigma_h_next))
                or not np.all(np.isfinite(dyn_next))
                or (raw_step > 1.0e-14 and mixed_step > float(opts.pulay_step_cap) * raw_step)
            )
            if unsafe:
                fallbacks += 1
                history.clear()
                a = float(opts.mixing)
                sigma_h_next = sigma_h + a * (sigma_h_out - sigma_h)
                dyn_next = dyn + a * (dyn_out - dyn)
            sigma_h = np.asarray(sigma_h_next)
            sigma_emb, sigma_imp = _unpack_dynamic(
                dyn_next, sigma_emb.shape, sigma_imp.shape, grid.nk
            )

        mu, G, _, _ = _solve_mu_matrix_fast(
            h0,
            sigma_h,
            sigma_emb,
            grid,
            float(target_filling),
            float(mu),
            float(opts.mu_tol),
            int(opts.mu_max_iter),
        )
        if converged:
            break

    expectation = onebody_expectation_from_lattice_G(
        G, K, h0, mu, sigma_h, grid
    )
    state = ClusterCovariantState(
        Sigma_H=np.asarray(sigma_h),
        Sigma_emb=np.asarray(sigma_emb),
        Sigma_imp=np.asarray(sigma_imp),
        G=np.asarray(G),
        mu=float(mu),
        bath=bath,
    )
    return ClusterCovariantSourceResult(
        state=state,
        expectation=float(expectation),
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
        impurity_mismatch=float(mismatch),
        bath_fit_error=float(bath_error),
        pulay_fallbacks=int(fallbacks),
    )


__all__ = [
    "ClusterCovariantState",
    "ClusterCovariantOptions",
    "ClusterCovariantSourceResult",
    "fit_finite_bath_complex",
    "onebody_expectation_from_lattice_G",
    "solve_cluster_source_warm",
]
