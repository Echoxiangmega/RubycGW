"""Multi-impurity finite-torus cluster ED+GW for commensurate finite-q order.

An Lx x Ly primitive-cell torus is folded to one 6*Lx*Ly orbital supercell.
Every primitive cell carries its own six-site finite-bath ED impurity, allowing
translation-breaking self-energies while retaining the same physical-pair
cluster interaction partition as the primitive orientation solver.

The embedding is

    Sigma_emb^SC = Sigma_GW^SC
                 + blockdiag_R(Sigma_ED^R - Sigma_GW^{C,R}).

This is the natural finite-q extension of the production physical-pair
cluster-ED+GW approximation on the same discrete torus used by the JF scan.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    build_impurity_one_body,
    cluster_gw_self_energy,
    fit_finite_bath,
    split_static_hybridization,
)
from .cluster_ed_gw_covariant import fit_finite_bath_complex
from .grids import MatsubaraGrid
from .impurity_ed import FiniteBathImpurityED
from .supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    hartree_self_energy_matrix,
)
from .supercell_gw_fast import _solve_mu_matrix_fast, solve_matrix_gw_fast
from .supercell_gw_split import (
    compute_sigma_gw_split_matrix,
    one_body_density_matrix_tail,
)


NSUB = 6


@dataclass(frozen=True)
class MultiCellEDGWOptions:
    max_iter: int = 120
    tol: float = 2e-6
    mixing: float = 0.45
    impurity_mixing: float = 1.0
    nbath: int = 6
    bath_fit_nfreq: int = 12
    bath_fit_max_nfev: int = 400
    bath_energy_window: float = 4.0
    bath_coupling_bound: float = 4.0
    bath_fit_xtol: float = 1e-9
    discard_weight_tol: float = 1e-11
    complex_bath: bool = True
    mu_tol: float = 1e-11
    mu_max_iter: int = 80
    verbose: bool = True


@dataclass
class MultiCellRestart:
    Sigma_H: np.ndarray
    Sigma_emb: np.ndarray
    Sigma_imp: np.ndarray
    G: np.ndarray
    mu: float
    bath_energies: np.ndarray
    bath_couplings: np.ndarray


@dataclass
class MultiCellEDGWResult:
    G: np.ndarray
    W: np.ndarray
    P: np.ndarray
    Sigma_H: np.ndarray
    Sigma_emb: np.ndarray
    Sigma_GW_lattice: np.ndarray
    Sigma_GW_cluster: np.ndarray
    Sigma_ED_cluster: np.ndarray
    G_cluster: np.ndarray
    G_impurity: np.ndarray
    mu: float
    density: np.ndarray
    bath_energies: np.ndarray
    bath_couplings: np.ndarray
    bath_fit_error: np.ndarray
    impurity_static_shift: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    impurity_mismatch: np.ndarray
    residual_history: np.ndarray
    elapsed_history: np.ndarray


def _maxabs(a):
    return float(np.max(np.abs(np.asarray(a)), initial=0.0))


def _relative_error(a, b):
    den = max(float(np.linalg.norm(np.asarray(b).ravel())), 1e-300)
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel()) / den)


def _block_slice(cell: int) -> slice:
    i0 = int(cell) * NSUB
    return slice(i0, i0 + NSUB)


def _bath_from_arrays(energies, couplings, cell, fit_error=np.nan):
    return BathParameters(
        np.asarray(energies[cell], dtype=float),
        np.asarray(couplings[cell], dtype=complex),
        float(np.asarray(fit_error).reshape(-1)[cell]) if np.size(fit_error) > cell else np.nan,
        0,
    )


def solve_multicell_cluster_ed_gw(
    h0: np.ndarray,
    Vq: np.ndarray,
    cluster_interactions,
    grid: MatsubaraGrid,
    *,
    target_particles: float,
    opts: MultiCellEDGWOptions = MultiCellEDGWOptions(),
    restart: MultiCellRestart | None = None,
):
    """Solve the finite-torus multi-impurity ED+GW fixed point."""
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    if grid.nk1 != 1 or grid.nk2 != 1:
        raise ValueError("multi-cell folded solver expects a one-point reduced BZ")
    norb = int(h0.shape[-1])
    if h0.shape != (1, 1, norb, norb) or Vq.shape != h0.shape:
        raise ValueError("h0/Vq must have shape (1,1,N,N)")
    if norb % NSUB != 0:
        raise ValueError("folded orbital dimension must be a multiple of six")
    ncell = norb // NSUB
    if not (0.0 < float(opts.mixing) <= 1.0):
        raise ValueError("mixing must lie in (0,1]")
    if not (0.0 < float(opts.impurity_mixing) <= 1.0):
        raise ValueError("impurity_mixing must lie in (0,1]")

    if restart is None:
        from .gw import GWOptions
        bg = solve_matrix_gw_fast(
            h0,
            Vq,
            grid,
            opts=GWOptions(
                target_filling=float(target_particles),
                max_iter=max(200, int(opts.max_iter)),
                tol=min(float(opts.tol), 1e-8),
                mixing=0.25,
                mixing_method="pulay",
                verbose=bool(opts.verbose),
                momentum_backend="fft",
                mu_tol=float(opts.mu_tol),
                mu_max_iter=int(opts.mu_max_iter),
            ),
        )
        if not bg.converged:
            raise RuntimeError(
                f"initial folded GW background did not converge: {bg.final_error:.3e}"
            )
        sigma_h = np.asarray(bg.Sigma_H, dtype=complex).copy()
        sigma_emb = np.asarray(bg.Sigma_GW, dtype=complex).copy()
        G = np.asarray(bg.G, dtype=complex).copy()
        mu = float(bg.mu)
        rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        sigma_imp = np.zeros((ncell, grid.nf, NSUB, NSUB), dtype=complex)
        baths: list[BathParameters | None] = [None] * ncell
        for c in range(ncell):
            sl = _block_slice(c)
            Gc = G[:, 0, 0, sl, sl]
            rhoc = rho[0, 0, sl, sl]
            sigma_imp[c], _, _ = cluster_gw_self_energy(
                Gc,
                rhoc,
                _cluster_matrix(cluster_interactions),
                grid,
            )
    else:
        sigma_h = np.asarray(restart.Sigma_H, dtype=complex).copy()
        sigma_emb = np.asarray(restart.Sigma_emb, dtype=complex).copy()
        sigma_imp = np.asarray(restart.Sigma_imp, dtype=complex).copy()
        G = np.asarray(restart.G, dtype=complex).copy()
        mu = float(restart.mu)
        if sigma_h.shape != (norb, norb):
            raise ValueError("restart Sigma_H shape mismatch")
        if sigma_emb.shape != (grid.nf, 1, 1, norb, norb):
            raise ValueError("restart Sigma_emb shape mismatch")
        if sigma_imp.shape != (ncell, grid.nf, NSUB, NSUB):
            raise ValueError("restart Sigma_imp shape mismatch")
        baths = [
            _bath_from_arrays(
                restart.bath_energies,
                restart.bath_couplings,
                c,
            )
            for c in range(ncell)
        ]

    Vc = _cluster_matrix(cluster_interactions)
    fitter = fit_finite_bath_complex if bool(opts.complex_bath) else fit_finite_bath
    eye6 = np.eye(NSUB, dtype=complex)

    converged = False
    err = np.inf
    residual_history = []
    elapsed_history = []
    P = np.zeros((grid.nb, 1, 1, norb, norb), dtype=complex)
    W = np.zeros_like(P)
    sigma_gw_lattice = np.zeros_like(G)
    sigma_cgw_all = np.zeros_like(sigma_imp)
    sigma_ed_all = np.zeros_like(sigma_imp)
    Gc_all = np.zeros((ncell, grid.nf, NSUB, NSUB), dtype=complex)
    Gimp_all = np.zeros_like(Gc_all)
    static_shift_all = np.zeros((ncell, NSUB, NSUB), dtype=complex)
    mismatch_all = np.full(ncell, np.inf)
    bath_error_all = np.full(ncell, np.inf)

    for it in range(1, int(opts.max_iter) + 1):
        t0 = perf_counter()
        rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        density = np.diagonal(rho[0, 0], axis1=-2, axis2=-1).real
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
        P = compute_polarization_matrix(G, grid, backend="fft")
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_lattice = compute_sigma_gw_split_matrix(
            G,
            W,
            Vq,
            grid,
            h0,
            mu,
            sigma_h,
            backend="fft",
        )
        sigma_emb_out = np.asarray(sigma_gw_lattice, dtype=complex).copy()

        for c in range(ncell):
            sl = _block_slice(c)
            Gc = np.asarray(G[:, 0, 0, sl, sl], dtype=complex)
            rhoc = np.asarray(rho[0, 0, sl, sl], dtype=complex)
            Gc_all[c] = Gc
            sigma_cgw, _, _ = cluster_gw_self_energy(Gc, rhoc, Vc, grid)
            sigma_cgw_all[c] = sigma_cgw

            g0_inv = np.linalg.inv(Gc) + sigma_imp[c]
            hloc = np.asarray(h0[0, 0, sl, sl], dtype=complex)
            raw_delta = (
                (1j * grid.omega[:, None, None] + float(mu))
                * eye6[None, :, :]
                - hloc[None, :, :]
                - g0_inv
            )
            static_shift, delta = split_static_hybridization(raw_delta, grid.omega)
            static_shift_all[c] = static_shift
            himp_corr = hloc + static_shift
            bath = fitter(
                delta,
                grid.omega,
                mu,
                nbath=int(opts.nbath),
                nfit=int(opts.bath_fit_nfreq),
                max_nfev=int(opts.bath_fit_max_nfev),
                energy_window=float(opts.bath_energy_window),
                coupling_bound=float(opts.bath_coupling_bound),
                xtol=float(opts.bath_fit_xtol),
                initial=baths[c],
                metric="delta",
                one_body=himp_corr,
            )
            baths[c] = bath
            bath_error_all[c] = float(bath.fit_error)

            himp = build_impurity_one_body(himp_corr, bath)
            impurity = FiniteBathImpurityED(
                himp,
                cluster_interactions,
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
            Gimp_all[c] = Gimp
            delta_fit = bath_hybridization(
                grid.omega, mu, bath.energies, bath.couplings
            )
            g0_fit_inv = (
                (1j * grid.omega[:, None, None] + float(mu))
                * eye6[None, :, :]
                - himp_corr[None, :, :]
                - delta_fit
            )
            sigma_raw = g0_fit_inv - np.linalg.inv(Gimp)
            bimp = float(opts.impurity_mixing)
            sigma_ed = (1.0 - bimp) * sigma_imp[c] + bimp * sigma_raw
            sigma_ed_all[c] = sigma_ed
            mismatch_all[c] = _relative_error(Gimp, Gc)
            sigma_emb_out[:, 0, 0, sl, sl] += sigma_ed - sigma_cgw

        err = max(
            _maxabs(sigma_h_out - sigma_h),
            _maxabs(sigma_emb_out - sigma_emb),
            _maxabs(sigma_ed_all - sigma_imp),
        )
        residual_history.append(float(err))
        elapsed_history.append(float(perf_counter() - t0))
        if opts.verbose:
            print(
                f"[multi-ED+GW] it={it:03d}, residual={err:.3e}, "
                f"max Gimp/Gc={np.max(mismatch_all):.3e}, "
                f"max bath={np.max(bath_error_all):.3e}, mu={mu:+.9f}, "
                f"dt={elapsed_history[-1]:.1f}s",
                flush=True,
            )
        if err < float(opts.tol):
            converged = True
            sigma_h = sigma_h_out
            sigma_emb = sigma_emb_out
            sigma_imp = sigma_ed_all.copy()
            mu, G, _, _ = _solve_mu_matrix_fast(
                h0,
                sigma_h,
                sigma_emb,
                grid,
                float(target_particles),
                mu,
                float(opts.mu_tol),
                int(opts.mu_max_iter),
            )
            break

        a = float(opts.mixing)
        sigma_h = sigma_h + a * (sigma_h_out - sigma_h)
        sigma_emb = sigma_emb + a * (sigma_emb_out - sigma_emb)
        sigma_imp = sigma_imp + a * (sigma_ed_all - sigma_imp)
        mu, G, _, _ = _solve_mu_matrix_fast(
            h0,
            sigma_h,
            sigma_emb,
            grid,
            float(target_particles),
            mu,
            float(opts.mu_tol),
            int(opts.mu_max_iter),
        )

    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density = np.diagonal(rho[0, 0], axis1=-2, axis2=-1).real

    bath_energies = np.stack([np.asarray(b.energies, dtype=float) for b in baths])
    bath_couplings = np.stack([np.asarray(b.couplings, dtype=complex) for b in baths])

    return MultiCellEDGWResult(
        G=np.asarray(G),
        W=np.asarray(W),
        P=np.asarray(P),
        Sigma_H=np.asarray(sigma_h),
        Sigma_emb=np.asarray(sigma_emb),
        Sigma_GW_lattice=np.asarray(sigma_gw_lattice),
        Sigma_GW_cluster=np.asarray(sigma_cgw_all),
        Sigma_ED_cluster=np.asarray(sigma_imp),
        G_cluster=np.asarray(Gc_all),
        G_impurity=np.asarray(Gimp_all),
        mu=float(mu),
        density=np.asarray(density),
        bath_energies=bath_energies,
        bath_couplings=bath_couplings,
        bath_fit_error=np.asarray(bath_error_all),
        impurity_static_shift=np.asarray(static_shift_all),
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
        impurity_mismatch=np.asarray(mismatch_all),
        residual_history=np.asarray(residual_history),
        elapsed_history=np.asarray(elapsed_history),
    )


def _cluster_matrix(interactions):
    from .cluster_ed_gw import cluster_interaction_matrix
    return cluster_interaction_matrix(interactions, NSUB)


__all__ = [
    "MultiCellEDGWOptions",
    "MultiCellRestart",
    "MultiCellEDGWResult",
    "solve_multicell_cluster_ed_gw",
]
