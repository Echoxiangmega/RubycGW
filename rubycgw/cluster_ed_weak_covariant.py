"""Finite-source covariant response for selectable weak+ED embedding."""
from __future__ import annotations

import numpy as np

from .cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    build_impurity_one_body,
    ruby_cluster_interactions,
)
from .cluster_ed_gw_covariant import (
    ClusterCovariantOptions,
    ClusterCovariantSourceResult,
    ClusterCovariantState,
    _maxabs,
    _pack_dynamic,
    _relative_error,
    _unpack_dynamic,
    fit_finite_bath_complex,
    onebody_expectation_from_lattice_G,
    solve_cluster_source_warm,
)
from .cluster_ed_weak_fast import canonical_weak_solver
from .gf2 import cluster_gf2_self_energy, compute_gf2_nonhartree
from .grids import MatsubaraGrid
from .gw import GWOptions, _mixed_self_energies
from .impurity_ed import FiniteBathImpurityED
from .model import NSUB, RubyParameters
from .sox_covariant import SOXOptions
from .supercell_gw import hartree_self_energy_matrix
from .supercell_gw_fast import _solve_mu_matrix_fast
from .supercell_gw_split import one_body_density_matrix_tail


def solve_cluster_source_warm_weak(
    h0_source: np.ndarray,
    Vq: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    K: np.ndarray,
    target_filling: float,
    initial: ClusterCovariantState,
    opts: ClusterCovariantOptions = ClusterCovariantOptions(),
    *,
    weak_solver: str = "gw",
    gf2_sox_opts: SOXOptions = SOXOptions(n_quad=64),
) -> ClusterCovariantSourceResult:
    """Relax one source-shifted embedding fixed point with GW or GF2 outside."""
    weak = canonical_weak_solver(weak_solver)
    if weak == "gw":
        return solve_cluster_source_warm(
            h0_source, Vq, params, grid, K, target_filling, initial, opts
        )

    gf2_sox_opts.validate()
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
        sigma_weak, _, _, _, _ = compute_gf2_nonhartree(
            G,
            Vq,
            h0,
            mu,
            sigma_h,
            grid,
            backend="fft",
            sox_opts=gf2_sox_opts,
        )

        Gc = np.mean(G, axis=(1, 2))
        sigma_cweak, _, _ = cluster_gf2_self_energy(
            Gc,
            rho_c,
            h_cluster,
            V_cluster,
            mu,
            grid,
            sox_opts=gf2_sox_opts,
        )
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
        delta_fit = bath_hybridization(grid.omega, mu, bath.energies, bath.couplings)
        g0_fit_inv = (
            (1j * grid.omega[:, None, None] + float(mu)) * eye[None, :, :]
            - h_cluster[None, :, :]
            - delta_fit
        )
        sigma_imp_out = g0_fit_inv - np.linalg.inv(Gimp)
        correction = sigma_imp_out - sigma_cweak
        sigma_emb_out = sigma_weak + correction[:, None, None, :, :]

        res_h = _maxabs(sigma_h_out - sigma_h)
        res_emb = _maxabs(sigma_emb_out - sigma_emb)
        res_imp = _maxabs(sigma_imp_out - sigma_imp)
        err = max(res_h, res_emb, res_imp)
        mismatch = _relative_error(Gimp, Gc)
        if opts.verbose:
            print(
                f"[cluster-cov-GF2] iter {it:02d}: residual={err:.3e} "
                f"(emb={res_emb:.3e}, imp={res_imp:.3e}), Gimp/Gc={mismatch:.3e}, "
                f"bath={bath_error:.3e}, mu={mu:+.9f}",
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
                sigma_h, dyn, sigma_h_out, dyn_out, mix_opts, it, history
            )
            raw_step = max(_maxabs(sigma_h_out - sigma_h), _maxabs(dyn_out - dyn))
            mixed_step = max(_maxabs(sigma_h_next - sigma_h), _maxabs(dyn_next - dyn))
            unsafe = (
                not np.all(np.isfinite(sigma_h_next))
                or not np.all(np.isfinite(dyn_next))
                or (
                    raw_step > 1.0e-14
                    and mixed_step > float(opts.pulay_step_cap) * raw_step
                )
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

    expectation = onebody_expectation_from_lattice_G(G, K, h0, mu, sigma_h, grid)
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


__all__ = ["solve_cluster_source_warm_weak"]
