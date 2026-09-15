"""C3-constrained normal-state cluster-ED+GW solver.

This module is intentionally separate from the production unconstrained solver.
It is used to construct a symmetry-restored *background* for susceptibility
analysis when the oriented six-site impurity correction drifts into a uniform
E-sector density-polarized branch.

The important point is that the Ruby primitive cell is not mapped to itself by
C3 through a simple on-site permutation: the B triangle is shifted by one
primitive cell.  We therefore constrain the physical lattice self-energy with
the full momentum-dependent C3 projector from :mod:`rubycgw.c3_constraint`.
The internal six-site impurity is left unrestricted; only the lattice field fed
back into Dyson is C3 projected.  This avoids imposing an incorrect local-only
C3 transformation on the cluster Green function.

The resulting state is a constrained normal-state reference.  The usual JF
scanner should be run *without* applying this projector to the response, so E
and current fluctuations remain available and can be compared around the
symmetric reference.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from .c3_constraint import (
    c3_lattice_residual,
    density_c3_spread,
    project_density_c3,
    project_lattice_c3,
)
from .cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    build_impurity_one_body,
    build_intracell_h0,
    cluster_gw_self_energy,
    fit_finite_bath,
    ruby_cluster_interactions,
)
from .cluster_ed_gw_fast import (
    ClusterEDGWFastOptions,
    ClusterEDGWFastResult,
    _maxabs,
    _pack_dynamic,
    _relative_error,
    _unpack_dynamic,
)
from .cluster_restart import ClusterEDGWRestartState
from .cluster_restart_solver import _continuation_background
from .grids import MatsubaraGrid
from .gw import GWOptions, _check_backend, _check_mixing_method, _mixed_self_energies
from .impurity_ed import FiniteBathImpurityED
from .model import NSUB, RubyParameters
from .supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
)
from .supercell_gw_fast import _solve_mu_matrix_fast, solve_matrix_gw_fast
from .supercell_gw_split import compute_sigma_gw_split_matrix, one_body_density_matrix_tail


def _constrain_hartree(sigma_h: np.ndarray) -> np.ndarray:
    """Project the diagonal Hartree field onto equal A/B triangle densities."""
    s = np.asarray(sigma_h, dtype=complex)
    out = np.zeros_like(s)
    d = np.real(np.diag(s))
    d[:3] = np.mean(d[:3])
    d[3:] = np.mean(d[3:])
    out[np.diag_indices(NSUB)] = d
    return out


def _refresh_green(
    h0,
    sigma_h,
    sigma_emb,
    grid,
    gw_opts,
    mu0: float,
):
    if gw_opts.target_filling is None:
        mu = float(mu0)
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_emb)
    else:
        mu, G, _, _ = _solve_mu_matrix_fast(
            h0,
            sigma_h,
            sigma_emb,
            grid,
            float(gw_opts.target_filling),
            float(mu0),
            float(gw_opts.mu_tol),
            int(gw_opts.mu_max_iter),
        )
    # In exact arithmetic this projection is redundant once h0/Sigma are C3
    # covariant.  Keeping it removes accumulated floating-point/bath-fit noise.
    return float(mu), project_lattice_c3(np.asarray(G, dtype=complex))


def solve_cluster_ed_gw_fast_c3_constrained(
    h0: np.ndarray,
    Vq: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    *,
    gw_opts: GWOptions = GWOptions(),
    embed_opts: ClusterEDGWFastOptions = ClusterEDGWFastOptions(),
    restart: ClusterEDGWRestartState | None = None,
) -> ClusterEDGWFastResult:
    """Solve a C3-constrained normal-state embedding fixed point.

    If ``restart`` is supplied, the saved embedded solution is used only as an
    initial condition and no standalone SC-GW solve is performed.  Its lattice
    self-energy is immediately projected onto the exact physical C3 subspace
    before the first Dyson solve.
    """
    backend = _check_backend(gw_opts.momentum_backend)
    method = _check_mixing_method(embed_opts.mixing_method)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    if h0.shape != (grid.nk1, grid.nk2, NSUB, NSUB):
        raise ValueError("cluster ED+GW expects the primitive 6-site lattice basis")
    if Vq.shape != h0.shape:
        raise ValueError("Vq/h0 shape mismatch")
    if int(grid.nk1) != int(grid.nk2):
        raise ValueError(
            "the exact Ruby C3 constraint requires Lx=Ly; "
            f"got {grid.nk1}x{grid.nk2}"
        )
    if not (0.0 < float(embed_opts.mixing) <= 1.0):
        raise ValueError("embedding mixing must lie in (0,1]")
    if not (0.0 < float(embed_opts.impurity_mixing) <= 1.0):
        raise ValueError("impurity mixing must lie in (0,1]")
    if int(embed_opts.pulay_history) < 2:
        raise ValueError("pulay_history must be at least 2")

    # Validate the C3 representation against the actual one-body Hamiltonian.
    h0_res = c3_lattice_residual(h0)
    if h0_res > 1.0e-10:
        raise ValueError(
            "requested C3 constraint is inconsistent with h0: "
            f"max residual={h0_res:.3e}.  Check t1=t2 and lattice geometry."
        )

    if restart is None:
        if embed_opts.verbose:
            print("[cluster-ED+GW:C3] solving initial lattice GW background ...", flush=True)
        background = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
        if not background.converged:
            raise RuntimeError(
                "initial lattice GW background is not converged: "
                f"{background.final_error:.3e}"
            )
        sigma_h = _constrain_hartree(background.Sigma_H)
        sigma_emb = project_lattice_c3(np.asarray(background.Sigma_GW, dtype=complex))
        mu = float(background.mu)
        bath: BathParameters | None = None
        sigma_imp = None
    else:
        background = _continuation_background(restart, grid)
        sigma_h = _constrain_hartree(restart.Sigma_H)
        sigma_emb = project_lattice_c3(np.asarray(restart.Sigma_emb, dtype=complex))
        mu = float(restart.mu)
        bath = restart.bath
        sigma_imp = np.asarray(restart.Sigma_imp, dtype=complex).copy()
        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW:C3] constrained continuation from "
                f"{restart.source_path or '<memory>'}; projecting lattice state "
                "and rebuilding Pulay history",
                flush=True,
            )

    mu, G = _refresh_green(h0, sigma_h, sigma_emb, grid, gw_opts, mu)

    h_cluster_strict = build_intracell_h0(params)
    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
    alias_norm = _maxabs(h_cluster - h_cluster_strict)
    if embed_opts.verbose and alias_norm > 1.0e-12:
        print(
            f"[cluster-ED+GW:C3] finite-torus local-block correction: "
            f"max|h_local-h_R0|={alias_norm:.3e}",
            flush=True,
        )

    interactions = ruby_cluster_interactions(params)
    V_cluster = np.asarray(Vq[0, 0], dtype=complex)

    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    Gc = np.mean(G, axis=(1, 2))
    rho_c = np.mean(rho_k, axis=(0, 1))
    density = project_density_c3(np.real(np.diag(rho_c)))
    rho_c = np.array(rho_c, copy=True)
    rho_c[np.diag_indices(NSUB)] = density
    sigma_cgw, _, _ = cluster_gw_self_energy(Gc, rho_c, V_cluster, grid)
    if sigma_imp is None:
        sigma_imp = np.array(sigma_cgw, copy=True)

    mix_opts = GWOptions(
        mixing=float(embed_opts.mixing),
        mixing_method=method,
        pulay_history=int(embed_opts.pulay_history),
        pulay_start=int(embed_opts.pulay_start),
        pulay_regularization=float(embed_opts.pulay_regularization),
    )
    mix_history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    fallbacks = 0

    residual_hist: list[float] = []
    imp_residual_hist: list[float] = []
    mismatch_hist: list[float] = []
    bath_hist: list[float] = []
    mu_hist: list[float] = []
    nfev_hist: list[int] = []
    elapsed_hist: list[float] = []
    c3_raw_hist: list[float] = []
    c3_removed_hist: list[float] = []
    density_raw_spread_hist: list[float] = []

    W = np.asarray(background.W)
    P = np.asarray(background.P)
    sigma_gw_lattice = project_lattice_c3(np.asarray(background.Sigma_GW, dtype=complex))
    Gimp = np.asarray(Gc)
    converged = False
    err = float("inf")
    mismatch = float("inf")
    it = 0

    for it in range(1, int(embed_opts.max_iter) + 1):
        t0 = perf_counter()
        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW:C3] outer {it:02d}: build constrained lattice map "
                f"(mix={method})",
                flush=True,
            )

        rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        rho_c = np.mean(rho_k, axis=(0, 1))
        density_raw = np.real(np.diag(rho_c))
        density_raw_spread_hist.append(density_c3_spread(density_raw))
        density = project_density_c3(density_raw)
        rho_c = np.array(rho_c, copy=True)
        rho_c[np.diag_indices(NSUB)] = density
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
        sigma_h_out = _constrain_hartree(sigma_h_out)

        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_lattice_raw = compute_sigma_gw_split_matrix(
            G, W, Vq, grid, h0, mu, sigma_h, backend=backend
        )
        sigma_gw_lattice = project_lattice_c3(sigma_gw_lattice_raw)

        Gc = np.mean(G, axis=(1, 2))
        sigma_cgw, _, _ = cluster_gw_self_energy(Gc, rho_c, V_cluster, grid)

        g0_inv = np.linalg.inv(Gc) + sigma_imp
        eye = np.eye(NSUB, dtype=complex)
        delta_target = (
            (1j * grid.omega[:, None, None] + float(mu)) * eye[None, :, :]
            - h_cluster[None, :, :]
            - g0_inv
        )

        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW:C3] outer {it:02d}: fit "
                f"{embed_opts.nbath} bath orbitals ...",
                flush=True,
            )
        bath = fit_finite_bath(
            delta_target,
            grid.omega,
            mu,
            nbath=int(embed_opts.nbath),
            nfit=int(embed_opts.bath_fit_nfreq),
            max_nfev=int(embed_opts.bath_fit_max_nfev),
            energy_window=float(embed_opts.bath_energy_window),
            coupling_bound=float(embed_opts.bath_coupling_bound),
            xtol=float(embed_opts.bath_fit_xtol),
            initial=bath,
        )
        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW:C3] outer {it:02d}: bath relerr={bath.fit_error:.3e}, "
                f"nfev={bath.nfev}; diagonalize impurity ...",
                flush=True,
            )

        himp = build_impurity_one_body(h_cluster, bath)
        impurity = FiniteBathImpurityED(
            himp,
            interactions,
            correlated_orbitals=tuple(range(NSUB)),
        )
        impurity.diagonalize()
        Gimp, selection = impurity.green_iomega(
            1j * grid.omega,
            mu,
            grid.T,
            orbitals=tuple(range(NSUB)),
            discard_weight_tol=float(embed_opts.discard_weight_tol),
        )
        delta_fit = bath_hybridization(
            grid.omega, mu, bath.energies, bath.couplings
        )
        g0_fit_inv = (
            (1j * grid.omega[:, None, None] + float(mu)) * eye[None, :, :]
            - h_cluster[None, :, :]
            - delta_fit
        )
        sigma_ed_raw = g0_fit_inv - np.linalg.inv(Gimp)

        beta_imp = float(embed_opts.impurity_mixing)
        sigma_imp_out = sigma_imp + beta_imp * (sigma_ed_raw - sigma_imp)
        correction = sigma_imp_out - sigma_cgw
        sigma_emb_raw = sigma_gw_lattice + correction[:, None, None, :, :]
        raw_c3 = c3_lattice_residual(sigma_emb_raw)
        sigma_emb_out = project_lattice_c3(sigma_emb_raw)
        c3_raw_hist.append(raw_c3)
        c3_removed_hist.append(_maxabs(sigma_emb_raw - sigma_emb_out))

        res_h = _maxabs(sigma_h_out - sigma_h)
        res_emb = _maxabs(sigma_emb_out - sigma_emb)
        res_imp = _maxabs(sigma_imp_out - sigma_imp)
        err = max(res_h, res_emb, res_imp)
        mismatch = _relative_error(Gimp, Gc)
        elapsed = perf_counter() - t0

        residual_hist.append(float(err))
        imp_residual_hist.append(float(res_imp))
        mismatch_hist.append(float(mismatch))
        bath_hist.append(float(bath.fit_error))
        mu_hist.append(float(mu))
        nfev_hist.append(int(bath.nfev))
        elapsed_hist.append(float(elapsed))

        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW:C3] outer {it:02d}: residual={err:.3e} "
                f"(emb={res_emb:.3e}, imp={res_imp:.3e}), "
                f"Gimp/Gc={mismatch:.3e}, bath={bath.fit_error:.3e}, "
                f"Ntot_imp={selection.average_particles:.6f}, mu={mu:+.9f}, "
                f"raw-C3={raw_c3:.3e}, dt={elapsed:.1f}s",
                flush=True,
            )

        if err < float(embed_opts.tol):
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
                mix_history,
            )

            raw_step = max(_maxabs(sigma_h_out - sigma_h), _maxabs(dyn_out - dyn))
            mixed_step = max(
                _maxabs(sigma_h_next - sigma_h),
                _maxabs(dyn_next - dyn),
            )
            cap = float(embed_opts.pulay_step_cap)
            unsafe = (
                not np.all(np.isfinite(sigma_h_next))
                or not np.all(np.isfinite(dyn_next))
                or (raw_step > 1e-14 and mixed_step > cap * raw_step)
            )
            if unsafe:
                fallbacks += 1
                mix_history.clear()
                a = float(embed_opts.mixing)
                sigma_h_next = sigma_h + a * (sigma_h_out - sigma_h)
                dyn_next = dyn + a * (dyn_out - dyn)
                if embed_opts.verbose:
                    print(
                        f"[cluster-ED+GW:C3] outer {it:02d}: Pulay safeguard -> "
                        f"linear fallback (mixed/raw={mixed_step/max(raw_step,1e-300):.2f})",
                        flush=True,
                    )

            sigma_h = _constrain_hartree(np.asarray(sigma_h_next))
            sigma_emb, sigma_imp = _unpack_dynamic(
                dyn_next,
                sigma_emb.shape,
                sigma_imp.shape,
                grid.nk,
            )
            sigma_emb = project_lattice_c3(sigma_emb)

        mu, G = _refresh_green(h0, sigma_h, sigma_emb, grid, gw_opts, mu)
        if converged:
            break

    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density_raw = np.real(np.diag(np.mean(rho_k, axis=(0, 1))))
    density = project_density_c3(density_raw)
    Gc = np.mean(G, axis=(1, 2))

    if bath is None:
        raise RuntimeError("embedding loop did not execute")

    result = ClusterEDGWFastResult(
        G=np.asarray(G),
        W=np.asarray(W),
        P=np.asarray(P),
        Sigma_H=np.asarray(sigma_h),
        Sigma_emb=np.asarray(sigma_emb),
        Sigma_GW_lattice=np.asarray(sigma_gw_lattice),
        Sigma_GW_cluster=np.asarray(sigma_cgw),
        Sigma_ED_cluster=np.asarray(sigma_imp),
        G_cluster=np.asarray(Gc),
        G_impurity=np.asarray(Gimp),
        mu=float(mu),
        density=np.asarray(density),
        bath=bath,
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
        impurity_mismatch=float(mismatch),
        bath_fit_error=float(bath.fit_error),
        background=background,
        mixing_method=method,
        residual_history=np.asarray(residual_hist, dtype=float),
        impurity_residual_history=np.asarray(imp_residual_hist, dtype=float),
        impurity_mismatch_history=np.asarray(mismatch_hist, dtype=float),
        bath_fit_history=np.asarray(bath_hist, dtype=float),
        mu_history=np.asarray(mu_hist, dtype=float),
        bath_nfev_history=np.asarray(nfev_hist, dtype=int),
        elapsed_history=np.asarray(elapsed_hist, dtype=float),
        pulay_fallbacks=int(fallbacks),
    )
    # Extra diagnostics are intentionally attached dynamically so the existing
    # result/checkpoint API remains compatible with the ordinary solver.
    result.c3_raw_residual_history = np.asarray(c3_raw_hist, dtype=float)
    result.c3_removed_norm_history = np.asarray(c3_removed_hist, dtype=float)
    result.c3_density_raw_spread_history = np.asarray(
        density_raw_spread_hist, dtype=float
    )
    result.c3_final_residual = float(c3_lattice_residual(result.Sigma_emb))
    result.c3_final_density_raw_spread = float(density_c3_spread(density_raw))
    result.c3_h0_residual = float(h0_res)
    return result


__all__ = ["solve_cluster_ed_gw_fast_c3_constrained"]
