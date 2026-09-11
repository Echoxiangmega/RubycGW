"""Pulay-accelerated 6-site cluster-ED + lattice-GW self-energy embedding.

This module keeps the physics of :mod:`rubycgw.cluster_ed_gw` but accelerates
its coupled fixed point.  The lattice embedded self-energy and the impurity
self-energy are mixed together, rather than damping the impurity map first and
then damping the lattice map a second time.

The dynamic state is

    X_dyn = (Sigma_emb(k,iw), sqrt(Nk) Sigma_imp(iw)),

where the sqrt(Nk) factor gives the local impurity block comparable weight in
Pulay's residual metric when the lattice contains Nk cluster momenta.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    build_impurity_one_body,
    build_intracell_h0,
    cluster_gw_self_energy,
    fit_finite_bath,
    ruby_cluster_interactions,
)
from .grids import MatsubaraGrid
from .gw import (
    GWOptions,
    GWResult,
    _check_backend,
    _check_mixing_method,
    _mixed_self_energies,
)
from .impurity_ed import FiniteBathImpurityED
from .model import NSUB, RubyParameters
from .supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
)
from .supercell_gw_fast import _solve_mu_matrix_fast, solve_matrix_gw_fast
from .supercell_gw_split import (
    compute_sigma_gw_split_matrix,
    one_body_density_matrix_tail,
)


@dataclass(frozen=True)
class ClusterEDGWFastOptions:
    max_iter: int = 30
    tol: float = 2.0e-5
    mixing: float = 0.40
    mixing_method: str = "pulay"  # "linear" or "pulay"
    pulay_history: int = 6
    pulay_start: int = 3
    pulay_regularization: float = 1.0e-7
    pulay_step_cap: float = 3.0
    # Optional pre-damping of the raw impurity map.  The default 1 means that
    # all damping/acceleration is handled by the coupled outer mixer.
    impurity_mixing: float = 1.0
    nbath: int = 6
    bath_fit_nfreq: int = 12
    bath_fit_max_nfev: int = 300
    bath_energy_window: float = 4.0
    bath_coupling_bound: float = 4.0
    bath_fit_xtol: float = 1.0e-9
    discard_weight_tol: float = 1.0e-11
    verbose: bool = True


@dataclass
class ClusterEDGWFastResult:
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
    bath: BathParameters
    converged: bool
    iterations: int
    final_error: float
    impurity_mismatch: float
    bath_fit_error: float
    background: GWResult
    mixing_method: str
    residual_history: np.ndarray
    impurity_residual_history: np.ndarray
    impurity_mismatch_history: np.ndarray
    bath_fit_history: np.ndarray
    mu_history: np.ndarray
    bath_nfev_history: np.ndarray
    elapsed_history: np.ndarray
    pulay_fallbacks: int


def _maxabs(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a)), initial=0.0))


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(b).ravel())), 1e-300)
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel()) / den)


def _pack_dynamic(
    sigma_emb: np.ndarray,
    sigma_imp: np.ndarray,
    nk: int,
) -> np.ndarray:
    """Pack lattice and impurity dynamic self-energies for one Pulay metric."""
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
    emb = flat[:nemb].reshape(sigma_emb_shape)
    imp = (flat[nemb:] / scale).reshape(sigma_imp_shape)
    return emb, imp


def solve_cluster_ed_gw_fast(
    h0: np.ndarray,
    Vq: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    *,
    gw_opts: GWOptions = GWOptions(),
    embed_opts: ClusterEDGWFastOptions = ClusterEDGWFastOptions(),
    background: GWResult | None = None,
) -> ClusterEDGWFastResult:
    """Solve the coupled lattice-GW / finite-bath ED fixed point with Pulay."""
    backend = _check_backend(gw_opts.momentum_backend)
    method = _check_mixing_method(embed_opts.mixing_method)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    if h0.shape != (grid.nk1, grid.nk2, NSUB, NSUB):
        raise ValueError("cluster ED+GW expects the primitive 6-site lattice basis")
    if Vq.shape != h0.shape:
        raise ValueError("Vq/h0 shape mismatch")
    if not (0.0 < float(embed_opts.mixing) <= 1.0):
        raise ValueError("embedding mixing must lie in (0,1]")
    if not (0.0 < float(embed_opts.impurity_mixing) <= 1.0):
        raise ValueError("impurity mixing must lie in (0,1]")
    if int(embed_opts.pulay_history) < 2:
        raise ValueError("pulay_history must be at least 2")

    if background is None:
        if embed_opts.verbose:
            print("[cluster-ED+GW] solving initial lattice GW background ...", flush=True)
        background = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    elif embed_opts.verbose:
        print(
            f"[cluster-ED+GW] using cached lattice GW background: "
            f"residual={background.final_error:.3e}",
            flush=True,
        )
    if not background.converged:
        raise RuntimeError(
            f"initial lattice GW background is not converged: {background.final_error:.3e}"
        )

    h_cluster_strict = build_intracell_h0(params)
    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
    alias_norm = _maxabs(h_cluster - h_cluster_strict)
    if embed_opts.verbose and alias_norm > 1.0e-12:
        print(
            f"[cluster-ED+GW] finite-torus local-block correction: "
            f"max|h_local-h_R0|={alias_norm:.3e}",
            flush=True,
        )

    interactions = ruby_cluster_interactions(params)
    V_cluster = np.asarray(Vq[0, 0], dtype=complex)

    sigma_h = np.asarray(background.Sigma_H, dtype=complex).copy()
    sigma_emb = np.asarray(background.Sigma_GW, dtype=complex).copy()
    mu = float(background.mu)
    G = np.asarray(background.G, dtype=complex).copy()
    bath: BathParameters | None = None

    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    Gc = np.mean(G, axis=(1, 2))
    rho_c = np.mean(rho_k, axis=(0, 1))
    sigma_cgw, _, _ = cluster_gw_self_energy(Gc, rho_c, V_cluster, grid)
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

    W = np.asarray(background.W)
    P = np.asarray(background.P)
    sigma_gw_lattice = np.asarray(background.Sigma_GW)
    Gimp = np.asarray(Gc)
    density = np.asarray(background.density)
    converged = False
    err = float("inf")
    mismatch = float("inf")
    it = 0

    for it in range(1, int(embed_opts.max_iter) + 1):
        t0 = perf_counter()
        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW] outer {it:02d}: build lattice GW map "
                f"(mix={method})",
                flush=True,
            )

        rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        rho_c = np.mean(rho_k, axis=(0, 1))
        density = np.real(np.diag(rho_c))
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_lattice = compute_sigma_gw_split_matrix(
            G, W, Vq, grid, h0, mu, sigma_h, backend=backend
        )

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
                f"[cluster-ED+GW] outer {it:02d}: fit {embed_opts.nbath} bath orbitals ...",
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
                f"[cluster-ED+GW] outer {it:02d}: bath relerr={bath.fit_error:.3e}, "
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
        sigma_emb_out = sigma_gw_lattice + correction[:, None, None, :, :]

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
                f"[cluster-ED+GW] outer {it:02d}: residual={err:.3e} "
                f"(emb={res_emb:.3e}, imp={res_imp:.3e}), "
                f"Gimp/Gc={mismatch:.3e}, bath={bath.fit_error:.3e}, "
                f"Ntot_imp={selection.average_particles:.6f}, "
                f"mu={mu:+.9f}, dt={elapsed:.1f}s",
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
                        f"[cluster-ED+GW] outer {it:02d}: Pulay safeguard -> "
                        f"linear fallback (mixed/raw={mixed_step/max(raw_step,1e-300):.2f})",
                        flush=True,
                    )

            sigma_h = np.asarray(sigma_h_next)
            sigma_emb, sigma_imp = _unpack_dynamic(
                dyn_next,
                sigma_emb.shape,
                sigma_imp.shape,
                grid.nk,
            )

        # Always update G so a converged return corresponds to the accepted
        # self-energy, and so the last non-converged iteration is not discarded.
        if gw_opts.target_filling is None:
            G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_emb)
        else:
            mu, G, _, _ = _solve_mu_matrix_fast(
                h0,
                sigma_h,
                sigma_emb,
                grid,
                float(gw_opts.target_filling),
                mu,
                float(gw_opts.mu_tol),
                int(gw_opts.mu_max_iter),
            )
        if converged:
            break

    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density = np.real(np.diag(np.mean(rho_k, axis=(0, 1))))
    Gc = np.mean(G, axis=(1, 2))

    if bath is None:
        raise RuntimeError("embedding loop did not execute")

    return ClusterEDGWFastResult(
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


__all__ = [
    "ClusterEDGWFastOptions",
    "ClusterEDGWFastResult",
    "solve_cluster_ed_gw_fast",
]
