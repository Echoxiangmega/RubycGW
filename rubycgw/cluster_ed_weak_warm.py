"""Cross-mesh warm starts for six-site ED + selectable weak-solver embedding.

A converged smaller-k-mesh result can seed a larger-k-mesh calculation without
interpolating the full k-dependent self-energy.  The mesh-independent cluster
correction

    DeltaSigma_C = Sigma_ED^C - Sigma_weak^C

is transplanted onto the newly converged weak background,

    Sigma_emb,new^(0)(k) = Sigma_weak,new(k) + DeltaSigma_C,old.

The impurity self-energy, bath, Hartree term and chemical potential are also
reused.  If the old and new meshes and interaction strength are identical, the
full saved G/Sigma_emb state is reused directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
from .cluster_ed_gw_fast import (
    ClusterEDGWFastOptions,
    _maxabs,
    _pack_dynamic,
    _relative_error,
    _unpack_dynamic,
)
from .cluster_ed_weak_fast import (
    ClusterEDWeakFastResult,
    canonical_weak_solver,
)
from .gf2 import (
    cluster_gf2_self_energy,
    compute_gf2_nonhartree,
    solve_matrix_gf2,
)
from .grids import MatsubaraGrid
from .gw import GWOptions, GWResult, _check_backend, _check_mixing_method, _mixed_self_energies
from .impurity_ed import FiniteBathImpurityED
from .model import NSUB, RubyParameters
from .sox_covariant import SOXOptions
from .supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
)
from .supercell_gw_fast import _solve_mu_matrix_fast, solve_matrix_gw_fast
from .supercell_gw_split import compute_sigma_gw_split_matrix, one_body_density_matrix_tail


@dataclass
class ClusterEDWarmStart:
    source: str
    weak_solver: str
    old_nk1: int
    old_nk2: int
    old_V: float
    old_T: float
    old_filling: float
    omega: np.ndarray
    Sigma_H: np.ndarray
    Sigma_imp: np.ndarray
    DeltaSigma_cluster: np.ndarray
    mu: float
    bath: BathParameters | None
    Sigma_emb_full: np.ndarray | None = None
    G_full: np.ndarray | None = None


def _scalar(z, name: str, default=None):
    if name in z:
        return np.asarray(z[name]).reshape(()).item()
    if default is not None:
        return default
    raise KeyError(name)


def load_cluster_ed_warm_start(
    path: str | Path,
    *,
    weak_solver: str,
    grid: MatsubaraGrid,
    filling: float,
    T: float,
    ti: float,
    t1: float,
    t2: float,
    V: float,
    nbath: int,
) -> ClusterEDWarmStart:
    """Load and validate a saved embedding state for mesh/V continuation.

    Old historical ``cluster_ed_gw_*.npz`` files are accepted; missing
    ``weak_solver``/``nk1``/``nk2`` metadata are interpreted as GW and Lx/Ly.
    The one-particle model, filling, temperature and Matsubara grid must match.
    ``V`` may differ intentionally for interaction-strength continuation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    weak = canonical_weak_solver(weak_solver)
    with np.load(p, allow_pickle=False) as z:
        old_weak = str(_scalar(z, "weak_solver", "gw"))
        old_weak = canonical_weak_solver(old_weak)
        if old_weak != weak:
            raise ValueError(
                f"warm-start weak solver {old_weak!r} != requested {weak!r}; "
                "cross-functional warm starts are disabled"
            )
        old_nk1 = int(_scalar(z, "nk1", _scalar(z, "Lx")))
        old_nk2 = int(_scalar(z, "nk2", _scalar(z, "Ly")))
        old_V = float(_scalar(z, "V"))
        old_T = float(_scalar(z, "T"))
        old_filling = float(_scalar(z, "filling"))
        for name, requested in (("ti", ti), ("t1", t1), ("t2", t2)):
            saved = float(_scalar(z, name))
            if not np.isclose(saved, float(requested), atol=1e-13, rtol=1e-12):
                raise ValueError(f"warm-start {name}={saved} != requested {requested}")
        if not np.isclose(old_T, float(T), atol=1e-13, rtol=1e-12):
            raise ValueError(f"warm-start T={old_T} != requested {T}")
        if not np.isclose(old_filling, float(filling), atol=1e-13, rtol=1e-12):
            raise ValueError(
                f"warm-start filling={old_filling} != requested {filling}"
            )
        omega = np.asarray(z["omega"], dtype=float)
        if omega.shape != grid.omega.shape or np.max(np.abs(omega - grid.omega)) > 1e-12:
            raise ValueError("warm-start Matsubara grid differs from requested grid")

        sigma_h = np.asarray(z["Sigma_H"], dtype=complex)
        sigma_imp = np.asarray(z["Sigma_ED_cluster"], dtype=complex)
        weak_key = "Sigma_weak_cluster" if "Sigma_weak_cluster" in z else "Sigma_GW_cluster"
        sigma_cweak = np.asarray(z[weak_key], dtype=complex)
        if sigma_imp.shape != (grid.nf, NSUB, NSUB):
            raise ValueError("warm-start impurity self-energy shape mismatch")
        if sigma_cweak.shape != sigma_imp.shape:
            raise ValueError("warm-start cluster weak self-energy shape mismatch")
        delta_sigma = sigma_imp - sigma_cweak
        mu = float(_scalar(z, "mu"))

        bath = None
        if "bath_energies" in z and "bath_couplings" in z:
            eps = np.asarray(z["bath_energies"], dtype=float)
            couplings = np.asarray(z["bath_couplings"], dtype=complex)
            if eps.shape == (int(nbath),) and couplings.shape == (NSUB, int(nbath)):
                bath = BathParameters(
                    energies=eps.copy(),
                    couplings=couplings.copy(),
                    fit_error=float(_scalar(z, "bath_fit_error", np.nan)),
                    nfev=0,
                )

        sigma_emb_full = None
        G_full = None
        if "Sigma_emb" in z and "G" in z:
            se = np.asarray(z["Sigma_emb"], dtype=complex)
            gg = np.asarray(z["G"], dtype=complex)
            old_shape = (grid.nf, old_nk1, old_nk2, NSUB, NSUB)
            if se.shape == old_shape and gg.shape == old_shape:
                sigma_emb_full = se.copy()
                G_full = gg.copy()

    return ClusterEDWarmStart(
        source=str(p),
        weak_solver=weak,
        old_nk1=old_nk1,
        old_nk2=old_nk2,
        old_V=old_V,
        old_T=old_T,
        old_filling=old_filling,
        omega=omega,
        Sigma_H=sigma_h.copy(),
        Sigma_imp=sigma_imp.copy(),
        DeltaSigma_cluster=delta_sigma.copy(),
        mu=mu,
        bath=bath,
        Sigma_emb_full=sigma_emb_full,
        G_full=G_full,
    )


def _initial_from_warm(
    warm: ClusterEDWarmStart,
    background: GWResult,
    h0: np.ndarray,
    grid: MatsubaraGrid,
    gw_opts: GWOptions,
    V: float,
):
    same_mesh = (warm.old_nk1, warm.old_nk2) == (grid.nk1, grid.nk2)
    same_V = np.isclose(float(warm.old_V), float(V), atol=1e-13, rtol=1e-12)
    full_ok = (
        same_mesh
        and same_V
        and warm.Sigma_emb_full is not None
        and warm.G_full is not None
        and warm.Sigma_emb_full.shape == background.Sigma_GW.shape
        and warm.G_full.shape == background.G.shape
    )

    sigma_h = np.asarray(warm.Sigma_H, dtype=complex).copy()
    sigma_imp = np.asarray(warm.Sigma_imp, dtype=complex).copy()
    bath = warm.bath
    if full_ok:
        sigma_emb = np.asarray(warm.Sigma_emb_full, dtype=complex).copy()
        G = np.asarray(warm.G_full, dtype=complex).copy()
        mu = float(warm.mu)
        mode = "full-state"
    else:
        sigma_emb = (
            np.asarray(background.Sigma_GW, dtype=complex)
            + np.asarray(warm.DeltaSigma_cluster, dtype=complex)[:, None, None, :, :]
        )
        mu_seed = float(warm.mu)
        if gw_opts.target_filling is None:
            mu = mu_seed
            G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_emb)
        else:
            mu, G, _, _ = _solve_mu_matrix_fast(
                h0,
                sigma_h,
                sigma_emb,
                grid,
                float(gw_opts.target_filling),
                mu_seed,
                float(gw_opts.mu_tol),
                int(gw_opts.mu_max_iter),
            )
        mode = "cluster-correction"
    return sigma_h, sigma_emb, sigma_imp, G, float(mu), bath, mode


def _weak_background(
    weak: str,
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    gw_opts: GWOptions,
    gf2_sox_opts: SOXOptions,
    background: GWResult | None,
):
    if background is not None:
        return background
    if weak == "gw":
        return solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    return solve_matrix_gf2(h0, Vq, grid, opts=gw_opts, sox_opts=gf2_sox_opts)


def solve_cluster_ed_weak_warm(
    h0: np.ndarray,
    Vq: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    *,
    warm_start: ClusterEDWarmStart,
    weak_solver: str = "gw",
    gw_opts: GWOptions = GWOptions(),
    embed_opts: ClusterEDGWFastOptions = ClusterEDGWFastOptions(),
    gf2_sox_opts: SOXOptions = SOXOptions(n_quad=64),
    background: GWResult | None = None,
) -> ClusterEDWeakFastResult:
    """Solve a warm-started embedding fixed point on an arbitrary new k mesh."""
    weak = canonical_weak_solver(weak_solver)
    if warm_start.weak_solver != weak:
        raise ValueError("warm-start weak solver mismatch")
    backend = _check_backend(gw_opts.momentum_backend)
    method = _check_mixing_method(embed_opts.mixing_method)
    gf2_sox_opts.validate()
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    if h0.shape != (grid.nk1, grid.nk2, NSUB, NSUB) or Vq.shape != h0.shape:
        raise ValueError("h0/Vq shape mismatch")

    background = _weak_background(
        weak, h0, Vq, grid, gw_opts, gf2_sox_opts, background
    )
    if not background.converged:
        raise RuntimeError(
            f"initial {weak.upper()} background is not converged: "
            f"{background.final_error:.3e}"
        )

    sigma_h, sigma_emb, sigma_imp, G, mu, bath, warm_mode = _initial_from_warm(
        warm_start, background, h0, grid, gw_opts, params.V
    )
    if embed_opts.verbose:
        print(
            f"[cluster-ED+{weak.upper()}] warm start: {warm_mode} from "
            f"{warm_start.old_nk1}x{warm_start.old_nk2}, V={warm_start.old_V:g}; "
            f"new mesh={grid.nk1}x{grid.nk2}, V={params.V:g}, "
            f"bath={'reused' if bath is not None else 'refit'}",
            flush=True,
        )

    h_cluster_strict = build_intracell_h0(params)
    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
    alias_norm = _maxabs(h_cluster - h_cluster_strict)
    if embed_opts.verbose and alias_norm > 1e-12:
        print(
            f"[cluster-ED+{weak.upper()}] discrete-k local-block correction: "
            f"max|h_local-h_R0|={alias_norm:.3e}",
            flush=True,
        )

    interactions = ruby_cluster_interactions(params)
    V_cluster = np.asarray(Vq[0, 0], dtype=complex)
    mix_opts = GWOptions(
        mixing=float(embed_opts.mixing),
        mixing_method=method,
        pulay_history=int(embed_opts.pulay_history),
        pulay_start=int(embed_opts.pulay_start),
        pulay_regularization=float(embed_opts.pulay_regularization),
    )
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
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
    sigma_weak_lattice = np.asarray(background.Sigma_GW)
    Gimp = np.mean(G, axis=(1, 2))
    sigma_cweak = np.asarray(warm_start.Sigma_imp - warm_start.DeltaSigma_cluster)
    density = np.asarray(background.density)
    converged = False
    err = float("inf")
    mismatch = float("inf")
    eye = np.eye(NSUB, dtype=complex)
    it = 0

    for it in range(1, int(embed_opts.max_iter) + 1):
        t0 = perf_counter()
        rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        rho_c = np.mean(rho_k, axis=(0, 1))
        density = np.real(np.diag(rho_c))
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])

        if weak == "gw":
            P = compute_polarization_matrix(G, grid, backend=backend)
            W = compute_screened_interaction_matrix(P, Vq)
            sigma_weak_lattice = compute_sigma_gw_split_matrix(
                G, W, Vq, grid, h0, mu, sigma_h, backend=backend
            )
        else:
            sigma_weak_lattice, P, W, _, _ = compute_gf2_nonhartree(
                G,
                Vq,
                h0,
                mu,
                sigma_h,
                grid,
                backend=backend,
                sox_opts=gf2_sox_opts,
            )

        Gc = np.mean(G, axis=(1, 2))
        if weak == "gw":
            sigma_cweak, _, _ = cluster_gw_self_energy(
                Gc, rho_c, V_cluster, grid
            )
        else:
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
        correction = sigma_imp_out - sigma_cweak
        sigma_emb_out = sigma_weak_lattice + correction[:, None, None, :, :]

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
                f"[cluster-ED+{weak.upper()}] outer {it:02d}: residual={err:.3e} "
                f"(emb={res_emb:.3e}, imp={res_imp:.3e}), "
                f"Gimp/Gc={mismatch:.3e}, bath={bath.fit_error:.3e}, "
                f"Ntot_imp={selection.average_particles:.6f}, mu={mu:+.9f}, "
                f"dt={elapsed:.1f}s",
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
                history,
            )
            raw_step = max(_maxabs(sigma_h_out - sigma_h), _maxabs(dyn_out - dyn))
            mixed_step = max(
                _maxabs(sigma_h_next - sigma_h),
                _maxabs(dyn_next - dyn),
            )
            unsafe = (
                not np.all(np.isfinite(sigma_h_next))
                or not np.all(np.isfinite(dyn_next))
                or (
                    raw_step > 1e-14
                    and mixed_step > float(embed_opts.pulay_step_cap) * raw_step
                )
            )
            if unsafe:
                fallbacks += 1
                history.clear()
                a = float(embed_opts.mixing)
                sigma_h_next = sigma_h + a * (sigma_h_out - sigma_h)
                dyn_next = dyn + a * (dyn_out - dyn)
            sigma_h = np.asarray(sigma_h_next)
            sigma_emb, sigma_imp = _unpack_dynamic(
                dyn_next, sigma_emb.shape, sigma_imp.shape, grid.nk
            )

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
    return ClusterEDWeakFastResult(
        weak_solver=weak,
        G=np.asarray(G),
        W=np.asarray(W),
        P=np.asarray(P),
        Sigma_H=np.asarray(sigma_h),
        Sigma_emb=np.asarray(sigma_emb),
        Sigma_weak_lattice=np.asarray(sigma_weak_lattice),
        Sigma_weak_cluster=np.asarray(sigma_cweak),
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
    "ClusterEDWarmStart",
    "load_cluster_ed_warm_start",
    "solve_cluster_ed_weak_warm",
]
