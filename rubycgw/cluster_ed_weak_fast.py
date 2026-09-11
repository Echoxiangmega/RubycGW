"""Pluggable weak-solver + six-site ED self-energy embedding.

The strong solver remains the same six-site finite-bath ED impurity.  The weak
lattice skeleton and its cluster double counting are selected by ``weak_solver``:

* ``gw``  : historical lattice GW / cluster GW embedding;
* ``gf2`` : self-consistent second-order Green-function theory, including both
            direct and exchange O(V^2) skeletons.

For GW this module delegates to the validated historical solver.  GF2 uses the
same outer Pulay/DIIS embedding loop and the same impurity/bath machinery, so
comparisons isolate the weak-solver choice.
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
    fit_finite_bath,
    ruby_cluster_interactions,
)
from .cluster_ed_gw_fast import (
    ClusterEDGWFastOptions,
    _maxabs,
    _pack_dynamic,
    _relative_error,
    _unpack_dynamic,
    solve_cluster_ed_gw_fast,
)
from .gf2 import cluster_gf2_self_energy, compute_gf2_nonhartree, solve_matrix_gf2
from .grids import MatsubaraGrid
from .gw import GWOptions, GWResult, _check_backend, _check_mixing_method, _mixed_self_energies
from .impurity_ed import FiniteBathImpurityED
from .model import NSUB, RubyParameters
from .sox_covariant import SOXOptions
from .supercell_gw import dyson_from_sigma_matrix, hartree_self_energy_matrix
from .supercell_gw_fast import _solve_mu_matrix_fast
from .supercell_gw_split import one_body_density_matrix_tail


WEAK_SOLVERS = ("gw", "gf2")


def canonical_weak_solver(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {"gw": "gw", "gf2": "gf2", "second_order": "gf2", "second-order": "gf2"}
    if key not in aliases:
        raise ValueError(f"weak_solver must be one of {WEAK_SOLVERS}")
    return aliases[key]


@dataclass
class ClusterEDWeakFastResult:
    weak_solver: str
    G: np.ndarray
    W: np.ndarray
    P: np.ndarray
    Sigma_H: np.ndarray
    Sigma_emb: np.ndarray
    Sigma_weak_lattice: np.ndarray
    Sigma_weak_cluster: np.ndarray
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

    # Backward-compatible aliases.  For weak_solver='gf2' these mean the generic
    # weak self-energy, not literal GW; new code should use Sigma_weak_*.
    @property
    def Sigma_GW_lattice(self) -> np.ndarray:
        return self.Sigma_weak_lattice

    @property
    def Sigma_GW_cluster(self) -> np.ndarray:
        return self.Sigma_weak_cluster


def _wrap_gw_result(result) -> ClusterEDWeakFastResult:
    return ClusterEDWeakFastResult(
        weak_solver="gw",
        G=np.asarray(result.G),
        W=np.asarray(result.W),
        P=np.asarray(result.P),
        Sigma_H=np.asarray(result.Sigma_H),
        Sigma_emb=np.asarray(result.Sigma_emb),
        Sigma_weak_lattice=np.asarray(result.Sigma_GW_lattice),
        Sigma_weak_cluster=np.asarray(result.Sigma_GW_cluster),
        Sigma_ED_cluster=np.asarray(result.Sigma_ED_cluster),
        G_cluster=np.asarray(result.G_cluster),
        G_impurity=np.asarray(result.G_impurity),
        mu=float(result.mu),
        density=np.asarray(result.density),
        bath=result.bath,
        converged=bool(result.converged),
        iterations=int(result.iterations),
        final_error=float(result.final_error),
        impurity_mismatch=float(result.impurity_mismatch),
        bath_fit_error=float(result.bath_fit_error),
        background=result.background,
        mixing_method=str(result.mixing_method),
        residual_history=np.asarray(result.residual_history),
        impurity_residual_history=np.asarray(result.impurity_residual_history),
        impurity_mismatch_history=np.asarray(result.impurity_mismatch_history),
        bath_fit_history=np.asarray(result.bath_fit_history),
        mu_history=np.asarray(result.mu_history),
        bath_nfev_history=np.asarray(result.bath_nfev_history),
        elapsed_history=np.asarray(result.elapsed_history),
        pulay_fallbacks=int(result.pulay_fallbacks),
    )


def solve_cluster_ed_weak_fast(
    h0: np.ndarray,
    Vq: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    *,
    weak_solver: str = "gw",
    gw_opts: GWOptions = GWOptions(),
    embed_opts: ClusterEDGWFastOptions = ClusterEDGWFastOptions(),
    gf2_sox_opts: SOXOptions = SOXOptions(n_quad=64),
    background: GWResult | None = None,
) -> ClusterEDWeakFastResult:
    """Solve ED embedding with a selectable GW or GF2 weak functional."""
    weak = canonical_weak_solver(weak_solver)
    if weak == "gw":
        return _wrap_gw_result(
            solve_cluster_ed_gw_fast(
                h0,
                Vq,
                params,
                grid,
                gw_opts=gw_opts,
                embed_opts=embed_opts,
                background=background,
            )
        )

    backend = _check_backend(gw_opts.momentum_backend)
    method = _check_mixing_method(embed_opts.mixing_method)
    gf2_sox_opts.validate()
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    if h0.shape != (grid.nk1, grid.nk2, NSUB, NSUB) or Vq.shape != h0.shape:
        raise ValueError("cluster ED+GF2 expects primitive 6-site h0/Vq on the chosen k mesh")
    if not (0.0 < float(embed_opts.mixing) <= 1.0):
        raise ValueError("embedding mixing must lie in (0,1]")
    if not (0.0 < float(embed_opts.impurity_mixing) <= 1.0):
        raise ValueError("impurity mixing must lie in (0,1]")

    if background is None:
        if embed_opts.verbose:
            print("[cluster-ED+GF2] solving initial lattice GF2 background ...", flush=True)
        background = solve_matrix_gf2(
            h0, Vq, grid, opts=gw_opts, sox_opts=gf2_sox_opts
        )
    elif embed_opts.verbose:
        print(
            f"[cluster-ED+GF2] using cached lattice GF2 background: residual={background.final_error:.3e}",
            flush=True,
        )
    if not background.converged:
        raise RuntimeError(
            f"initial lattice GF2 background is not converged: {background.final_error:.3e}"
        )

    h_cluster_strict = build_intracell_h0(params)
    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
    alias_norm = _maxabs(h_cluster - h_cluster_strict)
    if embed_opts.verbose and alias_norm > 1.0e-12:
        print(
            f"[cluster-ED+GF2] discrete-k local-block correction: max|h_local-h_R0|={alias_norm:.3e}",
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
    sigma_cweak, _, _ = cluster_gf2_self_energy(
        Gc,
        rho_c,
        h_cluster,
        V_cluster,
        mu,
        grid,
        sox_opts=gf2_sox_opts,
    )
    sigma_imp = np.array(sigma_cweak, copy=True)

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
    sigma_weak_lattice = np.asarray(background.Sigma_GW)
    Gimp = np.asarray(Gc)
    density = np.asarray(background.density)
    converged = False
    err = float("inf")
    mismatch = float("inf")
    it = 0

    for it in range(1, int(embed_opts.max_iter) + 1):
        t0 = perf_counter()
        if embed_opts.verbose:
            print(f"[cluster-ED+GF2] outer {it:02d}: build lattice GF2 map", flush=True)

        rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        rho_c = np.mean(rho_k, axis=(0, 1))
        density = np.real(np.diag(rho_c))
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
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
        eye = np.eye(NSUB, dtype=complex)
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
        delta_fit = bath_hybridization(grid.omega, mu, bath.energies, bath.couplings)
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
                f"[cluster-ED+GF2] outer {it:02d}: residual={err:.3e} "
                f"(emb={res_emb:.3e}, imp={res_imp:.3e}), Gimp/Gc={mismatch:.3e}, "
                f"bath={bath.fit_error:.3e}, Ntot_imp={selection.average_particles:.6f}, "
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
                sigma_h, dyn, sigma_h_out, dyn_out, mix_opts, it, mix_history
            )
            raw_step = max(_maxabs(sigma_h_out - sigma_h), _maxabs(dyn_out - dyn))
            mixed_step = max(_maxabs(sigma_h_next - sigma_h), _maxabs(dyn_next - dyn))
            unsafe = (
                not np.all(np.isfinite(sigma_h_next))
                or not np.all(np.isfinite(dyn_next))
                or (
                    raw_step > 1.0e-14
                    and mixed_step > float(embed_opts.pulay_step_cap) * raw_step
                )
            )
            if unsafe:
                fallbacks += 1
                mix_history.clear()
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
        weak_solver="gf2",
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
    "WEAK_SOLVERS",
    "canonical_weak_solver",
    "ClusterEDWeakFastResult",
    "solve_cluster_ed_weak_fast",
]
