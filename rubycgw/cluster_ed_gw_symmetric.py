"""C3- and time-reversal-constrained three-orientation cluster ED+GW.

This module constructs a symmetric parent branch for comparing ordering
susceptibilities.  The physical lattice GW functional is kept once, while the
strong six-site correction is the equal-weight average of the three C3-related
real A-B pair cuts,

    Sigma_emb = Sigma_GW^lat
              + (1/3) sum_r U_r^{-1}
                [Sigma_ED^(r) - Sigma_GW,C^(r)] U_r.

Each orientation contains the same six intra-triangle V bonds and exactly two
Vprime plus two Vcross bonds belonging to one real neighbouring A-B triangle
pair.  The factor 1/3 is therefore essential: this is the self-energy derivative
of the explicitly symmetrized functional

    Phi_sym = Phi_GW^lat
            + (1/3) sum_r [Phi_ED^(r) - Phi_GW,C^(r)].

The nonlinear iteration is additionally projected onto the C3- and spinless
time-reversal-invariant subspace.  This projection is a branch constraint used
only to obtain the unbroken parent solution; a response calculation must
linearize the unprojected Phi_sym functional so that both C3-breaking charge
order and time-reversal-breaking loop-current fluctuations remain available.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from . import cluster_ed_gw_fast as fast
from .c3_constraint import (
    c3_lattice_residual,
    project_density_c3,
    project_lattice_c3,
    rotate_lattice_c3,
)
from .cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    build_impurity_one_body,
    cluster_gw_self_energy,
    cluster_interaction_matrix,
    split_static_hybridization,
)
from .cluster_orientation import (
    build_oriented_lattice_fields,
    gauge_transform_lattice,
    orientation_b_shift,
)
from .grids import MatsubaraGrid
from .gw import GWOptions, GWResult, _mixed_self_energies
from .impurity_ed import FiniteBathImpurityED
from .model import NSUB, RubyParameters
from .models.ruby import physical_pair_cluster_interactions
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


@dataclass
class SymmetrizedClusterEDGWResult:
    G: np.ndarray
    W: np.ndarray
    P: np.ndarray
    Sigma_H: np.ndarray
    Sigma_emb: np.ndarray
    Sigma_GW_lattice: np.ndarray
    Sigma_GW_cluster_orientation: np.ndarray
    Sigma_ED_cluster_orientation: np.ndarray
    G_cluster_orientation: np.ndarray
    G_impurity_orientation: np.ndarray
    impurity_static_shift_orientation: np.ndarray
    baths: tuple[BathParameters, BathParameters, BathParameters]
    mu: float
    density: np.ndarray
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
    elapsed_history: np.ndarray
    orientation_spread_history: np.ndarray
    c3_residual_history: np.ndarray
    mixer_fallbacks: int


def _maxabs(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a)), initial=0.0))


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(b).ravel())), 1.0e-300)
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel()) / den)


def _negative_momentum_indices(n: int) -> np.ndarray:
    i = np.arange(int(n), dtype=int)
    return (-i) % int(n)


def project_fermion_time_reversal(field: np.ndarray) -> np.ndarray:
    """Project X(k,iw) onto spinless TR: X(k,iw)=X(-k,-iw)^*."""
    x = np.asarray(field, dtype=complex)
    if x.ndim != 5 or x.shape[-2:] != (NSUB, NSUB):
        raise ValueError("fermion field must have shape (nf,nk1,nk2,6,6)")
    i1 = _negative_momentum_indices(x.shape[1])
    i2 = _negative_momentum_indices(x.shape[2])
    partner = np.conj(x[::-1][:, i1][:, :, i2])
    return 0.5 * (x + partner)


def project_local_time_reversal(field: np.ndarray) -> np.ndarray:
    """Project a local Matsubara matrix onto X(iw)=X(-iw)^*."""
    x = np.asarray(field, dtype=complex)
    if x.ndim != 3 or x.shape[-2:] != (NSUB, NSUB):
        raise ValueError("local fermion field must have shape (nf,6,6)")
    return 0.5 * (x + np.conj(x[::-1]))


def _project_parent_dynamic(field: np.ndarray) -> np.ndarray:
    """Project a lattice fermion field onto the C3 x TR parent subspace."""
    return project_lattice_c3(project_fermion_time_reversal(field))


def _project_parent_hartree(sigma_h: np.ndarray) -> np.ndarray:
    """C3/TR projection for the static Hartree block."""
    h = np.asarray(sigma_h, dtype=complex)
    if h.shape != (NSUB, NSUB):
        raise ValueError("Hartree block must be 6x6")
    d = np.real(np.diag(h))
    d = project_density_c3(d)
    return np.diag(d.astype(complex))


def _local_to_common_lattice(
    local: np.ndarray,
    orientation: int,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Broadcast an orientation-local matrix and return it to ori0 gauge."""
    x = np.asarray(local, dtype=complex)
    if x.ndim < 2 or x.shape[-2:] != (NSUB, NSUB):
        raise ValueError("local field must end in (6,6)")
    lead = x.shape[:-2]
    lattice = np.broadcast_to(
        x.reshape(lead + (1, 1, NSUB, NSUB)),
        lead + (grid.nk1, grid.nk2, NSUB, NSUB),
    ).copy()
    return gauge_transform_lattice(
        lattice, -orientation_b_shift(int(orientation))
    )


def _impurity_average_lattice(
    sigma_imp: np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    s = np.asarray(sigma_imp, dtype=complex)
    if s.ndim != 4 or s.shape[0] != 3 or s.shape[-2:] != (NSUB, NSUB):
        raise ValueError("sigma_imp must have shape (3,nf,6,6)")
    out = np.zeros(
        (s.shape[1], grid.nk1, grid.nk2, NSUB, NSUB), dtype=complex
    )
    for r in range(3):
        out += _local_to_common_lattice(s[r], r, grid)
    return out / 3.0


def _pack_three_dynamic(
    sigma_emb: np.ndarray,
    sigma_imp: np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Pack a nonredundant common-lattice + three-impurity dynamic state."""
    emb = np.asarray(sigma_emb, dtype=complex)
    imp = np.asarray(sigma_imp, dtype=complex)
    imp_avg = _impurity_average_lattice(imp, grid)
    weak = emb - imp_avg
    scale = np.sqrt(float(grid.nk) / 3.0)
    return np.concatenate(
        [weak.ravel()] + [scale * imp[r].ravel() for r in range(3)]
    )


def _unpack_three_dynamic(
    packed: np.ndarray,
    sigma_emb_shape: tuple[int, ...],
    sigma_imp_shape: tuple[int, ...],
    grid: MatsubaraGrid,
) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(packed, dtype=complex).reshape(-1)
    nemb = int(np.prod(sigma_emb_shape))
    if len(sigma_imp_shape) != 4 or sigma_imp_shape[0] != 3:
        raise ValueError("sigma_imp_shape must start with orientation axis 3")
    nimp = int(np.prod(sigma_imp_shape[1:]))
    expected = nemb + 3 * nimp
    if flat.size != expected:
        raise ValueError(
            f"three-orientation packed size {flat.size} != expected {expected}"
        )
    weak = flat[:nemb].reshape(sigma_emb_shape)
    scale = np.sqrt(float(grid.nk) / 3.0)
    imp = np.empty(sigma_imp_shape, dtype=complex)
    offset = nemb
    for r in range(3):
        imp[r] = (flat[offset:offset + nimp] / scale).reshape(
            sigma_imp_shape[1:]
        )
        offset += nimp
    emb = weak + _impurity_average_lattice(imp, grid)
    return emb, imp


def _orientation_spread(corrections_common: list[np.ndarray]) -> float:
    """Mismatch from the exact C3 orbit of the orientation-0 correction."""
    ref = np.asarray(corrections_common[0], dtype=complex)
    worst = 0.0
    rotated = ref
    for r in (1, 2):
        rotated = rotate_lattice_c3(rotated)
        worst = max(worst, _maxabs(corrections_common[r] - rotated))
    return float(worst)


def solve_cluster_ed_gw_three_orientation_symmetric(
    h0: np.ndarray,
    Vq: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    *,
    gw_opts: GWOptions = GWOptions(),
    embed_opts: fast.ClusterEDGWFastOptions = fast.ClusterEDGWFastOptions(),
    background: GWResult | None = None,
) -> SymmetrizedClusterEDGWResult:
    """Solve the C3/TR-constrained parent branch of the symmetrized functional."""
    if grid.nk1 != grid.nk2:
        raise ValueError("three-orientation C3 parent requires a square k mesh")
    backend = fast._check_backend(gw_opts.momentum_backend)
    method = fast._check_embed_mixing_method(embed_opts.mixing_method)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    expected = (grid.nk1, grid.nk2, NSUB, NSUB)
    if h0.shape != expected or Vq.shape != expected:
        raise ValueError("h0 and Vq must both have shape (nk,nk,6,6)")
    if not (0.0 < float(embed_opts.mixing) <= 1.0):
        raise ValueError("embedding mixing must lie in (0,1]")
    if not (0.0 < float(embed_opts.impurity_mixing) <= 1.0):
        raise ValueError("impurity mixing must lie in (0,1]")

    if background is None:
        if embed_opts.verbose:
            print(
                "[cluster-ED+GW:sym] solving one canonical lattice SCGW background ...",
                flush=True,
            )
        background = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    if not background.converged:
        raise RuntimeError(
            "initial lattice GW background is not converged: "
            f"{background.final_error:.3e}"
        )

    # Start exactly inside the desired parent subspace even if accumulated
    # floating-point noise in the unconstrained SCGW background is asymmetric.
    sigma_h = _project_parent_hartree(background.Sigma_H)
    sigma_emb = _project_parent_dynamic(background.Sigma_GW)
    mu = float(background.mu)
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

    h0_oriented: list[np.ndarray] = []
    h_cluster: list[np.ndarray] = []
    interactions = []
    V_cluster = []
    for r in range(3):
        hr, _ = build_oriented_lattice_fields(h0, Vq, r)
        h0_oriented.append(np.asarray(hr))
        hc = np.mean(hr, axis=(0, 1))
        h_cluster.append(0.5 * (hc + hc.conj().T))
        ir = physical_pair_cluster_interactions(params, r)
        interactions.append(ir)
        V_cluster.append(cluster_interaction_matrix(ir, NSUB))

    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    sigma_imp = np.empty((3, grid.nf, NSUB, NSUB), dtype=complex)
    for r in range(3):
        Gr = gauge_transform_lattice(G, orientation_b_shift(r))
        rr = gauge_transform_lattice(rho_k, orientation_b_shift(r))
        Gcr = np.mean(Gr, axis=(1, 2))
        rhor = np.mean(rr, axis=(0, 1))
        sigma_imp[r], _, _ = cluster_gw_self_energy(
            Gcr, rhor, V_cluster[r], grid
        )
        sigma_imp[r] = project_local_time_reversal(sigma_imp[r])

    baths: list[BathParameters | None] = [None, None, None]
    static_shifts = np.zeros((3, NSUB, NSUB), dtype=complex)
    Gc_stack = np.zeros((3, grid.nf, NSUB, NSUB), dtype=complex)
    Gimp_stack = np.zeros_like(Gc_stack)
    sigma_cgw_stack = np.zeros_like(Gc_stack)

    mix_opts = GWOptions(
        mixing=float(embed_opts.mixing),
        mixing_method=("pulay" if method == "broyden" else method),
        pulay_history=int(embed_opts.pulay_history),
        pulay_start=int(embed_opts.pulay_start),
        pulay_regularization=float(embed_opts.pulay_regularization),
    )
    mix_history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    broyden = (
        fast._LimitedMemoryBroyden(
            alpha=float(embed_opts.mixing),
            history=int(embed_opts.broyden_history),
            regularization=float(embed_opts.broyden_regularization),
        )
        if method == "broyden"
        else None
    )
    fallbacks = 0

    residual_hist: list[float] = []
    imp_residual_hist: list[list[float]] = []
    mismatch_hist: list[list[float]] = []
    bath_hist: list[list[float]] = []
    mu_hist: list[float] = []
    elapsed_hist: list[float] = []
    spread_hist: list[float] = []
    c3_hist: list[float] = []

    W = np.asarray(background.W)
    P = np.asarray(background.P)
    sigma_gw_lattice = _project_parent_dynamic(background.Sigma_GW)
    density = project_density_c3(np.asarray(background.density, dtype=float))
    converged = False
    err = float("inf")
    it = 0

    for it in range(1, int(embed_opts.max_iter) + 1):
        t0 = perf_counter()
        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW:sym] outer {it:02d}: lattice GW + 3 pair impurities "
                f"(mix={method})",
                flush=True,
            )

        rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        rho_c_common = np.mean(rho_k, axis=(0, 1))
        density = project_density_c3(np.real(np.diag(rho_c_common)))
        sigma_h_out = _project_parent_hartree(
            hartree_self_energy_matrix(density, Vq[0, 0])
        )
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_lattice = compute_sigma_gw_split_matrix(
            G, W, Vq, grid, h0, mu, sigma_h, backend=backend
        )
        sigma_gw_lattice = _project_parent_dynamic(sigma_gw_lattice)

        sigma_imp_out = np.empty_like(sigma_imp)
        corrections_common: list[np.ndarray] = []
        imp_res = []
        mismatches = []
        bath_errors = []

        for r in range(3):
            shift = orientation_b_shift(r)
            Gr = gauge_transform_lattice(G, shift)
            rho_kr = gauge_transform_lattice(rho_k, shift)
            Gcr = np.mean(Gr, axis=(1, 2))
            rho_cr = np.mean(rho_kr, axis=(0, 1))
            Gc_stack[r] = Gcr

            sigma_cgw, _, _ = cluster_gw_self_energy(
                Gcr, rho_cr, V_cluster[r], grid
            )
            sigma_cgw = project_local_time_reversal(sigma_cgw)
            sigma_cgw_stack[r] = sigma_cgw

            g0_inv = np.linalg.inv(Gcr) + sigma_imp[r]
            eye = np.eye(NSUB, dtype=complex)
            delta_raw = (
                (1j * grid.omega[:, None, None] + float(mu))
                * eye[None, :, :]
                - h_cluster[r][None, :, :]
                - g0_inv
            )
            static_shift, delta_target = split_static_hybridization(
                delta_raw, grid.omega
            )
            static_shifts[r] = 0.5 * (
                static_shift + static_shift.conj().T
            )
            h_impurity = h_cluster[r] + static_shifts[r]

            if embed_opts.verbose:
                print(
                    f"    ori{r}: fit {embed_opts.nbath} bath orbitals ...",
                    flush=True,
                )
            bath = fast.fit_finite_bath(
                delta_target,
                grid.omega,
                mu,
                nbath=int(embed_opts.nbath),
                nfit=int(embed_opts.bath_fit_nfreq),
                max_nfev=int(embed_opts.bath_fit_max_nfev),
                energy_window=float(embed_opts.bath_energy_window),
                coupling_bound=float(embed_opts.bath_coupling_bound),
                xtol=float(embed_opts.bath_fit_xtol),
                initial=baths[r],
                metric=str(embed_opts.bath_fit_metric),
                one_body=h_impurity,
            )
            baths[r] = bath

            himp = build_impurity_one_body(h_impurity, bath)
            impurity = FiniteBathImpurityED(
                himp,
                interactions[r],
                correlated_orbitals=tuple(range(NSUB)),
            )
            impurity.diagonalize()
            Gimp, _ = impurity.green_iomega(
                1j * grid.omega,
                mu,
                grid.T,
                orbitals=tuple(range(NSUB)),
                discard_weight_tol=float(embed_opts.discard_weight_tol),
            )
            Gimp_stack[r] = Gimp
            delta_fit = bath_hybridization(
                grid.omega, mu, bath.energies, bath.couplings
            )
            g0_fit_inv = (
                (1j * grid.omega[:, None, None] + float(mu))
                * eye[None, :, :]
                - h_impurity[None, :, :]
                - delta_fit
            )
            sigma_ed_raw = g0_fit_inv - np.linalg.inv(Gimp)
            sigma_ed_raw = project_local_time_reversal(sigma_ed_raw)

            beta = float(embed_opts.impurity_mixing)
            sigma_imp_out[r] = sigma_imp[r] + beta * (
                sigma_ed_raw - sigma_imp[r]
            )
            sigma_imp_out[r] = project_local_time_reversal(sigma_imp_out[r])

            correction_local = sigma_imp_out[r] - sigma_cgw
            correction_common = _local_to_common_lattice(
                correction_local, r, grid
            )
            corrections_common.append(correction_common)
            imp_res.append(_maxabs(sigma_imp_out[r] - sigma_imp[r]))
            mismatches.append(_relative_error(Gimp, Gcr))
            bath_errors.append(float(bath.fit_error))

        correction_avg = sum(corrections_common) / 3.0
        sigma_emb_out = _project_parent_dynamic(
            sigma_gw_lattice + correction_avg
        )

        res_h = _maxabs(sigma_h_out - sigma_h)
        res_emb = _maxabs(sigma_emb_out - sigma_emb)
        err = max(res_h, res_emb, max(imp_res))
        spread = _orientation_spread(corrections_common)
        c3res = c3_lattice_residual(sigma_emb_out)
        elapsed = perf_counter() - t0

        residual_hist.append(float(err))
        imp_residual_hist.append([float(x) for x in imp_res])
        mismatch_hist.append([float(x) for x in mismatches])
        bath_hist.append([float(x) for x in bath_errors])
        mu_hist.append(float(mu))
        elapsed_hist.append(float(elapsed))
        spread_hist.append(float(spread))
        c3_hist.append(float(c3res))

        if embed_opts.verbose:
            print(
                f"[cluster-ED+GW:sym] outer {it:02d}: residual={err:.3e} "
                f"(H={res_h:.3e}, emb={res_emb:.3e}, "
                f"imp=max {max(imp_res):.3e}), mu={mu:+.9f}, dt={elapsed:.1f}s\n"
                f"    Gimp/Gc={np.asarray(mismatches)}\n"
                f"    bath={np.asarray(bath_errors)}\n"
                f"    ori-C3 spread={spread:.3e}, projected C3 residual={c3res:.3e}",
                flush=True,
            )

        if err < float(embed_opts.tol):
            sigma_h = np.asarray(sigma_h_out)
            sigma_emb = np.asarray(sigma_emb_out)
            sigma_imp = np.asarray(sigma_imp_out)
            converged = True
        else:
            dyn = _pack_three_dynamic(sigma_emb, sigma_imp, grid)
            dyn_out = _pack_three_dynamic(
                sigma_emb_out, sigma_imp_out, grid
            )

            if method == "broyden":
                if broyden is None:
                    raise RuntimeError("internal Broyden mixer missing")
                if (
                    len(residual_hist) >= 2
                    and residual_hist[-1]
                    > float(embed_opts.broyden_reset_growth)
                    * residual_hist[-2]
                ):
                    broyden.clear()
                    if embed_opts.verbose:
                        print(
                            f"[cluster-ED+GW:sym] outer {it:02d}: "
                            "Broyden history reset after residual growth",
                            flush=True,
                        )
                x, hscale = fast._pack_broyden_state(sigma_h, dyn)
                xout, hscale_out = fast._pack_broyden_state(
                    sigma_h_out, dyn_out
                )
                if hscale != hscale_out:
                    raise RuntimeError("Broyden state scaling changed")
                q = x - xout
                xnext = broyden.propose(x, q)
                sigma_h_next, dyn_next = fast._unpack_broyden_state(
                    xnext, sigma_h.shape, dyn.shape, hscale
                )
                cap = float(embed_opts.broyden_step_cap)
            else:
                sigma_h_next, dyn_next = _mixed_self_energies(
                    sigma_h,
                    dyn,
                    sigma_h_out,
                    dyn_out,
                    mix_opts,
                    it,
                    mix_history,
                )
                cap = float(embed_opts.pulay_step_cap)

            raw_step = max(
                _maxabs(sigma_h_out - sigma_h),
                _maxabs(dyn_out - dyn),
            )
            mixed_step = max(
                _maxabs(sigma_h_next - sigma_h),
                _maxabs(dyn_next - dyn),
            )
            finite = (
                np.all(np.isfinite(sigma_h_next))
                and np.all(np.isfinite(dyn_next))
            )
            oversized = raw_step > 1.0e-14 and mixed_step > cap * raw_step

            if method == "broyden" and finite and oversized:
                scale = (cap * raw_step) / max(mixed_step, 1.0e-300)
                sigma_h_next = sigma_h + scale * (
                    sigma_h_next - sigma_h
                )
                dyn_next = dyn + scale * (dyn_next - dyn)
                if embed_opts.verbose:
                    print(
                        f"[cluster-ED+GW:sym] outer {it:02d}: "
                        f"Broyden trust clip proposed/raw="
                        f"{mixed_step/max(raw_step,1e-300):.2f} -> {cap:.2f}",
                        flush=True,
                    )
            elif (not finite) or oversized:
                fallbacks += 1
                mix_history.clear()
                if broyden is not None:
                    broyden.clear()
                a = float(embed_opts.mixing)
                sigma_h_next = sigma_h + a * (
                    sigma_h_out - sigma_h
                )
                dyn_next = dyn + a * (dyn_out - dyn)
                if embed_opts.verbose:
                    print(
                        f"[cluster-ED+GW:sym] outer {it:02d}: "
                        "mixer safeguard -> linear fallback",
                        flush=True,
                    )

            sigma_h = _project_parent_hartree(sigma_h_next)
            sigma_emb, sigma_imp = _unpack_three_dynamic(
                dyn_next,
                sigma_emb.shape,
                sigma_imp.shape,
                grid,
            )
            sigma_emb = _project_parent_dynamic(sigma_emb)
            for r in range(3):
                sigma_imp[r] = project_local_time_reversal(sigma_imp[r])

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

    if any(b is None for b in baths):
        raise RuntimeError("embedding loop did not build all three baths")

    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density = project_density_c3(
        np.real(np.diag(np.mean(rho_k, axis=(0, 1))))
    )

    return SymmetrizedClusterEDGWResult(
        G=np.asarray(G),
        W=np.asarray(W),
        P=np.asarray(P),
        Sigma_H=np.asarray(sigma_h),
        Sigma_emb=np.asarray(sigma_emb),
        Sigma_GW_lattice=np.asarray(sigma_gw_lattice),
        Sigma_GW_cluster_orientation=np.asarray(sigma_cgw_stack),
        Sigma_ED_cluster_orientation=np.asarray(sigma_imp),
        G_cluster_orientation=np.asarray(Gc_stack),
        G_impurity_orientation=np.asarray(Gimp_stack),
        impurity_static_shift_orientation=np.asarray(static_shifts),
        baths=tuple(baths),  # type: ignore[arg-type]
        mu=float(mu),
        density=np.asarray(density),
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
        impurity_mismatch=float(np.max(mismatch_hist[-1])),
        bath_fit_error=float(np.max(bath_hist[-1])),
        background=background,
        mixing_method=method,
        residual_history=np.asarray(residual_hist, dtype=float),
        impurity_residual_history=np.asarray(
            imp_residual_hist, dtype=float
        ),
        impurity_mismatch_history=np.asarray(mismatch_hist, dtype=float),
        bath_fit_history=np.asarray(bath_hist, dtype=float),
        mu_history=np.asarray(mu_hist, dtype=float),
        elapsed_history=np.asarray(elapsed_hist, dtype=float),
        orientation_spread_history=np.asarray(spread_hist, dtype=float),
        c3_residual_history=np.asarray(c3_hist, dtype=float),
        mixer_fallbacks=int(fallbacks),
    )


__all__ = [
    "SymmetrizedClusterEDGWResult",
    "project_fermion_time_reversal",
    "project_local_time_reversal",
    "solve_cluster_ed_gw_three_orientation_symmetric",
]
