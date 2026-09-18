"""C3-symmetric three-orientation physical-pair cluster ED+GW embedding.

All three C3-related six-site cuts are evaluated on the same lattice Green
function at every outer iteration.  Each local cluster correction is mapped
back to one common primitive-cell gauge before the three corrections are
averaged,

    Sigma_emb(k) = Sigma_GW^lat(k)
                 + (1/3) sum_r D_r(k)^dagger
                     [Sigma_ED^r - Sigma_GW,cl^r] D_r(k).

This is deliberately different from solving three independent oriented
embeddings and averaging their final states: the nonlinear map itself remains
in the C3-symmetric subspace.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .c3_constraint import c3_lattice_residual, project_density_c3, project_lattice_c3
from .cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    build_impurity_one_body,
    cluster_gw_self_energy,
    cluster_interaction_matrix,
    fit_finite_bath,
    split_static_hybridization,
)
from .cluster_ed_gw_fast import (
    ClusterEDGWFastOptions,
    _LimitedMemoryBroyden,
    _check_embed_mixing_method,
    _maxabs,
    _pack_broyden_state,
    _relative_error,
    _unpack_broyden_state,
)
from .cluster_orientation import build_oriented_lattice_fields, transform_between_orientations
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
from .supercell_gw_split import compute_sigma_gw_split_matrix, one_body_density_matrix_tail


@dataclass
class ThreeOrientationEDGWResult:
    G: np.ndarray
    W: np.ndarray
    P: np.ndarray
    Sigma_H: np.ndarray
    Sigma_emb: np.ndarray
    Sigma_GW_lattice: np.ndarray
    Sigma_GW_cluster_by_orientation: np.ndarray
    Sigma_ED_cluster_by_orientation: np.ndarray
    G_cluster_by_orientation: np.ndarray
    G_impurity_by_orientation: np.ndarray
    impurity_static_shift_by_orientation: np.ndarray
    baths: tuple[BathParameters, BathParameters, BathParameters]
    mu: float
    density: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    impurity_mismatch_by_orientation: np.ndarray
    bath_fit_error_by_orientation: np.ndarray
    background: GWResult
    mixing_method: str
    residual_history: np.ndarray
    impurity_residual_history: np.ndarray
    impurity_mismatch_history: np.ndarray
    bath_fit_history: np.ndarray
    mu_history: np.ndarray
    elapsed_history: np.ndarray
    c3_residual_history: np.ndarray
    mixer_fallbacks: int


def _local_to_common(field: np.ndarray, orientation: int, grid: MatsubaraGrid) -> np.ndarray:
    """Map a k-independent local matrix in orientation r to common gauge 0."""
    x = np.asarray(field, dtype=complex)
    lead = x.shape[:-2]
    lat = np.broadcast_to(
        x.reshape(lead + (1, 1, NSUB, NSUB)),
        lead + (grid.nk1, grid.nk2, NSUB, NSUB),
    ).copy()
    return transform_between_orientations(lat, int(orientation), 0)


def _common_to_orientation(field: np.ndarray, orientation: int) -> np.ndarray:
    return transform_between_orientations(np.asarray(field, dtype=complex), 0, int(orientation))


def _average_local_fields_in_common(
    fields: list[np.ndarray] | tuple[np.ndarray, ...],
    grid: MatsubaraGrid,
) -> np.ndarray:
    return sum(_local_to_common(f, r, grid) for r, f in enumerate(fields)) / 3.0


def _pack_three_dynamic(
    sigma_emb: np.ndarray,
    sigma_imps: list[np.ndarray] | tuple[np.ndarray, ...],
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Pack common weak lattice piece plus the three local impurity pieces."""
    avg_imp = _average_local_fields_in_common(sigma_imps, grid)
    weak = np.asarray(sigma_emb, dtype=complex) - avg_imp
    scale = np.sqrt(float(max(grid.nk, 1)) / 3.0)
    chunks = [weak.ravel()]
    chunks.extend(scale * np.asarray(s, dtype=complex).ravel() for s in sigma_imps)
    return np.concatenate(chunks)


def _unpack_three_dynamic(
    packed: np.ndarray,
    sigma_emb_shape: tuple[int, ...],
    sigma_imp_shape: tuple[int, ...],
    grid: MatsubaraGrid,
) -> tuple[np.ndarray, list[np.ndarray]]:
    flat = np.asarray(packed, dtype=complex).reshape(-1)
    nemb = int(np.prod(sigma_emb_shape))
    nimp = int(np.prod(sigma_imp_shape))
    weak = flat[:nemb].reshape(sigma_emb_shape)
    scale = np.sqrt(float(max(grid.nk, 1)) / 3.0)
    imps = []
    pos = nemb
    for _ in range(3):
        imps.append((flat[pos:pos+nimp] / scale).reshape(sigma_imp_shape))
        pos += nimp
    if pos != flat.size:
        raise ValueError("three-orientation dynamic state size mismatch")
    emb = weak + _average_local_fields_in_common(imps, grid)
    return np.asarray(emb), imps


def _project_common_dynamic(field: np.ndarray) -> np.ndarray:
    return np.asarray(project_lattice_c3(np.asarray(field, dtype=complex)))


def solve_three_orientation_ed_gw(
    h0: np.ndarray,
    Vq: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
    *,
    gw_opts: GWOptions = GWOptions(),
    embed_opts: ClusterEDGWFastOptions = ClusterEDGWFastOptions(),
    background: GWResult | None = None,
) -> ThreeOrientationEDGWResult:
    """Solve the simultaneous C3-symmetric three-cut physical-pair embedding."""
    if grid.nk1 != grid.nk2:
        raise ValueError("three-orientation C3 projection requires a square k mesh")
    method = _check_embed_mixing_method(embed_opts.mixing_method)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    if h0.shape != (grid.nk1, grid.nk2, NSUB, NSUB) or Vq.shape != h0.shape:
        raise ValueError("expected primitive-cell h0/Vq arrays of shape (nk1,nk2,6,6)")

    if background is None:
        if embed_opts.verbose:
            print("[3ori-ED+GW] solving one common-gauge lattice SCGW background ...", flush=True)
        background = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    if not background.converged:
        raise RuntimeError(
            f"common SCGW background is not converged: residual={background.final_error:.3e}"
        )

    # One common lattice state.  Project tiny numerical C3 drift before entering
    # the constrained map and re-solve the fixed-filling Dyson equation.
    density0 = project_density_c3(np.asarray(background.density, dtype=float))
    sigma_h = hartree_self_energy_matrix(density0, Vq[0, 0])
    sigma_emb = _project_common_dynamic(np.asarray(background.Sigma_GW, dtype=complex))
    mu = float(background.mu)
    if gw_opts.target_filling is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_emb)
    else:
        mu, G, _, _ = _solve_mu_matrix_fast(
            h0, sigma_h, sigma_emb, grid, float(gw_opts.target_filling),
            mu, float(gw_opts.mu_tol), int(gw_opts.mu_max_iter)
        )

    oriented = [build_oriented_lattice_fields(h0, Vq, r) for r in range(3)]
    h_clusters = []
    interactions = []
    V_clusters = []
    for r in range(3):
        hr = np.asarray(oriented[r][0], dtype=complex)
        hc = np.mean(hr, axis=(0, 1))
        hc = 0.5 * (hc + hc.conj().T)
        h_clusters.append(hc)
        ir = physical_pair_cluster_interactions(params, r)
        interactions.append(ir)
        V_clusters.append(cluster_interaction_matrix(ir, NSUB))

    baths: list[BathParameters | None] = [None, None, None]
    sigma_imps: list[np.ndarray] = []
    for r in range(3):
        Gr = _common_to_orientation(G, r)
        rho_r = one_body_density_matrix_tail(
            Gr, grid, oriented[r][0], mu, sigma_h
        )
        Gc_r = np.mean(Gr, axis=(1, 2))
        rho_c_r = np.mean(rho_r, axis=(0, 1))
        scgw_r, _, _ = cluster_gw_self_energy(Gc_r, rho_c_r, V_clusters[r], grid)
        sigma_imps.append(np.asarray(scgw_r).copy())

    mix_opts = GWOptions(
        mixing=float(embed_opts.mixing),
        mixing_method=("pulay" if method == "broyden" else method),
        pulay_history=int(embed_opts.pulay_history),
        pulay_start=int(embed_opts.pulay_start),
        pulay_regularization=float(embed_opts.pulay_regularization),
    )
    mix_history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    broyden = (
        _LimitedMemoryBroyden(
            alpha=float(embed_opts.mixing),
            history=int(embed_opts.broyden_history),
            regularization=float(embed_opts.broyden_regularization),
        )
        if method == "broyden" else None
    )
    fallbacks = 0

    residual_hist: list[float] = []
    imp_residual_hist: list[list[float]] = []
    mismatch_hist: list[list[float]] = []
    bath_hist: list[list[float]] = []
    mu_hist: list[float] = []
    elapsed_hist: list[float] = []
    c3_hist: list[float] = []

    W = np.asarray(background.W)
    P = np.asarray(background.P)
    sigma_gw_lattice = np.asarray(background.Sigma_GW)
    sigma_cgws = [np.zeros_like(sigma_imps[0]) for _ in range(3)]
    Gimps = [np.zeros_like(sigma_imps[0]) for _ in range(3)]
    Gcs = [np.zeros_like(sigma_imps[0]) for _ in range(3)]
    static_shifts = [np.zeros((NSUB, NSUB), dtype=complex) for _ in range(3)]
    mismatches = [np.inf, np.inf, np.inf]
    bath_errors = [np.inf, np.inf, np.inf]
    density = np.asarray(density0)
    converged = False
    err = np.inf
    it = 0

    for it in range(1, int(embed_opts.max_iter) + 1):
        t0 = perf_counter()
        if embed_opts.verbose:
            print(f"[3ori-ED+GW] outer {it:02d}: common lattice map + 3 impurity cuts", flush=True)

        rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
        rho_c0 = np.mean(rho_k, axis=(0, 1))
        density = project_density_c3(np.real(np.diag(rho_c0)))
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])

        P = compute_polarization_matrix(G, grid, backend=str(gw_opts.momentum_backend))
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_lattice = compute_sigma_gw_split_matrix(
            G, W, Vq, grid, h0, mu, sigma_h,
            backend=str(gw_opts.momentum_backend),
        )
        sigma_gw_lattice = _project_common_dynamic(sigma_gw_lattice)

        sigma_imp_outs: list[np.ndarray] = []
        corrections_common: list[np.ndarray] = []
        imp_res = []
        mismatches = []
        bath_errors = []

        for r in range(3):
            Gr = _common_to_orientation(G, r)
            rho_r = one_body_density_matrix_tail(
                Gr, grid, oriented[r][0], mu, sigma_h
            )
            Gc_r = np.mean(Gr, axis=(1, 2))
            rho_c_r = np.mean(rho_r, axis=(0, 1))
            Gcs[r] = np.asarray(Gc_r)

            sigma_cgw_r, _, _ = cluster_gw_self_energy(
                Gc_r, rho_c_r, V_clusters[r], grid
            )
            sigma_cgws[r] = np.asarray(sigma_cgw_r)

            g0_inv = np.linalg.inv(Gc_r) + sigma_imps[r]
            eye = np.eye(NSUB, dtype=complex)
            delta_raw = (
                (1j * grid.omega[:, None, None] + float(mu)) * eye[None]
                - h_clusters[r][None]
                - g0_inv
            )
            static_shift, delta_target = split_static_hybridization(delta_raw, grid.omega)
            static_shifts[r] = np.asarray(static_shift)
            h_imp = h_clusters[r] + static_shift

            if embed_opts.verbose:
                print(f"[3ori-ED+GW] outer {it:02d}: ori{r} fit bath + ED", flush=True)
            bath_r = fit_finite_bath(
                delta_target, grid.omega, mu,
                nbath=int(embed_opts.nbath),
                nfit=int(embed_opts.bath_fit_nfreq),
                max_nfev=int(embed_opts.bath_fit_max_nfev),
                energy_window=float(embed_opts.bath_energy_window),
                coupling_bound=float(embed_opts.bath_coupling_bound),
                xtol=float(embed_opts.bath_fit_xtol),
                initial=baths[r],
                metric=str(embed_opts.bath_fit_metric),
                one_body=h_imp,
            )
            baths[r] = bath_r
            himp = build_impurity_one_body(h_imp, bath_r)
            impurity = FiniteBathImpurityED(
                himp, interactions[r], correlated_orbitals=tuple(range(NSUB))
            )
            impurity.diagonalize()
            Gimp_r, _ = impurity.green_iomega(
                1j * grid.omega, mu, grid.T,
                orbitals=tuple(range(NSUB)),
                discard_weight_tol=float(embed_opts.discard_weight_tol),
            )
            Gimps[r] = np.asarray(Gimp_r)
            delta_fit = bath_hybridization(
                grid.omega, mu, bath_r.energies, bath_r.couplings
            )
            g0_fit_inv = (
                (1j * grid.omega[:, None, None] + float(mu)) * eye[None]
                - h_imp[None] - delta_fit
            )
            sigma_ed_raw = g0_fit_inv - np.linalg.inv(Gimp_r)
            beta = float(embed_opts.impurity_mixing)
            sigma_imp_out_r = sigma_imps[r] + beta * (sigma_ed_raw - sigma_imps[r])
            sigma_imp_outs.append(np.asarray(sigma_imp_out_r))

            correction_r = sigma_imp_out_r - sigma_cgw_r
            corrections_common.append(_local_to_common(correction_r, r, grid))
            imp_res.append(_maxabs(sigma_imp_out_r - sigma_imps[r]))
            mismatches.append(_relative_error(Gimp_r, Gc_r))
            bath_errors.append(float(bath_r.fit_error))

        correction_sym = sum(corrections_common) / 3.0
        sigma_emb_out = _project_common_dynamic(sigma_gw_lattice + correction_sym)

        res_h = _maxabs(sigma_h_out - sigma_h)
        res_emb = _maxabs(sigma_emb_out - sigma_emb)
        err = max([res_h, res_emb] + imp_res)
        c3res = c3_lattice_residual(sigma_emb_out)
        elapsed = perf_counter() - t0

        residual_hist.append(float(err))
        imp_residual_hist.append([float(x) for x in imp_res])
        mismatch_hist.append([float(x) for x in mismatches])
        bath_hist.append([float(x) for x in bath_errors])
        mu_hist.append(float(mu))
        elapsed_hist.append(float(elapsed))
        c3_hist.append(float(c3res))

        if embed_opts.verbose:
            print(
                f"[3ori-ED+GW] outer {it:02d}: residual={err:.3e} "
                f"(H={res_h:.3e}, emb={res_emb:.3e}, "
                f"imp={max(imp_res):.3e}), mu={mu:+.9f}, "
                f"C3={c3res:.2e}, dt={elapsed:.1f}s\n"
                f"    mismatch ori0/1/2="
                f"{mismatches[0]:.3e}/{mismatches[1]:.3e}/{mismatches[2]:.3e}; "
                f"bath={bath_errors[0]:.3e}/{bath_errors[1]:.3e}/{bath_errors[2]:.3e}",
                flush=True,
            )

        if err < float(embed_opts.tol):
            sigma_h = np.asarray(sigma_h_out)
            sigma_emb = np.asarray(sigma_emb_out)
            sigma_imps = [np.asarray(x) for x in sigma_imp_outs]
            converged = True
        else:
            dyn = _pack_three_dynamic(sigma_emb, sigma_imps, grid)
            dyn_out = _pack_three_dynamic(sigma_emb_out, sigma_imp_outs, grid)

            if method == "broyden":
                if broyden is None:
                    raise RuntimeError("Broyden mixer not initialized")
                if (
                    len(residual_hist) >= 2
                    and residual_hist[-1]
                    > float(embed_opts.broyden_reset_growth) * residual_hist[-2]
                ):
                    broyden.clear()
                x, hscale = _pack_broyden_state(sigma_h, dyn)
                xout, hscale_out = _pack_broyden_state(sigma_h_out, dyn_out)
                if hscale != hscale_out:
                    raise RuntimeError("Broyden state scale changed")
                q = x - xout
                xnext = broyden.propose(x, q)
                sigma_h_next, dyn_next = _unpack_broyden_state(
                    xnext, sigma_h.shape, dyn.shape, hscale
                )
                cap = float(embed_opts.broyden_step_cap)
            else:
                sigma_h_next, dyn_next = _mixed_self_energies(
                    sigma_h, dyn, sigma_h_out, dyn_out,
                    mix_opts, it, mix_history,
                )
                cap = float(embed_opts.pulay_step_cap)

            raw_step = max(_maxabs(sigma_h_out-sigma_h), _maxabs(dyn_out-dyn))
            mixed_step = max(_maxabs(sigma_h_next-sigma_h), _maxabs(dyn_next-dyn))
            finite = np.all(np.isfinite(sigma_h_next)) and np.all(np.isfinite(dyn_next))
            oversized = raw_step > 1e-14 and mixed_step > cap * raw_step

            if method == "broyden" and finite and oversized:
                scale = (cap * raw_step) / max(mixed_step, 1e-300)
                sigma_h_next = sigma_h + scale * (sigma_h_next - sigma_h)
                dyn_next = dyn + scale * (dyn_next - dyn)
                if embed_opts.verbose:
                    print(
                        f"[3ori-ED+GW] outer {it:02d}: Broyden trust clip "
                        f"(proposed/raw={mixed_step/max(raw_step,1e-300):.2f}, "
                        f"accepted/raw={cap:.2f}, rank={broyden.rank})",
                        flush=True,
                    )
            elif (not finite) or oversized:
                fallbacks += 1
                mix_history.clear()
                if broyden is not None:
                    broyden.clear()
                a = float(embed_opts.mixing)
                sigma_h_next = sigma_h + a * (sigma_h_out - sigma_h)
                dyn_next = dyn + a * (dyn_out - dyn)

            sigma_h = np.asarray(sigma_h_next)
            sigma_emb, sigma_imps = _unpack_three_dynamic(
                dyn_next, sigma_emb.shape, sigma_imps[0].shape, grid
            )
            # Keep the common lattice state exactly in the C3-symmetric subspace.
            sigma_emb = _project_common_dynamic(sigma_emb)
            # sigma_h is diagonal for density interactions; averaging the three
            # diagonal entries within each triangle is the C3 projection.
            sigma_h = np.diag(
                np.r_[np.repeat(np.mean(np.diag(sigma_h).real[:3]), 3),
                      np.repeat(np.mean(np.diag(sigma_h).real[3:]), 3)]
            ).astype(complex)

        if gw_opts.target_filling is None:
            G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_emb)
        else:
            mu, G, _, _ = _solve_mu_matrix_fast(
                h0, sigma_h, sigma_emb, grid, float(gw_opts.target_filling),
                mu, float(gw_opts.mu_tol), int(gw_opts.mu_max_iter)
            )
        # Do not project G independently: with C3-projected self-energies the
        # Dyson result is already covariant up to roundoff, and keeping it
        # untouched preserves the exact Dyson identity.
        if converged:
            break

    if any(b is None for b in baths):
        raise RuntimeError("three-orientation embedding loop did not execute")

    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    density = project_density_c3(np.real(np.diag(np.mean(rho_k, axis=(0, 1)))))

    return ThreeOrientationEDGWResult(
        G=np.asarray(G),
        W=np.asarray(W),
        P=np.asarray(P),
        Sigma_H=np.asarray(sigma_h),
        Sigma_emb=np.asarray(sigma_emb),
        Sigma_GW_lattice=np.asarray(sigma_gw_lattice),
        Sigma_GW_cluster_by_orientation=np.asarray(sigma_cgws),
        Sigma_ED_cluster_by_orientation=np.asarray(sigma_imps),
        G_cluster_by_orientation=np.asarray(Gcs),
        G_impurity_by_orientation=np.asarray(Gimps),
        impurity_static_shift_by_orientation=np.asarray(static_shifts),
        baths=(baths[0], baths[1], baths[2]),  # type: ignore[arg-type]
        mu=float(mu),
        density=np.asarray(density),
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
        impurity_mismatch_by_orientation=np.asarray(mismatches, dtype=float),
        bath_fit_error_by_orientation=np.asarray(bath_errors, dtype=float),
        background=background,
        mixing_method=method,
        residual_history=np.asarray(residual_hist, dtype=float),
        impurity_residual_history=np.asarray(imp_residual_hist, dtype=float),
        impurity_mismatch_history=np.asarray(mismatch_hist, dtype=float),
        bath_fit_history=np.asarray(bath_hist, dtype=float),
        mu_history=np.asarray(mu_hist, dtype=float),
        elapsed_history=np.asarray(elapsed_hist, dtype=float),
        c3_residual_history=np.asarray(c3_hist, dtype=float),
        mixer_fallbacks=int(fallbacks),
    )


__all__ = ["ThreeOrientationEDGWResult", "solve_three_orientation_ed_gw"]
