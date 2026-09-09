#!/usr/bin/env python3
"""Twist-averaged finite-source current benchmark on a small Ruby torus.

For each boundary twist (theta1,theta2), solve the exact finite cluster and the
exactly matching finite-torus SC-GW problem for

    H(theta,h) = H0(theta) - h K_source + H_V.

Twist averaging reduces one-particle shell effects without enlarging the many-
body Hilbert space.  For an L1 x L2 torus, a twist shifts the allowed primitive
momenta to

    k_a = (m_a + theta_a/(2*pi)) / L_a,  m_a=0,...,L_a-1.

Thus an Ntheta1 x Ntheta2 uniform twist grid samples the same one-particle
momenta as an effective (L1*Ntheta1) x (L2*Ntheta2) primitive k mesh, although
for the interacting problem each twist is still a separate many-body boundary
condition and the final result is the twist average.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import benchmark_finite_source_current as bench
from diagnose_ed_warmstart_gw import _full_ed_seed, _relative_green_error, _current_from_gw
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import NSUB, RubyParameters, ruby_hoppings
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast

_SOURCE_TO_CHANNEL = {"same": "z_same", "opposite": "z_opposite"}


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--V", nargs="+", type=float, default=[1.0])
    p.add_argument("--source", nargs="+", choices=sorted(_SOURCE_TO_CHANNEL), default=["same"])
    p.add_argument("--h", nargs="+", type=float, default=[0.05, 0.02, 0.01])
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--ntheta1", type=int, default=4)
    p.add_argument("--ntheta2", type=int, default=8)
    p.add_argument("--twist-grid", choices=["midpoint", "gamma"], default="midpoint")
    p.add_argument("--methods", nargs="+", choices=["ed", "gw"], default=["ed", "gw"])
    p.add_argument("--gw-first-seed", choices=["zero", "ed-full"], default="ed-full")
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--gw-max-iter", type=int, default=500)
    p.add_argument("--gw-tol", type=float, default=2e-8)
    p.add_argument("--mixing", type=float, default=0.22)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--backend", choices=["fft", "direct"], default="direct")
    p.add_argument("--ed-discard-weight-tol", type=float, default=1e-12)
    p.add_argument("--ed-inverse-rcond", type=float, default=1e-12)
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/twist_averaged_finite_source"))
    return p.parse_args()


def _unique_descending_nonnegative(values):
    arr = np.asarray([float(x) for x in values], dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)) or np.any(arr < -1e-15):
        raise ValueError("--h must contain finite non-negative values")
    arr[np.abs(arr) < 1e-15] = 0.0
    return np.asarray(sorted(set(arr.tolist()), reverse=True), dtype=float)


def _twist_axis(n: int, kind: str) -> np.ndarray:
    if int(n) < 1:
        raise ValueError("twist-grid sizes must be positive")
    j = np.arange(int(n), dtype=float)
    if kind == "midpoint":
        # Symmetric in [-pi,pi), avoids the special PBC point.
        return 2.0 * np.pi * ((j + 0.5) / float(n) - 0.5)
    theta = 2.0 * np.pi * j / float(n)
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _twists(n1: int, n2: int, kind: str) -> np.ndarray:
    a = _twist_axis(n1, kind)
    b = _twist_axis(n2, kind)
    t1, t2 = np.meshgrid(a, b, indexing="ij")
    return np.stack([t1.ravel(), t2.ravel()], axis=-1)


def _site(L1: int, L2: int, r1: int, r2: int, a: int) -> int:
    return NSUB * ((int(r1) % int(L1)) * int(L2) + (int(r2) % int(L2))) + int(a)


def build_twisted_cluster_h0(L1: int, L2: int, params: RubyParameters, theta) -> np.ndarray:
    """Real-space finite-torus h0 with c_{r+L_a}=exp(i theta_a)c_r."""
    theta = np.asarray(theta, dtype=float).reshape(2)
    nsites = NSUB * int(L1) * int(L2)
    h = np.zeros((nsites, nsites), dtype=complex)
    p0 = RubyParameters(ti=params.ti, t1=params.t1, t2=params.t2, V=0.0)
    for r1 in range(int(L1)):
        for r2 in range(int(L2)):
            for i, j, R, amp in ruby_hoppings(p0):
                raw1 = r1 + int(R[0])
                raw2 = r2 + int(R[1])
                wrap1 = raw1 // int(L1)
                wrap2 = raw2 // int(L2)
                I = _site(L1, L2, r1, r2, int(i))
                J = _site(L1, L2, raw1, raw2, int(j))
                phase = np.exp(1j * (wrap1 * theta[0] + wrap2 * theta[1]))
                h[I, J] += complex(amp) * phase
    h = 0.5 * (h + h.conj().T)
    if np.max(np.abs(h - h.conj().T)) > 1e-11:
        raise RuntimeError("twisted one-body Hamiltonian is not Hermitian")
    return h


def _nanmean_std(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.nan, np.nan, 0
    x = arr[finite]
    return float(np.mean(x)), float(np.std(x)), int(x.size)


def main():
    args = _args()
    if args.T <= 0 or 6 * args.L1 * args.L2 > 16:
        raise ValueError("need T>0 and at most 16 ED sites")
    hvalues = _unique_descending_nonnegative(args.h)
    Vvalues = np.asarray(args.V, dtype=float)
    sources = list(dict.fromkeys(args.source))
    methods = set(args.methods)
    twists = _twists(args.ntheta1, args.ntheta2, args.twist_grid)
    ntwist = len(twists)
    ncell = int(args.L1) * int(args.L2)
    target = float(args.filling) * ncell

    params0 = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    base_exact = ExactSmallRubyThermal(args.L1, args.L2, params0)
    Vunit = np.asarray(base_exact.Vunit, dtype=complex)
    K_by_source = {
        src: np.asarray(base_exact.pseudospin_operator(_SOURCE_TO_CHANNEL[src], (0.0, 0.0)), dtype=complex)
        for src in sources
    }
    # Sanity check: theta=0 must reproduce the original PBC finite torus.
    h0_pbc = build_twisted_cluster_h0(args.L1, args.L2, params0, (0.0, 0.0))
    pbc_err = float(np.max(np.abs(h0_pbc - np.asarray(base_exact.h0, dtype=complex))))
    if pbc_err > 1e-11:
        raise RuntimeError(f"twist builder disagrees with PBC h0: {pbc_err:.3e}")

    grid = MatsubaraGrid(nk1=1, nk2=1, nw=args.nw, nOmega=args.nomega, T=args.T)
    gw_opts = GWOptions(
        target_filling=target,
        max_iter=args.gw_max_iter,
        tol=args.gw_tol,
        mixing=args.mixing,
        mixing_method=args.mixing_method,
        verbose=args.verbose,
        momentum_backend=args.backend,
    )

    shape = (len(sources), len(Vvalues), len(hvalues), ntwist)
    J_ed = np.full(shape, np.nan)
    mu_ed = np.full(shape, np.nan)
    J_gw = np.full(shape, np.nan)
    mu_gw = np.full(shape, np.nan)
    Gerr = np.full(shape, np.nan)
    gw_converged = np.zeros(shape, dtype=bool)
    gw_iterations = np.zeros(shape, dtype=int)
    gw_residual = np.full(shape, np.nan)

    print("=== twist-averaged finite-source benchmark ===")
    print(f"cluster={args.L1}x{args.L2}, T={args.T:g}, filling={args.filling:g}, target={target:g}")
    print(f"twist grid={args.ntheta1}x{args.ntheta2} ({args.twist_grid}), points={ntwist}")
    print(
        "effective primitive one-body k mesh="
        f"{args.L1*args.ntheta1}x{args.L2*args.ntheta2}"
    )
    print("GW still uses nk=1 in the full finite-torus orbital basis at each twist.")

    for isrc, src in enumerate(sources):
        K = K_by_source[src]
        for iv, V in enumerate(Vvalues):
            Vq = (float(V) * Vunit)[None, None]
            print(f"\n### source={src}, V={V:g} ###")
            for itw, theta in enumerate(twists):
                base_h0 = build_twisted_cluster_h0(args.L1, args.L2, params0, theta)
                continuation = None
                for ih, h in enumerate(hvalues):
                    hmat = 0.5 * ((base_h0 - float(h) * K) + (base_h0 - float(h) * K).conj().T)
                    h0h = hmat[None, None]
                    exact = ExactSmallRubyThermal(args.L1, args.L2, params0)
                    exact.h0 = hmat
                    exact.diagonalize(float(V))
                    mu_e = exact.solve_mu(target, args.T)
                    mu_ed[isrc, iv, ih, itw] = mu_e
                    J_ed[isrc, iv, ih, itw] = float(bench._exact_onebody_expectation(exact, K, mu_e, args.T).real)

                    if "gw" in methods:
                        ed_seed, G_ed, _, _, _ = _full_ed_seed(
                            exact, mu_e, args.T, Vq, h0h, grid,
                            args.ed_discard_weight_tol, args.ed_inverse_rcond,
                        )
                        if ih == 0:
                            initial = ed_seed if args.gw_first_seed == "ed-full" else None
                        else:
                            initial = continuation if continuation is not None else ed_seed
                        gw = solve_matrix_gw_fast(h0h, Vq, grid, opts=gw_opts, initial=initial)
                        gw_converged[isrc, iv, ih, itw] = bool(gw.converged)
                        gw_iterations[isrc, iv, ih, itw] = int(gw.iterations)
                        gw_residual[isrc, iv, ih, itw] = float(gw.final_error)
                        mu_gw[isrc, iv, ih, itw] = float(gw.mu)
                        J_gw[isrc, iv, ih, itw] = _current_from_gw(gw, K, Vq, grid, h0h, args.backend)
                        Gerr[isrc, iv, ih, itw] = _relative_green_error(gw.G, G_ed)
                        continuation = gw if gw.converged else None
                        if (not gw.converged) and (not args.allow_unconverged):
                            raise RuntimeError(
                                f"GW failed at source={src}, V={V:g}, h={h:g}, "
                                f"theta/2pi=({theta[0]/(2*np.pi):+.6f},{theta[1]/(2*np.pi):+.6f}), "
                                f"residual={gw.final_error:.3e}"
                            )
                if (itw + 1) % max(1, min(8, ntwist)) == 0 or itw + 1 == ntwist:
                    print(f"  completed twists {itw+1}/{ntwist}")

            for ih, h in enumerate(hvalues):
                ed_mean, ed_std, _ = _nanmean_std(J_ed[isrc, iv, ih])
                if "gw" in methods:
                    valid = gw_converged[isrc, iv, ih]
                    gwvals = np.where(valid, J_gw[isrc, iv, ih], np.nan)
                    gvals = np.where(valid, Gerr[isrc, iv, ih], np.nan)
                    gw_mean, gw_std, ngw = _nanmean_std(gwvals)
                    ge_mean, ge_std, _ = _nanmean_std(gvals)
                    print(
                        f"  h={h:g}: ED={ed_mean:+.9f} (twist std={ed_std:.3e}); "
                        f"GW={gw_mean:+.9f} (std={gw_std:.3e}, conv={ngw}/{ntwist}); "
                        f"Gerr={ge_mean:.3e} (std={ge_std:.3e})"
                    )
                else:
                    print(f"  h={h:g}: ED={ed_mean:+.9f} (twist std={ed_std:.3e})")

    J_ed_avg = np.nanmean(J_ed, axis=-1)
    J_ed_std = np.nanstd(J_ed, axis=-1)
    valid_gw = np.where(gw_converged, J_gw, np.nan)
    valid_ge = np.where(gw_converged, Gerr, np.nan)
    J_gw_avg = np.nanmean(valid_gw, axis=-1)
    J_gw_std = np.nanstd(valid_gw, axis=-1)
    Gerr_avg = np.nanmean(valid_ge, axis=-1)
    Gerr_std = np.nanstd(valid_ge, axis=-1)
    gw_converged_fraction = np.mean(gw_converged, axis=-1)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "twist_averaged_finite_source.npz",
        L1=args.L1, L2=args.L2, V=Vvalues, h=hvalues, sources=np.asarray(sources),
        filling=args.filling, T=args.T, ntheta1=args.ntheta1, ntheta2=args.ntheta2,
        twist_grid=args.twist_grid, twists=twists,
        effective_kmesh=np.asarray([args.L1*args.ntheta1, args.L2*args.ntheta2]),
        J_ed_twist=J_ed, mu_ed_twist=mu_ed,
        J_gw_twist=J_gw, mu_gw_twist=mu_gw, Gerr_twist=Gerr,
        gw_converged=gw_converged, gw_iterations=gw_iterations, gw_residual=gw_residual,
        J_ed_avg=J_ed_avg, J_ed_twist_std=J_ed_std,
        J_gw_avg=J_gw_avg, J_gw_twist_std=J_gw_std,
        Gerr_avg=Gerr_avg, Gerr_twist_std=Gerr_std,
        gw_converged_fraction=gw_converged_fraction,
    )

    with (args.out / "twist_average.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source","V","h","J_ed_avg","J_ed_twist_std","J_gw_avg","J_gw_twist_std","Gerr_avg","Gerr_twist_std","gw_converged_fraction"])
        for isrc, src in enumerate(sources):
            for iv, V in enumerate(Vvalues):
                for ih, h in enumerate(hvalues):
                    w.writerow([
                        src, float(V), float(h),
                        float(J_ed_avg[isrc,iv,ih]), float(J_ed_std[isrc,iv,ih]),
                        float(J_gw_avg[isrc,iv,ih]), float(J_gw_std[isrc,iv,ih]),
                        float(Gerr_avg[isrc,iv,ih]), float(Gerr_std[isrc,iv,ih]),
                        float(gw_converged_fraction[isrc,iv,ih]),
                    ])

    with (args.out / "twist_points.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source","V","h","twist_index","theta1","theta2","phi1","phi2","J_ed","J_gw","Gerr","gw_converged","gw_iterations","gw_residual"])
        for isrc, src in enumerate(sources):
            for iv, V in enumerate(Vvalues):
                for ih, h in enumerate(hvalues):
                    for itw, theta in enumerate(twists):
                        w.writerow([
                            src, float(V), float(h), itw, float(theta[0]), float(theta[1]),
                            float(theta[0]/(2*np.pi)), float(theta[1]/(2*np.pi)),
                            float(J_ed[isrc,iv,ih,itw]), float(J_gw[isrc,iv,ih,itw]),
                            float(Gerr[isrc,iv,ih,itw]), bool(gw_converged[isrc,iv,ih,itw]),
                            int(gw_iterations[isrc,iv,ih,itw]), float(gw_residual[isrc,iv,ih,itw]),
                        ])
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
