#!/usr/bin/env python3
"""Diagnose SC-GW basins using exact-ED warm starts on the same finite Ruby torus.

For every requested (source,V,h), solve the same finite-torus SC-GW problem from:
zero, h-continuation, an exact-ED static Hartree+Fock seed, and a full exact
Dyson self-energy seed reconstructed from the Lehmann G_ED(iw).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import benchmark_finite_source_current as bench
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_gw import dyson_from_sigma_matrix
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.supercell_gw_split import compute_sigma_gw_split_components

_SOURCE_TO_CHANNEL = {"same": "z_same", "opposite": "z_opposite"}
_SEED_MODES = ("zero", "continuation", "ed-static", "ed-full")


@dataclass
class _GWSeed:
    G: np.ndarray
    Sigma_H: np.ndarray
    Sigma_GW: np.ndarray
    mu: float


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--V", nargs="+", type=float, default=[1.0])
    p.add_argument("--source", nargs="+", choices=sorted(_SOURCE_TO_CHANNEL), default=["same"])
    p.add_argument("--h", nargs="+", type=float, default=[0.05,0.02,0.01,0.005,0.002,0.001,0.0])
    p.add_argument("--seed", nargs="+", choices=_SEED_MODES, default=list(_SEED_MODES))
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--gw-max-iter", type=int, default=500)
    p.add_argument("--gw-tol", type=float, default=2e-8)
    p.add_argument("--mixing", type=float, default=0.22)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--pulay-history", type=int, default=6)
    p.add_argument("--pulay-start", type=int, default=3)
    p.add_argument("--pulay-regularization", type=float, default=1e-10)
    p.add_argument("--backend", choices=["fft", "direct"], default="direct")
    p.add_argument("--ed-discard-weight-tol", type=float, default=1e-12)
    p.add_argument("--ed-inverse-rcond", type=float, default=1e-12)
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/ed_warmstart_gw_branches"))
    return p.parse_args()


def _unique_descending_nonnegative(values):
    arr = np.asarray([float(x) for x in values], dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)) or np.any(arr < -1e-15):
        raise ValueError("--h values must be finite, non-negative, and non-empty")
    arr[np.abs(arr) < 1e-15] = 0.0
    return np.asarray(sorted(set(arr.tolist()), reverse=True), dtype=float)


def _exact_density(exact, mu, T):
    probs, _, _ = exact._normalized_probabilities(float(mu), float(T))
    density = np.zeros(exact.n_sites, dtype=float)
    for sec, p in zip(exact.sectors, probs):
        if len(p) == 0 or not np.any(p > 0.0):
            continue
        basis_weight = (np.abs(sec.eigenvectors) ** 2) @ np.asarray(p, dtype=float)
        for site in range(exact.n_sites):
            occ = ((sec.basis >> np.uint64(site)) & np.uint64(1)).astype(float)
            density[site] += float(np.dot(basis_weight, occ))
    return density


def _exact_interaction_bond_rho(exact, mu, T):
    rho = np.zeros((exact.n_sites, exact.n_sites), dtype=complex)
    rho[np.diag_indices(exact.n_sites)] = _exact_density(exact, mu, T)
    for i, j in exact.interaction_pairs:
        K = np.zeros_like(rho)
        K[j, i] = 1.0  # <c_j^dag c_i> = rho[i,j]
        value = bench._exact_onebody_expectation(exact, K, mu, T)
        rho[i, j] = value
        rho[j, i] = np.conj(value)
    return 0.5 * (rho + rho.conj().T)


def _static_ed_seed(exact, mu_ed, T, Vq, h0h, grid):
    rho = _exact_interaction_bond_rho(exact, mu_ed, T)
    density = np.real(np.diag(rho))
    V0 = np.asarray(Vq[0, 0], dtype=complex)
    sigma_h = np.zeros_like(V0)
    sigma_h[np.diag_indices_from(sigma_h)] = V0 @ density
    sigma_f = -rho * V0.T
    sigma_f = 0.5 * (sigma_f + sigma_f.conj().T)
    sigma_gw = np.broadcast_to(
        sigma_f[None, None, None, :, :],
        (grid.nf, grid.nk1, grid.nk2, exact.n_sites, exact.n_sites),
    ).copy()
    Gseed = dyson_from_sigma_matrix(h0h, grid, mu_ed, sigma_h, sigma_gw)
    return _GWSeed(Gseed, sigma_h, sigma_gw, float(mu_ed))


def _invert_exact_green(G, rcond):
    Garr = np.asarray(G, dtype=complex)
    out = np.empty_like(Garr)
    worst_cond = 0.0
    fallback_count = 0
    for n, mat in enumerate(Garr):
        s = np.linalg.svd(mat, compute_uv=False)
        smax, smin = float(np.max(s)), float(np.min(s))
        cond = np.inf if smin <= 0.0 else smax / smin
        worst_cond = max(worst_cond, cond)
        if (not np.isfinite(cond)) or smin <= float(rcond) * max(smax, 1e-300):
            out[n] = np.linalg.pinv(mat, rcond=float(rcond))
            fallback_count += 1
        else:
            out[n] = np.linalg.inv(mat)
    return out, float(worst_cond), int(fallback_count)


def _full_ed_seed(exact, mu_ed, T, Vq, h0h, grid, discard_weight_tol, inverse_rcond):
    G_ed, selection = exact.green_iomega(
        1j * np.asarray(grid.omega), float(mu_ed), float(T),
        discard_weight_tol=float(discard_weight_tol),
    )
    Ginv, worst_cond, fallback_count = _invert_exact_green(G_ed, inverse_rcond)
    norb = exact.n_sites
    eye = np.eye(norb, dtype=complex)
    hmat = np.asarray(h0h[0, 0], dtype=complex)
    G0inv = ((1j*np.asarray(grid.omega)[:,None,None] + float(mu_ed))*eye[None,:,:]
             - hmat[None,:,:])
    sigma_total = G0inv - Ginv
    density = _exact_density(exact, mu_ed, T)
    V0 = np.asarray(Vq[0, 0], dtype=complex)
    sigma_h = np.zeros_like(V0)
    sigma_h[np.diag_indices_from(sigma_h)] = V0 @ density
    sigma_gw = (sigma_total - sigma_h[None,:,:])[:,None,None,:,:]
    seed = _GWSeed(G_ed[:,None,None,:,:], sigma_h, sigma_gw, float(mu_ed))
    return seed, G_ed, worst_cond, fallback_count, float(selection.discarded_weight)


def _relative_green_error(G_gw, G_ed):
    arr = np.asarray(G_gw, dtype=complex)[:,0,0]
    exact = np.asarray(G_ed, dtype=complex)
    denom = float(np.linalg.norm(exact.ravel()))
    return float(np.linalg.norm((arr-exact).ravel()) / max(denom, 1e-300))


def _current_from_gw(gw, K, Vq, grid, h0h, backend):
    _, sigma_f, _, _ = compute_sigma_gw_split_components(
        gw.G, gw.W, Vq, grid, h0h, gw.mu, gw.Sigma_H, backend=backend
    )
    h_static = h0h + gw.Sigma_H[None,None] + sigma_f
    return float(bench._bilinear_expectation_tail_completed(
        gw.G, K, grid, h_static, gw.mu
    ).real)


def main():
    args = _args()
    if args.T <= 0 or 6*args.L1*args.L2 > 16:
        raise ValueError("need T>0 and at most 16 ED sites")
    if not (0 < args.mixing <= 1):
        raise ValueError("--mixing must lie in (0,1]")
    if not (0 <= args.ed_discard_weight_tol < 1):
        raise ValueError("--ed-discard-weight-tol must lie in [0,1)")
    if args.ed_inverse_rcond <= 0:
        raise ValueError("--ed-inverse-rcond must be positive")

    hvalues = _unique_descending_nonnegative(args.h)
    Vvalues = np.asarray(args.V, dtype=float)
    sources = list(dict.fromkeys(args.source))
    seeds = list(dict.fromkeys(args.seed))
    ncell = int(args.L1)*int(args.L2)
    target = float(args.filling)*ncell

    params0 = RubyParameters(ti=args.ti,t1=args.t1,t2=args.t2,V=0.0)
    base_exact = ExactSmallRubyThermal(args.L1,args.L2,params0)
    base_h0 = np.asarray(base_exact.h0,dtype=complex)
    Vunit = np.asarray(base_exact.Vunit,dtype=complex)
    K_by_source = {src: np.asarray(base_exact.pseudospin_operator(
        _SOURCE_TO_CHANNEL[src], (0.0,0.0)), dtype=complex) for src in sources}

    grid = MatsubaraGrid(nk1=1,nk2=1,nw=args.nw,nOmega=args.nomega,T=args.T)
    gw_opts = GWOptions(
        target_filling=target,max_iter=args.gw_max_iter,tol=args.gw_tol,
        mixing=args.mixing,mixing_method=args.mixing_method,
        pulay_history=args.pulay_history,pulay_start=args.pulay_start,
        pulay_regularization=args.pulay_regularization,verbose=args.verbose,
        momentum_backend=args.backend,
    )

    shape = (len(seeds),len(sources),len(Vvalues),len(hvalues))
    eshape = (len(sources),len(Vvalues),len(hvalues))
    J_ed=np.full(eshape,np.nan); mu_ed=np.full(eshape,np.nan)
    J_gw=np.full(shape,np.nan); mu_gw=np.full(shape,np.nan)
    converged=np.zeros(shape,dtype=bool); iterations=np.zeros(shape,dtype=int)
    residual=np.full(shape,np.nan); g_relerr=np.full(shape,np.nan)
    seed_sigma_norm=np.full(shape,np.nan)
    ed_g_condition=np.full(eshape,np.nan); ed_g_pinv_count=np.zeros(eshape,dtype=int)
    ed_g_discarded_weight=np.full(eshape,np.nan)

    print("=== exact-ED warm-start / GW basin diagnostic ===")
    print(f"cluster={args.L1}x{args.L2}, T={args.T:g}, filling={args.filling:g}, target={target:g}")
    print(f"seeds={seeds}\nV={Vvalues.tolist()}\nh={hvalues.tolist()}")

    for isrc,src in enumerate(sources):
        Ksrc=K_by_source[src]
        print(f"\n##### source={src} ({_SOURCE_TO_CHANNEL[src]}) #####")
        for iv,V in enumerate(Vvalues):
            print(f"\n=== V={V:g} ===")
            Vq=(float(V)*Vunit)[None,None]
            continuation_initial=None
            for ih,h in enumerate(hvalues):
                print(f"\n  -- h={h:.8g} --")
                hmat=0.5*((base_h0-float(h)*Ksrc)+(base_h0-float(h)*Ksrc).conj().T)
                h0h=hmat[None,None]
                exact=ExactSmallRubyThermal(args.L1,args.L2,params0)
                exact.h0=np.asarray(hmat,dtype=complex)
                exact.diagonalize(float(V))
                mu_e=exact.solve_mu(target,args.T)
                mu_ed[isrc,iv,ih]=mu_e
                J_ed[isrc,iv,ih]=float(bench._exact_onebody_expectation(exact,Ksrc,mu_e,args.T).real)

                full_seed=static_seed=None; G_ed=None
                if "ed-full" in seeds:
                    full_seed,G_ed,cond,npinv,discarded=_full_ed_seed(
                        exact,mu_e,args.T,Vq,h0h,grid,args.ed_discard_weight_tol,args.ed_inverse_rcond)
                    ed_g_condition[isrc,iv,ih]=cond; ed_g_pinv_count[isrc,iv,ih]=npinv
                    ed_g_discarded_weight[isrc,iv,ih]=discarded
                    print(f"     ED G: worst cond={cond:.3e}, pinv={npinv}, discarded={discarded:.3e}")
                if "ed-static" in seeds:
                    static_seed=_static_ed_seed(exact,mu_e,args.T,Vq,h0h,grid)
                if G_ed is None:
                    G_ed,selection=exact.green_iomega(1j*np.asarray(grid.omega),mu_e,args.T,
                        discard_weight_tol=args.ed_discard_weight_tol)
                    ed_g_discarded_weight[isrc,iv,ih]=float(selection.discarded_weight)

                solved_cont=None
                for iseed,mode in enumerate(seeds):
                    initial={"zero":None,"continuation":continuation_initial,
                             "ed-static":static_seed,"ed-full":full_seed}[mode]
                    if initial is not None:
                        seed_sigma_norm[iseed,isrc,iv,ih]=float(np.linalg.norm(initial.Sigma_GW.ravel()))
                    gw=solve_matrix_gw_fast(h0h,Vq,grid,opts=gw_opts,initial=initial)
                    if mode=="continuation": solved_cont=gw
                    converged[iseed,isrc,iv,ih]=bool(gw.converged)
                    iterations[iseed,isrc,iv,ih]=int(gw.iterations)
                    residual[iseed,isrc,iv,ih]=float(gw.final_error)
                    mu_gw[iseed,isrc,iv,ih]=float(gw.mu)
                    J_gw[iseed,isrc,iv,ih]=_current_from_gw(gw,Ksrc,Vq,grid,h0h,args.backend)
                    g_relerr[iseed,isrc,iv,ih]=_relative_green_error(gw.G,G_ed)
                    status="OK" if gw.converged else "FAIL"
                    print(f"     {mode:12s} {status:4s} iter={gw.iterations:4d} r={gw.final_error:.3e} "
                          f"J={J_gw[iseed,isrc,iv,ih]:+.9f} Gerr={g_relerr[iseed,isrc,iv,ih]:.3e}")
                if "continuation" in seeds:
                    continuation_initial=solved_cont if (solved_cont is not None and solved_cont.converged) else None
                row_conv=converged[:,isrc,iv,ih]
                if np.any(row_conv):
                    vals=J_gw[:,isrc,iv,ih][row_conv]
                    print(f"     ED J={J_ed[isrc,iv,ih]:+.9f}; converged-seed J spread={np.ptp(vals):.3e}")
                if (not args.allow_unconverged) and np.any(~row_conv):
                    failed=[seeds[i] for i in np.flatnonzero(~row_conv)]
                    raise RuntimeError(f"GW seed(s) {failed} did not converge for source={src}, V={V:g}, h={h:g}; rerun with --allow-unconverged to continue")

    outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    npz_path=outdir/"ed_warmstart_gw_branches.npz"; csv_path=outdir/"ed_warmstart_gw_branches.csv"
    np.savez_compressed(npz_path,seeds=np.asarray(seeds),sources=np.asarray(sources),
        source_channels=np.asarray([_SOURCE_TO_CHANNEL[x] for x in sources]),V=Vvalues,h=hvalues,
        J_ed=J_ed,J_gw=J_gw,mu_ed=mu_ed,mu_gw=mu_gw,gw_converged=converged,
        gw_iterations=iterations,gw_residual=residual,g_relerr_vs_ed=g_relerr,
        seed_sigma_norm=seed_sigma_norm,ed_g_worst_condition=ed_g_condition,
        ed_g_pinv_count=ed_g_pinv_count,ed_g_discarded_weight=ed_g_discarded_weight,
        L1=args.L1,L2=args.L2,filling=args.filling,target_particles=target,T=args.T,
        ti=args.ti,t1=args.t1,t2=args.t2,nw=args.nw,nOmega=args.nomega,gw_tol=args.gw_tol,
        mixing=args.mixing,mixing_method=np.asarray(args.mixing_method),
        source_convention=np.asarray("H(h)=H0-h*K_source"))
    with csv_path.open("w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh); w.writerow(["seed","source","channel","V","h","J_ED","J_GW","mu_ED","mu_GW","converged","iterations","residual","G_relerr_vs_ED","seed_sigma_norm"])
        for iseed,seed in enumerate(seeds):
            for isrc,src in enumerate(sources):
                for iv,V in enumerate(Vvalues):
                    for ih,h in enumerate(hvalues):
                        w.writerow([seed,src,_SOURCE_TO_CHANNEL[src],f"{V:.16g}",f"{h:.16g}",
                            f"{J_ed[isrc,iv,ih]:.16g}",f"{J_gw[iseed,isrc,iv,ih]:.16g}",
                            f"{mu_ed[isrc,iv,ih]:.16g}",f"{mu_gw[iseed,isrc,iv,ih]:.16g}",
                            bool(converged[iseed,isrc,iv,ih]),int(iterations[iseed,isrc,iv,ih]),
                            f"{residual[iseed,isrc,iv,ih]:.6e}",f"{g_relerr[iseed,isrc,iv,ih]:.6e}",
                            f"{seed_sigma_norm[iseed,isrc,iv,ih]:.6e}"])
    print("\nsaved:",npz_path); print("saved:",csv_path)


if __name__ == "__main__":
    main()
