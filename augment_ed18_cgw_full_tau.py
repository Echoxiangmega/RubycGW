#!/usr/bin/env python3
"""Add full finite-frequency cGW curves to an existing ED/GG tau benchmark.

The expensive finite-T ED trace is NOT repeated.  This driver loads an NPZ
produced by ``benchmark_ed18_cgw.py``, rebuilds the same zero-source SC-GW
background, solves the full H/F/MT/AL covariant vertex independently for each
represented external bosonic Matsubara frequency, and Fourier transforms the
result to imaginary time.

Only the diagonal C_{mu,mu}(q,tau) curves needed by the original benchmark
figure are computed dynamically.  This reduces the number of vertex solves:
one driven full cGW vertex per channel and positive Matsubara frequency.  The
negative-frequency diagonal response is restored with

    chi_mu_mu(-iOmega) = chi_mu_mu(+iOmega)^*.

For primitive Q=(1/3,1/3), the right source is the exact complex harmonic

    K_Q = (K_Qc - i K_Qs)/sqrt(2),

and the left observable is K_Q^dagger.  Thus the calculation reproduces the
same complex-Q convention used by ED without solving a full 12x12 Qc/Qs
matrix at every frequency.

Example
-------
    python augment_ed18_cgw_full_tau.py ^
      --benchmark bench_n3_V1_T008.npz ^
      --nw 55 ^
      --vertex-workers 2 ^
      --out bench_n3_V1_T008_full_tau.npz

The output PNG overlays ED, GG, and full cGW on exactly the same tau panels.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.checkpoint import checkpoint_filename, load_supercell_checkpoint
from rubycgw.dynamic_cgw import solve_vertex_iomega, susceptibility_matrix_iomega
from rubycgw.ed_cgw_benchmark import bosonic_iomega_to_tau
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.pseudospin import supercell_pseudospin_harmonic_vertices
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    density_from_G_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
)
from rubycgw.supercell_gw_split import compute_sigma_gw_split_matrix


_CHANNEL_TEX = {
    "x_even": r"$x_{\rm even}$",
    "x_odd": r"$x_{\rm odd}$",
    "y_even": r"$y_{\rm even}$",
    "y_odd": r"$y_{\rm odd}$",
    "z_same": r"$z_{\rm same}$",
    "z_opposite": r"$z_{\rm opposite}$",
}


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, help="Existing benchmark_ed18_cgw NPZ.")
    p.add_argument("--nw", type=int, default=55, help="Fermion half-box used by the SC-GW checkpoint.")
    p.add_argument(
        "--mmax",
        type=int,
        default=None,
        help="Largest positive external bosonic index to solve. Default: benchmark nOmega.",
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-dir", default="results/supercell18/checkpoints")
    p.add_argument("--max-scgw-residual", type=float, default=1e-6)
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--vertex-max-iter", type=int, default=150)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-solver", choices=["gmres", "linear"], default="gmres")
    p.add_argument("--vertex-gmres-restart", type=int, default=12)
    p.add_argument("--vertex-mixing", type=float, default=0.25)
    p.add_argument("--vertex-workers", type=int, default=1)
    p.add_argument("--vertex-verbose", action="store_true")
    p.add_argument("--static-check-tol", type=float, default=5e-5)
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _load(path):
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def _scalar(data, key, cast=float):
    if key not in data:
        raise KeyError(f"benchmark is missing required key {key!r}")
    return cast(np.asarray(data[key]).reshape(()))


def _rebuild_scgw(seed, params, grid, backend):
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    G = dyson_from_sigma_matrix(h0, grid, seed.mu, seed.Sigma_H, seed.Sigma_GW)
    density = density_from_G_matrix(G, grid, h0=h0, mu=seed.mu, sigma_h=seed.Sigma_H)
    sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_split_matrix(
        G, W, Vq, grid, h0, seed.mu, seed.Sigma_H, backend=backend
    )
    rH = float(np.max(np.abs(sigma_h_out - seed.Sigma_H)))
    rGW = float(np.max(np.abs(sigma_gw_out - seed.Sigma_GW)))
    return Vq, G, W, density, rH, rGW


def _checkpoint_path(args, source, params, grid, filling):
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    if "scgw_checkpoint" in source:
        saved = Path(str(np.asarray(source["scgw_checkpoint"]).reshape(())))
        if saved.exists():
            return saved
    return Path(args.checkpoint_dir) / checkpoint_filename(
        float(params.V), float(filling), grid
    )


def _vertices(channels):
    KG, _, _ = supercell_pseudospin_harmonic_vertices(channels, harmonic="q0")
    KC, _, _ = supercell_pseudospin_harmonic_vertices(channels, harmonic="Qc")
    KS, _, _ = supercell_pseudospin_harmonic_vertices(channels, harmonic="Qs")
    root2 = np.sqrt(2.0)
    # O_Q = (Qc - i Qs)/sqrt(2); left observable is O_Q^dagger.
    KQ_right = (np.asarray(KC) - 1j * np.asarray(KS)) / root2
    KQ_left = (np.asarray(KC) + 1j * np.asarray(KS)) / root2
    return np.asarray(KG), KQ_left, KQ_right


def _solve_chain(
    qname,
    channel,
    Kleft,
    Kright,
    m_nonnegative,
    G,
    W,
    Vq,
    grid,
    opts,
):
    vals = np.zeros(len(m_nonnegative), dtype=complex)
    iterations = np.zeros(len(m_nonnegative), dtype=int)
    residuals = np.zeros(len(m_nonnegative), dtype=float)
    previous = None
    for j, m in enumerate(m_nonnegative):
        print(f"  dynamic full {qname:5s} {channel:12s} m={m:+d}")
        res = solve_vertex_iomega(
            G,
            W,
            Vq,
            Kright,
            int(m),
            grid,
            opts=opts,
            initial_gamma=previous,
        )
        if not res.converged:
            raise RuntimeError(
                f"dynamic vertex failed for {qname}/{channel}/m={m}: "
                f"residual={res.final_error:.3e}"
            )
        vals[j] = susceptibility_matrix_iomega(
            G,
            Kleft[None, ...],
            [res.Gamma],
            int(m),
            grid,
        )[0, 0]
        iterations[j] = int(res.iterations)
        residuals[j] = float(res.final_error)
        previous = res.Gamma
    return vals, iterations, residuals


def _plot_tau(tau, beta, ed_tau, gg_tau_diag, full_tau_diag, channels, out, dpi):
    x = np.asarray(tau, dtype=float) / float(beta)
    groups = [
        ("x", [i for i, c in enumerate(channels) if str(c).startswith("x_")]),
        ("y", [i for i, c in enumerate(channels) if str(c).startswith("y_")]),
        ("z/current", [i for i, c in enumerate(channels) if str(c).startswith("z_")]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 10.5), sharex=True)
    qtitles = [r"$\Gamma$", r"$Q=(1/3,1/3)$"]
    for row, (gname, idxs) in enumerate(groups):
        for iq in range(2):
            ax = axes[row, iq]
            for ic in idxs:
                label = _CHANNEL_TEX.get(str(channels[ic]), str(channels[ic]))
                ax.plot(x, np.real(ed_tau[iq, :, ic, ic]), label=f"ED {label}")
                ax.plot(x, np.real(gg_tau_diag[iq, :, ic]), linestyle="--", label=f"GG {label}")
                ax.plot(x, np.real(full_tau_diag[iq, :, ic]), linestyle=":", linewidth=2.0, label=f"full {label}")
            ax.axvline(0.5, linestyle=":", linewidth=0.8, alpha=0.5)
            ax.grid(alpha=0.2)
            ax.set_ylabel(r"$C_{\mu\mu}(q,\tau)$")
            if row == 0:
                ax.set_title(qtitles[iq])
            if row == 2:
                ax.set_xlabel(r"$\tau/\beta$")
            ax.legend(fontsize=7, ncol=2)
            ax.text(0.02, 0.05, gname, transform=ax.transAxes, fontsize=9)
    fig.suptitle(
        "same 18-site torus: ED vs GG vs full cGW imaginary-time correlation",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    source_path = Path(args.benchmark)
    source = _load(source_path)

    V = _scalar(source, "V")
    filling = _scalar(source, "filling")
    T = _scalar(source, "temperature")
    ti = _scalar(source, "ti")
    t1 = _scalar(source, "t1")
    t2 = _scalar(source, "t2")
    tau = np.asarray(source["tau_grid"], dtype=float)
    channels = tuple(str(x) for x in np.asarray(source["channels"]).tolist())
    ed_tau = np.asarray(source["ed_correlation_tau"], dtype=complex)
    gg_iw_G = np.asarray(source["gg_iomega_Gamma"], dtype=complex)
    gg_iw_Q = np.asarray(source["gg_iomega_Q"], dtype=complex)
    m_values_source = np.asarray(source["gg_m_values"], dtype=int)

    nOmega = int(np.max(np.abs(m_values_source)))
    mmax = nOmega if args.mmax is None else int(args.mmax)
    if mmax < 0 or mmax > nOmega:
        raise ValueError(f"--mmax must satisfy 0 <= mmax <= benchmark nOmega={nOmega}")
    if int(args.nw) <= nOmega:
        raise ValueError("--nw should be larger than nOmega for a useful shifted fermion window")

    grid = MatsubaraGrid(nk1=1, nk2=1, nw=int(args.nw), nOmega=nOmega, T=T)
    if not np.array_equal(np.asarray(grid.m_values), m_values_source):
        raise ValueError("reconstructed bosonic grid does not match source benchmark")

    params = RubyParameters(ti=ti, t1=t1, t2=t2, V=V)
    checkpoint = _checkpoint_path(args, source, params, grid, filling)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SC-GW checkpoint not found: {checkpoint}\n"
            "Pass --checkpoint explicitly if the benchmark was moved."
        )
    seed, meta, _ = load_supercell_checkpoint(checkpoint, params, grid, filling)
    if not bool(meta.get("converged", False)):
        raise ValueError("dynamic benchmark requires a converged zero-source SC-GW checkpoint")
    if abs(float(meta.get("source", 0.0))) > 1e-14:
        raise ValueError("dynamic benchmark requires a zero-source checkpoint")

    print("=== augment same-torus benchmark with full dynamic cGW ===")
    print(f"source benchmark: {source_path}")
    print(f"V={V:g}, filling={filling:g}, T={T:g}, nw={args.nw}, nOmega={nOmega}, mmax={mmax}")
    print("channels:", ", ".join(channels))
    print("checkpoint:", checkpoint)
    print("dynamic vertex: full H+F+MT+AL")
    print("negative diagonal frequencies are restored from chi(-iOmega)=chi(+iOmega)^*")

    Vq, G, W, density, rH, rGW = _rebuild_scgw(
        seed, params, grid, args.momentum_backend
    )
    sc_res = max(rH, rGW)
    print(f"SC-GW fixed-point check: rH={rH:.3e}, rGW={rGW:.3e}, max={sc_res:.3e}")
    if sc_res > float(args.max_scgw_residual):
        raise RuntimeError(
            f"checkpoint residual {sc_res:.3e} exceeds --max-scgw-residual"
        )

    KG, KQ_left, KQ_right = _vertices(channels)
    opts = SupercellVertexOptions(
        max_iter=int(args.vertex_max_iter),
        tol=float(args.vertex_tol),
        mixing=float(args.vertex_mixing),
        solver=str(args.vertex_solver),
        gmres_restart=int(args.vertex_gmres_restart),
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=bool(args.vertex_verbose),
        momentum_backend=str(args.momentum_backend),
    )

    m_nonnegative = np.arange(0, mmax + 1, dtype=int)
    nc = len(channels)
    pos = np.zeros((2, len(m_nonnegative), nc), dtype=complex)
    iterations = np.zeros((2, len(m_nonnegative), nc), dtype=int)
    residuals = np.zeros((2, len(m_nonnegative), nc), dtype=float)

    jobs = []
    for ic, ch in enumerate(channels):
        jobs.append((0, ic, "Gamma", ch, KG[ic], KG[ic]))
        jobs.append((1, ic, "Q", ch, KQ_left[ic], KQ_right[ic]))

    def run_job(job):
        iq, ic, qname, ch, kl, kr = job
        vals, its, resids = _solve_chain(
            qname, ch, kl, kr, m_nonnegative, G, W, Vq, grid, opts
        )
        return iq, ic, vals, its, resids

    workers = max(1, int(args.vertex_workers))
    if workers == 1:
        results = [run_job(job) for job in jobs]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(run_job, job) for job in jobs]
            for fut in as_completed(futs):
                results.append(fut.result())
    for iq, ic, vals, its, resids in results:
        pos[iq, :, ic] = vals
        iterations[iq, :, ic] = its
        residuals[iq, :, ic] = resids

    # Reconstruct the symmetric represented frequency window.  Frequencies
    # above mmax are deliberately omitted from BOTH GG and cGW tau transforms
    # so the plotted comparison uses the same Matsubara cutoff.
    full_iw_diag = np.zeros((2, grid.nb, nc), dtype=complex)
    keep = np.abs(np.asarray(grid.m_values)) <= mmax
    for im, m_raw in enumerate(grid.m_values):
        m = int(m_raw)
        if abs(m) > mmax:
            continue
        if m >= 0:
            full_iw_diag[:, im] = pos[:, m]
        else:
            full_iw_diag[:, im] = np.conj(pos[:, -m])

    gg_iw_diag = np.stack(
        [
            np.diagonal(gg_iw_G, axis1=-2, axis2=-1),
            np.diagonal(gg_iw_Q, axis1=-2, axis2=-1),
        ],
        axis=0,
    )
    gg_iw_diag_cut = np.where(keep[None, :, None], gg_iw_diag, 0.0)

    full_tau_diag = np.stack(
        [bosonic_iomega_to_tau(full_iw_diag[iq], grid, tau) for iq in range(2)],
        axis=0,
    )
    gg_tau_diag = np.stack(
        [bosonic_iomega_to_tau(gg_iw_diag_cut[iq], grid, tau) for iq in range(2)],
        axis=0,
    )

    # m=0 must reproduce the previously saved static full cGW diagonal when
    # that benchmark used stage=full. This is the most important regression
    # check for the new frequency routing and complex-Q source convention.
    static_rel = np.full(2, np.nan, dtype=float)
    if "cgw_static_chi" in source and str(np.asarray(source.get("cgw_stage", "")).reshape(())) == "full":
        static = np.asarray(source["cgw_static_chi"], dtype=complex)
        m0 = int(np.where(np.asarray(grid.m_values) == 0)[0][0])
        for iq, qname in enumerate(("Gamma", "Q")):
            ref = np.real(np.diag(static[iq]))
            got = np.real(full_iw_diag[iq, m0])
            denom = max(float(np.linalg.norm(ref)), 1e-300)
            static_rel[iq] = float(np.linalg.norm(got - ref) / denom)
            print(f"m=0 consistency {qname}: relative diagonal error={static_rel[iq]:.3e}")
            if static_rel[iq] > float(args.static_check_tol):
                raise RuntimeError(
                    f"dynamic m=0 does not reproduce saved static full cGW at {qname}: "
                    f"relative error={static_rel[iq]:.3e}"
                )

    mid = int(np.argmin(np.abs(tau - 0.5 / T)))
    print("\n=== tau=beta/2 diagonal comparison ===")
    for iq, qname in enumerate(("Gamma", "Q")):
        print(qname + ":")
        for ic, ch in enumerate(channels):
            e = float(np.real(ed_tau[iq, mid, ic, ic]))
            g = float(np.real(gg_tau_diag[iq, mid, ic]))
            f = float(np.real(full_tau_diag[iq, mid, ic]))
            print(f"  {ch:12s} ED={e:+.6e}  GG={g:+.6e}  full={f:+.6e}")

    out = Path(args.out) if args.out else source_path.with_name(source_path.stem + "_full_tau.npz")
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        source_benchmark=np.asarray(str(source_path)),
        V=float(V),
        filling=float(filling),
        temperature=float(T),
        beta=float(1.0 / T),
        ti=float(ti),
        t1=float(t1),
        t2=float(t2),
        nw=int(args.nw),
        nOmega=int(nOmega),
        dynamic_mmax=int(mmax),
        m_values=np.asarray(grid.m_values),
        Omega=np.asarray(grid.Omega),
        channels=np.asarray(channels),
        q_names=np.asarray(["Gamma", "Q"]),
        tau_grid=tau,
        ed_correlation_tau=ed_tau,
        gg_iomega_diagonal=gg_iw_diag_cut,
        gg_correlation_tau_diagonal=gg_tau_diag,
        full_cgw_iomega_diagonal=full_iw_diag,
        full_cgw_correlation_tau_diagonal=full_tau_diag,
        full_cgw_positive_m=np.asarray(m_nonnegative),
        full_cgw_vertex_iterations=iterations,
        full_cgw_vertex_residuals=residuals,
        m0_static_relative_diagonal_error=static_rel,
        scgw_checkpoint=np.asarray(str(checkpoint)),
        scgw_mu=float(seed.mu),
        scgw_density=np.asarray(density),
        scgw_residual_H=float(rH),
        scgw_residual_GW=float(rGW),
        negative_frequency_reconstruction=np.asarray(
            "diagonal Hermitian response: chi(-iOmega)=conj(chi(+iOmega))"
        ),
        dynamic_scope=np.asarray(
            "full H+F+MT+AL cGW at q_sc=0 for each external bosonic frequency; diagonal C_mu_mu only"
        ),
    )
    plot = out.with_name(out.stem + ".png")
    _plot_tau(tau, 1.0 / T, ed_tau, gg_tau_diag, full_tau_diag, channels, plot, args.dpi)
    print("saved:", out)
    print("saved:", plot)


if __name__ == "__main__":
    main()
