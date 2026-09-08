#!/usr/bin/env python3
"""Validate static cGW against direct finite-source SC-GW derivatives.

This is an end-to-end FDT/functional-derivative check of the *implementation*,
not a comparison to ED.  Starting from the converged zero-source SC-GW
checkpoint used by ``benchmark_ed18_cgw.py``, for each requested primitive-q=0
pseudospin channel K we compare

    chi_cGW = - int Tr[K G Gamma_K G]

with the direct central finite difference

    chi_FD(h) = [ <K>_{+h} - <K>_{-h} ] / (2 h),

where both +/- source states are fully re-solved SC-GW fixed points of

    H(h) = H(0) - h K.

By default the source solves keep the chemical potential fixed at the
zero-source checkpoint value.  This is the direct numerical derivative matched
by the present cGW vertex equation.  ``--ensemble fixed-filling`` is available
as a separate diagnostic, but it is not the strict identity being tested unless
a chemical-potential counterterm is also included in the vertex equation.

The default channels are the two normalized loop-current pseudospins
``z_same`` and ``z_opposite``.  The exact same harmonic vertices used by the
static cGW benchmark are used as finite source operators, so no legacy eta vs z
normalization conversion is involved.

Example
-------
    python validate_cgw_finite_source.py ^
      --benchmark bench_n3_V1_T008.npz ^
      --h-values 0.02 0.01 0.005 ^
      --gw-tol 1e-10 ^
      --out validate_n3_V1_current_fd.npz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.checkpoint import checkpoint_filename, load_supercell_checkpoint
from rubycgw.ed_cgw_benchmark import static_response_from_gammas
from rubycgw.finite_source_validation import (
    add_bilinear_source,
    bilinear_expectation,
    central_finite_difference,
    quadratic_zero_source_extrapolation,
    relative_error,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.pseudospin import (
    canonical_channel_name,
    supercell_pseudospin_harmonic_vertex,
)
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import SupercellVertexOptions, solve_vertex_q0
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    density_from_G_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
)
from rubycgw.supercell_gw_bootstrap import AndersonOptions, solve_matrix_gw_anderson
from rubycgw.supercell_gw_split import compute_sigma_gw_split_matrix


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, help="NPZ from benchmark_ed18_cgw.py")
    p.add_argument(
        "--channels",
        nargs="+",
        default=["z_same", "z_opposite"],
        help="Primitive q=0 pseudospin channels. Default: both current channels.",
    )
    p.add_argument(
        "--h-values",
        nargs="+",
        type=float,
        default=[0.02, 0.01, 0.005],
        help="Positive source magnitudes for symmetric central differences.",
    )
    p.add_argument(
        "--ensemble",
        choices=["fixed-mu", "fixed-filling"],
        default="fixed-mu",
        help=(
            "Source-solve constraint. fixed-mu is the direct identity matched by "
            "the current cGW vertex implementation."
        ),
    )
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-dir", default="results/supercell18/checkpoints")
    p.add_argument("--max-scgw-residual", type=float, default=1e-6)
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")

    p.add_argument("--gw-max-iter", type=int, default=350)
    p.add_argument("--gw-tol", type=float, default=1e-10)
    p.add_argument("--gw-mixing", type=float, default=0.20)
    p.add_argument("--mu-tol", type=float, default=1e-11)
    p.add_argument("--mu-max-iter", type=int, default=80)
    p.add_argument("--gw-verbose", action="store_true")

    p.add_argument("--vertex-max-iter", type=int, default=180)
    p.add_argument("--vertex-tol", type=float, default=1e-9)
    p.add_argument("--vertex-solver", choices=["gmres", "linear"], default="gmres")
    p.add_argument("--vertex-gmres-restart", type=int, default=12)
    p.add_argument("--vertex-mixing", type=float, default=0.25)
    p.add_argument("--vertex-verbose", action="store_true")

    p.add_argument("--out", default="validate_cgw_finite_source.npz")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def _scalar(data: dict[str, np.ndarray], key: str, cast=float):
    if key not in data:
        raise KeyError(f"benchmark is missing required key {key!r}")
    return cast(np.asarray(data[key]).reshape(()))


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


def _rebuild_zero_background(seed, params, grid, backend):
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    G = dyson_from_sigma_matrix(h0, grid, seed.mu, seed.Sigma_H, seed.Sigma_GW)
    density = density_from_G_matrix(
        G, grid, h0=h0, mu=seed.mu, sigma_h=seed.Sigma_H
    )
    sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_split_matrix(
        G, W, Vq, grid, h0, seed.mu, seed.Sigma_H, backend=backend
    )
    rH = float(np.max(np.abs(sigma_h_out - seed.Sigma_H)))
    rGW = float(np.max(np.abs(sigma_gw_out - seed.Sigma_GW)))
    return h0, Vq, G, W, density, rH, rGW


def _cgw_scalar(G, W, Vq, K, grid, args):
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
    result = solve_vertex_q0(G, W, Vq, K, grid, opts=opts)
    if not result.converged:
        raise RuntimeError(
            f"cGW vertex did not converge: residual={result.final_error:.3e}"
        )
    chi = static_response_from_gammas(
        G,
        np.asarray(K)[None, ...],
        [result.Gamma],
        grid,
    )[0, 0]
    return complex(chi), result


def _source_options(args, seed_mu, target_supercell):
    target = None if args.ensemble == "fixed-mu" else float(target_supercell)
    return GWOptions(
        mu=float(seed_mu),
        target_filling=target,
        max_iter=int(args.gw_max_iter),
        tol=float(args.gw_tol),
        mixing=float(args.gw_mixing),
        mixing_method="linear",
        pulay_history=6,
        pulay_start=3,
        pulay_regularization=1e-10,
        mu_tol=float(args.mu_tol),
        mu_max_iter=int(args.mu_max_iter),
        verbose=bool(args.gw_verbose),
        momentum_backend=str(args.momentum_backend),
    )


def _solve_source(base_h0, Vq, K, h, sign, seed, grid, opts):
    hsigned = float(sign) * float(h)
    h0 = add_bilinear_source(base_h0, K, hsigned)
    t0 = time.perf_counter()
    gw = solve_matrix_gw_anderson(
        h0,
        Vq,
        grid,
        opts=opts,
        initial=seed,
        anderson=AndersonOptions(),
    )
    runtime = time.perf_counter() - t0
    if not gw.converged:
        raise RuntimeError(
            f"SC-GW source solve failed at h={hsigned:+.6g}: "
            f"residual={gw.final_error:.3e}"
        )
    value = bilinear_expectation(gw.G, K, grid)
    return gw, value, runtime


def _plot(results, channels, out, dpi):
    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.8), squeeze=False)
    for i, ch in enumerate(channels):
        ax = axes[0, i]
        rows = results[ch]["rows"]
        h = np.array([x["h"] for x in rows], dtype=float)
        fd = np.array([x["chi_fd_re"] for x in rows], dtype=float)
        cgw = float(results[ch]["chi_cgw"].real)
        ext = float(results[ch]["chi_extrapolated"].real)
        ax.plot(h**2, fd, marker="o", label="finite source")
        ax.axhline(cgw, linestyle="--", label="cGW vertex")
        ax.scatter([0.0], [ext], marker="x", s=70, label=r"$h\to0$ fit")
        ax.set_xlabel(r"$h^2$")
        ax.set_ylabel(r"$\chi_{KK}$")
        ax.set_title(ch)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        text = (
            f"cGW={cgw:+.6g}\n"
            f"fit={ext:+.6g}\n"
            f"rel.err={results[ch]['relerr_extrapolated']:.3e}"
        )
        ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", fontsize=8)
    fig.suptitle("SC-GW finite-source derivative vs full static cGW", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    source = _load_npz(args.benchmark)

    V = _scalar(source, "V")
    filling = _scalar(source, "filling")
    T = _scalar(source, "temperature")
    ti = _scalar(source, "ti")
    t1 = _scalar(source, "t1")
    t2 = _scalar(source, "t2")
    m_values = np.asarray(source["gg_m_values"], dtype=int)
    nOmega = int(np.max(np.abs(m_values)))

    if args.gw_tol <= 0.0 or args.vertex_tol <= 0.0:
        raise ValueError("tolerances must be positive")
    h_values = np.asarray(sorted(set(float(x) for x in args.h_values), reverse=True))
    if h_values.size < 2 or np.any(h_values <= 0.0):
        raise ValueError("--h-values needs at least two positive values")

    channels = tuple(canonical_channel_name(x) for x in args.channels)
    params = RubyParameters(ti=ti, t1=t1, t2=t2, V=V)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=int(args.nw), nOmega=nOmega, T=T)
    target_supercell = 3.0 * float(filling)

    checkpoint = _checkpoint_path(args, source, params, grid, filling)
    if not checkpoint.exists():
        raise FileNotFoundError(f"SC-GW checkpoint not found: {checkpoint}")
    seed, meta, _ = load_supercell_checkpoint(
        checkpoint, params, grid, float(filling)
    )
    if not bool(meta.get("converged", False)):
        raise ValueError("finite-source validation requires a converged zero-source checkpoint")
    if abs(float(meta.get("source", 0.0))) > 1e-14:
        raise ValueError("finite-source validation requires a zero-source checkpoint")

    base_h0, Vq, G0, W0, density0, rH, rGW = _rebuild_zero_background(
        seed, params, grid, args.momentum_backend
    )
    sc_res = max(rH, rGW)
    if sc_res > float(args.max_scgw_residual):
        raise RuntimeError(
            f"zero-source checkpoint residual {sc_res:.3e} exceeds "
            f"--max-scgw-residual={args.max_scgw_residual:.3e}"
        )

    print("=== cGW finite-source derivative validation ===")
    print(
        f"V={V:g}, filling={filling:g}, T={T:g}, grid=1x1 supercell, "
        f"nw={grid.nw}, nOmega={grid.nOmega}"
    )
    print("checkpoint:", checkpoint)
    print(f"zero-source residual: rH={rH:.3e}, rGW={rGW:.3e}")
    print(f"zero-source mu={seed.mu:.12g}, total density={np.sum(density0):.12g}")
    print("source ensemble:", args.ensemble)
    if args.ensemble == "fixed-mu":
        print("  direct identity tested: derivative at fixed chemical potential")
    else:
        print(
            "  WARNING: fixed-filling finite difference is an ensemble diagnostic, "
            "not the strict derivative represented by the current cGW vertex equation."
        )

    opts_source = _source_options(args, seed.mu, target_supercell)
    results = {}
    csv_rows = []

    saved_static = None
    if "cgw_static_chi" in source:
        saved_static = np.asarray(source["cgw_static_chi"], dtype=complex)
    source_channels = tuple(str(x) for x in np.asarray(source.get("channels", np.array([]))).tolist())

    for ch in channels:
        K, label, _ = supercell_pseudospin_harmonic_vertex(ch, harmonic="q0")
        trace_K = complex(np.trace(K))
        if abs(trace_K) > 1e-12:
            raise ValueError(
                f"finite-box expectation helper assumes a traceless source; {ch} has TrK={trace_K}"
            )

        print("\n" + "-" * 78)
        print(f"channel {ch} ({label})")
        chi_cgw, vertex = _cgw_scalar(G0, W0, Vq, K, grid, args)
        m0 = bilinear_expectation(G0, K, grid)
        print(
            f"full cGW chi={chi_cgw.real:+.10e}{chi_cgw.imag:+.2e}i, "
            f"vertex residual={vertex.final_error:.3e}, <K>0={m0.real:+.3e}{m0.imag:+.1e}i"
        )

        if saved_static is not None and ch in source_channels:
            idx = source_channels.index(ch)
            saved = complex(saved_static[0, idx, idx])
            delta_saved = abs(saved - chi_cgw)
            print(
                f"saved benchmark static chi={saved.real:+.10e}{saved.imag:+.2e}i, "
                f"|new-saved|={delta_saved:.3e}"
            )

        rows = []
        fd_values = []
        for h in h_values:
            print(f"  finite source h={h:g}: solve +h")
            gp, mp, tp = _solve_source(
                base_h0, Vq, K, h, +1.0, seed, grid, opts_source
            )
            print(f"  finite source h={h:g}: solve -h")
            gm, mm, tm = _solve_source(
                base_h0, Vq, K, h, -1.0, seed, grid, opts_source
            )
            chi_fd = central_finite_difference(mp, mm, h)
            fd_values.append(chi_fd)
            even_offset = 0.5 * (mp + mm) - m0
            rel = relative_error(chi_cgw, chi_fd)
            row = {
                "channel": ch,
                "ensemble": args.ensemble,
                "h": float(h),
                "m_plus_re": float(mp.real),
                "m_plus_im": float(mp.imag),
                "m_minus_re": float(mm.real),
                "m_minus_im": float(mm.imag),
                "chi_fd_re": float(chi_fd.real),
                "chi_fd_im": float(chi_fd.imag),
                "chi_cgw_re": float(chi_cgw.real),
                "chi_cgw_im": float(chi_cgw.imag),
                "relative_error": float(rel),
                "even_offset_abs": float(abs(even_offset)),
                "mu_plus": float(gp.mu),
                "mu_minus": float(gm.mu),
                "density_plus": float(np.sum(gp.density)),
                "density_minus": float(np.sum(gm.density)),
                "residual_plus": float(gp.final_error),
                "residual_minus": float(gm.final_error),
                "runtime_plus_s": float(tp),
                "runtime_minus_s": float(tm),
            }
            rows.append(row)
            csv_rows.append(row)
            print(
                f"    <K>+={mp.real:+.9e}, <K>-={mm.real:+.9e}, "
                f"chi_FD={chi_fd.real:+.9e}{chi_fd.imag:+.1e}i, "
                f"rel.err={rel:.3e}, even-offset={abs(even_offset):.3e}"
            )
            print(
                f"    residuals (+/-)=({gp.final_error:.2e},{gm.final_error:.2e}), "
                f"mu(+/-)=({gp.mu:.10f},{gm.mu:.10f}), "
                f"N(+/-)=({np.sum(gp.density):.10f},{np.sum(gm.density):.10f})"
            )

        chi_ext, slope = quadratic_zero_source_extrapolation(h_values, fd_values)
        rel_ext = relative_error(chi_cgw, chi_ext)
        print(
            f"  h->0 fit: chi={chi_ext.real:+.10e}{chi_ext.imag:+.2e}i, "
            f"cGW={chi_cgw.real:+.10e}{chi_cgw.imag:+.2e}i, "
            f"relative error={rel_ext:.3e}"
        )
        results[ch] = {
            "chi_cgw": chi_cgw,
            "chi_extrapolated": chi_ext,
            "fit_slope": slope,
            "relerr_extrapolated": rel_ext,
            "m0": m0,
            "rows": rows,
        }

    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    channel_array = np.asarray(channels)
    chi_cgw_array = np.asarray([results[ch]["chi_cgw"] for ch in channels])
    chi_ext_array = np.asarray([results[ch]["chi_extrapolated"] for ch in channels])
    rel_ext_array = np.asarray([results[ch]["relerr_extrapolated"] for ch in channels])
    m0_array = np.asarray([results[ch]["m0"] for ch in channels])
    chi_fd_array = np.asarray(
        [[complex(x["chi_fd_re"], x["chi_fd_im"]) for x in results[ch]["rows"]] for ch in channels]
    )
    np.savez_compressed(
        out,
        source_benchmark=np.asarray(str(args.benchmark)),
        checkpoint=np.asarray(str(checkpoint)),
        ensemble=np.asarray(args.ensemble),
        V=float(V),
        filling=float(filling),
        temperature=float(T),
        ti=float(ti),
        t1=float(t1),
        t2=float(t2),
        nw=int(grid.nw),
        nOmega=int(grid.nOmega),
        channels=channel_array,
        h_values=h_values,
        chi_cgw=chi_cgw_array,
        chi_fd=chi_fd_array,
        chi_extrapolated=chi_ext_array,
        relative_error_extrapolated=rel_ext_array,
        zero_source_expectation=m0_array,
        zero_source_mu=float(seed.mu),
        zero_source_density=np.asarray(density0),
        zero_source_residual_H=float(rH),
        zero_source_residual_GW=float(rGW),
        identity_note=np.asarray(
            "fixed-mu central derivative is the direct numerical derivative matched by the present cGW vertex"
        ),
    )
    print("\nsaved:", out)

    csv_path = out.with_suffix(".csv")
    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print("saved:", csv_path)

    plot_path = out.with_name(out.stem + "_fd.png")
    _plot(results, channels, plot_path, args.dpi)
    print("saved:", plot_path)

    print("\n=== verdict ===")
    for ch in channels:
        print(
            f"{ch:12s}: cGW={results[ch]['chi_cgw'].real:+.9e}, "
            f"FD(h->0)={results[ch]['chi_extrapolated'].real:+.9e}, "
            f"rel.err={results[ch]['relerr_extrapolated']:.3e}"
        )
    if args.ensemble == "fixed-mu":
        print(
            "If the extrapolated finite-source derivatives agree with cGW within "
            "source/truncation/solver error, the H/F/MT/AL implementation is the "
            "correct derivative of the implemented SC-GW fixed-point map."
        )


if __name__ == "__main__":
    main()
