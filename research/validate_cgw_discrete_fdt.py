#!/usr/bin/env python3
"""Separate cGW implementation error from production Matsubara-tail error.

The production finite-source check compares cGW against the tail-corrected SC-GW
map and currently differs at the ~1% level for the n=3,V=1,T=0.08 benchmark.
That comparison mixes two effects:

1. whether H/F/MT/AL are the correct functional derivative of the discretized GW
   equations;
2. whether the finite Matsubara box used by cGW reproduces the analytic-tail
   equal-time sums used by production SC-GW.

This driver removes item 2.  It first re-equilibrates the loaded production
checkpoint to a diagnostic *finite-box* SC-GW fixed point at the same chemical
potential.  The finite-box map evaluates both Hartree density and static Fock
rho with the represented Matsubara sum only, exactly matching the X=G Gamma G
sums in ``supercell_cgw.py``.  It then compares

    chi_cGW = - int Tr[K G Gamma_K G]

against fully self-consistent +/-h finite-source derivatives of that same
finite-box map.

If these agree to solver/finite-difference precision while the production-tail
check remains at ~1%, the residual mismatch is numerical tail discretization,
not a missing H/F/MT/AL derivative.

Example
-------
    python validate_cgw_discrete_fdt.py ^
      --benchmark bench_n3_V1_T008.npz ^
      --h-values 0.02 0.01 0.005 ^
      --gw-tol 1e-10 ^
      --out validate_n3_V1_discrete_fdt.npz
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
from rubycgw.finite_box_gw import solve_matrix_gw_finite_box
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
from rubycgw.pseudospin import canonical_channel_name, supercell_pseudospin_harmonic_vertex
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import SupercellVertexOptions, solve_vertex_q0


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--channels", nargs="+", default=["z_same", "z_opposite"])
    p.add_argument("--h-values", nargs="+", type=float, default=[0.02, 0.01, 0.005])
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-dir", default="results/supercell18/checkpoints")
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--gw-max-iter", type=int, default=700)
    p.add_argument("--gw-tol", type=float, default=1e-10)
    p.add_argument("--gw-mixing", type=float, default=0.12)
    p.add_argument("--gw-verbose", action="store_true")
    p.add_argument("--vertex-max-iter", type=int, default=200)
    p.add_argument("--vertex-tol", type=float, default=1e-10)
    p.add_argument("--vertex-gmres-restart", type=int, default=12)
    p.add_argument("--vertex-verbose", action="store_true")
    p.add_argument("--out", default="validate_cgw_discrete_fdt.npz")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _load(path):
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def _scalar(data, key, cast=float):
    if key not in data:
        raise KeyError(f"benchmark missing {key!r}")
    return cast(np.asarray(data[key]).reshape(()))


def _checkpoint_path(args, source, params, grid, filling):
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    if "scgw_checkpoint" in source:
        saved = Path(str(np.asarray(source["scgw_checkpoint"]).reshape(())))
        if saved.exists():
            return saved
    return Path(args.checkpoint_dir) / checkpoint_filename(float(params.V), float(filling), grid)


def _gw_options(args, mu):
    return GWOptions(
        mu=float(mu),
        target_filling=None,
        max_iter=int(args.gw_max_iter),
        tol=float(args.gw_tol),
        mixing=float(args.gw_mixing),
        mixing_method="linear",
        pulay_history=6,
        pulay_start=3,
        pulay_regularization=1e-10,
        verbose=bool(args.gw_verbose),
        momentum_backend=str(args.momentum_backend),
    )


def _cgw_scalar(G, W, Vq, K, grid, args):
    opts = SupercellVertexOptions(
        max_iter=int(args.vertex_max_iter),
        tol=float(args.vertex_tol),
        solver="gmres",
        gmres_restart=int(args.vertex_gmres_restart),
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=bool(args.vertex_verbose),
        momentum_backend=str(args.momentum_backend),
    )
    res = solve_vertex_q0(G, W, Vq, K, grid, opts=opts)
    if not res.converged:
        raise RuntimeError(f"cGW vertex failed: residual={res.final_error:.3e}")
    chi = static_response_from_gammas(G, K[None, ...], [res.Gamma], grid)[0, 0]
    return complex(chi), res


def _plot(results, channels, out, dpi):
    fig, axes = plt.subplots(1, len(channels), figsize=(6.2 * len(channels), 4.8), squeeze=False)
    for i, ch in enumerate(channels):
        r = results[ch]
        h = np.asarray([row["h"] for row in r["rows"]], dtype=float)
        fd = np.asarray([row["chi_fd_re"] for row in r["rows"]], dtype=float)
        ax = axes[0, i]
        ax.plot(h * h, fd, marker="o", label="finite-box finite source")
        ax.axhline(r["chi_cgw"].real, linestyle="--", label="cGW vertex")
        ax.scatter([0.0], [r["chi_extrapolated"].real], marker="x", s=70, label=r"$h\to0$ fit")
        ax.set_xlabel(r"$h^2$")
        ax.set_ylabel(r"$\chi_{KK}$")
        ax.set_title(ch)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.text(
            0.04,
            0.96,
            f"cGW={r['chi_cgw'].real:+.8g}\nfit={r['chi_extrapolated'].real:+.8g}\nrel.err={r['relerr']:.3e}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    fig.suptitle("finite-box SC-GW derivative vs full static cGW", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    source = _load(args.benchmark)
    V = _scalar(source, "V")
    filling = _scalar(source, "filling")
    T = _scalar(source, "temperature")
    ti = _scalar(source, "ti")
    t1 = _scalar(source, "t1")
    t2 = _scalar(source, "t2")
    nOmega = int(np.max(np.abs(np.asarray(source["gg_m_values"], dtype=int))))
    h_values = np.asarray(sorted(set(float(x) for x in args.h_values), reverse=True))
    if h_values.size < 2 or np.any(h_values <= 0.0):
        raise ValueError("--h-values requires at least two positive values")

    params = RubyParameters(ti=ti, t1=t1, t2=t2, V=V)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=int(args.nw), nOmega=nOmega, T=T)
    channels = tuple(canonical_channel_name(x) for x in args.channels)
    checkpoint = _checkpoint_path(args, source, params, grid, filling)
    seed, meta, _ = load_supercell_checkpoint(checkpoint, params, grid, float(filling))
    if not bool(meta.get("converged", False)) or abs(float(meta.get("source", 0.0))) > 1e-14:
        raise ValueError("need a converged zero-source production checkpoint")

    base_h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    opts = _gw_options(args, seed.mu)

    print("=== exact discrete cGW/FDT validation ===")
    print(f"production seed: {checkpoint}")
    print(f"V={V:g}, filling label={filling:g}, T={T:g}, nw={grid.nw}, nOmega={grid.nOmega}")
    print(f"fixed mu={seed.mu:.12g}")
    print("re-equilibrating production checkpoint to the finite-box GW map ...")
    zero = solve_matrix_gw_finite_box(base_h0, Vq, grid, opts=opts, initial=seed)
    if not zero.converged:
        raise RuntimeError(f"finite-box zero-source solve failed: residual={zero.final_error:.3e}")
    print(
        f"finite-box zero source converged: it={zero.iterations}, residual={zero.final_error:.3e}, "
        f"N_box={np.sum(zero.density):.10f}"
    )

    results = {}
    csv_rows = []
    for ch in channels:
        K, label, _ = supercell_pseudospin_harmonic_vertex(ch, "q0")
        if abs(np.trace(K)) > 1e-12:
            raise ValueError(f"channel {ch} is not traceless")
        chi_cgw, vertex = _cgw_scalar(zero.G, zero.W, Vq, K, grid, args)
        m0 = bilinear_expectation(zero.G, K, grid)
        print("\n" + "-" * 76)
        print(f"{ch} ({label}): cGW={chi_cgw.real:+.12e}, vertex residual={vertex.final_error:.3e}")

        fd_values = []
        rows = []
        for h in h_values:
            vals = []
            states = []
            runtimes = []
            for sign in (+1.0, -1.0):
                hs = sign * float(h)
                h0 = add_bilinear_source(base_h0, K, hs)
                t0 = time.perf_counter()
                gw = solve_matrix_gw_finite_box(h0, Vq, grid, opts=opts, initial=zero)
                runtimes.append(time.perf_counter() - t0)
                if not gw.converged:
                    raise RuntimeError(f"finite-box source solve failed at h={hs:+g}: residual={gw.final_error:.3e}")
                states.append(gw)
                vals.append(bilinear_expectation(gw.G, K, grid))
            chi_fd = central_finite_difference(vals[0], vals[1], h)
            fd_values.append(chi_fd)
            even = 0.5 * (vals[0] + vals[1]) - m0
            rel = relative_error(chi_cgw, chi_fd)
            row = {
                "channel": ch,
                "h": float(h),
                "chi_fd_re": float(chi_fd.real),
                "chi_fd_im": float(chi_fd.imag),
                "chi_cgw_re": float(chi_cgw.real),
                "chi_cgw_im": float(chi_cgw.imag),
                "relative_error": float(rel),
                "even_offset_abs": float(abs(even)),
                "residual_plus": float(states[0].final_error),
                "residual_minus": float(states[1].final_error),
                "runtime_plus_s": float(runtimes[0]),
                "runtime_minus_s": float(runtimes[1]),
            }
            rows.append(row)
            csv_rows.append(row)
            print(
                f"  h={h:g}: FD={chi_fd.real:+.12e}, rel.err={rel:.3e}, "
                f"even-offset={abs(even):.3e}, residuals=({states[0].final_error:.1e},{states[1].final_error:.1e})"
            )

        ext, slope = quadratic_zero_source_extrapolation(h_values, fd_values)
        rel_ext = relative_error(chi_cgw, ext)
        print(f"  h->0: FD={ext.real:+.12e}, cGW={chi_cgw.real:+.12e}, rel.err={rel_ext:.3e}")
        results[ch] = {
            "chi_cgw": chi_cgw,
            "chi_extrapolated": ext,
            "slope": slope,
            "relerr": rel_ext,
            "rows": rows,
        }

    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        source_benchmark=np.asarray(str(args.benchmark)),
        checkpoint=np.asarray(str(checkpoint)),
        discretization=np.asarray("finite-box"),
        V=float(V),
        filling_label=float(filling),
        temperature=float(T),
        mu=float(zero.mu),
        nw=int(grid.nw),
        nOmega=int(grid.nOmega),
        channels=np.asarray(channels),
        h_values=h_values,
        chi_cgw=np.asarray([results[ch]["chi_cgw"] for ch in channels]),
        chi_fd=np.asarray([[complex(row["chi_fd_re"], row["chi_fd_im"]) for row in results[ch]["rows"]] for ch in channels]),
        chi_extrapolated=np.asarray([results[ch]["chi_extrapolated"] for ch in channels]),
        relative_error_extrapolated=np.asarray([results[ch]["relerr"] for ch in channels]),
        finite_box_zero_density=np.asarray(zero.density),
        finite_box_zero_residual=float(zero.final_error),
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
            f"{ch:12s}: cGW={results[ch]['chi_cgw'].real:+.12e}, "
            f"FD(h->0)={results[ch]['chi_extrapolated'].real:+.12e}, "
            f"rel.err={results[ch]['relerr']:.3e}"
        )
    print(
        "If this finite-box error is much smaller than the production-tail finite-source error, "
        "the latter is a Matsubara-tail discretization mismatch rather than a missing cGW diagram."
    )


if __name__ == "__main__":
    main()
