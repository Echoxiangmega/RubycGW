#!/usr/bin/env python3
"""Coarse finite-T 18-site Ruby ED soft-mode map in the (V,filling) plane.

The 18-site index-three torus is solved in the canonical fixed-N ensemble.
Finite-temperature traces use the existing stochastic thermal-typicality /
Krylov implementation.  The Hamiltonian itself is the exact 18-site many-body
Hamiltonian, including V, Vprime and Vcross.

Because the three-cell torus contains only Gamma and +/-Q with
Q=(1/3,1/3), this diagnostic cannot detect an M-point instability.

For visualization only, a point is labelled Normal when the largest physical
pseudospin susceptibility eigenvalue over Gamma,Q is below --chi-threshold.
Otherwise the label is taken from the dominant content of the leading soft
eigenvector.  This finite-cluster threshold is a plotting convention, not a
thermodynamic phase-transition criterion.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from rubycgw.ed18 import ED18_CHANNELS, ED18Solver, Q_GAMMA, Q_PERIOD3
from rubycgw.ed18_thermal import random_phase_trace_vectors, thermal_response
from rubycgw.models.ruby import ExtendedRubyParameters
from rubycgw.supercell import NSECTOR


PHASE_NAMES = np.asarray(
    ["Normal", "CO-even", "CO-odd", "LC-same", "LC-opposite"],
    dtype="<U16",
)
PHASE_CODES = {name: i for i, name in enumerate(PHASE_NAMES)}

# ED18_CHANNELS =
# x_even, x_odd, y_even, y_odd, z_same, z_opposite
CO_EVEN = (0, 2)
CO_ODD = (1, 3)
LC_SAME = (4,)
LC_OPPOSITE = (5,)


def _csv_floats(text: str) -> list[float]:
    out = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return out


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--V-values",
        type=_csv_floats,
        default=_csv_floats("0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8"),
    )
    p.add_argument(
        "--fillings",
        type=_csv_floats,
        default=_csv_floats("1,1.3333333333333333,1.6666666666666667,2,2.3333333333333335,2.6666666666666665,3"),
        help="Canonical fillings; 3*filling must be an integer on the 18-site torus.",
    )
    p.add_argument("--temperature", "--T", type=float, default=0.01)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--Vprime", "--Vp", type=float, default=-0.1)
    p.add_argument("--Vcross", "--Vx", type=float, default=-0.07)
    p.add_argument("--thermal-samples", type=int, default=4)
    p.add_argument("--tau-points", type=int, default=17)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--eig-tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=5000)
    p.add_argument(
        "--chi-threshold",
        type=float,
        default=20.0,
        help="Heuristic Normal/soft threshold for max susceptibility.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/ed18_phase_T001/ed18_softmode_phase.npz"),
    )
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--rerun", action="store_true")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Reuse saved raw susceptibilities and only reclassify/replot.",
    )
    return p.parse_args()


def _validate_fillings(fillings):
    out = []
    for n in fillings:
        nf = int(round(float(n) * NSECTOR))
        if abs(float(n) * NSECTOR - nf) > 1e-10:
            raise ValueError(
                f"filling={n:g} is not allowed in canonical 18-site ED: "
                "3*filling must be an integer"
            )
        if not 0 <= nf <= 18:
            raise ValueError(f"invalid particle number Nf={nf}")
        out.append(nf / float(NSECTOR))
    return np.asarray(out, dtype=float)


def _classify_one(chi_max: float, vec: np.ndarray, threshold: float):
    w = np.abs(np.asarray(vec, dtype=complex)) ** 2
    total = float(np.sum(w))
    if total > 0:
        w = w / total
    grouped = {
        "CO-even": float(np.sum(w[list(CO_EVEN)])),
        "CO-odd": float(np.sum(w[list(CO_ODD)])),
        "LC-same": float(np.sum(w[list(LC_SAME)])),
        "LC-opposite": float(np.sum(w[list(LC_OPPOSITE)])),
    }
    if float(chi_max) < float(threshold):
        name = "Normal"
    else:
        name = max(grouped, key=grouped.get)
    return PHASE_CODES[name], name, w, grouped


def _classify_grid(chi_max, lead_vec, threshold):
    shape = chi_max.shape
    code = np.zeros(shape, dtype=int)
    channel_weights = np.zeros(shape + (len(ED18_CHANNELS),), dtype=float)
    group_weights = np.zeros(shape + (4,), dtype=float)
    group_names = ("CO-even", "CO-odd", "LC-same", "LC-opposite")
    for idx in np.ndindex(shape):
        c, _, w, grouped = _classify_one(chi_max[idx], lead_vec[idx], threshold)
        code[idx] = c
        channel_weights[idx] = w
        group_weights[idx] = [grouped[x] for x in group_names]
    return code, channel_weights, group_weights


def _centers_to_edges(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        return np.asarray([x[0] - 0.5, x[0] + 0.5])
    mids = 0.5 * (x[:-1] + x[1:])
    return np.concatenate([[x[0] - (mids[0] - x[0])], mids, [x[-1] + (x[-1] - mids[-1])]])


def _plot(out, V, fillings, chi_max, q_soft, phase_code, threshold, temperature, dpi):
    phase_png = out.with_name(out.stem + f"_phase_chi{threshold:g}.png")
    chi_png = out.with_name(out.stem + "_chi_max.png")

    colors = ["#eeeeee", "#5aa9e6", "#247ba0", "#ef8354", "#d1495b"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(PHASE_NAMES) + 0.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    im = ax.pcolormesh(
        _centers_to_edges(V),
        _centers_to_edges(fillings),
        phase_code,
        shading="flat",
        cmap=cmap,
        norm=norm,
    )
    for iy, n in enumerate(fillings):
        for ix, v in enumerate(V):
            qtxt = r"$\Gamma$" if int(q_soft[iy, ix]) == 0 else "Q"
            ax.text(v, n, qtxt, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(PHASE_NAMES)))
    cbar.ax.set_yticklabels(PHASE_NAMES.tolist())
    ax.set_xlabel("V")
    ax.set_ylabel("primitive filling n")
    ax.set_title(
        f"18-site finite-T ED soft-mode map, T={temperature:g}, "
        rf"$\chi_{{\rm th}}$={threshold:g}"
    )
    fig.tight_layout()
    fig.savefig(phase_png, dpi=int(dpi))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    im = ax.pcolormesh(
        _centers_to_edges(V),
        _centers_to_edges(fillings),
        chi_max,
        shading="flat",
    )
    fig.colorbar(im, ax=ax, label=r"$\chi_{\max}$")
    if len(V) >= 2 and len(fillings) >= 2:\n        ax.contour(V, fillings, chi_max, levels=[float(threshold)], linewidths=1.4)
    ax.set_xlabel("V")
    ax.set_ylabel("primitive filling n")
    ax.set_title(f"18-site finite-T ED max susceptibility, T={temperature:g}")
    fig.tight_layout()
    fig.savefig(chi_png, dpi=int(dpi))
    plt.close(fig)
    return phase_png, chi_png


def _write_csv(path, V, fillings, n_particles, chi_max, q_soft, phase_code,
               channel_weights, group_weights):
    csv_path = path.with_suffix(".csv")
    group_names = ("CO-even", "CO-odd", "LC-same", "LC-opposite")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filling", "Nf", "V", "chi_max", "q_soft", "phase",
            *[f"w_{x}" for x in ED18_CHANNELS],
            *[f"W_{x}" for x in group_names],
        ])
        for iy, n in enumerate(fillings):
            for ix, v in enumerate(V):
                w.writerow([
                    f"{n:.12g}",
                    int(n_particles[iy]),
                    f"{v:.12g}",
                    f"{chi_max[iy, ix]:.12g}",
                    "Gamma" if int(q_soft[iy, ix]) == 0 else "Q",
                    str(PHASE_NAMES[int(phase_code[iy, ix])]),
                    *[f"{x:.12g}" for x in channel_weights[iy, ix]],
                    *[f"{x:.12g}" for x in group_weights[iy, ix]],
                ])
    return csv_path


def _save_raw(path, *, V, fillings, n_particles, dimensions, args, done,
              E0, gap, chi_eval, chi_vec, chi_max, q_soft, lead_vec):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(1, dtype=int),
        V=np.asarray(V, dtype=float),
        fillings=np.asarray(fillings, dtype=float),
        n_particles=np.asarray(n_particles, dtype=int),
        dimensions=np.asarray(dimensions, dtype=int),
        temperature=float(args.temperature),
        beta=float(1.0 / args.temperature),
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2),
        Vprime=float(args.Vprime), Vcross=float(args.Vcross),
        ensemble=np.asarray("canonical_fixed_N"),
        thermal_method=np.asarray("random_phase_typicality_Krylov_expm_multiply"),
        thermal_samples=int(args.thermal_samples),
        tau_points=int(args.tau_points),
        random_seed=int(args.seed),
        channels=np.asarray(ED18_CHANNELS),
        q_names=np.asarray(["Gamma", "Q"]),
        q_vectors=np.asarray([Q_GAMMA, Q_PERIOD3]),
        done=np.asarray(done, dtype=bool),
        E0=np.asarray(E0, dtype=float),
        ground_gap=np.asarray(gap, dtype=float),
        thermal_chi_eigenvalues=np.asarray(chi_eval, dtype=float),
        thermal_chi_eigenvectors=np.asarray(chi_vec, dtype=complex),
        chi_max=np.asarray(chi_max, dtype=float),
        q_soft=np.asarray(q_soft, dtype=int),
        leading_soft_vector=np.asarray(lead_vec, dtype=complex),
        cluster_warning=np.asarray(
            "18-site index-3 torus contains Gamma and +/-Q only; M is not commensurate"
        ),
    )


def _load_existing(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def main():
    args = _args()
    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")

    if args.plot_only:
        d = _load_existing(out)
        V = np.asarray(d["V"], dtype=float)
        fillings = np.asarray(d["fillings"], dtype=float)
        chi_max = np.asarray(d["chi_max"], dtype=float)
        q_soft = np.asarray(d["q_soft"], dtype=int)
        lead_vec = np.asarray(d["leading_soft_vector"], dtype=complex)
        phase_code, cw, gw = _classify_grid(chi_max, lead_vec, args.chi_threshold)
        p1, p2 = _plot(
            out, V, fillings, chi_max, q_soft, phase_code,
            args.chi_threshold, float(np.asarray(d["temperature"]).reshape(())), args.dpi,
        )
        csv_path = _write_csv(
            out, V, fillings, np.asarray(d["n_particles"], dtype=int),
            chi_max, q_soft, phase_code, cw, gw,
        )
        print("saved:", p1)
        print("saved:", p2)
        print("saved:", csv_path)
        return

    V = np.asarray(sorted(set(float(x) for x in args.V_values)), dtype=float)
    fillings = _validate_fillings(sorted(set(float(x) for x in args.fillings)))
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.thermal_samples < 1:
        raise ValueError("--thermal-samples must be positive")
    if args.tau_points < 3 or args.tau_points % 2 != 1:
        raise ValueError("--tau-points must be an odd integer >= 3")
    if args.chi_threshold <= 0:
        raise ValueError("--chi-threshold must be positive")

    nfills, nV = len(fillings), len(V)
    nq, nc = 2, len(ED18_CHANNELS)
    done = np.zeros((nfills, nV), dtype=bool)
    E0 = np.full((nfills, nV), np.nan)
    gap = np.full((nfills, nV), np.nan)
    chi_eval = np.full((nfills, nV, nq, nc), np.nan)
    chi_vec = np.full((nfills, nV, nq, nc, nc), np.nan + 1j * np.nan, dtype=complex)
    chi_max = np.full((nfills, nV), np.nan)
    q_soft = np.full((nfills, nV), -1, dtype=int)
    lead_vec = np.full((nfills, nV, nc), np.nan + 1j * np.nan, dtype=complex)
    n_particles = np.asarray(np.rint(fillings * NSECTOR), dtype=int)
    dimensions = np.zeros(nfills, dtype=int)

    if out.exists() and not args.rerun:
        old = _load_existing(out)
        if (
            np.array_equal(np.asarray(old["V"], dtype=float), V)
            and np.allclose(np.asarray(old["fillings"], dtype=float), fillings, rtol=0, atol=1e-12)
            and np.isclose(float(np.asarray(old["temperature"]).reshape(())), args.temperature)
            and np.isclose(float(np.asarray(old["Vprime"]).reshape(())), args.Vprime)
            and np.isclose(float(np.asarray(old["Vcross"]).reshape(())), args.Vcross)
        ):
            done = np.asarray(old["done"], dtype=bool)
            E0 = np.asarray(old["E0"], dtype=float)
            gap = np.asarray(old["ground_gap"], dtype=float)
            chi_eval = np.asarray(old["thermal_chi_eigenvalues"], dtype=float)
            chi_vec = np.asarray(old["thermal_chi_eigenvectors"], dtype=complex)
            chi_max = np.asarray(old["chi_max"], dtype=float)
            q_soft = np.asarray(old["q_soft"], dtype=int)
            lead_vec = np.asarray(old["leading_soft_vector"], dtype=complex)
            dimensions = np.asarray(old["dimensions"], dtype=int)
            print(f"[resume] loaded {np.count_nonzero(done)}/{done.size} completed points")

    params = ExtendedRubyParameters(
        ti=float(args.ti), t1=float(args.t1), t2=float(args.t2), V=0.0,
        Vprime=float(args.Vprime), Vcross=float(args.Vcross),
    )
    qvecs = (Q_GAMMA, Q_PERIOD3)

    print("=== 18-site finite-T ED coarse soft-mode map ===")
    print(f"T={args.temperature:g}, Vp={args.Vprime:g}, Vx={args.Vcross:g}")
    print(f"V={V.tolist()}")
    print(f"fillings={fillings.tolist()}")
    print(f"chi threshold={args.chi_threshold:g}")
    print("canonical fixed-N; available q: Gamma and Q=(1/3,1/3); M absent")

    for iy, n in enumerate(fillings):
        solver = ED18Solver(params, primitive_filling=float(n))
        dimensions[iy] = solver.dimension
        traces = random_phase_trace_vectors(
            solver.dimension,
            int(args.thermal_samples),
            seed=int(args.seed) + 1009 * int(solver.n_particles),
        )
        previous = None
        print(
            f"\n=== filling={n:g}, Nf={solver.n_particles}, dim={solver.dimension} ===",
            flush=True,
        )

        for ix, v in enumerate(V):
            if done[iy, ix] and not args.rerun:
                print(f"[skip] V={v:g}: already done")
                continue

            spec = solver.solve(
                float(v),
                n_eigs=6,
                tol=float(args.eig_tol),
                maxiter=int(args.maxiter),
                v0=previous,
            )
            previous = spec.eigenvectors[:, 0].real.copy()
            E0[iy, ix] = spec.energies[0]
            gap[iy, ix] = spec.gap_above_manifold

            for iq, q in enumerate(qvecs):
                resp = thermal_response(
                    solver,
                    float(v),
                    float(args.temperature),
                    q,
                    trace_vectors=traces,
                    tau_points=int(args.tau_points),
                    energy_shift=E0[iy, ix],
                )
                chi_eval[iy, ix, iq] = resp.susceptibility_eigenvalues
                chi_vec[iy, ix, iq] = resp.susceptibility_eigenvectors

            candidates = chi_eval[iy, ix, :, 0]
            iqstar = int(np.nanargmax(candidates))
            chi_max[iy, ix] = float(candidates[iqstar])
            q_soft[iy, ix] = iqstar
            lead_vec[iy, ix] = chi_vec[iy, ix, iqstar, :, 0]
            code, name, _w, grouped = _classify_one(
                chi_max[iy, ix], lead_vec[iy, ix], args.chi_threshold
            )
            qname = "Gamma" if iqstar == 0 else "Q"
            print(
                f"V={v:g}: chi_max={chi_max[iy,ix]:.6g} @ {qname}, "
                f"label={name}, "
                f"WcoE={grouped['CO-even']:.3f}, WcoO={grouped['CO-odd']:.3f}, "
                f"WlcS={grouped['LC-same']:.3f}, WlcO={grouped['LC-opposite']:.3f}",
                flush=True,
            )
            done[iy, ix] = True
            _save_raw(
                out,
                V=V, fillings=fillings, n_particles=n_particles,
                dimensions=dimensions, args=args, done=done, E0=E0, gap=gap,
                chi_eval=chi_eval, chi_vec=chi_vec, chi_max=chi_max,
                q_soft=q_soft, lead_vec=lead_vec,
            )

    phase_code, channel_weights, group_weights = _classify_grid(
        chi_max, lead_vec, args.chi_threshold
    )
    p1, p2 = _plot(
        out, V, fillings, chi_max, q_soft, phase_code,
        args.chi_threshold, args.temperature, args.dpi,
    )
    csv_path = _write_csv(
        out, V, fillings, n_particles, chi_max, q_soft, phase_code,
        channel_weights, group_weights,
    )
    print("\nsaved:", out)
    print("saved:", p1)
    print("saved:", p2)
    print("saved:", csv_path)
    print(
        "Threshold classification is heuristic.  Replot sensitivity cheaply with e.g. "
        "--plot-only --chi-threshold 15 or 25."
    )


if __name__ == "__main__":
    main()
