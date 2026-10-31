#!/usr/bin/env python3
"""Strong-coupling diagnostics directly from microscopic Ruby-lattice ED data.

This postprocessor does NOT diagonalize or compare to an effective pseudospin
Hamiltonian. It asks two simpler questions of an existing fixed-filling
scan_susceptibility.py data set:

1. Do the microscopic low-energy gaps scale as 1/V at strong coupling?
   We collect the globally ordered levels from each V_*/sectors.json and plot
   both Delta E_n and V*Delta E_n.

2. Does the microscopic ground state develop staggered correlations in the
   local triangle E-doublet pseudospin operators?

The local operators are Pauli-normalized proxies whose projection into the
one-fermion E doublet is exactly sigma_x,y,z:

  A: tau_x = 2 n0 - n1 - n2
     tau_y = sqrt(3) (n2 - n1)
     tau_z = -sqrt(3) I_A^CCW / t

  B: tau_x = 2 n3 - n4 - n5
     tau_y = sqrt(3) (n4 - n5)
     tau_z = +sqrt(3) I_B^CCW / t

The opposite signs in tau_z (and the B tau_y convention) account for the
opposite geometric handedness of the two triangles. I_A/B^CCW is the physical
counterclockwise triangle current averaged over the three edges, using the same
operator as lc_structure.py.

For the triangle-center bipartite lattice, epsilon_p=+1 on A and -1 on B,

  S_AF^a = (1/Ntri) sum_pq epsilon_p epsilon_q <tau_p^a tau_q^a>,

and m_AF,a^2 = S_AF^a / Ntri. Both raw and connected versions are saved.
The primary finite-size diagnostic is the raw ground-multiplet average. For a
unique symmetry-preserving ground state raw and connected are identical up to
roundoff.

Run from ed_benchmark/native_ed, for example:

  python analyze_strong_coupling.py --input results\\chi_signed_n2 --Vmin 1 --levels 8

Saved ground_states.npz files are reused. If a point lacks one, only that V is
re-solved unless --cache-only is supplied.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry import RubyPBC
from core import make_states
from lc_structure import triangle_index, one_triangle_action
from plot_current_pattern import load_scan_metadata, load_or_compute_ground
from scan_susceptibility import site_occupations, unique_dir


SQRT3 = float(np.sqrt(3.0))


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _scan_info(root: Path):
    meta = root / "scan_metadata.json"
    if not meta.exists():
        raise FileNotFoundError(f"No scan_metadata.json in {root}")
    info = json.loads(meta.read_text(encoding="utf-8"))
    needed = {"cluster", "t1", "t2", "nf"}
    missing = sorted(needed - set(info))
    if missing:
        raise KeyError(f"scan metadata missing {missing}")
    return info


def _available_points(root: Path):
    out = []
    for p in root.glob("V_*/responses.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append((float(d["V"]), p.parent))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    out.sort(key=lambda x: x[0])
    return out


def _select_values(points, requested=None, vmin=None, vmax=None):
    avail = np.array([v for v, _ in points], dtype=float)
    if requested is not None:
        chosen = []
        for x in requested:
            if not len(avail):
                raise ValueError("No completed V points")
            j = int(np.argmin(abs(avail - float(x))))
            if abs(avail[j] - float(x)) > 1e-8:
                raise ValueError(f"Requested V={x:g} is not a completed scan point")
            chosen.append(float(avail[j]))
    else:
        chosen = list(map(float, avail))
    if vmin is not None:
        chosen = [v for v in chosen if v >= float(vmin) - 1e-12]
    if vmax is not None:
        chosen = [v for v in chosen if v <= float(vmax) + 1e-12]
    chosen = sorted(set(chosen))
    if not chosen:
        raise ValueError("No V values remain after selection")
    return chosen


def _global_levels_from_sectors(path: Path):
    """Return globally sorted low energies, including symmetry degeneracies."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    levels = []
    for rec in data:
        vals = rec.get("energies", rec.get("eigenvalues", []))
        levels.extend(float(x) for x in vals)
    if not levels:
        raise ValueError(f"No low eigenvalues in {path}")
    return np.array(sorted(levels), dtype=float)


def spectrum_row(folder: Path, V: float, levels: int):
    response = json.loads((folder / "responses.json").read_text(encoding="utf-8"))
    row0 = response["row"]
    e0 = float(row0["E0"])
    energies = _global_levels_from_sectors(folder / "sectors.json")
    # Use the E0 recorded by the scan; sectors should agree.
    if abs(float(energies[0]) - e0) > 2e-7:
        raise AssertionError(f"E0 mismatch at V={V:g}: sectors={energies[0]}, response={e0}")
    out = {
        "V": float(V),
        "E0": e0,
        "ground_dim": int(row0.get("ground_dim", 1)),
        "gap_above_manifold": float(row0.get("gap_above_manifold", np.nan)),
        "V_gap_above_manifold": float(V * float(row0.get("gap_above_manifold", np.nan))),
    }
    for n in range(1, levels + 1):
        gap = float(energies[n] - e0) if n < len(energies) else np.nan
        out[f"gap_E{n}"] = gap
        out[f"V_gap_E{n}"] = float(V * gap) if np.isfinite(gap) else np.nan
    return out


def _triangle_xy_amplitudes(g, states, occ):
    """Diagonal microscopic tau_x/tau_y amplitudes for every triangle."""
    loops, cells, sub = triangle_index(g)
    amp = np.empty((len(loops), 2, len(states)), dtype=float)
    for p, (cell, ab) in enumerate(zip(cells, sub)):
        base = 6 * int(cell)
        if int(ab) == 0:
            sites = [base + 0, base + 1, base + 2]
            wx = np.array([2.0, -1.0, -1.0])
            wy = np.array([0.0, -SQRT3, +SQRT3])
        else:
            sites = [base + 3, base + 4, base + 5]
            wx = np.array([2.0, -1.0, -1.0])
            # Physical B frame: opposite geometric handedness.
            wy = np.array([0.0, +SQRT3, -SQRT3])
        amp[p, 0] = occ[:, sites] @ wx
        amp[p, 1] = occ[:, sites] @ wy
    return loops, cells, sub, amp


def pseudospin_actions(g, states, psi, xy_cache=None):
    """Return columns O_{p,a}|psi> in order (p,x),(p,y),(p,z)."""
    states = np.asarray(states, dtype=np.uint64)
    psi = np.asarray(psi, dtype=complex)
    if xy_cache is None:
        occ = site_occupations(states, g.Ns)
        loops, cells, sub, amp = _triangle_xy_amplitudes(g, states, occ)
    else:
        loops, cells, sub, amp = xy_cache
    nt = len(loops)
    out = np.empty((len(states), 3 * nt), dtype=complex)
    if abs(float(g.t["t"])) < 1e-14:
        raise ValueError("Pauli-normalized tau_z requires nonzero intra-triangle t")
    zscale = SQRT3 / float(g.t["t"])
    for p, loop in enumerate(loops):
        out[:, 3 * p + 0] = amp[p, 0] * psi
        out[:, 3 * p + 1] = amp[p, 1] * psi
        # Physical-CCW I has opposite sign relative to local E chirality on A,
        # and the same sign on B.
        frame = -1.0 if int(sub[p]) == 0 else +1.0
        out[:, 3 * p + 2] = frame * zscale * one_triangle_action(
            g, states, psi, loop
        )
    return out, (loops, cells, sub, amp)


def pseudospin_correlations(g, states, psis):
    """Ground-multiplet microscopic pseudospin Gram matrices and AF projections."""
    states = np.asarray(states, dtype=np.uint64)
    psis = np.asarray(psis, dtype=complex)
    if psis.ndim != 2 or psis.shape[0] != len(states):
        raise ValueError("Invalid ground-state matrix")
    d = psis.shape[1]
    if d < 1:
        raise ValueError("Empty ground manifold")
    if np.linalg.norm(psis.conj().T @ psis - np.eye(d)) > 2e-7:
        raise ValueError("Ground states are not orthonormal")

    occ = site_occupations(states, g.Ns)
    loops, cells, sub, amp = _triangle_xy_amplitudes(g, states, occ)
    cache = (loops, cells, sub, amp)
    nt = len(loops)
    nop = 3 * nt
    gram = np.zeros((nop, nop), dtype=complex)
    mean = np.zeros(nop, dtype=complex)

    for psi in psis.T:
        A, _ = pseudospin_actions(g, states, psi, cache)
        gram += (A.conj().T @ A) / d
        mean += (psi.conj() @ A) / d

    gram = 0.5 * (gram + gram.conj().T)
    if np.max(np.abs(gram.imag)) > 2e-7:
        # The full matrix can contain symmetry-allowed imaginary xy pieces,
        # but same-component AF projections below must be real.
        pass
    conn = gram - np.outer(mean.conj(), mean)
    conn = 0.5 * (conn + conn.conj().T)

    eps = np.where(sub == 0, 1.0, -1.0)
    uniform = np.ones(nt, dtype=float)

    def project(matrix, signs, axis):
        v = np.zeros(nop, dtype=complex)
        v[axis::3] = signs / np.sqrt(float(nt))
        z = np.vdot(v, matrix @ v)
        if abs(z.imag) > 1e-7 * (1 + abs(z.real)):
            raise AssertionError(f"Non-real structure factor: {z}")
        return float(z.real)

    names = ["x", "y", "z"]
    out = {
        "ground_dim": int(d),
        "Ntriangle": int(nt),
        "tau_mean_max_abs": float(np.max(np.abs(mean))),
    }
    for a, name in enumerate(names):
        out[f"S_AF_{name}"] = project(gram, eps, a)
        out[f"S_AF_{name}_conn"] = project(conn, eps, a)
        out[f"S_uniform_{name}"] = project(gram, uniform, a)
        out[f"S_uniform_{name}_conn"] = project(conn, uniform, a)

    out["S_AF_total"] = sum(out[f"S_AF_{a}"] for a in names)
    out["S_AF_avg"] = out["S_AF_total"] / 3.0
    out["S_AF_total_conn"] = sum(out[f"S_AF_{a}_conn"] for a in names)
    out["S_AF_avg_conn"] = out["S_AF_total_conn"] / 3.0
    out["m2_AF_total"] = out["S_AF_total"] / nt
    out["m2_AF_avg"] = out["S_AF_avg"] / nt
    out["m2_AF_z"] = out["S_AF_z"] / nt

    # Average microscopic local Pauli-proxy moment. In the ideal E doublet,
    # sum_a <tau_a^2> = 3.
    local_tau2 = 0.0
    for p in range(nt):
        for a in range(3):
            local_tau2 += float(gram[3 * p + a, 3 * p + a].real)
    out["local_tau2_avg"] = local_tau2 / nt

    # Strong-coupling quality: probability that each triangle contains exactly
    # one fermion, plus the average single-triangle probability.
    probs = np.mean(np.abs(psis) ** 2, axis=1)
    one = np.empty((len(states), nt), dtype=bool)
    for p, (cell, ab) in enumerate(zip(cells, sub)):
        base = 6 * int(cell) + (0 if int(ab) == 0 else 3)
        one[:, p] = np.sum(occ[:, base:base + 3], axis=1) == 1
    out["P_Ntriangle_eq_1_avg"] = float(
        np.mean(np.sum(probs[:, None] * one, axis=0))
    )
    out["P_all_triangles_N_eq_1"] = float(np.sum(probs * np.all(one, axis=1)))

    return out, {
        "C_raw": gram,
        "C_connected": conn,
        "mean": mean,
        "triangle_cell": cells,
        "triangle_sublattice": sub,
        "loops": np.asarray(loops, dtype=int),
    }


def _plot_spectrum(rows, out):
    rows = sorted(rows, key=lambda x: x["V"])
    V = np.array([r["V"] for r in rows], float)
    keys = [k for k in rows[0] if k.startswith("gap_E") and not k.startswith("gap_above")]
    keys.sort(key=lambda s: int(s[5:]))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key in keys:
        ax.plot(V, [r[key] for r in rows], "-o", ms=3.5, label=key.replace("_", " "))
    ax.set_xlabel("V")
    ax.set_ylabel(r"$E_n-E_0$")
    ax.set_title("Microscopic ED low-energy gaps")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "spectrum_raw_vs_V.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key in keys:
        vkey = "V_" + key
        ax.plot(V, [r[vkey] for r in rows], "-o", ms=3.5,
                label=r"$V\," + key.replace("gap_E", r"\Delta E_") + "$")
    ax.set_xlabel("V")
    ax.set_ylabel(r"$V(E_n-E_0)$")
    ax.set_title(r"Test of $1/V$ low-energy spectral collapse")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "spectrum_V_scaled.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = 1.0 / V
    for key in keys:
        ax.plot(x, [r["V_" + key] for r in rows], "o-", ms=3.5,
                label=key.replace("gap_E", "level "))
    ax.set_xlabel(r"$1/V$")
    ax.set_ylabel(r"$V(E_n-E_0)$")
    ax.set_title("Strong-coupling extrapolation: plateau at 1/V -> 0")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "spectrum_collapse_vs_inverseV.png", dpi=190)
    plt.close(fig)


def _plateau_diagnostics(rows, fit_vmin):
    use = [r for r in rows if r["V"] >= fit_vmin - 1e-12]
    if len(use) < 2:
        return []
    x = np.array([1.0 / r["V"] for r in use], float)
    keys = [k for k in rows[0] if k.startswith("gap_E") and not k.startswith("gap_above")]
    keys.sort(key=lambda s: int(s[5:]))
    out = []
    for key in keys:
        y = np.array([r["V_" + key] for r in use], float)
        mask = np.isfinite(y)
        if np.count_nonzero(mask) < 2:
            continue
        # V*Delta = c0 + c1/V is the simplest correction-to-scaling check.
        c1, c0 = np.polyfit(x[mask], y[mask], 1)
        pred = c0 + c1 * x[mask]
        rms = float(np.sqrt(np.mean((y[mask] - pred) ** 2)))
        spread = float(np.std(y[mask]) / (abs(np.mean(y[mask])) + 1e-30))
        out.append({
            "level": key,
            "Vmin_fit": float(fit_vmin),
            "points": int(np.count_nonzero(mask)),
            "plateau_intercept_1overV_to_0": float(c0),
            "correction_slope": float(c1),
            "rms_fit": rms,
            "relative_scaled_spread": spread,
        })
    return out


def _plot_pseudospin(rows, out):
    rows = sorted(rows, key=lambda x: x["V"])
    V = np.array([r["V"] for r in rows], float)

    fig, ax = plt.subplots(figsize=(7.3, 4.9))
    for a in ("x", "y", "z"):
        ax.plot(V, [r[f"S_AF_{a}"] for r in rows], "-o", ms=4,
                label=rf"$S_{{AF}}^{a}$")
    ax.plot(V, [r["S_AF_avg"] for r in rows], "--o", ms=4,
            label=r"$(S_x+S_y+S_z)/3$")
    ax.set_xlabel("V")
    ax.set_ylabel("staggered pseudospin structure factor")
    ax.set_title("Microscopic Pauli-proxy staggered pseudospin correlations")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pseudospin_staggered_vs_V.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 4.9))
    ax.plot(V, [r["m2_AF_avg"] for r in rows], "-o", label=r"$S_{AF}^{avg}/N_{tri}$")
    ax.plot(V, [r["m2_AF_z"] for r in rows], "-o", label=r"$S_{AF}^{z}/N_{tri}$")
    ax.plot(V, [r["P_Ntriangle_eq_1_avg"] for r in rows], "--o",
            label=r"$\langle P_{N_\triangle=1}\rangle$")
    ax.plot(V, [r["P_all_triangles_N_eq_1"] for r in rows], "--o",
            label=r"$P(\forall\triangle:N_\triangle=1)$")
    ax.set_xlabel("V")
    ax.set_ylabel("dimensionless")
    ax.set_title("AF order-density diagnostic and strong-coupling subspace weight")
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "pseudospin_order_density_and_projection.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 4.9))
    ax.plot(V, [r["local_tau2_avg"] for r in rows], "-o")
    ax.axhline(3.0, linestyle="--", linewidth=1.2, label="ideal Pauli E-doublet = 3")
    ax.set_xlabel("V")
    ax.set_ylabel(r"$N_{tri}^{-1}\sum_p\langle\tau_p^2\rangle$")
    ax.set_title("Local pseudospin-moment normalization")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pseudospin_local_moment.png", dpi=190)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Existing scan_susceptibility.py root")
    p.add_argument("--V", nargs="+", type=float, help="Selected completed V values")
    p.add_argument("--Vmin", type=float, default=None)
    p.add_argument("--Vmax", type=float, default=None)
    p.add_argument("--levels", type=int, default=8,
                   help="Number of ordered excited levels to include")
    p.add_argument("--fit-Vmin", type=float, default=None,
                   help="Vmin for V*gap = c0+c1/V plateau fits; default selected midpoint")
    p.add_argument("--output", help="Default: <input>/strong_coupling_diagnostics")
    p.add_argument("--cache-only", action="store_true",
                   help="Require saved ground_states.npz; never re-run ED")
    p.add_argument("--force-resolve", action="store_true",
                   help="Ignore saved ground_states.npz and solve selected points again")
    p.add_argument("--low", type=int, default=None,
                   help="Override symmetry-block low-state count if re-solving")
    p.add_argument("--max-basis", type=int, default=70000)
    a = p.parse_args(argv)

    if a.levels < 1:
        raise ValueError("--levels must be positive")
    root = Path(a.input)
    info = _scan_info(root)
    g = RubyPBC(info["cluster"], t=info.get("t", 1.0),
                t1=info["t1"], t2=info["t2"])
    if not np.isclose(float(g.t["t1"]), -float(g.t["t2"]), atol=1e-10, rtol=1e-10):
        print("WARNING: t1 != -t2; the diagnostics are still valid but this is not the signed-hopping case.")
    nf = int(info["nf"])
    if not np.isclose(nf / g.num_cells, 2.0, atol=1e-12):
        print("WARNING: filling is not n=2; Pauli-proxy operators remain microscopic observables, "
              "but the one-fermion-per-triangle strong-coupling interpretation is not expected.")

    points = _available_points(root)
    vs = _select_values(points, a.V, a.Vmin, a.Vmax)
    out = Path(a.output) if a.output else root / "strong_coupling_diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    spectrum = []
    pseudo = []
    cache = None
    for V in vs:
        folder, config, V0, existing_row = load_scan_metadata(root, V)
        spectrum.append(spectrum_row(folder, V, a.levels))

        ground_path = folder / "ground_states.npz"
        if a.cache_only and not ground_path.exists():
            raise FileNotFoundError(
                f"V={V:g} has no {ground_path}; rerun the scan with --save-state "
                "or omit --cache-only to solve only this point."
            )
        cfg = dict(info, max_basis=a.max_basis)
        if a.low is not None:
            cfg["low"] = int(a.low)
        gg, states, psis, e0, origin, cache = load_or_compute_ground(
            folder, cfg, V, cache, a.force_resolve
        )
        if gg.name != g.name:
            raise AssertionError("Geometry changed inside one scan")
        vals, matrices = pseudospin_correlations(g, states, psis)
        row = {"V": float(V), "E0": float(e0), "wavefunctions": origin, **vals}
        pseudo.append(row)
        point_out = out / f"V_{V:+.9f}"
        point_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            point_out / "pseudospin_structure.npz",
            V=float(V), E0=float(e0), Nf=nf, cluster=g.name,
            t=float(g.t["t"]), t1=float(g.t["t1"]), t2=float(g.t["t2"]),
            **matrices,
        )
        print(
            f"V={V:g}: V*gap={spectrum[-1].get('V_gap_above_manifold', np.nan):.7g}, "
            f"S_AF=(x,y,z)=({vals['S_AF_x']:.6g},{vals['S_AF_y']:.6g},{vals['S_AF_z']:.6g}), "
            f"S_AF_avg={vals['S_AF_avg']:.6g}, "
            f"P1(avg)={vals['P_Ntriangle_eq_1_avg']:.6g}, "
            f"P1(all)={vals['P_all_triangles_N_eq_1']:.6g} [{origin}]",
            flush=True,
        )

    _write_csv(out / "spectrum_collapse.csv", spectrum)
    _write_csv(out / "pseudospin_staggered.csv", pseudo)
    _plot_spectrum(spectrum, out)
    _plot_pseudospin(pseudo, out)

    fit_vmin = a.fit_Vmin
    if fit_vmin is None:
        fit_vmin = vs[len(vs) // 2]
    plateau = _plateau_diagnostics(spectrum, float(fit_vmin))
    _write_csv(out / "spectrum_plateau_fits.csv", plateau)

    summary = {
        "input": str(root),
        "cluster": g.name,
        "Nf": nf,
        "n_per_primitive_cell": nf / g.num_cells,
        "t": g.t,
        "V_values": vs,
        "pseudospin_normalization": "Pauli: projected E-doublet eigenvalues +/-1",
        "staggered_sign": "A=+1, B=-1 on triangle-center bipartite lattice",
        "S_AF_definition": "(1/Ntri) sum_pq eps_p eps_q <tau_p tau_q>",
        "spin_half_conversion": "For T=tau/2, divide every S component by 4.",
        "fit_Vmin": float(fit_vmin),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
