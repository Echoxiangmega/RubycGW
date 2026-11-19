#!/usr/bin/env python3
"""Plot site density, bond order, and bond current from saved ED/VMC states.

This reader is designed for outputs from
ed_benchmark/vmc_zeroT/run_projection_comparison.py.

Supported state files
---------------------
ED:
    <result_root>/ed/final_state.npz
    keys: psi_ground_manifold, native_states, ...

Raw/PAV/VAP:
    <result_root>/<method_dir>/final_state.npz
    keys: psi_native, native_states, ...

For a degenerate ED manifold, --mode average evaluates the equal-weight
incoherent ground-manifold density matrix.  This is basis-independent inside
that manifold.  --mode vector selects one saved column instead.

Outputs
-------
site_density.png/.svg
bond_order.png/.svg
bond_current.png/.svg
site_density.csv
bond_observables.csv
summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIVE = REPO / "ed_benchmark" / "native_ed"
sys.path.insert(0, str(NATIVE))

from geometry import RubyPBC, A, ORBITALS  # noqa: E402
from core import edge_triplets  # noqa: E402


SITE_CMAP = "coolwarm"
BOND_CMAP = "RdBu_r"
CURRENT_CMAP = "RdBu_r"


def load_geometry(result_root: Path) -> RubyPBC:
    path = result_root / "geometry.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing geometry.json: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    cluster = meta.get("name", meta.get("cluster"))
    if cluster is None:
        raise KeyError("geometry.json must contain 'name' or 'cluster'")

    # geometry.metadata() stores hopping amplitudes under the nested 't' dict.
    hopping = meta.get("t", {})
    if isinstance(hopping, dict):
        t = float(hopping.get("t", 1.0))
        t1 = float(hopping.get("t1", 0.5))
        t2 = float(hopping.get("t2", 0.5))
    else:
        # Backward compatibility with older flat metadata.
        t = float(meta.get("t", 1.0))
        t1 = float(meta.get("t1", 0.5))
        t2 = float(meta.get("t2", 0.5))
    return RubyPBC(cluster, t=t, t1=t1, t2=t2)


def load_saved_state(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        if "psi_native" in files:
            psi = np.asarray(z["psi_native"], np.complex128)
            states = np.asarray(z["native_states"], np.uint64)
            kind = "vector"
        elif "psi_ground_manifold" in files:
            psi = np.asarray(z["psi_ground_manifold"], np.complex128)
            states = np.asarray(z["native_states"], np.uint64)
            kind = "manifold"
        else:
            raise RuntimeError(
                f"Unrecognized state file {path}: expected psi_native or "
                "psi_ground_manifold"
            )
        metadata = {
            key: np.asarray(z[key]).tolist()
            for key in z.files
            if key not in ("psi_native", "psi_ground_manifold", "native_states")
        }
    return {"kind": kind, "psi": psi, "states": states, "metadata": metadata}


def normalize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.complex128)
    if x.ndim == 1:
        nrm = float(np.linalg.norm(x))
        if nrm < 1e-15:
            raise ValueError("Zero-norm state")
        return x / nrm
    if x.ndim != 2:
        raise ValueError(f"Expected state vector/manifold, got shape {x.shape}")
    out = x.copy()
    norms = np.linalg.norm(out, axis=0)
    if np.any(norms < 1e-15):
        raise ValueError("Ground manifold contains a zero-norm vector")
    out /= norms[None, :]
    return out


def select_representation(payload: dict, mode: str, column: int | None) -> dict:
    states = payload["states"]
    psi = normalize_columns(payload["psi"])

    if payload["kind"] == "vector":
        if mode == "average":
            # A one-dimensional manifold and a pure state are equivalent here.
            mode = "vector"
        return {
            "mode": "vector",
            "psi": psi,
            "states": states,
            "label": "single saved state",
        }

    if mode == "average":
        return {
            "mode": "average",
            "psi": psi,
            "states": states,
            "label": f"equal-weight ED ground-manifold average (d={psi.shape[1]})",
        }

    if column is None:
        column = 0
    if not 0 <= int(column) < psi.shape[1]:
        raise IndexError(
            f"--column {column} outside ground-manifold dimension {psi.shape[1]}"
        )
    return {
        "mode": "vector",
        "psi": psi[:, int(column)],
        "states": states,
        "label": f"ED ground-manifold column {int(column)}",
    }


def site_positions(g: RubyPBC):
    xy = np.zeros((g.Ns, 2), float)
    cells = np.zeros((g.Ns, 2), int)
    for site in range(g.Ns):
        ci, orb = divmod(site, 6)
        cell = np.asarray(g.reps[ci], int)
        cells[site] = cell
        xy[site] = A @ cell + ORBITALS[orb]
    return xy, cells


def edge_segments(g: RubyPBC, xy: np.ndarray, cells: np.ndarray):
    """Short physical representative for each independent PBC hopping bond."""
    starts = np.zeros((len(g.edges), 2), float)
    ends = np.zeros((len(g.edges), 2), float)
    crossed = np.zeros(len(g.edges), bool)
    for n, e in enumerate(g.edges):
        starts[n] = xy[e.i]
        ci = e.i // 6
        ends[n] = (
            A @ (cells[6 * ci] + np.asarray(e.delta, int))
            + ORBITALS[e.j % 6]
        )
        crossed[n] = np.linalg.norm(ends[n] - xy[e.j]) > 1e-8
    return starts, ends, crossed


def mixture_probabilities(rep: dict) -> np.ndarray:
    if rep["mode"] == "vector":
        return np.abs(rep["psi"]) ** 2
    return np.mean(np.abs(rep["psi"]) ** 2, axis=1)


def compute_site_density(g: RubyPBC, states: np.ndarray, rep: dict):
    prob = mixture_probabilities(rep)
    density = np.zeros(g.Ns, float)
    for i in range(g.Ns):
        bit = np.uint64(1 << (g.Ns - 1 - i))
        density[i] = float(prob @ ((states & bit) != 0))
    return density


def edge_rho_single(g: RubyPBC, states: np.ndarray, psi: np.ndarray):
    """rho_e=<c_i^dag c_j> for the stored oriented edge i,j."""
    rho = np.zeros(len(g.edges), np.complex128)
    for n, e in enumerate(g.edges):
        row, col, sign = edge_triplets(g, states, n)
        filled_j = (
            states[col] & np.uint64(1 << (g.Ns - 1 - e.j))
        ) != 0
        rho[n] = np.vdot(
            psi[row[filled_j]],
            sign[filled_j] * psi[col[filled_j]],
        )
    return rho


def compute_edge_observables(g: RubyPBC, states: np.ndarray, rep: dict):
    if rep["mode"] == "vector":
        rho = edge_rho_single(g, states, rep["psi"])
    else:
        manifold = rep["psi"]
        rho = sum(
            (edge_rho_single(g, states, manifold[:, j])
             for j in range(manifold.shape[1])),
            start=np.zeros(len(g.edges), np.complex128),
        ) / manifold.shape[1]

    bond = 2.0 * np.real(rho)
    current = np.empty(len(g.edges), float)
    for n, e in enumerate(g.edges):
        z = rho[n]
        current[n] = float(
            (1j * g.t[e.kind] * (z - np.conjugate(z))).real
        )
    return rho, bond, current


def export_site_csv(path, g, xy, density):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "site", "cell_index", "orbital_in_cell",
            "x", "y", "density",
        ])
        for site in range(g.Ns):
            cell, orb = divmod(site, 6)
            w.writerow([
                site, cell, orb,
                float(xy[site, 0]), float(xy[site, 1]),
                float(density[site]),
            ])


def export_bond_csv(path, g, starts, ends, crossed, rho, bond, current):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "edge_idx", "site_i", "site_j", "kind",
            "x_i", "y_i", "x_j_image", "y_j_image", "crosses_pbc",
            "rho_real", "rho_imag", "bond_order", "current_i_to_j",
        ])
        for n, e in enumerate(g.edges):
            w.writerow([
                n, e.i, e.j, e.kind,
                float(starts[n, 0]), float(starts[n, 1]),
                float(ends[n, 0]), float(ends[n, 1]),
                bool(crossed[n]),
                float(rho[n].real), float(rho[n].imag),
                float(bond[n]), float(current[n]),
            ])


def configure_axes(ax, xy, starts, ends):
    pts = np.vstack((xy, starts, ends))
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    pad = 0.5
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_base_bonds(ax, g, starts, ends, crossed, alpha=0.3):
    for n, e in enumerate(g.edges):
        ax.plot(
            [starts[n, 0], ends[n, 0]],
            [starts[n, 1], ends[n, 1]],
            lw=1.0,
            color="0.50",
            alpha=alpha,
            linestyle="--" if crossed[n] else "-",
            zorder=1,
        )


def draw_sites(ax, xy, values=None, cmap=None, norm=None, size=120):
    if values is None:
        ax.scatter(
            xy[:, 0], xy[:, 1], s=size, facecolors="white",
            edgecolors="black", linewidths=1.0, zorder=5,
        )
        artist = None
    else:
        artist = ax.scatter(
            xy[:, 0], xy[:, 1], s=size,
            c=values, cmap=cmap, norm=norm,
            edgecolors="black", linewidths=1.0, zorder=5,
        )
    for i, p in enumerate(xy):
        ax.text(
            p[0], p[1], str(i),
            ha="center", va="center", fontsize=7.5, zorder=6,
        )
    return artist


def save_both(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), dpi=230, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_site_density(g, xy, starts, ends, crossed, density, out, title):
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    draw_base_bonds(ax, g, starts, ends, crossed)
    nbar = float(np.mean(density))
    dev = max(float(np.max(np.abs(density - nbar))), 1e-12)
    norm = TwoSlopeNorm(vmin=nbar-dev, vcenter=nbar, vmax=nbar+dev)
    sc = draw_sites(
        ax, xy, density, SITE_CMAP, norm,
        size=190 + 900 * np.abs(density - nbar) / max(dev, 1e-12),
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.025)
    cb.set_label(r"$\langle n_i\rangle$")
    ax.set_title(f"{title}\nsite density", fontsize=13)
    configure_axes(ax, xy, starts, ends)
    fig.tight_layout()
    save_both(fig, out / "site_density")


def plot_bond_order(g, xy, starts, ends, crossed, bond, out, title):
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    maxabs = max(float(np.max(np.abs(bond))), 1e-12)
    norm = TwoSlopeNorm(vmin=-maxabs, vcenter=0.0, vmax=maxabs)
    cmap = plt.get_cmap(BOND_CMAP)
    for n, e in enumerate(g.edges):
        ax.plot(
            [starts[n, 0], ends[n, 0]],
            [starts[n, 1], ends[n, 1]],
            color=cmap(norm(bond[n])),
            lw=0.8 + 4.2 * abs(bond[n]) / maxabs,
            linestyle="--" if crossed[n] else "-",
            solid_capstyle="round",
            zorder=2,
        )
    draw_sites(ax, xy, size=105)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.025)
    cb.set_label(r"$2\,\mathrm{Re}\langle c_i^\dagger c_j\rangle$")
    ax.set_title(f"{title}\nbond order", fontsize=13)
    configure_axes(ax, xy, starts, ends)
    fig.tight_layout()
    save_both(fig, out / "bond_order")


def plot_bond_current(
    g, xy, starts, ends, crossed, current, out, title,
    threshold_fraction=0.02,
):
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    draw_base_bonds(ax, g, starts, ends, crossed, alpha=0.22)
    maxabs = max(float(np.max(np.abs(current))), 1e-14)
    norm = TwoSlopeNorm(vmin=-maxabs, vcenter=0.0, vmax=maxabs)
    cmap = plt.get_cmap(CURRENT_CMAP)

    for n, e in enumerate(g.edges):
        j = float(current[n])
        if abs(j) < threshold_fraction * maxabs:
            continue
        start = starts[n].copy()
        end = ends[n].copy()
        if j < 0:
            start, end = end, start
        tail = 0.23 * start + 0.77 * end
        head = 0.77 * start + 0.23 * end
        ax.add_patch(
            patches.FancyArrowPatch(
                tail, head,
                arrowstyle="-|>",
                mutation_scale=10 + 8 * abs(j) / maxabs,
                linewidth=1.0 + 3.0 * abs(j) / maxabs,
                color=cmap(norm(j)),
                linestyle="--" if crossed[n] else "-",
                zorder=4,
            )
        )

    draw_sites(ax, xy, size=105)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.025)
    cb.set_label(r"$J_{i\to j}$")
    ax.set_title(
        f"{title}\nbond current; max |J|={maxabs:.5g}",
        fontsize=13,
    )
    configure_axes(ax, xy, starts, ends)
    fig.tight_layout()
    save_both(fig, out / "bond_current")


def parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--result-root", required=True,
        help="Projection-comparison result directory containing geometry.json."
    )
    p.add_argument(
        "--state-file", default=None,
        help=(
            "Saved final_state.npz. If omitted, use "
            "<result-root>/ed/final_state.npz."
        ),
    )
    p.add_argument(
        "--mode", choices=["average", "vector"], default="average",
        help="How to interpret a degenerate ED ground manifold."
    )
    p.add_argument(
        "--column", type=int, default=None,
        help="Ground-manifold column used with --mode vector."
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Default: <result-root>/ed_maps or sibling <state-dir>_maps."
    )
    p.add_argument(
        "--current-threshold", type=float, default=0.02,
        help="Hide arrows below this fraction of max |current|."
    )
    return p


def main():
    a = parser().parse_args()
    root = Path(a.result_root).resolve()
    state_file = (
        Path(a.state_file).resolve()
        if a.state_file is not None
        else root / "ed" / "final_state.npz"
    )
    if not state_file.exists():
        raise FileNotFoundError(
            f"State file not found: {state_file}\n"
            "For ED, rerun run_projection_comparison.py with --ed-only "
            "after updating the repository."
        )

    if a.output_dir is not None:
        out = Path(a.output_dir).resolve()
    elif state_file.parent.name == "ed":
        out = root / "ed_maps"
    else:
        out = state_file.parent.parent / (state_file.parent.name + "_maps")
    out.mkdir(parents=True, exist_ok=True)

    g = load_geometry(root)
    payload = load_saved_state(state_file)
    rep = select_representation(payload, a.mode, a.column)
    states = np.asarray(rep["states"], np.uint64)
    if len(states) != len(rep["psi"]):
        raise RuntimeError(
            f"Basis/state mismatch: {len(states)} states versus "
            f"wavefunction leading dimension {len(rep['psi'])}"
        )

    xy, cells = site_positions(g)
    starts, ends, crossed = edge_segments(g, xy, cells)
    density = compute_site_density(g, states, rep)
    rho, bond, current = compute_edge_observables(g, states, rep)

    export_site_csv(out / "site_density.csv", g, xy, density)
    export_bond_csv(
        out / "bond_observables.csv",
        g, starts, ends, crossed, rho, bond, current,
    )

    title = f"{g.name}: {rep['label']}"
    plot_site_density(
        g, xy, starts, ends, crossed, density, out, title
    )
    plot_bond_order(
        g, xy, starts, ends, crossed, bond, out, title
    )
    plot_bond_current(
        g, xy, starts, ends, crossed, current, out, title,
        threshold_fraction=float(a.current_threshold),
    )

    summary = {
        "cluster": g.name,
        "Ns": int(g.Ns),
        "state_file": str(state_file),
        "representation": rep["label"],
        "mean_density": float(np.mean(density)),
        "density_min": float(np.min(density)),
        "density_max": float(np.max(density)),
        "max_abs_bond_order": float(np.max(np.abs(bond))),
        "max_abs_current": float(np.max(np.abs(current))),
        "note": (
            "For a unique spinless time-reversal-even ED ground state, "
            "all one-point bond currents should vanish up to numerical noise."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[saved-state maps] state={state_file}")
    print(f"[saved-state maps] representation={rep['label']}")
    print(f"[saved-state maps] output={out}")
    print(
        f"[saved-state maps] n=[{summary['density_min']:.6g},"
        f"{summary['density_max']:.6g}], "
        f"max|B|={summary['max_abs_bond_order']:.6g}, "
        f"max|J|={summary['max_abs_current']:.6g}"
    )


if __name__ == "__main__":
    main()
