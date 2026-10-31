#!/usr/bin/env python3
"""Visualize correlation-loop phases in Ruby ED scan ground states.

Each contractible triangle, square and hexagon is shown on the SAME
real-space lattice map. Phi_p = arg(product_<ij in CCW p> <c_i^dag c_j>).
This is a gauge-invariant correlation phase, NOT a physical magnetic flux.

Examples (from ed_benchmark/native_ed):
  python plot_loop_flux.py --input results\my_2x2_scan --V 1 1.45 2 --channel same
  python plot_loop_flux.py --input results\my_2x2_scan --channel opposite
  python plot_loop_flux.py --input results\my_2x2_scan\V_+1.000000000 --channel same

If --V is omitted for a scan root, all V points with responses.json are used.
Saved ground_states.npz is reused; a missing state is recomputed at that V.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np

from plot_current_pattern import (
    load_scan_metadata, load_or_compute_ground, optimize_state,
    selected_site_positions,
)

LOOP_TYPES = {3: "triangle", 4: "square", 6: "hexagon"}
SHORT_LABELS = {3: "T", 4: "S", 6: "H"}


def find_scan_values(root: Path) -> list[float]:
    """Read the actual scanned V values rather than guessing folder names."""
    values = []
    for path in root.glob("V_*/responses.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        values.append(float(record["V"]))
    if not values:
        raise ValueError(
            f"No V_*/responses.json in {root}; pass --V to solve selected points."
        )
    return sorted(set(values))


def channel_selection_ambiguous(dat, tol=1e-9) -> bool:
    """An identically zero or degenerate maximum does not select a unique state."""
    eig = np.sort(np.asarray(
        dat["projection"][dat["selected_channel"]]["eigenvalues"], dtype=float
    ))
    if eig.size == 0:
        return True
    return abs(eig[-1]) <= tol or (
        eig.size > 1 and abs(eig[-1] - eig[-2]) <= tol
    )


def rows_for_loops(dat, min_product=1e-12):
    """Keep all contractible elementary loops, even with undefined phases."""
    rows = []
    for i, lp in enumerate(dat["loop_data"]):
        length = int(lp["length"])
        if length not in LOOP_TYPES:
            continue
        strength = float(lp["rho_product_abs"])
        phi = float(lp["rho_flux"])
        valid = bool(np.isfinite(phi) and strength > min_product)
        rows.append({
            "loop_id": i,
            "length": length,
            "shape": LOOP_TYPES[length],
            "type": str(lp["type"]),
            "sites": " ".join(map(str, lp["sites"])),
            "phi_rad": phi if valid else float("nan"),
            "abs_phi_rad": abs(phi) if valid else float("nan"),
            "phi_over_pi": phi / np.pi if valid else float("nan"),
            "I_ccw": float(lp["I_ccw"]),
            "rho_product_abs": strength,
            "bare_flux_rad": float(lp["bare_flux"]),
            "valid_phase": valid,
            "polygon": np.asarray(lp["real_coordinates"], dtype=float),
        })
    return rows


def draw_map(g, rows, dat, filename, mode, ambiguous):
    """Overlay triangles, squares and hexagons on one common lattice plot."""
    if mode not in ("abs", "signed"):
        raise ValueError("mode must be 'abs' or 'signed'")
    xy, _, boundary, _, _, _ = selected_site_positions(g)
    norm = (Normalize(0, np.pi) if mode == "abs"
            else TwoSlopeNorm(vmin=-np.pi, vcenter=0, vmax=np.pi))
    cmap = plt.get_cmap("viridis" if mode == "abs" else "RdBu_r")
    sm = ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.add_patch(patches.Polygon(
        boundary, closed=True, fill=False, linestyle="--",
        edgecolor="#6F8292", linewidth=1.7, zorder=1,
    ))
    # Larger faces first, so smaller plaquettes are not hidden by their
    # periodic-image polygons. Values always use the SAME color scale.
    for row in sorted(rows, key=lambda r: (-r["length"], r["loop_id"])):
        vertices = row["polygon"]
        valid = row["valid_phase"]
        value = row["abs_phi_rad"] if mode == "abs" else row["phi_rad"]
        color = sm.to_rgba(value) if valid else (0.8, 0.8, 0.8, 0.7)
        ax.add_patch(patches.Polygon(
            vertices, closed=True, facecolor=color, alpha=0.82,
            edgecolor="#344759", linewidth=1.2, hatch=None if valid else "//",
            zorder=2 + (6 - row["length"]) / 4,
        ))
        label = SHORT_LABELS[row["length"]]
        if row["length"] == 3:
            label += row["type"]
        label += f" {value:+.2f}" if (valid and mode == "signed") else (
            f" {value:.2f}" if valid else " n/a"
        )
        center = vertices.mean(axis=0)
        ax.text(
            center[0], center[1], label, ha="center", va="center",
            fontsize=7.4, fontweight="bold", color="#111827",
            bbox=dict(facecolor="white", alpha=0.56, edgecolor="none", pad=0.5),
            zorder=7,
        )
    for point in xy:
        ax.plot(point[0], point[1], "o", ms=3.6,
                mec="#344759", mfc="white", mew=0.8, zorder=8)

    all_points = [xy, boundary] + [r["polygon"] for r in rows]
    extent = np.concatenate(all_points)
    lo, hi = extent.min(axis=0), extent.max(axis=0)
    ax.set_xlim(lo[0] - 0.55, hi[0] + 0.55)
    ax.set_ylim(lo[1] - 0.55, hi[1] + 0.55)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8).set_label(
        r"$|\Phi_p|$ (rad)" if mode == "abs" else r"$\Phi_p$ (rad, CCW)"
    )
    suffix = " (selection nonunique)" if ambiguous else ""
    ax.set_title(
        f"{g.name}: triangle / square / hexagon loop phases\n"
        f"{dat['selected_channel']}={dat['J_selected']:+.6g}{suffix}",
        fontsize=12,
    )
    fig.text(
        0.5, 0.015,
        r"$\Phi_p=\arg\prod_{(ij)\in p}\langle c_i^\dagger c_j\rangle$"
        "  |  T = triangle, S = square, H = hexagon; n/a = weak product",
        ha="center", fontsize=9,
    )
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_shape(rows, length):
    vals = np.array([
        row["abs_phi_rad"] for row in rows
        if row["length"] == length and row["valid_phase"]
    ], dtype=float)
    return {
        "count": sum(row["length"] == length for row in rows),
        "valid_count": int(vals.size),
        "max_abs_phi": float(np.max(vals)) if vals.size else float("nan"),
        "mean_abs_phi": float(np.mean(vals)) if vals.size else float("nan"),
    }


def export_one(g, states, psis, e0, V, output, channel, check_h, min_product):
    output.mkdir(parents=True, exist_ok=True)
    dat = optimize_state(
        g, states, psis, channel=channel, E0=e0, V=V, check_h=check_h
    )
    ambiguous = channel_selection_ambiguous(dat)
    if ambiguous:
        warnings.warn(
            f"V={V:g}: the LC_{channel} projected maximum does not select "
            "a unique ground state; local Phi values may depend on the "
            "chosen degenerate-state basis."
        )
    rows = rows_for_loops(dat, min_product)
    if not rows:
        raise RuntimeError(f"No elementary contractible loops found at V={V:g}")

    fields = [
        "loop_id", "length", "shape", "type", "sites", "phi_rad",
        "abs_phi_rad", "phi_over_pi", "I_ccw", "rho_product_abs",
        "bare_flux_rad", "valid_phase",
    ]
    with (output / "loop_flux.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    draw_map(g, rows, dat, output / "loop_flux_abs.png", "abs", ambiguous)
    draw_map(g, rows, dat, output / "loop_flux_signed.png", "signed", ambiguous)

    summary = {
        "V": float(V), "cluster": g.name, "nf": int(states[0]).bit_count(),
        "ground_dim": int(psis.shape[1]),
        "channel": dat["selected_channel"],
        "channel_J": float(dat["J_selected"]),
        "selection_nonunique": ambiguous,
        "min_loop_product": min_product,
        "shapes": {
            LOOP_TYPES[length]: summarize_shape(rows, length)
            for length in LOOP_TYPES
        },
        "note": (
            "Phi is the CCW phase of the one-body correlation product, "
            "not a physical magnetic field or electromagnetic flux. "
            "A pi phase alone does not imply time-reversal breaking."
        ),
    }
    (output / "loop_flux_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def plot_overview(summaries, output):
    """Max |Phi| for each loop shape versus V on one graph."""
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    vs = [row["V"] for row in summaries]
    for length, marker in ((3, "o"), (4, "s"), (6, "^")):
        name = LOOP_TYPES[length]
        vals = [r["shapes"][name]["max_abs_phi"] for r in summaries]
        ax.plot(vs, vals, marker=marker, linewidth=1.8, label=name)
    ambiguous = [r["V"] for r in summaries if r["selection_nonunique"]]
    for V in ambiguous:
        ax.axvline(V, color="grey", alpha=0.13, linewidth=1.2)
    ax.set_xlabel("V / t")
    ax.set_ylabel(r"$\max_{p\mathrm{\ of\ shape}}|\Phi_p|$ (rad)")
    ax.set_ylim(-0.05, np.pi + 0.1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--V", type=float, nargs="+", default=None)
    p.add_argument("--channel", choices=("same", "opposite"), default="same")
    p.add_argument("--output", default=None)
    p.add_argument("--force-resolve", action="store_true")
    p.add_argument("--no-check-H", action="store_true")
    p.add_argument("--low", type=int, default=None)
    p.add_argument("--max-basis", type=int, default=70_000)
    p.add_argument("--min-loop-product", type=float, default=1e-12)
    args = p.parse_args(argv)
    if args.min_loop_product < 0:
        p.error("--min-loop-product must be nonnegative")

    root = Path(args.input)
    if (root / "responses.json").is_file():
        record = json.loads((root / "responses.json").read_text(encoding="utf-8"))
        if args.V and any(abs(v - record["V"]) > 1e-9 for v in args.V):
            p.error("--V disagrees with the single-point input directory")
        values = [float(record["V"])]
        scan_root = root.parent
    else:
        if not (root / "scan_metadata.json").is_file():
            p.error("input must have scan_metadata.json or responses.json")
        values = sorted(set(args.V)) if args.V else find_scan_values(root)
        scan_root = root
    dest = Path(args.output) if args.output else scan_root / f"loop_flux_{args.channel}"
    dest.mkdir(parents=True, exist_ok=True)

    summaries, cache = [], None
    for V in values:
        folder, info, val, row = load_scan_metadata(root, V)
        info = dict(info, max_basis=args.max_basis)
        if args.low is not None:
            info["low"] = args.low
        g, states, psis, e0, source, cache = load_or_compute_ground(
            folder, info, val, cache=cache, force_resolve=args.force_resolve
        )
        if row and int(row.get("ground_dim", psis.shape[1])) != psis.shape[1]:
            raise RuntimeError(
                f"V={val}: scan ground_dim={row['ground_dim']}, "
                f"loaded/recomputed ground_dim={psis.shape[1]}"
            )
        sub = dest / f"V_{val:+.9f}"
        summary = export_one(
            g, states, psis, e0, val, sub, args.channel,
            not args.no_check_H, args.min_loop_product,
        )
        summary["state_source"] = source
        summaries.append(summary)
        counts = {k: v["count"] for k, v in summary["shapes"].items()}
        print(
            f"V={val:g}, LC_{args.channel}={summary['channel_J']:+.6g}, "
            f"loops={counts}, source={source}, nonunique="
            f"{summary['selection_nonunique']}",
            flush=True,
        )

    fields = ["V", "channel", "channel_J", "ground_dim",
              "selection_nonunique", "state_source"]
    for name in LOOP_TYPES.values():
        fields.extend([
            f"{name}_count", f"{name}_valid_count",
            f"{name}_max_abs_phi", f"{name}_mean_abs_phi",
        ])
    with (dest / "overview.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in summaries:
            out = {k: rec.get(k) for k in fields}
            for name, data in rec["shapes"].items():
                for key, value in data.items():
                    out[f"{name}_{key}"] = value
            writer.writerow(out)
    (dest / "overview.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    plot_overview(summaries, dest / "loop_flux_overview.png")
    print(f"Loop flux maps, CSV files and V overview saved in: {dest}")


if __name__ == "__main__":
    main()
