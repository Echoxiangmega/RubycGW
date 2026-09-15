#!/usr/bin/env python3
"""Plot an 18-site Ruby-supercell density as a repeated real-space pattern.

The script accepts either

1. ``density_profile.csv`` written by ``run_supercell_gw.py``; select a
   converged point with ``--V`` and, optionally, ``--source``; or
2. one zero-source checkpoint ``.npz`` written by the same driver.

The 18-site ordering is the project convention

    I = 6*s + a,   s = 0,1,2,   a = 0,...,5,

with sector ``s=(R1+R2) mod 3`` for a primitive cell R=(R1,R2).  The selected
18-site density is periodically repeated over a finite patch only for
visualization; no interpolation is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NSUB = 6
NSECTOR = 3
NSUP = NSUB * NSECTOR

# Real-space embedding of the Ruby graph used in rubycgw/model.py, rotated so
# a1 is horizontal.  This is used only to visualize the discrete site density.
A1 = np.array([1.38486563, 0.0])
A2 = np.array([0.69243282, 1.19932882])
BASIS = np.array(
    [
        [0.00000000, 0.00000000],   # 0
        [0.38788737, 0.92170678],   # 1
        [-0.60427780, 0.79677371],  # 2
        [-0.21634254, 0.71848049],  # 3
        [0.38793526, -0.07829322],  # 4
        [-0.60422991, -0.20322629], # 5
    ],
    dtype=float,
)

# Undirected bonds in exactly the model convention: (a,b,(dR1,dR2)).
BONDS = [
    (0, 1, (0, 0)),
    (0, 2, (0, 0)),
    (2, 1, (0, 0)),
    (3, 4, (0, 0)),
    (3, 5, (0, 0)),
    (4, 5, (0, 0)),
    (1, 4, (0, 0)),
    (5, 0, (0, -1)),
    (2, 3, (-1, 0)),
    (3, 1, (0, -1)),
    (2, 5, (0, 0)),
    (0, 4, (-1, 0)),
]


def _load_from_checkpoint(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        if "density" not in data:
            raise ValueError(f"{path} does not contain a density array.")
        density = np.asarray(data["density"], dtype=float).reshape(-1)
        meta = {}
        if "metadata_json" in data:
            meta = json.loads(str(data["metadata_json"].item()))
    if density.size != NSUP:
        raise ValueError(f"Expected {NSUP} density values, got {density.size}.")
    return density, meta


def _load_from_csv(
    path: Path,
    target_V: float,
    target_source: float,
    V_tol: float,
    source_tol: float,
) -> tuple[np.ndarray, dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty.")

    required = {"V", "source", "site", "density"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    available_V = sorted({float(row["V"]) for row in rows})
    matching = [
        row
        for row in rows
        if abs(float(row["V"]) - float(target_V)) <= float(V_tol)
        and abs(float(row["source"]) - float(target_source)) <= float(source_tol)
    ]
    if not matching:
        values = ", ".join(f"{v:g}" for v in available_V)
        raise ValueError(
            f"No rows found for V={target_V:g}, source={target_source:g}. "
            f"Available V values: {values}"
        )

    # A scan may contain several complete blocks with the same V/source.
    # Prefer the latest (v_step, source_step) block if those columns exist.
    if "v_step" in matching[0] and "source_step" in matching[0]:
        keys = {
            (int(row["v_step"]), int(row["source_step"]))
            for row in matching
        }
        chosen_key = max(keys)
        matching = [
            row
            for row in matching
            if (int(row["v_step"]), int(row["source_step"])) == chosen_key
        ]
    else:
        chosen_key = None

    density = np.full(NSUP, np.nan, dtype=float)
    for row in matching:
        site = int(row["site"])
        if 0 <= site < NSUP:
            density[site] = float(row["density"])

    if np.any(~np.isfinite(density)):
        missing_sites = np.flatnonzero(~np.isfinite(density)).tolist()
        raise ValueError(
            f"Selected V/source block does not contain all 18 sites; "
            f"missing {missing_sites}."
        )

    meta = {
        "V": float(target_V),
        "source": float(target_source),
        "block": chosen_key,
    }
    return density, meta


def load_density(
    path: Path,
    target_V: float | None,
    target_source: float,
    V_tol: float,
    source_tol: float,
) -> tuple[np.ndarray, dict]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        density, meta = _load_from_checkpoint(path)
        if target_V is not None and "V" in meta:
            if abs(float(meta["V"]) - float(target_V)) > float(V_tol):
                raise ValueError(
                    f"Checkpoint contains V={float(meta['V']):g}, "
                    f"not requested V={target_V:g}."
                )
        return density, meta
    if suffix == ".csv":
        if target_V is None:
            raise ValueError("--V is required when --input is a CSV file.")
        return _load_from_csv(
            path, target_V, target_source, V_tol, source_tol
        )
    raise ValueError("--input must be a density_profile.csv or checkpoint .npz.")


def cell_sector(R1: int, R2: int) -> int:
    return int((int(R1) + int(R2)) % NSECTOR)


def site_position(R1: int, R2: int, sublattice: int) -> np.ndarray:
    return int(R1) * A1 + int(R2) * A2 + BASIS[int(sublattice)]


def density_at_cell(density18: np.ndarray, R1: int, R2: int) -> np.ndarray:
    s = cell_sector(R1, R2)
    return density18[NSUB * s : NSUB * (s + 1)]


def _selected_cells(radius: int) -> list[tuple[int, int]]:
    return [
        (R1, R2)
        for R1 in range(-radius, radius + 1)
        for R2 in range(-radius, radius + 1)
    ]


def plot_density(
    density18: np.ndarray,
    meta: dict,
    radius: int,
    quantity: str,
    marker_size: float,
    annotate: bool,
    show_site_labels: bool,
    draw_bonds: bool,
    output: Path,
    dpi: int,
) -> None:
    cells = _selected_cells(radius)
    cell_set = set(cells)
    mean_density = float(np.mean(density18))

    positions = []
    values = []
    labels = []
    for R1, R2 in cells:
        local_density = density_at_cell(density18, R1, R2)
        for a in range(NSUB):
            positions.append(site_position(R1, R2, a))
            values.append(float(local_density[a]))
            labels.append((R1, R2, a))

    positions = np.asarray(positions, dtype=float)
    density_values = np.asarray(values, dtype=float)

    if quantity == "delta":
        plot_values = density_values - mean_density
        vmax = max(float(np.max(np.abs(plot_values))), 1.0e-14)
        vmin = -vmax
        cmap = "RdBu_r"
        cbar_label = r"$n_i-\bar n$"
    else:
        plot_values = density_values
        vmin = float(np.min(density_values))
        vmax = float(np.max(density_values))
        if np.isclose(vmin, vmax):
            pad = max(1.0e-6, abs(vmin) * 1.0e-6)
            vmin -= pad
            vmax += pad
        cmap = "viridis"
        cbar_label = r"$n_i$"

    fig, ax = plt.subplots(figsize=(8.2, 7.2))

    if draw_bonds:
        for R1, R2 in cells:
            for a, b, delta in BONDS:
                dest = (R1 + int(delta[0]), R2 + int(delta[1]))
                if dest not in cell_set:
                    continue
                p0 = site_position(R1, R2, a)
                p1 = site_position(dest[0], dest[1], b)
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    linewidth=0.8,
                    alpha=0.35,
                    color="0.45",
                    zorder=1,
                )

    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=plot_values,
        s=float(marker_size),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.45,
        zorder=2,
    )

    if annotate or show_site_labels:
        for (x, y), nval, (_, _, a) in zip(
            positions, density_values, labels
        ):
            pieces = []
            if show_site_labels:
                pieces.append(str(a))
            if annotate:
                pieces.append(f"{nval:.3f}")
            ax.text(
                x,
                y,
                "\n".join(pieces),
                ha="center",
                va="center",
                fontsize=6.5,
                zorder=3,
            )

    # Project-defined period-three amplitude, duplicated here so the plotting
    # utility can also be copied and run standalone.
    w = np.exp(2j * np.pi / 3.0)
    v6 = np.array([1.0, w, w**2, -1.0, -w, -(w**2)], dtype=complex)
    z = np.concatenate([v6 * (w**s) for s in range(NSECTOR)])
    delta18 = density18 - mean_density
    phi = complex(2.0 * np.vdot(z, delta18) / np.vdot(z, z))

    V_text = meta.get("V", None)
    source_text = meta.get("source", None)
    title_parts = ["Ruby real-space density"]
    if V_text is not None:
        title_parts.append(f"V={float(V_text):g}")
    if source_text is not None:
        title_parts.append(f"h={float(source_text):g}")
    title_parts.append(rf"$|\Phi|={abs(phi):.4f}$")
    ax.set_title(", ".join(title_parts))

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    ax.set_aspect("equal")
    ax.set_xlabel("real-space x (bond length units)")
    ax.set_ylabel("real-space y (bond length units)")
    ax.margins(0.05)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"mean density = {mean_density:.12g}")
    print(f"min/max density = {density18.min():.12g} / {density18.max():.12g}")
    print(f"Phi = {phi.real:+.8e}{phi.imag:+.8e}i")
    print(f"|Phi| = {abs(phi):.8e}")
    print(f"saved: {output}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="density_profile.csv or one supercell checkpoint .npz",
    )
    p.add_argument(
        "--V",
        type=float,
        default=None,
        help="V to select from CSV; optional consistency check for checkpoint input",
    )
    p.add_argument(
        "--source",
        type=float,
        default=0.0,
        help="Source h to select from CSV (default: 0)",
    )
    p.add_argument("--V-tol", type=float, default=1e-10)
    p.add_argument("--source-tol", type=float, default=1e-12)
    p.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Plot primitive cells R1,R2=-radius,...,+radius (default: 2)",
    )
    p.add_argument(
        "--quantity",
        choices=["density", "delta"],
        default="density",
        help="Color by n_i or n_i-mean(n) (default: density)",
    )
    p.add_argument("--marker-size", type=float, default=210.0)
    p.add_argument("--annotate", action="store_true", help="Write n_i on each site.")
    p.add_argument(
        "--site-labels",
        action="store_true",
        help="Write primitive sublattice label 0,...,5 on each site.",
    )
    p.add_argument("--no-bonds", action="store_true")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG/PDF/SVG output path. Default is density_realspace_V*.png",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    if args.radius < 0:
        raise ValueError("--radius must be non-negative.")
    if args.marker_size <= 0:
        raise ValueError("--marker-size must be positive.")

    density18, meta = load_density(
        args.input,
        args.V,
        args.source,
        args.V_tol,
        args.source_tol,
    )

    if args.output is None:
        V_value = meta.get("V", args.V)
        if V_value is None:
            stem = "density_realspace"
        else:
            stem = f"density_realspace_V{float(V_value):.6f}"
        args.output = Path(stem + ".png")

    plot_density(
        density18=density18,
        meta=meta,
        radius=args.radius,
        quantity=args.quantity,
        marker_size=args.marker_size,
        annotate=args.annotate,
        show_site_labels=args.site_labels,
        draw_bonds=not args.no_bonds,
        output=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
