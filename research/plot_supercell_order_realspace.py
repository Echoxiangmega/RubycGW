#!/usr/bin/env python3
"""Plot real-space density or triangle loop-current order from Ruby checkpoints.

The geometry matches the earlier ``ABC()`` convention used in the project:

    a1 = (sqrt(3)/2, 1/2)
    a2 = (-sqrt(3)/2, 1/2)

with the six orbital fractional coordinates specified below.  For the 18-site
Q=(1/3,1/3) supercell, the basis ordering and representatives are taken from
``rubycgw.supercell`` exactly:

    I = 6*s + a,  s=0,1,2,  a=0,...,5,
    R_s = (s,0).

GW checkpoints intentionally store only ``Sigma_H``, ``Sigma_GW``, ``density``
and metadata, not an equal-time density matrix or G.  Therefore current plots
reconstruct the Matsubara Green function from the stored converged self-energies
and evaluate the local triangle eta expectation values directly:

    eta_{A,s} = (T/Nk) sum_{k,n} Tr[K_{A,s} G(k,iw_n)],
    eta_{B,s} = (T/Nk) sum_{k,n} Tr[K_{B,s} G(k,iw_n)].

This is the same loop-current order parameter convention used throughout the
project.  Positive eta follows the oriented algebraic loops

    A: 0 -> 1 -> 2 -> 0,
    B: 3 -> 4 -> 5 -> 3.

Because A and B have opposite geometric handedness in this embedding, equal
algebraic signs correspond to opposite physical circulation, consistently with
``eta_plus = (eta_A+eta_B)/sqrt(2)`` = physical-opposite circulation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, eta_vertices
from rubycgw.supercell import (
    NSUP,
    NSECTOR,
    NSUB,
    SUPERCELL_MATRIX,
    SUPERCELL_REPRESENTATIVES,
    build_supercell_h0,
)
from rubycgw.supercell_gw import dyson_from_sigma_matrix


# -----------------------------------------------------------------------------
# Geometry: exactly the user's earlier ABC() embedding.
# -----------------------------------------------------------------------------


def primitive_geometry() -> tuple[np.ndarray, np.ndarray]:
    lat = np.array(
        [
            [np.sqrt(3.0) / 2.0, 0.5],
            [-np.sqrt(3.0) / 2.0, 0.5],
        ],
        dtype=float,
    )
    dis = 1.0 / 8.0
    orb = np.array(
        [
            [1.0 / 3.0 - dis, 2.0 / 3.0 + dis],
            [1.0 / 3.0 + 2.0 * dis, 2.0 / 3.0 + dis],
            [1.0 / 3.0 - dis, 2.0 / 3.0 - 2.0 * dis],
            [2.0 / 3.0 + dis, 1.0 / 3.0 - dis],
            [2.0 / 3.0 + dis, 1.0 / 3.0 + 2.0 * dis],
            [2.0 / 3.0 - 2.0 * dis, 1.0 / 3.0 - dis],
        ],
        dtype=float,
    )
    return lat, orb


def primitive_to_cart(frac: np.ndarray) -> np.ndarray:
    lat, _ = primitive_geometry()
    return np.asarray(frac, dtype=float) @ lat


def supercell_site_positions() -> np.ndarray:
    """Return positions with the exact internal ordering I=6*s+a."""
    _, orb = primitive_geometry()
    out = np.zeros((NSUP, 2), dtype=float)
    for s, Rs in enumerate(SUPERCELL_REPRESENTATIVES):
        for a in range(NSUB):
            out[NSUB * s + a] = primitive_to_cart(Rs + orb[a])
    return out


def supercell_vectors_cart() -> tuple[np.ndarray, np.ndarray]:
    # SUPERCELL_MATRIX columns are T1 and T2 in primitive coordinates.
    T1 = primitive_to_cart(SUPERCELL_MATRIX[:, 0])
    T2 = primitive_to_cart(SUPERCELL_MATRIX[:, 1])
    return T1, T2


def triangle_indices(sector: int, triangle: str) -> list[int]:
    base = NSUB * int(sector)
    if triangle == "A":
        local = (0, 1, 2)
    elif triangle == "B":
        local = (3, 4, 5)
    else:
        raise ValueError("triangle must be A or B")
    return [base + a for a in local]


def triangle_oriented_indices(sector: int, triangle: str, sign: float = 1.0) -> list[int]:
    """Project eta orientation onto the 18-site indices.

    Positive eta:
      A: 0->1->2->0
      B: 3->4->5->3
    Negative eta reverses the arrows.
    """
    inds = triangle_indices(sector, triangle)
    if sign < 0.0:
        return [inds[0], inds[2], inds[1]]
    return inds


# -----------------------------------------------------------------------------
# Checkpoint reconstruction.
# -----------------------------------------------------------------------------


def load_checkpoint(path: str | Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"metadata_json", "Sigma_H", "Sigma_GW", "density"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"checkpoint is missing required fields: {missing}")
        meta = json.loads(str(data["metadata_json"].item()))
        sigma_h = np.asarray(data["Sigma_H"], dtype=complex)
        sigma_gw = np.asarray(data["Sigma_GW"], dtype=complex)
        density = np.asarray(data["density"], dtype=float)
    if density.shape != (NSUP,):
        raise ValueError(f"this plotting driver expects an 18-site checkpoint; density shape={density.shape}")
    return meta, sigma_h, sigma_gw, density


def grid_and_params_from_metadata(meta: dict) -> tuple[MatsubaraGrid, RubyParameters]:
    grid = MatsubaraGrid(
        nk1=int(meta["nk1"]),
        nk2=int(meta["nk2"]),
        nw=int(meta["nw"]),
        nOmega=int(meta["nOmega"]),
        T=float(meta["T"]),
    )
    params = RubyParameters(
        ti=float(meta["ti"]),
        t1=float(meta["t1"]),
        t2=float(meta["t2"]),
        V=float(meta["V"]),
    )
    return grid, params


def reconstruct_G(
    meta: dict,
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
) -> tuple[np.ndarray, MatsubaraGrid, RubyParameters]:
    grid, params = grid_and_params_from_metadata(meta)
    source = float(meta.get("source", 0.0))
    if abs(source) > 1e-14:
        raise ValueError(
            "This checkpoint has a nonzero temporary source but the checkpoint metadata "
            "does not store which branch/source operator generated it. Plot a zero-source "
            "endpoint checkpoint instead."
        )
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    G = dyson_from_sigma_matrix(
        h0,
        grid,
        float(meta["mu"]),
        np.asarray(sigma_h, dtype=complex),
        np.asarray(sigma_gw, dtype=complex),
    )
    return G, grid, params


def local_triangle_eta(G: np.ndarray, grid: MatsubaraGrid) -> tuple[np.ndarray, float]:
    """Return eta[s, A/B] and the largest discarded imaginary part."""
    ka, kb, _, _ = eta_vertices()
    eta = np.zeros((NSECTOR, 2), dtype=complex)
    for s in range(NSECTOR):
        sl = slice(NSUB * s, NSUB * (s + 1))
        for itri, k6 in enumerate((ka, kb)):
            K = np.zeros((NSUP, NSUP), dtype=complex)
            K[sl, sl] = k6
            eta[s, itri] = (grid.T / grid.nk) * np.einsum(
                "ab,nxyba->", K, G, optimize=True
            )
    imag_max = float(np.max(np.abs(eta.imag)))
    return eta.real.astype(float), imag_max


# -----------------------------------------------------------------------------
# Plotting helpers.
# -----------------------------------------------------------------------------


def tile_shifts(radius: int) -> list[tuple[int, int, np.ndarray]]:
    radius = int(radius)
    T1, T2 = supercell_vectors_cart()
    out = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            out.append((i, j, i * T1 + j * T2))
    return out


def draw_triangle_bonds(ax, positions: np.ndarray, shift: np.ndarray, alpha: float = 0.55) -> None:
    for s in range(NSECTOR):
        for tri in ("A", "B"):
            inds = triangle_indices(s, tri)
            pts = positions[inds] + shift
            ax.add_patch(
                Polygon(
                    pts,
                    closed=True,
                    fill=False,
                    linewidth=1.15,
                    edgecolor="0.35",
                    alpha=alpha,
                    zorder=1,
                )
            )


def draw_supercell_boundary(ax, shift: np.ndarray, alpha: float = 0.25) -> None:
    T1, T2 = supercell_vectors_cart()
    pts = np.array([shift, shift + T1, shift + T1 + T2, shift + T2])
    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            fill=False,
            linewidth=1.0,
            linestyle="--",
            edgecolor="0.5",
            alpha=alpha,
            zorder=0,
        )
    )


def set_equal_limits(ax, all_points: np.ndarray, pad_fraction: float = 0.08) -> None:
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    pad = pad_fraction * max(dx, dy, 1.0)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")


def resolve_current_unit(spec: str, eta: np.ndarray) -> tuple[float, str]:
    vals = {
        f"{tri}{s}": float(eta[s, 0 if tri == "A" else 1])
        for s in range(NSECTOR)
        for tri in ("A", "B")
    }
    key = str(spec).strip()
    if key.lower() == "max":
        unit = float(np.max(np.abs(eta)))
        label = "max |eta|"
    elif key.upper() in vals:
        unit = abs(vals[key.upper()])
        label = f"|eta_{key.upper()}|"
    else:
        try:
            unit = abs(float(key))
        except ValueError as exc:
            allowed = "max, A0, B0, A1, B1, A2, B2, or a numeric value"
            raise ValueError(f"invalid --current-unit {spec!r}; use {allowed}") from exc
        label = f"{unit:g}"
    if not np.isfinite(unit) or unit < 1e-14:
        raise ValueError(
            f"chosen current normalization unit is too small ({unit:.3e}); "
            "choose --current-unit max or a nonzero triangle/reference value"
        )
    return unit, label


def draw_directed_triangle(
    ax,
    positions: np.ndarray,
    shift: np.ndarray,
    sector: int,
    triangle: str,
    value_norm: float,
    max_abs_norm: float,
) -> None:
    """Draw the physical arrow direction implied by the signed eta value."""
    if abs(value_norm) < 1e-13:
        return
    inds = triangle_oriented_indices(sector, triangle, sign=value_norm)
    pts = positions[inds] + shift
    strength = min(abs(value_norm) / max(max_abs_norm, 1e-14), 1.0)
    linewidth = 1.0 + 3.0 * strength
    mutation = 9.0 + 7.0 * strength
    color = "tab:red" if value_norm > 0.0 else "tab:blue"

    # Draw three slightly shortened directed edges.  Using the actual oriented
    # orbital sequence automatically accounts for A/B opposite handedness.
    for p0, p1 in zip(pts, np.roll(pts, -1, axis=0)):
        d = p1 - p0
        q0 = p0 + 0.12 * d
        q1 = p1 - 0.12 * d
        arrow = FancyArrowPatch(
            q0,
            q1,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=linewidth,
            color=color,
            alpha=0.92,
            zorder=4,
        )
        ax.add_patch(arrow)


def write_density_csv(path: Path, density: np.ndarray, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mean = float(np.mean(density))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sector", "sublattice", "site", "x", "y", "density", "density_minus_mean"])
        for s in range(NSECTOR):
            for a in range(NSUB):
                I = NSUB * s + a
                writer.writerow(
                    [s, a, I, positions[I, 0], positions[I, 1], density[I], density[I] - mean]
                )


def write_current_csv(path: Path, eta: np.ndarray, unit: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sector", "triangle", "eta", "eta_over_unit"])
        for s in range(NSECTOR):
            writer.writerow([s, "A", eta[s, 0], eta[s, 0] / unit])
            writer.writerow([s, "B", eta[s, 1], eta[s, 1] / unit])


def plot_density(args, density: np.ndarray) -> None:
    positions = supercell_site_positions()
    shifts = tile_shifts(args.tile_radius)

    if args.density_size_mode == "n":
        values = np.asarray(density, dtype=float)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        denom = max(vmax - vmin, 1e-14)
        scale = (values - vmin) / denom
        size_label = "n_i"
    else:
        values = np.asarray(density, dtype=float) - float(np.mean(density))
        vmax_abs = max(float(np.max(np.abs(values))), 1e-14)
        scale = np.abs(values) / vmax_abs
        size_label = "|n_i-<n>|"

    sizes = args.site_size_min + (args.site_size_max - args.site_size_min) * scale
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    scat = None
    all_points = []
    for ti, tj, shift in shifts:
        draw_supercell_boundary(ax, shift)
        draw_triangle_bonds(ax, positions, shift)
        pts = positions + shift
        all_points.append(pts)
        scat = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=sizes,
            c=density,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.55,
            zorder=3,
        )
        if args.annotate and ti == 0 and tj == 0:
            for s in range(NSECTOR):
                for a in range(NSUB):
                    I = NSUB * s + a
                    ax.text(
                        pts[I, 0] + 0.025,
                        pts[I, 1] + 0.025,
                        f"{s}:{a}",
                        fontsize=7,
                        zorder=6,
                    )
    cbar = fig.colorbar(scat, ax=ax, shrink=0.86)
    cbar.set_label("site density n_i")
    set_equal_limits(ax, np.concatenate(all_points, axis=0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Ruby 18-site real-space density  (point size ~ {size_label})")
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    if args.csv is not None:
        write_density_csv(Path(args.csv), density, positions)


def plot_current(
    args,
    meta: dict,
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
) -> None:
    G, grid, _ = reconstruct_G(meta, sigma_h, sigma_gw)
    eta, imag_max = local_triangle_eta(G, grid)
    unit, unit_label = resolve_current_unit(args.current_unit, eta)
    eta_n = eta / unit
    max_abs_norm = float(np.max(np.abs(eta_n)))

    positions = supercell_site_positions()
    shifts = tile_shifts(args.tile_radius)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    all_points = []

    for ti, tj, shift in shifts:
        draw_supercell_boundary(ax, shift)
        draw_triangle_bonds(ax, positions, shift, alpha=0.38)
        pts = positions + shift
        all_points.append(pts)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=32,
            facecolors="white",
            edgecolors="black",
            linewidths=0.65,
            zorder=3,
        )
        for s in range(NSECTOR):
            for itri, tri in enumerate(("A", "B")):
                value = float(eta_n[s, itri])
                draw_directed_triangle(
                    ax,
                    positions,
                    shift,
                    s,
                    tri,
                    value,
                    max_abs_norm,
                )
                inds = triangle_indices(s, tri)
                center = np.mean(positions[inds], axis=0) + shift
                if ti == 0 and tj == 0 or args.label_all_tiles:
                    ax.text(
                        center[0],
                        center[1],
                        f"{tri}{s}\n{value:+.3f}",
                        fontsize=8,
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.55", alpha=0.86),
                        zorder=7,
                    )
        if args.annotate and ti == 0 and tj == 0:
            for s in range(NSECTOR):
                for a in range(NSUB):
                    I = NSUB * s + a
                    ax.text(pts[I, 0] + 0.02, pts[I, 1] + 0.02, f"{s}:{a}", fontsize=6.5, zorder=8)

    set_equal_limits(ax, np.concatenate(all_points, axis=0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "Ruby 18-site local triangle loop-current order\n"
        f"labels = eta/unit,  unit={unit:.6g} ({unit_label});  max discarded Im={imag_max:.2e}"
    )
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    if args.csv is not None:
        write_current_csv(Path(args.csv), eta, unit)

    print("local triangle eta (rows s=0,1,2; columns A,B):")
    for s in range(NSECTOR):
        print(
            f"  s={s}: eta_A={eta[s,0]:+.12e}, eta_B={eta[s,1]:+.12e}, "
            f"normalized=({eta_n[s,0]:+.6f}, {eta_n[s,1]:+.6f})"
        )
    print(f"current normalization unit = {unit:.12e} ({unit_label})")
    print(f"max discarded Im(eta) = {imag_max:.3e}")


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=str, help="18-site Ruby GW checkpoint (.npz)")
    p.add_argument("--observable", choices=["density", "current"], required=True)
    p.add_argument("-o", "--output", required=True, help="Output PNG path")
    p.add_argument("--csv", default=None, help="Optional numeric CSV output")
    p.add_argument("--dpi", type=int, default=240)
    p.add_argument("--annotate", action="store_true", help="Label sites as sector:sublattice")
    p.add_argument(
        "--tile-radius",
        type=int,
        default=0,
        help="Replicate the 18-site supercell from -R..R along both T1,T2 for visual context.",
    )
    p.add_argument(
        "--label-all-tiles",
        action="store_true",
        help="For current plots, repeat numerical triangle labels on all displayed tiles.",
    )

    p.add_argument("--site-size-min", type=float, default=90.0)
    p.add_argument("--site-size-max", type=float, default=850.0)
    p.add_argument(
        "--density-size-mode",
        choices=["n", "delta"],
        default="n",
        help="Scale density point size by n_i or by |n_i-mean(n)|.",
    )
    p.add_argument(
        "--current-unit",
        default="max",
        help="Normalize eta by 'max', A0/B0/A1/B1/A2/B2, or a numeric value.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.tile_radius < 0:
        raise ValueError("--tile-radius must be nonnegative")

    meta, sigma_h, sigma_gw, density = load_checkpoint(args.checkpoint)
    print(
        f"checkpoint: V={float(meta['V']):g}, filling={float(meta['primitive_filling']):g}, "
        f"T={float(meta['T']):g}, mu={float(meta['mu']):.10f}, source={float(meta.get('source',0.0)):g}"
    )

    if args.observable == "density":
        plot_density(args, density)
    else:
        plot_current(args, meta, sigma_h, sigma_gw)

    print("output PNG:", args.output)
    if args.csv is not None:
        print("output CSV:", args.csv)


if __name__ == "__main__":
    main()
