#!/usr/bin/env python3
"""Reassess saved V-only Ruby static ED/cGW rank data without rerunning ED.

The original inverse-response spectrum is sensitive to almost-null response
directions.  This script keeps only bubble eigenvectors whose eigenvalues
exceed --floor * max(eigenvalue), then studies the projected inverse-response
correction and its LC/CO reconstruction.  This is a diagnostic effective
STATIC kernel and is not the full particle-hole irreducible vertex.

Example (Windows, repository root):
    python research\\reanalyze_v_only_rank_npz.py results\\v_only_vertex_rank\\V_0.6\\rank_diagnostic.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def analyze(path: Path, floor: float = 0.01):
    with np.load(path, allow_pickle=False) as z:
        obj = {k: z[k].copy() for k in z.files}
    chi_ed = (obj["chi_ed"] + obj["chi_ed"].T) / 2
    chi_cgw = (obj["chi_cgw"] + obj["chi_cgw"].T) / 2
    bubble = (obj["bubble_gw"] + obj["bubble_gw"].T) / 2
    w, U = np.linalg.eigh(bubble)
    keep = w > floor * w[-1]
    if not np.any(keep):
        raise ValueError("No well-resolved bubble modes")
    Q, w = U[:, keep], w[keep]
    C = Q.T @ chi_cgw @ Q
    E = Q.T @ chi_ed @ Q
    if np.linalg.eigvalsh(C)[0] <= 0 or np.linalg.eigvalsh(E)[0] <= 0:
        raise ValueError("Non-positive projected response: change floor/check data")
    root = np.sqrt(w)
    inv_c = np.linalg.inv(C)
    D = root[:, None] * (inv_c - np.linalg.inv(E)) * root[None, :]
    D = (D + D.T) / 2
    d, X = np.linalg.eigh(D)
    ids = np.argsort(-abs(d))
    d, X = d[ids], X[:, ids]
    correction_norm = float(np.linalg.norm(d))
    if np.linalg.norm(chi_ed - chi_cgw) / np.linalg.norm(chi_ed) < 1e-9:
        meaningful = False
    else:
        meaningful = correction_norm > 1e-9
    capture = (
        np.cumsum(d*d) / np.sum(d*d)
        if meaningful else np.zeros_like(d)
    )
    target = obj["target_vectors"]
    curves = []
    for r in range(len(d)+1):
        Dr = (X[:, :r] * d[:r]) @ X[:, :r].T
        inv_response = inv_c - (1/root)[:, None]*Dr*(1/root)[None, :]
        response = np.linalg.inv(inv_response)
        # Keep cGW unchanged outside the resolved bubble subspace.
        full = chi_cgw + Q @ (response - C) @ Q.T
        curves.append(np.einsum("ai,ij,aj->a", target, full, target))
    return {
        "V": float(obj["V"]),
        "names": obj["target_names"].astype(str),
        "chi_ed": np.einsum("ai,ij,aj->a", target, chi_ed, target),
        "chi_cgw": np.einsum("ai,ij,aj->a", target, chi_cgw, target),
        "target_curves": np.asarray(curves),
        "eigenvalues": d, "capture": capture, "retained": int(len(d)),
        "meaningful": meaningful, "original_capture": obj["capture"],
    }


def main():
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--floor", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=Path("results/v_only_rank_reanalysis"))
    args = parser.parse_args()
    if not (0 < args.floor < 1):
        parser.error("--floor must satisfy 0 < floor < 1")
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        a = analyze(path, floor=args.floor)
        tag = f"V_{a['V']:.6g}"
        print(f"V={a['V']:g} retained support {a['retained']}/36")
        if a["meaningful"]:
            for fraction in (.95, .99):
                orig = int(np.searchsorted(a["original_capture"], fraction)+1)
                regularized = int(np.searchsorted(a["capture"], fraction)+1)
                print(f"  {fraction:.0%} weight: old {orig}, filtered {regularized}")
        else:
            print("  ED and cGW coincide: rank is numerically meaningless")
        for j, name in enumerate(a["names"][:3]):
            print(f"  {name}: ED={a['chi_ed'][j]:.7g}, "
                  f"cGW={a['chi_cgw'][j]:.7g}, "
                  f"rank4={a['target_curves'][min(4,a['retained']),j]:.7g}")
        np.savez_compressed(args.out/f"{tag}_filtered.npz", **a)
        if not a["meaningful"]:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        r = np.arange(1, len(a["capture"])+1)
        ax.plot(r, a["capture"], label="filtered")
        ax.plot(np.arange(1, 37), a["original_capture"], ls=":", label="original")
        ax.axhline(.95, ls="--", color="gray")
        ax.set(xlabel="retained modes", ylabel="cumulative squared eigenvalue weight",
               title=f"V={a['V']:g}, floor={args.floor:g}", ylim=(0, 1.02))
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out/f"{tag}_capture.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in ("LC_same", "LC_opposite", "CO_AminusB"):
            j = int(np.flatnonzero(a["names"] == name)[0])
            line, = ax.plot(a["target_curves"][:, j], ".-", ms=3, label=name)
            ax.axhline(a["chi_ed"][j], ls="--", color=line.get_color(), alpha=.65)
        ax.set(xlabel="retained modes", ylabel="static susceptibility",
               title=f"V={a['V']:g}: solid reconstructed, dashed ED",
               xlim=(0, min(22, a["retained"])))
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out/f"{tag}_targets.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
