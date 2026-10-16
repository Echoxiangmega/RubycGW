#!/usr/bin/env python3
"""Robust low-rank diagnosis of saved original V-only Ruby ED/cGW data.

No inverse susceptibility is taken in the main analysis: the Hermitian
response difference DeltaChi = chi_ED - chi_cGW is diagonalized directly.

A cross-V transfer diagnostic uses eigenvectors learned at one V as a *fixed
subspace*, but refits the correction matrix inside that subspace using ED
at the TEST V. This is an oracle diagnostic, NOT a predictive approximation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def load_response(path):
    with np.load(path, allow_pickle=False) as z:
        required = ("V", "chi_ed", "chi_cgw", "target_names", "target_vectors")
        for key in required:
            if key not in z.files:
                raise ValueError(f"{path}: missing {key!r}")
        data = {k: np.array(z[k], copy=True) for k in required}
    ce = np.asarray(data["chi_ed"], dtype=float)
    cc = np.asarray(data["chi_cgw"], dtype=float)
    if ce.shape != (36, 36) or cc.shape != ce.shape:
        raise ValueError("This diagnostic expects 36x36 uniform-cell matrices")
    d = 0.5 * (ce + ce.T - cc - cc.T)
    eigenvalues, eigenvectors = np.linalg.eigh(d)
    order = np.argsort(-abs(eigenvalues))
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    energy = np.sum(eigenvalues**2)
    capture = (np.cumsum(eigenvalues**2) / energy
               if energy > 1e-20 else np.zeros_like(eigenvalues))
    target = np.asarray(data["target_vectors"], dtype=float)
    ed = np.einsum("ai,ij,aj->a", target, ce, target)
    cgw = np.einsum("ai,ij,aj->a", target, cc, target)
    curves = []
    for r in range(37):
        dr = (eigenvectors[:, :r] * eigenvalues[:r]) @ eigenvectors[:, :r].T
        curves.append(np.einsum("ai,ij,aj->a", target, cc + dr, target))
    data.update(delta=d, eigenvalues=eigenvalues, eigenvectors=eigenvectors,
                capture=capture, target_ed=ed, target_cgw=cgw,
                target_curves=np.array(curves), path=str(path))
    return data


def transfer_oracle(train, test):
    """Use training eigenvectors, refitting ONLY the test-V amplitudes via ED."""
    if train["target_names"].tolist() != test["target_names"].tolist():
        raise ValueError("Target names differ across parameter points")
    test_delta = test["delta"]
    target = test["target_vectors"]
    cc = test["chi_cgw"]
    out = []
    overlaps = []
    for r in range(37):
        u = train["eigenvectors"][:, :r]
        approximated_delta = u @ (u.T @ test_delta @ u) @ u.T
        out.append(np.einsum("ai,ij,aj->a",
                             target, cc + approximated_delta, target))
        overlaps.append(
            np.linalg.norm(approximated_delta)**2 /
            max(np.linalg.norm(test_delta)**2, 1e-300)
        )
    return np.asarray(out), np.asarray(overlaps)


def main():
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--out", type=Path,
                   default=Path("results/v_only_response_rank"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    datasets = sorted([load_response(f) for f in args.files],
                      key=lambda x: float(x["V"]))
    for a in datasets:
        v = float(a["V"])
        folder = args.out / f"V_{v:.6g}"
        folder.mkdir(parents=True, exist_ok=True)
        r95 = np.searchsorted(a["capture"], .95) + 1
        print(f"V={v:g}: direct-response R(4)={a['capture'][3]:.6f}, "
              f"rank95={r95}")
        for name, ed, cgw, r2, r7 in zip(
            a["target_names"], a["target_ed"], a["target_cgw"],
            a["target_curves"][2], a["target_curves"][7],
        ):
            print(f"  {str(name):16s} ED={ed: .7g} cGW={cgw: .7g} "
                  f"rank2={r2: .7g} rank7={r7: .7g}")
        np.savez_compressed(folder / "direct_response_rank.npz",
                            V=v, eigenvalues=a["eigenvalues"],
                            eigenvectors=a["eigenvectors"],
                            capture=a["capture"],
                            delta_chi=a["delta"],
                            target_names=a["target_names"],
                            target_ed=a["target_ed"],
                            target_cgw=a["target_cgw"],
                            target_curves=a["target_curves"])
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        if np.max(np.abs(a["eigenvalues"])) > 1e-9:
            ax.plot(np.arange(1, 37), a["capture"], ".-")
        ax.set(xlabel="Number of response-difference modes",
               ylabel="Captured ||Delta chi||_F^2",
               title=f"V={v:g}: direct response difference",
               ylim=(-.025, 1.025))
        ax.axhline(.95, ls="--", color="gray")
        fig.tight_layout()
        fig.savefig(folder / "capture.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for name in ("LC_same", "LC_opposite", "CO_AminusB"):
            match = np.flatnonzero(a["target_names"] == name)
            if not len(match):
                continue
            j = int(match[0])
            line, = ax.plot(a["target_curves"][:, j], label=name)
            ax.axhline(a["target_ed"][j], ls="--", color=line.get_color())
        ax.set(xlabel="Number of response-difference modes",
               ylabel="Static susceptibility",
               title=f"V={v:g}: dashed = ED, solid = rank-r reconstruction")
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / "target_reconstruction.png", dpi=170)
        plt.close(fig)

    if len(datasets) > 1:
        for train in datasets:
            if np.linalg.norm(train["delta"]) < 1e-9:
                continue
            for test in datasets:
                if train is test or np.linalg.norm(test["delta"]) < 1e-9:
                    continue
                curves, captured = transfer_oracle(train, test)
                tag = f"train_V{float(train['V']):.6g}_test_V{float(test['V']):.6g}"
                np.savez_compressed(args.out / f"{tag}.npz",
                                    ranks=np.arange(37), oracle_curves=curves,
                                    capture=captured,
                                    target_names=test["target_names"],
                                    target_ed=test["target_ed"],
                                    target_cgw=test["target_cgw"])
                print(f"TRANSFER (ED amplitudes refitted) {tag}: "
                      f"rank4 captured={captured[4]:.6f}, "
                      f"rank7 captured={captured[7]:.6f}")
                fig, ax = plt.subplots(figsize=(7, 4.3))
                for name in ("LC_same", "LC_opposite", "CO_AminusB"):
                    match = np.flatnonzero(test["target_names"] == name)
                    if not len(match):
                        continue
                    j = int(match[0])
                    line, = ax.plot(curves[:, j], label=name)
                    ax.axhline(test["target_ed"][j], ls="--",
                               color=line.get_color())
                ax.set(xlabel="Transferred source-space dimension r",
                       ylabel="Test-V static susceptibility",
                       title=tag + " (oracle amplitudes)")
                ax.legend()
                fig.tight_layout()
                fig.savefig(args.out / f"{tag}.png", dpi=170)
                plt.close(fig)


if __name__ == "__main__":
    main()
