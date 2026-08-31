"""Plot low-frequency effective bands from an 18-site supercell GW checkpoint.

This is a diagnostic for the insulating/gapped character of a converged
Matsubara-axis GW state.  It does *not* perform analytic continuation and
therefore should not be interpreted as the exact real-frequency GW
quasiparticle dispersion.

For the checkpoint self-energy we construct

    H_eff(k) = h0(k) + Sigma_H + Herm[Sigma_GW(i*pi*T,k)] - mu I,

where ``Herm[A]=(A+A^dagger)/2``.  The checkpoint stores Sigma_GW only on the
self-consistency k mesh, so along a continuous band path it is periodically
Fourier-interpolated from that mesh.  With a 3x3 GW mesh this interpolation is
necessarily coarse, but it is useful for diagnosing whether the Fermi level is
inside a substantial low-frequency gap.

Example
-------
python plot_supercell_bands.py ^
  --checkpoint results\supercell18\checkpoints\V1.730000_n3.000000_nk3x3_nw47_no10_T0.05.npz
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rubycgw.model import RubyParameters
from rubycgw.supercell import NSUP, build_supercell_h0


def _load_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"].item()))
        sigma_h = np.asarray(data["Sigma_H"], dtype=complex)
        sigma_gw = np.asarray(data["Sigma_GW"], dtype=complex)
        density = np.asarray(data["density"], dtype=float)

    if sigma_h.shape != (NSUP, NSUP):
        raise ValueError(f"Sigma_H has unexpected shape {sigma_h.shape}")
    if sigma_gw.ndim != 5 or sigma_gw.shape[-2:] != (NSUP, NSUP):
        raise ValueError(f"Sigma_GW has unexpected shape {sigma_gw.shape}")

    required = ["V", "T", "nw", "nk1", "nk2", "ti", "t1", "t2", "mu"]
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"Checkpoint is missing metadata keys: {missing}")

    expected_nf = 2 * int(meta["nw"])
    expected_shape = (
        expected_nf,
        int(meta["nk1"]),
        int(meta["nk2"]),
        NSUP,
        NSUP,
    )
    if sigma_gw.shape != expected_shape:
        raise ValueError(
            f"Sigma_GW shape {sigma_gw.shape} does not match metadata {expected_shape}"
        )

    return meta, sigma_h, sigma_gw, density


def _hermitian_low_frequency_sigma(sigma_gw: np.ndarray, nw: int) -> np.ndarray:
    """Hermitian zero-frequency proxy from the +/- pi*T Matsubara slices."""
    iw_plus = int(nw)      # n=0 -> +pi T
    iw_minus = int(nw) - 1  # n=-1 -> -pi T

    sp = sigma_gw[iw_plus]
    sm = sigma_gw[iw_minus]
    hp = 0.5 * (sp + np.swapaxes(sp.conj(), -1, -2))
    hm = 0.5 * (sm + np.swapaxes(sm.conj(), -1, -2))
    out = 0.5 * (hp + hm)
    return 0.5 * (out + np.swapaxes(out.conj(), -1, -2))


def _periodic_fourier_interpolate(field: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    """Periodic trigonometric interpolation from a regular reduced-k mesh."""
    field = np.asarray(field, dtype=complex)
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 2)
    nk1, nk2 = field.shape[:2]

    coeff = np.fft.fft2(field, axes=(0, 1)) / float(nk1 * nk2)
    r1 = np.fft.fftfreq(nk1, d=1.0 / nk1)
    r2 = np.fft.fftfreq(nk2, d=1.0 / nk2)

    phase = np.exp(
        2j
        * np.pi
        * (
            kpts[:, 0, None, None] * r1[None, :, None]
            + kpts[:, 1, None, None] * r2[None, None, :]
        )
    )
    out = np.einsum("pij,ijab->pab", phase, coeff, optimize=True)
    return 0.5 * (out + np.swapaxes(out.conj(), -1, -2))


def _band_path(points_per_segment: int):
    """Gamma-M-K-Gamma path in reduced supercell reciprocal coordinates."""
    n = int(points_per_segment)
    if n < 2:
        raise ValueError("points_per_segment must be at least 2")

    gamma = np.array([0.0, 0.0])
    m = np.array([0.5, 0.0])
    k = np.array([1.0 / 3.0, 1.0 / 3.0])
    nodes = [gamma, m, k, gamma]
    labels = [r"$\Gamma$", "M", "K", r"$\Gamma$"]

    pieces = []
    for start, stop in zip(nodes[:-1], nodes[1:]):
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        pieces.append((1.0 - t) * start[None, :] + t * stop[None, :])
    pieces.append(nodes[-1][None, :])
    path = np.concatenate(pieces, axis=0)
    x = np.arange(path.shape[0], dtype=float) / float(n)
    ticks = np.arange(4, dtype=float)
    return path, x, ticks, labels


def _effective_eigenvalues(
    kpts: np.ndarray,
    params: RubyParameters,
    sigma_h: np.ndarray,
    sigma0_grid: np.ndarray,
    mu: float,
) -> np.ndarray:
    h0 = build_supercell_h0(kpts, params).reshape(-1, NSUP, NSUP)
    sigma0 = _periodic_fourier_interpolate(sigma0_grid, kpts)
    eye = np.eye(NSUP, dtype=complex)
    heff = h0 + sigma_h[None, :, :] + sigma0 - float(mu) * eye[None, :, :]
    heff = 0.5 * (heff + np.swapaxes(heff.conj(), -1, -2))
    return np.linalg.eigvalsh(heff)


def _gap_diagnostic(
    params: RubyParameters,
    sigma_h: np.ndarray,
    sigma0_grid: np.ndarray,
    mu: float,
    dense_n: int,
):
    n = int(dense_n)
    if n < 3:
        raise ValueError("dense_n must be at least 3")
    u = np.arange(n, dtype=float) / float(n)
    k1, k2 = np.meshgrid(u, u, indexing="ij")
    kpts = np.stack([k1, k2], axis=-1).reshape(-1, 2)
    evals = _effective_eigenvalues(kpts, params, sigma_h, sigma0_grid, mu)

    below = evals[evals <= 0.0]
    above = evals[evals >= 0.0]
    vbm = float(np.max(below)) if below.size else float("nan")
    cbm = float(np.min(above)) if above.size else float("nan")
    gap = cbm - vbm if np.isfinite(vbm) and np.isfinite(cbm) else float("nan")
    min_abs = float(np.min(np.abs(evals)))
    return vbm, cbm, gap, min_abs


def _write_csv(path: Path, x: np.ndarray, kpts: np.ndarray, evals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_coordinate", "k1_sc", "k2_sc"] + [f"band_{i+1}" for i in range(NSUP)])
        for s, k, bands in zip(x, kpts, evals):
            writer.writerow([f"{s:.12g}", f"{k[0]:.12g}", f"{k[1]:.12g}"] + [f"{e:.12g}" for e in bands])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a low-frequency effective band diagnostic from a supercell GW checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a converged 18-site GW checkpoint (.npz).")
    parser.add_argument("--points", type=int, default=180, help="k points per high-symmetry segment (default: 180).")
    parser.add_argument("--dense", type=int, default=81, help="Dense k mesh for the indirect-gap diagnostic (default: 81).")
    parser.add_argument("--window", type=float, default=1.0, help="Plot energy window +/- value around the Fermi level (default: 1.0).")
    parser.add_argument("--output", default=None, help="Output PNG path. Default: next to checkpoint with _bands.png suffix.")
    parser.add_argument("--csv", default=None, help="Optional output CSV path. Default: same stem with _bands.csv.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    meta, sigma_h, sigma_gw, density = _load_checkpoint(checkpoint)

    params = RubyParameters(
        ti=float(meta["ti"]),
        t1=float(meta["t1"]),
        t2=float(meta["t2"]),
        V=float(meta["V"]),
    )
    mu = float(meta["mu"])
    T = float(meta["T"])
    nw = int(meta["nw"])
    nk1 = int(meta["nk1"])
    nk2 = int(meta["nk2"])

    sigma0_grid = _hermitian_low_frequency_sigma(sigma_gw, nw)
    kpath, x, ticks, labels = _band_path(args.points)
    evals = _effective_eigenvalues(kpath, params, sigma_h, sigma0_grid, mu)

    vbm, cbm, gap, min_abs = _gap_diagnostic(
        params, sigma_h, sigma0_grid, mu, args.dense
    )

    output = Path(args.output) if args.output else checkpoint.with_name(checkpoint.stem + "_bands.png")
    csv_path = Path(args.csv) if args.csv else checkpoint.with_name(checkpoint.stem + "_bands.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, x, kpath, evals)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for ib in range(NSUP):
        ax.plot(x, evals[:, ib], linewidth=1.0)
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    for tick in ticks:
        ax.axvline(tick, linewidth=0.6, alpha=0.35)
    ax.set_xticks(ticks, labels)
    ax.set_xlim(float(x[0]), float(x[-1]))
    window = abs(float(args.window))
    if window > 0.0:
        ax.set_ylim(-window, window)
    ax.set_ylabel(r"$E_{\mathrm{eff}}-\mu$")
    ax.set_title(
        f"18-site Ruby low-frequency effective bands, V={float(meta['V']):.3f}\n"
        f"Hermitian Sigma(i pi T), T={T:g}, GW mesh={nk1}x{nk2}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)

    print("=" * 78)
    print("18-site Ruby low-frequency effective-band diagnostic")
    print(f"checkpoint : {checkpoint}")
    print(f"V          : {float(meta['V']):.8g}")
    print(f"mu         : {mu:.10f}")
    print(f"T          : {T:.8g}")
    print(f"GW k mesh  : {nk1} x {nk2}")
    print(f"mean n     : {float(np.mean(density)):.10f} per supercell orbital")
    print(f"VBM        : {vbm:+.8e} relative to mu")
    print(f"CBM        : {cbm:+.8e} relative to mu")
    print(f"indirect gap proxy : {gap:+.8e}")
    print(f"min |E_eff-mu|     : {min_abs:.8e}")
    print(f"PNG        : {output}")
    print(f"CSV        : {csv_path}")
    print("NOTE: this is a Matsubara low-frequency static diagnostic, not an")
    print("      analytically-continued real-frequency GW quasiparticle band structure.")
    if nk1 <= 3 or nk2 <= 3:
        print("NOTE: Sigma(k) is Fourier-interpolated from a very coarse GW k mesh;")
        print("      use the result mainly to diagnose a robust gap, not fine dispersion.")
    print("=" * 78)


if __name__ == "__main__":
    main()
