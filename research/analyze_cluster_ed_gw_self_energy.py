#!/usr/bin/env python3
"""Decompose exact, GW, and cluster-ED+GW Dyson kernels into local/nonlocal parts.

The physically gauge-invariant one-particle kernel is

    K(k,iw) = Sigma(k,iw) - mu I
            = iw I - h0(k) - G(k,iw)^{-1}.

Using K rather than raw Sigma avoids spurious differences caused only by the
slightly different chemical potentials of ED and approximate solutions.

For a finite Lx x Ly torus,

    K_loc(iw) = (1/Nk) sum_k K(k,iw),
    K_nonloc(k,iw) = K(k,iw) - K_loc(iw).

The script also performs two hybrid reconstructions:

  1. exact local + approximate nonlocal,
  2. approximate local + exact nonlocal,

and compares their Green functions with exact ED.  This directly diagnoses
whether the remaining one-particle error is dominated by local or nonlocal
self-energy structure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.model import RubyParameters, build_h0


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("results/cluster_ed_gw/cluster_ed_gw_L2x1_V1_fill2.npz"),
        help="converged run_cluster_ed_gw.py output",
    )
    p.add_argument(
        "--low-nfreq",
        type=int,
        default=4,
        help="number of positive Matsubara frequencies (and matching negative ones) in low-frequency metrics",
    )
    p.add_argument("--out", type=Path, default=None, help="output NPZ; default is INPUT with _sigma_decomp suffix")
    return p.parse_args()


def realspace_to_k(GR: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    """Transform a finite-torus site-basis Green matrix to 6x6 primitive k blocks.

    This is the inverse of the convention used by run_cluster_ed_gw.py:

        G_RR' = (1/Nk) sum_k exp(+ik.(R-R')) G_k.
    """
    arr = np.asarray(GR, dtype=complex)
    ncell = int(Lx) * int(Ly)
    nsite = 6 * ncell
    if arr.ndim != 3 or arr.shape[1:] != (nsite, nsite):
        raise ValueError("real-space Green function has incompatible shape")
    nf = int(arr.shape[0])
    cells = [(r1, r2) for r1 in range(int(Lx)) for r2 in range(int(Ly))]
    out = np.zeros((nf, int(Lx), int(Ly), 6, 6), dtype=complex)
    for i in range(int(Lx)):
        for j in range(int(Ly)):
            block = np.zeros((nf, 6, 6), dtype=complex)
            for c, (r1, r2) in enumerate(cells):
                for d, (s1, s2) in enumerate(cells):
                    phase = np.exp(
                        -2j * np.pi * (
                            (i / float(Lx)) * (r1 - s1)
                            + (j / float(Ly)) * (r2 - s2)
                        )
                    )
                    block += phase * arr[:, 6*c:6*(c+1), 6*d:6*(d+1)]
            out[:, i, j] = block / float(ncell)
    return out


def k_to_realspace(Gk: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    """Forward finite-torus transform, used as a convention/round-trip check."""
    arr = np.asarray(Gk, dtype=complex)
    if arr.ndim != 5 or arr.shape[1:3] != (int(Lx), int(Ly)) or arr.shape[-2:] != (6, 6):
        raise ValueError("k-space Green function has incompatible shape")
    nf = int(arr.shape[0])
    ncell = int(Lx) * int(Ly)
    cells = [(r1, r2) for r1 in range(int(Lx)) for r2 in range(int(Ly))]
    out = np.zeros((nf, 6*ncell, 6*ncell), dtype=complex)
    for c, (r1, r2) in enumerate(cells):
        for d, (s1, s2) in enumerate(cells):
            block = np.zeros((nf, 6, 6), dtype=complex)
            for i in range(int(Lx)):
                for j in range(int(Ly)):
                    phase = np.exp(
                        2j * np.pi * (
                            (i / float(Lx)) * (r1 - s1)
                            + (j / float(Ly)) * (r2 - s2)
                        )
                    )
                    block += phase * arr[:, i, j]
            out[:, 6*c:6*(c+1), 6*d:6*(d+1)] = block / float(ncell)
    return out


def dyson_kernel(Gk: np.ndarray, h0: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Return K = Sigma - mu I = iw I - h0 - G^{-1}."""
    G = np.asarray(Gk, dtype=complex)
    h = np.asarray(h0, dtype=complex)
    w = np.asarray(omega, dtype=float).reshape(-1)
    if G.shape != (len(w),) + h.shape:
        raise ValueError("G/h0/omega shape mismatch")
    eye = np.eye(6, dtype=complex)
    return (
        (1j * w[:, None, None, None, None]) * eye[None, None, None]
        - h[None, :, :, :, :]
        - np.linalg.inv(G)
    )


def decompose_kernel(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(K, dtype=complex)
    loc = np.mean(arr, axis=(1, 2))
    nonloc = arr - loc[:, None, None, :, :]
    return loc, nonloc


def green_from_kernel(K: np.ndarray, h0: np.ndarray, omega: np.ndarray) -> np.ndarray:
    arr = np.asarray(K, dtype=complex)
    h = np.asarray(h0, dtype=complex)
    w = np.asarray(omega, dtype=float).reshape(-1)
    eye = np.eye(6, dtype=complex)
    invg = (
        (1j * w[:, None, None, None, None]) * eye[None, None, None]
        - h[None, :, :, :, :]
        - arr
    )
    return np.linalg.inv(invg)


def _relerr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    den = max(float(np.linalg.norm(bb.ravel())), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def _frac(a: np.ndarray, b: np.ndarray) -> float:
    den = max(float(np.linalg.norm(np.asarray(b).ravel())), 1e-300)
    return float(np.linalg.norm(np.asarray(a).ravel()) / den)


def _low_indices(omega: np.ndarray, npos: int) -> np.ndarray:
    npos = int(npos)
    if npos < 1:
        raise ValueError("--low-nfreq must be positive")
    w = np.asarray(omega, dtype=float)
    nkeep = min(2 * npos, len(w))
    return np.sort(np.argsort(np.abs(w))[:nkeep])


def _metrics(prefix: str, K: np.ndarray, L: np.ndarray, N: np.ndarray,
             Kex: np.ndarray, Lex: np.ndarray, Nex: np.ndarray,
             idx: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, sel in (("all", slice(None)), ("low", idx)):
        out[f"{prefix}_K_total_relerr_{label}"] = _relerr(K[sel], Kex[sel])
        out[f"{prefix}_K_local_relerr_{label}"] = _relerr(L[sel], Lex[sel])
        out[f"{prefix}_K_nonlocal_relerr_{label}"] = _relerr(N[sel], Nex[sel])
        # Put local and nonlocal errors on the same denominator: ||K_exact||.
        locdiff = np.broadcast_to(
            (L[sel] - Lex[sel])[:, None, None, :, :],
            Kex[sel].shape,
        )
        out[f"{prefix}_local_error_over_exact_total_{label}"] = _frac(locdiff, Kex[sel])
        out[f"{prefix}_nonlocal_error_over_exact_total_{label}"] = _frac(N[sel] - Nex[sel], Kex[sel])
    return out


def main():
    args = _args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    with np.load(args.input, allow_pickle=False) as z:
        required = [
            "Lx", "Ly", "ti", "t1", "t2", "omega",
            "G", "G_background", "G_ed", "Gerr_embedded", "Gerr_background",
        ]
        missing = [k for k in required if k not in z]
        if missing:
            raise KeyError(f"input NPZ is missing required arrays: {missing}")
        Lx = int(z["Lx"])
        Ly = int(z["Ly"])
        params = RubyParameters(
            ti=float(z["ti"]),
            t1=float(z["t1"]),
            t2=float(z["t2"]),
            V=float(z["V"]) if "V" in z else 0.0,
        )
        omega = np.asarray(z["omega"], dtype=float)
        Gemb = np.asarray(z["G"], dtype=complex)
        Ggw = np.asarray(z["G_background"], dtype=complex)
        Ged_real = np.asarray(z["G_ed"], dtype=complex)
        gerr_emb_saved = float(z["Gerr_embedded"])
        gerr_gw_saved = float(z["Gerr_background"])
        mu = float(z["mu"]) if "mu" in z else np.nan
        mu_ed = float(z["mu_ed"]) if "mu_ed" in z else np.nan
        mu_gw = float(z["mu_background"]) if "mu_background" in z else np.nan
        converged = bool(z["converged"]) if "converged" in z else False

    if not converged:
        print("warning: input embedding run is not marked converged", flush=True)

    kmesh = np.zeros((Lx, Ly, 2), dtype=float)
    for i in range(Lx):
        for j in range(Ly):
            kmesh[i, j] = (i / float(Lx), j / float(Ly))
    h0 = build_h0(kmesh, params)

    Ged = realspace_to_k(Ged_real, Lx, Ly)
    roundtrip = _relerr(k_to_realspace(Ged, Lx, Ly), Ged_real)
    if roundtrip > 1e-10:
        raise RuntimeError(f"ED real/k-space transform round-trip failed: {roundtrip:.3e}")

    Kex = dyson_kernel(Ged, h0, omega)
    Kgw = dyson_kernel(Ggw, h0, omega)
    Kemb = dyson_kernel(Gemb, h0, omega)
    Lex, Nex = decompose_kernel(Kex)
    Lgw, Ngw = decompose_kernel(Kgw)
    Lemb, Nemb = decompose_kernel(Kemb)

    idx = _low_indices(omega, args.low_nfreq)
    metrics: dict[str, float] = {
        "ed_transform_roundtrip": roundtrip,
        "Gerr_GW_saved": gerr_gw_saved,
        "Gerr_embedded_saved": gerr_emb_saved,
        "Gerr_GW_kspace": _relerr(Ggw, Ged),
        "Gerr_embedded_kspace": _relerr(Gemb, Ged),
        "exact_nonlocal_fraction_all": _frac(Nex, Kex),
        "exact_nonlocal_fraction_low": _frac(Nex[idx], Kex[idx]),
    }
    metrics.update(_metrics("GW", Kgw, Lgw, Ngw, Kex, Lex, Nex, idx))
    metrics.update(_metrics("embedded", Kemb, Lemb, Nemb, Kex, Lex, Nex, idx))

    # Hybrid diagnostics: hold one sector exact and leave the other approximate.
    hybrids = {}
    for name, Lapp, Napp in (
        ("GW", Lgw, Ngw),
        ("embedded", Lemb, Nemb),
    ):
        K_exactloc_appnon = Lex[:, None, None, :, :] + Napp
        K_apploc_exactnon = Lapp[:, None, None, :, :] + Nex
        G_exactloc_appnon = green_from_kernel(K_exactloc_appnon, h0, omega)
        G_apploc_exactnon = green_from_kernel(K_apploc_exactnon, h0, omega)
        metrics[f"{name}_Gerr_exactlocal_appnonlocal"] = _relerr(G_exactloc_appnon, Ged)
        metrics[f"{name}_Gerr_applocal_exactnonlocal"] = _relerr(G_apploc_exactnon, Ged)
        metrics[f"{name}_Gerr_exactlocal_appnonlocal_low"] = _relerr(
            G_exactloc_appnon[idx], Ged[idx]
        )
        metrics[f"{name}_Gerr_applocal_exactnonlocal_low"] = _relerr(
            G_apploc_exactnon[idx], Ged[idx]
        )
        hybrids[f"G_{name}_exactlocal_appnonlocal"] = G_exactloc_appnon
        hybrids[f"G_{name}_applocal_exactnonlocal"] = G_apploc_exactnon

    print("=== exact local/nonlocal self-energy diagnostic ===", flush=True)
    print(f"input={args.input}", flush=True)
    print(
        "comparison uses K = Sigma - mu I = iw I - h0 - G^{-1} "
        "(chemical-potential gauge removed)",
        flush=True,
    )
    print(
        f"exact nonlocal fraction: all={metrics['exact_nonlocal_fraction_all']:.6e}, "
        f"low={metrics['exact_nonlocal_fraction_low']:.6e}",
        flush=True,
    )
    for name, label in (("GW", "GW"), ("embedded", "clusterED+GW")):
        print(
            f"[{label}] K relerr all: total={metrics[name + '_K_total_relerr_all']:.6e}, "
            f"local={metrics[name + '_K_local_relerr_all']:.6e}, "
            f"nonlocal={metrics[name + '_K_nonlocal_relerr_all']:.6e}",
            flush=True,
        )
        print(
            f"[{label}] K relerr low: total={metrics[name + '_K_total_relerr_low']:.6e}, "
            f"local={metrics[name + '_K_local_relerr_low']:.6e}, "
            f"nonlocal={metrics[name + '_K_nonlocal_relerr_low']:.6e}",
            flush=True,
        )
        print(
            f"[{label}] hybrid Gerr: exact-local+approx-nonlocal="
            f"{metrics[name + '_Gerr_exactlocal_appnonlocal']:.6e}, "
            f"approx-local+exact-nonlocal="
            f"{metrics[name + '_Gerr_applocal_exactnonlocal']:.6e}",
            flush=True,
        )

    out = args.out
    if out is None:
        out = args.input.with_name(args.input.stem + "_sigma_decomp.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        source=str(args.input),
        Lx=Lx,
        Ly=Ly,
        omega=omega,
        low_indices=idx,
        h0=h0,
        G_exact_k=Ged,
        K_exact=Kex,
        K_GW=Kgw,
        K_embedded=Kemb,
        K_exact_local=Lex,
        K_exact_nonlocal=Nex,
        K_GW_local=Lgw,
        K_GW_nonlocal=Ngw,
        K_embedded_local=Lemb,
        K_embedded_nonlocal=Nemb,
        mu=mu,
        mu_ed=mu_ed,
        mu_GW=mu_gw,
        **{k: np.asarray(v) for k, v in metrics.items()},
        **hybrids,
    )
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
