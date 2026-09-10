"""Small persistent cache for expensive exact-diagonalization benchmarks.

The cache is intentionally conservative: a cached result is reused only when a
canonical parameter signature and the fermionic Matsubara grid match exactly.
This keeps ED reference data separate from the approximate solver settings, so
changing GW mixing, channel choice, vertex tolerance, or nOmega does not force
ED to be recomputed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


ED_CACHE_VERSION = 1


def canonical_ed_signature(signature: dict) -> str:
    """Return a stable JSON representation of an ED benchmark signature."""
    payload = {"cache_version": ED_CACHE_VERSION, **dict(signature)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def ed_cache_path(cache_dir: Path | str, signature: dict) -> Path:
    """Return the deterministic cache path for ``signature``."""
    cache_dir = Path(cache_dir)
    canonical = canonical_ed_signature(signature)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    L1 = int(signature["L1"])
    L2 = int(signature["L2"])
    V = float(signature["V"])
    filling = float(signature["filling"])
    T = float(signature["T"])
    nw = int(signature["nw"])
    stem = f"ed_{L1}x{L2}_V{V:g}_fill{filling:g}_T{T:g}_nw{nw}_{digest}.npz"
    return cache_dir / stem


def load_ed_cache(
    cache_dir: Path | str,
    signature: dict,
    expected_omega: np.ndarray,
    nsite: int,
):
    """Load a validated cache entry, or return ``None`` on any mismatch."""
    path = ed_cache_path(cache_dir, signature)
    if not path.exists():
        return None
    expected_sig = canonical_ed_signature(signature)
    try:
        with np.load(path, allow_pickle=False) as data:
            stored_sig = str(np.asarray(data["signature_json"]).item())
            omega = np.asarray(data["omega"], dtype=float)
            mu_ed = float(np.asarray(data["mu_ed"]).item())
            G_ed = np.asarray(data["G_ed"], dtype=complex)
            chi_ed = np.asarray(data["chi_ed"], dtype=float)
    except (OSError, ValueError, KeyError, TypeError):
        return None

    expected_omega = np.asarray(expected_omega, dtype=float)
    if stored_sig != expected_sig:
        return None
    if omega.shape != expected_omega.shape or not np.array_equal(omega, expected_omega):
        return None
    if G_ed.shape != (expected_omega.size, int(nsite), int(nsite)):
        return None
    if chi_ed.shape != (2, 2):
        return None
    if not np.isfinite(mu_ed):
        return None
    return {
        "path": path,
        "mu_ed": mu_ed,
        "G_ed": G_ed,
        "chi_ed": chi_ed,
    }


def save_ed_cache(
    cache_dir: Path | str,
    signature: dict,
    omega: np.ndarray,
    mu_ed: float,
    G_ed: np.ndarray,
    chi_ed: np.ndarray,
) -> Path:
    """Atomically save an ED reference result and return its cache path."""
    path = ed_cache_path(cache_dir, signature)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(
            tmp_path,
            signature_json=np.asarray(canonical_ed_signature(signature)),
            omega=np.asarray(omega, dtype=float),
            mu_ed=float(mu_ed),
            G_ed=np.asarray(G_ed, dtype=complex),
            chi_ed=np.asarray(chi_ed, dtype=float),
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path
