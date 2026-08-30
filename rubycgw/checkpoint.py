"""Checkpoint helpers for the 18-site periodic Ruby GW continuation.

Only the self-energy state required to restart a GW fixed-point iteration is
loaded back into memory: ``Sigma_H``, ``Sigma_GW`` and ``mu``.  The checkpoint
also stores densities and enough metadata to reject incompatible numerical
settings.  The interaction ``V`` is intentionally *not* required to match, since
continuation in V is the main use case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .grids import MatsubaraGrid
from .model import RubyParameters
from .supercell import NSUP, charge_order_parameter


CHECKPOINT_VERSION = 1


@dataclass
class GWCheckpointSeed:
    """Minimal object accepted by ``solve_supercell_gw(..., initial=...)``."""

    Sigma_H: np.ndarray
    Sigma_GW: np.ndarray
    mu: float


def _metadata(
    gw,
    params: RubyParameters,
    grid: MatsubaraGrid,
    primitive_filling: float,
    source: float,
) -> dict:
    phi = charge_order_parameter(np.asarray(gw.density, dtype=float))
    return {
        "version": CHECKPOINT_VERSION,
        "matrix_dimension": NSUP,
        "V": float(params.V),
        "source": float(source),
        "primitive_filling": float(primitive_filling),
        "T": float(grid.T),
        "nk1": int(grid.nk1),
        "nk2": int(grid.nk2),
        "nw": int(grid.nw),
        "nOmega": int(grid.nOmega),
        "ti": float(params.ti),
        "t1": float(params.t1),
        "t2": float(params.t2),
        "mu": float(gw.mu),
        "charge_order_re": float(phi.real),
        "charge_order_im": float(phi.imag),
        "charge_order_abs": float(abs(phi)),
        "final_error": float(gw.final_error),
        "converged": bool(gw.converged),
    }


def checkpoint_filename(
    V: float,
    primitive_filling: float,
    grid: MatsubaraGrid,
) -> str:
    """Stable filename that keeps different numerical grids separate."""
    return (
        f"V{float(V):.6f}_n{float(primitive_filling):.6f}_"
        f"nk{grid.nk1}x{grid.nk2}_nw{grid.nw}_no{grid.nOmega}_T{grid.T:.6g}.npz"
    )


def save_supercell_checkpoint(
    path: str | Path,
    gw,
    params: RubyParameters,
    grid: MatsubaraGrid,
    primitive_filling: float,
    source: float = 0.0,
) -> Path:
    """Save a converged supercell GW state for later continuation."""
    if not bool(gw.converged):
        raise ValueError("Refusing to checkpoint a non-converged GW state.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = _metadata(gw, params, grid, primitive_filling, source)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(meta)),
        Sigma_H=np.asarray(gw.Sigma_H, dtype=complex),
        Sigma_GW=np.asarray(gw.Sigma_GW, dtype=complex),
        density=np.asarray(gw.density, dtype=float),
    )
    return path


def read_checkpoint_metadata(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"].item()))


def _float_match(a: float, b: float, atol: float = 1e-12) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=atol))


def checkpoint_compatibility_error(
    meta: dict,
    params: RubyParameters,
    grid: MatsubaraGrid,
    primitive_filling: float,
) -> str | None:
    """Return None when a checkpoint can seed the requested calculation."""
    exact = {
        "version": CHECKPOINT_VERSION,
        "matrix_dimension": NSUP,
        "nk1": grid.nk1,
        "nk2": grid.nk2,
        "nw": grid.nw,
        "nOmega": grid.nOmega,
    }
    for key, expected in exact.items():
        if int(meta.get(key, -999999)) != int(expected):
            return f"{key}: checkpoint={meta.get(key)!r}, requested={expected!r}"

    floats = {
        "T": grid.T,
        "primitive_filling": primitive_filling,
        "ti": params.ti,
        "t1": params.t1,
        "t2": params.t2,
    }
    for key, expected in floats.items():
        if key not in meta or not _float_match(meta[key], expected):
            return f"{key}: checkpoint={meta.get(key)!r}, requested={expected!r}"

    # V is deliberately omitted: a checkpoint is meant to seed a different V.
    return None


def load_supercell_checkpoint(
    path: str | Path,
    params: RubyParameters,
    grid: MatsubaraGrid,
    primitive_filling: float,
) -> tuple[GWCheckpointSeed, dict, np.ndarray]:
    """Load and validate a checkpoint for the requested numerical setup."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"].item()))
        error = checkpoint_compatibility_error(meta, params, grid, primitive_filling)
        if error is not None:
            raise ValueError(f"Incompatible checkpoint {path}: {error}")

        sigma_h = np.asarray(data["Sigma_H"], dtype=complex)
        sigma_gw = np.asarray(data["Sigma_GW"], dtype=complex)
        density = np.asarray(data["density"], dtype=float)

    expected_h = (NSUP, NSUP)
    expected_gw = (grid.nf, grid.nk1, grid.nk2, NSUP, NSUP)
    if sigma_h.shape != expected_h:
        raise ValueError(f"Checkpoint Sigma_H shape {sigma_h.shape} != {expected_h}")
    if sigma_gw.shape != expected_gw:
        raise ValueError(f"Checkpoint Sigma_GW shape {sigma_gw.shape} != {expected_gw}")
    if density.shape != (NSUP,):
        raise ValueError(f"Checkpoint density shape {density.shape} != {(NSUP,)}")

    seed = GWCheckpointSeed(
        Sigma_H=np.array(sigma_h, copy=True),
        Sigma_GW=np.array(sigma_gw, copy=True),
        mu=float(meta["mu"]),
    )
    return seed, meta, density


def find_nearest_compatible_checkpoint(
    directory: str | Path,
    target_V: float,
    params: RubyParameters,
    grid: MatsubaraGrid,
    primitive_filling: float,
) -> Path | None:
    """Find the largest compatible zero-source V not exceeding target_V."""
    directory = Path(directory)
    if not directory.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for path in directory.glob("*.npz"):
        try:
            meta = read_checkpoint_metadata(path)
        except Exception:
            continue
        if checkpoint_compatibility_error(meta, params, grid, primitive_filling) is not None:
            continue
        if not bool(meta.get("converged", False)):
            continue
        if not _float_match(meta.get("source", np.nan), 0.0, atol=1e-14):
            continue
        V = float(meta.get("V", np.nan))
        if np.isfinite(V) and V <= float(target_V) + 1e-12:
            candidates.append((V, path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]
