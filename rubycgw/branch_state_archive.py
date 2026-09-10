"""Persistent full-state history for pseudo-arclength continuation branches.

The lightweight PAC checkpoint only needs the last two states to resume.  For
post-processing a non-monotonic branch, however, we also want every accepted
continuation state.  This module stores those full real codec vectors together
with their V values and CSV step identifiers in one atomic NPZ archive.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np


@dataclass
class BranchStateArchive:
    signature: str
    step: np.ndarray
    kind: np.ndarray
    V: np.ndarray
    X: np.ndarray

    @property
    def nstate(self) -> int:
        return int(self.X.shape[0])

    @property
    def state_size(self) -> int:
        return int(self.X.shape[1])


def _validated_arrays(steps, kinds, V_values, X_values):
    step = np.asarray(steps, dtype=np.int64).reshape(-1)
    kind = np.asarray(kinds, dtype=str).reshape(-1)
    V = np.asarray(V_values, dtype=float).reshape(-1)
    X = np.asarray(X_values, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("X history must be a 2D array of codec states")
    n = int(X.shape[0])
    if step.size != n or kind.size != n or V.size != n:
        raise ValueError("step/kind/V/X histories must have the same length")
    if n == 0:
        raise ValueError("branch-state archive cannot be empty")
    if not np.all(np.isfinite(V)) or not np.all(np.isfinite(X)):
        raise ValueError("branch-state archive contains non-finite values")
    return step, kind, V, X


def save_branch_state_archive(
    path: Path | str,
    signature: str,
    steps,
    kinds,
    V_values,
    X_values,
) -> Path:
    """Atomically save all accepted full PAC states."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    step, kind, V, X = _validated_arrays(steps, kinds, V_values, X_values)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(
            tmp_path,
            signature=np.asarray(str(signature)),
            step=step,
            kind=kind,
            V=V,
            X=X,
            state_size=np.asarray(X.shape[1], dtype=np.int64),
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def load_branch_state_archive(
    path: Path | str,
    *,
    expected_signature: str | None = None,
    expected_state_size: int | None = None,
) -> BranchStateArchive:
    """Load and validate a full PAC state archive."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        signature = str(np.asarray(data["signature"]).item())
        step = np.asarray(data["step"], dtype=np.int64)
        kind = np.asarray(data["kind"], dtype=str)
        V = np.asarray(data["V"], dtype=float)
        X = np.asarray(data["X"], dtype=float)
    step, kind, V, X = _validated_arrays(step, kind, V, X)
    if expected_signature is not None and signature != str(expected_signature):
        raise RuntimeError("branch-state archive signature does not match this run")
    if expected_state_size is not None and X.shape[1] != int(expected_state_size):
        raise RuntimeError("branch-state archive codec size does not match this run")
    return BranchStateArchive(signature, step, kind, V, X)


def append_state_in_memory(
    archive: BranchStateArchive,
    step: int,
    kind: str,
    V: float,
    x: np.ndarray,
) -> BranchStateArchive:
    """Return an archive with one accepted state appended."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != archive.state_size:
        raise ValueError("appended state has the wrong codec size")
    return BranchStateArchive(
        archive.signature,
        np.concatenate([archive.step, np.asarray([int(step)], dtype=np.int64)]),
        np.concatenate([archive.kind, np.asarray([str(kind)], dtype=str)]),
        np.concatenate([archive.V, np.asarray([float(V)], dtype=float)]),
        np.concatenate([archive.X, x[None, :]], axis=0),
    )


def states_match(x1, V1, x2, V2, *, atol: float = 1e-12) -> bool:
    """Whether two archived/checkpoint states are numerically identical."""
    a = np.asarray(x1, dtype=float).reshape(-1)
    b = np.asarray(x2, dtype=float).reshape(-1)
    return bool(
        a.shape == b.shape
        and abs(float(V1) - float(V2)) <= float(atol)
        and np.allclose(a, b, rtol=0.0, atol=float(atol))
    )


__all__ = [
    "BranchStateArchive",
    "save_branch_state_archive",
    "load_branch_state_archive",
    "append_state_in_memory",
    "states_match",
]
