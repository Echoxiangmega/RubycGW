"""Restart-state helpers for self-consistent cluster ED+GW runs.

The production runner saves all expensive dynamic variables needed to continue
an embedding solve.  This module validates a saved ``.npz`` checkpoint against
the requested physical problem and reconstructs the warm-start state used by
:func:`rubycgw.cluster_ed_gw_fast.solve_cluster_ed_gw_fast`.

Solver-control parameters such as the embedding tolerance, Pulay history length,
or maximum iteration count are deliberately *not* required to match.  A restart
is intended precisely for continuing the same physical calculation with tighter
or improved convergence settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cluster_ed_gw import BathParameters
from .grids import MatsubaraGrid
from .model import RubyParameters


@dataclass
class ClusterEDGWRestartState:
    """Dynamic state required to continue a cluster-ED+GW fixed point."""

    G: np.ndarray
    Sigma_H: np.ndarray
    Sigma_emb: np.ndarray
    Sigma_imp: np.ndarray
    mu: float
    bath: BathParameters
    source_path: str = ""


def _scalar(z, key: str, cast=float):
    if key not in z:
        raise ValueError(f"restart checkpoint is missing required field {key!r}")
    return cast(np.asarray(z[key]).reshape(()))


def _require_close(name: str, saved: float, requested: float, *, atol: float = 1e-12) -> None:
    if not np.isclose(float(saved), float(requested), rtol=1e-11, atol=atol):
        raise ValueError(
            f"restart {name} mismatch: checkpoint has {saved!r}, requested {requested!r}"
        )


def load_cluster_ed_gw_restart(
    path: str | Path,
    *,
    Lx: int,
    Ly: int,
    filling: float,
    T: float,
    params: RubyParameters,
    grid: MatsubaraGrid,
    nbath: int,
) -> ClusterEDGWRestartState:
    """Load and validate a saved cluster-ED+GW result for continuation.

    The physical lattice, interaction parameters, filling, temperature,
    Matsubara grids and bath size must match.  Numerical convergence settings
    may be changed freely after restart.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"restart checkpoint does not exist: {p}")

    with np.load(p, allow_pickle=False) as z:
        if _scalar(z, "Lx", int) != int(Lx) or _scalar(z, "Ly", int) != int(Ly):
            raise ValueError(
                "restart lattice-size mismatch: "
                f"checkpoint is {_scalar(z, 'Lx', int)}x{_scalar(z, 'Ly', int)}, "
                f"requested {int(Lx)}x{int(Ly)}"
            )
        _require_close("V", _scalar(z, "V"), float(params.V))
        _require_close("ti", _scalar(z, "ti"), float(params.ti))
        _require_close("t1", _scalar(z, "t1"), float(params.t1))
        _require_close("t2", _scalar(z, "t2"), float(params.t2))
        _require_close("filling", _scalar(z, "filling"), float(filling))
        _require_close("T", _scalar(z, "T"), float(T))

        omega = np.asarray(z["omega"], dtype=float) if "omega" in z else None
        Omega = np.asarray(z["Omega"], dtype=float) if "Omega" in z else None
        if omega is None or Omega is None:
            raise ValueError("restart checkpoint is missing omega/Omega grids")
        if omega.shape != np.asarray(grid.omega).shape or not np.allclose(
            omega, np.asarray(grid.omega), rtol=1e-12, atol=1e-13
        ):
            raise ValueError("restart fermionic Matsubara grid does not match requested grid")
        if Omega.shape != np.asarray(grid.Omega).shape or not np.allclose(
            Omega, np.asarray(grid.Omega), rtol=1e-12, atol=1e-13
        ):
            raise ValueError("restart bosonic Matsubara grid does not match requested grid")

        required = (
            "G",
            "Sigma_H",
            "Sigma_emb",
            "Sigma_ED_cluster",
            "mu",
            "bath_energies",
            "bath_couplings",
        )
        missing = [key for key in required if key not in z]
        if missing:
            raise ValueError(f"restart checkpoint is missing fields: {', '.join(missing)}")

        G = np.asarray(z["G"], dtype=complex)
        sigma_h = np.asarray(z["Sigma_H"], dtype=complex)
        sigma_emb = np.asarray(z["Sigma_emb"], dtype=complex)
        sigma_imp = np.asarray(z["Sigma_ED_cluster"], dtype=complex)
        mu = float(np.asarray(z["mu"]).reshape(()))
        energies = np.asarray(z["bath_energies"], dtype=float).reshape(-1)
        couplings = np.asarray(z["bath_couplings"], dtype=complex)

    expected_g = (grid.nf, int(Lx), int(Ly), 6, 6)
    expected_sigma_h = (6, 6)
    expected_imp = (grid.nf, 6, 6)
    if G.shape != expected_g:
        raise ValueError(f"restart G shape {G.shape} != expected {expected_g}")
    if sigma_emb.shape != expected_g:
        raise ValueError(
            f"restart Sigma_emb shape {sigma_emb.shape} != expected {expected_g}"
        )
    if sigma_h.shape != expected_sigma_h:
        raise ValueError(
            f"restart Sigma_H shape {sigma_h.shape} != expected {expected_sigma_h}"
        )
    if sigma_imp.shape != expected_imp:
        raise ValueError(
            f"restart Sigma_ED_cluster shape {sigma_imp.shape} != expected {expected_imp}"
        )
    if energies.size != int(nbath):
        raise ValueError(
            f"restart bath-size mismatch: checkpoint has {energies.size}, requested {int(nbath)}"
        )
    if couplings.shape != (6, int(nbath)):
        raise ValueError(
            f"restart bath_couplings shape {couplings.shape} != expected {(6, int(nbath))}"
        )
    arrays = (G, sigma_h, sigma_emb, sigma_imp, energies, couplings)
    if not all(np.all(np.isfinite(a)) for a in arrays) or not np.isfinite(mu):
        raise ValueError("restart checkpoint contains non-finite dynamic state")

    bath = BathParameters(
        energies=np.array(energies, copy=True),
        couplings=np.array(couplings, copy=True),
        fit_error=np.nan,
        nfev=0,
    )
    return ClusterEDGWRestartState(
        G=np.array(G, copy=True),
        Sigma_H=np.array(sigma_h, copy=True),
        Sigma_emb=np.array(sigma_emb, copy=True),
        Sigma_imp=np.array(sigma_imp, copy=True),
        mu=float(mu),
        bath=bath,
        source_path=str(p),
    )
