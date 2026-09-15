"""Standardized NPZ IO for public cluster-background workflows."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..workflows.background import BackgroundRun

FORMAT_VERSION = 1


def save_background(path: str | Path, run: BackgroundRun) -> Path:
    """Save a :class:`BackgroundRun` in the established checkpoint-compatible format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = run.result
    m = run.model
    c = run.config
    np.savez_compressed(
        path,
        format_name=np.asarray("rubycgw_cluster_background"),
        format_version=np.asarray(FORMAT_VERSION, dtype=int),
        restart_mode=np.asarray(run.restart_mode),
        restart_source=np.asarray(run.restart_source),
        Lx=int(c.grid.Lx),
        Ly=int(c.grid.Ly),
        V=float(m.V),
        Vprime=float(m.Vprime),
        Vp=float(m.Vprime),
        Vcross=float(m.Vcross),
        Vx=float(m.Vcross),
        filling=float(c.filling),
        T=float(c.grid.T),
        ti=float(m.ti),
        t1=float(m.t1),
        t2=float(m.t2),
        omega=np.asarray(run.grid.omega),
        Omega=np.asarray(run.grid.Omega),
        converged=bool(r.converged),
        iterations=int(r.iterations),
        final_error=float(r.final_error),
        impurity_mismatch=float(r.impurity_mismatch),
        bath_fit_error=float(r.bath_fit_error),
        mixing_method=np.asarray(str(r.mixing_method)),
        pulay_fallbacks=int(r.pulay_fallbacks),
        residual_history=np.asarray(r.residual_history),
        impurity_residual_history=np.asarray(r.impurity_residual_history),
        impurity_mismatch_history=np.asarray(r.impurity_mismatch_history),
        bath_fit_history=np.asarray(r.bath_fit_history),
        mu_history=np.asarray(r.mu_history),
        bath_nfev_history=np.asarray(r.bath_nfev_history),
        elapsed_history=np.asarray(r.elapsed_history),
        mu=float(r.mu),
        density=np.asarray(r.density),
        G=np.asarray(r.G),
        W=np.asarray(r.W),
        P=np.asarray(r.P),
        Sigma_H=np.asarray(r.Sigma_H),
        Sigma_emb=np.asarray(r.Sigma_emb),
        Sigma_GW_lattice=np.asarray(r.Sigma_GW_lattice),
        Sigma_GW_cluster=np.asarray(r.Sigma_GW_cluster),
        Sigma_ED_cluster=np.asarray(r.Sigma_ED_cluster),
        G_cluster=np.asarray(r.G_cluster),
        G_impurity=np.asarray(r.G_impurity),
        bath_energies=np.asarray(r.bath.energies),
        bath_couplings=np.asarray(r.bath.couplings),
        G_background=np.asarray(r.background.G),
        mu_background=float(r.background.mu),
    )
    return path


def background_metadata(path: str | Path) -> dict[str, object]:
    """Read lightweight scalar metadata without exposing the full arrays."""
    keys = (
        "format_name",
        "format_version",
        "Lx",
        "Ly",
        "V",
        "Vprime",
        "Vcross",
        "filling",
        "T",
        "ti",
        "t1",
        "t2",
        "converged",
        "iterations",
        "final_error",
        "impurity_mismatch",
        "bath_fit_error",
        "mu",
    )
    out: dict[str, object] = {}
    with np.load(Path(path), allow_pickle=False) as z:
        for key in keys:
            if key not in z:
                continue
            a = np.asarray(z[key])
            out[key] = a.reshape(()).item() if a.size == 1 else a.copy()
    return out


__all__ = ["FORMAT_VERSION", "save_background", "background_metadata"]
