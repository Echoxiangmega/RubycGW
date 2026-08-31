"""Fourier-interpolate 18-site GW checkpoints between regular k meshes.

A converged checkpoint on one supercell k mesh can be used as a warm start on
another mesh without changing the target GW equations.  Sigma_H, mu and the
18-site density are copied, while the full Matsubara-dependent Sigma_GW(k) is
periodically trigonometric-interpolated in the two reduced supercell momenta.

The interpolated file is deliberately marked ``converged=False``.  It is only a
warm start and must be passed explicitly to ``run_supercell_gw.py`` so the new
k mesh is self-consistently converged before the state is used physically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .checkpoint import CHECKPOINT_VERSION, checkpoint_filename
from .grids import MatsubaraGrid
from .supercell import NSUP


def interpolate_sigma_gw_kmesh(
    sigma_gw: np.ndarray,
    target_nk1: int,
    target_nk2: int,
) -> np.ndarray:
    """Periodic Fourier interpolation of Sigma_GW from one regular mesh to another.

    Parameters
    ----------
    sigma_gw
        Array with shape ``(nf,nk1,nk2,NSUP,NSUP)`` sampled at reduced momenta
        ``k_i=j_i/nk_i``.
    target_nk1, target_nk2
        Dimensions of the new regular reduced-k mesh.

    Notes
    -----
    The interpolation uses the same phase convention as the Ruby model,
    ``exp(+2*pi*i*k.R)``.  Therefore it exactly reproduces all Fourier modes
    representable on the source mesh and exactly reproduces the original data
    when evaluated again on the source mesh.
    """
    sigma = np.asarray(sigma_gw, dtype=complex)
    if sigma.ndim != 5 or sigma.shape[-2:] != (NSUP, NSUP):
        raise ValueError(
            "sigma_gw must have shape (nf,nk1,nk2,NSUP,NSUP); "
            f"got {sigma.shape}"
        )

    target_nk1 = int(target_nk1)
    target_nk2 = int(target_nk2)
    if target_nk1 < 1 or target_nk2 < 1:
        raise ValueError("target k-mesh dimensions must be positive")

    _, nk1, nk2, _, _ = sigma.shape
    if target_nk1 == nk1 and target_nk2 == nk2:
        return np.array(sigma, copy=True)

    # If f(k)=sum_R c_R exp(+2*pi*i*k.R), numpy.fft.fft2 returns
    # (nk1*nk2)*c_R on the regular k mesh.
    coeff = np.fft.fft2(sigma, axes=(1, 2)) / float(nk1 * nk2)
    r1 = np.fft.fftfreq(nk1, d=1.0 / nk1)
    r2 = np.fft.fftfreq(nk2, d=1.0 / nk2)

    k1 = np.arange(target_nk1, dtype=float) / float(target_nk1)
    k2 = np.arange(target_nk2, dtype=float) / float(target_nk2)
    phase1 = np.exp(2j * np.pi * k1[:, None] * r1[None, :])
    phase2 = np.exp(2j * np.pi * k2[:, None] * r2[None, :])

    out = np.einsum(
        "xi,yj,nijab->nxyab",
        phase1,
        phase2,
        coeff,
        optimize=True,
    )
    return np.asarray(out, dtype=complex)


def _load_raw_checkpoint(path: str | Path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"].item()))
        sigma_h = np.asarray(data["Sigma_H"], dtype=complex)
        sigma_gw = np.asarray(data["Sigma_GW"], dtype=complex)
        density = np.asarray(data["density"], dtype=float)

    required = {
        "version",
        "matrix_dimension",
        "V",
        "source",
        "primitive_filling",
        "T",
        "nk1",
        "nk2",
        "nw",
        "nOmega",
        "ti",
        "t1",
        "t2",
        "mu",
    }
    missing = sorted(required.difference(meta))
    if missing:
        raise ValueError(f"Checkpoint is missing metadata keys: {missing}")
    if int(meta["version"]) != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version {meta['version']} != supported {CHECKPOINT_VERSION}"
        )
    if int(meta["matrix_dimension"]) != NSUP:
        raise ValueError("Checkpoint is not an 18-site supercell state")
    if sigma_h.shape != (NSUP, NSUP):
        raise ValueError(f"Unexpected Sigma_H shape {sigma_h.shape}")

    expected = (
        2 * int(meta["nw"]),
        int(meta["nk1"]),
        int(meta["nk2"]),
        NSUP,
        NSUP,
    )
    if sigma_gw.shape != expected:
        raise ValueError(
            f"Sigma_GW shape {sigma_gw.shape} does not match metadata {expected}"
        )
    if density.shape != (NSUP,):
        raise ValueError(f"Unexpected density shape {density.shape}")
    return meta, sigma_h, sigma_gw, density


def interpolated_checkpoint_default_path(
    source_path: str | Path,
    target_nk1: int,
    target_nk2: int,
) -> Path:
    """Return the ordinary checkpoint filename for the target k mesh."""
    source_path = Path(source_path)
    meta, _, _, _ = _load_raw_checkpoint(source_path)
    grid = MatsubaraGrid(
        nk1=int(target_nk1),
        nk2=int(target_nk2),
        nw=int(meta["nw"]),
        nOmega=int(meta["nOmega"]),
        T=float(meta["T"]),
    )
    return source_path.with_name(
        checkpoint_filename(float(meta["V"]), float(meta["primitive_filling"]), grid)
    )


def write_interpolated_checkpoint(
    source_path: str | Path,
    target_nk1: int,
    target_nk2: int,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict]:
    """Create a target-mesh warm-start checkpoint from an existing checkpoint.

    The output uses normal checkpoint metadata for the *target* mesh so it can
    be loaded by the existing driver with ``--restart-from``.  It is marked as
    non-converged and assigned a large placeholder residual, forcing the driver
    to self-consistently solve the target mesh even if V is unchanged.
    """
    source_path = Path(source_path)
    meta, sigma_h, sigma_gw, density = _load_raw_checkpoint(source_path)

    target_nk1 = int(target_nk1)
    target_nk2 = int(target_nk2)
    old_nk1 = int(meta["nk1"])
    old_nk2 = int(meta["nk2"])
    if target_nk1 < 1 or target_nk2 < 1:
        raise ValueError("target k-mesh dimensions must be positive")
    if target_nk1 == old_nk1 and target_nk2 == old_nk2:
        raise ValueError(
            f"Source checkpoint already uses {old_nk1}x{old_nk2}; "
            "choose a different target mesh"
        )

    target_sigma_gw = interpolate_sigma_gw_kmesh(
        sigma_gw, target_nk1, target_nk2
    )

    out = (
        Path(output_path)
        if output_path is not None
        else interpolated_checkpoint_default_path(
            source_path, target_nk1, target_nk2
        )
    )
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out}. Pass overwrite=True/--force to replace it."
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    target_meta = dict(meta)
    target_meta["nk1"] = target_nk1
    target_meta["nk2"] = target_nk2
    target_meta["converged"] = False
    # This is intentionally large so a same-V explicit restart is always run.
    target_meta["final_error"] = 1.0
    target_meta["interpolated_warm_start"] = True
    target_meta["interpolated_from"] = str(source_path)
    target_meta["interpolated_from_nk1"] = old_nk1
    target_meta["interpolated_from_nk2"] = old_nk2
    target_meta["interpolated_source_final_error"] = float(
        meta.get("final_error", np.nan)
    )

    np.savez_compressed(
        out,
        metadata_json=np.asarray(json.dumps(target_meta)),
        Sigma_H=np.asarray(sigma_h, dtype=complex),
        Sigma_GW=np.asarray(target_sigma_gw, dtype=complex),
        density=np.asarray(density, dtype=float),
    )
    return out, target_meta
