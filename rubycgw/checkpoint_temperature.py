"""Temperature-interpolate 18-site GW checkpoints for warm starts.

A converged Matsubara-axis GW checkpoint at one temperature can seed a nearby
calculation at another temperature without changing the target GW equations.
``Sigma_H``, ``mu`` and the 18-site density are copied, while the full
frequency-dependent ``Sigma_GW(iw,k)`` is interpolated from the source
fermionic Matsubara frequencies to the target ones.

Because the split-GW self-energy has the high-frequency form

    Sigma_GW(iw,k) = Sigma_inf(k) + O(1/iw),

we first estimate the static limit from symmetric high-|w| pairs.  The dynamic
remainder is linearly interpolated inside the source frequency window and is
continued as 1/w outside that window.  The output is deliberately marked
``converged=False`` and must be reconverged at the target temperature.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .checkpoint import CHECKPOINT_VERSION, checkpoint_filename
from .grids import MatsubaraGrid
from .supercell import NSUP


def _fermionic_omega(nw: int, T: float) -> np.ndarray:
    nw = int(nw)
    T = float(T)
    if nw < 1:
        raise ValueError("nw must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    n = np.arange(-nw, nw, dtype=float)
    return (2.0 * n + 1.0) * np.pi * T


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


def estimate_sigma_infinity(
    sigma_gw: np.ndarray,
    tail_pairs: int = 4,
) -> np.ndarray:
    """Estimate the static high-frequency self-energy from +/-iw tail pairs.

    The Matsubara identity ``Sigma(-iw)=Sigma(iw)^dagger`` implies that the
    average of opposite-frequency values removes the leading odd 1/iw tail.
    Averaging several largest-|w| pairs therefore provides a stable warm-start
    estimate of the frequency-independent Fock-like limit.
    """
    sigma = np.asarray(sigma_gw, dtype=complex)
    if sigma.ndim != 5 or sigma.shape[-2:] != (NSUP, NSUP):
        raise ValueError(
            "sigma_gw must have shape (nf,nk1,nk2,NSUP,NSUP); "
            f"got {sigma.shape}"
        )
    nf = int(sigma.shape[0])
    if nf % 2 != 0:
        raise ValueError("fermionic Matsubara axis must have even length")
    nw = nf // 2
    npairs = min(max(int(tail_pairs), 1), nw)

    even_tail = []
    for p in range(nw - npairs, nw):
        ip = nw + p
        im = nw - 1 - p
        even_tail.append(0.5 * (sigma[ip] + sigma[im]))
    sigma_inf = np.mean(np.stack(even_tail, axis=0), axis=0)
    return 0.5 * (sigma_inf + np.swapaxes(sigma_inf.conj(), -1, -2))


def _enforce_matsubara_hermiticity(field: np.ndarray) -> np.ndarray:
    out = np.array(field, dtype=complex, copy=True)
    nf = int(out.shape[0])
    if nf % 2 != 0:
        raise ValueError("fermionic Matsubara axis must have even length")
    nw = nf // 2
    for p in range(nw):
        ip = nw + p
        im = nw - 1 - p
        plus = 0.5 * (out[ip] + np.swapaxes(out[im].conj(), -1, -2))
        out[ip] = plus
        out[im] = np.swapaxes(plus.conj(), -1, -2)
    return out


def interpolate_sigma_gw_temperature(
    sigma_gw: np.ndarray,
    source_T: float,
    target_T: float,
    target_nw: int | None = None,
    tail_pairs: int = 4,
) -> np.ndarray:
    """Interpolate ``Sigma_GW(iw,k)`` between fermionic Matsubara grids.

    Inside the source frequency window the dynamic part is linearly
    interpolated in the physical Matsubara frequency ``omega``.  Outside that
    window it is continued with the leading high-frequency ``1/omega`` law.
    This is only a warm-start construction; the target-temperature GW equations
    must subsequently be solved self-consistently.
    """
    sigma = np.asarray(sigma_gw, dtype=complex)
    if sigma.ndim != 5 or sigma.shape[-2:] != (NSUP, NSUP):
        raise ValueError(
            "sigma_gw must have shape (nf,nk1,nk2,NSUP,NSUP); "
            f"got {sigma.shape}"
        )
    nf = int(sigma.shape[0])
    if nf % 2 != 0:
        raise ValueError("fermionic Matsubara axis must have even length")
    source_nw = nf // 2
    target_nw = source_nw if target_nw is None else int(target_nw)
    if target_nw < 1:
        raise ValueError("target_nw must be positive")
    source_T = float(source_T)
    target_T = float(target_T)
    if source_T <= 0.0 or target_T <= 0.0:
        raise ValueError("source and target temperatures must be positive")

    if np.isclose(source_T, target_T, rtol=0.0, atol=0.0) and target_nw == source_nw:
        return np.array(sigma, copy=True)

    old_omega = _fermionic_omega(source_nw, source_T)
    new_omega = _fermionic_omega(target_nw, target_T)
    sigma_inf = estimate_sigma_infinity(sigma, tail_pairs=tail_pairs)
    dynamic = sigma - sigma_inf[None, ...]

    out_dynamic = np.empty(
        (2 * target_nw,) + sigma.shape[1:],
        dtype=complex,
    )

    for it, w in enumerate(new_omega):
        if w < old_omega[0]:
            ratio = old_omega[0] / w
            out_dynamic[it] = dynamic[0] * ratio
            continue
        if w > old_omega[-1]:
            ratio = old_omega[-1] / w
            out_dynamic[it] = dynamic[-1] * ratio
            continue

        hi = int(np.searchsorted(old_omega, w, side="left"))
        if hi == 0:
            out_dynamic[it] = dynamic[0]
        elif hi >= old_omega.size:
            out_dynamic[it] = dynamic[-1]
        elif np.isclose(w, old_omega[hi], rtol=0.0, atol=1e-15):
            out_dynamic[it] = dynamic[hi]
        else:
            lo = hi - 1
            denom = old_omega[hi] - old_omega[lo]
            alpha = float((w - old_omega[lo]) / denom)
            out_dynamic[it] = (1.0 - alpha) * dynamic[lo] + alpha * dynamic[hi]

    out = sigma_inf[None, ...] + out_dynamic
    return _enforce_matsubara_hermiticity(out)


def temperature_checkpoint_default_path(
    source_path: str | Path,
    target_T: float,
    target_nw: int | None = None,
) -> Path:
    source_path = Path(source_path)
    meta, _, _, _ = _load_raw_checkpoint(source_path)
    nw = int(meta["nw"]) if target_nw is None else int(target_nw)
    grid = MatsubaraGrid(
        nk1=int(meta["nk1"]),
        nk2=int(meta["nk2"]),
        nw=nw,
        nOmega=int(meta["nOmega"]),
        T=float(target_T),
    )
    return source_path.with_name(
        checkpoint_filename(float(meta["V"]), float(meta["primitive_filling"]), grid)
    )


def write_temperature_interpolated_checkpoint(
    source_path: str | Path,
    target_T: float,
    target_nw: int | None = None,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    tail_pairs: int = 4,
) -> tuple[Path, dict]:
    """Create a target-temperature warm-start checkpoint.

    The output metadata describes the target temperature/frequency grid so the
    ordinary driver can load it explicitly with ``--restart-from``.  It is
    marked non-converged with a large placeholder residual to force a genuine
    target-temperature self-consistency cycle.
    """
    source_path = Path(source_path)
    meta, sigma_h, sigma_gw, density = _load_raw_checkpoint(source_path)

    source_T = float(meta["T"])
    target_T = float(target_T)
    if target_T <= 0.0:
        raise ValueError("target_T must be positive")
    old_nw = int(meta["nw"])
    new_nw = old_nw if target_nw is None else int(target_nw)
    if new_nw < 1:
        raise ValueError("target_nw must be positive")
    if np.isclose(target_T, source_T, rtol=0.0, atol=0.0) and new_nw == old_nw:
        raise ValueError("Target temperature/frequency grid is identical to the source")

    target_sigma_gw = interpolate_sigma_gw_temperature(
        sigma_gw,
        source_T=source_T,
        target_T=target_T,
        target_nw=new_nw,
        tail_pairs=tail_pairs,
    )

    out = (
        Path(output_path)
        if output_path is not None
        else temperature_checkpoint_default_path(
            source_path,
            target_T=target_T,
            target_nw=new_nw,
        )
    )
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out}. Pass overwrite=True/--force to replace it."
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    target_meta = dict(meta)
    target_meta["T"] = target_T
    target_meta["nw"] = new_nw
    target_meta["converged"] = False
    target_meta["final_error"] = 1.0
    target_meta["temperature_interpolated_warm_start"] = True
    target_meta["temperature_interpolated_from"] = str(source_path)
    target_meta["temperature_interpolated_from_T"] = source_T
    target_meta["temperature_interpolated_from_nw"] = old_nw
    target_meta["temperature_interpolation_tail_pairs"] = int(tail_pairs)
    target_meta["temperature_interpolated_source_final_error"] = float(
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
