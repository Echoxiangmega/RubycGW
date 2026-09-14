#!/usr/bin/env python3
"""Run the production all-q cluster-JF response for a V-prime background.

The command-line interface is the same as ``scan_cluster_ed_gw_jf_q.py``.  The
input NPZ must have been produced by ``run_cluster_ed_gw_vprime.py`` and contain
``Vprime`` (or ``Vp``).  The six-channel response then determines q, Pauli
character and A/B parity without preselecting a current pattern.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vprime_study.model import VPrimeParameters, build_vprime_interaction
from vprime_study.patches import install_vprime_cluster_hooks


install_vprime_cluster_hooks()

import scan_cluster_ed_gw_jf_q as _driver  # noqa: E402


def _load_vprime(path: Path) -> float:
    with np.load(path, allow_pickle=False) as z:
        if "Vprime" in z:
            return float(np.asarray(z["Vprime"]).reshape(()))
        if "Vp" in z:
            return float(np.asarray(z["Vp"]).reshape(()))
    raise KeyError("V-prime response input needs a Vprime (or Vp) scalar")


def _tag(x: float) -> str:
    return f"{float(x):g}"


def _append_metadata(path: Path, vp: float) -> None:
    with np.load(path, allow_pickle=False) as z:
        payload = {k: np.asarray(z[k]) for k in z.files}
    payload["Vprime"] = np.asarray(float(vp))
    payload["Vp"] = np.asarray(float(vp))
    payload["interaction_model"] = np.asarray("V_intra_plus_Vprime_intertriangle")
    payload["cluster_projection"] = np.asarray("q0_primitive_cell_projection")
    np.savez_compressed(path, **payload)


def main():
    args = _driver._args()
    vp = _load_vprime(Path(args.input))

    def _params_factory(*, ti=0.4, t1=0.2, t2=0.2, V=0.2, **_ignored):
        return VPrimeParameters(
            ti=float(ti), t1=float(t1), t2=float(t2),
            V=float(V), Vprime=float(vp),
        )

    if Path(args.out) == Path("cluster_ed_gw_jf_q.npz"):
        args.out = Path("results/vprime_response") / (
            f"{Path(args.input).stem}_jf_q_Vp{_tag(vp)}.npz"
        )

    original_args = _driver._args
    original_params = _driver.RubyParameters
    original_build_interaction = _driver.build_interaction
    _driver._args = lambda: args
    _driver.RubyParameters = _params_factory
    _driver.build_interaction = build_vprime_interaction
    try:
        _driver.main()
    finally:
        _driver._args = original_args
        _driver.RubyParameters = original_params
        _driver.build_interaction = original_build_interaction

    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    _append_metadata(out, vp)
    print(f"V-prime metadata appended: V'={vp:g}", flush=True)


if __name__ == "__main__":
    main()
