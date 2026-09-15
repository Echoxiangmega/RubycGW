#!/usr/bin/env python3
"""Run the production finite-q cluster-JF response for a V' + Vx background."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vprime_study.cross_model import (
    VPrimeCrossParameters,
    build_vprime_vcross_interaction,
    vprime_vcross_cluster_interactions,
)
from vprime_study.patches import install_cluster_interaction_hooks


install_cluster_interaction_hooks(vprime_vcross_cluster_interactions)

import scan_cluster_ed_gw_jf_q as _driver  # noqa: E402


def _load_interactions(path: Path) -> tuple[float, float]:
    with np.load(path, allow_pickle=False) as z:
        if "Vprime" in z:
            vp = float(np.asarray(z["Vprime"]).reshape(()))
        elif "Vp" in z:
            vp = float(np.asarray(z["Vp"]).reshape(()))
        else:
            raise KeyError("V'+Vx response input needs a Vprime (or Vp) scalar")
        if "Vcross" in z:
            vx = float(np.asarray(z["Vcross"]).reshape(()))
        elif "Vx" in z:
            vx = float(np.asarray(z["Vx"]).reshape(()))
        else:
            vx = 0.0
    return vp, vx


def _tag(x: float) -> str:
    return f"{float(x):g}"


def _append_metadata(path: Path, vp: float, vx: float) -> None:
    with np.load(path, allow_pickle=False) as z:
        payload = {k: np.asarray(z[k]) for k in z.files}
    payload["Vprime"] = np.asarray(float(vp))
    payload["Vp"] = np.asarray(float(vp))
    payload["Vcross"] = np.asarray(float(vx))
    payload["Vx"] = np.asarray(float(vx))
    payload["interaction_model"] = np.asarray(
        "V_intra_plus_Vprime_straight_plus_Vcross_diagonal"
    )
    payload["cluster_projection"] = np.asarray(
        "q0_primitive_cell_projection_sum_repeated_cross_pairs"
    )
    np.savez_compressed(path, **payload)


def main():
    args = _driver._args()
    vp, vx = _load_interactions(Path(args.input))

    def _params_factory(*, ti=0.4, t1=0.2, t2=0.2, V=0.2, **_ignored):
        return VPrimeCrossParameters(
            ti=float(ti), t1=float(t1), t2=float(t2),
            V=float(V), Vprime=float(vp), Vcross=float(vx),
        )

    if Path(args.out) == Path("cluster_ed_gw_jf_q.npz"):
        args.out = Path("results/vprime_vcross_response") / (
            f"{Path(args.input).stem}_jf_q_Vp{_tag(vp)}_Vx{_tag(vx)}.npz"
        )

    original_args = _driver._args
    original_params = _driver.RubyParameters
    original_build_interaction = _driver.build_interaction
    _driver._args = lambda: args
    _driver.RubyParameters = _params_factory
    _driver.build_interaction = build_vprime_vcross_interaction
    try:
        try:
            _driver.main()
        except BaseException:
            partial = _driver._partial_path(Path(args.out))
            if partial.exists():
                _append_metadata(partial, vp, vx)
                print(
                    f"V'+Vx metadata appended to partial checkpoint: V'={vp:g}, Vx={vx:g}",
                    flush=True,
                )
            raise
    finally:
        _driver._args = original_args
        _driver.RubyParameters = original_params
        _driver.build_interaction = original_build_interaction

    out = _driver._normalise_out(Path(args.out))
    _append_metadata(out, vp, vx)
    print(f"V'+Vx metadata appended: V'={vp:g}, Vx={vx:g}", flush=True)


if __name__ == "__main__":
    main()
