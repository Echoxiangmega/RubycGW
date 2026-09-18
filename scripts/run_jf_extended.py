#!/usr/bin/env python3
"""Run the production finite-q JF response for a Vprime/Vcross background.

This is a thin compatibility front-end around ``scripts/run_jf.py``.  It reads
the extended interaction parameters and cluster orientation from the background,
installs the same physical A-B pair interaction used by that impurity, reuses
the saved static Weiss shift, and reconstructs the full oriented lattice V(q).
"""
from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from rubycgw.cluster_orientation import gauge_transform_lattice, orientation_b_shift
from rubycgw.models.ruby import (
    ExtendedRubyParameters,
    build_extended_interaction,
    physical_pair_cluster_interactions,
)
from rubycgw.model import build_h0 as build_base_h0
from vprime_study.patches import install_cluster_interaction_hooks

# When this file is executed as ``python scripts/run_jf_extended.py``, the
# scripts directory is on sys.path and the maintained baseline driver can be
# imported directly.
import run_jf as _driver  # noqa: E402


def _load_background_metadata(path: Path) -> tuple[float, float, int, str]:
    with np.load(path, allow_pickle=False) as z:
        if "Vprime" in z:
            vp = float(np.asarray(z["Vprime"]).reshape(()))
        elif "Vp" in z:
            vp = float(np.asarray(z["Vp"]).reshape(()))
        else:
            vp = 0.0
        if "Vcross" in z:
            vx = float(np.asarray(z["Vcross"]).reshape(()))
        elif "Vx" in z:
            vx = float(np.asarray(z["Vx"]).reshape(()))
        else:
            vx = 0.0
        orientation = int(np.asarray(z["cluster_orientation"]).reshape(())) if "cluster_orientation" in z else 0
        projection = (
            str(np.asarray(z["cluster_projection"]).reshape(()))
            if "cluster_projection" in z else "legacy_unknown"
        )
    return vp, vx, orientation, projection


def _tag(x: float) -> str:
    return f"{float(x):g}"


def _append_metadata(path: Path, vp: float, vx: float, orientation: int) -> None:
    with np.load(path, allow_pickle=False) as z:
        payload = {k: np.asarray(z[k]) for k in z.files}
    payload["Vprime"] = np.asarray(float(vp))
    payload["Vp"] = np.asarray(float(vp))
    payload["Vcross"] = np.asarray(float(vx))
    payload["Vx"] = np.asarray(float(vx))
    payload["interaction_model"] = np.asarray(
        "lattice_full_V_Vprime_Vcross__cluster_ED_physical_pair"
    )
    payload["cluster_projection"] = np.asarray("physical_pair_no_intercell_collapse")
    payload["embedding_scheme"] = np.asarray(
        "ED(V+one_real_pair_Vprime_Vcross)+GW(full_lattice)"
    )
    payload["cluster_orientation"] = np.asarray(int(orientation))
    np.savez_compressed(path, **payload)


def main() -> None:
    args = _driver._args()
    vp, vx, orientation, projection = _load_background_metadata(Path(args.input))
    if projection != "physical_pair_no_intercell_collapse":
        raise ValueError(
            "input checkpoint is not from the physical-pair ED+GW partition: "
            f"cluster_projection={projection!r}. Recompute the background with the current main branch."
        )

    def _params_factory(*, ti=0.4, t1=0.2, t2=0.2, V=0.2, **_ignored):
        return ExtendedRubyParameters(
            ti=float(ti),
            t1=float(t1),
            t2=float(t2),
            V=float(V),
            Vprime=float(vp),
            Vcross=float(vx),
        )

    install_cluster_interaction_hooks(
        lambda p: physical_pair_cluster_interactions(p, int(orientation))
    )

    shift = orientation_b_shift(orientation)

    def _build_h0_oriented(kpts, params):
        return gauge_transform_lattice(build_base_h0(kpts, params), shift)

    def _build_interaction_oriented(qpts, params):
        return gauge_transform_lattice(build_extended_interaction(qpts, params), shift)

    if Path(args.out) == Path("cluster_ed_gw_jf_q.npz"):
        args.out = Path("results/jf_response") / (
            f"{Path(args.input).stem}_jf_q_Vp{_tag(vp)}_Vx{_tag(vx)}.npz"
        )

    original_args = _driver._args
    original_params = _driver.RubyParameters
    original_build_h0 = _driver.build_h0
    original_build_interaction = _driver.build_interaction
    _driver._args = lambda: args
    _driver.RubyParameters = _params_factory
    _driver.build_h0 = _build_h0_oriented
    _driver.build_interaction = _build_interaction_oriented
    try:
        try:
            _driver.main()
        except BaseException:
            partial = _driver._partial_path(Path(args.out))
            if partial.exists():
                _append_metadata(partial, vp, vx, orientation)
            raise
    finally:
        _driver._args = original_args
        _driver.RubyParameters = original_params
        _driver.build_h0 = original_build_h0
        _driver.build_interaction = original_build_interaction

    out = _driver._normalise_out(Path(args.out))
    _append_metadata(out, vp, vx, orientation)
    print(
        f"extended JF metadata: Vprime={vp:g}, Vcross={vx:g}, orientation={orientation}, "
        "cluster=physical-pair",
        flush=True,
    )


if __name__ == "__main__":
    main()
