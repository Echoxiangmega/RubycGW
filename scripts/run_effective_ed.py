#!/usr/bin/env python3
"""Run finite-size ED of the Ruby strong-coupling pseudospin model."""
from __future__ import annotations

import argparse

from rubycgw.analysis import summarize_effective_ed
from rubycgw.api import EffectiveEDConfig, RubyModel, run_effective_ed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--t", type=float, default=0.2)
    p.add_argument("--V", type=float, default=1.8)
    p.add_argument("--Vp", "--Vprime", dest="Vprime", type=float, default=-0.1)
    p.add_argument("--Vx", "--Vcross", dest="Vcross", type=float, default=-0.05)
    p.add_argument("--nev", type=int, default=4)
    p.add_argument("--tol", type=float, default=1e-10)
    return p.parse_args()


def main():
    a = parse_args()
    model = RubyModel(t1=a.t, t2=a.t, V=a.V, Vprime=a.Vprime, Vcross=a.Vcross)
    result = run_effective_ed(
        model,
        EffectiveEDConfig(Lx=a.Lx, Ly=a.Ly, nev=a.nev, tol=a.tol),
    )
    s = summarize_effective_ed(result)
    jn, jm, jz = model.effective_couplings()
    print(f"Jn={jn:+.8e}, Jm={jm:+.8e}, Jz={jz:+.8e}")
    print(
        f"E0={s['ground_energy']:+.10e}, gap={s['gap']:.6e}, "
        f"degeneracy={s['ground_degeneracy']}"
    )
    print(
        f"leading q={s['leading_q']}, mode={s['leading_component']}_{s['leading_parity']}, "
        f"lambda={s['leading_lambda']:.8f}"
    )
    print(
        f"z_same(Gamma)={s['z_same_at_gamma']:.8f}, xy_max={s['xy_max']:.8f}"
    )


if __name__ == "__main__":
    main()
