#!/usr/bin/env python3
"""Regression tests for spatial + spinless-time-reversal benchmark restoration."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(NATIVE))

import geometry
import observables as native_observables

from benchmark_observables import benchmark_observables
from run_benchmark import solve_ed
from run_projection_comparison import (
    build_ed_sectors,
    build_tr_targets,
    spatial_pair_project,
    restore_time_reversal_lowest,
    target_support_residual,
    time_reversal_residual,
    one_lanczos_step,
)


def main():
    g=geometry.RubyPBC("ring3",t=1.0,t1=0.5,t2=-0.5)
    H,_,gs,_=solve_ed(g,6,2.0,k=8,degeneracy_tol=1e-8)
    assert gs.shape[1]==1

    group,_,sectors=build_ed_sectors(g,H,gs,1e-8)
    targets=build_tr_targets(sectors)
    assert len(targets)==1
    target=targets[0]

    rng=np.random.default_rng(314159)
    x=rng.normal(size=len(H.states))+1j*rng.normal(size=len(H.states))
    x/=np.linalg.norm(x)

    pair,w=spatial_pair_project(group,target,x)
    assert pair is not None and w>0
    y,info=restore_time_reversal_lowest(pair,H)

    tr0=time_reversal_residual(y)
    sp0=target_support_residual(group,target,y)
    obs=benchmark_observables(g,H.states,y,native_observables)
    print("post-TR diagnostics:")
    print("  spatial support residual =",sp0)
    print("  time-reversal residual   =",tr0)
    print("  J_same                   =",obs.lc_same_mean)
    print("  J_opposite               =",obs.lc_opposite_mean)

    assert tr0<1e-10,tr0
    assert sp0<1e-10,sp0
    assert abs(obs.lc_same_mean)<1e-10,obs.lc_same_mean
    assert abs(obs.lc_opposite_mean)<1e-10,obs.lc_opposite_mean

    z,_=one_lanczos_step(y,H)
    tr1=time_reversal_residual(z)
    sp1=target_support_residual(group,target,z)
    print("post-1LS diagnostics:")
    print("  spatial support residual =",sp1)
    print("  time-reversal residual   =",tr1)
    assert tr1<1e-10,tr1
    assert sp1<1e-10,sp1
    obs2=benchmark_observables(g,H.states,z,native_observables)
    assert abs(obs2.lc_same_mean)<1e-10,obs2.lc_same_mean
    assert abs(obs2.lc_opposite_mean)<1e-10,obs2.lc_opposite_mean

    # Also verify that a degenerate ED manifold can be organized into
    # self-conjugate sectors and/or Gamma/Gamma* TR pairs without omissions.
    H3,_,gs3,_=solve_ed(g,3,2.5,k=12,degeneracy_tol=1e-8)
    group3,_,sectors3=build_ed_sectors(g,H3,gs3,1e-8)
    targets3=build_tr_targets(sectors3)
    covered=sum(t["dimension"] for t in targets3)
    assert covered==gs3.shape[1],(covered,gs3.shape[1])

    print("spatial + time-reversal restoration tests: PASS")
    print("unique target:",target["k"],target["c3"])
    print("TR residual =",time_reversal_residual(y))
    print("J_same =",obs.lc_same_mean,"J_opposite =",obs.lc_opposite_mean)
    print("degenerate ground dim =",gs3.shape[1],"TR targets =",len(targets3))


if __name__=="__main__":
    main()
