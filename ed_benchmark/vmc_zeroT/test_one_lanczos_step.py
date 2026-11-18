#!/usr/bin/env python3
"""Regression tests for the exact one-Lanczos-step Hex18 postprocessor."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(NATIVE))

import geometry
from run_benchmark import solve_ed
from run_projection_comparison import one_lanczos_step
from symmetry_projection import (
    build_ordinary_space_group,
    characterize_unique_state,
    project_to_characters,
    projected_sector_residual,
)


def energy(psi,H):
    psi=np.asarray(psi,complex)
    psi/=np.linalg.norm(psi)
    return float(np.vdot(psi,H.operator@psi).real)


def main():
    g=geometry.RubyPBC("ring3",t=1.0,t1=0.5,t2=-0.5)
    H,_,gs,row=solve_ed(g,6,2.0,k=8,degeneracy_tol=1e-8)
    assert gs.shape[1]==1

    # Exact eigenstate is unchanged.
    psi0=np.asarray(gs[:,0],complex)
    psi1,info=one_lanczos_step(psi0,H)
    assert abs(energy(psi1,H)-energy(psi0,H))<1e-10
    assert 1-abs(np.vdot(psi0,psi1))<1e-9

    # A generic state can only improve or stay equal.
    rng=np.random.default_rng(123)
    x=rng.normal(size=len(H.states))+1j*rng.normal(size=len(H.states))
    x/=np.linalg.norm(x)
    e0=energy(x,H)
    y,info=one_lanczos_step(x,H)
    e1=energy(y,H)
    assert e1<=e0+1e-10,(e0,e1)

    # A projected state remains in the same exact symmetry sector after H action.
    group=build_ordinary_space_group(g,H.states)
    chars=characterize_unique_state(group,psi0,tol=1e-8)
    xp,w=project_to_characters(group,chars,x)
    assert xp is not None and w>0
    yp,_=one_lanczos_step(xp,H)
    res=projected_sector_residual(group,chars,yp)
    assert res<1e-9,res

    print("one-Lanczos-step tests: PASS")
    print("random-state energy:",e0,"->",e1)
    print("projected-sector residual after 1LS =",res)


if __name__=="__main__":
    main()
