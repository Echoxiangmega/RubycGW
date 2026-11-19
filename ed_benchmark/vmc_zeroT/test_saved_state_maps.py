#!/usr/bin/env python3
"""Regression tests for saved-state site/bond/current map observables."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(NATIVE))

import geometry
import core

from plot_saved_state_maps import (
    compute_site_density,
    compute_edge_observables,
)


def main():
    g=geometry.RubyPBC("ring3",t=1.0,t1=0.5,t2=-0.5)
    nf=2
    states=core.make_states(g.Ns,nf)
    rng=np.random.default_rng(20261119)
    psi=rng.normal(size=len(states))+1j*rng.normal(size=len(states))
    psi=psi/np.linalg.norm(psi)

    rep={"mode":"vector","psi":psi,"states":states}
    rep_tr={"mode":"vector","psi":psi.conjugate(),"states":states}
    density=compute_site_density(g,states,rep)
    density_tr=compute_site_density(g,states,rep_tr)
    rho,bond,current=compute_edge_observables(g,states,rep)
    rho_tr,bond_tr,current_tr=compute_edge_observables(g,states,rep_tr)

    assert abs(np.sum(density)-nf)<1e-12
    assert np.max(np.abs(density-density_tr))<1e-12
    assert np.max(np.abs(bond-bond_tr))<1e-12
    assert np.max(np.abs(current+current_tr))<1e-12

    # Equal-weight mixture of a state and its time reverse must have zero
    # one-point current while retaining density and real bond order.
    manifold=np.column_stack((psi,psi.conjugate()))
    rep_mix={"mode":"average","psi":manifold,"states":states}
    density_mix=compute_site_density(g,states,rep_mix)
    _,bond_mix,current_mix=compute_edge_observables(g,states,rep_mix)
    assert np.max(np.abs(density_mix-density))<1e-12
    assert np.max(np.abs(bond_mix-bond))<1e-12
    assert np.max(np.abs(current_mix))<1e-12

    print("saved-state map observable tests: PASS")
    print("sum density =",np.sum(density))
    print("max |J(psi)+J(T psi)| =",np.max(np.abs(current+current_tr)))
    print("max |J(TR mixture)| =",np.max(np.abs(current_mix)))


if __name__=="__main__":
    main()
