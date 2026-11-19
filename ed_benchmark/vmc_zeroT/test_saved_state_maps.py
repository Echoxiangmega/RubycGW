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
    select_representation,
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
    _,bond,current=compute_edge_observables(g,states,rep)
    _,bond_tr,current_tr=compute_edge_observables(g,states,rep_tr)

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

    # Artificial exactly degenerate manifold: one basis state occupies two
    # A-triangle sites and the other two B-triangle sites. The charge modes
    # must select an actual normalized state inside this manifold.
    def fock_state(occupied):
        return np.uint64(sum(1 << (g.Ns-1-i) for i in occupied))

    sA=fock_state([0,1])
    sB=fock_state([3,4])
    iA=int(np.searchsorted(states,sA))
    iB=int(np.searchsorted(states,sB))
    assert states[iA]==sA and states[iB]==sB

    deg=np.zeros((len(states),2),dtype=np.complex128)
    deg[iA,0]=1.0
    deg[iB,1]=1.0
    payload={"kind":"manifold","psi":deg,"states":states,"metadata":{}}

    sel_ab=select_representation(payload,"charge_ab",None,g)
    d_ab=compute_site_density(g,states,sel_ab)
    contrast_ab=float(
        np.sum(d_ab[[0,1,2]])-np.sum(d_ab[[3,4,5]])
    )
    assert abs(contrast_ab)>1.5,contrast_ab
    assert sel_ab["selection"]["selection_available"]
    assert abs(np.linalg.norm(sel_ab["psi"])-1.0)<1e-12

    sel_lead=select_representation(payload,"charge_leading",None,g)
    d_lead=compute_site_density(g,states,sel_lead)
    ensemble=0.5*(
        compute_site_density(
            g,states,{"mode":"vector","psi":deg[:,0],"states":states}
        )
        + compute_site_density(
            g,states,{"mode":"vector","psi":deg[:,1],"states":states}
        )
    )
    assert np.max(np.abs(d_lead-ensemble))>0.4
    assert sel_lead["selection"]["selection_available"]
    assert "leading_q" in sel_lead["selection"]

    # A unique state cannot be turned into a broken-order branch by request.
    unique_payload={
        "kind":"manifold",
        "psi":deg[:,:1],
        "states":states,
        "metadata":{},
    }
    unique_lc=select_representation(
        unique_payload,"lc_opposite",None,g
    )
    assert not unique_lc["selection"]["selection_available"]
    assert unique_lc["selection"]["ground_dim"]==1

    print("saved-state map observable tests: PASS")
    print("sum density =",np.sum(density))
    print("max |J(psi)+J(T psi)| =",np.max(np.abs(current+current_tr)))
    print("max |J(TR mixture)| =",np.max(np.abs(current_mix)))
    print("charge_ab contrast =",contrast_ab)
    print("charge_leading q =",sel_lead["selection"]["leading_q"])


if __name__=="__main__":
    main()
