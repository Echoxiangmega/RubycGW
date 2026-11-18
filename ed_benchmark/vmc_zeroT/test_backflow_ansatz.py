#!/usr/bin/env python3
"""Regression tests for bond-aware complex backflow on Hex18."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(NATIVE))

import geometry

from benchmark_observables import ruby_star_formfactor
from vmc_ansatz import (
    bond_backflow_channels,
    jastrow_channels,
    one_body_seed,
    build_netket_model,
)
from run_benchmark import make_netket_hamiltonian


def main():
    import jax
    jax.config.update("jax_enable_x64",True)
    import netket as nk

    g=geometry.RubyPBC("ring3",t=1.0,t1=0.5,t2=-0.5)
    labels,bf,counts=bond_backflow_channels(g)
    assert labels==["eta_t","eta_t1","eta_t2"],labels
    assert list(map(int,counts))==[18,9,9],counts
    assert bf.shape==(3,18,18)
    for k in range(3):
        assert np.max(np.abs(bf[k]-bf[k].T))<1e-14
        assert np.max(np.abs(np.diag(bf[k])))<1e-14

    # No graph pair occurs in two backflow channels.
    total=np.sum(bf,axis=0)
    iu=np.triu_indices(18,k=1)
    assert np.max(total[iu])<=1.0+1e-14

    # At eta=0, the backflow model must be exactly the old SJ model.
    hi,_,_,graph=make_netket_hamiltonian(g,3,2.5)
    star=ruby_star_formfactor(g)
    M0=one_body_seed(g,3,"normal",0.08,star)
    _,_,shells,_=jastrow_channels(g,mode="bond_aware",n_shells=4)

    model_sj=build_netket_model(M0,shells,True,backflow_matrices=None)
    model_bf=build_netket_model(M0,shells,True,backflow_matrices=bf)

    sampler=nk.sampler.MetropolisFermionHop(hi,graph=graph,n_chains=2)
    vs_sj=nk.vqs.MCState(
        sampler,model_sj,n_samples=64,n_discard_per_chain=2,
        chunk_size=64,seed=1234,sampler_seed=4321
    )
    vs_bf=nk.vqs.MCState(
        sampler,model_bf,n_samples=64,n_discard_per_chain=2,
        chunk_size=64,seed=1234,sampler_seed=4321
    )

    eta=np.asarray(vs_bf.parameters["eta"],complex)
    assert np.max(np.abs(eta))<1e-15,eta

    psi_sj=np.asarray(vs_sj.to_array(normalize=True))
    psi_bf=np.asarray(vs_bf.to_array(normalize=True))
    overlap=abs(np.vdot(psi_sj,psi_bf))
    residual=np.linalg.norm(
        psi_bf-psi_sj*np.vdot(psi_sj,psi_bf)
    )
    assert 1.0-overlap<2e-10,overlap
    assert residual<2e-10,residual

    print("bond-aware backflow tests: PASS")
    for lab,n in zip(labels,counts):
        print(f"  {lab:8s} pairs={int(n):3d}")
    print("  eta(initial) =",eta)
    print("  |<SJ|BF(eta=0)>| =",overlap)
    print("  zero-eta residual =",residual)


if __name__=="__main__":
    main()
