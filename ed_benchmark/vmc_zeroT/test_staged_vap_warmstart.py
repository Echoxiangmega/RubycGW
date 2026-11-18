#!/usr/bin/env python3
"""Regression test for staged VAP warm-start parameter transfer."""
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
    jastrow_channels,
    bond_backflow_channels,
    one_body_seed,
    build_projected_netket_model,
)
from run_benchmark import make_netket_hamiltonian
from symmetry_projection import build_translation_group, characterize_unique_state
from run_benchmark import solve_ed


def main():
    import jax
    jax.config.update("jax_enable_x64",True)
    import netket as nk

    g=geometry.RubyPBC("ring3",t=1.0,t1=0.5,t2=-0.5)
    H,_,gs,_=solve_ed(g,3,2.5,k=12,degeneracy_tol=1e-8)
    tg=build_translation_group(g,H.states)

    # Pick one ED ground vector merely to define a translation target.
    psi=np.asarray(gs[:,0],complex)
    chars=characterize_unique_state(tg,psi,tol=1e-8)

    hi,_,_,graph=make_netket_hamiltonian(g,3,2.5)
    star=ruby_star_formfactor(g)
    M0=one_body_seed(g,3,"normal",0.08,star)
    _,_,shells,_=jastrow_channels(g,mode="bond_aware",n_shells=4)
    _,bf,_=bond_backflow_channels(g)

    v0=np.linspace(-0.07,0.05,shells.shape[0])
    eta0=np.asarray([0.01+0.02j,-0.03+0.01j,0.02-0.04j])
    maps=np.asarray([op.site_map for op in tg],np.int32)

    model=build_projected_netket_model(
        M0,shells,True,maps,np.asarray(chars,complex),
        backflow_matrices=bf,
        initial_v=v0,
        initial_eta=eta0,
    )
    sampler=nk.sampler.MetropolisFermionHop(hi,graph=graph,n_chains=2)
    vs=nk.vqs.MCState(
        sampler,model,n_samples=64,n_discard_per_chain=2,
        chunk_size=64,seed=41,sampler_seed=43
    )

    got_v=np.asarray(vs.parameters["v"],float)
    got_eta=np.asarray(vs.parameters["eta"],complex)
    assert np.max(np.abs(got_v-v0))<1e-14,(got_v,v0)
    assert np.max(np.abs(got_eta-eta0))<1e-14,(got_eta,eta0)

    print("staged VAP warm-start test: PASS")
    print("  v =",got_v)
    print("  eta =",got_eta)


if __name__=="__main__":
    main()
