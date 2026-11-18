#!/usr/bin/env python3
"""Tests for C3-constrained, translation-breaking Hex18 orbital ansatz."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(NATIVE))

import geometry
from benchmark_observables import ruby_star_formfactor
from vmc_ansatz import one_body_seed, minimum_image_shells
from c3_constrained_ansatz import (
    c3_seed_subspace,
    one_particle_c3_matrix,
    build_c3_constrained_netket_model,
)


def main():
    g=geometry.RubyPBC("ring3",t=1.0,t1=0.5,t2=-0.5)
    star=ruby_star_formfactor(g)
    R=one_particle_c3_matrix(g)

    for target_m in range(3):
        Qseed=one_body_seed(g,3,"normal",0.08,star)
        data=c3_seed_subspace(g,Qseed,3,target_manybody_m=target_m)
        Q=data["Q0"]
        P=Q@Q.conj().T
        assert np.max(np.abs(Q.conj().T@Q-np.eye(3)))<1e-10
        assert np.max(np.abs(R@P@R.conj().T-P))<1e-10
        assert data["manybody_m"]==target_m
        assert sum(data["counts"])==3
        print(
            f"C3:m{target_m} counts={data['counts']} "
            f"physical_real_dof={data['physical_real_dof']}"
        )

    # Model construction catches Flax/JAX shape/scope errors.
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64",True)
        _,shells,_=minimum_image_shells(g,n_shells=4)
        data=c3_seed_subspace(
            g,one_body_seed(g,3,"normal",0.08,star),3,target_manybody_m=0
        )
        model=build_c3_constrained_netket_model(data,shells,True)
        x=np.zeros((1,g.Ns),dtype=np.int8);x[0,:3]=1
        variables=model.init(jax.random.PRNGKey(0),jnp.asarray(x))
        y=np.asarray(model.apply(variables,jnp.asarray(x)))
        assert np.all(np.isfinite(y)),y
    except ImportError:
        print("JAX/NetKet unavailable: skipped model-instantiation test")

    print("C3-constrained ansatz tests: PASS")


if __name__=="__main__":
    main()
