#!/usr/bin/env python3
"""Regression tests for the ring3-supercell translation-built-in ansatz."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(NATIVE))

import geometry

from benchmark_observables import ruby_star_formfactor
from vmc_ansatz import one_body_seed
from supercell_ansatz import (
    ring3_periodic_star_formfactor,
    seed_bloch_blocks,
    assemble_bloch_orbitals_numpy,
    _sparse_jastrow_channels,
    _sparse_backflow_channels,
)
from symmetry_projection import (
    c3v_site_maps,
    c3v_irrep_characters,
    internal_translation_site_maps,
    internal_c3v_space_group_site_maps,
    internal_gamma_c3v_characters,
    ordinary_space_group_site_maps,
)


def projector(Q):
    Q=np.asarray(Q,complex)
    return Q@Q.conj().T


def main():
    # 1x1 must be geometrically identical to the historical Hex18 ring3 torus.
    old=geometry.RubyPBC("ring3",t=1.0,t1=0.5,t2=-0.5)
    g=geometry.make_ring3_supercell(1,1,t=1.0,t1=0.5,t2=-0.5)
    assert g.Ns==18
    assert np.array_equal(g.M,old.M)
    assert list(g.reps)==list(old.reps)
    assert [(e.i,e.j,e.kind,e.delta,e.winding) for e in g.edges] == [
        (e.i,e.j,e.kind,e.delta,e.winding) for e in old.edges
    ]

    ops1=c3v_site_maps(g)
    assert len(ops1)==6
    assert len({tuple(map(int,p)) for _,p in ops1})==6
    names1=[name for name,_ in ops1]
    assert names1[:3]==["E","C3","C3^2"]
    assert all(name.startswith("M") for name in names1[3:])
    assert np.allclose(c3v_irrep_characters("A1"),np.ones(6))
    assert np.allclose(
        c3v_irrep_characters("A2"),
        np.asarray([1,1,1,-1,-1,-1],complex),
    )
    tint1=internal_translation_site_maps(g)
    full1=internal_c3v_space_group_site_maps(g)
    assert len(tint1)==3
    assert len(full1)==18
    assert len({tuple(map(int,p)) for _,p in full1})==18
    old_full=ordinary_space_group_site_maps(g)
    assert len(old_full)==18
    assert {
        tuple(map(int,p)) for _,p in full1
    } == {
        tuple(map(int,p)) for _,p in old_full
    }
    assert np.allclose(
        internal_gamma_c3v_characters("A2"),
        np.tile(np.asarray([1,1,1,-1,-1,-1],complex),3),
    )

    f_old=ruby_star_formfactor(old)
    f_new=ring3_periodic_star_formfactor(g,1,1)
    assert np.max(np.abs(f_old-f_new))<1e-12

    for seed in ("normal","sd","isd","lc_same","lc_opposite"):
        ref=one_body_seed(old,6,seed,0.08,f_old)
        data=seed_bloch_blocks(g,6,1,1,seed,0.08)
        assert data["occupations_per_k"].tolist()==[6]
        Q=assemble_bloch_orbitals_numpy(
            data["blocks"],data["sc_coords"],data["internal_sites"],data["kpoints"]
        )
        assert Q.shape==(18,6)
        # Eigenvectors may rotate within degenerate occupied subspaces; compare
        # the physical occupied projector, not individual columns.
        err=np.linalg.norm(projector(Q)-projector(ref))
        assert err<1e-9,(seed,err)

    # 2x2 ring3 superlattice = 72 sites.  The periodic SD/ISD pattern must
    # repeat exactly four times and the occupied projector must commute with
    # the two ring3-supercell translations.
    big=geometry.make_ring3_supercell(2,2,t=1.0,t1=0.5,t2=-0.5)
    assert big.Ns==72 and big.num_cells==12
    ops2=c3v_site_maps(big)
    assert len(ops2)==6
    assert len({tuple(map(int,p)) for _,p in ops2})==6
    tint2=internal_translation_site_maps(big)
    full2=internal_c3v_space_group_site_maps(big)
    assert len(tint2)==3
    assert len(full2)==18
    assert len({tuple(map(int,p)) for _,p in full2})==18
    data=seed_bloch_blocks(big,24,2,2,"sd",0.08)
    assert sum(data["occupations_per_k"])==24
    Q=assemble_bloch_orbitals_numpy(
        data["blocks"],data["sc_coords"],data["internal_sites"],data["kpoints"]
    )
    P=projector(Q)
    for dr in (tuple(geometry.RING3_MATRIX[:,0]),tuple(geometry.RING3_MATRIX[:,1])):
        p=big.translate(dr)
        assert np.linalg.norm(P[np.ix_(p,p)]-P)<1e-9

    f=data["star_formfactor"]
    # Each of the four 18-site supercells carries the same 36-edge reference
    # pattern, hence every reference-edge value appears four times.
    refvals=np.sort(ring3_periodic_star_formfactor(g,1,1))
    assert len(f)==4*len(refvals)
    assert np.allclose(np.sort(f),np.sort(np.tile(refvals,4)))

    # Sparse Jastrow/backflow compression must reproduce the dense formulas.
    rng=np.random.default_rng(260119)
    ns=18
    nch=4
    mats=np.zeros((nch,ns,ns),float)
    for s in range(nch):
        for i in range(ns-1):
            if (i+s)%3==0:
                mats[s,i,i+1]=mats[s,i+1,i]=1.0+0.1*s
    dn=rng.normal(size=ns)
    jch,ji,jj,jv=_sparse_jastrow_channels(mats)
    dense_j=0.5*np.einsum("i,sij,j->s",dn,mats,dn)
    sparse_j=np.zeros(nch,float)
    np.add.at(sparse_j,jch,jv*dn[ji]*dn[jj])
    assert np.max(np.abs(dense_j-sparse_j))<1e-12

    bf=np.zeros((3,ns,ns),float)
    for k in range(3):
        for i in range(ns):
            j=(i+k+1)%ns
            bf[k,i,j]=1.0
    eta=rng.normal(size=3)+1j*rng.normal(size=3)
    Q=rng.normal(size=(ns,5))+1j*rng.normal(size=(ns,5))
    bch,bdst,bsrc,bv=_sparse_backflow_channels(bf)
    dense_b=np.einsum("k,kij,j,ja->ia",eta,bf,dn,Q)
    sparse_b=np.zeros_like(Q,dtype=complex)
    np.add.at(
        sparse_b,bdst,
        eta[bch,None]*bv[:,None]*dn[bsrc,None]*Q[bsrc,:],
    )
    assert np.max(np.abs(dense_b-sparse_b))<1e-12

    print("supercell ansatz tests: PASS")
    print("1x1: exact Hex18 one-body variational space")
    print("2x2 ring3 superlattice: Ns=",big.Ns,
          "occupations_per_k=",data["occupations_per_k"].tolist())
    print("C3v residual point group: 6 distinct site maps on 1x1 and 2x2")
    print("Residual Tint: 3 maps; Tint x C3v: 18 maps independent of supercell size")


if __name__=="__main__":
    main()
