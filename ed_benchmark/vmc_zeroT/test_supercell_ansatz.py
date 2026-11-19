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
)
from symmetry_projection import c3v_site_maps


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

    print("supercell ansatz tests: PASS")
    print("1x1: exact Hex18 one-body variational space")
    print("2x2 ring3 superlattice: Ns=",big.Ns,
          "occupations_per_k=",data["occupations_per_k"].tolist())
    print("C3v residual point group: 6 distinct site maps on 1x1 and 2x2")


if __name__=="__main__":
    main()
