#!/usr/bin/env python3
"""Regression tests for vectorized Fock maps/operator application and ED bridge."""
from pathlib import Path
from itertools import combinations
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(NATIVE))

import geometry
import core
import observables

from symmetry_projection import second_quantize_site_map
from run_benchmark import solve_ed


def reference_map(states,ns,p):
    target=np.empty(len(states),dtype=np.int64)
    phase=np.empty(len(states),dtype=np.int8)
    for a,s0 in enumerate(states):
        s=int(s0)
        occ=[i for i in range(ns) if s & (1 << (ns-1-i))]
        mapped=[int(p[i]) for i in occ]
        inv=sum(mapped[i]>mapped[j] for i in range(len(mapped)) for j in range(i+1,len(mapped)))
        st=sum(1 << (ns-1-j) for j in mapped)
        b=int(np.searchsorted(states,np.uint64(st)))
        target[a]=b
        phase[a]=-1 if inv%2 else 1
    return target,phase


def reference_apply_general(ket,ket_states,bra_states,ns,ops):
    out=np.zeros(len(bra_states),dtype=np.complex128)
    index={int(s):i for i,s in enumerate(bra_states)}
    for a,amp in enumerate(ket):
        if abs(amp)<=1e-16:
            continue
        new=int(ket_states[a]); sign=1
        for what,site in reversed(ops):
            bit=1 << (ns-1-site)
            filled=(new&bit)!=0
            if (what=='+' and filled) or (what=='-' and not filled):
                sign=0; break
            if (new >> (ns-site)).bit_count()%2:
                sign=-sign
            if what=='+':
                new|=bit
            else:
                new^=bit
        if sign and new in index:
            out[index[new]]+=sign*amp
    return out


def reference_apply(ket,states,ns,ops):
    return reference_apply_general(ket,states,states,ns,ops)


def main():
    g=geometry.RubyPBC("ring3",t=1.0,t1=0.5,t2=-0.5)
    states=core.make_states(g.Ns,3)
    p=g.point_maps()["C3"]
    op=second_quantize_site_map(states,g.Ns,p,name="C3-test")
    tref,pref=reference_map(states,g.Ns,p)
    assert np.array_equal(op.target_index,tref)
    assert np.array_equal(op.fermion_phase,pref)

    rng=np.random.default_rng(20261119)
    x=rng.normal(size=len(states))+1j*rng.normal(size=len(states))
    x/=np.linalg.norm(x)
    ops=[("+",0),("-",5)]
    got=observables.apply_ops(x,states,states,g.Ns,ops)
    ref=reference_apply(x,states,g.Ns,ops)
    assert np.linalg.norm(got-ref)<1e-12

    lower=np.array([
        sum(1 << (g.Ns-1-i) for i in occ)
        for occ in combinations(range(g.Ns),1)
    ],dtype=np.uint64)
    lower=np.sort(lower)[::-1]
    ops=[("-",1),("-",7)]
    got=observables.apply_ops(x,states,lower,g.Ns,ops)
    ref=reference_apply_general(x,states,lower,g.Ns,ops)
    assert np.linalg.norm(got-ref)<1e-12

    # Native symmetry ED must reproduce the full-basis result on a small case.
    _,_,_,full=solve_ed(g,3,1.3,k=6,degeneracy_tol=1e-8,backend="full")
    _,_,_,sym=solve_ed(g,3,1.3,k=6,degeneracy_tol=1e-8,backend="symmetry")
    assert abs(full["E"]-sym["E"])<1e-9,(full["E"],sym["E"])

    g2=geometry.RubyPBC("2x2",t=1.0,t1=0.5,t2=-0.5)
    assert g2.Ns==24
    assert len(core.make_states(24,8))==735471
    assert g2.point_maps()["C3"] is not None
    assert len(g2.point_maps()["mirrors"])>=3
    print("2x2 benchmark plumbing tests: PASS")


if __name__=="__main__":
    main()
