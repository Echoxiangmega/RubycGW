#!/usr/bin/env python3
"""Regression tests for bond-aware Ruby density-Jastrow channels."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(NATIVE))

import geometry
from vmc_ansatz import bond_aware_jastrow_channels


def main():
    g=geometry.RubyPBC("ring3",t=1.0,t1=0.5,t2=-0.5)
    labels,distances,mats,counts=bond_aware_jastrow_channels(
        g,n_nonbond_shells=4
    )

    assert labels[:3]==["bond:t","bond:t1","bond:t2"],labels
    assert list(map(int,counts[:3]))==[18,9,9],counts
    assert mats.shape[0]==7,(labels,mats.shape)
    assert mats.shape[1:]==(18,18)

    # Every channel is symmetric with zero diagonal.
    for label,m in zip(labels,mats):
        assert np.max(np.abs(m-m.T))<1e-12,label
        assert np.max(np.abs(np.diag(m)))<1e-12,label

    # No unordered pair belongs to two channels.
    total=np.sum(mats,axis=0)
    iu=np.triu_indices(g.Ns,k=1)
    assert np.max(total[iu])<=1.0+1e-12

    # All graph edges are represented exactly once among the first 3 channels.
    graph=np.zeros((g.Ns,g.Ns),float)
    for e in g.edges:
        graph[e.i,e.j]=graph[e.j,e.i]=1.0
    assert np.max(np.abs(np.sum(mats[:3],axis=0)-graph))<1e-12

    print("bond-aware Jastrow test: PASS")
    for i,(lab,d,n) in enumerate(zip(labels,distances,counts)):
        print(f"  v[{i}] {lab:16s} pairs={int(n):3d} representative_r={d:.12g}")


if __name__=="__main__":
    main()
