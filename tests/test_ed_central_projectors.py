"""Regression for non-Abelian ED-driven VAP projectors (no NetKet needed)."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VMC=ROOT/"ed_benchmark"/"vmc_zeroT"
NATIVE=ROOT/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(VMC))
sys.path.insert(0,str(NATIVE))

from ed_central_projectors import ed_central_sectors, pair_time_reversal_sectors, central_projector_diagnostics


def test_c3v_e_projector():
    from itertools import permutations
    from types import SimpleNamespace
    # Three-site permutation representation = A1 + E.
    group=[]
    for p in permutations(range(3)):
        p=np.array(p)
        group.append(SimpleNamespace(site_map=p,name=str(p),
                     apply=lambda x,p=p: x[np.argsort(p)]))
    sectors=ed_central_sectors(group,np.eye(3))
    assert sorted(s["irrep_dimension"] for s in sectors)==[1,2]
    E=next(s for s in sectors if s["irrep_dimension"]==2)
    a=np.array([1.,-1.,0.])
    weight,res=central_projector_diagnostics(group,E["characters"],a)
    assert abs(weight-1)<1e-10 and res<1e-10
    weight,res=central_projector_diagnostics(group,E["characters"],np.ones(3))
    assert weight<1e-10 and abs(res-1)<1e-10
    assert len(pair_time_reversal_sectors(sectors))==2


def test_c3_conjugate_pair():
    from types import SimpleNamespace
    group=[]
    for i in range(3):
        p=np.roll(np.arange(3),i)
        group.append(SimpleNamespace(site_map=p,name=str(p),
                     apply=lambda x,p=p: x[np.argsort(p)]))
    ground=np.linalg.qr(np.array([[1.,1.],[-1.,1.],[0.,-2.]]))[0]
    sectors=ed_central_sectors(group,ground)
    assert len(sectors)==2
    pair=pair_time_reversal_sectors(sectors)
    assert len(pair)==1 and pair[0]["tr_paired"]
    for x in ground.T:
        w,r=central_projector_diagnostics(group,pair[0]["characters"],x)
        assert abs(w-1)<1e-10 and r<1e-10


def test_ruby_nf4_reported_ground_manifold():
    from geometry import make_ring3_supercell
    from run_benchmark import solve_ed
    from run_supercell_benchmark import c3v_targets_from_ed
    from symmetry_projection import build_internal_c3v_space_group, build_internal_translation_group
    g=make_ring3_supercell(1,1,t=1.,t1=.5,t2=-.5)
    H,_,gs,row=solve_ed(g,4,8.6,k=12,degeneracy_tol=1e-8,backend="full")
    assert not row.get("ground_manifold_truncated")
    assert gs.shape[1]>1
    _,point=c3v_targets_from_ed(g,H,gs)
    assert sum(t["ground_dimension"] for t in point)==gs.shape[1]
    for group in (build_internal_translation_group(g,H.states),
                  build_internal_c3v_space_group(g,H.states)):
        sectors=ed_central_sectors(group,gs)
        assert sum(s["ground_dimension"] for s in sectors)==gs.shape[1]
        paired=pair_time_reversal_sectors(sectors)
        for x in gs.T:
            weights=[central_projector_diagnostics(group,t["characters"],x)[0]
                     for t in paired]
            assert abs(sum(weights)-1)<1e-7
