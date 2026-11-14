"""Check that the triangle3 partition has no artificial CO symmetry constraint.

This is an architectural audit, NOT a claim that triangle3 and cell6
will have the same ordered phase at finite V. Both should match the
bare Hartree/Fock interaction, but intertriangle correlations beyond GW
differ by design.
"""
import numpy as np
from numpy.testing import assert_allclose

from rubycgw.cluster_ed_gw import cluster_interaction_matrix, ruby_cluster_interactions
from rubycgw.fierz_cluster_embedding import triangle_cluster_interactions
from rubycgw.grids import MatsubaraGrid
from rubycgw.hex18_order import order_observables, seed_matrix
from rubycgw.model import RubyParameters
from rubycgw.supercell_gw import hartree_self_energy_matrix
from rubycgw.supercell_gw_split import (
    compute_static_fock_matrix, one_body_density_matrix_tail,
)
from rubycgw.three_torus_benchmark import build_torus_geometry, noninteracting_cluster_seed
from research.scan_hex18_cluster_order_vn import _fresh_bootstrap, _inject_seed, parse_args


def test_six_triangle_blocks_exactly_cover_all_interacting_pairs_and_static_HF():
    """At first order in V, triangle3 and cell6 are exactly identical."""
    p = RubyParameters(ti=1.,t1=-.5,t2=.5,V=.5)
    geom=build_torus_geometry("hex18",p)
    H=geom.h0[0,0]
    ints=ruby_cluster_interactions(p)
    assert geom.nsites==18
    assert len(ints)==6
    assert np.max(abs(H[:3,3:6]))>0.
    assert np.max(abs(H[:3,6:]))>0.
    rng=np.random.default_rng(42)
    rho=np.diag(np.linspace(.09,.39,18).astype(complex))
    raw=rng.standard_normal((18,18))+1j*rng.standard_normal((18,18))
    rho+=.008*(raw+raw.conj().T)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=4,nOmega=2,T=.125)
    sh_full=hartree_self_energy_matrix(np.diag(rho).real,geom.vq[0,0])
    sf_full=compute_static_fock_matrix(
        rho[None,None],geom.vq,grid,backend="direct"
    )[0,0]
    sh_sum=np.zeros((18,18),complex)
    sf_sum=np.zeros((18,18),complex)
    v_sum=np.zeros((18,18),complex)
    for c in range(6):
        sl=slice(3*c,3*(c+1))
        terms=triangle_cluster_interactions(ints,c%2)
        Vc=cluster_interaction_matrix(terms,3)
        v_sum[sl,sl]=Vc
        sh_sum[sl,sl]=hartree_self_energy_matrix(
            np.diag(rho[sl,sl]).real,Vc
        )
        sf_sum[sl,sl]=compute_static_fock_matrix(
            rho[sl,sl][None,None],
            Vc[None,None],
            grid,backend="direct"
        )[0,0]
    assert_allclose(v_sum,geom.vq[0,0],rtol=0,atol=1e-13)
    assert_allclose(sh_sum,sh_full,rtol=0,atol=1e-13)
    assert_allclose(sf_sum,sf_full,rtol=0,atol=1e-13)
    assert np.max(abs(sh_full[0:3,0:3]-np.eye(3)*sh_full[0,0]))>1e-4


def test_weiss_field_keeps_all_intertriangle_hoppings_exactly_at_V0():
    """Projected Weiss contains the full 15-site environment by Schur complement."""
    p=RubyParameters(ti=1.,t1=-.5,t2=.5,V=0.)
    geom=build_torus_geometry("hex18",p)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=5,nOmega=2,T=.125)
    free=noninteracting_cluster_seed(geom,grid,target_particles=3.9)
    H=geom.h0[0,0]
    for c in range(6):
        loc=np.arange(3*c,3*c+3)
        env=np.asarray([i for i in range(18) if i not in loc],int)
        gc=free.G[:,0,0][:,loc][:,:,loc]
        delta=(1j*grid.omega[:,None,None]+free.mu)*np.eye(3)[None] - H[np.ix_(loc,loc)][None] - np.linalg.inv(gc)
        link=H[np.ix_(loc,env)]
        Hen=H[np.ix_(env,env)]
        expected=np.asarray([
            link@np.linalg.inv((1j*w+free.mu)*np.eye(15)-Hen)@link.conj().T
            for w in grid.omega
        ])
        assert_allclose(delta,expected,rtol=2e-10,atol=2e-10)
    # No real-space hopping is removed when building the 3-site Weiss field.


def test_CO_seed_enters_triangle3_with_independent_A_B_Weiss_fields():
    """The seed survives Dyson and can break both translation and intratriangle C3."""
    params=RubyParameters(ti=1.,t1=-.5,t2=.5,V=.5)
    geom=build_torus_geometry("hex18",params)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=7,nOmega=3,T=.125)
    args=parse_args([
        "--cluster-partition","triangle3",
        "--V-scan",".5","--fillings","1.3",
        "--nw","7","--nomega","3","--out","unused",
    ])
    init,kind=_fresh_bootstrap(geom,params,grid,3.9,"density",args)
    assert kind=="free"
    assert init.Sigma_imp.shape==(6,grid.nf,3,3)
    assert init.bath_couplings.shape==(6,3,3)
    source=seed_matrix(geom,params,"co_qK_A0")
    assert np.any(np.abs(source[:3,:3])>0)
    assert np.any(np.abs(source[6:9,6:9])>0)
    pert=_inject_seed(init,source,.3,geom.h0,grid,3.9)
    assert_allclose(
        pert.Sigma_emb[:,0,0]-init.Sigma_emb[:,0,0],
        np.broadcast_to(.3*source, (grid.nf,18,18)),
        rtol=0,atol=1e-13
    )
    rho=one_body_density_matrix_tail(pert.G,grid,geom.h0,pert.mu,pert.Sigma_H)[0,0]
    obs=order_observables(rho,geom,params,threshold=.002)
    assert obs["co_amplitude"]>.002
    assert obs["phase_family"]=="CO"
    assert abs(float(np.trace(rho).real)-3.9)<1e-8
    # It is the self-consistency that may restore symmetry, not the partition.
