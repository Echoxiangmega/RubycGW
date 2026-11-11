"""Operator-level tests for exact A1+E pseudospin Fierz and momentum ED."""
import numpy as np
import pytest
from scipy import sparse

from rubycgw.model import RubyParameters
from rubycgw.fierz_channel_gw import (
    build_channel_definition, channel_static_self_energy,
)
from rubycgw.three_torus_interaction_channels import (
    build_torus_interaction_channel,
)
from rubycgw.three_torus_benchmark import build_torus_geometry
from rubycgw.momentum_ed import MomentumTorusED
from rubycgw.ed_cluster import RubyEDClusterSolver


def _onebody_fock(K):
    K=np.asarray(K,dtype=complex)
    norb=K.shape[0]
    d=1<<norb
    O=np.zeros((d,d),dtype=complex)
    for s in range(d):
        for i,j in np.argwhere(abs(K)>1e-13):
            i,j=int(i),int(j)
            if i==j:
                if s&(1<<i):
                    O[s,s]+=K[i,i]
                continue
            if not(s&(1<<j)) or s&(1<<i):
                continue
            a,b=sorted((i,j))
            between=((1<<b)-1)^((1<<(a+1))-1)
            sign=-1 if ((s&between).bit_count()%2) else 1
            new=s^(1<<i)^(1<<j)
            O[new,s]+=sign*K[i,j]
    return O


@pytest.mark.parametrize("V",[-0.8,0.0,0.9,2.0])
def test_full_triangle_v_operator_exact_a1_plus_e_fierz(V):
    pairs=[(i,j) for q in (0,3)
           for i,j in ((q,q+1),(q,q+2),(q+1,q+2))]
    definition=build_channel_definition(pairs,6,V,"pseudospin")
    assert len(definition.labels)==10  # five per triangle
    lhs=np.zeros((64,64),dtype=complex)
    for state in range(64):
        lhs[state,state]=V*sum(
            int(bool(state&(1<<i)) and bool(state&(1<<j)))
            for i,j in pairs
        )
    op=[_onebody_fock(K) for K in definition.vertices]
    rhs=np.zeros_like(lhs)
    for tri in range(2):
        base=tri*5
        spin=op[base:base+3]
        nA=op[base+3]
        nE=op[base+4]
        rhs-=V/6*sum(S@S for S in spin)
        rhs+=V*(nA@nE)+V/2*nE
        # n_A1 and n_E are orthogonal projectors; they commute.
        np.testing.assert_allclose(nA@nE,nE@nA,atol=1e-12)
    np.testing.assert_allclose(rhs,lhs,rtol=0,atol=5e-12)
    # Exact HF map agrees with the original density representation.
    rng=np.random.default_rng(4)
    A=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6))
    rho=.35*np.eye(6)+.03*(A+A.conj().T)
    density=build_channel_definition(pairs,6,V,"density")
    np.testing.assert_allclose(
        channel_static_self_energy(rho,definition)[0],
        channel_static_self_energy(rho,density)[0],atol=1e-12
    )


@pytest.mark.parametrize("geometry,N",[
    ("3x1",2),("hex18",2),("2x2",2)
])
def test_projected_momentum_hamiltonian_spectrum_and_orthonormality(geometry,N):
    params=RubyParameters(ti=1,t1=-.5,t2=.5,V=.65)
    geom=build_torus_geometry(geometry,params)
    accelerated=MomentumTorusED(geom,params,n_particles=N)
    D=accelerated.full_dimension
    P=sparse.hstack([b.projector for b in accelerated.blocks],format="csr")
    assert P.shape==(D,D)
    np.testing.assert_allclose((P.getH()@P).toarray(),np.eye(D),atol=1e-12)
    spec=accelerated.ground_spectrum(.65,n_eigs=4)
    full=accelerated.solver
    if geometry=="hex18":
        fullH=full.hamiltonian(.65)
    else:
        fullH=sparse.csc_matrix(np.column_stack([
            full.apply_hamiltonian(.65,np.eye(D)[:,j]) for j in range(D)
        ]))
    ref=np.linalg.eigvalsh(fullH.toarray())
    np.testing.assert_allclose(spec.energies[0],ref[0],atol=2e-10)
    reduced=np.concatenate([
        np.linalg.eigvalsh(accelerated.hamiltonian(.65,k).toarray())
        for k in range(len(accelerated.blocks))
    ])
    np.testing.assert_allclose(np.sort(reduced),ref,rtol=0,atol=3e-10)


def test_finite_q_operator_transfers_manybody_momentum():
    params=RubyParameters(ti=1,t1=-.5,t2=.5,V=.3)
    geom=build_torus_geometry("3x1",params)
    ed=MomentumTorusED(geom,params,n_particles=1)
    Q=np.asarray((1/3,0.))
    M=ed.solver.operator_matrix("z_same",Q)
    O=np.column_stack([
        ed.solver.apply_onebody_matrix(M,np.eye(ed.full_dimension)[:,j])
        for j in range(ed.full_dimension)
    ])
    for ik,sector in enumerate(ed.blocks):
        kto=ed.k_index(np.asarray(ed.k_values[ik])-Q)
        P=sector.projector.toarray()
        out=O@P
        target=ed.blocks[kto].projector.toarray()
        projected=target@(target.conj().T@out)
        np.testing.assert_allclose(out,projected,atol=1e-11)


def test_exact_momentum_block_thermal_trace_matches_full_ED():
    params=RubyParameters(ti=1,t1=-.5,t2=.5,V=.4)
    geom=build_torus_geometry("3x1",params)
    ed=MomentumTorusED(geom,params,n_particles=1)
    T=.55
    Q=np.asarray((1/3,0.))
    ans=ed.thermal_response(
        .4,T,Q,("z_same",),n_trace_vectors=2,
        tau_points=9,trace_mode="exact"
    )
    src=ed.solver
    D=src.dimension
    fullH=np.column_stack([
        src.apply_hamiltonian(.4,np.eye(D)[:,j])
        for j in range(D)
    ])
    ev,U=np.linalg.eigh(fullH)
    O=np.column_stack([
        src.apply_onebody_matrix(
            src.operator_matrix("z_same",Q),np.eye(D)[:,j]
        )
        for j in range(D)
    ])
    o=U.conj().T@O@U
    beta=1/T
    weights=np.exp(-beta*(ev-ev.min()))
    Z=np.sum(weights)
    C=[]
    for tau in ans.tau_grid:
        ex=np.exp(-tau*(ev[:,None]-ev[None,:]))
        # O_nm = <n|O|m>; operator transfers m -> n.
        raw=np.sum((np.abs(o)**2)*ex*weights[None,:])/Z
        C.append(raw)
    np.testing.assert_allclose(
        ans.correlation_tau_matrix[:,0,0].real,C,
        rtol=0,atol=3e-9
    )
    from scipy.integrate import simpson
    np.testing.assert_allclose(
        ans.susceptibility_matrix[0,0].real,
        simpson(C,x=ans.tau_grid),atol=3e-9
    )
