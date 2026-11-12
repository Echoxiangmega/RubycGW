"""Same finite-torus infrastructure for ED, GW, cGW, post-GW and 6-site ED+GW.

The three geometries are periodic Ruby tori: rectangular 3x1 and 2x2,
and the index-three skew/hexagonal 18-site torus of supercell.py.  This is
NOT the separate open-boundary six-triangle ring (hex18_ed.py).

The ED Green function is a controlled finite-temperature Lanczos approximation
in selected particle-number sectors; its error controls must be reported.
The cGW vertex improves response on the GW background, not the one-particle G.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from types import SimpleNamespace
import numpy as np

from .model import NSUB, RubyParameters, build_h0, build_interaction
from .grids import MatsubaraGrid
from .folded_torus import fold_static_k_field, torus_cells
from .supercell import (
    SUPERCELL_REPRESENTATIVES, build_supercell_h0, build_supercell_interaction,
)
from .ed18 import ED18Solver
from .ed_cluster import RubyEDClusterSolver
from .pseudospin import primitive_pseudospin_vertex
from .supercell_gw import (
    compute_polarization_matrix, compute_screened_interaction_matrix,
    hartree_self_energy_matrix, dyson_from_sigma_matrix,
)
from .supercell_gw_fast import _solve_mu_matrix_fast
from .supercell_gw_split import (
    one_body_density_matrix_tail, compute_sigma_gw_split_matrix,
    compute_sigma_gw_split_components,
)
from .cluster_ed_gw import (
    BathParameters, ruby_cluster_interactions, cluster_interaction_matrix,
    cluster_gw_self_energy, fit_finite_bath, split_static_hybridization,
    build_impurity_one_body, bath_hybridization,
)
from .impurity_ed import FiniteBathImpurityED
from .ed_cgw_benchmark import relative_frobenius_error
from .production_cgw import solve_vertex_q0_tail
from .production_cgw_sox import (
    static_gg_tail_completed, static_response_tail_completed,
)
from .response_tail import build_tail_reference
from .supercell_cgw import SupercellVertexOptions


@dataclass
class TorusGeometry:
    name: str
    cells: np.ndarray
    h0: np.ndarray
    vq: np.ndarray
    nsites: int
    ncell: int
    def validate_momentum(self, q, *, atol=1e-8):
        q = np.asarray(q, dtype=float).reshape(2)
        if self.name == "3x1":
            components = np.array([3*q[0], q[1]])
        elif self.name == "2x2":
            components = np.array([2*q[0], 2*q[1]])
        else:
            components = np.array([q[0]-q[1], q[0]+2*q[1]])
        if np.max(np.abs(components - np.rint(components))) > float(atol):
            raise ValueError(
                f"K or Q={tuple(q)} is not commensurate with {self.name} PBC; "
                f"allowed momenta: {self.allowed_momenta()}"
            )
        return q
    def allowed_momenta(self):
        if self.name == "3x1":
            return ((0.,0.), (1/3,0.), (2/3,0.))
        if self.name == "2x2":
            return ((0.,0.), (.5,0.), (0.,.5), (.5,.5))
        return ((0.,0.), (1/3,1/3), (2/3,2/3))


def build_torus_geometry(name: str, params: RubyParameters) -> TorusGeometry:
    name = str(name).lower().replace("_","").replace("-","")
    aliases = {"3x1":"3x1","2x2":"2x2","hex":"hex18","hex18":"hex18","ring3":"hex18"}
    if name not in aliases:
        raise ValueError("geometry must be 3x1, 2x2 or hex18 (periodic skew 18-site torus)")
    name = aliases[name]
    if name == "hex18":
        cells = np.asarray(SUPERCELL_REPRESENTATIVES, dtype=int)
        h = build_supercell_h0(np.zeros((1,2)), params)[0]
        v = build_supercell_interaction(np.zeros((1,2)), params)[0]
    else:
        L1,L2 = (3,1) if name=="3x1" else (2,2)
        mesh = MatsubaraGrid(nk1=L1,nk2=L2,nw=1,nOmega=1,T=.2)
        h = fold_static_k_field(build_h0(mesh.kmesh(), params))
        v = fold_static_k_field(build_interaction(mesh.qmesh(), params))
        cells = np.asarray(torus_cells(L1,L2), dtype=int)
    if np.max(np.abs(h-h.conj().T)) > 1e-10:
        raise RuntimeError("folded hopping matrix is not Hermitian")
    if np.max(np.abs(v-v.conj().T)) > 1e-10:
        raise RuntimeError("folded V matrix is not Hermitian")
    ncell=len(cells)
    return TorusGeometry(name,cells,h[None,None],v[None,None],6*ncell,ncell)


def k_projector(geom: TorusGeometry, K) -> np.ndarray:
    q=geom.validate_momentum(K)
    U=np.zeros((geom.nsites,NSUB),dtype=complex)
    for c,R in enumerate(geom.cells):
        sl=slice(NSUB*c,NSUB*(c+1))
        U[sl,:]=np.exp(2j*np.pi*np.dot(q,R))*np.eye(NSUB)/np.sqrt(geom.ncell)
    return U


def project_k_green(G: np.ndarray, geom: TorusGeometry, K) -> np.ndarray:
    a=np.asarray(G,dtype=complex)
    if a.ndim==5:
        if a.shape[1:3]!=(1,1):
            raise ValueError("GW G must use exactly one reduced-supercell k point")
        a=a[:,0,0]
    if a.ndim!=3 or a.shape[-2:]!=(geom.nsites,geom.nsites):
        raise ValueError("G should have shape (nf,18/24,18/24)")
    U=k_projector(geom,K)
    return np.einsum("ia,nij,jb->nab",U.conj(),a,U,optimize=True)


def hermitian_source_components(geom: TorusGeometry, channel: str, Q) -> list[np.ndarray]:
    """Hermitian cos/sin source operators with O(Q)=Ocos-i Osin."""
    q=geom.validate_momentum(Q)
    vertex=primitive_pseudospin_vertex(channel)
    components=[]
    for trig in (np.cos,np.sin):
        M=np.zeros((geom.nsites,geom.nsites),dtype=complex)
        for ic,R in enumerate(geom.cells):
            angle=2*np.pi*float(np.dot(q,R))
            weight=float(trig(angle))/np.sqrt(float(geom.ncell))
            sl=slice(6*ic,6*(ic+1))
            M[sl,sl]=weight*vertex
        if np.max(np.abs(M))>1e-12:
            components.append(M)
    return components


def green_k_tau(
    G_iomega: np.ndarray, omega: np.ndarray, T: float, mu: float,
    h0_K: np.ndarray, tau: np.ndarray,
) -> np.ndarray:
    """Fermionic inverse Matsubara with an exact noninteracting reference tail.

    G_ref=(iw+mu-h0_K)^-1 and G_ref(tau)=-exp[-tau(e-mu)](1-f(e-mu)).
    This exactly enforces its discontinuity at tau=0 and anti-periodicity,
    while the finite frequency box is applied only to G-G_ref.
    """
    G=np.asarray(G_iomega,dtype=complex)
    w=np.asarray(omega,dtype=float).reshape(-1)
    t=np.asarray(tau,dtype=float).reshape(-1)
    H=np.asarray(h0_K,dtype=complex)
    E,U=np.linalg.eigh(0.5*(H+H.conj().T))
    d=E-float(mu)
    denom=1j*w[:,None]-d[None,:]
    Gref=np.einsum("ia,na,ja->nij",U,1/denom,U.conj(),optimize=True)
    # numerically stable -(1-f)*exp(-tau*d)
    from scipy.special import expit
    occupation=expit(-d/float(T))
    ref_t=-np.exp(np.clip(-t[:,None]*d[None,:],-700,700))*(1-occupation)[None,:]
    # For d << -T and tau near beta, direct 0*infinity loses precision.
    # Equivalent exp(-tau*d)/(1+exp(-beta*d)) in log form is stable.
    logamp=-t[:,None]*d[None,:]-np.logaddexp(0.0,-d[None,:]/float(T))
    ref_t=-np.exp(logamp)
    ref=np.einsum("ia,ta,ja->tij",U,ref_t,U.conj(),optimize=True)
    phase=np.exp(-1j*np.outer(t,w))
    corrected=ref+float(T)*np.einsum("tn,nij->tij",phase,G-Gref,optimize=True)
    return np.asarray(corrected.real,dtype=float)


def finite_temperature_energy_estimate(
    G: np.ndarray, sigma_h: np.ndarray, sigma_dynamic: np.ndarray,
    h0: np.ndarray, Vq: np.ndarray, grid: MatsubaraGrid, mu: float,
) -> dict[str,float]:
    """Galitskii-Migdal finite-T estimate, NOT a zero-temperature E0.

    Input sigma_dynamic includes Fock plus correlation; split using the same
    static Fock convention as production GW.
    """
    rho=one_body_density_matrix_tail(G,grid,h0,mu,sigma_h)
    _,fock,corr,_=compute_sigma_gw_split_components(
        G,compute_screened_interaction_matrix(
            compute_polarization_matrix(G,grid),Vq
        ),Vq,grid,h0,mu,sigma_h,backend="fft"
    )
    ek=np.einsum("xyab,xyba->",h0,rho,optimize=True).real/grid.nk
    eh=.5*np.einsum("ab,xyba->",sigma_h,rho,optimize=True).real/grid.nk
    ef=.5*np.einsum("xyab,xyba->",fock,rho,optimize=True).real/grid.nk
    ec=.5*grid.T*np.einsum("nxyab,nxyba->",corr,G,optimize=True).real/grid.nk
    return dict(kinetic=float(ek),hartree=float(eh),fock=float(ef),
                correlation=float(ec),total=float(ek+eh+ef+ec))


def gw_energy_from_state(state, geom: TorusGeometry, grid: MatsubaraGrid):
    rho=one_body_density_matrix_tail(
        state.G,grid,geom.h0,float(state.mu),state.Sigma_H
    )
    _,fock,corr,_=compute_sigma_gw_split_components(
        state.G,state.W,geom.vq,grid,geom.h0,float(state.mu),
        state.Sigma_H,backend="fft"
    )
    ek=float(np.einsum("xyab,xyba->",geom.h0,rho).real/grid.nk)
    eh=.5*float(np.einsum("ab,xyba->",state.Sigma_H,rho).real/grid.nk)
    ef=.5*float(np.einsum("xyab,xyba->",fock,rho).real/grid.nk)
    ec=.5*float((grid.T/grid.nk)*np.einsum(
        "nxyab,nxyba->",corr,state.G).real)
    return float(ek+eh+ef+ec)


def embedded_gm_energy(state, geom: TorusGeometry, grid: MatsubaraGrid):
    """A non-variational finite-T diagnostic for ED+GW, not certified E0."""
    G=np.asarray(state.G)
    H=np.asarray(state.Sigma_H)
    sig=np.asarray(state.Sigma_emb)+H[None,None,None]
    rho=one_body_density_matrix_tail(G,grid,geom.h0,state.mu,H)
    # Separate the even high-frequency constant before the Matsubara trace.
    w=np.asarray(grid.omega)
    ip=int(np.argmax(w)); im=int(np.argmin(w))
    stat=.5*(sig[ip]+sig[im])
    stat=.5*(stat+stat.swapaxes(-1,-2).conj())
    kinetic=np.einsum("xyab,xyba->",geom.h0,rho).real/grid.nk
    static=.5*np.einsum("xyab,xyba->",stat,rho).real/grid.nk
    corr=.5*grid.T*np.einsum(
        "nxyab,nxyba->",sig-stat[None],G
    ).real/grid.nk
    return float(kinetic+static+corr)


def cluster_seed_from_density_gw(
    background, geom: TorusGeometry, grid: MatsubaraGrid,
    target_particles: float, *, mu_tol: float = 1e-10,
):
    """Turn a density-channel GW iterate into a valid embedding *initial guess*.

    A preliminary lattice GW fixed point is not required for convergence of
    the subsequent coupled cluster-ED+GW equations: every embedding iteration
    rebuilds lattice P, W and Sigma_GW from the current embedded G.

    The channel GW result reports freshly evaluated output self-energies, which
    can differ from the Dyson self-energies behind its G when the preliminary
    fixed point was NOT reached. Re-solve the *one-body Dyson equation only*
    in that case to make G, Sigma_H, Sigma_GW and filling consistent before
    passing them into the embedding. This is not another GW iteration.
    """
    h=np.asarray(background.Sigma_tadpole,dtype=complex)
    f=np.asarray(background.Sigma_exchange,dtype=complex)
    sc=np.asarray(background.Sigma_c,dtype=complex)
    norb=int(geom.nsites)
    expected=(grid.nf,1,1,norb,norb)
    if h.shape!=(norb,norb) or f.shape!=(norb,norb) or sc.shape!=expected:
        raise ValueError("density-GW seed self-energy shape does not match torus")
    if any(not np.all(np.isfinite(v)) for v in (h,f,sc,background.G)):
        raise ValueError("density-GW seed contains nonfinite self-energy or G")
    sigma_gw=f[None,None,None,:,:]+sc
    if background.converged:
        G=np.asarray(background.G,dtype=complex).copy()
        mu=float(background.mu)
        rebuilt=False
    else:
        # The state returned after max_iter may contain Sigma[updated] but
        # G[mixed]. A single Dyson solve reconciles this harmlessly.
        mu,G,_,_=_solve_mu_matrix_fast(
            geom.h0,h,sigma_gw,grid,float(target_particles),
            float(background.mu),float(mu_tol),100
        )
        rebuilt=True
    if not np.isfinite(mu) or not np.all(np.isfinite(G)):
        raise ValueError("density-GW seed Dyson solve produced nonfinite G/mu")
    rho=one_body_density_matrix_tail(G,grid,geom.h0,mu,h)
    density=np.asarray(np.diag(rho[0,0]).real,dtype=float)
    if not np.all(np.isfinite(density)):
        raise ValueError("density-GW seed density is nonfinite")
    return SimpleNamespace(
        G=np.asarray(G),mu=float(mu),
        Sigma_H=h.copy(),Sigma_GW=sigma_gw.copy(),density=density,
        converged=bool(background.converged),
        final_error=float(background.final_error),
        seed_dyson_rebuilt=rebuilt,
        seed_origin="density-GW-preiterate" if not background.converged
                    else "density-GW-converged",
        seed_density_error=float(np.sum(density)-float(target_particles)),
    )


def uniform_primitive_cluster_ed_gw(
    geom: TorusGeometry, grid: MatsubaraGrid, background, *,
    nbath=6, max_iter=45, tol=3e-5, mixing=.35,
    bath_nfit=12, bath_max_nfev=200, verbose=False,
    target_particles: float | None = None,
):
    """One equivalent six-site impurity replicated over an arbitrary torus.

    This is a Q=0 / primitive-translation-invariant Ansatz. For Q!=0
    susceptibility use the separate multi-impurity finite-source solver.
    """
    G=np.array(background.G,copy=True)
    mu=float(background.mu)
    sigma_h=np.array(background.Sigma_H,copy=True)
    sigma_emb=np.array(background.Sigma_GW,copy=True)
    ncell=geom.ncell
    filling_target=(float(np.sum(background.density)) if target_particles is None
                    else float(target_particles))
    if not np.isfinite(filling_target):
        raise ValueError("cluster-ED+GW target filling must be finite")
    chunks=[slice(6*c,6*(c+1)) for c in range(ncell)]
    one=np.eye(6,dtype=complex)
    hloc=np.mean([geom.h0[0,0,sl,sl] for sl in chunks],axis=0)
    ints=ruby_cluster_interactions(
        RubyParameters(ti=1,t1=-.5,t2=.5,V=float(np.real(geom.vq[0,0,0,1])))
    )
    Vc=cluster_interaction_matrix(ints,6)
    bath=None
    rho=one_body_density_matrix_tail(G,grid,geom.h0,mu,sigma_h)
    Gc=np.mean([G[:,0,0,sl,sl] for sl in chunks],axis=0)
    rhoc=np.mean([rho[0,0,sl,sl] for sl in chunks],axis=0)
    sigma_imp=cluster_gw_self_energy(Gc,rhoc,Vc,grid)[0]
    err=float("inf")
    mismatch=float("inf")
    bath_err=float("inf")
    converged=False
    sigma_gw=np.array(background.Sigma_GW,copy=True)
    for it in range(1,int(max_iter)+1):
        rho=one_body_density_matrix_tail(G,grid,geom.h0,mu,sigma_h)
        n=np.diagonal(rho[0,0]).real
        sh_new=hartree_self_energy_matrix(n,geom.vq[0,0])
        P=compute_polarization_matrix(G,grid)
        W=compute_screened_interaction_matrix(P,geom.vq)
        sigma_gw=compute_sigma_gw_split_matrix(
            G,W,geom.vq,grid,geom.h0,mu,sigma_h,backend="fft"
        )
        Gc=np.mean([G[:,0,0,sl,sl] for sl in chunks],axis=0)
        rhoc=np.mean([rho[0,0,sl,sl] for sl in chunks],axis=0)
        scgw,_,_=cluster_gw_self_energy(Gc,rhoc,Vc,grid)
        g0inv=np.linalg.inv(Gc)+sigma_imp
        raw=(1j*grid.omega[:,None,None]+mu)*one[None]-hloc[None]-g0inv
        static,delta=split_static_hybridization(raw,grid.omega)
        himp=hloc+static
        bath=fit_finite_bath(
            delta,grid.omega,mu,nbath=int(nbath),nfit=int(bath_nfit),
            max_nfev=int(bath_max_nfev),initial=bath,
            metric="delta",one_body=himp
        )
        solver=FiniteBathImpurityED(
            build_impurity_one_body(himp,bath),ints,
            correlated_orbitals=tuple(range(6))
        )
        solver.diagonalize()
        Gimp,_=solver.green_iomega(1j*grid.omega,mu,grid.T,
                                   discard_weight_tol=1e-11)
        deltafit=bath_hybridization(
            grid.omega,mu,bath.energies,bath.couplings
        )
        sigma_new=(1j*grid.omega[:,None,None]+mu)*one[None]-himp[None]-deltafit-np.linalg.inv(Gimp)
        emb_new=np.array(sigma_gw,copy=True)
        for sl in chunks:
            emb_new[:,0,0,sl,sl]+=sigma_new-scgw
        mismatch=float(np.linalg.norm(Gimp-Gc)/max(np.linalg.norm(Gc),1e-300))
        bath_err=float(bath.fit_error)
        err=float(max(
            np.max(np.abs(sh_new-sigma_h)),
            np.max(np.abs(emb_new-sigma_emb)),
            np.max(np.abs(sigma_new-sigma_imp))
        ))
        if verbose:
            print(f"[uniform ED+GW] it={it}, residual={err:.3e}, bath={bath_err:.3e}, Gimp/Gc={mismatch:.3e}",flush=True)
        a=1.0 if err<tol else float(mixing)
        sigma_h=sigma_h+a*(sh_new-sigma_h)
        sigma_emb=sigma_emb+a*(emb_new-sigma_emb)
        sigma_imp=sigma_imp+a*(sigma_new-sigma_imp)
        mu,G,_,_=_solve_mu_matrix_fast(
            geom.h0,sigma_h,sigma_emb,grid,
            filling_target,mu,1e-10,100
        )
        if err<tol:
            converged=True
            break
    return SimpleNamespace(
        G=G,mu=float(mu),Sigma_H=sigma_h,Sigma_emb=sigma_emb,
        Sigma_ED_cluster=sigma_imp,Sigma_GW_lattice=sigma_gw,
        bath_fit_error=bath_err,impurity_mismatch=mismatch,
        bath=bath,
        iterations=it,final_error=err,converged=converged
    )


class SkewEDSectorAdapter:
    """Offer the ED18 sparse fixed-N solver through the Lanczos sector API."""
    def __init__(self, params: RubyParameters, n_particles: int):
        ed=ED18Solver(params,n_particles=int(n_particles))
        self.ed=ed
        self.basis=np.asarray(ed.basis,dtype=np.uint64)
        self.dimension=int(ed.dimension)
        self._cached_v=None
        self._cached_h=None

    def _state_indices(self, states):
        return np.asarray([self.ed.index[int(x)] for x in states],dtype=np.int64)

    def hamiltonian_operator(self,V):
        if self._cached_h is None or self._cached_v!=float(V):
            self._cached_h=self.ed.hamiltonian(float(V))
            self._cached_v=float(V)
        return self._cached_h

    def apply_hamiltonian(self,V,x):
        return self.hamiltonian_operator(V)@x


def max_required_ed_dimension(nsites: int, n_particles: int, padding: int) -> int:
    if not 0 < n_particles < nsites:
        raise ValueError("fixed-particle ED requires 0<N<Nsites")
    lo=max(1,n_particles-padding)
    hi=min(nsites-1,n_particles+padding)
    return max(comb(nsites,N) for N in range(max(0,lo-1),min(nsites,hi+1)+1))


def channel_static_response(
    state, geom: TorusGeometry, grid: MatsubaraGrid, Q,
    channels: tuple[str,...], *, include_cgw=True,
    vertex_opts: SupercellVertexOptions | None = None,
    strict=True,
) -> tuple[dict[str,float],dict[str,float],dict[str,bool]]:
    """GW bubble and cGW static response in exactly matched channel/Q units."""
    _,sigF,_,_=compute_sigma_gw_split_components(
        state.G,state.W,geom.vq,grid,geom.h0,state.mu,state.Sigma_H
    )
    hstatic=geom.h0+state.Sigma_H[None,None]+sigF
    reference=build_tail_reference(geom.h0,state.mu,state.Sigma_H,grid)
    vertex_opts=vertex_opts or SupercellVertexOptions(verbose=False)
    bubble={}
    covariant={}
    status={}
    for ch in channels:
        components=hermitian_source_components(geom,ch,Q)
        chi_gg=0.
        chi_cov=0.
        all_conv=True
        for K in components:
            Karr=K[None]
            gg=static_gg_tail_completed(
                state.G,Karr,grid,hstatic,state.mu
            )["completed"][0,0].real
            chi_gg+=float(gg)
            if include_cgw:
                out=solve_vertex_q0_tail(
                    state.G,state.W,geom.vq,K,grid,reference,
                    opts=vertex_opts
                )
                all_conv=all_conv and bool(out.converged)
                if strict and not out.converged:
                    raise RuntimeError(
                        f"cGW vertex failed for {ch}, residual={out.final_error:.3e}"
                    )
                chi_cov+=float(static_response_tail_completed(
                    state.G,Karr,[out.Gamma],grid,hstatic,state.mu
                )["completed"][0,0].real)
        bubble[ch]=float(chi_gg)
        if include_cgw:
            covariant[ch]=float(chi_cov)
            status[ch]=bool(all_conv)
    return bubble,covariant,status
