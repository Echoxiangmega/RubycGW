"""Analytic static linear response of the finite-bath cluster-ED+GW fixed point.

The physical map is differentiated at a converged zero-source multicell
fixed point (same complex bath parameter manifold as finite-source chi).
All diagrams use the same frequency boxes and finite-torus GW skeleton
as cluster_ed_gw_multicell. The only approximation in the linearization
is the original finite Matsubara/bath discretization and the numerical
pseudoinverse of redundant bath parameters.

One ED eigensystem per inequivalent cluster is reused for every GMRES
matvec and every source; no ED is re-diagonalized under source fields.
The calculation is static (Omega=0) and assumes a smooth *interior*
stationary bath fit, not an active-bound KKT point.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .cluster_ed_gw import (
    BathParameters,bath_hybridization,build_impurity_one_body,
    cluster_interaction_matrix,ruby_cluster_interactions,
    split_static_hybridization,
)
from .cluster_bath_linear_response import (
    bath_delta_tangent,complex_bath_fit_tangent
)
from .grids import MatsubaraGrid,frequency_shift_slices
from .impurity_ed import FiniteBathImpurityED
from .impurity_linear_response import impurity_green_tangent
from .model import RubyParameters
from .supercell_gw import (
    _fermi,compute_polarization_matrix,
    compute_screened_interaction_matrix,
    compute_sigma_gw_matrix,hartree_self_energy_matrix,
)


def _density_tangent_cache(h0,mu,sigma_h,G,grid):
    h=np.asarray(h0,complex)[0,0]+np.asarray(sigma_h,complex)
    h=.5*(h+h.conj().T)
    e,u=np.linalg.eigh(h)
    f=_fermi(e-float(mu),grid.T)
    diff=e[:,None]-e[None,:]
    fd=(f[:,None]-f[None,:])/np.where(abs(diff)>1.e-9,diff,1.)
    fmid=_fermi((e[:,None]+e[None,:])*.5-float(mu),grid.T)
    fd=np.where(abs(diff)>1.e-9,fd,-fmid*(1-fmid)/grid.T)
    denom=1./(1j*grid.omega[:,None]+float(mu)-e[None,:])
    gref=np.einsum("aj,nj,bj->nab",u,denom,u.conj(),optimize=True)
    state=dict(e=e,u=u,fd=fd,gref=gref,G=np.asarray(G[:,0,0]),
               T=float(grid.T),norb=len(e))
    def rho_tangent(dg,dh_eff):
        du=u.conj().T@dh_eff@u
        drho_ref=u@(fd*du)@u.conj().T
        dgr=np.matmul(np.matmul(gref,dh_eff),gref)
        delta=drho_ref+float(grid.T)*np.sum(dg-dgr,axis=0)
        return .5*(delta+delta.conj().T)
    g0=state["G"]
    dg_mu=-np.matmul(g0,g0)
    b=float(np.trace(rho_tangent(dg_mu,-np.eye(len(e)))).real)
    state["rho_tangent"]=rho_tangent
    state["dn_dmu"]=b
    return state


def _polarization_tangent(G,dG,grid):
    A=np.asarray(G,complex)[:,0,0]
    B=np.asarray(dG,complex)[:,0,0]
    n=A.shape[-1]
    out=np.zeros((grid.nb,1,1,n,n),complex)
    for j,m in enumerate(grid.m_values):
        src,dst=frequency_shift_slices(grid.nf,int(m))
        if src.stop==src.start:
            continue
        out[j,0,0]=float(grid.T)*(
            np.einsum("nab,nba->ab",B[src],A[dst],optimize=True)+
            np.einsum("nab,nba->ab",A[src],B[dst],optimize=True)
        )
    return out


def _gw_tangent(G,dG,rho,dRho,W,Vq,grid):
    """GW derivative with Hartree stored separately.

    dP = (dG)G+G(dG), dW=W dP W;
    dSigma_F = -dRho V^T;
    dSigma_c = -(dG)(W-V)-G(dW).
    """
    Vq=np.asarray(Vq,complex)
    dP=_polarization_tangent(G,dG,grid)
    dW=np.matmul(np.matmul(W,dP),W)
    dF=-np.asarray(dRho,complex)*Vq[0,0].T
    Wc=np.asarray(W,complex)-Vq[None]
    dyn=(
        compute_sigma_gw_matrix(dG,Wc,grid,backend="fft")+
        compute_sigma_gw_matrix(G,dW,grid,backend="fft")
    )
    return dyn+dF[None,None,None],dP,dW


def _static_hermitian_freq(z,grid):
    """Project numerical noise onto Sigma(-iw)=Sigma(iw)^dagger."""
    a=np.asarray(z,complex).copy()
    nw=grid.nw
    positive=.5*(a[nw:]+a[:nw][::-1].conj().transpose(0,1,2,4,3))
    a[nw:]=positive
    a[:nw]=positive[::-1].conj().transpose(0,1,2,4,3)
    return a


@dataclass
class ClusterAnalyticResponse:
    chi: float
    converged: bool
    residual: float
    iterations: int
    time_seconds: float
    dmu: float
    dn_total: float
    bath_stationarity: float
    bath_linear_error: float


class StaticClusterEDGWLinearizer:
    """Matrix-free real-linear derivative of the converged multicell map."""

    def __init__(self,geom,params,grid,reference, *,
                 bath_nfit=12,discard_weight_tol=1.e-11,
                 bath_stationarity_tol=2.e-5):
        if not reference.converged:
            raise ValueError("analytic chi requires a converged zero-source multi-ED+GW")
        self.geom=geom
        self.params=params
        self.grid=grid
        self.ref=reference
        self.nbath=int(np.asarray(reference.bath_energies).shape[-1])
        self.norb=int(geom.nsites)
        self.ncell=int(geom.ncell)
        self.nfit=int(bath_nfit)
        self.discard_tol=float(discard_weight_tol)
        self.bath_stationarity_tol=float(bath_stationarity_tol)
        self.g=np.asarray(reference.G,complex)
        self.w=np.asarray(reference.W,complex)
        self.mu=float(reference.mu)
        self.h0=np.asarray(geom.h0,complex)
        self.sh=np.asarray(reference.Sigma_H,complex)
        self.sigma=np.asarray(reference.Sigma_emb,complex)
        self.sigma_imp=np.asarray(reference.Sigma_ED_cluster,complex)
        self.vq=np.asarray(geom.vq,complex)
        self.tail=_density_tangent_cache(self.h0,self.mu,self.sh,self.g,grid)
        self.ints=ruby_cluster_interactions(params)
        self.vc=cluster_interaction_matrix(self.ints,6)
        self.cg=MatsubaraGrid(nk1=1,nk2=1,nw=grid.nw,
                              nOmega=grid.nOmega,T=grid.T)
        self.cells=[]
        self.bath_linear_max=0.
        self.bath_stationarity_max=0.
        for c in range(self.ncell):
            sl=slice(6*c,6*(c+1))
            hl=np.asarray(self.h0[0,0,sl,sl],complex)
            S=np.asarray(reference.impurity_static_shift[c],complex)
            bath=BathParameters(
                np.asarray(reference.bath_energies[c],float),
                np.asarray(reference.bath_couplings[c],complex),
                float(reference.bath_fit_error[c]),0
            )
            himp=build_impurity_one_body(hl+S,bath)
            impurity=FiniteBathImpurityED(himp,self.ints,
                                            correlated_orbitals=tuple(range(6)))
            impurity.diagonalize()
            Gimp,_=impurity.green_iomega(
                1j*grid.omega,self.mu,grid.T,
                discard_weight_tol=self.discard_tol
            )
            Gc=self.g[:,0,0,sl,sl]
            pc=compute_polarization_matrix(Gc[:,None,None],self.cg)
            wc=compute_screened_interaction_matrix(pc,self.vc[None,None])
            self.cells.append(dict(
                sl=sl,hl=hl,S=S,bath=bath,impurity=impurity,
                Gimp=Gimp,invGimp=np.linalg.inv(Gimp),
                Gc=Gc,invGc=np.linalg.inv(Gc),pc=pc,wc=wc
            ))

    def _pack(self,dh,dSigma,dImp):
        """Real map coordinates: diagonal Hartree, positive-iw complex Sigma."""
        dhdiag=np.asarray(np.diag(dh).real,float)
        out=[dhdiag]
        for value in (dSigma,dImp):
            z=np.asarray(value,complex)[self.grid.nw:].ravel()
            out.append(z.real)
            out.append(z.imag)
        return np.concatenate(out)

    def _unpack(self,vec):
        x=np.asarray(vec,float).reshape(-1)
        nf,norb,ncell=self.grid.nf,self.norb,self.ncell
        nw=self.grid.nw
        hdiag=x[:norb]
        pos=norb
        def one(shape):
            nonlocal pos
            n=int(np.prod(shape))
            re=x[pos:pos+n];pos+=n
            im=x[pos:pos+n];pos+=n
            zp=(re+1j*im).reshape(shape)
            z=np.empty((nf,)+shape[1:],complex)
            z[nw:]=zp
            z[:nw]=zp[::-1].conj().swapaxes(-1,-2)
            return z
        ds=one((nw,1,1,norb,norb))
        di=one((nw,ncell,6,6))
        if pos!=len(x):
            raise ValueError("linearized response vector size mismatch")
        return np.diag(hdiag.astype(complex)),ds,di

    @property
    def dimension(self):
        return self.norb+2*self.grid.nw*(
            self.norb*self.norb+self.ncell*36
        )

    def lattice_direction(self,dh,dsh,dse):
        """Enforce fixed mean filling analytically via dmu."""
        G0=self.g[:,0,0]
        X=np.asarray(dh[0,0]+dsh,complex)[None]+dse[:,0,0]
        dg0=np.matmul(np.matmul(G0,X),G0)
        drho0=self.tail["rho_tangent"](dg0,dh[0,0]+dsh)
        b=float(self.tail["dn_dmu"])
        a=float(np.trace(drho0).real)
        if abs(b)<1.e-12:
            if abs(a)>1.e-8:
                raise RuntimeError("fixed-filling derivative has vanishing dN/dmu")
            dmu=0.
        else:
            dmu=-a/b
        dg=dg0-dmu*np.matmul(G0,G0)
        drho=self.tail["rho_tangent"](
            dg,dh[0,0]+dsh-dmu*np.eye(self.norb)
        )
        return dg[:,None,None],drho,dmu

    def direction(self,packed,dh):
        """Differentiated fixed-point output for a source and input tangent."""
        dsh,dse,di=self._unpack(packed)
        dg,drho,dmu=self.lattice_direction(dh,dsh,dse)
        dout_h=hartree_self_energy_matrix(
            np.diag(drho).real,self.vq[0,0]
        )
        dgw,_,_=_gw_tangent(
            self.g,dg,drho,self.w,self.vq,self.grid
        ) if False else _gw_tangent(
            self.g,dg,drho,drho,self.w,self.vq,self.grid
        )
        dout_emb=np.asarray(dgw,complex).copy()
        dout_imp=np.zeros_like(di)
        bstation=0.
        berr=0.
        for c,part in enumerate(self.cells):
            sl=part["sl"]
            dgc=dg[:,0,0,sl,sl]
            drc=drho[sl,sl]
            gC=part["Gc"][:,None,None]
            dgC=dgc[:,None,None]
            dsc,_,_=_gw_tangent(
                gC,dgC,drc,drc,part["wc"],self.vc[None,None],
                self.cg
            )
            dshc=hartree_self_energy_matrix(
                np.diag(drc).real,self.vc
            )
            dsc=dsc[:,0,0]+dshc[None]
            # Differentiate the actual Weiss target used in the code.
            inv=part["invGc"]
            dtarget=(
                dmu*np.eye(6)[None]-
                dh[0,0,sl,sl][None]-
                di[:,c]+
                np.matmul(np.matmul(inv,dgc),inv)
            )
            dS,dDelta=split_static_hybridization(
                dtarget,self.grid.omega
            )
            dhloc=dh[0,0,sl,sl]+dS
            # The reference dynamic fit target is reconstructed from the
            # exact (converged) zero-field Weiss map.
            raw=(
                (1j*self.grid.omega[:,None,None]+self.mu)
                *np.eye(6)[None]
                -part["hl"][None]-part["S"][None]
                -part["invGc"]-self.sigma_imp[c]
            )
            dx,dv,diag=complex_bath_fit_tangent(
                raw,dDelta,self.grid.omega,self.mu,part["bath"],
                nfit=self.nfit,stationarity_tol=self.bath_stationarity_tol
            )
            bstation=max(bstation,diag.stationarity)
            berr=max(berr,diag.linearized_residual)
            # Grand-canonical one-body tangent d(Himp-mu*N):
            # bath pole xi=eps-mu, hence bath diagonal tangent is dxi.
            nb=self.nbath
            dim=6+nb
            dm=np.zeros((dim,dim),complex)
            dm[:6,:6]=dhloc-dmu*np.eye(6)
            dm[6:,6:]=np.diag(dx)
            dm[:6,6:]=dv
            dm[6:,:6]=dv.conj().T
            dGimp,_=impurity_green_tangent(
                part["impurity"],self.grid.omega,self.mu,self.grid.T,dm,
                discard_weight_tol=self.discard_tol
            )
            dDeltaFit=bath_delta_tangent(
                self.grid.omega,self.mu,part["bath"],dx,dv
            )
            dSigmaED=(
                dmu*np.eye(6)[None]-dhloc[None]-dDeltaFit
                +np.matmul(
                    np.matmul(part["invGimp"],dGimp),
                    part["invGimp"]
                )
            )
            dout_imp[:,c]=dSigmaED
            dout_emb[:,0,0,sl,sl]+=dSigmaED-dsc
        self.bath_stationarity_max=max(
            self.bath_stationarity_max,bstation
        )
        self.bath_linear_max=max(self.bath_linear_max,berr)
        return self._pack(dout_h,dout_emb,dout_imp)

    def response(self,K, *, rtol=1.e-5,maxiter=10,restart=8):
        """Solve (I-dF/dSigma) dSigma=(dF/dh), then recover dG and chi."""
        t0=perf_counter()
        K=np.asarray(K,complex)
        if K.shape!=(self.norb,self.norb):
            raise ValueError("source matrix must have full torus dimensions")
        if np.max(abs(K-K.conj().T))>1.e-10:
            raise ValueError("static source must be Hermitian")
        source=np.zeros_like(self.h0)
        source[0,0]=-K
        zero=np.zeros(self.dimension,float)
        b=self.direction(zero,source)
        count=[0]
        def matvec(x):
            return np.asarray(x,float)-self.direction(x,np.zeros_like(source))
        operator=LinearOperator((self.dimension,self.dimension),
                                matvec=matvec,dtype=float)
        def callback(_):
            count[0]+=1
        sol,status=gmres(operator,b,x0=zero,
                         rtol=float(rtol),atol=0.,
                         restart=int(restart),maxiter=int(maxiter),
                         callback=callback,callback_type="pr_norm")
        residual=float(np.linalg.norm(matvec(sol)-b)/
                       max(np.linalg.norm(b),1.e-30))
        converged=bool(status==0 and residual<max(2*rtol,1.e-7))
        if not converged:
            raise RuntimeError(
                f"analytic cluster chi GMRES did not converge: "
                f"status={status}, residual={residual:.3e}, "
                f"iterations={count[0]}"
            )
        dsh,dse,_=self._unpack(sol)
        _,drho,dmu=self.lattice_direction(source,dsh,dse)
        chi=float(np.einsum("ab,ba->",K,drho,optimize=True).real)
        return ClusterAnalyticResponse(
            chi=chi,converged=True,residual=residual,
            iterations=count[0],time_seconds=float(perf_counter()-t0),
            dmu=float(dmu),dn_total=float(np.trace(drho).real),
            bath_stationarity=float(self.bath_stationarity_max),
            bath_linear_error=float(self.bath_linear_max),
        )


__all__=["StaticClusterEDGWLinearizer","ClusterAnalyticResponse"]
