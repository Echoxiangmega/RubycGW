"""Exact implicit static derivative of a complex finite Anderson bath fit.

For F=||r(theta,D)||^2/2 with REAL parameter vector
theta=(eps-mu, Re V, Im V), differentiation of the stationary least
squares equations gives H dtheta = J^T dD_weighted.
The Hessian H=J^T J+sum_i r_i Hessian(r_i) is analytic (NOT Gauss
Newton unless the latter term vanishes). We use an SVD pseudoinverse
for phase/gauge redundancies and require a well-fitted interior optimum.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .cluster_ed_gw import BathParameters,bath_hybridization
from .cluster_ed_gw_covariant import _complex_bath_residual_jacobian


@dataclass
class BathTangentDiagnostics:
    stationarity: float
    least_squares_rank: int
    hessian_condition: float
    linearized_residual: float
    active_bounds: bool


def _weights(omega,nfit):
    w=np.asarray(omega,float).reshape(-1)
    pos=np.flatnonzero(w>0)
    pos=pos[np.argsort(w[pos])[:min(int(nfit),len(pos))]]
    if not len(pos):
        raise ValueError("positive Matsubara frequencies needed")
    wf=w[pos]
    w0=max(float(wf[0]),1.e-12)
    weights=1./np.sqrt(wf*wf+w0*w0)
    weights/=np.max(weights)
    return pos,wf,weights


def complex_bath_fit_tangent(
    target_delta: np.ndarray,
    dtarget_delta: np.ndarray,
    omega: np.ndarray, mu: float,
    bath: BathParameters, *, nfit: int=12,
    energy_window: float=4.,coupling_bound: float=4.,
    svd_cutoff: float=1.e-9,
    stationarity_tol: float=2.e-5,
):
    """Derivative of a complex bath least-squares *stationary optimum*.

    The energy derivative returned is d(eps-mu). Physical bath energies
    satisfy d eps = dxi + dmu. Optimizing in xi eliminates an explicit
    mu derivative of the rational hybridization and retains exactly
    the same objective and moving energy bounds as the original fitter.
    """
    target=np.asarray(target_delta,complex)
    dtarget=np.asarray(dtarget_delta,complex)
    w=np.asarray(omega,float)
    if target.shape!=dtarget.shape:
        raise ValueError("target and response hybridization shapes differ")
    pos,wfit,weights=_weights(w,nfit)
    eps=np.asarray(bath.energies,float)
    v=np.asarray(bath.couplings,complex)
    n,nb=v.shape
    if target.shape!=(len(w),n,n):
        raise ValueError("finite bath hybridization dimensions mismatch")
    model=bath_hybridization(wfit,mu,eps,v)
    r=(model-target[pos])*weights[:,None,None]
    residual=np.r_[r.real.ravel(),r.imag.ravel()]
    J=_complex_bath_residual_jacobian(wfit,mu,eps,v,weights)
    g=J.T@residual
    stationarity=float(np.max(abs(g),initial=0.))
    if not np.isfinite(stationarity):
        raise ValueError("nonfinite bath stationarity")
    if stationarity>float(stationarity_tol):
        raise RuntimeError(
            f"complex bath fit is not stationary: ||J^T r||_max={stationarity:.3e}"
        )
    xi=eps-float(mu)
    bound=float(coupling_bound)
    hit=bool(np.any(abs(xi)>float(energy_window)-1.e-6)
             or np.any(abs(v.real)>bound-1.e-6)
             or np.any(abs(v.imag)>bound-1.e-6))
    if hit:
        raise RuntimeError(
            "finite bath fit has active parameter bounds; smooth implicit "
            "Hessian derivative is not valid without active-set differentiation"
        )

    # Exact second-derivative contraction with the residual.
    # For one bath pole:
    #   d^2 Delta/deps^2 = 2 D^3 vv^dag;
    #   d^2 Delta/deps dv = D^2(dv v^dag+v dv^dag);
    #   d^2 Delta/dv dw = D(dv dw^dag+dw dv^dag).
    H=J.T@J
    weighted_conj=r.conj()*weights[:,None,None]
    def contraction(second):
        return float(np.real(np.sum(weighted_conj*second)))

    nvc=n*nb
    for p in range(nb):
        D=1./(1j*wfit+float(mu)-eps[p])
        vp=v[:,p]
        outer=np.outer(vp,vp.conj())
        H[p,p]+=contraction(2*(D**3)[:,None,None]*outer[None])
        u=[]
        inds=[]
        for a in range(n):
            er=np.zeros(n,complex); er[a]=1.
            ei=1j*er
            u.extend((er,ei))
            inds.extend((nb+a*nb+p,nb+nvc+a*nb+p))
        for b,ub in zip(inds,u):
            tens=np.outer(ub,vp.conj())+np.outer(vp,ub.conj())
            val=contraction((D**2)[:,None,None]*tens[None])
            H[p,b]+=val
            H[b,p]+=val
        for ii,(i,ui) in enumerate(zip(inds,u)):
            for j,uj in zip(inds[ii:],u[ii:]):
                tens=np.outer(ui,uj.conj())+np.outer(uj,ui.conj())
                val=contraction(D[:,None,None]*tens[None])
                H[i,j]+=val
                if i!=j:
                    H[j,i]+=val

    H=.5*(H+H.T)
    dd=dtarget[pos]*weights[:,None,None]
    rhs=J.T@np.r_[dd.real.ravel(),dd.imag.ravel()]
    U,sv,Vh=np.linalg.svd(H)
    threshold=max(svd_cutoff*max(float(sv[0]),1.),1.e-12)
    keep=sv>threshold
    solution=(Vh.T[:,keep]@(
        (U.T@rhs)[keep]/sv[keep]
    ) if np.any(keep) else np.zeros_like(rhs))
    residual_linear=float(np.linalg.norm(H@solution-rhs)/
                          max(float(np.linalg.norm(rhs)),1.e-30))
    return (
        np.asarray(solution[:nb],float),
        np.asarray(solution[nb:nb+nvc].reshape(n,nb)
                   +1j*solution[nb+nvc:].reshape(n,nb),complex),
        BathTangentDiagnostics(
            stationarity=stationarity,
            least_squares_rank=int(np.sum(keep)),
            hessian_condition=(float(sv[0]/sv[keep][-1])
                               if np.any(keep) else float("inf")),
            linearized_residual=residual_linear,
            active_bounds=False,
        ),
    )


def bath_delta_tangent(omega,mu,bath,dxi,dv):
    """Analytic dDelta for relative bath energies xi=eps-mu."""
    w=np.asarray(omega,float)
    xi=np.asarray(bath.energies,float)-float(mu)
    v=np.asarray(bath.couplings,complex)
    out=np.zeros((len(w),v.shape[0],v.shape[0]),complex)
    for p in range(len(xi)):
        den=1./(1j*w-xi[p])
        out+=(den*den*float(dxi[p]))[:,None,None]*np.outer(v[:,p],v[:,p].conj())
        out+=den[:,None,None]*(
            np.outer(dv[:,p],v[:,p].conj())
            +np.outer(v[:,p],dv[:,p].conj())
        )
    return out


__all__=[
    "BathTangentDiagnostics","complex_bath_fit_tangent","bath_delta_tangent"
]
