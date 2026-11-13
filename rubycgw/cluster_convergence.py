"""Diagnostics and safeguarded mixing for direct strong-V cluster ED+GW.

No interaction continuation, and no modification of the impurity or GW maps.
"""
from __future__ import annotations

import numpy as np

from .gw import GWOptions, _mixed_self_energies
from .cluster_ed_gw import BathParameters


def maxabs(a):
    return float(np.max(np.abs(np.asarray(a)), initial=0.0))


class CoupledSelfEnergyMixer:
    """Pulay for (static, lattice embedding, impurity) with bounded steps.

    Treat the impurity residual as an independent block of the DIIS metric,
    rather than allowing its small array size to hide it inside the large
    lattice self-energy. The history is transient and resets after restart.
    """
    def __init__(self, *, method="pulay", alpha=0.3, history=6, start=3,
                 regularization=1e-7, step_cap=2.0, reset_growth=2.0,
                 impurity_weight=1.0):
        if method not in ("linear", "pulay"):
            raise ValueError("mixing method must be linear or pulay")
        if not 0 < float(alpha) <= 1:
            raise ValueError("mixing alpha must be in (0,1]")
        if int(history) < 2 or int(start) < 1:
            raise ValueError("Pulay history >=2 and start >=1 are required")
        if float(step_cap) <= 0 or float(reset_growth) <= 1:
            raise ValueError("invalid Pulay safeguard parameters")
        if float(impurity_weight) <= 0:
            raise ValueError("impurity_weight must be positive")
        self.opts=GWOptions(
            mixing=float(alpha),mixing_method=method,
            pulay_history=int(history),pulay_start=int(start),
            pulay_regularization=float(regularization),
        )
        self.history=[]
        self.step_cap=float(step_cap)
        self.reset_growth=float(reset_growth)
        self.weight=float(impurity_weight)
        self.last_residual=None
        self.fallbacks=0
        self.resets=0

    def propose(self,sh,emb,imp,sh_out,emb_out,imp_out,iteration):
        sh=np.asarray(sh,complex)
        emb=np.asarray(emb,complex)
        imp=np.asarray(imp,complex)
        sh_out=np.asarray(sh_out,complex)
        emb_out=np.asarray(emb_out,complex)
        imp_out=np.asarray(imp_out,complex)
        nemb=emb.size
        scale=self.weight
        old=np.concatenate([emb.ravel(),scale*imp.ravel()])
        new=np.concatenate([emb_out.ravel(),scale*imp_out.ravel()])
        residual=max(maxabs(sh_out-sh),maxabs(emb_out-emb),
                     maxabs(imp_out-imp))
        if (self.last_residual is not None and
                residual > self.reset_growth*max(self.last_residual,1e-30)
                and self.history):
            self.history.clear()
            self.resets+=1
        self.last_residual=residual
        try:
            shnext,dynnext=_mixed_self_energies(
                sh,old,sh_out,new,self.opts,int(iteration),self.history
            )
            embnext=dynnext[:nemb].reshape(emb.shape)
            impnext=(dynnext[nemb:]/scale).reshape(imp.shape)
            raw=max(maxabs(sh_out-sh),maxabs(new-old))
            step=max(maxabs(shnext-sh),maxabs(dynnext-old))
            safe=(np.all(np.isfinite(shnext)) and
                  np.all(np.isfinite(dynnext)) and
                  (raw<=1e-14 or step<=self.step_cap*raw))
            if not safe:
                raise ArithmeticError("unsafe or oversized Pulay step")
            return shnext,embnext,impnext
        except (np.linalg.LinAlgError,ValueError,ArithmeticError):
            self.history.clear()
            self.fallbacks+=1
            a=float(self.opts.mixing)
            return (sh+a*(sh_out-sh),emb+a*(emb_out-emb),
                    imp+a*(imp_out-imp))


def screening_diagnostic(P, Vq, channels, grid):
    """Return the smallest singular value of the actual GW screening kernel."""
    if channels.mode=="density":
        pol=np.asarray(P,complex)[:,0,0]
        coupling=np.asarray(Vq,complex)[0,0]
    else:
        pol=np.asarray(P,complex)
        coupling=np.asarray(channels.lattice.coupling,complex)
    mat=np.eye(coupling.shape[0])[None]-np.matmul(coupling[None],pol)
    singular=np.linalg.svd(mat,compute_uv=False)[:,-1]
    index=int(np.argmin(singular))
    return dict(
        min_singular_value=float(singular[index]),
        bosonic_m=int(grid.m_values[index]),
        omega=float(grid.Omega[index]),
    )


def fit_bath_with_restarts(fitter,delta,omega,mu,*,initial=None,
                           restarts=0,retry_error=0.03,seed=0,**kwargs):
    """Re-fit only when optimization fails, hits a bound, or residual is poor.

    Keep the lowest-residual result; do not report an unsuccessful optimization
    as successfully converged. Additional initial guesses are reproducible.
    """
    best=fitter(delta,omega,mu,initial=initial,**kwargs)
    attempts=1
    if (int(restarts)<=0 or
            (bool(best.success) and not bool(best.active_bounds) and
             float(best.fit_error)<=float(retry_error))):
        return best,attempts
    rng=np.random.default_rng(int(seed))
    n=int(kwargs["nbath"])
    window=float(kwargs.get("energy_window",4.0))
    bound=float(kwargs.get("coupling_bound",4.0))
    for j in range(int(restarts)):
        if j==0:
            trial=None  # deterministic moment-based fresh starting point
        else:
            ref=initial if initial is not None else best
            eps=np.asarray(ref.energies,float).copy()
            eps+=rng.normal(scale=.08*window,size=n)
            eps=np.clip(eps,mu-window+1e-6,mu+window-1e-6)
            couplings=np.asarray(ref.couplings,complex).copy()
            couplings+=rng.normal(size=couplings.shape)*.06*max(
                np.linalg.norm(couplings)/np.sqrt(couplings.size),1e-3
            )
            if np.max(np.abs(couplings.imag),initial=0.)<1e-15:
                couplings=couplings.real
            couplings=np.clip(couplings.real,-bound+1e-6,bound-1e-6) + (
                1j*np.clip(couplings.imag,-bound+1e-6,bound-1e-6)
                if np.iscomplexobj(couplings) else 0.
            )
            trial=BathParameters(eps,couplings,float("inf"),0)
        outcome=fitter(delta,omega,mu,initial=trial,**kwargs)
        attempts+=1
        if ((bool(outcome.success) and not bool(best.success)) or
            (bool(outcome.success)==bool(best.success) and
             outcome.fit_error < best.fit_error)):
            best=outcome
    return best,attempts
