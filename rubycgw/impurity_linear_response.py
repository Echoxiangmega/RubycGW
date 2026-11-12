"""Analytic static directional derivative of finite-temperature ED impurity G.

Differentiates the finite-bath Lehmann resolvent at a *single* ED
eigensystem: grand-canonical weights, occupied source states, and the
two-resolvent insertion of the Hamiltonian perturbation in the
opposite particle-number sector. No finite source diagonalizations.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .impurity_ed import FiniteBathImpurityED


@dataclass
class ImpurityTangentDiagnostics:
    retained_weight: float
    source_states: int
    degenerate_blocks: int
    max_energy_derivative: float


def _direction_data(impurity, dh, *, degeneracy_tol=1.e-9):
    """Rotate degenerate eigenspaces to diagonalize the direction dK."""
    tangent=FiniteBathImpurityED(
        dh,(),correlated_orbitals=impurity.correlated_orbitals
    )
    states=[]
    ndegen=0
    for n,sec in enumerate(impurity.sectors):
        E=np.asarray(sec.energies,float).copy()
        U=np.asarray(sec.eigenvectors,complex).copy()
        dH=tangent.dense_hamiltonian(n)
        i=0
        while i<len(E):
            j=i+1
            while j<len(E) and abs(E[j]-E[i])<=degeneracy_tol:
                j+=1
            if j-i>1:
                u=U[:,i:j]
                a=u.conj().T@dH@u
                _,Z=np.linalg.eigh(.5*(a+a.conj().T))
                U[:,i:j]=u@Z
                ndegen+=1
            i=j
        dHU=dH@U
        dE=np.einsum("ij,ij->j",U.conj(),dHU,optimize=True).real
        states.append(dict(E=E,U=U,dH=dH,dHU=dHU,dE=dE))
    return states,ndegen


def impurity_green_tangent(
    impurity: FiniteBathImpurityED,
    omega: np.ndarray, mu: float, T: float,
    d_one_body_minus_mu: np.ndarray, *,
    discard_weight_tol: float=1.e-11,
    degeneracy_tol: float=1.e-9,
):
    """Directional derivative of G_imp(iw) at a static Hermitian d(H-mu N).

    The argument d_one_body_minus_mu includes -dmu on all impurity+bath
    orbital diagonals. For an existing finite-bath ED eigensystem no
    new many-body eigenproblem is solved. The positive/negative
    thermal Lehmann branches are differentiated separately.
    """
    if impurity._sectors is None:
        raise RuntimeError("diagonalize unperturbed impurity before tangent")
    dh=np.asarray(d_one_body_minus_mu,complex)
    norb=int(impurity.n_orbitals)
    if dh.shape!=(norb,norb):
        raise ValueError("impurity perturbation shape mismatch")
    if np.max(abs(dh-dh.conj().T))>1.e-9:
        raise ValueError("static ED perturbation must be Hermitian")
    if T<=0:
        raise ValueError("T must be positive")
    z=1j*np.asarray(omega,float).reshape(-1)
    sites=tuple(impurity.correlated_orbitals)
    nd=len(sites)
    states,ndeg=_direction_data(impurity,dh,degeneracy_tol=degeneracy_tol)
    probabilities,_=impurity._normalized_probabilities(mu,T)
    selection=impurity.thermal_selection(
        mu,T,discard_weight_tol=discard_weight_tol
    )
    energies=[d["E"]-mu*n for n,d in enumerate(states)]
    averaged=float(sum(
        np.dot(p,d["dE"]) for p,d in zip(probabilities,states)
    ))
    dp=[-p*(d["dE"]-averaged)/T
        for p,d in zip(probabilities,states)]
    retained=[]
    for n,(selection_n,d) in enumerate(zip(selection.kept_indices,states)):
        idx=set(map(int,selection_n))
        for j in tuple(idx):
            idx.update(int(k) for k in np.flatnonzero(
                abs(d["E"]-d["E"][j])<=degeneracy_tol
            ))
        retained.append(sorted(idx))

    out=np.zeros((len(z),nd,nd),complex)
    nsource=0

    def dot_vector(data,index):
        E,U=data["E"],data["U"]
        overlaps=U.conj().T@data["dHU"][:,index]
        gap=E[index]-E
        coeff=np.zeros(len(E),complex)
        good=abs(gap)>degeneracy_tol
        coeff[good]=overlaps[good]/gap[good]
        return U@coeff

    for N in range(norb):
        low,up=states[N],states[N+1]
        Ul,Uu=low["U"],up["U"]
        El,Eu=energies[N],energies[N+1]
        for j in retained[N]:
            p=float(probabilities[N][j])
            pd=float(dp[N][j])
            if not p and not pd:
                continue
            nsource+=1
            v=Ul[:,j]
            dv=dot_vector(low,j)
            cr=np.column_stack([
                impurity._apply_creation_batch(N,a,v)[:,0] for a in sites
            ])
            dcr=np.column_stack([
                impurity._apply_creation_batch(N,a,dv)[:,0] for a in sites
            ])
            B=Uu.conj().T@cr
            dB=Uu.conj().T@dcr
            diff=El[j]-Eu
            for iw,w in enumerate(z):
                r=1./(w+diff)
                rb=r[:,None]*B
                g=B.conj().T@rb
                dterm=dB.conj().T@rb+B.conj().T@(r[:,None]*dB)
                # R(dK_up-dxi_low)R: no derivative of upper
                # eigenvectors is needed.
                right=Uu@rb
                left=Uu@(r.conj()[:,None]*B)
                dterm+=left.conj().T@(up["dH"]@right)
                dterm-=low["dE"][j]*(B.conj().T@((r*r)[:,None]*B))
                out[iw]+=p*dterm+pd*g

        for j in retained[N+1]:
            p=float(probabilities[N+1][j])
            pd=float(dp[N+1][j])
            if not p and not pd:
                continue
            nsource+=1
            v=Uu[:,j]
            dv=dot_vector(up,j)
            an=np.column_stack([
                impurity._apply_annihilation_batch(N+1,a,v)[:,0]
                for a in sites
            ])
            dan=np.column_stack([
                impurity._apply_annihilation_batch(N+1,a,dv)[:,0]
                for a in sites
            ])
            A=Ul.conj().T@an
            dA=Ul.conj().T@dan
            diff=El-Eu[j]
            for iw,w in enumerate(z):
                r=1./(w+diff)
                ra=r[:,None]*A.conj()
                g=A.T@ra
                dterm=dA.T@ra+A.T@(r[:,None]*dA.conj())
                # R(dxi_up-dK_low)R.
                # Upper-source Lehmann term has A_a R A_b^*, i.e.
                # a TRANSPOSED low-sector resolvent. The inserted dK
                # must therefore be transposed too. Using dH without
                # .T gives an incorrect imaginary off-diagonal response
                # for complex loop-current sources, even though real
                # hopping tests pass.
                right=Ul.conj()@ra
                left=Ul.conj()@(r.conj()[:,None]*A.conj())
                dterm-=left.conj().T@(low["dH"].T@right)
                dterm+=up["dE"][j]*(A.T@((r*r)[:,None]*A.conj()))
                out[iw]+=p*dterm+pd*g

    return out,ImpurityTangentDiagnostics(
        retained_weight=float(selection.kept_weight),
        source_states=int(nsource),
        degenerate_blocks=int(ndeg),
        max_energy_derivative=max(
            float(np.max(abs(d["dE"]))) for d in states
        ),
    )


__all__=["ImpurityTangentDiagnostics","impurity_green_tangent"]
