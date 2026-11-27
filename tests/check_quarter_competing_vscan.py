"""Run: OPENBLAS_NUM_THREADS=1 python tests/check_quarter_competing_vscan.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'ed_benchmark'/'native_ed'),
              str(REPO/'ed_benchmark'/'vmc_zeroT'),str(REPO/'research')]

from core import FixedHamiltonian,make_states  # noqa: E402
from geometry import RubyPBC,make_ring3_supercell  # noqa: E402
from observables import apply_ops  # noqa: E402
from quarter_competing_vscan import Trial,exact_filling,_add,_remove  # noqa: E402
from ruby_did_vmc import pairing_source,pfaffian  # noqa: E402
from vmc_ansatz import jastrow_channels  # noqa: E402


def check():
    assert exact_filling(2,2,'1/4')==(72,18)
    assert np.all(make_ring3_supercell(2,2).M%2==0)
    for shape in ((3,3),(2,1)):
        try:exact_filling(*shape,'1/4')
        except ValueError:pass
        else:raise AssertionError('Missing three-M/even-particle check')

    rng=np.random.default_rng(4)
    for n in (4,6,10):
        a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        a=a-a.T
        assert np.allclose(np.exp(Trial.logpf(a)),pfaffian(a),rtol=1e-11)

    g=RubyPBC('2x2',t=1,t1=-.5,t2=.5)
    H=FixedHamiltonian(g,6,1.5,method='sparse')
    _,_,mats,_=jastrow_channels(g,mode='bond_aware',n_shells=4)
    dplus=pairing_source(g,.2,'d+')
    dminus=pairing_source(g,.2,'d-')
    assert np.allclose(dplus.conj(),dminus)
    paired=dict(delta=.2,mu=-1.,v=[.01]*7)
    trial_plus=Trial(g,6,'pf_dplus',paired,mats)
    trial_minus=Trial(g,6,'pf_dminus',paired,mats)
    configs=(np.array([0,1,5,9,13,20]),np.array([0,3,6,10,14,21]))
    zp=[trial_plus.logpsi(x) for x in configs]
    zm=[trial_minus.logpsi(x) for x in configs]
    assert np.allclose(np.exp(zp[1]-zp[0]),
                       np.exp(np.conj(zm[1]-zm[0])),rtol=1e-10)
    occ=np.array([0,1,5,9,13,20])
    bit=sum(1<<(g.Ns-1-int(i)) for i in occ)
    row=int(np.flatnonzero(H.states==bit)[0])
    # Check the four-fermion pair estimator's chronological CAR signs against
    # the existing exact Fock-space operator implementation.
    annihilate=(0,5)
    create=(2,6)
    stage,s1=_remove(occ,annihilate[1]);stage,s2=_remove(stage,annihilate[0])
    stage,s3=_add(stage,create[0]);stage,s4=_add(stage,create[1])
    root=np.zeros(len(H.states),complex);root[row]=1.
    bra4=make_states(g.Ns,4)
    intermediate=apply_ops(root,H.states,bra4,g.Ns,
                           [('-',annihilate[0]),('-',annihilate[1])])
    result=apply_ops(intermediate,bra4,H.states,g.Ns,
                     [('+',create[1]),('+',create[0])])
    target_bit=sum(1<<(g.Ns-1-int(i)) for i in stage)
    target_row=int(np.flatnonzero(H.states==target_bit)[0])
    assert result[target_row]==s1*s2*s3*s4

    for seed,p in (
        ('sdw',dict(mx=.08,my=.03,mz=.04,B_sign=1,v=[.01]*7)),
        ('pf_s',dict(delta=.2,mu=-1.,v=[.01]*7)),
        ('pf_dplus',dict(delta=.2,mu=-1.,v=[.01]*7)),
        ('pf_dminus',dict(delta=.2,mu=-1.,v=[.01]*7)),
    ):
        trial=Trial(g,6,seed,p,mats)
        z=trial.logpsi(occ)
        sparse=H.operator.getrow(row)
        expected=0j
        for j,h in zip(sparse.indices,sparse.data):
            state=int(H.states[j])
            config=np.flatnonzero([(state>>(g.Ns-1-i))&1
                                   for i in range(g.Ns)])
            expected+=h*np.exp(trial.logpsi(config)-z)
        measured=trial.local_energy(occ,z,1.5)
        assert np.allclose(measured,expected,rtol=2e-11,atol=1e-10),seed

        # Long accepted-hop chains must retain the exact Jastrow features and
        # fixed-N determinant/Pfaffian amplitudes after every cache update.
        chain=occ.copy();cache=trial.hop_cache(chain)
        for _ in range(24):
            src=int(rng.choice(chain))
            dst=int(rng.choice(np.setdiff1d(np.arange(g.Ns),chain)))
            ratio,_,target=trial.hop_ratio(chain,src,dst,cache)
            # The reference logpf intentionally marks near-singular matrices
            # as outside its numerical support.
            if not np.isfinite(trial.logpsi(target).real):continue
            direct=np.exp(trial.logpsi(target)-trial.logpsi(chain))
            assert np.allclose(ratio,direct,rtol=2e-10,atol=2e-11),(
                seed,src,dst,chain,ratio,direct,cache[1]-trial.features(chain))
            cache=trial.hop_cache(target,previous=cache,move=(src,dst))
            chain=target
            assert np.allclose(cache[1],trial.features(chain),atol=1e-10),seed
    print('Pfaffian, filling/M compatibility, and four local energies: OK')


if __name__=='__main__':check()
