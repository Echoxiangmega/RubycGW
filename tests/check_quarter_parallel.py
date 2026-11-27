"""Check parallel SDW candidates against serial, with identical RNG streams.

Run: OPENBLAS_NUM_THREADS=1 python tests/check_quarter_parallel.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'ed_benchmark'/'native_ed'),
              str(REPO/'ed_benchmark'/'vmc_zeroT'),str(REPO/'research')]

from geometry import make_ring3_supercell  # noqa: E402
from quarter_competing_vscan import optimize  # noqa: E402
from vmc_ansatz import jastrow_channels  # noqa: E402


def check():
    g=make_ring3_supercell(2,2,t=1.,t1=-.5,t2=.5)
    _,_,mats,_=jastrow_channels(g,mode='bond_aware',n_shells=4)
    options=SimpleNamespace(
        seed_amplitude=.09,train_samples=16,burn=2,rng_seed=2026,
        refine_steps=1,eval_samples=32,eval_burn=2,interval=1,
        pair_samples=0,candidate_workers=1,
    )
    serial=optimize(g,18,mats,'sdw',1.5,[],options,np.random.default_rng(34))
    options.candidate_workers=2
    parallel=optimize(g,18,mats,'sdw',1.5,[],options,np.random.default_rng(34))
    for key in ('mc_E','mc_E_error','acceptance','mx','my','mz','B_sign'):
        assert np.isclose(serial[key],parallel[key],rtol=1e-10,atol=1e-10),key
    assert np.allclose(serial['v'],parallel['v'],rtol=1e-10,atol=1e-10)
    print('Serial/parallel SDW candidate energies and final state agree')


if __name__=='__main__':check()
