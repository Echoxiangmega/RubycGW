#!/usr/bin/env python3
"""NetKet-level consistency test for the Hex18 VAP projected ansatz."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIVE = REPO / "ed_benchmark" / "native_ed"
sys.path.insert(0, str(NATIVE))

import geometry as native_geometry

from benchmark_observables import ruby_star_formfactor
from vmc_ansatz import minimum_image_shells, one_body_seed, build_projected_netket_model
from run_benchmark import make_netket_hamiltonian, reorder_netket_vector_to_native, solve_ed
from run_vap import initial_base_state_native, make_projectable_seed
from symmetry_projection import (
    build_ordinary_space_group,
    characterize_unique_state,
    project_to_characters,
    projected_sector_residual,
)


def main():
    import jax
    jax.config.update("jax_enable_x64", True)
    import netket as nk

    g = native_geometry.RubyPBC("ring3", t=1.0, t1=0.5, t2=-0.5)
    H, _, ed_gs, _ = solve_ed(g, 6, 2.0, k=6, degeneracy_tol=1e-8)
    assert ed_gs.shape[1] == 1

    group = build_ordinary_space_group(g, H.states)
    chars = characterize_unique_state(group, ed_gs[:, 0], tol=1e-8)

    hi, _, _, graph = make_netket_hamiltonian(g, 6, 2.0)
    star = ruby_star_formfactor(g)
    _, shells, _ = minimum_image_shells(g, n_shells=4)
    print("initial seed target-sector weights:")
    raw = {}
    for seed in ("normal", "sd", "isd", "lc_same", "lc_opposite"):
        Mseed = one_body_seed(g, 6, seed, 0.08, star)
        base_seed = initial_base_state_native(Mseed, H.states, g.Ns, 6)
        _, wseed = project_to_characters(group, chars, base_seed)
        raw[seed] = float(wseed)
        print(f"  {seed:12s} w={wseed:.12g}")

    M0_raw = one_body_seed(g, 6, "lc_same", 0.08, star)
    M0, w_start, jitter = make_projectable_seed(
        M0_raw, H, group, chars, rng_seed=2718,
        min_weight=1e-8, initial_jitter=1e-4
    )
    print(f"lc_same projectable initialization: w={w_start:.12g}, jitter={jitter:.3e}")
    maps = np.asarray([op.site_map for op in group], dtype=np.int32)

    model = build_projected_netket_model(
        M0, shells, True, maps, np.asarray(chars, complex)
    )
    sampler = nk.sampler.MetropolisFermionHop(hi, graph=graph, n_chains=4)
    vstate = nk.vqs.MCState(
        sampler, model, n_samples=128, n_discard_per_chain=4,
        chunk_size=128, seed=2718, sampler_seed=3141
    )

    psi_nk = np.asarray(vstate.to_array(normalize=True))
    psi_model = reorder_netket_vector_to_native(hi, psi_nk, H.states, g.Ns)

    base = initial_base_state_native(M0, H.states, g.Ns, 6)
    psi_ref, weight = project_to_characters(group, chars, base)
    assert psi_ref is not None, "projectable-seed helper failed"
    overlap = abs(np.vdot(psi_ref, psi_model))
    residual = projected_sector_residual(group, chars, psi_model)

    assert 1.0 - overlap < 2e-8, overlap
    assert residual < 2e-8, residual
    print("VAP projected-model test: PASS")
    print("initial unprojected target-sector weight =", weight)
    print("|<independent PAV | NetKet VAP-init>| =", overlap)
    print("projected-sector residual =", residual)


if __name__ == "__main__":
    main()
