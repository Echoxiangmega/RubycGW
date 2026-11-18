#!/usr/bin/env python3
"""Regression tests for the symmetry-tied one-body parameterization."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIVE = REPO / "ed_benchmark" / "native_ed"
sys.path.insert(0, str(NATIVE))

import geometry
from benchmark_observables import ruby_star_formfactor
from vmc_ansatz import one_body_seed_hamiltonian, one_body_seed, minimum_image_shells
from parametric_ansatz import (
    build_symmetry_tied_basis,
    seed_tied_parameters,
    reconstruct_one_body,
    build_orbital_rotation_data,
    build_stable_parametric_netket_model,
)


def transform_h(h, p):
    p = np.asarray(p, int)
    return h[np.ix_(p, p)]


def main():
    g = geometry.RubyPBC("ring3", t=1.0, t1=0.5, t2=-0.5)
    basis = build_symmetry_tied_basis(g)
    print("site_orbits =", basis.site_orbits)
    print("bond_orbits =", basis.bond_orbits)
    print("n_onsite =", basis.n_onsite)
    print("n_complex_bond =", basis.n_bond)
    print("n_one_body_parameters =", basis.n_one_body_parameters)

    # Every basis matrix must be invariant under translations x C3.
    mats = [basis.h0]
    mats += [m.astype(complex) for m in basis.onsite_matrices]
    mats += [m.astype(complex) for m in basis.bond_real_matrices]
    mats += [m.astype(complex) for m in basis.bond_imag_matrices]
    for im, h in enumerate(mats):
        for p in basis.subgroup_maps:
            err = np.max(np.abs(transform_h(h, p) - h))
            assert err < 1e-10, (im, err)

    # The imaginary bond basis must genuinely allow TR-breaking hopping.
    assert basis.n_bond > 0
    assert max(np.max(np.abs(np.imag(m))) for m in basis.bond_imag_matrices) > 0.5

    # Physical seeds can be projected into the tied parameter manifold.
    star = ruby_star_formfactor(g)
    for seed in ("normal", "sd", "isd", "lc_same", "lc_opposite"):
        hseed = one_body_seed_hamiltonian(g, seed, 0.08, star)
        p = seed_tied_parameters(basis, hseed)
        h = reconstruct_one_body(basis, p)
        assert np.max(np.abs(h-h.conj().T)) < 1e-12
        for q in basis.subgroup_maps:
            assert np.max(np.abs(transform_h(h, q)-h)) < 1e-10

    # Instantiate the stable reduced NetKet model so scope/shape errors are
    # caught before a long VMC job.
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        star = ruby_star_formfactor(g)
        Q0 = one_body_seed(g, 3, "normal", 0.08, star)
        rot = build_orbital_rotation_data(basis, Q0)
        _, shells, _ = minimum_image_shells(g, n_shells=4)
        model = build_stable_parametric_netket_model(
            rot["Q0"], rot["tangent_basis"], shells, use_jastrow=True,
            initial_theta=np.zeros(rot["tangent_rank"])
        )
        x = np.zeros((1, g.Ns), dtype=np.int8)
        x[0, :3] = 1
        variables = model.init(jax.random.PRNGKey(0), jnp.asarray(x))
        y = np.asarray(model.apply(variables, jnp.asarray(x)))
        assert np.all(np.isfinite(y)), y
    except ImportError:
        print("NetKet/JAX unavailable: skipped stable-model instantiation check")

    print("symmetry-tied parameterization tests: PASS")


if __name__ == "__main__":
    main()
