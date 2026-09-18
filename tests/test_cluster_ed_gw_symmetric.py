import numpy as np

from rubycgw.c3_constraint import project_lattice_c3
from rubycgw.cluster_ed_gw_symmetric import (
    _local_to_common_lattice,
    _pack_three_dynamic,
    _unpack_three_dynamic,
    project_fermion_time_reversal,
    project_local_time_reversal,
)
from rubycgw.cluster_orientation import rotate_local_between_orientations
from rubycgw.grids import MatsubaraGrid
from rubycgw.models.ruby import (
    ExtendedRubyParameters,
    physical_pair_cluster_interactions,
)


def test_fermion_time_reversal_projection():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(6, 3, 3, 6, 6)) + 1j * rng.normal(
        size=(6, 3, 3, 6, 6)
    )
    y = project_fermion_time_reversal(x)
    neg = np.asarray([0, 2, 1], dtype=int)
    partner = np.conj(y[::-1][:, neg][:, :, neg])
    assert np.max(np.abs(y - partner)) < 1e-12


def test_local_time_reversal_projection():
    rng = np.random.default_rng(456)
    x = rng.normal(size=(8, 6, 6)) + 1j * rng.normal(size=(8, 6, 6))
    y = project_local_time_reversal(x)
    assert np.max(np.abs(y - np.conj(y[::-1]))) < 1e-12


def test_three_dynamic_pack_roundtrip():
    rng = np.random.default_rng(789)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.1)
    emb = rng.normal(size=(grid.nf, 2, 2, 6, 6)) + 1j * rng.normal(
        size=(grid.nf, 2, 2, 6, 6)
    )
    imp = rng.normal(size=(3, grid.nf, 6, 6)) + 1j * rng.normal(
        size=(3, grid.nf, 6, 6)
    )
    packed = _pack_three_dynamic(emb, imp, grid)
    emb2, imp2 = _unpack_three_dynamic(
        packed, emb.shape, imp.shape, grid
    )
    assert np.max(np.abs(emb2 - emb)) < 1e-12
    assert np.max(np.abs(imp2 - imp)) < 1e-12


def test_three_orientation_average_equals_c3_projection_for_exact_orbit():
    rng = np.random.default_rng(2468)
    grid = MatsubaraGrid(nk1=3, nk2=3, nw=2, nOmega=1, T=0.1)
    local0 = rng.normal(size=(grid.nf, 6, 6)) + 1j * rng.normal(
        size=(grid.nf, 6, 6)
    )

    common = []
    for r in range(3):
        local_r = rotate_local_between_orientations(
            local0, 0, r, nk1=grid.nk1, nk2=grid.nk2
        )
        common.append(_local_to_common_lattice(local_r, r, grid))

    avg = sum(common) / 3.0
    base = np.broadcast_to(
        local0[:, None, None, :, :],
        (grid.nf, grid.nk1, grid.nk2, 6, 6),
    ).copy()
    expected = project_lattice_c3(base)
    assert np.max(np.abs(avg - expected)) < 1e-11


def test_each_physical_pair_has_one_full_v_cluster_and_one_real_pair():
    p = ExtendedRubyParameters(
        V=1.8, Vprime=-0.1, Vcross=-0.07
    )
    for r in range(3):
        terms = physical_pair_cluster_interactions(p, r)
        intra = [(i, j, u) for i, j, u in terms if (i < 3) == (j < 3)]
        inter = [(i, j, u) for i, j, u in terms if (i < 3) != (j < 3)]
        assert len(intra) == 6
        assert len(inter) == 4
        assert sum(np.isclose(u, p.Vprime) for _, _, u in inter) == 2
        assert sum(np.isclose(u, p.Vcross) for _, _, u in inter) == 2
