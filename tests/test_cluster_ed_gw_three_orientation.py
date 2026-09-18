import numpy as np

from rubycgw.c3_constraint import c3_lattice_residual
from rubycgw.cluster_ed_gw_three_orientation import (
    _average_local_fields_in_common,
    _c3_local_orbit,
    _pack_representative_dynamic,
    _pack_three_dynamic,
    _unpack_representative_dynamic,
    _unpack_three_dynamic,
)
from rubycgw.cluster_orientation import rotate_local_between_orientations
from rubycgw.grids import MatsubaraGrid


def test_three_orientation_dynamic_pack_roundtrip():
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.1)
    rng = np.random.default_rng(12345)
    emb = (
        rng.normal(size=(grid.nf, 2, 2, 6, 6))
        + 1j * rng.normal(size=(grid.nf, 2, 2, 6, 6))
    )
    imps = [
        rng.normal(size=(grid.nf, 6, 6))
        + 1j * rng.normal(size=(grid.nf, 6, 6))
        for _ in range(3)
    ]
    packed = _pack_three_dynamic(emb, imps, grid)
    emb2, imps2 = _unpack_three_dynamic(
        packed, emb.shape, imps[0].shape, grid
    )
    assert np.max(np.abs(emb2 - emb)) < 1e-12
    for a, b in zip(imps2, imps):
        assert np.max(np.abs(a - b)) < 1e-12


def test_average_of_c3_related_local_corrections_is_c3_invariant():
    grid = MatsubaraGrid(nk1=3, nk2=3, nw=2, nOmega=1, T=0.1)
    rng = np.random.default_rng(54321)
    x0 = (
        rng.normal(size=(grid.nf, 6, 6))
        + 1j * rng.normal(size=(grid.nf, 6, 6))
    )
    xs = [
        rotate_local_between_orientations(
            x0, 0, r, nk1=grid.nk1, nk2=grid.nk2
        )
        for r in (0, 1, 2)
    ]
    avg = _average_local_fields_in_common(xs, grid)
    assert c3_lattice_residual(avg) < 1e-11


def test_representative_dynamic_pack_roundtrip():
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.1)
    rng = np.random.default_rng(24680)
    imp0 = (
        rng.normal(size=(grid.nf, 6, 6))
        + 1j * rng.normal(size=(grid.nf, 6, 6))
    )
    # Build a C3-symmetric embedded field from an arbitrary weak piece plus
    # the exact local C3 orbit of one representative impurity.
    weak = (
        rng.normal(size=(grid.nf, 2, 2, 6, 6))
        + 1j * rng.normal(size=(grid.nf, 2, 2, 6, 6))
    )
    orbit = _c3_local_orbit(imp0, grid)
    from rubycgw.cluster_ed_gw_three_orientation import _average_local_fields_in_common
    emb = weak + _average_local_fields_in_common(orbit, grid)
    packed = _pack_representative_dynamic(emb, imp0, grid)
    emb2, imp02 = _unpack_representative_dynamic(
        packed, emb.shape, imp0.shape, grid
    )
    assert np.max(np.abs(emb2 - emb)) < 1e-12
    assert np.max(np.abs(imp02 - imp0)) < 1e-12


def test_local_c3_orbit_closes_after_three_orientations():
    grid = MatsubaraGrid(nk1=3, nk2=3, nw=2, nOmega=1, T=0.1)
    rng = np.random.default_rng(13579)
    x0 = (
        rng.normal(size=(grid.nf, 6, 6))
        + 1j * rng.normal(size=(grid.nf, 6, 6))
    )
    xs = _c3_local_orbit(x0, grid)
    assert len(xs) == 3
    # Mapping orientation 2 once more to orientation 0 must recover the
    # representative exactly.
    from rubycgw.cluster_orientation import rotate_local_between_orientations
    xback = rotate_local_between_orientations(
        xs[2], 2, 0, nk1=grid.nk1, nk2=grid.nk2
    )
    assert np.max(np.abs(xback - x0)) < 1e-11
