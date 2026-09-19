import numpy as np

from research.track_cluster_ed_gw_vn_modes import _crossings, _track_group


def _modes(vectors, lambdas):
    n = len(lambdas)
    return {
        "vector": np.asarray(vectors, dtype=complex),
        "lambda": np.asarray(lambdas, dtype=complex),
        "co_even": np.ones(n),
        "co_odd": np.zeros(n),
        "lc_same": np.zeros(n),
        "lc_opposite": np.zeros(n),
        "uniform": np.zeros(n),
    }


def test_overlap_tracking_survives_arpack_order_swap():
    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    m0 = _modes([e0, e1], [0.8, 1.4])
    m1 = _modes([e1, e0], [1.2, 1.1])

    records = []
    _track_group(
        [(0.0, m0, np.array([0, 1])), (1.0, m1, np.array([0, 1]))],
        filling=2.0,
        q=(0, 0),
        sector="TR_even",
        min_overlap=0.5,
        records=records,
        next_branch=0,
    )

    by_branch = {}
    for r in records:
        by_branch.setdefault(r["branch_id"], []).append(r)
    assert len(by_branch) == 2

    tracks = [
        [complex(x["lambda_value"]).real for x in sorted(rr, key=lambda z: z["V"])]
        for rr in by_branch.values()
    ]
    assert [0.8, 1.1] in tracks
    assert [1.4, 1.2] in tracks


def test_crossing_uses_signed_mass_not_nearest_distance():
    records = [
        dict(
            filling=2.0, V=0.8, q1=0, q2=0, sector="TR_even", branch_id=3,
            lambda_value=1.4 + 0j, overlap_previous=np.nan,
            co_even=1.0, co_odd=0.0, lc_same=0.0, lc_opposite=0.0, uniform=0.0,
        ),
        dict(
            filling=2.0, V=1.0, q1=0, q2=0, sector="TR_even", branch_id=3,
            lambda_value=0.8 + 0j, overlap_previous=0.95,
            co_even=1.0, co_odd=0.0, lc_same=0.0, lc_opposite=0.0, uniform=0.0,
        ),
    ]
    crossings = _crossings(records, imag_tol=1e-6)
    assert len(crossings) == 1
    assert np.isclose(crossings[0]["V_cross"], 0.8 + 0.2 * (0.4 / 0.6))
    assert crossings[0]["channel_group"] == "CO"
