import numpy as np

from rubycgw.branch_state_archive import (
    BranchStateArchive,
    append_state_in_memory,
    find_target_crossings,
    load_branch_state_archive,
    save_branch_state_archive,
    states_match,
)


def test_branch_state_archive_roundtrip_and_append(tmp_path):
    signature = '{"mode":"weighted_nbj","lambda":0.5}'
    x0 = np.array([1.0, 2.0, 3.0])
    x1 = np.array([1.1, 2.1, 3.1])
    archive = BranchStateArchive(
        signature,
        np.array([0, 1], dtype=np.int64),
        np.array(["seed", "seed"]),
        np.array([0.05, 0.10]),
        np.stack([x0, x1]),
    )
    archive = append_state_in_memory(
        archive, 2, "pac", 0.13, np.array([1.2, 2.2, 3.2])
    )
    path = tmp_path / "branch_states.npz"
    save_branch_state_archive(
        path, archive.signature, archive.step, archive.kind, archive.V, archive.X
    )
    got = load_branch_state_archive(
        path, expected_signature=signature, expected_state_size=3
    )
    np.testing.assert_array_equal(got.step, [0, 1, 2])
    np.testing.assert_array_equal(got.kind, ["seed", "seed", "pac"])
    np.testing.assert_allclose(got.V, [0.05, 0.10, 0.13])
    np.testing.assert_allclose(got.X, archive.X)
    assert states_match(got.X[-1], got.V[-1], archive.X[-1], archive.V[-1])


def test_branch_state_archive_rejects_wrong_signature(tmp_path):
    path = tmp_path / "branch_states.npz"
    save_branch_state_archive(
        path, "sig-a", [0], ["seed"], [0.05], np.zeros((1, 4))
    )
    try:
        load_branch_state_archive(path, expected_signature="sig-b")
    except RuntimeError:
        pass
    else:
        raise AssertionError("signature mismatch was accepted")


def test_find_target_crossings_finds_every_fold_crossing():
    # The branch crosses V=1 three times as it folds back and forward.
    V = np.array([0.7, 1.2, 0.8, 1.3])
    X = np.column_stack([V, np.arange(V.size, dtype=float)])
    archive = BranchStateArchive(
        "sig",
        np.arange(V.size, dtype=np.int64),
        np.array(["pac"] * V.size),
        V,
        X,
    )
    crossings = find_target_crossings(archive, 1.0)
    assert len(crossings) == 3
    assert [(c.left_index, c.right_index) for c in crossings] == [(0, 1), (1, 2), (2, 3)]
    for c in crossings:
        assert abs(c.x_guess[0] - 1.0) < 1e-14
        assert 0.0 < c.alpha < 1.0


def test_find_target_crossings_does_not_double_count_exact_hit():
    V = np.array([0.7, 1.0, 1.3, 0.8])
    X = np.column_stack([V, np.arange(V.size, dtype=float)])
    archive = BranchStateArchive(
        "sig",
        np.arange(V.size, dtype=np.int64),
        np.array(["pac"] * V.size),
        V,
        X,
    )
    crossings = find_target_crossings(archive, 1.0)
    # One exact archived root at index 1, plus the later 1.3 -> 0.8 crossing.
    assert len(crossings) == 2
    assert crossings[0].left_index == crossings[0].right_index == 1
    assert (crossings[1].left_index, crossings[1].right_index) == (2, 3)
