from types import SimpleNamespace

from decompose_ed18_cgw_stages import _canonical_stages, _stage_options


def _args():
    return SimpleNamespace(
        vertex_max_iter=150,
        vertex_tol=1e-8,
        vertex_mixing=0.25,
        vertex_solver="gmres",
        vertex_gmres_restart=12,
        vertex_verbose=False,
        momentum_backend="fft",
    )


def test_all_stage_order_is_canonical():
    assert _canonical_stages(["all"]) == ["gg", "hf", "split-mt", "full"]
    assert _canonical_stages(["full", "hf"]) == ["gg", "hf", "full"]


def test_stage_kernel_flags():
    args = _args()
    hf = _stage_options("hf", args)
    mt = _stage_options("split-mt", args)
    full = _stage_options("full", args)

    assert hf.include_hartree and hf.include_fock
    assert not hf.include_mt and not hf.include_al
    assert mt.include_hartree and mt.include_fock and mt.include_mt
    assert not mt.include_al
    assert full.include_hartree and full.include_fock and full.include_mt and full.include_al
