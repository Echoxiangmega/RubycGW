import numpy as np

from rubycgw.benchmark_plot import plot_benchmark_npz


def _matrix_series(scale):
    out = np.zeros((2, 2, 2), dtype=complex)
    for iv in range(2):
        for ic in range(2):
            out[iv, ic, ic] = scale * (1.0 + 0.4 * iv + 0.1 * ic)
    return out


def test_plotter_writes_only_response_summary_with_post_gw(tmp_path):
    path = tmp_path / "benchmark.npz"
    np.savez_compressed(
        path,
        V=np.array([0.1, 0.2]),
        channels=np.array(["x_even", "z_same"]),
        ed=_matrix_series(1.00).real,
        gg_completed=_matrix_series(0.90),
        cgw_completed=_matrix_series(1.08),
        cgw_sox_completed=_matrix_series(1.02),
        post_gw_chi_completed=_matrix_series(1.01),
        g_relerr_gw=np.array([0.2, 0.25]),
        g_relerr_post_gw=np.array([0.12, 0.15]),
        g_lowfreq_relerr_gw=np.array([0.3, 0.35]),
        g_lowfreq_relerr_post_gw=np.array([0.18, 0.20]),
    )
    made = plot_benchmark_npz(path)
    assert [p.name for p in made] == ["benchmark_response_summary.png"]
    assert made[0].exists()
