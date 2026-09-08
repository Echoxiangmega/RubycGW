import numpy as np

from rubycgw.benchmark_plot import plot_benchmark_npz


def _matrix_series(scale):
    out = np.zeros((2, 2, 2), dtype=complex)
    for iv in range(2):
        for ic in range(2):
            out[iv, ic, ic] = scale * (1.0 + 0.4 * iv + 0.1 * ic)
    return out


def test_plotter_shows_updated_post_state_chi_and_green_panels(tmp_path):
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
    names = {p.name for p in made}
    assert "benchmark_post_gw_chi_x_even.png" in names
    assert "benchmark_post_gw_chi_z_same.png" in names
    assert "benchmark_post_gw_chi_summary.png" in names
    assert "benchmark_post_gw_chi_relative_error.png" in names
    assert "benchmark_green_relative_error.png" in names
    assert "benchmark_green_lowfreq_error.png" in names
    assert "benchmark_bubble_summary.png" not in names
    assert all(p.exists() for p in made)
