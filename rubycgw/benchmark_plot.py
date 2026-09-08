"""Plot helper for the ED/GW/cGW/SOX/post-GW benchmark files."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _diag(arr: np.ndarray) -> np.ndarray:
    return np.real(np.diagonal(np.asarray(arr), axis1=1, axis2=2))


def _plot_channel_summary(V, channels, series, ylabel, path):
    fig, axes = plt.subplots(1, len(channels), figsize=(5.1 * len(channels), 4.3))
    if len(channels) == 1:
        axes = [axes]
    for ic, (ax, ch) in enumerate(zip(axes, channels)):
        for label, values in series.items():
            ax.plot(V, values[:, ic], marker="o", label=label)
        ax.set_xlabel("V")
        ax.set_title(ch)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_benchmark_npz(
    npz_path: str | Path,
    *,
    outdir: str | Path | None = None,
) -> list[Path]:
    """Generate only ``benchmark_response_summary.png``.

    The summary always contains the baseline ED/GG/cGW/cGW+SOX responses.  If
    the NPZ contains ordinary post-GW and/or post-(GW+SOX) susceptibilities,
    those updated-state responses are added to the same figure.
    """
    npz_path = Path(npz_path)
    out = npz_path.parent if outdir is None else Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    V = np.asarray(data["V"], dtype=float)
    channels = [str(x) for x in data["channels"]]

    response_series = {
        "ED": _diag(data["ed"]),
        "GG": _diag(data["gg_completed"]),
        "cGW": _diag(data["cgw_completed"]),
        "cGW+SOX": _diag(data["cgw_sox_completed"]),
    }
    if "post_gw_chi_completed" in data.files:
        response_series["post-GW"] = _diag(data["post_gw_chi_completed"])
    if "post_gw_sox_chi_completed" in data.files:
        response_series["post-(GW+SOX)"] = _diag(
            data["post_gw_sox_chi_completed"]
        )

    path = out / "benchmark_response_summary.png"
    _plot_channel_summary(
        V,
        channels,
        response_series,
        r"static susceptibility $\chi$",
        path,
    )
    return [path]


__all__ = ["plot_benchmark_npz"]
