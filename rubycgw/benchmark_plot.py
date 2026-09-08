"""Plot helpers for the ED/GW/cGW/SOX/post-GW benchmark files."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _diag(arr: np.ndarray) -> np.ndarray:
    return np.real(np.diagonal(np.asarray(arr), axis1=1, axis2=2))


def plot_benchmark_npz(
    npz_path: str | Path,
    *,
    outdir: str | Path | None = None,
) -> list[Path]:
    """Generate response curves and optional post-GW Green-error curves."""
    npz_path = Path(npz_path)
    if outdir is None:
        out = npz_path.parent
    else:
        out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    V = np.asarray(data["V"], dtype=float)
    channels = [str(x) for x in data["channels"]]
    series = {
        "ED": _diag(data["ed"]),
        "GG": _diag(data["gg_completed"]),
        "cGW": _diag(data["cgw_completed"]),
        "cGW+SOX": _diag(data["cgw_sox_completed"]),
    }
    made: list[Path] = []

    for ic, ch in enumerate(channels):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for label, values in series.items():
            ax.plot(V, values[:, ic], marker="o", label=label)
        ax.set_xlabel("V")
        ax.set_ylabel(r"static susceptibility $\chi$")
        ax.set_title(ch)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out / f"benchmark_{ch}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(path)

    fig, axes = plt.subplots(1, len(channels), figsize=(5.1 * len(channels), 4.3))
    if len(channels) == 1:
        axes = [axes]
    for ic, (ax, ch) in enumerate(zip(axes, channels)):
        for label, values in series.items():
            ax.plot(V, values[:, ic], marker="o", label=label)
        ax.set_xlabel("V")
        ax.set_title(ch)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"static susceptibility $\chi$")
    axes[-1].legend()
    fig.tight_layout()
    path = out / "benchmark_response_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    # Relative response error makes the weak-coupling SOX improvement and the
    # strong-coupling over-correction visible on one common scale.
    ed = series["ED"]
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    for label in ("GG", "cGW", "cGW+SOX"):
        rel = np.mean(np.abs((series[label] - ed) / ed), axis=1)
        ax.plot(V, rel, marker="o", label=label)
    ax.set_xlabel("V")
    ax.set_ylabel("mean relative susceptibility error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out / "benchmark_response_relative_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    green_keys = [
        ("g_relerr_gw", "GW"),
        ("g_relerr_gw_sox", "GW+SOX"),
        ("g_relerr_post_gw", "post-GW"),
        ("g_relerr_post_gw_sox", "post-(GW+SOX)"),
    ]
    available = [(key, label) for key, label in green_keys if key in data.files]
    if available:
        fig, ax = plt.subplots(figsize=(6.6, 4.7))
        for key, label in available:
            values = np.asarray(data[key], dtype=float)
            if np.any(np.isfinite(values)):
                ax.plot(V, values, marker="o", label=label)
        ax.set_xlabel("V")
        ax.set_ylabel(r"$\|G-G_{ED}\|_F/\|G_{ED}\|_F$")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out / "benchmark_green_relative_error.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(path)

    low_keys = [
        ("g_lowfreq_relerr_gw", "GW"),
        ("g_lowfreq_relerr_gw_sox", "GW+SOX"),
        ("g_lowfreq_relerr_post_gw", "post-GW"),
        ("g_lowfreq_relerr_post_gw_sox", "post-(GW+SOX)"),
    ]
    available_low = [(key, label) for key, label in low_keys if key in data.files]
    if available_low:
        fig, ax = plt.subplots(figsize=(6.6, 4.7))
        for key, label in available_low:
            values = np.asarray(data[key], dtype=float)
            if np.any(np.isfinite(values)):
                ax.plot(V, values, marker="o", label=label)
        ax.set_xlabel("V")
        ax.set_ylabel("low-frequency relative Green-function error")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out / "benchmark_green_lowfreq_error.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(path)

    return made


__all__ = ["plot_benchmark_npz"]
