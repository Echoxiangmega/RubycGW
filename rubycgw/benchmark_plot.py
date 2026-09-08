"""Plot helpers for the ED/GW/cGW/SOX/post-GW benchmark files."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _diag(arr: np.ndarray) -> np.ndarray:
    return np.real(np.diagonal(np.asarray(arr), axis1=1, axis2=2))


def _finite_series(data, specs):
    out = []
    for key, label in specs:
        if key not in data.files:
            continue
        values = np.asarray(data[key])
        if np.any(np.isfinite(values)):
            out.append((key, label, values))
    return out


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
    """Generate baseline chi, selected post-chi, and Green-error plots."""
    npz_path = Path(npz_path)
    out = npz_path.parent if outdir is None else Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    V = np.asarray(data["V"], dtype=float)
    channels = [str(x) for x in data["channels"]]
    made: list[Path] = []

    # ------------------------------------------------------------------
    # Original production/static response benchmark.
    # ------------------------------------------------------------------
    response_series = {
        "ED": _diag(data["ed"]),
        "GG": _diag(data["gg_completed"]),
        "cGW": _diag(data["cgw_completed"]),
        "cGW+SOX": _diag(data["cgw_sox_completed"]),
    }
    for ic, ch in enumerate(channels):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for label, values in response_series.items():
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

    path = out / "benchmark_response_summary.png"
    _plot_channel_summary(
        V,
        channels,
        response_series,
        r"static susceptibility $\chi$",
        path,
    )
    made.append(path)

    ed = response_series["ED"]
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    for label in ("GG", "cGW", "cGW+SOX"):
        rel = np.mean(np.abs((response_series[label] - ed) / ed), axis=1)
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

    # ------------------------------------------------------------------
    # Ordinary post-GW: compare the original cGW susceptibility with chi
    # recomputed on the updated state (G_post,W_post).  No GG[post] and no
    # fixed-G / W_post-only diagnostic are plotted.
    # ------------------------------------------------------------------
    if "post_gw_chi_completed" in data.files:
        post_series = {
            "ED": _diag(data["ed"]),
            "cGW": _diag(data["cgw_completed"]),
            "post-GW chi": _diag(data["post_gw_chi_completed"]),
        }
        for ic, ch in enumerate(channels):
            fig, ax = plt.subplots(figsize=(6.5, 4.7))
            for label, values in post_series.items():
                ax.plot(V, values[:, ic], marker="o", label=label)
            ax.set_xlabel("V")
            ax.set_ylabel(r"static susceptibility $\chi$")
            ax.set_title(f"post-GW: {ch}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = out / f"benchmark_post_gw_chi_{ch}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            made.append(path)
        path = out / "benchmark_post_gw_chi_summary.png"
        _plot_channel_summary(
            V,
            channels,
            post_series,
            r"static susceptibility $\chi$",
            path,
        )
        made.append(path)

        fig, ax = plt.subplots(figsize=(6.6, 4.7))
        for label in ("cGW", "post-GW chi"):
            rel = np.mean(np.abs((post_series[label] - ed) / ed), axis=1)
            ax.plot(V, rel, marker="o", label=label)
        ax.set_xlabel("V")
        ax.set_ylabel("mean relative susceptibility error")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out / "benchmark_post_gw_chi_relative_error.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(path)

    # ------------------------------------------------------------------
    # Post-(GW+SOX), independently selected.
    # ------------------------------------------------------------------
    if "post_gw_sox_chi_completed" in data.files:
        post_sox_series = {
            "ED": _diag(data["ed"]),
            "cGW+SOX": _diag(data["cgw_sox_completed"]),
            "post-(GW+SOX) chi": _diag(data["post_gw_sox_chi_completed"]),
        }
        for ic, ch in enumerate(channels):
            fig, ax = plt.subplots(figsize=(6.5, 4.7))
            for label, values in post_sox_series.items():
                ax.plot(V, values[:, ic], marker="o", label=label)
            ax.set_xlabel("V")
            ax.set_ylabel(r"static susceptibility $\chi$")
            ax.set_title(f"post-(GW+SOX): {ch}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = out / f"benchmark_post_gw_sox_chi_{ch}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            made.append(path)
        path = out / "benchmark_post_gw_sox_chi_summary.png"
        _plot_channel_summary(
            V,
            channels,
            post_sox_series,
            r"static susceptibility $\chi$",
            path,
        )
        made.append(path)

    # ------------------------------------------------------------------
    # Green-function information is kept in separate figures.
    # ------------------------------------------------------------------
    green_specs = [
        ("g_relerr_gw", "GW"),
        ("g_relerr_post_gw", "post-GW"),
        ("g_relerr_gw_sox", "GW+SOX"),
        ("g_relerr_post_gw_sox", "post-(GW+SOX)"),
    ]
    available = _finite_series(data, green_specs)
    if available:
        fig, ax = plt.subplots(figsize=(6.6, 4.7))
        for _, label, values in available:
            ax.plot(V, np.asarray(values, dtype=float), marker="o", label=label)
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

    low_specs = [
        ("g_lowfreq_relerr_gw", "GW"),
        ("g_lowfreq_relerr_post_gw", "post-GW"),
        ("g_lowfreq_relerr_gw_sox", "GW+SOX"),
        ("g_lowfreq_relerr_post_gw_sox", "post-(GW+SOX)"),
    ]
    available_low = _finite_series(data, low_specs)
    if available_low:
        fig, ax = plt.subplots(figsize=(6.6, 4.7))
        for _, label, values in available_low:
            ax.plot(V, np.asarray(values, dtype=float), marker="o", label=label)
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
