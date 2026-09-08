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
    """Generate production-response, post-response, bubble, and Green plots."""
    npz_path = Path(npz_path)
    out = npz_path.parent if outdir is None else Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    V = np.asarray(data["V"], dtype=float)
    channels = [str(x) for x in data["channels"]]
    made: list[Path] = []

    # ------------------------------------------------------------------
    # Production/static response: keep the original comparison uncluttered.
    # ------------------------------------------------------------------
    response_series = {
        "ED": _diag(data["ed"]),
        "GG[GW]": _diag(data["gg_completed"]),
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
    for label in ("GG[GW]", "cGW", "cGW+SOX"):
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
    # Bubble-only comparison: isolates how much the one-particle G changes chi.
    # ED is shown only as the exact full-response reference, not as an ED bubble.
    # ------------------------------------------------------------------
    bubble_series = {
        "ED exact chi": _diag(data["ed"]),
        "GG[GW]": _diag(data["gg_completed"]),
    }
    if "gg_gw_sox_completed" in data.files:
        bubble_series["GG[GW+SOX]"] = _diag(data["gg_gw_sox_completed"])
    if "gg_post_gw_completed" in data.files:
        bubble_series["GG[post-GW]"] = _diag(data["gg_post_gw_completed"])
    if "gg_post_gw_sox_completed" in data.files:
        bubble_series["GG[post-(GW+SOX)]"] = _diag(data["gg_post_gw_sox_completed"])

    path = out / "benchmark_bubble_summary.png"
    _plot_channel_summary(
        V,
        channels,
        bubble_series,
        r"bubble susceptibility $\chi^{GG}$",
        path,
    )
    made.append(path)

    # ------------------------------------------------------------------
    # Ordinary post-GW response diagnostics.  Show separately from production
    # response because cGW on the post background is not the strict derivative
    # of the complete one-shot post map.
    # ------------------------------------------------------------------
    if "cgw_wpost_completed" in data.files or "cgw_post_gw_completed" in data.files:
        post_series = {
            "ED": _diag(data["ed"]),
            "cGW[GW,W_GW]": _diag(data["cgw_completed"]),
        }
        if "cgw_wpost_completed" in data.files:
            post_series["cGW[G_GW,W_post]"] = _diag(data["cgw_wpost_completed"])
        if "cgw_post_gw_completed" in data.files:
            post_series["cGW[G_post,W_post]"] = _diag(data["cgw_post_gw_completed"])
        for ic, ch in enumerate(channels):
            fig, ax = plt.subplots(figsize=(6.6, 4.7))
            for label, values in post_series.items():
                ax.plot(V, values[:, ic], marker="o", label=label)
            ax.set_xlabel("V")
            ax.set_ylabel(r"static susceptibility $\chi$")
            ax.set_title(f"post-GW diagnostic: {ch}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = out / f"benchmark_post_gw_response_{ch}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            made.append(path)
        path = out / "benchmark_post_gw_response_summary.png"
        _plot_channel_summary(
            V,
            channels,
            post_series,
            r"static susceptibility $\chi$",
            path,
        )
        made.append(path)

    # ------------------------------------------------------------------
    # Post-(GW+SOX) response diagnostics, independently selected.
    # ------------------------------------------------------------------
    if "cgw_sox_wpost_completed" in data.files or "cgw_sox_post_completed" in data.files:
        post_sox_series = {
            "ED": _diag(data["ed"]),
            "cGW+SOX[bg]": _diag(data["cgw_sox_completed"]),
        }
        if "cgw_sox_wpost_completed" in data.files:
            post_sox_series["cGW+SOX[G_bg,W_post]"] = _diag(
                data["cgw_sox_wpost_completed"]
            )
        if "cgw_sox_post_completed" in data.files:
            post_sox_series["cGW+SOX[G_post,W_post]"] = _diag(
                data["cgw_sox_post_completed"]
            )
        for ic, ch in enumerate(channels):
            fig, ax = plt.subplots(figsize=(6.6, 4.7))
            for label, values in post_sox_series.items():
                ax.plot(V, values[:, ic], marker="o", label=label)
            ax.set_xlabel("V")
            ax.set_ylabel(r"static susceptibility $\chi$")
            ax.set_title(f"post-(GW+SOX) diagnostic: {ch}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = out / f"benchmark_post_gw_sox_response_{ch}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            made.append(path)
        path = out / "benchmark_post_gw_sox_response_summary.png"
        _plot_channel_summary(
            V,
            channels,
            post_sox_series,
            r"static susceptibility $\chi$",
            path,
        )
        made.append(path)

    # ------------------------------------------------------------------
    # Green-function error plots are separate from all chi diagnostics.
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
