"""Plots for zero-temperature 18-site ED static susceptibilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .ed18_plot import (
    _add_transition_lines,
    _channel_labels,
    leading_subspace_weights,
    load_ed18_scan,
    transition_midpoints,
)


def _require_chi(data: dict[str, np.ndarray]) -> None:
    needed = (
        "chi_regular_matrix",
        "chi_regular_eigenvalues",
        "chi_regular_eigenvectors",
        "chi_ground_singular_matrix",
        "chi_ground_singular_eigenvalues",
        "chi_ground_singular_eigenvectors",
        "chi_is_singular",
    )
    missing = [k for k in needed if k not in data]
    if missing:
        raise KeyError(
            "scan file has no zero-temperature susceptibility data; rerun "
            "scan_ed18_vs_V.py with --with-chi. Missing: " + ", ".join(missing)
        )


def save_ed18_chi_plot(
    data_or_path: dict[str, np.ndarray] | str | Path,
    *,
    chi_path: str | Path,
    top_modes: int = 3,
    dpi: int = 180,
) -> Path:
    """Plot regular T=0 chi and the exact-GS singular coefficient at Gamma/Q."""
    if isinstance(data_or_path, (str, Path)):
        data = load_ed18_scan(data_or_path)
    else:
        data = {k: np.asarray(v) for k, v in data_or_path.items()}
    _require_chi(data)

    V = np.asarray(data["V"], dtype=float)
    channels = np.asarray(data["channels"]).astype(str)
    q_names = np.asarray(data["q_names"]).astype(str)
    reg = np.asarray(data["chi_regular_matrix"], dtype=complex)
    revals = np.asarray(data["chi_regular_eigenvalues"], dtype=float)
    revecs = np.asarray(data["chi_regular_eigenvectors"], dtype=complex)
    sing = np.asarray(data["chi_ground_singular_matrix"], dtype=complex)
    sevals = np.asarray(data["chi_ground_singular_eigenvalues"], dtype=float)
    sevecs = np.asarray(data["chi_ground_singular_eigenvectors"], dtype=complex)
    is_singular = np.asarray(data["chi_is_singular"], dtype=bool)

    reg_diag = np.real(np.diagonal(reg, axis1=-2, axis2=-1))
    reg_weights, reg_dim = leading_subspace_weights(revals, revecs)
    sing_weights, sing_dim = leading_subspace_weights(sevals, sevecs)
    mids = transition_midpoints(data)
    labels = _channel_labels(channels)

    iq_gamma = int(np.where(q_names == "Gamma")[0][0])
    iq_q = int(np.where(q_names == "Q")[0][0])
    q_indices = [iq_gamma, iq_q]
    q_titles = [r"$\Gamma$", r"$Q=(1/3,1/3)$"]
    nshow = max(1, min(int(top_modes), revals.shape[-1]))

    fig, axes = plt.subplots(5, 2, figsize=(14.5, 17.0), sharex="col")
    for col, (iq, qtitle) in enumerate(zip(q_indices, q_titles)):
        for m in range(nshow):
            axes[0, col].plot(V, revals[:, iq, m], marker="o", ms=2.5, label=rf"$\chi^{{reg}}_{m+1}$")
        axes[0, col].set_title(qtitle)
        axes[0, col].set_ylabel(r"regular $T=0$ $\chi$ eigenvalues")
        axes[0, col].legend(fontsize=8)

        for ic, label in enumerate(labels):
            axes[1, col].plot(V, reg_diag[:, iq, ic], label=label)
        axes[1, col].set_ylabel(r"diagonal $\chi^{reg}_{\mu\mu}$")
        axes[1, col].legend(ncol=2, fontsize=8)

        for ic, label in enumerate(labels):
            axes[2, col].plot(V, reg_weights[:, iq, ic], label=label)
        axes[2, col].set_ylabel("leading regular-chi channel weight")
        axes[2, col].set_ylim(-0.03, 1.03)
        axes[2, col].legend(ncol=2, fontsize=8)
        dvals = sorted(set(int(x) for x in reg_dim[:, iq]))
        axes[2, col].set_title("regular leading eigenspace d=" + ",".join(map(str, dvals)), fontsize=9)

        for m in range(nshow):
            axes[3, col].plot(V, sevals[:, iq, m], marker="o", ms=2.5, label=rf"$c^{{GS}}_{m+1}$")
        axes[3, col].set_ylabel(r"GS singular coefficient $C^{GS}$")
        axes[3, col].legend(fontsize=8)
        # Mark V points where strict T=0 susceptibility is nonanalytic/divergent.
        bad = is_singular[:, iq]
        if np.any(bad):
            ytop = max(float(np.nanmax(sevals[:, iq, 0])), 1e-12)
            axes[3, col].scatter(V[bad], np.full(np.count_nonzero(bad), ytop), marker="x", label="T=0 singular")

        for ic, label in enumerate(labels):
            axes[4, col].plot(V, sing_weights[:, iq, ic], label=label)
        axes[4, col].set_ylabel("leading singular-mode channel weight")
        axes[4, col].set_xlabel(r"$V$")
        axes[4, col].set_ylim(-0.03, 1.03)
        axes[4, col].legend(ncol=2, fontsize=8)
        sdvals = sorted(set(int(x) for x in sing_dim[:, iq]))
        axes[4, col].set_title("singular leading eigenspace d=" + ",".join(map(str, sdvals)), fontsize=9)

        for row in range(5):
            _add_transition_lines(axes[row, col], mids)
            axes[row, col].grid(alpha=0.2)

    fig.suptitle(
        "18-site Ruby ED: exact fixed-N zero-temperature static susceptibility\n"
        r"nonzero $C^{GS}$ means the strict $T=0$ response is singular; $\chi^{reg}$ is the finite excited-state part",
        y=0.995,
    )
    fig.tight_layout()
    chi_path = Path(chi_path)
    chi_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chi_path, dpi=int(dpi))
    plt.close(fig)
    return chi_path
