"""Plotting utilities for 18-site Ruby exact-diagonalization V scans.

The scan stores the complete six-channel equal-time pseudospin structure
matrix at Gamma and Q=(1/3,1/3).  These helpers turn that information into
plots that distinguish:

* the leading structure-factor eigenvalue S_max(q);
* the individual diagonal channel correlations S_{mu,mu}(q);
* the leading eigenmode composition |v_mu(q)|^2;
* the first few structure-factor eigenvalues, which reveal near-degeneracies;
* the low-energy many-body spectrum and ground-manifold changes.

No ED rerun is required: an existing ``scan_ed18_vs_V.py`` NPZ file contains
all information needed by :func:`save_ed18_plots`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .ed18 import translation_phase_reduced


_CHANNEL_TEX = {
    "x_even": r"$x_{\rm even}$",
    "x_odd": r"$x_{\rm odd}$",
    "y_even": r"$y_{\rm even}$",
    "y_odd": r"$y_{\rm odd}$",
    "z_same": r"$z_{\rm same}$",
    "z_opposite": r"$z_{\rm opposite}$",
}


def load_ed18_scan(path: str | Path) -> dict[str, np.ndarray]:
    """Load one ED18 scan NPZ into an ordinary dictionary."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _channel_labels(channels: np.ndarray) -> list[str]:
    return [_CHANNEL_TEX.get(str(ch), str(ch)) for ch in channels]


def ground_sector_signature(
    translation_eigenvalues: np.ndarray,
    *,
    tol: float = 1e-8,
) -> tuple[float, ...]:
    """Return the sorted reduced primitive-translation phases of the GS manifold."""
    vals = np.asarray(translation_eigenvalues, dtype=complex).reshape(-1)
    good = vals[np.isfinite(vals.real) & np.isfinite(vals.imag) & (np.abs(vals) > tol)]
    phases = sorted(round(translation_phase_reduced(z), 6) for z in good)
    return tuple(phases)


def ground_sector_label(translation_eigenvalues: np.ndarray) -> str:
    """Human-readable label for the common 18-site Gamma / +/-Q sectors."""
    sig = ground_sector_signature(translation_eigenvalues)
    if sig == (0.0,):
        return r"$\Gamma$"
    if len(sig) == 2 and np.allclose(sig, (-1.0 / 3.0, 1.0 / 3.0), atol=2e-5):
        return r"$\pm Q$"
    if sig == (round(1.0 / 3.0, 6),):
        return r"$+Q$"
    if sig == (round(-1.0 / 3.0, 6),):
        return r"$-Q$"
    if not sig:
        return "?"
    return ",".join(f"{x:+.3f}" for x in sig)


def transition_midpoints(data: dict[str, np.ndarray]) -> np.ndarray:
    """Locate intervals where the finite-size ground-manifold signature changes."""
    V = np.asarray(data["V"], dtype=float)
    deg = np.asarray(data["ground_multiplicity"], dtype=int)
    trans = np.asarray(data["translation_eigenvalues"], dtype=complex)
    sigs = [
        (int(deg[i]), ground_sector_signature(trans[i]))
        for i in range(len(V))
    ]
    mids = []
    for i in range(len(V) - 1):
        if sigs[i + 1] != sigs[i]:
            mids.append(0.5 * (V[i] + V[i + 1]))
    return np.asarray(mids, dtype=float)


def _add_transition_lines(ax, mids: np.ndarray) -> None:
    for x in np.asarray(mids, dtype=float):
        ax.axvline(x, linestyle=":", linewidth=1.0, alpha=0.65)


def _validate(data: dict[str, np.ndarray]) -> None:
    needed = (
        "V",
        "energies",
        "ground_multiplicity",
        "gap_above_manifold",
        "interaction_expectation",
        "translation_eigenvalues",
        "channels",
        "q_names",
        "structure_matrix",
        "structure_eigenvalues",
        "structure_eigenvectors",
    )
    missing = [key for key in needed if key not in data]
    if missing:
        raise KeyError("ED18 scan file is missing required arrays: " + ", ".join(missing))


def derived_mode_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return convenient channel-resolved arrays derived from the saved matrices."""
    _validate(data)
    S = np.asarray(data["structure_matrix"], dtype=complex)
    evecs = np.asarray(data["structure_eigenvectors"], dtype=complex)
    diag = np.real(np.diagonal(S, axis1=-2, axis2=-1))
    leading_weights = np.abs(evecs[..., 0]) ** 2
    leading_weights /= np.maximum(
        np.sum(leading_weights, axis=-1, keepdims=True), 1e-300
    )
    return {
        "structure_diagonal": np.asarray(diag, dtype=float),
        "leading_mode_weights": np.asarray(leading_weights, dtype=float),
    }


def save_ed18_plots(
    data_or_path: dict[str, np.ndarray] | str | Path,
    *,
    summary_path: str | Path,
    modes_path: str | Path,
    top_spectrum: int = 6,
    top_structure: int = 3,
    dpi: int = 180,
) -> tuple[Path, Path]:
    """Create a summary figure and a channel/eigenmode figure.

    ``summary_path`` contains the many-body energy, low-energy spectrum,
    gap/multiplicity, interaction expectation, and S_max(Gamma/Q).

    ``modes_path`` resolves the structure factor into the six pseudospin
    channels and shows the leading eigenvector weights.
    """
    if isinstance(data_or_path, (str, Path)):
        data = load_ed18_scan(data_or_path)
    else:
        data = {k: np.asarray(v) for k, v in data_or_path.items()}
    _validate(data)

    V = np.asarray(data["V"], dtype=float)
    energies = np.asarray(data["energies"], dtype=float)
    deg = np.asarray(data["ground_multiplicity"], dtype=int)
    gaps = np.asarray(data["gap_above_manifold"], dtype=float)
    dint = np.asarray(data["interaction_expectation"], dtype=float)
    trans = np.asarray(data["translation_eigenvalues"], dtype=complex)
    channels = np.asarray(data["channels"]).astype(str)
    q_names = np.asarray(data["q_names"]).astype(str)
    Sevals = np.asarray(data["structure_eigenvalues"], dtype=float)
    derived = derived_mode_data(data)
    Sdiag = derived["structure_diagonal"]
    weights = derived["leading_mode_weights"]
    mids = transition_midpoints(data)

    summary_path = Path(summary_path)
    modes_path = Path(modes_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    modes_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure 1: many-body / global summary.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(9.2, 13.0), sharex=True)

    axes[0].plot(V, energies[:, 0], marker="o", ms=3)
    axes[0].set_ylabel(r"$E_0$")
    axes[0].set_title(
        "18-site Ruby ED, n=2 (Gamma/Q benchmark; M not commensurate)"
    )

    nshow = max(2, min(int(top_spectrum), energies.shape[1]))
    for n in range(1, nshow):
        axes[1].plot(
            V,
            energies[:, n] - energies[:, 0],
            marker="o",
            ms=2.5,
            label=rf"$E_{n}-E_0$",
        )
    axes[1].set_ylabel("low-energy splitting")
    axes[1].legend(ncol=min(3, nshow - 1), fontsize=8)

    axes[2].plot(V, gaps, marker="o", ms=3, label="gap above GS manifold")
    axes[2].set_ylabel("manifold gap")
    axdeg = axes[2].twinx()
    axdeg.step(V, deg, where="mid", label="GS multiplicity")
    axdeg.set_ylabel("GS multiplicity")
    axdeg.set_yticks(sorted(set(int(x) for x in deg)))
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = axdeg.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    axes[3].plot(V, dint, marker="s", ms=3)
    axes[3].set_ylabel(r"$\langle D\rangle=dE_0/dV$")
    axes[3].set_title(
        r"$D=\sum_{\langle ij\rangle_\triangle} n_i n_j$"
    )

    iq_gamma = int(np.where(q_names == "Gamma")[0][0])
    iq_q = int(np.where(q_names == "Q")[0][0])
    axes[4].plot(
        V, Sevals[:, iq_gamma, 0], marker="o", ms=3, label=r"$S_{\max}(\Gamma)$"
    )
    axes[4].plot(
        V, Sevals[:, iq_q, 0], marker="s", ms=3, label=r"$S_{\max}(Q)$"
    )
    axes[4].set_ylabel("leading structure eigenvalue")
    axes[4].set_xlabel(r"$V$")
    axes[4].legend()

    for ax in axes:
        _add_transition_lines(ax, mids)
        ax.grid(alpha=0.2)

    # Label the finite-size GS sector on the gap panel.
    sector_labels = [ground_sector_label(trans[i]) for i in range(len(V))]
    segments = []
    start = 0
    for i in range(1, len(V) + 1):
        if i == len(V) or sector_labels[i] != sector_labels[start]:
            segments.append((start, i - 1, sector_labels[start]))
            start = i
    ymax = axes[2].get_ylim()[1]
    for i0, i1, label in segments:
        x = 0.5 * (V[i0] + V[i1])
        axes[2].text(x, 0.92 * ymax, label, ha="center", va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(summary_path, dpi=int(dpi))
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: what physical mode is responsible for S_max?
    # ------------------------------------------------------------------
    labels = _channel_labels(channels)
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 11.5), sharex="col")
    q_indices = [iq_gamma, iq_q]
    q_titles = [r"$\Gamma$", r"$Q=(1/3,1/3)$"]

    nstruct = max(1, min(int(top_structure), Sevals.shape[-1]))
    for col, (iq, qtitle) in enumerate(zip(q_indices, q_titles)):
        # Top: first few eigenvalues of the full 6x6 structure matrix.
        for m in range(nstruct):
            axes[0, col].plot(
                V,
                Sevals[:, iq, m],
                marker="o",
                ms=2.5,
                label=rf"$s_{m+1}$",
            )
        axes[0, col].set_title(qtitle)
        axes[0, col].set_ylabel("structure eigenvalues")
        axes[0, col].legend(fontsize=8)

        # Middle: diagonal channel correlations. These are NOT generally the
        # eigenvalues when channels mix, but show which named observables
        # fluctuate strongly.
        for ic, label in enumerate(labels):
            axes[1, col].plot(V, Sdiag[:, iq, ic], label=label)
        axes[1, col].set_ylabel(r"diagonal $S_{\mu\mu}$")
        axes[1, col].legend(ncol=2, fontsize=8)

        # Bottom: composition of the leading collective eigenmode.
        for ic, label in enumerate(labels):
            axes[2, col].plot(V, weights[:, iq, ic], label=label)
        axes[2, col].set_ylabel(r"leading weight $|v_\mu|^2$")
        axes[2, col].set_xlabel(r"$V$")
        axes[2, col].set_ylim(-0.03, 1.03)
        axes[2, col].legend(ncol=2, fontsize=8)

        for row in range(3):
            _add_transition_lines(axes[row, col], mids)
            axes[row, col].grid(alpha=0.2)

    fig.suptitle(
        "18-site Ruby ED: channel-resolved equal-time pseudospin correlations",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(modes_path, dpi=int(dpi))
    plt.close(fig)

    return summary_path, modes_path
