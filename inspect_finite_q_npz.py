#!/usr/bin/env python3
"""Inspect finite-q susceptibility NPZ files and identify every eigenmode.

This utility is intended for outputs of ``scan_primitive_cgw_q.py``.  For each
stored q point it diagonalizes the Hermitian susceptibility matrix, prints all
susceptibility eigenvalues, the corresponding curvature values ``r=1/lambda``,
and the channel composition of every eigenvector.  Negative eigenvalues are
also collected in a separate summary so that an unstable/post-pole mode is not
hidden by a positive ``lambda_max``.

Examples
--------

    python inspect_finite_q_npz.py full6_M.npz
    python inspect_finite_q_npz.py full6_Gamma.npz full6_Q.npz full6_M.npz
    python inspect_finite_q_npz.py gg_all6.npz --negative-only

The eigenvector global phase is arbitrary.  Before printing, each mode is
phase-fixed so that its largest-magnitude component is real and positive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", help="One or more .npz files to inspect.")
    p.add_argument(
        "--negative-only",
        action="store_true",
        help="Print only negative-eigenvalue modes instead of the full spectrum.",
    )
    p.add_argument(
        "--weight-cut",
        type=float,
        default=1e-3,
        help="Hide channel components with |v_i|^2 below this threshold.",
    )
    p.add_argument(
        "--zero-tol",
        type=float,
        default=1e-12,
        help="Treat |lambda| below this threshold as numerically zero.",
    )
    p.add_argument(
        "--q-index",
        nargs=2,
        type=int,
        metavar=("IQ1", "IQ2"),
        help="Inspect only one stored q index when a file contains a full q scan.",
    )
    return p.parse_args()


def _decode_strings(arr: np.ndarray) -> list[str]:
    out: list[str] = []
    for x in np.asarray(arr).reshape(-1):
        if isinstance(x, bytes):
            out.append(x.decode())
        else:
            out.append(str(x))
    return out


def _scalar(d, key: str, default=None):
    if key not in d:
        return default
    arr = np.asarray(d[key])
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def _phase_fix(vec: np.ndarray) -> np.ndarray:
    """Fix arbitrary eigenvector phase by making its largest entry real positive."""
    v = np.asarray(vec, dtype=complex).copy()
    if v.size == 0:
        return v
    pivot = int(np.argmax(np.abs(v)))
    if abs(v[pivot]) == 0.0:
        return v
    v *= np.exp(-1j * np.angle(v[pivot]))
    if v[pivot].real < 0.0:
        v *= -1.0
    # Remove insignificant numerical imaginary parts after phase fixing.
    v.real[np.abs(v.real) < 5e-15] = 0.0
    v.imag[np.abs(v.imag) < 5e-15] = 0.0
    return v


def _format_amp(z: complex) -> str:
    z = complex(z)
    if abs(z.imag) < 5e-10:
        return f"{z.real:+.6f}"
    if abs(z.real) < 5e-10:
        return f"{z.imag:+.6f}i"
    return f"({z.real:+.6f}{z.imag:+.6f}i)"


def _mode_expression(
    channels: list[str], vec: np.ndarray, weight_cut: float
) -> str:
    weights = np.abs(vec) ** 2
    order = np.argsort(weights)[::-1]
    pieces: list[str] = []
    for i in order:
        if float(weights[i]) < float(weight_cut):
            continue
        pieces.append(f"{_format_amp(vec[i])} {channels[i]}")
    return "  ".join(pieces) if pieces else "(all printed weights below cutoff)"


def _print_mode(
    rank: int,
    lam: float,
    vec: np.ndarray,
    channels: list[str],
    weight_cut: float,
    zero_tol: float,
    indent: str = "",
) -> None:
    if abs(lam) <= zero_tol:
        curvature = "singular / |lambda| ~ 0"
    else:
        curvature = f"{1.0 / lam:+.10e}"

    status = "NEGATIVE" if lam < -zero_tol else ("ZERO" if abs(lam) <= zero_tol else "positive")
    print(
        f"{indent}mode {rank:2d}: lambda={lam:+.10e}  "
        f"r=1/lambda={curvature}  [{status}]"
    )
    print(f"{indent}  combination: {_mode_expression(channels, vec, weight_cut)}")

    weights = np.abs(vec) ** 2
    order = np.argsort(weights)[::-1]
    for i in order:
        if float(weights[i]) < float(weight_cut):
            continue
        print(
            f"{indent}    {channels[i]:15s} "
            f"amp={_format_amp(vec[i]):>22s}  weight={weights[i]:.6f}"
        )


def _find_q_positions(d, requested_q_index):
    nq = int(np.asarray(d["chi_raw"]).shape[0])
    if requested_q_index is None:
        return list(range(nq))
    if "q_indices" not in d:
        raise KeyError("file has no q_indices field")
    target = tuple(int(x) for x in requested_q_index)
    q_indices = np.asarray(d["q_indices"], dtype=int)
    found = [i for i, q in enumerate(q_indices) if tuple(int(x) for x in q) == target]
    if not found:
        raise ValueError(f"requested q-index {target} is not stored in this file")
    return found


def _hermitian_matrices(d) -> tuple[np.ndarray, str]:
    if "chi_hermitian_qpair" in d:
        return np.asarray(d["chi_hermitian_qpair"], dtype=complex), "chi_hermitian_qpair"
    raw = np.asarray(d["chi_raw"], dtype=complex)
    herm = 0.5 * (raw + np.swapaxes(raw.conj(), -1, -2))
    return herm, "local 0.5*(chi_raw+chi_raw^dagger)"


def inspect_file(path: str, args: argparse.Namespace) -> None:
    p = Path(path)
    d = np.load(p, allow_pickle=False)

    print("\n" + "=" * 88)
    print(f"FILE: {p}")
    print("=" * 88)

    stage = _scalar(d, "stage", "unknown")
    if isinstance(stage, bytes):
        stage = stage.decode()
    print(
        f"V={_scalar(d, 'V', 'n/a')}  filling={_scalar(d, 'filling', 'n/a')}  "
        f"T={_scalar(d, 'T', 'n/a')}  stage={stage}"
    )
    print(
        f"mesh={_scalar(d, 'nk1', 'n/a')}x{_scalar(d, 'nk2', 'n/a')}  "
        f"mu={_scalar(d, 'mu', 'n/a')}  "
        f"background_residual={_scalar(d, 'background_residual', 'n/a')}"
    )

    left = _decode_strings(d["left_channels"])
    right = _decode_strings(d["right_channels"])
    if left != right:
        raise ValueError(
            "eigenmode analysis requires the same left/right channel basis; "
            f"got left={left}, right={right}"
        )
    channels = left
    print("channels:", ", ".join(channels))

    chi, chi_source = _hermitian_matrices(d)
    print("eigenanalysis matrix:", chi_source)

    q_indices = np.asarray(d["q_indices"], dtype=int) if "q_indices" in d else None
    q_reduced = np.asarray(d["q_reduced"], dtype=float) if "q_reduced" in d else None
    q_centered = np.asarray(d["q_centered"], dtype=float) if "q_centered" in d else None
    positions = _find_q_positions(d, args.q_index)

    negative_summary: list[tuple[int, int, float, np.ndarray]] = []

    for pos in positions:
        mat = np.asarray(chi[pos], dtype=complex)
        mat = 0.5 * (mat + mat.conj().T)
        evals, evecs = np.linalg.eigh(mat)  # ascending eigenvalues

        # Print from largest positive down to most negative for ordinary view.
        order = np.argsort(evals)[::-1]

        print("\n" + "-" * 88)
        q_parts = []
        if q_indices is not None:
            q_parts.append(f"q_index={tuple(int(x) for x in q_indices[pos])}")
        if q_reduced is not None:
            q_parts.append(f"q_reduced={tuple(float(x) for x in q_reduced[pos])}")
        if q_centered is not None:
            q_parts.append(f"q_centered={tuple(float(x) for x in q_centered[pos])}")
        print("  ".join(q_parts) if q_parts else f"q position {pos}")

        neg_here = [int(i) for i in np.where(evals < -float(args.zero_tol))[0]]
        print(
            f"spectrum: {len(evals)} modes, "
            f"negative={len(neg_here)}, positive={int(np.sum(evals > args.zero_tol))}"
        )

        if not args.negative_only:
            print("\nALL EIGENMODES (sorted by lambda descending):")
            for display_rank, idx in enumerate(order, start=1):
                vec = _phase_fix(evecs[:, idx])
                _print_mode(
                    display_rank,
                    float(evals[idx].real),
                    vec,
                    channels,
                    args.weight_cut,
                    args.zero_tol,
                    indent="  ",
                )

        if neg_here:
            print("\nNEGATIVE MODES (most negative first):")
            for neg_rank, idx in enumerate(neg_here, start=1):
                vec = _phase_fix(evecs[:, idx])
                _print_mode(
                    neg_rank,
                    float(evals[idx].real),
                    vec,
                    channels,
                    args.weight_cut,
                    args.zero_tol,
                    indent="  ",
                )
                negative_summary.append((pos, idx, float(evals[idx].real), vec))
        else:
            print("\nNEGATIVE MODES: none")

    print("\n" + "=" * 88)
    print("NEGATIVE-MODE SUMMARY")
    print("=" * 88)
    if not negative_summary:
        print("No eigenvalue below -zero_tol in the inspected q points.")
    else:
        for pos, _, lam, vec in sorted(negative_summary, key=lambda x: x[2]):
            if q_centered is not None:
                qtxt = tuple(float(x) for x in q_centered[pos])
            elif q_indices is not None:
                qtxt = tuple(int(x) for x in q_indices[pos])
            else:
                qtxt = pos
            print(
                f"q={qtxt}  lambda={lam:+.10e}  "
                f"mode={_mode_expression(channels, vec, args.weight_cut)}"
            )


def main() -> None:
    args = _parse_args()
    if args.weight_cut < 0.0 or args.weight_cut > 1.0:
        raise ValueError("--weight-cut must lie between 0 and 1")
    if args.zero_tol <= 0.0:
        raise ValueError("--zero-tol must be positive")
    for path in args.files:
        inspect_file(path, args)


if __name__ == "__main__":
    main()
