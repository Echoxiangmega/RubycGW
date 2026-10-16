#!/usr/bin/env python3
"""Static effective vertex-rank diagnostic for the original V-only Ruby model.

Run from repository root:
    python research/diagnose_v_only_vertex_rank.py --V 0.2 0.8

This computes all 36 intracell q=0 Hermitian bilinears using exact
grand-canonical thermal ED and the full H/F/MT/AL1/AL2 cGW response.
It diagnoses a STATIC PROJECTED response correction, not the full
frequency-dependent irreducible parquet vertex.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Support both "python -m research.diagnose_v_only_vertex_rank" and direct
# invocation "python research/diagnose_v_only_vertex_rank.py" from a checkout
# that has not been installed with "pip install -e .".
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rubycgw.ed_cgw_benchmark import relative_frobenius_error
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters, eta_vertices
from rubycgw.production_cgw import solve_vertex_q0_tail
from rubycgw.production_cgw_sox import static_gg_tail_completed, static_response_tail_completed
from rubycgw.response_tail import build_tail_reference
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.supercell_gw_split import compute_sigma_gw_split_components


def hermitian_form_factors(n=6):
    """36 orthonormal Hermitian single-cell matrices (trace inner product)."""
    mats, labels, kinds = [], [], []
    for i in range(n):
        k = np.zeros((n, n), dtype=complex)
        k[i, i] = 1
        mats.append(k)
        labels.append(f"n{i}")
        kinds.append("density")
    for i in range(n):
        for j in range(i + 1, n):
            k = np.zeros((n, n), dtype=complex)
            k[i, j] = k[j, i] = 1 / np.sqrt(2)
            mats.append(k)
            labels.append(f"B{i}{j}")
            kinds.append("real_bond")
    for i in range(n):
        for j in range(i + 1, n):
            k = np.zeros((n, n), dtype=complex)
            k[i, j] = 1j / np.sqrt(2)
            k[j, i] = -1j / np.sqrt(2)
            mats.append(k)
            labels.append(f"J{i}{j}")
            kinds.append("imag_bond")
    return np.asarray(mats), np.asarray(labels), np.asarray(kinds)


def embed_q0(mats, cells):
    """Uniform q=0 sources normalized by sqrt(number of primitive cells)."""
    a = np.asarray(mats)
    cells, n = int(cells), a.shape[-1]
    out = np.zeros((len(a), cells * n, cells * n), dtype=complex)
    for c in range(cells):
        out[:, c*n:(c+1)*n, c*n:(c+1)*n] = a / np.sqrt(cells)
    return out


def symmetric(a):
    a = np.asarray(a)
    return 0.5 * (a.real + a.real.T)


def psd_reference(chi, rcond=1e-7):
    chi = symmetric(chi)
    v, U = np.linalg.eigh(chi)
    maxev = max(float(np.max(np.abs(v))), 1e-300)
    if float(np.min(v)) < -rcond * maxev:
        raise ValueError("Reference GW bubble is not PSD; increase Matsubara cutoffs")
    good = v > rcond * maxev
    if not np.any(good):
        raise ValueError("Reference bubble has no resolved positive modes")
    return U[:, good], v[good], v


def stable_inverse(a, rcond=1e-7):
    """Spectrally truncated inverse in the chosen common support."""
    vals, U = np.linalg.eigh(symmetric(a))
    cut = rcond * max(float(np.max(np.abs(vals))), 1e-300)
    if float(np.min(vals)) < -cut:
        raise ValueError("Resolved negative susceptibility eigenvalue")
    inv = np.divide(1., vals, out=np.zeros_like(vals), where=vals > cut)
    return (U * inv) @ U.T, vals, int(np.sum(vals > cut))


def effective_rank_diagnostic(chi_ed, chi_cgw, bubble_gw, bubble_ed=None,
                              rcond=1e-7):
    """SVD of weighted STATIC inverse-response correction.

    Main diagnostic: Delta=chi_cGW^{-1}-chi_ED^{-1} in a common positive
    bubble_GW support. This includes both G and vertex errors, not merely
    missing fully irreducible diagrams.

    Optional DeltaK = (chi0_ED^{-1}-chi_ED^{-1})
                    - (chi0_GW^{-1}-chi_cGW^{-1})
    is an effective STATIC kernel difference, not a full four-leg I_ph.
    """
    Z, vals, allvals = psd_reference(bubble_gw, rcond)
    ce, cc, bg = (symmetric(x) for x in (chi_ed, chi_cgw, bubble_gw))
    e, c = Z.T @ ce @ Z, Z.T @ cc @ Z
    ie, evals_ed, ne = stable_inverse(e, rcond)
    ic, evals_cgw, nc = stable_inverse(c, rcond)
    delta_inv = ic - ie
    diag_sqrt = np.diag(np.sqrt(vals))
    diag_invsqrt = np.diag(1 / np.sqrt(vals))
    D = symmetric(diag_sqrt @ delta_inv @ diag_sqrt)
    eig, vec = np.linalg.eigh(D)
    idx = np.argsort(-np.abs(eig))
    eig, vec = eig[idx], vec[:, idx]
    total = float(np.sum(eig ** 2))
    capture = np.cumsum(eig ** 2) / total if total > 1e-30 else np.ones(len(eig))

    modes = Z @ diag_invsqrt @ vec
    modes /= np.maximum(np.linalg.norm(modes, axis=0, keepdims=True), 1e-300)

    reconstructed = []
    for r in range(len(eig) + 1):
        Dr = (vec[:, :r] * eig[:r]) @ vec[:, :r].T
        correction = diag_invsqrt @ Dr @ diag_invsqrt
        try:
            chi_r, _, _ = stable_inverse(ic - correction, rcond)
            reconstructed.append(Z @ chi_r @ Z.T)
        except ValueError:
            # A rank-truncated kernel can temporarily be non-positive.
            reconstructed.append(np.full_like(ce, np.nan))
    result = dict(
        response_eigenvalues=eig, capture=capture, modes=modes,
        chi_reconstructed=np.asarray(reconstructed),
        bubble_eigenvalues=allvals, retained_dim=len(vals),
        chi_ed_eigenvalues=evals_ed, chi_cgw_eigenvalues=evals_cgw,
        chi_ed_rank=ne, chi_cgw_rank=nc,
        delta_inv_response=Z @ delta_inv @ Z.T,
    )
    if bubble_ed is not None:
        be = Z.T @ symmetric(bubble_ed) @ Z
        ibe, _, _ = stable_inverse(be, rcond)
        ibg, _, _ = stable_inverse(Z.T @ bg @ Z, rcond)
        diff = (ibe - ie) - (ibg - ic)
        result["delta_K_effective"] = Z @ diff @ Z.T
        result["delta_bubble_inverse"] = Z @ (ibe - ibg) @ Z.T
    return result


def target_vectors(mats):
    ka, kb, opposite, same = eta_vertices()
    co = np.diag([1., 1., 1., -1., -1., -1.]).astype(complex)
    refs = {
        "LC_same": same,
        "LC_opposite": opposite,
        "CO_AminusB": co,
        "triangle_A": ka,
        "triangle_B": kb,
    }
    out = {}
    for key, mat in refs.items():
        coords = np.einsum("aij,ij->a", mats.conj(), mat, optimize=True).real
        out[key] = coords / np.linalg.norm(coords)
    return out


def response_for_sources(gw, h0, vq, grid, mats, vopts, edge_points):
    _, sigma_f, _, _ = compute_sigma_gw_split_components(
        gw.G, gw.W, vq, grid, h0, gw.mu, gw.Sigma_H,
        backend=vopts.momentum_backend,
    )
    hstatic = h0 + gw.Sigma_H[None, None] + sigma_f
    reference = build_tail_reference(h0, gw.mu, gw.Sigma_H, grid)
    gammas, errors = [], []
    for index, op in enumerate(mats):
        sol = solve_vertex_q0_tail(
            gw.G, gw.W, vq, op, grid, reference, opts=vopts
        )
        errors.append(sol.final_error)
        if not sol.converged:
            raise RuntimeError(
                f"cGW source {index} did not converge: {sol.final_error:.3e}"
            )
        gammas.append(sol.Gamma)
    chi = static_response_tail_completed(
        gw.G, mats, gammas, grid, hstatic, gw.mu, edge_points=edge_points
    )["completed"]
    bubble = static_gg_tail_completed(
        gw.G, mats, grid, hstatic, gw.mu, edge_points=edge_points
    )["completed"]
    return chi, bubble, np.asarray(errors)


def save_plots(data, folder, V):
    import matplotlib.pyplot as plt
    eig = np.asarray(data["response_eigenvalues"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, len(eig)+1), np.abs(eig)+1e-15, ".-")
    ax.set(xlabel="Retained mode index", ylabel="|weighted eigenvalue|",
           title=f"V={V:g}: static inverse-response correction")
    fig.tight_layout()
    fig.savefig(folder / "spectrum.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    capture = np.asarray(data["capture"])
    ax.plot(np.arange(1, len(capture)+1), capture, ".-")
    ax.axhline(.95, ls="--", color="grey")
    ax.axhline(.99, ls=":", color="grey")
    ax.set(xlabel="Retained modes r", ylabel="Frobenius capture R(r)",
           ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(folder / "rank_capture.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(np.abs(data["modes"][:, :8]), aspect="auto", origin="lower")
    ax.set(xlabel="Ranked mode", ylabel="36 Hermitian form factors")
    fig.colorbar(im, ax=ax, label="|mode coefficient|")
    fig.tight_layout()
    fig.savefig(folder / "mode_weights.png", dpi=180)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L1", type=int, default=1)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--V", nargs="+", type=float, default=[0.2])
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--filling", type=float, default=3.)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--gw-max-iter", type=int, default=140)
    p.add_argument("--vertex-max-iter", type=int, default=180)
    p.add_argument("--rcond", type=float, default=1e-7)
    p.add_argument("--out", type=Path,
                   default=Path("results/v_only_vertex_rank"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.L1 * args.L2 * 6 > 16:
        raise ValueError("Full thermal ED requires no more than 16 sites")
    mats, labels, kinds = hermitian_form_factors()
    refs = target_vectors(mats)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=args.nw,
                         nOmega=args.nomega, T=args.T)
    for V in args.V:
        print(f"==== Original V-only Ruby: V={V:g}, "
              f"{args.L1}x{args.L2} ====", flush=True)
        params = RubyParameters(ti=args.ti, t1=args.t1,
                                t2=args.t2, V=V)
        ed = ExactSmallRubyThermal(args.L1, args.L2, params)
        ed.diagonalize(V)
        filling = args.filling * ed.n_cells
        mu = ed.solve_mu(filling, args.T)
        sources = embed_q0(mats, ed.n_cells)
        chi_ed, means = ed.static_susceptibility_matrix(
            sources, mu, args.T
        )
        print("ED static 36x36 complete", flush=True)
        h0 = ed.h0[None, None]
        vq = (V * ed.Vunit)[None, None]
        opts = GWOptions(
            target_filling=filling, max_iter=args.gw_max_iter,
            tol=2e-8, mixing=0.22, mixing_method="pulay",
            verbose=False, momentum_backend="direct",
        )
        gw = solve_matrix_gw_fast(h0, vq, grid, opts=opts)
        if not gw.converged:
            raise RuntimeError(
                f"GW did not converge: {gw.final_error:.3e}"
            )
        vopts = SupercellVertexOptions(
            max_iter=args.vertex_max_iter, tol=2e-8,
            solver="gmres", gmres_restart=12, verbose=False,
            momentum_backend="direct",
            include_hartree=True, include_fock=True,
            include_mt=True, include_al=True,
        )
        print("GW converged; solving 36 full H/F/MT/AL responses",
              flush=True)
        chi_cgw, bubble_gw, residuals = response_for_sources(
            gw, h0, vq, grid, sources, vopts, edge_points=2
        )
        exact_g, selection = ed.green_iomega(
            1j * grid.omega, mu, args.T
        )
        bubble_ed = static_gg_tail_completed(
            exact_g[:, None, None], sources, grid, h0, mu,
            edge_points=2,
        )["completed"]
        chi_cgw, bubble_gw, bubble_ed = map(
            symmetric, (chi_cgw, bubble_gw, bubble_ed)
        )
        rank = effective_rank_diagnostic(
            chi_ed, chi_cgw, bubble_gw, bubble_ed,
            rcond=args.rcond
        )
        folder = args.out / f"V_{V:.6g}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, vec in refs.items():
            print(f" {name:14s} ED={vec@chi_ed@vec:+.7g} "
                  f"cGW={vec@chi_cgw@vec:+.7g}", flush=True)
        print(" bubble ED/GW relative error:",
              relative_frobenius_error(bubble_ed, bubble_gw))
        print(" rank at 95% and 99%:",
              int(np.searchsorted(rank["capture"], .95)+1),
              int(np.searchsorted(rank["capture"], .99)+1))
        projected = np.asarray([
            [v @ m @ v for m in rank["chi_reconstructed"]]
            for v in refs.values()
        ])
        np.savez_compressed(
            folder / "rank_diagnostic.npz",
            V=V, T=args.T, mu_ed=mu, mu_gw=gw.mu,
            L1=args.L1, L2=args.L2,
            labels=labels, kinds=kinds, operators=mats,
            chi_ed=chi_ed, chi_cgw=chi_cgw,
            bubble_ed=bubble_ed, bubble_gw=bubble_gw,
            means_ed=means, vertex_residuals=residuals,
            gw_residual=gw.final_error,
            exact_ed_discard_weight=selection.discarded_weight,
            target_names=np.asarray(list(refs)),
            target_vectors=np.asarray(list(refs.values())),
            target_reconstructed_chi=projected,
            **rank,
        )
        save_plots(rank, folder, V)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, curve in zip(refs, projected):
            if name in ("LC_same", "LC_opposite", "CO_AminusB"):
                ax.plot(np.arange(len(curve)), curve, ".-", label=name)
        ax.set(xlabel="Retained correction modes r",
               ylabel="Reconstructed static susceptibility")
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / "target_reconstruction.png", dpi=180)
        plt.close(fig)
        print(" saved", folder / "rank_diagnostic.npz", flush=True)


if __name__ == "__main__":
    main()
