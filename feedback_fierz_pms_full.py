#!/usr/bin/env python3
"""Feed the full n/B/J covariant response back into a refined full-PMS root.

The input is ``*_full_pms.npz`` from ``refine_fierz_pms_full.py``.  The Fierz
weights are held fixed at the selected two-dimensional PMS root.  The script
then iterates the Gamma_P diagnostic closure

    G -> chi_cov[n/B/J] -> W_Gamma = g - g chi_cov g -> Sigma -> G

on the same folded finite torus and compares the resulting Green function and
q=0 current response against the upstream ED reference.

This is intentionally *not* a re-optimization of the PMS weights and does not
report a new free energy: the Gamma_P screening-feedback closure is not the
same Phi-derivable GW functional used to define the original PMS free energy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rubycgw.fierz_channel_gamma import solve_channel_gamma_feedback
from rubycgw.fierz_channel_gw import (
    ChannelGWResult,
    channel_static_self_energy,
    solve_channel_vertex_q0,
    susceptibility_from_vertex_q0,
)
from rubycgw.fierz_mixed import FierzWeights, build_weighted_nbj_definition
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.hedin_gamma_fast import GammaPFeedbackOptions
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw_split import one_body_density_matrix_tail


CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-pms-npz", type=Path, required=True)
    p.add_argument("--candidate", type=int, default=0)
    p.add_argument("--feedback-max", type=int, default=20)
    p.add_argument("--feedback-tol", type=float, default=2e-6)
    p.add_argument("--feedback-mixing", type=float, default=0.15)
    p.add_argument("--feedback-w-mixing", type=float, default=None)
    p.add_argument(
        "--m-max", type=int, default=0,
        help="max |bosonic Matsubara index| solved covariantly; use -1 for all",
    )
    p.add_argument("--mixing-method", choices=("linear", "pulay"), default="pulay")
    p.add_argument("--pulay-history", type=int, default=6)
    p.add_argument("--pulay-start", type=int, default=3)
    p.add_argument("--vertex-max-iter", type=int, default=300)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-gmres-restart", type=int, default=16)
    p.add_argument("--tail-edge-points", type=int, default=2)
    p.add_argument("--low-count", type=int, default=8)
    p.add_argument("--mu-tol", type=float, default=1e-10)
    p.add_argument("--mu-max-iter", type=int, default=100)
    p.add_argument("--allow-unconverged-vertex", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/fierz_full_gamma_feedback"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _relerr(a, b):
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    den = max(float(np.linalg.norm(bb.ravel())), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def _green_errors(G, Ged, grid, low_count):
    arr = np.asarray(G, dtype=complex)
    if arr.ndim == 5:
        arr = arr[:, 0, 0]
    exact = np.asarray(Ged, dtype=complex)
    idx = np.argsort(np.abs(np.asarray(grid.omega)))[: min(int(low_count), grid.nf)]
    return _relerr(arr, exact), _relerr(arr[idx], exact[idx])


def _physical_chi(chi):
    herm = 0.5 * (np.asarray(chi) + np.asarray(chi).conj().T)
    return np.asarray(herm.real, dtype=float), float(np.max(np.abs(herm.imag)))


def _load(full_path: Path, candidate: int):
    with np.load(full_path, allow_pickle=False) as d:
        nroot = len(np.asarray(d["a_star"]))
        if candidate < 0 or candidate >= nroot:
            raise IndexError(f"candidate {candidate} outside [0,{nroot-1}]")
        source_target = Path(str(np.asarray(d["source_target_npz"]).item()))
        data = {
            "V": float(np.asarray(d["V"]).item()),
            "source_root": int(np.asarray(d["source_root"]).item()),
            "a": float(np.asarray(d["a_star"])[candidate]),
            "s": float(np.asarray(d["s_star"])[candidate]),
            "ln": float(np.asarray(d["lambda_n"])[candidate]),
            "lb": float(np.asarray(d["lambda_B"])[candidate]),
            "lj": float(np.asarray(d["lambda_J"])[candidate]),
            "mu": float(np.asarray(d["mu"])[candidate]),
            "G": np.asarray(d["G"])[candidate],
            "P": np.asarray(d["P"])[candidate],
            "W": np.asarray(d["W"])[candidate],
            "Sigma_static": np.asarray(d["Sigma_static"])[candidate],
            "Sigma_c": np.asarray(d["Sigma_c"])[candidate],
            "chi_before": np.asarray(d["chi"])[candidate] if "chi" in d else None,
            "Gerr_before_saved": (
                float(np.asarray(d["Gerr"])[candidate]) if "Gerr" in d else np.nan
            ),
        }
    if not source_target.exists():
        raise RuntimeError(f"source target NPZ not found: {source_target}")
    with np.load(source_target, allow_pickle=False) as d:
        source_checkpoint = Path(str(np.asarray(d["source_checkpoint"]).item()))
        G_ed = np.asarray(d["G_ed"], dtype=complex) if "G_ed" in d else np.empty((0,), complex)
        chi_ed = np.asarray(d["chi_ed"], dtype=float) if "chi_ed" in d else np.empty((0,), float)
    if not source_checkpoint.exists():
        raise RuntimeError(f"source checkpoint not found: {source_checkpoint}")
    with np.load(source_checkpoint, allow_pickle=False) as d:
        meta = json.loads(str(np.asarray(d["signature"]).item()))
    return data, source_target, meta, G_ed, chi_ed


def _background(data, definition, h0, grid):
    rho = one_body_density_matrix_tail(
        data["G"], grid, h0, data["mu"], data["Sigma_static"]
    )
    rho0 = np.asarray(rho[0, 0], dtype=complex)
    _, tad, exchange = channel_static_self_energy(rho0, definition)
    return ChannelGWResult(
        G=np.asarray(data["G"]),
        W=np.asarray(data["W"]),
        P=np.asarray(data["P"]),
        Sigma_static=np.asarray(data["Sigma_static"]),
        Sigma_tadpole=np.asarray(tad),
        Sigma_exchange=np.asarray(exchange),
        Sigma_c=np.asarray(data["Sigma_c"]),
        mu=float(data["mu"]),
        rho=rho0,
        density=np.real(np.diag(rho0)),
        converged=True,
        iterations=0,
        final_error=0.0,
        mixing_method="full-pms-newton",
        mode=definition.mode,
        min_screening_singular_value=np.nan,
    )


def _q0_current_chi(bg, definition, operators, grid, vopts):
    raw = np.full((2, 2), np.nan + 0j, dtype=complex)
    residuals = []
    ok = True
    for b, (name, Ksrc) in enumerate(zip(CHANNELS, operators)):
        print(f"final q=0 cGW vertex: {name} ...")
        vr = solve_channel_vertex_q0(bg, definition, Ksrc, grid, opts=vopts)
        residuals.append(float(vr.final_error))
        if not vr.converged:
            ok = False
            print(f"WARNING: final {name} vertex failed: {vr.final_error:.3e}")
            break
        for a, Kleft in enumerate(operators):
            raw[a, b] = susceptibility_from_vertex_q0(bg.G, Kleft, vr.Gamma, grid)
    if not ok:
        return np.full((2, 2), np.nan), raw, np.nan, max(residuals) if residuals else np.nan
    chi, imag = _physical_chi(raw)
    return chi, raw, imag, max(residuals) if residuals else 0.0


def main():
    args = _args()
    data, source_target, meta, G_ed, chi_ed = _load(args.full_pms_npz, args.candidate)
    params = RubyParameters(
        ti=float(meta["ti"]), t1=float(meta["t1"]), t2=float(meta["t2"]), V=0.0
    )
    geometry = ExactSmallRubyThermal(int(meta["L1"]), int(meta["L2"]), params)
    grid = MatsubaraGrid(
        nk1=1,
        nk2=1,
        nw=int(meta["nw"]),
        nOmega=int(meta["nomega"]),
        T=float(meta["T"]),
    )
    h0 = np.asarray(geometry.h0, dtype=complex)[None, None]
    weights = FierzWeights(data["ln"], data["lb"], data["lj"])
    definition = build_weighted_nbj_definition(
        geometry.interaction_pairs, int(geometry.n_sites), data["V"], weights
    )
    bg = _background(data, definition, h0, grid)
    operators = np.stack([
        np.asarray(geometry.pseudospin_operator(ch, (0.0, 0.0)), dtype=complex)
        for ch in CHANNELS
    ])

    gw_opts = GWOptions(
        mu=float(data["mu"]),
        target_filling=float(meta["target"]),
        mu_tol=float(args.mu_tol),
        mu_max_iter=int(args.mu_max_iter),
        verbose=bool(args.verbose),
        momentum_backend="direct",
    )
    vopts = SupercellVertexOptions(
        max_iter=int(args.vertex_max_iter),
        tol=float(args.vertex_tol),
        solver="gmres",
        gmres_restart=int(args.vertex_gmres_restart),
        verbose=bool(args.verbose),
        momentum_backend="direct",
    )
    m_max = None if int(args.m_max) < 0 else int(args.m_max)
    fopts = GammaPFeedbackOptions(
        max_iter=int(args.feedback_max),
        tol=float(args.feedback_tol),
        mixing=float(args.feedback_mixing),
        w_mixing=(None if args.feedback_w_mixing is None else float(args.feedback_w_mixing)),
        m_max=m_max,
        allow_unconverged_vertex=bool(args.allow_unconverged_vertex),
        verbose=True,
        mixing_method=str(args.mixing_method),
        pulay_history=int(args.pulay_history),
        pulay_start=int(args.pulay_start),
    )

    print("=== Fixed-full-PMS n/B/J Gamma_P feedback ===")
    print(
        f"V={data['V']:g}, source root={data['source_root']}, candidate={args.candidate}, "
        f"a={data['a']:+.10f}, s={data['s']:.10f}"
    )
    print(
        f"weights=(n,B,J)=({data['ln']:.9f},{data['lb']:.9f},{data['lj']:.9f}), "
        f"nchannel={len(definition.labels)}, m_max={'all' if m_max is None else m_max}"
    )

    g_before = g_before_low = np.nan
    if G_ed.size:
        g_before, g_before_low = _green_errors(bg.G, G_ed, grid, args.low_count)
        print(f"before feedback: Gerr={g_before:.6e}, Gerr_low={g_before_low:.6e}")
    if data["chi_before"] is not None:
        c0 = np.asarray(data["chi_before"], dtype=float)
        cerr0 = _relerr(c0, chi_ed) if chi_ed.shape == (2, 2) else np.nan
        print(
            f"before feedback: chi=({c0[0,0]:+.9f},{c0[1,1]:+.9f}), "
            f"chi_relerr={cerr0:.6e}"
        )

    result = solve_channel_gamma_feedback(
        h0,
        definition,
        grid,
        gw_opts=gw_opts,
        vertex_opts=vopts,
        feedback_opts=fopts,
        background=bg,
        tail_edge_points=int(args.tail_edge_points),
    )
    print(
        f"feedback fixed point: converged={result.converged}, iterations={result.iterations}, "
        f"residual={result.final_error:.3e}, W_identity={result.screening_identity_error:.3e}"
    )

    _, tad, exchange = channel_static_self_energy(result.rho, definition)
    final_bg = ChannelGWResult(
        G=np.asarray(result.G),
        W=np.asarray(result.W),
        P=np.asarray(result.P_gamma),
        Sigma_static=np.asarray(result.Sigma_static),
        Sigma_tadpole=np.asarray(tad),
        Sigma_exchange=np.asarray(exchange),
        Sigma_c=np.asarray(result.Sigma_c),
        mu=float(result.mu),
        rho=np.asarray(result.rho),
        density=np.real(np.diag(result.rho)),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        final_error=float(result.final_error),
        mixing_method="fierz-gamma-feedback",
        mode=definition.mode,
        min_screening_singular_value=np.nan,
    )
    chi, chi_raw, chi_imag, q0_vertex_res = _q0_current_chi(
        final_bg, definition, operators, grid, vopts
    )

    gerr = gerr_low = np.nan
    if G_ed.size:
        gerr, gerr_low = _green_errors(result.G, G_ed, grid, args.low_count)
    chi_err = _relerr(chi, chi_ed) if chi_ed.shape == (2, 2) else np.nan
    print(
        f"after feedback: Gerr={gerr:.6e}, Gerr_low={gerr_low:.6e}, "
        f"chi=({chi[0,0]:+.9f},{chi[1,1]:+.9f}), chi_relerr={chi_err:.6e}"
    )
    if np.isfinite(g_before) and np.isfinite(gerr):
        print(f"Gerr ratio after/before = {gerr/g_before:.6f}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.full_pms_npz.stem.replace("_full_pms", "")
    tag = f"{stem}_cand{args.candidate}_gammaP_m{'all' if m_max is None else m_max}"
    outfile = args.out / f"{tag}.npz"
    np.savez_compressed(
        outfile,
        V=float(data["V"]),
        source_root=int(data["source_root"]),
        candidate=int(args.candidate),
        a_star=float(data["a"]),
        s_star=float(data["s"]),
        lambda_n=float(data["ln"]),
        lambda_B=float(data["lb"]),
        lambda_J=float(data["lj"]),
        channel_labels=np.asarray(definition.labels),
        m_max=np.asarray(-1 if m_max is None else m_max),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        final_error=float(result.final_error),
        screening_identity_error=float(result.screening_identity_error),
        mu=float(result.mu),
        G=np.asarray(result.G),
        W=np.asarray(result.W),
        P_gamma=np.asarray(result.P_gamma),
        chi_channel_cov=np.asarray(result.chi_cov),
        chi_channel_raw=np.asarray(result.response.chi_raw),
        chi_channel_tail=np.asarray(result.response.tail_correction),
        transfer_converged=np.asarray(result.response.transfer_converged),
        transfer_max_error=np.asarray(result.response.transfer_max_error),
        fallback_mask=np.asarray(result.fallback_mask),
        Sigma_static=np.asarray(result.Sigma_static),
        Sigma_c=np.asarray(result.Sigma_c),
        rho=np.asarray(result.rho),
        chi=np.asarray(chi),
        chi_raw=np.asarray(chi_raw),
        chi_ed=np.asarray(chi_ed),
        chi_relerr=float(chi_err),
        chi_max_imag=float(chi_imag),
        q0_vertex_max_residual=float(q0_vertex_res),
        Gerr=float(gerr),
        Gerr_low=float(gerr_low),
        Gerr_before=float(g_before),
        Gerr_low_before=float(g_before_low),
        G_ed=np.asarray(G_ed),
        source_full_pms_npz=np.asarray(str(args.full_pms_npz)),
        source_target_npz=np.asarray(str(source_target)),
        free_energy_valid=np.asarray(False),
        free_energy_note=np.asarray(
            "Gamma_P feedback is a screening-feedback diagnostic; the original GW PMS free-energy functional is not reused."
        ),
    )
    print(f"saved {outfile}")
    print("NOTE: no feedback free energy is reported; this closure is not the original Phi_GW PMS functional.")


if __name__ == "__main__":
    main()
