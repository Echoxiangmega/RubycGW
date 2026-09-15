#!/usr/bin/env python3
"""Evaluate weighted n/B/J cGW and free energy at every target-V crossing.

The weighted PAC scan stores every accepted codec state in ``*_states.npz``.
This postprocessor searches the whole archived branch for every encounter with a
requested V, refines each candidate at exactly fixed V with Newton-Krylov,
deduplicates refined roots, and evaluates both q=0 current susceptibilities and
the matching multichannel GW Luttinger-Ward free energy for every distinct root.

For fixed filling the reported thermodynamic quantity is

    F_GW = Omega_GW + mu_GW N_target.

The exact 2x1 thermal reference is evaluated in the same grand-canonical
ensemble at the chemical potential satisfying <N>=N_target, then transformed as
F_ED=Omega_ED+mu_ED N_target.  Thus folded PAC roots at one V can be ranked by
DeltaF_branch while each approximate F can also be compared with ED.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rubycgw.branch_state_archive import find_target_crossings, load_branch_state_archive
from rubycgw.ed_benchmark_cache import load_ed_cache, save_ed_cache
from rubycgw.fierz_channel_gw import (
    ChannelGWResult,
    channel_static_self_energy,
    solve_channel_vertex_q0,
    susceptibility_from_vertex_q0,
)
from rubycgw.fierz_free_energy import (
    evaluate_channel_gw_free_energy,
    evaluate_exact_thermal_free_energy,
)
from rubycgw.fierz_mixed import (
    FierzWeights,
    WeightedFierzGWResidual,
    build_weighted_nbj_definition,
    soft_mode_sector_fractions,
)
from rubycgw.fierz_pac import soft_mode_overlap
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.pseudo_arclength import PACOptions, refine_fixed_parameter
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.supercell_cgw import SupercellVertexOptions


CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--states", type=Path, default=None,
        help="full-X PAC archive; default is inferred from *_checkpoint.npz",
    )
    p.add_argument("--V", type=float, required=True, help="exact target interaction")
    p.add_argument(
        "--crossing-atol", type=float, default=1e-10,
        help="absolute V tolerance for treating an archived point as an exact target hit",
    )
    p.add_argument(
        "--root-dedup-rtol", type=float, default=1e-7,
        help="relative codec-state distance below which refined roots are merged",
    )
    p.add_argument("--newton-tol", type=float, default=1e-10)
    p.add_argument("--newton-max", type=int, default=18)
    p.add_argument("--fd-eps", type=float, default=3e-7)
    p.add_argument("--gmres-rtol", type=float, default=5e-4)
    p.add_argument("--gmres-maxiter", type=int, default=80)
    p.add_argument("--gmres-restart", type=int, default=20)
    p.add_argument("--line-search-min", type=float, default=1.0 / 4096.0)
    p.add_argument("--screening-floor", type=float, default=1e-8)
    p.add_argument("--vertex-max-iter", type=int, default=300)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-gmres-restart", type=int, default=16)
    p.add_argument("--low-count", type=int, default=8)
    p.add_argument("--skip-ed", action="store_true")
    p.add_argument("--refresh-ed-cache", action="store_true")
    p.add_argument("--ed-cache-dir", type=Path, default=Path("results/ed_cache"))
    p.add_argument("--out", type=Path, default=Path("results/ed_fierz_weighted_cgw"))
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
    idx = np.argsort(np.abs(np.asarray(grid.omega)))[:min(int(low_count), grid.nf)]
    return _relerr(arr, exact), _relerr(arr[idx], exact[idx])


def _physical_chi(chi):
    herm = 0.5 * (np.asarray(chi) + np.asarray(chi).conj().T)
    return np.asarray(herm.real, dtype=float), float(np.max(np.abs(herm.imag)))


def _ed_signature(meta, V):
    return {
        "L1": int(meta["L1"]), "L2": int(meta["L2"]), "V": float(V),
        "filling": float(meta["filling"]), "T": float(meta["T"]),
        "ti": float(meta["ti"]), "t1": float(meta["t1"]), "t2": float(meta["t2"]),
        "nw": int(meta["nw"]), "channels": list(CHANNELS),
    }


def _lambda_tag(lam: float) -> str:
    return f"{float(lam):.6f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _infer_states_path(checkpoint: Path) -> Path:
    suffix = "_checkpoint.npz"
    name = checkpoint.name
    if name.endswith(suffix):
        return checkpoint.with_name(name[:-len(suffix)] + "_states.npz")
    return checkpoint.with_name(checkpoint.stem + "_states.npz")


def _root_distance(x1, x2) -> float:
    a = np.asarray(x1, dtype=float).reshape(-1)
    b = np.asarray(x2, dtype=float).reshape(-1)
    if a.shape != b.shape:
        return float("inf")
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0)
    return float(np.linalg.norm(a - b) / scale)


def _make_background(problem, definition, x, ev, root):
    sigma_static, sigma_c, mu = problem.codec.decode(x)
    _, tad, exchange = channel_static_self_energy(ev.rho, definition)
    return ChannelGWResult(
        G=np.asarray(ev.G), W=np.asarray(ev.W), P=np.asarray(ev.P),
        Sigma_static=np.asarray(sigma_static),
        Sigma_tadpole=np.asarray(tad), Sigma_exchange=np.asarray(exchange),
        Sigma_c=np.asarray(sigma_c), mu=float(mu), rho=np.asarray(ev.rho),
        density=np.real(np.diag(ev.rho)), converged=bool(root.converged),
        iterations=int(root.newton_iterations), final_error=float(ev.physical_residual),
        mixing_method="newton-krylov", mode=definition.mode,
        min_screening_singular_value=float(ev.smin),
    )


def _source_segment(crossing) -> dict:
    return {
        "left_index": int(crossing.left_index),
        "right_index": int(crossing.right_index),
        "left_step": int(crossing.left_step),
        "right_step": int(crossing.right_step),
        "V_left": float(crossing.V_left),
        "V_right": float(crossing.V_right),
        "alpha": float(crossing.alpha),
    }


def _ed_thermo_dict(omega, F, N, npc):
    return {
        "Omega": float(omega),
        "F": float(F),
        "N": float(N),
        "Omega_pc": float(omega) / float(npc),
        "F_pc": float(F) / float(npc),
    }


def _load_or_compute_ed(args, meta, geometry, operators, grid, V_target, norb):
    if args.skip_ed:
        return None, None, np.nan, "", None

    ed_sig = _ed_signature(meta, V_target)
    target_N = float(meta["target"])
    npc = int(meta["L1"]) * int(meta["L2"])
    cached = None if args.refresh_ed_cache else load_ed_cache(
        args.ed_cache_dir, ed_sig, np.asarray(grid.omega), norb
    )

    if cached is not None:
        print(f"ED cache hit: {cached['path']}")
        Ged = np.asarray(cached["G_ed"], dtype=complex)
        chi_ed = np.asarray(cached["chi_ed"], dtype=float)
        mu_ed = float(cached["mu_ed"])
        if bool(cached.get("has_thermo", False)):
            thermo = _ed_thermo_dict(
                cached["omega_ed"], cached["F_ed"], cached["N_ed"], npc
            )
            return Ged, chi_ed, mu_ed, str(cached["path"]), thermo

        print("ED cache has no thermodynamic scalars; diagonalizing once to upgrade it ...")
        geometry.diagonalize(V_target)
        exact = evaluate_exact_thermal_free_energy(
            geometry, mu_ed, float(meta["T"]), target_particles=target_N,
            primitive_cells_per_supercell=npc,
        )
        path = save_ed_cache(
            args.ed_cache_dir, ed_sig, np.asarray(grid.omega), mu_ed, Ged, chi_ed,
            omega_ed=exact.grand_potential,
            F_ed=exact.helmholtz_free_energy,
            N_ed=exact.particle_number_actual,
        )
        thermo = _ed_thermo_dict(
            exact.grand_potential, exact.helmholtz_free_energy,
            exact.particle_number_actual, npc,
        )
        print(f"ED cache upgraded with Omega/F: {path}")
        return Ged, chi_ed, mu_ed, str(path), thermo

    print("ED cache miss; computing exact G, chi and thermodynamics once at target V ...")
    geometry.diagonalize(V_target)
    mu_ed = geometry.solve_mu(target_N, float(meta["T"]))
    chi_ed, _ = geometry.static_susceptibility_matrix(operators, mu_ed, float(meta["T"]))
    chi_ed = np.asarray(chi_ed, dtype=float)
    Ged, _ = geometry.green_iomega(1j * np.asarray(grid.omega), mu_ed, float(meta["T"]))
    Ged = np.asarray(Ged, dtype=complex)
    exact = evaluate_exact_thermal_free_energy(
        geometry, mu_ed, float(meta["T"]), target_particles=target_N,
        primitive_cells_per_supercell=npc,
    )
    path = save_ed_cache(
        args.ed_cache_dir, ed_sig, np.asarray(grid.omega), mu_ed, Ged, chi_ed,
        omega_ed=exact.grand_potential,
        F_ed=exact.helmholtz_free_energy,
        N_ed=exact.particle_number_actual,
    )
    thermo = _ed_thermo_dict(
        exact.grand_potential, exact.helmholtz_free_energy,
        exact.particle_number_actual, npc,
    )
    print(f"ED cache saved: {path}")
    return Ged, chi_ed, float(mu_ed), str(path), thermo


def _write_summary_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _args()
    if args.crossing_atol < 0.0:
        raise ValueError("--crossing-atol must be non-negative")
    if args.root_dedup_rtol <= 0.0:
        raise ValueError("--root-dedup-rtol must be positive")

    with np.load(args.checkpoint, allow_pickle=False) as data:
        signature = str(np.asarray(data["signature"]).item())
        meta = json.loads(signature)

    required = (
        "mode", "lambda", "lambda_n", "lambda_B", "lambda_J",
        "L1", "L2", "filling", "target", "T", "ti", "t1", "t2",
        "nw", "nomega", "norb",
    )
    missing = [k for k in required if k not in meta]
    if missing:
        raise RuntimeError(f"weighted PAC checkpoint signature missing keys: {missing}")
    if str(meta["mode"]).lower() != "weighted_nbj":
        raise RuntimeError("checkpoint is not from scan_fierz_weighted_pseudo_arclength.py")

    weights = FierzWeights(
        float(meta["lambda_n"]), float(meta["lambda_B"]), float(meta["lambda_J"])
    )
    V_target = float(args.V)
    target_N = float(meta["target"])
    npc = int(meta["L1"]) * int(meta["L2"])
    states_path = args.states if args.states is not None else _infer_states_path(args.checkpoint)
    if not states_path.exists():
        raise RuntimeError(
            f"full-X branch archive not found: {states_path}. "
            "Resume or rerun scan_fierz_weighted_pseudo_arclength.py with the current code first."
        )

    params = RubyParameters(
        ti=float(meta["ti"]), t1=float(meta["t1"]), t2=float(meta["t2"]), V=0.0
    )
    geometry = ExactSmallRubyThermal(int(meta["L1"]), int(meta["L2"]), params)
    norb = int(geometry.n_sites)
    if norb != int(meta["norb"]):
        raise RuntimeError("checkpoint orbital count does not match reconstructed geometry")
    grid = MatsubaraGrid(
        nk1=1, nk2=1, nw=int(meta["nw"]), nOmega=int(meta["nomega"]), T=float(meta["T"])
    )
    h0 = np.asarray(geometry.h0, dtype=complex)[None, None]
    problem = WeightedFierzGWResidual(
        h0, geometry.interaction_pairs, weights, grid, target_N
    )
    archive = load_branch_state_archive(
        states_path, expected_signature=signature, expected_state_size=problem.codec.size
    )
    crossings = find_target_crossings(archive, V_target, atol=float(args.crossing_atol))

    print("=== All-crossing weighted n/B/J Fierz-GW + cGW + free energy ===")
    print(
        f"lambda={float(meta['lambda']):g}, weights=(n,B,J)="
        f"({weights.density:g},{weights.bond:g},{weights.current:g}), "
        f"target V={V_target:.12g}, Ntarget={target_N:g}"
    )
    print(
        f"archive={states_path}, states={archive.nstate}, "
        f"V range=[{np.min(archive.V):.9g},{np.max(archive.V):.9g}], "
        f"target candidates={len(crossings)}"
    )
    if int(archive.step[0]) > 0:
        print(
            "WARNING: this archive starts from a legacy resume point; crossings before its "
            "first archived state cannot be recovered."
        )
    if not crossings:
        raise RuntimeError(
            f"the archived branch contains no crossing of V={V_target:g}; "
            "continue the branch farther or choose a covered target"
        )

    for ic, c in enumerate(crossings):
        if c.left_index == c.right_index:
            desc = f"exact archived state index={c.left_index}, step={c.left_step}, V={c.V_left:.12g}"
        else:
            desc = (
                f"indices={c.left_index}->{c.right_index}, steps={c.left_step}->{c.right_step}, "
                f"V={c.V_left:.12g}->{c.V_right:.12g}, alpha={c.alpha:.6f}"
            )
        print(f"candidate {ic}: {desc}")

    def guarded_residual(x, V):
        ev = problem.evaluate(x, V)
        if ev.smin < float(args.screening_floor):
            raise FloatingPointError("screening matrix below numerical floor")
        return ev.residual

    opts = PACOptions(
        tol=args.newton_tol, max_newton=args.newton_max, fd_eps=args.fd_eps,
        gmres_rtol=args.gmres_rtol, gmres_maxiter=args.gmres_maxiter,
        gmres_restart=args.gmres_restart, line_search_min=args.line_search_min,
        verbose=args.verbose,
    )

    roots = []
    candidate_failures = []
    duplicate_count = 0
    for ic, crossing in enumerate(crossings):
        print(f"\n--- refine candidate {ic}/{len(crossings)-1} at exact V={V_target:g} ---")
        root = refine_fixed_parameter(
            np.asarray(crossing.x_guess, dtype=float), V_target, guarded_residual, opts=opts
        )
        if not root.converged:
            print(
                f"candidate {ic} FAILED refinement: |R_scaled|={root.residual_norm:.3e}; "
                "continuing with other crossings"
            )
            candidate_failures.append(ic)
            continue

        ev = problem.evaluate(root.x, V_target)
        duplicate_of = None
        duplicate_distance = np.nan
        for ir, item in enumerate(roots):
            dist = _root_distance(root.x, item["root"].x)
            if dist < float(args.root_dedup_rtol):
                duplicate_of = ir
                duplicate_distance = dist
                break
        if duplicate_of is not None:
            duplicate_count += 1
            roots[duplicate_of]["sources"].append(_source_segment(crossing))
            print(
                f"candidate {ic} -> duplicate of root {duplicate_of} "
                f"(relative X distance={duplicate_distance:.3e}); merged"
            )
            continue

        roots.append({
            "root": root,
            "ev": ev,
            "representative_candidate": int(ic),
            "representative_crossing": crossing,
            "sources": [_source_segment(crossing)],
        })
        print(
            f"candidate {ic} -> NEW root {len(roots)-1}: Newton={root.newton_iterations}, "
            f"GMRES={root.gmres_iterations}, |R_scaled|={root.residual_norm:.3e}, "
            f"phys={ev.physical_residual:.3e}, Nerr={ev.filling_error:+.3e}, smin={ev.smin:.3e}"
        )

    if not roots:
        raise RuntimeError(
            "all target-crossing candidates failed exact fixed-V refinement; "
            "increase solver limits or run PAC with smaller ds around the target"
        )

    definition = build_weighted_nbj_definition(
        geometry.interaction_pairs, norb, V_target, weights
    )
    operators = np.stack([
        np.asarray(geometry.pseudospin_operator(ch, (0.0, 0.0)), dtype=complex)
        for ch in CHANNELS
    ])
    Ged, chi_ed, mu_ed, ed_cache_path, ed_thermo = _load_or_compute_ed(
        args, meta, geometry, operators, grid, V_target, norb
    )
    if ed_thermo is not None:
        print(
            f"ED thermo: mu={mu_ed:+.10f}, Omega={ed_thermo['Omega']:+.12e}, "
            f"F={ed_thermo['F']:+.12e}, F/cell={ed_thermo['F_pc']:+.12e}, "
            f"N={ed_thermo['N']:.12g}"
        )

    vopts = SupercellVertexOptions(
        max_iter=args.vertex_max_iter, tol=args.vertex_tol, solver="gmres",
        gmres_restart=args.vertex_gmres_restart, verbose=args.verbose,
        momentum_backend="direct",
    )

    summary_rows = []
    for ir, item in enumerate(roots):
        root = item["root"]
        ev = item["ev"]
        crossing = item["representative_crossing"]
        bg = _make_background(problem, definition, root.x, ev, root)
        overlap = [soft_mode_overlap(K, definition, ev.soft_mode) for K in operators]
        frac = soft_mode_sector_fractions(definition, ev.soft_mode)
        thermo = evaluate_channel_gw_free_energy(
            bg, definition, h0, grid, target_particles=target_N,
            primitive_cells_per_supercell=npc,
        )

        print(f"\n=== root {ir}/{len(roots)-1} at exact V={V_target:g} ===")
        print(
            f"source candidate={item['representative_candidate']}, "
            f"segment steps={crossing.left_step}->{crossing.right_step}, "
            f"merged source crossings={len(item['sources'])}"
        )
        print(
            f"mu={bg.mu:+.10f}, smin={ev.smin:.3e}, "
            f"soft m={ev.soft_m:+d}, Omega={ev.soft_Omega:+.6g}, "
            f"sector(n,B,J)=({frac['n']:.6f},{frac['B']:.6f},{frac['J']:.6f}), "
            f"overlap(same,opp)=({overlap[0]:.6f},{overlap[1]:.6f})"
        )
        print(
            f"root {ir} GW thermo: Omega={thermo.grand_potential:+.12e}, "
            f"F={thermo.helmholtz_free_energy:+.12e}, "
            f"F/cell={thermo.free_energy_per_primitive_cell:+.12e}, "
            f"N={thermo.particle_number_actual:.12g}"
        )
        print(
            f"  LW pieces: Omega0={thermo.omega0:+.6e}, ferm={thermo.fermionic_lw:+.6e}, "
            f"Phi_tad={thermo.phi_tadpole:+.6e}, Phi_x={thermo.phi_exchange:+.6e}, "
            f"Phi_corr={thermo.phi_correlation:+.6e}"
        )

        chi_raw = np.full((2, 2), np.nan + 0j, dtype=complex)
        vertex_conv = []
        vertex_res = []
        vertex_ok = True
        for b, (name, Ksrc) in enumerate(zip(CHANNELS, operators)):
            print(f"root {ir} cGW vertex: {name} ...")
            vr = solve_channel_vertex_q0(bg, definition, Ksrc, grid, opts=vopts)
            vertex_conv.append(bool(vr.converged))
            vertex_res.append(float(vr.final_error))
            if not vr.converged:
                vertex_ok = False
                print(
                    f"WARNING: root {ir} vertex {name} failed: residual={vr.final_error:.3e}; "
                    "chi for this root will be NaN"
                )
                break
            for a, Kleft in enumerate(operators):
                chi_raw[a, b] = susceptibility_from_vertex_q0(bg.G, Kleft, vr.Gamma, grid)

        if vertex_ok:
            chi_phys, chi_imag = _physical_chi(chi_raw)
            print(
                f"root {ir} cGW chi: same={chi_phys[0,0]:+.9f}, "
                f"opposite={chi_phys[1,1]:+.9f}, hermitian-imag={chi_imag:.3e}, "
                f"vertex max residual={max(vertex_res):.3e}"
            )
        else:
            chi_phys = np.full((2, 2), np.nan, dtype=float)
            chi_imag = np.nan

        gerr = np.nan
        gerr_low = np.nan
        chi_err = np.nan
        F_minus_ed = np.nan
        F_minus_ed_pc = np.nan
        if Ged is not None:
            gerr, gerr_low = _green_errors(bg.G, Ged, grid, args.low_count)
            if vertex_ok:
                chi_err = _relerr(chi_phys, chi_ed)
            if ed_thermo is not None:
                F_minus_ed = thermo.helmholtz_free_energy - ed_thermo["F"]
                F_minus_ed_pc = thermo.free_energy_per_primitive_cell - ed_thermo["F_pc"]
            print(
                f"root {ir} ED comparison: Gerr={gerr:.6e}, Gerr_low={gerr_low:.6e}, "
                f"chi_ED=({chi_ed[0,0]:.9f},{chi_ed[1,1]:.9f}), chi_relerr={chi_err:.6e}, "
                f"F_GW-F_ED={F_minus_ed:+.6e} ({F_minus_ed_pc:+.6e}/cell)"
            )

        sigma_static, sigma_c, mu = problem.codec.decode(root.x)
        item.update({
            "bg": bg,
            "frac": frac,
            "overlap": np.asarray(overlap, dtype=float),
            "chi": np.asarray(chi_phys),
            "chi_raw": np.asarray(chi_raw),
            "chi_imag": float(chi_imag),
            "vertex_converged": np.asarray(vertex_conv, dtype=bool),
            "vertex_residual": np.asarray(vertex_res, dtype=float),
            "gerr": float(gerr),
            "gerr_low": float(gerr_low),
            "chi_err": float(chi_err),
            "sigma_static": np.asarray(sigma_static),
            "sigma_c": np.asarray(sigma_c),
            "mu": float(mu),
            "thermo": thermo,
            "F_minus_ed": float(F_minus_ed),
            "F_minus_ed_pc": float(F_minus_ed_pc),
        })
        summary_rows.append({
            "root": ir,
            "representative_candidate": item["representative_candidate"],
            "source_crossings": len(item["sources"]),
            "left_step": crossing.left_step,
            "right_step": crossing.right_step,
            "V_left": f"{crossing.V_left:.16g}",
            "V_right": f"{crossing.V_right:.16g}",
            "mu": f"{mu:.16g}",
            "smin": f"{ev.smin:.8e}",
            "soft_n": f"{frac['n']:.8e}",
            "soft_B": f"{frac['B']:.8e}",
            "soft_J": f"{frac['J']:.8e}",
            "chi_same": f"{chi_phys[0,0]:.16g}",
            "chi_opposite": f"{chi_phys[1,1]:.16g}",
            "Gerr": f"{gerr:.8e}",
            "Gerr_low": f"{gerr_low:.8e}",
            "chi_relerr": f"{chi_err:.8e}",
            "Omega_GW": f"{thermo.grand_potential:.16g}",
            "F_GW": f"{thermo.helmholtz_free_energy:.16g}",
            "F_GW_per_cell": f"{thermo.free_energy_per_primitive_cell:.16g}",
            "Omega0": f"{thermo.omega0:.16g}",
            "LW_fermionic": f"{thermo.fermionic_lw:.16g}",
            "Phi_tad": f"{thermo.phi_tadpole:.16g}",
            "Phi_x": f"{thermo.phi_exchange:.16g}",
            "Phi_corr": f"{thermo.phi_correlation:.16g}",
            "Omega_ED": f"{ed_thermo['Omega']:.16g}" if ed_thermo is not None else "nan",
            "F_ED": f"{ed_thermo['F']:.16g}" if ed_thermo is not None else "nan",
            "F_ED_per_cell": f"{ed_thermo['F_pc']:.16g}" if ed_thermo is not None else "nan",
            "F_minus_ED": f"{F_minus_ed:.16g}",
            "F_minus_ED_per_cell": f"{F_minus_ed_pc:.16g}",
            "DeltaF_branch": "nan",
            "DeltaF_branch_per_cell": "nan",
            "physical_residual": f"{ev.physical_residual:.8e}",
            "newton_iter": root.newton_iterations,
            "gmres_iter": root.gmres_iterations,
        })

    # Thermodynamic ranking is meaningful among roots of the same lambda/functional.
    Fvals = np.asarray([item["thermo"].helmholtz_free_energy for item in roots], dtype=float)
    Fmin = float(np.min(Fvals))
    delta_branch = Fvals - Fmin
    for ir, (item, row) in enumerate(zip(roots, summary_rows)):
        item["deltaF_branch"] = float(delta_branch[ir])
        item["deltaF_branch_pc"] = float(delta_branch[ir] / npc)
        row["DeltaF_branch"] = f"{delta_branch[ir]:.16g}"
        row["DeltaF_branch_per_cell"] = f"{delta_branch[ir] / npc:.16g}"

    print(
        "\nNOTE: chi uses the existing analytic finite-Matsubara cGW convention; the pure-J "
        "finite-source test showed a ~5.7% FDT mismatch at nw=55 pending tail-derivative correction."
    )
    print(
        "NOTE: DeltaF_branch ranks roots only within this fixed lambda GW functional.  Absolute "
        "F_GW-F_ED is also reported, but changing lambda changes the truncated Phi functional, so "
        "cross-lambda free-energy ordering is itself Fierz dependent."
    )

    args.out.mkdir(parents=True, exist_ok=True)
    lamtag = _lambda_tag(float(meta["lambda"]))
    tag = (
        f"V{V_target:g}_fill{float(meta['filling']):g}_{int(meta['L1'])}x{int(meta['L2'])}_"
        f"nbj_lam{lamtag}_allcrossings"
    )
    outfile = args.out / f"{tag}.npz"
    csvfile = args.out / f"{tag}.csv"

    max_sources = max(len(item["sources"]) for item in roots)
    source_segments_json = np.asarray(
        [json.dumps(item["sources"], sort_keys=True) for item in roots], dtype=str
    )
    rep = [item["representative_crossing"] for item in roots]

    np.savez_compressed(
        outfile,
        V=V_target,
        mode=np.asarray("weighted_nbj"),
        lambda_n=float(weights.density), lambda_B=float(weights.bond), lambda_J=float(weights.current),
        lambda_value=float(meta["lambda"]), filling=float(meta["filling"]), T=float(meta["T"]),
        L1=int(meta["L1"]), L2=int(meta["L2"]), omega=np.asarray(grid.omega), Omega=np.asarray(grid.Omega),
        nroot=len(roots), candidate_count=len(crossings), failed_candidate_count=len(candidate_failures),
        duplicate_candidate_count=duplicate_count, failed_candidates=np.asarray(candidate_failures, dtype=np.int64),
        representative_candidate=np.asarray([item["representative_candidate"] for item in roots], dtype=np.int64),
        crossing_left_index=np.asarray([c.left_index for c in rep], dtype=np.int64),
        crossing_right_index=np.asarray([c.right_index for c in rep], dtype=np.int64),
        crossing_left_step=np.asarray([c.left_step for c in rep], dtype=np.int64),
        crossing_right_step=np.asarray([c.right_step for c in rep], dtype=np.int64),
        crossing_V_left=np.asarray([c.V_left for c in rep], dtype=float),
        crossing_V_right=np.asarray([c.V_right for c in rep], dtype=float),
        crossing_alpha=np.asarray([c.alpha for c in rep], dtype=float),
        source_segments_json=source_segments_json, max_source_crossings=max_sources,
        X_target=np.stack([item["root"].x for item in roots]),
        Sigma_static=np.stack([item["sigma_static"] for item in roots]),
        Sigma_c=np.stack([item["sigma_c"] for item in roots]),
        G=np.stack([item["bg"].G for item in roots]),
        P=np.stack([item["bg"].P for item in roots]),
        W=np.stack([item["bg"].W for item in roots]),
        rho=np.stack([item["bg"].rho for item in roots]),
        mu=np.asarray([item["mu"] for item in roots], dtype=float),
        smin=np.asarray([item["ev"].smin for item in roots], dtype=float),
        soft_m=np.asarray([item["ev"].soft_m for item in roots], dtype=np.int64),
        soft_Omega=np.asarray([item["ev"].soft_Omega for item in roots], dtype=float),
        soft_mode=np.stack([item["ev"].soft_mode for item in roots]),
        soft_sector_n=np.asarray([item["frac"]["n"] for item in roots], dtype=float),
        soft_sector_B=np.asarray([item["frac"]["B"] for item in roots], dtype=float),
        soft_sector_J=np.asarray([item["frac"]["J"] for item in roots], dtype=float),
        soft_overlap_same=np.asarray([item["overlap"][0] for item in roots], dtype=float),
        soft_overlap_opposite=np.asarray([item["overlap"][1] for item in roots], dtype=float),
        fixed_scaled_residual=np.asarray([item["root"].residual_norm for item in roots], dtype=float),
        physical_residual=np.asarray([item["ev"].physical_residual for item in roots], dtype=float),
        filling_error=np.asarray([item["ev"].filling_error for item in roots], dtype=float),
        newton_iterations=np.asarray([item["root"].newton_iterations for item in roots], dtype=np.int64),
        gmres_iterations=np.asarray([item["root"].gmres_iterations for item in roots], dtype=np.int64),
        chi=np.stack([item["chi"] for item in roots]), chi_raw=np.stack([item["chi_raw"] for item in roots]),
        vertex_converged=np.asarray([bool(np.all(item["vertex_converged"])) for item in roots], dtype=bool),
        vertex_max_residual=np.asarray([
            float(np.max(item["vertex_residual"])) if item["vertex_residual"].size else np.nan
            for item in roots
        ], dtype=float),
        omega0_gw=np.asarray([item["thermo"].omega0 for item in roots], dtype=float),
        lw_fermionic_gw=np.asarray([item["thermo"].fermionic_lw for item in roots], dtype=float),
        phi_tad_gw=np.asarray([item["thermo"].phi_tadpole for item in roots], dtype=float),
        phi_exchange_gw=np.asarray([item["thermo"].phi_exchange for item in roots], dtype=float),
        phi_corr_gw=np.asarray([item["thermo"].phi_correlation for item in roots], dtype=float),
        omega_gw=np.asarray([item["thermo"].grand_potential for item in roots], dtype=float),
        F_gw=np.asarray([item["thermo"].helmholtz_free_energy for item in roots], dtype=float),
        F_gw_per_cell=np.asarray([item["thermo"].free_energy_per_primitive_cell for item in roots], dtype=float),
        deltaF_branch=np.asarray([item["deltaF_branch"] for item in roots], dtype=float),
        deltaF_branch_per_cell=np.asarray([item["deltaF_branch_pc"] for item in roots], dtype=float),
        N_gw=np.asarray([item["thermo"].particle_number_actual for item in roots], dtype=float),
        G_ed=np.asarray(Ged) if Ged is not None else np.empty((0,), dtype=complex),
        chi_ed=np.asarray(chi_ed) if chi_ed is not None else np.empty((0,), dtype=float), mu_ed=float(mu_ed),
        omega_ed=float(ed_thermo["Omega"]) if ed_thermo is not None else np.nan,
        F_ed=float(ed_thermo["F"]) if ed_thermo is not None else np.nan,
        F_ed_per_cell=float(ed_thermo["F_pc"]) if ed_thermo is not None else np.nan,
        N_ed=float(ed_thermo["N"]) if ed_thermo is not None else np.nan,
        F_minus_ed=np.asarray([item["F_minus_ed"] for item in roots], dtype=float),
        F_minus_ed_per_cell=np.asarray([item["F_minus_ed_pc"] for item in roots], dtype=float),
        gerr=np.asarray([item["gerr"] for item in roots], dtype=float),
        gerr_low=np.asarray([item["gerr_low"] for item in roots], dtype=float),
        chi_err=np.asarray([item["chi_err"] for item in roots], dtype=float),
        ed_cache_path=np.asarray(ed_cache_path), source_checkpoint=np.asarray(str(args.checkpoint)),
        source_states=np.asarray(str(states_path)),
    )
    _write_summary_csv(csvfile, summary_rows)

    print("\n=== target-V all-root summary ===")
    print(
        f"candidates={len(crossings)}, distinct roots={len(roots)}, "
        f"duplicates merged={duplicate_count}, refinement failures={len(candidate_failures)}"
    )
    for row in summary_rows:
        print(
            f"root {row['root']}: steps={row['left_step']}->{row['right_step']}, "
            f"chi(same,opp)=({row['chi_same']},{row['chi_opposite']}), "
            f"F/cell={row['F_GW_per_cell']}, DeltaF_branch/cell={row['DeltaF_branch_per_cell']}, "
            f"F-F_ED/cell={row['F_minus_ED_per_cell']}, Gerr={row['Gerr']}"
        )
    print(f"saved {outfile}")
    print(f"saved {csvfile}")


if __name__ == "__main__":
    main()
