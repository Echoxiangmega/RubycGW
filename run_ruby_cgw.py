#!/usr/bin/env python3
"""Staged Ruby response calculation: G0G0 -> GG -> GW+MT -> full cGW."""

import numpy as np

from rubycgw import (
    RubyParameters,
    MatsubaraGrid,
    GWOptions,
    VertexOptions,
    build_interaction,
    eta_vertices,
    solve_gw,
    solve_noninteracting,
    solve_vertex_q0,
    chi_eta,
)


def _real_if_close(z, tol=1e-10):
    z = complex(z)
    return z.real if abs(z.imag) < tol else z


def _print_response_table(rows):
    print("\n=== staged eta susceptibilities, q=(0,0) ===")
    print(f"{'level':<18s} {'opposite (+)':>18s} {'same (-)':>18s} {'same-opposite':>18s}")
    print("-" * 76)
    for name, cp, cm in rows:
        cp = _real_if_close(cp)
        cm = _real_if_close(cm)
        d = _real_if_close(cm - cp)
        print(f"{name:<18s} {cp:>18.10g} {cm:>18.10g} {d:>18.10g}")


def _print_vertex_norms(label, vertex):
    print(f"\n--- full cGW vertex diagnostics: {label} ---")
    print("converged       :", vertex.converged)
    print("iterations      :", vertex.iterations)
    print("max |Gamma_H|   :", np.max(np.abs(vertex.Gamma_H)))
    print("max |Gamma_MT|  :", np.max(np.abs(vertex.Gamma_MT)))
    print("max |Gamma_AL1| :", np.max(np.abs(vertex.Gamma_AL1)))
    print("max |Gamma_AL2| :", np.max(np.abs(vertex.Gamma_AL2)))


def main():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.10)
    grid = MatsubaraGrid(nk1=4, nk2=4, nw=16, nOmega=6, T=0.05)
    target_filling = 2.0

    _, _, K_plus, K_minus = eta_vertices()

    bare = solve_noninteracting(
        params,
        grid,
        mu=0.0,
        target_filling=target_filling,
    )
    chi_plus_g0 = chi_eta(bare.G0, K_plus, grid)
    chi_minus_g0 = chi_eta(bare.G0, K_minus, grid)

    print("=== noninteracting reference ===")
    print("mu0:", bare.mu)
    print("density per site:", bare.density)
    print("total filling:", np.sum(bare.density))

    gw_opts = GWOptions(
        mu=bare.mu,
        target_filling=target_filling,
        max_iter=100,
        tol=1e-8,
        mixing=0.20,
        verbose=True,
    )
    gw = solve_gw(params, grid, gw_opts)

    print("\n=== GW summary ===")
    print("converged:", gw.converged)
    print("iterations:", gw.iterations)
    print("mu_GW:", gw.mu)
    print("density per site:", gw.density)
    print("total filling:", np.sum(gw.density))

    chi_plus_gg = chi_eta(gw.G, K_plus, grid)
    chi_minus_gg = chi_eta(gw.G, K_minus, grid)
    Vq0 = build_interaction(grid.qmesh(), params)[0, 0]

    mt_opts = VertexOptions(
        max_iter=150,
        tol=1e-8,
        mixing=0.20,
        include_hartree=True,
        include_mt=True,
        include_al=False,
        verbose=False,
    )

    print("\n=== solve GW+MT vertices ===")
    vp_mt = solve_vertex_q0(gw.G, gw.W, Vq0, K_plus, grid, mt_opts)
    vm_mt = solve_vertex_q0(gw.G, gw.W, Vq0, K_minus, grid, mt_opts)
    chi_plus_mt = chi_eta(gw.G, K_plus, grid, Gamma=vp_mt.Gamma)
    chi_minus_mt = chi_eta(gw.G, K_minus, grid, Gamma=vm_mt.Gamma)
    print("plus/opposite vertex converged:", vp_mt.converged, "iterations:", vp_mt.iterations)
    print("minus/same vertex converged    :", vm_mt.converged, "iterations:", vm_mt.iterations)

    full_opts = VertexOptions(
        max_iter=150,
        tol=1e-8,
        mixing=0.20,
        include_hartree=True,
        include_mt=True,
        include_al=True,
        verbose=False,
    )

    print("\n=== solve full cGW vertices ===")
    # The full solution is very close to the already converged MT solution for
    # the present model, so use MT as a warm start instead of restarting at K.
    vp_full = solve_vertex_q0(
        gw.G, gw.W, Vq0, K_plus, grid, full_opts,
        initial_gamma=vp_mt.Gamma,
    )
    vm_full = solve_vertex_q0(
        gw.G, gw.W, Vq0, K_minus, grid, full_opts,
        initial_gamma=vm_mt.Gamma,
    )
    chi_plus_full = chi_eta(gw.G, K_plus, grid, Gamma=vp_full.Gamma)
    chi_minus_full = chi_eta(gw.G, K_minus, grid, Gamma=vm_full.Gamma)

    rows = [
        ("G0G0 (bare)", chi_plus_g0, chi_minus_g0),
        ("GG", chi_plus_gg, chi_minus_gg),
        ("GW + MT", chi_plus_mt, chi_minus_mt),
        ("full cGW", chi_plus_full, chi_minus_full),
    ]
    _print_response_table(rows)

    _print_vertex_norms("plus = physical opposite", vp_full)
    _print_vertex_norms("minus = physical same", vm_full)

    print("\n=== interpretation labels ===")
    print("eta_plus  = physical opposite circulation")
    print("eta_minus = physical same circulation")
    print("positive same-opposite means the same-current channel has larger susceptibility")


if __name__ == "__main__":
    main()
