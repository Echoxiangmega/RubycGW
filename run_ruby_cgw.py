#!/usr/bin/env python3
"""Small-grid reference run for Ruby GW -> GG bubble -> full q=0 cGW."""

import numpy as np

from rubycgw import (
    RubyParameters,
    MatsubaraGrid,
    GWOptions,
    VertexOptions,
    build_interaction,
    eta_vertices,
    solve_gw,
    solve_vertex_q0,
    chi_eta,
)


def main():
    # Keep the defaults deliberately small: this is a transparent reference
    # implementation. Increase nk/nw/nOmega only after the checks pass.
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.10)
    grid = MatsubaraGrid(nk1=4, nk2=4, nw=16, nOmega=6, T=0.05)

    gw_opts = GWOptions(
        mu=0.0,
        target_filling=2.0,
        max_iter=80,
        tol=1e-7,
        mixing=0.20,
        verbose=True,
    )

    _, _, K_plus, K_minus = eta_vertices()
    gw = solve_gw(params, grid, gw_opts)

    print("\n=== GW summary ===")
    print("converged:", gw.converged)
    print("iterations:", gw.iterations)
    print("mu:", gw.mu)
    print("density per site:", gw.density)
    print("total filling:", np.sum(gw.density))

    chi_plus_gg = chi_eta(gw.G, K_plus, grid)
    chi_minus_gg = chi_eta(gw.G, K_minus, grid)
    print("\n=== dressed GG bubbles ===")
    print("chi_plus  (physical opposite) =", chi_plus_gg)
    print("chi_minus (physical same)     =", chi_minus_gg)
    print("same - opposite               =", chi_minus_gg - chi_plus_gg)

    Vq0 = build_interaction(grid.qmesh(), params)[0, 0]
    vopts = VertexOptions(
        max_iter=60,
        tol=1e-7,
        mixing=0.20,
        include_hartree=True,
        include_mt=True,
        include_al=True,
        verbose=True,
    )

    print("\n=== solve cGW vertex: plus / physical opposite ===")
    vp = solve_vertex_q0(gw.G, gw.W, Vq0, K_plus, grid, vopts)
    chi_plus_cgw = chi_eta(gw.G, K_plus, grid, Gamma=vp.Gamma)

    print("\n=== solve cGW vertex: minus / physical same ===")
    vm = solve_vertex_q0(gw.G, gw.W, Vq0, K_minus, grid, vopts)
    chi_minus_cgw = chi_eta(gw.G, K_minus, grid, Gamma=vm.Gamma)

    print("\n=== cGW susceptibilities ===")
    print("chi_plus  (physical opposite) =", chi_plus_cgw)
    print("chi_minus (physical same)     =", chi_minus_cgw)
    print("same - opposite               =", chi_minus_cgw - chi_plus_cgw)
    print("Hartree vertex max (+):", np.max(np.abs(vp.Gamma_H)))
    print("Hartree vertex max (-):", np.max(np.abs(vm.Gamma_H)))


if __name__ == "__main__":
    main()
