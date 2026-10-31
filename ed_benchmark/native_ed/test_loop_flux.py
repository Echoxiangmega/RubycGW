#!/usr/bin/env python3
"""Lightweight tests for the mixed-shape ED loop-phase plots.

Run: python test_loop_flux.py
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json

import numpy as np

from geometry import RubyPBC
from plot_current_pattern import plaquette_data, selected_site_positions
from plot_loop_flux import (
    channel_selection_ambiguous,
    draw_map,
    fock_flux_for_loop,
    rows_for_loops,
    summarize_shape,
)


class TestLoopFlux(unittest.TestCase):
    def test_all_shapes_and_zero_phase(self):
        for cluster in ("2x2", "ring3"):
            g = RubyPBC(cluster, t1=0.5, t2=0.5)
            coords = selected_site_positions(g)[0]
            # All rho positive: every CCW loop phase must vanish.
            rho = np.ones(len(g.edges), dtype=complex) * 0.25
            j = np.zeros(len(g.edges), dtype=float)
            loops = plaquette_data(g, rho, j, coords)
            rows = rows_for_loops({"loop_data": loops})
            lengths = {row["length"] for row in rows}
            self.assertEqual(lengths, {3, 4, 6})
            self.assertTrue(all(row["valid_phase"] for row in rows))
            self.assertTrue(all(abs(row["phi_rad"]) < 1e-12 for row in rows))
            for length in (3, 4, 6):
                stats = summarize_shape(rows, length)
                self.assertGreater(stats["count"], 0)
                self.assertEqual(stats["max_abs_phi"], 0.0)

    def test_direction_and_weak_product(self):
        g = RubyPBC("2x2", t1=0.5, t2=0.5)
        coords = selected_site_positions(g)[0]
        rho = np.ones(len(g.edges), dtype=complex) * 0.25
        rho[0] *= np.exp(0.33j)
        loops = plaquette_data(g, rho, np.zeros(len(rho)), coords)
        rows = rows_for_loops({"loop_data": loops})
        self.assertTrue(any(abs(r["phi_rad"]) > 0.1 for r in rows))
        # All phases should be disabled at deliberately excessive threshold.
        undefined = rows_for_loops({"loop_data": loops}, min_product=100.0)
        self.assertTrue(all(not row["valid_phase"] for row in undefined))

    def test_selection_ambiguity(self):
        def dat(e):
            return {
                "selected_channel": "LC_same",
                "projection": {"LC_same": {"eigenvalues": e}},
            }
        self.assertTrue(channel_selection_ambiguous(dat([0.0, 0.0])))
        self.assertTrue(channel_selection_ambiguous(dat([0.1, 0.1])))
        self.assertFalse(channel_selection_ambiguous(dat([-0.1, 0.1])))


    def test_effective_fock_flux(self):
        g = RubyPBC("2x2", t1=0.5, t2=0.5)
        coords = selected_site_positions(g)[0]
        rho = np.ones(len(g.edges), dtype=complex) * 0.20
        # Complex nearest-neighbour contraction only on one triangle edge.
        one_t = next(k for k, e in enumerate(g.edges) if e.kind == "t")
        rho[one_t] = 0.20 + 0.15j
        loops = plaquette_data(g, rho, np.zeros(len(rho)), coords)
        dat = {"loop_data": loops, "rho_edges": rho}

        zero = rows_for_loops(dat, g=g, V=0.0, flux_mode="fock")
        self.assertTrue(all(r["valid_phase"] for r in zero))
        self.assertTrue(all(abs(r["phi_rad"]) < 1e-12 for r in zero))

        finite = rows_for_loops(dat, g=g, V=1.0, flux_mode="fock")
        self.assertTrue(any(abs(r["phi_rad"]) > 0.01
                            for r in finite if r["length"] == 3))
        self.assertTrue(all(abs(r["phi_rad"]) < 1e-12
                            for r in finite if r["length"] == 6))
        self.assertTrue(all(isinstance(r["valid_phase"], bool) for r in finite))
        # Hermitian conjugation reverses each oriented loop phase.
        dat_reversed = {"loop_data": loops, "rho_edges": np.conjugate(rho)}
        for lp in loops:
            phase, _ = fock_flux_for_loop(g, dat, list(lp["sites"]), 1.0)
            reversed_phase, _ = fock_flux_for_loop(
                g, dat_reversed, list(lp["sites"]), 1.0
            )
            self.assertAlmostEqual(phase, -reversed_phase, places=12)

    def test_ambiguity_serializes_to_json(self):
        dat = {
            "selected_channel": "LC_same",
            "projection": {"LC_same": {"eigenvalues": np.array([0., 0.])}},
        }
        flagged = channel_selection_ambiguous(dat)
        self.assertIs(type(flagged), bool)
        self.assertEqual(json.loads(json.dumps({"nonunique": flagged})),
                         {"nonunique": True})

    def test_plot_writes_png(self):
        g = RubyPBC("2x2", t1=0.5, t2=0.5)
        coords = selected_site_positions(g)[0]
        rho = np.ones(len(g.edges), dtype=complex) * 0.25
        loops = plaquette_data(g, rho, np.zeros(len(rho)), coords)
        rows = rows_for_loops({"loop_data": loops})
        dat = {"selected_channel": "LC_same", "J_selected": 0.0}
        with TemporaryDirectory() as temp:
            for mode in ("abs", "signed"):
                dest = Path(temp) / f"{mode}.png"
                draw_map(g, rows, dat, dest, mode, ambiguous=True, flux_mode="correlation")
                self.assertGreater(dest.stat().st_size, 1024)


if __name__ == "__main__":
    unittest.main()
