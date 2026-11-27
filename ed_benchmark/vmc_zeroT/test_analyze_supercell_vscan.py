"""Regression checks for phase-level energy comparisons in quarter V scans."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ed_benchmark.vmc_zeroT import analyze_supercell_vscan as analysis


def row(seed, phase, V, energy):
    return {
        "branch_seed":seed,
        "phase_label":phase,
        "scan_V":V,
        "mc_E":energy,
        "mc_E_error":0.01,
    }


class ComparisonGroupingTest(unittest.TestCase):
    def test_merge_old_phases_but_keep_four_new_ansatzes(self):
        rows=[
            row("normal","CO",2.5,-10.0),
            row("sd","CO",2.5,-10.4),
            row("lc_same","LC_same",2.5,-10.2),
            row("sdw","SDW_M_CO",2.5,-10.5),
            row("pf_s","Pf_s_candidate",2.5,-9.5),
            row("pf_dplus","Pf_d_chiral_candidate",2.5,-10.3),
            row("pf_dminus","Pf_d_chiral_candidate",2.5,-10.31),
            row("normal","LC_same",2.7,-11.0),
            row("sd","CO",2.7,-10.8),
        ]
        table=analysis._comparison_table(rows)
        at_v=[r for r in table if r["V"]==2.5]
        self.assertEqual(len(at_v),6)
        self.assertEqual(
            {r["candidate"] for r in at_v},
            {"phase:CO","phase:LC_same","ansatz:sdw","ansatz:pf_s",
             "ansatz:pf_dplus","ansatz:pf_dminus"},
        )
        co=next(r for r in at_v if r["candidate"]=="phase:CO")
        self.assertEqual(co["selected_seed"],"sd")
        self.assertEqual(co["mc_E"],-10.4)

        ranking=analysis._phase_rows(rows,2,expected_seeds=(
            "normal","sd","lc_same","sdw","pf_s","pf_dplus","pf_dminus"
        ))
        self.assertEqual(ranking[0]["winner_candidate"],"ansatz:sdw")
        self.assertEqual(ranking[0]["second_candidate"],"phase:CO")
        self.assertEqual(ranking[0]["n_competitors"],6)
        self.assertEqual(ranking[0]["n_phases"],5)
        self.assertTrue(ranking[1]["incomplete"])

        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            analysis._energy_plot(rows,out,4,4)
            analysis._relative_energy_plot(rows,out,4,4)
            analysis._reference_energy_plots(rows,out,4,4)
            analysis._phase_envelope_plot(rows,out,4,4)
            for name in ("energy_branches.png","energy_relative.png",
                         "energy_reference_per_site.png",
                         "energy_phase_envelopes.png"):
                self.assertTrue((out/name).is_file(),name)


if __name__=="__main__":
    unittest.main()
