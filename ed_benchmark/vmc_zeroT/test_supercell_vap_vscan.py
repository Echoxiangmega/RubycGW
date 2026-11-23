#!/usr/bin/env python3
"""Unit tests for run_supercell_vap_vscan.py (no NetKet required).

Run:
    python -m unittest ed_benchmark/vmc_zeroT/test_supercell_vap_vscan.py -v
or:
    python ed_benchmark/vmc_zeroT/test_supercell_vap_vscan.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_supercell_vap_vscan as driver


class TestFastVapSeedArray(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.out = self.root / "vap"
        self.fake_runner = self.root / "mock_fast_vap.py"
        self.fake_runner.write_text("# synthetic runner\n", encoding="utf-8")
        self._saved_runner = driver.FAST_VAP
        driver.FAST_VAP = self.fake_runner
        self.addCleanup(setattr, driver, "FAST_VAP", self._saved_runner)

    def args(self, *extra):
        return [
            "--L1", "3", "--L2", "3", "--nf", "54",
            "--t", "1", "--t1", "-0.5", "--t2", "-0.5",
            "--seeds", "sd", "isd", "lc_same", "lc_opposite",
            "--scan-root", str(self.raw),
            "--output-root", str(self.out),
            "--mode", "continuation", "--direction", "down",
            *extra,
        ]

    def raw_checkpoint(self, seed, v):
        directory = (self.raw / "L3x3" / f"seed_{seed}" /
                     "continuation_down" / driver.vtag(v) /
                     f"large_raw_backflow__{seed}")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "final_params.npz"
        path.touch()
        return path

    def test_v_tags_match_raw(self):
        self.assertEqual(driver.vtag(3.6), "V_p3p6")
        self.assertEqual(driver.vtag(4.0), "V_p4")
        self.assertEqual(driver.vtag(-0.25), "V_m0p25")

    def test_array_index_selects_seed(self):
        args, _ = driver.parse_args(self.args("--V", "3.6", "--array-index", "2"))
        self.assertEqual(driver.select_seed(args), (2, "lc_same"))
        with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "3"}):
            args, _ = driver.parse_args(self.args("--V", "3.6"))
            self.assertEqual(driver.select_seed(args), (3, "lc_opposite"))

    def test_multi_seed_needs_index_outside_slurm(self):
        with patch.dict(os.environ, {}, clear=True):
            args, _ = driver.parse_args(self.args("--V", "3.6"))
            with self.assertRaisesRegex(ValueError, "array-index"):
                driver.select_seed(args)

    def test_independent_seed_paths_and_passthrough(self):
        self.raw_checkpoint("isd", 3.6)
        self.raw_checkpoint("isd", 3.5)
        commands = []

        def fake_run(cmd, check=True):
            self.assertTrue(check)
            commands.append(cmd)
            return None

        with patch("run_supercell_vap_vscan.subprocess.run", side_effect=fake_run):
            result = driver.run(self.args(
                "--V-values", "3.5", "3.6",
                "--array-index", "1",
                "--sampler", "projected-cache",
                "--sweep-size", "54", "--spatial-sweep-size", "27",
                "--seed-amplitude", "0.09",
                "--irreps", "A1", "A2", "E",
            ))
        self.assertEqual(result, 0)
        self.assertEqual(len(commands), 2)
        for cmd in commands:
            self.assertIn("--seed", cmd)
            self.assertEqual(cmd[cmd.index("--seed") + 1], "isd")
            self.assertIn("--sampler", cmd)
            self.assertEqual(cmd[cmd.index("--sampler") + 1], "projected-cache")
            self.assertIn("--seed-amplitude", cmd)
            self.assertIn("--raw-stage", cmd)
            self.assertIn("--output", cmd)
        # downward sweep must run V=3.6 first
        self.assertEqual(commands[0][commands[0].index("--V") + 1], "3.6")
        self.assertEqual(commands[1][commands[1].index("--V") + 1], "3.5")
        status = (self.out / "L3x3" / "seed_isd" /
                  "continuation_down" / "vap_scan_status.csv")
        self.assertTrue(status.exists())
        with status.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["status"] for r in rows], ["ok", "ok"])

    def test_skip_missing_raw_and_fail_if_all_missing(self):
        self.raw_checkpoint("sd", 3.6)
        calls = []
        with patch("run_supercell_vap_vscan.subprocess.run",
                   side_effect=lambda cmd, check=True: calls.append(cmd)):
            ret = driver.run(self.args(
                "--array-index", "0", "--V-values", "3.6", "3.5",
                "--missing-raw", "skip",
            ))
        self.assertEqual(ret, 0)
        self.assertEqual(len(calls), 1)
        with patch("run_supercell_vap_vscan.subprocess.run") as sp:
            ret = driver.run(self.args("--array-index", "3", "--V", "3.6"))
        self.assertEqual(ret, 2)
        sp.assert_not_called()

    def test_no_cross_seed_parallel_ambiguity(self):
        with self.assertRaisesRegex(ValueError, "Duplicate seeds"):
            args, _ = driver.parse_args([
                "--L1", "3", "--L2", "3", "--V", "3.6",
                "--seeds", "sd", "sd", "--scan-root", str(self.raw),
                "--output-root", str(self.out), "--array-index", "0",
            ])
            driver.select_seed(args)

    def test_no_cli_collision(self):
        with self.assertRaisesRegex(ValueError, "Duplicate/unsupported"):
            driver.parse_args(self.args(
                "--V", "3.6", "--array-index", "0",
                "--raw-stage", "/unwanted/checkpoint",
            ))

    def test_dry_run_does_not_spawn(self):
        self.raw_checkpoint("sd", 3.6)
        with patch("run_supercell_vap_vscan.subprocess.run") as sp:
            ret = driver.run(self.args("--array-index", "0", "--V", "3.6",
                                        "--dry-run", "--sampler", "standard"))
        self.assertEqual(ret, 0)
        sp.assert_not_called()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
