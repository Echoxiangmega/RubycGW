import subprocess
import sys


def test_jf_bath_convergence_help_smoke():
    proc = subprocess.run(
        [sys.executable, "scan_cluster_ed_gw_jf_bath_convergence.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "bath" in proc.stdout.lower()
    assert "--bath-rank" in proc.stdout
