import subprocess
import sys


def test_jf_q_scan_help_smoke():
    proc = subprocess.run(
        [sys.executable, "scan_cluster_ed_gw_jf_q.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Jacob" in proc.stdout or "pseudospin" in proc.stdout
    assert "--bath-rank" in proc.stdout
