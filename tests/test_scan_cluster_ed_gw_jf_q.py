import subprocess
import sys


def test_jf_q_scan_help_smoke():
    proc = subprocess.run(
        [sys.executable, "scripts/run_jf.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Jacob" in proc.stdout or "pseudospin" in proc.stdout
    assert "--bath-rank" in proc.stdout
