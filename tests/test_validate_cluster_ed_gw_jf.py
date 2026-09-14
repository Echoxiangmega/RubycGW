import subprocess
import sys


def test_validate_cluster_ed_gw_jf_help_smoke():
    proc = subprocess.run(
        [sys.executable, "validate_cluster_ed_gw_jf.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "finite-source" in proc.stdout.lower() or "finite sources" in proc.stdout.lower()
    assert "--bath-rank" in proc.stdout
