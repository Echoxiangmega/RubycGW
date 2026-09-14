import subprocess
import sys


def test_run_cluster_ed_gw_accel_restart_help():
    proc = subprocess.run(
        [sys.executable, "run_cluster_ed_gw_accel.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--restart-from" in proc.stdout
    assert "continue from a saved cluster-ED+GW" in proc.stdout
