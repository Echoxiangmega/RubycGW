"""Readable benchmark rankings must stay numerically honest."""
import csv
from rubycgw.benchmark_report import make_comparison_scorecards


def test_report_separates_green_and_susceptibility_without_false_score(tmp_path):
    rows = [
        dict(method="ED",quantity="chi[x_even](Q)",value=2, status="ok"),
        dict(method="ED",quantity="chi[y_even](Q)",value=4, status="ok"),
        dict(method="ED",quantity="chi[z_same](Q)",value=3, status="ok"),
        dict(method="ED",quantity="chi[z_opposite](Q)",value=2.5,status="ok"),
        dict(method="GW[bj]",quantity="G_rel_Frobenius_K",value=.03,status="ok"),
        dict(method="GW[bj]",quantity="G_tau_curve_relative_L2_error",value=.02,status="ok"),
        dict(method="GW[bj]",quantity="total_method_runtime_seconds",value=60,status="ok"),
        dict(method="post-GW[bj]",quantity="G_rel_Frobenius_K",value=.025,status="ok"),
        dict(method="post-GW[bj]",quantity="G_tau_curve_relative_L2_error",value=.023,status="ok"),
        dict(method="post-GW[bj]",quantity="total_method_runtime_seconds",value=500,status="ok"),
        dict(method="cGW[bj]",quantity="total_method_runtime_seconds",value=200,status="ok"),
    ]
    for channel, value in (
        ("x_even",2.05),("y_even",3.9),("z_same",3.1),("z_opposite",2.6)
    ):
        rows.append(dict(method="cGW[bj]",quantity=f"chi[{channel}](Q)",
                         value=value,status="ok"))
    wide=make_comparison_scorecards(rows,tmp_path)
    by={r["method"]:r for r in wide}
    assert abs(by["cGW[bj]"]["chi_mean_4_error_pct"]-
               100*((.05/2+.1/4+.1/3+.1/2.5)/4))<1e-10
    assert abs(by["cGW[bj]"]["G_iomega_error_pct"]-3)<1e-12
    assert by["cGW[bj]"]["G_shared_with_GW"]
    with (tmp_path/"ranking_green.csv").open(encoding="utf-8-sig") as handle:
        ranked=list(csv.DictReader(handle))
    assert [r["method"] for r in ranked]==["post-GW[bj]","GW[bj]"]
    with (tmp_path/"ranking_susceptibility.csv").open(encoding="utf-8-sig") as handle:
        ranked=list(csv.DictReader(handle))
    assert [r["method"] for r in ranked]==["cGW[bj]"]
    assert (tmp_path/"comparison_wide.csv").is_file()
    assert (tmp_path/"comparison_summary.md").is_file()


def test_report_does_not_rank_partial_or_unconverged_susceptibility(tmp_path):
    rows=[
        dict(method="ED",quantity="chi[x_even](Q)",value=1,status="ok"),
        dict(method="ED",quantity="chi[y_even](Q)",value=2,status="ok"),
        dict(method="ED",quantity="chi[z_same](Q)",value=3,status="ok"),
        dict(method="ED",quantity="chi[z_opposite](Q)",value=4,status="ok"),
        dict(method="GW[density]",quantity="method_status",
             value=None,status="unconverged",notes="residual 1e-3"),
        dict(method="cGW[density]",quantity="method_status",
             value=None,status="skipped",notes="requires GW"),
        dict(method="GW[bj]",quantity="chi_GG[x_even](Q)",value=.9,status="ok"),
        dict(method="GW[bj]",quantity="chi_GG[y_even](Q)",value=1.9,status="ok"),
        dict(method="GW[bj]",quantity="chi_GG[z_same](Q)",value=2.9,status="ok"),
        dict(method="GW[bj]",quantity="chi_GG[z_opposite](Q)",value=None,
             status="unconverged"),
    ]
    values=make_comparison_scorecards(rows,tmp_path)
    by={r["method"]:r for r in values}
    assert by["GW[density]"]["status"]=="unconverged"
    assert by["cGW[density]"]["status"]=="skipped"
    assert by["GW[bj]"]["chi_mean_4_error_pct"] is None
    with (tmp_path/"ranking_susceptibility.csv").open(encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == 0
