"""Ruby-lattice self-consistent GW and covariant-GW reference code."""

from .model import RubyParameters, build_h0, build_interaction, eta_vertices
from .grids import MatsubaraGrid
from .gw import (
    GWOptions,
    GWResult,
    NonInteractingResult,
    solve_gw,
    solve_noninteracting,
)
from .cgw import VertexOptions, VertexResult, solve_vertex_q0
from .susceptibility import chi_eta, channel_summary
from .supercell import (
    NSUP,
    build_supercell_h0,
    build_supercell_interaction,
    charge_order_parameter,
    period3_complex_mode,
    period3_real_pattern,
)
from .supercell_gw import solve_matrix_gw, solve_supercell_gw
from .supercell_gw_fast import solve_matrix_gw_fast, solve_supercell_gw_fast

__all__ = [
    "RubyParameters", "MatsubaraGrid", "GWOptions", "GWResult",
    "NonInteractingResult", "VertexOptions", "VertexResult", "build_h0",
    "build_interaction", "eta_vertices", "solve_gw", "solve_noninteracting",
    "solve_vertex_q0", "chi_eta", "channel_summary",
    "NSUP", "build_supercell_h0", "build_supercell_interaction",
    "charge_order_parameter", "period3_complex_mode", "period3_real_pattern",
    "solve_matrix_gw", "solve_supercell_gw",
    "solve_matrix_gw_fast", "solve_supercell_gw_fast",
]
