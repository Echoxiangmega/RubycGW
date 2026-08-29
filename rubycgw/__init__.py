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

__all__ = [
    "RubyParameters", "MatsubaraGrid", "GWOptions", "GWResult",
    "NonInteractingResult", "VertexOptions", "VertexResult", "build_h0",
    "build_interaction", "eta_vertices", "solve_gw", "solve_noninteracting",
    "solve_vertex_q0", "chi_eta", "channel_summary",
]
