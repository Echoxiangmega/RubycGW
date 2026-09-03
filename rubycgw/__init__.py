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
from .orbital_moment import (
    OrbitalMomentResult,
    analyze_checkpoint_orbital_moments,
    equal_time_density_matrix,
    triangle_signed_area,
)
from .electromagnetic import (
    EM_CHANNELS,
    ElectromagneticBackground,
    ElectromagneticResponse,
    FiniteDifferenceResponse,
    build_supercell_h0_peierls,
    peierls_flux_vertex,
    load_electromagnetic_background,
    solve_electromagnetic_response,
    finite_difference_electromagnetic_response,
    compare_covariant_to_finite_difference,
)
from .bulk_orbital_magnetization import (
    BulkOrbitalMagnetizationResult,
    analyze_checkpoint_bulk_orbital_magnetization,
    bulk_orbital_magnetization_from_arrays,
    spectral_cartesian_covariant_derivatives,
    supercell_h0_cartesian_derivatives,
)

__all__ = [
    "RubyParameters", "MatsubaraGrid", "GWOptions", "GWResult",
    "NonInteractingResult", "VertexOptions", "VertexResult", "build_h0",
    "build_interaction", "eta_vertices", "solve_gw", "solve_noninteracting",
    "solve_vertex_q0", "chi_eta", "channel_summary",
    "NSUP", "build_supercell_h0", "build_supercell_interaction",
    "charge_order_parameter", "period3_complex_mode", "period3_real_pattern",
    "solve_matrix_gw", "solve_supercell_gw",
    "solve_matrix_gw_fast", "solve_supercell_gw_fast",
    "OrbitalMomentResult", "analyze_checkpoint_orbital_moments",
    "equal_time_density_matrix", "triangle_signed_area",
    "EM_CHANNELS", "ElectromagneticBackground", "ElectromagneticResponse",
    "FiniteDifferenceResponse", "build_supercell_h0_peierls",
    "peierls_flux_vertex", "load_electromagnetic_background",
    "solve_electromagnetic_response", "finite_difference_electromagnetic_response",
    "compare_covariant_to_finite_difference",
    "BulkOrbitalMagnetizationResult",
    "analyze_checkpoint_bulk_orbital_magnetization",
    "bulk_orbital_magnetization_from_arrays",
    "spectral_cartesian_covariant_derivatives",
    "supercell_h0_cartesian_derivatives",
]