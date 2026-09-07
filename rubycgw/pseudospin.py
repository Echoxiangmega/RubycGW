"""Pseudospin/orbital-order vertices for the Ruby-lattice E doublet.

For one triangle with the low-energy chiral basis |+>, |->,

    tau_x = 2 n0 - n1 - n2,
    tau_y = sqrt(3) (n2 - n1),
    tau_z = loop chirality,

where tau_x,tau_y are TR-even intra-triangle E-type charge/orbital
polarizations and tau_z is TR-odd loop current.  The two elementary triangles
in the Ruby primitive cell have opposite geometric handedness.  We therefore
use a *physical* pseudospin frame in which B is obtained from the algebraic
triangle basis by |+> <-> |->.  Consequently x_B is unchanged while y_B and
z_B change sign relative to the purely algebraic convention.

The z vertices are normalized so that their projection onto the E doublet has
Pauli eigenvalues +/-1.  The legacy eta/current vertices are larger by sqrt(3),
so diagonal z susceptibilities from this module are one third of the legacy
eta susceptibilities.  This normalization is intentional: it allows direct
comparison of chi_xx, chi_yy, and chi_zz as pseudospin components.
"""

from __future__ import annotations

import numpy as np

from .model import NSUB, eta_vertices
from .supercell import NSECTOR, NSUP
from .supercell_cgw import sector_harmonic_matrix


SQRT3 = float(np.sqrt(3.0))


def primitive_triangle_pseudospin_vertices() -> dict[str, np.ndarray]:
    """Return physical A/B pseudospin vertices as 6x6 matrices.

    Returned keys are ``Ax, Ay, Az, Bx, By, Bz``.  In the physical chiral
    basis each triple projects to the ordinary Pauli matrices (tau_x,tau_y,
    tau_z).  ``Az`` and ``Bz`` include the opposite geometric handedness of the
    two triangles.
    """
    ka, kb, _, _ = eta_vertices()

    ax = np.zeros((NSUB, NSUB), dtype=complex)
    ay = np.zeros_like(ax)
    bx = np.zeros_like(ax)
    by = np.zeros_like(ax)

    # A triangle: sites (0,1,2).
    ax[0, 0], ax[1, 1], ax[2, 2] = 2.0, -1.0, -1.0
    ay[0, 0], ay[1, 1], ay[2, 2] = 0.0, -SQRT3, +SQRT3

    # B physical pseudospin frame.  Swapping algebraic |+> and |-> leaves
    # tau_x invariant and flips tau_y,tau_z.
    bx[3, 3], bx[4, 4], bx[5, 5] = 2.0, -1.0, -1.0
    by[3, 3], by[4, 4], by[5, 5] = 0.0, +SQRT3, -SQRT3

    az = -ka / SQRT3
    bz = +kb / SQRT3

    return {
        "Ax": ax,
        "Ay": ay,
        "Az": az,
        "Bx": bx,
        "By": by,
        "Bz": bz,
    }


def primitive_cell_pseudospin_channels() -> dict[str, np.ndarray]:
    """Return named 6x6 pseudospin channels for one primitive cell.

    Besides local A/B components, ``x_even/x_odd`` etc. are normalized A/B
    combinations in the physical pseudospin frame.  For z specifically,

        z_even = physical same circulation,
        z_odd  = physical opposite circulation.

    ``z_same`` and ``z_opposite`` are aliases of these two channels.
    """
    v = primitive_triangle_pseudospin_vertices()
    out = dict(v)
    root2 = np.sqrt(2.0)
    for comp in "xyz":
        A = v[f"A{comp}"]
        B = v[f"B{comp}"]
        out[f"{comp}_even"] = (A + B) / root2
        out[f"{comp}_odd"] = (A - B) / root2

    out["z_same"] = out["z_even"]
    out["z_opposite"] = out["z_odd"]
    return out


_CHANNEL_ALIASES = {
    "same": "z_same",
    "opposite": "z_opposite",
    "zeven": "z_even",
    "zodd": "z_odd",
    "xeven": "x_even",
    "xodd": "x_odd",
    "yeven": "y_even",
    "yodd": "y_odd",
}


def canonical_channel_name(name: str) -> str:
    raw = str(name).strip()
    if raw in primitive_cell_pseudospin_channels():
        return raw
    low = raw.lower().replace("-", "_")
    if low in _CHANNEL_ALIASES:
        return _CHANNEL_ALIASES[low]
    # Preserve A/B capitalization for convenient lower-case CLI input.
    if len(low) == 2 and low[0] in ("a", "b") and low[1] in "xyz":
        cand = low[0].upper() + low[1]
        if cand in primitive_cell_pseudospin_channels():
            return cand
    if low in primitive_cell_pseudospin_channels():
        return low
    raise ValueError(
        f"unknown pseudospin channel {name!r}; available channels are: "
        + ", ".join(available_pseudospin_channels())
    )


def available_pseudospin_channels() -> list[str]:
    """Return canonical public channel names."""
    return [
        "Ax", "Ay", "Az", "Bx", "By", "Bz",
        "x_even", "x_odd", "y_even", "y_odd",
        "z_even", "z_odd", "z_same", "z_opposite",
    ]


def primitive_pseudospin_vertex(name: str) -> np.ndarray:
    """Resolve one named primitive-cell pseudospin vertex."""
    canonical = canonical_channel_name(name)
    return np.array(primitive_cell_pseudospin_channels()[canonical], copy=True)


def supercell_pseudospin_vertices(
    channel_names: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Embed requested primitive-cell channels into the three supercell sectors.

    The returned order is channel-major, then sector::

        channel0_s0, channel0_s1, channel0_s2,
        channel1_s0, ...
    """
    canonical = [canonical_channel_name(x) for x in channel_names]
    vertices: list[np.ndarray] = []
    labels: list[str] = []
    for ch in canonical:
        k6 = primitive_pseudospin_vertex(ch)
        for s in range(NSECTOR):
            mat = np.zeros((NSUP, NSUP), dtype=complex)
            sl = slice(NSUB * s, NSUB * (s + 1))
            mat[sl, sl] = k6
            vertices.append(mat)
            labels.append(f"{ch}_s{s}")
    if not vertices:
        raise ValueError("at least one pseudospin channel is required")
    return np.stack(vertices, axis=0), labels, canonical


def pseudospin_harmonic_transform(n_channels: int) -> np.ndarray:
    """Block-diagonal (s0,s1,s2)->(q0,Qc,Qs) transform for all channels."""
    n_channels = int(n_channels)
    if n_channels < 1:
        raise ValueError("n_channels must be positive")
    u = sector_harmonic_matrix()
    T = np.zeros((NSECTOR * n_channels, NSECTOR * n_channels), dtype=float)
    for i in range(n_channels):
        sl = slice(NSECTOR * i, NSECTOR * (i + 1))
        T[sl, sl] = u
    return T


def pseudospin_harmonic_labels(channel_names: list[str] | tuple[str, ...]) -> list[str]:
    canonical = [canonical_channel_name(x) for x in channel_names]
    labels: list[str] = []
    for ch in canonical:
        labels.extend([f"{ch}_q0", f"{ch}_Qc", f"{ch}_Qs"])
    return labels


def harmonic_block_indices(n_channels: int, harmonic: str) -> np.ndarray:
    """Indices of one harmonic block in channel-major harmonic ordering."""
    harmonic = str(harmonic).strip()
    mapping = {"q0": 0, "Qc": 1, "Qs": 2, "qc": 1, "qs": 2}
    if harmonic not in mapping:
        raise ValueError("harmonic must be one of q0,Qc,Qs")
    h = mapping[harmonic]
    return np.asarray([NSECTOR * i + h for i in range(int(n_channels))], dtype=int)
