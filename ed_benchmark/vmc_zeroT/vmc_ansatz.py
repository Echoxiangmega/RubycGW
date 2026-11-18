"""Varma-style complex Slater x density-Jastrow ansatz for Hex18."""
from __future__ import annotations

from functools import partial
from itertools import product
import numpy as np


def minimum_image_shells(g, n_shells: int = 4, tol: float = 1e-8):
    """Construct symmetric pair matrices grouped by PBC Euclidean distance."""
    ns = g.Ns
    records = []
    for i in range(ns):
        for j in range(i + 1, ns):
            d = min(
                np.linalg.norm(
                    np.asarray(g.positions[j]) + np.asarray(g.B) @ np.asarray(w, int)
                    - np.asarray(g.positions[i])
                )
                for w in product(range(-2, 3), repeat=2)
            )
            records.append((float(d), i, j))
    distances = []
    for d, _, _ in sorted(records):
        if not distances or abs(d - distances[-1]) > tol:
            distances.append(d)
    if n_shells > 0:
        distances = distances[:n_shells]
    mats = np.zeros((len(distances), ns, ns), float)
    counts = np.zeros(len(distances), int)
    for d, i, j in records:
        k = next((k for k, x in enumerate(distances) if abs(d - x) <= tol), None)
        if k is not None:
            mats[k, i, j] = mats[k, j, i] = 1.0
            counts[k] += 1
    return np.asarray(distances), mats, counts


def one_body_seed(g, nf: int, seed: str, seed_amplitude: float, star_formfactor) -> np.ndarray:
    """Return occupied orbitals of a physically seeded auxiliary one-body H.

    The orbitals are initial conditions only. During VMC the full complex
    orbital matrix is variational; it is not constrained by HF self-consistency.
    """
    ns = g.Ns
    h = np.zeros((ns, ns), complex)
    for e in g.edges:
        t = float(g.t[e.kind])
        h[e.i, e.j] += t
        h[e.j, e.i] += t

    s = seed.lower()
    amp = float(seed_amplitude)
    if s in ("sd", "isd"):
        sign = 1.0 if s == "sd" else -1.0
        f = np.asarray(star_formfactor, float)
        for fb, e in zip(f, g.edges):
            z = sign * amp * fb
            h[e.i, e.j] += z
            h[e.j, e.i] += z
    elif s in ("lc_same", "lc_opposite"):
        for loop in [x for x in g.cycles() if len(x) == 3]:
            is_a = loop[0] % 6 < 3
            chirality = 1.0 if (s == "lc_same" or is_a) else -1.0
            for i, j in zip(loop, loop[1:] + loop[:1]):
                h[i, j] += 1j * chirality * amp
                h[j, i] -= 1j * chirality * amp
    elif s != "normal":
        raise ValueError(f"Unknown VMC seed: {seed}")

    _, u = np.linalg.eigh(h)
    return np.asarray(u[:, :nf], dtype=np.complex128)


def build_netket_model(M0: np.ndarray, shell_matrices: np.ndarray, use_jastrow: bool):
    """Create a Flax-Linen log-wavefunction model lazily."""
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import netket as nk

    M0 = np.asarray(M0, np.complex128)
    shells_np = np.asarray(shell_matrices, np.float64)
    ns, nf = M0.shape

    class LogSlaterDensityJastrow(nn.Module):
        @nn.compact
        def __call__(self, n: jax.Array) -> jax.Array:
            def init_M(key):
                del key
                return jnp.asarray(M0)

            M = self.param("M", init_M)
            if use_jastrow:
                v = self.param(
                    "v", nn.initializers.zeros_init(), (shells_np.shape[0],), jnp.float64
                )
            else:
                v = None

            Q, _ = jnp.linalg.qr(M, mode="reduced")
            shells = jnp.asarray(shells_np)
            filling = nf / ns

            @partial(jnp.vectorize, signature="(n)->()")
            def logpsi(x):
                occupied = jnp.nonzero(x, size=nf)[0]
                A = Q[occupied]
                logdet = nk.jax.logdet_cmplx(A)
                if use_jastrow and shells.shape[0] > 0:
                    dn = x - filling
                    features = 0.5 * jnp.einsum("i,sij,j->s", dn, shells, dn)
                    return logdet + jnp.dot(v, features)
                return logdet

            return logpsi(n)

    return LogSlaterDensityJastrow()


def build_projected_netket_model(
    M0: np.ndarray,
    shell_matrices: np.ndarray,
    use_jastrow: bool,
    site_maps: np.ndarray,
    characters: np.ndarray,
):
    """Create a symmetry-projected log wavefunction for variation after projection.

    For a target one-dimensional irrep with characters chi_g, the model is

        Psi_Gamma(x) = sum_g chi_g^* (U_g Psi)(x)

    up to an irrelevant overall 1/|G| factor. For the site permutation
    c_i^dag -> c_{p(i)}^dag,

        (U_g Psi)(x_target) = s_g(x_source) Psi(x_source),
        x_source[i] = x_target[p(i)],

    where s_g is the fermionic reordering parity. The same complex Slater
    matrix and real density-Jastrow parameters are shared by all group images.
    """
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import netket as nk

    M0 = np.asarray(M0, np.complex128)
    shells_np = np.asarray(shell_matrices, np.float64)
    maps_np = np.asarray(site_maps, np.int32)
    chars_np = np.asarray(characters, np.complex128)
    ns, nf = M0.shape
    if maps_np.ndim != 2 or maps_np.shape[1] != ns:
        raise ValueError("site_maps must have shape (|G|, Ns)")
    if chars_np.shape != (maps_np.shape[0],):
        raise ValueError("characters length does not match site_maps")

    class LogProjectedSlaterDensityJastrow(nn.Module):
        @nn.compact
        def __call__(self, n: jax.Array) -> jax.Array:
            def init_M(key):
                del key
                return jnp.asarray(M0)

            M = self.param("M", init_M)
            if use_jastrow:
                v = self.param(
                    "v", nn.initializers.zeros_init(), (shells_np.shape[0],), jnp.float64
                )
            else:
                v = None

            Q, _ = jnp.linalg.qr(M, mode="reduced")
            shells = jnp.asarray(shells_np)
            maps = jnp.asarray(maps_np)
            chars = jnp.asarray(chars_np)
            filling = nf / ns

            def base_logpsi(x):
                occupied = jnp.nonzero(x, size=nf)[0]
                A = Q[occupied]
                out = nk.jax.logdet_cmplx(A)
                if use_jastrow and shells.shape[0] > 0:
                    dn = x - filling
                    features = 0.5 * jnp.einsum("i,sij,j->s", dn, shells, dn)
                    out = out + jnp.dot(v, features)
                return out

            def fock_sign(p, source):
                occupied = jnp.nonzero(source, size=nf)[0]
                mapped = p[occupied]
                # Number of inversions in mapped creation-operator order.
                cmp = mapped[:, None] > mapped[None, :]
                inv = jnp.sum(jnp.triu(cmp, k=1), dtype=jnp.int32)
                return jnp.where((inv & 1) == 0, 1.0, -1.0)

            def projected_logpsi(x):
                # source[i] = target[p(i)] for U_p acting on amplitudes.
                sources = jax.vmap(lambda p: x[p])(maps)
                logs = jax.vmap(base_logpsi)(sources)
                signs = jax.vmap(fock_sign)(maps, sources)
                coeff = jnp.conj(chars) * signs
                shift = jnp.max(jnp.real(logs))
                z = jnp.sum(coeff * jnp.exp(logs - shift))
                # Exact symmetry nodes are allowed. A tiny complex floor avoids
                # undefined log/gradient at accidental floating-point zero only.
                tiny = jnp.asarray(1e-300 + 0.0j, dtype=z.dtype)
                z_safe = jnp.where(jnp.abs(z) > 1e-300, z, tiny)
                return shift + jnp.log(z_safe)

            return jax.vmap(projected_logpsi)(n)

    return LogProjectedSlaterDensityJastrow()
