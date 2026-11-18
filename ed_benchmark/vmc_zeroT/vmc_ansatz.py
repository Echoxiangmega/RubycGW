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



def bond_aware_jastrow_channels(g, n_nonbond_shells: int = 4, tol: float = 1e-8):
    """Density-Jastrow channels that distinguish t, t1 and t2 graph bonds.

    Channels are:
      1) all graph bonds of kind t,
      2) all graph bonds of kind t1,
      3) all graph bonds of kind t2,
      4+) non-graph pairs grouped by minimum-image Euclidean distance.

    Graph-bond pairs are excluded from the distance shells, so no pair is
    counted twice. This is useful on the Ruby lattice because physically
    distinct hopping/interacting bonds can have the same geometric distance.
    """
    ns = g.Ns
    mats = []
    labels = []
    distances = []
    counts = []

    edge_pairs = set()
    for kind in ("t", "t1", "t2"):
        m = np.zeros((ns, ns), float)
        n = 0
        ds = []
        for e in g.edges:
            if e.kind != kind:
                continue
            i, j = int(e.i), int(e.j)
            key = tuple(sorted((i, j)))
            if key in edge_pairs:
                raise RuntimeError(f"Duplicate graph pair in bond-aware Jastrow: {key}")
            edge_pairs.add(key)
            m[i, j] = m[j, i] = 1.0
            n += 1
            d = min(
                np.linalg.norm(
                    np.asarray(g.positions[j]) + np.asarray(g.B) @ np.asarray(w, int)
                    - np.asarray(g.positions[i])
                )
                for w in product(range(-2, 3), repeat=2)
            )
            ds.append(float(d))
        mats.append(m)
        labels.append(f"bond:{kind}")
        distances.append(float(np.mean(ds)) if ds else float("nan"))
        counts.append(n)

    records = []
    for i in range(ns):
        for j in range(i + 1, ns):
            if (i, j) in edge_pairs:
                continue
            d = min(
                np.linalg.norm(
                    np.asarray(g.positions[j]) + np.asarray(g.B) @ np.asarray(w, int)
                    - np.asarray(g.positions[i])
                )
                for w in product(range(-2, 3), repeat=2)
            )
            records.append((float(d), i, j))

    shell_distances = []
    for d, _, _ in sorted(records):
        if not shell_distances or abs(d-shell_distances[-1]) > tol:
            shell_distances.append(d)
    if n_nonbond_shells > 0:
        shell_distances = shell_distances[:n_nonbond_shells]

    for ishell, d0 in enumerate(shell_distances, 1):
        m = np.zeros((ns, ns), float)
        n = 0
        for d, i, j in records:
            if abs(d-d0) <= tol:
                m[i, j] = m[j, i] = 1.0
                n += 1
        mats.append(m)
        labels.append(f"nonbond:r{ishell}")
        distances.append(float(d0))
        counts.append(n)

    mats = np.asarray(mats, dtype=float)
    counts = np.asarray(counts, dtype=int)
    distances = np.asarray(distances, dtype=float)

    # Every included unordered pair appears in exactly one channel.
    total = np.sum(mats, axis=0)
    iu = np.triu_indices(ns, k=1)
    if np.any(total[iu] > 1.0 + 1e-12):
        raise RuntimeError("Bond-aware Jastrow channels overlap")
    return labels, distances, mats, counts


def jastrow_channels(g, mode: str = "distance", n_shells: int = 4):
    """Unified Jastrow channel constructor.

    mode='distance':
        old behavior, all pairs grouped only by distance.
    mode='bond_aware':
        separate t/t1/t2 graph bonds, then n_shells nonbond distance shells.
    """
    mode = str(mode).lower()
    if mode == "distance":
        distances, mats, counts = minimum_image_shells(g, n_shells=n_shells)
        labels = [f"distance:r{i+1}" for i in range(len(distances))]
        return labels, distances, mats, counts
    if mode == "bond_aware":
        return bond_aware_jastrow_channels(g, n_nonbond_shells=n_shells)
    raise ValueError(f"Unknown Jastrow mode: {mode}")



def bond_backflow_channels(g):
    """Three symmetric nearest-neighbor backflow channels: t, t1 and t2.

    The backflow orbital deformation is
        Q_BF[i,a](x) = Q[i,a]
          + sum_k eta_k sum_j B_k[i,j] (n_j-nu) Q[j,a],
    with complex eta_k and k in {t,t1,t2}.
    """
    ns = g.Ns
    labels = ["t", "t1", "t2"]
    mats = np.zeros((3, ns, ns), dtype=float)
    counts = np.zeros(3, dtype=int)
    kind_to_k = {name: k for k, name in enumerate(labels)}
    seen = set()
    for e in g.edges:
        if e.kind not in kind_to_k:
            continue
        i, j = int(e.i), int(e.j)
        key = tuple(sorted((i, j)))
        if key in seen:
            raise RuntimeError(f"Duplicate graph pair in backflow channels: {key}")
        seen.add(key)
        k = kind_to_k[e.kind]
        mats[k, i, j] = mats[k, j, i] = 1.0
        counts[k] += 1
    return [f"eta_{x}" for x in labels], mats, counts


def one_body_seed_hamiltonian(g, seed: str, seed_amplitude: float, star_formfactor) -> np.ndarray:
    """Auxiliary one-body Hamiltonian used only to initialize variational states."""
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
    return h


def one_body_seed(g, nf: int, seed: str, seed_amplitude: float, star_formfactor) -> np.ndarray:
    """Return occupied orbitals of the physically seeded auxiliary one-body H."""
    h = one_body_seed_hamiltonian(g, seed, seed_amplitude, star_formfactor)
    _, u = np.linalg.eigh(h)
    return np.asarray(u[:, :nf], dtype=np.complex128)


def build_netket_model(
    M0: np.ndarray,
    shell_matrices: np.ndarray,
    use_jastrow: bool,
    backflow_matrices: np.ndarray | None = None,
):
    """Create a Flax-Linen log-wavefunction model lazily."""
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import netket as nk

    M0 = np.asarray(M0, np.complex128)
    shells_np = np.asarray(shell_matrices, np.float64)
    backflow_np = (
        None if backflow_matrices is None
        else np.asarray(backflow_matrices, np.float64)
    )
    ns, nf = M0.shape
    if backflow_np is not None and backflow_np.shape[1:] != (ns, ns):
        raise ValueError("backflow_matrices must have shape (n_channels,Ns,Ns)")

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
            if backflow_np is not None:
                eta = self.param(
                    "eta", nn.initializers.zeros_init(),
                    (backflow_np.shape[0],), jnp.complex128
                )
                backflow = jnp.asarray(backflow_np)
            else:
                eta = None
                backflow = None

            Q, _ = jnp.linalg.qr(M, mode="reduced")
            shells = jnp.asarray(shells_np)
            filling = nf / ns

            @partial(jnp.vectorize, signature="(n)->()")
            def logpsi(x):
                occupied = jnp.nonzero(x, size=nf)[0]
                dn = x - filling
                if backflow is not None:
                    dQ = jnp.einsum("k,kij,j,ja->ia", eta, backflow, dn, Q)
                    Qx = Q + dQ
                else:
                    Qx = Q
                A = Qx[occupied]
                logdet = nk.jax.logdet_cmplx(A)
                if use_jastrow and shells.shape[0] > 0:
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
    partner_characters: np.ndarray | None = None,
    time_reversal_even: bool = False,
    initial_tr_phase: float = 0.0,
    backflow_matrices: np.ndarray | None = None,
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
    backflow_np = (
        None if backflow_matrices is None
        else np.asarray(backflow_matrices, np.float64)
    )
    maps_np = np.asarray(site_maps, np.int32)
    chars_np = np.asarray(characters, np.complex128)
    partner_chars_np = (
        None if partner_characters is None
        else np.asarray(partner_characters, np.complex128)
    )
    ns, nf = M0.shape
    if backflow_np is not None and backflow_np.shape[1:] != (ns, ns):
        raise ValueError("backflow_matrices must have shape (n_channels,Ns,Ns)")
    if maps_np.ndim != 2 or maps_np.shape[1] != ns:
        raise ValueError("site_maps must have shape (|G|, Ns)")
    if chars_np.shape != (maps_np.shape[0],):
        raise ValueError("characters length does not match site_maps")
    if partner_chars_np is not None and partner_chars_np.shape != chars_np.shape:
        raise ValueError("partner_characters shape does not match characters")

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
            if backflow_np is not None:
                eta = self.param(
                    "eta", nn.initializers.zeros_init(),
                    (backflow_np.shape[0],), jnp.complex128
                )
                backflow = jnp.asarray(backflow_np)
            else:
                eta = None
                backflow = None

            Q, _ = jnp.linalg.qr(M, mode="reduced")
            shells = jnp.asarray(shells_np)
            maps = jnp.asarray(maps_np)
            chars = jnp.asarray(chars_np)
            partner_chars = (
                None if partner_chars_np is None else jnp.asarray(partner_chars_np)
            )
            if time_reversal_even:
                tr_phase = self.param(
                    "tr_phase",
                    lambda key: jnp.asarray(initial_tr_phase, dtype=jnp.float64),
                )
            else:
                tr_phase = None
            filling = nf / ns

            def base_logpsi(x):
                occupied = jnp.nonzero(x, size=nf)[0]
                dn = x - filling
                if backflow is not None:
                    dQ = jnp.einsum("k,kij,j,ja->ia", eta, backflow, dn, Q)
                    Qx = Q + dQ
                else:
                    Qx = Q
                A = Qx[occupied]
                out = nk.jax.logdet_cmplx(A)
                if use_jastrow and shells.shape[0] > 0:
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
                shift = jnp.max(jnp.real(logs))
                amps = jnp.exp(logs - shift)
                z = jnp.sum(jnp.conj(chars) * signs * amps)
                if partner_chars is not None:
                    z = z + jnp.sum(jnp.conj(partner_chars) * signs * amps)
                if time_reversal_even:
                    # Spinless time reversal is complex conjugation K in the
                    # occupation basis. The trainable phase spans all real
                    # (T-invariant up to a global phase) combinations of the
                    # projected state and its time-reversed partner.
                    phase = jnp.exp(1j * tr_phase)
                    z = phase * z + jnp.conj(phase * z)
                # Exact symmetry/TR nodes are allowed. A tiny complex floor
                # avoids undefined log/gradient only at floating-point zero.
                tiny = jnp.asarray(1e-300 + 0.0j, dtype=z.dtype)
                z_safe = jnp.where(jnp.abs(z) > 1e-300, z, tiny)
                return shift + jnp.log(z_safe)

            # Support arbitrary leading batch/sample dimensions, as in the
            # unprojected NetKet model.
            return jnp.vectorize(projected_logpsi, signature="(n)->()")(n)

    return LogProjectedSlaterDensityJastrow()
