"""Cheap lower-rank views of a precomputed cluster-ED+GW bath tangent.

The expensive part of the JF construction is the ED finite difference for every
retained SVD direction.  Rank-convergence studies should therefore build the
largest requested tangent once and obtain smaller ranks by truncating the
already-computed singular basis.

For the retained bath coordinates the local implicit equation is

    (I - M) a = b.

Because the SVD modes are ordered, restricting to the first ``r`` modes simply
uses the leading principal block of ``I-M`` together with the first ``r``
columns/vectors.  No ED solve is repeated.
"""
from __future__ import annotations

import numpy as np

from .cluster_ed_gw_jf import BathTangentModel


def effective_bath_tangent_rank(model: BathTangentModel, rank: int | None) -> int:
    """Return the usable rank, with ``None``/non-positive meaning full rank."""
    full = int(model.rank)
    if full < 1:
        raise ValueError("bath tangent model has no retained modes")
    if rank is None or int(rank) <= 0:
        return full
    return min(int(rank), full)


def truncate_bath_tangent_model(
    model: BathTangentModel,
    rank: int | None,
) -> BathTangentModel:
    """Return a lower-rank view without recomputing any impurity ED derivative.

    ``rank<=0`` means the full precomputed tangent.  If ``rank`` is larger than
    the available SVD rank, the original model is returned unchanged.
    """
    r = effective_bath_tangent_rank(model, rank)
    if r == int(model.rank):
        return model

    inner = np.asarray(model.inner_matrix)[:r, :r]
    cond = float(np.linalg.cond(inner))
    return BathTangentModel(
        theta0=np.asarray(model.theta0),
        mode_vectors=np.asarray(model.mode_vectors)[:, :r],
        singular_values=np.asarray(model.singular_values)[:r],
        Ufit=np.asarray(model.Ufit)[:, :r],
        sigma_modes=np.asarray(model.sigma_modes)[:r],
        inner_matrix=inner,
        fit_indices=np.asarray(model.fit_indices),
        fit_weights=np.asarray(model.fit_weights),
        fit_metric=str(model.fit_metric),
        model_g0_fit=np.asarray(model.model_g0_fit),
        Gc_inv=np.asarray(model.Gc_inv),
        rank=r,
        condition_number=cond,
        build_seconds=float(model.build_seconds),
        fd_step=float(model.fd_step),
    )


__all__ = [
    "effective_bath_tangent_rank",
    "truncate_bath_tangent_model",
]
