"""Generic pseudo-arclength continuation with a matrix-free Newton-Krylov corrector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


@dataclass(frozen=True)
class PACOptions:
    tol: float = 1e-8
    max_newton: int = 10
    fd_eps: float = 1e-6
    gmres_rtol: float = 2e-3
    gmres_maxiter: int = 24
    gmres_restart: int = 12
    line_search_min: float = 1.0 / 1024.0
    armijo: float = 1e-4
    verbose: bool = False


@dataclass
class PACStepResult:
    x: np.ndarray
    parameter: float
    tangent: np.ndarray
    predictor: np.ndarray
    converged: bool
    newton_iterations: int
    gmres_iterations: int
    residual_norm: float
    line_search_reductions: int


@dataclass
class FixedParameterResult:
    x: np.ndarray
    converged: bool
    newton_iterations: int
    gmres_iterations: int
    residual_norm: float
    line_search_reductions: int


def _call_gmres(A, b, opts: PACOptions, callback):
    kwargs = dict(
        restart=int(opts.gmres_restart),
        maxiter=int(opts.gmres_maxiter),
        callback=callback,
    )
    try:
        return gmres(
            A, b, rtol=float(opts.gmres_rtol), atol=0.0,
            callback_type="legacy", **kwargs,
        )
    except TypeError:
        return gmres(A, b, tol=float(opts.gmres_rtol), **kwargs)


def _newton_krylov_real(
    x_initial: np.ndarray,
    residual: Callable[[np.ndarray], np.ndarray],
    *,
    opts: PACOptions,
    label: str,
) -> FixedParameterResult:
    """Matrix-free Newton-Krylov solve for a square real residual map."""
    x = np.asarray(x_initial, dtype=float).reshape(-1).copy()
    gmres_total = 0
    ls_total = 0
    converged = False
    rnorm = float("inf")
    nit = 0

    def evaluate(xx):
        rr = np.asarray(residual(np.asarray(xx, dtype=float)), dtype=float).reshape(-1)
        if rr.shape != x.shape:
            raise ValueError("residual must have same size as x")
        if not np.all(np.isfinite(rr)):
            raise FloatingPointError("non-finite Newton-Krylov residual")
        return rr

    for nit in range(1, int(opts.max_newton) + 1):
        f0 = evaluate(x)
        rnorm = float(np.linalg.norm(f0))
        if opts.verbose:
            print(f"    {label} Newton {nit:2d}: |R|={rnorm:.3e}")
        if rnorm < float(opts.tol):
            converged = True
            break

        n = x.size
        xnorm = max(1.0, float(np.linalg.norm(x)))

        def matvec(v):
            vv = np.asarray(v, dtype=float)
            vnorm = float(np.linalg.norm(vv))
            if vnorm <= 1e-30:
                return np.zeros_like(vv)
            eps = float(opts.fd_eps) * xnorm / vnorm
            return (evaluate(x + eps * vv) - f0) / eps

        A = LinearOperator((n, n), matvec=matvec, dtype=float)
        gmres_count = [0]

        def cb(_):
            gmres_count[0] += 1

        delta, info = _call_gmres(A, -f0, opts, cb)
        gmres_total += gmres_count[0]
        if not np.all(np.isfinite(delta)):
            if opts.verbose:
                print("      GMRES returned a non-finite direction")
            break

        if info != 0:
            try:
                lin_ratio = float(np.linalg.norm(A @ delta + f0) / max(rnorm, 1e-300))
            except Exception:
                lin_ratio = float("inf")
            if opts.verbose:
                print(f"      GMRES info={info}, linear residual ratio={lin_ratio:.3e}")
            if not np.isfinite(lin_ratio) or lin_ratio >= 0.95:
                break

        lam = 1.0
        accepted = False
        while lam >= float(opts.line_search_min):
            trial = x + lam * delta
            try:
                ft = evaluate(trial)
                nt = float(np.linalg.norm(ft))
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                nt = float("inf")
            if np.isfinite(nt) and nt <= (1.0 - float(opts.armijo) * lam) * rnorm:
                x = trial
                accepted = True
                break
            lam *= 0.5
            ls_total += 1
        if not accepted:
            if opts.verbose:
                print("      line search failed; stopping corrector")
            break

    if not converged:
        try:
            rnorm = float(np.linalg.norm(evaluate(x)))
            converged = bool(rnorm < float(opts.tol))
        except Exception:
            rnorm = float("inf")

    return FixedParameterResult(
        x=np.asarray(x, dtype=float),
        converged=converged,
        newton_iterations=int(nit),
        gmres_iterations=int(gmres_total),
        residual_norm=float(rnorm),
        line_search_reductions=int(ls_total),
    )


def refine_fixed_parameter(
    x_initial: np.ndarray,
    parameter: float,
    residual: Callable[[np.ndarray, float], np.ndarray],
    *,
    opts: PACOptions = PACOptions(),
) -> FixedParameterResult:
    """Refine a continuation state at an exactly fixed parameter value.

    This solves ``R(x, parameter)=0`` with the same matrix-free finite-difference
    Newton-Krylov machinery and backtracking used by the pseudo-arclength corrector.
    It is intended for projecting a PAC branch onto a requested physical value such
    as exactly V=1 after the branch has crossed that value.
    """
    p = float(parameter)
    return _newton_krylov_real(
        x_initial,
        lambda xx: residual(xx, p),
        opts=opts,
        label=f"fixed p={p:g}",
    )


def pseudo_arclength_step(
    x_prev: np.ndarray,
    parameter_prev: float,
    x_curr: np.ndarray,
    parameter_curr: float,
    ds: float,
    residual: Callable[[np.ndarray, float], np.ndarray],
    *,
    parameter_scale: float = 1.0,
    opts: PACOptions = PACOptions(),
) -> PACStepResult:
    """Take one predictor-corrector pseudo-arclength step.

    ``x`` should already be represented in a scaled Euclidean coordinate system.
    ``residual(x,p)`` must return an array with exactly ``x.size`` real entries.
    ``parameter_scale`` controls the relative metric weight of the continuation
    parameter by using p/parameter_scale in the augmented coordinate vector.
    """
    x0 = np.asarray(x_prev, dtype=float).reshape(-1)
    x1 = np.asarray(x_curr, dtype=float).reshape(-1)
    if x0.shape != x1.shape:
        raise ValueError("x_prev/x_curr shape mismatch")
    pscale = float(parameter_scale)
    if not np.isfinite(pscale) or pscale <= 0.0:
        raise ValueError("parameter_scale must be positive")
    if ds <= 0.0:
        raise ValueError("ds must be positive")

    y0 = np.concatenate([x0, [float(parameter_prev) / pscale]])
    y1 = np.concatenate([x1, [float(parameter_curr) / pscale]])
    tangent = y1 - y0
    tn = float(np.linalg.norm(tangent))
    if not np.isfinite(tn) or tn <= 1e-14:
        raise ValueError("seed points are too close for a secant tangent")
    tangent /= tn
    ypred = y1 + float(ds) * tangent
    y = ypred.copy()

    def unpack_y(yy):
        return yy[:-1], float(yy[-1] * pscale)

    def augmented(yy):
        xx, pp = unpack_y(yy)
        rr = np.asarray(residual(xx, pp), dtype=float).reshape(-1)
        if rr.shape != xx.shape:
            raise ValueError("residual must have same size as x")
        if not np.all(np.isfinite(rr)):
            raise FloatingPointError("non-finite pseudo-arclength residual")
        arc = float(np.dot(tangent, yy - ypred))
        return np.concatenate([rr, [arc]])

    gmres_total = 0
    ls_total = 0
    converged = False
    rnorm = float("inf")
    nit = 0

    for nit in range(1, int(opts.max_newton) + 1):
        f0 = augmented(y)
        rnorm = float(np.linalg.norm(f0))
        if opts.verbose:
            print(f"    PAC Newton {nit:2d}: |F|={rnorm:.3e}")
        if rnorm < float(opts.tol):
            converged = True
            break

        n = y.size
        ynorm = max(1.0, float(np.linalg.norm(y)))

        def matvec(v):
            vv = np.asarray(v, dtype=float)
            vnorm = float(np.linalg.norm(vv))
            if vnorm <= 1e-30:
                return np.zeros_like(vv)
            eps = float(opts.fd_eps) * ynorm / vnorm
            return (augmented(y + eps * vv) - f0) / eps

        A = LinearOperator((n, n), matvec=matvec, dtype=float)
        gmres_count = [0]

        def cb(_):
            gmres_count[0] += 1

        delta, info = _call_gmres(A, -f0, opts, cb)
        gmres_total += gmres_count[0]
        if not np.all(np.isfinite(delta)):
            if opts.verbose:
                print("      GMRES returned a non-finite direction")
            break

        # Matrix-free finite-difference Jacobians are only approximately linear,
        # so SciPy can return info!=0 even when the inexact Newton direction is
        # useful. Accept it when it still reduces the linearized residual.
        if info != 0:
            try:
                lin_ratio = float(np.linalg.norm(A @ delta + f0) / max(rnorm, 1e-300))
            except Exception:
                lin_ratio = float("inf")
            if opts.verbose:
                print(f"      GMRES info={info}, linear residual ratio={lin_ratio:.3e}")
            if not np.isfinite(lin_ratio) or lin_ratio >= 0.95:
                break

        lam = 1.0
        accepted = False
        while lam >= float(opts.line_search_min):
            trial = y + lam * delta
            try:
                ft = augmented(trial)
                nt = float(np.linalg.norm(ft))
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                nt = float("inf")
            if np.isfinite(nt) and nt <= (1.0 - float(opts.armijo) * lam) * rnorm:
                y = trial
                accepted = True
                break
            lam *= 0.5
            ls_total += 1
        if not accepted:
            if opts.verbose:
                print("      line search failed; stopping corrector")
            break

    if not converged:
        try:
            rnorm = float(np.linalg.norm(augmented(y)))
            converged = bool(rnorm < float(opts.tol))
        except Exception:
            rnorm = float("inf")

    xout, pout = unpack_y(y)
    return PACStepResult(
        x=np.asarray(xout, dtype=float),
        parameter=float(pout),
        tangent=np.asarray(tangent, dtype=float),
        predictor=np.asarray(ypred, dtype=float),
        converged=converged,
        newton_iterations=int(nit),
        gmres_iterations=int(gmres_total),
        residual_norm=float(rnorm),
        line_search_reductions=int(ls_total),
    )
