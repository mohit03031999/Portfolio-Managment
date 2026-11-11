"""Optimal rotationally invariant estimator of a correlation matrix (nonlinear shrinkage).

Clipping, in :mod:`market_analyzer.matrices`, keeps the sample eigenvectors
and replaces every eigenvalue inside the noise band by one common value.
The estimator here also keeps the sample eigenvectors but gives *every*
eigenvalue its own corrected value. The correction is the one that
minimises the squared error to the truth among all estimators that share
the sample eigenvectors, which is why the literature calls it the optimal
rotationally invariant estimator (RIE).

The remarkable fact, due to Ledoit and Péché (2011), is that in the large-N
limit the correction needs nothing but the sample spectrum. With
``q = N / T`` and ``z = λ − iη`` a point just below the real axis,

    ξ(λ) = λ / |1 − q + q z g(z)|²,        g(z) = (1/N) Σ_j 1 / (z − λ_j)

where ``g`` is the Stieltjes transform of the sample eigenvalues. Bun,
Bouchaud and Potters (2017) showed the same formula holds for the outliers
(market and sector modes), chose ``η = N^(-1/2)`` for the finite-N
evaluation, and added two practical fixes:

* **Inverse-Wishart regularisation.** At finite N the formula drives the
  smallest eigenvalues toward zero instead of lifting them, exactly where a
  minimum-variance optimiser looks hardest. The fix calibrates an
  inverse-Wishart prior so that its lower edge sits at the observed smallest
  eigenvalue, and multiplies each small ``ξ`` by the factor that would make
  the formula exact if the truth really were inverse-Wishart (for which the
  optimal cleaning is linear shrinkage ``1 + α (λ − 1)``). Only eigenvalues
  below one are touched and only ever lifted.
* **Sorting.** The cleaned eigenvalues are expected to be monotone in the
  sample eigenvalues in the large-N limit, and sorting them measurably
  lowers the error to the oracle in their simulations (their Table 1).

The trace is preserved: after cleaning, the eigenvalues are rescaled so
their sum equals the sum of the sample eigenvalues (``N`` for a
correlation matrix).

**Exponentially weighted samples.** When the correlation is built from
weighted returns, ``E = Σ_t w_t y_t y_tᵀ``, the noise is no longer
Marchenko-Pastur. Its R-transform is ``R(x) = Σ_t w_t / (1 − N w_t x)``
(each weighted rank-one term is free of the others), its S-transform
solves ``S · R(ω S) = 1``, and the optimal cleaning for any such
multiplicative noise is

    ξ(λ) = Im[ω S(ω)] / Im[g(z)],       ω = z g(z) − 1,

which reduces to the formula above when the weights are flat (then
``S(ω) = 1 / (1 + q ω)``). For exponential weights with decay ``ε`` and
``N (1 − ε) = q_e`` this is ``S(ω) = (1 − e^(−q_e ω)) / (q_e ω)``, the
spectrum of Pafka, Potters and Kondor (2004): a softer lower edge and a
longer upper tail than Marchenko-Pastur with the same variance
``q_eff = N Σ w_t²``. The inverse-Wishart lift is applied with ``q_eff``
and the flat-noise formulas, as a regulariser only.

When ``T ≤ N`` the sample correlation has ``N − T`` zero eigenvalues and
the formula gives 0/0 there. Those slots take the Ledoit-Wolf value for the
null eigenvalues, ``1 / ((q − 1) m̆(0))`` with ``m̆(0)`` the mean inverse of
the non-zero eigenvalues over the ``T`` companion dimensions; it equals one
when the truth is the identity.

References: Ledoit & Péché, "Eigenvectors of some large sample covariance
matrix ensembles", PTRF 2011; Bun, Bouchaud & Potters, "Cleaning large
correlation matrices: tools from random matrix theory", Physics Reports
2017, Sections 6.2, 6.5 and 8.1 (Algorithm 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ZERO_TOLERANCE = 1e-10


@dataclass(frozen=True)
class RIESpectrum:
    """The sample eigenvalues and what the estimator did to each of them."""

    eigenvalues: np.ndarray
    """Sample eigenvalues, largest first."""
    raw: np.ndarray
    """Ledoit-Péché values before regularisation, sorting and trace rescaling."""
    cleaned: np.ndarray
    """Final cleaned eigenvalues in the order of ``eigenvalues``."""
    q: float
    """``N / T`` for a flat sample, ``N Σ w_t²`` for a weighted one."""
    eta: float
    kappa: float
    """Inverse-Wishart parameter calibrated to the smallest eigenvalue; inf when the pure-noise prior was used, nan without regularisation."""
    alpha: float
    """Slope ``1 / (1 + 2 q κ)`` of the matching linear shrinkage; nan without regularisation."""
    n_regularized: int
    """How many eigenvalues the inverse-Wishart step lifted."""
    n_zero: int
    """How many sample eigenvalues were zero (``T ≤ N``)."""
    trace_scale: float
    """Factor applied at the end so the trace matches the sample's."""


def rie_eigenvalues(
    eigenvalues: np.ndarray,
    q: float | None = None,
    *,
    weights: np.ndarray | None = None,
    regularize: bool = True,
    sort: bool = True,
    eta: float | None = None,
) -> RIESpectrum:
    """Clean a sorted (largest first) correlation spectrum with the optimal RIE.

    ``q`` is ``N / T`` with the same ``T`` used to form the matrix (the
    repo uses ``T − 1`` because Pearson correlation de-means). ``eta``
    defaults to ``N^(-1/2)``. ``regularize`` applies the inverse-Wishart
    lift to the small eigenvalues and ``sort`` makes the cleaned values
    monotone; both off gives the bare Ledoit-Péché formula.

    ``weights`` are the observation weights (summing to one) when the
    matrix was built from a weighted sample: the noise model is then the
    weighted Wishart of the module docstring instead of Marchenko-Pastur,
    ``q`` may be omitted and defaults to ``N Σ w_t²``, which is also the
    ``q`` the inverse-Wishart lift uses.
    """
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Need at least two eigenvalues.")
    if np.any(np.diff(values) > 1e-8):
        raise ValueError("Pass eigenvalues sorted from largest to smallest.")
    if values.min() < -1e-8:
        raise ValueError("Correlation eigenvalues must be non-negative.")
    values = np.clip(values, 0.0, None)
    n_assets = values.size
    if weights is not None:
        weights = _normalised_weights(weights)
        q_weighted = float(n_assets * np.sum(weights**2))
        q = q_weighted if q is None else float(q)
    if q is None:
        raise ValueError("Pass q, or weights from which q follows.")
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q must be positive.")
    eta = float(n_assets**-0.5) if eta is None else float(eta)
    if eta <= 0:
        raise ValueError("eta must be positive.")

    zero = values <= _ZERO_TOLERANCE * max(values[0], _ZERO_TOLERANCE)
    n_zero = int(zero.sum())
    nonzero = ~zero
    z = values - 1j * eta
    g = stieltjes_transform(z, values)
    raw = shrink(z, q, g) if weights is None else weighted_shrink(z, g, weights, n_assets)
    if n_zero:
        if q <= 1.0:
            raise ValueError("Zero eigenvalues need q > 1 (fewer observations than assets).")
        # Ledoit-Wolf value for the null eigenvalues when N > T. The companion
        # matrix has N / q dimensions and the non-zero eigenvalues.
        companion_mean_inverse = float((q / n_assets) * np.sum(1.0 / values[nonzero]))
        raw[zero] = 1.0 / ((q - 1.0) * companion_mean_inverse)
    cleaned = raw.copy()

    kappa = alpha = float("nan")
    n_regularized = 0
    if regularize:
        smallest = float(values[nonzero].min())
        kappa = inverse_wishart_kappa(smallest, q)
        if np.isfinite(kappa):
            alpha = 1.0 / (1.0 + 2.0 * q * kappa)
            g_prior = inverse_wishart_stieltjes(z, q, kappa)
        else:
            # The smallest eigenvalue sits at or above the Marchenko-Pastur
            # edge, so no inverse-Wishart prior is wide enough: use the pure
            # noise prior (κ = ∞), under which the truth is the identity.
            kappa, alpha = float("inf"), 0.0
            g_prior = marchenko_pastur_stieltjes(z, q)
        gamma = np.ones_like(values)
        gamma[nonzero] = (1.0 + alpha * (values[nonzero] - 1.0)) / shrink(z[nonzero], q, g_prior[nonzero])
        lift = (gamma > 1.0) & (values < 1.0)
        cleaned[lift] = cleaned[lift] * gamma[lift]
        n_regularized = int(lift.sum())
    if sort:
        cleaned = np.sort(cleaned)[::-1]
    total = float(cleaned.sum())
    trace_scale = float(values.sum() / total) if total > 0 else 1.0
    cleaned = cleaned * trace_scale
    return RIESpectrum(
        eigenvalues=values,
        raw=raw,
        cleaned=cleaned,
        q=float(q),
        eta=eta,
        kappa=kappa,
        alpha=alpha,
        n_regularized=n_regularized,
        n_zero=n_zero,
        trace_scale=trace_scale,
    )


def stieltjes_transform(z: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """``g(z_i) = (1/(N−1)) Σ_{j≠i} 1 / (z_i − λ_j)``, one value per eigenvalue.

    Each eigenvalue is left out of its own sum, as in Bun, Bouchaud and
    Potters' Algorithm 1, so the self term ``1 / (−iη)`` does not swamp
    the others.
    """
    points = np.asarray(z, dtype=complex)
    values = np.asarray(eigenvalues, dtype=float)
    if points.shape != values.shape:
        raise ValueError("z must hold one point per eigenvalue.")
    full = np.sum(1.0 / (points[:, None] - values[None, :]), axis=1)
    own = 1.0 / (points - values)
    return (full - own) / (values.size - 1)


def shrink(z: np.ndarray, q: float, g: np.ndarray) -> np.ndarray:
    """The Ledoit-Péché map ``Re(z) / |1 − q + q z g|²``."""
    points = np.asarray(z, dtype=complex)
    return np.real(points) / np.abs(1.0 - q + q * points * np.asarray(g, dtype=complex)) ** 2


def inverse_wishart_edges(q: float, kappa: float) -> tuple[float, float]:
    """Edges of the sample spectrum when the truth is inverse-Wishart with parameter ``kappa``."""
    if kappa <= 0 or q <= 0:
        raise ValueError("q and kappa must be positive.")
    root = np.sqrt((2.0 * kappa + 1.0) * (2.0 * q * kappa + 1.0))
    centre = (1.0 + q) * kappa + 1.0
    return float((centre - root) / kappa), float((centre + root) / kappa)


def inverse_wishart_stieltjes(z: np.ndarray, q: float, kappa: float) -> np.ndarray:
    """Stieltjes transform of the sample spectrum under an inverse-Wishart truth.

    Bun, Bouchaud and Potters' equation (3.41): the discriminant of the
    quadratic factorises as ``κ² (z − λ₊)(z − λ₋)`` with the edges of
    :func:`inverse_wishart_edges`. The branch is the one with ``g ~ 1/z``
    at infinity and a positive imaginary part just below the real axis. As
    ``κ → ∞`` it tends to the Marchenko-Pastur transform.
    """
    points = np.asarray(z, dtype=complex)
    lower, upper = inverse_wishart_edges(q, kappa)
    root = np.sqrt(points - upper) * np.sqrt(points - lower)
    numerator = points * (1.0 + kappa) - kappa * (1.0 - q) - kappa * root
    return numerator / (points * (points + 2.0 * q * kappa))


def marchenko_pastur_stieltjes(z: np.ndarray, q: float) -> np.ndarray:
    """Stieltjes transform of the Marchenko-Pastur law with unit variance, the ``κ → ∞`` limit above."""
    points = np.asarray(z, dtype=complex)
    lower, upper = (1.0 - np.sqrt(q)) ** 2, (1.0 + np.sqrt(q)) ** 2
    root = np.sqrt(points - upper) * np.sqrt(points - lower)
    return (points + q - 1.0 - root) / (2.0 * q * points)


def inverse_wishart_kappa(smallest: float, q: float) -> float:
    """The ``kappa`` whose inverse-Wishart lower edge equals the observed smallest eigenvalue.

    Solving ``λ₋(κ) = λ_N`` gives ``κ = 2 λ_N / ((1 − q − λ_N)² − 4 q λ_N)``.
    The denominator is positive only when ``λ_N`` lies below the
    Marchenko-Pastur lower edge ``(1 − √q)²``; otherwise the spectrum is
    no wider than pure noise and nan is returned, which the caller reads
    as ``κ = ∞``.
    """
    if smallest <= 0:
        return float("nan")
    denominator = (1.0 - q - smallest) ** 2 - 4.0 * q * smallest
    if denominator <= 0:
        return float("nan")
    return float(2.0 * smallest / denominator)


# ----------------------------------------------------------------------------- weighted samples


def exponential_weights(n_obs: int, halflife: float | None) -> np.ndarray:
    """Normalised exponential weights, oldest observation first, newest largest.

    ``halflife`` is in observations; ``None`` or ``inf`` gives flat weights.
    The weight ``halflife`` observations back is half the newest one.
    """
    if n_obs < 1:
        raise ValueError("Need at least one observation.")
    if halflife is None or not np.isfinite(halflife):
        return np.full(n_obs, 1.0 / n_obs)
    if halflife <= 0:
        raise ValueError("halflife must be positive.")
    lags = np.arange(n_obs)[::-1]
    weights = 0.5 ** (lags / float(halflife))
    return weights / weights.sum()


def effective_observations(weights: np.ndarray) -> float:
    """Kish's effective sample size ``(Σ w)² / Σ w²``: ``T`` for flat weights, about ``2.9 × halflife`` for long exponential ones."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2))


def weighted_r_transform(x: np.ndarray, weights: np.ndarray, n_assets: int) -> np.ndarray:
    """``R(x) = Σ_t w_t / (1 − N w_t x)``, the R-transform of ``Σ_t w_t y_t y_tᵀ`` for iid unit-normal ``y_t``.

    With flat weights ``1/T`` this is the Marchenko-Pastur ``1 / (1 − q x)``.
    """
    points = np.asarray(x, dtype=complex)
    w = np.asarray(weights, dtype=float)
    return np.sum(w[None, :] / (1.0 - n_assets * w[None, :] * points[:, None]), axis=1)


def _weighted_r_prime(x: np.ndarray, weights: np.ndarray, n_assets: int) -> np.ndarray:
    points = np.asarray(x, dtype=complex)
    w = np.asarray(weights, dtype=float)
    return np.sum(n_assets * w[None, :] ** 2 / (1.0 - n_assets * w[None, :] * points[:, None]) ** 2, axis=1)


def weighted_s_transform(omega: np.ndarray, weights: np.ndarray, n_assets: int, tol: float = 1e-12, max_iter: int = 100) -> np.ndarray:
    """``S(ω)`` of the weighted noise, the root of ``S · R(ω S) = 1`` that continues ``S(0) = 1``.

    ``R`` has one pole per observation, so the equation has many roots;
    the physical one is the analytic continuation from ``ω = 0``. It is
    followed by continuation in ``ω`` with a damped Newton step at each
    stage, then polished at the target. (Starting Newton from the
    Marchenko-Pastur guess ``1 / (1 + q_eff ω)`` fails when ``q_eff > 1``
    and ``ω`` is near ``−1``, the regime of the smallest eigenvalues.)
    """
    om = np.asarray(omega, dtype=complex)
    w = _normalised_weights(weights)
    q_eff = n_assets * float(np.sum(w**2))
    stages = max(4, int(np.ceil(4.0 * float(np.max(np.abs(q_eff * om))))))
    s = np.ones_like(om)
    for stage in range(1, stages + 1):
        s = _newton_s(om * (stage / stages), s, w, n_assets, 1e-9, 25)
    return _newton_s(om, s, w, n_assets, tol, max_iter)


def _newton_s(om: np.ndarray, s: np.ndarray, w: np.ndarray, n_assets: int, tol: float, iters: int) -> np.ndarray:
    for _ in range(iters):
        x = om * s
        r = weighted_r_transform(x, w, n_assets)
        step = (s * r - 1.0) / (r + s * om * _weighted_r_prime(x, w, n_assets))
        # Never move more than half of |s| in one step; keeps the iterate on the physical branch.
        damping = np.minimum(1.0, 0.5 * np.abs(s) / np.maximum(np.abs(step), 1e-300))
        s = s - damping * step
        if np.max(np.abs(step)) < tol:
            break
    return s


def weighted_shrink(z: np.ndarray, g: np.ndarray, weights: np.ndarray, n_assets: int) -> np.ndarray:
    """``ξ = Im[ω S(ω)] / Im[g]`` for the weighted noise, with ``ω = z g − 1``.

    Bun, Bouchaud and Potters evaluate the flat-noise formula with
    ``Re(z) = λ`` in the numerator rather than ``Im(z g) / Im(g)``, which
    at finite ``η`` differs by ``η Re(g) / Im(g)`` and is the same size as
    ``λ`` for an isolated eigenvalue. The same convention is used here:
    ``ω`` takes the imaginary part ``λ Im(g)``.
    """
    points = np.asarray(z, dtype=complex)
    gg = np.asarray(g, dtype=complex)
    omega = (np.real(points * gg) - 1.0) + 1j * np.real(points) * np.imag(gg)
    s = weighted_s_transform(omega, weights, n_assets)
    return np.imag(omega * s) / np.imag(gg)


def _normalised_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).ravel()
    if w.size < 2 or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("weights must be at least two non-negative finite numbers with a positive sum.")
    return w / w.sum()
