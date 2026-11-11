"""Covariance estimators and the minimum-variance weights they imply.

The backtest in :mod:`market_analyzer.backtest` compares several ways of
turning one window of daily returns into a covariance matrix:

``sample``
    The plain sample covariance. Every entry is taken at face value.
``clipped``
    The Marchenko-Pastur cleaned covariance from
    :class:`market_analyzer.matrices.CorrelationModel` with the fitted
    noise edge. Correlation eigenvalues inside the noise band are replaced
    by their average.
``clipped_unit``
    The same clipping with the textbook edge, noise variance one.
``clipped_market``
    The same clipping with the noise variance left after the market mode,
    as in Laloux, Cizeau, Bouchaud and Potters (1999).
``diagonal``
    The sample variances with every correlation set to zero. Minimum
    variance on this matrix is inverse-variance weighting.
``ledoit_wolf``
    The sample covariance shrunk toward a scaled identity, with the
    intensity from Ledoit and Wolf (2004), "A well-conditioned estimator
    for large-dimensional covariance matrices".
``rie``
    The optimal rotationally invariant estimator (nonlinear shrinkage) of
    :mod:`market_analyzer.rie`: every correlation eigenvalue gets its own
    corrected value from the Ledoit-Péché formula, with the inverse-Wishart
    lift of the smallest eigenvalues and the sorting of Bun, Bouchaud and
    Potters (2017), rescaled by sample volatilities.
``rie_raw``
    The same formula without the lift and the sorting, kept so the effect
    of the regularisation can be measured.
``rie_ewma``
    Exponentially weighted returns (half-life 252 sessions) cleaned with
    the RIE under the weighted-noise model, so recent days count more
    while the eigenvalue correction knows how much noise that costs. Use
    :func:`ewma_estimator` for other half-lives or for weighted samples
    cleaned by clipping or not at all.
``rie_standardized``
    Each day's return divided by that stock's own volatility at the time
    (an exponentially weighted estimate from earlier days, half-life 20
    sessions), so storm days and calm days enter the correlation with the
    same weight; the correlation is cleaned with the RIE and scaled back by
    each stock's latest volatility. Use :func:`standardized_estimator` for
    other half-lives, cleanings or weighted days.

Each estimator returns a :class:`CovarianceEstimate`: the matrix plus a few
diagnostics. Two portfolio rules turn a matrix into weights, neither needing
an expected-return input, so both are tests of whether a covariance matrix
can be trusted: unconstrained minimum variance, ``w ∝ Σ⁻¹ 1``
(:func:`minimum_variance_weights`), the sharpest test because the optimiser
may short whatever looks like a hedge; and long-only minimum variance
(:func:`long_only_minimum_variance_weights`), the same objective with every
weight held at or above zero, solved by an active-set method because there
is no closed form once the sign constraints bind.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_analyzer.matrices import CorrelationModel, _cleaned_correlation, _decompose, _symmetrize, clip_eigenvalues
from market_analyzer.rie import effective_observations, exponential_weights, rie_eigenvalues

_WEIGHT_TOLERANCE = 1e-10
_RELATIVE_RANK_TOLERANCE = 1e-10


@dataclass(frozen=True)
class CovarianceEstimate:
    """An N × N covariance matrix and a few numbers about how it was built."""

    covariance: pd.DataFrame
    info: dict[str, float] = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        return [str(ticker) for ticker in self.covariance.index]


Estimator = Callable[[pd.DataFrame], CovarianceEstimate]


def sample_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    """Sample covariance with ``ddof=1``."""
    panel = _dense(returns)
    return CovarianceEstimate(panel.cov())


def clipped_covariance(returns: pd.DataFrame, sigma2: float | str = "fitted") -> CovarianceEstimate:
    """Marchenko-Pastur clipped correlation, rescaled by sample volatilities.

    ``sigma2`` picks the noise edge, see :mod:`market_analyzer.matrices`.
    """
    model = CorrelationModel.from_returns(_dense(returns), sigma2=sigma2)
    info = {
        "q": model.q,
        "sigma2": model.sigma2,
        "lambda_plus": model.lambda_plus,
        "n_signal": float(model.n_signal),
    }
    return CovarianceEstimate(model.cleaned_covariance, info)


def clipped_unit_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    """Clipping with the textbook edge, noise variance one."""
    return clipped_covariance(returns, sigma2=1.0)


def clipped_market_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    """Clipping with the noise variance left after removing the market mode."""
    return clipped_covariance(returns, sigma2="market")


def diagonal_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    """Sample variances on the diagonal, zero everywhere else."""
    panel = _dense(returns)
    variances = panel.var(ddof=1).to_numpy(dtype=float)
    matrix = pd.DataFrame(np.diag(variances), index=panel.columns, columns=panel.columns)
    return CovarianceEstimate(matrix)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    """Linear shrinkage of the sample covariance toward a scaled identity.

    With ``S`` the sample covariance, ``m`` its average variance, and
    ``x_t`` the de-meaned return rows, the Ledoit-Wolf intensity is

        delta² = ‖S − m I‖² / N
        beta²  = min(delta², (1/T²) Σ_t ‖x_t x_tᵀ − S‖² / N)
        alpha  = beta² / delta²

    and the estimate is ``(1 − alpha) S + alpha m I``. The sum over dates
    collapses to ``Σ_t ‖x_t‖⁴ − T ‖S‖²`` because ``Σ_t x_t x_tᵀ = T S``, so
    nothing larger than N × N is ever formed. The result is scaled by
    ``T / (T − 1)`` so that at zero shrinkage it equals the ``ddof=1`` sample
    covariance used by the other estimators.
    """
    panel = _dense(returns)
    x = panel.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    observations, assets = x.shape
    sample = x.T @ x / observations
    identity = np.eye(assets)
    mean_variance = float(np.trace(sample) / assets)
    delta2 = float(np.sum((sample - mean_variance * identity) ** 2) / assets)
    squared_norms = np.sum(x**2, axis=1)
    beta2_bar = float((np.sum(squared_norms**2) - observations * np.sum(sample**2)) / (assets * observations**2))
    beta2 = min(max(beta2_bar, 0.0), delta2)
    shrinkage = beta2 / delta2 if delta2 > 0 else 1.0
    shrunk = (1.0 - shrinkage) * sample + shrinkage * mean_variance * identity
    shrunk *= observations / (observations - 1)
    matrix = pd.DataFrame(shrunk, index=panel.columns, columns=panel.columns)
    return CovarianceEstimate(matrix, {"shrinkage": shrinkage, "target_variance": mean_variance})


def rie_covariance(
    returns: pd.DataFrame,
    regularize: bool = True,
    sort: bool = True,
    eta: float | None = None,
) -> CovarianceEstimate:
    """Optimal rotationally invariant estimator of the correlation, rescaled by sample volatilities.

    The sample correlation is eigendecomposed, each eigenvalue is mapped
    through :func:`market_analyzer.rie.rie_eigenvalues` with ``q = N / (T − 1)``,
    the matrix is rebuilt on the sample eigenvectors with its diagonal reset
    to one (as the clipped estimator does), and the result is scaled by the
    sample volatilities. ``info`` records ``q``, ``eta``, the inverse-Wishart
    ``kappa`` and ``alpha`` (``kappa`` is ``inf`` when the pure-noise prior
    was used and ``nan`` without regularisation), how many eigenvalues were
    lifted, how many were zero, the trace rescaling factor, and the smallest
    sample and cleaned eigenvalues.
    """
    panel = _dense(returns)
    if panel.shape[1] < 2:
        raise ValueError("Need at least two tickers to clean a correlation matrix.")
    x = panel.to_numpy(dtype=float)
    observations, assets = x.shape
    x = x - x.mean(axis=0)
    volatilities = np.sqrt(np.sum(x**2, axis=0) / (observations - 1))
    if np.any(volatilities == 0):
        raise ValueError("Some tickers have zero variance over the window, so they have no correlation.")
    standardized = x / volatilities
    correlation = _symmetrize(standardized.T @ standardized / (observations - 1))
    eigenvalues, eigenvectors = _decompose(correlation)
    q = assets / (observations - 1)
    spectrum = rie_eigenvalues(eigenvalues, q, regularize=regularize, sort=sort, eta=eta)
    cleaned = _cleaned_correlation(spectrum.cleaned, eigenvectors)
    matrix = pd.DataFrame(
        volatilities[:, None] * cleaned * volatilities[None, :],
        index=panel.columns,
        columns=panel.columns,
    )
    info = {
        "q": q,
        "eta": spectrum.eta,
        "kappa": spectrum.kappa,
        "alpha": spectrum.alpha,
        "n_regularized": float(spectrum.n_regularized),
        "n_zero": float(spectrum.n_zero),
        "trace_scale": spectrum.trace_scale,
        "min_eigenvalue": float(spectrum.eigenvalues[-1]),
        "min_cleaned": float(spectrum.cleaned.min()),
    }
    return CovarianceEstimate(matrix, info)


def rie_raw_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    """The bare Ledoit-Péché map: no inverse-Wishart lift, no sorting."""
    return rie_covariance(returns, regularize=False, sort=False)


def james_stein_rie_covariance(returns: pd.DataFrame, shrink: float | None = None) -> CovarianceEstimate:
    """The RIE with the market eigenvector corrected by James-Stein shrinkage.

    The sample correlation is cleaned by the RIE as in :func:`rie_covariance`.
    Then the top eigenvector ``h`` (sign chosen so its entries average
    positive) is turned into loadings ``b = √λ₁ h``, roughly each stock's
    beta in correlation units. Noise makes those loadings look more spread
    out than the true ones: ``V̂ = V_true + s²``, with the noise variance of
    one loading ``s² = ℓ̄ / (T − 1)`` and ``ℓ̄`` the average of the other
    eigenvalues. Each loading's distance from their average is shrunk by
    ``c = 1 − s² / V̂`` (clipped to [0, 1]), the result renormalised to the
    corrected direction ``h̃``. The matrix is rebuilt as
    ``ξ₁ h̃ h̃ᵀ + P R P``, where ``R`` holds the other cleaned modes and
    ``P = I − h̃ h̃ᵀ`` keeps them perpendicular to the new market direction,
    with the diagonal reset to one before scaling by sample volatilities.

    ``shrink`` overrides the data-driven ``c`` (1 gives back the RIE, 0
    replaces the loadings by their average). ``info`` adds ``c`` (the factor
    used), ``c_data`` (the James-Stein value), ``noise`` (``s²``),
    ``spread`` (``V̂``) and ``turn_degrees`` (the angle between ``h`` and ``h̃``).
    """
    if shrink is not None and not 0.0 <= shrink <= 1.0:
        raise ValueError("shrink must be between 0 (all loadings equal) and 1 (no correction).")
    panel = _dense(returns)
    if panel.shape[1] < 3:
        raise ValueError("Need at least three tickers for James-Stein shrinkage.")
    x = panel.to_numpy(dtype=float)
    observations, assets = x.shape
    x = x - x.mean(axis=0)
    volatilities = np.sqrt(np.sum(x**2, axis=0) / (observations - 1))
    if np.any(volatilities == 0):
        raise ValueError("Some tickers have zero variance over the window, so they have no correlation.")
    standardized = x / volatilities
    correlation = _symmetrize(standardized.T @ standardized / (observations - 1))
    eigenvalues, eigenvectors = _decompose(correlation)
    q = assets / (observations - 1)
    spectrum = rie_eigenvalues(eigenvalues, q)

    top = eigenvectors[:, 0]
    top = top if top.sum() >= 0 else -top
    loadings = np.sqrt(eigenvalues[0]) * top
    mean = float(loadings.mean())
    spread = float(np.mean((loadings - mean) ** 2))
    noise = float(eigenvalues[1:].mean() / (observations - 1))
    c_data = float(np.clip(1.0 - noise / spread, 0.0, 1.0)) if spread > 0 else 0.0
    c = c_data if shrink is None else float(shrink)
    corrected = mean + c * (loadings - mean)
    direction = corrected / np.linalg.norm(corrected)

    rest = eigenvectors[:, 1:] @ np.diag(spectrum.cleaned[1:]) @ eigenvectors[:, 1:].T
    a = direction @ rest
    rest = rest - np.outer(direction, a) - np.outer(a, direction) + float(a @ direction) * np.outer(direction, direction)
    cleaned = spectrum.cleaned[0] * np.outer(direction, direction) + rest
    cleaned = (cleaned + cleaned.T) / 2.0
    scale = np.sqrt(np.maximum(np.diag(cleaned), 1e-12))
    cleaned = _symmetrize(cleaned / np.outer(scale, scale))
    matrix = pd.DataFrame(volatilities[:, None] * cleaned * volatilities[None, :], index=panel.columns, columns=panel.columns)
    info = {
        "q": q,
        "c": c,
        "c_data": c_data,
        "noise": noise,
        "spread": spread,
        "turn_degrees": float(np.degrees(np.arccos(np.clip(abs(direction @ top), 0.0, 1.0)))),
        "min_cleaned": float(spectrum.cleaned.min()),
    }
    return CovarianceEstimate(matrix, info)


def ewma_covariance(
    returns: pd.DataFrame,
    halflife: float | None,
    cleaning: str = "rie",
    noise: str = "weighted",
    regularize: bool = True,
    sort: bool = True,
) -> CovarianceEstimate:
    """Exponentially weighted covariance of the window, optionally cleaned.

    Rows are weighted ``0.5^(lag / halflife)``, newest largest, normalised
    to one; ``halflife=None`` is the flat window. Means, variances and the
    correlation are all weighted, and the covariance is divided by
    ``1 − Σ w_t²`` so the flat case equals the ``ddof=1`` sample covariance.

    ``cleaning`` is ``"none"`` (the weighted covariance itself),
    ``"clipped"`` (Marchenko-Pastur clipping with the textbook edge at the
    effective ``q = N Σ w_t²``, as in Pafka, Potters and Kondor 2004), or
    ``"rie"``. For the RIE, ``noise="weighted"`` uses the exact
    weighted-sample noise model and ``noise="effective"`` the flat formula
    at the effective ``q``. ``info`` records the half-life, the effective
    number of observations, ``q`` and the cleaning diagnostics.
    """
    panel = _dense(returns)
    if panel.shape[1] < 2:
        raise ValueError("Need at least two tickers.")
    x = panel.to_numpy(dtype=float)
    weights = exponential_weights(x.shape[0], halflife)
    covariance, volatilities = _weighted_covariance(x, weights)
    info: dict[str, float] = {
        "halflife": float("inf") if halflife is None or not np.isfinite(halflife) else float(halflife),
        "effective_observations": effective_observations(weights),
        "q": float(x.shape[1] * np.sum(weights**2)),
    }
    if cleaning == "none":
        matrix = covariance
    else:
        cleaned, cleaning_info = _cleaned_correlation_of(covariance, volatilities, weights, cleaning, noise, regularize, sort)
        info.update(cleaning_info)
        matrix = volatilities[:, None] * cleaned * volatilities[None, :]
    frame = pd.DataFrame(matrix, index=panel.columns, columns=panel.columns)
    return CovarianceEstimate(frame, info)


def ewma_volatility(returns: pd.DataFrame, halflife: float, min_periods: int | None = None, floor: float = 0.1) -> pd.DataFrame:
    """Each stock's volatility on each day, from the days strictly before it.

    The RiskMetrics recipe: the variance is an exponentially weighted
    average of past squared returns (zero mean assumed), with weights that
    halve every ``halflife`` days and are normalised over the days seen so
    far. The value on day ``t`` uses returns up to ``t − 1`` only, so a
    day never sees its own move. The first ``min_periods`` days (default
    ``max(20, halflife)``) are left missing, as a burn-in.

    ``floor`` keeps the estimate at or above that fraction of the stock's
    standard deviation over the whole frame. A stock that did not trade
    for weeks (a price series filled forward before its listing, say) has
    an exponentially weighted volatility that decays toward zero, and the
    first real return after the gap would otherwise be divided by almost
    nothing.
    """
    panel = _dense(returns)
    variance = _ewma_variance(panel, halflife, min_periods).shift(1)
    return _floored(np.sqrt(variance), panel, floor)


def _ewma_variance(panel: pd.DataFrame, halflife: float, min_periods: int | None) -> pd.DataFrame:
    if not np.isfinite(halflife) or halflife <= 0:
        raise ValueError("halflife must be a positive number of days.")
    burn = int(max(20, round(halflife))) if min_periods is None else int(min_periods)
    return (panel**2).ewm(halflife=float(halflife), adjust=True, min_periods=burn).mean()


def _floored(sigma, panel: pd.DataFrame, floor: float):
    if floor < 0 or floor >= 1:
        raise ValueError("floor must be in [0, 1).")
    if floor == 0:
        return sigma
    lower = floor * panel.std(ddof=1)
    if isinstance(sigma, pd.DataFrame):
        return sigma.clip(lower=lower, axis=1)
    return sigma.clip(lower=lower)


def standardized_covariance(
    returns: pd.DataFrame,
    vol_halflife: float = 20.0,
    cleaning: str = "rie",
    halflife: float | None = None,
    noise: str = "weighted",
    regularize: bool = True,
    sort: bool = True,
    min_periods: int | None = None,
    floor: float = 0.1,
    mode: str = "stock",
    standardize: bool = True,
    scale: str = "latest",
    trust: float = 0.0,
) -> CovarianceEstimate:
    """Covariance from volatility-standardised returns, scaled back by a volatility forecast.

    Two recipes for the volatility of a day:

    ``mode="stock"``
        Each stock's own :func:`ewma_volatility` (half-life ``vol_halflife``,
        past days only, burn-in dropped). The forecast for the day after
        the window is the same estimate including the window's last return.
    ``mode="market"``
        One factor per day for every stock: the cross-sectional RMS of the
        returns, each in units of its own window standard deviation, as
        Bun, Bouchaud and Potters do before cleaning. Pooled over N stocks
        it is nearly noise-free, and it needs no lag because it is used
        only to equalise days. The forecast is each stock's window
        standard deviation times the exponentially weighted forecast of
        the factor.

    The volatility does two jobs, which the flags separate so each can be
    judged on its own: ``standardize`` decides whether the correlation is
    built from the standardised returns (``True``) or the raw ones;
    ``scale`` decides whether the cleaned correlation is scaled back by the
    forecast (``"latest"``) or by the window standard deviations
    (``"window"``). ``standardize=False, scale="window"`` is the plain
    estimator. The correlation itself is built with the same day weighting
    (``halflife``) and cleaning as :func:`ewma_covariance`.

    ``trust`` (market mode only) lets each stock's own recent volatility
    move its forecast part of the way from the market-factor value. With
    ``ρ_i`` the stock's exponentially weighted volatility divided by the
    market-factor forecast ``σ̄_i · m``, the forecast is
    ``σ̄_i · m · (1 + trust · (ρ_i − 1))``: ``trust = 0`` is the market
    factor alone, ``trust = 1`` is the stock's own volatility, the same
    number the stock mode forecasts with. The dial changes the forecast
    only; the correlation is still built from the market-standardised
    returns.

    ``info`` adds the burn-in length, the pooled excess kurtosis of the raw
    and the standardised returns, the mean variance of the standardised
    returns (near one when the volatility keeps up), for the market mode
    the latest factor, and with ``trust > 0`` the spread of the ratios
    ``ρ_i``. ``floor`` is passed to :func:`ewma_volatility`.
    """
    if mode not in ("stock", "market"):
        raise ValueError("mode must be 'stock' or 'market'.")
    if scale not in ("latest", "window"):
        raise ValueError("scale must be 'latest' or 'window'.")
    if not 0.0 <= trust <= 1.0:
        raise ValueError("trust must be between 0 (market factor only) and 1 (each stock's own volatility).")
    if trust > 0 and mode != "market":
        raise ValueError("trust applies to the market mode only; the stock mode already uses each stock's own volatility.")
    panel = _dense(returns)
    if panel.shape[1] < 2:
        raise ValueError("Need at least two tickers.")
    raw = panel.to_numpy(dtype=float)
    window_sigma = raw.std(axis=0, ddof=1)
    if np.any(window_sigma <= 0):
        raise ValueError("Some tickers have zero variance over the window, so they have no correlation.")
    extra: dict[str, float] = {}
    if mode == "stock":
        sigma = ewma_volatility(panel, vol_halflife, min_periods, floor)
        keep = sigma.notna().all(axis=1).to_numpy()
        z = (raw / sigma.to_numpy(dtype=float))[keep]
        latest = _floored(np.sqrt(_ewma_variance(panel, vol_halflife, min_periods).iloc[-1]), panel, floor).to_numpy(dtype=float)
    else:
        units = raw / window_sigma
        factor = np.sqrt(np.mean(units**2, axis=1))
        keep = np.ones(raw.shape[0], dtype=bool)
        z = units / factor[:, None]
        forecast = float(np.sqrt(pd.Series(factor**2).ewm(halflife=float(vol_halflife), adjust=True).mean().iloc[-1]))
        latest = window_sigma * forecast
        extra = {"factor_latest": forecast, "factor_min": float(factor.min()), "factor_max": float(factor.max())}
        if trust > 0:
            own = _floored(np.sqrt(_ewma_variance(panel, vol_halflife, min_periods).iloc[-1]), panel, floor).to_numpy(dtype=float)
            ratio = own / latest
            latest = latest * (1.0 + trust * (ratio - 1.0))
            extra.update({"ratio_median": float(np.median(ratio)), "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max())})
    if z.shape[0] < 3:
        raise ValueError("The window is too short for the volatility burn-in; use a longer window or a shorter vol_halflife.")
    if not np.all(np.isfinite(latest)) or np.any(latest <= 0):
        raise ValueError("Could not form a volatility forecast for every ticker.")
    source = z if standardize else raw[keep]
    weights = exponential_weights(source.shape[0], halflife)
    covariance_s, scale_s = _weighted_covariance(source, weights)
    info: dict[str, float] = {
        "vol_halflife": float(vol_halflife),
        "market_mode": 1.0 if mode == "market" else 0.0,
        "standardize": 1.0 if standardize else 0.0,
        "scale_latest": 1.0 if scale == "latest" else 0.0,
        "trust": float(trust),
        "burn_in": float(raw.shape[0] - z.shape[0]),
        "halflife": float("inf") if halflife is None or not np.isfinite(halflife) else float(halflife),
        "effective_observations": effective_observations(weights),
        "q": float(source.shape[1] * np.sum(weights**2)),
        "standardized_variance": float(np.mean(np.var(z, axis=0, ddof=1))),
        "kurtosis_raw": _pooled_excess_kurtosis(raw),
        "kurtosis_standardized": _pooled_excess_kurtosis(z),
        **extra,
    }
    if cleaning == "none":
        cleaned = covariance_s / np.outer(scale_s, scale_s)
    else:
        cleaned, cleaning_info = _cleaned_correlation_of(covariance_s, scale_s, weights, cleaning, noise, regularize, sort)
        info.update(cleaning_info)
    d = latest if scale == "latest" else window_sigma
    matrix = d[:, None] * cleaned * d[None, :]
    frame = pd.DataFrame(matrix, index=panel.columns, columns=panel.columns)
    return CovarianceEstimate(frame, info)


def standardized_estimator(vol_halflife: float = 20.0, cleaning: str = "rie", **options) -> Estimator:
    """An :data:`Estimator` for :func:`standardized_covariance` with its choices fixed, for the backtest registry."""

    def estimate(returns: pd.DataFrame) -> CovarianceEstimate:
        return standardized_covariance(returns, vol_halflife, cleaning=cleaning, **options)

    estimate.__name__ = f"standardized_{cleaning}_{vol_halflife:g}"
    return estimate


def _weighted_covariance(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted covariance (weighted mean removed, divided by ``1 − Σ w²``) and its volatilities."""
    centred = x - weights @ x
    covariance = (centred * weights[:, None]).T @ centred / (1.0 - float(np.sum(weights**2)))
    variances = np.diag(covariance)
    if np.any(variances <= 0):
        raise ValueError("Some tickers have zero variance over the window, so they have no correlation.")
    return covariance, np.sqrt(variances)


def _cleaned_correlation_of(
    covariance: np.ndarray,
    volatilities: np.ndarray,
    weights: np.ndarray,
    cleaning: str,
    noise: str,
    regularize: bool,
    sort: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    """Clean the correlation implied by a (weighted) covariance; returns the cleaned correlation and diagnostics."""
    if cleaning not in ("none", "clipped", "rie"):
        raise ValueError("cleaning must be 'none', 'clipped' or 'rie'.")
    if noise not in ("weighted", "effective"):
        raise ValueError("noise must be 'weighted' or 'effective'.")
    correlation = _symmetrize(covariance / np.outer(volatilities, volatilities))
    eigenvalues, eigenvectors = _decompose(correlation)
    q = float(correlation.shape[0] * np.sum(weights**2))
    if cleaning == "none":
        return correlation, {}
    if cleaning == "clipped":
        cleaned_values, sigma2, _, lambda_plus, n_signal = clip_eigenvalues(eigenvalues, q, 1.0)
        info = {"sigma2": sigma2, "lambda_plus": lambda_plus, "n_signal": float(n_signal)}
    else:
        spectrum = rie_eigenvalues(
            eigenvalues,
            q,
            weights=weights if noise == "weighted" else None,
            regularize=regularize,
            sort=sort,
        )
        cleaned_values = spectrum.cleaned
        info = {
            "eta": spectrum.eta,
            "kappa": spectrum.kappa,
            "alpha": spectrum.alpha,
            "n_regularized": float(spectrum.n_regularized),
            "n_zero": float(spectrum.n_zero),
            "trace_scale": spectrum.trace_scale,
            "min_eigenvalue": float(spectrum.eigenvalues[-1]),
            "min_cleaned": float(spectrum.cleaned.min()),
        }
    return _cleaned_correlation(cleaned_values, eigenvectors), info


def _pooled_excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis of all entries pooled after removing each column's mean and scale."""
    centred = x - x.mean(axis=0)
    scale = centred.std(axis=0)
    scale[scale == 0] = np.nan
    z = (centred / scale).ravel()
    z = z[np.isfinite(z)]
    if z.size < 4:
        return float("nan")
    return float(np.mean(z**4) / np.mean(z**2) ** 2 - 3.0)


def ewma_estimator(halflife: float | None, cleaning: str = "rie", **options) -> Estimator:
    """An :data:`Estimator` for :func:`ewma_covariance` with the half-life and cleaning fixed, for the backtest registry."""

    def estimate(returns: pd.DataFrame) -> CovarianceEstimate:
        return ewma_covariance(returns, halflife, cleaning=cleaning, **options)

    estimate.__name__ = f"ewma_{cleaning}_{'flat' if halflife is None else halflife}"
    return estimate


ESTIMATORS: dict[str, Estimator] = {
    "sample": sample_covariance,
    "clipped": clipped_covariance,
    "clipped_unit": clipped_unit_covariance,
    "clipped_market": clipped_market_covariance,
    "diagonal": diagonal_covariance,
    "ledoit_wolf": ledoit_wolf_covariance,
    "rie": rie_covariance,
    "rie_raw": rie_raw_covariance,
    "rie_js": james_stein_rie_covariance,
    "rie_ewma": ewma_estimator(252, "rie"),
    "rie_standardized": standardized_estimator(20.0, "rie"),
}


def minimum_variance_weights(covariance: pd.DataFrame) -> pd.Series:
    """Unconstrained minimum-variance weights, ``w = Σ⁻¹1 / (1ᵀΣ⁻¹1)``.

    Weights sum to one and may be negative. When the matrix is rank
    deficient, which happens to the sample covariance whenever there are
    more tickers than independent dates, the pseudo-inverse is used instead
    and ``weights.attrs["singular"]`` is set. ``weights.attrs["condition_number"]``
    holds the ratio of the largest to the smallest positive eigenvalue.
    """
    matrix = _square(covariance)
    eigenvalues = np.linalg.eigvalsh(matrix)
    largest = float(eigenvalues[-1])
    if not np.isfinite(largest) or largest <= 0:
        raise ValueError("Covariance matrix has no positive eigenvalue.")
    cutoff = largest * _RELATIVE_RANK_TOLERANCE
    positive = eigenvalues[eigenvalues > cutoff]
    singular = positive.size < eigenvalues.size
    ones = np.ones(matrix.shape[0])
    if singular:
        raw = np.linalg.pinv(matrix, rcond=_RELATIVE_RANK_TOLERANCE, hermitian=True) @ ones
    else:
        raw = np.linalg.solve(matrix, ones)
    total = float(raw.sum())
    if not np.isfinite(total) or abs(total) < _WEIGHT_TOLERANCE:
        raise ValueError("Minimum-variance weights are undefined: 1ᵀΣ⁻¹1 is zero or not finite.")
    weights = pd.Series(raw / total, index=covariance.index, name="weight")
    weights.attrs["singular"] = bool(singular)
    weights.attrs["condition_number"] = float(largest / positive.min())
    return weights


def long_only_minimum_variance_weights(covariance: pd.DataFrame) -> pd.Series:
    """Minimum-variance weights with no shorting: ``min wᵀΣw`` over ``w ≥ 0``, ``Σw = 1``.

    There is no closed form once a sign constraint binds, but the problem is
    a non-negative least-squares problem in disguise. Minimising
    ``yᵀΣy − 2·1ᵀy`` over ``y ≥ 0`` has the optimality conditions
    ``Σy − 1 = μ`` with ``μ ≥ 0`` and ``μᵢ yᵢ = 0``. Dividing by ``1ᵀy``
    turns them into the conditions of the budgeted problem with multiplier
    ``λ = 1 / 1ᵀy`` on ``Σw = 1``, so ``w = y / 1ᵀy``. ``y`` comes from the
    Lawson–Hanson active-set algorithm (1974) in its normal-equations form:
    start with every weight at zero; admit the stock whose entry of
    ``1 − Σy`` is largest (buying it lowers the objective fastest); solve the
    unconstrained problem on the admitted stocks; if that solution turns a
    weight negative, step back toward the previous point until the first
    weight reaches zero and drop that stock; repeat until no excluded stock
    would help. The objective falls at every step, so it ends after finitely
    many of them, in practice after about as many as there are positions.

    Weights are non-negative and sum to one, and most of them are exactly
    zero. ``attrs`` carries ``singular`` and ``condition_number`` as
    :func:`minimum_variance_weights` does (a rank-deficient matrix gets a
    ridge of ``1e-10`` times its largest eigenvalue so every sub-problem is
    solvable), ``n_positions`` (weights above ``1e-10``), and ``kkt_gap``:
    how far the worst excluded stock is from its optimality condition
    ``(Σw)ᵢ ≥ wᵀΣw``, relative to ``wᵀΣw``; zero up to rounding at the
    optimum.
    """
    matrix = _square(covariance)
    assets = matrix.shape[0]
    eigenvalues = np.linalg.eigvalsh(matrix)
    largest = float(eigenvalues[-1])
    if not np.isfinite(largest) or largest <= 0:
        raise ValueError("Covariance matrix has no positive eigenvalue.")
    cutoff = largest * _RELATIVE_RANK_TOLERANCE
    positive = eigenvalues[eigenvalues > cutoff]
    singular = positive.size < eigenvalues.size
    scale = float(np.trace(matrix) / assets)
    sigma = matrix / scale
    if singular:
        sigma = sigma + np.eye(assets) * (largest / scale) * _RELATIVE_RANK_TOLERANCE
    raw = _nnls_normal_equations(sigma, np.ones(assets))
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Long-only minimum-variance weights are undefined: the solver returned no position.")
    vector = raw / total
    gradient = sigma @ vector
    variance = float(vector @ gradient)
    weights = pd.Series(vector, index=covariance.index, name="weight")
    weights.attrs["singular"] = bool(singular)
    weights.attrs["condition_number"] = float(largest / positive.min())
    weights.attrs["n_positions"] = int(np.sum(vector > _WEIGHT_TOLERANCE))
    weights.attrs["kkt_gap"] = float(max(0.0, variance - float(gradient.min())) / variance) if variance > 0 else float("nan")
    return weights


def _nnls_normal_equations(sigma: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Minimise ``yᵀΣy − 2cᵀy`` over ``y ≥ 0`` by Lawson–Hanson on the normal equations.

    ``sigma`` is symmetric positive (semi)definite. The passive set holds the
    coordinates allowed to be positive; the rest sit at zero.
    """
    n = len(c)
    tolerance = _WEIGHT_TOLERANCE * max(float(np.abs(c).max()), 1.0)
    passive = np.zeros(n, dtype=bool)
    y = np.zeros(n)
    gradient = c.copy()  # c − Σy: positive entries are coordinates whose increase lowers the objective
    budget = 20 * n + 100
    steps = 0
    while True:
        candidates = np.where(passive, -np.inf, gradient)
        entering = int(np.argmax(candidates))
        if candidates[entering] <= tolerance:
            break
        passive[entering] = True
        while True:
            steps += 1
            if steps > budget:
                raise RuntimeError("The long-only optimiser did not converge; the covariance matrix may be badly conditioned.")
            idx = np.flatnonzero(passive)
            trial = np.zeros(n)
            trial[idx] = _solve_block(sigma[np.ix_(idx, idx)], c[idx])
            if np.all(trial[idx] > 0.0):
                y = trial
                break
            blocking = idx[trial[idx] <= 0.0]
            alpha = float(np.min(y[blocking] / (y[blocking] - trial[blocking])))
            y = y + alpha * (trial - y)
            passive &= y > tolerance
            y[~passive] = 0.0
        gradient = c - sigma @ y
    return y


def _solve_block(block: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(block, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(block, rhs, rcond=None)[0]


def gross_capped_minimum_variance_weights(covariance: pd.DataFrame, cap: float) -> pd.Series:
    """Minimum-variance weights with gross exposure at most ``cap``: ``min wᵀΣw`` over ``Σw = 1``, ``Σ|wᵢ| ≤ cap``.

    ``cap`` is the total long plus short exposure per unit of capital:
    ``cap = 1`` is long-only, ``1.5`` a 125/25 book, ``2`` a 150/50 book,
    ``inf`` the unconstrained optimiser. Since the weights sum to one, the
    total short is ``(cap − 1) / 2``.

    The optimality conditions (Fan, Zhang and Yu 2012) put every stock's
    risk contribution ``(Σw)ᵢ`` in a band: ``η − μ`` for the longs, ``η + μ``
    for the shorts, anywhere inside ``[η − μ, η + μ]`` for the stocks left
    out, with ``η`` the multiplier on the budget and ``μ ≥ 0`` the shadow
    price of exposure (zero when the cap does not bind). The code is a
    primal active-set method on that structure. It starts from the long-only
    portfolio, feasible for every ``cap ≥ 1``. Each step solves the
    equality-constrained problem on the stocks currently held with their
    current signs (the budget, plus the cap once it binds), moves toward
    that solution until a weight would change sign or the exposure would
    reach the cap, and records that event. At a sub-problem's minimiser it
    releases the excluded stock furthest outside the band (long if below,
    short if above), drops the cap if its shadow price has turned negative,
    or stops when every condition holds. If the unconstrained portfolio
    already satisfies the cap it is returned directly.

    ``attrs`` carries ``singular`` and ``condition_number`` as the other
    rules do, ``n_positions``, ``n_short``, ``cap``, ``cap_active`` (did the
    constraint bind), ``cap_multiplier`` (``μ`` in the covariance's units)
    and ``kkt_gap`` (the worst violation of the band, relative to ``η``).
    """
    if not np.isfinite(cap):
        weights = minimum_variance_weights(covariance)
        vector = weights.to_numpy(dtype=float)
        weights.attrs.update(
            {
                "n_positions": int(np.sum(np.abs(vector) > _WEIGHT_TOLERANCE)),
                "n_short": int(np.sum(vector < -_WEIGHT_TOLERANCE)),
                "cap": float("inf"),
                "cap_active": False,
                "cap_multiplier": 0.0,
            }
        )
        return weights
    if cap < 1.0:
        raise ValueError("cap must be at least 1: the weights sum to one, so gross exposure cannot be lower.")
    if cap == 1.0:
        weights = long_only_minimum_variance_weights(covariance)
        weights.attrs.update({"n_short": 0, "cap": 1.0, "cap_active": True, "cap_multiplier": float("nan")})
        return weights
    matrix = _square(covariance)
    assets = matrix.shape[0]
    eigenvalues = np.linalg.eigvalsh(matrix)
    largest = float(eigenvalues[-1])
    if not np.isfinite(largest) or largest <= 0:
        raise ValueError("Covariance matrix has no positive eigenvalue.")
    cutoff = largest * _RELATIVE_RANK_TOLERANCE
    positive = eigenvalues[eigenvalues > cutoff]
    singular = positive.size < eigenvalues.size
    scale = float(np.trace(matrix) / assets)
    sigma = matrix / scale
    if singular:
        sigma = sigma + np.eye(assets) * (largest / scale) * _RELATIVE_RANK_TOLERANCE
    ones = np.ones(assets)
    unconstrained = _solve_block(sigma, ones)
    unconstrained = unconstrained / unconstrained.sum()
    if np.abs(unconstrained).sum() <= cap:
        vector, eta, mu, active = unconstrained, float(unconstrained @ sigma @ unconstrained), 0.0, False
    else:
        start = _nnls_normal_equations(sigma, ones)
        vector, eta, mu, active = _active_set_gross_cap(sigma, start / start.sum(), cap)
    gradient = sigma @ vector
    held = np.abs(vector) > _WEIGHT_TOLERANCE
    violation = 0.0
    if (~held).any():
        violation = float((np.abs(gradient[~held] - eta) - max(mu, 0.0)).max())
    weights = pd.Series(vector, index=covariance.index, name="weight")
    weights.attrs["singular"] = bool(singular)
    weights.attrs["condition_number"] = float(largest / positive.min())
    weights.attrs["n_positions"] = int(held.sum())
    weights.attrs["n_short"] = int(np.sum(vector < -_WEIGHT_TOLERANCE))
    weights.attrs["cap"] = float(cap)
    weights.attrs["cap_active"] = bool(active)
    weights.attrs["cap_multiplier"] = float(mu * scale)
    weights.attrs["kkt_gap"] = float(max(0.0, violation) / eta) if eta > 0 else float("nan")
    return weights


def _active_set_gross_cap(sigma: np.ndarray, start: np.ndarray, cap: float) -> tuple[np.ndarray, float, float, bool]:
    """Primal active-set method for ``min wᵀΣw`` over ``Σw = 1``, ``Σ|w| ≤ cap``, from a feasible ``start``.

    Returns the weights, the budget multiplier ``η``, the cap's shadow price
    ``μ`` and whether the cap binds. ``sigma`` must be positive definite.
    """
    n = len(start)
    tolerance = _WEIGHT_TOLERANCE
    w = np.where(np.abs(start) > tolerance, start, 0.0)
    sign = np.sign(w)
    active = False
    eta = mu = 0.0
    for _ in range(50 * n + 100):
        free = np.flatnonzero(sign)
        g = sign[free]
        k = len(free)
        a = np.array([np.ones(k), g]) if active else np.ones((1, k))
        m = a.shape[0]
        kkt = np.zeros((k + m, k + m))
        kkt[:k, :k] = sigma[np.ix_(free, free)]
        kkt[:k, k:] = -a.T
        kkt[k:, :k] = a
        rhs = np.concatenate([np.zeros(k), [1.0, cap][:m]])
        solution = _solve_block(kkt, rhs)
        target = np.zeros(n)
        target[free] = solution[:k]
        multipliers = solution[k:]
        direction = target - w
        alpha, blocking, hits_cap = 1.0, None, False
        crossing = free[g * target[free] <= 0.0]  # weights that would vanish or change sign on the way
        if crossing.size:
            steps = w[crossing] / (w[crossing] - target[crossing])
            best = int(np.argmin(steps))
            if steps[best] < alpha:
                alpha, blocking = float(steps[best]), int(crossing[best])
        if not active:
            growth = float(g @ direction[free])
            if growth > tolerance:
                to_cap = (cap - float(np.abs(w).sum())) / growth
                if to_cap < alpha:
                    alpha, blocking, hits_cap = max(to_cap, 0.0), None, True
        w = w + alpha * direction
        if blocking is not None:
            w[blocking] = 0.0
            sign[blocking] = 0.0
            continue
        if hits_cap:
            active = True
            continue
        eta = float(multipliers[0])
        mu = -float(multipliers[1]) if active else 0.0
        if active and mu < -tolerance:
            active, mu = False, 0.0
            continue
        excluded = np.flatnonzero(sign == 0)
        if excluded.size == 0:
            break
        offset = sigma[excluded] @ w - eta
        band = max(mu, 0.0)
        slack = max(abs(eta), 1.0) * 1e-9
        wants_long = -(offset + band)  # positive when the stock sits below the band
        wants_short = offset - band  # positive when the stock sits above the band
        i_long, i_short = int(np.argmax(wants_long)), int(np.argmax(wants_short))
        if wants_long[i_long] <= slack and wants_short[i_short] <= slack:
            break
        if wants_long[i_long] >= wants_short[i_short]:
            sign[excluded[i_long]] = 1.0
        else:
            sign[excluded[i_short]] = -1.0
    else:
        raise RuntimeError("The gross-exposure optimiser did not converge; the covariance matrix may be badly conditioned.")
    return w, eta, mu, active


def equal_weights(tickers: Iterable[str]) -> pd.Series:
    """One over N in every name."""
    names = [str(ticker) for ticker in tickers]
    if not names:
        raise ValueError("Need at least one ticker.")
    weights = pd.Series(1.0 / len(names), index=names, name="weight")
    weights.attrs["singular"] = False
    weights.attrs["condition_number"] = 1.0
    return weights


def portfolio_variance(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """``wᵀ Σ w`` with the covariance aligned to the weight index."""
    aligned = covariance.loc[weights.index, weights.index].to_numpy(dtype=float)
    vector = weights.to_numpy(dtype=float)
    return float(vector @ aligned @ vector)


def weight_statistics(weights: pd.Series) -> dict[str, float]:
    """Concentration numbers: gross leverage, largest position, shorts, positions held, effective N."""
    vector = weights.to_numpy(dtype=float)
    return {
        "gross_leverage": float(np.abs(vector).sum()),
        "max_weight": float(np.abs(vector).max()),
        "n_short": int(np.sum(vector < -_WEIGHT_TOLERANCE)),
        "n_positions": int(np.sum(np.abs(vector) > _WEIGHT_TOLERANCE)),
        "effective_n": float(1.0 / np.sum(vector**2)),
    }


def turnover(new: pd.Series, previous: pd.Series | None) -> float:
    """Sum of absolute weight changes. From cash, it is the gross leverage of ``new``."""
    if previous is None:
        return float(np.abs(new.to_numpy(dtype=float)).sum())
    union = new.index.union(previous.index)
    delta = new.reindex(union, fill_value=0.0) - previous.reindex(union, fill_value=0.0)
    return float(np.abs(delta.to_numpy(dtype=float)).sum())


def _dense(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame with one column per ticker.")
    if returns.shape[0] < 2:
        raise ValueError("Need at least two return dates to estimate a covariance matrix.")
    if returns.shape[1] < 1:
        raise ValueError("Need at least one ticker.")
    if returns.columns.duplicated().any():
        raise ValueError("Return columns must be unique tickers.")
    if returns.isna().any().any():
        raise ValueError("The return window has missing values. Keep only tickers with a complete window.")
    panel = returns.copy()
    panel.columns = [str(column) for column in panel.columns]
    return panel


def _square(covariance: pd.DataFrame) -> np.ndarray:
    if not isinstance(covariance, pd.DataFrame):
        raise ValueError("covariance must be a DataFrame.")
    if covariance.shape[0] != covariance.shape[1] or covariance.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix.")
    if not covariance.index.equals(covariance.columns):
        raise ValueError("covariance index and columns must list the same tickers in the same order.")
    matrix = covariance.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("covariance has non-finite entries.")
    return (matrix + matrix.T) / 2.0
