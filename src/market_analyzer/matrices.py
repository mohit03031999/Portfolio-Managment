"""Correlation, covariance, and a noise-cleaned correlation.

A portfolio optimizer needs a covariance matrix. The one you compute from
T daily returns and N stocks is a noisy estimate. Random matrix theory
describes that noise: the Marchenko-Pastur law gives the band where
eigenvalues of a purely random correlation matrix must land. Eigenvalues
above the band are the market and sector structure worth keeping. Eigenvalues
inside the band are replaced by their average. That cleaned correlation,
rescaled by each stock's own volatility, is the covariance to use later
for portfolio weights.

The ratio used here is q = N / (T - 1). Pearson correlation de-means each
column and divides by T - 1, so T - 1 is the matching sample size. Papers
that build the Gram matrix as (1/T) X'X often write q = N/T instead. The
two edges agree once T is large.

Where the noise edge sits depends on the noise variance ``sigma2`` fed to
the Marchenko-Pastur formula, and three conventions are supported:

``"fitted"``
    Default. Find the smallest number of leading eigenvalues such that the
    average of everything below them sets an edge that everything below
    them respects. On real equity data this can label a large share of the
    spectrum as signal, because the empirical bulk is wider than the
    theoretical one.
``1.0`` or any positive number
    A fixed noise variance. ``1.0`` is the textbook edge for a correlation
    matrix and keeps roughly the market plus the sector modes.
``"market"``
    The average of every eigenvalue except the largest, which is the noise
    variance left once the market mode has taken its share of the trace.
    This is the convention of Laloux, Cizeau, Bouchaud and Potters (1999).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_EDGE_TOLERANCE = 1e-8
EDGE_FITTED = "fitted"
EDGE_MARKET = "market"


@dataclass(frozen=True)
class CorrelationModel:
    """Sample risk matrices for one dense return panel, plus the cleaned copies."""

    observations: int
    assets: int
    q: float
    sigma2: float
    lambda_minus: float
    lambda_plus: float
    n_signal: int
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    eigenvalues: pd.DataFrame
    cleaned_correlation: pd.DataFrame
    cleaned_covariance: pd.DataFrame
    signal_loadings: pd.DataFrame
    edge_rule: str = EDGE_FITTED

    @classmethod
    def from_returns(cls, returns: pd.DataFrame, sigma2: float | str = EDGE_FITTED) -> CorrelationModel:
        """Estimate the matrices from a date × ticker return panel.

        The panel has to be complete. A missing day in one column means the
        columns were not observed on the same dates, and the Marchenko-Pastur
        edge no longer applies. Pull with ``--align intersection`` for that
        panel.
        """
        panel = _complete_returns(returns)
        tickers = [str(column) for column in panel.columns]
        observations, assets = panel.shape
        effective = observations - 1
        q = assets / effective

        covariance = panel.cov()
        correlation = panel.corr()
        matrix = _symmetrize(correlation.to_numpy(dtype=float))
        raw_eigenvalues, eigenvectors = _decompose(matrix)
        edge_rule = _edge_rule(sigma2)
        clipped, noise_variance, lambda_minus, lambda_plus, n_signal = clip_eigenvalues(raw_eigenvalues, q, sigma2)
        cleaned = _cleaned_correlation(clipped, eigenvectors)
        cleaned_correlation = pd.DataFrame(cleaned, index=tickers, columns=tickers)
        volatilities = panel.std(ddof=1).to_numpy(dtype=float)
        cleaned_covariance = pd.DataFrame(
            volatilities[:, None] * cleaned * volatilities[None, :],
            index=tickers,
            columns=tickers,
        )
        covariance.index = tickers
        covariance.columns = tickers
        correlation.index = tickers
        correlation.columns = tickers

        return cls(
            observations=observations,
            assets=assets,
            q=q,
            sigma2=noise_variance,
            lambda_minus=lambda_minus,
            lambda_plus=lambda_plus,
            n_signal=n_signal,
            covariance=covariance,
            correlation=correlation,
            eigenvalues=_spectrum(raw_eigenvalues, clipped, n_signal),
            cleaned_correlation=cleaned_correlation,
            cleaned_covariance=cleaned_covariance,
            signal_loadings=_loadings(tickers, eigenvectors, n_signal),
            edge_rule=edge_rule,
        )

    def note(self) -> str:
        """One-line description of how much of the spectrum survived the noise cut."""
        text = (
            f"Correlation spectrum: {self.n_signal} of {self.assets} eigenvalues "
            f"sit above the Marchenko-Pastur noise edge {self.lambda_plus:.3f} "
            f"(q={self.q:.3f}, noise variance={self.sigma2:.3f}, edge rule {self.edge_rule})."
        )
        if self.observations - 1 < self.assets:
            text += " Fewer independent dates than tickers, so use cleaned_covariance for weights."
        return text

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Write the six matrices next to each other."""
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        paths = {
            "covariance": folder / "covariance.csv",
            "correlation": folder / "correlation.csv",
            "eigenvalues": folder / "eigenvalues.csv",
            "cleaned_correlation": folder / "cleaned_correlation.csv",
            "cleaned_covariance": folder / "cleaned_covariance.csv",
            "signal_loadings": folder / "signal_loadings.csv",
        }
        self.covariance.to_csv(paths["covariance"])
        self.correlation.to_csv(paths["correlation"])
        self.eigenvalues.to_csv(paths["eigenvalues"])
        self.cleaned_correlation.to_csv(paths["cleaned_correlation"])
        self.cleaned_covariance.to_csv(paths["cleaned_covariance"])
        self.signal_loadings.to_csv(paths["signal_loadings"])
        return paths


def marchenko_pastur_edges(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Return the lower and upper edges of the noise band.

    For variance ``sigma2`` and ``q = N / T_eff``, the edges are
    ``sigma2 * (1 ± sqrt(q))²``.
    """
    if q <= 0:
        raise ValueError("q must be positive.")
    if sigma2 < 0:
        raise ValueError("sigma2 must be non-negative.")
    width = np.sqrt(q)
    return float(sigma2 * (1.0 - width) ** 2), float(sigma2 * (1.0 + width) ** 2)


def marchenko_pastur_density(values: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur density on the continuous noise band.

    The density is zero outside ``[λ-, λ+]``. For ``q < 1`` it integrates to 1.
    For ``q > 1`` the same curve integrates to ``1/q`` and the missing mass sits
    at eigenvalue 0.
    """
    grid = np.asarray(values, dtype=float)
    density = np.zeros(grid.shape, dtype=float)
    if q <= 0 or sigma2 <= 0:
        return density
    lower, upper = marchenko_pastur_edges(q, sigma2)
    span = (upper - grid) * (grid - lower)
    inside = (grid > 0) & (span > 0)
    density[inside] = np.sqrt(span[inside]) / (2.0 * np.pi * sigma2 * q * grid[inside])
    return density


def clip_eigenvalues(
    eigenvalues: np.ndarray,
    q: float,
    sigma2: float | str = EDGE_FITTED,
) -> tuple[np.ndarray, float, float, float, int]:
    """Replace bulk eigenvalues with their average.

    ``eigenvalues`` must be sorted from largest to smallest. ``sigma2`` is
    the noise variance that sets the edge: ``"fitted"`` searches for a
    self-consistent split, ``"market"`` uses the average eigenvalue after
    removing the largest one, and a positive number is used as given.

    Returns the clipped eigenvalues, the noise variance, the two edges,
    and the number of eigenvalues left unchanged.
    """
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Need at least two eigenvalues.")
    if np.any(np.diff(values) > _EDGE_TOLERANCE):
        raise ValueError("Pass eigenvalues sorted from largest to smallest.")
    if values.min() < -1e-8:
        raise ValueError("Correlation eigenvalues must be non-negative.")
    values = np.clip(values, 0.0, None)

    rule = _edge_rule(sigma2)
    if rule == EDGE_FITTED:
        split = _consistent_split(values, q)
        if split is None:
            noise_variance = 1.0
            lambda_minus, lambda_plus = marchenko_pastur_edges(q, noise_variance)
            n_signal = int(np.sum(values > lambda_plus))
        else:
            n_signal, noise_variance, lambda_minus, lambda_plus = split
    else:
        if rule == EDGE_MARKET:
            noise_variance = float((values.sum() - values[0]) / (values.size - 1))
        else:
            noise_variance = float(sigma2)
        lambda_minus, lambda_plus = marchenko_pastur_edges(q, noise_variance)
        n_signal = int(np.sum(values > lambda_plus + _EDGE_TOLERANCE))

    clipped = values.copy()
    # n_signal == 0 still replaces the whole spectrum. Every eigenvalue is
    # noise, so the cleaned correlation collapses toward the identity.
    if n_signal < values.size:
        clipped[n_signal:] = values[n_signal:].sum() / (values.size - n_signal)
    return clipped, noise_variance, lambda_minus, lambda_plus, n_signal


def _edge_rule(sigma2: float | str) -> str:
    """Normalise the ``sigma2`` argument to ``fitted``, ``market``, or a number as text."""
    if isinstance(sigma2, str):
        if sigma2 in (EDGE_FITTED, EDGE_MARKET):
            return sigma2
        try:
            value = float(sigma2)
        except ValueError as exc:
            raise ValueError(
                f"sigma2 must be {EDGE_FITTED!r}, {EDGE_MARKET!r}, or a positive number, got {sigma2!r}."
            ) from exc
    elif isinstance(sigma2, (int, float)) and not isinstance(sigma2, bool):
        value = float(sigma2)
    else:
        raise ValueError(f"sigma2 must be {EDGE_FITTED!r}, {EDGE_MARKET!r}, or a positive number.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError("A numeric sigma2 must be positive.")
    return f"sigma2={value:g}"


def _consistent_split(
    eigenvalues: np.ndarray,
    q: float,
) -> tuple[int, float, float, float] | None:
    """Find how many leading eigenvalues clear the edge set by the rest."""
    count = eigenvalues.size
    for n_signal in range(count):
        n_bulk = count - n_signal
        sigma2 = float(eigenvalues[n_signal:].sum() / n_bulk)
        lambda_minus, lambda_plus = marchenko_pastur_edges(q, sigma2)
        largest_bulk = float(eigenvalues[n_signal])
        smallest_signal = np.inf if n_signal == 0 else float(eigenvalues[n_signal - 1])
        bulk_is_inside = largest_bulk <= lambda_plus + _EDGE_TOLERANCE
        signal_is_outside = smallest_signal > lambda_plus
        if bulk_is_inside and signal_is_outside:
            return n_signal, sigma2, lambda_minus, lambda_plus
    return None


def _complete_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame with one column per ticker.")
    panel = returns.copy()
    if panel.shape[0] < 2:
        raise ValueError("Need at least two return dates to estimate a correlation matrix.")
    if panel.shape[1] < 2:
        raise ValueError("Need at least two tickers to estimate a correlation matrix.")
    if panel.columns.duplicated().any():
        raise ValueError("Return columns must be unique tickers.")
    if panel.isna().any().any():
        raise ValueError(
            "The return matrix has missing values. Denoising needs every ticker on every date. "
            "Use --align intersection when you pull the basket."
        )
    volatility = panel.std(ddof=1)
    flat = [str(ticker) for ticker in volatility.index if volatility[ticker] == 0 or pd.isna(volatility[ticker])]
    if flat:
        raise ValueError("These columns have zero variance, so they have no correlation: " + ", ".join(flat))
    return panel


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    np.fill_diagonal(symmetric, 1.0)
    return symmetric


def _decompose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _cleaned_correlation(clipped: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    cleaned = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    cleaned = (cleaned + cleaned.T) / 2.0
    scale = np.sqrt(np.maximum(np.diag(cleaned), 1e-12))
    cleaned = cleaned / np.outer(scale, scale)
    cleaned = (cleaned + cleaned.T) / 2.0
    np.fill_diagonal(cleaned, 1.0)
    return cleaned


def _spectrum(eigenvalues: np.ndarray, clipped: np.ndarray, n_signal: int) -> pd.DataFrame:
    total = float(eigenvalues.sum())
    share = eigenvalues / total if total else np.zeros_like(eigenvalues)
    flags = np.zeros(eigenvalues.size, dtype=bool)
    flags[:n_signal] = True
    frame = pd.DataFrame(
        {
            "eigenvalue": eigenvalues,
            "variance_share": share,
            "cumulative_share": np.cumsum(share),
            "above_noise_edge": flags,
            "clipped_eigenvalue": clipped,
        },
        index=pd.Index(range(1, eigenvalues.size + 1), name="mode"),
    )
    return frame


def _loadings(tickers: list[str], eigenvectors: np.ndarray, n_signal: int) -> pd.DataFrame:
    keep = n_signal if n_signal else 1
    columns = [f"mode_{mode}" for mode in range(1, keep + 1)]
    frame = pd.DataFrame(eigenvectors[:, :keep], index=tickers, columns=columns)
    frame.index.name = "ticker"
    return frame
