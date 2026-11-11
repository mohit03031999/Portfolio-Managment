from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_analyzer.cli import main
from market_analyzer.matrices import (
    CorrelationModel,
    clip_eigenvalues,
    marchenko_pastur_density,
    marchenko_pastur_edges,
)
from market_analyzer.models import MarketData
from market_analyzer.pipeline import save_market_data


def test_marchenko_pastur_edges_at_unit_variance():
    lower, upper = marchenko_pastur_edges(q=0.25, sigma2=1.0)
    assert lower == pytest.approx(0.25)
    assert upper == pytest.approx(2.25)


def test_marchenko_pastur_density_integrates_to_one_when_q_is_below_one():
    q, sigma2 = 0.25, 1.0
    lower, upper = marchenko_pastur_edges(q, sigma2)
    grid = np.linspace(lower, upper, 20001)
    mass = np.trapezoid(marchenko_pastur_density(grid, q, sigma2), grid)
    assert mass == pytest.approx(1.0, abs=1e-3)


def test_clip_keeps_the_eigenvalue_above_the_bulk():
    clipped, sigma2, lambda_minus, lambda_plus, n_signal = clip_eigenvalues(
        np.array([3.0, 0.5, 0.3, 0.2]),
        q=0.25,
    )
    assert n_signal == 1
    assert sigma2 == pytest.approx(1.0 / 3.0)
    assert lambda_plus == pytest.approx((1.0 / 3.0) * 2.25)
    assert lambda_minus == pytest.approx((1.0 / 3.0) * 0.25)
    assert clipped == pytest.approx([3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def test_clip_flattens_a_spectrum_that_is_all_noise():
    clipped, sigma2, _, lambda_plus, n_signal = clip_eigenvalues(
        np.array([1.2, 1.1, 0.9, 0.8]),
        q=0.25,
    )
    assert n_signal == 0
    assert sigma2 == pytest.approx(1.0)
    assert lambda_plus == pytest.approx(2.25)
    assert clipped == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_correlation_matches_the_pearson_gram_matrix():
    returns = pd.DataFrame(
        np.random.default_rng(0).normal(size=(40, 4)),
        columns=list("ABCD"),
    )
    model = CorrelationModel.from_returns(returns)
    standardized = (returns - returns.mean()) / returns.std(ddof=1)
    gram = (standardized.T @ standardized) / (len(returns) - 1)
    assert model.q == pytest.approx(4 / 39)
    assert model.correlation.to_numpy() == pytest.approx(gram.to_numpy())
    assert model.covariance.to_numpy() == pytest.approx(returns.cov().to_numpy())


def test_factor_model_keeps_the_market_mode_and_the_variances():
    rng = np.random.default_rng(7)
    market = rng.normal(size=600)
    noise = rng.normal(size=(600, 25))
    returns = pd.DataFrame(
        market[:, None] + 0.5 * noise,
        columns=[f"S{i:02d}" for i in range(25)],
    )
    model = CorrelationModel.from_returns(returns)

    assert model.n_signal >= 1
    assert bool(model.eigenvalues.iloc[0]["above_noise_edge"])
    assert int(model.eigenvalues["above_noise_edge"].sum()) == model.n_signal
    assert np.diag(model.cleaned_correlation) == pytest.approx(np.ones(25))
    assert np.diag(model.cleaned_covariance) == pytest.approx(returns.var(ddof=1).to_numpy())
    assert np.linalg.norm(model.signal_loadings.iloc[:, 0].to_numpy()) == pytest.approx(1.0)
    assert np.linalg.eigvalsh(model.cleaned_correlation.to_numpy()).min() > -1e-8
    assert "Marchenko-Pastur" in model.note()


def test_missing_and_flat_columns_are_rejected():
    dated = pd.DataFrame({"A": [0.01, 0.02, None], "B": [0.01, -0.01, 0.03]})
    with pytest.raises(ValueError, match="missing values"):
        CorrelationModel.from_returns(dated)
    flat = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="zero variance"):
        CorrelationModel.from_returns(flat)


def test_save_writes_the_cleaned_matrices(tmp_path: Path):
    rng = np.random.default_rng(1)
    market = rng.normal(size=80)
    noise = rng.normal(size=(80, 6))
    returns = pd.DataFrame(0.8 * market[:, None] + 0.5 * noise, columns=list("ABCDEF"))
    returns.index = pd.bdate_range("2024-01-01", periods=80)
    returns.index.name = "Date"
    data = MarketData(
        name="factors",
        start="2024-01-01",
        end="2024-06-01",
        requested_tickers=list(returns.columns),
        missing_tickers=[],
        prices=returns,
        returns=returns,
    )
    paths = save_market_data(data, tmp_path)
    assert paths["cleaned_covariance"].is_file()
    assert paths["eigenvalues"].is_file()
    assert any("Marchenko-Pastur" in note for note in data.notes)
    cleaned = pd.read_csv(paths["cleaned_correlation"], index_col=0)
    assert cleaned.to_numpy().diagonal() == pytest.approx(np.ones(6))


def test_matrices_command_reads_a_saved_return_file(tmp_path: Path, capsys):
    rng = np.random.default_rng(3)
    market = rng.normal(size=100)
    noise = rng.normal(size=(100, 8))
    returns = pd.DataFrame(market[:, None] + 0.4 * noise, columns=[f"S{i}" for i in range(8)])
    returns.index = pd.bdate_range("2024-01-01", periods=100)
    returns.index.name = "Date"
    source = tmp_path / "returns.csv"
    returns.to_csv(source)

    code = main(["matrices", str(source)])
    assert code == 0
    assert "Marchenko-Pastur" in capsys.readouterr().out
    assert (tmp_path / "cleaned_covariance.csv").is_file()
    assert (tmp_path / "signal_loadings.csv").is_file()


def test_fixed_and_market_edges_count_only_what_clears_them():
    spectrum = np.array([3.0, 0.5, 0.3, 0.2])
    clipped, sigma2, _, lambda_plus, n_signal = clip_eigenvalues(spectrum, q=0.25, sigma2=1.0)
    assert sigma2 == 1.0
    assert lambda_plus == pytest.approx(2.25)
    assert n_signal == 1
    assert clipped == pytest.approx([3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    _, sigma2, _, lambda_plus, n_signal = clip_eigenvalues(spectrum, q=0.25, sigma2="market")
    assert sigma2 == pytest.approx(1.0 / 3.0)
    assert lambda_plus == pytest.approx(2.25 / 3.0)
    assert n_signal == 1

    _, sigma2, _, lambda_plus, n_signal = clip_eigenvalues(spectrum, q=0.25, sigma2="2.0")
    assert sigma2 == 2.0
    assert n_signal == 0

    with pytest.raises(ValueError, match="sigma2 must be"):
        clip_eigenvalues(spectrum, q=0.25, sigma2="huge")
    with pytest.raises(ValueError, match="positive"):
        clip_eigenvalues(spectrum, q=0.25, sigma2=-1.0)


def test_unit_edge_keeps_fewer_modes_than_the_fitted_edge_on_a_wide_bulk():
    rng = np.random.default_rng(11)
    n_dates, n_assets = 260, 120
    market = rng.normal(size=n_dates)
    loadings = rng.uniform(0.5, 1.5, size=n_assets)
    # Idiosyncratic noise with unequal variances widens the bulk beyond the pure Marchenko-Pastur shape.
    noise = rng.normal(size=(n_dates, n_assets)) * rng.uniform(0.5, 2.5, size=n_assets)
    returns = pd.DataFrame(market[:, None] * loadings[None, :] + noise, columns=[f"S{i:03d}" for i in range(n_assets)])
    fitted = CorrelationModel.from_returns(returns)
    unit = CorrelationModel.from_returns(returns, sigma2=1.0)
    assert fitted.edge_rule == "fitted"
    assert unit.edge_rule == "sigma2=1"
    assert unit.sigma2 == 1.0
    assert 1 <= unit.n_signal <= fitted.n_signal
    assert "edge rule" in unit.note()
    assert np.diag(unit.cleaned_covariance) == pytest.approx(returns.var(ddof=1).to_numpy())
