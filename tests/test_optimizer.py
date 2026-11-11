import itertools

import numpy as np
import pandas as pd
import pytest

from market_analyzer.optimizer import (
    ESTIMATORS,
    clipped_covariance,
    diagonal_covariance,
    equal_weights,
    gross_capped_minimum_variance_weights,
    ledoit_wolf_covariance,
    long_only_minimum_variance_weights,
    minimum_variance_weights,
    portfolio_variance,
    sample_covariance,
    turnover,
    weight_statistics,
)


def _factor_panel(seed: int, n_dates: int, n_assets: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.01, size=n_dates)
    betas = rng.uniform(0.6, 1.4, size=n_assets)
    idio = rng.uniform(0.008, 0.02, size=n_assets)
    returns = market[:, None] * betas[None, :] + rng.normal(size=(n_dates, n_assets)) * idio
    frame = pd.DataFrame(returns, columns=[f"S{i:03d}" for i in range(n_assets)])
    frame.index = pd.bdate_range("2018-01-01", periods=n_dates)
    return frame


def test_two_asset_minimum_variance_matches_the_closed_form():
    sigma1, sigma2, rho = 0.2, 0.3, 0.4
    covariance = pd.DataFrame(
        [[sigma1**2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2**2]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    weights = minimum_variance_weights(covariance)
    expected_a = (sigma2**2 - rho * sigma1 * sigma2) / (sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
    assert weights["A"] == pytest.approx(expected_a)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.attrs["singular"] is False
    # Any other fully invested weight has higher variance.
    for alternative in (0.0, 0.5, 1.0):
        other = pd.Series([alternative, 1 - alternative], index=["A", "B"])
        assert portfolio_variance(weights, covariance) <= portfolio_variance(other, covariance) + 1e-12


def test_diagonal_covariance_gives_inverse_variance_weights():
    variances = np.array([0.01, 0.04, 0.02])
    covariance = pd.DataFrame(np.diag(variances), index=list("ABC"), columns=list("ABC"))
    weights = minimum_variance_weights(covariance)
    expected = (1 / variances) / (1 / variances).sum()
    assert weights.to_numpy() == pytest.approx(expected)


def test_rank_deficient_covariance_falls_back_to_the_pseudo_inverse():
    panel = _factor_panel(0, n_dates=3, n_assets=5)
    covariance = sample_covariance(panel).covariance
    weights = minimum_variance_weights(covariance)
    assert weights.attrs["singular"] is True
    assert weights.sum() == pytest.approx(1.0)
    assert np.isfinite(weights.to_numpy()).all()


def test_ledoit_wolf_shortcut_matches_the_brute_force_sum():
    panel = _factor_panel(1, n_dates=40, n_assets=6)
    estimate = ledoit_wolf_covariance(panel)
    x = panel.to_numpy() - panel.to_numpy().mean(axis=0)
    t, n = x.shape
    sample = x.T @ x / t
    mean_variance = np.trace(sample) / n
    delta2 = np.sum((sample - mean_variance * np.eye(n)) ** 2) / n
    beta2_bar = sum(np.sum((np.outer(row, row) - sample) ** 2) / n for row in x) / t**2
    shrinkage = min(beta2_bar, delta2) / delta2
    assert estimate.info["shrinkage"] == pytest.approx(shrinkage)
    expected = ((1 - shrinkage) * sample + shrinkage * mean_variance * np.eye(n)) * t / (t - 1)
    assert estimate.covariance.to_numpy() == pytest.approx(expected)


def test_ledoit_wolf_shrinks_less_as_the_sample_grows():
    short = ledoit_wolf_covariance(_factor_panel(2, n_dates=30, n_assets=20))
    long = ledoit_wolf_covariance(_factor_panel(2, n_dates=3000, n_assets=20))
    assert 0.0 < long.info["shrinkage"] < short.info["shrinkage"] < 1.0
    assert np.linalg.eigvalsh(short.covariance.to_numpy()).min() > 0


def test_every_estimator_returns_an_aligned_square_matrix():
    panel = _factor_panel(3, n_dates=300, n_assets=12)
    sample_variances = panel.var(ddof=1).to_numpy()
    for key, estimator in ESTIMATORS.items():
        estimate = estimator(panel)
        matrix = estimate.covariance
        assert matrix.shape == (12, 12), key
        assert list(matrix.index) == list(panel.columns), key
        assert list(matrix.columns) == list(panel.columns), key
        assert matrix.to_numpy() == pytest.approx(matrix.to_numpy().T), key
    for estimator in (sample_covariance, clipped_covariance, diagonal_covariance):
        assert np.diag(estimator(panel).covariance.to_numpy()) == pytest.approx(sample_variances)
    assert clipped_covariance(panel).info["n_signal"] >= 1


def test_estimators_reject_missing_values():
    panel = _factor_panel(4, n_dates=50, n_assets=4)
    panel.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="missing"):
        sample_covariance(panel)


def _two_asset(sigma1: float, sigma2: float, rho: float) -> pd.DataFrame:
    return pd.DataFrame(
        [[sigma1**2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2**2]],
        index=["A", "B"],
        columns=["A", "B"],
    )


def _brute_force_long_only(matrix: np.ndarray) -> np.ndarray:
    """Try every support: the unconstrained solution on a subset that comes out non-negative is a candidate."""
    n = matrix.shape[0]
    best, best_variance = None, np.inf
    for mask in range(1, 2**n):
        idx = [i for i in range(n) if mask >> i & 1]
        block = matrix[np.ix_(idx, idx)]
        raw = np.linalg.solve(block, np.ones(len(idx)))
        if raw.sum() <= 0 or (raw < -1e-12).any():
            continue
        w = np.zeros(n)
        w[idx] = raw / raw.sum()
        variance = w @ matrix @ w
        if variance < best_variance:
            best, best_variance = w, variance
    return best


def test_long_only_drops_the_short_leg_in_the_two_asset_example():
    # Unconstrained wants +124% / −24%; long-only puts everything in the calm stock.
    weights = long_only_minimum_variance_weights(_two_asset(0.2, 0.3, 0.8))
    assert weights.to_numpy() == pytest.approx([1.0, 0.0])
    assert weights.attrs["n_positions"] == 1
    assert weights.attrs["kkt_gap"] == pytest.approx(0.0, abs=1e-10)


def test_long_only_equals_unconstrained_when_no_weight_is_negative():
    covariance = _two_asset(0.2, 0.3, 0.4)
    unconstrained = minimum_variance_weights(covariance)
    assert (unconstrained > 0).all()
    constrained = long_only_minimum_variance_weights(covariance)
    assert constrained.to_numpy() == pytest.approx(unconstrained.to_numpy())
    variances = np.array([0.01, 0.04, 0.02])
    diagonal = pd.DataFrame(np.diag(variances), index=list("ABC"), columns=list("ABC"))
    assert long_only_minimum_variance_weights(diagonal).to_numpy() == pytest.approx((1 / variances) / (1 / variances).sum())


def test_long_only_matches_the_brute_force_search_on_small_problems():
    for seed in range(12):
        panel = _factor_panel(100 + seed, n_dates=40, n_assets=6)
        covariance = sample_covariance(panel).covariance
        weights = long_only_minimum_variance_weights(covariance)
        expected = _brute_force_long_only(covariance.to_numpy())
        assert weights.to_numpy() == pytest.approx(expected, abs=1e-9), seed
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0).all()
        assert weights.attrs["kkt_gap"] == pytest.approx(0.0, abs=1e-9)
        assert weights.attrs["n_positions"] == int((expected > 1e-10).sum())


def test_long_only_is_feasible_optimal_and_never_beats_unconstrained_in_sample():
    panel = _factor_panel(7, n_dates=300, n_assets=12)
    covariance = sample_covariance(panel).covariance
    constrained = long_only_minimum_variance_weights(covariance)
    unconstrained = minimum_variance_weights(covariance)
    assert (constrained >= 0).all()
    assert constrained.sum() == pytest.approx(1.0)
    assert weight_statistics(constrained)["n_short"] == 0
    assert weight_statistics(constrained)["gross_leverage"] == pytest.approx(1.0)
    assert portfolio_variance(constrained, covariance) >= portfolio_variance(unconstrained, covariance) - 1e-14
    # The optimality condition: every stock held has (Σw)ᵢ = wᵀΣw, every stock excluded has (Σw)ᵢ ≥ wᵀΣw.
    gradient = covariance.to_numpy() @ constrained.to_numpy()
    variance = portfolio_variance(constrained, covariance)
    held = constrained.to_numpy() > 1e-10
    assert gradient[held] == pytest.approx(variance, rel=1e-8)
    assert (gradient[~held] >= variance * (1 - 1e-8)).all()


def test_long_only_handles_a_rank_deficient_covariance():
    panel = _factor_panel(0, n_dates=3, n_assets=5)
    covariance = sample_covariance(panel).covariance
    weights = long_only_minimum_variance_weights(covariance)
    assert weights.attrs["singular"] is True
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    assert np.isfinite(weights.to_numpy()).all()


def _brute_force_gross_cap(matrix: np.ndarray, cap: float) -> np.ndarray:
    """Try every sign pattern, with the cap slack and binding: the optimum is one of these equality-constrained solutions."""
    n = matrix.shape[0]
    best, best_variance = None, np.inf
    for pattern in itertools.product((-1.0, 0.0, 1.0), repeat=n):
        g = np.array(pattern)
        free = np.flatnonzero(g)
        if free.size == 0:
            continue
        for binding in (False, True):
            if binding and np.all(g[free] == g[free][0]):
                continue  # budget and cap rows coincide: no solution with the cap binding
            a = np.array([np.ones(free.size), g[free]]) if binding else np.ones((1, free.size))
            k, m = free.size, a.shape[0]
            kkt = np.zeros((k + m, k + m))
            kkt[:k, :k] = matrix[np.ix_(free, free)]
            kkt[:k, k:] = -a.T
            kkt[k:, :k] = a
            rhs = np.concatenate([np.zeros(k), [1.0, cap][:m]])
            try:
                solution = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                continue
            w = np.zeros(n)
            w[free] = solution[:k]
            if np.any(g[free] * w[free] <= 0) or np.abs(w).sum() > cap + 1e-9:
                continue
            variance = w @ matrix @ w
            if variance < best_variance:
                best, best_variance = w, variance
    return best


def test_gross_cap_matches_the_brute_force_search_on_small_problems():
    binding = 0
    for seed in range(6):
        panel = _factor_panel(200 + seed, n_dates=40, n_assets=6)
        covariance = sample_covariance(panel).covariance
        for cap in (1.2, 1.5, 2.0):
            weights = gross_capped_minimum_variance_weights(covariance, cap)
            expected = _brute_force_gross_cap(covariance.to_numpy(), cap)
            assert weights.to_numpy() == pytest.approx(expected, abs=1e-8), (seed, cap)
            assert weights.sum() == pytest.approx(1.0)
            assert np.abs(weights).sum() <= cap + 1e-9
            assert weights.attrs["kkt_gap"] == pytest.approx(0.0, abs=1e-8)
            binding += int(weights.attrs["cap_active"])
    assert binding > 0


def test_gross_cap_interpolates_between_long_only_and_unconstrained():
    panel = _factor_panel(9, n_dates=400, n_assets=15)
    covariance = sample_covariance(panel).covariance
    unconstrained = minimum_variance_weights(covariance)
    long_only = long_only_minimum_variance_weights(covariance)
    gross = float(np.abs(unconstrained).sum())
    assert gross > 1.5
    loose = gross_capped_minimum_variance_weights(covariance, gross + 1.0)
    assert loose.to_numpy() == pytest.approx(unconstrained.to_numpy())
    assert loose.attrs["cap_active"] is False
    assert gross_capped_minimum_variance_weights(covariance, float("inf")).to_numpy() == pytest.approx(unconstrained.to_numpy())
    assert gross_capped_minimum_variance_weights(covariance, 1.0).to_numpy() == pytest.approx(long_only.to_numpy())
    previous = portfolio_variance(long_only, covariance)
    matrix = covariance.to_numpy()
    for cap in (1.1, 1.25, 1.5, 2.0, 3.0):
        weights = gross_capped_minimum_variance_weights(covariance, cap)
        assert weights.sum() == pytest.approx(1.0)
        assert np.abs(weights).sum() == pytest.approx(min(cap, gross), abs=1e-8)
        assert weights.attrs["cap_active"] is (cap < gross)
        variance = portfolio_variance(weights, covariance)
        assert portfolio_variance(unconstrained, covariance) - 1e-14 <= variance <= previous + 1e-14
        previous = variance
        # The band: longs share one risk contribution, shorts a higher one, excluded stocks sit in between.
        w = weights.to_numpy()
        gradient = matrix @ w
        longs, shorts, out = w > 1e-10, w < -1e-10, np.abs(w) <= 1e-10
        low = float(gradient[longs].mean())
        assert gradient[longs] == pytest.approx(low, rel=1e-6)
        high = low
        if shorts.any():
            high = float(gradient[shorts].mean())
            assert gradient[shorts] == pytest.approx(high, rel=1e-6)
            assert low <= high + 1e-12
        if out.any():
            assert (gradient[out] >= low - 1e-8 * abs(low)).all()
            assert (gradient[out] <= high + 1e-8 * abs(high)).all()
    with pytest.raises(ValueError, match="at least 1"):
        gross_capped_minimum_variance_weights(covariance, 0.5)


def test_weight_statistics_and_turnover():
    weights = pd.Series([0.7, 0.5, -0.2], index=list("ABC"))
    stats = weight_statistics(weights)
    assert stats["gross_leverage"] == pytest.approx(1.4)
    assert stats["max_weight"] == pytest.approx(0.7)
    assert stats["n_short"] == 1
    assert stats["n_positions"] == 3
    assert stats["effective_n"] == pytest.approx(1 / (0.49 + 0.25 + 0.04))
    assert turnover(weights, None) == pytest.approx(1.4)
    previous = pd.Series([0.5, 0.5], index=list("AB"))
    assert turnover(weights, previous) == pytest.approx(0.2 + 0.0 + 0.2)
    equal = equal_weights(["A", "B", "C", "D"])
    assert equal.to_numpy() == pytest.approx([0.25] * 4)
