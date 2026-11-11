import numpy as np
import pandas as pd
import pytest

from market_analyzer.optimizer import ESTIMATORS, minimum_variance_weights, rie_covariance, rie_raw_covariance
from market_analyzer.rie import (
    inverse_wishart_edges,
    inverse_wishart_kappa,
    inverse_wishart_stieltjes,
    marchenko_pastur_stieltjes,
    rie_eigenvalues,
    shrink,
    stieltjes_transform,
)


def _sorted_spectrum(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _sample_correlation(rng: np.random.Generator, truth: np.ndarray, n_dates: int) -> np.ndarray:
    x = rng.standard_normal((n_dates, truth.shape[0])) @ np.linalg.cholesky(truth).T
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    return x.T @ x / (n_dates - 1)


def _factor_truth(rng: np.random.Generator, n_assets: int) -> np.ndarray:
    betas = rng.uniform(0.5, 1.5, n_assets)
    sector = np.zeros(n_assets)
    sector[: n_assets // 3] = 1.0
    truth = 0.3 * np.outer(betas, betas) + 0.15 * np.outer(sector, sector) + np.diag(rng.uniform(0.3, 1.5, n_assets))
    scale = np.sqrt(np.diag(truth))
    return truth / np.outer(scale, scale)


def test_inverse_wishart_transform_turns_the_formula_into_linear_shrinkage():
    # For an inverse-Wishart truth the optimal cleaning is 1 + alpha (lambda - 1),
    # Bun, Bouchaud and Potters eq. 6.24; this checks the transform and its branch.
    q, kappa = 0.5, 2.0
    lower, upper = inverse_wishart_edges(q, kappa)
    alpha = 1.0 / (1.0 + 2.0 * q * kappa)
    grid = np.linspace(lower + 1e-3, upper - 1e-3, 25)
    z = grid - 1e-9j
    cleaned = shrink(z, q, inverse_wishart_stieltjes(z, q, kappa))
    assert cleaned == pytest.approx(1.0 + alpha * (grid - 1.0), abs=1e-6)
    assert inverse_wishart_kappa(lower, q) == pytest.approx(kappa)


def test_marchenko_pastur_is_the_large_kappa_limit():
    q = 0.4
    z = np.array([0.5, 1.0, 2.0]) - 1e-6j
    big = inverse_wishart_stieltjes(z, q, 1e7)
    assert big == pytest.approx(marchenko_pastur_stieltjes(z, q), abs=1e-5)
    # Pure noise maps to one everywhere inside the bulk.
    assert shrink(z, q, marchenko_pastur_stieltjes(z, q)) == pytest.approx(np.ones(3), abs=1e-4)


def test_kappa_is_undefined_when_the_spectrum_is_narrower_than_noise():
    q = 0.5
    edge = (1.0 - np.sqrt(q)) ** 2
    assert np.isfinite(inverse_wishart_kappa(edge * 0.9, q))
    assert np.isnan(inverse_wishart_kappa(edge * 1.1, q))
    assert np.isnan(inverse_wishart_kappa(0.0, q))


def test_stieltjes_transform_leaves_each_eigenvalue_out_of_its_own_sum():
    values = np.array([3.0, 1.0, 0.5])
    z = values - 0.1j
    g = stieltjes_transform(z, values)
    expected = np.array([sum(1.0 / (z[i] - values[j]) for j in range(3) if j != i) / 2.0 for i in range(3)])
    assert g == pytest.approx(expected)
    assert np.all(g.imag > 0)


def test_pure_noise_is_pulled_toward_one_and_the_lift_fixes_the_left_edge():
    rng = np.random.default_rng(0)
    n_assets, n_dates = 200, 400
    sample = _sample_correlation(rng, np.eye(n_assets), n_dates)
    eigenvalues, _ = _sorted_spectrum(sample)
    q = n_assets / (n_dates - 1)
    raw = rie_eigenvalues(eigenvalues, q, regularize=False, sort=False)
    cleaned = rie_eigenvalues(eigenvalues, q)
    assert np.median(raw.raw) == pytest.approx(1.0, abs=0.05)
    assert raw.raw.min() < 0.7  # the finite-N dive at the left edge
    assert cleaned.cleaned.min() > 0.85
    assert cleaned.cleaned.max() < 1.25
    assert cleaned.kappa == np.inf and cleaned.alpha == 0.0
    assert cleaned.n_regularized > 0
    assert cleaned.cleaned.sum() == pytest.approx(eigenvalues.sum())
    assert np.all(np.diff(cleaned.cleaned) <= 1e-12)


def test_rie_tracks_the_oracle_and_its_portfolio_is_as_good_as_the_oracle_portfolio():
    rng = np.random.default_rng(1)
    n_assets, n_dates = 240, 480
    truth = _factor_truth(rng, n_assets)
    sample = _sample_correlation(rng, truth, n_dates)
    eigenvalues, eigenvectors = _sorted_spectrum(sample)
    oracle = np.einsum("ij,jk,ki->i", eigenvectors.T, truth, eigenvectors)
    q = n_assets / (n_dates - 1)
    raw = rie_eigenvalues(eigenvalues, q, regularize=False, sort=False).cleaned
    cleaned = rie_eigenvalues(eigenvalues, q).cleaned

    def squared_error(values: np.ndarray) -> float:
        return float(np.mean((values - oracle) ** 2))

    assert squared_error(raw) < squared_error(eigenvalues) / 5
    # The smallest eigenvalues are where the raw formula is biased low and the lift helps.
    tail = slice(-20, None)
    assert np.mean(raw[tail]) < np.mean(oracle[tail]) - 0.1
    assert abs(np.mean(cleaned[tail]) - np.mean(oracle[tail])) < 0.1

    ones = np.ones(n_assets)
    optimum = np.sqrt(1.0 / (ones @ np.linalg.solve(truth, ones)))

    def true_over_optimal(values: np.ndarray) -> float:
        matrix = eigenvectors @ np.diag(values) @ eigenvectors.T
        weights = np.linalg.solve(matrix, ones)
        weights /= weights.sum()
        return float(np.sqrt(weights @ truth @ weights) / optimum)

    assert true_over_optimal(cleaned) < true_over_optimal(raw) < true_over_optimal(eigenvalues)
    assert true_over_optimal(cleaned) == pytest.approx(true_over_optimal(oracle), abs=0.01)


def test_a_huge_sample_leaves_the_spectrum_alone():
    rng = np.random.default_rng(2)
    n_assets, n_dates = 20, 100_000
    sample = _sample_correlation(rng, _factor_truth(rng, n_assets), n_dates)
    eigenvalues, _ = _sorted_spectrum(sample)
    q = n_assets / (n_dates - 1)
    raw = rie_eigenvalues(eigenvalues, q, regularize=False, sort=False)
    assert np.max(np.abs(raw.cleaned - eigenvalues)) < 1e-3
    assert np.max(np.abs(rie_eigenvalues(eigenvalues, q).cleaned - eigenvalues)) < 0.05


def test_more_assets_than_dates_fills_the_null_eigenvalues():
    rng = np.random.default_rng(3)
    n_assets, n_dates = 60, 40
    sample = _sample_correlation(rng, np.eye(n_assets), n_dates)
    eigenvalues, _ = _sorted_spectrum(sample)
    q = n_assets / (n_dates - 1)
    spectrum = rie_eigenvalues(eigenvalues, q)
    assert spectrum.n_zero == n_assets - (n_dates - 1)
    assert spectrum.raw[-1] == pytest.approx(1.0, abs=0.2)
    assert spectrum.cleaned.min() > 0.5
    assert spectrum.cleaned.sum() == pytest.approx(eigenvalues.sum())
    with pytest.raises(ValueError, match="q > 1"):
        rie_eigenvalues(eigenvalues, 0.5)


def test_rie_rejects_bad_input():
    with pytest.raises(ValueError, match="sorted"):
        rie_eigenvalues(np.array([0.5, 1.0, 2.0]), 0.5)
    with pytest.raises(ValueError, match="q must be positive"):
        rie_eigenvalues(np.array([2.0, 1.0, 0.5]), 0.0)
    with pytest.raises(ValueError, match="eta"):
        rie_eigenvalues(np.array([2.0, 1.0, 0.5]), 0.5, eta=0.0)


def _panel(seed: int, n_dates: int, n_assets: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    truth = _factor_truth(rng, n_assets)
    vols = rng.uniform(0.01, 0.03, n_assets)
    x = rng.standard_normal((n_dates, n_assets)) @ np.linalg.cholesky(truth).T * vols
    frame = pd.DataFrame(x, columns=[f"S{i:03d}" for i in range(n_assets)])
    frame.index = pd.bdate_range("2018-01-01", periods=n_dates)
    return frame


def test_rie_estimator_keeps_variances_and_is_registered():
    panel = _panel(4, n_dates=300, n_assets=60)
    estimate = rie_covariance(panel)
    matrix = estimate.covariance
    assert list(matrix.index) == list(panel.columns) and list(matrix.columns) == list(panel.columns)
    assert np.diag(matrix.to_numpy()) == pytest.approx(panel.var(ddof=1).to_numpy())
    assert np.linalg.eigvalsh(matrix.to_numpy()).min() > 0
    assert estimate.info["q"] == pytest.approx(60 / 299)
    assert estimate.info["eta"] == pytest.approx(60**-0.5)
    assert estimate.info["n_regularized"] >= 1
    assert ESTIMATORS["rie"] is rie_covariance and ESTIMATORS["rie_raw"] is rie_raw_covariance
    weights = minimum_variance_weights(matrix)
    assert weights.sum() == pytest.approx(1.0)


def test_regularised_rie_is_better_conditioned_than_the_raw_formula():
    panel = _panel(5, n_dates=300, n_assets=120)
    raw = rie_raw_covariance(panel)
    cleaned = rie_covariance(panel)
    assert np.isnan(raw.info["kappa"]) and raw.info["n_regularized"] == 0
    assert cleaned.info["min_cleaned"] > raw.info["min_cleaned"]
    assert minimum_variance_weights(cleaned.covariance).attrs["condition_number"] < minimum_variance_weights(raw.covariance).attrs["condition_number"]


# ----------------------------------------------------------------------------- weighted samples

from market_analyzer.optimizer import ewma_covariance, ewma_estimator, sample_covariance  # noqa: E402
from market_analyzer.rie import (  # noqa: E402
    effective_observations,
    exponential_weights,
    weighted_r_transform,
    weighted_s_transform,
    weighted_shrink,
)


def test_exponential_weights_halve_every_halflife_and_sum_to_one():
    weights = exponential_weights(600, 100)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[-1] > weights[0]
    assert weights[-1] / weights[-101] == pytest.approx(2.0)
    assert exponential_weights(50, None) == pytest.approx(np.full(50, 0.02))
    assert effective_observations(exponential_weights(50, None)) == pytest.approx(50.0)
    # Long window: Kish's effective size tends to (1 + eps) / (1 - eps) ~ 2.885 half-lives.
    assert effective_observations(exponential_weights(20_000, 252)) == pytest.approx(2.885 * 252, rel=0.01)


def test_flat_weights_give_the_marchenko_pastur_transforms():
    n_assets, n_obs = 300, 600
    flat = np.full(n_obs, 1.0 / n_obs)
    q = n_assets / n_obs
    x = np.array([0.3 + 0.2j, -0.5 + 0.1j, 2.0 + 1.0j])
    assert weighted_r_transform(x, flat, n_assets) == pytest.approx(1.0 / (1.0 - q * x))
    assert weighted_s_transform(x, flat, n_assets) == pytest.approx(1.0 / (1.0 + q * x))
    # And the weighted map reproduces the Ledoit-Peche closed form up to the O(eta) finite-size
    # convention, which differs most near the lower edge where Re(g) is large (about 10% there).
    rng = np.random.default_rng(5)
    eigenvalues, _ = _sorted_spectrum(np.corrcoef(rng.standard_normal((n_obs, n_assets)), rowvar=False))
    z = eigenvalues - 1j * n_assets**-0.5
    g = stieltjes_transform(z, eigenvalues)
    assert weighted_shrink(z, g, flat, n_assets) == pytest.approx(shrink(z, q, g), rel=0.12)
    assert weighted_shrink(z, g, flat, n_assets)[:200] == pytest.approx(shrink(z, q, g)[:200], rel=0.03)


def test_s_transform_matches_the_closed_form_for_long_exponential_weights():
    # With an effectively infinite past, w_t = (1 - eps) eps^t and N (1 - eps) = q_e, the S-transform is
    # (1 - exp(-q_e omega)) / (q_e omega). Near omega = -1 with q_eff > 1 the Marchenko-Pastur start
    # sits next to a pole, which is where the plain Newton solve used to diverge.
    n_assets, n_obs, halflife = 300, 20_000, 100
    weights = exponential_weights(n_obs, halflife)
    eps = 0.5 ** (1.0 / halflife)
    q_e = n_assets * (1.0 - eps)
    omega = np.array([0.3 + 0.2j, -0.5 + 0.1j, -0.95 + 0.01j, -0.999 + 1e-4j, 0.01 + 1e-5j])
    closed_form = (1.0 - np.exp(-q_e * omega)) / (q_e * omega)
    solved = weighted_s_transform(omega, weights, n_assets)
    # The closed form is the eps -> 1 continuum limit; with eps = 0.993 the discrete sum sits within 1%.
    assert solved == pytest.approx(closed_form, rel=1e-2)
    assert np.isfinite(solved).all()
    residual = solved * weighted_r_transform(omega * solved, weights, n_assets) - 1.0
    assert np.max(np.abs(residual)) < 1e-10  # but it is an exact root of the discrete equation
    x = np.array([0.1 + 0.05j, -0.3 + 0.1j, -2.0 + 0.2j])
    assert weighted_r_transform(x, weights, n_assets) == pytest.approx(-np.log(1 - q_e * x) / (q_e * x), rel=1e-2)
    assert n_assets * np.sum(weights**2) > 1.0  # q_eff above one, the hard regime


def test_weighted_rie_stays_finite_with_tiny_eigenvalues_and_q_above_one():
    rng = np.random.default_rng(9)
    n_assets, n_obs = 200, 500
    truth = _factor_truth(rng, n_assets)
    truth[:, 1] = truth[:, 0] * 0.999 + truth[:, 1] * 0.001  # a near-duplicate pair: a true eigenvalue near zero
    truth[1, :] = truth[:, 1]
    truth[1, 1] = 1.0
    truth = (truth + truth.T) / 2 + 1e-3 * np.eye(n_assets)
    scale = np.sqrt(np.diag(truth))
    truth = truth / np.outer(scale, scale)
    weights = exponential_weights(n_obs, 40)  # q_eff about 1.7
    x = rng.standard_normal((n_obs, n_assets)) @ np.linalg.cholesky(truth).T
    x = x - weights @ x
    covariance = (x * weights[:, None]).T @ x / (1 - np.sum(weights**2))
    d = np.sqrt(np.diag(covariance))
    eigenvalues, _ = _sorted_spectrum(covariance / np.outer(d, d))
    spectrum = rie_eigenvalues(eigenvalues, weights=weights)
    assert spectrum.q > 1.0
    assert np.isfinite(spectrum.cleaned).all()
    assert spectrum.cleaned.min() > 0
    assert spectrum.cleaned.sum() == pytest.approx(eigenvalues.sum())


def test_exponentially_weighted_noise_is_wider_than_marchenko_pastur_at_the_same_variance():
    rng = np.random.default_rng(0)
    n_assets, n_obs = 400, 8000
    weights = exponential_weights(n_obs, 277.3)  # Kish T_eff = 800, q_eff = 0.5
    q_eff = n_assets * np.sum(weights**2)
    assert q_eff == pytest.approx(0.5, abs=0.01)
    y = rng.standard_normal((n_obs, n_assets))
    eigenvalues = np.linalg.eigvalsh((y * weights[:, None]).T @ y)
    assert eigenvalues.mean() == pytest.approx(1.0, abs=0.02)
    assert eigenvalues.var() == pytest.approx(q_eff, rel=0.05)
    lower, upper = (1 - np.sqrt(q_eff)) ** 2, (1 + np.sqrt(q_eff)) ** 2
    assert eigenvalues.min() > lower * 1.5  # softer lower edge
    assert eigenvalues.max() > upper * 1.05  # longer upper tail


def test_weighted_rie_tracks_the_oracle_where_the_effective_q_shortcut_does_not():
    rng = np.random.default_rng(1)
    n_assets, n_obs = 300, 6000
    truth = _factor_truth(rng, n_assets)
    weights = exponential_weights(n_obs, 104)  # q_eff ~ 1
    x = rng.standard_normal((n_obs, n_assets)) @ np.linalg.cholesky(truth).T
    x = x - weights @ x
    covariance = (x * weights[:, None]).T @ x / (1 - np.sum(weights**2))
    scale = np.sqrt(np.diag(covariance))
    eigenvalues, eigenvectors = _sorted_spectrum(covariance / np.outer(scale, scale))
    oracle = np.einsum("ij,jk,ki->i", eigenvectors.T, truth, eigenvectors)
    exact = rie_eigenvalues(eigenvalues, weights=weights, regularize=False, sort=False)
    shortcut = rie_eigenvalues(eigenvalues, exact.q, regularize=False, sort=False)
    assert exact.q == pytest.approx(n_assets * np.sum(weights**2))
    assert np.mean((exact.cleaned - oracle) ** 2) < 0.2 * np.mean((shortcut.cleaned - oracle) ** 2)
    ones = np.ones(n_assets)
    optimum = np.sqrt(1.0 / (ones @ np.linalg.solve(truth, ones)))

    def true_over_optimal(values: np.ndarray) -> float:
        matrix = eigenvectors @ np.diag(values) @ eigenvectors.T
        w = np.linalg.solve(matrix, ones)
        w /= w.sum()
        return float(np.sqrt(w @ truth @ w) / optimum)

    assert true_over_optimal(exact.cleaned) == pytest.approx(true_over_optimal(oracle), abs=0.03)


def test_ewma_covariance_with_a_flat_window_is_the_sample_covariance():
    panel = _panel(6, n_dates=300, n_assets=30)
    flat = ewma_covariance(panel, None, cleaning="none")
    assert flat.covariance.to_numpy() == pytest.approx(sample_covariance(panel).covariance.to_numpy())
    assert flat.info["effective_observations"] == pytest.approx(300.0)
    weighted = ewma_covariance(panel, 40, cleaning="none")
    assert weighted.info["effective_observations"] < 150
    assert np.linalg.eigvalsh(weighted.covariance.to_numpy()).min() > 0
    # Recent days dominate: the oldest row carries 0.5^(299/40) of the newest one's weight.
    bumped = panel.copy()
    bumped.iloc[0] *= 5
    flat_move = np.abs(ewma_covariance(bumped, None, cleaning="none").covariance.to_numpy() - flat.covariance.to_numpy()).max()
    weighted_move = np.abs(ewma_covariance(bumped, 40, cleaning="none").covariance.to_numpy() - weighted.covariance.to_numpy()).max()
    assert weighted_move < 0.05 * flat_move


def test_ewma_estimators_clean_and_are_registered():
    panel = _panel(7, n_dates=800, n_assets=80)
    for cleaning in ("clipped", "rie"):
        estimate = ewma_covariance(panel, 126, cleaning=cleaning)
        matrix = estimate.covariance.to_numpy()
        assert matrix == pytest.approx(matrix.T)
        assert np.linalg.eigvalsh(matrix).min() > 0
        assert estimate.info["halflife"] == 126
        assert estimate.info["q"] == pytest.approx(80 * np.sum(exponential_weights(800, 126) ** 2))
    assert "n_signal" in ewma_covariance(panel, 126, cleaning="clipped").info
    assert ewma_covariance(panel, 126, cleaning="rie").info["n_regularized"] >= 0
    effective = ewma_covariance(panel, 126, cleaning="rie", noise="effective")
    assert effective.covariance.shape == (80, 80)
    assert ESTIMATORS["rie_ewma"](panel).info["halflife"] == 252
    weights = minimum_variance_weights(ewma_estimator(63, "rie")(panel).covariance)
    assert weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="cleaning"):
        ewma_covariance(panel, 126, cleaning="magic")


# ----------------------------------------------------------------------------- volatility standardisation

from market_analyzer.optimizer import ewma_volatility, rie_covariance, standardized_covariance, standardized_estimator  # noqa: E402


def test_ewma_volatility_uses_only_the_past_and_tracks_a_constant_level():
    rng = np.random.default_rng(11)
    panel = pd.DataFrame(rng.standard_normal((2000, 3)) * np.array([0.01, 0.02, 0.03]), columns=list("ABC"))
    sigma = ewma_volatility(panel, 20)
    assert sigma.iloc[:20].isna().all().all()  # burn-in
    assert sigma.iloc[20:].notna().all().all()
    # The level is right on average, within the noise of ~60 effective days.
    assert sigma.iloc[200:].mean().to_numpy() == pytest.approx([0.01, 0.02, 0.03], rel=0.05)
    # No peeking: changing a return moves the estimate only from the next day on.
    bumped = panel.copy()
    bumped.iloc[1000, 0] = 0.5
    other = ewma_volatility(bumped, 20)
    assert other.iloc[:1001].equals(sigma.iloc[:1001]) or np.allclose(other.iloc[:1001].to_numpy(), sigma.iloc[:1001].to_numpy(), equal_nan=True)
    assert other.iloc[1001, 0] > 5 * sigma.iloc[1001, 0]


def test_standardised_returns_have_unit_variance_and_lose_their_fat_tails():
    rng = np.random.default_rng(12)
    n_dates, n_assets = 1500, 40
    truth = _factor_truth(rng, n_assets)
    # A common volatility clock: calm, then a storm, then calm.
    clock = np.ones(n_dates)
    clock[700:800] = 4.0
    x = (rng.standard_normal((n_dates, n_assets)) @ np.linalg.cholesky(truth).T) * clock[:, None] * 0.01
    panel = pd.DataFrame(x, columns=[f"S{i:02d}" for i in range(n_assets)])
    estimate = standardized_covariance(panel, 20, cleaning="none")
    assert estimate.info["standardized_variance"] == pytest.approx(1.0, abs=0.15)
    assert estimate.info["kurtosis_raw"] > 5
    assert estimate.info["kurtosis_standardized"] < 1.5
    assert estimate.info["burn_in"] == 20


def test_standardised_covariance_scales_with_the_latest_volatility_not_the_window_average():
    rng = np.random.default_rng(13)
    n_dates, n_assets = 1000, 30
    truth = _factor_truth(rng, n_assets)
    clock = np.where(np.arange(n_dates) < 800, 1.0, 3.0)  # the last 200 days are three times as volatile
    x = (rng.standard_normal((n_dates, n_assets)) @ np.linalg.cholesky(truth).T) * clock[:, None] * 0.01
    panel = pd.DataFrame(x, columns=[f"S{i:02d}" for i in range(n_assets)])
    plain = rie_covariance(panel)
    standardized = standardized_covariance(panel, 20, cleaning="rie")
    plain_vol = np.sqrt(np.diag(plain.covariance.to_numpy())).mean()
    std_vol = np.sqrt(np.diag(standardized.covariance.to_numpy())).mean()
    recent = panel.iloc[-100:].std().mean()
    assert std_vol == pytest.approx(recent, rel=0.2)
    assert plain_vol < 0.7 * recent  # the window average is dragged down by the calm 800 days
    # Correlations are unaffected by the common clock, so the two cleaned correlations agree closely.
    def corr(estimate):
        m = estimate.covariance.to_numpy()
        d = np.sqrt(np.diag(m))
        return m / np.outer(d, d)
    assert np.abs(corr(plain) - corr(standardized)).max() < 0.25
    assert np.linalg.eigvalsh(standardized.covariance.to_numpy()).min() > 0


def test_standardised_estimator_is_registered_and_composes_with_day_weights():
    panel = _panel(14, n_dates=600, n_assets=50)
    estimate = ESTIMATORS["rie_standardized"](panel)
    assert estimate.info["vol_halflife"] == 20.0
    assert estimate.covariance.shape == (50, 50)
    weighted = standardized_estimator(40, "clipped", halflife=126)(panel)
    assert weighted.info["halflife"] == 126 and "n_signal" in weighted.info
    assert weighted.info["effective_observations"] < 600 - weighted.info["burn_in"]
    weights = minimum_variance_weights(estimate.covariance)
    assert weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="burn-in"):
        standardized_covariance(panel.iloc[:15], 20)
    # The flat EWMA path is unchanged by the refactor.
    assert ewma_covariance(panel, None, cleaning="none").covariance.to_numpy() == pytest.approx(sample_covariance(panel).covariance.to_numpy())


def test_market_mode_and_the_two_jobs_of_volatility_can_be_separated():
    rng = np.random.default_rng(15)
    n_dates, n_assets = 1000, 30
    truth = _factor_truth(rng, n_assets)
    clock = np.where(np.arange(n_dates) < 800, 1.0, 3.0)
    x = (rng.standard_normal((n_dates, n_assets)) @ np.linalg.cholesky(truth).T) * clock[:, None] * 0.01
    panel = pd.DataFrame(x, columns=[f"S{i:02d}" for i in range(n_assets)])
    market = standardized_covariance(panel, 20, cleaning="none", mode="market")
    assert market.info["market_mode"] == 1.0 and market.info["burn_in"] == 0
    assert market.info["standardized_variance"] == pytest.approx(1.0, abs=0.05)
    # The latest factor is the storm level in units of the window sigma: 3 / sqrt((800 + 200 * 9) / 1000).
    assert market.info["factor_latest"] == pytest.approx(3.0 / np.sqrt(2.6), rel=0.1)
    recent = panel.iloc[-100:].std().mean()
    assert np.sqrt(np.diag(market.covariance.to_numpy())).mean() == pytest.approx(recent, rel=0.2)
    # scale="window" keeps the window variances on the diagonal; standardize=False with scale="window" is the plain estimator.
    window_scaled = standardized_covariance(panel, 20, cleaning="none", mode="market", scale="window")
    assert np.diag(window_scaled.covariance.to_numpy()) == pytest.approx(panel.var(ddof=1).to_numpy())
    plain = standardized_covariance(panel, 20, cleaning="none", mode="market", standardize=False, scale="window")
    assert plain.covariance.to_numpy() == pytest.approx(sample_covariance(panel).covariance.to_numpy())
    with pytest.raises(ValueError, match="mode"):
        standardized_covariance(panel, 20, mode="galaxy")


def test_trust_dial_moves_the_forecast_from_the_market_factor_toward_each_stocks_own_volatility():
    # Forty stocks at 1% a day; in the last 60 days the first one alone turns three times as volatile.
    rng = np.random.default_rng(11)
    n_days, n_assets = 400, 40
    sigma = np.full((n_days, n_assets), 0.01)
    sigma[-60:, 0] *= 3.0
    panel = pd.DataFrame(rng.standard_normal((n_days, n_assets)) * sigma, columns=[f"S{i}" for i in range(n_assets)])

    def vols(estimate):
        return np.sqrt(np.diag(estimate.covariance.to_numpy()))

    market = standardized_covariance(panel, 20, cleaning="none", mode="market")
    own = standardized_covariance(panel, 20, cleaning="none", mode="stock")
    zero = standardized_covariance(panel, 20, cleaning="none", mode="market", trust=0.0)
    half = standardized_covariance(panel, 20, cleaning="none", mode="market", trust=0.5)
    full = standardized_covariance(panel, 20, cleaning="none", mode="market", trust=1.0)
    # trust 0 is the market factor; trust 1 is each stock's own volatility, the number the stock mode forecasts with.
    assert np.allclose(vols(zero), vols(market))
    assert np.allclose(vols(full), vols(own))
    # In between, the forecast is linear in the volatility: half trust is the midpoint.
    assert np.allclose(vols(half), 0.5 * (vols(market) + vols(own)))
    # The first stock's own storm shows as a ratio well above one (its window σ̄ and the market factor absorb part of it).
    assert half.info["trust"] == 0.5 and half.info["ratio_max"] > 1.5
    assert vols(market)[0] < vols(half)[0] < vols(full)[0]
    # The dial changes the forecast only: the correlation is the market-standardised one in every case.
    corr = lambda e: e.covariance.to_numpy() / np.outer(vols(e), vols(e))
    assert np.allclose(corr(half), corr(market))
    with pytest.raises(ValueError):
        standardized_covariance(panel, 20, mode="market", trust=1.5)
    with pytest.raises(ValueError):
        standardized_covariance(panel, 20, mode="stock", trust=0.5)


from market_analyzer.optimizer import james_stein_rie_covariance  # noqa: E402


def _one_factor_panel(n_assets: int, n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    betas = rng.uniform(0.3, 1.7, n_assets)
    market = rng.standard_normal(n_days)
    x = market[:, None] * betas[None, :] + 1.5 * rng.standard_normal((n_days, n_assets))
    return pd.DataFrame(x, columns=[f"S{i}" for i in range(n_assets)])


def test_james_stein_rie_pulls_the_market_loadings_toward_their_average_and_falls_back_to_the_rie():
    panel = _one_factor_panel(60, 200, seed=3)
    plain = rie_covariance(panel).covariance
    unchanged = james_stein_rie_covariance(panel, shrink=1.0)
    assert np.allclose(unchanged.covariance.to_numpy(), plain.to_numpy(), atol=1e-10)
    assert unchanged.info["turn_degrees"] < 1e-6

    estimate = james_stein_rie_covariance(panel)
    info = estimate.info
    assert 0.0 < info["c_data"] < 1.0 and info["c"] == info["c_data"]
    assert info["turn_degrees"] > 0
    matrix = estimate.covariance.to_numpy()
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), panel.var(ddof=1).to_numpy())
    assert np.linalg.eigvalsh(matrix).min() > 0

    def loading_spread(cov: np.ndarray) -> float:
        vol = np.sqrt(np.diag(cov))
        values, vectors = np.linalg.eigh(cov / np.outer(vol, vol))
        top = vectors[:, -1] * np.sign(vectors[:, -1].sum())
        return float(top.std() / top.mean())

    assert loading_spread(matrix) < loading_spread(plain.to_numpy())
    assert ESTIMATORS["rie_js"] is james_stein_rie_covariance
    with pytest.raises(ValueError):
        james_stein_rie_covariance(panel, shrink=1.5)
