# Market analyzer

Pull adjusted prices and a daily return matrix for **any companies you choose**. A basket is a list of [Yahoo Finance](https://finance.yahoo.com) symbols. It can be one stock, a personal watchlist, or a whole exchange list you point at yourself. Nothing is hardcoded to the S&P 500.

The guide for finding a symbol, exchange suffixes, and the return matrix is [docs/getting-stock-data.md](docs/getting-stock-data.md).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If `python -m venv` reports that `ensurepip` is unavailable, `uv` does the same job:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

`python -m market_analyzer pull ...` is the same command as `market-analyzer pull ...`.

## Pull a basket

Pass symbols directly:

```bash
market-analyzer pull \
  --tickers AAPL RELIANCE.NS HSBA.L \
  --start 2020-01-01 \
  --end 2025-01-01 \
  --name my-basket
```

Or edit a copy of `config/universe.example.yaml` and run:

```bash
market-analyzer pull --universe config/universe.example.yaml
```

Leave `end` unset when you want prices through the latest session. Yahoo Finance treats an end date as exclusive, so `2025-01-01` stops at 2024-12-31.

Files land in `data/<name>/`:

| File | What it is |
| --- | --- |
| `prices.csv` | Adjusted close, one column per company |
| `returns.csv` | Daily simple returns. Rows are dates (T), columns are companies (N) |
| `summary.csv` | Total return, annualized return, volatility, max drawdown |
| `covariance.csv` | Sample covariance of the return columns |
| `correlation.csv` | Pearson correlation of the return columns |
| `eigenvalues.csv` | Correlation eigenvalues, largest first, and the noise-band cut |
| `cleaned_correlation.csv` | Correlation after Marchenko-Pastur clipping |
| `cleaned_covariance.csv` | Cleaned correlation rescaled by each stock's volatility |
| `signal_loadings.csv` | Eigenvector weights of modes above the noise edge. If none clear it, this file holds the leading eigenvector only |

From Python:

```python
from market_analyzer import build_market_data

data = build_market_data(
    tickers=["AAPL", "RELIANCE.NS"],
    start="2020-01-01",
    end="2025-01-01",
)
data.returns  # shape (T, N)

from market_analyzer import CorrelationModel

model = CorrelationModel.from_returns(data.returns)
model.cleaned_covariance  # risk matrix for a later optimizer
```

Rebuild the matrices from a saved return file:

```bash
market-analyzer matrices data/my-basket/returns.csv
```

`cleaned_covariance.csv` is the matrix portfolio weights should use. The raw covariance treats estimation noise as if it were a real risk relationship. Clipping replaces correlation eigenvalues inside the Marchenko-Pastur noise band with their average and keeps the ones that stick out (typically the market, sometimes a sector). This step improves the risk model. It does not forecast returns. A profit objective needs an expected-return model on top of this covariance, and that is a separate piece of work.

## Backtest

`cleaned_covariance.csv` is only worth having if it produces better portfolios out of sample. The `backtest` command checks that. On each decision date it looks back `--window` sessions, keeps every ticker with a complete history over that window, builds a covariance matrix with each estimator, turns each into unconstrained minimum-variance weights, and holds them for the next `--holding` sessions. Nothing from the holding period touches the estimate. It then slides forward and repeats.

```bash
market-analyzer backtest data/my-basket/returns.csv --window 500 --holding 21
```

Pass `--from-prices` to hand it a price file instead. A ticker whose history starts late joins once it has a full window.

Strategies, all fully invested with shorts allowed:

| Strategy | Covariance it inverts |
| --- | --- |
| `minvar_sample` | Plain sample covariance |
| `minvar_clipped` | Marchenko-Pastur clipped correlation with the fitted noise edge, rescaled by volatility |
| `minvar_clipped_unit` | The same clipping with the textbook edge, noise variance 1 |
| `minvar_diagonal` | Variances only. This is inverse-variance weighting |
| `minvar_ledoit_wolf` | Sample covariance shrunk toward a scaled identity |
| `minvar_rie` | Optimal rotationally invariant estimator (nonlinear shrinkage): every eigenvalue gets its own corrected value, with the inverse-Wishart lift and sorting of Bun, Bouchaud and Potters. Not in the default set |
| `minvar_rie_ewma` | Exponentially weighted returns (half-life 252 sessions) cleaned by the RIE under the weighted-noise model. Not in the default set; `ewma_estimator(halflife, cleaning)` builds other half-lives |
| `minvar_rie_standardized` | Each return divided by the stock's own volatility at the time (exponentially weighted, half-life 20, past days only) before the correlation step, RIE-cleaned, scaled back by the latest volatility. Not in the default set; `standardized_estimator(...)` builds the variants, including the market-wide factor |
| `equal_weight` | None. One over N, with predicted risk read off the sample covariance |

Files land in `<folder>/backtest/`:

| File | What it is |
| --- | --- |
| `periods.csv` | One row per strategy and holding period: predicted and realised volatility, their ratio, turnover, gross leverage, largest weight, shorts, effective N |
| `daily_returns.csv` | Stitched out-of-sample daily returns, one column per strategy |
| `summary.csv` | Per strategy: realised volatility, median ratio with quartiles, mean turnover and leverage, return, Sharpe, drawdown |
| `weights_<strategy>.csv` | Weights chosen at each decision date |

The number to read first is `median_vol_ratio`, realised over predicted volatility. A risk model that tells the truth sits near 1. The sample covariance lets the optimizer load up on directions that only look riskless because of estimation noise, so its realised volatility runs well above what it predicted and its weights swing between rebalances. Cleaning pulls the ratio back toward 1 and calms the weights.

`examples/sp500_minvar_backtest.py` runs this on the S&P 500 closes cached by `examples/sp500_marchenko_pastur.py` at two window lengths and draws growth, rolling volatility, predicted against realised volatility, the ratio through time, turnover, and leverage under `output/sp500_minvar_backtest/`.

From Python:

```python
from market_analyzer import BacktestConfig, run_backtest

result = run_backtest(returns, BacktestConfig(window=500, holding=21))
result.summary()          # one row per strategy
result.periods            # one row per strategy and holding period
result.daily_returns      # date × strategy
```

`minvar_clipped_market`, which sets the noise variance to what is left after the market mode, and `minvar_rie_raw`, the Ledoit-Péché formula without the lift and the sorting, are available but not in the default set. Nothing was added to `DEFAULT_STRATEGIES`.

### Nonlinear shrinkage

`market_analyzer.rie` holds the optimal rotationally invariant estimator. Clipping gives every noisy eigenvalue the same value; this estimator gives each one its own, from a formula that needs only the sample spectrum:

```
ξ(λ) = λ / |1 − q + q z g(z)|²,   z = λ − i N^(−1/2),   g(z) = (1/N) Σ_j 1 / (z − λ_j)
```

At finite N the formula drives the smallest eigenvalues toward zero, where a minimum-variance optimiser looks hardest, so the smallest ones are lifted by the factor an inverse-Wishart truth would need (κ calibrated to the observed smallest eigenvalue), the cleaned values are sorted, and the trace is preserved. `examples/sp500_rie_study.py` runs both variants through the three tests of the earlier studies and draws the cleaning functions themselves; results under `output/sp500_rie/`.

Volatility standardisation separates the two jobs of the volatility in Σ = D C D. `ewma_volatility` gives each stock a running volatility from past days only; `standardized_covariance` divides each return by it before the correlation step (`mode="stock"`), or by one market-wide factor per day (`mode="market"`, the cross-sectional RMS of returns in units of each stock's window volatility), cleans the correlation, and scales back by the latest volatility (`scale="latest"`) or the window one (`scale="window"`). `examples/sp500_standardisation_study.py` runs the variants on three panels, the third being the Gaussian panel multiplied by the market's daily volatility factor, so volatility clustering is isolated from everything else; results under `output/sp500_standardisation/`. Finding: standardising the correlation's inputs is harmless, scaling the forecast by a short per-stock volatility estimate is harmful, and the market-wide factor keeps honesty and risk while cutting leverage and turnover. The market mode has a `trust` dial: the forecast is the market-factor value moved a fraction `trust` of the way toward each stock's own recent volatility (`trust=0` the factor alone, `trust=1` the stock's own). `examples/sp500_vol_trust_study.py` measures that fraction from history (about 0.4 on the S&P 500, zero on Gaussian data, so stock-specific calm and storm are real) and then shows that every setting above zero makes the optimiser less honest, in all three worlds: the per-stock forecast improves only a little, while the optimiser's damage grows with the square of the noise the dial lets in. Results under `output/sp500_vol_trust/`.

Exponential weighting lets recent days count more: a day's weight halves every `halflife` sessions. The noise of a weighted sample is not Marchenko-Pastur (Pafka, Potters and Kondor 2004), so `rie.py` carries the weighted-noise model: the R-transform `R(x) = Σ_t w_t / (1 − N w_t x)`, its S-transform, and the general cleaning `ξ = Im[ω S(ω)] / Im g` with `ω = z g − 1`, which reduces to the formula above for flat weights. `ewma_covariance(returns, halflife, cleaning)` builds the weighted covariance and cleans it with nothing, clipping at the effective q, or the RIE. `examples/sp500_ewma_study.py` sweeps the half-life on the Gaussian truth and on the real market; results under `output/sp500_ewma/`.

Long-only weights need a quadratic-program solver and are not implemented yet. Nothing here forecasts returns; the test is about risk.

### Where the noise edge sits

The clipping needs a noise variance to place the Marchenko-Pastur edge, and the choice changes how many eigenvalues count as signal. `matrices` and `CorrelationModel.from_returns` take `--sigma2` / `sigma2`:

| Value | Edge | On the S&P 500, T = 500, N = 474 |
| --- | --- | --- |
| `fitted` (default) | Smallest leading set such that the rest is consistent with pure noise | 106 modes kept |
| `1.0` | Textbook edge for a correlation matrix | 13 modes kept |
| `market` | Noise variance left after the largest eigenvalue, as in Laloux et al. (1999) | 17 modes kept |

The fitted rule cascades on real data because the empirical bulk is wider than the theoretical one, so it treats a large share of the spectrum as signal. The backtest shows which rule produces better portfolios out of sample.

## The web page

`app/streamlit_app.py` is a Streamlit page over the same code: pick stocks from 3,368 US common stocks, see how much an equal split would swing against the minimum-variance mix, how the mix spreads the money and why, and what would have happened over the last three years with monthly re-optimisation. It runs on the market pack in `data/pack/`, built by `examples/build_market_pack.py`. Run it with `.venv/bin/streamlit run app/streamlit_app.py`; the design, the data rules and the deployment steps are in [docs/webpage.md](docs/webpage.md).

## Layout

```
config/universe.example.yaml      tickers you type
config/html_table.example.yaml    optional symbols from any HTML table
docs/getting-stock-data.md        how to get data for any company
docs/covariance_cleaning_review.md  survey of cleaning methods (RIE, exponential weighting and volatility standardisation implemented; factor, clustering, robust still to try)
docs/why_the_gap_persists.md        why every fix stalls: documented limits of each method, and the shuffled-days split of the remaining gap
docs/fixing_the_directions.md       methods that change eigenvectors, where the RIE's direction error sits, and untested combinations
docs/algorithm_catalogue.md         every published covariance algorithm for minimum variance, in plain words, with our results where tested
docs/long_only_minimum_variance.md  the long-only optimiser: why weights sum to one, the no-shorting rule as a data-driven shrinkage, how the active-set solver works, and the real-market and lab results
docs/gross_exposure_cap.md          the gross-exposure dial between long-only and unconstrained: the budget-for-shorting idea, the L² bound, the band optimality conditions, the active-set solver, and the sweep results
docs/published_comparison.md        our long-only, gross-cap and cleaning results against Jagannathan–Ma, DeMiguel et al., Fan–Zhang–Yu, Ledoit–Wolf and Bun–Bouchaud–Potters, numbers read from their tables; which libraries implement the constraints
docs/webpage.md                     the web page: what it shows, the architecture, the market pack and its rules with counts, the engine, running, deployment, what the presets showed
app/streamlit_app.py              the web page (Streamlit): pick stocks, see the risk, the mix, and what would have happened
data/pack/                        the market pack the page runs on: daily returns, per-stock facts, manifest, build report
src/market_analyzer/              download, returns, summary, risk matrices, nonlinear shrinkage (rie.py), optimizer (unconstrained, long-only, gross cap), backtest, market universe and pack (market_universe.py, market_pack.py), basket engine for the page (basket.py)
examples/pull_custom_basket.py    short script for a three-name basket
examples/sp500_marchenko_pastur.py  S&P 500 spectrum at three window lengths
examples/sp500_minvar_backtest.py   rolling minimum-variance test, raw against cleaned
examples/sp500_eigenvector_stability.py  do the rolling eigenvectors move? real vs Gaussian from a fixed covariance
examples/sp500_simulated_backtest.py     the backtest on Gaussian returns from a fixed covariance, with an oracle
examples/sp500_honesty_vs_window.py      Monte Carlo sweep of the window length: noise vs bias per estimator
examples/sp500_window_sweep.py           rolling backtest at six window lengths, real vs Gaussian, common period
examples/sp500_rie_study.py              the optimal RIE through all three tests, plus the cleaning functions on one window
examples/sp500_ewma_study.py             exponential weighting with the RIE: half-life sweeps on the Gaussian truth and the real market
examples/sp500_standardisation_study.py  volatility standardisation through three worlds: constant volatility, the market's volatility clock, the real market
examples/sp500_vol_trust_study.py        the stock dial on the market factor: how much of a stock's own volatility to trust, measured from history and tested in three worlds
examples/sp500_shuffle_test.py           real days in a random order: splits the real-market gap into fat tails and time order
examples/sp500_eigenvector_error.py      hands the RIE the true top or quietest directions and splits its extra risk by true direction
examples/sp500_james_stein_study.py      James–Stein shrinkage of the market eigenvector on top of the RIE (rie_js), exact-truth worlds and the real backtest
examples/sp500_persistence_study.py      what carries over between periods: volatilities, correlations, market, sector and middle subspaces, real vs no-change baselines
examples/sp500_rie_noise_by_direction.py how much variance the RIE removes or adds along each sample direction, graded against the truth in a Gaussian lab
examples/kendall_tiny_example.py         Kendall's tau by hand on five real days, one company-news day's effect on Pearson vs Kendall, the Gaussian dictionary
examples/sp500_kendall_study.py          Kendall's tau as the source of directions: aim vs truth, plain Kendall and Kendall-room hybrids vs the RIE, lab worlds and real market
examples/sp500_memory_length.py          how long a memory for the skeleton: room forecasts vs history length, real vs shuffled, and whether sizes along old directions carry over
examples/sp500_two_speed_study.py        design A, directions from the past and sizes from the present, vs the 1000- and 250-day RIEs and two volatility-swap controls, real market and Gaussian lab
examples/sp500_long_only_study.py        long-only minimum variance against the unconstrained optimiser: sample covariance and RIE at 1000 and 250 days on the same dates, equal weight, real market and Gaussian lab with an oracle
examples/sp500_gross_cap_study.py        the gross-exposure dial from long-only to unconstrained: caps 1, 1.25, 1.5, 2, 3, 5 and none, sample covariance and RIE at 1000 and 250 days, first half against second half, real market and Gaussian lab with an oracle
examples/sp500_fat_tails.py              four pictures of fat tails: histogram, clustering, the standardisation fix, concentration
examples/build_market_pack.py            builds the market pack: NASDAQ Trader symbol files and screener in, listing rules, batched Yahoo download, history checks, pack and report out
```

`data/` is created when you pull and is gitignored.

## Tests

```bash
pytest
```

The tests use synthetic prices. A real pull needs a network connection to Yahoo Finance.
