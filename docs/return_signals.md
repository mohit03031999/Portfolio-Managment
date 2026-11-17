# Return signals: the signal lab

The risk side of the platform cleans a covariance matrix and finds the
quietest mix. It never asks which stocks will go up. The return side needs a
second ingredient, a forecast of each stock's expected return, and the
Markowitz weights become the cleaned inverse covariance applied to that
forecast. Minimum variance is the special case where every forecast is the
same. The forecast is the hard half: a stock's average return is measured by
where it ends up, so its error shrinks with calendar years, not with the
number of daily prices. Nine years of data pin a 30%-volatility stock's
annual mean to about ±10 points, on a true value of perhaps 8. Chopra and
Ziemba (1993) put errors in means at about ten times the cost of errors in
variances, and DeMiguel, Garlappi and Uppal (2009) found that mean-variance
on sample means needs centuries of data to beat an equal split.

So before any forecast goes near the optimiser, the lab asks each candidate
one question: **on each month-end, if we rank every stock by the signal, does
the ranking line up with the stocks' returns over the next month?** A single
portfolio track cannot answer this (a nine-year Sharpe ratio carries an error
bar of about ±0.4), but a cross-section of thousands of stocks, repeated every
month, can detect an edge as small as a rank correlation of 0.03.

## 1. What the code is

| Piece | File | Role |
|---|---|---|
| Signals | `src/market_analyzer/signals.py` | One panel (dates × tickers) per signal, computed from past days only; the registry `SIGNALS` carries each signal's orientation (`higher_is_better`), and `oriented` flips the sign so that higher always means "expect a higher return"; `forward_return` is the outcome and the only function that looks ahead |
| Scoring | `src/market_analyzer/signal_lab.py` | Per-date cross-sectional statistics (`rank_correlation`, `cross_sectional_slope`, `decile_returns`, `top_decile_turnover`), `score_signal` → `SignalScore`, `summary_table`, `decision_dates`, `sharpe_with_error` |
| Study | `examples/signal_lab.py` | Runs every registry signal on the market pack, adds the controls and the liquid-universe split, writes `output/signal_lab/` |
| Tests | `tests/test_signals.py` | Hand-checked forward returns and momentum, point-in-time (change the future, the past must not move), beta of a stock equal to the market is one, a planted edge is recovered by the IC and both slopes, shuffled outcomes score zero, deciles are monotone on a planted edge, per-date mask |

The library part is additive: nothing in the optimiser, the backtest or the
page changed.

### Signals

Windows are counted in sessions (252 a year, 21 a month); a stock gets a
value when at least 90% of the window is present.

| Key | Definition | Orientation | What the record says |
|---|---|---|---|
| `momentum` | return over the past 252 sessions ending 21 sessions ago ("12-1") | higher is better | the most robust signal known; crashes in sharp rebounds |
| `reversal` | return over the past 21 sessions | lower is better | real, usually eaten by costs |
| `low_vol` | annualised standard deviation of daily returns over the past year | lower is better | similar return at lower risk |
| `low_beta` | slope on the equal-weight market over the past year, from rolling means with the market masked to the stock's own days | lower is better | cousin of low volatility |
| `mean_1y` | mean daily return over the past year × 21 | higher is better | the textbook forecast, the negative control |
| `mean_3y` | mean daily return over the past three years × 21 | higher is better | long horizons reverse (De Bondt and Thaler) |

### Outcome and measurements

The outcome is each stock's return over the 21 sessions after the decision
date, missing unless all 21 are present. Decision dates are the last session
of each month. Per signal and date, across the stocks with both a signal and
an outcome:

- **IC**, the Spearman rank correlation between signal and outcome.
- **Deciles**: equal-weight outcome of each tenth of the ranking; the
  **spread** is decile 10 minus decile 1, a long-short portfolio whose
  monthly series gives an annualised Sharpe ratio with Lo's (2002) error
  bar √((1 + SR²/2)/n).
- **Calibration slope**: OLS slope of the demeaned outcome on the z-scored
  signal, in return per standard deviation of signal. For the two
  historical-mean signals, which are already in 21-session return units,
  the raw slope is also kept: the **trust fraction**, the share of the
  forecast that survived (1 = trust fully, 0 = nothing).
- **Decay**: IC against the return of the k-th month ahead, k = 1, 2, 3, 6,
  12, which says how often the ranking would need refreshing.
- **Turnover** of the top decile from one month to the next, and
  **coverage**, the number of stocks scored.

Dates are then averaged (Fama–MacBeth): the mean across months and a
t-statistic from the spread of the monthly values, which absorbs the
correlation of stocks on the same day.

Two controls run alongside: the historical mean, which the record says
carries nothing, and momentum scored against outcomes shuffled among the
stocks of each date, which must score zero. Every signal is also scored
inside the **liquid 1000**, the thousand stocks with the highest trailing
six-month median dollar volume *as of each date* (a point-in-time split from
the raw cache; a split by today's market cap would pick the period's winners,
and did: it made high volatility look like a strong positive signal among
"large caps").

## 2. Running it

```bash
python examples/signal_lab.py            # ~25 s on the pack
python examples/signal_lab.py --replot   # redraw from the saved tables
```

Outputs in `output/signal_lab/`: `summary.txt` / `summary.csv`,
`ic_monthly.csv`, `spread_monthly.csv`, `decay.csv`, `ic_by_year.csv`,
`pack_equal_weight_by_year.csv`, `deciles/<key>.csv`, and the figures
`spread_growth.png`, `deciles.png`, `ic_by_year.png`, `decay.png`. The
liquid-universe rows need `data/raw/adj_close.parquet` and `volume.parquet`
from the pack build; without them the script skips those rows and says so.

## 3. Results on the market pack (2026-11-12)

3,368 stocks, 94 month-ends from 2018-12-31 to 2026-09-30, 21-session
outcomes, about 2,800 stocks scored per month. One return of −182% (a bad
adjusted price) is treated as missing.

| Signal | IC | t | spread / yr | Sharpe | bp per sd | trust | turnover |
|---|---|---|---|---|---|---|---|
| Momentum (12-1) | +0.008 | +0.5 | −13.0% | −0.49 ± 0.36 | +3 | · | 0.25 |
| Momentum · liquid 1000 | +0.003 | +0.1 | +7.1% | +0.22 ± 0.36 | +33 | · | 0.26 |
| Reversal (1 month) | −0.005 | −0.4 | +0.6% | +0.03 ± 0.36 | −3 | · | 0.82 |
| Low volatility | +0.022 | +1.1 | −39.0% | −1.01 ± 0.36 | −87 | · | 0.08 |
| Low volatility · liquid 1000 | +0.006 | +0.2 | −17.0% | −0.40 ± 0.36 | −48 | · | 0.09 |
| Low beta | +0.009 | +0.4 | −14.1% | −0.38 ± 0.36 | −40 | · | 0.09 |
| Historical mean (1 year) | +0.005 | +0.3 | +1.3% | +0.05 ± 0.36 | +5 | +0.02 | 0.22 |
| Historical mean (3 years) | −0.019 | −1.2 | −21.9% | −0.94 ± 0.41 | −41 | −0.18 | 0.14 |
| Momentum vs shuffled outcomes | −0.002 | −1.3 | −0.9% | −0.17 ± 0.36 | +3 | · | 0.25 |

The full table, the half-by-half ICs and the decay rows are in
`summary.txt`.

What the numbers say, signal by signal:

- **The machinery behaves.** The shuffled control scores −0.002 with a flat
  decay, the planted-edge tests recover a known slope, and the point-in-time
  test holds for every signal.
- **Nothing is significant.** Every |t| is below 1.7. With a monthly IC
  spread of about 0.15, 94 months give an IC error bar of about ±0.015, so
  the lab could just detect the record's 0.03 momentum edge if it were
  there. Momentum shows +0.008.
- **The historical mean does what the record says.** One-year mean: trust
  +0.02, essentially nothing. Three-year mean: trust −0.18 and IC −0.019,
  the textbook forecast points the wrong way, which is the long-horizon
  reversal of De Bondt and Thaler.
- **Low volatility is split down the middle.** Its IC is the most positive
  in the table (+0.022, low-vol stocks rank higher next month) while its
  decile spread is the most negative (−39% a year): the highest-volatility
  decile's *mean* return was 48% a year. Rank-based and mean-based
  measures disagree because a few extreme winners dominate the mean, and
  the next section says where those winners come from.
- **Momentum's bottom decile returned 40% a year**, more than every decile
  but the top. Last year's biggest losers, among the stocks that are still
  listed today, recovered hard.

## 4. The finding that matters: this pack cannot score returns

The pack was built for the risk side from stocks listed today. Companies
delisted since 2018, by bankruptcy or by takeover, are not in it. For
volatilities and correlations this hardly matters. For returns it is
decisive, and the pack shows it on its own: the equal-weight return of every
stock in the pack, by year, against three index funds (adjusted closes from
Yahoo Finance, fetched 2026-11-12):

| Year | Pack, equal weight | IWM (Russell 2000) | RSP (S&P 500 equal weight) | SPY (S&P 500) |
|---|---|---|---|---|
| 2019 | +32% | +25% | +29% | +31% |
| 2020 | +35% | +20% | +13% | +18% |
| 2021 | +27% | +15% | +29% | +29% |
| 2022 | −15% | −21% | −12% | −18% |
| 2023 | +26% | +17% | +14% | +26% |
| 2024 | +21% | +11% | +13% | +25% |
| 2025 | +25% | +13% | +11% | +18% |
| 2019–2025, per year | **+20%** | +10.5% | +13% | +17% |

An equal-weight basket of 3,000 US stocks, most of them small, compounding
seven to ten points a year faster than the small-cap index is not a
discovery; it is the dead stocks' absence. The bias is not uniform across
the ranking, which is why it wrecks the signals rather than merely lifting
them: the stocks that die are concentrated in the high-volatility decile
and in the past-losers decile, exactly the buckets that the low-volatility
and momentum signals sell. Removing the dead makes those buckets look like
the best places to be. The liquid-1000 rows halve the effect (the
low-volatility spread goes from −39% to −17%) but cannot remove it, because
takeovers remove liquid names too, and takeover targets are not a random
sample either.

So the honest conclusion is about data, not models: **return signals
cannot be scored on this pack**. The risk results stand (they are measured
within the pack and survivorship barely touches second moments); the return
pathway needs a universe with the dead included before step two.

## 5. Where to get a universe with the dead included

- **Point-in-time S&P 500 membership.** The index's additions and removals
  since 2018 are public and well documented. Members at each month-end
  give a survivorship-free large-cap universe of 500, if the removed
  names' prices can still be fetched. Free price feeds drop many delisted
  tickers, so the first job is to count how many of the removed names are
  retrievable; that count is the measurable gap. Smaller cross-section
  than the pack (500 against 2,800), so the IC error bar roughly doubles,
  but the numbers would be honest.
- **A survivorship-free commercial feed** (Sharadar on Nasdaq Data Link,
  EODHD, Norgate), which carry delisted stocks with their final prices.
  Paid, a few tens of dollars a month, and the only route to an honest
  all-cap test.
- **CRSP**, the academic standard, through a university subscription.

Until one of these is in place, every return number from the pack carries
the survivorship caveat printed in `summary.txt`.

## 6. Reading the lab's output in general

- The **IC** is the headline; it is rank-based and immune to a few extreme
  outcomes. A long-run monthly IC of 0.03–0.05 is what published momentum
  earns; 0.10 would be exceptional.
- The **decile spread** is what a long-short portfolio on the ranking
  would have made before costs, equal weight. Its Sharpe error bar is the
  reminder of how little a time series can say.
- **bp per sd** is the size of the edge in the optimiser's units: the
  expected relative return next month for a stock one standard deviation
  above average on the signal. It is what a forecast vector would carry.
- The **trust fraction** is the shrinkage the data itself asks for: a
  forecast in return units whose slope on realised returns is 0.1 should
  be used at a tenth of its face value.
- **Decay** against **turnover** is the cost question: a signal that keeps
  its IC for six months but replaces a quarter of its top decile every
  month can be traded quarterly instead.
