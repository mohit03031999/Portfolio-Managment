# Long-only minimum variance

*Written 2026-11-09. Code: `long_only_minimum_variance_weights` in `src/market_analyzer/optimizer.py`; strategy names `longonly_<estimator>` in the backtest; study script `examples/sp500_long_only_study.py`; results in `output/sp500_long_only/`.*

## 0. In one paragraph

Every method in this project so far changed the covariance matrix and left the optimiser alone. The optimiser takes the matrix at face value and puts money wherever the matrix says risk is lowest, hedging with short positions, so every error in the matrix becomes a position and the book runs at 5 to 8 times gross leverage. Long-only keeps the same goal (the lowest-risk fully invested mix) and adds one rule: no weight may be negative. There is no shorting, so there is no borrowing and no gearing; leverage is exactly 1. The optimiser loses the tool it uses to exploit noise. This document explains what the rule is, why it works as a kind of cleaning, how we solve it without a closed form, and what it did on the S&P 500.

## 1. The rule

Unconstrained minimum variance (what every earlier study used):

```
minimise   wᵀ Σ w
subject to Σᵢ wᵢ = 1
```

The answer is one formula, `w = Σ⁻¹1 / (1ᵀΣ⁻¹1)`. Weights may be negative (short positions), and typically about half of them are.

Long-only minimum variance:

```
minimise   wᵀ Σ w
subject to Σᵢ wᵢ = 1
           wᵢ ≥ 0   for every stock
```

Two points that are easy to get wrong:

- **The weights still sum to 1, not 0.** "Sum to 1" means all the capital is invested. "Sum to 0" would be a dollar-neutral long–short fund (a dollar sold short for every dollar bought), the opposite of long-only. Without the budget constraint the lowest-risk portfolio is trivially "hold cash".
- **Leverage is exactly 1.** Gross leverage is Σ|wᵢ|. With no negatives it equals Σwᵢ = 1. An unconstrained book with gross 6.9 holds about $395 long and $295 short per $100 of capital; the long-only book holds $100 of stocks.

## 2. The two-stock example

Stock A: 20% volatility. Stock B: 30%. Correlation 0.8.

Unconstrained: `w = [+124%, −24%]`, predicted risk 19.5%. B behaves like 1.2 copies of A plus its own noise, so a short in B cancels part of A's movement.

Long-only: the short is forbidden, B goes to zero, `w = [100%, 0%]`, risk 20.0%.

Now suppose the 0.8 was estimation noise and the true correlation is 0.6:

| | predicted | realised | honesty |
|---|---|---|---|
| unconstrained | 19.5% | 21.2% | 1.09 |
| long-only | 20.0% | 20.0% | 1.00 |

The constraint gave up 0.5 points on paper and in return made the portfolio immune to the wrong number. With 456 stocks the optimiser finds hundreds of such "hedges", and each one is built on a noisy correlation.

## 3. Why no-shorting is a form of cleaning

Jagannathan and Ma (2003, *Journal of Finance*) proved that the long-only solution on a covariance Σ is **the same portfolio** as the unconstrained solution on a modified matrix

```
Σ̃ = Σ − δ 1ᵀ − 1 δᵀ        i.e.   σ̃ᵢⱼ = σᵢⱼ − δᵢ − δⱼ
```

where δᵢ ≥ 0 is the Lagrange multiplier on the constraint wᵢ ≥ 0. It is zero for every stock the optimiser holds and positive exactly for the stocks it wanted to short. So the constraint lowers the covariances of the "short" stocks with everything else (and their variances, by 2δᵢ) until the optimiser no longer wants to short them. The stocks the unconstrained optimiser shorts are the ones whose sample covariances with the rest came out highest; those are also the ones most likely to be inflated by noise. The constraint shrinks exactly those entries, by an amount the data picks, one stock at a time.

This is why long-only works even on the raw sample covariance, and why Jagannathan and Ma found that once the constraint is on, every cleaning method performs about the same as the sample covariance: the constraint is already doing most of the cleaning.

## 4. How the solver finds the answer

With the sign constraints there is no formula. The code uses the classical active-set method of Lawson and Hanson (1974), through a small trick that turns our problem into theirs.

**The trick.** Minimise `yᵀΣy − 2·Σᵢyᵢ` over `y ≥ 0`, with no budget constraint. Its optimality conditions are `Σy − 1 = μ`, `μ ≥ 0`, `μᵢyᵢ = 0`. Divide everything by `s = Σᵢyᵢ`: with `w = y/s` they become `Σw − (1/s)·1 = μ/s`, which are exactly the optimality conditions of the budgeted long-only problem with multiplier `λ = 1/s` on the budget. So solve the unbudgeted problem and normalise. The unbudgeted problem is a non-negative least-squares problem, the kind Lawson and Hanson solve.

**The algorithm, in plain words.**

1. Start with every weight at zero. Compute `g = 1 − Σy`. A positive `gᵢ` means buying a little of stock i lowers the objective.
2. Admit the stock with the largest `gᵢ`.
3. Solve the *unconstrained* problem on the admitted stocks only (a small linear system, the same closed form as before).
4. If every admitted weight came out positive, accept it. Otherwise step from the previous point toward the new solution only as far as the first weight that reaches zero, drop that stock, and solve again on the rest.
5. Recompute `g`. If no excluded stock has `gᵢ > 0`, stop: no purchase would help. Otherwise go to 2.

The objective falls at every step, so the loop ends after finitely many steps. On the 456-stock RIE it admits about 60 to 90 stocks and finishes in 0.02 seconds. At the end the code checks the optimality condition directly: for every held stock `(Σw)ᵢ = wᵀΣw` and for every excluded stock `(Σw)ᵢ ≥ wᵀΣw`. The worst violation, relative to `wᵀΣw`, is stored as `attrs["kkt_gap"]`; it is at the level of rounding (1e-15) on real matrices. `attrs["n_positions"]` counts the stocks held.

A rank-deficient matrix (the sample covariance with fewer days than stocks) gets a ridge of 1e-10 times its largest eigenvalue so every sub-problem is solvable; `attrs["singular"]` is set, as the unconstrained function does.

## 5. How to run it

Every estimator has a long-only strategy in the backtest: `longonly_sample`, `longonly_rie`, `longonly_clipped`, and so on, beside `minvar_sample`, `minvar_rie`, `minvar_clipped`. Nothing in the defaults changed; the new names are opt-in.

```bash
python -m market_analyzer backtest --strategies minvar_rie longonly_rie equal_weight ...
```

The study script runs both windows on the same decision dates and reports them side by side:

```bash
python examples/sp500_long_only_study.py                       # real market, T = 1000 and 250
python examples/sp500_long_only_study.py --panel gaussian_raw  # Gaussian lab with an oracle
python examples/sp500_long_only_study.py --report              # summary.txt and long_only.png
```

The `summary()` of a backtest result has a new column `mean_n_positions`; `weight_statistics` reports `n_positions`.

## 6. Results

Setup: 456 S&P 500 stocks with a complete history, 93 holds of 21 days from 2018-12-24 to 2026-10-01, the same decision dates for both windows (the 250-day run is fed the panel from day 750 so its first decision falls on the 1000-day run's first date). Realised volatility is the standard deviation of the stitched daily out-of-sample returns, annualised. Honesty is the median over holds of realised / predicted volatility; a model that tells the truth sits at 1. Positions are stocks with a positive weight. Figure: `output/sp500_long_only/long_only.png`.

### 6.1 Real market

T = 1000 days:

| strategy | realised vol | honesty (q25–q75) | turnover / month | leverage | positions | max weight |
|---|---|---|---|---|---|---|
| sample, unconstrained | 14.81% | 1.97 (1.73–2.35) | 2.69 | 11.0 | 456 | 14.0% |
| sample, long-only | 14.52% | **0.86** (0.72–1.09) | 0.14 | 1.0 | 36 | 13.4% |
| RIE, unconstrained | **13.30%** | 1.31 (1.12–1.60) | 1.34 | 6.9 | 456 | 8.6% |
| RIE, long-only | 14.62% | **0.85** (0.69–1.07) | 0.14 | 1.0 | 37 | 13.8% |
| equal weight | 19.85% | 0.68 (0.57–0.90) | 0 | 1.0 | 456 | 0.2% |

T = 250 days:

| strategy | realised vol | honesty (q25–q75) | turnover / month | leverage | positions | max weight |
|---|---|---|---|---|---|---|
| sample, unconstrained | 16.24% | 2.59 (2.18–3.04) | 7.61 | 9.3 | 456 | 8.6% |
| sample, long-only | 13.66% | **1.02** (0.79–1.36) | 0.43 | 1.0 | 37 | 13.2% |
| RIE, unconstrained | **13.26%** | 1.63 (1.35–2.10) | 1.87 | 4.8 | 456 | 7.4% |
| RIE, long-only | 13.73% | **1.00** (0.78–1.35) | 0.40 | 1.0 | 47 | 11.5% |
| equal weight | 19.85% | 0.78 (0.59–1.00) | 0 | 1.0 | 456 | 0.2% |

Share of holds in which realised volatility exceeded the prediction by more than 50%: long-only 9% (T = 1000) and 17% (T = 250); unconstrained RIE 31% and 61%.

The two long-only portfolios hold almost the same stocks: the weight overlap between the sample and RIE versions averages 0.92 (T = 1000) and 0.79 (T = 250). The RIE long-only book at T = 1000 held SW, KDP, KR, VZ, BMY, GIS, WMT, CBOE, NEM and CHRW in more than 80% of the holds: staples, telecom, exchanges, gold, the classic low-volatility basket.

Returns are not what minimum variance optimises, but for the record over the whole track at T = 1000: growth of 1 to 3.05× (RIE, unconstrained), 2.49× (RIE, long-only), 2.57× (sample, long-only), 3.71× (equal weight); Sharpe 1.15, 0.88, 0.91, 0.95. At T = 250 the long-only Sharpe (1.09) is above the unconstrained RIE's (0.98).

### 6.2 Gaussian lab

Gaussian days drawn from the raw full-sample covariance (seed 1), so nothing drifts and the true risk of every portfolio is known. "True vol" is the risk under the true matrix, averaged over holds; the oracle rows use the true matrix as the estimate.

| strategy | T = 1000: true vol | honesty | T = 250: true vol | honesty |
|---|---|---|---|---|
| sample, unconstrained | 11.27% | 1.82 | 14.45% | 2.10 |
| sample, long-only | 12.07% | **1.01** | 12.63% | 1.12 |
| RIE, unconstrained | **9.78%** | 1.24 | **11.15%** | 1.57 |
| RIE, long-only | 12.09% | **1.00** | 12.61% | 1.09 |
| equal weight | 17.86% | 0.98 | 17.86% | 0.98 |
| truth, unconstrained | 8.30% | 0.98 | 8.30% | 0.98 |
| truth, long-only | 11.84% | 0.99 | 11.84% | 0.99 |

## 7. What we learned

1. **The optimism was the optimiser's doing. Confirmed.** Forbid shorting and the honesty ratio drops from 1.31–2.59 to 0.85–1.02 on real data and to 1.00–1.01 in the lab, where the oracle itself sits at 0.99. The estimation noise in the covariance is still there (the sample covariance is as noisy as ever); what changed is that the optimiser can no longer turn that noise into long–short bets. At T = 1000 the long-only ratios sit below 1 (0.85) in the same direction as equal weight (0.68): a 1000-day window carried the 2020 crash for years and over-predicted risk. That is forecast error of the window, not optimiser behaviour.

2. **Realised risk did not fall below the unconstrained RIE.** 14.62% against 13.30% at T = 1000 (+10%), 13.73% against 13.26% at T = 250 (+3.5%). The lab says why: when the matrix is right, shorting is worth a lot (true-covariance portfolios: 8.30% unconstrained, 11.84% long-only, a 43% difference). On real data the noise and the drift eat most of that advantage, but not all of it: hedging with shorts still buys 0.5–1.3 points of risk per year, even on noisy correlations. Against the raw sample covariance the constraint does lower risk (16.24% → 13.66% at T = 250), because there the shorts were mostly noise.

3. **Cleaning stops mattering once shorting is forbidden.** Sample and RIE long-only portfolios differ by 0.1 points of risk in every world and hold mostly the same stocks. Jagannathan and Ma's finding holds on our data: the constraint is already doing the cleaning. The RIE's work is only visible when the optimiser is allowed to use the whole matrix.

4. **The portfolio becomes something one could hold.** Leverage exactly 1, 36–47 stocks, turnover 0.14 per month at T = 1000 and 0.40 at T = 250 against 1.3–1.9 unconstrained, a fifth to a tenth of the trading. The price is concentration: the largest position is 11–14% of the book, so a position cap is the obvious next constraint.

5. **A short window helps long-only on real data and hurts it in the lab.** 13.7% against 14.6% real, 12.6% against 12.1% lab. The real-market gain is weather, fresh volatilities, the same lesson as in the memory-length study; and under long-only it costs nothing in honesty (1.00 at T = 250), unlike the unconstrained case where T = 250 pushed the ratio to 1.63.

6. **Where this leaves the design.** The two extremes are now measured: full freedom (leverage 7, risk 13.3%, honesty 1.3, turnover 1.3) and no shorting (leverage 1, risk 13.7–14.6%, honesty 1.0, turnover 0.14–0.40). Real information lives in the shorts, and so does the noise. The middle ground, a gross-exposure cap somewhere between 1 and 7 as in Fan, Zhang and Yu (2012), or a position cap on the long-only book, is the natural next experiment. Neither is built.
