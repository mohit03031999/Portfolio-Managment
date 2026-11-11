# The gross-exposure cap

*Written 2026-11-09. Code: `gross_capped_minimum_variance_weights` in `src/market_analyzer/optimizer.py`; strategy names `gross<cap>_<estimator>` in the backtest (`gross2_rie` is a 150/50 book); study script `examples/sp500_gross_cap_study.py`; results in `output/sp500_gross_cap/`. Companion to `docs/long_only_minimum_variance.md`.*

## 0. In one paragraph

The long-only test measured the two ends of a dial. With shorting allowed, the optimiser hedged well (13.3% realised risk) but lied about it (forecasts 1.3 to 2.6 times too optimistic) and ran at 5 to 11 times gross leverage. With shorting forbidden, it told the truth (honesty 1.0) but lost the hedges (13.7 to 14.6%). The shorts carried real information and the noise at the same time. A cap on gross exposure, `Σ|wᵢ| ≤ L` with the weights still summing to one, gives the optimiser a budget for shorting and lets the size of the payoff decide which hedges survive. This document explains the rule, why a budget separates information from noise, the optimality conditions the solver works on, and what the sweep over L found.

## 1. The rule

```
minimise   wᵀ Σ w
subject to Σᵢ wᵢ = 1          (fully invested)
           Σᵢ |wᵢ| ≤ L        (total long plus short exposure at most L)
```

Since the weights sum to one, long − short = 1 and long + short ≤ L, so the total short is at most (L − 1)/2:

| L | longs / shorts per $100 | name |
|---|---|---|
| 1 | 100 / 0 | long-only |
| 1.25 | 112.5 / 12.5 | |
| 1.5 | 125 / 25 | a "125/25" fund |
| 2 | 150 / 50 | a "150/50" fund, what a margin account allows |
| 3 | 200 / 100 | |
| 5 | 300 / 200 | |
| none | | unconstrained; our RIE book ran at about 400 / 300 |

## 2. Why a budget separates information from noise

Give the optimiser a limited amount of shorting and it has to spend it where the hedge pays the most. The first dollars go to the large, obvious hedges: offsetting market exposure, pairing the strongest sector moves. Those rest on correlations we measure well (the sizes along the top directions are near-exact). The last dollars, the stretch from 25% short to 300% short, go to hundreds of small bets, each one resting on a small difference between two noisy numbers.

Two facts make the trade favourable:

1. **The benefit of shorting flattens.** The first unit of hedge removes a lot of variance, the next less, and so on. Most of the hedging value is bought with the first part of the budget.
2. **The damage from noise grows with the stretch.** Fan, Zhang and Yu (2012, *JASA*) proved that the gap between the risk the model predicts and the true risk of the chosen portfolio is at most (the worst single-entry error of the covariance) × L². At L = 7 the factor is 49, at L = 2 it is 4, at L = 1 it is 1. The optimiser's power to find errors scales with how far it may stretch.

So the cap does not try to tell good numbers from bad ones inside the matrix (every matrix method we tried failed at that). It limits how much the optimiser can *act* on any number, and lets the payoff decide which bets survive.

## 3. The two-stock example with the dial

A: 20% risk, B: 30%, estimated correlation 0.8. If the true correlation is 0.6:

| setting | weights | predicted | realised | honesty |
|---|---|---|---|---|
| unconstrained | +124 / −24 | 19.5% | 21.2% | 1.09 |
| cap 1.2 (short ≤ 10%) | +110 / −10 | 19.7% | 20.3% | 1.03 |
| long-only | +100 / 0 | 20.0% | 20.0% | 1.00 |

If the correlation is right, the cap keeps 60% of the hedging benefit (0.3 of the 0.5 points). If it is wrong, the cap takes only 35% of the damage (0.6 of the 1.7 points). The first bit of hedge is the cheapest; the rest is where the noise lives.

## 4. The optimality conditions, and why the solution is sparse

Write ρᵢ = (Σw)ᵢ for the risk contribution of stock i (how much the portfolio's variance changes per unit of stock i). With η the multiplier on the budget and μ ≥ 0 the shadow price of exposure, the solution satisfies

```
ρᵢ = η − μ        for every stock held long
ρᵢ = η + μ        for every stock held short
η − μ ≤ ρᵢ ≤ η + μ   for every stock left out
```

Every risk contribution sits in a band of half-width μ around η. Stocks below the band would be worth buying; the optimiser buys them until they reach the floor. Stocks above the band would be worth shorting; it shorts them until they reach the ceiling. Stocks inside the band are not worth a position at the current price of exposure, so they sit at exactly zero. When the cap does not bind, μ = 0, the band has no width, and every stock is held: the unconstrained solution. As L shrinks, μ rises, the band widens and more stocks fall inside it. That is why a capped portfolio holds a few dozen longs and a handful of shorts rather than 456 signed positions.

There is also the shrinkage reading, the generalisation of Jagannathan and Ma: the capped solution is the unconstrained solution of a modified matrix in which an amount μ is added to the covariances among the longs and subtracted among the shorts, by exactly enough to make stretching the book unattractive. The constraint is a cleaning in disguise, one that acts on bets rather than on entries, with the data choosing μ.

## 5. How the solver works

With the sign constraints there is no formula. `gross_capped_minimum_variance_weights` is a primal active-set method built directly on the band conditions:

1. **Start feasible.** The long-only portfolio (from the Lawson–Hanson solver) has gross exposure 1 ≤ L. If the unconstrained portfolio already satisfies the cap, return it: nothing to do.
2. **Solve on the current working set.** For the stocks currently held, with their current signs, solve the equality-constrained problem: the budget `Σw = 1`, plus `Σ sign(wᵢ) wᵢ = L` once the cap binds. This is a small linear system (the KKT system) whose solution also gives η and μ.
3. **Move toward that solution, but stop at the first event.** Either a held weight would reach zero (drop that stock from the working set) or the exposure would reach L (mark the cap as binding). Then solve again.
4. **At a sub-problem's minimiser, check the conditions.** If the cap binds with μ < 0, it should not bind: release it. Otherwise look at every excluded stock: the one furthest below the band enters as a long, the one furthest above enters as a short. If none is outside the band, stop.

The objective falls at every step, so it ends after finitely many; on the 456-stock RIE it takes 0.03 s at L = 1.25 and 0.5 s at L = 5, where it admits about 440 stocks. `attrs["kkt_gap"]` records the worst violation of the band (rounding level, 1e-16, on real matrices); `attrs["cap_active"]` says whether the cap bound; `attrs["cap_multiplier"]` is μ in the covariance's units. `cap = 1` delegates to the long-only solver and `cap = inf` to the closed form, so one function covers the whole dial.

## 6. How to run it

```bash
python -m market_analyzer backtest --strategies minvar_rie gross2_rie gross1.5_rie longonly_rie equal_weight ...
python examples/sp500_gross_cap_study.py                       # real market, caps 1 … none, T = 1000 and 250
python examples/sp500_gross_cap_study.py --panel gaussian_raw  # Gaussian lab with an oracle at every cap
python examples/sp500_gross_cap_study.py --report              # summary.txt and gross_cap.png
```

## 7. Results

Setup as in the long-only study: 456 stocks, 93 holds of 21 days from 2018-12-24 to 2026-10-01, the same decision dates for both windows. Realised volatility is the standard deviation of the stitched daily out-of-sample returns, annualised; honesty is the median over holds of realised / predicted volatility. The halves split at 2022-11-08: the first half holds the 2020 crash and the 2022 bear market, the second half is calmer. Figure: `output/sp500_gross_cap/gross_cap.png`.

### 7.1 Real market, RIE

T = 1000 days:

| cap | realised vol | honesty | turnover / month | longs / shorts | max weight | first half | second half |
|---|---|---|---|---|---|---|---|
| 1 (long-only) | 14.62% | 0.85 | 0.14 | 37 / 0 | 13.8% | 17.79% | 10.55% |
| 1.25 | 14.11% | 0.89 | 0.19 | 50 / 13 | 13.3% | 17.15% | 10.22% |
| 1.5 | 13.81% | 0.92 | 0.24 | 63 / 24 | 12.5% | 16.75% | 10.05% |
| 2 | 13.45% | **0.99** | 0.35 | 87 / 47 | 11.3% | 16.25% | 9.90% |
| 3 | 13.07% | 1.09 | 0.53 | 131 / 89 | 9.9% | 15.68% | **9.80%** |
| 5 | **12.99%** | 1.22 | 0.93 | 201 / 166 | 8.6% | **15.44%** | 9.96% |
| none (6.9) | 13.30% | 1.31 | 1.34 | 243 / 213 | 8.6% | 15.67% | 10.40% |

T = 250 days:

| cap | realised vol | honesty | turnover / month | longs / shorts | max weight | first half | second half |
|---|---|---|---|---|---|---|---|
| 1 (long-only) | 13.73% | **1.00** | 0.40 | 47 / 0 | 11.5% | 16.84% | 9.68% |
| 1.25 | 13.56% | 1.07 | 0.54 | 63 / 18 | 10.5% | 16.63% | 9.56% |
| 1.5 | 13.34% | 1.17 | 0.66 | 77 / 33 | 9.9% | 16.30% | 9.51% |
| 2 | 13.15% | 1.28 | 0.89 | 106 / 63 | 9.2% | 16.04% | **9.43%** |
| 3 | **13.05%** | 1.46 | 1.29 | 164 / 122 | 8.2% | **15.87%** | 9.44% |
| 5 (4.5 reached) | 13.22% | 1.63 | 1.79 | 236 / 200 | 7.4% | 16.05% | 9.60% |
| none (4.8) | 13.26% | 1.63 | 1.87 | 246 / 210 | 7.4% | 16.10% | 9.62% |

### 7.2 Real market, sample covariance

| cap | T = 1000: realised | honesty | T = 250: realised | honesty |
|---|---|---|---|---|
| 1 | 14.52% | 0.86 | 13.66% | 1.02 |
| 1.25 | 13.99% | 0.91 | 13.46% | 1.15 |
| 1.5 | 13.68% | 0.96 | 13.25% | 1.27 |
| 2 | 13.32% | 1.05 | **13.00%** | 1.57 |
| 3 | **13.12%** | 1.18 | 13.33% | 2.22 |
| 5 | 13.15% | 1.43 | 14.95% | 5.47 |
| none | 14.81% | 1.97 | 16.24% | 2.59 |

At 1000 days the capped sample covariance lands within 0.15 points of the capped RIE at every cap up to 3: the cap is doing the cleaning. At 250 days the sample covariance is singular (456 stocks, 250 days); with a cap of 2 it still produces the lowest realised number in the whole table (13.00%) but with a forecast off by 57%, and from cap 3 upward it finds in-sample directions of nearly zero variance and the forecasts become meaningless (honesty 2.2, then 5.5).

### 7.3 Gaussian lab

Gaussian days drawn from the raw full-sample covariance (seed 1): nothing drifts, no fat tails, and the true risk of every portfolio is known. "True vol" is the risk of the chosen weights under the true matrix, averaged over holds. The true-covariance rows show the pure price of the constraint.

| cap | truth: true vol | RIE, T = 1000: true vol (honesty) | sample, T = 1000 | RIE, T = 250 | sample, T = 250 |
|---|---|---|---|---|---|
| 1 | 11.84% | 12.09% (1.00) | 12.07% (1.01) | 12.61% (1.09) | 12.63% (1.12) |
| 1.25 | 11.07% | 11.41% (1.00) | 11.40% (1.02) | 12.10% (1.13) | 12.15% (1.21) |
| 1.5 | 10.55% | 10.96% (1.02) | 10.95% (1.05) | 11.79% (1.19) | 11.90% (1.31) |
| 2 | 9.91% | 10.45% (1.06) | 10.47% (1.09) | 11.47% (1.29) | 11.72% (1.53) |
| 3 | 9.17% | 9.92% (1.13) | 10.00% (1.18) | 11.20% (1.46) | 11.87% (2.05) |
| 5 | 8.49% | **9.59%** (1.19) | 9.79% (1.37) | 11.14% (1.57) | 12.90% (4.05) |
| none | 8.30% | 9.78% (1.24) | 11.27% (1.82) | 11.15% (1.57) | 14.45% (2.10) |

## 8. What we learned

1. **The dial works, and the U is broad.** For the 1000-day RIE on real data, realised risk falls from 14.6% (long-only) to 13.0% at cap 5 and rises again to 13.3% unconstrained. Both halves of the track put the minimum between caps 3 and 5, so the choice is not a fluke of one episode. Picking the cap on the first half (5) gives 9.96% on the second half against 10.40% unconstrained; picking on the second half (3) gives 15.68% on the first, level with the unconstrained 15.67%. The gain over unconstrained is small (0.2–0.3 points); the gain over long-only is large (1.6 points).

2. **Cap 2 is the honest point.** The 150/50 RIE book on 1000 days realises 13.45% with honesty 0.99, turnover 0.35 per month (a quarter of the unconstrained book's), 87 longs and 47 shorts, largest position 11%. It gives up 0.15 points of risk against the unconstrained optimiser and in return tells the truth about its risk, runs at 2× instead of 6.9× leverage, and trades a quarter as much. That is the portfolio a margin account can hold. Caps 3 and 5 squeeze out another 0.4 points at the price of honesty 1.09–1.22.

3. **The cap does the cleaning, until the matrix is singular.** At 1000 days the capped sample covariance lands within 0.15 points of the capped RIE for every cap up to 3, and in the lab a cap of 5 on the raw sample reaches the unconstrained RIE's true risk (9.79% against 9.78%). Cap and cleaning are substitutes when there are many more days than stocks. At 250 days the sample covariance is singular; with a cap of 2 it still finds the lowest real number in the table (13.00%) but its forecast is off by 57%, and above cap 2 it finds in-sample directions of almost zero variance and collapses (honesty 2.2, then 5.5, realised 15.0% at cap 5) while the RIE stays at 13.0–13.3%. When the window is short, cap and cleaning are complements.

4. **The price of the constraint when the truth is known is steep.** Lab, true covariance: 11.84% at cap 1, 9.91% at cap 2, 9.17% at cap 3, 8.49% at cap 5, 8.30% unconstrained. Shorting is worth 3.5 points of risk when the matrix is right. In the lab the RIE barely benefits from a cap (9.59% at cap 5 against 9.78%; nothing at 250 days), because its noise on Gaussian data is small. On real data the benefit appears earlier and is larger. The difference is the drift and the fat tails of the real market, which the lab does not have and which the cap protects against.

5. **For a capped book, the long window is the better one.** At 250 days honesty degrades fast as the cap loosens (1.07 at 1.25, 1.28 at 2, 1.46 at 3): the short window's optimism returns as soon as shorting is allowed. Its best real number (13.05% at cap 3) is no better than the 1000-day RIE's (13.07% at cap 3, 12.99% at cap 5) and far less honest. The freshness advantage of 250 days was about volatilities; the 1000-day book at cap 2 beats everything at 250 days on honesty at the same risk.

6. **What the project's question now looks like.** Weeks of work on the covariance moved the unconstrained portfolio from 14.8% to 13.3% and the forecast ratio from 2.0 to 1.3. One constraint on the optimiser moved the forecast ratio to 1.0 at 13.45%, and a looser one to 13.0%. Once the optimiser is constrained, the cleaning matters much less (finding 3), which means the RIE's remaining value for a holdable book is in the short-window and singular cases, and in honesty rather than risk. Open next: a position cap (the largest weight is 11% at cap 2); whether the cap's real-market gain comes from fat tails or from drift (the shuffled-days test would say); sector labels; and the clipping default.
