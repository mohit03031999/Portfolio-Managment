# Our results against the published record

*Written 2026-11-09. Numbers below were read from the papers' own tables (PDF text extracted the same day), not from memory or second-hand summaries. Our numbers are from `output/sp500_long_only/` and `output/sp500_gross_cap/` (93 holds of 21 days, 2018-12-24 to 2026-10-01, 456 S&P 500 stocks).*

## 0. In one paragraph

Four independent studies on US stocks, with different decades, data frequencies and estimators, show the same three patterns we found: once shorting is forbidden every covariance estimator gives about the same portfolio; a well-cleaned matrix with shorting allowed beats long-only by roughly 10–15%; and a raw or singular matrix with shorting allowed loses to long-only. On the gross-exposure cap, Fan, Zhang and Yu's U-shape with its minimum at a cap of 2 is exactly what our sample covariance does on a 250-day window, and the flat curve of their factor model is what our RIE does. The size of the RIE's gain over the raw sample covariance (10%) matches Bun, Bouchaud and Potters to the decimal in relative terms, and the raw sample covariance's forecast error in our Gaussian lab (1.82) matches the random-matrix prediction 1/(1 − q) = 1.84. What the record does not contain, and we add, is the forecast-honesty ratio along the constraint dial and the price of each constraint when the covariance is known exactly.

## 1. The setups, side by side

| study | universe | data | window | rebalance | period | risk scored as |
|---|---|---|---|---|---|---|
| **ours** | 456 S&P 500 stocks with a complete history | daily | 1000 d (q = 0.46) and 250 d (q = 1.8) | 21 days | 2018–2026 | annualised std of stitched daily out-of-sample returns; plus realised / predicted |
| Jagannathan & Ma 2003 (JM) | 500 random NYSE stocks above the 20th size percentile, redrawn yearly | monthly (daily for some estimators) | 60 months | yearly, held 12 months | 1968–1998 | annualised std of monthly returns |
| DeMiguel, Garlappi, Nogales & Uppal 2009 (DGNU) | 500 random CRSP stocks | monthly | 120 months (singular sample) | monthly | 1968–2005 | monthly variance |
| Fan, Zhang & Yu 2012 (FZY) | 600 random Russell 3000 stocks; also 100 Fama–French portfolios | daily | 24 months ≈ 500 d (singular sample) | monthly | 2003–2007 | annualised std of daily returns |
| Ledoit & Wolf 2017 (LW) | N = 30 … 500 largest CRSP stocks | daily | 250 d (singular sample at N ≥ 250) | monthly | 1973–2011 | annualised std of daily returns |
| Bun, Bouchaud & Potters 2017 (BBP) | 500 most liquid US stocks | daily, standardised by each day's cross-sectional volatility | 1000 d (q = 0.5) | 60-day test blocks | 1966–2012 | annualised std in standardised units |
| Clarke, de Silva & Thorley 2006 (CDT) | 1000 largest US stocks | monthly | — | — | 1968–2005 | realised std vs the cap-weighted market |

Levels are not comparable across rows: different decades (ours holds the 2020 crash), different universes, monthly against daily data, and BBP's standardised units. Ratios within a study are comparable, and that is what the sections below compare.

## 2. Long-only against unconstrained

| study | long-only | unconstrained, good estimator | unconstrained, raw or singular | reading |
|---|---|---|---|---|
| JM, Table 5 | sample 12.43%, 1-factor 12.62%, Ledoit 12.29%, FF3 12.38%, daily sample 12.34%: **a 0.3-point band** | Ledoit 10.76%, daily sample 10.64%, 1-factor 11.69%, FF3 11.35% | monthly sample: not computable (60 obs, 500 stocks) | unconstrained beats long-only by 1.5–2 points (13–15%); the estimator stops mattering under long-only |
| LW, Tables 1 and 8, N = 500 | sample 10.57%, linear 10.59%, nonlinear 10.74% | nonlinear 9.65%, linear 10.20% | sample: not available (singular) | unconstrained nonlinear beats long-only by 1.0 point (10%); "disallowing short sales helps Sample but hurts Lin, NonLin" |
| DGNU, Table 3, 500 CRSP | 10.2% (MINC) | Ledoit–Wolf 8.9% (MINL), norm-constrained 8.8–9.4% | sample 11.2% (MINU, 120 months for 500 stocks) | the singular sample loses to long-only; shrinkage beats both |
| **ours**, T = 1000 | sample 14.52%, RIE 14.62%: **a 0.1-point band** | RIE 13.30% | sample 14.81% | unconstrained RIE beats long-only by 1.3 points (10%) |
| **ours**, T = 250 | sample 13.66%, RIE 13.73% | RIE 13.26% | sample 16.24% | the singular-ish sample loses to long-only by 2.6 points |

Three agreements:

1. **Under long-only, the estimator stops mattering.** JM's band across five estimators is 0.3 points; LW's across three is 0.2; ours across two is 0.1. JM's abstract says it in one sentence: once the constraint is imposed, the monthly sample covariance performs as well as factor models, shrinkage estimators and daily data. We confirmed it on 2018–2026 daily data.
2. **A cleaned unconstrained matrix beats long-only by about 10–15%.** JM 13–15%, LW 10%, ours 10%. In the lab, with the covariance known exactly, the difference is 30% (8.30% against 11.84%); on real data noise and drift eat two-thirds of it in every study.
3. **A raw or singular unconstrained matrix loses to long-only.** DGNU (monthly, 120 obs for 500 stocks), FZY (see below), and our 250-day sample covariance. JM could not even compute it.

A fourth, loose check: CDT report the long-only minimum-variance portfolio at about three-quarters of the market's realised risk. Ours is 14.6% against 19.85% for equal weight, a ratio of 0.74; JM's is 12.4% against 15.6% value-weighted (0.80) or 17.5% equal-weighted (0.71).

## 3. The gross-exposure cap

| study | cap 1 | 1.5 | 2 | 3 | 4–5 | unconstrained | minimum at |
|---|---|---|---|---|---|---|---|
| FZY, Table 3, sample covariance (600 stocks, singular) | 9.28 | — | **8.20** | 8.43 | 8.94 / 9.66 | 12.20 (c = 8) | 2 |
| FZY, Table 3, factor model | 9.08 | — | **8.31** | 8.65 | 8.66 / 8.66 | 8.66 | 2, then flat |
| FZY, Table 3, RiskMetrics | 9.78 | — | **8.44** | 8.95 | 9.43 / 10.04 | 11.06 | 2 |
| FZY, Table 2, sample covariance (100 FF portfolios) | 10.14 | 8.41 | 7.56 | **7.13** | — | 7.82 | 3 |
| LW, Table 1, N = 500: sample with c = 2 ("FZY" column) | 10.57 (long-only sample) | — | **10.21** | — | — | nonlinear 9.65 | — |
| **ours**, sample, T = 250 (singular-ish) | 13.66 | 13.25 | **13.00** | 13.33 | 14.95 (5) | 16.24 | 2 |
| **ours**, sample, T = 1000 | 14.52 | 13.68 | 13.32 | **13.12** | 13.15 (5) | 14.81 | 3 |
| **ours**, RIE, T = 1000 | 14.62 | 13.81 | 13.45 | 13.07 | **12.99** (5) | 13.30 | 3–5, flat |
| **ours**, RIE, T = 250 | 13.73 | 13.34 | 13.15 | **13.05** | 13.22 (5) | 13.26 | 2–3, flat |

Readings:

- **The sharp U belongs to the noisy matrix.** FZY's sample covariance (singular, 600 stocks on 500 days) bottoms at c = 2 and climbs 50% by c = 8. Our 250-day sample covariance does the same: bottom at c = 2, up 25% by c = 5. FZY's own words: the ex-post volatilities of the sample covariance and RiskMetrics "shoot up quickly beyond c = 2".
- **The flat curve belongs to the cleaned matrix.** FZY's factor model is flat from c = 2 on; our RIE keeps improving gently to c = 3–5 and then gives back 0.3 points. Our 1000-day sample covariance sits in between (minimum at 3, mild rise), because 1000 days for 456 stocks is noisy but not singular.
- **Cap 2 on the sample covariance against unconstrained nonlinear shrinkage.** LW: 10.21% against 9.65%, shrinkage wins by 6%. Ours at T = 1000: 13.32% against 13.30%, a tie; at T = 250: 13.00% against 13.26%, the cap wins by 2%. The two are close substitutes everywhere; which one edges ahead depends on the window.
- **Cap 2 against long-only, same matrix.** FZY −12% (sample), −8% (factor); LW −3%; ours −5% (sample, 250 d), −8% (sample, 1000 d), −8% (RIE, 1000 d). Allowing 50% of shorting always buys several points of risk.

What the published sweeps do not show is the forecast. Along our dial the RIE's honesty goes 0.85 (cap 1), 0.99 (cap 2), 1.09 (cap 3), 1.22 (cap 5), 1.31 (unconstrained). None of the papers reports realised against predicted risk, so none can say where on the dial the model still tells the truth. Ours says cap 2.

## 4. Cleaning without constraints

| comparison | published | ours |
|---|---|---|
| raw sample → RIE, minimum variance, US stocks, T = 1000 | BBP Table 4: 11.6% → 10.4%, **−10.3%** (clipping 10.6%, −8.6%; linear shrinkage 10.5%, −9.5%) | 14.81% → 13.30%, **−10.2%** (clipping 13.5%, −8.8%) |
| linear → nonlinear shrinkage, N = 500, T = 250 | LW Table 1: 10.20% → 9.65%, −5.4%; adding a market factor to nonlinear (NL-SF) 8.61%, −15.6% | not re-run in this round |
| forecast error of the raw sample covariance | BBP §7.1: the in-sample risk understates the realised risk by (1 − q)² in variance, so realised / predicted volatility = 1/(1 − q) = **1.84** at our q = 0.456 | Gaussian lab, T = 1000: **1.82**; real market: 1.97 |

The relative gain of the RIE over the raw sample covariance is the same in BBP's 1966–2012 standardised data and in our 2018–2026 raw data, and clipping sits at the same fraction of that gain in both. The random-matrix forecast of the sample covariance's optimism is reproduced in our lab to the second decimal; the real market adds 0.15 on top, which is the fat-tails-and-drift part we measured separately in the shuffle test.

## 5. What the record does not contain

- **Honesty along the dial.** No paper reports realised / predicted risk for constrained portfolios. Our finding that the forecast is exact at cap 2 and 30% optimistic unconstrained is new to us; we have not seen it published.
- **The price of each constraint when nothing is noisy.** FZY prove a bound and say the constraint "decreases drastically the oracle and actual risks" as c grows; our Gaussian lab puts numbers on it for a realistic covariance: 11.84% (cap 1), 9.91% (2), 9.17% (3), 8.49% (5), 8.30% (none).
- **The cap on a nonlinearly cleaned matrix.** FZY sweep the sample covariance, a factor model and RiskMetrics; LW run the cap only at c = 2 and only on the sample covariance. The flat RIE curve with its minimum at 3–5 is ours.
- **Where the gain comes from.** No paper separates fat tails from drift in the constraint's benefit. The shuffled-days world would; it is not run.

Caveats on our side: one period (2018–2026, holding the 2020 crash and the 2022 bear market), one universe with survivorship bias (stocks with a complete history as of 2026), realised volatility of 21-day stitched returns rather than monthly returns, and 93 holds, so differences under about 0.2 points between our own rows are within noise.

## 6. Is this already implemented elsewhere?

Yes, as constraint options in the general-purpose optimisers; no, none of them ships results for it.

| library | long-only | gross cap | notes |
|---|---|---|---|
| skfolio (`MeanRisk`) | `min_weights=0` (the default) | `max_long` and `max_short` cap the two legs separately, so the gross cap is `max_long + max_short`; `l1_coef` is the penalty form | `budget=1` is our Σw = 1 |
| cvxportfolio | `LongOnly()` | `LeverageLimit(L)`: exactly Σ|wᵢ| ≤ L, cash excluded | also `MaxWeights` / `MinWeights` for position caps |
| PyPortfolioOpt (`EfficientFrontier`) | `weight_bounds=(0, 1)` (the default) | not native; `add_constraint` with a cvxpy expression on the weights | `L2_reg` is a ridge, not a gross cap |
| Riskfolio-Lib | has short-sale flags and long/short exposure limits in its `Portfolio` class | | its documentation page was not reachable today, so the parameter names are not verified here |

All of them hand the problem to a generic convex solver (CVXPY and friends). Ours solves the two problems directly with active-set methods on the Lawson–Hanson and Fan–Zhang–Yu optimality conditions, without a dependency, and is verified against a brute-force search over every support and sign pattern on small problems. None of the libraries' documentation reports an out-of-sample comparison of long-only, capped and unconstrained portfolios; the comparisons live in the papers above.

## 7. Sources

- Jagannathan, R., Ma, T. (2003). Risk reduction in large portfolios: why imposing the wrong constraints helps. *Journal of Finance* 58(4), 1651–1683. NBER WP 8922, Table 5. https://www.nber.org/system/files/working_papers/w8922/w8922.pdf
- DeMiguel, V., Garlappi, L., Nogales, F. J., Uppal, R. (2009). A generalized approach to portfolio optimization: improving performance by constraining portfolio norms. *Management Science* 55(5), 798–812. Numbers from the authors' slide deck, Table 3. https://www.cim.pt/files/Followup_Optim_Finance/deMiguel.pdf
- Fan, J., Zhang, J., Yu, K. (2012). Vast portfolio selection with gross-exposure constraints. *JASA* 107(498), 592–606, Tables 2–3 and Theorem 1. https://pmc.ncbi.nlm.nih.gov/articles/PMC3535429
- Ledoit, O., Wolf, M. (2017). Nonlinear shrinkage of the covariance matrix for portfolio selection: Markowitz meets Goldilocks. *Review of Financial Studies* 30(12), 4349–4388, Tables 1 and 8. https://www.econ.uzh.ch/dam/jcr:ee86799b-f478-4495-b6c5-9e3ee0b07a51/rfs_2017.pdf
- Bun, J., Bouchaud, J.-P., Potters, M. (2017). Cleaning large correlation matrices: tools from random matrix theory. *Physics Reports* 666, 1–109, §7.1 and Table 4. https://arxiv.org/abs/1610.08104
- Clarke, R., de Silva, H., Thorley, S. (2006). Minimum-variance portfolios in the U.S. equity market. *Journal of Portfolio Management* 33(1), 10–24 (abstract-level figure only; full text not reached).
- Library documentation: https://skfolio.org/generated/skfolio.optimization.MeanRisk.html, https://www.cvxportfolio.com/en/stable/constraints.html, https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html
