# Algorithm catalogue: every published way to build the covariance for a minimum-variance portfolio

A reference to read before designing a method of our own. Each algorithm gets the same five lines:

- **Idea**: the thought behind it, in one or two sentences.
- **Steps**: what you actually compute.
- **Changes**: which part of the covariance it touches. *Sizes* means eigenvalues, *directions* means eigenvectors, *time* means how old and new days are weighed, *portfolio* means it leaves the matrix alone and constrains the weights.
- **How it did**: our own test on 456 S&P 500 stocks where we ran it (realised volatility, honesty), otherwise the best published number near our size.
- **Weak point.**

Our reference numbers, real market, T = 1000, 21-day holds, December 2018 to October 2026: look-ahead oracle 9.2%, best plain method so far 12.7% (RIE on 100–250 days), RIE on 1000 days 13.3%.

The picture from our persistence study (`fixing_the_directions.md` §7.2) is a useful lens. The covariance has a **slow skeleton** (market arrow, sector room), **fast weather** (volatilities, current strength of co-movement) and **fog** (the middle and quiet directions, where nothing carries over). Each algorithm below is tagged with which of the three it works on.

---

## Map

| Family | Algorithms | Works on |
|---|---|---|
| A. Baselines | sample covariance, diagonal, equal weight | — |
| B. Fix the sizes | linear shrinkage, MP clipping, nonlinear shrinkage, optimal RIE, cross-validated eigenvalues, Average Oracle, Barra eigen-adjustment | fog |
| C. Impose a shape | single-index and factor models, PCA factors, POET, factor + nonlinear shrinkage, trees (nested factors, HPCA, BAHC), sparse inverse | skeleton |
| D. Pull arrows toward an anchor | James–Stein eigenvector, MAPS, strategy-specific shrinkage, quadratic optimisation bias, augmented James–Stein | skeleton |
| E. Better raw data | Kendall's tau, Tyler / R-NL, open-high-low-close returns, intraday realised covariance | skeleton + weather (via fat tails) |
| F. Model time | rolling window, EWMA, EWMA + RMT, volatility standardisation, GARCH, DCC, DCC-NL | weather |
| G. Constrain the portfolio | long-only, norm caps, gross-exposure cap, partial minimum variance, resampling, hierarchical risk parity | none (bypasses the matrix) |
| H. Learn it | end-to-end neural network | fog + weather |

---

## A. Baselines

### A1. Sample covariance
- **Idea:** average the products of returns over the window.
- **Steps:** S = (1/(T−1)) Σₜ (rₜ − r̄)(rₜ − r̄)ᵀ.
- **Changes:** nothing; this is the raw input to everything else.
- **How it did (ours):** 14.8%, honesty 1.97. The optimiser trusts every coincidence.
- **Weak point:** about 104,000 numbers from about four observations each; the inverse is inflated by roughly T/(T − N).

### A2. Diagonal (ignore all correlations)
- **Idea:** keep each stock's variance, set every correlation to zero.
- **Changes:** throws away the whole skeleton.
- **Weak point:** cannot hedge the market; very far from minimum variance.

### A3. Equal weight (1/N)
- **Idea:** no estimation at all.
- **How it did (ours):** 19.9%, honesty 0.68.
- **Weak point:** ignores risk entirely; the benchmark every method must beat.

---

## B. Fix the sizes (eigenvalues), keep the sample directions

All of these are *rotation-equivariant*: they keep the sample eigenvectors and change only eigenvalues. Section 1 of `fixing_the_directions.md` explains why that sets a floor.

### B1. Linear shrinkage (Ledoit, Wolf 2003, 2004)
- **Idea:** blend the noisy matrix with a simple, sturdy one.
- **Steps:** Σ̂ = (1 − δ)·S + δ·F, with the blend δ chosen from the data to minimise expected error. The target F can be the identity scaled to the average variance (2004, *J. Multivariate Analysis*), constant correlation (2004, "Honey, I shrunk…"), or a single-index model (2003, *J. Empirical Finance*).
- **Changes:** with the identity target, sizes only. With the other targets, also directions, because the target has different eigenvectors.
- **How it did (ours, identity target):** 13.45%, honesty 1.55.
- **Weak point:** one dial for all eigenvalues: shrinks big and small by the same rule.

### B2. Marchenko–Pastur clipping (Laloux, Cizeau, Bouchaud, Potters 1999; Plerou et al. 1999)
- **Idea:** eigenvalues inside the band pure noise would produce are noise; flatten them.
- **Steps:** correlation matrix, eigenvalues; every eigenvalue below the MP edge (1 + √(N/T))² is replaced by their average; rebuild, reset the diagonal to 1, rescale by volatilities.
- **Changes:** sizes (fog flattened, skeleton kept).
- **How it did (ours):** 13.52%, honesty 1.47.
- **Weak point:** hard cut; keeps the large eigenvalues at their inflated sample values.

### B3. Nonlinear shrinkage (Ledoit, Wolf 2012 QuEST, 2020 analytical; *Annals of Statistics*)
- **Idea:** each eigenvalue gets its own correction, derived from RMT.
- **Steps:** for each sample direction uᵢ, estimate the true variance along it, uᵢᵀΣuᵢ, from the sample spectrum alone.
- **Changes:** sizes, optimally among methods that keep the sample directions.
- **How it did:** N = 500, T = 1260, CRSP: 8.63% (De Nard, Ledoit, Wolf 2021). Beats linear shrinkage by 2–5%.
- **Weak point:** the eigenvector floor; assumes the truth does not change in the window.

### B4. Optimal RIE (Bun, Allez, Bouchaud, Potters 2016; Bun, Bouchaud, Potters 2017)
- **Idea:** the physicists' version of B3, through free probability; same target uᵢᵀΣuᵢ.
- **Steps:** Ledoit–Péché formula on the correlation spectrum, plus an inverse-Wishart lift for the smallest eigenvalues (our `rie.py`).
- **Changes:** sizes.
- **How it did (ours):** 13.30%, honesty 1.31 at 1000 days; 12.7% at 100–250 days on the same period.
- **Weak point:** same floor as B3.

### B5. Cross-validated eigenvalues (Bartz 2016; Tan, Zohren 2020)
- **Idea:** instead of a formula, measure each sample direction's variance on held-out days.
- **Steps:** split the window into folds; directions from the training folds, variances measured on the held-out fold; average, then force the result to be monotone (isotonic regression).
- **Changes:** sizes; captures some real change because held-out days differ.
- **How it did:** within 2–5% of nonlinear shrinkage in published tests.
- **Weak point:** noisy with few folds; still keeps sample directions.

### B6. Average Oracle (Bongiorno, Challet, Loeper 2023)
- **Idea:** accept that the truth moves; learn from history how much variance each rank of direction *actually* had next period.
- **Steps:** over thousands of past window pairs, record for each rank k the variance the past window's k-th direction had in the next window; average by rank; apply those fixed numbers to today's directions.
- **Changes:** sizes, calibrated to real drift.
- **How it did:** n = 100, 20-day rebalancing: 10.9% on a 240-day window against 11.5% for nonlinear shrinkage on 1200 days; much lower turnover.
- **Weak point:** one set of numbers for all market regimes; needs a long history to calibrate.

### B7. Barra eigenfactor risk adjustment (Menchero, Orr, Wang 2011)
- **Idea:** simulate how much each eigen-portfolio's risk is underestimated, then scale it up.
- **Steps:** simulate returns from the estimated matrix, re-estimate, measure the bias per eigenvalue rank, apply the correction (plus a 1.4× boost for fat tails and drift).
- **Changes:** sizes; aimed at honesty.
- **How it did:** in Barra's data the quietest eigenfactors were 40% riskier than predicted before the fix.
- **Weak point:** does what the RIE does analytically; leaves the directions alone.

---

## C. Impose a shape on the whole matrix

### C1. Single-index model (Sharpe 1963)
- **Idea:** every stock = beta × market + its own noise; stocks are linked only through the market.
- **Steps:** Σ̂ = σₘ² ββᵀ + diag(residual variances).
- **Changes:** directions (skeleton reduced to one arrow).
- **Weak point:** ignores sectors entirely.

### C2. Fundamental factor models (Barra and similar)
- **Idea:** returns explained by known characteristics: industry, size, value, momentum.
- **Steps:** Σ̂ = B F Bᵀ + D, with exposures B from company data, factor covariance F estimated, D diagonal.
- **Changes:** directions, using outside information (company characteristics).
- **Weak point:** "residuals independent" is too strict; needs data we do not have.

### C3. Statistical (PCA) factor model
- **Idea:** let the data pick the factors: keep the top k eigenvectors, call the rest independent noise.
- **Steps:** Σ̂ = Σᵢ≤k λᵢuᵢuᵢᵀ + diag(residual variances).
- **Changes:** directions (the diagonal residual rotates the small eigenvectors).
- **Weak point:** inherits the tilt of the sample factors (same window).

### C4. POET (Fan, Liao, Mincheva 2013)
- **Idea:** PCA factors plus a residual matrix where small entries are set to zero.
- **Changes:** directions.
- **How it did:** 8.48% at N = 500, T = 1260 against 8.63% for nonlinear shrinkage (De Nard et al. 2021).
- **Weak point:** threshold choice; versions without factors blow up.

### C5. Factor + nonlinear shrinkage (Ledoit, Wolf 2017 NL-SF; De Nard, Ledoit, Wolf 2021 AFM-NL)
- **Idea:** remove the market (or a few factors), clean what is left with nonlinear shrinkage, add the factors back.
- **How it did:** a strong gain at N/T = 2 (8.61% against 9.65%), **exactly zero** at our N/T (8.63% = 8.63%). Helps on top of a dynamic model (8.11% against 8.26%).
- **Weak point:** the factor comes from the same window, so it brings no new information when the window is long.

### C6. Hierarchically nested factor model (Tumminello, Lillo, Mantegna 2007)
- **Idea:** cluster stocks into a tree; each branch is a factor shared by the stocks below it.
- **Steps:** average-linkage clustering of the correlation; correlation of two stocks = the value at the node where they join; prune unreliable nodes by bootstrap.
- **Changes:** directions (skeleton = tree).
- **How it did:** no portfolio test; tree filters 12.3–13.0% against 12.2% for clipping on 90 stocks (Pantaleo et al. 2011).
- **Weak point:** the tree is noisy; never used as a portfolio covariance.

### C7. Hierarchical PCA (Avellaneda, Serur 2020)
- **Idea:** trust correlations inside a sector; link sectors only through each sector's top eigenvector.
- **Changes:** directions, using sector labels.
- **Weak point:** no minimum-variance test.

### C8. BAHC and k-BAHC (Bongiorno, Challet 2020–21)
- **Idea:** build the tree many times on bootstrap resamples and average the filtered matrices.
- **How it did:** beats cross-validated eigenvalues only for windows under ~200–300 days at n = 100; under 1% at each method's best window.
- **Weak point:** advantage disappears at long windows.

### C9. Sparse inverse: hedging regressions (Stevens 1998; Goto, Xu 2015; Callot et al. 2021) and graphical lasso (Lee, Seregina 2024)
- **Idea:** minimum variance needs Σ⁻¹, and row i of Σ⁻¹ is "regress stock i on all others". Force each stock to be hedged by only a few others.
- **Steps:** lasso regression of each stock on the rest (nodewise), or penalised likelihood on Σ⁻¹ (graphical lasso), optionally after removing factors.
- **Changes:** directions (through the inverse).
- **How it did:** about 10.6% against about 9.5% for linear shrinkage on 452 S&P stocks, one year (Callot et al.).
- **Weak point:** sparsity is not how equity hedges look; weak at large N.

### C10. Blockbuster (De Nard 2022)
- **Idea:** find blocks of related stocks from the data, then shrink within and between blocks.
- **How it did:** beats nonlinear shrinkage only for small and medium universes (abstract).

---

## D. Pull the arrows toward an anchor

### D1. James–Stein for the leading eigenvector (Goldberg, Papanicolaou, Shkolnik 2022; Goldberg, Kercheval 2023)
- **Idea:** noise makes the stocks' betas look more spread out than they are; shrink each one toward the average.
- **Steps:** loadings b = √λ₁·h; c = 1 − noise/spread; b̃ = mean + c·(b − mean); renormalise.
- **Changes:** directions (market arrow).
- **How it did (ours, `rie_js`):** worse at every window from 25 to 1000 days (+0.3% to +4.7%), honesty slightly better. At short windows it does turn the arrow toward the truth, but even the true arrow made short-window portfolios worse.
- **Weak point:** for the S&P 500 the market arrow's error is a leak into the sector arrows, not extra dispersion.

### D2. Multiple anchor point shrinkage, MAPS (Gurdogan, Kercheval 2022)
- **Idea:** same as D1, but pull toward a richer target: sector groups, beta-ranked groups, or last window's arrow.
- **How it did:** simulation only; sector anchors beat the single average anchor.
- **Weak point:** no real-data test.

### D3. Strategy-specific eigenvector shrinkage (Goldberg, Gurdogan, Kercheval 2025)
- **Idea:** anchor on the portfolio's own constraint vectors, because the error is amplified exactly along them.
- **How it did:** simulation only.

### D4. Quadratic optimisation bias (Gurdogan, Shkolnik 2024)
- **Idea:** correct several factor arrows at once.
- **How it did:** simulation, 7 factors, p = 500: 11.43% → 10.75%. Proves that the rotation *inside* the factor room cannot be estimated.

### D5. Augmented James–Stein (Seong, Hong, Jung 2026)
- **Idea:** include the other large sample eigenvectors in the anchor, so the market arrow is separated from the sector arrows; built for fixed N/T, our regime.
- **How it did:** alignment simulations only, no portfolio test.
- **Weak point:** brand new preprint.

---

## E. Better raw data for the directions

### E1. Kendall's tau (Espana, Le Coz, Smerlak 2024)
- **Idea:** for each pair of days, each stock votes "up or down relative to the other day"; huge days get no extra say.
- **Steps:** divide each day by its cross-sectional size and each stock by its volatility; Kendall correlation matrix; eigenvalues by cross-validation (or none at all).
- **Changes:** directions, robust to fat tails.
- **How it did:** ~500 North American stocks, T = 1000: −14.5% from the directions alone, the best real-data result at our size (one paper, weak RIE baseline).
- **How it did (ours, 2026-11-09):** real market 13.63% against RIE 13.30% (+2.5%), honesty 1.24 against 1.32; +2% to +3% riskier in both lab worlds; its directions are no better aimed than Pearson's (top-20 room 0.74 against 0.69 on fat-tailed days, slightly worse on Gaussian days); moving the RIE's top-k room to Kendall's room changes risk by ±3% of the gap, which is noise. The published gain did not transfer; see `fixing_the_directions.md` §7.3.
- **Weak point:** no RIE exists for Kendall matrices; gain fades with longer windows; measures typical-day co-movement, while money risk is made on the big days.

### E2. Tyler's estimator with nonlinear shrinkage, R-NL (Hediger, Näf, Wolf 2023)
- **Idea:** scale every day to unit length, so only its pattern counts, then fit directions to the patterns.
- **How it did:** N = 500, T = 1260: 9.35% → 9.12%, mostly from cleaning the correlation, which our RIE already does.

### E3. Open-high-low-close returns (De Nard, Engle, Ledoit, Wolf 2022)
- **Idea:** the day's range carries more information than the close; tame big moves to a sign and rescale by the range.
- **How it did:** N = 500, T = 1260: 9.01% → 7.44% against DCC-NL.
- **Weak point:** needs open/high/low prices we do not store (`yfinance` provides them).

### E4. Intraday realised covariance (Hautsch, Kyj, Oomen 2012; Hautsch, Kyj, Malec 2015)
- **Idea:** many returns per day, so far more observations.
- **How it did:** a tie before 2008, −10.7% during the crisis, against a weak daily baseline.
- **Weak point:** needs tick data; asynchronous trading.

---

## F. Model time

### F1. Rolling window
- **Idea:** use only the last T days.
- **How it did (ours, RIE, same period):** 25 days 13.1%, 100 days 12.7%, 250 days 12.7%, 1000 days 13.3%.
- **Weak point:** short = fresh but noisy and far too optimistic (honesty 2.4 at 100 days); long = stable but stale.

### F2. Exponentially weighted (RiskMetrics 1996)
- **Idea:** every day counts, but weight halves every h days.
- **How it did (ours, h = 252 with a weighted-noise RIE):** 13.0%, honesty 1.29.
- **Weak point:** one speed for every piece.

### F3. Exponential weighting + RMT cleaning (Pafka, Potters, Kondor 2004)
- **Idea:** the noise law changes with the weights; clean with the right one.
- **How it did (ours):** this is F2's RIE.

### F4. Volatility standardisation (ours, after the DCC idea)
- **Idea:** separate volatility (weather) from correlation (skeleton): divide returns by a volatility forecast, clean the correlation, scale back.
- **How it did (ours):** per-stock forecasts hurt; a single market-wide factor kept risk (13.5%, honesty 1.35) and cut leverage and turnover. Any per-stock trust above zero hurt.

### F5. GARCH volatilities (Bollerslev 1986)
- **Idea:** each stock's volatility follows a recursion: today's variance = long-run level + reaction to yesterday's shock + persistence.
- **Weak point:** many parameters across 456 stocks.

### F6. DCC (Engle 2002) and DCC-NL (Engle, Ledoit, Wolf 2019)
- **Idea:** GARCH volatilities plus a correlation matrix that drifts day by day toward recent co-movements, anchored on a long-run target (cleaned by nonlinear shrinkage in DCC-NL).
- **Steps:** Qₜ = (1 − α − β)·C + α·sₜ₋₁sₜ₋₁ᵀ + β·Qₜ₋₁, normalised to a correlation.
- **How it did:** N = 500: 9.64% against 9.74% with a fixed correlation. The moving correlation is worth about 1%; most of the gain is from the GARCH volatilities.
- **Weak point:** two numbers (α, β) govern all 104,000 pairs; turnover 3–10× higher.

---

## G. Constrain the portfolio instead of fixing the matrix

### G1. Long-only (Jagannathan, Ma 2003)
- **Idea:** forbid short positions; mathematically the same as shrinking the covariance of the stocks you would have shorted.
- **How it did:** every good cleaner performs about the same as the sample covariance once long-only is imposed. Their Table 5 (500 random NYSE stocks, monthly data, 60-month windows, 1968–1998): long-only 12.3–12.6% for every estimator (sample, one-factor, Ledoit shrinkage, Fama–French, daily sample); unconstrained 10.6–11.7% with the factor, shrinkage and daily-data estimators; equal weight 17.5%. Ledoit–Wolf (2017, Table 8, N = 500 daily) find the same: long-only 10.57% (sample) to 10.74% (nonlinear shrinkage) against 9.65% for unconstrained nonlinear shrinkage.
- **How it did (ours, 2026-11-09):** built as `long_only_minimum_variance_weights` (Lawson–Hanson active set on the normal equations) and the `longonly_<estimator>` strategies. 93 holds, 456 stocks: honesty 0.85–1.02 on real data (unconstrained 1.31–2.59) and 1.00 in the Gaussian lab, so the optimiser's optimism is gone; realised risk 14.6% at T = 1000 and 13.7% at T = 250 against 13.3% for the unconstrained RIE, so shorting still carried real information; sample and RIE long-only within 0.1 points of each other (Jagannathan–Ma confirmed); leverage 1, 36–47 stocks, max weight 11–14%, turnover 0.14–0.40 per month against 1.3–1.9. Lab: true-covariance portfolios 8.30% unconstrained against 11.84% long-only, the price of the constraint when nothing is noisy. Details in `docs/long_only_minimum_variance.md`.
- **Weak point:** raises the best achievable risk by 1–2 points.

### G2. Norm caps and partial minimum variance (DeMiguel, Garlappi, Nogales, Uppal 2009)
- **Idea:** limit how extreme the weights can be, or take only a few steps from equal weights toward minimum variance so the noisiest directions are never reached.
- **How it did:** 500 stocks, monthly data: 8.8% against 11.2% unconstrained, similar to Ledoit–Wolf.

### G3. Gross-exposure cap (Fan, Zhang, Yu 2012)
- **Idea:** cap total long plus short; with the cap, only the largest single entry error matters, not the eigenvectors.
- **How it did:** Table 3 (600 random Russell 3000 stocks, daily, 24-month windows, 2003–2007), sample covariance: 9.28% long-only, 8.20% at c = 2 (the minimum), 8.43% at c = 3, 9.66% at c = 5, 12.20% at c = 8 (their stand-in for unconstrained); the factor model goes 9.08% → 8.31% at c = 2 and stays flat. Table 2 (100 Fama–French portfolios): 10.14% → 8.41% → 7.56% → 7.13% at c = 1, 1.5, 2, 3 against 7.82% unconstrained. Ledoit–Wolf (2017, Table 1) re-ran c = 2 on the sample covariance for N = 500 daily: 10.21% against 10.57% long-only and 9.65% for unconstrained nonlinear shrinkage. (An earlier version of this line quoted "400 stocks, 8.62% at c = 2.5 against 14.4%", which does not match the paper; corrected 2026-11-09 from the paper's tables.)
- **How it did (ours, 2026-11-09):** built as `gross_capped_minimum_variance_weights` (primal active set on the band conditions) and the `gross<cap>_<estimator>` strategies; sweep over caps 1, 1.25, 1.5, 2, 3, 5, none on 93 holds. RIE, 1000 days: 14.6% (cap 1) → 13.5% at cap 2 with honesty 0.99 → 13.1% at cap 3 (honesty 1.09) → 13.0% at cap 5 → 13.3% unconstrained (honesty 1.31); turnover 0.35 at cap 2 against 1.34 unconstrained. Both halves of the track put the minimum between caps 3 and 5, and a cap picked on one half beats the unconstrained book on the other. Sample covariance with a cap of 2–3 lands within 0.15 points of the RIE at 1000 days (the cap does the cleaning), but at 250 days the singular sample blows up above cap 2 (honesty 2.2 at cap 3, 5.5 at cap 5) while the RIE stays at 13.0–13.3%. Lab: the cap costs real hedging when the truth is known (true covariance 11.8% at cap 1, 9.9% at cap 2, 8.3% unconstrained), and a cap of 5 on the raw sample covariance reaches the unconstrained RIE's true risk (9.79% against 9.78%). Details in `docs/gross_exposure_cap.md`.

### G4. Subset resampling (Shen, Wang 2017)
- **Idea:** compute minimum variance on many random small groups of stocks and average the weights.
- **How it did:** 434 S&P stocks, 500-day window: 6.98% against 7.78% for Ledoit–Wolf.
- **Weak point:** never compared with nonlinear shrinkage or an RIE.

### G5. Michaud resampling (Michaud 1998)
- **Idea:** simulate many plausible return histories, optimise each, average the portfolios.
- **How it did:** loses to Ledoit–Wolf, adds nothing on top of it (Wolf 2007).

### G6. Hierarchical risk parity, HRP (López de Prado 2016)
- **Idea:** cluster stocks into a tree, then split money top-down so each branch gets less when it is riskier; never inverts the matrix.
- **Weak point:** not a minimum-variance method; robust but not the lowest risk.

---

## H. Learn the whole thing

### H1. End-to-end neural network (Bongiorno, Manolakis, Mantegna 2026)
- **Idea:** train a network to output the eigenvalues, volatilities and day weights that minimise realised portfolio risk directly.
- **How it did:** n = 300, long-short: 10.9% against 11.1% for nonlinear shrinkage, with 2.5× the turnover; long-only 13.5% against about 14.2%.
- **Weak point:** keeps the sample directions; needs a long training history.

---

## I. Our own attempts

### I1. Two-speed estimator: directions from the past, sizes from the present (2026-11-09)
- **Idea:** the skeleton drifts slowly, so take the directions from a long history; sizes and volatilities move fast, so measure them on recent days along those directions. A size measured along a direction not fitted to those days is honest by construction, so no random-matrix formula is needed.
- **Steps:** eigenvectors of the correlation of the history; size of each = variance of the last L days along it, isotonic in the old rank; renormalise the diagonal; scale by the recent volatilities.
- **Changes:** directions (from older data) and sizes (out of sample).
- **How it did (ours):** Gaussian lab 9.53% against RIE 9.78% (best estimator we ran, honesty 1.18); real market 15.9–16.6% against 13.30% with honesty 2.2. Two controls that swap only the volatilities between the 1000- and 250-day RIEs are both worse than either pure window (+3.7%, +5.6%).
- **Weak point:** the market arrow drifts enough (overlap ≈ 0.92 with the next period from older data against 0.99 from the fresh window) that an old arrow mis-hedges a 100-unit factor; volatilities and correlations split across windows give an incoherent matrix. See `fixing_the_directions.md` §7.5.

## How the families line up with what we measured

| Our finding | Families that address it | Our evidence so far |
|---|---|---|
| Market arrow stable; its small tilt leaks into the sector arrows | D (anchors), C (factors) | D1 failed; C5 gives 0% at our N/T |
| Sector room real and lasting, but hard to measure; single arrows blur | C (sectors, trees), D2/D5 | untested on our data |
| Fat tails ruin the factor directions | E (Kendall, Tyler) | Kendall untested on our data |
| Volatilities and correlations drift over years | F (time) | F2 small gain; F4 per-stock hurt; F1 fresher windows −4.6% |
| Middle directions carry nothing | B (flatten) | done by the RIE |
| The optimiser amplifies every error | G (constraints) | long-only and the gross-exposure cap built and tested 2026-11-09: cap 2 on the RIE is honest (0.99) at 13.5%, caps 3–5 reach 13.0–13.1% against 13.3% unconstrained; a position cap is untested |

The untested combinations from the literature are listed in `fixing_the_directions.md` §6.

## Sources

The full source list with links is in `fixing_the_directions.md` §9 and `why_the_gap_persists.md` §8. Additional classic references used here: Sharpe (1963), A simplified model for portfolio analysis, *Management Science* 9; Laloux, Cizeau, Bouchaud, Potters (1999), Noise dressing of financial correlation matrices, *Phys. Rev. Lett.* 83; Plerou et al. (1999), Universal and nonuniversal properties of cross correlations in financial time series, *Phys. Rev. Lett.* 83; Ledoit, Wolf (2004), A well-conditioned estimator for large-dimensional covariance matrices, *J. Multivariate Analysis* 88; Ledoit, Wolf (2012), Nonlinear shrinkage estimation of large-dimensional covariance matrices, *Annals of Statistics* 40; Ledoit, Wolf (2020), Analytical nonlinear shrinkage of large-dimensional covariance matrices, *Annals of Statistics* 48; Bun, Allez, Bouchaud, Potters (2016), Rotational invariant estimator for general noisy matrices, *IEEE Trans. Information Theory* 62; Bartz (2016), Cross-validation based nonlinear shrinkage, arXiv:1611.00798; J.P. Morgan/Reuters (1996), RiskMetrics technical document; Bollerslev (1986), Generalized autoregressive conditional heteroskedasticity, *J. Econometrics* 31; Engle (2002), Dynamic conditional correlation, *J. Business & Economic Statistics* 20; Michaud (1998), *Efficient Asset Management*; López de Prado (2016), Building diversified portfolios that outperform out of sample, *J. Portfolio Management* 42.
