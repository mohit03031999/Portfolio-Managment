# Fixing the directions: what the literature does to eigenvectors, and where our error sits

## 0. The short answer, in plain words

**Yes: the directions are the biggest piece we have not touched.** But the error is not spread evenly, and fixing it needs information from outside the window.

What we found on our own data (section 2):
- **The market direction is almost perfect**, off by only 2–3 degrees. But it is so big that even this small tilt spills risk onto quiet directions and fools the optimiser. Fixing the market direction alone would close 13–18% of the gap.
- **The next 20 directions blur into each other**, because several of them are almost the same size. Fat tails make this much worse. Fixing the top 20 perfectly would close 38–45% of the gap. The rest is spread over directions 20 to 200.

What the literature says (sections 3–4):
- **You cannot fix directions for free.** Every method brings outside information: a direction you trust (all stocks equal, sectors), a shape (factors, blocks, trees), better data (ranks instead of raw returns, daily high/low prices), or a model of time.
- **What worked on real data at our size:** Kendall's rank correlation, about 15% less risk from the directions alone (one paper; tested on our panel 2026-11-09 and it did not transfer, +2.5% risk, section 7.3); daily high/low prices, 17% (one paper, needs data we do not keep).
- **What did not work at our size:** a static factor model (exactly 0%), sample splitting (0%), trees (only short windows), sparse inverses (worse than simple shrinkage).
- **Strong in simulation but never tried on real data:** the James–Stein methods that pull the market direction toward "all stocks equal".

Room for your own idea (section 6): several combinations nobody has tested. Examples: James–Stein on the top directions with the RIE on the rest; Kendall directions with an RIE built for Kendall; fixing the *subspace* of the top 20 rather than each direction; using earlier windows as an anchor; or attacking the spillover itself. Section 5 gives five levers to describe any idea with.


## 1. Why the directions cannot be fixed for free

The RIE already gives every sample direction the best possible variance. What it cannot do is move a direction. That is not laziness. If you know nothing about which way the truth points, every rotation is equally likely, and the sample directions are the best guess there is. Ledoit and Wolf say they keep the sample eigenvectors "only because we do not know how to improve upon them".

So every method that improves the directions brings in **outside information**: something believed before looking at the window. The literature uses five kinds:

1. **An anchor.** A direction you trust in advance, such as "all stocks move with the market" (the vector of ones) or "stocks in one sector move together". The noisy sample direction is pulled toward it.
2. **A shape.** An assumption about the matrix as a whole: a few factors plus independent leftovers, blocks by sector, a tree of clusters, or an inverse with many zeros.
3. **Better raw material.** Statistics that fat tails disturb less (ranks, Kendall's tau, Tyler's estimator), more information per day (intraday prices), or a longer history for the parts that move slowly.
4. **A model of time.** Directions are allowed to move, but smoothly, or old directions are forgotten gently.
5. **Skipping the matrix.** Leave the estimate alone and regularise the portfolio itself: cap the weights, drop noisy directions from the inverse, or average many portfolios.

When you design your own method, the first question is which outside information you trust, and for which directions.

## 2. Where the error sits on our data

Before reading what others do, we measured where the RIE's direction error is on our own 456 stocks (`examples/sp500_eigenvector_error.py`, results in `output/sp500_eigenvector_error/`). We used two worlds in which the truth is known exactly:

- **Gaussian days** drawn from the raw full-sample covariance: pure estimation noise.
- **Real days, shuffled**: fat tails and jumps, no time order. For each window, the truth is the covariance of the days outside it, because the hold days are drawn from those. This makes the truth stricter than in the staircase of `why_the_gap_persists.md`, so the "best possible" here is 7.80% instead of 8.25%.

Window T = 1000, the same 93 decision dates as the backtest, two runs per world. Realised risk is the exact expected variance under the truth.

### 2.1 Hand the RIE some true directions

We replace the RIE's matrix by the truth inside the span of the true top-*k* directions and keep the RIE everywhere else. We then do the same starting from the quietest directions. *k* = 0 is the RIE, *k* = 456 is the truth. This is a thought experiment, because no estimator knows the true directions. It says how much a perfect fix of each part would be worth.

| True directions handed over | Gaussian days: volatility | gap closed | Real days, shuffled: volatility | gap closed |
|---|---|---|---|---|
| none (the RIE) | 9.77% | 0% | 11.53% | 0% |
| the market only (top 1) | 9.58% | 13% | 10.87% | 18% |
| top 3 | 9.44% | 22% | 10.47% | 28% |
| top 10 | 9.33% | 30% | 10.25% | 34% |
| top 20 | 9.21% | 38% | 9.86% | 45% |
| top 50 | 9.04% | 49% | 9.35% | 58% |
| top 100 | 8.78% | 67% | 8.80% | 73% |
| top 200 | 8.53% | 84% | 8.38% | 84% |
| quietest 20 | 9.66% | 7% | 11.27% | 7% |
| quietest 100 | 9.27% | 34% | 10.31% | 33% |
| all 456 (the truth) | 8.30% | 100% | 7.80% | 100% |

Honesty moves with risk. In the Gaussian world it goes from 1.27 for the RIE to 1.16 with the top 10 and 1.04 with the top 100. In the shuffled world it goes from 1.50 to 1.30 with the top 10 and 1.03 with the top 100.

A second version gives only the true top-*k* **subspace**, the room the big directions live in, and keeps the RIE's own numbers inside it. Up to *k* = 20 it closes almost as much as the full version: 12% for the market, 29% for the top 10 and 35% for the top 20 on Gaussian days. On shuffled days the figures are 19%, 35% and 42%. For the big directions, the gain comes from knowing *where* they point, not *how big* they are.

![Which true directions close the gap, and where the extra variance lands](../output/sp500_eigenvector_error/eigenvector_error.png)

### 2.2 How well each sample direction lines up with the true one

Squared overlap of each sample correlation eigenvector with the true one. A value of 1 means the same direction; about 0.002 means two random directions.

| Mode | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gaussian days | 1.00 | 0.98 | 0.97 | 0.92 | 0.87 | 0.89 | 0.88 | 0.62 | 0.58 | 0.71 | 0.47 | 0.41 | 0.19 | 0.15 | 0.09 |
| Real days, shuffled | 0.99 | 0.94 | 0.91 | 0.75 | 0.44 | 0.65 | 0.44 | 0.12 | 0.12 | 0.25 | 0.10 | 0.11 | 0.05 | 0.06 | 0.05 |
| Textbook, one isolated spike | 1.00 | 0.99 | 0.99 | 0.98 | 0.97 | 0.97 | 0.96 | 0.95 | 0.95 | 0.94 | 0.92 | 0.90 | 0.89 | 0.88 | 0.87 |

### 2.3 What this says, in plain words

- **The market direction is almost perfect, and it still costs a lot.** Its overlap is 0.998, a tilt of only 2.6 degrees. But the market carries 34% of all the variance in the correlation matrix (eigenvalue 155 out of 456). A 2.6-degree tilt spills 155 × 0.002 ≈ 0.31 units of that variance onto a direction that is really quiet, while the quietest true directions hold only about 0.20 units each. The optimiser then thinks that quiet direction is risky and avoids it. Fixing the market direction alone closes 13% of the gap on Gaussian days. On shuffled real days the tilt is 5.7 degrees, the spill about 1.5 units, and fixing the market alone closes 18%. **A big direction with a small tilt does damage out of all proportion to its tilt.**
- **The next directions blur into each other.** Modes 8 and 9 have true sizes 4.66 and 4.44; modes 16 to 19 sit between 2.06 and 2.29. When two true directions are almost the same size, the sample cannot tell them apart and mixes them. That is why the measured overlaps beyond mode 7 are far below the textbook value, which assumes each spike stands alone. The *room* they span is better estimated than each single direction (section 2.1), so a method can aim at the subspace rather than at each vector.
- **Fat tails damage the factor directions.** On shuffled real days the overlaps of modes 4 to 7 fall to 0.44–0.75, and beyond mode 8 to 0.05–0.25, close to noise. The share of extra variance landing in the top 10 directions rises from 6% to 25%. So a tail-robust statistic is an eigenvector fix too, and it is aimed at the top 20.
- **The quiet end does not help on its own.** Knowing the quietest 20 directions exactly closes only 7% of the gap. The extra variance does land in the quiet directions (79% in modes 51–456 on Gaussian days), but its cause sits mostly higher up: the big directions spill into them.
- **A ceiling for anchors and factor priors.** Knowing the top 20 directions perfectly closes 38% of the gap on Gaussian days and 45% on shuffled days. Any method that only fixes the market and sector directions is capped near there. The rest lives in the middle, modes 20 to 200, which no simple anchor describes.
- **The dispersion bias is small at our window.** The spread of the sample market eigenvector's entries is 1.01 times the true spread on Gaussian days and 1.05 times on shuffled days. A correction aimed only at the dispersion bias fixes a small part of the market tilt at T = 1000; its published gains come from shorter windows or more stocks.

## 3. What the literature does to the directions

Each family below is described the same way: the idea in plain words, the core formula, the outside information it brings, what it achieved, and its limits. Section 7 says how well each number was checked.

### 3.1 Anchors: pull a direction toward one you trust

**The idea.** The sample market direction is tilted, and the tilt is not random. Its entries, which are roughly each stock's beta, are more spread out than the true betas. This is the *dispersion bias*. The optimiser treats the false spread as real. It thinks it can hedge the market by holding stocks whose beta looks low only because of noise, so the portfolio is riskier than it believes. The fix pulls the entries back toward their average, which is the same as pulling the direction toward "all stocks equal" (the vector of ones). It pulls hard when the noise is large and gently when it is small. This is the James–Stein idea applied to an eigenvector.

**The formula** (Goldberg and Kercheval 2023). For the top eigenvector *h* with eigenvalue λ, over *p* stocks and *n* days:

  h_JSE = m·1 + c·(h − m·1),  with m = mean of h's entries,  c = 1 − ν² / s²(h),
  s²(h) = (1/p) Σᵢ (λhᵢ − λm)²,  ν² = (tr S − λ²) / (p(n − 1)).

Shkolnik (2022) notes that when *p* < *n*, as for us, the divisor *n* − 1 becomes *p* − 1. The general form replaces "the ones vector" by any anchor subspace *C*. It shrinks the part of *h* that lies outside *C* by t = (1 − ρ̂²)/(1 − ‖P_C h‖²), where ρ̂² is the estimated squared overlap of *h* with the truth (Seong, Hong, Jung 2026).

| Paper | Anchor | Evidence |
|---|---|---|
| Goldberg, Papanicolaou, Shkolnik, *The dispersion bias* (SIAM J. Financial Math. 2022) | the ones vector | Simulation only. One factor, p = 500, n = 50: true minimum-variance volatility 8.49% (PCA) → 6.15% (corrected); the best possible is 4.85%. Honesty 2.16 → 1.04. With four factors, N = 500, T = 250 (arXiv v4): 5.10% → 4.82%, honesty about 1.20 → 1.03. |
| Goldberg, Kercheval, *James–Stein for the leading eigenvector* (PNAS 2023) | the ones vector | Shows the correction above is James–Stein. Simulations of the angle error; the portfolio test (p = 500, n = 252, fat-tailed specific returns) is shown only as boxplots: eigenvalue cleaning "makes no improvement", the eigenvector correction a "substantial" one. Code: github.com/kercheval-a/JSE. |
| Gurdogan, Kercheval, *Multiple anchor point shrinkage* (MAPS, SIAM J. Financial Math. 2022) | several anchors: sectors, beta-ranked groups, last window's eigenvector | Simulation, including real S&P 500 betas with simulated returns: 11 sector anchors beat the single ones-vector anchor; beta-ranked groups do best. Random anchors neither help nor hurt. Last window's eigenvector as an anchor helps when betas persist. |
| Goldberg, Gurdogan, Kercheval, *Portfolio optimisation via strategy-specific eigenvector shrinkage* (Finance and Stochastics 2025) | the constraint vectors of the portfolio itself | Simulation, p = 3000, n = 24: improvement positive in all 10,000 runs. |
| Gurdogan, Shkolnik, *The quadratic optimization bias of large covariance matrices* (arXiv 2024) | the ones vector, for many factors at once | Simulation, 7 factors, n = 120. Minimum-variance volatility at p = 500: optimum 7.69%, PCA 11.43%, fix factor 1 only 10.82%, fix all 10.75%. Much larger gains at p = 2000 and 8000. Proves that the rotation *inside* the factor subspace cannot be estimated when there are two or more factors. |
| Seong, Hong, Jung, *Augmented James–Stein estimation for leading eigenvectors and eigenspaces* (arXiv 2026) | an anchor plus the other big sample eigenvectors | The only version built for fixed N/T, our regime. Alignment simulations only, at p/n = 1 and 2. No portfolio test. |
| Monasson, Villamaina, *Estimating the principal components of correlation matrices from all their empirical eigenvectors* (EPL 2015) | a prior on the eigenvector's shape (how concentrated it is) | Sample eigenvectors just below a spike each carry a little of the true spike's direction; combining them beats the top sample eigenvector. N = 80, no portfolio test. |
| Barra USE4 eigenfactor risk adjustment (Menchero, Orr, Wang 2011) | none: changes eigenvalues, not eigenvectors | Simulates how much each sample eigen-portfolio's variance is underestimated and scales it up, plus a 1.4× boost for fat tails and drift. On real data the quietest eigenfactors were 40% riskier than predicted before the adjustment. This is the RIE's job done by simulation; it does not touch the directions. |

**Limits.** All the portfolio evidence is simulated, with a few spikes on a flat noise floor. No paper backtests on real returns, combines the correction with a full eigenvalue cleaner such as the RIE, or tests it at fixed N/T on portfolios. On our data the dispersion bias is small at T = 1000 (section 2.3), because the market eigenvalue is huge relative to the noise. The published gains are at n = 24–252 days, where the tilt is much larger.

### 3.2 Shapes: factors, sectors, trees, a sparse inverse

**The idea.** Assume the matrix has a simple form and estimate only that form. Examples: a few factors plus independent leftovers, correlation blocks by sector, a tree of clusters, or an inverse with many zeros (each stock hedged by only a few others). Shrinking toward such a target also rotates the eigenvectors, because the target has different ones.

| Paper | Shape | Evidence on minimum variance |
|---|---|---|
| De Nard, Ledoit, Wolf, *Factor models for portfolio selection in large dimensions* (J. Financial Econometrics 2021) | Fama–French factors, then nonlinear shrinkage of the leftovers | **The closest test to our setup:** CRSP, largest N stocks, T = 1260, rebalanced every 21 days, 1978–2017, long-short. At N = 500, nonlinear shrinkage 8.63%, with 1 factor 8.63%, with 5 factors 8.63%. **A static factor prior adds nothing.** It helps only on top of a dynamic model: DCC-NL 8.26%, one factor + DCC-NL 8.11%. POET 8.48%. |
| Ledoit, Wolf, *Markowitz meets Goldilocks* (RFS 2017), from `why_the_gap_persists.md` | one factor, then nonlinear shrinkage | N = 500, T = 250 (N/T = 2): 8.61% against 9.65%. The factor helps when the window is short. |
| Ledoit, Wolf (J. Empirical Finance 2003) | blend with a one-factor matrix | Monthly data, N > T (909–1314 stocks, 120 months): 9.55% against 10.21% for shrinking to the identity. Not our regime. |
| Fan, Liao, Mincheva, POET (JRSS B 2013) | top factors + thresholded leftovers | Own test on 100 portfolios, not stocks. Versions that threshold without factors blow up (De Nard et al.). |
| Avellaneda, Serur, *Hierarchical PCA* (arXiv 2020) | keep correlations inside a sector; link sectors only through each sector's top eigenvector | No minimum-variance test. |
| Tumminello, Lillo, Mantegna, *Hierarchically nested factor model* (EPL 2007) | a tree from clustering, one factor per branch | No portfolio test. In Pantaleo et al. 2011 (90 stocks, 1-year window, long-short), tree methods 12.3–13.0% against 12.2% for RMT clipping and 13.6% for the sample. |
| Bongiorno, Challet, BAHC and k-BAHC (2020–21), from `why_the_gap_persists.md` | bootstrapped trees | Wins only for windows under ~200–300 days at n = 100. |
| Callot, Caner, Önder, Ulaşan, *Nodewise regression* (JBES 2021) | sparse inverse: each stock regressed on a few others | S&P 500, 452 stocks, 964-day window, one year out of sample: about 10.6% against about 9.5% for linear shrinkage (annualised by our agent from daily variances). |
| Lee, Seregina, *Factor graphical lasso* (J. Financial Econometrics 2024) | factors + sparse inverse of the leftovers | ~420 S&P 500 stocks, T = 1260: about 11.1% against about 13.4% for their factor + nonlinear shrinkage, but their nonlinear-shrinkage number is far worse than De Nard et al.'s, so the comparison is doubtful. |
| Fan, Furger, Xiu (JBES 2016) | leftovers correlated only inside GICS sectors | Intraday data. Sector blocks mainly repair a too-strict "leftovers independent" assumption; they do not beat a good full estimator. |
| De Nard, *Blockbuster meets shrinkage* (J. Financial Econometrics 2022) | data-driven blocks, then shrinkage | Beats nonlinear shrinkage only for small and medium universes (abstract). |

**Limits.** A shape helps when the window is short compared with the number of stocks, or when it repairs an even stricter model. At N/T ≈ 0.4–0.5, against good eigenvalue cleaning, no static shape has been shown to help. The reason fits our section 2. A factor model estimates the market and factor directions *from the same window*, so it inherits the same tilt. Its only new information is "the leftovers are uncorrelated", and the truth's middle modes say that is false.

### 3.3 Better raw material: robust statistics and more data per day

**The idea.** Fat-tailed days pull the directions around (section 2.3: on shuffled real days the factor directions 4–20 are badly lost). A statistic that gives each day less say, or more information per day, gives better directions from the same window.

| Paper | What changes | Evidence |
|---|---|---|
| Espana, Le Coz, Smerlak, *Kendall correlation coefficients for portfolio optimization* (arXiv 2024) | Kendall's tau: for every pair of days each stock casts a vote (up or down relative to the other day), so no single huge day dominates. Each day is first divided by its cross-sectional size, and each stock by its standard deviation. | ~500 North American stocks, T = 1000, 60-day holds, 2000–2024, long-short. **Swapping only the directions, Kendall for Pearson, with the same cross-validated eigenvalues: −14.5% in the US**, −7.7% in Europe, −3.9% in Asia-Pacific. Against their RIE −15.3%, but their RIE did worse than clipping. Plain Kendall without any cleaning did as well as cleaned Kendall. Risk measured on standardised returns; no significance tests. In simulation the gain comes from fat tails: with Gaussian data Kendall and Pearson directions are equally good. |
| Bandeira, Lodhia, Rigollet (Electron. Commun. Probab. 2017) | noise law for Kendall | For independent stocks, Kendall eigenvalues follow (2/3)·Marchenko–Pastur + 1/3. Gives the noise edge for clipping Kendall; there is no RIE for Kendall yet. |
| Hediger, Näf, Wolf, *R-NL* (IEEE Trans. Signal Processing 2023) | Tyler's estimator: each day scaled to unit length, directions fitted to the scaled days | CRSP, 21-day rebalancing, long-short. N = 500, T = 1260: nonlinear shrinkage 9.35%, Tyler version 9.32%, Tyler on standardised stocks 9.12%. The authors say the gain comes mainly from shrinking the *correlation* rather than the covariance. Our RIE already works on the correlation, so we already have most of that gain. |
| Di Luzio, Morelli, *Dynamic conditional SKEPTIC* (arXiv 2025) | Kendall and Spearman as a dynamic correlation target | 429 S&P 500 stocks, 2013–2025: DCC-NL 15.70%, Kendall version 15.81%, Spearman 17.13%. No risk gain; much lower turnover. |
| De Nard, Engle, Ledoit, Wolf, *Enhancements based on intraday data* (J. Banking & Finance 2022) | daily open, high, low and close prices: a tail-tamed return, sign-like for big moves, rescaled by the day's high-low range | N = 500, T = 1260, 21-day rebalancing: 9.01% → 7.44% (−17%). Baseline is the dynamic DCC-NL. Needs open/high/low prices; `yfinance` provides them, but our pipeline keeps only the adjusted close. |
| Hautsch, Kyj, Malec (J. Applied Econometrics 2015) | intraday realised covariance | 350 S&P stocks, 2007–09: a tie before the crisis, −10.7% during it, against a weak daily baseline (q ≈ 1.4). |
| Lam, NERCOME (Annals of Statistics 2016); Abadir, Distaso, Žikeš (J. Econometrics 2014) | sample splitting | Eigenvalues only, or eigenvectors barely changed. 100 stocks: identical to nonlinear shrinkage (0.264 for all three). |

**Limits.** Kendall is the strongest real-data result in this whole review that changes only the directions, at our size and window. It comes from one paper with a weak RIE baseline and no significance tests, and its gain fades as the window grows (by q ≈ 0.2 in the US). Our q is 0.46, inside the region where it helped.

### 3.4 A model of time: directions that move

**The idea.** On the real market the true directions themselves move. Our stability study found that modes beyond 20 overlap with themselves a month later at about 0.25. A method can let the directions drift, borrow from earlier windows, or account for the drift in some other way.

| Paper | What it does with time | Evidence |
|---|---|---|
| Engle, Ledoit, Wolf, *Large dynamic covariance matrices* (JBES 2019) | DCC-NL: GARCH volatility per stock; the correlation drifts day by day toward recent co-movements, anchored on a nonlinear-shrinkage target | CRSP, T = 1250, 21-day rebalancing, 1986–2015, N = 500: DCC-NL 9.64% against 9.74% for the same model with a *fixed* correlation. **The part that rotates the directions is worth 0.1 points**; the GARCH volatilities and the cleaned target carry the gain. |
| Allez, Bouchaud, *Eigenvector dynamics* (Phys. Rev. E 2012) | Measures how far the top-5 subspace moves between windows, against how far noise alone would move it | S&P 500, 2000–2010: the top subspaces really move, beyond noise. The window that estimates them most stably is about 700 days for the S&P 500. Single eigenvectors "hybridise" when two eigenvalues come close, so subspaces should be compared, not single vectors. In sell-offs the market mode tilts toward "all stocks equal" (Nikkei). No tracking method and no portfolio test. |
| Bun, Bouchaud, Potters, *Overlaps between eigenvectors of correlated random matrices* (Phys. Rev. E 2018) | The overlap two windows *should* have if nothing changed, computed from their spectra | 300 S&P stocks: bulk matches the theory; the top eigenvectors overlap less than noise allows, the market mode least. Overlapping rolling windows, our case, are "the subject of ongoing investigations". |
| Bongiorno, Challet, Loeper, *Average Oracle* (J. Stat. Mech. 2023), and Bongiorno, Challet (Quant. Finance 2024) | Keeps the recent window's directions; the eigenvalue of each rank is the historical average of how much variance that rank's direction *actually had next period*. Real rotation between windows is folded into the eigenvalues. | n = 100, T = 1200, 20-day rebalancing: Average Oracle on 240 days 10.9%, nonlinear shrinkage on 1200 days 11.5%, DCC + shrinkage 11.0%. Changes eigenvalues, not directions. |
| Bongiorno, Manolakis, Mantegna, end-to-end network (J. Finance & Data Science 2026) | Learns eigenvalues, volatilities and how much to weight old days | Keeps the sample directions; the authors say it "cannot … denoise the time-varying information carried by eigenvector dynamics". n = 300 long-short: 10.9% against 11.1% for nonlinear shrinkage, with 2.5× the turnover. |
| Trucíos et al. (JBES 2023); Zumbach (2009) | dynamic factor model; empirical study | Long-only only, no shrinkage benchmark. Zumbach: "most of the dynamics occur in the eigenvectors, including deep in the spectrum" (abstract). |

**Limits.** No paper smooths, aligns or averages eigenvectors across rolling windows and reports minimum-variance results. That is a genuine gap. The time models that exist move directions only through a two-number recursion (DCC), and that part is worth very little.

### 3.5 Skipping the matrix: regularise the portfolio

**The idea.** Leave the matrix alone and stop the optimiser from betting on noisy directions: cap the weights, take only a few steps from equal weights toward minimum variance, or average the portfolios of many random sub-universes.

| Paper | Method | Evidence |
|---|---|---|
| DeMiguel, Garlappi, Nogales, Uppal (Management Science 2009) | Cap the weight norm; or "partial minimum variance": a few conjugate-gradient steps from 1/N, which never reaches the noisiest directions | 500 CRSP stocks, monthly data, 120-month window (workshop slides): partial 8.8%, 2-norm cap 8.9%, Ledoit–Wolf 8.9%, unconstrained 11.2%. A 2-norm cap is equivalent to shrinking toward the identity. |
| Fan, Zhang, Yu (JASA 2012) | Gross-exposure cap | With a cap, only the largest entry-wise error matters, not the eigenvectors. 400 Russell stocks: 8.62% at a cap of 2.5 against 14.40% unconstrained. |
| Shen, Wang, *Portfolio selection via subset resampling* (AAAI 2017) | Average the minimum-variance weights of many random subsets of n^0.7 stocks | 434 S&P stocks, 500-day window, daily rebalancing: 6.98% against 7.78% for Ledoit–Wolf and 13.18% for the sample. Never compared with nonlinear shrinkage or an RIE. |
| Carrasco, Noumon (working paper 2010) | Spectral filters on the inverse (ridge, cut-off, Landweber) | Not a minimum-variance study (Sharpe ratios on portfolio data). |
| Wolf (2007) | Michaud resampling | Loses to Ledoit–Wolf shrinkage; adds nothing on top of it. |

**Limits.** None of these has been benchmarked against nonlinear shrinkage or an RIE on a large universe. They are substitutes for cleaning, not additions to it. The long-only evidence in `why_the_gap_persists.md` says the same.

## 4. How the families compare on real data, at or near our size

| What changes the directions | Real-data gain against good eigenvalue cleaning | Our N/T? | Confidence |
|---|---|---|---|
| Kendall's tau instead of Pearson | −14.5% (US), −7.7% (Europe), −3.9% (Asia); **ours: +2.5% (section 7.3)** | yes, q = 0.5 | one paper, weak RIE baseline, no significance test; did not transfer to our panel |
| Daily open/high/low/close, tail-tamed returns | −17% against DCC-NL | N = 500, T = 1260 | one paper; needs data we do not keep |
| Tyler's estimator on standardised stocks | −2.5% at N = 500, T = 1260; mostly from cleaning the correlation, which we already do | yes | one paper, significant |
| One factor + dynamic correlation | −1.8% against DCC-NL, −6% against static NL | yes | one group |
| Dynamic correlation (DCC) | −1% against a fixed correlation with the same volatilities | yes | one group |
| Static factor prior | 0% (8.63% = 8.63%) | yes | one group |
| Sample splitting | 0% | n = 100 | one paper |
| Trees (BAHC) | below 1% at their best window; only short windows | no | one group |
| Sparse inverse | worse than linear shrinkage | yes | one paper, one year |
| James–Stein anchors | large in simulation; never tested on real returns | theory exists since 2026 | simulation only |
| Subset resampling | −10% against Ledoit–Wolf | close (T = 500) | one paper; never against an RIE |

## 5. The design space for a method of your own

Every method above is a choice on five levers. Writing your idea down as five answers is the quickest way to see whether it is new.

1. **Which directions to fix.** The market (worth up to 13–18% of the gap on our data), the next 20 (up to 38–45% together with the market), the middle 20–200 (the rest), or the quiet end (little on its own).
2. **What outside information.** An anchor (ones vector, sectors, beta-ranked groups, last window's directions), a shape (factors, blocks, tree, sparse inverse), better raw material (ranks, unit-length days, open/high/low prices), or time (earlier windows, drift).
3. **How strongly to apply it.** From theory (James–Stein's c from the noise level), by cross-validation (Kendall paper, isotonic regression), from history (Average Oracle, Barra's simulation), or by hand.
4. **In which space.** Covariance, correlation (cleaning the correlation is where R-NL's gain came from, and our RIE already does it), the inverse (hedging regressions), or the weights.
5. **Vectors or subspaces.** The rotation *inside* a group of similar-sized factors cannot be estimated (Gurdogan, Shkolnik 2024; Allez, Bouchaud 2012), but the subspace can. Our test agrees: knowing the true top-20 subspace is worth almost as much as knowing the true top-20 matrix.

One rule comes out of the comparison. **Only information the window does not already contain, or the window's information used better, can move a direction.** A factor model fitted on the same window inherits the same tilt, which is why the static factor prior gave exactly 0%. Sector labels, earlier windows, intraday ranges and a prior on beta dispersion are new information. Ranks and Tyler use the same data better when tails are fat.

## 6. Combinations nobody has tested

These come from the papers' own "open" lists and from where our section 2 puts the error. None has a published minimum-variance test on real returns.

- **James–Stein or MAPS on the top directions, the RIE on the rest.** Every anchor paper uses a flat noise floor instead of an eigenvalue cleaner, and none uses real returns. Theory for fixed N/T appeared in 2026 (Seong, Hong, Jung). On our data the dispersion part of the market tilt is small at T = 1000, so the anchor would have to be richer than the ones vector: sectors, beta-ranked groups, or last window's subspace.
- **Kendall directions with an RIE built for Kendall.** The Kendall paper picks eigenvalues by cross-validation because no RIE exists for Kendall matrices. Its noise law is known (Bandeira et al. 2017). Our check says fat tails are what ruin directions 4–20, which is exactly where Kendall helps.
- **Fix the subspace, not the vectors.** Estimate the room of the top 10–20 directions as well as possible, for example robustly or by averaging across overlapping windows, and leave the RIE inside and outside it. This is supported by our subspace test, by Allez and Bouchaud's hybridisation, and by the theory that the inner rotation cannot be estimated.
- **Earlier windows as an anchor.** MAPS proves last window's eigenvector helps when betas persist. Allez and Bouchaud put the most stable top subspace at about 700 days. No paper smooths eigenvectors across rolling windows with a minimum-variance test.
- **Spillover control.** Section 2.3 shows the damage comes from big directions spilling variance onto quiet ones through small tilts. A method could aim at the spill itself: estimate how much variance a tilt of the measured size would leak, and remove that from the quiet directions. This is our own idea, not from a paper.
- **Weight-side averaging on top of the RIE.** Subset resampling beat Ledoit–Wolf by 10% but was never stacked on an RIE.

## 7. Test: James–Stein on the market direction, on top of the RIE (2026-11-05)

`james_stein_rie_covariance` in `optimizer.py` (registered as `rie_js`; defaults unchanged) cleans the correlation with the RIE, turns the market eigenvector into loadings b = √λ₁·h, shrinks each loading's distance from their average by c = 1 − s²/V̂ with noise s² = ℓ̄/(T − 1) (ℓ̄ = average of the other eigenvalues), and rebuilds the matrix with the other modes made perpendicular to the new direction. Study: `examples/sp500_james_stein_study.py`, outputs in `output/sp500_james_stein/`.

Realised volatility (median honesty in brackets):

| World | Window | RIE | RIE + James–Stein | best fixed c | true market direction (ceiling) |
|---|---|---|---|---|---|
| Gaussian days | 250 | 11.12% (1.61) | 11.40% (1.56) | 1 (no shrink) | 10.98% |
| Gaussian days | 500 | 10.35% (1.40) | 10.46% (1.37) | 1 | 10.12% |
| Gaussian days | 1000 | 9.77% (1.27) | 9.81% (1.26) | 1 | 9.58% |
| Real days, shuffled | 250 | 12.51% (2.06) | 12.73% (2.00) | 1 | 12.34% |
| Real days, shuffled | 500 | 11.95% (1.71) | 12.07% (1.68) | 1 | 11.41% |
| Real days, shuffled | 1000 | 11.53% (1.50) | 11.58% (1.48) | 1 | 10.87% |
| Real market, backtest | 250 | 11.59% (1.70) | 11.76% (1.67) | — | — |
| Real market, backtest | 500 | 12.06% (1.44) | 12.15% (1.44) | — | — |
| Real market, backtest | 1000 | 13.30% (1.31) | 13.34% (1.29) | — | — |

(The real-market rows at different windows cover different periods, so compare across a row, not down a column.)

**Result: James–Stein makes the portfolio slightly riskier everywhere (+0.3% to +2.5%) and slightly more honest (honesty down by 0.01–0.06).** The data-chosen c was 0.82–0.85 at T = 250 and 0.95 at T = 1000, as predicted, but no shrinkage at all (c = 1) was best among c = 0.5, 0.8, 0.9 and 1 in every world and window. The market direction's overlap with the truth got slightly *worse* on Gaussian days (0.9949 → 0.9940 at T = 250). Meanwhile, handing the RIE the true market direction would have cut risk by 1–6%, so the tilt is real; James–Stein just corrects the wrong part of it.

**Why it fails here** (Gaussian days, measured against the known truth):

1. **The noise is not spread evenly.** 29% of the market direction's error lies along the true sector directions 2–21, against about 4% if the noise had no favourite direction. The market arrow mostly tilts *toward the sector arrows*. Shrinking toward the average removes spread equally in every direction, so it shaves real beta differences as much as noise.
2. **The noise formula overstates the noise.** At T = 250, ℓ̄/T predicts a noise spread of 0.0026, but the actual spread of the error is 0.0015. So c comes out too small: it shrinks too much.
3. **The dispersion bias is small for the S&P 500 at these windows.** The market eigenvalue is huge relative to the noise (155 against ~0.66), so the market arrow is already 99.5–99.9% right. The published gains come from settings with far fewer days than stocks (n = 24–252 for p = 500–3000), where the tilt is mainly dispersion.

**Lesson for designing a fix.** The market arrow's error is a *leak into the sector arrows*, not extra beta dispersion. A better anchor for the market arrow is the sector subspace: separate market from sectors cleanly, for example with sector labels or with the "augmented" James–Stein of Seong, Hong and Jung, which uses the other big eigenvectors as part of the anchor. That moves the idea from "fix one arrow" to "fix the top subspace", which section 2 already pointed to.

### 7.1 Short windows: 25, 50 and 100 days (2026-11-06)

James–Stein was built for few days and many stocks, and short windows are fresher, so we repeated the test at T = 25, 50 and 100 (fewer days than stocks; the RIE fills the empty directions). In the real market every window is now scored on the same days, from 2018-12-24 on, so "short and fresh" and "long and stale" race over the same period.

Real market, same period, realised volatility (median honesty; turnover per rebalance):

| Window | RIE | RIE + James–Stein | James–Stein c |
|---|---|---|---|
| 25 | 13.08% (5.89; 2.68) | 13.69% (5.06; 3.33) | 0.48 |
| 50 | 13.60% (3.59; 2.51) | 13.84% (3.34; 2.92) | 0.66 |
| 100 | **12.69%** (2.45; 2.31) | 13.03% (2.35; 2.52) | 0.78 |
| 250 | **12.68%** (1.68; 1.86) | 12.87% (1.66; 1.93) | 0.88 |
| 500 | 12.95% (1.46; 1.53) | 13.05% (1.45; 1.56) | 0.94 |
| 1000 | 13.30% (1.31; 1.39) | 13.34% (1.29; 1.40) | 0.96 |

Exact-truth worlds (Gaussian / shuffled), realised volatility:

| Window | RIE | RIE + James–Stein | RIE + true market direction | market overlap, RIE → James–Stein |
|---|---|---|---|---|
| 25 | 14.23% / 14.76% | 15.57% / 15.18% | 16.75% / 17.35% | 0.940 → 0.968 / 0.865 → 0.928 |
| 50 | 13.20% / 13.93% | 14.39% / 14.36% | 14.65% / 15.69% | 0.976 → 0.982 / 0.929 → 0.959 |
| 100 | 12.34% / 13.32% | 13.11% / 13.72% | 12.85% / 14.17% | 0.989 → 0.989 / 0.966 → 0.976 |

What this shows:

1. **Freshness helps the RIE's risk.** On the same days, 100 and 250 days give 12.7% against 13.3% for 1000 days (−4.6%). Even 25 days (13.1%) beats 1000. Staleness is real. The price: the forecast becomes far too optimistic (honesty 1.7 at 250 days, 2.4 at 100, 5.9 at 25) and turnover rises from 1.4 to 1.9–2.7 per rebalance. This is one path with no significance test; differences under about 3% are within the noise seen between simulation draws.
2. **At short windows James–Stein does turn the market arrow toward the truth**, as the papers promise: overlap 0.940 → 0.968 at 25 days on Gaussian days, 0.865 → 0.928 on shuffled real days. Yet the portfolio gets riskier (+3% to +9%).
3. **Even the true market arrow makes short-window portfolios worse** (+18% at 25 days). At short windows the matrix's errors are interlocked: the other directions were estimated around the tilted market arrow, and correcting that one arrow alone breaks the balance between them. A better part does not make a better whole. This is the same lesson as the per-stock volatility forecasts: a more accurate input is not automatically a better portfolio.
4. **James–Stein improves honesty at every window** (5.89 → 5.06 at 25 days, real market), but never risk.

**Lesson.** The gain from freshness is real and comes from the plain RIE at 100–250 days, not from James–Stein. A short-window method needs to fix the whole top subspace consistently, or fix the over-optimistic forecast separately, rather than repair one arrow.

### 7.2 What carries over from one period to the next (2026-11-06)

`examples/sp500_persistence_study.py` (outputs in `output/sp500_persistence/`) cuts history into back-to-back windows that do not overlap and measures how similar each piece of the covariance is in the next window. The same is done on shuffled real days (fat tails, no change) and Gaussian days (noise only). Real similarity below the shuffled one is genuine market change; this is how RMT separates change from noise (Allez, Bouchaud 2012; Bun, Bouchaud, Potters 2018).

Adjacent windows; similarity 1 = identical. The change share is (shuffled − real)/(1 − real): the part of the real drop that noise does not explain.

| Piece | 126 days: real / no change | change share | 250 days | change share | 500 days | change share |
|---|---|---|---|---|---|---|
| Volatilities (rank correlation) | 0.75 / 0.75 | ~0% | 0.75 / 0.84 | 35% | 0.70 / 0.90 | 66% |
| Pairwise correlations | 0.58 / 0.54 | ~0% | 0.64 / 0.68 | 10% | 0.65 / 0.79 | 39% |
| Market arrow | 0.91 / 0.94 | 31% | 0.94 / 0.97 | 47% | 0.95 / 0.98 | 63% |
| Sector arrows, one by one | 0.07 / 0.06 | ~0% | 0.08 / 0.10 | 2% | 0.07 / 0.16 | 9% |
| Sector room (modes 2–20; random 0.04) | 0.27 / 0.30 | 3% | 0.36 / 0.41 | 8% | 0.43 / 0.52 | 16% |
| Middle room (modes 21–100; random 0.18) | 0.22 / 0.22 | 0% | 0.24 / 0.25 | 1% | 0.28 / 0.29 | 3% |

Fade with time (126-day windows, real market; the no-change baseline stays flat): volatilities 0.75 → 0.59 after 2 years → 0.52 after 5; correlations 0.58 → 0.44 → 0.34; market arrow 0.91 → 0.88 → 0.85; sector room 0.27 → 0.25 → 0.23; middle room 0.22 throughout.

What this says:

1. **At half-year windows almost all the difference between periods is noise, not change.** Real and shuffled agree for every piece except the market arrow. Fat tails are a large part of that noise: Gaussian days agree far more (volatilities 0.94 against 0.75).
2. **Real change shows up over longer horizons**: it is two thirds of the drop in volatilities and the market arrow at 500 days. Volatilities and correlations keep fading over five years; they are the fast-moving pieces.
3. **The market arrow is the stable skeleton**: 0.85 similar even five years apart.
4. **The sector room is real, lasting structure**: five years later it is still at 0.23, six times the random level, but even with no change two half-years agree only 0.30. It is hard to *measure*, not quick to change. The individual sector arrows carry almost nothing (0.07): they blur, as section 2 found.
5. **The middle room carries nothing over** beyond the random level even in a world with no change. Nothing can be learned there at these windows, so keeping it flat is right.
6. **Neighbouring periods resemble each other more than random periods do** (real correlations 0.58 against 0.54 at the first lag): calm and stormy spells last a while.

Caveat: few real pairs (22 at 126 days, 10 at 250, 4 at 500), so the 500-day row is rough.

### 7.3 Kendall's tau: what it is, and what it does on our data (2026-11-09)

**What it measures, in plain words.** Pearson's correlation multiplies the two stocks' returns day by day and averages the products, so a day where both moved 10% counts a hundred times more than a day where both moved 1%. Kendall's tau looks instead at every *pair of days* and asks one question: between those two days, did stock A go the same way as stock B? Count the agreements and the disagreements:

> tau = (agreements − disagreements) / number of day pairs.

Every pair of days casts exactly one vote, so a huge day has no more say than a quiet one. With 1000 days there are 499,500 pairs.

*Worked example, real returns.* AAPL against XOM, the five trading days 3–9 January 2024 (returns −0.75, −1.27, −0.40, +2.42, −0.23 % and +0.84, −0.87, +0.30, −1.67, −1.24 %). Of the ten day pairs, two agree and eight disagree, so tau = (2 − 8)/10 = −0.60; Pearson gives −0.58 on the same five days. Now a company-news day: GL fell 53.1% on 11 April 2024 while the market moved −0.06%. Over the 60 days around it, against its closest partner PRU (−1.2% that day), Pearson is +0.22 with that day and +0.11 without it: one day is half the measured correlation. Kendall is +0.26 with and +0.24 without. (Script: `examples/kendall_tiny_example.py`, on the saved panel `output/sp500_marchenko_pastur/prices.csv`.)

**The paper's recipe** (Espana, Le Coz, Smerlak 2024, arXiv 2410.17366; steps and numbers read from the arXiv full text):

1. remove each stock's mean;
2. divide each day by its cross-sectional size √(Σⱼ xⱼₜ²), so a stormy day is shrunk to the size of a calm one before the votes are cast — this does change the Kendall matrix, because it changes how that day compares with other days (scaling one *stock* changes nothing; scaling one *day* does);
3. divide each stock by its standard deviation (irrelevant for Kendall, needed for the Pearson they compare against);
4. Kendall's tau for every pair of stocks: a 456 × 456 matrix with ones on the diagonal. No sine transform — they tried the Gaussian map ρ = sin(πτ/2) and dropped it because it produced negative eigenvalues;
5. eigenvalues either left as they are ("Kendall") or re-estimated by ten-fold cross-validation ("Kendall ICVC": directions from nine folds, the variance along each direction measured on the tenth, averaged over folds, then forced to be decreasing by isotonic regression);
6. covariance Σᵢⱼ = σᵢ σⱼ Ξᵢⱼ, minimum variance with the budget constraint only (long-short), T = 1000 (q = 0.5), 60-day holds, the 500 most liquid stocks of each region, 2000–2024.

Their minimum-variance results (America; annualised, their units): Kendall 33.4 against RIE 39.2 (−14.8%); Kendall ICVC 33.2 against Pearson ICVC 38.9 (−14.5%). Europe −8%, Asia-Pacific −6.5%. Caveats: their σᵢ and their out-of-sample risk are both computed on the day-standardised returns, so the risk is measured in units of "a typical day's size", not in money; no significance test; no turnover; the advantage fades for q below about 0.2 in the US (longer windows).

**Two facts that explain why a plain Kendall matrix can serve as a covariance at all.**

- *The Gaussian dictionary.* If returns were Gaussian, tau = (2/π)·arcsin ρ: ρ = 0.5 → tau 0.33, 0.9 → 0.71, 0.99 → 0.91. The Kendall matrix is therefore a *shrunk* correlation matrix: every entry is pulled toward zero, the large ones least. Shrinkage toward the identity is built in.
- *The noise law.* For independent stocks the Kendall eigenvalues follow (2/3)·Marchenko–Pastur + 1/3 (Bandeira, Lodhia, Rigollet 2017; confirmed in the text). At our q = 0.456 the Pearson noise bulk runs from 0.105 to 2.81, the Kendall bulk from 0.40 to 2.21. The quietest directions arrive lifted about fourfold *before any cleaning*: Kendall starts where the RIE has to work to get.

Neither of the two Marchenko–Pastur-for-Kendall papers (Bandeira et al. 2017; Bousseyroux, Espana, Smerlak 2025) gives a cleaning formula or an RIE for Kendall matrices, only the limit law for independent stocks. That is why the paper falls back on cross-validation for the eigenvalues.

**What we tested** (`examples/sp500_kendall_study.py`, outputs in `output/sp500_kendall/`; 93 decision dates, T = 1000, 21-day holds, one seed per lab world). The matrix follows the paper's steps 1, 2 and 4. As everywhere in our pipeline, the correlation-type matrix is rescaled by the raw window's sample volatilities, and realised risk is measured on raw returns. Estimators: plain Kendall; Kendall without the per-day standardisation; Kendall's directions with the RIE's eigenvalues assigned by rank; and the RIE with its top-k room moved to where Kendall puts it (k = 1 to 100), the Kendall twin of the perfect-room hand-over of section 2. Not run: Kendall ICVC (in the paper it is within 1% of plain Kendall).

Realised volatility (median honesty in brackets):

| World | RIE (Pearson) | Kendall, plain | Kendall, no day standardisation | Kendall directions + RIE sizes | RIE, top-20 room from Kendall | truth |
|---|---|---|---|---|---|---|
| Gaussian days | 9.78% (1.26) | 10.02% (1.31) | 9.80% (1.25) | 9.67% (1.31) | 9.72% (1.30) | 8.30% |
| Real days, shuffled | 11.44% (1.49) | 11.81% (1.65) | 11.72% (1.56) | 11.79% (1.66) | 11.50% (1.56) | 7.87% |
| Real market | 13.30% (1.32) | 13.63% (1.24) | 13.54% (1.23) | 13.66% (1.46) | 13.25% (1.38) | — |

Share of the RIE-to-truth gap closed by moving the RIE's top-k room to Kendall's room (the perfect room from section 2 in brackets): Gaussian days k = 1: +5% (12%), 3: +6% (22%), 10: +5% (29%), 20: +4% (35%), 50: +2% (43%), 100: +1% (50%); shuffled real days k = 1: −1% (19%), 3: −2% (30%), 10: −3% (35%), 20: −2% (42%), 50: −3% (45%), 100: −2% (46%).

How well the directions aim (squared overlap with the truth, mean over dates):

| | market | modes 2–3 | modes 4–7 | modes 8–20 | top-10 room | top-20 room |
|---|---|---|---|---|---|---|
| Gaussian days, Pearson | 1.00 | 0.98 | 0.87–0.90 | 0.08–0.80 | 0.954 | 0.859 |
| Gaussian days, Kendall | 1.00 | 0.97–0.98 | 0.71–0.88 | 0.10–0.77 | 0.943 | 0.848 |
| Shuffled days, Pearson | 0.99 | 0.90–0.94 | 0.25–0.63 | 0.02–0.20 | 0.828 | 0.686 |
| Shuffled days, Kendall | 0.99 | 0.89–0.95 | 0.06–0.76 | 0.02–0.45 | 0.830 | 0.744 |

**What this says, in plain words.**

1. **The published gain did not transfer.** Plain Kendall is 2.5% riskier than the RIE on the real market, 3% riskier on shuffled real days and 2% on Gaussian days. Its forecast is more honest on the real market (1.24 against 1.32): the built-in shrinkage makes it more cautious, not better. Every hybrid sits within ±3% of the gap, which is the noise between draws.
2. **Kendall's directions are not better aimed.** On Gaussian days they are slightly worse: Kendall uses only the order of the days, which for Gaussian data is like having about 9% fewer days (its efficiency relative to Pearson is 9/π² ≈ 0.91). On fat-tailed days the top-20 room is a little better placed (0.74 against 0.69) but the single arrows are not, and the portfolio does not notice. The blur in modes 8–20 comes from near-equal eigenvalues, that is from noise, not from a few big days; re-weighting the same days cannot un-blur them. Section 7.2 said the same from the other side: the single sector arrows carry nothing between periods even in a world with no change.
3. **Why the paper saw 15% and we see nothing, as far as we can tell.** (a) They measure out-of-sample risk on day-standardised returns, every day shrunk to a common size; those are exactly the units in which Kendall counts, one vote per day pair. We measure risk in money, where the big days dominate, and the co-movement *on big days* is what a minimum-variance portfolio has to get right. Pearson weights exactly those days. Kendall measures typical-day co-movement, which is a different quantity. (b) The RIE they beat is the bare Ledoit–Péché formula; ours includes the inverse-Wishart lift, which repairs the quiet end — the part where Kendall's built-in lift ((2/3)·MP + 1/3) would otherwise help. (c) One paper, no significance test, and the gain was already fading as the window grew.
4. **The lesson is the same as for James–Stein and the per-stock volatility forecasts.** A more robust input is not automatically a better portfolio. The optimiser needs the covariance of the days that will actually hurt, and those are the big ones. Big days belong in the directions as well as in the sizes, as long as they are shared moves; what we would want to remove is the single-stock news day, and Kendall removes the weight of *every* big day, shared or not.

Caveats: one seed per lab world; differences under about 3% are within draw-to-draw noise; the Kendall ICVC variant was not run; the paper rescales with the volatilities of day-standardised returns, we with raw sample volatilities; Kendall's top eigenvalue is about 89 in correlation units against roughly 110–140 for Pearson, as the Gaussian dictionary predicts.

### 7.4 Each piece at its own speed: how long a memory? (2026-11-09)

Mohit's idea: take the slow pieces (market arrow, sector room) from a long history and the fast pieces (volatilities, sizes) from recent days. Before building it, `examples/sp500_memory_length.py` (outputs in `output/sp500_memory_length/`) measures the premise: does a longer history predict the *next* period's top room better, or does drift cancel the extra data? For each of 35 decision dates (21 days apart, the last three years, the only stretch with 2000 days of history before it and 250 after), the top-k room of the correlation matrix from the last L days is compared with the room of the next 250 days. The same on shuffled real days, where nothing drifts, gives the noise-only ceiling.

Share of the next 250 days' top room captured (1 = same room; the target is itself noisy, so compare across L, not with 1):

| Room | 250 d | 500 d | 1000 d | 1500 d | 2000 d | random |
|---|---|---|---|---|---|---|
| market, real | 0.923 | 0.930 | 0.925 | 0.915 | 0.916 | 0.002 |
| market, shuffled | 0.962 | 0.973 | 0.979 | 0.982 | 0.983 | |
| top 3, real | 0.646 | 0.661 | 0.641 | 0.637 | 0.638 | 0.007 |
| top 3, shuffled | 0.764 | 0.826 | 0.863 | 0.875 | 0.879 | |
| top 10, real | 0.532 | 0.566 | 0.554 | 0.540 | 0.554 | 0.022 |
| top 10, shuffled | 0.550 | 0.598 | 0.637 | 0.663 | 0.675 | |
| top 20, real | 0.402 | 0.450 | 0.447 | 0.442 | 0.455 | 0.044 |
| top 20, shuffled | 0.439 | 0.489 | 0.524 | 0.540 | 0.548 | |

Sizes: variance (correlation units) along the top directions, summed per group, in the recent 1000 days and in the next 250 days, with the directions taken either from the recent window itself (fitted to those days) or from a 2000-day history (not fitted to them):

| Directions from | market (1) | sector room (2–20) | quietest 50 |
|---|---|---|---|
| recent 1000 days (fitted to the window), real | 171.5 → 108.6 (−37%) | 118.6 → 106.2 (−10%) | 1.7 → 13.9 (×8) |
| 2000-day history (*contains* the window), real | 170.4 → 107.8 (−37%) | 112.6 → 106.9 (−5%) | 3.2 → 11.8 (×3.7) |
| the 1000 days *before* the window (disjoint), real | 165.4 → 105.3 (−36%) | 87.7 → 94.0 (+7%) | 12.0 → 16.2 (+35%) |
| recent 1000 days, shuffled | 143.6 → 148.8 (+4%) | 124.8 → 99.1 (−21%) | 2.0 → 12.3 (×6) |
| 2000-day history (contains the window), shuffled | 143.2 → 149.3 (+4%) | 119.3 → 101.9 (−15%) | 3.8 → 9.6 (×2.5) |
| the 1000 days before the window, shuffled | 142.1 → 149.1 (+5%) | 104.2 → 98.8 (−5%) | 11.9 → 11.9 (0%) |

*Correction (same day).* The first version of this section used only the first two rows and concluded that older directions find "genuinely quieter" quiet directions and give honest sizes. That was contamination: the 2000-day history contains the 1000-day window, so its directions were half fitted to the sizing days. The disjoint rows were added after the two-speed test (section 7.5) exposed the problem. The conclusions below are the corrected ones.

What this says:

1. **For the top room, memory beyond about two years buys nothing.** On shuffled days the room forecast keeps improving with history (top 20: 0.44 → 0.55); on the real market it is flat from 500 days on (0.45 at 500, 0.455 at 2000). The gain from more data and the loss from drift cancel almost exactly. The skeleton is stable enough to survive, not stable enough to profit from a longer look.
2. **Sizes measured along directions that were not fitted to those days are honest, apart from drift.** With disjoint directions the realised-to-seen ratio is 1.0–1.4 at every rank on the real market (1.22 at rank 200, 1.34 at 300, 1.37 at 400) and 1.0 on shuffled days. The window's own quiet directions, by contrast, look eight times quieter than they turn out (1.7 → 13.9 over the quietest 50): the classic over-fit. So the cross-validation-in-time idea works as arithmetic.
3. **But older directions are worse directions.** The quietest 50 directions of the days *before* the window realise 16.2 next period, against 13.9 for the window's own quietest, mirage and all; the old sector room captures 88 units of the recent co-movement against 119 for the window's own. The quiet end drifts (and is partly noise, as section 7.2 found), so "quiet five years ago" is not "quiet next month". The earlier claim that a longer history finds quieter directions came from the contaminated rows.
4. **Market strength is weather.** The market share of co-movement fell by 37% from the recent window to the next period in this stretch (post-2022 calm), whichever directions are used. The size of the market mode has to come from recent days.

Direction by direction (`sizes_by_rank.png`): the size seen inside the window is honest (realised ÷ seen ≈ 1, weather aside) down to about rank 100 and becomes a mirage below rank 150: with the window's own directions the ratio is 1.7 at rank 200, 3.0 at 300 and 6.2 at 400; with the contaminated 2000-day directions 1.5, 2.4 and 3.5; with directions from the days before the window 1.2, 1.3 and 1.4, which is drift, not over-fit. Shuffled days show the same pattern for the fitted directions and a flat 1.0 for the disjoint ones. (The very last ranks are real: near-duplicate share classes such as GOOG/GOOGL make directions with variance 0.005 in any period.)

Weather (`weather.png`, 126-day blocks of the real market): the market's share of co-movement swings between 79 and 284 (median 148) and tracks the market's own volatility (correlation 0.88 across blocks): stormy half-years are market-driven half-years. The sector room's summed share moves the other way (117–210), because the shares add up to 456. Direction 2 alone runs 22–56. The 2026 blocks are the most sector-driven of the decade (market 84, sector room 210).

*The paragraph and the options below are the design as it stood before the test, kept as the record of the decision; section 7.5 has the result and the corrected lesson.* The idea seemed to survive with the roles changed: a long history for the *quiet* directions as much as for the skeleton, the recent window for the *sizes*, measured along the old directions, and for the volatilities. Design options to test, in order of simplicity:

- **A. Directions old, sizes recent.** Directions from the last L_dir days (2000 or all history); eigenvalue of each direction = variance of the last L_size days (250 or 500) along it, forced to be decreasing in the old rank (isotonic regression, as in ICVC) to tame the noise; volatilities from the recent window or the market-wide factor. No RIE needed: the sizes are out-of-sample.
- **B. Long RIE, recent volatilities.** The RIE on 2000 days for the whole correlation structure (directions and relative sizes), volatilities from the last 250–500 days. The simplest two-speed version; the window sweep already says plain 2000-day flat windows lose on real data, so the question is whether fresh volatilities alone rescue them.
- **C. Room swap.** The recent RIE with its top-k room moved to the 2000-day room. The table above predicts this does nothing, so it is the control.

Judge all three on the real market over the 93 holds against the RIE's 13.30% (and 12.7% at 250 days), on honesty and turnover; and in the Gaussian lab, where nothing drifts, to check that the machinery at least beats the 1000-day RIE when more data can only help.

### 7.5 Test: design A, directions from the past and sizes from the present (2026-11-09)

`examples/sp500_two_speed_study.py` (outputs in `output/sp500_two_speed/`). Recipe: directions = eigenvectors of the correlation matrix of a long history; size of each direction = variance of the last L days along it (correlation units), forced to be decreasing in the old rank by isotonic regression; correlation rebuilt and renormalised to ones on the diagonal; scaled by the sample volatilities of the same L days. No random-matrix formula. Two versions of "long history": every day before the date (which *contains* the sizing window, so the sizes are partly in-sample) and every day before the sizing window (disjoint, truly out of sample). Two controls swap only the volatilities between the 1000-day and 250-day RIEs. 93 holds of 21 days, December 2018 to October 2026; the Gaussian lab has the same dates with a fixed truth.

Realised volatility (median honesty; turnover per rebalance):

| Estimator | Real market | Gaussian lab (truth 8.30%) |
|---|---|---|
| RIE, 1000 days | **13.30%** (1.32; 1.34) | 9.78% (1.26) |
| RIE, 250 days | 13.26% (1.69; 1.87) | 11.15% (1.62) |
| control: RIE 1000-day correlation, 250-day vols | 13.79% (1.83; 1.80) | 10.04% (1.35) |
| control: RIE 250-day correlation, 1000-day vols | 14.05% (1.81; 1.66) | 11.00% (1.55) |
| A, directions all history (contains window), sizes 1000 d | 13.63% (1.62; 1.09) | 9.79% (1.27) |
| A, directions all history (contains window), sizes 500 d | 14.01% (1.83; 1.28) | 9.88% (1.32) |
| A, directions all history (contains window), sizes 250 d | 14.01% (2.13; 1.71) | 10.08% (1.40) |
| A, same without isotonic smoothing | 13.98% (2.15; 1.84) | 10.09% (1.41) |
| A, directions from the days before the window, sizes 500 d | 16.61% (2.24; 0.65) | **9.53%** (1.18) |
| A, directions from the days before the window, sizes 250 d | 15.88% (2.22; 1.00) | 9.68% (1.24) |

On the 45 dates with 2000 days of history the picture is the same: real market RIE 1000 d 10.24%, RIE 250 d 9.38%, A (2000-day directions, sizes 250 d) 10.77%, A disjoint 11.61%; Gaussian lab RIE 9.73%, A disjoint 9.51–9.61%.

What this says:

1. **Design A fails on the real market and works in the lab.** With truly out-of-sample sizes it is 19–25% riskier than the RIE on the real market, with a forecast that is half the realised risk (honesty 2.2); in the Gaussian lab the same construction is the best estimator we have run, 2.5% below the RIE and more honest (1.18), closing 16% of the gap. The machinery is right; the world drifts.
2. **The cause is the market arrow, not the quiet end.** At the last real window the old market direction carries 60 units of the last 250 days' co-movement, where the window's own direction carries 117; the missing variance is smeared over ranks 5–10 (14.6 and 6.6 against the RIE's 9.5 and 3.8). A tilted description of a 100-unit direction leaves the optimiser hedging the wrong market, and that costs more than everything else combined. A's quiet sizes are *higher* than the RIE's (0.33 against 0.18 at rank 400), so the quiet end is not where the optimism comes from.
3. **The tolerance for the market arrow is about 0.998, not 0.9.** Section 7.2 called the market arrow a stable skeleton because it stays 0.85 similar over five years. For a 100-unit direction an overlap of 0.92 (what the days before the window give for the next period) spills 8 units into other directions, forty times a quiet eigenvalue. "Stable" by the standards of eigenvector overlap is nowhere near stable enough for a minimum-variance optimiser. The fresh window's own arrow (0.99 with the next period) is the one thing that must not be replaced by anything older.
4. **Volatilities and correlations must come from the same days.** Both controls are worse than either pure window (+3.7% and +5.6% on the real market, +2.6% and +12.5% in the lab). Fresh volatilities on a stale correlation, or the reverse, give an incoherent matrix: when volatilities are high, correlations are high too, and splitting them breaks that link. This also rules out design B (long RIE, fresh volatilities) before running it, and explains why the market-wide factor was the only standardisation that did not hurt (section 3.3 of the review): it rescales everything by one number and leaves the structure coherent.
5. **Isotonic smoothing is nearly irrelevant** (13.98% against 14.01%): the noise in 456 sizes measured on 250 days is not the problem.
6. **Why the 250-day RIE is good.** Fresh directions, fresh sizes, fresh volatilities, all from the same days, with the RIE repairing the damage of q = 1.8. Its forecast is optimistic (1.69) and it trades more (1.87), but its 13.26% is the best plain window on these holds, and it beats every two-speed mixture.

**Lesson.** The one-sentence version of this idea was "slow pieces from long history, fast pieces from recent days". The test says the pieces cannot be separated along the time axis: the market arrow needs the freshest data because its tilt is so expensive, and volatilities and correlations need the *same* data because they move together. What does carry over honestly across time is a *size along a given direction*, which is useful arithmetic (it is how the lab version wins), but only if the direction itself is fresh. That leaves cross-validation *within* the recent window (directions from some of the days, sizes from the others, as the Kendall paper's ICVC does with random folds) as the honest-sizes route, and the published results put that within a few percent of nonlinear shrinkage.

## 8. Evidence quality

- **Our own check** (section 2) uses two Gaussian draws and two shuffles, 93 windows each. The "hand over the true directions" numbers are ceilings: no estimator knows the true directions. The real market adds drift on top, which these worlds leave out.
- **Every arXiv paper cited here was checked against the arXiv listing**: it exists, with the title and authors given. The journal references were confirmed by the agents on publisher or author pages.
- **The numbers** were read from the full text by one research agent each and not re-checked independently, except where `why_the_gap_persists.md` had already verified them (Ledoit, Wolf 2017; BAHC). Exceptions: Goto and Xu 2015 and the Barra FAJ paper (abstract only, no numbers used); DeMiguel et al. 2009 (workshop slides); Lunde, Shephard, Sheppard (no copy found, not used). Several annualised figures were converted from daily or monthly variances by the agents.
- **Most real-data results come from one research group each**: nonlinear shrinkage and factor models from Ledoit, Wolf and coauthors; BAHC and the Average Oracle from Bongiorno and Challet; anchors from Goldberg, Kercheval, Shkolnik and coauthors. The 2026 preprints (Seong et al.; Bernstein et al.; Gurdogan, Shkolnik) are recent and not yet peer reviewed.
- **Leads that turned out wrong** and were corrected: "Honey, I shrunk" has no minimum-variance test; Barra's eigen-adjustment changes eigenvalues, not eigenvectors; the R-NL gain of 8.23% → 7.04% belongs to the standardised variant at N = 1000, T = 252; Carrasco and Noumon is not a minimum-variance paper; the Kendall paper uses no sine transform.

## 9. Sources

Anchors
- Goldberg, Papanicolaou, Shkolnik (2022), The dispersion bias, *SIAM J. Financial Math.* 13(2), 521–550. https://arxiv.org/abs/1711.05360
- Goldberg, Kercheval (2023), James–Stein for the leading eigenvector, *PNAS* 120(2). https://pmc.ncbi.nlm.nih.gov/articles/PMC9926287/
- Shkolnik (2022), James–Stein estimation of the first principal component, *Stat* 11(1). https://arxiv.org/abs/2109.01975
- Gurdogan, Kercheval (2022), Multiple anchor point shrinkage for the sample covariance matrix, *SIAM J. Financial Math.* 13(3). https://arxiv.org/abs/2109.00148
- Goldberg, Gurdogan, Kercheval (2025), Portfolio optimisation via strategy-specific eigenvector shrinkage, *Finance and Stochastics* 29(3). https://www.math.fsu.edu/~kercheva/papers/strategy_specific_shrinkage.pdf
- Gurdogan, Shkolnik (2024), The quadratic optimization bias of large covariance matrices. https://arxiv.org/abs/2410.03053
- Seong, Hong, Jung (2026), Augmented James–Stein estimation for leading eigenvectors and eigenspaces in high dimensions. https://arxiv.org/abs/2609.34315
- Bernstein, Goldberg, Gunther, Kercheval, Lan (2026), Principal component error in high-dimensional factor models. https://arxiv.org/abs/2609.20550
- Gurdogan, Shkolnik (2026), Recovery limits for eigenvector alignment in HDLSS spiked covariance models. https://arxiv.org/abs/2609.23081
- Monasson, Villamaina (2015), Estimating the principal components of correlation matrices from all their empirical eigenvectors, *EPL* 112. https://arxiv.org/abs/1503.00287
- Menchero, Orr, Wang (2011), Barra US Equity Model (USE4) methodology notes. https://www.top1000funds.com/wp-content/uploads/2011/09/USE4_Methodology_Notes_August_2011.pdf
- Paul (2007), Asymptotics of sample eigenstructure for a large dimensional spiked covariance model, *Statistica Sinica* 17. https://www3.stat.sinica.edu.tw/statistica/oldpdf/A17n418.pdf

Shapes
- De Nard, Ledoit, Wolf (2021), Factor models for portfolio selection in large dimensions: the good, the better and the ugly, *J. Financial Econometrics* 19(2). https://www.econ.uzh.ch/dam/jcr:7576bfe6-1261-465a-ae0e-0197b781f5b8/jfec_2021.pdf
- Ledoit, Wolf (2003), Improved estimation of the covariance matrix of stock returns with an application to portfolio selection, *J. Empirical Finance* 10. https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffff9961f70f/jef.pdf
- Fan, Liao, Mincheva (2013), Large covariance estimation by thresholding principal orthogonal complements, *JRSS B* 75(4). https://arxiv.org/abs/1201.0175
- Avellaneda, Serur (2020), Hierarchical PCA and modeling asset correlations. https://arxiv.org/abs/2010.04140
- Tumminello, Lillo, Mantegna (2007), Hierarchically nested factor model from multivariate data, *EPL* 78. https://arxiv.org/abs/cond-mat/0511726
- Tumminello, Lillo, Mantegna (2007), Shrinkage and spectral filtering of correlation matrices: a comparison via the Kullback–Leibler distance, *Acta Physica Polonica B* 38. https://arxiv.org/abs/0710.0576
- Pantaleo, Tumminello, Lillo, Mantegna (2011), When do improved covariance matrix estimators enhance portfolio optimization?, *Quantitative Finance* 11(7). https://arxiv.org/abs/1004.4272
- Callot, Caner, Önder, Ulaşan (2021), A nodewise regression approach to estimating large portfolios, *JBES* 39(2). https://arxiv.org/abs/1611.07347
- Lee, Seregina (2024), Optimal portfolio using factor graphical lasso, *J. Financial Econometrics* 22(3). https://arxiv.org/abs/2011.00435
- Fan, Furger, Xiu (2016), Incorporating Global Industrial Classification Standard into portfolio allocation, *JBES* 34(4). https://dachxiu.chicagobooth.edu/download/GICS.pdf
- Stevens (1998), On the inverse of the covariance matrix in portfolio analysis, *J. Finance* 53(5). https://www.federalreserve.gov/econres/ifdp/on-the-inverse-of-the-covariance-matrix-in-portfolio-analysis.htm
- Goto, Xu (2015), Improving mean variance optimization through sparse hedging restrictions, *JFQA* 50(6) (abstract only).
- De Nard (2022), Oops! I shrunk the sample covariance matrix again: Blockbuster meets shrinkage, *J. Financial Econometrics* 20(4) (abstract only).

Better raw material
- Espana, Le Coz, Smerlak (2024), Kendall correlation coefficients for portfolio optimization. https://arxiv.org/abs/2410.17366
- Bandeira, Lodhia, Rigollet (2017), Marčenko–Pastur law for Kendall's tau, *Electron. Commun. Probab.* 22. https://arxiv.org/abs/1611.04505
- Bousseyroux, Espana, Smerlak (2025), Another Marčenko–Pastur law for Kendall's tau. https://arxiv.org/abs/2503.18645
- Hediger, Näf, Wolf (2023), R-NL: covariance matrix estimation for elliptical distributions based on nonlinear shrinkage, *IEEE Trans. Signal Processing* 71. https://arxiv.org/abs/2210.14854
- Frahm, Jaekel (2005), Random matrix theory and robust covariance matrix estimation for financial data. https://arxiv.org/abs/physics/0503007
- Yang, Couillet, McKay (2015), A robust statistics approach to minimum variance portfolio optimization, *IEEE Trans. Signal Processing* 63(24). https://arxiv.org/abs/1503.08013
- Di Luzio, Morelli (2025), Dynamic conditional SKEPTIC. https://arxiv.org/abs/2512.11648
- De Nard, Engle, Ledoit, Wolf (2022), Large dynamic covariance matrices: enhancements based on intraday data, *J. Banking & Finance* 138.
- Hautsch, Kyj, Malec (2015), Do high-frequency data improve high-dimensional portfolio allocations?, *J. Applied Econometrics* 30(2).
- Lam (2016), Nonparametric eigenvalue-regularized precision or covariance matrix estimator, *Annals of Statistics* 44(3).
- Abadir, Distaso, Žikeš (2014), Design-free estimation of variance matrices, *J. Econometrics* 181(2).

Time
- Engle, Ledoit, Wolf (2019), Large dynamic covariance matrices, *JBES* 37(2). https://www.econ.uzh.ch/apps/workingpapers/wp/econwp231.pdf
- Allez, Bouchaud (2012), Eigenvector dynamics: general theory and some applications, *Phys. Rev. E* 86. https://arxiv.org/abs/1203.6228
- Bun, Bouchaud, Potters (2018), Overlaps between eigenvectors of correlated random matrices, *Phys. Rev. E* 98. https://arxiv.org/abs/1603.04364
- Bongiorno, Challet, Loeper (2023), Cleaning the covariance matrix of strongly nonstationary systems with time-independent eigenvalues, *J. Stat. Mech.* https://arxiv.org/abs/2111.13109
- Bongiorno, Challet (2024), Covariance matrix filtering and portfolio optimisation: the Average Oracle vs non-linear shrinkage and all the variants of DCC-NLS, *Quantitative Finance* 24(9). https://arxiv.org/abs/2309.17219
- Bongiorno, Manolakis, Mantegna (2026), End-to-end large portfolio optimization for variance minimization with neural networks through covariance cleaning, *J. Finance and Data Science* 12. https://arxiv.org/abs/2507.01918
- Zumbach (2009), The empirical properties of large covariance matrices. https://arxiv.org/abs/0903.1525
- Trucíos, Mazzeu, Hallin, Hotta, Valls Pereira, Zevallos (2023), *JBES* 41(1).

Portfolio side
- DeMiguel, Garlappi, Nogales, Uppal (2009), A generalized approach to portfolio optimization: improving performance by constraining portfolio norms, *Management Science* 55(5).
- Fan, Zhang, Yu (2012), Vast portfolio selection with gross-exposure constraints, *JASA* 107. https://arxiv.org/abs/0812.2604
- Shen, Wang (2017), Portfolio selection via subset resampling, *AAAI-17*, 1517–1523.
- Carrasco, Noumon (2010), Optimal portfolio selection using regularization, working paper. https://cireqmontreal.com/wp-content/uploads/2012/04/carrasco.pdf
- Wolf (2007), Resampling vs. shrinkage for benchmarked managers, *Wilmott Magazine*.
