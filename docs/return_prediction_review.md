# Predicting returns and using the prediction: a literature review for beginners

Written 2026-11-12. This is the reading behind the question "our risk side
works; how do people add *which stocks will go up*?". It covers about fifty
papers in five groups. Every number was read from the paper itself during
this review unless the entry says otherwise (section 9 lists the
exceptions), and every journal reference was checked against the Crossref
database.

How to read it: section 1 is a word list, section 2 is the whole story in
one page, sections 3 to 7 are the five groups of papers, section 8 is what
it means for this project. Each paper gets the same four lines: the
**idea**, **how it works**, the **result**, and the **catch**.

## 1. Word list

| Word | Plain meaning |
|---|---|
| Return | How much a stock's price changed over a period, as a percentage |
| Volatility | How much a stock's daily returns bounce around; our measure of risk |
| Weights | The share of your money in each stock |
| Long / short | Long: you own the stock and gain if it rises. Short: you borrow and sell it, and gain if it falls |
| Signal (also predictor, characteristic) | One number per stock that is supposed to say which stocks will do better, for example "return over the past year" |
| Decile, quintile | Sort all stocks by a signal and cut the list into 10 (or 5) equal groups |
| Spread, or long-short portfolio | Own the top group, short the bottom group. Its return is "top minus bottom", so the market's own move cancels out |
| Equal-weighted / value-weighted | Equal: the same money in every stock. Value: money in proportion to company size. Equal weighting gives tiny companies a big say |
| Microcaps | The tiniest listed companies. About 60% of all names but only 3% of the market's value, and expensive to trade |
| Sharpe ratio | Return earned per unit of risk: yearly return above cash, divided by yearly volatility. The US stock market's long-run figure is about 0.4. Above 1 is very good. Above 2 *before costs* usually means tiny stocks or very heavy trading |
| Information ratio | The same idea, used by some papers for "return divided by volatility" of a strategy |
| Alpha | The part of a return that is left after removing what the market's own move explains |
| t-statistic | How many error bars a result sits away from zero. 2 is the usual bar; the papers in section 5 argue for 3 |
| In-sample / out-of-sample | In-sample: tested on the same data used to build the rule (flattering). Out-of-sample: tested on later data the rule never saw (honest) |
| Turnover | The share of the portfolio that is replaced each period. 100% a month means everything is sold and rebought monthly |
| Basis point (bp) | One hundredth of a percent. 10 bp = 0.10% |
| Shrinkage | Pulling a noisy estimate toward a safe default, in proportion to how noisy it is |
| Survivorship bias | Testing on companies that still exist today, which silently leaves out the ones that failed |
| Covariance matrix, risk model | The table of how every pair of stocks moves together. This is what our project has built and cleaned |
| Mean-variance optimisation (Markowitz) | The textbook recipe: given a forecast return for every stock and the covariance matrix, find the weights with the best return for a given risk |
| Minimum variance | The same recipe with *no* forecast: find the calmest mix. This is our current portfolio |
| 1/N | Split the money equally across all N assets. The simplest benchmark |
| CRSP | The standard academic database of US stock prices. It includes companies that later died, which is why academics do not have our survivorship problem |

## 2. The story in one page

```
 prices ──► SIGNAL ──► FORECAST ──► OPTIMISER ──► PORTFOLIO ──► after COSTS
            a score     a shrunk     forecast +      weights       what you
            per stock   version of   risk model                    really earn
                        the score
            section 4   section 3    section 6                     section 5
            section 7
```

1. **Forecasting returns is the hard half** (section 3). A stock's risk can
   be measured well from a year of daily prices. Its average return cannot
   be measured well even from decades. A textbook optimiser fed noisy
   return forecasts does worse than splitting the money equally. Every
   repair is the same move: start from a safe portfolio that needs no
   forecast, and lean toward the forecast only a little.
2. **A handful of simple price-based rules have worked historically**
   (section 4): buy last year's winners (momentum), avoid the wildest
   stocks (low risk), and the reverse rules at very short and very long
   horizons.
3. **Most published rules shrink when checked** (section 5): by about a
   quarter on fresh data, by more than half after publication, and often to
   nothing once trading costs and tiny stocks are accounted for. Momentum
   is the main price-based survivor. This section also holds the free
   "answer key" that shows how far off our own first test was.
4. **There is a published recipe for adding a signal to a cleaned risk
   model like ours** (section 6). With momentum and 500 large US stocks it
   raised the Sharpe ratio from about 0.7 to about 0.9, and the gain
   disappears at realistic trading costs unless turnover is controlled.
5. **Machine learning finds more, mostly where it cannot be traded**
   (section 7). The headline numbers come from equal-weighted portfolios of
   tiny stocks that are replaced every month. Among large stocks, after
   costs, little is left.

## 3. Group A: why forecasting returns is hard, and how to tone a forecast down

**Merton (1980), "On estimating the expected return on the market".**
- Idea: sampling more often sharpens your estimate of risk, not of return.
- How it works: a mathematical argument. The precision of an average return
  depends only on how many *years* you have. The precision of a volatility
  depends on how many *observations* you have, so daily data helps.
- Result: over the same period, volatility estimated from yearly data is
  about nineteen times less precise than from daily data. Even with 52
  years of US market data, reasonable methods disagree about the market's
  expected return by 2.2 points a year. Raw sample averages are "almost
  useless".
- Catch: an exploratory paper about the whole market, not single stocks.

**Chopra and Ziemba (1993), "The effect of errors in means, variances, and
covariances on optimal portfolio choice".**
- Idea: which mistake hurts an optimiser most?
- How it works: take ten stocks, treat their history as the truth, add
  random errors to the returns, then to the risks, then to the
  co-movements, and see how much each kind of error damages the optimised
  portfolio.
- Result: errors in returns cost about 3, 11 and 21 times as much as
  errors in risks, for cautious, medium and bold investors. Against errors
  in co-movements the ratios are 5, 22 and 57.
- Catch: ten stocks, one decade, an in-sample exercise. Their advice
  matters for us: without good forecasts "it may be preferable to set all
  forecasts equal", which is exactly minimum variance.

**Michaud (1989), "The Markowitz optimization enigma: is 'optimized'
optimal?".**
- Idea: an optimiser is an "estimation-error maximiser".
- How it works: an essay. The optimiser puts the most money in stocks with
  the highest forecast return and lowest forecast risk, and those are the
  stocks whose forecasts are most likely wrong.
- Result (quoting a Jobson and Korkie experiment with 20 stocks and 5
  years of data): the "optimal" portfolio thought its own Sharpe ratio was
  0.29. Its true value was 0.08. The equal split scored 0.27 and the
  truly optimal portfolio 0.34.
- Catch: no shorting limits in that experiment. Michaud also notes that
  when return inputs are set to zero "there is no error maximization
  resulting from errors in the return estimates".

**Best and Grauer (1991), "On the sensitivity of mean-variance-efficient
portfolios to changes in asset means".** Abstract only. A "surprisingly
small" increase in one asset's forecast drives half the securities out of
the portfolio.

**DeMiguel, Garlappi and Uppal (2009), "Optimal versus naive
diversification: how inefficient is the 1/N portfolio strategy?".**
- Idea: do fourteen clever portfolio models beat splitting the money
  equally?
- How it works: seven real datasets of 3 to 24 assets, monthly data. Each
  model is re-estimated every month on the past 10 years and held for one
  month, so the test is out-of-sample.
- Result: none beats the equal split consistently. On the first dataset
  the equal split has a monthly Sharpe ratio of 0.19 and the textbook
  optimiser 0.08; in-sample the same optimiser showed 0.38. The textbook
  optimiser would need about 3,000 months (250 years) of data to beat the
  equal split with 25 assets, and about 6,000 months with 50. The models
  that ignore return forecasts, minimum variance among them, are the ones
  that most often match or beat the equal split.
- Catch: the assets are baskets of stocks, not single stocks, and there
  are few of them. The authors' own conclusion: "it may be best to ignore
  the data on expected returns, but still exploit the correlation
  structure".

**Jorion (1986), "Bayes-Stein estimation for portfolio analysis".**
- Idea: do not trust each stock's own average; pull all of them toward a
  common value.
- How it works: forecast = (1 − v) × the stock's own past average + v ×
  the average return of the minimum-variance portfolio. The pull v is
  large when the stocks' averages are statistically hard to tell apart.
- Result: Jorion's own tables could not be read (paywalled). In Kan and
  Zhou's independent test with 10 assets and 10 years of data, the plain
  optimiser loses 1.5% a month in usefulness to the investor, Jorion's
  version loses 0.2%.
- Catch: on real data it barely improves on the plain optimiser
  (DeMiguel et al.).

**Black and Litterman (1992), "Global portfolio optimization".**
- Idea: start from the forecasts that would make the *market's own*
  portfolio the best one, then adjust only where you have a view.
- How it works: (1) work backwards from market weights to the returns
  that justify them; (2) state views such as "A will beat B by 2%", each
  with a confidence; (3) blend the two with a formula that weights each by
  its confidence; (4) run the usual optimiser. The result is the market
  portfolio plus a tilt for each view.
- Result: with raw historical averages the optimiser demanded positions
  like −96% in Canadian bonds and +55% in US bonds. With the blended
  forecasts the positions are moderate. The backtest is shown only as
  charts.
- Catch: needs every company's market value, which prices and volumes do
  not give us.

**Kan and Zhou (2007), "Optimal portfolio choice with parameter
uncertainty".**
- Idea: when forecasts are noisy, the best portfolio is a mix of three
  funds: cash, the minimum-variance portfolio, and the textbook optimal
  portfolio.
- How it works: a formula for the mix. The more stocks you have relative
  to months of data, the more weight goes to minimum variance.
- Result (simulation, 25 assets, 10 years of data, usefulness to the
  investor in % per month): textbook optimiser −6.5, minimum variance
  +0.5, three-fund rule +0.6, perfect knowledge +2.0.
- Catch: a simulation with well-behaved returns. On real data the rule
  beats the equal split on only 2 of 6 datasets (DeMiguel et al.).

**Tu and Zhou (2011), "Markowitz meets Talmud".**
- Idea: mix the equal split with a sophisticated rule.
- How it works: weights = (1 − δ) × equal split + δ × rule. δ is set by
  comparing how wrong the equal split is on average with how noisy the
  rule is.
- Result: the equal split mixed with the three-fund rule beats the equal
  split alone on 5 of 7 real datasets with 10 years of data, and 6 of 7
  with 20 years.
- Catch: the big wins are on datasets that contain ready-made long-short
  portfolios; on plain industry and country baskets the equal split still
  wins.

**What group A says:** every repair is the same move. Pick a safe
portfolio that needs no forecast (equal split, minimum variance, or the
market), and lean toward the forecast by an amount that shrinks as the
forecast gets noisier. Our minimum-variance portfolio is that safe
starting point.

## 4. Group B: simple rules that need only prices

All numbers here are *before* trading costs unless stated.

| Rule | Paper | What you rank on | Result | Catch |
|---|---|---|---|---|
| **Momentum**: last year's winners keep winning | Jegadeesh and Titman (1993) | Return over the past 3 to 12 months | Winners beat losers by 0.95% a month for 6-month ranking and holding (t = 3.1), 1.49% for the best variant. Still 9.3% a year after a 0.5% trading cost | Loses about 7% in an average January; the gain fades after a year and partly reverses |
| The same, tested later | Jegadeesh and Titman (2001) | Same, 6 months | 1.01% a month in 1990–1997, after the first paper was known | Reverses in years 4 and 5 |
| The same, large stocks only | Asness, Moskowitz and Pedersen (2013) | Past year, skipping the last month | 5.4% a year, Sharpe 0.33, among the largest US stocks 1972–2011 | Much weaker than the all-stocks figures above |
| Momentum crashes | Daniel and Moskowitz (2016) | Past year, skipping a month | Long-run Sharpe 0.60, but the worst months are −74% and −61% (1932) and −46% (April 2009) | Crashes come when a beaten-down market rebounds: last year's losers snap back hardest |
| Momentum with a risk brake | Barroso and Santa-Clara (2015) | Same, but hold less when momentum itself has been volatile lately | Sharpe rises from 0.53 to 0.97; worst month improves from −79% to −28% | No costs; numbers from the thesis version of the paper |
| Near the 52-week high | George and Hwang (2004) | Today's price divided by the highest price of the past year | 0.45% a month; 1.23% outside January | Similar to momentum |
| **Short-term reversal**: last week's or month's losers bounce | Lehmann (1990); Jegadeesh (1990) | Return over the past week or month | 1.79% a *week* before costs; 2.49% a month between extreme deciles | Trades the whole book weekly. With a one-day delay and a 0.4% cost the profit is statistically zero |
| **Long-term reversal** | De Bondt and Thaler (1985) | Return over the past 3 to 5 years | Past losers beat past winners by 24.6% over the next 3 years | Mostly a January effect in low-priced stocks |
| **Low risk**: avoid the wildest stocks | Ang, Hodrick, Xing and Zhang (2006) | Last month's stock-specific volatility | The wildest fifth of stocks earned −0.02% a month, the calmest fifth 1.04% | The wildest fifth is only 1.9% of the market's value |
| Low risk via market sensitivity | Frazzini and Pedersen (2014) | Beta: how much a stock moves when the market moves | Long low-beta, short high-beta: 0.70% a month, Sharpe 0.78 | Needs borrowing; a later paper argues the profit leans on tiny stocks |
| Scale the whole portfolio by recent volatility | Moreira and Muir (2017) | Last month's market volatility | Holding less after volatile months adds 4.9% a year of alpha for the market portfolio | A later study of 103 strategies finds the real-time version generally does *not* beat the unmanaged one |
| Trend following | Moskowitz, Ooi and Pedersen (2012) | An asset's own past-year return, up or down | Sharpe above 1 across 58 futures markets | Futures, not single stocks |
| Overnight versus daytime | Lou, Polk and Skouras (2019) | Split each day's return into overnight and daytime parts | All of momentum's profit is earned overnight | Needs opening prices and precise timing; costs would eat much of it |

**What group B says:** the published before-cost gap between top and
bottom groups is about 0.5% to 1.5% a month, mostly from equal-weighted
tests that ended by 2013. Among large stocks the gap is about half. Low
risk is something our minimum-variance portfolio already leans on, so
adding it as a forecast would count it twice. Short-term reversal is best
used as a reason to skip the latest month when measuring momentum.

## 5. Group C: do published rules survive?

**McLean and Pontiff (2016), "Does academic research destroy stock return
predictability?"** 97 published rules. Average top-minus-bottom return:
0.58% a month in the original study period, 0.40% in later data before
publication, 0.26% after publication. That is 26% lower on fresh data and
58% lower once published. Price-and-volume rules started highest and fell
furthest.

**Harvey, Liu and Zhu (2016), "... and the cross-section of expected
returns".** 316 published "factors". When hundreds of rules have been
tried, some look good by luck, so the bar should rise from t = 2 to about
t = 3. By their count between 80 and 158 of 296 published findings are
likely false.

**Hou, Xue and Zhang (2020), "Replicating anomalies".** They rebuilt 452
rules with value-weighted portfolios so tiny stocks cannot dominate. 65%
fail the t = 2 bar. If tiny stocks are allowed to dominate (equal
weights), only 43% fail. Microcaps are 61% of companies and 3.3% of market
value. Short-term reversal shrinks from 1.99% to 0.26% a month. Six-month
momentum survives at 0.82% a month (t = 3.5). (The detailed numbers are
from the 2017 working paper.)

**Jensen, Kelly and Pedersen (2023), "Is there a replication crisis in
finance?"** A more forgiving design (153 rules, 93 countries, alpha
instead of raw return, related rules judged together) finds 82% replicate.
The two papers differ in method more than in data; both agree the effects
are weaker than first published.

**Novy-Marx and Velikov (2016), "A taxonomy of anomalies and their trading
costs".** Value-weighted strategies charged a realistic cost per trade.

| Strategy | Before costs, % a month | Share replaced monthly, each side | Cost, % a month | After costs, % a month |
|---|---|---|---|---|
| Value | 0.47 | 3% | 0.05 | 0.42 |
| Momentum | 1.33 | 35% | 0.65 | 0.68 |
| Momentum, trading only when a stock moves well out of its group | 1.20 | 19% | 0.35 | 0.85 |
| Low stock-specific volatility | 0.63 | 25% | 0.52 | 0.11 |
| Short-term reversal | 0.37 | 91% | 1.65 | −1.28 |

**Frazzini, Israel and Moskowitz (2018), "Trading costs".** Real trades by
one large manager cost far less than academic estimates: about 12 bp per
trade. Momentum costs 3.5% a year at 127% monthly turnover. (Numbers from
the 2012 draft; the 2018 paper was read in abstract only.)

**Shumway (1997), "The delisting bias in CRSP data".** Abstract only. Even
the academic database was missing the final, usually terrible, return of
stocks removed for failure. Later papers plug in −30%.

**Elton, Gruber and Blake (1996), "Survivor bias and mutual fund
performance".** For mutual funds, studying only survivors overstates
returns by 0.4% a year over 10 years and about 1% a year over 20. (Working
paper version.) Our stock universe showed a far larger effect; see below.

**Lo (2002), "The statistics of Sharpe ratios".** The error bar of a
Sharpe ratio is about √((1 + SR²/2) / number of periods). With seven years
of monthly data, a true yearly Sharpe of 0.5 comes with an error bar of
about ±0.38 (our arithmetic from his formula), so seven years cannot tell
0.5 from zero.

**Bailey and López de Prado (2014), "The deflated Sharpe ratio".** If you
try many strategies and report the best, its Sharpe ratio is inflated;
they give a formula to deflate it by the number of tries.

**Chen and Zimmermann (2022), "Open source cross-sectional asset
pricing".** Free code and data for about 200 published rules' portfolio
returns, updated yearly.

### The answer key: our first test against honest data

Kenneth French (Dartmouth) publishes, free, the monthly returns of ten
groups of US stocks sorted by past-year return, past-month return and
recent volatility. They are built on CRSP, so the dead companies are in.
We recomputed the comparison ourselves from his files (data through
September 2026).

| Quantity, 2019 onward | Our pack (survivors only) | French library (dead included) |
|---|---|---|
| All stocks, equal weight, per year, compounded 2019–2025 | 20.3% | 9.0% |
| Last year's biggest losers, per year | 40.5% | 6.3% |
| The most volatile stocks, per year | 48.1% | 5.2% |
| Momentum, winners minus losers | −13% a year | +0.87% a month equal-weighted (t = 1.2); +0.52% value-weighted (t = 0.5) |
| Low volatility, calm minus wild | −39% a year | about +0.3% a month equal-weighted, not significant |

The groups are not cut identically (French uses New York Stock Exchange
cut-offs and 60 days for volatility; we used equal-sized groups and a
year), and the months differ by one. The message survives those
differences: with the dead included, last year's biggest losers and the
wildest stocks were the *worst* places to be, not the best. Our first
test had the sign backwards because the failures were missing. It also
shows the true size of what we are hunting: since 2019 momentum's gap was
positive but not statistically distinguishable from zero.

### Where to get data with the dead included

Prices read from the vendors' own pages on 2026-11-12; check before
buying.

| Source | Dead companies? | History | Cheapest plan that includes them |
|---|---|---|---|
| Kenneth French data library | Yes, but ready-made groups only, not single stocks | From 1926 | Free |
| Sharadar (prices table) | Yes, about 21,000 tickers | From 1997 | $19 a month for 10 years of history, $39 for all |
| EODHD | Yes, "in any of our packages" | Mostly from 2000 | $19.99 a month |
| Norgate Data | Platinum level and above | From 1990 | $630 a year |
| Wikipedia, "Historical components of the S&P 500" | Dated list of additions and removals, about 20 a year | From 1976 | Free (membership only, no prices) |
| GitHub `fja05680/sp500` | Daily S&P 500 membership | From 1996 | Free (membership only) |

## 6. Group D: turning a score into a portfolio

This is the group closest to what we would build.

**Grinold (1989), "The fundamental law of active management"; Clarke, de
Silva and Thorley (2002), "Portfolio constraints and the fundamental
law".** Not read from the originals (paywalled); from a later paper's
restatement and an abstract. The law: a strategy's Sharpe-like ratio ≈
(how well the scores line up with outcomes) × √(number of independent
bets). A tiny edge spread over many stocks adds up. Constraints such as
"no shorting" let only part of the edge through; in practice 30% to 80%.

**Ledoit and Wolf (2017), "Nonlinear shrinkage of the covariance matrix
for portfolio selection: Markowitz meets Goldilocks".** The recipe.
- Idea: use a cleaned covariance matrix (theirs is a close relative of
  our RIE) with a momentum score.
- How it works:
  1. Each month take the 500 largest US stocks (also 30 to 250).
  2. Score each stock by its return over the past year, skipping the
     latest month. Call the scores m.
  3. Estimate the covariance matrix from the past 250 days and clean it.
  4. Find the weights with the *lowest risk* that satisfy two conditions:
     the weights sum to 1, and the portfolio's average score equals a
     target b. The target is the average score of the top fifth of
     stocks by momentum. The answer is a mix of two portfolios,
     "inverse covariance × ones" (our minimum-variance portfolio) and
     "inverse covariance × scores".
  5. Hold for 21 days, repeat. Tested 1973–2011.
- Result (500 stocks; return and risk in % a year):

  | Portfolio | Return | Risk | Sharpe |
  |---|---|---|---|
  | Top fifth by momentum, equal weight (the benchmark) | 14.7 | 21.5 | 0.68 |
  | Recipe with no risk model (all stocks treated alike) | 13.6 | 19.6 | 0.69 |
  | Recipe with the raw covariance matrix | fails with 500 stocks and 250 days | | |
  | Recipe with simple (linear) cleaning | 8.9 | 11.8 | 0.75 |
  | Recipe with their nonlinear cleaning | 9.9 | 11.1 | 0.90 |
  | Minimum variance, no signal, same cleaning | | 9.7 | 0.55 |

  The signal lifts the Sharpe ratio from 0.55 to 0.90 by adding return
  for a little extra risk. With shorting forbidden the Sharpe ratio is
  0.94 with nonlinear cleaning and 0.90 with the raw matrix: as we found
  on the risk side, constraints make the cleaning matter much less.
- Catch: the recipe replaces about three times the portfolio's value
  every month. Sharpe after costs, 500 stocks:

  | Cost per trade (bid-ask spread) | Benchmark | Simple cleaning | Nonlinear cleaning |
  |---|---|---|---|
  | 3 bp | 0.66 | 0.62 | 0.81 |
  | 5 bp | 0.65 | 0.52 | 0.75 |
  | 10 bp | 0.62 | 0.29 | 0.59 |
  | 20 bp | 0.57 | −0.16 | 0.27 |

  Above roughly 10 bp the plain benchmark wins. The theory also assumes
  the score is unrelated to the errors in the covariance matrix. The
  numbers are from the authors' working paper appendix.

**Engle, Ledoit and Wolf (2019), "Large dynamic covariance matrices".**
The same recipe with a covariance model that updates daily, 1986–2015.
With 1,000 stocks: benchmark 0.76, their best model 1.62 (return ÷ risk).
No costs reported.

**De Nard, Ledoit and Wolf (2021), "Factor models for portfolio selection
in large dimensions".** The same recipe, 1978–2017, 1,000 stocks:
benchmark 0.87, static nonlinear cleaning 1.71, best dynamic model 2.01.
Monthly turnover 1.3 times the portfolio for the static model and about 2
for the dynamic ones; total long plus short positions about 4 times the
money invested.

**Bun, Bouchaud and Potters (2017), "Cleaning large correlation
matrices".** The source of our RIE.
- Theory: for a score that is unrelated to the noise in the matrix, the
  risk the optimiser predicts, the true risk and the risk actually
  realised are in fixed proportion: predicted ÷ (1 − q) = true = (1 − q) ×
  realised, where q is stocks ÷ days. The "oracle" cleaning that our RIE
  approximates is the best possible for *any* such score, not only for
  minimum variance.
- Test (500 US stocks, 1966–2012, realised risk in % a year):

  | Score used | RIE | Clipping | Linear shrinkage | No risk model | Raw matrix |
  |---|---|---|---|---|---|
  | None (minimum variance) | 10.4 | 10.6 | 10.5 | 15.0 | 11.6 |
  | Yesterday's return, reversed | 7.97 | 8.11 | 8.13 | 17.7 | 9.75 |
  | Perfect foresight | 10.9 | 11.1 | 11.1 | 17.3 | 13.4 |

- Catch: only risk is reported, never return, and they state that scores
  that line up with the main directions of the matrix are outside the
  theory.

**Pedersen, Babu and Levine (2021), "Enhanced portfolio optimization".**
- Idea: the optimiser's worst bets sit in the quietest directions of the
  covariance matrix, exactly the ones our eigenvector studies found
  unreliable. Fix it by shrinking all correlations toward zero.
- How it works: replace every correlation by (1 − w) × itself. w = 0 is
  the textbook optimiser; w = 1 ignores correlations and just sizes each
  asset by score ÷ variance. Pick w by what worked on past data.
- Result (55 futures markets, 1985–2018, Sharpe): textbook optimiser
  0.87, signal-only benchmark 1.09, shrunk at w = 75%: 1.32. Fixing the
  risk model alone needs w of 5% to 10%; allowing for noise in the
  *forecast* needs about 75%. With that much shrinkage, a random-matrix
  risk model gives the same result as a basic one (1.32 against 1.32).
- Catch: futures and industry baskets, at most 98 assets; no costs.

**Kozak, Nagel and Santosh (2020), "Shrinking the cross-section".** The
same insight, derived differently: shrink the bets on quiet directions
most, by a factor that depends on each direction's size.

**Brandt, Santa-Clara and Valkanov (2009), "Parametric portfolio
policies".**
- Idea: skip the forecast and the covariance matrix altogether. Write the
  weights directly as "market weight + a tilt proportional to the stock's
  scores" and tune three tilt strengths.
- Result (size, value and momentum scores, 1964–2002): Sharpe 0.96
  against 0.36 for the market; 0.69 to 0.94 out-of-sample; 0.56 to 0.58
  with shorting forbidden. (Working paper version.)
- Catch: no risk model, so nothing limits hidden bets on one sector.

**DeMiguel, Martín-Utrera, Nogales and Uppal (2020), "A transaction-cost
perspective on the multitude of firm characteristics".** Trading several
scores together is cheaper than trading each alone, because their trades
partly cancel: turnover drops from 24% to 7% a month. Sharpe after costs:
1.36 with shorting, 1.06 with shorting limited to half, 0.67 with no
shorting.

**Gârleanu and Pedersen (2013), "Dynamic trading with predictable returns
and transaction costs".** Do not jump to the ideal portfolio each period;
move part of the way, and give more weight to slow-changing signals. In
their test, jumping all the way had a Sharpe of 0.83 before costs and −9.8
after; the partial rule 0.62 before and 0.58 after.

**Ao, Li and Zheng (2019), "Approaching mean-variance efficiency for large
portfolios".** Abstract only.

**What group D says:** the published way to add a signal to a risk model
like ours is the Ledoit–Wolf two-condition recipe. Its gains are real and
depend on the cleaning when shorting is allowed, and they are fragile to
trading costs. Shrinking correlations hard, trading only part of the way,
and forbidding or limiting shorts all trade a little paper performance
for a lot of robustness.

## 7. Group E: many signals, machine learning, and news

| Paper | Inputs → method → output | Result (before costs) | Catch |
|---|---|---|---|
| Lewellen (2015) | 3, 7 or 15 scores per stock → each month, fit how returns lined up with scores, average those fits over the past 10 years → forecast of next month's return | Realised returns move 0.74 to 0.80 for each 1.00 of forecast (all stocks), 0.44 to 0.66 for large stocks. Top-minus-bottom tenth: 2.4% a month equal-weighted, 0.9% for large stocks value-weighted | In the last ten years of the sample the figure for large stocks is −0.37: the forecasts stopped working |
| Green, Hand and Zhang (2017) | 94 scores at once | 12 matter outside the tiniest stocks; since 2003 only 2 | Abstract only |
| Gu, Kelly and Xiu (2020) | 94 scores × 8 economy-wide variables + industry = 920 inputs → 13 methods up to 5-layer neural networks → next month's return; trained 1957–1986, tested 1987–2016 | Best network explains 0.40% of the variation in monthly returns. Top-minus-bottom Sharpe 1.35 value-weighted, 2.45 equal-weighted | Replaces more than 110% of the portfolio monthly; no costs; the most useful inputs are price-based (recent returns, volatility, trading volume) |
| Freyberger, Neuhierl and Weber (2020) | 62 scores → a flexible curve per score, with automatic selection | 13 of 62 scores matter. Sharpe 2.83 for all stocks, 0.86 once the smallest fifth is excluded | Working paper numbers; no costs |
| Kelly, Pruitt and Su (2019) | 36 scores → a few hidden common drivers whose effect on each stock depends on its scores | Sharpe up to 4.05 on fresh data; 2.61 for large stocks. 10 scores do the work, most of them price or volume based | "High turnover, implying high implementation costs" |
| Chen, Pelger and Zhu (2024) | 46 scores + 178 economy-wide series → two competing neural networks | Sharpe about 2.6 for all stocks, about 0.9 for the largest 550 | Training Sharpe far above test Sharpe |
| Avramov, Cheng and Metzker (2023) | Re-examines the machine-learning strategies | Profits sit in hard-to-trade stocks; they shrink a lot without microcaps and distressed firms, and further after costs | Abstract only |
| Jiang, Kelly and Xiu (2023) | **Prices and volume only**: a picture of the last 5, 20 or 60 days' price bars → image-recognition network → probability the stock rises | Weekly: Sharpe 7.2 equal-weighted, 1.5 value-weighted. Monthly: 2.4 and 0.5. Largest 500 stocks, weekly: about 1.0 | Weekly version replaces 690% of the portfolio a month; authors say it suits "market makers" |
| Kelly, Malamud and Zhou (2024) | 15 economy-wide predictors blown up into 12,000 features → timing the whole market | Sharpe 0.47 | A heavily shrunk simple model gets 0.46; a later paper argues the method reduces to momentum |
| Tetlock (2007) | Daily count of gloomy words in a newspaper column | A gloomy day predicts −8 bp on the index next day, reversed within the week | Trading it earns about the cost of trading it |
| Ke, Kelly and Xiu (2019) | 22 million news articles → words that go with price rises and falls → daily score | Sharpe 4.3 equal-weighted, 1.3 value-weighted; the news is fully in the price within about two days | Replaces 95% of the portfolio daily; 2.3 at best after costs |
| Lopez-Lira and Tang (2023–2025) | Ask a language model whether each headline is good or bad news | Sharpe 3.0 equal-weighted, 0.6 value-weighted; fell from 6.5 in late 2021 to 1.2 in early 2024 | Negative for large stocks after a 5 bp cost |

**What group E says:** the same pattern in every row. Equal-weighted,
before-cost numbers are spectacular; value-weighted or large-stock numbers
are a third to a half as large; turnover is extreme; later years are
weaker. These methods train on 20 to 30 years with dozens of inputs. With
nine years of prices and volumes there is room only for simple, heavily
shrunk models.

## 8. What this means for our project

1. **Our minimum-variance portfolio is the literature's recommended
   answer when forecasts are weak.** That is the explicit advice of
   Chopra and Ziemba and of DeMiguel et al. It is the safe portfolio every
   method in group A leans away from.
2. **Fix the data before testing any forecast.** The answer key shows our
   first test had the sign wrong. Options, cheapest first: use French's
   ready-made groups as the truth for "does this rule work"; build a
   point-in-time S&P 500 list; subscribe to a feed with dead companies
   (about $19 to $39 a month).
3. **Momentum is the one price-based signal worth building on.** It
   survives replication and costs, and it needs a risk brake, which our
   volatility machinery can supply. Expect something like half the
   published 1% a month, less in large stocks, and long stretches of
   nothing: since 2019 it has not been distinguishable from zero.
4. **The recipe to copy is Ledoit and Wolf's**, because it is built for a
   cleaned covariance matrix like ours and published with full tables to
   compare against. Run it in our long-only and gross-capped modes too.
5. **Budget for costs from the start.** The published recipe trades three
   times the portfolio a month and loses to a plain benchmark at 10 to 20
   bp. Use the known remedies: trade part of the way, only when a stock
   moves well out of its group, and shrink the forecast hard.
6. **Judge it the way the papers do**: Sharpe ratio on later data against
   the "top fifth by momentum" benchmark and against minimum variance,
   with the error bar stated, value-weighted or large-stock results next
   to any equal-weighted one, after costs.

## 9. What was and was not verified

- Read in full from the published article or the authors' final draft:
  the large majority; numbers carry table references in the working notes.
- **Working-paper versions read instead of the journal article** (numbers
  may differ slightly): Merton 1980; Jegadeesh and Titman 2001; Lehmann
  1990; Ang et al. 2006; Daniel and Moskowitz 2016; Barroso and
  Santa-Clara 2015 (thesis chapter); Harvey, Liu and Zhu 2016; Hou, Xue
  and Zhang 2020; Elton, Gruber and Blake 1996; Brandt, Santa-Clara and
  Valkanov 2009; Freyberger, Neuhierl and Weber 2020; Kelly, Malamud and
  Zhou 2024; Frazzini, Israel and Moskowitz (2012 draft); Ledoit and Wolf
  2017 (momentum tables from working paper 137, appendix E).
- **Abstract only** (no numbers taken from the body): Jorion 1986 (formula
  taken from two papers that restate it); Jobson and Korkie 1980; Best and
  Grauer 1991; Jegadeesh 1990; Shumway 1997; Shumway and Warther 1999;
  Brown et al. 1992; Clarke, de Silva and Thorley 2002; Ao, Li and Zheng
  2019; Avramov, Cheng and Metzker 2023; Green, Hand and Zhang 2017; Gu,
  Kelly and Xiu 2021.
- **Not verified from the original**: Grinold 1989 and 1994 (stated from a
  later paper's restatement).
- The French-library comparison in section 5 is our own calculation from
  his files, not a published number. Vendor prices were read from vendor
  pages on 2026-11-12.
- Oddities found in the sources: Daniel and Moskowitz's working paper
  gives 0.71, 0.60 and 0.59 for the same Sharpe ratio in different places;
  one Sharpe entry in Ledoit and Wolf's table (30 stocks, single-factor
  model, 0.48) does not match its own return and risk (0.63).

## 10. References

Group A. Merton, R. (1980), J. Financial Economics 8(4), 323–361,
doi:10.1016/0304-405X(80)90007-0. Chopra, V. and Ziemba, W. (1993), J.
Portfolio Management 19(2), 6–11, doi:10.3905/jpm.1993.409440. Michaud, R.
(1989), Financial Analysts J. 45(1), 31–42, doi:10.2469/faj.v45.n1.31.
Jobson, J. and Korkie, B. (1980), J. American Statistical Association
75(371), 544–554, doi:10.1080/01621459.1980.10477507. Best, M. and Grauer,
R. (1991), Review of Financial Studies 4(2), 315–342,
doi:10.1093/rfs/4.2.315. DeMiguel, V., Garlappi, L. and Uppal, R. (2009),
Review of Financial Studies 22(5), 1915–1953, doi:10.1093/rfs/hhm075.
Jorion, P. (1986), J. Financial and Quantitative Analysis 21(3), 279–292,
doi:10.2307/2331042. Black, F. and Litterman, R. (1992), Financial Analysts
J. 48(5), 28–43, doi:10.2469/faj.v48.n5.28. He, G. and Litterman, R.
(1999), "The intuition behind Black-Litterman model portfolios", Goldman
Sachs. Kan, R. and Zhou, G. (2007), J. Financial and Quantitative Analysis
42(3), 621–656, doi:10.1017/S0022109000004129. Tu, J. and Zhou, G. (2011),
J. Financial Economics 99(1), 204–215, doi:10.1016/j.jfineco.2010.08.013.

Group B. Jegadeesh, N. and Titman, S. (1993), J. Finance 48(1), 65–91,
doi:10.1111/j.1540-6261.1993.tb04702.x; (2001), J. Finance 56(2), 699–720,
doi:10.1111/0022-1082.00342. Jegadeesh, N. (1990), J. Finance 45(3),
881–898, doi:10.1111/j.1540-6261.1990.tb05110.x. Lehmann, B. (1990),
Quarterly J. Economics 105(1), 1–28, doi:10.2307/2937816. De Bondt, W. and
Thaler, R. (1985), J. Finance 40(3), 793–805, doi:10.2307/2327804. Ang, A.,
Hodrick, R., Xing, Y. and Zhang, X. (2006), J. Finance 61(1), 259–299,
doi:10.1111/j.1540-6261.2006.00836.x. Frazzini, A. and Pedersen, L. (2014),
J. Financial Economics 111(1), 1–25, doi:10.1016/j.jfineco.2013.10.005.
Novy-Marx, R. and Velikov, M. (2022), J. Financial Economics 143(1),
80–106, doi:10.1016/j.jfineco.2021.05.023. Daniel, K. and Moskowitz, T.
(2016), J. Financial Economics 122(2), 221–247,
doi:10.1016/j.jfineco.2015.12.002. Barroso, P. and Santa-Clara, P. (2015),
J. Financial Economics 116(1), 111–120, doi:10.1016/j.jfineco.2014.11.010.
Moreira, A. and Muir, T. (2017), J. Finance 72(4), 1611–1644,
doi:10.1111/jofi.12513. Cederburg, S., O'Doherty, M., Wang, F. and Yan, X.
(2020), J. Financial Economics 138(1), 95–117,
doi:10.1016/j.jfineco.2020.04.015. Asness, C., Moskowitz, T. and Pedersen,
L. (2013), J. Finance 68(3), 929–985, doi:10.1111/jofi.12021. Moskowitz,
T., Ooi, Y. and Pedersen, L. (2012), J. Financial Economics 104(2),
228–250, doi:10.1016/j.jfineco.2011.11.003. George, T. and Hwang, C.
(2004), J. Finance 59(5), 2145–2176,
doi:10.1111/j.1540-6261.2004.00695.x. Lou, D., Polk, C. and Skouras, S.
(2019), J. Financial Economics 134(1), 192–213,
doi:10.1016/j.jfineco.2019.03.011.

Group C. McLean, R. and Pontiff, J. (2016), J. Finance 71(1), 5–32,
doi:10.1111/jofi.12365. Harvey, C., Liu, Y. and Zhu, H. (2016), Review of
Financial Studies 29(1), 5–68, doi:10.1093/rfs/hhv059. Hou, K., Xue, C. and
Zhang, L. (2020), Review of Financial Studies 33(5), 2019–2133,
doi:10.1093/rfs/hhy131. Jensen, T., Kelly, B. and Pedersen, L. (2023), J.
Finance 78(5), 2465–2518, doi:10.1111/jofi.13249. Novy-Marx, R. and
Velikov, M. (2016), Review of Financial Studies 29(1), 104–147,
doi:10.1093/rfs/hhv063. Frazzini, A., Israel, R. and Moskowitz, T. (2018),
"Trading costs", AQR working paper. Shumway, T. (1997), J. Finance 52(1),
327–340, doi:10.1111/j.1540-6261.1997.tb03818.x. Shumway, T. and Warther,
V. (1999), J. Finance 54(6), 2361–2379, doi:10.1111/0022-1082.00192. Brown,
S., Goetzmann, W., Ibbotson, R. and Ross, S. (1992), Review of Financial
Studies 5(4), 553–580, doi:10.1093/rfs/5.4.553. Elton, E., Gruber, M. and
Blake, C. (1996), Review of Financial Studies 9(4), 1097–1120,
doi:10.1093/rfs/9.4.1097. Lo, A. (2002), Financial Analysts J. 58(4),
36–52, doi:10.2469/faj.v58.n4.2453. Bailey, D. and López de Prado, M.
(2014), J. Portfolio Management 40(5), 94–107,
doi:10.3905/jpm.2014.40.5.094. Chen, A. and Zimmermann, T. (2022), Critical
Finance Review 11(2), 207–264, doi:10.1561/104.00000112. French, K., data
library, mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html.

Group D. Grinold, R. (1989), J. Portfolio Management 15(3), 30–37,
doi:10.3905/jpm.1989.409211; (1994), J. Portfolio Management 20(4), 9–16,
doi:10.3905/jpm.1994.409482. Clarke, R., de Silva, H. and Thorley, S.
(2002), Financial Analysts J. 58(5), 48–66, doi:10.2469/faj.v58.n5.2468.
Ledoit, O. and Wolf, M. (2017), Review of Financial Studies 30(12),
4349–4388, doi:10.1093/rfs/hhx052 (tables from University of Zurich
Economics working paper 137, February 2017). Engle, R., Ledoit, O. and
Wolf, M. (2019), J. Business and Economic Statistics 37(2), 363–375,
doi:10.1080/07350015.2017.1345683. De Nard, G., Ledoit, O. and Wolf, M.
(2021), J. Financial Econometrics 19(2), 236–257,
doi:10.1093/jjfinec/nby033. Bun, J., Bouchaud, J.-P. and Potters, M.
(2017), Physics Reports 666, 1–109, doi:10.1016/j.physrep.2016.10.005,
arXiv:1610.08104. Pedersen, L., Babu, A. and Levine, A. (2021), Financial
Analysts J. 77(2), 124–151, doi:10.1080/0015198X.2020.1854543. Kozak, S.,
Nagel, S. and Santosh, S. (2020), J. Financial Economics 135(2), 271–292,
doi:10.1016/j.jfineco.2019.06.008. Brandt, M., Santa-Clara, P. and
Valkanov, R. (2009), Review of Financial Studies 22(9), 3411–3447,
doi:10.1093/rfs/hhp003. DeMiguel, V., Martín-Utrera, A., Nogales, F. and
Uppal, R. (2020), Review of Financial Studies 33(5), 2180–2222,
doi:10.1093/rfs/hhz085. Gârleanu, N. and Pedersen, L. (2013), J. Finance
68(6), 2309–2340, doi:10.1111/jofi.12080. Ao, M., Li, Y. and Zheng, X.
(2019), Review of Financial Studies 32(7), 2890–2919,
doi:10.1093/rfs/hhy105.

Group E. Lewellen, J. (2015), Critical Finance Review 4(1), 1–44,
doi:10.1561/104.00000024. Green, J., Hand, J. and Zhang, X. F. (2017),
Review of Financial Studies 30(12), 4389–4436, doi:10.1093/rfs/hhx019. Gu,
S., Kelly, B. and Xiu, D. (2020), Review of Financial Studies 33(5),
2223–2273, doi:10.1093/rfs/hhaa009; (2021), J. Econometrics 222(1),
429–450, doi:10.1016/j.jeconom.2020.07.009. Freyberger, J., Neuhierl, A.
and Weber, M. (2020), Review of Financial Studies 33(5), 2326–2377,
doi:10.1093/rfs/hhz123. Kelly, B., Pruitt, S. and Su, Y. (2019), J.
Financial Economics 134(3), 501–524, doi:10.1016/j.jfineco.2019.05.001.
Chen, L., Pelger, M. and Zhu, J. (2024), Management Science 70(2),
714–750, doi:10.1287/mnsc.2023.4695. Avramov, D., Cheng, S. and Metzker, L.
(2023), Management Science 69(5), 2587–2619, doi:10.1287/mnsc.2022.4449.
Jiang, J., Kelly, B. and Xiu, D. (2023), J. Finance 78(6), 3193–3249,
doi:10.1111/jofi.13268. Kelly, B., Malamud, S. and Zhou, K. (2024), J.
Finance 79(1), 459–503, doi:10.1111/jofi.13298. Nagel, S. (2025),
"Seemingly virtuous complexity in return prediction", NBER working paper
34104. Tetlock, P. (2007), J. Finance 62(3), 1139–1168,
doi:10.1111/j.1540-6261.2007.01232.x. Ke, Z., Kelly, B. and Xiu, D.
(2019), "Predicting returns with text data", NBER working paper 26186.
Lopez-Lira, A. and Tang, Y. (2023, revised 2025), "Can ChatGPT forecast
stock price movements?", arXiv:2304.07619.
