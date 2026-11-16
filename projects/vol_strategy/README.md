# SPX volatility strategy backtest

A backtest of the user's volatility strategy (the "Buy High, Sell Low" page): 30-day SPX options
struck at 15 delta, delta-hedged daily with ES futures. It covers all six versions on the page:

| Key | Version | Legs (per unit) |
|---|---|---|
| `book` | Short (put + strangle) | sell 2 × 15-delta puts, sell 1 × 15-delta call |
| `strangle` | Short strangle | sell 15-delta put and call |
| `condor` | Iron condor | short strangle + buy 5-delta put and call wings |
| `longstr` | Long strangle | buy 15-delta put and call |
| `longput` | Long put | buy 15-delta put |
| `longcall` | Long call | buy 15-delta call |

Data: daily S&P 500 and VIX closes from FMP (June 2009 – Nov 2026), and real SPY/SPXW option
trades from Alpaca (Feb 2024 onwards) to calibrate and check the option prices.

## Quick start

```bash
cd projects/vol_strategy
pip install numpy pandas scipy matplotlib pyarrow requests pytest
python -m pytest -q
```

```bash
FMP_API_KEY=... python -m volstrat --output-dir output
```

Add `--calibrate` (with `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` set) to rebuild the surface
calibration from Alpaca option data first. That takes about an hour. The fitted parameters are
already in `calibration/`.

## How the backtest works

- **Entry:** when flat, open one unit at the close if the previous close's VIX was between 12 and
  18 (the page's starting condition). Strikes are picked at 15 delta (5 delta for the condor's wings)
  on the day's volatility surface, rounded to 5-point SPX strikes. Expiry is the last trading day on
  or before 30 calendar days out.
- **Hedge:** each close, ES is re-set to the position's Black-Scholes delta (fractional contracts).
- **Exits:** hold to expiry (the primary rule), or one of `tp` (take profit at 50% of the credit,
  or +100% of the debit for long versions), `tp_sl` (plus a stop at a loss of 2× the credit, or 50%
  of the debit), `dte` (close with 10 days left), or `tp_dte`. A new position can open the same day.
- **Costs:** option half-spread of max(0.05 points, 1% of the price) plus $1.50 per contract each
  way; nothing at cash-settled expiry. ES: $0.175 per $1/point traded (half a tick plus commission).
- **Returns:** daily P&L divided by one SPX contract's notional (index × $100). Sharpe ratios do
  not depend on this choice. Return on margin is much higher than return on notional, because
  margin is only a fraction of notional.

### Option prices: the calibrated volatility surface

There is no free history of SPX option prices back to 2010, so options are priced with
Black-Scholes on a smile driven by the VIX: `sigma(K) = atm - a*m + b*m^2`, `m = ln(K/S)`.

The page uses ATM = 0.9 × VIX, a = 0.6, b = 1.0 at every tenor. Real SPY option trades
(`volstrat/calibrate.py`, 699 days of 30-day smiles plus about 350 days each of 10- and 20-day
smiles) show that this is too rich on calls and too cheap on the 5-delta put wings, especially for
short-dated options in stress. The calibrated surface (`calibration/surface_params.json`) instead
lets the level and skew depend on the VIX and on time to expiry, and holds the smile flat outside
the range of strikes it was fitted on:

- ATM (vol points) = 0.73 × VIX + 2.1, or about 0.87 × VIX at VIX 15
- skew slope a = 0.48 + 0.012 × VIX, curvature b = 3.5 − 0.08 × VIX
- shorter options: skew slope × (30/T)^0.27, curvature × (30/T)^1.24, ATM × (30/T)^(−0.22 + 0.0098 × VIX)

Median model-minus-real implied vol at the traded strikes (vol points) is within ±1 point for
VIX below 30 at 10, 20 and 30 days (`calibration/fit_errors.csv`). The page's model is off by up
to +3.8 points on calls, and by −5.5 to −17 points on short-dated 5-delta puts in stress
(`calibration/fit_errors_page_model.csv`).

**Check on real SPX options.** At the backtest's own entries since Feb 2024 (33 distinct option
legs that traded on SPXW that day; none of the condor's 5-delta wings did), the calibrated surface is on average 0.3 (puts) to 0.8 (calls) vol points richer than
the traded prices, and the page's model is 1.5 points richer on calls (`results/spxw_check.csv`).
So the calibrated results below probably still slightly flatter the short-vol versions. The
"calibrated −0.5" rows show the effect of correcting for that.

## Results

Primary setup, fixed before any results were seen: hedged daily, VIX 12–18 entry filter, held to
expiry. Calibrated surface, after costs, Jan 2010 – Nov 2026.

| Version | Sharpe | Return / yr on notional | Max drawdown on notional | Win rate | Worst trade (× premium) | Beta / corr. to SPX | Sharpe, surface −0.5 vol |
|---|---|---|---|---|---|---|---|
| Short (put + strangle) | 0.69 | 1.83% | −8.7% | 84% | −5.8 | 0.06 / 0.25 | 0.35 |
| Short strangle | 0.85 | 1.30% | −4.8% | 79% | −4.9 | 0.04 / 0.26 | 0.47 |
| Iron condor | 0.43 | 0.46% | −2.8% | 70% | −4.2 | 0.03 / 0.30 | 0.05 |
| Long strangle | −1.08 | −1.63% | −30.2% | 17% | −1.5 | −0.04 / −0.26 | −0.69 |
| Long put | −0.58 | −0.70% | −13.8% | 15% | −1.9 | −0.02 / −0.21 | −0.33 |
| Long call | −1.35 | −0.95% | −16.9% | 26% | −3.5 | −0.01 / −0.19 | −0.94 |

151 completed trades per version. A hedged long option can lose more than its premium when the
hedge loses on a trending, low-volatility path. Full tables: `results/grid_calibrated.csv`,
`results/grid_page.csv` and `results/sensitivity.csv`.

### What the results say

1. **The premium is real.** On 85% of days the VIX was above the S&P 500's realised volatility
   over the next 21 trading days, by 3.9 points on average (`results/vrp.png`). Selling options
   earns that premium. Buying them pays it, and the long versions lose money in every setup tested.
2. **Short volatility makes money, but the edge is thin.** It is worth about one volatility point.
   Moving the whole calibrated surface down by 1 point takes the book's Sharpe from 0.69 to about 0;
   moving it up by 1 point gives 1.34. The real-SPX check above suggests the truth is a little below
   the calibrated surface. The page's own pricing model makes every short version look better than
   the calibrated one, and the iron condor most of all (0.96 vs 0.43), because the page prices its
   protective wings too cheaply.
3. **The edge has faded.** Calibrated Sharpe for 2010–2017 vs 2018–2026: book 1.64 → 0.24,
   strangle 1.75 → 0.38, condor 1.09 → 0.06 (`results/subperiods.csv`). With the −0.5 vol
   correction from the SPX check, 2018–2026 is about break-even (book −0.02, strangle 0.07). Since
   Feb 2018 there have been more volatility sellers and sharper shocks.
4. **The losses come in bursts.** Stress windows cost the book 3.4–7.2% of notional each
   (Aug 2015, Feb 2018, Mar 2020, Apr 2025; `results/episodes.png`). That is one to two years of
   average gains. April 2025 was the worst: the options expired worthless, but the daily ES hedge
   sold into the crash and bought back into the rebound ("buy high, sell low").
5. **Ranking:** the short strangle has the best risk-adjusted result. The book earns more per unit
   but takes about twice the tail risk. The iron condor's wings cost most of its premium. The long
   versions are insurance: they pay in crashes (up to +3.7% of notional in April 2025) and lose
   steadily otherwise.
6. **Exit rules:** in the grid (`results/sharpe_grid.png`), take-profit plus stop (`tp_sl`) looks
   best for the book and strangle, and closing early (`dte`) looks worst. Treat these comparisons
   with care: early exits depend on how the model prices an option part-way through its life,
   which is the least certain part of the calibration. The hold-to-expiry results depend only on the
   entry price, the hedge and the settlement.
7. **The VIX filter helps the short versions.** Without it, entries continue into high-VIX
   regimes, and the hedged book's Sharpe drops from 0.69 to 0.51.
8. **"Hedged" is not market-neutral.** The hedged short books keep a correlation of 0.25–0.30
   with the S&P 500 (beta 0.03–0.06 on the calibrated surface, about 0.1 on the page's model).
   Short vega loses when the VIX jumps in a selloff, and a Black-Scholes delta hedge does not
   offset that.

### Status at the last data point (13 Nov 2026)

The VIX closed at 19.3, above the 12–18 band, so no new position would be opened at the next
close. Under the primary rules the backtest holds positions opened on 12 Nov (VIX 17.5 the day
before), expiring 11 Dec. See `results/status.json` for the strikes and hedge sizes.

## Limitations

- **Pricing is modelled, not historical.** The surface is calibrated on 2024–2026 SPY options and
  applied to 2010–2026. If skew was flatter or steeper before 2024, earlier results shift.
- **Daily closes only.** Re-hedging happens at the close, so intraday moves and overnight gaps
  are captured only close to close. Real gap risk is somewhat larger.
- **Margin is not modelled.** Short SPX options need margin that jumps in stress. A forced
  reduction at the worst moment would make results worse.
- **One unit at a time, fractional hedge.** Real sizes would round to whole ES or MES contracts.
- **Early-exit marks** depend on the fitted term structure of skew.

## How the code was checked

- 115 unit tests: pricing (put-call parity, Greeks against finite differences), the strike
  solver, accounting (daily P&L equals trade P&L, costs charged exactly once), exits, the
  look-ahead-free entry filter, and a Monte Carlo test showing that a hedged short option breaks
  even when implied equals realised volatility and earns the gap when implied is higher.
- A six-agent code review (pricing, timing and look-ahead, accounting, realism, reporting, tests)
  with a skeptic trying to refute each finding. 12 of 14 findings were confirmed and fixed. They
  included a bug that gave the iron condor long-vol exit levels, a strike solver that broke with
  realistic smile curvature, and the overly rich pricing model. Each has a regression test in
  `tests/test_review_fixes.py`.

## Layout

```
volstrat/bs.py           Black-Scholes, Greeks, the VIX-driven smile, strike-for-delta solver
volstrat/data.py         S&P 500 and VIX from FMP
volstrat/strategies.py   the six positions
volstrat/backtest.py     daily engine, costs, exits, summary statistics
volstrat/analysis.py     grids, variance risk premium, stress episodes
volstrat/calibrate.py    Alpaca option data, implied vols, surface fit, SPXW price check
volstrat/surfaces.py     page and calibrated surface presets
volstrat/__main__.py     runs everything and writes results/
calibration/             SPY smile data and fitted surface parameters
results/                 tables and charts from the last full run
```

## Next steps

- **Chapter 9:** forecast realised volatility (GARCH) and enter only when the VIX is well above the
  forecast, rather than using a fixed 12–18 band.
- **Chapter 24 and candlesticks:** test chart conditions as entry filters, e.g. skip new short-vol
  positions after a close below the lower Bollinger band or a bearish engulfing day.
- **Chapter 5:** position sizing with stress-based margin.
- **Paper trading** through Alpaca to measure real fills against the model.
