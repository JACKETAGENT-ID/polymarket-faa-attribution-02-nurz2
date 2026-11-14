# FAA attribution forecast — nurz v2

Python 3.13, standard library only. `main.py` reads each case in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`): for every market, 240 rows of
five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the Yes price.

## Approach

A fast-adapting volatility random walk with separately priced tails:

- **Two-speed, two-shape volatility.** Body sigma blends a robust
  MAD-based sigma with an RMS sigma under a short 30-minute EWMA
  half-life (plus 10% of a slow whole-history estimate), so spreads
  react within the hour when a market wakes up. The outer quantiles
  use an RMS-only twin sigma — jump-sensitive, because outer pinball
  loss is where single large ticks land. Per-minute sigma is floored
  at 0.0012 and capped at 0.006 (the cap doubles as a jump cap), then
  scaled by `2*sqrt(x(1-x))` floored at 0.20.
- **Horizon-banded term structure.** The `sqrt(h)` fan is multiplied
  per band (≤10, ≤30, ≤60, ≤120, >120 min) by
  (0.95, 1.15, 1.10, 1.02, 1.00) — the 10–60 minute bands were
  under-covered at the plain persistence shape.
- **Centre.** Last price minus 0.3× the last 1-minute move (partial
  reversal of book bounce), bleeding toward zero at 0.04% of price
  per minute through a saturating 90-minute exponential, capped at
  0.35σ/min. Inner quantiles use a 0.9 z-scale.

## Local measurement

No local grader ships with the kit, so losses were measured by
walk-forward on the practice cases themselves: the actuals for case
p_k are the last 240 prices of the same market in case p_{k+1}
(cases are exactly 4 hours apart), scored with the rules' pinball
loss over p00–p04 (124,800 items):

- house baseline: 0.3925
- this account's earlier EWMA entry: 0.3031 (server dev loss 0.2678)
- this entry: 0.2965

An independent internal backtest (pseudo-checkpoints in the last
2,000 minutes of p05) agrees: 0.6367 vs the earlier entry's 0.6420.
Ideas that were tried and rejected on the same measurements:
empirical h-step quantile refits, multi-scale vol curves, centre
smoothing toward an EWMA level, vol-burst factors, and fitted
band recalibrations (they failed split-half validation).
