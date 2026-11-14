# polymarket-faa-attribution-02 entry (nurz, v2) — Python 3.13, stdlib only.
#
# Forecast: a fast-adapting volatility random walk whose tails are priced
# separately from its body. Three ideas, each measured on the practice cases
# against this account's earlier EWMA entry (local walk-forward loss 0.3031
# -> 0.2965; see README):
#
# 1. Two-speed, two-shape volatility. The body sigma blends a robust
#    median-absolute-deviation sigma with an RMS sigma, both EWMA-weighted
#    with a SHORT 30-minute half-life (plus a 10% weight on a slow
#    whole-history estimate), so the spread reacts within the hour when a
#    market wakes up instead of averaging over yesterday's regime. The
#    OUTER quantiles (5%/95%) do not reuse that body sigma: they use an
#    RMS-only twin sigma, which is jump-sensitive, because the outer
#    pinball loss is exactly where a single large tick lands. Per-minute
#    sigma is floored (0.0012) and capped (0.006) — the cap is the jump
#    cap: no single minute's implied vol, however violent, widens the
#    whole 4-hour fan beyond it — and scaled by 2*sqrt(x(1-x)) floored at
#    0.20, since a price near 0 or 1 cannot travel as far.
#
# 2. Horizon-banded term structure. The sqrt(h) fan is multiplied per
#    horizon band (<=10, <=30, <=60, <=120, >120 minutes) by
#    (0.95, 1.15, 1.10, 1.02, 1.00): the 10-60 minute bands were
#    systematically under-covered at the persistence shape, the first
#    minutes slightly over-covered.
#
# 3. Centre: last price, de-bounced and bled. The centre starts at
#    x - 0.3 * (last 1-minute move) — a partial reversal of the final
#    tick, which is mostly book bounce — and then decays toward zero at
#    0.04% of the price per minute, applied through a saturating
#    90-minute exponential so the total bleed stays bounded. The drift
#    is capped at 0.35 sigma per minute.
#
# Inner quantiles (25%/75%) use a 0.9 z-scale. Deterministic: no
# randomness, no network, no clock. Reads every *.in in
# $MAKEFUN_INPUT_DIR (else /input), writes /output/<id>.out (else
# $MAKEFUN_OUTPUT_DIR), one forecast per market, 240 rows x 5 quantiles.

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# --- tunables (fixed in advance; chosen on the practice cases) -----------
SIGMA_FLOOR = 0.0012      # per-minute sigma floor (price units)
SIGMA_CAP = 0.0060        # per-minute sigma cap = the jump cap
LEVEL_FLOOR = 0.20        # floor on the price-level volatility factor
HL_CORE = 30.0            # EWMA half-life (min) of the fast sigma
HL_LONG = 480.0           # EWMA half-life (min) of the slow sigma
LONG_WEIGHT = 0.10        # weight of the slow sigma in the blend
VOL_WINDOW = 720          # minutes of recent history for the fast sigma
MAD_WEIGHT = 0.65         # body sigma: MAD weight (rest is RMS)
INNER_ZSCALE = 0.9        # inner quantiles (25%/75%) slightly narrower
BAND_MULT = (0.95, 1.15, 1.10, 1.02, 1.00)  # h: <=10, <=30, <=60, <=120, >120
REV_K = 0.3               # partial reversal of the last 1-minute move
DECAY_RATE = 0.0004       # centre bleed toward zero, per minute
DRIFT_TAU = 90.0          # saturation horizon (min) of the centre drift
DRIFT_CAP_SIG = 0.35      # drift cap, in sigmas per minute


def _price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def _ewma_mad_rms(diffs, half_life):
    """EWMA (recent-weighted) mean|d| and mean d^2 of 1-minute diffs."""
    lam = 0.5 ** (1.0 / half_life)
    w = 1.0
    wsum = 0.0
    wabs = 0.0
    wsq = 0.0
    for d in reversed(diffs):
        wsum += w
        wabs += w * abs(d)
        wsq += w * d * d
        w *= lam
    if wsum <= 0.0:
        return 0.0, 0.0
    return wabs / wsum, wsq / wsum


def _band(h):
    if h <= 10:
        return 0
    if h <= 30:
        return 1
    if h <= 60:
        return 2
    if h <= 120:
        return 3
    return 4


def _market_params(prices):
    """Return (centre0, drift_per_min, sigma_body, sigma_outer)."""
    x = prices[-1]
    n = len(prices)
    if n < 2:
        return x, 0.0, SIGMA_FLOOR, SIGMA_FLOOR
    diffs = [prices[i] - prices[i - 1] for i in range(1, n)]

    # --- fast + slow MAD/RMS statistics ----------------------------------
    ma_r, ms_r = _ewma_mad_rms(diffs[-VOL_WINDOW:], HL_CORE)
    ma_l, ms_l = _ewma_mad_rms(diffs, HL_LONG)
    # Mean |d| of a normal is sigma*sqrt(2/pi); invert for a MAD sigma.
    mad_r = ma_r * math.sqrt(math.pi / 2.0)
    mad_l = ma_l * math.sqrt(math.pi / 2.0)
    body_recent = MAD_WEIGHT * mad_r + (1.0 - MAD_WEIGHT) * math.sqrt(ms_r)
    body_long = MAD_WEIGHT * mad_l + (1.0 - MAD_WEIGHT) * math.sqrt(ms_l)
    sigma = (1.0 - LONG_WEIGHT) * body_recent + LONG_WEIGHT * body_long
    # Outer sigma: RMS only, same two-speed blend — jump-sensitive tails.
    sigma_outer = ((1.0 - LONG_WEIGHT) * math.sqrt(ms_r)
                   + LONG_WEIGHT * math.sqrt(ms_l))

    sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
    sigma_outer = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma_outer))

    # --- price-level scaling: a price near 0 or 1 cannot move as far -----
    level = 2.0 * math.sqrt(max(x, 0.0) * max(1.0 - x, 0.0))
    lf = max(LEVEL_FLOOR, min(1.0, level))
    sigma *= lf
    sigma_outer *= lf

    # --- centre: de-bounced last price + bounded bleed toward zero -------
    centre0 = x - REV_K * (prices[-1] - prices[-2])
    drift = -DECAY_RATE * x
    dcap = DRIFT_CAP_SIG * max(sigma, 1e-6)
    drift = max(-dcap, min(dcap, drift))
    return centre0, drift, sigma, sigma_outer


def forecast_market(prices, horizon):
    """240 rows x 5 quantiles for one market."""
    centre0, drift, sigma, sigma_outer = _market_params(prices)
    rows = []
    for h in range(1, horizon + 1):
        bm = BAND_MULT[_band(h)]
        spread = sigma * math.sqrt(h) * bm
        spread_outer = sigma_outer * math.sqrt(h) * bm
        centre = centre0 + drift * DRIFT_TAU * (1.0 - math.exp(-h / DRIFT_TAU))
        row = []
        for j, z in enumerate(Z):
            if j in (0, 4):
                q = centre + z * spread_outer
            elif j in (1, 3):
                q = centre + z * INNER_ZSCALE * spread
            else:
                q = centre
            row.append(_price6(q))
        # Enforce non-decreasing quantiles after rounding.
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir()
                   if p.is_file() and p.name.endswith(".in"))
    for path in files:
        case = json.loads(path.read_text(encoding="utf-8"))
        horizon = int(case["horizon_minutes"])
        forecasts = []
        for m in case["markets"]:
            prices = [float(v) for v in m["prices"]]
            if not prices:
                raise ValueError("market %s has no prices" % m["id"])
            forecasts.append({"id": m["id"],
                              "q": forecast_market(prices, horizon)})
        out = out_dir / (path.stem + ".out")
        out.write_text(json.dumps({"forecasts": forecasts}) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
