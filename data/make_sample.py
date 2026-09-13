"""Generate data/sample_weekly.csv, a synthetic weekly dataset in the schema the app expects.

Nothing here comes from client data. Scale, channel mix, seasonality and media response
are chosen to resemble a mid-size retail advertiser so the app produces sensible-looking
results end to end (train, forecast, backtest). Run:

    python data/make_sample.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260913
START, N_WEEKS = "2023-06-14", 118
OUT = Path(__file__).with_name("sample_weekly.csv")

# channel: (budget share at start, share at end, cost per impression,
#           adstock decay, effect size, half-saturation point)
CHANNELS = {
    "audio":     (0.020, 0.010, 0.140,   0.4, 0.004, 1.0),
    "search":    (0.680, 0.600, 0.0125,  0.2, 0.080, 0.8),
    "social":    (0.010, 0.008, 0.016,   0.3, 0.003, 1.0),
    "streaming": (0.050, 0.070, 0.0055,  0.5, 0.030, 1.2),
    "tv":        (0.190, 0.250, 0.00078, 0.6, 0.090, 1.3),
    "video":     (0.050, 0.062, 0.0009,  0.5, 0.020, 1.0),
}


def adstock(x, decay):
    out, carry = np.zeros_like(x), 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def main():
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, periods=N_WEEKS, freq="7D")
    n = N_WEEKS
    t = np.linspace(0, 1, n)
    ang = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25

    # Yearly seasonality peaking mid-December, plus a mild decline over the period
    season = 1 + 0.16 * np.cos(ang - 2 * np.pi * 350 / 365.25) + 0.05 * np.cos(2 * ang)
    drift = 1 - 0.07 * t
    demand = season * drift * (1 + rng.normal(0, 0.04, n))
    gqv = 1600 * demand  # query volume moves with underlying demand

    weekly_budget = 6.6e6 / 13 * (1 - 0.10 * t)
    spend, impr = {}, {}
    for ch, (s0, s1, cpi, _, _, _) in CHANNELS.items():
        base = weekly_budget * (s0 + (s1 - s0) * t)
        if ch == "search":
            base = base * (0.6 + 0.4 * demand)  # search spend tracks demand: the confound the control removes
        sp = base * rng.lognormal(0, 0.35 if ch in ("audio", "social") else 0.12, n)
        if ch == "audio":
            sp[rng.random(n) < 0.15] = 0.0  # dark weeks on the smallest channel
        cpi_t = cpi * (1 + 0.06 * np.sin(ang)) * rng.lognormal(0, 0.05, n)
        spend[ch] = sp
        impr[ch] = np.where(sp > 0, sp / cpi_t, 0.0)

    base_conv = 34000 * demand
    media_conv = np.zeros(n)
    for ch, (_, _, _, decay, beta, half) in CHANNELS.items():
        x = adstock(impr[ch], decay)
        x = x / x.mean()
        media_conv += beta * base_conv.mean() * x**1.5 / (x**1.5 + half**1.5)  # Hill saturation
    conversions = base_conv + media_conv + rng.normal(0, 1500, n)
    rev_per_conv = 113 * (1 + 0.03 * t) * (1 + rng.normal(0, 0.012, n))

    df = pd.DataFrame({"time": dates.strftime("%Y-%m-%d")})
    for ch in CHANNELS:
        df[f"{ch}_spend"] = spend[ch].round(2)
        df[f"{ch}_impression"] = impr[ch].round().astype(int)
    df["conversions"] = conversions.round().astype(int)
    df["revenue_per_conversion"] = rev_per_conv.round(4)
    df["gqv_control"] = gqv.round(2)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} weeks, {df.shape[1]} columns, media ~ {media_conv.sum() / conversions.sum():.0%} of conversions)")


if __name__ == "__main__":
    main()
