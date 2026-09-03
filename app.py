"""
MMM Budget Optimizer — Streamlit UI (Colab edition)
Runs inside Google Colab. Upload data, map columns, train the model,
and explore optimization results — all from the browser.

v3.0 — redesigned UI: top navigation, card layout, KPI tiles, unified chart theme.
Requires Streamlit >= 1.46 (st.navigation position="top").
"""

APP_VERSION = "3.0"

import html as _html
import json
import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Streamlit compat: use_container_width was deprecated in favour of width="stretch"
# ---------------------------------------------------------------------------

try:
    _VER = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _VER = (0, 0)
_FULL = {"width": "stretch"} if _VER >= (1, 49) else {"use_container_width": True}


def _df(data, **kw):
    try:
        return st.dataframe(data, **_FULL, **kw)
    except Exception:
        return st.dataframe(data, use_container_width=True, **kw)


def _btn(label, **kw):
    try:
        return st.button(label, **_FULL, **kw)
    except Exception:
        kw.pop("icon", None)
        return st.button(label, use_container_width=True, **kw)


def _plot(fig):
    try:
        return st.plotly_chart(fig, **_FULL, config={"displayModeBar": False})
    except Exception:
        return st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Heavy imports (cached — loaded once per process)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_deps():
    import tensorflow as tf
    import tensorflow_probability as tfp
    from meridian import constants
    from meridian.analysis import analyzer, optimizer
    from meridian.data import data_frame_input_data_builder
    from meridian.model import model as meridian_model, prior_distribution, spec
    from meridian.schema.serde import meridian_serde
    return dict(
        tf=tf, tfp=tfp, constants=constants,
        analyzer=analyzer, optimizer=optimizer,
        builder_cls=data_frame_input_data_builder.DataFrameInputDataBuilder,
        model_cls=meridian_model.Meridian,
        prior_cls=prior_distribution.PriorDistribution,
        spec_cls=spec.ModelSpec,
        serde=meridian_serde,
    )


# ---------------------------------------------------------------------------
# Page config + design system
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MMM Budget Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Design tokens (mirrored in CSS :root below)
INK, MUTED, LABEL, LINE, CANVAS = "#131916", "#6E756F", "#8B928C", "#E4E8E5", "#EEF0EE"
GREEN, LIME, YELLOW, PINK = "#16B34A", "#A6DD0F", "#F0CF1C", "#E5335A"
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="184" height="32" viewBox="0 0 184 32">
<rect width="32" height="32" rx="9" fill="#131916"/>
<rect x="8" y="20" width="16" height="4" rx="2" fill="#fff"/>
<rect x="8" y="14" width="11" height="4" rx="2" fill="#A6DD0F"/>
<rect x="8" y="8" width="6" height="4" rx="2" fill="#fff"/>
<text x="42" y="21.5" font-family="Inter,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif" font-size="15" font-weight="700" fill="#131916" letter-spacing="-0.2">MMM Optimizer</text>
</svg>"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{
  --bg:#EEF0EE; --card:#FFFFFF; --line:#E4E8E5; --ink:#131916; --muted:#6E756F; --label:#8B928C;
  --green:#16B34A; --lime:#A6DD0F; --yellow:#F0CF1C; --pink:#E5335A;
  --green-soft:#E6F7EB; --pink-soft:#FDE8ED; --gray-soft:#F3F5F3;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  font-family: Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: #fff; border-bottom: 1px solid var(--line); }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 1.1rem; padding-bottom: 4rem; max-width: 1440px; }

/* ---- top navigation: underline tabs ---- */
a[data-testid="stTopNavLink"] {
  background: transparent !important; border-radius: 0 !important;
  padding: 2px 2px 4px !important; margin: 0 9px !important;
  border-bottom: 2px solid transparent;
}
a[data-testid="stTopNavLink"] span { font-size: 13.5px !important; font-weight: 500 !important; color: var(--muted) !important; }
a[data-testid="stTopNavLink"]:hover span { color: var(--ink) !important; }
a[data-testid="stTopNavLink"][aria-current="page"] { border-bottom-color: var(--ink); }
a[data-testid="stTopNavLink"][aria-current="page"] span { color: var(--ink) !important; font-weight: 600 !important; }

/* ---- cards: any st.container(key="card…") ---- */
div[class*="st-key-card"] {
  background: var(--card) !important; border: 1px solid var(--line) !important;
  border-radius: 16px !important; padding: 18px 20px 20px !important;
  box-shadow: 0 1px 2px rgba(19,25,22,.04);
}
div[class*="st-key-card"] div[class*="st-key-card"] { box-shadow: none; }

/* ---- status strip under the header ---- */
.strip { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; margin: 0 0 14px; }
.chips { display:flex; gap:8px; flex-wrap:wrap; }
.chip { display:inline-flex; align-items:center; gap:7px; font-size:12px; font-weight:500; color:var(--ink);
        background:#fff; border:1px solid var(--line); border-radius:999px; padding:5px 11px; }
.chip.muted { color:var(--muted); }
.chip .dot { width:7px; height:7px; border-radius:50%; background:var(--green); }
.chip .dot.off { background:#C6CBC7; }
.chip .k { color:var(--label); font-weight:500; }
.meta { font-size:12px; color:var(--label); }

/* ---- page + card titles ---- */
.ptitle { font-size:22px; font-weight:700; letter-spacing:-0.01em; color:var(--ink); margin:2px 0 2px; line-height:1.2; }
.psub { font-size:13px; color:var(--muted); margin:0 0 16px; max-width:80ch; line-height:1.5; }
.ctitle { display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin: 0 0 6px; }
.ctitle h3 { font-size:15px; font-weight:600; color:var(--ink); margin:0; letter-spacing:-0.005em; }
.ctitle .hint { font-size:12px; color:var(--label); }
.ctitle .step { display:inline-flex; width:20px; height:20px; border-radius:6px; background:var(--ink); color:#fff; font-size:11px; font-weight:600;
                align-items:center; justify-content:center; margin-right:8px; vertical-align:middle; position:relative; top:-1px; }
.ctitle .step.done { background:var(--green); }

/* ---- KPI tiles ---- */
.kpis { display:flex; flex-wrap:wrap; }
.kpi { flex:1 1 150px; min-width:150px; padding: 2px 20px 2px 0; }
.kpi + .kpi { border-left:1px solid var(--line); padding-left:20px; }
.kpi-label { font-size:10.5px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--label); margin-bottom:6px; white-space:nowrap; }
.kpi-value { font-size:22px; font-weight:650; color:var(--ink); letter-spacing:-0.02em; line-height:1.15; display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; }
.kpi-delta { font-size:12px; font-weight:600; }
.kpi-delta.up { color:var(--green); } .kpi-delta.down { color:var(--pink); } .kpi-delta.flat { color:var(--muted); }
.kpi-sub { font-size:12px; color:var(--muted); margin-top:5px; }
.kpi-bar { height:4px; border-radius:99px; background:var(--gray-soft); margin-top:10px; overflow:hidden; }
.kpi-bar span { display:block; height:100%; border-radius:99px; }

/* ---- share (stacked) bars ---- */
.share { margin: 2px 0 4px; }
.share-name { font-size:12px; font-weight:600; color:var(--ink); margin-bottom:8px; }
.share-bar { display:flex; height:6px; border-radius:99px; overflow:hidden; gap:2px; background:var(--gray-soft); }
.share-bar span { display:block; height:100%; }
.share-rows { margin-top:12px; display:grid; grid-template-columns: 1fr; row-gap:7px; }
.share-row { display:flex; align-items:center; gap:10px; font-size:12.5px; color:var(--muted); }
.share-row .sw { width:8px; height:8px; border-radius:2px; flex:none; }
.share-row .nm { color:var(--ink); flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.share-row .val { color:var(--muted); font-variant-numeric: tabular-nums; }
.share-row .pct { color:var(--ink); font-weight:600; min-width:44px; text-align:right; font-variant-numeric: tabular-nums; }

/* ---- clean tables ---- */
table.mt { width:100%; border-collapse:collapse; font-size:13px; margin: 2px 0 4px; }
table.mt th { text-align:left; font-size:11px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--label);
              padding: 6px 10px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }
table.mt td { padding: 9px 10px; border-bottom:1px solid var(--gray-soft); color:var(--ink); font-variant-numeric: tabular-nums; }
table.mt tr:last-child td { border-bottom:none; }
table.mt th.r, table.mt td.r { text-align:right; }
table.mt .muted { color:var(--muted); }
.up { color:var(--green); font-weight:600; } .down { color:var(--pink); font-weight:600; } .flat { color:var(--muted); }
.ok { color:var(--green); font-weight:700; } .bad { color:var(--pink); font-weight:700; }
.tag { display:inline-block; font-size:11px; font-weight:600; padding:2px 8px; border-radius:999px; background:var(--gray-soft); color:var(--muted); }
.tag.g { background:var(--green-soft); color:#0E7A33; } .tag.p { background:var(--pink-soft); color:#B01F44; }

/* ---- empty state ---- */
.empty { text-align:left; padding: 6px 0 2px; }
.empty h3 { font-size:16px; font-weight:600; margin:0 0 4px; color:var(--ink); }
.empty p { font-size:13px; color:var(--muted); margin:0 0 10px; }

/* ---- buttons: black primary (matches config.toml primaryColor; fallback if config missing) ---- */
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
  background: var(--ink) !important; border-color: var(--ink) !important; color: #fff !important; font-weight: 600; }
.stButton button[kind="primary"]:hover { background: #2A322D !important; border-color: #2A322D !important; }
.stButton button[kind="primary"]:disabled { background: #C6CBC7 !important; border-color: #C6CBC7 !important; color:#fff !important; }
.stButton button[kind="secondary"] { font-weight: 500; }

/* ---- native widgets: quieter ---- */
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding: 12px 16px; }
[data-testid="stMetricLabel"] p { font-size:11px !important; letter-spacing:.06em; text-transform:uppercase; color:var(--label) !important; font-weight:600; }
[data-testid="stExpander"] { background:#fff; border-radius:12px; }
[data-testid="stExpander"] details { border-radius:12px; }
[data-testid="stAlert"] { border-radius:12px; }
div[data-testid="stCaptionContainer"] p, .stCaption { color: var(--muted); }
[data-testid="stWidgetLabel"] p { font-size:12.5px !important; font-weight:500; color:var(--ink); }
hr { margin: .6rem 0 1rem; border-color: var(--line); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

_DEFAULTS = dict(
    uploaded_df=None,
    uploaded_id=None,
    raw_df=None,
    long_defaults=None,
    national_df=None,
    mmm=None,
    model_trained=False,
    cfg=None,
    trend_multipliers=None,
    quarterly_df=None,
    last_results=None,
    backtest_results=None,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def S(key):
    return st.session_state[key]


def is_text(series):
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def model_ready():
    return S("model_trained") and S("mmm") is not None


# ---------------------------------------------------------------------------
# Quarter helpers
# ---------------------------------------------------------------------------

def quarter_to_date_range(label):
    year, q = int(label[:4]), int(label[-1])
    starts = {1: f"{year}-01-01", 2: f"{year}-04-01", 3: f"{year}-07-01", 4: f"{year}-10-01"}
    ends = {1: f"{year}-03-31", 2: f"{year}-06-30", 3: f"{year}-09-30", 4: f"{year}-12-31"}
    return starts[q], ends[q]


def get_corresponding_quarter(label):
    return f"{int(label[:4]) - 1}{label[-2:]}"


def quarter_week_counts(df, time_col):
    dt = pd.to_datetime(df[time_col])
    return dt.dt.to_period("Q").value_counts().sort_index()


def available_quarters(df, time_col):
    return [str(q) for q in quarter_week_counts(df, time_col).index]


def complete_quarters(df, time_col, frac=0.75):
    """Quarters with at least `frac` of a full quarter's weeks (drops partial first/last)."""
    wc = quarter_week_counts(df, time_col)
    if len(wc) == 0:
        return []
    threshold = frac * wc.max()
    return [str(q) for q, n in wc.items() if n >= threshold]


def forecastable_quarters(df, time_col, n_ahead=4):
    """Quarters that can be forecast: the quarter the data ends in (only if it is
    still in progress, i.e. fewer than 13 weeks observed) plus up to n_ahead
    quarters beyond it. Quarters already fully in the data are never offered —
    evaluating those is what Backtest is for. Each option's same-quarter-last-year
    must be a complete quarter. Returns (options, default_index) with default =
    first quarter after the data ends."""
    wc = quarter_week_counts(df, time_col)
    full = set(complete_quarters(df, time_col))
    if wc.empty:
        return [], 0
    last = pd.Period(str(wc.index[-1]), freq="Q")
    candidates = [str(last + i) for i in range(1, n_ahead + 1)]
    if wc.iloc[-1] < 13:  # data ends mid-quarter: the rest of that quarter is still ahead
        candidates.insert(0, str(last))
    options = [q for q in candidates if get_corresponding_quarter(q) in full]
    if not options:
        return [], 0
    first_future = str(last + 1)
    default = next((i for i, q in enumerate(options) if q >= first_future), 0)  # ISO-like labels sort correctly
    return options, default


def n_weeks_in_range(df, time_col, start, end):
    dt = pd.to_datetime(df[time_col])
    return int(((dt >= start) & (dt <= end)).sum())


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def normalize_time(df, time_col):
    """ISO date strings, sorted, deduplicated index."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col]).dt.strftime("%Y-%m-%d")
    return df.sort_values(time_col).reset_index(drop=True)


def season_col_names(k):
    """Names of the generated seasonal control columns for k harmonics."""
    return [f"season_{f}{i}" for i in range(1, k + 1) for f in ("sin", "cos")]


def season_values(dates, k):
    """Fourier terms of the yearly cycle, shape (n, 2k), ordered sin1, cos1, sin2, cos2, ...
    A given calendar date gets the same values every year, which is what lets the
    model learn 'December is high' from past Decembers and apply it to future ones."""
    doy = pd.DatetimeIndex(dates).dayofyear.values.astype(float)
    ang = 2 * np.pi * doy / 365.25
    cols = []
    for i in range(1, k + 1):
        cols += [np.sin(i * ang), np.cos(i * ang)]
    return np.column_stack(cols) if cols else np.zeros((len(doy), 0))


def add_season_cols(df, time_col, k):
    df = df.copy()
    names = season_col_names(k)
    if names:
        vals = season_values(pd.to_datetime(df[time_col]), k)
        for i, n in enumerate(names):
            df[n] = vals[:, i]
    return df


def aggregate_to_national(df, cfg):
    channels = cfg["channels"]
    media_cols = [f"{ch}{cfg['impression_suffix']}" for ch in channels]
    spend_cols = [f"{ch}{cfg['spend_suffix']}" for ch in channels]

    df = df.copy()
    df["_revenue"] = df[cfg["kpi_col"]] * df[cfg["rev_per_kpi_col"]]

    sum_cols = media_cols + spend_cols + cfg["organic_cols"] + [cfg["kpi_col"], "_revenue"]
    agg = {c: "sum" for c in sum_cols if c in df.columns}
    for c in cfg["non_media_cols"] + cfg["control_cols"]:
        if c in df.columns:
            agg[c] = "mean"

    nat = df.groupby(cfg["time_col"]).agg(agg).reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        nat[cfg["rev_per_kpi_col"]] = nat["_revenue"] / nat[cfg["kpi_col"]]
    nat[cfg["rev_per_kpi_col"]] = nat[cfg["rev_per_kpi_col"]].replace([np.inf, -np.inf], 0).fillna(0)
    return nat.drop(columns=["_revenue"])


def validate_national(df, cfg):
    """Pre-flight checks. Returns (errors, warnings)."""
    errors, warns = [], []
    channels = cfg["channels"]
    tcol = cfg["time_col"]

    # Time
    try:
        dt = pd.to_datetime(df[tcol])
    except Exception:
        errors.append(f"Time column `{tcol}` cannot be parsed as dates.")
        return errors, warns
    if dt.duplicated().any():
        errors.append(f"Duplicate dates in `{tcol}`. Set a geo column or remove duplicate rows.")
    if len(df) < 2:
        errors.append("Fewer than 2 rows of data.")
        return errors, warns
    gaps = dt.sort_values().diff().dropna().dt.days
    if not gaps.empty and (gaps.nunique() > 1 or gaps.iloc[0] != 7):
        warns.append(f"Dates are not evenly weekly (gaps: {sorted(gaps.unique().tolist())} days). Meridian assumes regular weekly periods.")
    if len(df) < 52:
        warns.append(f"{len(df)} weeks of data. Estimates may be unreliable with fewer than 52 weeks.")

    # Required numeric columns
    req = ([cfg["kpi_col"], cfg["rev_per_kpi_col"]]
           + [f"{ch}{cfg['impression_suffix']}" for ch in channels]
           + [f"{ch}{cfg['spend_suffix']}" for ch in channels]
           + cfg["non_media_cols"] + cfg["organic_cols"] + cfg["control_cols"])
    for c in req:
        if c not in df.columns:
            errors.append(f"Column `{c}` not found.")
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            errors.append(f"Column `{c}` is not numeric.")
            continue
        n_nan = int(df[c].isna().sum())
        if n_nan:
            errors.append(f"Column `{c}` has {n_nan} missing values.")
        if (df[c] < 0).any():
            warns.append(f"Column `{c}` has negative values.")

    if errors:
        return errors, warns

    for ch in channels:
        if df[f"{ch}{cfg['spend_suffix']}"].sum() <= 0:
            errors.append(f"Channel `{ch}` has zero total spend. Remove it from Channels.")
        if df[f"{ch}{cfg['impression_suffix']}"].sum() <= 0:
            errors.append(f"Channel `{ch}` has zero total impressions. Remove it from Channels.")
    if df[cfg["kpi_col"]].sum() <= 0:
        errors.append("KPI column sums to zero.")
    zero_kpi = int((df[cfg["kpi_col"]] <= 0).sum())
    if zero_kpi:
        warns.append(f"{zero_kpi} weeks have zero KPI.")
    if df[cfg["rev_per_kpi_col"]].sum() <= 0:
        errors.append("Revenue-per-KPI column sums to zero.")

    return errors, warns


TREND_MIN, TREND_MAX = 0.5, 2.0  # each year-over-year ratio is winsorized to halving / doubling


def compute_trend_multipliers(df, cfg):
    """Per-variable growth multipliers from complete quarters only.

    For each variable, every same-quarter year-over-year ratio in the data is
    winsorized to [TREND_MIN, TREND_MAX] and the geometric mean is taken (rates
    compound, so the geometric mean is the appropriate estimator; the cap keeps
    a near-zero denominator or one anomalous quarter on a small channel from
    dominating). With fewer than two years of complete quarters no year-over-year
    ratio exists and all multipliers are 1.0: quarter-over-quarter ratios are not
    used because they mix seasonality into the growth estimate.
    Returns (multipliers, quarterly_df)."""
    channels = cfg["channels"]
    tcol = cfg["time_col"]
    variables = (
        [f"{ch}_spend" for ch in channels]
        + [f"{ch}_cpi" for ch in channels]
        + cfg["non_media_cols"] + cfg["organic_cols"] + cfg["control_cols"] + ["rev_per_kpi"]
    )
    neutral = {v: 1.0 for v in variables}

    df = df.copy()
    df["time_dt"] = pd.to_datetime(df[tcol])
    df["quarter"] = df["time_dt"].dt.to_period("Q").astype(str)
    keep = set(complete_quarters(df, tcol))
    df = df[df["quarter"].isin(keep)]
    if df.empty:
        return neutral, pd.DataFrame()

    rows = []
    for q, g in df.groupby("quarter"):
        p = pd.Period(q, freq="Q")
        r = {"quarter": q, "year": p.year, "q_num": p.quarter, "n_weeks": len(g)}
        for ch in channels:
            ti = g[f"{ch}{cfg['impression_suffix']}"].sum()
            ts = g[f"{ch}{cfg['spend_suffix']}"].sum()
            r[f"{ch}_spend"] = ts / len(g)  # weekly rate, so 13- and 14-week quarters compare like for like
            r[f"{ch}_cpi"] = ts / ti if ti > 0 else 0
        for c in cfg["non_media_cols"] + cfg["organic_cols"] + cfg["control_cols"]:
            r[c] = g[c].mean()
        tk = g[cfg["kpi_col"]].sum()
        tr = (g[cfg["kpi_col"]] * g[cfg["rev_per_kpi_col"]]).sum()
        r["rev_per_kpi"] = tr / tk if tk > 0 else 0
        rows.append(r)
    qdf = pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)

    if qdf["year"].nunique() < 2:
        return neutral, qdf

    tm = {}
    for var in variables:
        rates = []
        for qn in qdf["q_num"].unique():
            vals = qdf[qdf["q_num"] == qn].sort_values("year")[var].values
            rates += [vals[i] / vals[i - 1] for i in range(1, len(vals)) if vals[i - 1] > 0]
        if not rates:
            tm[var] = 1.0
            continue
        capped = np.clip(np.asarray(rates, dtype=float), TREND_MIN, TREND_MAX)
        tm[var] = float(np.exp(np.mean(np.log(capped))))
    return tm, qdf


def build_forecast_config(df, trend_multipliers, cfg, corresponding_quarter):
    channels = cfg["channels"]
    df = df.copy()
    df["time_dt"] = pd.to_datetime(df[cfg["time_col"]])
    start, end = corresponding_quarter
    base = df[(df["time_dt"] >= start) & (df["time_dt"] <= end)]
    if base.empty:
        return None

    spend_cols = [f"{ch}{cfg['spend_suffix']}" for ch in channels]
    naive_budget = float(base[spend_cols].sum().sum())
    if naive_budget <= 0:
        return None
    naive_pct = (base[spend_cols].sum() / naive_budget).values

    naive_cpi = {}
    for ch in channels:
        ts = base[f"{ch}{cfg['spend_suffix']}"].sum()
        ti = base[f"{ch}{cfg['impression_suffix']}"].sum()
        naive_cpi[ch] = float(ts / ti) if ti > 0 else 0.0

    tk = base[cfg["kpi_col"]].sum()
    tr = (base[cfg["kpi_col"]] * base[cfg["rev_per_kpi_col"]]).sum()
    naive_rpk = float(tr / tk) if tk > 0 else 0.0
    naive_nm = {c: float(base[c].mean()) for c in cfg["non_media_cols"]}
    naive_org = {c: float(base[c].mean()) for c in cfg["organic_cols"]}
    naive_ctl = {c: float(base[c].mean()) for c in cfg["control_cols"]}

    g = lambda k: trend_multipliers.get(k, 1.0)
    # total budget grows at the spend-weighted average of the channel growth rates,
    # so a small channel's trend cannot swing the total
    spend_growth = float(sum(base[f"{ch}{cfg['spend_suffix']}"].sum() * g(f"{ch}_spend") for ch in channels) / naive_budget)
    return {
        "n_baseline_weeks": len(base),
        "total_budget": naive_budget * spend_growth,
        "spend_growth": spend_growth,
        "total_budget_naive": naive_budget,
        "spend_pct": {ch: float(p) for ch, p in zip(channels, naive_pct)},
        "cost_per_impression": {ch: naive_cpi[ch] * g(f"{ch}_cpi") for ch in channels},
        "cost_per_impression_naive": naive_cpi,
        "rev_per_kpi": naive_rpk * g("rev_per_kpi"),
        "rev_per_kpi_naive": naive_rpk,
        "non_media": {c: naive_nm[c] * g(c) for c in cfg["non_media_cols"]},
        "non_media_naive": naive_nm,
        "organic": {c: naive_org[c] * g(c) for c in cfg["organic_cols"]},
        "organic_naive": naive_org,
        "controls": {c: naive_ctl[c] * g(c) for c in cfg["control_cols"]},
        "controls_naive": naive_ctl,
        "n_future_weeks": cfg["n_future_weeks"],
    }


def build_future_data_tensors(fc, cfg, last_date):
    deps = _load_deps()
    tf = deps["tf"]
    channels = cfg["channels"]
    n_weeks, n_ch = fc["n_future_weeks"], len(channels)
    weekly = fc["total_budget"] / n_weeks

    ms = np.zeros((1, n_weeks, n_ch))
    m = np.zeros((1, n_weeks, n_ch))
    for c, ch in enumerate(channels):
        s = weekly * fc["spend_pct"][ch]
        cpi = fc["cost_per_impression"][ch]
        ms[0, :, c] = s
        m[0, :, c] = s / cpi if cpi > 0 else 0

    rpk = np.full((1, n_weeks), fc["rev_per_kpi"])
    nm = np.zeros((1, n_weeks, len(cfg["non_media_cols"])))
    for i, c in enumerate(cfg["non_media_cols"]):
        nm[0, :, i] = fc["non_media"][c]
    org = np.zeros((1, n_weeks, len(cfg["organic_cols"])))
    for i, c in enumerate(cfg["organic_cols"]):
        org[0, :, i] = fc["organic"][c]
    last_dt = pd.to_datetime(last_date)
    dates = [(last_dt + pd.Timedelta(weeks=w + 1)).strftime("%Y-%m-%d") for w in range(n_weeks)]

    season_cols = cfg.get("season_cols", [])
    ctl = np.zeros((1, n_weeks, len(cfg["control_cols"]) + len(season_cols)))
    for i, c in enumerate(cfg["control_cols"]):
        ctl[0, :, i] = fc["controls"][c]
    if season_cols:  # order must match training: user controls first, then seasonal terms
        ctl[0, :, len(cfg["control_cols"]):] = season_values(pd.to_datetime(dates), len(season_cols) // 2)

    td = dict(
        media=tf.convert_to_tensor(m, dtype=tf.float32),
        media_spend=tf.convert_to_tensor(ms, dtype=tf.float32),
        revenue_per_kpi=tf.convert_to_tensor(rpk, dtype=tf.float32),
        time=tf.convert_to_tensor(dates, dtype=tf.string),
    )
    if cfg["non_media_cols"]:
        td["non_media_treatments"] = tf.convert_to_tensor(nm, dtype=tf.float32)
    if cfg["organic_cols"]:
        td["organic_media"] = tf.convert_to_tensor(org, dtype=tf.float32)
    if ctl.shape[-1]:
        td["controls"] = tf.convert_to_tensor(ctl, dtype=tf.float32)
    return deps["analyzer"].DataTensors(**td), dates


MAX_SHIFT_PCT = 500  # slider ceiling; at this level the optimizer is effectively unconstrained


def constraint_label(c):
    """Human-readable spend constraint. Below 100% it is symmetric (±c); above,
    a channel may be cut to zero and grown by up to +c."""
    return f"±{c:.0%}" if c < 1 else f"0 to +{c:.0%}"


def shift_slider(key, label="Max shift per channel (%)"):
    return st.slider(label, 5, MAX_SHIFT_PCT, 30, 5, key=key,
                     help="Maximum change in each channel's spend relative to status quo. Up to 100% the bound is "
                          "symmetric. Above 100% a channel may be reduced to zero and increased by up to this amount; "
                          f"{MAX_SHIFT_PCT}% is effectively unconstrained.") / 100


def run_optimizer(mmm, fc, cfg, constraint=0.3, last_date=None):
    deps = _load_deps()
    if last_date is None:
        last_date = pd.to_datetime(S("national_df")[cfg["time_col"]]).max()
    future, _ = build_future_data_tensors(fc, cfg, last_date)
    bo = deps["optimizer"].BudgetOptimizer(mmm)
    return bo.optimize(
        new_data=future,
        budget=fc["total_budget"],
        pct_of_spend=list(fc["spend_pct"].values()),
        spend_constraint_lower=min(constraint, 1.0),  # Meridian: lower bound is a fraction of spend, max 1.0 (= zero)
        spend_constraint_upper=constraint,            # upper may exceed 1.0 (e.g. 3.0 = up to 4x current)
    )


def extract_results(res, fc, quarter, constraint):
    nonopt, opt = res.nonoptimized_data, res.optimized_data
    sq_t = float(nonopt.attrs["total_incremental_outcome"])
    op_t = float(opt.attrs["total_incremental_outcome"])
    return {
        "quarter": quarter,
        "sq_pct": nonopt.pct_of_spend.values * 100,
        "op_pct": opt.pct_of_spend.values * 100,
        "sq_spend": nonopt.spend.values,
        "op_spend": opt.spend.values,
        "sq_rev": nonopt.incremental_outcome.sel(metric="mean").values,
        "op_rev": opt.incremental_outcome.sel(metric="mean").values,
        "sq_roi": nonopt.roi.sel(metric="mean").values,
        "op_roi": opt.roi.sel(metric="mean").values,
        "sq_total": sq_t, "op_total": op_t, "gain": op_t - sq_t,
        "budget": fc["total_budget"], "constraint": constraint,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(national_df, cfg, log, holdout_mask=None):
    """Train a Meridian model. `holdout_mask` (bool array, one per week) marks weeks
    whose KPI is excluded from the likelihood — Meridian still sees their media
    (adstock carries over) but never their outcomes. Used by the backtest."""
    deps = _load_deps()
    channels = cfg["channels"]
    media_cols = [f"{ch}{cfg['impression_suffix']}" for ch in channels]
    spend_cols = [f"{ch}{cfg['spend_suffix']}" for ch in channels]

    log("Building input data")
    # Meridian's builder wants a 'time' column and will treat any 'geo' column as geo data,
    # so pass a clean frame containing only what the model needs.
    keep = [cfg["time_col"], cfg["kpi_col"], cfg["rev_per_kpi_col"]] + media_cols + spend_cols \
           + cfg["non_media_cols"] + cfg["organic_cols"] + cfg["control_cols"] + cfg.get("season_cols", [])
    df = national_df[keep].copy().rename(columns={cfg["time_col"]: "time"})
    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
    for c in df.columns:
        if c != "time":
            df[c] = df[c].astype(float)

    builder = deps["builder_cls"](
        kpi_type="non_revenue",
        default_kpi_column=cfg["kpi_col"],
        default_revenue_per_kpi_column=cfg["rev_per_kpi_col"],
    )
    builder = (builder.with_kpi(df).with_revenue_per_kpi(df)
               .with_media(df, media_cols=media_cols, media_spend_cols=spend_cols, media_channels=channels))
    if cfg["non_media_cols"]:
        builder = builder.with_non_media_treatments(df, non_media_treatment_cols=cfg["non_media_cols"])
    if cfg["organic_cols"]:
        builder = builder.with_organic_media(df, organic_media_cols=cfg["organic_cols"],
                                             organic_media_channels=cfg["organic_names"])
    ctl_cols = cfg["control_cols"] + cfg.get("season_cols", [])  # user controls first, then seasonal terms
    if ctl_cols:
        builder = builder.with_controls(df, control_cols=ctl_cols)
    data = builder.build()

    prior = deps["prior_cls"](
        roi_m=deps["tfp"].distributions.LogNormal(cfg["roi_mu"], cfg["roi_sigma"], name=deps["constants"].ROI_M)
    )
    spec_kw = dict(prior=prior)
    if holdout_mask is not None:
        spec_kw["holdout_id"] = np.asarray(holdout_mask, dtype=bool)
        log(f"Holdout: {int(spec_kw['holdout_id'].sum())} of {len(spec_kw['holdout_id'])} weeks excluded from the likelihood")
    mmm = deps["model_cls"](input_data=data, model_spec=deps["spec_cls"](**spec_kw))

    log(f"Sampling prior · {cfg['n_prior_samples']} draws")
    mmm.sample_prior(cfg["n_prior_samples"])

    log(f"Sampling posterior · {cfg['n_chains']} chains × {cfg['n_keep']} draws "
        f"({cfg['n_adapt']} adaptation, {cfg['n_burnin']} burn-in)")
    t0 = time.time()
    mmm.sample_posterior(n_chains=cfg["n_chains"], n_adapt=cfg["n_adapt"],
                         n_burnin=cfg["n_burnin"], n_keep=cfg["n_keep"], seed=cfg.get("seed", 42))
    log(f"Posterior sampling complete · {(time.time() - t0) / 60:.1f} min")
    return mmm


# ---------------------------------------------------------------------------
# Backtest engine (true holdout)
# ---------------------------------------------------------------------------

def accuracy_metrics(pred, act):
    pred, act = np.asarray(pred, float), np.asarray(act, float)
    nz = act != 0
    mape = float(np.mean(np.abs(pred[nz] - act[nz]) / np.abs(act[nz])) * 100) if nz.any() else float("nan")
    denom = float(np.sum(np.abs(act)))
    wmape = float(np.sum(np.abs(pred - act)) / denom * 100) if denom > 0 else float("nan")
    bias = float((pred.sum() - act.sum()) / act.sum() * 100) if act.sum() != 0 else float("nan")
    ss_tot = float(np.sum((act - act.mean()) ** 2))
    r2 = float(1 - np.sum((pred - act) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return dict(mape=mape, wmape=wmape, bias=bias, r2=r2)


def run_backtest(bt_q, national_df, cfg, quick, log):
    """True holdout backtest for one quarter.

    1. Keep only data up to the end of `bt_q`; nothing after it exists.
    2. Retrain the MMM with that quarter's weeks marked as holdout: the model sees
       the spend that ran (needed for adstock) but never the KPI.
    3. Model test — predict weekly revenue for the holdout weeks from the actual
       spend and compare with actual revenue.
    4. Forecast-layer test — trend multipliers and the same-quarter-last-year
       baseline are computed from data strictly before the quarter, exactly as
       the Forecast page would have at the time, and compared with what happened.
    """
    deps = _load_deps()
    channels, tcol = cfg["channels"], cfg["time_col"]
    sfx_s, sfx_i = cfg["spend_suffix"], cfg["impression_suffix"]
    q_start, q_end = quarter_to_date_range(bt_q)
    corr = quarter_to_date_range(get_corresponding_quarter(bt_q))

    dt_all = pd.to_datetime(national_df[tcol])
    upto = national_df[dt_all <= q_end].reset_index(drop=True)
    dt = pd.to_datetime(upto[tcol])
    mask = (dt >= q_start).values
    hist, actual = upto[~mask], upto[mask]
    if actual.empty or hist.empty:
        return None

    # Forecast layer from history only
    bt_trends, _ = compute_trend_multipliers(hist, cfg)
    fc = build_forecast_config(hist, bt_trends, cfg, corr)
    if fc is None:
        return None

    bt_cfg = dict(cfg)
    if quick:
        bt_cfg.update(n_chains=min(cfg["n_chains"], 2), n_keep=min(cfg["n_keep"], 250),
                      n_adapt=min(cfg["n_adapt"], 500), n_burnin=min(cfg["n_burnin"], 250))
    log(f"Fitting on {len(upto)} weeks through {upto[tcol].max()} · {int(mask.sum())} holdout weeks")
    mmm_bt = train_model(upto, bt_cfg, log, holdout_mask=mask)

    # Model test: expected revenue per week given the media that actually ran
    log("Computing expected revenue")
    an = deps["analyzer"].Analyzer(mmm_bt)
    eo = an.expected_outcome(aggregate_geos=True, aggregate_times=False, use_kpi=False)
    eo = eo.numpy() if hasattr(eo, "numpy") else np.asarray(eo)
    flat = eo.reshape(-1, eo.shape[-1])  # (chains*draws, n_times)
    model_times = pd.to_datetime(mmm_bt.input_data.time.values).strftime("%Y-%m-%d").tolist()
    dates = dt.dt.strftime("%Y-%m-%d").tolist()
    order = [model_times.index(d) for d in dates]  # align model time axis to our frame
    pred = flat[:, order]
    pred_mean, pred_lo, pred_hi = pred.mean(0), np.percentile(pred, 5, axis=0), np.percentile(pred, 95, axis=0)
    act_rev = (upto[cfg["kpi_col"]] * upto[cfg["rev_per_kpi_col"]]).values.astype(float)

    # Forecast-layer test: assumptions vs actuals
    spend_cols = [f"{ch}{sfx_s}" for ch in channels]
    a_spend = actual[spend_cols].sum()
    a_budget = float(a_spend.sum())
    a_pct = {ch: float(a_spend[f"{ch}{sfx_s}"] / a_budget) if a_budget > 0 else 0.0 for ch in channels}
    a_cpi = {}
    for ch in channels:
        ti = float(actual[f"{ch}{sfx_i}"].sum())
        a_cpi[ch] = float(a_spend[f"{ch}{sfx_s}"] / ti) if ti > 0 else 0.0
    a_kpi = float(actual[cfg["kpi_col"]].sum())
    a_rpk = float((actual[cfg["kpi_col"]] * actual[cfg["rev_per_kpi_col"]]).sum() / a_kpi) if a_kpi > 0 else 0.0
    a_ctl = {c: float(actual[c].mean()) for c in cfg.get("control_cols", [])}

    return {
        "quarter": bt_q, "q_start": q_start, "q_end": q_end,
        "n_train_weeks": int((~mask).sum()), "n_holdout_weeks": int(mask.sum()),
        "quick": bool(quick),
        "dates": dates, "mask": mask, "actual_rev": act_rev,
        "pred_mean": pred_mean, "pred_lo": pred_lo, "pred_hi": pred_hi,
        "test": accuracy_metrics(pred_mean[mask], act_rev[mask]),
        "train": accuracy_metrics(pred_mean[~mask], act_rev[~mask]),
        "fc": fc,
        "actual_budget": a_budget, "actual_pct": a_pct, "actual_cpi": a_cpi, "actual_rpk": a_rpk, "actual_controls": a_ctl,
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def default_save_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive"
    if os.path.isdir("/data"):
        return "/data"
    return os.getcwd()


def _jsonable(obj):
    """Recursively convert numpy containers/scalars so json.dump accepts the object."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _restore_arrays(obj):
    """Inverse of _jsonable for result dicts: lists of numbers/bools become numpy arrays."""
    if isinstance(obj, dict):
        return {k: _restore_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(isinstance(v, (bool, int, float)) for v in obj):
            return np.array(obj)
        return [_restore_arrays(v) for v in obj]
    return obj


def save_bundle(path):
    deps = _load_deps()
    deps["serde"].save_meridian(S("mmm"), path)
    cfg_path = path.replace(".binpb", "_config.json")
    export = dict(S("cfg"))
    export["app_version"] = APP_VERSION
    export["national_data"] = json.loads(S("national_df").to_json(orient="records", date_format="iso"))
    export["trend_multipliers"] = S("trend_multipliers")
    export["quarterly_data"] = json.loads(S("quarterly_df").to_json(orient="records"))
    export["last_results"] = _jsonable(S("last_results"))
    export["backtest_results"] = _jsonable(S("backtest_results"))
    with open(cfg_path, "w") as f:
        json.dump(export, f)
    return cfg_path


def load_bundle(path):
    deps = _load_deps()
    cfg_path = path.replace(".binpb", "_config.json")
    mmm = deps["serde"].load_meridian(path)
    with open(cfg_path) as f:
        saved = json.load(f)
    keys = ["time_col", "geo_col", "kpi_col", "rev_per_kpi_col", "channels", "impression_suffix",
            "spend_suffix", "non_media_cols", "organic_cols", "organic_names", "control_cols", "seasonality_k", "season_cols",
            "roi_mu", "roi_sigma", "n_chains", "n_adapt", "n_burnin", "n_keep", "n_prior_samples", "n_future_weeks", "seed"]
    cfg = {k: saved[k] for k in keys if k in saved}
    cfg.setdefault("control_cols", [])  # bundles saved before controls were supported
    cfg.setdefault("seasonality_k", 0)
    cfg.setdefault("season_cols", [])
    ndf = pd.DataFrame(saved["national_data"])
    for c in ndf.columns:
        if c != cfg["time_col"]:
            ndf[c] = pd.to_numeric(ndf[c], errors="coerce")
    ndf = normalize_time(ndf, cfg["time_col"])
    st.session_state.mmm = mmm
    st.session_state.model_trained = True
    st.session_state.cfg = cfg
    st.session_state.national_df = ndf
    st.session_state.trend_multipliers = saved["trend_multipliers"]
    st.session_state.quarterly_df = pd.DataFrame(saved["quarterly_data"])
    st.session_state.last_results = _restore_arrays(saved.get("last_results"))
    st.session_state.backtest_results = _restore_arrays(saved.get("backtest_results"))




# ---------------------------------------------------------------------------
# Chart theme
# ---------------------------------------------------------------------------

COLORS = {"sq": INK, "opt": GREEN, "gain": LIME, "cost": PINK}
PALETTE = [GREEN, LIME, YELLOW, PINK, INK, "#1BB8B0", "#F28C1A", "#8B5CF6"]


def money_scale(*arrays):
    """Pick a unit so the largest value reads nicely."""
    mx = max((float(np.nanmax(np.abs(a))) for a in arrays if len(a)), default=0)
    if mx >= 1e9:
        return 1e9, "$B"
    if mx >= 1e6:
        return 1e6, "$M"
    if mx >= 1e3:
        return 1e3, "$K"
    return 1.0, "$"


def fmt_money(v):
    """Compact money: $1.2M, $84.5K, $920."""
    v = float(v)
    a = abs(v)
    if a >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if a >= 1e6:
        return f"${v / 1e6:,.2f}M"
    if a >= 1e3:
        return f"${v / 1e3:,.1f}K"
    return f"${v:,.0f}"


def fmt_growth(mult):
    return f"{(mult - 1) * 100:+.1f}%"


def style_fig(fig, height=340, unified=True, legend=True):
    fig.update_layout(
        height=height, font=dict(family=FONT, size=12, color=MUTED),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30 if legend else 12, b=36, l=48, r=12),
        showlegend=legend,
        legend=dict(orientation="h", y=1.14, x=0, xanchor="left", bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12, color=INK), itemsizing="constant"),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(family=FONT, color=INK, size=12)),
        hovermode="x unified" if unified else "closest",
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=LINE, zeroline=False,
                     tickfont=dict(size=11, color=LABEL), title_font=dict(size=11, color=LABEL))
    fig.update_yaxes(showgrid=True, gridcolor=CANVAS, zeroline=False, showline=False,
                     tickfont=dict(size=11, color=LABEL), title_font=dict(size=11, color=LABEL))
    fig.update_annotations(font=dict(size=13, color=INK, family=FONT))
    return fig


def _round_bars(fig, r=5):
    try:
        fig.update_traces(marker_cornerradius=r, selector=dict(type="bar"))
    except Exception:
        pass


def _bar_pair(channels, sq, opt, ylabel):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Status quo", x=channels, y=sq, marker_color=COLORS["sq"]))
    fig.add_trace(go.Bar(name="Optimized", x=channels, y=opt, marker_color=COLORS["opt"]))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, yaxis_title=ylabel)
    _round_bars(fig)
    return style_fig(fig, height=320)


def _hex_rgba(h, a):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# ---------------------------------------------------------------------------
# HTML components (design system)
# ---------------------------------------------------------------------------

class H(str):
    """Marker: string is trusted HTML, don't escape."""


def _esc(v):
    return str(v) if isinstance(v, H) else _html.escape(str(v))


def md(s):
    st.markdown(s, unsafe_allow_html=True)


def card(key):
    """White rounded card. Usage: with card('card_x'): ..."""
    return st.container(border=False, key=key)


def page_title(title, sub=None):
    md(f'<div class="ptitle">{_esc(title)}</div>' + (f'<div class="psub">{_esc(sub)}</div>' if sub else '<div style="height:12px"></div>'))


def card_title(title, hint=None, step=None, done=False):
    badge = f'<span class="step{" done" if done else ""}">{step}</span>' if step is not None else ""
    hint_html = f'<span class="hint">{_esc(hint)}</span>' if hint else ""
    md(f'<div class="ctitle"><h3>{badge}{_esc(title)}</h3>{hint_html}</div>')


def kpi_row(items):
    """items: list of dicts with label, value, delta (str), delta_dir ('up'|'down'|'flat'), sub, bar (0..1), bar_color."""
    parts = []
    for it in items:
        d = ""
        if it.get("delta"):
            d = f'<span class="kpi-delta {it.get("delta_dir", "flat")}">{_esc(it["delta"])}</span>'
        sub = f'<div class="kpi-sub">{_esc(it["sub"])}</div>' if it.get("sub") else ""
        bar = ""
        if it.get("bar") is not None:
            pct = max(0.0, min(1.0, float(it["bar"]))) * 100
            bar = f'<div class="kpi-bar"><span style="width:{pct:.1f}%;background:{it.get("bar_color", GREEN)}"></span></div>'
        parts.append(f'<div class="kpi"><div class="kpi-label">{_esc(it["label"])}</div>'
                     f'<div class="kpi-value">{_esc(it["value"])}{d}</div>{sub}{bar}</div>')
    md('<div class="kpis">' + "".join(parts) + "</div>")


def share_bar(name, labels, pcts, values=None):
    """Stacked horizontal bar + legend rows (like 'Costs by category')."""
    segs, rows = [], []
    for i, (lab, p) in enumerate(zip(labels, pcts)):
        c = PALETTE[i % len(PALETTE)]
        segs.append(f'<span style="width:{max(float(p), 0):.2f}%;background:{c}"></span>')
        val = f'<span class="val">{_esc(values[i])}</span>' if values is not None else ""
        rows.append(f'<div class="share-row"><span class="sw" style="background:{c}"></span>'
                    f'<span class="nm">{_esc(lab)}</span>{val}<span class="pct">{float(p):.1f}%</span></div>')
    md(f'<div class="share"><div class="share-name">{_esc(name)}</div><div class="share-bar">{"".join(segs)}</div>'
       f'<div class="share-rows">{"".join(rows)}</div></div>')


def table(rows, right=(), cols=None):
    """rows: list of dicts (values may be H() for trusted html). right: column names to right-align."""
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    th = "".join(f'<th class="{"r" if c in right else ""}">{_esc(c)}</th>' for c in cols)
    body = "".join("<tr>" + "".join(f'<td class="{"r" if c in right else ""}">{_esc(r.get(c, ""))}</td>' for c in cols) + "</tr>" for r in rows)
    md(f'<table class="mt"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>')


def arrow(v, thr=0.5):
    """Colored ↑ / ↓ / — as trusted HTML."""
    if v > thr:
        return H('<span class="up">↑</span>')
    if v < -thr:
        return H('<span class="down">↓</span>')
    return H('<span class="flat">—</span>')


def signed(v, fmt="{:+.1f}", suffix=""):
    cls = "up" if v > 0 else ("down" if v < 0 else "flat")
    return H(f'<span class="{cls}">{fmt.format(v)}{suffix}</span>')


def header_strip():
    if model_ready():
        _c, _n = S("cfg"), S("national_df")
        tc = _c["time_col"]
        chips = [
            '<span class="chip"><span class="dot"></span>Model trained</span>',
            f'<span class="chip muted"><span class="k">Data</span>{_n[tc].nunique()} weeks</span>',
            f'<span class="chip muted"><span class="k">Range</span>{_esc(_n[tc].min())} → {_esc(_n[tc].max())}</span>',
            f'<span class="chip muted"><span class="k">Channels</span>{_esc(", ".join(_c["channels"]))}</span>',
        ]
    else:
        chips = ['<span class="chip muted"><span class="dot off"></span>No model loaded</span>']
    md(f'<div class="strip"><div class="chips">{"".join(chips)}</div>'
       f'<div class="meta">v{APP_VERSION} · Streamlit {st.__version__}</div></div>')


def require_model():
    if model_ready():
        return
    with card("card_empty"):
        md('<div class="empty"><h3>No model loaded</h3><p>Train a model or load a saved one to use this page.</p></div>')
        st.page_link(PAGE_CONFIG, label="Go to Configuration", icon=":material/arrow_forward:")
    st.stop()


def show_error(msg, exc=True):
    st.error(msg)
    if exc:
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# ===================================================================
# PAGE 1: Configuration
# ===================================================================

def page_config():
    page_title("Configuration")

    trained = model_ready()

    # ---- Model status + Load ----
    top_l, top_r = st.columns([3, 2], gap="medium")
    with top_l:
        with card("card_status"):
            card_title("Model status")
            if trained:
                _c, _n = S("cfg"), S("national_df")
                tc = _c["time_col"]
                n_q = len(complete_quarters(_n, tc))
                kpi_row([
                    {"label": "Status", "value": "Trained", "delta": "● in memory", "delta_dir": "up"},
                    {"label": "Weeks", "value": f"{_n[tc].nunique()}", "sub": f"{_n[tc].min()} → {_n[tc].max()}"},
                    {"label": "Complete quarters", "value": f"{n_q}"},
                    {"label": "Channels", "value": f"{len(_c['channels'])}", "sub": ", ".join(_c["channels"])},
                ])
            else:
                kpi_row([
                    {"label": "Status", "value": "None", "delta": "not loaded", "delta_dir": "flat"},
                    {"label": "Weeks", "value": "—"},
                    {"label": "Complete quarters", "value": "—"},
                    {"label": "Channels", "value": "—"},
                ])
    with top_r:
        with card("card_load"):
            card_title("Load a saved model")
            load_path = st.text_input("Model path", value=os.path.join(default_save_dir(), "saved_mmm.binpb"),
                                      label_visibility="collapsed", placeholder="/content/drive/MyDrive/saved_mmm.binpb")
            if _btn("Load model", icon=":material/folder_open:", type="primary" if not trained else "secondary"):
                loaded_ok = False
                try:
                    with st.spinner("Loading…"):
                        load_bundle(load_path)
                    loaded_ok = True
                except Exception as e:
                    show_error(f"Load failed: {e}", exc=False)
                if loaded_ok:
                    st.toast("Model loaded", icon=":material/check_circle:")
                    st.rerun()

    # ---- 1. Upload ----
    with card("card_upload"):
        card_title("Upload dataset", "CSV · weekly rows", step=1, done=S("uploaded_df") is not None)
        uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            uid = (uploaded.name, uploaded.size)
            if S("uploaded_id") != uid:  # only parse when a new file arrives
                st.session_state.uploaded_df = pd.read_csv(uploaded)
                st.session_state.uploaded_id = uid
                st.session_state.raw_df = None

        up_df = S("uploaded_df")
        if up_df is None:
            st.stop()

        st.caption(f"{up_df.shape[0]:,} rows × {up_df.shape[1]} columns")
        _df(up_df.head(8), height=250)
        up_cols = list(up_df.columns)

        # ---- Layout detection ----
        def _looks_long(df):
            """Long format = a date-like column with repeated values + a low-cardinality label column."""
            date_col = None
            for c in df.columns:
                if is_text(df[c]):
                    try:
                        parsed = pd.to_datetime(df[c], errors="coerce")
                        if parsed.notna().mean() > 0.9:
                            date_col = c
                            break
                    except Exception:
                        pass
            if date_col is None or not df[date_col].duplicated().any():
                return False
            return any(is_text(df[c]) and c != date_col and df[c].nunique() <= 30 for c in df.columns)

        lay_default = "Long" if _looks_long(up_df) else "Wide"
        lc0, lc1 = st.columns([1, 3], gap="medium")
        with lc0:
            layout = st.segmented_control("Data layout", ["Wide", "Long"], default=lay_default, key="layout_mode") or lay_default

        if layout == "Long":
            def _idx(name, fallback=0):
                return up_cols.index(name) if name in up_cols else fallback

            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                l_time = st.selectbox("Time column", up_cols, index=_idx("week"), key="l_time")
                l_channel = st.selectbox("Channel column", up_cols, index=_idx("channel"), key="l_ch")
            with lc2:
                l_spend = st.selectbox("Spend column", up_cols, index=_idx("cost"), key="l_spend")
                l_impr = st.selectbox("Media metric column (impressions or clicks)", up_cols, index=_idx("impr"), key="l_impr")
            with lc3:
                l_geo = st.selectbox("Geo column (optional)", ["(none)"] + up_cols, index=0, key="l_geo")
                l_geo = None if l_geo == "(none)" else l_geo

            if len({l_time, l_channel, l_spend, l_impr}) < 4:
                st.error("Time, channel, spend and media-metric must be four different columns.")
                st.stop()

            keys = [l_time] + ([l_geo] if l_geo else [])
            context_cols = [c for c in up_cols if c not in {l_time, l_channel, l_spend, l_impr, l_geo}]

            ch_spend = up_df.groupby(l_channel)[l_spend].sum()
            dead = ch_spend[ch_spend <= 0].index.tolist()
            if dead:
                st.warning(f"Channels with zero total spend excluded: {', '.join(map(str, dead))}")
            keep = up_df[~up_df[l_channel].isin(dead)]
            if keep.empty:
                st.error("No channels with spend.")
                st.stop()

            wide_spend = keep.pivot_table(index=keys, columns=l_channel, values=l_spend, aggfunc="sum")
            wide_impr = keep.pivot_table(index=keys, columns=l_channel, values=l_impr, aggfunc="sum")
            wide_spend.columns = [f"{c}_spend" for c in wide_spend.columns]
            wide_impr.columns = [f"{c}_impression" for c in wide_impr.columns]

            week_level, per_channel = [], []
            if context_cols:
                nun = up_df.groupby(keys)[context_cols].nunique()
                week_level = [c for c in context_cols if (nun[c] <= 1).all()]
                per_channel = [c for c in context_cols if c not in week_level]

            parts = [wide_impr, wide_spend]
            if week_level:
                parts.insert(0, up_df.groupby(keys)[week_level].first())
            for c in per_channel:
                p = keep.pivot_table(index=keys, columns=l_channel, values=c, aggfunc="sum")
                p.columns = [f"{ch}_{c}" for ch in p.columns]
                parts.append(p)
            if per_channel:
                st.caption(f"Per-channel columns pivoted: {', '.join(per_channel)}")

            raw_df = pd.concat(parts, axis=1).reset_index().fillna(0)
            st.session_state.raw_df = raw_df
            st.session_state.long_defaults = dict(time=l_time, week_level=week_level, geo=l_geo)
            st.success(f"Pivoted to wide layout: {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")
            _df(raw_df.head(5), height=200)
        else:
            st.session_state.raw_df = up_df
            st.session_state.long_defaults = None

    raw_df = S("raw_df")
    all_cols = list(raw_df.columns)

    # ---- 2. Map columns ----
    with card("card_map"):
        card_title("Map columns", step=2)
        ld = S("long_defaults") or {}

        sx1, sx2 = st.columns(2)
        with sx1:
            imp_suffix = st.text_input("Impression column suffix", value="_impression")
        with sx2:
            spend_suffix = st.text_input("Spend column suffix", value="_spend")
        if not imp_suffix or not spend_suffix:
            st.error("Suffixes cannot be empty.")
            st.stop()

        detected = sorted(
            c[:-len(imp_suffix)] for c in all_cols
            if c.endswith(imp_suffix) and c[:-len(imp_suffix)] + spend_suffix in all_cols
        )
        media_like = {f"{ch}{imp_suffix}" for ch in detected} | {f"{ch}{spend_suffix}" for ch in detected}

        # Sensible defaults: time = long-format time col or first date-like col; KPI/rpk = first numeric non-media cols
        def _first_date_col():
            if ld.get("time") in all_cols:
                return ld["time"]
            for c in all_cols:
                if is_text(raw_df[c]):
                    try:
                        if pd.to_datetime(raw_df[c], errors="coerce").notna().mean() > 0.9:
                            return c
                    except Exception:
                        pass
            return all_cols[0]

        t_default = _first_date_col()
        numeric_ctx = [c for c in (ld.get("week_level") or all_cols)
                       if c in all_cols and c != t_default and c not in media_like
                       and pd.api.types.is_numeric_dtype(raw_df[c])]
        kpi_default = numeric_ctx[0] if numeric_ctx else all_cols[0]
        rpk_default = numeric_ctx[1] if len(numeric_ctx) > 1 else kpi_default

        def _sb(label, options, default, key):
            return st.selectbox(label, options, index=options.index(default) if default in options else 0, key=key)

        mc1, mc2 = st.columns(2)
        with mc1:
            time_col = _sb("Time column", all_cols, t_default, "m_time")
            kpi_col = _sb("KPI column (e.g. leads, conversions)", all_cols, kpi_default, "m_kpi")
            rpk_col = _sb("Revenue per KPI column", all_cols, rpk_default, "m_rpk")
        with mc2:
            geo_opts = ["(none)"] + all_cols
            geo_default = ld.get("geo") or ("geo" if "geo" in all_cols else "(none)")
            geo_col = _sb("Geo column (leave as none if national)", geo_opts, geo_default, "m_geo")
            geo_col = None if geo_col == "(none)" else geo_col
            channels = st.multiselect("Channels (auto-detected from suffixes)", detected, default=detected, key="m_ch")

        if len({time_col, kpi_col, rpk_col}) < 3:
            st.error("Time, KPI and revenue-per-KPI must be three different columns.")
            st.stop()
        if not detected:
            st.error(f"No channel pairs found. Need columns like `X{imp_suffix}` and `X{spend_suffix}`.")
            st.stop()
        if not channels:
            st.warning("Select at least one channel.")
            st.stop()

        used = {time_col, kpi_col, rpk_col, geo_col} | {f"{ch}{imp_suffix}" for ch in channels} | {f"{ch}{spend_suffix}" for ch in channels}
        remaining = [c for c in all_cols if c not in used and pd.api.types.is_numeric_dtype(raw_df[c])]
        rm1, rm2, rm3 = st.columns(3)
        with rm1:
            non_media_cols = st.multiselect("Non-media treatment columns", remaining, key="m_nm",
                                            help="Treatments under the advertiser's control, e.g. price or promotion depth. An incremental effect is estimated.")
        with rm2:
            organic_cols = st.multiselect("Organic media columns", [c for c in remaining if c not in non_media_cols], key="m_org")
        with rm3:
            control_cols = st.multiselect("Control columns", [c for c in remaining if c not in non_media_cols and c not in organic_cols],
                                          key="m_ctl", help="External covariates, e.g. Google query volume, weather, macro indices. "
                                                            "Included as controls; no incremental effect is estimated.")
        organic_names = [c[:-len(imp_suffix)] if c.endswith(imp_suffix) else c for c in organic_cols]

    # ---- 3. Model settings ----
    with card("card_settings"):
        card_title("Model settings", step=3)
        ms1, ms2, ms3, ms4 = st.columns(4)
        with ms1:
            roi_mu = st.number_input("ROAS prior μ (log-normal)", value=0.2, step=0.1, format="%.2f")
        with ms2:
            roi_sigma = st.number_input("ROAS prior σ", value=0.9, min_value=0.05, step=0.1, format="%.2f")
        with ms3:
            n_future_weeks = st.number_input("Forecast horizon (weeks)", value=13, min_value=1, max_value=52)
        with ms4:
            season_k = st.number_input("Seasonality harmonics", value=2, min_value=0, max_value=4,
                                       help="Yearly sine/cosine terms added as control variables (0 = off). Each harmonic adds two terms.")
        with st.expander("Sampling (MCMC)"):
            sm1, sm2, sm3, sm4 = st.columns(4)
            with sm1:
                n_chains = st.number_input("Chains", value=4, min_value=1, max_value=10)
            with sm2:
                n_keep = st.number_input("Samples kept per chain", value=500, min_value=100, step=100)
            with sm3:
                n_adapt = st.number_input("Adaptation steps", value=1500, min_value=100, step=100)
            with sm4:
                n_burnin = st.number_input("Burn-in steps", value=500, min_value=100, step=100)

    cfg = dict(
        time_col=time_col, geo_col=geo_col, kpi_col=kpi_col, rev_per_kpi_col=rpk_col,
        channels=channels, impression_suffix=imp_suffix, spend_suffix=spend_suffix,
        non_media_cols=non_media_cols, organic_cols=organic_cols, organic_names=organic_names,
        control_cols=control_cols, seasonality_k=int(season_k), season_cols=season_col_names(int(season_k)),
        roi_mu=float(roi_mu), roi_sigma=float(roi_sigma),
        n_chains=int(n_chains), n_adapt=int(n_adapt), n_burnin=int(n_burnin), n_keep=int(n_keep),
        n_prior_samples=500, n_future_weeks=int(n_future_weeks), seed=42,
    )

    # ---- 4. Pre-flight + train ----
    with card("card_train"):
        card_title("Train model", step=4, done=trained)

        # Build the national frame now so we can validate before the user commits to a 10-minute train
        if geo_col and raw_df[geo_col].nunique() > 1:
            candidate = aggregate_to_national(raw_df, cfg)
            geo_note = f"{raw_df[geo_col].nunique()} geos aggregated to national"
        else:
            candidate = raw_df.drop(columns=[geo_col]) if geo_col else raw_df.copy()
            geo_note = "national data"
        try:
            candidate = normalize_time(candidate, time_col)
        except Exception as e:
            st.error(f"Time column `{time_col}` cannot be parsed as dates: {e}")
            st.stop()

        candidate = add_season_cols(candidate, time_col, cfg["seasonality_k"])
        errors, warns = validate_national(candidate, cfg)
        n_q = len(complete_quarters(candidate, time_col))

        kpi_row([
            {"label": "Weeks", "value": f"{len(candidate)}", "sub": f"{candidate[time_col].min()} → {candidate[time_col].max()}"},
            {"label": "Complete quarters", "value": f"{n_q}", "sub": geo_note,
             "delta": "" if n_q >= 8 else "< 2 years, no growth", "delta_dir": "down"},
            {"label": "Channels", "value": f"{len(channels)}", "sub": ", ".join(channels)},
            {"label": "Pre-flight", "value": "Ready" if not errors else f"{len(errors)} error{'s' if len(errors) > 1 else ''}",
             "delta": f"{len(warns)} warning{'s' if len(warns) != 1 else ''}" if warns else "",
             "delta_dir": "flat", "bar": 1.0 if not errors else 0.25, "bar_color": GREEN if not errors else PINK},
        ])
        for w in warns:
            st.warning(w)
        for e in errors:
            st.error(e)
        if n_q < 8:
            st.caption("Fewer than two years of complete quarters: no year-over-year ratios, so all growth multipliers are 1.0.")

        if _btn("Train model", icon=":material/rocket_launch:", type="primary", disabled=bool(errors)):
            st.session_state.national_df = candidate
            st.session_state.cfg = cfg
            tm, qdf = compute_trend_multipliers(candidate, cfg)
            st.session_state.trend_multipliers = tm
            st.session_state.quarterly_df = qdf

            ok = False
            with st.status("Training model…", expanded=True) as status:
                try:
                    mmm = train_model(candidate, cfg, lambda m: st.write(m))
                    st.session_state.mmm = mmm
                    st.session_state.model_trained = True
                    for k in ("last_results", "backtest_results"):
                        st.session_state[k] = None
                    status.update(label="Model trained", state="complete", expanded=False)
                    ok = True
                except Exception as e:
                    status.update(label="Training failed", state="error", expanded=True)
                    show_error(f"Training failed: {e}")
            if ok:
                st.toast("Model trained", icon=":material/check_circle:")
                st.rerun()

    # ---- 5. Save ----
    if trained:
        with card("card_save"):
            card_title("Save model", step=5)
            if not os.path.isdir("/content/drive/MyDrive") and os.path.isdir("/content"):
                st.warning("Google Drive is not mounted. Files saved to the runtime are lost when it disconnects.")
            sc1, sc2 = st.columns([4, 1], vertical_alignment="bottom")
            with sc1:
                save_path = st.text_input("Save path", value=os.path.join(default_save_dir(), "saved_mmm.binpb"))
            with sc2:
                do_save = _btn("Save model", icon=":material/save:", type="primary")
            if do_save:
                try:
                    with st.spinner("Saving…"):
                        cfg_path = save_bundle(save_path)
                    st.toast("Model saved", icon=":material/check_circle:")
                    st.success(f"Saved `{save_path}` and `{cfg_path}`")
                except Exception as e:
                    show_error(f"Save failed: {e}", exc=False)


# ===================================================================
# PAGE 2: Forecast
# ===================================================================

def page_forecast():
    page_title("Forecast & optimize")
    require_model()

    mmm, cfg = S("mmm"), S("cfg")
    national_df, trend_multipliers = S("national_df"), S("trend_multipliers")
    channels = cfg["channels"]

    fq_opts, fq_def = forecastable_quarters(national_df, cfg["time_col"])
    if not fq_opts:
        st.error("Need at least one complete quarter plus its same-quarter-last-year to forecast.")
        st.stop()

    # ---- Setup ----
    with card("card_fc_setup"):
        card_title("Forecast setup")
        q1, q2, q3 = st.columns([1, 1, 1], gap="medium")
        with q1:
            if "fc_q" not in st.session_state and S("last_results") and S("last_results").get("quarter") in fq_opts:
                fq_def = fq_opts.index(S("last_results")["quarter"])  # after a load, open on the quarter the saved result is for
            forecast_q = st.selectbox("Forecast quarter", fq_opts, index=fq_def, key="fc_q")
        with q2:
            growth_mult = st.slider("Growth rate multiplier", 0.8, 1.3, 1.0, 0.01, key="fc_gm",
                                    help="Scales all growth multipliers. 1.0 applies historical growth; 0.8 applies 80% of it.")
        with q3:
            constraint = shift_slider("fc_con")

        corr_label = get_corresponding_quarter(forecast_q)
        corr_quarter = quarter_to_date_range(corr_label)
        adj_trends = {k: 1.0 + (v - 1.0) * growth_mult for k, v in trend_multipliers.items()}
        base = build_forecast_config(national_df, adj_trends, cfg, corr_quarter)
        if base is None:
            st.error(f"No usable baseline data for {corr_label}.")
            st.stop()


    # ---- Assumptions ----
    wk = f"{forecast_q}_{growth_mult:.2f}"  # inputs keyed on (quarter, growth) so they reset when either changes
    with card("card_fc_params"):
        card_title("Assumptions", f"Defaults from {corr_label}")
        spend_g = base["spend_growth"]
        kpi_row([
            {"label": "Budget", "value": fmt_money(base["total_budget"]),
             "delta": fmt_growth(spend_g), "delta_dir": "up" if spend_g >= 1 else "down",
             "sub": f"Prior year: ${base['total_budget_naive']:,.0f}"},
            {"label": "Revenue per KPI", "value": f"${base['rev_per_kpi']:,.2f}",
             "delta": fmt_growth(adj_trends.get("rev_per_kpi", 1.0)),
             "delta_dir": "up" if adj_trends.get("rev_per_kpi", 1.0) >= 1 else "down",
             "sub": f"Prior year: ${base['rev_per_kpi_naive']:,.4f}"},
            {"label": "Horizon", "value": f"{base['n_future_weeks']} wk", "sub": f"Baseline {base['n_baseline_weeks']} wk"},
            {"label": "Channels", "value": f"{len(channels)}", "sub": f"{constraint_label(constraint)} max shift"},
        ])
        md('<div style="height:6px"></div>')
        p1, p2 = st.columns(2)
        with p1:
            total_budget = st.number_input("Total budget ($)", min_value=0.0, value=float(base["total_budget"]),
                                           step=10000.0, format="%.0f", key=f"budget_{wk}")
        with p2:
            rev_per_kpi = st.number_input("Revenue per KPI ($)", min_value=0.0, value=float(base["rev_per_kpi"]),
                                          step=0.01, format="%.4f", key=f"rpk_{wk}")

        with st.expander("Per-channel cost per impression"):
            cpi = {}
            cols = st.columns(min(len(channels), 3))
            for i, ch in enumerate(channels):
                with cols[i % len(cols)]:
                    cpi[ch] = st.number_input(ch, min_value=0.0, value=float(base["cost_per_impression"][ch]),
                                              step=0.0001, format="%.6f", key=f"cpi_{ch}_{wk}")
                    st.caption(f"Prior year: {base['cost_per_impression_naive'][ch]:.6f}")

        nm = {}
        if cfg["non_media_cols"]:
            with st.expander("Non-media treatment levels"):
                cols = st.columns(min(len(cfg["non_media_cols"]), 3))
                for i, c in enumerate(cfg["non_media_cols"]):
                    with cols[i % len(cols)]:
                        nm[c] = st.number_input(c, value=float(base["non_media"][c]), step=0.01, format="%.4f", key=f"nm_{c}_{wk}")
                        st.caption(f"Prior year: {base['non_media_naive'][c]:.4f}")

        ctl = {}
        if cfg.get("control_cols"):
            with st.expander("Control levels"):
                cols = st.columns(min(len(cfg["control_cols"]), 3))
                for i, c in enumerate(cfg["control_cols"]):
                    with cols[i % len(cols)]:
                        ctl[c] = st.number_input(c, value=float(base["controls"][c]), step=0.01, format="%.4f", key=f"ctl_{c}_{wk}")
                        st.caption(f"Prior year: {base['controls_naive'][c]:.4f}")

        with st.expander("Growth multipliers applied"):
            rows = [{"Variable": f"{ch} spend", "Historical": fmt_growth(trend_multipliers.get(f"{ch}_spend", 1)),
                     "Applied": signed((adj_trends.get(f"{ch}_spend", 1) - 1) * 100, suffix="%")} for ch in channels]
            rows += [{"Variable": f"{ch} CPI", "Historical": fmt_growth(trend_multipliers.get(f"{ch}_cpi", 1)),
                      "Applied": signed((adj_trends.get(f"{ch}_cpi", 1) - 1) * 100, suffix="%")} for ch in channels]
            rows.append({"Variable": "Revenue per KPI", "Historical": fmt_growth(trend_multipliers.get("rev_per_kpi", 1)),
                         "Applied": signed((adj_trends.get("rev_per_kpi", 1) - 1) * 100, suffix="%")})
            rows += [{"Variable": c, "Historical": fmt_growth(trend_multipliers.get(c, 1)),
                      "Applied": signed((adj_trends.get(c, 1) - 1) * 100, suffix="%")}
                     for c in cfg["non_media_cols"] + cfg["organic_cols"] + cfg.get("control_cols", [])]
            table(rows, right={"Historical", "Applied"})

        final = dict(base)
        final["total_budget"] = float(total_budget)
        final["rev_per_kpi"] = float(rev_per_kpi)
        final["cost_per_impression"] = cpi
        if nm:
            final["non_media"] = nm
        if ctl:
            final["controls"] = ctl

        if _btn("Run optimizer", icon=":material/play_arrow:", type="primary", disabled=total_budget <= 0):
            try:
                with st.spinner("Optimizing…"):
                    res = run_optimizer(mmm, final, cfg, constraint)
                st.session_state.last_results = extract_results(res, final, forecast_q, constraint)
            except Exception as e:
                show_error(f"Optimizer failed: {e}")

    # ---- Results ----
    r = S("last_results")
    if not r:
        return

    if r["quarter"] != forecast_q:
        st.info(f"Results shown for {r['quarter']}. Run the optimizer to update for {forecast_q}.")

    gp = r["gain"] / r["sq_total"] * 100 if r["sq_total"] else 0
    with card("card_fc_kpis"):
        card_title(f"Results — {r['quarter']}",
                   f"${r['budget']:,.0f} budget · channels held within {constraint_label(r['constraint'])} of status quo")
        kpi_row([
            {"label": "Status quo incremental revenue", "value": fmt_money(r["sq_total"]),
             "bar": r["sq_total"] / max(r["op_total"], r["sq_total"], 1e-9), "bar_color": INK},
            {"label": "Optimized incremental revenue", "value": fmt_money(r["op_total"]),
             "bar": r["op_total"] / max(r["op_total"], r["sq_total"], 1e-9), "bar_color": GREEN},
            {"label": "Gain from reallocation", "value": fmt_money(r["gain"]),
             "delta": f"{gp:+.1f}%", "delta_dir": "up" if r["gain"] > 0 else ("down" if r["gain"] < 0 else "flat")},
            {"label": "Blended ROAS", "value": f"{r['op_total'] / r['budget']:.2f}x" if r["budget"] else "—",
             "sub": f"Status quo {r['sq_total'] / r['budget']:.2f}x" if r["budget"] else ""},
        ])

    a1, a2 = st.columns([2, 3], gap="medium")
    with a1:
        with card("card_fc_alloc"):
            card_title("Budget allocation", "% of budget")
            share_bar("Status quo", channels, r["sq_pct"], [fmt_money(v) for v in r["sq_spend"]])
            md('<div style="height:10px"></div>')
            share_bar("Optimized", channels, r["op_pct"], [fmt_money(v) for v in r["op_spend"]])
    with a2:
        with card("card_fc_table"):
            card_title("Channel results")
            rows = []
            for i, ch in enumerate(channels):
                sq, op = r["sq_pct"][i], r["op_pct"][i]
                rows.append({"Channel": ch,
                             "Status quo": H(f"{sq:.1f}% <span class='muted'>({fmt_money(r['sq_spend'][i])})</span>"),
                             "Optimized": H(f"{op:.1f}% <span class='muted'>({fmt_money(r['op_spend'][i])})</span>"),
                             "Change": signed(op - sq, suffix=" pp"),
                             "SQ ROAS": f"{r['sq_roi'][i]:.2f}",
                             "Opt ROAS": f"{r['op_roi'][i]:.2f}"})
            table(rows, right={"Status quo", "Optimized", "Change", "SQ ROAS", "Opt ROAS"})

    div, unit = money_scale(r["sq_rev"], r["op_rev"])
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with card("card_fc_ch1"):
            card_title("Allocation", "% of budget")
            _plot(_bar_pair(channels, r["sq_pct"], r["op_pct"], "% of budget"))
    with c2:
        with card("card_fc_ch2"):
            card_title("Revenue contribution", f"Incremental, {unit}")
            _plot(_bar_pair(channels, r["sq_rev"] / div, r["op_rev"] / div, unit))
    with c3:
        with card("card_fc_ch3"):
            card_title("Return on ad spend", "ROAS")
            _plot(_bar_pair(channels, r["sq_roi"], r["op_roi"], "ROAS"))


# ===================================================================
# PAGE 3: Backtest
# ===================================================================

def page_backtest():
    page_title("Backtest")
    require_model()

    cfg = S("cfg")
    national_df = S("national_df")
    channels = cfg["channels"]
    tcol = cfg["time_col"]

    full = complete_quarters(national_df, tcol)
    eligible = [q for q in full if get_corresponding_quarter(q) in full]
    if not eligible:
        st.warning("Need at least two complete years of data for backtesting.")
        st.stop()

    with card("card_bt_setup"):
        card_title("Backtest setup")
        b1, b2 = st.columns([3, 1], gap="medium", vertical_alignment="bottom")
        with b1:
            bt_qs = st.multiselect("Quarters to backtest", eligible, default=[eligible[-1]])
        with b2:
            quick = st.toggle("Quick sampling", value=False, key="bt_quick",
                              help="At most 2 chains × 250 draws.")

        if _btn("Run backtest", icon=":material/replay:", type="primary", disabled=not bt_qs):
            out = []
            with st.status("Backtesting…", expanded=True) as status:
                try:
                    for bt_q in bt_qs:
                        st.write(f"**{bt_q}**")
                        r = run_backtest(bt_q, national_df, cfg, quick, lambda m: st.write(m))
                        if r is None:
                            st.warning(f"{bt_q} skipped: insufficient history.")
                            continue
                        out.append(r)
                    status.update(label="Backtest complete", state="complete", expanded=False)
                    st.session_state.backtest_results = out
                except Exception as e:
                    status.update(label="Backtest failed", state="error", expanded=True)
                    show_error(f"Backtest failed: {e}")

    if not S("backtest_results"):
        return

    summary = []
    for j, bt in enumerate(S("backtest_results")):
        m = bt["mask"]
        t, tr = bt["test"], bt["train"]
        pred_q, act_q = float(bt["pred_mean"][m].sum()), float(bt["actual_rev"][m].sum())
        good = t["wmape"] <= 15

        with card(f"card_bt_{j}"):
            card_title(f"Quarter {bt['quarter']}",
                       f"Fitted on {bt['n_train_weeks']} weeks · {bt['n_holdout_weeks']} holdout weeks"
                       + (" · reduced sampling" if bt["quick"] else ""))
            kpi_row([
                {"label": "Predicted revenue (holdout)", "value": fmt_money(pred_q),
                 "delta": signed(t["bias"], suffix="%"), "delta_dir": "flat" if abs(t["bias"]) <= 5 else "down",
                 "sub": "Expected revenue under actual spend"},
                {"label": "Actual revenue", "value": fmt_money(act_q), "sub": f"{bt['q_start']} → {bt['q_end']}"},
                {"label": "Holdout wMAPE", "value": f"{t['wmape']:.1f}%", "sub": f"In-sample {tr['wmape']:.1f}%",
                 "bar": max(0.0, 1 - t["wmape"] / 50), "bar_color": GREEN if good else PINK},
                {"label": "Holdout R²", "value": f"{t['r2']:.2f}", "sub": f"In-sample {tr['r2']:.2f}"},
            ])

            # Weekly predicted vs actual, last year of training + holdout
            n_show = min(len(bt["dates"]), bt["n_holdout_weeks"] + 52)
            sl = slice(len(bt["dates"]) - n_show, None)
            x = bt["dates"][sl]
            div, unit = money_scale(bt["actual_rev"][sl], bt["pred_hi"][sl])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x + x[::-1], y=list(bt["pred_hi"][sl] / div) + list(bt["pred_lo"][sl] / div)[::-1],
                                     fill="toself", fillcolor=_hex_rgba(GREEN, 0.12), line=dict(width=0),
                                     hoverinfo="skip", name="90% interval"))
            fig.add_trace(go.Scatter(x=x, y=bt["pred_mean"][sl] / div, mode="lines", name="Predicted",
                                     line=dict(color=GREEN, width=2)))
            fig.add_trace(go.Scatter(x=x, y=bt["actual_rev"][sl] / div, mode="lines+markers", name="Actual",
                                     line=dict(color=INK, width=2), marker=dict(size=4)))
            fig.add_vrect(x0=bt["q_start"], x1=bt["q_end"], fillcolor=_hex_rgba(INK, 0.04), line_width=0,
                          annotation_text="holdout", annotation_position="top left",
                          annotation_font=dict(size=11, color=LABEL))
            fig.update_layout(yaxis_title=f"Weekly revenue ({unit})")
            _plot(style_fig(fig, height=320))

            md('<div class="ctitle"><h3>Forecast assumptions vs actual</h3><span class="hint">From data preceding the quarter</span></div>')
            fc = bt["fc"]
            def _err(f, a):
                return signed((f - a) / a * 100, suffix="%") if a else "—"
            rows = [{"Variable": "Budget", "Forecast": fmt_money(fc["total_budget"]),
                     "Actual": fmt_money(bt["actual_budget"]), "Error": _err(fc["total_budget"], bt["actual_budget"])},
                    {"Variable": "Revenue per KPI", "Forecast": f"${fc['rev_per_kpi']:,.2f}",
                     "Actual": f"${bt['actual_rpk']:,.2f}", "Error": _err(fc["rev_per_kpi"], bt["actual_rpk"])}]
            rows += [{"Variable": f"{ch} share", "Forecast": f"{fc['spend_pct'][ch] * 100:.1f}%",
                      "Actual": f"{bt['actual_pct'][ch] * 100:.1f}%",
                      "Error": signed((fc["spend_pct"][ch] - bt["actual_pct"][ch]) * 100, suffix=" pp")} for ch in channels]
            rows += [{"Variable": f"{ch} CPI", "Forecast": f"{fc['cost_per_impression'][ch]:.6f}",
                      "Actual": f"{bt['actual_cpi'][ch]:.6f}",
                      "Error": _err(fc["cost_per_impression"][ch], bt["actual_cpi"][ch])} for ch in channels]
            rows += [{"Variable": c, "Forecast": f"{fc['controls'][c]:.4f}", "Actual": f"{bt['actual_controls'][c]:.4f}",
                      "Error": _err(fc["controls"][c], bt["actual_controls"][c])} for c in cfg.get("control_cols", [])]
            table(rows, right={"Forecast", "Actual", "Error"})

        summary.append({"Quarter": bt["quarter"], "Predicted": fmt_money(pred_q), "Actual": fmt_money(act_q),
                        "Bias": signed(t["bias"], suffix="%"), "wMAPE": f"{t['wmape']:.1f}%",
                        "MAPE": f"{t['mape']:.1f}%", "R²": f"{t['r2']:.2f}", "In-sample wMAPE": f"{tr['wmape']:.1f}%"})

    if len(summary) > 1:
        with card("card_bt_summary"):
            card_title("Summary", "Holdout accuracy per quarter")
            table(summary, right={"Predicted", "Actual", "Bias", "wMAPE", "MAPE", "R²", "In-sample wMAPE"})



# ===================================================================
# Navigation (top bar)
# ===================================================================

PAGE_CONFIG = st.Page(page_config, title="Configuration", url_path="configure", default=True)
PAGES = [
    PAGE_CONFIG,
    st.Page(page_forecast, title="Forecast", url_path="forecast"),
    st.Page(page_backtest, title="Backtest", url_path="backtest"),
]

st.logo(LOGO_SVG, size="large")
pg = st.navigation(PAGES, position="top")
header_strip()
pg.run()
