# MMM Budget Optimizer: technical documentation

Internship deliverable, September 2026. Author: Raees Kabir. Supervisor: Shanavas Kavu.
Repository: https://github.com/rqees/mmm-forecast

This document is written for the data scientist who inherits the tool. It assumes familiarity with regression and basic Bayesian ideas, and explains marketing-mix-modelling terms on first use. It is organised by what you are trying to do:

| If you want to… | Read |
|---|---|
| Understand what was built and why | 1. Overview, 2. Problem and scope |
| Know what data it needs | 3. Data |
| Run it once, end to end | 4. Getting started |
| Do a specific task | 5. How-to guides |
| Look something up | 6. Reference |
| Understand the method and the reasoning | 7. Method and design decisions |
| See what it produced during the internship | 8. Results |
| Know where it stops | 9. Limitations and known issues, 10. Next steps |

---

## 1. Overview

The MMM Budget Optimizer is a Streamlit web application that recommends how to split a fixed quarterly media budget across channels to maximise incremental revenue, and validates its own model on held-out quarters. It wraps Google Meridian, an open-source Bayesian marketing mix model (MMM), and adds three things Meridian does not provide:

1. **A forecast layer** that builds next quarter's inputs (budget, channel mix, cost per impression, revenue per conversion, control levels) from the same quarter of the prior year and year-over-year growth, with every value editable.
2. **A future-period optimizer path.** Meridian's optimizer cannot natively optimise a period outside the fitted data. The app constructs the future period's data tensors itself and passes them in, which lets the optimizer work on a quarter that has not happened yet.
3. **A holdout backtest** that refits the model with a past quarter's outcomes hidden, predicts that quarter's revenue from the spend that actually ran, and compares with actual revenue. It also compares the forecast layer's assumptions with what happened, so model error and assumption error can be told apart.

It runs in Google Colab on a GPU, takes a weekly CSV, and is brand-agnostic: channels are detected from column suffixes, so a different client's file needs a mapping step, not a code change.

## 2. Problem and scope

### 2.1 Business question

Given a set budget for the coming quarter and weekly history of spend, impressions, conversions and revenue per conversion by channel, how should the budget be split across channels to maximise incremental revenue?

### 2.2 Why a model is needed

Three properties of media response make the answer non-obvious from a spreadsheet:

- **Diminishing returns.** The first dollar in a channel earns more than the millionth. The rate at which returns fall differs by channel.
- **Carryover.** An impression this week keeps working for several weeks (adstock).
- **Confounding.** Paid search spend rises when organic demand rises, so naive attribution credits search with conversions that would have happened anyway.

An MMM estimates all three from history. Meridian was chosen because it is the current open-source standard, handles the three properties explicitly, and has a budget optimizer built in.

### 2.3 Objectives and success criteria

| Objective | Criterion | Outcome |
|---|---|---|
| Recommend a channel allocation for a future quarter | Optimizer runs on a future period with user-set budget and constraint | Met |
| Use seasonality and all available year-over-year history in the forecast | Same-quarter-prior-year baseline; YoY growth multipliers; seasonal controls in the model | Met |
| Use control variables | Google query volume as a control; generated seasonal controls | Met |
| Validate on held-out data | Per-quarter holdout backtest with in-sample comparison | Met: two quarters within 11% wMAPE, no systematic bias |
| Geo-level model | Geo rows aggregated to national in v1 | Not met; deferred (Section 10) |
| Work on any client's file | Column mapping UI, wide and long layouts | Met |

### 2.4 Scope decisions made with the supervisor

- **National model for v1.** Geo rows are aggregated before training. Reason: simplify the first iteration and focus on channel effects. Cost: Meridian's partial pooling across regions is forgone.
- **External signals dropped for v1, then Google query volume added.** Competitor sales, sentiment and population columns from the initial sample dataset were left out. Query volume was added later as the one control that de-confounds paid search.
- **Streamlit on Colab instead of a FastAPI service.** The original scope described a Python backend with an optional front end. A browser tool that runs where the GPU is was judged the more useful deliverable. The pipeline is a single module and can be wrapped in a service later.
- **Fixed budget.** The tool reallocates a given total. It does not choose the total.

## 3. Data

### 3.1 Sources used during the internship

| Phase | Dataset | Notes |
|---|---|---|
| Build | Meridian's synthetic sample (`notebooks/meridian_getting_started.ipynb`) | Five channels, geo-level, includes promotion, organic media, competitor and sentiment columns. Used to build and test the pipeline. |
| Delivery | Client weekly data (not in the repository) | Six channels: audio, search, social, streaming, tv, video. Conversions as KPI, revenue per conversion, Google query volume as control. 118 weeks, 2023-06-14 to 2025-09-10. |
| Public sample | `data/sample_weekly.csv`, generated by `data/make_sample.py` | Synthetic. Same schema, date range and approximate scale as the client file, with its own seasonality, channel drift and media response. No client rows. Use it to try the app; results on it are not the client results in Section 8. |

### 3.2 Required schema (wide layout)

One row per week. Column names are configurable; the defaults are shown.

| Column | Type | Meaning |
|---|---|---|
| time column (e.g. `time`, `week`) | date | Week start or end date. Must parse as a date. Weekly spacing expected. |
| `<channel>_spend` | numeric | Spend for the channel that week. |
| `<channel>_impression` | numeric | Impressions (or clicks) for the channel that week. |
| KPI column (e.g. `conversions`) | numeric | The outcome the model explains. |
| revenue-per-KPI column | numeric | Revenue per unit of KPI that week. Revenue = KPI × this. |
| control columns (e.g. `gqv_control`) | numeric | External drivers. Optional. |
| non-media treatment columns (e.g. `Promo`) | numeric | Levers the advertiser sets. Optional. |
| organic media columns | numeric | Unpaid impressions. Optional. |
| geo column | text | Region identifier. Optional; rows are aggregated to national. |

A channel is any prefix that has both a `_spend` and an `_impression` column. The suffixes are editable.

### 3.3 Long layout

One row per week and channel, with columns for time, channel name, spend and a media metric, optionally geo. The app detects this layout, pivots to wide, and carries any extra columns through: week-level extras (same value for every channel in a week) become single columns, channel-level extras are pivoted per channel.

### 3.4 Transformations

- **Geo aggregation.** Spend, impressions, KPI and organic media are summed across geos per week. Revenue per KPI is recomputed as total revenue divided by total KPI, which is the correctly weighted average. Controls and non-media treatments are averaged.
- **Seasonal controls.** For seasonality harmonics *k*, the app generates 2*k* columns `season_sin1, season_cos1, …` from day-of-year: sin(2πi·d/365.25) and cos(·) for i = 1…k. The same calendar date gets the same values every year.
- **Complete quarters.** A quarter is complete if it has at least 75% of the weeks of the fullest quarter in the data. Partial first and last quarters are excluded from growth estimation.

### 3.5 Known data issues and checks

Pre-flight validation runs before training. Errors block training; warnings do not.

| Check | Level |
|---|---|
| Time column does not parse as dates | error |
| Duplicate dates (usually a geo column not mapped) | error |
| Fewer than 2 rows | error |
| Missing, non-numeric or NaN values in any required column | error |
| Channel with zero total spend or zero total impressions | error |
| KPI or revenue-per-KPI sums to zero | error |
| Irregular week spacing | warning |
| Fewer than 52 weeks | warning |
| Negative values | warning |
| Weeks with zero KPI | warning |

Observed in the client data: two small channels (audio, social) under 1% of spend, and audio spend falling to zero in some quarters. Both make those channels' estimates uncertain; see Section 9.

## 4. Getting started

This walks through one complete run in Colab. Budget about 45 minutes, most of it waiting.

1. Open a Colab notebook with a GPU runtime (Runtime → Change runtime type → T4 GPU).
2. Upload `app.py` to `/content`, or copy it from Drive.
3. Paste the contents of `colab_launcher.py` into a cell and run it. It installs `google-meridian`, `streamlit` and `plotly`, mounts Drive, writes the Streamlit theme, starts the app, opens a Cloudflare tunnel, and prints a public URL. Keep the cell running.
4. Open the URL. You are on **Configuration**.
5. Upload the weekly CSV, or `data/sample_weekly.csv` from the repository to try the tool. Check the preview, then the detected layout.
6. In **Map columns**, confirm the suffixes, the time, KPI and revenue-per-KPI columns, the channels, and select any control columns.
7. Leave **Model settings** at defaults for a first run.
8. In **Train model**, check Pre-flight says Ready, then click Train. About ten minutes on a T4.
9. Go to **Forecast**. Pick the quarter, leave the growth multiplier at 1.0 and max shift at 30%, review the assumptions, click Run optimizer. A few minutes.
10. Go to **Backtest**. Select the most recent complete quarter, leave Quick sampling off, click Run backtest. About ten minutes per quarter.
11. Back on **Configuration**, click Save model. The model, its data and all results are written to Drive and can be restored with Load model in seconds.

## 5. How-to guides

### 5.1 Prepare a new client's file

Produce one row per week with a date column, a `_spend` and `_impression` pair per channel, a KPI column, a revenue-per-KPI column, and any controls. If the file is one row per week and channel, leave it; the app pivots it. If there is a region column, keep it and map it as the geo column so the app can aggregate.

### 5.2 Add a control such as query volume

Include it as a numeric weekly column. Select it under **Control columns** in the mapping step. Do not select it as a channel or a non-media treatment. If the file has geo rows, the control is averaged across regions.

### 5.3 Add a promotion or price series

Select it under **Non-media treatment columns**. Meridian estimates an incremental effect for it, relative to a baseline level, and the forecast layer projects its level for next quarter. Use this for anything the advertiser sets; use a control for anything the market sets.

### 5.4 Change the return prior

**ROAS prior μ** and **σ** set a log-normal prior on each channel's return on ad spend (incremental revenue per dollar). Defaults 0.2 and 0.9 give a median around 1.2x with a wide spread. Lower σ makes the prior stronger. Per-channel priors from incrementality tests are not exposed; see Section 10.

### 5.5 Forecast a quarter with a known budget

On **Forecast**, type the budget into **Total budget ($)**. Everything else can stay at the trend default. The optimizer splits that exact figure.

### 5.6 Run the constraint ladder

Run the optimizer at 30%, 60% and 500% with the same budget. Note search share, TV share and the gain at each. Report all three: the direction should hold at every setting and the gain per unit of movement should fall.

### 5.7 Validate on a held-out quarter

On **Backtest**, select one or more complete quarters whose same quarter a year earlier is also complete. Leave Quick sampling off for reportable numbers. Read holdout wMAPE against in-sample wMAPE, and bias. Ignore holdout R² unless the question is about within-quarter shape.

### 5.8 Save, load and hand over

Save writes `<name>.binpb` (the Meridian model) and `<name>_config.json` (column mapping, settings, the national data, growth multipliers, quarterly table, last forecast result and all backtest results). Load restores all of it. Both files travel together.

### 5.9 Run outside Colab

Any machine with a CUDA GPU, Python 3.10+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `config.toml` at `.streamlit/config.toml` next to `app.py` for the intended theme.

## 6. Reference

### 6.1 Pages and elements

**Configuration**

| Element | Meaning |
|---|---|
| Model status card | Status, weeks (with date range), complete quarters, channels of the model in memory |
| Model path, Load model | Path to `.binpb`; the `_config.json` is read from the same folder |
| CSV file | Weekly rows, wide or long |
| Data layout | Wide / Long, auto-detected |
| Long-layout selectors | Time, channel, spend, media metric, geo columns |
| Impression / Spend column suffix | Default `_impression`, `_spend` |
| Time, KPI, Revenue-per-KPI, Geo column | Column roles |
| Channels | Detected pairs; deselect to drop |
| Non-media treatment, Organic media, Control columns | Covariate roles (Section 7.1) |
| ROAS prior μ, σ | Log-normal prior parameters |
| Forecast horizon (weeks) | Length of the future period, default 13 |
| Seasonality harmonics | Number of yearly sine/cosine pairs, default 2, 0 disables |
| Sampling (MCMC): Chains, Samples kept per chain, Adaptation steps, Burn-in steps | Defaults 4, 500, 1500, 500; seed fixed at 42 |
| Train model | Disabled while pre-flight errors exist |
| Save path, Save model | Writes model and config bundle |

**Forecast**

| Element | Meaning |
|---|---|
| Forecast quarter | In-progress quarter if the data ends mid-quarter, plus the next four. Past quarters are not offered. |
| Growth rate multiplier (0.8–1.3) | Scales every growth multiplier's deviation from 1.0 |
| Max shift per channel (5–500%) | Bound on each channel's spend relative to status quo. ≤100% symmetric; >100% lower bound is zero |
| Assumptions tiles | Budget with growth and prior-year value; revenue per KPI; horizon and baseline weeks; channel count and constraint |
| Total budget ($), Revenue per KPI ($) | Editable |
| Per-channel cost per impression | Editable, prior-year value shown |
| Non-media treatment levels, Control levels | Editable quarter-average levels |
| Growth multipliers applied | Historical rate and applied rate per variable |
| Run optimizer | Meridian budget optimizer on the future period |
| Results tiles | Status quo incremental revenue, optimized incremental revenue, gain, blended ROAS |
| Budget allocation, Channel results, three charts | Shares, dollars, change, ROAS per channel |

**Backtest**

| Element | Meaning |
|---|---|
| Quarters to backtest | Complete quarters with a complete same-quarter-prior-year |
| Quick sampling | At most 2 chains × 250 draws (500 adaptation, 250 burn-in) |
| Per-quarter tiles | Predicted holdout revenue with bias, actual revenue, holdout wMAPE with in-sample, holdout R² with in-sample |
| Chart | Last 52 training weeks plus holdout; actual, predicted mean, 90% interval, holdout shaded |
| Forecast assumptions vs actual | Budget, revenue per KPI, channel shares, CPIs, controls: forecast from pre-quarter data vs actual |
| Summary | Per-quarter predicted, actual, bias, wMAPE, MAPE, R², in-sample wMAPE |

### 6.2 Metrics

| Metric | Definition |
|---|---|
| Incremental revenue | Revenue attributable to media, per Meridian's counterfactual (media on minus media off). Not total revenue. |
| ROAS | Incremental revenue ÷ spend. Meridian's documentation calls this ROI. True ROI would divide incremental profit by spend. |
| Gain | Optimized minus status quo incremental revenue at the same budget. |
| Status quo | Prior-year same-quarter channel mix applied to the current budget. |
| wMAPE | Σ\|predicted − actual\| ÷ Σ actual over the weeks in the window. Weighted by revenue. |
| MAPE | Mean of \|predicted − actual\| ÷ actual per week. |
| Bias | (Σ predicted − Σ actual) ÷ Σ actual. |
| R² | 1 − SS_res ÷ SS_tot on weekly values within the window. |
| Growth multiplier | Geometric mean of a variable's same-quarter year-over-year ratios, each winsorized to [0.5, 2.0]. 1.0 when fewer than two years of complete quarters exist. |

### 6.3 Save bundle (`<name>_config.json`)

| Key | Content |
|---|---|
| `time_col`, `geo_col`, `kpi_col`, `rev_per_kpi_col`, `channels`, `impression_suffix`, `spend_suffix` | Column mapping |
| `non_media_cols`, `organic_cols`, `organic_names`, `control_cols`, `seasonality_k`, `season_cols` | Covariates |
| `roi_mu`, `roi_sigma`, `n_chains`, `n_adapt`, `n_burnin`, `n_keep`, `n_prior_samples`, `n_future_weeks`, `seed` | Model settings |
| `national_data` | The aggregated weekly frame used for training, as records |
| `trend_multipliers`, `quarterly_data` | Growth multipliers and the per-quarter table behind them |
| `last_results`, `backtest_results` | Forecast and backtest results, arrays serialised as lists |
| `app_version` | App version string |

Bundles saved by earlier versions load with missing keys defaulted (no controls, no seasonality).

### 6.4 Code map (`app.py`)

| Area | Functions |
|---|---|
| Quarter helpers | `quarter_to_date_range`, `get_corresponding_quarter`, `quarter_week_counts`, `complete_quarters`, `forecastable_quarters` |
| Data pipeline | `normalize_time`, `season_col_names`, `season_values`, `add_season_cols`, `aggregate_to_national`, `validate_national` |
| Forecast layer | `compute_trend_multipliers`, `build_forecast_config`, `build_future_data_tensors` |
| Optimization | `run_optimizer`, `extract_results`, `constraint_label`, `shift_slider` |
| Training | `train_model` (accepts `holdout_mask`) |
| Backtest | `accuracy_metrics`, `run_backtest` |
| Persistence | `save_bundle`, `load_bundle`, `_jsonable`, `_restore_arrays` |
| UI | `page_config`, `page_forecast`, `page_backtest`, plus card, table and chart helpers |

Dependencies: `streamlit ≥ 1.30`, `google-meridian[and-cuda,schema]`, `plotly ≥ 5.18`. Meridian pulls TensorFlow and TensorFlow Probability.

## 7. Method and design decisions

### 7.1 Model

Meridian fits, per channel, a geometric adstock (carryover) and a Hill saturation curve, with a Bayesian prior on each channel's return. The app configures it as:

- **National model.** One geo after aggregation. Meridian's default baseline for a national model is a single level over time (one knot); the seasonal controls give it a yearly shape.
- **KPI type non-revenue with revenue per KPI**, so incremental outcomes are reported in revenue.
- **Prior.** Log-normal on ROAS for every channel, μ = 0.2, σ = 0.9 by default. Weakly informative; median about 1.2x, 90% of mass roughly 0.3x to 5x.
- **Covariates in three roles.** *Controls* (query volume, seasonal terms) enter additively with a coefficient and get no incremental effect. *Non-media treatments* (promotions, price) get an incremental effect relative to a baseline. *Organic media* gets adstock and saturation but has no spend and is not optimised.
- **Sampling.** NUTS via Meridian: 500 prior draws; 4 chains × 500 kept draws after 1,500 adaptation and 500 burn-in steps, seed 42. About ten minutes on a T4.

### 7.2 Forecast layer

Next quarter's inputs are built from the same quarter of the prior year (the baseline) scaled by growth multipliers:

1. **Growth multipliers.** For each variable (spend per channel as a weekly rate, cost per impression per channel, revenue per KPI, each control and treatment level), take every same-quarter year-over-year ratio across complete quarters, winsorize each to [0.5, 2.0], and take the geometric mean. With fewer than two years of complete quarters, every multiplier is 1.0. The **Growth rate multiplier** slider scales each multiplier's distance from 1.0.
2. **Budget default.** Prior-year quarter spend times the spend-weighted average of the channel spend multipliers, so a small channel's trend cannot swing the total. The user can enter any budget.
3. **Channel mix (status quo).** The prior-year quarter's shares of spend, unscaled.
4. **Cost per impression.** Prior-year quarter spend ÷ impressions per channel, times that channel's CPI multiplier.
5. **Revenue per KPI.** Prior-year quarter revenue ÷ KPI, times its multiplier.
6. **Control and treatment levels.** Prior-year quarter mean times multiplier, held constant across the horizon.
7. **Future tensors.** Weekly spend per channel = budget × share ÷ horizon weeks; impressions = spend ÷ CPI; controls as above plus seasonal terms computed from the future dates; dates start the week after the data ends. These are passed to Meridian as `new_data`.

### 7.3 Optimizer

Meridian's `BudgetOptimizer.optimize` with the future tensors, the budget, the status-quo shares, and symmetric spend constraints up to 100%. Above 100% the lower bound is fixed at 1.0 (spend may fall to zero) while the upper bound continues to rise; 500% is effectively unconstrained. Results are read from the optimizer's non-optimized and optimized datasets: shares, spend, incremental revenue and ROAS per channel, totals and gain.

Controls do not affect the optimizer's answer: they enter additively and cancel in the counterfactual difference. They matter during training, where they shape attribution.

### 7.4 Validation design

For each selected quarter:

1. Truncate the data at the quarter's end, so nothing after it exists.
2. Compute growth multipliers and the baseline from data strictly before the quarter, exactly as the Forecast page would have at the time.
3. Refit Meridian on the truncated data with the quarter's weeks passed as `holdout_id`. Meridian excludes those weeks' KPI from the likelihood but keeps their media, so adstock carries over into the holdout. Truncating instead would sever the carryover and make the first holdout week look like a cold start.
4. Compute expected revenue per week for all weeks with `Analyzer.expected_outcome(aggregate_times=False)`, take the posterior mean and 5th/95th percentiles, align to the data's time axis.
5. Score holdout weeks and training weeks separately: wMAPE, MAPE, bias, R².
6. Compare the forecast layer's assumptions with the quarter's actual budget, mix, CPIs, revenue per KPI and control levels.

The backtest validates prediction. It cannot validate the optimizer's recommendation, because the revenue a mix nobody ran would have earned is unobserved. This is true of every MMM. The earlier version of the page scored whether the client happened to move channels in the recommended direction; that was removed because it measured agreement, not accuracy.

### 7.5 Decisions and reasoning

| Decision | Alternative considered | Reason |
|---|---|---|
| Same-quarter-prior-year baseline × growth | Time-series model of each input (SARIMA, Prophet) | With about two years of weekly data a time-series model would estimate seasonality from two cycles. The baseline is interpretable, cannot diverge, and carries input seasonality by construction. |
| Year-over-year ratios, complete quarters only | Quarter-over-quarter ratios | QoQ folds seasonality into growth. A partial quarter distorted totals during development, hence the completeness threshold. |
| Geometric mean of winsorized ratios | Arithmetic mean (first version), median (second version) | Rates compound; the arithmetic mean of ratios is biased upward and unstable on small denominators. The first version produced a 730% CPI forecast error on a small channel. The cap bounds the damage from any single anomalous quarter. |
| No QoQ fallback under two years | Fall back to QoQ | A wrong growth estimate is worse than none. |
| Seasonality as Fourier controls | External baseline forecast injected into the model; more knots | Learned inside the model from past years; composes with `new_data`; no separate forecasting step. Knots would fit last year's shape but cannot repeat it twelve months later. |
| Query volume as a control, not a treatment | Omit it | Removes the demand confound on paid search. Meridian recommends it. |
| Forecast picker offers future quarters only | Any quarter | Forecasting a quarter already in the data used a model trained on that quarter: a leaky retrospective. Past quarters belong to the backtest. |
| Spend constraint up to 500% | Cap at 100% | At 100% a channel can at most double; TV was pinned at that cap and the unconstrained optimum was invisible. |
| Persist results in the save bundle | Recompute on load | Backtests take ten minutes per quarter; a restart should not require rerunning them. |
| Blended ROAS at fixed budget | Profit-maximising budget | Out of scope by brief; noted in Section 10. |

## 8. Results

All numbers from the client dataset, national model, seasonality harmonics = 2, default prior and sampling.

### 8.1 Validation

Holdout backtests, full sampling:

| Quarter | Predicted | Actual | Bias | Holdout wMAPE | MAPE | R² | In-sample wMAPE |
|---|---|---|---|---|---|---|---|
| 2025Q1 | $66.37M | $65.32M | +1.6% | 9.0% | 9.1% | −1.16 | 7.3% |
| 2025Q2 | $54.62M | $56.18M | −2.8% | 11.0% | 11.1% | −0.01 | 7.4% |

Before seasonal controls were added, the same test gave bias +6.8%, +2.0% and +4.5% on 2024Q4, 2025Q1 and 2025Q2, with holdout wMAPE 9.5%, 5.7% and 7.6%. Seasonal controls removed the systematic over-prediction and improved in-sample fit, at the cost of a few points of holdout error, because with two years of history the seasonal terms are learned from a single prior cycle.

Negative holdout R² on 13 weekly values indicates the model does not track within-quarter shape; the flat baseline plus two harmonics cannot follow week-to-week swings. Level accuracy (wMAPE, bias) is the relevant measure for quarterly planning.

Total quarterly revenue is $55M to $66M; the model attributes about $6M of it to media. The backtest therefore validates the total more strongly than the individual channel curves. Consistency across quarters and the stability of the recommendation across constraints are the additional evidence for the curves.

### 8.2 Recommendation for 2026Q1

Budget $6.08M (trend default). Baseline 2025Q1 mix.

| Max shift | Search share | TV share | Optimized incremental revenue | Gain | Blended ROAS |
|---|---|---|---|---|---|
| Status quo | 64.5% | 22.0% | $6.05M | – | 0.99x |
| 30% | 53.8% | 28.6% | $6.38M | +$335K (+5.5%) | 1.05x |
| 60% | 43.6% | 35.2% | $6.59M | +$538K (+8.9%) | 1.08x |
| Unconstrained | 31.2% | 43.4% | $6.86M | +$811K (+13.4%) | 1.13x |

Per channel at 30%:

| Channel | Status quo | Optimized | ROAS before → after |
|---|---|---|---|
| audio | 1.2% | 1.6% | 0.95 → 0.92 |
| search | 64.5% | 53.8% | 0.65 → 0.73 |
| social | 0.7% | 1.0% | 3.25 → 3.05 |
| streaming | 5.6% | 7.3% | 1.42 → 1.24 |
| tv | 22.0% | 28.6% | 1.83 → 1.59 |
| video | 5.9% | 7.7% | 0.99 → 0.87 |

Reading: the direction is the same at every constraint (out of search, into TV, streaming and video). The gain per percentage point moved out of search falls from about $31K to $26K to $24K across the ladder, which is the saturation curve made visible. The unconstrained mix puts TV at 43% of budget, beyond any share in the history, and increases audio and social five to six-fold from under 1% of budget, where the model has little data. The recommended path is one rung per quarter with re-estimation between.

### 8.3 Assumption accuracy

In the backtests, revenue per KPI was forecast within 1% to 4%, query volume within 2%, channel CPIs within 10% to 26%, and the total budget within 18% in both directions. The budget misses reflect planning decisions the client made (a cut in 2025Q1, an increase in 2025Q2) that no trend could anticipate; this is why the budget is a user input.

### 8.4 Environment

Colab T4 GPU. Training about 10 minutes; optimizer run 1 to 3 minutes; backtest about 10 minutes per quarter at full sampling.

## 9. Limitations and known issues

- **National only.** Geo variation and population are aggregated away; partial pooling is forgone.
- **Forecast-layer inputs are point estimates.** Meridian carries posterior uncertainty, but budget, CPIs, revenue per KPI and growth multipliers enter as single numbers, so the reported interval on incremental revenue understates total uncertainty. The growth slider is a manual sensitivity check, not a distribution.
- **Few year-over-year observations.** With two years each variable's multiplier rests on at most four ratios; winsorization limits but does not remove the noise.
- **Baseline is flat plus seasonality.** No trend component of its own. Holdout error exceeds in-sample by a few points for this reason.
- **Uniform spend within the quarter.** The optimizer sets a cross-channel split; each channel's spend is constant across the 13 weeks.
- **ROAS, not ROI.** Returns are revenue-based; break-even on a profit basis is 1 ÷ gross margin, not 1.0.
- **No convergence diagnostics in the UI.** R-hat and trace plots are available from Meridian's inference data but are not surfaced.
- **One prior for all channels.** Meridian's intended use is per-channel priors from incrementality tests.
- **Small channels are uncertain.** Audio and social are under 1% of spend; their ROAS estimates (0.95x, 3.25x) are prior-driven and should not anchor decisions.
- **Query volume may over-correct.** Brand searches caused by media are absorbed by the control. A brand-excluded series would be cleaner.

## 10. Recommended next steps

In priority order, with rough effort:

1. **Geo-level model** (days). Pass the geo and population columns to Meridian instead of aggregating. Restores partial pooling and multiplies the observations behind every curve. Forecast layer, optimizer and backtest apply unchanged.
2. **Propagate assumption uncertainty** (days). The backtest already measures how wrong each assumption has historically been. Sample growth multipliers from those error distributions, rerun the optimizer per draw, and report best/base/worst.
3. **Spend timing within the quarter** (days). With seasonality in the model, let the optimizer concentrate spend into high-baseline weeks.
4. **Convergence diagnostics** (hours). Surface R-hat per parameter and flag values above 1.05 after training.
5. **Per-channel priors** (hours, given test data). Expose μ and σ per channel; populate from geo or incrementality tests.
6. **Flexible budget** (hours). Meridian's target-marginal-ROI mode chooses the total spend at which the last dollar returns a target; add as an alternative optimizer mode.
7. **Brand-excluded query volume** (data request). Replace or supplement the current control.
8. **Re-run the backtest each quarter** (process). Add each newly complete quarter as it arrives; a third year of data is the single biggest improvement available.

## 11. Glossary

- **Adstock**: carryover of media effect into later weeks, modelled as geometric decay.
- **Hill curve**: S-shaped saturation function mapping media volume to effect.
- **Control**: a covariate the advertiser does not set; explains outcome, gets no incremental credit.
- **Non-media treatment**: a lever the advertiser sets that is not media; gets incremental credit.
- **Organic media**: unpaid impressions; modelled like media, not optimised.
- **Incremental revenue**: revenue attributable to media, relative to a no-media counterfactual.
- **ROAS**: incremental revenue ÷ spend.
- **Status quo**: prior-year same-quarter mix at the current budget.
- **Holdout**: weeks whose outcomes are excluded from model fitting so predictions for them are out of sample.
- **wMAPE**: revenue-weighted mean absolute percentage error.
- **Winsorize**: clip values to a fixed range before averaging.
- **GQV**: Google query volume, weekly search interest for the brand or category.

## 12. Decision record

| Date | Change | Reason |
|---|---|---|
| Aug 2026 | Pipeline built on Meridian sample data; Streamlit app on Colab | Scope agreed with supervisor |
| Sep 2 | Control columns added; forecast picker restricted to future quarters | GQV had no correct slot; past-quarter "forecasts" leaked |
| Sep 2 | Spend constraint extended past 100% | Unconstrained optimum was invisible |
| Sep 2 | Backtest rewritten as true holdout | Previous version used the full model and scored agreement, not accuracy |
| Sep 3 | Seasonal Fourier controls | Model ran systematically hot; no calendar in a national baseline |
| Sep 3 | Growth multipliers: geometric mean of winsorized YoY ratios, weekly-rate spend, no QoQ fallback | Arithmetic mean produced extreme CPI forecasts; 13- vs 14-week quarters biased ratios |
| Sep 3 | Budget default spend-weighted; ROAS labelling; results persisted in bundle; Sensitivity and Trends pages removed | Reviewer feedback and demo preparation |
