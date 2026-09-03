# MMM Budget Optimizer

Streamlit application for quarterly media budget allocation on top of Google Meridian.
Given weekly historical data and a fixed budget, it forecasts the next quarter and
recommends the channel allocation that maximizes incremental revenue, with holdout
backtesting to validate the model.

## Running in Colab

1. Open a GPU runtime (Runtime → Change runtime type → T4 GPU).
2. Upload `app.py` to `/content`, or copy it from Drive.
3. Paste the contents of `colab_launcher.py` into a cell and run it. It installs
   dependencies, mounts Drive, starts Streamlit, and prints a public URL.
4. Keep the cell running while the app is in use.

## Pages

- **Configuration**: upload weekly data, map columns, set model options, train,
  save, and load. A saved bundle is a `.binpb` model plus a `_config.json` that
  carries the data, growth multipliers, and any forecast or backtest results.
- **Forecast**: choose a future quarter, review the assumptions built from the
  same quarter of the prior year and growth multipliers, edit any of them, set the
  per-channel constraint, and run the optimizer.
- **Backtest**: holdout validation. For each selected quarter the model is refitted
  with that quarter's outcomes excluded, and expected revenue under actual spend is
  compared with actual revenue.

## Data

Weekly rows, wide layout: one row per week with `<channel>_spend` and
`<channel>_impression` columns per channel, a KPI column, a revenue-per-KPI column,
and optional control columns such as Google query volume. A long layout (one row
per week × channel) is pivoted automatically. Multiple geos are aggregated to
national.

## Method

- **Model**: Meridian national MMM with adstock and Hill saturation per channel,
  a log-normal ROI prior, user-supplied controls, and generated yearly Fourier
  terms as seasonal controls.
- **Forecast layer**: next-quarter inputs are the same quarter of the prior year
  scaled by growth multipliers. Each multiplier is the geometric mean of the
  variable's same-quarter year-over-year ratios, with every ratio winsorized to
  [0.5, 2.0]. With fewer than two years of complete quarters all multipliers are 1.0.
- **Optimizer**: Meridian budget optimizer at fixed total budget with symmetric
  per-channel bounds up to 100%, and lower bound zero above that.
- **Validation**: per-quarter holdout via Meridian `holdout_id`, reporting wMAPE,
  MAPE, bias, and R² on the held-out weeks against in-sample values.

## Limitations

- **National only.** Geos are aggregated before training and population is unused.
  A geo-level model is the natural next step and Meridian supports it directly.
- **Forecast-layer inputs are point estimates.** The MMM carries posterior
  uncertainty, but the growth multipliers, prior-year baseline, and cost per
  impression do not. Uncertainty in the recommendation from those inputs is not
  propagated; the growth multiplier slider on the Forecast page is a manual
  sensitivity check, not a distribution.
- **Few year-over-year observations.** With two years of data each variable has at
  most four ratios behind its multiplier, which is why they are winsorized.
- **Flat baseline plus Fourier seasonality.** The national baseline is a single
  level with generated seasonal terms; it has no trend component of its own.
- **Fixed budget.** The optimizer reallocates a given total. It does not choose the
  total, for example by growing spend until marginal ROI reaches a target.
