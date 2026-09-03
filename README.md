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
  scaled by growth multipliers, the median year-over-year ratio per variable capped
  to [0.5, 2.0].
- **Optimizer**: Meridian budget optimizer at fixed total budget with symmetric
  per-channel bounds up to 100%, and lower bound zero above that.
- **Validation**: per-quarter holdout via Meridian `holdout_id`, reporting wMAPE,
  MAPE, bias, and R² on the held-out weeks against in-sample values.
