# Bitcoin Lab 3.0

An open, research-first BTC/EUR forecasting dashboard. It is an educational experiment, **not investment advice or a trading system**.

## What it does
- Publishes a static dashboard suitable for GitHub Pages.
- Refreshes BTC/EUR daily prices using CoinGecko.
- Produces 30- and 90-day forecasts once per day and stores each forecast with its origin date, before its outcome is known.
- Evaluates matured forecasts against a zero-return baseline using MAE, RMSE, directional accuracy and the fraction of cases where the model beats zero on absolute error.
- Keeps a visible separation between the forecast signal and the risk-adjusted research score.
- Offers CSV exports and records data freshness/source information.

## Scientific guardrails
- Forecasts are **prospective**: the ledger begins when the workflow is first run. It does not pretend that today's model generated historical forecasts.
- Do not tune parameters on the same observations used to report performance. Keep a final holdout period untouched.
- A positive score is not evidence of predictive skill. Compare out-of-sample forecasts with zero-return and simple historical-mean baselines.
- A missing API key, failed update, stale data, or empty history should be visible—not silently converted into a successful forecast.
- Crypto prices are volatile; historical relationships may fail. Nothing here recommends buying or selling Bitcoin.

## Deploy
1. In GitHub, open **Settings → Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**, select `main` and `/ (root)`, then Save.
3. If the repository is private, check whether your GitHub plan allows Pages for a private repository. To publish a public dashboard most simply, change the repository visibility to **Public** in Settings → General → Danger Zone. Public source code must never contain secrets.
4. Open **Actions** and enable workflows if GitHub asks.
5. Run **Bitcoin Lab daily update → Run workflow** once manually. The workflow also runs daily (GitHub may delay scheduled runs).
6. Wait for the workflow to finish, then reload the Pages site.

## Optional CoinGecko key
The updater can try the public endpoint without a key, subject to rate limits. For more reliable access, create a CoinGecko Demo API key and add it under **Settings → Secrets and variables → Actions → New repository secret** with the name `COINGECKO_DEMO_API_KEY`. Never put the key in HTML, source files, screenshots, or commit history.

## Run locally
Requires Python 3.11+ and no third-party packages:
```bash
python scripts/update_lab.py
python -m unittest discover -s tests -v
```
The updater writes `data/latest.json` and `data/forecasts.json`.

## Method summary
The baseline model uses a shrunk historical daily log-return mean over a trailing window. For horizon (h), its expected log return is the trailing mean multiplied by (h) and a shrinkage factor. The risk-adjusted score subtracts a downside-tail proxy, a volatility uncertainty penalty and an assumed round-trip cost. These are transparent research knobs, not calibrated probabilities or validated trading edges.

The dashboard's current interactive parameters are for **exploration only**; changing them does not rewrite the immutable forecasts already in the ledger. Forecast evaluation uses stored predictions and prices available after each forecast origin.

## Data and attribution
Market data is requested from [CoinGecko](https://www.coingecko.com/). Availability, granularity and rate limits can change. Data is denominated in EUR; it is not an exchange execution price.
