#!/usr/bin/env python3
"""Fetch BTC/EUR daily prices and maintain a prospective, auditable forecast ledger."""
from __future__ import annotations
import json, math, os, statistics, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "data" / "latest.json"
LEDGER = ROOT / "data" / "forecasts.json"
API = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=eur&days=730&interval=daily"
HORIZONS = (30, 90)
LOOKBACK = 180
SHRINKAGE = 0.35
LAMBDA = 1.0
GAMMA = 0.5
COST = 0.005

def read_json(path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

def fetch_prices():
    headers = {"User-Agent": "BitcoinLab3/1.0 (+research dashboard)", "accept": "application/json"}
    key = os.environ.get("COINGECKO_DEMO_API_KEY", "").strip()
    if key:
        headers["x-cg-demo-api-key"] = key
    req = Request(API, headers=headers)
    try:
        with urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"CoinGecko request failed: {exc}") from exc
    raw = payload.get("prices")
    if not isinstance(raw, list) or len(raw) < 100:
        raise RuntimeError("CoinGecko returned too little price history.")
    # Keep the final observation for each UTC calendar date.
    daily = {}
    for item in raw:
        if not isinstance(item, list) or len(item) < 2:
            continue
        dt = datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc).date().isoformat()
        price = float(item[1])
        if math.isfinite(price) and price > 0:
            daily[dt] = price
    prices = [{"date": d, "close": round(daily[d], 6)} for d in sorted(daily)]
    if len(prices) < 100:
        raise RuntimeError("Could not normalize CoinGecko prices into daily observations.")
    return prices

def log_returns(prices):
    out = []
    for prev, cur in zip(prices, prices[1:]):
        if prev["close"] > 0 and cur["close"] > 0:
            out.append((cur["date"], math.log(cur["close"] / prev["close"])))
    return out

def quantile(values, q):
    vals = sorted(values)
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

def build_forecast(prices, horizon, now_date):
    rets = log_returns(prices)
    window = [r for _, r in rets[-LOOKBACK:]]
    if len(window) < 60:
        raise RuntimeError("At least 60 daily returns are required before forecasting.")
    daily_mean = statistics.fmean(window)
    daily_vol = statistics.pstdev(window)
    mu = daily_mean * horizon * SHRINKAGE
    downside_tail = max(0.0, -quantile(window, 0.05)) * math.sqrt(horizon)
    uncertainty = 1.645 * daily_vol * math.sqrt(horizon)
    score = mu - LAMBDA * downside_tail - GAMMA * uncertainty - COST
    last = prices[-1]
    origin = datetime.fromisoformat(last["date"]).date()
    target = origin + timedelta(days=horizon)
    return {
        "id": f"{last['date']}-{horizon}d",
        "origin_date": last["date"],
        "target_date": target.isoformat(),
        "horizon_days": horizon,
        "origin_price_eur": last["close"],
        "model": "shrunk_trailing_mean_v1",
        "lookback_days": LOOKBACK,
        "shrinkage": SHRINKAGE,
        "predicted_log_return": round(mu, 8),
        "zero_baseline_log_return": 0.0,
        "downside_tail_proxy": round(downside_tail, 8),
        "uncertainty_penalty": round(uncertainty, 8),
        "assumed_cost": COST,
        "risk_adjusted_score": round(score, 8),
        "status": "pending",
        "realized_log_return": None,
        "realized_date": None,
        "realized_price_eur": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")
    }

def mature_forecasts(ledger, prices):
    by_date = {p["date"]: p for p in prices}
    dates = [p["date"] for p in prices]
    changed = False
    for f in ledger["forecasts"]:
        if f.get("status") != "pending":
            continue
        target = f.get("target_date")
        # Use the first available close on/after target, no more than 3 calendar days later.
        match = next((d for d in dates if target <= d <= (datetime.fromisoformat(target).date() + timedelta(days=3)).isoformat()), None)
        if match and f.get("origin_price_eur", 0) > 0:
            f["realized_date"] = match
            f["realized_price_eur"] = by_date[match]["close"]
            f["realized_log_return"] = round(math.log(by_date[match]["close"] / f["origin_price_eur"]), 8)
            f["status"] = "matured"
            changed = True
    return changed

def main():
    prices = fetch_prices()
    latest = {
        "schema_version": 1,
        "source": "CoinGecko",
        "currency": "EUR",
        "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prices": prices,
        "status": "ok"
    }
    ledger = read_json(LEDGER, {"schema_version": 1, "created_note": "Prospective ledger; entries begin on first successful scheduled run.", "forecasts": []})
    if not isinstance(ledger.get("forecasts"), list):
        raise RuntimeError("Forecast ledger has an invalid schema.")
    changed = mature_forecasts(ledger, prices)
    existing = {f.get("id") for f in ledger["forecasts"]}
    for horizon in HORIZONS:
        f = build_forecast(prices, horizon, datetime.now(timezone.utc).date())
        if f["id"] not in existing:
            ledger["forecasts"].append(f)
            changed = True
    ledger["forecasts"].sort(key=lambda f: (f.get("origin_date", ""), f.get("horizon_days", 0)))
    ledger["last_updated"] = latest["last_updated"]
    write_json(LATEST, latest)
    write_json(LEDGER, ledger)
    matured = sum(f.get("status") == "matured" for f in ledger["forecasts"])
    pending = sum(f.get("status") == "pending" for f in ledger["forecasts"])
    print(f"Updated {len(prices)} daily prices; ledger has {len(ledger['forecasts'])} forecasts ({matured} matured, {pending} pending).")
    print("Note: forecasts start prospectively; this run does not backfill historical predictions.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
