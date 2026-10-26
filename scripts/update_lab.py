#!/usr/bin/env python3
"""Fetch daily BTC/EUR prices and maintain an auditable prospective forecast ledger."""
from __future__ import annotations
import json, math, os, statistics, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "data" / "latest.json"
LEDGER = ROOT / "data" / "forecasts.json"
COINGECKO_API = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=eur&days=730&interval=daily"
BINANCE_API = "https://api.binance.com/api/v3/klines?symbol=BTCEUR&interval=1d&limit=1000"
KRAKEN_API = "https://api.kraken.com/0/public/OHLC?pair=XBTEUR&interval=1440"
HORIZONS = (30, 90)
LOOKBACK = 180
SHRINKAGE = 0.35
LAMBDA = 1.0
GAMMA = 0.5
COST = 0.005
MODEL_VERSION = "1.0"

def read_json(path, fallback):
    """Read JSON without ever treating corruption as an empty persistent record."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return fallback
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Refusing to overwrite malformed JSON at {path} (line {exc.lineno}, column {exc.colno}). Original file preserved.") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Refusing to use {path}: expected a JSON object; original file preserved.")
    return value

def write_json(path, obj):
    """Atomically replace a JSON artifact and skip byte-identical writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    try:
        if path.read_text(encoding="utf-8") == content:
            return False
    except FileNotFoundError:
        pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    return True

def request_json(url, headers=None):
    req = Request(url, headers=headers or {})
    try:
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"HTTP request failed for {url.split('?')[0]}: {exc}") from exc

def fetch_kraken_prices():
    payload = request_json(KRAKEN_API, {"User-Agent": "BitcoinLab3/1.0"})
    errors = payload.get("error", [])
    if errors:
        raise RuntimeError("Kraken API error: " + ", ".join(errors))
    result = payload.get("result", {})
    pair_key = next((k for k in result if k != "last"), None)
    candles = result.get(pair_key, []) if pair_key else []
    now_seconds = int(datetime.now(timezone.utc).timestamp())
    daily = {}
    for candle in candles:
        if not isinstance(candle, list) or len(candle) < 5:
            continue
        open_seconds, close_price = int(candle[0]), float(candle[4])
        # Only accept fully closed UTC daily candles.
        if open_seconds + 86400 > now_seconds or not math.isfinite(close_price) or close_price <= 0:
            continue
        date = datetime.fromtimestamp(open_seconds, tz=timezone.utc).date().isoformat()
        daily[date] = close_price
    prices = [{"date": d, "close": round(daily[d], 6)} for d in sorted(daily)]
    if len(prices) < 100:
        raise RuntimeError("Kraken returned fewer than 100 completed daily BTC/EUR candles.")
    return prices

def fetch_binance_prices():
    payload = request_json(BINANCE_API, {"User-Agent": "BitcoinLab3/1.0"})
    if not isinstance(payload, list):
        raise RuntimeError("Binance returned an unexpected payload.")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    daily = {}
    for candle in payload:
        if not isinstance(candle, list) or len(candle) < 7:
            continue
        open_ms, close_price, close_ms = int(candle[0]), float(candle[4]), int(candle[6])
        # Exclude the still-open daily candle to avoid using a moving close as a completed observation.
        if close_ms >= now_ms or not math.isfinite(close_price) or close_price <= 0:
            continue
        date = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc).date().isoformat()
        daily[date] = close_price
    prices = [{"date": d, "close": round(daily[d], 6)} for d in sorted(daily)]
    if len(prices) < 100:
        raise RuntimeError("Binance returned fewer than 100 completed daily BTC/EUR candles.")
    return prices

def fetch_coingecko_prices():
    headers = {"User-Agent": "BitcoinLab3/1.0 (+research dashboard)", "accept": "application/json"}
    key = os.environ.get("COINGECKO_DEMO_API_KEY", "").strip()
    if key:
        headers["x-cg-demo-api-key"] = key
    payload = request_json(COINGECKO_API, headers)
    raw = payload.get("prices")
    if not isinstance(raw, list) or len(raw) < 100:
        raise RuntimeError("CoinGecko returned too little price history.")
    daily = {}
    now_date = datetime.now(timezone.utc).date().isoformat()
    for item in raw:
        if not isinstance(item, list) or len(item) < 2:
            continue
        date = datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc).date().isoformat()
        price = float(item[1])
        if date < now_date and math.isfinite(price) and price > 0:
            daily[date] = price
    prices = [{"date": d, "close": round(daily[d], 6)} for d in sorted(daily)]
    if len(prices) < 100:
        raise RuntimeError("Could not normalize CoinGecko prices into completed daily observations.")
    return prices

def fetch_prices():
    # Prefer the public exchange endpoint; fall back to CoinGecko if the exchange endpoint is unavailable.
    errors = []
    for name, fetcher in (("Kraken", fetch_kraken_prices), ("Binance", fetch_binance_prices), ("CoinGecko", fetch_coingecko_prices)):
        try:
            return fetcher(), name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("All market-data providers failed. " + " | ".join(errors))

def log_returns(prices):
    """Return only true one-calendar-day log returns; gaps are never mislabeled daily."""
    out = []
    for prev, cur in zip(prices, prices[1:]):
        try:
            prev_day = datetime.fromisoformat(prev["date"]).date()
            cur_day = datetime.fromisoformat(cur["date"]).date()
            prev_close, cur_close = float(prev["close"]), float(cur["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if (cur_day - prev_day).days != 1:
            continue
        if math.isfinite(prev_close) and math.isfinite(cur_close) and prev_close > 0 and cur_close > 0:
            out.append((cur["date"], math.log(cur_close / prev_close)))
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
        "id": f"{last['date']}-{horizon}d-shrunk_trailing_mean_v{MODEL_VERSION}",
        "origin_date": last["date"],
        "data_cutoff_date": last["date"],
        "target_date": target.isoformat(),
        "horizon_days": horizon,
        "origin_price_eur": last["close"],
        "model": "shrunk_trailing_mean_v1",
        "model_version": MODEL_VERSION,
        "evaluation_eligible": True,
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
        "matured_at_utc": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")
    }

def annotate_ledger_records(ledger):
    """Retain legacy records but mark unknown provenance and exclude equivalent duplicates."""
    records = ledger["forecasts"]
    changed = False
    for forecast in records:
        if "model_version" not in forecast:
            forecast["model_version"] = "legacy_unknown"
            forecast["provenance_status"] = "legacy_unknown"
            changed = True
        forecast.setdefault("evaluation_eligible", True)
    groups = {}
    for index, forecast in enumerate(records):
        key = (
            forecast.get("origin_date"), forecast.get("target_date"),
            forecast.get("horizon_days"), forecast.get("model"),
            forecast.get("origin_price_eur"), forecast.get("predicted_log_return"),
        )
        groups.setdefault(key, []).append((index, forecast))
    for entries in groups.values():
        if len(entries) < 2:
            continue
        # Prefer explicit model-version provenance; never delete or rewrite prediction values.
        entries.sort(key=lambda pair: pair[1].get("model_version") == "legacy_unknown", reverse=False)
        for index, forecast in entries[1:]:
            if forecast.get("evaluation_eligible") is not False:
                forecast["evaluation_eligible"] = False
                forecast["evaluation_exclusion_reason"] = "equivalent_forecast_record_retained_for_history_but_excluded_from_metrics"
                changed = True
    return changed

def mature_forecasts(ledger, prices):
    """Mature only against the exact target-date close; never silently shift the horizon."""
    by_date = {p["date"]: p for p in prices}
    changed = False
    for f in ledger["forecasts"]:
        if f.get("status") != "pending":
            continue
        target = f.get("target_date")
        if not isinstance(target, str) or target not in by_date:
            continue
        origin_price = f.get("origin_price_eur")
        target_price = by_date[target].get("close")
        if (not isinstance(origin_price, (int, float)) or not math.isfinite(origin_price)
                or origin_price <= 0 or not isinstance(target_price, (int, float))
                or not math.isfinite(target_price) or target_price <= 0):
            continue
        f["realized_date"] = target
        f["realized_price_eur"] = target_price
        f["realized_log_return"] = round(math.log(target_price / origin_price), 8)
        f["status"] = "matured"
        f["matured_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        changed = True
    return changed

def main():
    prices, source = fetch_prices()
    previous_latest = read_json(LATEST, {})
    previous_ledger = read_json(LEDGER, {"schema_version": 2, "created_note": "Prospective ledger; entries begin on first successful scheduled run.", "forecasts": []})
    # Validate the incoming series before any persistent artifact is written.
    dates = [p.get("date") for p in prices]
    if len(dates) < 100 or dates != sorted(set(dates)):
        raise RuntimeError("Incoming price series has too few observations, duplicate dates, or non-chronological dates; no data files were written.")
    for p in prices:
        if not isinstance(p.get("close"), (int, float)) or not math.isfinite(p["close"]) or p["close"] <= 0:
            raise RuntimeError(f"Invalid close price on {p.get('date')}; no data files were written.")
    if not isinstance(previous_ledger.get("forecasts"), list):
        raise RuntimeError("Forecast ledger has an invalid schema; original file preserved.")
    if any(not isinstance(f, dict) for f in previous_ledger["forecasts"]):
        raise RuntimeError("Forecast ledger contains a non-object record; original file preserved.")
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prices_changed = previous_latest.get("prices") != prices or previous_latest.get("source") != source
    latest = {
        "schema_version": 2,
        "source": source,
        "source_switch_detected": bool(previous_latest.get("source") and previous_latest.get("source") != source),
        "historical_observation_provenance": "unknown_before_observation_level_tracking",
        "currency": "EUR",
        "last_updated": checked_at if prices_changed or not previous_latest.get("last_updated") else previous_latest["last_updated"],
        "prices": prices,
        "status": "ok"
    }
    ledger = previous_ledger
    previous_schema = int(ledger.get("schema_version", 1))
    ledger_changed = annotate_ledger_records(ledger)
    ledger_changed = mature_forecasts(ledger, prices) or ledger_changed
    # Upgrade the envelope only; existing historical forecast records are preserved as-is.
    ledger["schema_version"] = max(2, previous_schema)
    ledger_changed = ledger_changed or ledger["schema_version"] != previous_schema
    existing = {f.get("id") for f in ledger["forecasts"]}
    for horizon in HORIZONS:
        f = build_forecast(prices, horizon, datetime.now(timezone.utc).date())
        if f["id"] not in existing:
            ledger["forecasts"].append(f)
            existing.add(f["id"])
            ledger_changed = True
    ledger_changed = annotate_ledger_records(ledger) or ledger_changed
    ledger["forecasts"].sort(key=lambda f: (f.get("origin_date", ""), f.get("horizon_days", 0), f.get("model_version", "")))
    if ledger_changed or not ledger.get("last_updated"):
        ledger["last_updated"] = checked_at
    write_json(LATEST, latest)
    write_json(LEDGER, ledger)
    matured = sum(f.get("status") == "matured" for f in ledger["forecasts"])
    pending = sum(f.get("status") == "pending" for f in ledger["forecasts"])
    print(f"Updated {len(prices)} daily prices from {source}; ledger has {len(ledger['forecasts'])} forecasts ({matured} matured, {pending} pending).")
    print("Note: forecasts start prospectively; this run does not backfill historical predictions.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
