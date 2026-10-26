#!/usr/bin/env python3
"""Leakage-conscious walk-forward BTC/EUR model comparison; never writes the live ledger."""
from __future__ import annotations
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "data" / "latest.json"
OUTPUT = ROOT / "data" / "model_arena.json"
LOOKBACK = 180
SHRINKAGE = 0.35
MODEL_VERSIONS = {
    "zero_return_v1": "Always predicts zero log-return.",
    "historical_mean_180_v1": "Trailing 180-return mean multiplied by horizon.",
    "shrunk_trailing_mean_v1.0": "Trailing 180-return mean multiplied by horizon and 0.35.",
    "momentum_30_v1": "Trailing 30-return mean multiplied by horizon and 0.35.",
}

def load_prices(path=LATEST):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("prices")
    if not isinstance(raw, list):
        raise ValueError("latest.json has no prices list")
    prices = []
    for item in raw:
        try:
            date, close = item["date"], float(item["close"])
            if not isinstance(date, str) or not math.isfinite(close) or close <= 0:
                continue
            prices.append({"date": date, "close": close})
        except (KeyError, TypeError, ValueError):
            continue
    prices.sort(key=lambda p: p["date"])
    # A date is kept once; prefer the last occurrence in the source.
    by_date = {p["date"]: p for p in prices}
    prices = [by_date[d] for d in sorted(by_date)]
    if len(prices) < LOOKBACK + 32:
        raise ValueError("Not enough valid price history for walk-forward evaluation")
    return prices

def summarize(pairs):
    """Pairs are (predicted log return, realized log return), all on a common sample."""
    if not pairs:
        return {"n": 0, "mae": None, "rmse": None, "directional_accuracy": None,
                "directional_n": 0, "zero_baseline_mae": None, "zero_baseline_rmse": None,
                "skill_vs_zero_mae": None}
    errors = [pred - actual for pred, actual in pairs]
    mae = statistics.fmean(abs(e) for e in errors)
    rmse = math.sqrt(statistics.fmean(e * e for e in errors))
    directional = [(pred, actual) for pred, actual in pairs if actual != 0.0]
    da = (sum(1 for pred, actual in directional if pred != 0.0 and (pred > 0) == (actual > 0))
          / len(directional)) if directional else None
    zero_mae = statistics.fmean(abs(actual) for _, actual in pairs)
    zero_rmse = math.sqrt(statistics.fmean(actual * actual for _, actual in pairs))
    skill = (1.0 - mae / zero_mae) if zero_mae > 0 else None
    return {"n": len(pairs), "mae": round(mae, 10), "rmse": round(rmse, 10),
            "directional_accuracy": round(da, 6) if da is not None else None,
            "directional_n": len(directional), "zero_baseline_mae": round(zero_mae, 10),
            "zero_baseline_rmse": round(zero_rmse, 10),
            "skill_vs_zero_mae": round(skill, 6) if skill is not None else None}

def predict_at_origin(prices, origin_index, horizon, lookback=LOOKBACK):
    """Return predictions and realized outcome; prediction inputs stop at origin_index."""
    if origin_index < lookback or origin_index >= len(prices):
        return None
    origin = prices[origin_index]
    target_date = datetime.fromisoformat(origin["date"]).date()
    target = (target_date.fromordinal(target_date.toordinal() + horizon)).isoformat()
    by_date = {p["date"]: (i, p) for i, p in enumerate(prices)}
    target_item = by_date.get(target)
    if not target_item:
        return None
    target_i, target_price = target_item
    if target_i != origin_index + horizon:
        return None
    returns = [math.log(prices[j]["close"] / prices[j-1]["close"])
               for j in range(origin_index-lookback+1, origin_index+1)]
    if len(returns) != lookback:
        return None
    mean_window = statistics.fmean(returns)
    mean_momentum = statistics.fmean(returns[-30:])
    predictions = {
        "zero_return_v1": 0.0,
        "historical_mean_180_v1": mean_window * horizon,
        "shrunk_trailing_mean_v1.0": mean_window * horizon * SHRINKAGE,
        "momentum_30_v1": mean_momentum * horizon * SHRINKAGE,
    }
    realized = math.log(target_price["close"] / origin["close"])
    return predictions, realized, target

def run_walk_forward(prices, horizons=(30, 90), lookback=LOOKBACK):
    """At origin i, predictions only use closes <= i; target is exactly i+h calendar days."""
    records = {h: {name: [] for name in MODEL_VERSIONS} for h in horizons}
    nonoverlap = {h: {name: [] for name in MODEL_VERSIONS} for h in horizons}
    origins = {h: [] for h in horizons}
    for i in range(lookback, len(prices)):
        for h in horizons:
            forecast = predict_at_origin(prices, i, h, lookback)
            if forecast is None:
                continue
            predictions, actual, target = forecast
            for name, pred in predictions.items():
                records[h][name].append((pred, actual))
            origins[h].append({"origin_index": i, "origin_date": prices[i]["date"], "target_date": target})
    results = {}
    for h in horizons:
        # Use a fixed step of h days from the first eligible origin to reduce overlap.
        origin_indices = [x["origin_index"] for x in origins[h]]
        selected = set(origin_indices[::h]) if origin_indices else set()
        for name in MODEL_VERSIONS:
            # Pair alignment is guaranteed because every model shares the same valid origins.
            pairs = records[h][name]
            nonoverlap[h][name] = [pair for idx, pair in zip(origin_indices, pairs) if idx in selected]
        results[str(h)] = {
            "horizon_days": h,
            "sample_start": origins[h][0]["origin_date"] if origins[h] else None,
            "sample_end": origins[h][-1]["target_date"] if origins[h] else None,
            "overlapping": {name: summarize(records[h][name]) for name in MODEL_VERSIONS},
            "non_overlapping": {name: summarize(nonoverlap[h][name]) for name in MODEL_VERSIONS},
            "note": "Daily-origin results overlap. Non-overlapping subset samples origins every horizon days; neither is proof of future performance."
        }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_start": prices[0]["date"] if prices else None,
        "data_end": prices[-1]["date"] if prices else None,
        "price_observations": len(prices),
        "lookback_returns": lookback,
        "model_definitions": MODEL_VERSIONS,
        "method": "Expanding-time walk-forward origins with fixed trailing windows; features at origin i use only prices through i. Outcomes use exact calendar target dates.",
        "horizons": results,
    }

def main():
    prices = load_prices()
    result = run_walk_forward(prices)
    latest_payload = json.loads(LATEST.read_text(encoding="utf-8"))
    # Keep the artifact byte-stable when the underlying completed-price dataset is unchanged.
    result["generated_at_utc"] = latest_payload.get("last_updated", result["generated_at_utc"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUTPUT)
    print("Walk-forward arena updated:", len(prices), "price observations;",
          ", ".join(f"{h}D={result['horizons'][str(h)]['overlapping']['zero_return_v1']['n']} outcomes" for h in (30, 90)))

if __name__ == "__main__":
    main()
