#!/usr/bin/env python3
"""Leakage-conscious BTC/EUR model evaluation with experimental uncertainty and regimes."""
from __future__ import annotations

import json
import math
import statistics
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "data" / "latest.json"
OUTPUT = ROOT / "data" / "model_arena.json"
LOOKBACK = 180
SHRINKAGE = 0.35
MIN_INTERVAL_SAMPLE = 60
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
    by_date = {}
    for item in raw:
        try:
            day = date.fromisoformat(item["date"])
            close = float(item["close"])
            if not math.isfinite(close) or close <= 0:
                continue
            by_date[day.isoformat()] = {"date": day.isoformat(), "close": close}
        except (KeyError, TypeError, ValueError):
            continue
    prices = [by_date[d] for d in sorted(by_date)]
    if len(prices) < LOOKBACK + 32:
        raise ValueError("Not enough valid price history for walk-forward evaluation")
    return prices


def quantile(values, q):
    vals = sorted(values)
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def summarize(pairs):
    """Pairs are predicted and realized log returns on the same origin sample."""
    if not pairs:
        return {"n": 0, "mae": None, "rmse": None, "directional_accuracy": None,
                "directional_n": 0, "zero_baseline_mae": None, "zero_baseline_rmse": None,
                "skill_vs_zero_mae": None, "units": "log-return"}
    errors = [pred - actual for pred, actual in pairs]
    mae = statistics.fmean(abs(e) for e in errors)
    rmse = math.sqrt(statistics.fmean(e * e for e in errors))
    directional = [(pred, actual) for pred, actual in pairs if actual != 0.0]
    predicts_direction = any(pred != 0.0 for pred, _ in directional)
    da = (sum(1 for pred, actual in directional
              if pred != 0.0 and (pred > 0) == (actual > 0)) / len(directional)
          if directional and predicts_direction else None)
    zero_mae = statistics.fmean(abs(actual) for _, actual in pairs)
    zero_rmse = math.sqrt(statistics.fmean(actual * actual for _, actual in pairs))
    skill = 1.0 - mae / zero_mae if zero_mae > 0 else None
    return {"n": len(pairs), "mae": round(mae, 10), "rmse": round(rmse, 10),
            "directional_accuracy": round(da, 6) if da is not None else None,
            "directional_n": len(directional), "zero_baseline_mae": round(zero_mae, 10),
            "zero_baseline_rmse": round(zero_rmse, 10),
            "skill_vs_zero_mae": round(skill, 6) if skill is not None else None,
            "units": "log-return (not simple percentage return)"}


def _consecutive(segment):
    return all((date.fromisoformat(b["date"]) - date.fromisoformat(a["date"])).days == 1
               for a, b in zip(segment, segment[1:]))


def predict_at_origin(prices, origin_index, horizon, lookback=LOOKBACK):
    """Features stop at origin_index; require consecutive history and exact calendar target."""
    if origin_index < lookback or origin_index >= len(prices):
        return None
    origin = prices[origin_index]
    origin_day = date.fromisoformat(origin["date"])
    target_day = date.fromordinal(origin_day.toordinal() + horizon)
    target = target_day.isoformat()
    by_date = {p["date"]: (i, p) for i, p in enumerate(prices)}
    target_item = by_date.get(target)
    if not target_item:
        return None
    target_i, target_price = target_item
    if target_i != origin_index + horizon or (target_day - origin_day).days != horizon:
        return None
    window_prices = prices[origin_index - lookback:origin_index + 1]
    if len(window_prices) != lookback + 1 or not _consecutive(window_prices):
        return None
    returns = [math.log(prices[j]["close"] / prices[j - 1]["close"])
               for j in range(origin_index - lookback + 1, origin_index + 1)]
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


def empirical_horizon_interval(prices, origin_index, horizon, min_samples=MIN_INTERVAL_SAMPLE):
    """Past-only empirical 5th–95th percentiles of fully realized horizon-matched log returns."""
    origin_day = date.fromisoformat(prices[origin_index]["date"])
    historical = []
    for j in range(LOOKBACK, origin_index - horizon + 1):
        start_day = date.fromisoformat(prices[j]["date"])
        target_day = date.fromisoformat(prices[j + horizon]["date"])
        if (target_day - start_day).days != horizon or target_day > origin_day:
            continue
        segment = prices[j:j + horizon + 1]
        if len(segment) != horizon + 1 or not _consecutive(segment):
            continue
        historical.append(math.log(prices[j + horizon]["close"] / prices[j]["close"]))
    if len(historical) < min_samples:
        return {"status": "insufficient_sample", "n_calibration": len(historical),
                "lower_log_return": None, "upper_log_return": None, "nominal_coverage": 0.90}
    return {"status": "experimental_unvalidated", "n_calibration": len(historical),
            "lower_log_return": round(quantile(historical, 0.05), 10),
            "upper_log_return": round(quantile(historical, 0.95), 10),
            "nominal_coverage": 0.90}


def classify_regime(prices, origin_index):
    """Past-only descriptive label; no observation after origin_index is inspected."""
    if origin_index < LOOKBACK:
        return "insufficient_history"
    window = prices[origin_index - LOOKBACK:origin_index + 1]
    if not _consecutive(window):
        return "insufficient_history"
    returns = [math.log(window[i]["close"] / window[i - 1]["close"])
               for i in range(1, len(window))]
    vol_30 = statistics.pstdev(returns[-30:])
    vol_180 = statistics.pstdev(returns)
    if vol_180 <= 0 or vol_30 / vol_180 <= 0.75:
        volatility = "low"
    elif vol_30 / vol_180 >= 1.25:
        volatility = "high"
    else:
        volatility = "normal"
    trend_return = math.log(prices[origin_index]["close"] / prices[origin_index - 30]["close"])
    trend = "up" if trend_return > 0.02 else ("down" if trend_return < -0.02 else "flat")
    return trend + "_" + volatility + "_volatility"


def select_nonoverlap_origin_dates(origins, horizon):
    """Greedily select sorted origins separated by at least horizon calendar days."""
    selected = set()
    last_selected = None
    for item in origins:
        current = date.fromisoformat(item["origin_date"])
        if last_selected is None or (current - last_selected).days >= horizon:
            selected.add(item["origin_date"])
            last_selected = current
    return selected


def run_walk_forward(prices, horizons=(30, 90), lookback=LOOKBACK):
    records = {h: {name: [] for name in MODEL_VERSIONS} for h in horizons}
    nonoverlap = {h: {name: [] for name in MODEL_VERSIONS} for h in horizons}
    origins = {h: [] for h in horizons}
    interval_records = {h: [] for h in horizons}
    regime_records = {h: [] for h in horizons}
    for i in range(lookback, len(prices)):
        for h in horizons:
            forecast = predict_at_origin(prices, i, h, lookback)
            if forecast is None:
                continue
            predictions, actual, target = forecast
            for name, pred in predictions.items():
                records[h][name].append((pred, actual))
            origin_date = prices[i]["date"]
            origins[h].append({"origin_index": i, "origin_date": origin_date, "target_date": target})
            interval = empirical_horizon_interval(prices, i, h)
            if interval["lower_log_return"] is not None:
                interval_records[h].append({
                    **interval, "origin_date": origin_date, "target_date": target,
                    "actual_log_return": actual,
                    "covered": interval["lower_log_return"] <= actual <= interval["upper_log_return"],
                    "width_log_return": interval["upper_log_return"] - interval["lower_log_return"],
                })
            regime_records[h].append({
                "origin_date": origin_date, "target_date": target,
                "regime": classify_regime(prices, i), "actual_log_return": actual,
                "prediction": predictions["shrunk_trailing_mean_v1.0"],
            })

    results = {}
    for h in horizons:
        # Use actual origin dates, not every h-th row, to handle missing/invalid observations safely.
        selected_dates = select_nonoverlap_origin_dates(origins[h], h)
        origin_dates = [x["origin_date"] for x in origins[h]]
        for name in MODEL_VERSIONS:
            nonoverlap[h][name] = [
                pair for origin_date, pair in zip(origin_dates, records[h][name])
                if origin_date in selected_dates
            ]

        intervals = interval_records[h]
        interval_n = len(intervals)
        coverage = (sum(1 for item in intervals if item["covered"]) / interval_n) if interval_n else None
        width = statistics.fmean(item["width_log_return"] for item in intervals) if interval_n else None
        groups = {}
        for row in regime_records[h]:
            bucket = groups.setdefault(row["regime"], {"n": 0, "absolute_error_sum": 0.0,
                                                        "directional_correct": 0, "directional_n": 0})
            bucket["n"] += 1
            bucket["absolute_error_sum"] += abs(row["prediction"] - row["actual_log_return"])
            if row["actual_log_return"] != 0 and row["prediction"] != 0:
                bucket["directional_n"] += 1
                if (row["prediction"] > 0) == (row["actual_log_return"] > 0):
                    bucket["directional_correct"] += 1
        for bucket in groups.values():
            bucket["mae_log_return"] = round(bucket.pop("absolute_error_sum") / bucket["n"], 10)
            bucket["directional_accuracy"] = (
                round(bucket["directional_correct"] / bucket["directional_n"], 6)
                if bucket["directional_n"] else None
            )
            bucket["interpretation"] = "descriptive_only; small groups are not conclusive"

        results[str(h)] = {
            "horizon_days": h,
            "sample_start": origins[h][0]["origin_date"] if origins[h] else None,
            "sample_end": origins[h][-1]["target_date"] if origins[h] else None,
            "overlapping": {name: summarize(records[h][name]) for name in MODEL_VERSIONS},
            "non_overlapping": {name: summarize(nonoverlap[h][name]) for name in MODEL_VERSIONS},
            "uncertainty": {
                "method": "Empirical 5th–95th percentiles of earlier fully realized horizon-matched log returns available at each origin.",
                "nominal_coverage": 0.90, "status": "experimental_unvalidated",
                "n_evaluated": interval_n,
                "realized_coverage": round(coverage, 6) if coverage is not None else None,
                "mean_interval_width_log_return": round(width, 10) if width is not None else None,
                "minimum_calibration_sample": MIN_INTERVAL_SAMPLE,
                "warning": "Not calibrated probabilities; calibration samples overlap and coverage is descriptive.",
            },
            "regimes": {
                "definition": "Trend uses 30-day cumulative log return thresholds ±2%; volatility compares trailing 30-day population standard deviation with trailing 180-day standard deviation (high >=1.25x, low <=0.75x, otherwise normal). Past-only and descriptive.",
                "groups": groups,
            },
            "note": "Daily-origin outcomes overlap. Non-overlap origins are selected by actual calendar spacing; neither evaluation proves future performance.",
        }
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_start": prices[0]["date"] if prices else None,
        "data_end": prices[-1]["date"] if prices else None,
        "price_observations": len(prices),
        "lookback_returns": lookback,
        "model_definitions": MODEL_VERSIONS,
        "method": "Walk-forward origins use only prices available at origin; outcomes require exact calendar target dates and consecutive daily history.",
        "experiment_policy": {
            "registered_candidates": list(MODEL_VERSIONS),
            "production_model_changed": False,
            "promotion": "No automatic promotion. Candidates require independent chronological holdout evidence and adequate samples.",
            "selection_bias_warning": "Comparing multiple candidates on the same historical sample creates selection bias.",
        },
        "horizons": results,
    }


def main():
    prices = load_prices()
    result = run_walk_forward(prices)
    latest_payload = json.loads(LATEST.read_text(encoding="utf-8"))
    result["generated_at_utc"] = latest_payload.get("last_updated", result["generated_at_utc"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    try:
        unchanged = OUTPUT.read_text(encoding="utf-8") == text
    except FileNotFoundError:
        unchanged = False
    if not unchanged:
        tmp = OUTPUT.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(OUTPUT)
    print("Walk-forward arena updated:", len(prices), "price observations;",
          ", ".join(f"{h}D={result['horizons'][str(h)]['overlapping']['zero_return_v1']['n']} outcomes" for h in (30, 90)))


if __name__ == "__main__":
    main()
