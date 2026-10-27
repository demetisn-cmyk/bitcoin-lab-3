#!/usr/bin/env python3
"""Past-only adaptive research loop; writes evidence and recommendations, never changes live forecasts."""
from __future__ import annotations

import json
import math
import statistics
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "data" / "latest.json"
LEDGER_PATH = ROOT / "data" / "forecasts.json"
OUTPUT_PATH = ROOT / "data" / "adaptive_research.json"
CURRENT_MODEL = "shrunk_trailing_mean_v1"
CURRENT_VERSION = "1.0"
CANDIDATES = ("expanding_mean_v1",)
HORIZONS = (30, 90)
TRAIN_FRACTION = 0.60
MIN_NONOVERLAP_N = 30
MIN_RELATIVE_MAE_GAIN = 0.05
MAX_SIGN_TEST_P = 0.05


def _prices(payload):
    by_day = {}
    for row in payload.get("prices", []):
        try:
            day = date.fromisoformat(row["date"])
            close = float(row["close"])
            if math.isfinite(close) and close > 0:
                by_day[day] = close
        except (KeyError, TypeError, ValueError):
            continue
    return [(d, by_day[d]) for d in sorted(by_day)]


def _log_return(a, b):
    return math.log(b / a)


def _predict(prices, origin, model):
    # The slice ends at origin. No target or later observation is available to fitting.
    history = [_log_return(prices[i - 1][1], prices[i][1])
               for i in range(1, origin + 1)]
    if not history:
        return None
    if model == "zero_return_v1":
        return 0.0
    if model == "expanding_mean_v1":
        return statistics.fmean(history)
    if model == "shrunk_trailing_mean_v1":
        window = history[-180:]
        return statistics.fmean(window) * 0.35
    raise ValueError(f"unknown model: {model}")


def _sign_test(candidate_errors, comparator_errors):
    # One-sided exact paired sign test: candidate must win strictly.
    wins = sum(a < b for a, b in zip(candidate_errors, comparator_errors))
    n = len(candidate_errors)
    if not n:
        return None, wins
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return p, wins


def _evaluate(prices, horizon):
    n_prices = len(prices)
    split = max(181, int(n_prices * TRAIN_FRACTION))
    samples = []
    # origins are spaced by the target horizon, preventing overlapping validation outcomes.
    for origin in range(split, n_prices - horizon, horizon):
        target = origin + horizon
        if (prices[target][0] - prices[origin][0]).days != horizon:
            continue
        segment = prices[origin:target + 1]
        if any((b[0] - a[0]).days != 1 for a, b in zip(segment, segment[1:])):
            continue
        actual = _log_return(prices[origin][1], prices[target][1])
        row = {"origin_date": prices[origin][0].isoformat(),
               "target_date": prices[target][0].isoformat(), "actual": actual}
        for model in ("zero_return_v1", CURRENT_MODEL, *CANDIDATES):
            prediction = _predict(prices, origin, model)
            if prediction is not None:
                row[model] = prediction
        if all(model in row for model in ("zero_return_v1", CURRENT_MODEL, *CANDIDATES)):
            samples.append(row)

    def mae(key):
        return statistics.fmean(abs(row[key] - row["actual"]) for row in samples) if samples else None

    current_mae, zero_mae = mae(CURRENT_MODEL), mae("zero_return_v1")
    candidate_results = {}
    for candidate in CANDIDATES:
        c_errors = [abs(row[candidate] - row["actual"]) for row in samples]
        current_errors = [abs(row[CURRENT_MODEL] - row["actual"]) for row in samples]
        zero_errors = [abs(row["actual"]) for row in samples]
        p_current, wins_current = _sign_test(c_errors, current_errors)
        p_zero, wins_zero = _sign_test(c_errors, zero_errors)
        c_mae = mae(candidate)
        gains = []
        for base in (current_mae, zero_mae):
            gains.append((base - c_mae) / base if base and c_mae is not None else None)
        passed = (len(samples) >= MIN_NONOVERLAP_N and
                  all(g is not None and g >= MIN_RELATIVE_MAE_GAIN for g in gains) and
                  p_current is not None and p_zero is not None and
                  p_current <= MAX_SIGN_TEST_P and p_zero <= MAX_SIGN_TEST_P)
        candidate_results[candidate] = {
            "mae_log_return": c_mae,
            "relative_mae_gain_vs_current": gains[0],
            "relative_mae_gain_vs_zero": gains[1],
            "paired_sign_test_p_vs_current": p_current,
            "paired_sign_test_p_vs_zero": p_zero,
            "wins_vs_current": wins_current, "wins_vs_zero": wins_zero,
            "recommendation": "eligible_for_human_review" if passed else "retain_current_model",
        }
    return {
        "horizon_days": horizon, "validation_start": samples[0]["origin_date"] if samples else None,
        "validation_end": samples[-1]["target_date"] if samples else None,
        "non_overlapping_n": len(samples), "current_model": CURRENT_MODEL,
        "current_model_version": CURRENT_VERSION, "current_mae_log_return": current_mae,
        "zero_baseline_mae_log_return": zero_mae, "candidates": candidate_results,
    }


def run(prices_payload, ledger_payload, generated_at=None):
    prices = _prices(prices_payload)
    matured = [r for r in ledger_payload.get("forecasts", [])
               if r.get("status") == "matured" and r.get("evaluation_eligible", True)
               and isinstance(r.get("realized_log_return"), (int, float))]
    result = {
        "schema_version": 1,
        "generated_at_utc": generated_at or datetime.now(timezone.utc).isoformat(),
        "method": "Chronological 60% initial training period; later validation origins spaced by horizon; each forecast uses data through its origin only; exact calendar targets and consecutive daily prices required.",
        "matured_live_forecasts_evaluated": len(matured),
        "candidate_models": {
            "expanding_mean_v1": "Mean of all daily log returns available before the forecast origin."
        },
        "evidence_policy": {
            "minimum_non_overlapping_origins": MIN_NONOVERLAP_N,
            "minimum_mae_gain_vs_current_and_zero": MIN_RELATIVE_MAE_GAIN,
            "maximum_one_sided_paired_sign_test_p": MAX_SIGN_TEST_P,
            "promotion_mode": "recommendation_only; production model and forecast ledger are never changed",
        },
        "model_versions": [{"model": CURRENT_MODEL, "version": CURRENT_VERSION, "status": "active"}],
        "horizons": {str(h): _evaluate(prices, h) for h in HORIZONS},
        "limitations": [
            "Historical validation is a simulation, not forecasts actually issued at those origins.",
            "Overlapping outcomes are reduced by horizon spacing but market dependence remains.",
            "Small samples retain the current model; passing evidence is not proof of future performance.",
            "No trading recommendation or automated production promotion is made."
        ],
    }
    return result


def main():
    prices = json.loads(PRICES_PATH.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    result = run(prices, ledger)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
