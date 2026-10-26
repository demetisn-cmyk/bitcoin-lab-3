#!/usr/bin/env python3
"""Deterministic integrity sentinel for Bitcoin Lab's public research artifacts.

Critical failures exit non-zero. The report is written atomically; protected input
artifacts are never rewritten by this module.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "data" / "latest.json"
LEDGER = ROOT / "data" / "forecasts.json"
REPORT = ROOT / "data" / "quality_report.json"
MAX_STALENESS_DAYS = 3
JUMP_WARNING_FRACTION = 0.25


def load_object(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required artifact is missing: {path.relative_to(ROOT)}") from exc
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Malformed JSON in {path.relative_to(ROOT)} at line {exc.lineno}, "
            f"column {exc.colno}; file preserved."
        ) from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"{path.relative_to(ROOT)} must contain a JSON object; file preserved.")
    return obj


def atomic_write(path: Path, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def inspect_artifacts(latest: dict, ledger: dict, today: date | None = None) -> dict:
    today = today or datetime.now(timezone.utc).date()
    warnings: list[dict] = []
    failures: list[dict] = []
    prices = latest.get("prices")
    if not isinstance(prices, list) or not prices:
        failures.append({"code": "prices_missing", "message": "latest.json must contain a non-empty prices list."})
        prices = []

    parsed_dates: list[date] = []
    valid_prices: list[tuple[date, float]] = []
    seen: set[str] = set()
    for index, item in enumerate(prices):
        if not isinstance(item, dict):
            failures.append({"code": "price_record_invalid", "index": index, "message": "Price record is not an object."})
            continue
        raw_date = item.get("date")
        try:
            day = date.fromisoformat(raw_date)
        except (TypeError, ValueError):
            failures.append({"code": "date_invalid", "index": index, "message": f"Invalid ISO date: {raw_date!r}."})
            continue
        if raw_date in seen:
            failures.append({"code": "duplicate_date", "date": raw_date, "message": "Duplicate daily observation."})
        seen.add(raw_date)
        close = item.get("close")
        if isinstance(close, bool) or not isinstance(close, (int, float)) or not math.isfinite(close) or close <= 0:
            failures.append({"code": "close_invalid", "date": raw_date, "message": "Close must be finite and greater than zero."})
            continue
        parsed_dates.append(day)
        valid_prices.append((day, float(close)))

    if parsed_dates and parsed_dates != sorted(parsed_dates):
        failures.append({"code": "dates_unsorted", "message": "Price observations must be chronological."})
    if len(valid_prices) < 100:
        failures.append({"code": "history_too_short", "count": len(valid_prices), "message": "Fewer than 100 valid daily closes."})

    ordered = sorted(valid_prices)
    for (prev_day, prev_close), (cur_day, cur_close) in zip(ordered, ordered[1:]):
        gap = (cur_day - prev_day).days
        if gap > 1:
            warnings.append({"code": "date_gap", "from": prev_day.isoformat(), "to": cur_day.isoformat(), "missing_days": gap - 1})
        if gap == 1:
            move = abs(cur_close / prev_close - 1.0)
            if move >= JUMP_WARNING_FRACTION:
                warnings.append({"code": "implausible_jump_review", "date": cur_day.isoformat(), "simple_return": round(cur_close / prev_close - 1.0, 6), "message": "Large daily move flagged for review, not automatically rejected."})

    if ordered:
        newest = ordered[-1][0]
        age = (today - newest).days
        if age > MAX_STALENESS_DAYS:
            failures.append({"code": "prices_stale", "latest_date": newest.isoformat(), "age_days": age, "message": "Latest close is older than the configured 3-day tolerance."})
        elif age > 1:
            warnings.append({"code": "prices_not_recent", "latest_date": newest.isoformat(), "age_days": age})
    source = latest.get("source")
    if not isinstance(source, str) or not source:
        warnings.append({"code": "source_unknown", "message": "Dataset-level provider is missing; observation-level historical provenance is unknown."})
    if latest.get("source_switch_detected"):
        warnings.append({"code": "provider_switch", "source": source, "message": "Provider differs from the previous refresh. Historical prices may not be directly comparable."})
    if latest.get("historical_observation_provenance") == "unknown_before_observation_level_tracking":
        warnings.append({"code": "historical_provenance_unknown", "message": "Provider attribution for older individual observations is not known and is not inferred."})

    forecasts = ledger.get("forecasts")
    if not isinstance(forecasts, list):
        failures.append({"code": "ledger_schema_invalid", "message": "forecasts.json must contain a forecasts list."})
        forecasts = []
    else:
        ids: set[str] = set()
        for i, forecast in enumerate(forecasts):
            if not isinstance(forecast, dict):
                failures.append({"code": "ledger_record_invalid", "index": i, "message": "Forecast record is not an object."})
                continue
            fid = forecast.get("id")
            if not isinstance(fid, str) or not fid:
                failures.append({"code": "forecast_id_missing", "index": i, "message": "Forecast record has no non-empty ID."})
            elif fid in ids:
                warnings.append({"code": "duplicate_forecast_id", "id": fid, "message": "Duplicate ID retained for audit; exclude duplicates from metrics."})
            ids.add(fid)

    return {
        "schema_version": 1,
        "status": "fail" if failures else ("warn" if warnings else "ok"),
        "generated_at_utc": latest.get("last_updated") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_date": ordered[-1][0].isoformat() if ordered else None,
        "source": source,
        "historical_observation_provenance": latest.get("historical_observation_provenance", "unknown"),
        "checks": {
            "price_observations": len(prices),
            "valid_price_observations": len(valid_prices),
            "forecast_records": len(forecasts),
            "max_staleness_days": MAX_STALENESS_DAYS,
            "implausible_jump_warning_threshold": JUMP_WARNING_FRACTION,
        },
        "warnings": warnings,
        "failures": failures,
    }


def main() -> int:
    latest = load_object(LATEST)
    ledger = load_object(LEDGER)
    if not isinstance(ledger.get("forecasts"), list):
        raise RuntimeError("Forecast ledger schema invalid; original file preserved.")
    report = inspect_artifacts(latest, ledger)
    atomic_write(REPORT, report)
    print(f"Data-quality sentinel: {report['status']} ({len(report['warnings'])} warnings, {len(report['failures'])} failures).")
    for issue in report["failures"]:
        print(f"FAIL [{issue['code']}]: {issue['message']}", file=sys.stderr)
    for issue in report["warnings"]:
        print(f"WARN [{issue['code']}]: {issue.get('message', issue.get('date', issue.get('id', 'review required')))}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
