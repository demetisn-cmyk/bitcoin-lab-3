import importlib.util
import json
import math
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "update_lab.py"
spec = importlib.util.spec_from_file_location("update_lab", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ARENA_PATH = Path(__file__).resolve().parents[1] / "scripts" / "model_arena.py"
arena_spec = importlib.util.spec_from_file_location("model_arena", ARENA_PATH)
arena = importlib.util.module_from_spec(arena_spec)
arena_spec.loader.exec_module(arena)

SENTINEL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_sentinel.py"
sentinel_spec = importlib.util.spec_from_file_location("research_sentinel", SENTINEL_PATH)
sentinel = importlib.util.module_from_spec(sentinel_spec)
sentinel_spec.loader.exec_module(sentinel)

class ForecastPipelineTests(unittest.TestCase):
    def test_log_return(self):
        prices = [{"date":"2026-01-01","close":100.0},{"date":"2026-01-02","close":110.0}]
        got = module.log_returns(prices)
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0][1], math.log(1.1), places=10)

    def test_kraken_parser_excludes_open_daily_candle(self):
        from datetime import datetime, timezone
        now = int(datetime(2026, 10, 26, 12, tzinfo=timezone.utc).timestamp())
        payload = {"error": [], "result": {"XXBTZEUR": [
            [int(datetime(2026, 10, 24, tzinfo=timezone.utc).timestamp()), "1", "1", "1", "100", "1", "1", 1],
            [int(datetime(2026, 10, 26, tzinfo=timezone.utc).timestamp()), "1", "1", "1", "999", "1", "1", 1],
        ], "last": 0}}
        daily = {}
        for candle in payload["result"]["XXBTZEUR"]:
            opened, close = int(candle[0]), float(candle[4])
            if opened + 86400 <= now:
                daily[datetime.fromtimestamp(opened, tz=timezone.utc).date().isoformat()] = close
        self.assertEqual(len(daily), 1)
        self.assertEqual(list(daily.values()), [100.0])

    def test_quantile_bounds(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(module.quantile(values, 0), 1.0)
        self.assertEqual(module.quantile(values, 1), 5.0)
        self.assertAlmostEqual(module.quantile(values, 0.5), 3.0)

    def test_forecast_is_prospective_and_bounded(self):
        prices = []
        p = 100.0
        from datetime import date, timedelta
        start = date(2025, 1, 1)
        for i in range(250):
            p *= 1.001 if i % 3 else 0.998
            prices.append({"date":(start + timedelta(days=i)).isoformat(),"close":p})
        f = module.build_forecast(prices, 30, start)
        self.assertEqual(f["status"], "pending")
        self.assertIsNone(f["realized_log_return"])
        self.assertEqual(f["horizon_days"], 30)
        self.assertTrue(math.isfinite(f["risk_adjusted_score"]))

    def test_forecast_model_version_and_cutoff_are_recorded(self):
        prices = []
        p = 100.0
        start = date(2025, 1, 1)
        for i in range(250):
            p *= 1.001 if i % 3 else 0.998
            prices.append({"date": (start + timedelta(days=i)).isoformat(), "close": p})
        forecast = module.build_forecast(prices, 30, start)
        self.assertEqual(forecast["model_version"], "1.0")
        self.assertEqual(forecast["data_cutoff_date"], forecast["origin_date"])
        self.assertIn("shrunk_trailing_mean_v1.0", forecast["id"])
        repeated = module.build_forecast(prices, 30, start)
        self.assertEqual(forecast["id"], repeated["id"])

    def test_maturity_requires_exact_target_date(self):
        ledger = {"forecasts": [{
            "status": "pending", "target_date": "2026-01-04",
            "origin_price_eur": 100.0, "realized_log_return": None
        }]}
        prices = [
            {"date": "2026-01-01", "close": 100.0},
            {"date": "2026-01-02", "close": 101.0},
            {"date": "2026-01-03", "close": 102.0},
            {"date": "2026-01-05", "close": 105.0},
        ]
        self.assertFalse(module.mature_forecasts(ledger, prices))
        self.assertEqual(ledger["forecasts"][0]["status"], "pending")

    def test_maturity_uses_exact_target_close(self):
        ledger = {"forecasts": [{
            "status": "pending", "target_date": "2026-01-02",
            "origin_price_eur": 100.0, "realized_log_return": None
        }]}
        prices = [
            {"date": "2026-01-01", "close": 100.0},
            {"date": "2026-01-02", "close": 110.0},
        ]
        self.assertTrue(module.mature_forecasts(ledger, prices))
        forecast = ledger["forecasts"][0]
        self.assertEqual(forecast["status"], "matured")
        self.assertAlmostEqual(forecast["realized_log_return"], math.log(1.1), places=8)
        self.assertEqual(forecast["realized_date"], "2026-01-02")
        self.assertIsNotNone(forecast["matured_at_utc"])

    def test_evaluation_metrics_and_zero_direction(self):
        result = arena.summarize([(0.1, 0.2), (-0.1, -0.2), (0.0, 0.1)])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mae"], 0.1)
        self.assertAlmostEqual(result["rmse"], math.sqrt(0.01))
        self.assertEqual(result["directional_n"], 3)
        self.assertAlmostEqual(result["directional_accuracy"], 2 / 3, places=5)
        empty = arena.summarize([])
        self.assertIsNone(empty["mae"])
        self.assertIsNone(empty["directional_accuracy"])

    def test_walk_forward_predictions_do_not_use_future_prices(self):
        prices = []
        start = date(2025, 1, 1)
        p = 100.0
        for i in range(60):
            p *= 1.01 if i % 2 else 0.995
            prices.append({"date": (start + timedelta(days=i)).isoformat(), "close": p})
        origin = 25
        original = arena.predict_at_origin(prices, origin, 5, lookback=10)
        changed = [dict(item) for item in prices]
        for i in range(origin + 1, len(changed)):
            changed[i]["close"] *= 1.5
        modified = arena.predict_at_origin(changed, origin, 5, lookback=10)
        self.assertEqual(original[0], modified[0])
        self.assertNotEqual(original[1], modified[1])

    def test_walk_forward_outputs_separate_horizons_and_nonoverlap(self):
        prices = []
        start = date(2025, 1, 1)
        p = 100.0
        for i in range(240):
            p *= 1.003 if i % 4 else 0.99
            prices.append({"date": (start + timedelta(days=i)).isoformat(), "close": p})
        result = arena.run_walk_forward(prices, horizons=(10, 20), lookback=30)
        self.assertEqual(set(result["horizons"]), {"10", "20"})
        for horizon in ("10", "20"):
            section = result["horizons"][horizon]
            self.assertGreater(section["overlapping"]["zero_return_v1"]["n"], 0)
            self.assertGreater(section["non_overlapping"]["zero_return_v1"]["n"], 0)
            self.assertIn("momentum_30_v1", section["overlapping"])


    def test_corrupt_json_fails_closed_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forecasts.json"
            original = b'{"forecasts": [broken'
            path.write_bytes(original)
            with self.assertRaises(RuntimeError):
                module.read_json(path, {"forecasts": []})
            self.assertEqual(path.read_bytes(), original)

    def test_log_returns_skips_missing_calendar_days(self):
        prices = [
            {"date": "2026-01-01", "close": 100.0},
            {"date": "2026-01-02", "close": 110.0},
            {"date": "2026-01-04", "close": 121.0},
            {"date": "2026-01-05", "close": 120.0},
        ]
        got = module.log_returns(prices)
        self.assertEqual([item[0] for item in got], ["2026-01-02", "2026-01-05"])
        self.assertAlmostEqual(got[0][1], math.log(1.1))

    def test_zero_return_baseline_direction_is_unavailable(self):
        result = arena.summarize([(0.0, 0.2), (0.0, -0.1), (0.0, 0.05)])
        self.assertIsNone(result["directional_accuracy"])
        self.assertEqual(result["n"], 3)

    def test_empirical_interval_uses_only_past_outcomes(self):
        prices = []
        start = date(2024, 1, 1)
        p = 100.0
        for i in range(360):
            p *= 1.002 if i % 5 else 0.994
            prices.append({"date": (start + timedelta(days=i)).isoformat(), "close": p})
        origin = 300
        first = arena.empirical_horizon_interval(prices, origin, 10, min_samples=10)
        changed = [dict(x) for x in prices]
        for i in range(origin + 1, len(changed)):
            changed[i]["close"] *= 3.0
        second = arena.empirical_horizon_interval(changed, origin, 10, min_samples=10)
        self.assertEqual(first, second)

    def test_regime_classification_is_past_only(self):
        prices = []
        start = date(2024, 1, 1)
        p = 100.0
        for i in range(250):
            p *= 1.003 if i % 3 else 0.995
            prices.append({"date": (start + timedelta(days=i)).isoformat(), "close": p})
        origin = 220
        before = arena.classify_regime(prices, origin)
        altered = [dict(x) for x in prices]
        for i in range(origin + 1, len(altered)):
            altered[i]["close"] *= 0.1
        self.assertEqual(before, arena.classify_regime(altered, origin))

    def test_sentinel_reports_duplicates_gaps_and_provider_switch(self):
        today = date(2026, 1, 10)
        prices = [
            {"date": "2026-01-07", "close": 100.0},
            {"date": "2026-01-08", "close": 101.0},
            {"date": "2026-01-08", "close": 101.0},
            {"date": "2026-01-10", "close": 102.0},
        ]
        latest = {"prices": prices, "source": "Binance", "source_switch_detected": True,
                  "historical_observation_provenance": "unknown_before_observation_level_tracking",
                  "last_updated": "2026-01-10T00:00:00+00:00"}
        report = sentinel.inspect_artifacts(latest, {"forecasts": []}, today=today)
        codes = {item["code"] for item in report["warnings"] + report["failures"]}
        self.assertIn("duplicate_date", codes)
        self.assertIn("date_gap", codes)
        self.assertIn("provider_switch", codes)
        self.assertEqual(report["status"], "fail")

    def test_production_kraken_parser_with_mocked_provider_response(self):
        now = datetime.now(timezone.utc)
        base = (now.date() - timedelta(days=120))
        candles = []
        for i in range(110):
            opened = datetime.combine(base + timedelta(days=i), datetime.min.time(), tzinfo=timezone.utc)
            candles.append([int(opened.timestamp()), "1", "1", "1", str(100 + i), "1", "1", 1])
        today_open = datetime.combine(now.date(), datetime.min.time(), tzinfo=timezone.utc)
        candles.append([int(today_open.timestamp()), "1", "1", "1", "999999", "1", "1", 1])
        payload = {"error": [], "result": {"XXBTZEUR": candles, "last": 0}}
        with patch.object(module, "request_json", return_value=payload):
            prices = module.fetch_kraken_prices()
        self.assertEqual(len(prices), 110)
        self.assertNotEqual(prices[-1]["close"], 999999.0)

    def test_atomic_json_write_is_noop_for_identical_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            self.assertTrue(module.write_json(path, {"ok": True}))
            before = path.stat().st_mtime_ns
            self.assertFalse(module.write_json(path, {"ok": True}))
            self.assertEqual(path.stat().st_mtime_ns, before)


    def test_legacy_duplicate_is_preserved_but_excluded_from_metrics(self):
        legacy = {"id": "2026-01-01-30d", "origin_date": "2026-01-01",
                  "target_date": "2026-01-31", "horizon_days": 30,
                  "origin_price_eur": 100.0, "model": "shrunk_trailing_mean_v1",
                  "predicted_log_return": 0.02, "status": "matured",
                  "realized_log_return": 0.01}
        versioned = dict(legacy, id="2026-01-01-30d-shrunk-v1", model_version="1.0")
        ledger = {"forecasts": [legacy, versioned]}
        self.assertTrue(module.annotate_ledger_records(ledger))
        self.assertEqual(len(ledger["forecasts"]), 2)
        self.assertEqual(ledger["forecasts"][0]["model_version"], "legacy_unknown")
        self.assertFalse(ledger["forecasts"][0]["evaluation_eligible"])
        self.assertTrue(ledger["forecasts"][1]["evaluation_eligible"])
        self.assertEqual(ledger["forecasts"][0]["evaluation_exclusion_reason"],
                         "equivalent_forecast_record_retained_for_history_but_excluded_from_metrics")


    def test_nonoverlap_uses_calendar_days_not_row_count(self):
        origins = [
            {"origin_date": "2025-01-01"},
            {"origin_date": "2025-01-03"},
            {"origin_date": "2025-01-07"},
            {"origin_date": "2025-01-10"},
            {"origin_date": "2025-01-12"},
        ]
        selected = arena.select_nonoverlap_origin_dates(origins, 5)
        self.assertEqual(selected, {"2025-01-01", "2025-01-07", "2025-01-12"})

    def test_schema_invalid_ledger_is_rejected_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forecasts.json"
            original = b'{"forecasts":[{"id":"bad"}]}'
            path.write_bytes(original)
            with self.assertRaises(RuntimeError):
                module.validate_persistent_schemas({}, {"forecasts": [{"id": "bad"}]})
            self.assertEqual(path.read_bytes(), original)


    def test_horizon_returns_require_exact_consecutive_dates(self):
        prices = [
            {"date": "2026-01-01", "close": 100.0},
            {"date": "2026-01-02", "close": 110.0},
            {"date": "2026-01-04", "close": 121.0},
            {"date": "2026-01-05", "close": 120.0},
            {"date": "2026-01-06", "close": 132.0},
        ]
        got = module.horizon_log_returns(prices, 2)
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0], math.log(132.0 / 120.0))

    def test_forecast_score_uses_horizon_matched_method_version(self):
        prices = []
        start = date(2025, 1, 1)
        p = 100.0
        for i in range(260):
            p *= 1.001 if i % 3 else 0.998
            prices.append({"date": (start + timedelta(days=i)).isoformat(), "close": p})
        forecast = module.build_forecast(prices, 30, start)
        self.assertEqual(forecast["risk_score_version"], "empirical_horizon_v1")
        self.assertTrue(math.isfinite(forecast["downside_tail_proxy"]))
        self.assertTrue(math.isfinite(forecast["uncertainty_penalty"]))

if __name__ == "__main__":
    unittest.main()
