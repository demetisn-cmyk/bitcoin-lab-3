import importlib.util
import math
import unittest
from datetime import date, timedelta
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "update_lab.py"
spec = importlib.util.spec_from_file_location("update_lab", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ARENA_PATH = Path(__file__).resolve().parents[1] / "scripts" / "model_arena.py"
arena_spec = importlib.util.spec_from_file_location("model_arena", ARENA_PATH)
arena = importlib.util.module_from_spec(arena_spec)
arena_spec.loader.exec_module(arena)

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

if __name__ == "__main__":
    unittest.main()
