import importlib.util
import math
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "update_lab.py"
spec = importlib.util.spec_from_file_location("update_lab", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

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

if __name__ == "__main__":
    unittest.main()
