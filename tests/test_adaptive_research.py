import importlib.util
import unittest
from datetime import date, timedelta
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts" / "adaptive_research.py"
spec = importlib.util.spec_from_file_location("adaptive_research", PATH)
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


class AdaptiveResearchTests(unittest.TestCase):
    def test_training_prediction_uses_only_past_prices(self):
        prices = [(date(2020, 1, 1) + timedelta(days=i), 100 + i) for i in range(220)]
        before = loop._predict(prices, 180, "expanding_mean_v1")
        prices[181] = (prices[181][0], 1_000_000)
        after = loop._predict(prices, 180, "expanding_mean_v1")
        self.assertEqual(before, after)

    def test_insufficient_evidence_retains_current_model(self):
        prices = [{"date": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
                   "close": 100 + i} for i in range(220)]
        result = loop.run({"prices": prices}, {"forecasts": []}, "fixed")
        for horizon in ("30", "90"):
            candidate = result["horizons"][horizon]["candidates"]["expanding_mean_v1"]
            self.assertEqual(candidate["recommendation"], "retain_current_model")

    def test_matured_forecast_count_excludes_pending_and_ineligible(self):
        ledger = {"forecasts": [
            {"status": "matured", "realized_log_return": 0.1},
            {"status": "pending", "realized_log_return": None},
            {"status": "matured", "realized_log_return": 0.1, "evaluation_eligible": False},
        ]}
        result = loop.run({"prices": []}, ledger, "fixed")
        self.assertEqual(result["matured_live_forecasts_evaluated"], 1)


if __name__ == "__main__":
    unittest.main()
