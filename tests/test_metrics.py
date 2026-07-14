import unittest

from evaluation.metrics import compute_metrics


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

        for name in (
            "accuracy",
            "average_precision",
            "real_accuracy",
            "fake_accuracy",
            "f1",
            "auc",
        ):
            self.assertAlmostEqual(metrics[name], 1.0)
        self.assertAlmostEqual(metrics["eer"], 0.0)

    def test_single_class_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "both real"):
            compute_metrics([0, 0], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
