import unittest
import numpy as np
import pandas as pd
from src.matching_features.feature_schema import FEATURE_COLUMNS
from src.matching_model.trainer import train_model, predict_probabilities
from src.matching_model.evaluation import evaluate_model_at_threshold, sweep_thresholds
from src.matching_model.candidate_recall_analysis import compute_recall_ceilings
from src.matching_model.error_analysis import analyze_model_errors

class TestMatchingModel(unittest.TestCase):
    def setUp(self):
        # Create a small synthetic training and validation dataset
        np.random.seed(42)
        n_samples = 100
        data = {col: np.random.uniform(0, 1, n_samples) for col in FEATURE_COLUMNS}
        data["source1_entity_id"] = [f"S1-{i//5}" for i in range(n_samples)]
        data["target_entity_id"] = [f"S2-{i}" for i in range(n_samples)]
        data["target_source"] = ["S2"] * n_samples
        # Make label correlate with some features
        labels = (data["name_token_jaccard"] > 0.5) & (data["addr_token_jaccard"] > 0.4)
        data["label"] = labels.astype(int)

        self.df = pd.DataFrame(data)
        self.ground_truth = {
            f"S1-{i}": {f"S2-{i*5 + j}" for j in range(5) if labels[i*5 + j]}
            for i in range(20)
        }

    # 1. Probability prediction range [0, 1]
    def test_probability_prediction(self):
        model, _ = train_model("MODEL-001", self.df)
        probs, _ = predict_probabilities(model, self.df)
        self.assertEqual(len(probs), len(self.df))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    # 2. Thresholding logic
    def test_thresholding(self):
        probs = np.array([0.1, 0.4, 0.6, 0.8])
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-2"],
            "target_entity_id": ["T1", "T2", "T3", "T4"],
            "label": [0, 0, 1, 1]
        })
        gt = {"S1-1": set(), "S1-2": {"T3", "T4"}}
        res = evaluate_model_at_threshold(df, probs, gt, threshold=0.5)
        self.assertEqual(res["predicted_match_count"], 2)
        self.assertEqual(res["macro_f05"], 1.0)

    # 3. Zero-match entity handling
    def test_zero_match_entity(self):
        probs = np.array([0.2, 0.3])
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1"],
            "target_entity_id": ["T1", "T2"],
            "label": [0, 0]
        })
        gt = {"S1-1": set()}
        res = evaluate_model_at_threshold(df, probs, gt, threshold=0.5)
        # Predicted empty, true empty -> F0.5 = 1.0
        self.assertEqual(res["macro_f05"], 1.0)
        self.assertEqual(res["zero_match_accuracy"], 1.0)

    # 4. Zero-match entity penalty when predicted non-empty
    def test_zero_match_penalty(self):
        probs = np.array([0.7])
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1"],
            "target_entity_id": ["T1"],
            "label": [0]
        })
        gt = {"S1-1": set()}
        res = evaluate_model_at_threshold(df, probs, gt, threshold=0.5)
        self.assertEqual(res["macro_f05"], 0.0)
        self.assertEqual(res["zero_match_accuracy"], 0.0)

    # 5. Singleton entity evaluation
    def test_singleton_entity(self):
        probs = np.array([0.8, 0.2])
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1"],
            "target_entity_id": ["T1", "T2"],
            "label": [1, 0]
        })
        gt = {"S1-1": {"T1"}}
        res = evaluate_model_at_threshold(df, probs, gt, threshold=0.5)
        self.assertEqual(res["singleton_f05"], 1.0)

    # 6. Multi-match entity evaluation
    def test_multi_match_entity(self):
        probs = np.array([0.9, 0.85, 0.1])
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-1"],
            "target_entity_id": ["T1", "T2", "T3"],
            "label": [1, 1, 0]
        })
        gt = {"S1-1": {"T1", "T2"}}
        res = evaluate_model_at_threshold(df, probs, gt, threshold=0.5)
        self.assertEqual(res["multi_match_f05"], 1.0)

    # 7. Macro averaging
    def test_macro_averaging(self):
        # S1-1 is perfect (F0.5 = 1.0), S1-2 has false positive (F0.5 = 0.0)
        probs = np.array([0.8, 0.8])
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "target_entity_id": ["T1", "T2"],
            "label": [1, 0]
        })
        gt = {"S1-1": {"T1"}, "S1-2": set()}
        res = evaluate_model_at_threshold(df, probs, gt, threshold=0.5)
        self.assertAlmostEqual(res["macro_f05"], 0.5, places=3)

    # 8. Deterministic inference
    def test_deterministic_inference(self):
        model, _ = train_model("MODEL-001", self.df, random_state=42)
        p1, _ = predict_probabilities(model, self.df)
        p2, _ = predict_probabilities(model, self.df)
        np.testing.assert_array_almost_equal(p1, p2)

    # 9. Schema compatibility
    def test_schema_compatibility(self):
        model, _ = train_model("MODEL-003", self.df, feature_cols=FEATURE_COLUMNS[:10])
        probs, _ = predict_probabilities(model, self.df, feature_cols=FEATURE_COLUMNS[:10])
        self.assertEqual(len(probs), len(self.df))

    # 10. Recall ceiling calculation
    def test_recall_ceiling_calculation(self):
        probs = np.array([0.9, 0.2])
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1"],
            "target_entity_id": ["T1", "T2"],
            "label": [1, 1]
        })
        gt = {"S1-1": {"T1", "T2", "T3"}}  # T3 was blocked / never generated
        ceilings = compute_recall_ceilings(df, probs, gt, threshold=0.5)
        self.assertAlmostEqual(ceilings["candidate_generation_recall_ceiling_pct"], 66.67, places=1)
        self.assertAlmostEqual(ceilings["model_recall_within_candidates_pct"], 50.0, places=1)
        self.assertAlmostEqual(ceilings["final_end_to_end_recall_pct"], 33.33, places=1)

if __name__ == "__main__":
    unittest.main()
