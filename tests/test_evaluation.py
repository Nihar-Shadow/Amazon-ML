import os
import tempfile
import unittest
from src.data.ground_truth import parse_ground_truth, validate_entity_id
from src.evaluation.metrics import (
    compute_single_s1_metrics,
    evaluate_predictions,
    evaluate_candidate_blocking
)
from src.evaluation.validation import create_deterministic_split

class TestGroundTruthParser(unittest.TestCase):
    
    def test_parse_valid_ground_truth(self):
        content = (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-101\tS2-201,S3-301\n"
            "S1-102\t\n"  # Singleton
            "S1-103\tS2-202\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv", encoding="utf-8") as f:
            f.write(content)
            temp_path = f.name
            
        try:
            gt = parse_ground_truth(temp_path)
            self.assertEqual(len(gt), 3)
            self.assertEqual(gt["S1-101"], {"S2-201", "S3-301"})
            self.assertEqual(gt["S1-102"], set())  # Empty set for singleton
            self.assertEqual(gt["S1-103"], {"S2-202"})
        finally:
            os.remove(temp_path)
            
    def test_empty_match_list_is_empty_set(self):
        content = "source1_entity_id\tmatched_entity_ids\nS1-999\t\n"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv", encoding="utf-8") as f:
            f.write(content)
            temp_path = f.name
        try:
            gt = parse_ground_truth(temp_path)
            self.assertIn("S1-999", gt)
            self.assertEqual(gt["S1-999"], set())
        finally:
            os.remove(temp_path)
            
    def test_duplicate_target_id_raises_error(self):
        content = "source1_entity_id\tmatched_entity_ids\nS1-101\tS2-201,S2-201\n"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv", encoding="utf-8") as f:
            f.write(content)
            temp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_ground_truth(temp_path)
            self.assertIn("Duplicate target ID", str(ctx.exception))
        finally:
            os.remove(temp_path)

    def test_malformed_s1_id_raises_error(self):
        content = "source1_entity_id\tmatched_entity_ids\nINVALID_101\tS2-201\n"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv", encoding="utf-8") as f:
            f.write(content)
            temp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_ground_truth(temp_path)
            self.assertIn("Malformed S1 entity ID", str(ctx.exception))
        finally:
            os.remove(temp_path)

    def test_malformed_target_id_raises_error(self):
        content = "source1_entity_id\tmatched_entity_ids\nS1-101\tS4-999\n"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv", encoding="utf-8") as f:
            f.write(content)
            temp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_ground_truth(temp_path)
            self.assertIn("Malformed target entity ID", str(ctx.exception))
        finally:
            os.remove(temp_path)

    def test_id_format_regex(self):
        self.assertTrue(validate_entity_id("S1-12345", is_s1=True))
        self.assertFalse(validate_entity_id("S2-12345", is_s1=True))
        self.assertTrue(validate_entity_id("S2-12345", is_s1=False))
        self.assertTrue(validate_entity_id("S3-999", is_s1=False))
        self.assertFalse(validate_entity_id("S1-abc", is_s1=True))


class TestF05Metric(unittest.TestCase):

    def test_perfect_prediction(self):
        true_set = {"S2-1", "S3-1"}
        pred_set = {"S2-1", "S3-1"}
        m = compute_single_s1_metrics(true_set, pred_set, beta=0.5)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f_beta"], 1.0)
        self.assertEqual(m["exact_match"], 1.0)

    def test_singleton_scoring_correct(self):
        # Empty prediction for singleton -> 1.0
        true_set = set()
        pred_set = set()
        m = compute_single_s1_metrics(true_set, pred_set, beta=0.5)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f_beta"], 1.0)
        self.assertEqual(m["exact_match"], 1.0)

    def test_singleton_scoring_incorrect(self):
        # Non-empty prediction for singleton -> 0.0
        true_set = set()
        pred_set = {"S2-1"}
        m = compute_single_s1_metrics(true_set, pred_set, beta=0.5)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f_beta"], 0.0)
        self.assertEqual(m["exact_match"], 0.0)

    def test_non_singleton_empty_prediction(self):
        true_set = {"S2-1"}
        pred_set = set()
        m = compute_single_s1_metrics(true_set, pred_set, beta=0.5)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f_beta"], 0.0)
        self.assertEqual(m["exact_match"], 0.0)

    def test_f05_precision_weighting(self):
        # Case A: Precision = 1.0, Recall = 0.5 (TP=1, Pred=1, True=2)
        true_set_a = {"S2-1", "S2-2"}
        pred_set_a = {"S2-1"}
        m_a = compute_single_s1_metrics(true_set_a, pred_set_a, beta=0.5)
        # P = 1.0, R = 0.5
        # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.8333
        self.assertAlmostEqual(m_a["f_beta"], 5.0 / 6.0, places=4)

        # Case B: Precision = 0.5, Recall = 1.0 (TP=1, Pred=2, True=1)
        true_set_b = {"S2-1"}
        pred_set_b = {"S2-1", "S2-2"}
        m_b = compute_single_s1_metrics(true_set_b, pred_set_b, beta=0.5)
        # P = 0.5, R = 1.0
        # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = 0.5556
        self.assertAlmostEqual(m_b["f_beta"], 5.0 / 9.0, places=4)

        # Because F0.5 is precision-heavy, Case A (P=1.0, R=0.5) must score higher than Case B (P=0.5, R=1.0)!
        self.assertGreater(m_a["f_beta"], m_b["f_beta"])

    def test_macro_evaluation(self):
        gt = {
            "S1-1": {"S2-1"},       # Will get exact match: F0.5 = 1.0
            "S1-2": set(),           # Singleton, pred=empty: F0.5 = 1.0
            "S1-3": {"S2-2", "S3-2"} # Pred has 1 of 2: P=1.0, R=0.5 -> F0.5 = 5/6 = 0.8333
        }
        preds = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": {"S2-2"}
        }
        res = evaluate_predictions(gt, preds, beta=0.5)
        expected_macro_f05 = (1.0 + 1.0 + (5.0 / 6.0)) / 3.0
        self.assertAlmostEqual(res["macro_f05"], expected_macro_f05, places=4)
        self.assertAlmostEqual(res["exact_match_accuracy"], 2.0 / 3.0, places=4)
        self.assertEqual(res["singleton_count"], 1)
        self.assertEqual(res["singleton_accuracy"], 1.0)


class TestCandidateBlockingMetrics(unittest.TestCase):

    def test_candidate_recall_and_coverage(self):
        gt = {
            "S1-1": {"S2-1", "S3-1"},  # 2 true links
            "S1-2": {"S2-2"},          # 1 true link
            "S1-3": set()              # Singleton (0 links)
        }
        candidates = {
            "S1-1": {"S2-1", "S3-1", "S2-99"}, # Captured both true links!
            "S1-2": {"S2-99"},                 # Missed S2-2
            "S1-3": set()                      # 0 candidates for singleton
        }
        res = evaluate_candidate_blocking(gt, candidates)
        
        # Total true links = 3 (S2-1, S3-1, S2-2)
        # Retrieved = 2 (S2-1, S3-1)
        # Overall recall = 2/3 = 0.6667
        self.assertAlmostEqual(res["candidate_recall_overall"], 2.0 / 3.0, places=4)
        
        # S2 recall: true S2 = S2-1, S2-2 (total 2); retrieved S2 = S2-1 (1) -> 1/2 = 0.5
        self.assertAlmostEqual(res["candidate_recall_s2"], 0.5, places=4)
        
        # S3 recall: true S3 = S3-1 (total 1); retrieved S3 = S3-1 (1) -> 1/1 = 1.0
        self.assertAlmostEqual(res["candidate_recall_s3"], 1.0, places=4)
        
        # S1 coverage: non-singleton S1 count = 2 (S1-1, S1-2). S1-1 is fully covered, S1-2 is not -> 1/2 = 0.5
        self.assertAlmostEqual(res["s1_coverage"], 0.5, places=4)
        
        # Singleton diagnostics: 1 singleton, 0 candidates generated -> 100% clean
        self.assertEqual(res["singleton_s1_count"], 1)
        self.assertEqual(res["singleton_pct_clean_zero_cands"], 100.0)


class TestDeterministicValidationSplit(unittest.TestCase):

    def test_deterministic_split_reproducibility(self):
        ids = [f"S1-{i}" for i in range(1000)]
        train1, val1 = create_deterministic_split(ids, val_ratio=0.20, seed=42)
        train2, val2 = create_deterministic_split(ids, val_ratio=0.20, seed=42)
        
        # Exactly identical across runs with same seed
        self.assertEqual(train1, train2)
        self.assertEqual(val1, val2)
        
        # Exact 80/20 proportions
        self.assertEqual(len(val1), 200)
        self.assertEqual(len(train1), 800)
        
        # Zero overlap between train and val
        self.assertEqual(len(set(train1) & set(val1)), 0)
        
        # Union matches original
        self.assertEqual(set(train1) | set(val1), set(ids))

    def test_different_seed_different_split(self):
        ids = [f"S1-{i}" for i in range(1000)]
        train1, val1 = create_deterministic_split(ids, val_ratio=0.20, seed=42)
        train2, val2 = create_deterministic_split(ids, val_ratio=0.20, seed=123)
        self.assertNotEqual(train1, train2)
        self.assertNotEqual(val1, val2)

if __name__ == "__main__":
    unittest.main()
