import unittest
import tempfile
import os
import shutil
from utils.validate_submission import validate_submission

class TestValidateSubmission(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.s1_file = os.path.join(self.test_dir, "test_source1.tsv")
        self.s2_file = os.path.join(self.test_dir, "test_source2.tsv")
        self.s3_file = os.path.join(self.test_dir, "test_source3.tsv")

        with open(self.s1_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-001\tAlpha Corp\t100 Main St\tUS\n")
            f.write("S1-002\tBeta LLC\t200 Oak Ave\tUS\n")

        with open(self.s2_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-001\tAlpha Inc\t100 Main St\tUS\n")

        with open(self.s3_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-001\tBeta Corp\t200 Oak Ave\tUS\n")

        self.cand_file = os.path.join(self.test_dir, "candidate_pairs.tsv")
        self.match_file = os.path.join(self.test_dir, "matching_results.tsv")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_valid_submission(self):
        with open(self.cand_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-001\tS2-001\n")
            f.write("S1-002\tS3-001\n")

        with open(self.match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS2-001\n")
            f.write("S1-002\t\n")

        ok = validate_submission(self.match_file, self.cand_file, self.test_dir)
        self.assertTrue(ok)

    def test_prediction_not_in_candidates_fails(self):
        with open(self.cand_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-001\t\n")
            f.write("S1-002\tS3-001\n")

        with open(self.match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS2-001\n")  # S2-001 not in candidates!
            f.write("S1-002\t\n")

        ok = validate_submission(self.match_file, self.cand_file, self.test_dir)
        self.assertFalse(ok)

    def test_self_match_fails(self):
        with open(self.cand_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-001\tS2-001\n")
            f.write("S1-002\t\n")

        with open(self.match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS1-001\n")  # Self match!
            f.write("S1-002\t\n")

        ok = validate_submission(self.match_file, self.cand_file, self.test_dir)
        self.assertFalse(ok)

    def test_unknown_target_fails(self):
        with open(self.cand_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-001\tS2-999\n")  # S2-999 does not exist!
            f.write("S1-002\t\n")

        with open(self.match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\t\n")
            f.write("S1-002\t\n")

        ok = validate_submission(self.match_file, self.cand_file, self.test_dir)
        self.assertFalse(ok)

if __name__ == "__main__":
    unittest.main()
