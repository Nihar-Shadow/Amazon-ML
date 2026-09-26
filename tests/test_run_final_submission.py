import unittest
import tempfile
import shutil
import os
from scripts.run_final_submission import run_submission_pipeline

class TestFinalSubmissionPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.temp_dir, "test")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.test_data_dir, exist_ok=True)

        # Write miniature test files
        s1_file = os.path.join(self.test_data_dir, "test_source1.tsv")
        s2_file = os.path.join(self.test_data_dir, "test_source2.tsv")
        s3_file = os.path.join(self.test_data_dir, "test_source3.tsv")

        with open(s1_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-100\tOmega Electronics Corp\t123 Tech Blvd, Austin, TX\tUS\n")
            f.write("S1-101\tZeta Technologies Inc\t456 Cyber Way, Dallas, TX\tUS\n")

        with open(s2_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-100\tOmega Electronics\t123 Tech Blvd, Austin, TX\tUS\n")

        with open(s3_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-101\tZeta Technologies\t456 Cyber Way, Dallas, TX\tUS\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        zip_p = os.path.join(os.getcwd(), "pipeline_test_submission.zip")
        if os.path.exists(zip_p):
            try:
                os.remove(zip_p)
            except OSError:
                pass

    def test_end_to_end_pipeline(self):
        diag = run_submission_pipeline(
            test_dir=self.test_data_dir,
            output_dir=self.output_dir,
            team_name="pipeline_test",
            create_zip=True
        )
        self.assertTrue(diag["validation_passed"])
        self.assertEqual(diag["processed_s1_count"], 2)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "candidate_pairs.tsv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "matching_results.tsv")))
        self.assertTrue(os.path.exists(diag["zip_path"]))

if __name__ == "__main__":
    unittest.main()
