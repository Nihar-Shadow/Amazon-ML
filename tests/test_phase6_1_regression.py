import unittest
import tempfile
import shutil
import os
import hashlib
from src.candidate_generation.normalizer import normalize_country, normalize_entity_record
from src.candidate_generation.blocking import BlockingIndex
from scripts.run_final_submission import run_submission_pipeline

class TestPhase61Regression(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.temp_dir, "test")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.test_data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        zip_p = os.path.join(os.getcwd(), "regression_test_submission.zip")
        if os.path.exists(zip_p):
            try:
                os.remove(zip_p)
            except OSError:
                pass

    def test_country_normalization_france(self):
        """1. 'France' == normalized 'FRANCE'"""
        self.assertEqual(normalize_country("France"), "FRANCE")
        self.assertEqual(normalize_country("france"), "FRANCE")
        self.assertEqual(normalize_country("FRANCE"), "FRANCE")

    def test_country_normalization_india(self):
        """2. 'India' == normalized 'INDIA'"""
        self.assertEqual(normalize_country("India"), "INDIA")
        self.assertEqual(normalize_country("india"), "INDIA")
        self.assertEqual(normalize_country("INDIA"), "INDIA")

    def test_country_normalization_us(self):
        """3. 'US' == normalized 'US'"""
        self.assertEqual(normalize_country("US"), "US")
        self.assertEqual(normalize_country("us"), "US")
        self.assertEqual(normalize_country("Us"), "US")

    def test_arbitrary_future_country(self):
        """4. Open-set normalization handles arbitrary future country values without hardcoding."""
        self.assertEqual(normalize_country("Germany"), "GERMANY")
        self.assertEqual(normalize_country("brazil "), "BRAZIL")
        self.assertEqual(normalize_country("  Japan  "), "JAPAN")
        self.assertEqual(normalize_country("uk"), "UK")

    def test_no_cross_country_candidates(self):
        """5. Country partitioning guarantees no cross-country candidate generation."""
        index = BlockingIndex(max_block_size=500)
        # Target in France
        t_rec_fr = normalize_entity_record("S2-FR1", "Acme Bakery", "10 Rue Paris", "France")
        index.add_target_record(t_rec_fr)
        
        # Query in India
        q_rec_in = normalize_entity_record("S1-IN1", "Acme Bakery", "10 Rue Paris", "India")
        cands = index.retrieve_candidates_for_query(q_rec_in)
        self.assertEqual(len(cands), 0, "Cross-country candidate must never be generated")

        # Query in France
        q_rec_fr = normalize_entity_record("S1-FR1", "Acme Bakery", "10 Rue Paris", "France")
        cands_fr = index.retrieve_candidates_for_query(q_rec_fr)
        self.assertIn("S2-FR1", cands_fr)

    def test_country_partitioning_pipeline(self):
        """6 & 7. Country partitioning & bounded streamed S1 processing in pipeline."""
        s1_file = os.path.join(self.test_data_dir, "test_source1.tsv")
        s2_file = os.path.join(self.test_data_dir, "test_source2.tsv")
        s3_file = os.path.join(self.test_data_dir, "test_source3.tsv")

        with open(s1_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-FR1\tBoulangerie Paul Sarl\t12 Rue de Rivoli, Paris\tFrance\n")
            f.write("S1-IN1\tTata Consultancy Pvt Ltd\tNariman Point, Mumbai\tIndia\n")
            f.write("S1-US1\tApple Computer Inc\t1 Infinite Loop, Cupertino, CA\tUS\n")

        with open(s2_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-FR1\tBoulangerie Paul\t12 Rue de Rivoli, Paris\tFrance\n")
            f.write("S2-IN1\tTata Consultancy\tNariman Point, Mumbai\tIndia\n")

        with open(s3_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-US1\tApple Computer\t1 Infinite Loop, Cupertino, CA\tUS\n")

        diag = run_submission_pipeline(
            test_dir=self.test_data_dir,
            output_dir=self.output_dir,
            team_name="regression_test",
            create_zip=False
        )

        self.assertTrue(diag["validation_passed"])
        self.assertEqual(diag["processed_s1_count"], 3)
        self.assertIn("FRANCE", diag["country_diagnostics"])
        self.assertIn("INDIA", diag["country_diagnostics"])
        self.assertIn("US", diag["country_diagnostics"])
        self.assertGreater(diag["country_diagnostics"]["FRANCE"]["targets"], 0)
        self.assertGreater(diag["country_diagnostics"]["INDIA"]["targets"], 0)
        self.assertGreater(diag["country_diagnostics"]["US"]["targets"], 0)

    def test_candidate_set_determinism(self):
        """8. Candidate set and matching results remain 100% deterministic across runs."""
        s1_file = os.path.join(self.test_data_dir, "test_source1.tsv")
        s2_file = os.path.join(self.test_data_dir, "test_source2.tsv")
        s3_file = os.path.join(self.test_data_dir, "test_source3.tsv")

        with open(s1_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-FR1\tBoulangerie Paul Sarl\t12 Rue de Rivoli, Paris\tFrance\n")
            f.write("S1-IN1\tTata Consultancy Pvt Ltd\tNariman Point, Mumbai\tIndia\n")

        with open(s2_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-FR1\tBoulangerie Paul\t12 Rue de Rivoli, Paris\tFrance\n")

        with open(s3_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-IN1\tTata Consultancy\tNariman Point, Mumbai\tIndia\n")

        out1 = os.path.join(self.temp_dir, "out1")
        out2 = os.path.join(self.temp_dir, "out2")

        run_submission_pipeline(test_dir=self.test_data_dir, output_dir=out1, create_zip=False)
        run_submission_pipeline(test_dir=self.test_data_dir, output_dir=out2, create_zip=False)

        def get_sha256(p):
            with open(p, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        h_cand1 = get_sha256(os.path.join(out1, "candidate_pairs.tsv"))
        h_cand2 = get_sha256(os.path.join(out2, "candidate_pairs.tsv"))
        self.assertEqual(h_cand1, h_cand2, "Candidate pairs must be bit-for-bit identical")

        h_match1 = get_sha256(os.path.join(out1, "matching_results.tsv"))
        h_match2 = get_sha256(os.path.join(out2, "matching_results.tsv"))
        self.assertEqual(h_match1, h_match2, "Matching results must be bit-for-bit identical")

if __name__ == "__main__":
    unittest.main()
