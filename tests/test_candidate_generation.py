import os
import tempfile
import unittest
from src.candidate_generation.normalizer import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    normalize_entity_record,
    strip_accents
)
from src.candidate_generation.blocking import BlockingIndex, extract_blocking_keys
from src.candidate_generation.candidate_pool import CandidatePool
from src.candidate_generation.candidate_refiner import CandidateRefiner, score_candidate_pair
from src.candidate_generation.candidate_generator import CandidateGenerator
from src.candidate_generation.diagnostics import compute_detailed_candidate_metrics

class TestCandidateGenerationSuite(unittest.TestCase):

    # 1. Normalization
    def test_normalization_basic(self):
        norm = normalize_business_name("  Zephay Labs, Inc.  ")
        self.assertEqual(norm["normalized"], "zephay labs")
        self.assertEqual(norm["compact"], "zephaylabs")
        self.assertIn("zephay", norm["tokens"])
        self.assertIn("labs", norm["tokens"])

    # 2. Exact blocking
    def test_exact_blocking(self):
        index = BlockingIndex()
        target = normalize_entity_record("S2-101", "Acme Hardware", "123 Main St", "US")
        index.add_target_record(target, strategies=["exact_name"])
        
        query = normalize_entity_record("S1-1", "Acme Hardware LLC", "123 Main St", "US")
        cands = index.retrieve_candidates_for_query(query, strategies=["exact_name"])
        self.assertIn("S2-101", cands)

    # 3. Multi-block union
    def test_multi_block_union(self):
        index = BlockingIndex()
        t1 = normalize_entity_record("S2-1", "Acme Corp", "123 Main St", "US")
        t2 = normalize_entity_record("S2-2", "Global Logistics", "123 Main St", "US")
        index.add_target_record(t1)
        index.add_target_record(t2)
        
        # Query matches t1 on name, and matches t2 on address combo
        query = normalize_entity_record("S1-1", "Acme Inc", "123 Main Street", "US")
        cands = index.retrieve_candidates_for_query(query)
        self.assertIn("S2-1", cands)
        self.assertIn("S2-2", cands)

    # 4. Duplicate candidate removal
    def test_duplicate_candidate_removal(self):
        pool = CandidatePool()
        pool.add_candidate("S1-1", "S2-100", "exact_name")
        pool.add_candidate("S1-1", "S2-100", "sorted_tokens")
        pool.add_candidate("S1-1", "S2-100", "name_prefix")
        
        cands = pool.get_candidate_ids("S1-1")
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands, {"S2-100"})
        # Provenance contains all 3
        prov = pool.get_provenance("S1-1", "S2-100")
        self.assertEqual(prov, {"exact_name", "sorted_tokens", "name_prefix"})

    # 5. Candidate refinement
    def test_candidate_refinement(self):
        refiner = CandidateRefiner(min_score_threshold=0.20, max_candidates_per_query=10)
        q = normalize_entity_record("S1-1", "Alpha Omega Clinic", "500 Elm Street", "US")
        
        # High similarity target
        good_t = normalize_entity_record("S2-1", "Alpha Omega Clinic LLC", "500 Elm St", "US")
        # Irrelevant target retrieved by generic token
        bad_t = normalize_entity_record("S2-2", "Zeta Beta Bakery", "999 Oak Ave", "US")
        
        recs = {"S2-1": good_t, "S2-2": bad_t}
        filtered = refiner.refine_candidates_for_query(q, recs)
        self.assertIn("S2-1", filtered)
        self.assertNotIn("S2-2", filtered)

    # 6. Missing address in query/target
    def test_missing_address_handling(self):
        q = normalize_entity_record("S1-1", "Apex Tech", "", "US")
        t = normalize_entity_record("S2-1", "Apex Tech LLC", "100 Broadway", "US")
        score = score_candidate_pair(q, t)
        self.assertGreater(score, 0.20)

    # 7. Empty address both sides
    def test_empty_address_both_sides(self):
        q = normalize_entity_record("S1-1", "Starlight Cafe", "", "India")
        t = normalize_entity_record("S2-1", "Starlight Cafe Pvt Ltd", "", "India")
        score = score_candidate_pair(q, t)
        self.assertGreaterEqual(score, 0.40)

    # 8. Duplicate names across targets
    def test_duplicate_names(self):
        index = BlockingIndex()
        t1 = normalize_entity_record("S2-1", "Subway", "101 Main St", "US")
        t2 = normalize_entity_record("S2-2", "Subway", "202 Elm St", "US")
        index.add_target_record(t1)
        index.add_target_record(t2)
        
        q = normalize_entity_record("S1-1", "Subway", "101 Main Street", "US")
        cands = index.retrieve_candidates_for_query(q, strategies=["exact_name"])
        self.assertIn("S2-1", cands)
        self.assertIn("S2-2", cands)

    # 9. Duplicate addresses
    def test_duplicate_addresses(self):
        index = BlockingIndex()
        t1 = normalize_entity_record("S2-1", "Acme Health", "100 Medical Center Drive", "US")
        t2 = normalize_entity_record("S2-2", "Zenith Optical", "100 Medical Center Drive", "US")
        index.add_target_record(t1)
        index.add_target_record(t2)
        
        q = normalize_entity_record("S1-1", "Acme Health Clinic", "100 Medical Center Dr", "US")
        cands = index.retrieve_candidates_for_query(q)
        self.assertIn("S2-1", cands)

    # 10. Token reordering
    def test_token_reordering(self):
        index = BlockingIndex()
        target = normalize_entity_record("S2-1", "Monroe Laitinen Center", "400 Pine Rd", "US")
        index.add_target_record(target, strategies=["sorted_name_tokens"])
        
        query = normalize_entity_record("S1-1", "Laitinen Monroe Center", "400 Pine Rd", "US")
        cands = index.retrieve_candidates_for_query(query, strategies=["sorted_name_tokens"])
        self.assertIn("S2-1", cands)

    # 11. Punctuation variation
    def test_punctuation_variation(self):
        n1 = normalize_business_name("A & B + C Logistics - Co.")
        n2 = normalize_business_name("A and B plus C Logistics Co")
        self.assertEqual(n1["compact"], n2["compact"])

    # 12. Legal suffix variation
    def test_legal_suffix_variation(self):
        n1 = normalize_business_name("Evergreen Solutions [L.L.C.]")
        n2 = normalize_business_name("Evergreen Solutions Private Limited")
        n3 = normalize_business_name("Evergreen Solutions SARL")
        self.assertEqual(n1["normalized"], "evergreen solutions")
        self.assertEqual(n2["normalized"], "evergreen solutions")
        self.assertEqual(n3["normalized"], "evergreen solutions")

    # 13. Transliteration / diacritics
    def test_transliteration_and_accents(self):
        self.assertEqual(strip_accents("Léarning Cénter"), "Learning Center")
        self.assertEqual(strip_accents("Boutique Cœur"), "Boutique Coeur")
        n = normalize_business_name("Café Étoile")
        self.assertEqual(n["normalized"], "cafe etoile")

    # 14. Unseen country values
    def test_unseen_country_open_set(self):
        # Open-set countries: e.g. France, Germany, Japan, Mars
        c_france = normalize_country("france")
        c_germany = normalize_country("de")
        c_any = normalize_country("Neverland")
        self.assertEqual(c_france, "FRANCE")
        self.assertEqual(c_germany, "DE")
        self.assertEqual(c_any, "NEVERLAND")
        
        # Verify index handles unseen country cleanly
        index = BlockingIndex()
        t = normalize_entity_record("S2-1", "Boutique Soleil", "12 Rue de Paris", "France")
        index.add_target_record(t)
        q = normalize_entity_record("S1-1", "Boutique Soleil SARL", "12 R. de Paris", "France")
        cands = index.retrieve_candidates_for_query(q)
        self.assertIn("S2-1", cands)

    # 15. Zero-candidate handling
    def test_zero_candidate_handling(self):
        index = BlockingIndex()
        t = normalize_entity_record("S2-1", "Hardware Store", "100 Main", "US")
        index.add_target_record(t)
        
        # Query with completely disjoint name and country
        q = normalize_entity_record("S1-999", "Unique Quantum Bakery", "999 Space Way", "Mars")
        cands = index.retrieve_candidates_for_query(q)
        self.assertEqual(len(cands), 0)

    # 16. Candidate recall calculation
    def test_candidate_recall_calculation(self):
        gt = {
            "S1-1": {"S2-1", "S3-1"},
            "S1-2": {"S2-2"},
            "S1-3": set()  # Singleton
        }
        cands = {
            "S1-1": {"S2-1", "S3-1", "S2-999"}, # Retrieved both
            "S1-2": {"S2-999"},                 # Missed S2-2
            "S1-3": set()
        }
        metrics = compute_detailed_candidate_metrics(gt, cands)
        # 2 out of 3 true links retrieved = 0.6667
        self.assertAlmostEqual(metrics["candidate_recall_overall"], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(metrics["candidate_recall_s2"], 0.5, places=4)
        self.assertAlmostEqual(metrics["candidate_recall_s3"], 1.0, places=4)

    # 17. Candidate count statistics
    def test_candidate_count_statistics(self):
        gt = {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}}
        cands = {
            "S1-1": {"S2-1", "S3-1", "S3-2"}, # 3 candidates
            "S1-2": {"S2-2"}                   # 1 candidate
        }
        metrics = compute_detailed_candidate_metrics(gt, cands, total_target_records=100)
        self.assertEqual(metrics["candidate_count_mean"], 2.0)
        self.assertEqual(metrics["candidate_count_median"], 2.0)
        self.assertEqual(metrics["candidate_count_min"], 1)
        self.assertEqual(metrics["candidate_count_max"], 3)
        self.assertEqual(metrics["total_final_candidate_pairs"], 4)

    # 18. Candidate_pairs schema
    def test_candidate_pairs_schema(self):
        candidates = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": set()
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv", encoding="utf-8") as f:
            temp_tsv = f.name
            
        try:
            CandidateGenerator.write_candidate_pairs_tsv(candidates, output_file_path=temp_tsv)
            with open(temp_tsv, "r", encoding="utf-8") as f:
                header = f.readline().rstrip("\r\n").split("\t")
                row1 = f.readline().rstrip("\r\n").split("\t")
                row2 = f.readline().rstrip("\r\n").split("\t")
                
            self.assertEqual(header, ["source1_entity_id", "candidate_entity_ids"])
            self.assertEqual(row1[0], "S1-1")
            self.assertEqual(row1[1], "S2-10,S3-20")
            self.assertEqual(row2[0], "S1-2")
            self.assertEqual(row2[1], "")
        finally:
            os.remove(temp_tsv)

    # 19. No duplicate candidate pairs in candidate_pairs.tsv
    def test_no_duplicate_candidate_pairs(self):
        candidates = {
            "S1-1": {"S2-1", "S2-1", "S3-1"}
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv", encoding="utf-8") as f:
            temp_tsv = f.name
        try:
            CandidateGenerator.write_candidate_pairs_tsv(candidates, output_file_path=temp_tsv)
            with open(temp_tsv, "r", encoding="utf-8") as f:
                f.readline() # header
                line = f.readline().rstrip("\r\n").split("\t")
                cand_list = line[1].split(",")
                self.assertEqual(len(cand_list), len(set(cand_list)))
        finally:
            os.remove(temp_tsv)

    # 20. Every final match must be representable in candidate_pairs
    def test_every_final_match_representable(self):
        # Even 11 matches (max observed in EDA) representable in candidate list
        target_ids = {f"S2-{i}" for i in range(6)} | {f"S3-{i}" for i in range(5)}
        candidates = {"S1-100": target_ids}
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv", encoding="utf-8") as f:
            temp_tsv = f.name
        try:
            CandidateGenerator.write_candidate_pairs_tsv(candidates, output_file_path=temp_tsv)
            with open(temp_tsv, "r", encoding="utf-8") as f:
                f.readline()
                row = f.readline().rstrip("\r\n").split("\t")
                reconstructed = set(row[1].split(","))
                self.assertEqual(reconstructed, target_ids)
        finally:
            os.remove(temp_tsv)

    # 21. Leetspeak normalization and blocking regression
    def test_leetspeak_normalization_and_blocking(self):
        index = BlockingIndex()
        t = normalize_entity_record("S2-1", "@r0yaldevelopers", "Kechery Road", "India")
        index.add_target_record(t, strategies=["name_leet_compact"])
        
        q = normalize_entity_record("S1-1", "Royal Developers Private Limited", "Kechery Road", "India")
        cands = index.retrieve_candidates_for_query(q, strategies=["name_leet_compact"])
        self.assertIn("S2-1", cands)

    # 22. DBA and honorific alias extraction regression
    def test_dba_alias_extraction_and_blocking(self):
        index = BlockingIndex()
        t = normalize_entity_record("S3-1", "Ectonyla dba Orane Brothers Limited", "Borivali West", "India")
        index.add_target_record(t, strategies=["name_dba_alias"])
        
        q = normalize_entity_record("S1-1", "Orane Brothers Limited", "Borivali West", "India")
        cands = index.retrieve_candidates_for_query(q, strategies=["name_dba_alias"])
        self.assertIn("S3-1", cands)

    # 23. Address number leading zero tolerance regression
    def test_address_number_leading_zero_tolerance(self):
        index = BlockingIndex()
        t = normalize_entity_record("S3-1", "Tova Inc", "00501 Elton Court, Smithtown", "US")
        index.add_target_record(t, strategies=["name_address_combo", "address_exact_combo"])
        
        q = normalize_entity_record("S1-1", "Tova Inc", "501 Elton Court, Smithtown", "US")
        cands = index.retrieve_candidates_for_query(q, strategies=["name_address_combo"])
        self.assertIn("S3-1", cands)

    # 24. Salient address locality blocking for cross-script entities
    def test_salient_address_locality_blocking(self):
        index = BlockingIndex()
        # Target has Hindi name and English address with plot/locality
        t = normalize_entity_record("S2-1", "गोल्ड एंटरप्राइजेज", "Office No.18, Greater Noida", "India")
        index.add_target_record(t, strategies=["address_salient_combo"])
        
        q = normalize_entity_record("S1-1", "Gold Enterprises Private Limited", "Office No. 18, Greater Noida", "India")
        cands = index.retrieve_candidates_for_query(q, strategies=["address_salient_combo"])
        self.assertIn("S2-1", cands)

if __name__ == "__main__":
    unittest.main()

