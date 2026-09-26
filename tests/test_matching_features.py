import unittest
import numpy as np
import pandas as pd
from src.candidate_generation.normalizer import normalize_entity_record
from src.matching_features.feature_schema import FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS, TOTAL_FEATURE_COUNT
from src.matching_features.feature_extractor import PairwiseFeatureExtractor
from src.matching_features.frequency_features import FrequencyFeatureStore

class TestMatchingFeatures(unittest.TestCase):
    def setUp(self):
        self.extractor = PairwiseFeatureExtractor()

    # 1. Exact name match
    def test_exact_name_match(self):
        q = normalize_entity_record("S1-1", "Alpha Logistics Inc", "100 Main St, Austin, TX", "US")
        t = normalize_entity_record("S2-1", "Alpha Logistics Inc", "100 Main St, Austin, TX", "US")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["name_normalized_exact"], 1.0)
        self.assertEqual(feats["name_compact_exact"], 1.0)
        self.assertEqual(feats["name_token_jaccard"], 1.0)
        self.assertEqual(feats["name_edit_similarity"], 1.0)

    # 2. Typo name
    def test_typo_name(self):
        q = normalize_entity_record("S1-1", "Alpha Logistics", "100 Main St", "US")
        t = normalize_entity_record("S2-1", "Alppha Logstics", "100 Main St", "US")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["name_normalized_exact"], 0.0)
        self.assertGreater(feats["name_edit_similarity"], 0.7)
        self.assertGreaterEqual(feats["name_char_3gram_jaccard"], 0.5)

    # 3. Token reorder
    def test_token_reorder(self):
        q = normalize_entity_record("S1-1", "Blue River Health", "100 Main St", "US")
        t = normalize_entity_record("S2-1", "Health Blue River", "100 Main St", "US")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["name_token_jaccard"], 1.0)
        self.assertEqual(feats["name_sorted_token_jaccard"], 1.0)

    # 4. Legal suffix difference
    def test_legal_suffix_difference(self):
        q = normalize_entity_record("S1-1", "Apex Global Private Limited", "100 Main St", "India")
        t = normalize_entity_record("S2-1", "Apex Global LLC", "100 Main St", "India")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["name_normalized_exact"], 1.0)
        self.assertEqual(feats["name_compact_exact"], 1.0)

    # 5. DBA alias
    def test_dba_alias(self):
        q = normalize_entity_record("S1-1", "Orane Brothers", "Borivali, Mumbai", "India")
        t = normalize_entity_record("S3-1", "Ectonyla dba Orane Brothers", "Borivali, Mumbai", "India")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["view_dba_alias_match"], 1.0)

    # 6. Leetspeak
    def test_leetspeak(self):
        q = normalize_entity_record("S1-1", "Royal Developers", "Thrissur", "India")
        t = normalize_entity_record("S2-1", "@r0yaldevelopers", "Thrissur", "India")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["view_leet_name_exact"], 1.0)

    # 7. Exact address
    def test_exact_address(self):
        q = normalize_entity_record("S1-1", "Acme Corp", "501 Elton Court, Smithtown, NY", "US")
        t = normalize_entity_record("S2-1", "Acme Corp", "501 Elton Court, Smithtown, NY", "US")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["addr_normalized_exact"], 1.0)
        self.assertEqual(feats["addr_token_jaccard"], 1.0)
        self.assertEqual(feats["addr_exact_number_match"], 1.0)

    # 8. Address abbreviation
    def test_address_abbreviation(self):
        q = normalize_entity_record("S1-1", "Acme Corp", "501 Elton Street, Suite 4", "US")
        t = normalize_entity_record("S2-1", "Acme Corp", "501 Elton St, Ste 4", "US")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["addr_normalized_exact"], 1.0)
        self.assertEqual(feats["addr_token_jaccard"], 1.0)

    # 9. Leading-zero number
    def test_leading_zero_number(self):
        q = normalize_entity_record("S1-1", "Acme Corp", "501 Elton Court", "US")
        t = normalize_entity_record("S2-1", "Acme Corp", "00501 Elton Court", "US")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["addr_leading_zero_number_match"], 1.0)

    # 10. Missing address
    def test_missing_address(self):
        q = normalize_entity_record("S1-1", "Acme Corp", "501 Elton Court", "US")
        t = normalize_entity_record("S2-1", "Acme Corp", "", "US")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["addr_is_missing_either"], 1.0)
        self.assertEqual(feats["addr_normalized_exact"], 0.0)
        self.assertEqual(feats["addr_token_jaccard"], 0.0)

    # 11. Cross-script
    def test_cross_script(self):
        q = normalize_entity_record("S1-1", "Gold Enterprises", "Office 18, Greater Noida", "India")
        t = normalize_entity_record("S2-1", "गोल्ड एंटरप्राइजेज", "Office 18, Greater Noida", "India")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["qual_mixed_script_pair"], 1.0)
        self.assertEqual(feats["addr_exact_number_match"], 1.0)

    # 12. Country agreement
    def test_country_agreement(self):
        q = normalize_entity_record("S1-1", "Alpha", "100 Main", "France")
        t = normalize_entity_record("S2-1", "Alpha", "100 Main", "France")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["country_exact_match"], 1.0)
        self.assertEqual(feats["country_conflict"], 0.0)

    # 13. Country conflict
    def test_country_conflict(self):
        q = normalize_entity_record("S1-1", "Alpha", "100 Main", "US")
        t = normalize_entity_record("S2-1", "Alpha", "100 Main", "India")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(feats["country_exact_match"], 0.0)
        self.assertEqual(feats["country_conflict"], 1.0)

    # 14. Provenance features
    def test_provenance_features(self):
        q = normalize_entity_record("S1-1", "Alpha", "100 Main", "US")
        t = normalize_entity_record("S2-1", "Alpha", "100 Main", "US")
        strats = {"exact_name", "name_address_combo", "address_exact_combo"}
        feats = self.extractor.extract_pair_features(q, t, provenance_strategies=strats)
        self.assertEqual(feats["prov_exact_name"], 1.0)
        self.assertEqual(feats["prov_name_address"], 1.0)
        self.assertEqual(feats["prov_address_exact"], 1.0)
        self.assertEqual(feats["prov_name_leet"], 0.0)
        self.assertEqual(feats["prov_strategy_count"], 3.0)

    # 15. Refinement features
    def test_refinement_features(self):
        q = normalize_entity_record("S1-1", "Alpha Logistics", "100 Main St", "US")
        t = normalize_entity_record("S2-1", "Alpha Logistics", "100 Main St", "US")
        strats = {"exact_name"}
        feats = self.extractor.extract_pair_features(q, t, provenance_strategies=strats)
        self.assertEqual(feats["refine_provenance_bypass"], 1.0)
        self.assertEqual(feats["refine_composite_score"], 1.0)

    # 16. Frequency leakage protection
    def test_frequency_leakage_protection(self):
        store = FrequencyFeatureStore()
        train_rec = normalize_entity_record("S1-1", "Apex Corp", "100 Main", "US")
        store.fit([train_rec])
        
        # Validation query entity not in train
        val_q = normalize_entity_record("S1-99", "Unique New Corp", "100 Main", "US")
        val_t = normalize_entity_record("S2-99", "Apex Corp", "100 Main", "US")
        
        extractor = PairwiseFeatureExtractor(frequency_store=store)
        feats = extractor.extract_pair_features(val_q, val_t)
        self.assertEqual(feats["freq_q_name"], 0.0)
        self.assertEqual(feats["freq_t_name"], 1.0)

    # 17. Validation isolation
    def test_validation_isolation(self):
        # Ensure fitting on train does not change when validation is scored
        store = FrequencyFeatureStore()
        train_records = [normalize_entity_record(f"S1-{i}", "Common Corp", "100 Main", "US") for i in range(5)]
        store.fit(train_records)
        
        count_before = store.name_counts["common"]
        self.assertEqual(count_before, 5)
        
        # Extract features for validation pairs
        val_q = normalize_entity_record("S1-VAL", "Common Corp", "100 Main", "US")
        val_t = normalize_entity_record("S2-VAL", "Common Corp", "100 Main", "US")
        extractor = PairwiseFeatureExtractor(frequency_store=store)
        _ = extractor.extract_pair_features(val_q, val_t)
        
        count_after = store.name_counts["common"]
        self.assertEqual(count_before, count_after)

    # 18. Feature schema integrity
    def test_feature_schema(self):
        self.assertEqual(len(FEATURE_COLUMNS), TOTAL_FEATURE_COUNT)
        self.assertEqual(TOTAL_FEATURE_COUNT, 87)
        q = normalize_entity_record("S1-1", "Alpha", "100 Main", "US")
        t = normalize_entity_record("S2-1", "Beta", "200 Main", "US")
        feats = self.extractor.extract_pair_features(q, t)
        self.assertEqual(len(feats), TOTAL_FEATURE_COUNT)
        self.assertEqual(set(feats.keys()), set(FEATURE_COLUMNS))

    # 19. NaN/Inf protection
    def test_nan_inf_protection(self):
        q = normalize_entity_record("S1-1", "", "", "")
        t = normalize_entity_record("S2-1", "", "", "")
        feats = self.extractor.extract_pair_features(q, t)
        for k, v in feats.items():
            self.assertFalse(np.isnan(v), f"Feature {k} is NaN")
            self.assertFalse(np.isinf(v), f"Feature {k} is Inf")

    # 20. Deterministic feature generation
    def test_deterministic_generation(self):
        q = normalize_entity_record("S1-1", "Starlight Systems", "350 5th Ave, New York", "US")
        t = normalize_entity_record("S2-1", "Starlight Systems Inc", "350 Fifth Avenue, NYC", "US")
        f1 = self.extractor.extract_pair_features(q, t)
        f2 = self.extractor.extract_pair_features(q, t)
        self.assertEqual(f1, f2)

if __name__ == "__main__":
    unittest.main()
