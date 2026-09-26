# Versioned Feature Schema for Phase 4 Pairwise Matching
FEATURE_SCHEMA_VERSION = "phase4-v1"

# Explicitly ordered list of all feature column names
FEATURE_COLUMNS = [
    # Feature Group 1: Name Similarity (15 features)
    "name_normalized_exact",
    "name_compact_exact",
    "name_token_jaccard",
    "name_sorted_token_jaccard",
    "name_char_2gram_jaccard",
    "name_char_3gram_jaccard",
    "name_char_4gram_jaccard",
    "name_edit_similarity",
    "name_levenshtein_compact_sim",
    "name_compact_containment",
    "name_prefix_similarity",
    "name_suffix_similarity",
    "name_token_count_diff",
    "name_shared_token_count",
    "name_longest_token_match",

    # Feature Group 2: Address Similarity (15 features)
    "addr_normalized_exact",
    "addr_compact_exact",
    "addr_token_jaccard",
    "addr_char_3gram_jaccard",
    "addr_char_4gram_jaccard",
    "addr_edit_similarity",
    "addr_shared_numbers_count",
    "addr_exact_number_match",
    "addr_leading_zero_number_match",
    "addr_postal_code_match",
    "addr_locality_overlap",
    "addr_state_overlap",
    "addr_containment",
    "addr_token_count_diff",
    "addr_is_missing_either",

    # Feature Group 3: Cross-Field Features (8 features)
    "cross_strong_name_strong_addr",
    "cross_strong_name_weak_addr",
    "cross_weak_name_strong_addr",
    "cross_exact_number_similar_name",
    "cross_exact_locality_similar_name",
    "cross_same_country_name_sim",
    "cross_same_country_addr_sim",
    "cross_agreement_count",

    # Feature Group 4: Country Features (4 features)
    "country_exact_match",
    "country_normalized_match",
    "country_missing_either",
    "country_conflict",

    # Feature Group 5: Normalization-View Features (5 features)
    "view_leet_name_exact",
    "view_leet_name_sim",
    "view_dba_alias_match",
    "view_compact_name_match",
    "view_compact_addr_match",

    # Feature Group 6: Blocking Provenance Features (11 features)
    "prov_exact_name",
    "prov_sorted_tokens",
    "prov_first_two_tokens",
    "prov_longest_token",
    "prov_name_prefix",
    "prov_name_address",
    "prov_address_exact",
    "prov_name_leet",
    "prov_name_dba",
    "prov_address_salient",
    "prov_strategy_count",

    # Feature Group 7: Refinement Features (7 features)
    "refine_composite_score",
    "refine_ngram_sim",
    "refine_token_sim",
    "refine_containment",
    "refine_addr_token_sim",
    "refine_number_match",
    "refine_provenance_bypass",

    # Feature Group 8: Record Quality Features (16 features)
    "qual_q_name_len",
    "qual_t_name_len",
    "qual_q_addr_len",
    "qual_t_addr_len",
    "qual_q_name_token_count",
    "qual_t_name_token_count",
    "qual_q_addr_token_count",
    "qual_t_addr_token_count",
    "qual_addr_missing_q",
    "qual_addr_missing_t",
    "qual_non_latin_script_q",
    "qual_non_latin_script_t",
    "qual_mixed_script_pair",
    "qual_has_url_domain",
    "qual_has_legal_suffix",
    "qual_has_dba",

    # Feature Group 9: Entity Cardinality / Frequency Features (6 features)
    "freq_q_name",
    "freq_t_name",
    "freq_q_compact",
    "freq_t_compact",
    "freq_q_country_name",
    "freq_t_country_name"
]

TOTAL_FEATURE_COUNT = len(FEATURE_COLUMNS)  # Exactly 87 features
