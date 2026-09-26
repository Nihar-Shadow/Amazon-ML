# Technical Specification

## 1. Runtime

Recommended environment:

- Python 3.x
- pandas or equivalent TSV processing library
- NumPy
- SciPy where useful
- scikit-learn where useful
- rapidfuzz where useful
- pytest for tests

Pin versions in `requirements.txt`.

## 2. TSV Reading

Always explicitly specify:

```python
pd.read_csv(path, sep="\t")
```

Do not rely on automatic delimiter detection.

All IDs and textual fields must remain strings.

## 3. Canonical Data Types

Source record:

```python
{
    "entity_id": str,
    "business_name": str,
    "business_address": str,
    "country": str
}
```

Ground truth:

```python
{
    "source1_entity_id": str,
    "matched_entity_ids": set[str]
}
```

## 4. Normalized Fields

Never overwrite original fields.

Recommended representation:

```text
business_name
business_name_normalized
business_address
business_address_normalized
country
```

Potential normalization steps:

### Name

1. Unicode normalization (NFKD, accent stripping, ligature decomposition `œ` -> `oe`, `æ` -> `ae`)
2. case folding
3. whitespace normalization
4. punctuation normalization
5. common legal suffix normalization (US, India, France)
6. URL/domain artifact normalization
7. token extraction and order-invariant token sorting
8. leetspeak normalization view (`@` -> `a`, `0` -> `o`, `8` -> `b`, `1` -> `l`)
9. DBA and honorific alias extraction (`dba`, `t/a`, `trading as`, `m/s`, `shri`)

### Address

1. Unicode normalization
2. case folding
3. punctuation normalization
4. whitespace normalization
5. common street/unit abbreviation normalization
6. numeric token extraction with leading-zero tolerance
7. salient locality/city token extraction (excluding generic street terms)

### Active Blocking Strategies (EXP-013 Standard)
1. `exact_name`: Country + exact compact name
2. `sorted_name_tokens`: Country + sorted unique tokens
3. `first_two_name_tokens`: Country + first 2 significant name tokens
4. `longest_name_token`: Country + longest name token (min length 4)
5. `name_prefix_6`: Country + first 6 characters of compact name
6. `name_address_combo`: Country + first name token + first address number
7. `address_exact_combo`: Country + primary address number + first address token
8. `name_leet_compact`: Country + leetspeak-normalized compact name
9. `name_dba_alias`: Country + extracted DBA / trade name alias
10. `address_salient_combo`: Country + primary address number + salient address token

Normalization must be conservative. Do not delete information merely because it is unfamiliar.

## 5. Country

Country is a string label.

Do not hard-code the allowed set to US/India.

The test set contains France.

Country equality can be used as a candidate restriction only after validation supports it. The training EDA found no cross-country matches, but this is an observed training property rather than a statement that hidden labels are guaranteed by the problem statement.

## 6. Candidate Representation & Contract

Preferred in-memory logical structure:

```python
dict[str, set[str]]
```

where:

```text
key = S1 entity ID
value = candidate S2/S3 IDs
```

### `candidate_pairs.tsv` Specification (Official Contract)
`candidate_pairs.tsv` MUST contain the EXACT FINAL CANDIDATE SET passed to the ML model:
- Output path: `output/candidate_pairs.tsv`
- Schema:
  ```text
  source1_entity_id\tcandidate_entity_ids
  ```
- Separator: `\t` (tab)
- Multiplicity: Exactly one line per test Source 1 entity.
- Values: Comma-separated list of candidate S2/S3 entity IDs. Empty string if no candidates.
- Guarantee: No duplicate candidate pairs, no early unrefined pairs, strictly test S2/S3 IDs.
- Subsetting Rule: Every final match in `matching_results.tsv` MUST be a subset of `candidate_pairs.tsv`.

## 7. Candidate Recall

For each true link:

```text
true_target ∈ generated_candidates[source1]
```

Candidate recall:

```text
# true links retained
---------------------
# true links total
```

Also calculate S1-level full coverage:

```text
S1 is fully covered if every true target is in its candidate set.
```

## 8. F_0.5

Per entity:

```python
precision = TP / predicted
recall = TP / actual
```

For empty cases, follow the challenge definition carefully.

```text
F0.5 = 1.25PR / (0.25P + R)
```

Macro average over S1 entities.

## 9. Training Labels

For candidate pair `(S1, target)`:

```text
label = 1 if target appears in S1 ground truth
label = 0 otherwise
```

Negative sampling must be designed carefully.

Easy negatives alone are insufficient.

Hard negatives should eventually include candidates with:

- similar names
- similar addresses
- same country
- same normalized tokens
- same/common business names

## 10. Feature Schema (`phase4-v1`, 87 Features)

The pairwise matching feature extractor produces exactly 87 deterministic numerical features across 9 distinct feature groups:

1. **Name Similarity (15 features):** `name_normalized_exact`, `name_compact_exact`, `name_token_jaccard`, `name_sorted_token_jaccard`, `name_char_2gram_jaccard`, `name_char_3gram_jaccard`, `name_char_4gram_jaccard`, `name_edit_similarity`, `name_levenshtein_compact_sim`, `name_compact_containment`, `name_prefix_similarity`, `name_suffix_similarity`, `name_token_count_diff`, `name_shared_token_count`, `name_longest_token_match`.
2. **Address Similarity (15 features):** `addr_normalized_exact`, `addr_compact_exact`, `addr_token_jaccard`, `addr_char_3gram_jaccard`, `addr_char_4gram_jaccard`, `addr_edit_similarity`, `addr_shared_numbers_count`, `addr_exact_number_match`, `addr_leading_zero_number_match`, `addr_postal_code_match`, `addr_locality_overlap`, `addr_state_overlap`, `addr_containment`, `addr_token_count_diff`, `addr_is_missing_either`.
3. **Cross-Field Interaction (8 features):** `cross_strong_name_strong_addr`, `cross_strong_name_weak_addr`, `cross_weak_name_strong_addr`, `cross_exact_number_similar_name`, `cross_exact_locality_similar_name`, `cross_same_country_name_sim`, `cross_same_country_addr_sim`, `cross_agreement_count`.
4. **Country (4 features):** `country_exact_match`, `country_normalized_match`, `country_missing_either`, `country_conflict`. (Open-set support).
5. **Normalization Views (5 features):** `view_leet_name_exact`, `view_leet_name_sim`, `view_dba_alias_match`, `view_compact_name_match`, `view_compact_addr_match`.
6. **Blocking Provenance (11 features):** `prov_exact_name`, `prov_sorted_tokens`, `prov_first_two_tokens`, `prov_longest_token`, `prov_name_prefix`, `prov_name_address`, `prov_address_exact`, `prov_name_leet`, `prov_name_dba`, `prov_address_salient`, `prov_strategy_count`.
7. **Refinement Heuristics (7 features):** `refine_composite_score`, `refine_ngram_sim`, `refine_token_sim`, `refine_containment`, `refine_addr_token_sim`, `refine_number_match`, `refine_provenance_bypass`.
8. **Record Quality & Artifacts (16 features):** `qual_q_name_len`, `qual_t_name_len`, `qual_q_addr_len`, `qual_t_addr_len`, `qual_q_name_token_count`, `qual_t_name_token_count`, `qual_q_addr_token_count`, `qual_t_addr_token_count`, `qual_addr_missing_q`, `qual_addr_missing_t`, `qual_non_latin_script_q`, `qual_non_latin_script_t`, `qual_mixed_script_pair`, `qual_has_url_domain`, `qual_has_legal_suffix`, `qual_has_dba`.
9. **Entity Cardinality / Frequency (6 features, Strictly Leakage-Safe):** `freq_q_name`, `freq_t_name`, `freq_q_compact`, `freq_t_compact`, `freq_q_country_name`, `freq_t_country_name`. Fitted only on training entities. Zero validation leakage.

## 11. Model Strategy & Selected Champion

Empirically validated model families:
1. `MODEL-001`: Logistic Regression (Macro F0.5 = 0.9328 at threshold 0.85)
2. `MODEL-002`: Random Forest (Macro F0.5 = 0.9310 at threshold 0.60)
3. `MODEL-003`: HistGradientBoosting (Macro F0.5 = **0.9333** at threshold **0.70**) [CHAMPION]
4. `MODEL-004`: ExtraTrees Classifier (Macro F0.5 = 0.9310 at threshold 0.55)

**Selected Production Model:** `MODEL-003` (`HistGradientBoostingClassifier`, LightGBM-equivalent)
- `max_iter`: 100
- `max_depth`: 8
- `learning_rate`: 0.1
- `min_samples_leaf`: 20
- Model discrimination recall within candidates: **98.31%**
- Total inference runtime: ~0.20s per 156k pairs

## 12. Threshold Strategy

A comprehensive threshold sweep across [0.05, 0.95] established:
- Default 0.50 threshold yields Macro F0.5 = 0.9315
- **Optimal Threshold:** **0.70**
- Optimal Macro F0.5: **0.9333**
- Macro Precision: **0.9646**
- Macro Recall: **0.8757**
- Pairwise Precision: **0.9869**
- Pairwise Recall: **0.9831**
- Zero-match Accuracy: **0.9617**

## 13. Memory Strategy

For millions of rows:

- use chunks where necessary
- avoid repeated DataFrame copies
- use compact dtypes
- index normalized keys
- process candidate blocks incrementally
- persist intermediate data only when justified

## 14. Submission Validation

Before submission:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir datasets/test
```

A valid submission must satisfy all 15 official challenge integrity checks. The validator operates in streaming mode with $O(1)$ memory.

## 15. Production Inference & Submission Assembly Pipeline

Entry point: `scripts/run_final_submission.py`

Command:
```bash
python scripts/run_final_submission.py \
  --test-dir datasets/test \
  --model-path models/production_matching_model.pkl \
  --freq-path models/production_frequency_store.pkl \
  --output-dir output \
  --team-name amazon_ml_submission
```

Key characteristics:
- **Memory Envelope:** Processed by country partitions (`France`, `US`, `India`) to bound working memory < 800 MB.
- **Model Freezing:** Strictly uses frozen `HistGradientBoostingClassifier` at threshold $\theta = 0.70$.
- **Subsetting Rule:** Enforces `predicted_match ∈ final_candidate_set`.
- **Deterministic Ordering:** Lexicographical sort on `source1_entity_id`.
- **Packaging:** Assembles `<team_name>_submission.zip` matching required tree structure with zero bloat.
