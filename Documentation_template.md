# Amazon ML Challenge 2026: Business Entity Resolution
## Final Methodology & Technical Documentation

**Team Name:** amazon_ml_submission  
**Challenge Track:** Business Entity Resolution  
**Date:** September 2026  
**Primary Evaluation Metric:** Macro $F_{0.5}$ (Anchor-level Source 1 Entity Resolution)

---

## 1. Executive Summary

This submission implements a high-precision, low-latency machine learning pipeline for large-scale multi-source business entity resolution across Source 1 (query anchors) and Target sources (Source 2 and Source 3). The pipeline strictly satisfies all Amazon ML Challenge 2026 official constraints:
- **No external data, APIs, or geocoding:** 100% self-contained within permitted datasets.
- **Candidate subsetting guarantee:** Every predicted match in `matching_results.tsv` is strictly a subset of `candidate_pairs.tsv`.
- **Model compliance:** Built exclusively with scikit-learn (`HistGradientBoostingClassifier`, <1M parameters, MIT/BSD open-source license, well below 8B limit).
- **Macro $F_{0.5}$ alignment:** Optimized specifically for precision-heavy ranking with an empirically tuned frozen decision threshold ($\theta = 0.70$).
- **Candidate Generator:** EXP-013 (Hardened 10-Strategy Multi-Block Union with Composite Refinement, $K=50$, search-space reduction $>99.97\%$).
- **Pairwise Feature Schema:** `phase4-v1` (87 deterministic, non-leaking features across 9 orthogonal groups).

---

## 2. Problem Formulation & End-to-End Pipeline

```
Raw Test Data (test_source1, test_source2, test_source3)
                       │
                       ▼
Deterministic Entity Normalization & View Generation
(Decomposition, Lowercase, Compact, Soundex, Leet, Suffixes)
                       │
                       ▼
Partitioning by Country (India, US, France, Open-Set)
                       │
                       ▼
EXP-013 Blocking & Inverted Index Retrieval (10 Strategies)
                       │
                       ▼
Composite Deterministic Refinement (Threshold = 0.15, Max K = 50)
                       │
                       ▼
candidate_pairs.tsv (Exact Model Input Set)
                       │
                       ▼
87-Dimensional Pairwise Feature Extraction (phase4-v1 Schema)
                       │
                       ▼
Frozen ML Model Inference (HistGradientBoostingClassifier)
                       │
                       ▼
Decision Threshold Filtering (P(Match) >= 0.70)
                       │
                       ▼
matching_results.tsv (Exactly 1 Row per Test S1 Anchor)
                       │
                       ▼
Automated Submission Validation (utils/validate_submission.py)
```

---

## 3. Data Preprocessing & Normalization

To handle real-world entity variations without external knowledge, records undergo deterministic normalization:
1. **Unicode Canonical Decomposition:** Converts accents, diacritics, and varied encodings into normalized ASCII equivalents.
2. **Case Folding & Punctuation Stripping:** Uniform lowercase representations.
3. **Legal Entity Suffix Standardization:** Cleans company corporate suffixes (e.g., `llc`, `inc`, `pvt ltd`, `sarl`, `corp`, `gmbh`) while retaining them in secondary views.
4. **Secondary Views:**
   - `compact`: Alphanumeric-only concatenated representation.
   - `leet_compact`: Normalized leetspeak substitutions (e.g., `@` -> `a`, `0` -> `o`, `3` -> `e`, `$` -> `s`).
   - `dba_alias`: Extracts "doing business as" (DBA) or "aka" trade names.
   - `address_compact`: Strips standard street designations and normalizes numeric house/unit numbers.

---

## 4. Candidate Generation (EXP-013)

To avoid evaluating a $1.73\text{M} \times 9.97\text{M}$ ($1.7 \times 10^{13}$) Cartesian search space, EXP-013 utilizes a 10-strategy multi-block inverted index with provenance tracking:
1. `exact_name`: Strict country + compact name match.
2. `sorted_name_tokens`: Country + alphabetically sorted name tokens (word-order invariant).
3. `first_two_name_tokens`: Country + first two significant name tokens.
4. `longest_name_token`: Country + longest salient name token ($\ge 5$ characters).
5. `name_prefix_6`: Country + first 6 characters of compact name (typo/suffix tolerant).
6. `name_address_combo`: Country + first name token + first address token.
7. `address_exact_combo`: Country + compact address.
8. `name_leet_compact`: Country + leet-transformed compact name.
9. `name_dba_alias`: Country + parsed DBA alias.
10. `address_salient_combo`: Country + street number + postal token.

### Candidate Refinement & Budget Constraint
- Each retrieved candidate is scored via a lightweight composite heuristic:
  $$\text{Score} = 0.35 \times \text{Jaccard}_{\text{ngrams}} + 0.30 \times \text{Jaccard}_{\text{tokens}} + 0.20 \times \text{Containment} + 0.10 \times \text{Jaccard}_{\text{addr}} + 0.05 \times \text{NumberMatch}$$
- Provenance Bypass: Exact/sorted/leet/DBA matches bypass heuristic filtering.
- Filter: Candidates scoring $< 0.15$ are pruned.
- Candidate Cap: Strictly top-$K=50$ candidates per query anchor.
- **Search-Space Reduction:** $99.9797\%$ reduction ratio ($4,929.9\times$ speedup).

---

## 5. Feature Engineering (phase4-v1 Schema)

For every candidate pair $(S_1, \text{Target})$, an 87-dimensional pairwise feature vector is extracted across 9 orthogonal groups:
1. **Name Similarity (15 features):** Exact match, compact match, token Jaccard, sorted token Jaccard, character 2/3/4-gram Jaccards, Levenshtein edit similarity, common prefix/suffix lengths, token containment, Monge-Elkan asymmetric similarity.
2. **Address Similarity (15 features):** Normalized exact, token Jaccard, 2/3-gram Jaccards, edit similarity, numeric overlap, building number match, postal code match.
3. **Country Consistency (4 features):** Exact country match, missing country indicators, open-set cross-country flags.
4. **Cross-Field Interactions (8 features):** Name $\times$ Address composite similarity, harmonic mean of name/address sim, high-name-low-address conflict flags.
5. **View Similarity (5 features):** Leetspeak match, DBA alias match, raw unnormalized match.
6. **Blocking Provenance (11 features):** Binary indicators for which of the 10 blocking strategies retrieved the candidate, plus total blocking strategy count.
7. **Refinement Heuristics (7 features):** Composite refinement score, token sim, ngram sim, address sim, provenance bypass indicator.
8. **Record Quality (16 features):** String lengths, token counts, non-Latin character indicators, digit counts for query and target.
9. **Frequency & Uniqueness (6 features):** Query/target name frequency, compact frequency, country-specific name frequency (fitted strictly on training records to prevent leakage).

---

## 6. Model Architecture & Training

- **Model:** `HistGradientBoostingClassifier` (scikit-learn)
- **Hyperparameters:**
  - `max_iter`: 100
  - `max_depth`: 8
  - `learning_rate`: 0.1
  - `min_samples_leaf`: 20
  - `random_state`: 42
  - `class_weight`: None
- **Training Data:** Labeled candidate pairs generated by EXP-013 on training Source 1 anchors against training Source 2/Source 3 targets, labeled using `train_ground_truth.tsv`.
- **Validation Macro $F_{0.5}$:** **0.9333** (Validation Macro Precision: 0.9646, Validation Macro Recall: 0.8757, PR-AUC: 0.9985).

---

## 7. Decision Threshold & Multi-Match Policy

Because the primary competition metric is Macro $F_{0.5}$, Precision is weighted twice as heavily as Recall ($\beta = 0.5$). Systematic threshold ablation across $[0.30, 0.90]$ proved that $\theta = 0.70$ achieves the optimal trade-off:
- Predictions are **NOT** constrained to top-1: a Source 1 entity can match 0, 1, 2, or multiple targets from Source 2 and/or Source 3.
- Predictions with probability $P(\text{Match} \mid S_1, T) \ge 0.70$ are retained; all others are discarded.
- Singletons (entities with 0 matches above threshold) output an empty match field.

---

## 8. Verification & Submission Compliance

The generated output files strictly conform to the official specification:
1. `candidate_pairs.tsv`: Exactly one row per test Source 1 entity containing all candidates passed to the model.
2. `matching_results.tsv`: Exactly one row per test Source 1 entity containing all predicted matches ($\ge 0.70$).
3. **Subsetting Guarantee:** 100% of predicted matches exist in `candidate_pairs.tsv`.
4. **Validator Execution:** Automated end-to-end verification via `utils/validate_submission.py` passed with 0 errors.
