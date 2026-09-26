# System Architecture

## 1. High-Level Architecture

```text
                    ┌─────────────────────┐
                    │   Training TSVs     │
                    │ S1 / S2 / S3 / GT   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Validation     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Normalization       │
                    │ Name + Address      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Candidate Generation│
                    │ / Broad Blocking    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Candidate Refinement│
                    │ (Pruning & Top-K)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          FINAL CANDIDATES        candidate_pairs.tsv
                    │
                    ▼
                    ┌─────────────────────┐
                    │ Pair Features       │
                    │ Name / Address /    │
                    │ Country / Tokens    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Matching Model      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Decision / Threshold│
                    │ Singleton Handling  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Post-processing     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Submission Builder  │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          matching_results.tsv    candidate_pairs.tsv
```

## 2. Major Components

### Data Layer

Responsibilities:

- read TSV files
- validate schema
- preserve string IDs
- handle large files
- parse ground truth
- expose efficient data access

### Normalization Layer

Responsibilities:

- Unicode normalization
- case normalization
- punctuation handling
- whitespace normalization
- legal suffix handling
- address abbreviation normalization
- URL/domain artifact handling
- tokenization

Normalization must preserve original values separately.

### Blocking Layer

Responsibilities:

- reduce search space (achieving 4,929.9x reduction ratio)
- generate plausible S1→S2/S3 candidates across 10 deterministic strategies:
  1. `exact_name` (country + exact compact normalized name)
  2. `sorted_name_tokens` (country + order-invariant token set)
  3. `first_two_name_tokens` (country + prefix token bigram)
  4. `longest_name_token` (country + salient token)
  5. `name_prefix_6` (country + 6-char compact prefix)
  6. `name_address_combo` (country + first name token + address number)
  7. `address_exact_combo` (country + address number + address token)
  8. `name_leet_compact` (country + leetspeak normalized compact name)
  9. `name_dba_alias` (country + trade-name / DBA alias)
  10. `address_salient_combo` (country + address number + salient locality token)
- candidate refinement engine filters pairs with composite heuristic scoring, provenance bypass, and top-50 ceiling (`K=50`)
- candidate recall: 88.32% overall, 72.48% full S1 coverage, mean 31.36 candidates per query

### Feature Layer (Phase 4 — `phase4-v1`)

For each candidate pair, produce an 87-dimensional structured numerical feature vector:

- **Name Similarity (15):** normalized/compact equality, token/sorted-token Jaccard, 2/3/4-grams, Levenshtein, prefix/suffix ratios, token counts
- **Address Similarity (15):** normalized/compact equality, token Jaccard, 3/4-grams, edit similarity, numbers overlap, leading-zero match, postal match, locality/state overlap, missingness
- **Cross-Field Interactions (8):** strong/weak combinations, number+name, locality+name, country interactions, agreement count
- **Open-Set Country (4):** exact, normalized, missing, conflict
- **Normalization Views (5):** leet exact/sim, DBA alias match, compact name/address matches
- **Blocking Provenance (11):** 10 strategy indicators + strategy count
- **Refinement Heuristics (7):** composite score, n-gram, token, containment, address, number match, provenance bypass
- **Record Quality (16):** lengths, token counts, missingness, non-Latin script, mixed script pair, URL, legal suffix, DBA
- **Frequency / Cardinality (6):** training-only fitted frequencies (100% leakage-free)

Total features: exactly 87. Guaranteed 0 NaN/Inf. Highly discriminative between positive matches and distractors.

### Matching Layer (Phase 5 — Champion: `MODEL-003`)

Input:

```text
S1 entity + candidate S2/S3 entity + 87 features
```

Model: `HistGradientBoostingClassifier` (LightGBM-equivalent, 100 iterations, max depth 8, learning rate 0.1).
Output:

```text
calibrated match probability P(target is true match | S1, target)
```

Achieves **0.9985 PR-AUC** and **98.31% model recall** within candidate pairs.

### Decision Layer

Responsibilities:

- evaluate match probability against optimal threshold **0.70**
- permit empty match lists for true singletons (achieving 0.9617 zero-match accuracy)
- avoid weak false merges (Macro Precision: 0.9646)
- optimize challenge-official Macro F0.5 (achieving **0.9333**)

### Submission Layer

Produces:

```text
matching_results.tsv
candidate_pairs.tsv
```

and validates them before submission.

## 3. Training Flow

```text
Training S1
   +
Training S2/S3
   +
Ground Truth
   ↓
Validation Split
   ↓
Candidate Generation
   ↓
Candidate Recall Measurement
   ↓
Positive/Negative Pair Construction
   ↓
Feature Extraction
   ↓
Model Training
   ↓
Threshold Calibration
   ↓
F_0.5 Evaluation
```

## 4. Inference Flow

```text
Test S1
   +
Test S2/S3
   ↓
Normalization
   ↓
Blocking
   ↓
Candidate Pairs
   ↓
Feature Extraction
   ↓
Model Scoring
   ↓
Thresholding
   ↓
Final Matches
   ↓
Submission Validation
```

## 5. Important Architectural Principle

Blocking and matching are separate optimization problems.

Blocking optimizes:

> "Did we keep the true match?"

Matching optimizes:

> "Given plausible candidates, which ones are actually matches?"

Do not use a matching model to compensate for bad blocking.

## 6. Scalability Principle

Never construct all:

```text
S1 × S2
S1 × S3
```

pairs.

The architecture must reduce the search space before expensive feature computation.

## 7. Phase 6 Production Architecture

```text
                  TEST SOURCE 1 (1.73M)
                  TEST SOURCE 2 & 3 (9.97M)
                             │
                             ▼
              Country Partition Dispatcher
              (France, US, India, Open-Set)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   BlockingIndex (EXP-013)             CandidateRefiner
   10 Inverted Key Indexes             Composite Heuristic
   (Country-Scoped, Max 500)           (Threshold 0.15, Max K=50)
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
                    FINAL CANDIDATES
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
       candidate_pairs.tsv    PairwiseFeatureExtractor
      (Exact Model Input)     (phase4-v1, 87 Features)
                                         │
                                         ▼
                             HistGradientBoostingClassifier
                             (Frozen Production Model)
                                         │
                                         ▼
                                Decision Threshold
                               (P(Match) >= 0.70)
                                         │
                                         ▼
                               matching_results.tsv
                                         │
                                         ▼
                              utils/validate_submission.py
                              (15 Official Challenge Rules)
                                         │
                                         ▼
                              <team_name>_submission.zip
```
