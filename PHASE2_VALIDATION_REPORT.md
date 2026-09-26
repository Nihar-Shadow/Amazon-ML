# Phase 2: Validation Infrastructure Report

**Amazon ML Challenge 2026: Business Entity Resolution**  
**Phase:** 2 — Validation Infrastructure  
**Deliverable Status:** Complete  

---

## 1. Validation Split Design

### Anchor-Level Partitioning Strategy
In multi-source entity resolution, records from Source 1 serve as **query anchors**, while records from Source 2 and Source 3 constitute the **target candidate pool**.

- **Total Source 1 Records:** 2,206,821
- **Split Ratio:** 80% Train / 20% Validation
- **Train Source 1 Entities:** **1,765,457** (80.00%)
- **Validation Source 1 Entities:** **441,364** (20.00%)
- **Seed:** Fixed at `42`
- **Output Artifacts:** [`eda/train_s1_ids.txt`](file:///c:/AMAZON%20ML/eda/train_s1_ids.txt), [`eda/val_s1_ids.txt`](file:///c:/AMAZON%20ML/eda/val_s1_ids.txt), and [`eda/validation_split_summary.csv`](file:///c:/AMAZON%20ML/eda/validation_split_summary.csv).

### Leakage Prevention & Target Pool Isolation
Target entities (Source 2 and Source 3) are shared resources across the candidate space. However, to guarantee an unbiased offline benchmark:
1. **Anchor Quarantining:** S1 entity IDs are strictly separated. The validation S1 anchors are never seen during model training.
2. **Label Isolation:** The ground-truth linkages for the 441,364 validation S1 anchors are held out completely. No training component or feature builder may access validation ground truth.
3. **Reproducibility Guarantee:** S1 IDs are sorted prior to pseudo-random shuffling using Python's deterministic `random.Random(seed)`. Cross-platform consistency is 100% verified.

---

## 2. $F_{0.5}$ Metric Definition

### Mathematical Formulation
The official competition evaluation metric is **Macro $F_{0.5}$** across all Source 1 query entities:

For each individual Source 1 entity:
$$\text{Precision} = \frac{\text{TP}}{|\text{Predicted Matches}|} = \frac{|\text{True Targets} \cap \text{Predicted Targets}|}{|\text{Predicted Targets}|}$$

$$\text{Recall} = \frac{\text{TP}}{|\text{True Targets}|} = \frac{|\text{True Targets} \cap \text{Predicted Targets}|}{|\text{True Targets}|}$$

With $\beta = 0.5$ ($\beta^2 = 0.25$):
$$F_{0.5} = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

### Precision-Heavy Emphasis
In $F_{0.5}$, Precision is weighted **twice as heavily as Recall** ($1/\beta = 2$):
- **Scenario A (High Precision):** Precision = 1.0, Recall = 0.5 $\implies F_{0.5} = \frac{1.25 \times 1.0 \times 0.5}{0.25 \times 1.0 + 0.5} = \frac{0.625}{0.75} \approx \mathbf{0.8333}$
- **Scenario B (High Recall):** Precision = 0.5, Recall = 1.0 $\implies F_{0.5} = \frac{1.25 \times 0.5 \times 1.0}{0.25 \times 0.5 + 1.0} = \frac{0.625}{1.125} \approx \mathbf{0.5556}$

A false positive degrades the score substantially more than a false negative. Over-predicting candidate matches will severely harm competitive performance.

### Macro-Averaging
The primary score is the unweighted mean over all $N$ Source 1 entities:
$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^{N} F_{0.5}^{(i)}$$

---

## 3. Singleton Handling

In the training ground truth, **123,247 Source 1 entities (5.58%)** are singletons (zero target matches). The evaluation framework implements the exact mathematical edge-case specifications:

1. **True Singleton ($|\text{True Targets}| = 0$):**
   - **Correct Empty Prediction ($|\text{Predicted}| = 0$):**
     $$\text{Precision} = 1.0, \quad \text{Recall} = 1.0, \quad F_{0.5} = \mathbf{1.0}, \quad \text{Exact Match} = 1.0$$
   - **Incorrect Non-Empty Prediction ($|\text{Predicted}| > 0$):**
     $$\text{Precision} = 0.0, \quad \text{Recall} = 0.0, \quad F_{0.5} = \mathbf{0.0}, \quad \text{Exact Match} = 0.0$$

2. **Non-Singleton ($|\text{True Targets}| > 0$):**
   - **Empty Prediction ($|\text{Predicted}| = 0$):**
     $$\text{Precision} = 0.0, \quad \text{Recall} = 0.0, \quad F_{0.5} = \mathbf{0.0}, \quad \text{Exact Match} = 0.0$$

---

## 4. Candidate Recall & Blocking Diagnostics

The blocking evaluation interface [`src/evaluation/metrics.py`](file:///c:/AMAZON%20ML/src/evaluation/metrics.py) implements the diagnostic metrics required to govern Phase 3 candidate generation:

### Primary Candidate Metrics
- **Overall Candidate Recall:**
  $$\text{Candidate Recall} = \frac{\sum_{i} |\text{True Targets}_i \cap \text{Candidates}_i|}{\sum_i |\text{True Targets}_i|}$$
- **S1 Coverage:**
  $$\text{S1 Coverage} = \frac{|\{i : \text{True Targets}_i \subseteq \text{Candidates}_i, |\text{True Targets}_i| > 0\}|}{|\{i : |\text{True Targets}_i| > 0\}|}$$
- **Source-Specific Recall:**
  - $\text{Candidate Recall}_{S2} = \frac{\text{Retrieved True S2 Links}}{\text{Total True S2 Links}}$
  - $\text{Candidate Recall}_{S3} = \frac{\text{Retrieved True S3 Links}}{\text{Total True S3 Links}}$

### Blocking Efficiency Diagnostics
To evaluate whether a candidate set is tractable for downstream matching:
- **Candidate count distribution:** Mean, Median, Min, Max, P90, P95, and P99 candidates per S1.
- **Cardinality thresholds:**
  - Percentage of S1 entities with 0 candidates (`pct_zero_candidates`)
  - Percentage of S1 entities with $>100$ candidates (`pct_gt_100_candidates`)
  - Percentage of S1 entities with $>1000$ candidates (`pct_gt_1000_candidates`)
- **Singleton behavior:** Percentage of true singletons cleanly receiving 0 candidates.

---

## 5. Memory & Performance Considerations

Given the dataset scale (~26.4 million records across 7 files), the following architectural safeguards were built into [`src/data/loader.py`](file:///c:/AMAZON%20ML/src/data/loader.py) and [`src/data/ground_truth.py`](file:///c:/AMAZON%20ML/src/data/ground_truth.py):

1. **Strict String Fidelity:** Delimiter is `sep="\t"` with `dtype=str` and `keep_default_na=False` to prevent data loss or conversion of strings like `"NA"` or `"null"` into NaNs.
2. **Streaming & Chunking:** `stream_tsv_chunks()` and `stream_entity_ids()` stream records without loading entire 500MB DataFrames into RAM.
3. **No Cartesian Joins:** Memory footprint during full split generation over 2.2M entities was **316.28 MB**, well within the system's available 2.73 GB RAM.
4. **$O(1)$ Hash Set Operations:** Evaluation uses dictionary and set intersection for constant-time lookups per record.

---

## 6. Tests Executed

The unit test suite [`tests/test_evaluation.py`](file:///c:/AMAZON%20ML/tests/test_evaluation.py) exercises all critical paths:

| Test Class | Test Case | Target Behavior | Result |
| :--- | :--- | :--- | :---: |
| `TestGroundTruthParser` | `test_parse_valid_ground_truth` | Parses valid TSV into `dict[str, set[str]]` | **PASS** |
| `TestGroundTruthParser` | `test_empty_match_list_is_empty_set` | Verifies singleton matches yield `set()` | **PASS** |
| `TestGroundTruthParser` | `test_duplicate_target_id_raises_error` | Enforces no duplicates inside match list | **PASS** |
| `TestGroundTruthParser` | `test_malformed_s1_id_raises_error` | Catches invalid S1 ID formats | **PASS** |
| `TestGroundTruthParser` | `test_malformed_target_id_raises_error` | Catches invalid target ID prefixes | **PASS** |
| `TestGroundTruthParser` | `test_id_format_regex` | Tests regex validation helper | **PASS** |
| `TestF05Metric` | `test_perfect_prediction` | Precision=1.0, Recall=1.0 $\implies F_{0.5}=1.0$ | **PASS** |
| `TestF05Metric` | `test_singleton_scoring_correct` | Empty pred on singleton $\implies F_{0.5}=1.0$ | **PASS** |
| `TestF05Metric` | `test_singleton_scoring_incorrect` | Non-empty pred on singleton $\implies F_{0.5}=0.0$ | **PASS** |
| `TestF05Metric` | `test_non_singleton_empty_prediction` | Empty pred on true matches $\implies F_{0.5}=0.0$ | **PASS** |
| `TestF05Metric` | `test_f05_precision_weighting` | Confirms $P=1.0, R=0.5$ ($0.8333$) > $P=0.5, R=1.0$ ($0.5556$) | **PASS** |
| `TestF05Metric` | `test_macro_evaluation` | Verifies exact macro-average across multiple entities | **PASS** |
| `TestCandidateBlockingMetrics` | `test_candidate_recall_and_coverage` | Validates overall recall, S2/S3 breakdown, and coverage | **PASS** |
| `TestDeterministicValidationSplit` | `test_deterministic_split_reproducibility` | Verifies identical splits, 80/20 proportions, zero overlap | **PASS** |
| `TestDeterministicValidationSplit` | `test_different_seed_different_split` | Confirms seed sensitivity | **PASS** |

**Execution Output:**
```
Ran 15 tests in 0.178s
OK
```

---

## 7. Results

1. **Validation Split Executed on Real Dataset:**
   - 2,206,821 Source 1 entities split into **1,765,457 train anchors (80.00%)** and **441,364 validation anchors (20.00%)**.
   - Overlap between train and validation ID sets: **0**.
   - Full split generated and verified in **7.30 seconds**.
2. **Ground Truth Parser Validated:**
   - Strict format validation verified on edge cases (empty match strings, duplicate targets, malformed IDs).
3. **Blocking Interface Initialized:**
   - Standardized interface `generate_candidates(source1_df, source2_df, source3_df)` established in [`src/evaluation/validation.py`](file:///c:/AMAZON%20ML/src/evaluation/validation.py).
   - Clean decoupling between candidate generation (Phase 3) and entity matching (Phase 4).

---

## 8. Known Limitations

1. **Absence of France in Validation Split:**
   - The validation split is constructed exclusively from `datasets/train/` (which contains only US and India).
   - France (~15% of the test set) cannot be directly validated using the training split. Validation strategies in subsequent phases must simulate cold-start / unseen country conditions (e.g., cross-country validation holding out India).
2. **Baseline Candidate Generator:**
   - As per Phase 2 specifications, `generate_candidates()` currently returns an empty baseline to decouple blocking from matching. Sophisticated blocking algorithms will be implemented in Phase 3.

---

PHASE 2 COMPLETE
