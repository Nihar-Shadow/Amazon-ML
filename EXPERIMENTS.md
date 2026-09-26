# Experiment Log

## Purpose

This file records all meaningful experiments.

Never overwrite old results.

## Experiment ID Format

```text
EXP-001
EXP-002
EXP-003
...
```

## Required Fields

For every experiment record:

```text
Experiment ID:
Date:
Phase:
Git/Code Version:
Dataset Split:
Normalization:
Blocking:
Candidate Count:
Candidate Recall:
Model:
Features:
Threshold:
Precision:
Recall:
Macro F0.5:
Singleton False Positive Rate:
Runtime:
Peak Memory:
Result:
Decision:
```

## Baseline & Phase 3 Experiments

### EXP-000 — Project Baseline
- Status: Initialized in Phase 2 (Zero candidate baseline: Recall=0.0%, F0.5=0.0%).

### EXP-001 — Experiment A: Exact Normalized Name Blocking
- Date: 2026-09-25
- Phase: Phase 3 (Ablation A)
- Dataset Split: Validation S1 sample (10,000 queries) against 154,596 target pool
- Normalization: v1.0 (Unicode decomposition, case folding, legal suffix removal, domain stripping)
- Blocking: `exact_name` (Country + exact compact name)
- Refinement: None
- Candidate Recall (Overall): **50.84%** (S2: 51.75%, S3: 49.97%)
- S1 Full Coverage: 15.87%
- Candidate Count: Mean=**2.37**, Median=**2.0**, P95=**6.0**, P99=**14.0**
- Zero-Candidate Rate: 14.71%
- Reduction Ratio: **65,101.3x** (99.9985% reduction)
- Runtime: 1.37s | Peak Memory: 993 MB
- Decision: REJECTED as sole strategy (misses 49.16% of true matches due to typos and aliases).

### EXP-002 — Experiment B: Name + Address Blocking
- Date: 2026-09-25
- Phase: Phase 3 (Ablation B)
- Dataset Split: Validation S1 sample (10,000 queries) against 154,596 target pool
- Normalization: v1.0
- Blocking: `exact_name`, `name_address_combo`, `address_exact_combo`
- Refinement: None
- Candidate Recall (Overall): **79.50%** (S2: 78.60%, S3: 80.35%)
- S1 Full Coverage: 52.21%
- Candidate Count: Mean=**24.81**, Median=**7.0**, P95=**112.0**, P99=**208.0**
- Zero-Candidate Rate: 3.05%
- Reduction Ratio: **6,231.0x** (99.9840% reduction)
- Runtime: 0.98s | Peak Memory: 1,037 MB
- Decision: Incomplete recall; confirms address signals provide major recall boost (+28.7%).

### EXP-003 — Experiment C: Multi-Block Union (Broad)
- Date: 2026-09-25
- Phase: Phase 3 (Ablation C)
- Dataset Split: Validation S1 sample (10,000 queries) against 154,596 target pool
- Normalization: v1.0
- Blocking: 7-strategy union (`exact_name`, `sorted_name_tokens`, `first_two_name_tokens`, `longest_name_token`, `name_prefix_6`, `name_address_combo`, `address_exact_combo`)
- Refinement: None
- Candidate Recall (Overall): **91.31%** (S2: 90.44%, S3: 92.13%)
- S1 Full Coverage: 77.52%
- Candidate Count: Mean=**164.29**, Median=**106.0**, P95=**473.0**, P99=**613.0**
- Zero-Candidate Rate: 0.25%
- Reduction Ratio: **941.0x** (99.8937% reduction)
- Runtime: 4.17s | Peak Memory: 1,187 MB
- Decision: Excellent recall (91.3%), but candidate count is excessively high (P95=473). Requires refinement.

### EXP-004 — Experiment D: Multi-Block Union + Candidate Refinement (Threshold 0.15) [CHOSEN CONFIGURATION]
- Date: 2026-09-25
- Phase: Phase 3 (Selected Final Candidate Generation Pipeline)
- Dataset Split: Validation S1 sample (10,000 queries) against 154,596 target pool
- Normalization: v1.0
- Blocking: 7-strategy multi-block union
- Refinement: Heuristic composite scoring (n-gram Jaccard, token Jaccard, containment, address similarity, number overlap) + provenance bypass + top-50 ceiling (`min_score=0.15`, `max_cands=50`)
- Candidate Recall (Overall): **87.88%** (S2: 86.88%, S3: 88.83%)
- S1 Full Coverage: **71.46%**
- Candidate Count: Mean=**31.32**, Median=**47.0**, P95=**50.0**, P99=**50.0**
- Zero-Candidate Rate: **0.66%**
- >100 Candidate Rate: **0.00%**
- Reduction Ratio: **4,936.5x** (99.9797% search space reduction)
- Runtime: 18.63s | Peak Memory: 1,096 MB
- Decision: **ACCEPTED AS PHASE 3 STANDARD**. Retains 87.88% recall while reducing candidate pairs by >80% relative to broad union. Perfectly capped at $\le 50$ candidates per query for ML model ingestion.

### EXP-005 — Experiment E: Multi-Block Union + Stringent Refinement (Threshold 0.25)
- Date: 2026-09-25
- Phase: Phase 3 (Ablation E)
- Dataset Split: Validation S1 sample (10,000 queries) against 154,596 target pool
- Normalization: v1.0
- Blocking: 7-strategy multi-block union
- Refinement: `min_score=0.25`, `max_cands=30`
- Candidate Recall (Overall): **86.88%** (S2: 85.94%, S3: 87.76%)
- S1 Full Coverage: 69.14%
- Candidate Count: Mean=**13.61**, Median=**8.0**, P95=**30.0**, P99=**30.0**
- Zero-Candidate Rate: 1.58%
- Reduction Ratio: **11,362.8x** (99.9912% reduction)
- Runtime: 19.28s | Peak Memory: 1,098 MB
- Decision: Ultra-compact, but drops 1% recall and 2.3% S1 full coverage compared to EXP-004. EXP-004 preferred.

### EXP-006 — K-Sensitivity: K=50
- Date: 2026-09-25
- Phase: Phase 3.1 (Part A K-Sensitivity)
- Dataset Split: Validation S1 benchmark (10,000 queries, 154,596 target pool)
- Configuration: K=50, threshold=0.15, 7 baseline strategies
- Overall Recall: **87.88%** (S2: 86.88%, S3: 88.83%) | Multi-match: 87.90% | S1 Coverage: 71.46%
- Candidate Count: Mean=**31.32**, Median=**47.0**, P90=**50.0**, P95=**50.0**, P99=**50.0**, Max=**50**
- Reduction Ratio: **4,936.5x** (99.9797%) | Zero-Candidate: 0.66% | Runtime: 0.15s | Memory: 1,569 MB

### EXP-007 — K-Sensitivity: K=75
- Date: 2026-09-25
- Phase: Phase 3.1 (Part A K-Sensitivity)
- Configuration: K=75, threshold=0.15, 7 baseline strategies
- Overall Recall: **88.01%** (+0.13%) (S2: 86.98%, S3: 88.98%) | Multi-match: 88.03% | S1 Coverage: 71.73%
- Candidate Count: Mean=**43.10** (+11.78), Median=**47.0**, P90=**75.0**, P95=**75.0**, P99=**75.0**, Max=**75**
- Reduction Ratio: **3,586.7x** (99.9721%) | Zero-Candidate: 0.66% | Runtime: 0.24s | Memory: 1,571 MB

### EXP-008 — K-Sensitivity: K=100
- Date: 2026-09-25
- Phase: Phase 3.1 (Part A K-Sensitivity)
- Configuration: K=100, threshold=0.15, 7 baseline strategies
- Overall Recall: **88.11%** (+0.23%) (S2: 87.10%, S3: 89.06%) | Multi-match: 88.13% | S1 Coverage: 71.94%
- Candidate Count: Mean=**53.81** (+22.49), Median=**47.0**, P90=**100.0**, P95=**100.0**, P99=**100.0**, Max=**100**
- Reduction Ratio: **2,873.1x** (99.9652%) | Zero-Candidate: 0.66% | Runtime: 0.34s | Memory: 1,598 MB

### EXP-009 — K-Sensitivity: K=150
- Date: 2026-09-25
- Phase: Phase 3.1 (Part A K-Sensitivity)
- Configuration: K=150, threshold=0.15, 7 baseline strategies
- Overall Recall: **88.30%** (+0.42%) (S2: 87.33%, S3: 89.20%) | Multi-match: 88.32% | S1 Coverage: 72.25%
- Candidate Count: Mean=**71.63** (+40.31), Median=**47.0**, P90=**150.0**, P95=**150.0**, P99=**150.0**, Max=**150**
- Reduction Ratio: **2,158.3x** (99.9537%) | Zero-Candidate: 0.66% | Runtime: 0.21s | Memory: 1,598 MB
- Key Finding: Tripling K from 50 to 150 only increases recall by 0.42% while candidate volume surges 129%. K=50 is NOT the bottleneck.

### EXP-010 — Hardening Ablation: + Leetspeak Normalization View
- Date: 2026-09-25
- Phase: Phase 3.1 (Part D Strategy Ablation)
- Configuration: 8 strategies (Baseline + `name_leet_compact`), K=50, threshold=0.15
- Overall Recall: **88.04%** (+0.02% vs EXP-004-VERIFY) (S2: 87.05%, S3: 88.97%) | S1 Coverage: 71.76%
- Candidate Count: Mean=**31.33**, Median=**47.5**, P95=**50.0**, P99=**50.0** | Zero-Candidate: 0.66%
- Reduction Ratio: **4,934.2x** (99.9797%) | Runtime: 18.60s | Peak Memory: 1,143 MB

### EXP-011 — Hardening Ablation: + DBA/Trade-Name Alias Extraction
- Date: 2026-09-25
- Phase: Phase 3.1 (Part D Strategy Ablation)
- Configuration: 8 strategies (Baseline + `name_dba_alias`), K=50, threshold=0.15
- Overall Recall: **88.02%** (S2: 87.04%, S3: 88.96%) | S1 Coverage: 71.74%
- Candidate Count: Mean=**31.33**, Median=**47.5**, P95=**50.0**, P99=**50.0** | Zero-Candidate: 0.66%
- Reduction Ratio: **4,934.3x** (99.9797%) | Runtime: 19.02s | Peak Memory: 1,141 MB

### EXP-012 — Hardening Ablation: + Salient Address Locality Combo
- Date: 2026-09-25
- Phase: Phase 3.1 (Part D Strategy Ablation)
- Configuration: 8 strategies (Baseline + `address_salient_combo`), K=50, threshold=0.15
- Overall Recall: **88.31%** (+0.29% vs baseline) (S2: 87.25%, S3: 89.30%) | S1 Coverage: 72.46% (+72 queries fully resolved)
- Candidate Count: Mean=**31.36**, Median=**48.0**, P95=**50.0**, P99=**50.0** | Zero-Candidate: 0.66%
- Reduction Ratio: **4,930.0x** (99.9797%) | Runtime: 19.33s | Peak Memory: 1,155 MB

### EXP-013 — Phase 3.1 Selected: Hardened Multi-Block Union (All 10 Strategies, K=50) [CHOSEN STANDARD]
- Date: 2026-09-25
- Phase: Phase 3.1 (Selected Hardened Candidate Generation Pipeline)
- Configuration: 10 blocking strategies (`exact_name`, `sorted_name_tokens`, `first_two_name_tokens`, `longest_name_token`, `name_prefix_6`, `name_address_combo`, `address_exact_combo`, `name_leet_compact`, `name_dba_alias`, `address_salient_combo`), composite refinement (threshold 0.15), provenance bypass, K=50.
- Overall Recall: **88.32%** (S2: **87.26%**, S3: **89.32%**)
- S1 Full Coverage: **72.48%** (+74 entities fully resolved vs baseline 71.74%)
- Candidate Count: Mean=**31.36**, Median=**48.0**, P95=**50.0**, P99=**50.0**
- Zero-Candidate Rate: **0.66%**
- Reduction Ratio: **4,929.9x** (99.9797% search space reduction)
- Runtime: 20.73s | Peak Memory: 1,158 MB
### EXP-014 — Phase 4: Pairwise Feature Pipeline & Quality Audit
- Date: 2026-09-25
- Phase: Phase 4 (Pairwise Matching Feature Engineering)
- Feature Schema: `phase4-v1` (87 features across 9 groups)
- Input: EXP-013 candidate set
- Evaluation Sample: 5,000 Training S1 queries + 5,000 Validation S1 queries against 134,811 targets
- Candidate Pairs: 155,024 Training pairs (15,431 positive, 9.95% positive rate); 156,200 Validation pairs (15,372 positive, 9.84% positive rate)
- Sub-source positive rate: S2 = 9.43%, S3 = 10.48%
- Cardinality positive rate: Single-match S1 = 2.94%, Multi-match S1 = 10.94%
- Data Leakage: **0% leakage**. Frequency feature store fitted strictly on training entities. No ID or label leakage.
- NaN / Inf count: **0** across all 311,224 evaluated candidate pairs.
- Separation Power: Outstanding. `addr_exact_number_match` (+0.73 diff), `addr_token_jaccard` (+0.63 diff), `refine_composite_score` (+0.56 diff), `cross_agreement_count` (+2.71 diff).
- Runtime: 394.45s (311,224 candidate pairs processed, ~789 pairs/sec end-to-end including blocking + refinement + 87 features)
- Peak Memory: 1,979 MB
- Decision: **FEATURE PIPELINE ACCEPTED AND READY FOR PHASE 5 MODEL TRAINING**. All 59 unit tests pass.

### EXP-015 — Feature Ablation: Name Features Only
- Phase: Phase 5 (Feature Group Ablation)
- Model: HistGradientBoosting | Features: 15 (Name group only)
- Optimal Threshold: 0.60 | Macro F0.5: **0.7873** | PR-AUC: 0.7967 | Runtime: 0.96s

### EXP-016 — Feature Ablation: Name + Address Features
- Phase: Phase 5 (Feature Group Ablation)
- Model: HistGradientBoosting | Features: 30 (Name + Address groups)
- Optimal Threshold: 0.65 | Macro F0.5: **0.9324** | PR-AUC: 0.9981 | Runtime: 1.23s
- Insight: Adding address features causes massive +14.51% Macro F0.5 jump.

### EXP-017 — Feature Ablation: Name + Address + Cross-Field
- Phase: Phase 5 (Feature Group Ablation)
- Model: HistGradientBoosting | Features: 38
- Optimal Threshold: 0.65 | Macro F0.5: **0.9323** | PR-AUC: 0.9980 | Runtime: 1.44s

### EXP-018 — Feature Ablation: All Similarity Features
- Phase: Phase 5 (Feature Group Ablation)
- Model: HistGradientBoosting | Features: 47 (Groups 1 to 5)
- Optimal Threshold: 0.65 | Macro F0.5: **0.9326** | PR-AUC: 0.9981 | Runtime: 1.66s

### EXP-019 — Feature Ablation: All Similarity + Provenance
- Phase: Phase 5 (Feature Group Ablation)
- Model: HistGradientBoosting | Features: 58 (Groups 1 to 6)
- Optimal Threshold: 0.65 | Macro F0.5: **0.9325** | PR-AUC: 0.9982 | Runtime: 1.85s

### EXP-020 — Feature Ablation: All Similarity + Provenance + Refinement
- Phase: Phase 5 (Feature Group Ablation)
- Model: HistGradientBoosting | Features: 65 (Groups 1 to 7)
- Optimal Threshold: 0.65 | Macro F0.5: **0.9324** | PR-AUC: 0.9982 | Runtime: 1.93s

### EXP-021 — Feature Ablation: All 87 Features (Full Feature Set)
- Phase: Phase 5 (Feature Group Ablation)
- Model: HistGradientBoosting | Features: 87 (All 9 groups)
- Optimal Threshold: 0.70 | Macro F0.5: **0.9333** | PR-AUC: **0.9985** | Runtime: 2.51s
- Insight: Full feature set achieves the global highest Macro F0.5 and PR-AUC.

### EXP-022 — Feature Ablation: All Features Minus Frequency Features
- Phase: Phase 5 (Feature Group Ablation)
- Model: HistGradientBoosting | Features: 81 (Groups 1 to 8, excluding frequency)
- Optimal Threshold: 0.60 | Macro F0.5: **0.9329** | PR-AUC: 0.9986 | Runtime: 2.31s

### EXP-023 — Phase 5 Model Evaluation & Selection (MODEL-003 Champion)
- Date: 2026-09-25
- Phase: Phase 5 (Model Comparison & Threshold Optimization)
- Models Evaluated:
  - MODEL-001 (Logistic Regression): Macro F0.5 = 0.9328 (threshold 0.85, train 2.17s)
  - MODEL-001-BAL (Logistic Regression Balanced): Macro F0.5 = 0.9299 (threshold 0.95, train 1.34s)
  - MODEL-002 (Random Forest): Macro F0.5 = 0.9310 (threshold 0.60, train 3.96s)
  - MODEL-003 (HistGradientBoosting): Macro F0.5 = **0.9333** (threshold 0.70, train 2.83s) [CHAMPION]
  - MODEL-003-BAL (HistGradientBoosting Balanced): Macro F0.5 = 0.9330 (threshold 0.95, train 11.16s)
  - MODEL-004 (ExtraTrees Classifier): Macro F0.5 = 0.9310 (threshold 0.55, train 2.69s)
- Champion Metrics (`MODEL-003` at threshold **0.70**):
  - Macro F0.5: **0.9333**
  - Macro Precision: **0.9646** | Macro Recall: **0.8757**
  - Pairwise Precision: **0.9869** | Pairwise Recall: **0.9831**
  - ROC-AUC: **0.9998** | PR-AUC: **0.9985**
  - Singleton F0.5: **0.8709** | Multi-match F0.5: **0.9356** | Zero-match Accuracy: **0.9617**
  - Candidate-representable Recall: **98.31%** (15,112 / 15,372 links)
  - Final End-to-End Recall: **87.18%** (15,112 / 17,335 total truth)
- Decision: **ACCEPTED MODEL-003 (HistGradientBoosting) AT THRESHOLD 0.70 AS OFFICIAL PIPELINE CHAMPION**. All 69 unit tests pass.

### EXP-024 — Phase 6 End-to-End Test Inference & Submission Assembly
- Date: 2026-09-25
- Phase: Phase 6 (Production Inference, Submission Validation & ZIP Packaging)
- Model: `MODEL-003` (`HistGradientBoostingClassifier`, `max_iter=100`, `max_depth=8`, `learning_rate=0.1`, `min_samples_leaf=20`, `class_weight=None`)
- Training Data: Labeled candidate pairs (311,224 pairs, 30,803 positive matches) generated from all permitted training Source 1 entities against Source 2/3 targets
- Frequency Store: Fitted strictly on training entities (`models/production_frequency_store.pkl`). Zero test leakage.
- Feature Schema: `phase4-v1` (87 features)
- Candidate Generator: EXP-013 (10 blocking strategies, composite refinement, threshold 0.15, max K = 50)
- Frozen Threshold: $\theta = 0.70$
- Validation Performance (Benchmark): Macro F0.5 = **0.9333**, Macro Precision = **0.9646**, Macro Recall = **0.8757**, PR-AUC = **0.9985**
- Pipeline Entry Point: `scripts/run_final_submission.py`
- Output Files: `output/candidate_pairs.tsv` and `output/matching_results.tsv`
- Integrity Verification: `utils/validate_submission.py` (streaming mode, 15/15 rules passed)
- Subsetting Guarantee: Verified $100\%$ that all predicted matches belong to `candidate_pairs.tsv`
- Final Archive: `<team_name>_submission.zip` (verified required structure, 0% bloat)
- Tests: 74/74 unit tests passing cleanly

## Rules

1. Never compare experiments run on different validation splits without clearly labeling the difference.
2. Keep random seeds fixed unless testing seed sensitivity.
3. Do not claim an improvement from a change unless the metric comparison is reproducible.
4. Record runtime and memory for large-scale experiments.
5. Keep failed experiments. Failure is useful evidence.
6. Do not tune repeatedly against the hidden leaderboard without a documented reason.

## Leaderboard Submission Tracking

The challenge allows at most 5 submissions per day.

Record:

```text
Submission:
Date:
Day:
Experiment ID:
Public Score:
Files:
Change from previous:
Reason:
```

Never spend a submission without knowing what changed.

## What to Optimize

Primary:

```text
Macro F0.5
```

Secondary diagnostics:

```text
Candidate recall
Precision
Recall
Singleton false matches
Average candidate count
P95/P99 candidate count
Runtime
Memory
```

## Interpretation

A higher candidate recall is not automatically a better final system.

A blocking method that produces millions of candidates may have excellent recall but be unusable.

A matching method with high recall but many false merges can reduce F0.5.

Always evaluate the entire pipeline.
