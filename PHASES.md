# Phase Plan

## Phase 0 — Project Setup

Status: COMPLETE

Tasks:

- create repository
- establish dataset structure
- establish documentation
- protect raw data

## Phase 1 — Dataset EDA

Status: COMPLETE

Verified:

- file sizes
- row counts
- schemas
- missing values
- uniqueness
- country distributions
- name/address characteristics
- ground-truth multiplicity
- examples of real-world variation
- test distribution shift

EDA report: `eda_report.md`

## Phase 2 — Validation Infrastructure

Status: COMPLETE

Verified:
- [x] ground-truth parser with format and duplicate validations
- [x] deterministic 80/20 train/validation split (1,765,457 train / 441,364 val S1 anchors)
- [x] exact macro F_0.5 metric with precision weighting
- [x] singleton edge-case handling (empty/empty -> 1.0, empty/non-empty -> 0.0)
- [x] candidate recall and S1 coverage metrics
- [x] candidate count diagnostics (mean, median, P90, P95, P99, cardinality flags)
- [x] 15 unit tests passing
- [x] Phase 2 report: `PHASE2_VALIDATION_REPORT.md`

## Phase 3 — Candidate Generation & Blocking (Multi-Block + Refinement)

Status: COMPLETE

Implemented & Verified:
- [x] Unicode decomposition, accent/ligature stripping (`œ` -> `oe`), case folding
- [x] URL/domain normalization (`.com`, `http://`, `www.`)
- [x] International/US/Indian/French corporate legal suffix handling
- [x] Address structural abbreviation normalization (St, Rd, Ave, Blvd, etc.)
- [x] Open-set country preservation
- [x] 7-strategy multi-block inverted index (`exact_name`, `sorted_name_tokens`, `first_two_name_tokens`, `longest_name_token`, `name_prefix_6`, `name_address_combo`, `address_exact_combo`)
- [x] Candidate pool union with provenance tracking
- [x] Deterministic candidate refinement with heuristic scoring & top-50 ceiling
- [x] Ablation experiments (EXP-001 through EXP-005)
- [x] Candidate pairs contract (`candidate_pairs.tsv` represents the exact final candidate set)
- [x] Search space reduction ratio of 4,936.5x (99.98% reduction) with 87.88% candidate recall

## Phase 3.1 — Candidate Recall Hardening

Status: COMPLETE

Implemented & Verified:
- [x] Evaluated K sensitivity bottleneck (EXP-006 through EXP-009): Proved K=50 is NOT the bottleneck (K=150 only yields +0.42% recall for 129% candidate bloat).
- [x] Candidate-loss attribution: Diagnosed 4,192 missing links -> 71.71% BLOCKING_MISS, 19.66% REFINEMENT_MISS, 8.64% TOP_K_MISS, 0.00% COUNTRY_FILTER_MISS.
- [x] Failure taxonomy established: Character leetspeak/typos, DBA/honorific aliases, cross-script Latin/Devanagari, municipal plot/locality variations.
- [x] Developed 3 targeted blocking views:
  1. `name_leet_compact`: Character/digit substitution normalization (@ -> a, 0 -> o, 8 -> b, 1 -> l).
  2. `name_dba_alias`: Trade-name / DBA / doing-business-as alias extraction.
  3. `address_salient_combo`: Number + salient non-generic locality/city token combination.
- [x] Evaluated ablations (EXP-010 through EXP-013): EXP-013 achieves 88.32% overall recall (+0.30% gain, +74 full S1 coverage) with mean candidates stable at 31.36 and 99.9797% reduction ratio.
- [x] Preserved open-set country and no external data rules.
- [x] 39/39 unit tests pass cleanly (`tests/test_candidate_generation.py`).
- [x] Upgraded project standard for Phase 4 ingestion to EXP-013.

## Phase 4 — Pairwise Matching Feature Engineering

Status: COMPLETE

Implemented & Verified:
- [x] Implemented modular feature extraction architecture in `src/matching_features/`:
  - `name_features.py` (15 features: exact normalized/compact, token Jaccard, sorted token Jaccard, 2/3/4-grams, edit similarities, containment, prefix/suffix, token counts)
  - `address_features.py` (15 features: exact/compact address, token Jaccard, 3/4-grams, edit sim, shared numbers, exact/leading-zero number match, postal match, locality/state overlap, missing indicator)
  - `cross_field_features.py` (8 features: strong name + strong address, strong name + weak address, weak name + strong address, exact number + similar name, exact locality + similar name, same country interaction, agreement count)
  - `country_features.py` (4 features: open-set country match, normalized match, missing indicator, conflict indicator)
  - `view_features.py` (5 features: leet name exact/sim, DBA alias match, compact name match, compact address match)
  - `provenance_features.py` (11 features: 10 blocking strategy indicators + strategy count)
  - `refinement_features.py` (7 features: heuristic composite score, n-gram sim, token sim, containment, address sim, number match, provenance bypass)
  - `quality_features.py` (16 features: lengths, token counts, missing address indicators, non-Latin script, mixed script pair, URL/domain, legal suffix, DBA)
  - `frequency_features.py` (6 features: entity cardinality and frequency features fitted strictly on training anchors)
- [x] Feature schema versioned (`FEATURE_SCHEMA_VERSION = "phase4-v1"`, exactly 87 features).
- [x] Training labels constructed cleanly (`label = 1` for ground-truth match, `0` otherwise; 155,024 training pairs evaluated, 9.95% positive rate).
- [x] Strict validation isolation: validation anchors never leak into feature logic or frequency tables (156,200 validation candidate pairs evaluated, 9.84% positive rate).
- [x] Zero NaN or Infinite values across all 87 features.
- [x] Comprehensive feature quality, correlation, and leakage audits completed (`experiments/phase4_feature_audit.json`).
- [x] Verified high separation power between positive and negative matches (e.g. `addr_exact_number_match` diff +0.73, `addr_token_jaccard` diff +0.63, `refine_composite_score` diff +0.56).
- [x] 20 unit tests added in `tests/test_matching_features.py` (59 total project tests passing).

## Phase 5 — Model Training, Validation & Threshold Optimization

Status: COMPLETE

Implemented & Verified:
- [x] Trained 4 supervised model families:
  1. `MODEL-001`: Logistic Regression (Standardized, L2 penalty & Class-Weighted variants)
  2. `MODEL-002`: Random Forest (100 trees, max_depth=14, min_samples_leaf=5)
  3. `MODEL-003`: HistGradientBoosting (LightGBM-equivalent histogram boosting, max_depth=8, lr=0.1)
  4. `MODEL-004`: ExtraTrees Classifier (100 trees, max_depth=14)
- [x] Evaluated on 156,200 validation candidate pairs across 5,000 isolated S1 anchors.
- [x] Fine-grained threshold sweep across 19 thresholds [0.05 to 0.95].
- [x] Entity-level Macro F0.5 optimization: Selected `MODEL-003` at threshold **0.70**, achieving **0.9333 Macro F0.5** (Macro Precision: 0.9646, Macro Recall: 0.8757, PR-AUC: 0.9985).
- [x] Candidate recall ceiling: 88.68% candidate generation ceiling, **98.31% model recall within candidates** (15,112 / 15,372 true links recovered), final end-to-end recall 87.18%.
- [x] Feature group ablations (EXP-015 to EXP-022): Confirmed Name+Address achieves 0.9324; full 87 features achieves peak 0.9333 Macro F0.5.
- [x] Forensic error analysis: 201 false positives (principally generic name collisions and legal suffix artifacts) and 260 false negatives (missing addresses and severe typos).
- [x] Interactive Jupyter Notebook created: `notebooks/phase5_model_training.ipynb` (14 cells).
- [x] Reusable model logic modularized in `src/matching_model/`.
- [x] 10 unit tests added in `tests/test_matching_model.py` (69 total project tests passing).

## Phase 6 — End-to-End Test Inference & Submission Assembly

Status: COMPLETE

Implemented & Verified:
- [x] Production Model Freezing: `HistGradientBoostingClassifier` (`max_iter=100`, `max_depth=8`, `learning_rate=0.1`, `min_samples_leaf=20`, `class_weight=None`) trained on all 311,224 labeled candidate pairs from training data (`models/production_matching_model.pkl`).
- [x] Leakage Prevention: Frequency feature store fitted strictly on training entities (`models/production_frequency_store.pkl`). Zero test-derived frequencies.
- [x] Feature Schema: Exact `phase4-v1` 87-feature schema frozen and verified.
- [x] Decision Threshold: Frozen at optimal validation threshold $\theta = 0.70$.
- [x] Candidate Generator: EXP-013 unchanged (10 blocking strategies, composite refinement, threshold 0.15, max K = 50).
- [x] Candidate Pairs Contract: `candidate_pairs.tsv` represents the exact model input candidate set.
- [x] Partitioning Strategy: Country-wise memory-safe execution (`France`, `US`, `India`) keeps peak RAM under 800 MB while preserving open-set countries.
- [x] Output Generation: `output/candidate_pairs.tsv` and `output/matching_results.tsv` generated with deterministic lexicographical ordering.
- [x] Subsetting Guarantee: 100% of predicted matches belong strictly to `candidate_pairs.tsv`.
- [x] Official Submission Validator: Automated streaming validator `utils/validate_submission.py` enforcing all 15 challenge integrity checks with $O(1)$ memory usage.
- [x] Submission ZIP Packaging: Automated packaging of `<team_name>_submission.zip` matching official required directory layout.
- [x] Reproducibility Entry Point: `scripts/run_final_submission.py` coordinates the full test pipeline from code without notebook dependency.
- [x] Unit Tests: 74/74 unit tests passing cleanly (`python -m unittest discover tests/`).
- [x] Methodology Documentation: `Documentation_template.md` fully detailed and verified.
