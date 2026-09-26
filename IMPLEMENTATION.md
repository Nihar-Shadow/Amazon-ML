# Implementation Guide

## Current Objective
Phase 6: End-to-End Test Inference & Submission Assembly is COMPLETE.
Production Pipeline: `scripts/run_final_submission.py`
Champion Model: `MODEL-003` (`HistGradientBoostingClassifier`, $\theta=0.70$, 87 features, `phase4-v1`).
Validation Macro F0.5: 0.9333.
Official Submission Validation: PASSED.

## Repository Layout
```text
src/
├── candidate_generation/
│   ├── __init__.py
│   ├── normalizer.py
│   ├── blocking.py
│   ├── candidate_pool.py
│   ├── candidate_refiner.py
│   ├── candidate_generator.py
│   └── diagnostics.py
├── matching_features/
│   ├── __init__.py
│   ├── feature_schema.py
│   ├── string_metrics.py
│   ├── name_features.py
│   ├── address_features.py
│   ├── cross_field_features.py
│   ├── country_features.py
│   ├── view_features.py
│   ├── provenance_features.py
│   ├── refinement_features.py
│   ├── quality_features.py
│   ├── frequency_features.py
│   └── feature_extractor.py
├── matching_model/
│   ├── __init__.py
│   ├── trainer.py
│   ├── evaluation.py
│   ├── feature_ablations.py
│   ├── error_analysis.py
│   └── candidate_recall_analysis.py
├── data/
│   ├── __init__.py
│   ├── loader.py
│   └── ground_truth.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── validation.py
└── utils/
    ├── __init__.py
    └── logging_utils.py

scripts/
├── run_final_submission.py
├── audit_phase4_features.py
├── generate_phase5_notebook.py
└── benchmark_test_throughput.py

utils/
└── validate_submission.py

tests/
├── test_evaluation.py
├── test_candidate_generation.py
├── test_matching_features.py
├── test_matching_model.py
├── test_validate_submission.py
└── test_run_final_submission.py
```


## Implementation Order

### Step 1 — Loader

Implement safe TSV readers.

Requirements:

- explicit `sep="\t"`
- string IDs
- no dataset mutation
- chunking support

### Step 2 — Ground Truth

Parse:

```text
source1_entity_id -> set(matched_entity_ids)
```

Validate IDs against training target pools.

### Step 3 — Validation Split

Split Source 1 anchors deterministically.

Do not randomly split individual pairs.

Reason:

The ground truth is defined around S1 entities. Validation must preserve the entity-level prediction task.

### Step 4 — Metrics

Implement:

- precision
- recall
- F0.5
- macro F0.5
- singleton scoring

### Step 5 — Candidate Metrics

Implement:

- link-level candidate recall
- S1 full coverage
- candidate count statistics

### Step 6 — Tests

Test tiny synthetic examples before large-data execution.

Example:

```text
truth = {A, B}
prediction = {A, C}

precision = 1/2
recall = 1/2
```

F0.5 should be computed exactly according to the challenge formula.

## Coding Rules

- Type annotate public functions.
- Keep functions small.
- Avoid hidden global state.
- Log useful progress.
- Avoid printing millions of rows.
- Never silently catch data corruption.
- Raise explicit errors for malformed inputs.

## Large Dataset Rules

Before executing a new operation against millions of records, estimate:

- memory
- time
- output size

Never create:

```python
for s1 in source1:
    for s2 in source2:
        ...
```

or equivalent Cartesian operations.

## Baseline Candidate Generator

The first baseline may be intentionally simple.

Its purpose is to verify the candidate-recall framework, not to win the competition.

Do not optimize baseline performance until the metrics are trusted.

## Definition of Done for Phase 2

Phase 2 is complete only when:

```text
[ ] Ground truth loads
[ ] Validation split is deterministic
[ ] F0.5 passes synthetic tests
[ ] Singleton cases pass tests
[ ] Candidate recall works
[ ] Candidate diagnostics work
[ ] No raw data files changed
[ ] PHASE2_VALIDATION_REPORT.md exists
```
