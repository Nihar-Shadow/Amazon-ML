# Amazon ML Challenge 2026 — Business Entity Resolution

## Goal

Resolve business identities across three noisy data sources.

Source 1 is the reference/query source.

For every S1 entity, predict matching S2/S3 entities.

## Dataset

```text
datasets/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## Documentation

Read these before changing the project:

1. `AGENT_RULES.md`
2. `PROJECT_CONTEXT.md`
3. `ARCHITECTURE.md`
4. `TECH_SPEC.md`
5. `PHASES.md`
6. `IMPLEMENTATION.md`
7. `DESIGN.md`
8. `EXPERIMENTS.md`
9. `SUBMISSION.md`

## Current Status

**Phase 6 — End-to-End Test Inference & Submission Assembly is COMPLETE.**

Champion Architecture:
- Candidate Generation: EXP-013 (10-strategy multi-block inverted index + composite refinement, $K=50$, $\ge 0.15$)
- Pairwise Features: `phase4-v1` schema (87 features across 9 orthogonal groups, 0% validation leakage)
- ML Model: `MODEL-003` (`HistGradientBoostingClassifier`, `max_iter=100`, `max_depth=8`, `learning_rate=0.1`, `min_samples_leaf=20`)
- Decision Threshold: $\theta = 0.70$ (Validation Macro F0.5 = 0.9333, Macro Precision = 0.9646, Macro Recall = 0.8757)
- Validation: 74/74 unit tests passing (`python -m unittest discover tests/`)
- Submission Validator: `utils/validate_submission.py` (all 15 integrity rules passing)

## Running the Production Submission Pipeline

```bash
python scripts/run_final_submission.py \
  --test-dir datasets/test \
  --model-path models/production_matching_model.pkl \
  --freq-path models/production_frequency_store.pkl \
  --output-dir output \
  --team-name amazon_ml_submission
```

## Core Pipeline

```text
Data
 ↓
Validation
 ↓
Normalization
 ↓
Blocking
 ↓
Candidate Features
 ↓
Matching Model
 ↓
Threshold Calibration
 ↓
Post-processing
 ↓
Submission Validation
```

## Metric

Primary metric:

**Macro F_0.5**

```text
F0.5 = (1.25 * Precision * Recall) /
       (0.25 * Precision + Recall)
```

Precision is weighted more heavily than recall.

## Important

This project must use only challenge-provided data for entity resolution.

No external business lookup or data enrichment.

## Development Principle

Measure first.

Implement second.

Do not add complexity without validation evidence.
