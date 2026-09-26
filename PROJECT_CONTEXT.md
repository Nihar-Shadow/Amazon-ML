# Amazon ML Challenge 2026 — Project Context

## 1. Project Identity

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Goal:** Build an ML/data-science pipeline that resolves noisy business records across three independent sources.

Source 1 is the deduplicated reference/query source. For every Source 1 entity, the system must identify matching records from Source 2 and/or Source 3. A Source 1 entity can have zero, one, or many matches.

## 2. Source of Truth

The following documents are authoritative for challenge rules:

- Amazon ML Challenge 2026 Guidelines
- Amazon ML Challenge 2026 Business Entity Resolution Problem Statement
- The actual supplied dataset files
- The project's verified EDA report

Do not invent challenge rules.

## 3. Challenge Window

The provided guidelines state:

- Start: 25 September 2026, 12:00 AM IST
- End: 27 September 2026, 11:59 PM IST
- Maximum submissions: 5 per day for 3 days.
- Submission history must be maintained.

## 4. Dataset Structure

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

All source files are tab-separated.

Source schema:

```text
entity_id
business_name
business_address
country
```

Ground truth schema:

```text
source1_entity_id
matched_entity_ids
```

`matched_entity_ids` is a comma-separated list of S2/S3 IDs and can be empty.

## 5. Dataset Scale

Verified EDA reported:

- train_source1: 2,206,821
- train_source2: 5,034,616
- train_source3: 5,285,603
- train_ground_truth: 2,206,821
- test_source1: 1,732,544
- test_source2: 4,887,273
- test_source3: 5,082,316

Total records analyzed: 26,435,994.

This scale makes Cartesian/brute-force pair comparison infeasible.

## 6. Important Data Characteristics

Verified EDA findings:

- `business_name` has no missing values.
- Source 1 addresses have no missing values.
- Source 2 and Source 3 have approximately 3% missing addresses.
- Training contains US and India.
- Test additionally contains France.
- Names contain abbreviations, legal suffix differences, punctuation variation, typos, URL/domain forms, and script variation.
- Addresses contain abbreviations, reordered components, truncation, missing components, landmarks, and regional-script variation.
- Source 1 has a high repeated-name rate.
- Training ground truth contains zero-match S1 entities and many S1 entities with multiple matches.
- The EDA observed that no training target entity was linked to more than one S1 entity.

## 7. Critical Challenge Constraints

Never use:

- external business databases
- government business-registration lookup
- commercial entity-resolution APIs
- geocoding APIs
- internet-based entity lookup
- external data augmentation

The solution must be based on the supplied challenge data.

Model constraint from the problem statement:

- Final model must use a MIT or Apache 2.0 licensed model.
- Maximum model size: 8 billion parameters.

## 8. Evaluation

The leaderboard evaluates `matching_results.tsv` using macro F_0.5.

F_0.5:

```text
F0.5 = (1.25 * Precision * Recall) /
       (0.25 * Precision + Recall)
```

It is calculated per Source 1 entity and macro-averaged.

F_0.5 weights precision more heavily than recall.

Singleton behavior:

- true matches = empty + prediction empty => score 1.0
- true matches = empty + prediction non-empty => score 0.0

Do not optimize ordinary accuracy instead of the actual metric.

## 9. Required Outputs

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

Every test Source 1 entity must appear exactly once in `matching_results.tsv`.

Final matches must be a subset of candidates in `candidate_pairs.tsv`.

## 10. Engineering Philosophy

The project must be built incrementally:

1. Data integrity
2. EDA
3. Validation infrastructure
4. Blocking/candidate generation
5. Pair features
6. Matching model
7. Threshold calibration
8. Inference
9. Submission validation
10. Documentation
11. Controlled leaderboard experiments

Never skip validation just to reach a model quickly.

## 11. Current Project State

EDA is complete.

Next active phase:

**Phase 2 — Validation Infrastructure**

The next implementation should establish:

- ground-truth parsing
- deterministic train/validation split
- exact F_0.5 metric
- singleton handling
- candidate-recall metrics
- candidate-count diagnostics
- tests

Do not jump to final model training before Phase 2 is validated.
