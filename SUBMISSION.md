# Submission Specification

## 1. Leaderboard Upload

During the challenge, upload:

```text
matching_results.tsv
```

This is the file scored by the leaderboard.

## 2. Final Package

Required:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
│
└── Documentation_template.md
```

## 3. matching_results.tsv

Columns:

```text
source1_entity_id
matched_entity_ids
```

Rules:

- exactly one row for every test Source 1 entity
- empty match list when no matches
- no duplicate IDs
- matched IDs must exist in test S2/S3
- no S1 IDs as matches

## 4. candidate_pairs.tsv

Columns:

```text
source1_entity_id
candidate_entity_ids
```

Rules:

- exactly one row per test S1
- candidates are the final set passed into the matching model
- no duplicates
- only test S2/S3 IDs
- every final match must appear in candidates

## 5. Official Validator

Use:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir datasets/test
```

Do this before consuming a leaderboard submission.

## 6. Methodology Document

Must describe:

- methodology
- candidate generation/blocking
- model architecture
- feature engineering
- other relevant approach information

There is no page limit in the problem statement; technical clarity is preferred.

## 7. Model Compliance

Final model must satisfy:

- MIT or Apache 2.0 license
- no more than 8B parameters

## 8. Fair Play

Do not use:

- external entity lookup
- external databases
- geocoding
- commercial ER APIs
- internet data augmentation

The final package may be reviewed for compliance.

## 9. Final Submission Assembly & Reproducibility

The complete submission package is assembled using the unified runner:

```bash
python scripts/run_final_submission.py \
  --test-dir datasets/test \
  --model-path models/production_matching_model.pkl \
  --freq-path models/production_frequency_store.pkl \
  --output-dir output \
  --team-name amazon_ml_submission
```

This single command executes:
1. Model and training-fitted frequency store loading
2. EXP-013 candidate generation across test sources
3. Exact candidate pair generation (`candidate_pairs.tsv`)
4. 87-feature extraction (`phase4-v1` schema)
5. Supervised probability inference using frozen `HistGradientBoostingClassifier`
6. Decision threshold filtering at $\theta = 0.70$
7. Deterministic match output generation (`matching_results.tsv`)
8. Automated validation via `utils/validate_submission.py`
9. Clean archive packaging into `<team_name>_submission.zip` matching official structure:
   - `output/matching_results.tsv`
   - `output/candidate_pairs.tsv`
   - `code/business_entity_resolution/src/`
   - `code/business_entity_resolution/README.md`
   - `code/business_entity_resolution/requirements.txt`
   - `Documentation_template.md`
