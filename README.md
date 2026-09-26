# Amazon ML Challenge 2026 — Business Entity Resolution

[![Tests](https://img.shields.io/badge/tests-81%20passed-brightgreen.svg)](#running-tests)
[![Feature Schema](https://img.shields.io/badge/features-phase4--v1%20(87)-blue.svg)](#champion-architecture)
[![Decision Threshold](https://img.shields.io/badge/threshold-%CE%B8%20%3D%200.70-orange.svg)](#champion-architecture)

End-to-end machine learning system for resolving noisy business entity identities across multiple data sources (Source 1 query entities matched against Source 2 and Source 3 target entities).

---

## Table of Contents
1. [Overview & Architecture](#champion-architecture)
2. [Quickstart (Clone & Run Inference)](#quickstart-clone--run-inference)
3. [Repository Structure](#repository-structure)
4. [Dataset Setup](#dataset-setup)
5. [Training the Model from Scratch](#training-the-model-from-scratch)
6. [Generating Final Submission Outputs](#generating-final-submission-outputs)
7. [Submission Validation](#submission-validation)
8. [Running Tests](#running-tests)
9. [Metric](#evaluation-metric)
10. [Documentation Index](#documentation-index)

---

## Champion Architecture

- **Candidate Generation (EXP-013):** 10-strategy composite inverted index (`BlockingIndex`) with on-the-fly candidate refinement ($K=50$, score $\ge 0.15$). Memory-bounded country-partitioned streaming.
- **Pairwise Features (`phase4-v1`):** 87 features across 9 orthogonal groups (string similarities, token overlaps, address salient n-grams, character-level metrics, frequency TF-IDF weighting, provenance flags, and quality signals) with 0% data leakage.
- **Matching Model (`MODEL-003`):** `HistGradientBoostingClassifier` (`max_depth=8`, `learning_rate=0.1`, `max_iter=100`, `min_samples_leaf=20`). Pre-trained model weights are checked into [`models/`](models/).
- **Decision Threshold:** $\theta = 0.70$ calibrated on validation split (Validation Macro $F_{0.5} = 0.9333$, Macro Precision = 0.9646, Macro Recall = 0.8757).

---

## Quickstart (Clone & Run Inference)

Pre-trained production models and frequency stores are included in the repository under [`models/`](models/), so you can run test inference directly after cloning without retraining:

### 1. Clone the repository
```bash
git clone https://github.com/Nihar-Shadow/Amazon-ML.git
cd Amazon-ML
```

### 2. Set up environment & install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Place test datasets
Ensure test datasets are placed in `datasets/test/`:
```text
datasets/test/
├── test_source1.tsv
├── test_source2.tsv
└── test_source3.tsv
```

### 4. Run test inference
```bash
python scripts/run_final_submission.py
```
This produces the final submission artifacts in `output/`:
- `output/matching_results.tsv` (Predicted matches per S1 query $\ge 0.70$)
- `output/candidate_pairs.tsv` (Top candidates per S1 query)
- `amazon_ml_submission_submission.zip` (Submission package ready for upload)

---

## Repository Structure

```text
├── models/                     # Production model artifacts (pre-trained)
│   ├── production_matching_model.pkl
│   ├── production_frequency_store.pkl
│   └── production_model_metadata.json
├── src/                        # Core ML & data processing modules
│   ├── candidate_generation/   # BlockingIndex (EXP-013), normalizer, candidate refiner
│   ├── matching_features/      # 87-feature extraction engine (phase4-v1)
│   ├── matching_model/         # Model trainer, threshold evaluation, error analysis
│   ├── data/                   # Data loaders and ground-truth parsers
│   ├── evaluation/             # Macro F0.5 metrics and validation split logic
│   └── utils/                  # Logging, memory profiling utilities
├── scripts/                    # CLI execution pipelines
│   ├── run_final_submission.py # Production inference & submission assembly
│   ├── run_phase6_1_diagnostic.py # Multi-country correctness & memory diagnostic
│   ├── train_production_model.py  # Retrains production HistGradientBoosting
│   ├── train_and_evaluate_phase5.py # Phase 5 model benchmark and threshold sweep
│   └── generate_validation_split.py # Generates train/validation splits
├── tests/                      # Pytest unit & regression tests (81 tests)
├── utils/
│   └── validate_submission.py  # 15-rule submission integrity validator
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Dataset Setup

Place competition TSV files under `datasets/` (excluded from git via `.gitignore`):

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

TSV schema expected:
- `source1.tsv`: `entity_id \t business_name \t business_address \t country`
- `source2.tsv`: `entity_id \t business_name \t business_address \t country`
- `source3.tsv`: `entity_id \t business_name \t business_address \t country`
- `train_ground_truth.tsv`: `source1_entity_id \t matching_entity_ids` (comma-separated or empty)

---

## Training the Model from Scratch

If you wish to retrain or train from raw datasets:

### Step 1: Generate stratified validation split
```bash
python scripts/generate_validation_split.py
```
Outputs validation and training S1 ID lists in `eda/train_s1_ids.txt` and `eda/val_s1_ids.txt`.

### Step 2: Extract candidate features & benchmark models
```bash
python scripts/train_and_evaluate_phase5.py
```
This extracts candidate pairs via EXP-013 blocking, computes the 87 features, saves caches in `data_cache/`, and sweeps decision thresholds $\theta \in [0.10, 0.90]$.

### Step 3: Train final production model
```bash
python scripts/train_production_model.py
```
Trains the frozen `HistGradientBoostingClassifier` on the combined candidate dataset and exports:
- `models/production_matching_model.pkl`
- `models/production_frequency_store.pkl`
- `models/production_model_metadata.json`

---

## Generating Final Submission Outputs

Run the memory-bounded, country-partitioned inference pipeline:

```bash
python scripts/run_final_submission.py \
  --test-dir datasets/test \
  --model-path models/production_matching_model.pkl \
  --freq-path models/production_frequency_store.pkl \
  --output-dir output \
  --team-name amazon_ml_submission
```

### CLI Arguments:
- `--test-dir`: Path to folder containing test TSVs (default: `datasets/test`).
- `--output-dir`: Output destination directory (default: `output`).
- `--max-queries`: Optional integer cap on S1 queries (useful for profiling/smoke tests, e.g. `--max-queries 500`).
- `--skip-zip`: Generate TSVs without building the final ZIP archive.

### Produced Outputs:
1. `output/matching_results.tsv`:
   ```tsv
   source1_entity_id	matching_entity_ids
   s1_00000001	s2_00123456,s3_00789012
   s1_00000002	
   ```
2. `output/candidate_pairs.tsv`: Top candidate pool per S1 query ($K \le 50$).
3. `amazon_ml_submission_submission.zip`: Verified competition submission zip.

---

## Submission Validation

Validate generated submission TSVs against official competition constraints:

```bash
python utils/validate_submission.py \
  --results-file output/matching_results.tsv \
  --candidates-file output/candidate_pairs.tsv \
  --test-s1 datasets/test/test_source1.tsv \
  --test-s2 datasets/test/test_source2.tsv \
  --test-s3 datasets/test/test_source3.tsv
```

Checks 15 integrity rules:
- Exact 1:1 row count matching Test S1
- Deterministic lexicographical ordering of S1 IDs and predicted target IDs
- Strict candidate subset constraint (every predicted match must be in candidates)
- Zero cross-country entity matching
- Valid entity ID format and existence

---

## Running Tests

Run the full pytest suite (81 tests covering blocking, normalizer, feature extractors, model inference, and Phase 6.1 memory regression):

```bash
pytest tests/
```

To run a specific test suite:
```bash
pytest tests/test_phase6_1_regression.py
pytest tests/test_matching_features.py
```

---

## Evaluation Metric

The primary competition metric is **Macro $F_{0.5}$** computed across all Source 1 entities:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- Precision is weighted $2\times$ more heavily than Recall ($F_{0.5}$ penalizes false positive matches strongly).
- Unmatched S1 entities (empty predictions) are correctly handled and credited.

---

## Documentation Index

Detailed architectural specs, experiment logs, and diagnostic reports:
- [ARCHITECTURE.md](ARCHITECTURE.md) — System design and end-to-end pipeline diagrams
- [TECH_SPEC.md](TECH_SPEC.md) — Technical specifications, data models, and schemas
- [PHASES.md](PHASES.md) — Multi-phase development roadmap and milestones
- [EXPERIMENTS.md](EXPERIMENTS.md) — Blocking ablation studies (EXP-001 through EXP-013)
- [PHASE_6_1_PRODUCTION_FIX_REPORT.md](PHASE_6_1_PRODUCTION_FIX_REPORT.md) — Production inference memory bounding & country normalization audit
- [AGENT_RULES.md](AGENT_RULES.md) — Operational guidelines and design principles
