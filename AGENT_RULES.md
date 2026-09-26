# AI Agent Rules — Amazon ML Challenge 2026

## Mission

You are an engineering agent working on the Amazon ML Challenge 2026 Business Entity Resolution project.

Your job is to produce a reproducible, competition-compliant ML pipeline.

## Absolute Rules

### Rule 1 — Preserve Dataset Integrity

Never:

- modify original TSV files
- rename original dataset files
- delete original dataset files
- overwrite original dataset files
- add predictions into the dataset directories

Generated artifacts belong under dedicated directories such as:

```text
output/
eda/
models/
experiments/
reports/
```

### Rule 2 — No External Entity Lookup

Never use external information to resolve a business identity.

Forbidden:

- Google/Bing business searches
- government registration lookup
- geocoding
- Google Maps/business APIs
- commercial ER APIs
- external business databases
- internet-derived business attributes
- external enrichment datasets

Allowed:

- computation over supplied data
- standard local Python libraries
- ML libraries
- string similarity
- tokenization
- TF-IDF
- local deterministic normalization
- local ML models that comply with challenge licensing/size rules

### Rule 3 — Do Not Invent Data

If a statistic has not been measured, label it as unknown.

Do not convert an assumption into a fact.

### Rule 4 — Separate Blocking from Matching

The architecture must distinguish:

```text
Raw Data
   ↓
Normalization
   ↓
Blocking / Candidate Generation
   ↓
Pair Feature Extraction
   ↓
Matching Model
   ↓
Decision / Thresholding
   ↓
Post-processing
   ↓
Submission
```

### Rule 5 — Candidate Recall Comes First

A candidate generator that removes a true match makes that match impossible for the downstream model to recover.

Always measure candidate recall.

### Rule 6 — Precision Matters

The challenge uses macro F_0.5, which is precision-heavy.

Do not blindly increase recall by accepting weak matches.

### Rule 7 — Validate Every Change

Every major change must be evaluated against a fixed validation split.

Record:

- experiment ID
- code/config version
- candidate recall
- precision
- recall
- F_0.5
- average candidates
- P95/P99 candidate counts
- false-positive behavior
- singleton behavior

### Rule 8 — No Premature Complexity

Do not add:

- deep learning
- LLMs
- embeddings
- large models
- complex graph optimization

unless a measured baseline shows that the added complexity is justified.

### Rule 9 — Memory Awareness

The dataset contains tens of millions of records.

Avoid:

- full Cartesian joins
- all-pairs fuzzy matching
- unnecessary DataFrame copies
- Python objects for every pair
- materializing enormous intermediate matrices

Prefer:

- indexes
- dictionaries
- arrays
- chunking
- vectorized operations
- sparse representations
- compact feature matrices

### Rule 10 — France Must Not Be Hard-Coded Away

France is present in test data but absent from training.

Never:

```python
country in {"US", "India"}
```

as a filter.

Country handling must be open-set.

### Rule 11 — Do Not Treat EDA Inferences as Guaranteed Test Rules

A property observed in training must not automatically become a hard test constraint unless the challenge statement guarantees it.

For example, if training shows that a target appears under at most one S1, treat this as an observed training property and validate whether using it as post-processing is safe.

### Rule 12 — Reproducibility

Use:

- fixed random seeds
- pinned dependencies
- deterministic preprocessing where possible
- documented configurations
- saved experiment metadata

## Agent Workflow

Before coding:

1. Read `PROJECT_CONTEXT.md`
2. Read `ARCHITECTURE.md`
3. Read `TECH_SPEC.md`
4. Read `PHASES.md`
5. Read `IMPLEMENTATION.md`
6. Read `EXPERIMENTS.md`
7. Check current phase status
8. Inspect existing code before creating new code

Before changing architecture:

- explain why the change is needed
- identify affected modules
- define validation criteria

After implementation:

- run tests
- run relevant validation
- update phase status
- update experiment log
- document discovered issues

## Stop Conditions

Stop and ask for clarification if:

- dataset paths differ from the documented structure
- required challenge files are missing
- a proposed technique requires external data
- licensing of a model is unclear
- a change could invalidate previous experiment comparisons
- a result contradicts the verified data and cannot be explained

## Definition of Done

A phase is not complete merely because code exists.

A phase is complete only when:

1. implementation exists
2. tests pass
3. validation runs
4. metrics are recorded
5. documentation is updated
6. no challenge constraint is violated
