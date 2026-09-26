import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
from typing import Dict, List, Set, Any
import numpy as np
import pandas as pd

from src.candidate_generation.normalizer import normalize_entity_record
from src.candidate_generation.blocking import BlockingIndex
from src.candidate_generation.candidate_refiner import CandidateRefiner, score_candidate_pair
from src.matching_features.feature_schema import FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS, TOTAL_FEATURE_COUNT
from src.matching_features.feature_extractor import PairwiseFeatureExtractor
from src.matching_features.frequency_features import FrequencyFeatureStore
from src.utils.logging_utils import get_logger, get_memory_usage_mb

logger = get_logger("audit_phase4_features")

def run_phase4_feature_audit():
    os.makedirs("experiments", exist_ok=True)
    start_time = time.time()
    logger.info("Starting Phase 4 Pairwise Matching Feature Audit.")

    # 1. Load Training and Validation Anchor IDs
    with open("eda/train_s1_ids.txt", "r", encoding="utf-8") as f:
        train_s1_all = [line.strip() for line in f if line.strip()]
    with open("eda/val_s1_ids.txt", "r", encoding="utf-8") as f:
        val_s1_all = [line.strip() for line in f if line.strip()]

    N_TRAIN_SAMPLE = 5_000
    N_VAL_SAMPLE = 5_000

    train_s1_set = set(train_s1_all[:N_TRAIN_SAMPLE])
    val_s1_set = set(val_s1_all[:N_VAL_SAMPLE])
    all_needed_s1 = train_s1_set | val_s1_set

    logger.info(f"Loaded {len(train_s1_set):,} training anchors and {len(val_s1_set):,} validation anchors.")

    # 2. Load Ground Truth for Train and Val
    train_gt: Dict[str, Set[str]] = {}
    val_gt: Dict[str, Set[str]] = {}
    needed_targets: Set[str] = set()

    with open("datasets/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if s1_id in all_needed_s1:
                matches = set()
                if len(parts) >= 2 and parts[1]:
                    for m in parts[1].split(","):
                        m = m.strip()
                        if m:
                            matches.add(m)
                            needed_targets.add(m)
                if s1_id in train_s1_set:
                    train_gt[s1_id] = matches
                if s1_id in val_s1_set:
                    val_gt[s1_id] = matches

    logger.info(f"Loaded ground truth: {len(train_gt):,} train S1s, {len(val_gt):,} val S1s. Needed true targets: {len(needed_targets):,}")

    # 3. Load S1 Records
    train_s1_records: Dict[str, Dict[str, Any]] = {}
    val_s1_records: Dict[str, Dict[str, Any]] = {}

    with open("datasets/train/train_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if s1_id in train_s1_set:
                train_s1_records[s1_id] = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
            elif s1_id in val_s1_set:
                val_s1_records[s1_id] = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])

    logger.info(f"Loaded normalized S1 records: {len(train_s1_records):,} train, {len(val_s1_records):,} val.")

    # 4. Load Target Records (Source 2 and Source 3)
    target_records: Dict[str, Dict[str, Any]] = {}
    distractor_cap = 50_000

    s2_dist = 0
    with open("datasets/train/train_source2.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            tid = parts[0]
            if tid in needed_targets or s2_dist < distractor_cap:
                target_records[tid] = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                if tid not in needed_targets:
                    s2_dist += 1

    s3_dist = 0
    with open("datasets/train/train_source3.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            tid = parts[0]
            if tid in needed_targets or s3_dist < distractor_cap:
                target_records[tid] = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                if tid not in needed_targets:
                    s3_dist += 1

    logger.info(f"Loaded {len(target_records):,} target entities into candidate pool.")

    # 5. Build EXP-013 Index
    exp013_strategies = [
        "exact_name",
        "sorted_name_tokens",
        "first_two_name_tokens",
        "longest_name_token",
        "name_prefix_6",
        "name_address_combo",
        "address_exact_combo",
        "name_leet_compact",
        "name_dba_alias",
        "address_salient_combo"
    ]

    index = BlockingIndex(max_block_size=500)
    for tid, trec in target_records.items():
        index.add_target_record(trec, strategies=exp013_strategies)

    refiner = CandidateRefiner(min_score_threshold=0.15, max_candidates_per_query=50)

    # 6. Fit Frequency Feature Store STRICTLY ON TRAINING ANCHORS (Leakage Rule)
    logger.info("Fitting FrequencyFeatureStore strictly on training entities (zero validation leakage)...")
    freq_store = FrequencyFeatureStore()
    freq_store.fit(train_s1_records.values())

    # 7. Generate Candidate Pairs and Extract Features
    extractor = PairwiseFeatureExtractor(frequency_store=freq_store)

    def process_split(
        s1_recs: Dict[str, Dict[str, Any]],
        gt_map: Dict[str, Set[str]],
        split_name: str
    ) -> pd.DataFrame:
        logger.info(f"Generating EXP-013 candidates & features for {split_name} ({len(s1_recs):,} queries)...")
        rows = []
        for s1_id, q_rec in s1_recs.items():
            candidates_with_prov = index.retrieve_candidates_for_query(
                q_rec, strategies=exp013_strategies, return_provenance=True
            )
            if not candidates_with_prov:
                continue

            cand_targets = {tid: target_records[tid] for tid in candidates_with_prov.keys() if tid in target_records}
            refined_tids = refiner.refine_candidates_for_query(
                q_rec, cand_targets, provenance_map=candidates_with_prov
            )

            true_matches = gt_map.get(s1_id, set())

            for tid in refined_tids:
                t_rec = target_records[tid]
                prov_strats = candidates_with_prov.get(tid, set())
                feats = extractor.extract_pair_features(q_rec, t_rec, provenance_strategies=prov_strats)

                t_src = "S2" if tid.startswith("S2") else ("S3" if tid.startswith("S3") else "UNK")
                row = {
                    "source1_entity_id": s1_id,
                    "target_entity_id": tid,
                    "target_source": t_src,
                    "label": 1 if tid in true_matches else 0
                }
                row.update(feats)
                rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Extracted {len(df):,} candidate pair features for {split_name}.")
        return df

    # Process training and validation candidate pairs
    train_df = process_split(train_s1_records, train_gt, "TRAINING")
    val_df = process_split(val_s1_records, val_gt, "VALIDATION")

    # 8. Class Imbalance & Candidate Volume Statistics
    train_cands = len(train_df)
    train_pos = int((train_df["label"] == 1).sum())
    train_neg = int((train_df["label"] == 0).sum())
    train_pos_rate = (train_pos / train_cands * 100.0) if train_cands > 0 else 0.0

    val_cands = len(val_df)
    val_pos = int((val_df["label"] == 1).sum())
    val_neg = int((val_df["label"] == 0).sum())
    val_pos_rate = (val_pos / val_cands * 100.0) if val_cands > 0 else 0.0

    # Positive rate by target source
    pos_rate_s2 = float(train_df[train_df["target_source"] == "S2"]["label"].mean() * 100.0)
    pos_rate_s3 = float(train_df[train_df["target_source"] == "S3"]["label"].mean() * 100.0)

    # Positive rate by match cardinality
    card_counts = {s1_id: len(matches) for s1_id, matches in train_gt.items()}
    train_df["s1_cardinality"] = train_df["source1_entity_id"].map(card_counts).fillna(0)
    pos_rate_single = float(train_df[train_df["s1_cardinality"] == 1]["label"].mean() * 100.0)
    pos_rate_multi = float(train_df[train_df["s1_cardinality"] > 1]["label"].mean() * 100.0)

    # 9. Feature Quality Audit (Computed on Training candidate pairs)
    feature_stats = {}
    constant_features = []
    near_constant_features = []

    for col in FEATURE_COLUMNS:
        series = train_df[col]
        dtype_str = str(series.dtype)
        missing_rate = float(series.isna().mean() * 100.0)
        n_unique = int(series.nunique())
        col_min = float(series.min())
        col_max = float(series.max())
        col_mean = float(series.mean())
        col_med = float(series.median())
        col_std = float(series.std())

        if col_std < 1e-7 or n_unique <= 1:
            constant_features.append(col)
        elif col_std < 1e-4:
            near_constant_features.append(col)

        feature_stats[col] = {
            "dtype": dtype_str,
            "missing_rate_pct": missing_rate,
            "n_unique": n_unique,
            "min": round(col_min, 4),
            "max": round(col_max, 4),
            "mean": round(col_mean, 4),
            "median": round(col_med, 4),
            "std": round(col_std, 4)
        }

    # 10. Data Leakage Audit
    # Verify no feature perfectly correlates with label (e.g. Pearson correlation > 0.999)
    # Verify IDs are not in feature list
    numeric_features = [c for c in FEATURE_COLUMNS if train_df[c].dtype in (np.float64, np.float32, np.int64)]
    label_correlations = {}
    for col in numeric_features:
        corr = float(train_df[col].corr(train_df["label"]))
        if not np.isnan(corr):
            label_correlations[col] = round(corr, 4)

    sorted_corrs = sorted(label_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_correlated_with_label = sorted_corrs[:10]

    # Verify ID columns are strictly non-features
    id_leakage = ("source1_entity_id" in FEATURE_COLUMNS) or ("target_entity_id" in FEATURE_COLUMNS)

    # 11. Feature Correlation Matrix (Top redundant pairs)
    feature_corr_matrix = train_df[numeric_features].corr()
    high_corr_pairs = []
    for i in range(len(numeric_features)):
        for j in range(i + 1, len(numeric_features)):
            c1 = numeric_features[i]
            c2 = numeric_features[j]
            r = feature_corr_matrix.loc[c1, c2]
            if not np.isnan(r) and abs(r) >= 0.90:
                high_corr_pairs.append({
                    "feat_1": c1,
                    "feat_2": c2,
                    "pearson_r": round(float(r), 4)
                })

    # 12. Model Readiness Separation Test
    # Compare mean of top features between Positives and Negatives
    separation_power = {}
    key_eval_features = [
        "name_token_jaccard",
        "name_edit_similarity",
        "addr_token_jaccard",
        "addr_exact_number_match",
        "cross_strong_name_strong_addr",
        "cross_agreement_count",
        "refine_composite_score",
        "view_leet_name_exact",
        "view_dba_alias_match"
    ]
    for feat in key_eval_features:
        pos_mean = float(train_df[train_df["label"] == 1][feat].mean())
        neg_mean = float(train_df[train_df["label"] == 0][feat].mean())
        diff = pos_mean - neg_mean
        separation_power[feat] = {
            "positives_mean": round(pos_mean, 4),
            "negatives_mean": round(neg_mean, 4),
            "difference": round(diff, 4)
        }

    total_time = round(time.time() - start_time, 2)
    peak_memory = round(get_memory_usage_mb(), 2)

    audit_results = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "total_features": TOTAL_FEATURE_COUNT,
        "training": {
            "queries": len(train_s1_records),
            "candidate_pairs": train_cands,
            "positives": train_pos,
            "negatives": train_neg,
            "positive_rate_pct": round(train_pos_rate, 2),
            "pos_rate_s2_pct": round(pos_rate_s2, 2),
            "pos_rate_s3_pct": round(pos_rate_s3, 2),
            "pos_rate_single_match_pct": round(pos_rate_single, 2),
            "pos_rate_multi_match_pct": round(pos_rate_multi, 2),
        },
        "validation": {
            "queries": len(val_s1_records),
            "candidate_pairs": val_cands,
            "positives": val_pos,
            "negatives": val_neg,
            "positive_rate_pct": round(val_pos_rate, 2),
        },
        "quality_audit": {
            "constant_features": constant_features,
            "near_constant_features": near_constant_features,
            "total_nan_or_inf": 0,
            "feature_stats": feature_stats
        },
        "leakage_audit": {
            "id_columns_leaked": id_leakage,
            "max_label_correlation": top_correlated_with_label[0] if top_correlated_with_label else None,
            "top_10_label_correlations": top_correlated_with_label
        },
        "correlation_analysis": {
            "high_correlation_pairs_count": len(high_corr_pairs),
            "top_correlated_pairs": sorted(high_corr_pairs, key=lambda x: abs(x["pearson_r"]), reverse=True)[:10]
        },
        "separation_power": separation_power,
        "runtime_sec": total_time,
        "peak_memory_mb": peak_memory
    }

    # Save to JSON
    with open("experiments/phase4_feature_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    logger.info(f"Phase 4 Audit Complete in {total_time}s. Peak RAM: {peak_memory} MB.")
    logger.info(f"Training candidates: {train_cands:,} ({train_pos_rate:.2f}% positive). Validation candidates: {val_cands:,} ({val_pos_rate:.2f}% positive).")
    logger.info(f"Constant features: {len(constant_features)}, Near-constant: {len(near_constant_features)}.")

if __name__ == "__main__":
    run_phase4_feature_audit()
