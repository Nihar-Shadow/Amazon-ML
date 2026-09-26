import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import pickle
import csv
from typing import Dict, List, Set, Any
import numpy as np
import pandas as pd

from src.candidate_generation.normalizer import normalize_entity_record
from src.candidate_generation.blocking import BlockingIndex
from src.candidate_generation.candidate_refiner import CandidateRefiner
from src.matching_features.feature_schema import FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS, TOTAL_FEATURE_COUNT
from src.matching_features.feature_extractor import PairwiseFeatureExtractor
from src.matching_features.frequency_features import FrequencyFeatureStore
from src.matching_model.trainer import train_model, predict_probabilities
from src.matching_model.evaluation import evaluate_model_at_threshold, sweep_thresholds
from src.matching_model.feature_ablations import FEATURE_ABLATION_CONFIGS
from src.matching_model.error_analysis import analyze_model_errors
from src.matching_model.candidate_recall_analysis import compute_recall_ceilings
from src.utils.logging_utils import get_logger, get_memory_usage_mb

logger = get_logger("train_and_evaluate_phase5")

def get_or_extract_feature_data():
    os.makedirs("data_cache", exist_ok=True)
    train_cache = "data_cache/train_features.pkl"
    val_cache = "data_cache/val_features.pkl"
    gt_cache = "data_cache/ground_truth.pkl"

    if os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(gt_cache):
        logger.info("Loading cached feature datasets from data_cache/...")
        with open(train_cache, "rb") as f:
            train_df = pickle.load(f)
        with open(val_cache, "rb") as f:
            val_df = pickle.load(f)
        with open(gt_cache, "rb") as f:
            train_gt, val_gt = pickle.load(f)
        logger.info(f"Loaded train_df: {len(train_df):,}, val_df: {len(val_df):,}")
        return train_df, val_df, train_gt, val_gt

    logger.info("Cached datasets not found. Extracting candidate pairs and features...")
    # 1. Load S1 IDs
    with open("eda/train_s1_ids.txt", "r", encoding="utf-8") as f:
        train_s1_all = [line.strip() for line in f if line.strip()]
    with open("eda/val_s1_ids.txt", "r", encoding="utf-8") as f:
        val_s1_all = [line.strip() for line in f if line.strip()]

    N_TRAIN = 5_000
    N_VAL = 5_000
    train_s1_set = set(train_s1_all[:N_TRAIN])
    val_s1_set = set(val_s1_all[:N_VAL])
    all_needed_s1 = train_s1_set | val_s1_set

    # 2. Load Ground Truth
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

    # 4. Load Target Records
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

    # 5. EXP-013 Index
    exp013_strategies = [
        "exact_name", "sorted_name_tokens", "first_two_name_tokens", "longest_name_token",
        "name_prefix_6", "name_address_combo", "address_exact_combo", "name_leet_compact",
        "name_dba_alias", "address_salient_combo"
    ]
    index = BlockingIndex(max_block_size=500)
    for tid, trec in target_records.items():
        index.add_target_record(trec, strategies=exp013_strategies)

    refiner = CandidateRefiner(min_score_threshold=0.15, max_candidates_per_query=50)

    # 6. Fit Frequency store on train only
    freq_store = FrequencyFeatureStore()
    freq_store.fit(train_s1_records.values())

    extractor = PairwiseFeatureExtractor(frequency_store=freq_store)

    def extract_split(s1_recs, gt_map):
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
        return pd.DataFrame(rows)

    train_df = extract_split(train_s1_records, train_gt)
    val_df = extract_split(val_s1_records, val_gt)

    with open(train_cache, "wb") as f:
        pickle.dump(train_df, f)
    with open(val_cache, "wb") as f:
        pickle.dump(val_df, f)
    with open(gt_cache, "wb") as f:
        pickle.dump((train_gt, val_gt), f)

    logger.info("Saved feature datasets to data_cache/.")
    return train_df, val_df, train_gt, val_gt

def run_phase5_experiments():
    start_total_time = time.time()
    logger.info("Executing Phase 5 Model Training, Validation & Threshold Optimization.")

    train_df, val_df, train_gt, val_gt = get_or_extract_feature_data()
    os.makedirs("experiments", exist_ok=True)

    # All candidate thresholds to sweep
    thresholds = [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    ]

    model_candidates = [
        ("MODEL-001", "Logistic Regression (Standardized, L2)", "MODEL-001", None),
        ("MODEL-001-BAL", "Logistic Regression (Class-Weighted)", "MODEL-001", "balanced"),
        ("MODEL-002", "Random Forest (100 trees, max_depth=14)", "MODEL-002", None),
        ("MODEL-003", "HistGradientBoosting (LightGBM-Equivalent)", "MODEL-003", None),
        ("MODEL-003-BAL", "HistGradientBoosting (Class-Weighted)", "MODEL-003", "balanced"),
        ("MODEL-004", "ExtraTrees Classifier (100 trees, max_depth=14)", "MODEL-004", None),
    ]

    model_results = []
    trained_models = {}
    validation_probs = {}

    logger.info("--- 1. Training and Comparing Model Families ---")
    for m_id, m_name, base_type, cw in model_candidates:
        logger.info(f"Training {m_id}: {m_name}...")
        model, tr_time = train_model(base_type, train_df, class_weight=cw)
        probs, inf_time = predict_probabilities(model, val_df)

        trained_models[m_id] = model
        validation_probs[m_id] = probs

        # Threshold sweep for this model
        best_t_res, all_t_res = sweep_thresholds(val_df, probs, val_gt, thresholds=thresholds)

        res_entry = {
            "model_id": m_id,
            "model_name": m_name,
            "threshold": best_t_res["threshold"],
            "macro_f05": best_t_res["macro_f05"],
            "macro_precision": best_t_res["macro_precision"],
            "macro_recall": best_t_res["macro_recall"],
            "pairwise_precision": best_t_res["pairwise_precision"],
            "pairwise_recall": best_t_res["pairwise_recall"],
            "roc_auc": best_t_res["roc_auc"],
            "pr_auc": best_t_res["pr_auc"],
            "singleton_f05": best_t_res["singleton_f05"],
            "multi_match_f05": best_t_res["multi_match_f05"],
            "zero_match_accuracy": best_t_res["zero_match_accuracy"],
            "predicted_match_count": best_t_res["predicted_match_count"],
            "train_runtime_sec": tr_time,
            "val_runtime_sec": inf_time,
            "all_threshold_results": all_t_res
        }
        model_results.append(res_entry)
        logger.info(f"{m_id} [{m_name}] -> Best Threshold: {best_t_res['threshold']} | Macro F0.5: {best_t_res['macro_f05']:.4f} | PR-AUC: {best_t_res['pr_auc']:.4f}")

    # Sort models by Macro F0.5
    model_results.sort(key=lambda x: x["macro_f05"], reverse=True)
    best_model_entry = model_results[0]
    best_model_id = best_model_entry["model_id"]
    best_probs = validation_probs[best_model_id]
    best_threshold = best_model_entry["threshold"]

    logger.info(f"Best Model Selected: {best_model_id} ({best_model_entry['model_name']}) with Macro F0.5 = {best_model_entry['macro_f05']:.4f} at threshold = {best_threshold}")

    # --- 2. Controlled Feature Ablation Study on Best Model Family ---
    logger.info("--- 2. Controlled Feature Ablation Experiments ---")
    ablation_results = []
    
    for exp_id, cfg in FEATURE_ABLATION_CONFIGS.items():
        logger.info(f"Running Ablation {exp_id}: {cfg['name']} ({cfg['count']} features)...")
        m, tr_t = train_model("MODEL-003", train_df, feature_cols=cfg["columns"])
        p, inf_t = predict_probabilities(m, val_df, feature_cols=cfg["columns"])
        best_t, _ = sweep_thresholds(val_df, p, val_gt, thresholds=thresholds)

        ablation_entry = {
            "experiment_id": exp_id,
            "experiment_name": cfg["name"],
            "feature_count": cfg["count"],
            "optimal_threshold": best_t["threshold"],
            "macro_f05": best_t["macro_f05"],
            "macro_precision": best_t["macro_precision"],
            "macro_recall": best_t["macro_recall"],
            "roc_auc": best_t["roc_auc"],
            "pr_auc": best_t["pr_auc"],
            "train_runtime_sec": tr_t,
            "val_runtime_sec": inf_t
        }
        ablation_results.append(ablation_entry)
        logger.info(f"{exp_id}: {cfg['name']} -> Macro F0.5: {best_t['macro_f05']:.4f} | PR-AUC: {best_t['pr_auc']:.4f}")

    # --- 3. Forensic Error Analysis on Best Model ---
    logger.info("--- 3. Error Analysis on Best Model ---")
    error_summary = analyze_model_errors(val_df, best_probs, val_gt, threshold=best_threshold)

    # --- 4. Candidate Recall Ceiling & Attribution ---
    logger.info("--- 4. Candidate Recall Ceiling vs Model Discrimination Recall ---")
    recall_ceilings = compute_recall_ceilings(val_df, best_probs, val_gt, threshold=best_threshold)

    total_proc_time = round(time.time() - start_total_time, 2)
    peak_ram = round(get_memory_usage_mb(), 2)

    # --- 5. Export JSON and Summary CSV ---
    final_output = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "selected_model": best_model_entry,
        "all_models_comparison": model_results,
        "feature_ablations": ablation_results,
        "error_analysis": error_summary,
        "recall_ceilings": recall_ceilings,
        "total_runtime_sec": total_proc_time,
        "peak_memory_mb": peak_ram
    }

    with open("experiments/phase5_model_results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    # Write summary CSV
    with open("experiments/phase5_model_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Threshold", "Macro F0.5", "Macro Precision", "Macro Recall",
            "Pairwise Precision", "Pairwise Recall", "ROC-AUC", "PR-AUC",
            "Singleton F0.5", "Multi-match F0.5", "Zero-match Accuracy",
            "Predicted Matches", "Train Runtime (s)", "Memory (MB)"
        ])
        for m in model_results:
            writer.writerow([
                m["model_name"], m["threshold"], m["macro_f05"], m["macro_precision"],
                m["macro_recall"], m["pairwise_precision"], m["pairwise_recall"],
                m["roc_auc"], m["pr_auc"], m["singleton_f05"], m["multi_match_f05"],
                m["zero_match_accuracy"], m["predicted_match_count"],
                m["train_runtime_sec"], peak_ram
            ])

    logger.info("Phase 5 Model Training & Evaluation Complete.")
    logger.info(f"Results saved to experiments/phase5_model_results.json and phase5_model_summary.csv")

if __name__ == "__main__":
    run_phase5_experiments()
