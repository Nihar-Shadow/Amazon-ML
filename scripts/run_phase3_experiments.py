import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import csv
from typing import Dict, List, Set, Any
from src.candidate_generation.normalizer import normalize_entity_record
from src.candidate_generation.blocking import BlockingIndex
from src.candidate_generation.candidate_refiner import CandidateRefiner
from src.candidate_generation.diagnostics import compute_detailed_candidate_metrics, analyze_false_negatives
from src.utils.logging_utils import get_logger, log_step, get_memory_usage_mb

logger = get_logger("phase3_experiments")

def run_experiments():
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("eda", exist_ok=True)

    # 1. Load validation S1 IDs
    val_s1_ids_file = "eda/val_s1_ids.txt"
    if not os.path.exists(val_s1_ids_file):
        raise FileNotFoundError(f"Missing {val_s1_ids_file}. Run scripts/generate_validation_split.py first.")

    with open(val_s1_ids_file, "r", encoding="utf-8") as f:
        val_s1_ids_all = [line.strip() for line in f if line.strip()]

    # Use first 10,000 validation S1 entities for rapid, reproducible ablation benchmarking
    N_BENCHMARK = 10_000
    val_s1_benchmark_set = set(val_s1_ids_all[:N_BENCHMARK])
    logger.info(f"Loaded {len(val_s1_benchmark_set):,} benchmark validation S1 anchors.")

    # 2. Load ground truth for benchmark S1 entities
    gt_benchmark: Dict[str, Set[str]] = {}
    needed_targets: Set[str] = set()

    with open("datasets/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline() # header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if s1_id in val_s1_benchmark_set:
                matches = set()
                if len(parts) >= 2 and parts[1]:
                    for m in parts[1].split(","):
                        m = m.strip()
                        if m:
                            matches.add(m)
                            needed_targets.add(m)
                gt_benchmark[s1_id] = matches

    logger.info(
        f"Benchmark GT loaded: {len(gt_benchmark):,} S1s, "
        f"{sum(len(v) for v in gt_benchmark.values()):,} true links, "
        f"{len(needed_targets):,} unique target entities."
    )

    # 3. Load S1 benchmark records
    s1_records: Dict[str, Dict[str, Any]] = {}
    s1_countries: Dict[str, str] = {}
    with open("datasets/train/train_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if s1_id in val_s1_benchmark_set:
                rec = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                s1_records[s1_id] = rec
                s1_countries[s1_id] = rec["country"]

    # 4. Load target records: all needed true targets + 100,000 realistic distractors
    target_records: Dict[str, Dict[str, Any]] = {}
    distractor_cap_per_source = 60_000
    
    # Read from train_source2
    s2_distractor_count = 0
    with open("datasets/train/train_source2.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            tid = parts[0]
            if tid in needed_targets or s2_distractor_count < distractor_cap_per_source:
                rec = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                target_records[tid] = rec
                if tid not in needed_targets:
                    s2_distractor_count += 1

    # Read from train_source3
    s3_distractor_count = 0
    with open("datasets/train/train_source3.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            tid = parts[0]
            if tid in needed_targets or s3_distractor_count < distractor_cap_per_source:
                rec = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                target_records[tid] = rec
                if tid not in needed_targets:
                    s3_distractor_count += 1

    logger.info(f"Target candidate pool initialized: {len(target_records):,} total target records.")

    # 5. Define Ablation Configurations
    experiments = [
        {
            "id": "EXP-001",
            "name": "Experiment A: Exact Normalized Name",
            "strategies": ["exact_name"],
            "refine": False,
            "min_score": 0.0,
            "max_cands": 100
        },
        {
            "id": "EXP-002",
            "name": "Experiment B: Name + Address Blocking",
            "strategies": ["exact_name", "name_address_combo", "address_exact_combo"],
            "refine": False,
            "min_score": 0.0,
            "max_cands": 100
        },
        {
            "id": "EXP-003",
            "name": "Experiment C: Multi-Block Union (Broad)",
            "strategies": [
                "exact_name",
                "sorted_name_tokens",
                "first_two_name_tokens",
                "longest_name_token",
                "name_prefix_6",
                "name_address_combo",
                "address_exact_combo"
            ],
            "refine": False,
            "min_score": 0.0,
            "max_cands": 100
        },
        {
            "id": "EXP-004",
            "name": "Experiment D: Multi-Block Union + Candidate Refinement (Threshold 0.15)",
            "strategies": [
                "exact_name",
                "sorted_name_tokens",
                "first_two_name_tokens",
                "longest_name_token",
                "name_prefix_6",
                "name_address_combo",
                "address_exact_combo"
            ],
            "refine": True,
            "min_score": 0.15,
            "max_cands": 50
        },
        {
            "id": "EXP-005",
            "name": "Experiment E: Multi-Block Union + Stringent Refinement (Threshold 0.25)",
            "strategies": [
                "exact_name",
                "sorted_name_tokens",
                "first_two_name_tokens",
                "longest_name_token",
                "name_prefix_6",
                "name_address_combo",
                "address_exact_combo"
            ],
            "refine": True,
            "min_score": 0.25,
            "max_cands": 30
        },
    ]

    results_summary = []
    failure_reports = {}

    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        strats = exp["strategies"]
        do_refine = exp["refine"]
        min_sc = exp["min_score"]
        max_k = exp["max_cands"]
        
        logger.info(f"--- Running {exp_id}: {exp_name} ---")
        t0 = time.time()
        start_mem = get_memory_usage_mb()

        # Build index for active strategies
        index = BlockingIndex(max_block_size=500)
        for tid, trec in target_records.items():
            index.add_target_record(trec, strategies=strats)

        refiner = CandidateRefiner(
            min_score_threshold=min_sc,
            max_candidates_per_query=max_k,
            preserve_provenance_bypass=True
        )

        candidates: Dict[str, Set[str]] = {}

        for s1_id, q_rec in s1_records.items():
            cands_with_prov = index.retrieve_candidates_for_query(
                q_rec, strategies=strats, return_provenance=True
            )
            
            if do_refine and cands_with_prov:
                cand_trecs = {tid: target_records[tid] for tid in cands_with_prov if tid in target_records}
                filtered = refiner.refine_candidates_for_query(
                    q_rec, cand_trecs, provenance_map=cands_with_prov
                )
                candidates[s1_id] = filtered
            else:
                candidates[s1_id] = set(cands_with_prov.keys())

        elapsed = time.time() - t0
        peak_mem = get_memory_usage_mb()

        # Compute diagnostics
        diag = compute_detailed_candidate_metrics(
            ground_truth=gt_benchmark,
            candidates=candidates,
            s1_countries=s1_countries,
            total_target_records=len(target_records)
        )

        # Analyze false negatives
        fn_analysis = analyze_false_negatives(
            ground_truth=gt_benchmark,
            candidates=candidates,
            s1_records=s1_records,
            target_records=target_records,
            max_samples=10
        )
        failure_reports[exp_id] = fn_analysis

        exp_result = {
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "strategies": strats,
            "refinement_enabled": do_refine,
            "min_score": min_sc,
            "max_candidates": max_k,
            "runtime_sec": round(elapsed, 2),
            "peak_memory_mb": round(peak_mem, 2),
            "candidate_recall_overall": round(diag["candidate_recall_overall"], 4),
            "candidate_recall_s2": round(diag["candidate_recall_s2"], 4),
            "candidate_recall_s3": round(diag["candidate_recall_s3"], 4),
            "s1_coverage": round(diag["s1_coverage"], 4),
            "candidate_mean": round(diag["candidate_count_mean"], 2),
            "candidate_median": round(diag["candidate_count_median"], 2),
            "candidate_p95": round(diag["candidate_count_p95"], 2),
            "candidate_p99": round(diag["candidate_count_p99"], 2),
            "pct_zero_candidates": round(diag["pct_zero_candidates"], 2),
            "pct_gt_100_candidates": round(diag["pct_gt_100_candidates"], 2),
            "reduction_ratio": round(diag["reduction_ratio"], 1),
            "reduction_percentage": round(diag["reduction_percentage"], 4)
        }
        results_summary.append(exp_result)

        logger.info(
            f"Finished {exp_id} in {elapsed:.2f}s | "
            f"Recall: {exp_result['candidate_recall_overall']*100:.2f}% | "
            f"Mean Cands: {exp_result['candidate_mean']} | "
            f"P95: {exp_result['candidate_p95']} | "
            f"Reduction: {exp_result['reduction_percentage']:.2f}%"
        )

    # Save results to JSON and CSV
    with open("experiments/phase3_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump({"experiments": results_summary, "failures": failure_reports}, f, indent=2)

    with open("experiments/phase3_ablation_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment_id", "experiment_name", "candidate_recall_overall",
            "candidate_recall_s2", "candidate_recall_s3", "s1_coverage",
            "candidate_mean", "candidate_median", "candidate_p95", "candidate_p99",
            "pct_zero_candidates", "reduction_ratio", "reduction_percentage",
            "runtime_sec", "peak_memory_mb"
        ])
        writer.writeheader()
        for r in results_summary:
            row = {k: r[k] for k in writer.fieldnames}
            writer.writerow(row)

    logger.info("All ablation experiments completed. Summary written to experiments/")

if __name__ == "__main__":
    run_experiments()
