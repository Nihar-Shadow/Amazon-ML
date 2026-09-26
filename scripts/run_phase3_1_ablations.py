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
from src.candidate_generation.candidate_refiner import CandidateRefiner, score_candidate_pair
from src.candidate_generation.diagnostics import compute_detailed_candidate_metrics, analyze_false_negatives
from src.utils.logging_utils import get_logger, get_memory_usage_mb

logger = get_logger("phase3_1_ablations")

def run_phase3_1():
    os.makedirs("experiments", exist_ok=True)

    # 1. Load benchmark validation S1 IDs
    with open("eda/val_s1_ids.txt", "r", encoding="utf-8") as f:
        val_s1_ids_all = [line.strip() for line in f if line.strip()]
        
    N_BENCHMARK = 10_000
    val_s1_benchmark_set = set(val_s1_ids_all[:N_BENCHMARK])

    # 2. Load ground truth
    gt_benchmark: Dict[str, Set[str]] = {}
    needed_targets: Set[str] = set()
    with open("datasets/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
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

    # 3. Load S1 records
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

    # 4. Load target records
    target_records: Dict[str, Dict[str, Any]] = {}
    distractor_cap = 60_000
    
    s2_distractors = 0
    with open("datasets/train/train_source2.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            tid = parts[0]
            if tid in needed_targets or s2_distractors < distractor_cap:
                rec = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                target_records[tid] = rec
                if tid not in needed_targets:
                    s2_distractors += 1

    s3_distractors = 0
    with open("datasets/train/train_source3.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            tid = parts[0]
            if tid in needed_targets or s3_distractors < distractor_cap:
                rec = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                target_records[tid] = rec
                if tid not in needed_targets:
                    s3_distractors += 1

    logger.info(f"Loaded {len(s1_records):,} S1s, {len(target_records):,} targets.")

    baseline_7_strats = [
        "exact_name",
        "sorted_name_tokens",
        "first_two_name_tokens",
        "longest_name_token",
        "name_prefix_6",
        "name_address_combo",
        "address_exact_combo"
    ]

    # Define Phase 3.1 Ablations
    ablations = [
        {
            "id": "EXP-004-VERIFY",
            "name": "EXP-004 Baseline (7 strategies, K=50)",
            "strategies": baseline_7_strats,
            "k": 50,
            "min_score": 0.15
        },
        {
            "id": "EXP-010",
            "name": "EXP-010: Baseline + Leetspeak Normalization View",
            "strategies": baseline_7_strats + ["name_leet_compact"],
            "k": 50,
            "min_score": 0.15
        },
        {
            "id": "EXP-011",
            "name": "EXP-011: Baseline + DBA/Trade-Name Alias Extraction",
            "strategies": baseline_7_strats + ["name_dba_alias"],
            "k": 50,
            "min_score": 0.15
        },
        {
            "id": "EXP-012",
            "name": "EXP-012: Baseline + Salient Address Locality Combo",
            "strategies": baseline_7_strats + ["address_salient_combo"],
            "k": 50,
            "min_score": 0.15
        },
        {
            "id": "EXP-013",
            "name": "EXP-013: Hardened Multi-Block Pipeline (All 10 Strategies, K=50)",
            "strategies": baseline_7_strats + ["name_leet_compact", "name_dba_alias", "address_salient_combo"],
            "k": 50,
            "min_score": 0.15
        }
    ]

    results = []

    for ab in ablations:
        exp_id = ab["id"]
        exp_name = ab["name"]
        strats = ab["strategies"]
        k_val = ab["k"]
        min_sc = ab["min_score"]
        
        logger.info(f"--- Running {exp_id}: {exp_name} ---")
        t0 = time.time()
        start_mem = get_memory_usage_mb()

        # Build index
        index = BlockingIndex(max_block_size=500)
        for tid, trec in target_records.items():
            index.add_target_record(trec, strategies=strats)

        refiner = CandidateRefiner(
            min_score_threshold=min_sc,
            max_candidates_per_query=k_val,
            preserve_provenance_bypass=True
        )

        candidates: Dict[str, Set[str]] = {}

        for s1_id, q_rec in s1_records.items():
            cands_with_prov = index.retrieve_candidates_for_query(
                q_rec, strategies=strats, return_provenance=True
            )
            
            if cands_with_prov:
                cand_trecs = {tid: target_records[tid] for tid in cands_with_prov if tid in target_records}
                filtered = refiner.refine_candidates_for_query(
                    q_rec, cand_trecs, provenance_map=cands_with_prov
                )
                candidates[s1_id] = filtered
            else:
                candidates[s1_id] = set()

        elapsed = time.time() - t0
        peak_mem = get_memory_usage_mb()

        diag = compute_detailed_candidate_metrics(
            ground_truth=gt_benchmark,
            candidates=candidates,
            s1_countries=s1_countries,
            total_target_records=len(target_records)
        )

        res = {
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "strategies_count": len(strats),
            "candidate_recall_overall": round(diag["candidate_recall_overall"], 4),
            "candidate_recall_s2": round(diag["candidate_recall_s2"], 4),
            "candidate_recall_s3": round(diag["candidate_recall_s3"], 4),
            "s1_coverage": round(diag["s1_coverage"], 4),
            "multi_match_recall": round(diag["candidate_recall_multi_match"], 4),
            "candidate_mean": round(diag["candidate_count_mean"], 2),
            "candidate_median": round(diag["candidate_count_median"], 2),
            "candidate_p95": round(diag["candidate_count_p95"], 2),
            "candidate_p99": round(diag["candidate_count_p99"], 2),
            "pct_zero_candidates": round(diag["pct_zero_candidates"], 2),
            "reduction_ratio": round(diag["reduction_ratio"], 1),
            "reduction_percentage": round(diag["reduction_percentage"], 4),
            "runtime_sec": round(elapsed, 2),
            "peak_memory_mb": round(peak_mem, 2)
        }
        results.append(res)
        logger.info(
            f"Finished {exp_id} in {elapsed:.2f}s | "
            f"Recall: {res['candidate_recall_overall']*100:.2f}% | "
            f"Coverage: {res['s1_coverage']*100:.2f}% | "
            f"Mean Cands: {res['candidate_mean']} | "
            f"P95: {res['candidate_p95']}"
        )

    # Save summary
    with open("experiments/phase3_1_hardening_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open("experiments/phase3_1_hardening_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment_id", "experiment_name", "strategies_count",
            "candidate_recall_overall", "candidate_recall_s2", "candidate_recall_s3",
            "s1_coverage", "candidate_mean", "candidate_median", "candidate_p95",
            "candidate_p99", "pct_zero_candidates", "reduction_ratio",
            "reduction_percentage", "runtime_sec", "peak_memory_mb"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    logger.info("Phase 3.1 hardening experiments complete.")

if __name__ == "__main__":
    run_phase3_1()
