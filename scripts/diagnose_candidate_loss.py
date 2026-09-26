import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import csv
from collections import defaultdict
from typing import Dict, List, Set, Any
from src.candidate_generation.normalizer import normalize_entity_record
from src.candidate_generation.blocking import BlockingIndex
from src.candidate_generation.candidate_refiner import CandidateRefiner, score_candidate_pair
from src.candidate_generation.diagnostics import compute_detailed_candidate_metrics
from src.utils.logging_utils import get_logger, get_memory_usage_mb

logger = get_logger("diagnostics_runner")

def diagnose():
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

    # 5. Build Index (EXP-004 baseline strategies)
    baseline_strategies = [
        "exact_name",
        "sorted_name_tokens",
        "first_two_name_tokens",
        "longest_name_token",
        "name_prefix_6",
        "name_address_combo",
        "address_exact_combo"
    ]
    index = BlockingIndex(max_block_size=500)
    for tid, trec in target_records.items():
        index.add_target_record(trec, strategies=baseline_strategies)

    # ==========================================================
    # PART A — K SENSITIVITY EXPERIMENTS (K=50, 75, 100, 150)
    # ==========================================================
    k_experiments = [
        {"id": "EXP-006", "k": 50},
        {"id": "EXP-007", "k": 75},
        {"id": "EXP-008", "k": 100},
        {"id": "EXP-009", "k": 150},
    ]

    k_results = []
    # Precompute broad candidates and scored candidate lists
    raw_candidates_with_prov: Dict[str, Dict[str, Set[str]]] = {}
    scored_candidates_by_s1: Dict[str, List[tuple]] = {}

    t0_retrieval = time.time()
    for s1_id, q_rec in s1_records.items():
        cands_with_prov = index.retrieve_candidates_for_query(
            q_rec, strategies=baseline_strategies, return_provenance=True
        )
        raw_candidates_with_prov[s1_id] = cands_with_prov
        
        # Score candidates
        scored_list = []
        for tid, strats in cands_with_prov.items():
            if tid not in target_records:
                continue
            trec = target_records[tid]
            score = score_candidate_pair(q_rec, trec, provenance_strategies=strats)
            if score < 0.0:
                continue
            if "exact_name" in strats or "sorted_name_tokens" in strats:
                score = max(score, 1.0)
            if score >= 0.15:
                scored_list.append((tid, score))
                
        scored_list.sort(key=lambda x: x[1], reverse=True)
        scored_candidates_by_s1[s1_id] = scored_list

    for exp in k_experiments:
        exp_id = exp["id"]
        k_val = exp["k"]
        t0 = time.time()
        
        cands_k: Dict[str, Set[str]] = {}
        for s1_id in s1_records.keys():
            scored_list = scored_candidates_by_s1[s1_id]
            cands_k[s1_id] = {tid for tid, _ in scored_list[:k_val]}

        elapsed = time.time() - t0
        diag = compute_detailed_candidate_metrics(
            ground_truth=gt_benchmark,
            candidates=cands_k,
            s1_countries=s1_countries,
            total_target_records=len(target_records)
        )
        
        res = {
            "experiment_id": exp_id,
            "k": k_val,
            "candidate_recall_overall": round(diag["candidate_recall_overall"], 4),
            "candidate_recall_s2": round(diag["candidate_recall_s2"], 4),
            "candidate_recall_s3": round(diag["candidate_recall_s3"], 4),
            "s1_coverage": round(diag["s1_coverage"], 4),
            "multi_match_recall": round(diag["candidate_recall_multi_match"], 4),
            "candidate_mean": round(diag["candidate_count_mean"], 2),
            "candidate_median": round(diag["candidate_count_median"], 2),
            "candidate_p90": round(diag["candidate_count_p90"], 2),
            "candidate_p95": round(diag["candidate_count_p95"], 2),
            "candidate_p99": round(diag["candidate_count_p99"], 2),
            "candidate_max": diag["candidate_count_max"],
            "pct_zero_candidates": round(diag["pct_zero_candidates"], 2),
            "reduction_ratio": round(diag["reduction_ratio"], 1),
            "reduction_percentage": round(diag["reduction_percentage"], 4),
            "runtime_sec": round(elapsed, 2),
            "peak_memory_mb": round(get_memory_usage_mb(), 2)
        }
        k_results.append(res)
        logger.info(
            f"K={k_val} ({exp_id}): Recall={res['candidate_recall_overall']*100:.2f}%, "
            f"Mean Cands={res['candidate_mean']}, P95={res['candidate_p95']}"
        )

    # Save K sensitivity table
    with open("experiments/k_sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(k_results, f, indent=2)

    # ==========================================================
    # PART B — CANDIDATE LOSS ATTRIBUTION (for K=50 baseline)
    # ==========================================================
    # For every validation true match missing from FINAL candidates (K=50):
    k50_candidates = {s1: {tid for tid, _ in scored_candidates_by_s1[s1][:50]} for s1 in s1_records}
    
    total_true_links = 0
    total_missing_links = 0
    
    loss_categories = {
        "BLOCKING_MISS": 0,
        "REFINEMENT_MISS": 0,
        "TOP_K_MISS": 0,
        "COUNTRY_FILTER_MISS": 0,
        "OTHER": 0
    }
    
    loss_examples = defaultdict(list)

    for s1_id, true_set in gt_benchmark.items():
        if not true_set:
            continue
        total_true_links += len(true_set)
        final_cands = k50_candidates.get(s1_id, set())
        broad_cands_prov = raw_candidates_with_prov.get(s1_id, {})
        
        q_rec = s1_records[s1_id]
        
        for tid in true_set:
            if tid in final_cands:
                continue
                
            total_missing_links += 1
            trec = target_records.get(tid)
            if not trec:
                loss_categories["OTHER"] += 1
                continue
                
            # Case 1: Was it retrieved by broad blocking?
            if tid not in broad_cands_prov:
                loss_categories["BLOCKING_MISS"] += 1
                if len(loss_examples["BLOCKING_MISS"]) < 10:
                    loss_examples["BLOCKING_MISS"].append({
                        "s1_id": s1_id,
                        "target_id": tid,
                        "s1_name": q_rec["name"]["raw"],
                        "target_name": trec["name"]["raw"],
                        "s1_address": q_rec["address"]["raw"],
                        "target_address": trec["address"]["raw"],
                        "s1_country": q_rec["country"],
                        "target_country": trec["country"]
                    })
            else:
                # Target was retrieved by broad blocking!
                strats = broad_cands_prov[tid]
                score = score_candidate_pair(q_rec, trec, provenance_strategies=strats)
                
                if score < 0.0:
                    loss_categories["COUNTRY_FILTER_MISS"] += 1
                    if len(loss_examples["COUNTRY_FILTER_MISS"]) < 10:
                        loss_examples["COUNTRY_FILTER_MISS"].append({
                            "s1_id": s1_id,
                            "target_id": tid,
                            "s1_country": q_rec["country"],
                            "target_country": trec["country"]
                        })
                elif score < 0.15:
                    loss_categories["REFINEMENT_MISS"] += 1
                    if len(loss_examples["REFINEMENT_MISS"]) < 10:
                        loss_examples["REFINEMENT_MISS"].append({
                            "s1_id": s1_id,
                            "target_id": tid,
                            "score": round(score, 4),
                            "strategies": list(strats),
                            "s1_name": q_rec["name"]["raw"],
                            "target_name": trec["name"]["raw"],
                            "s1_address": q_rec["address"]["raw"],
                            "target_address": trec["address"]["raw"]
                        })
                else:
                    # Score was >= 0.15, so it was dropped by K=50 ceiling!
                    loss_categories["TOP_K_MISS"] += 1
                    if len(loss_examples["TOP_K_MISS"]) < 10:
                        loss_examples["TOP_K_MISS"].append({
                            "s1_id": s1_id,
                            "target_id": tid,
                            "score": round(score, 4),
                            "rank_in_scored": [t for t, _ in scored_candidates_by_s1[s1_id]].index(tid) + 1,
                            "total_candidates_above_threshold": len(scored_candidates_by_s1[s1_id])
                        })

    attribution_report = {
        "total_true_links": total_true_links,
        "total_missing_links": total_missing_links,
        "missing_rate_pct": round((total_missing_links / total_true_links) * 100, 2),
        "loss_breakdown": {
            k: {
                "count": v,
                "percentage_of_missing": round((v / total_missing_links) * 100, 2) if total_missing_links > 0 else 0.0,
                "percentage_of_all_true_links": round((v / total_true_links) * 100, 2) if total_true_links > 0 else 0.0
            }
            for k, v in loss_categories.items()
        },
        "samples": loss_examples
    }

    with open("experiments/candidate_loss_attribution.json", "w", encoding="utf-8") as f:
        json.dump(attribution_report, f, indent=2)

    logger.info("=== CANDIDATE LOSS ATTRIBUTION SUMMARY ===")
    logger.info(f"Total True Links: {total_true_links:,}")
    logger.info(f"Total Missing Links: {total_missing_links:,} ({attribution_report['missing_rate_pct']}%)")
    for k, v in attribution_report["loss_breakdown"].items():
        logger.info(f"  {k}: {v['count']:,} ({v['percentage_of_missing']}%)")

if __name__ == "__main__":
    diagnose()
