from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
from src.utils.logging_utils import get_logger

logger = get_logger("candidate_diagnostics")

def compute_detailed_candidate_metrics(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    s1_countries: Optional[Dict[str, str]] = None,
    total_target_records: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive candidate generation metrics:
    - Candidate recall (overall, S2, S3, by country, singletons, multi-match, match-count buckets)
    - Candidate count distribution (mean, median, min, max, P90, P95, P99)
    - Cardinality flags (0, >100, >1,000, >10,000)
    - Reduction ratio vs theoretical Cartesian search space
    """
    total_s1 = len(ground_truth)
    if total_s1 == 0:
        return {}

    # 1. Candidate count distribution
    all_cand_counts = []
    s2_cand_counts = []
    s3_cand_counts = []
    
    total_cand_pairs = 0
    zero_cand_s1 = 0
    gt_100_s1 = 0
    gt_1000_s1 = 0
    gt_10000_s1 = 0

    for s1_id in ground_truth.keys():
        cand_set = candidates.get(s1_id, set())
        n = len(cand_set)
        all_cand_counts.append(n)
        total_cand_pairs += n
        
        n_s2 = sum(1 for tid in cand_set if tid.startswith("S2-"))
        n_s3 = sum(1 for tid in cand_set if tid.startswith("S3-"))
        s2_cand_counts.append(n_s2)
        s3_cand_counts.append(n_s3)
        
        if n == 0:
            zero_cand_s1 += 1
        if n > 100:
            gt_100_s1 += 1
        if n > 1000:
            gt_1000_s1 += 1
        if n > 10000:
            gt_10000_s1 += 1

    arr_all = np.array(all_cand_counts, dtype=np.int32)
    arr_s2 = np.array(s2_cand_counts, dtype=np.int32)
    arr_s3 = np.array(s3_cand_counts, dtype=np.int32)

    # 2. Candidate Recall Breakdown
    total_true_links = 0
    retrieved_true_links = 0
    
    s2_true_links = 0
    s2_retrieved_links = 0
    s3_true_links = 0
    s3_retrieved_links = 0
    
    # By country
    country_true: Dict[str, int] = defaultdict(int)
    country_retrieved: Dict[str, int] = defaultdict(int)
    
    # Singletons vs Multi-match
    singleton_s1_count = 0
    singleton_clean_zero_cands = 0
    
    multi_match_s1_count = 0
    multi_match_true_links = 0
    multi_match_retrieved_links = 0
    
    # By match count bucket (1, 2, 3, 4, 5+)
    bucket_true: Dict[str, int] = defaultdict(int)
    bucket_retrieved: Dict[str, int] = defaultdict(int)

    # Fully covered S1s
    non_singleton_s1_count = 0
    fully_covered_s1_count = 0

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        n_true = len(true_set)
        country = s1_countries.get(s1_id, "UNKNOWN") if s1_countries else "UNKNOWN"
        
        if n_true == 0:
            singleton_s1_count += 1
            if len(cand_set) == 0:
                singleton_clean_zero_cands += 1
        else:
            non_singleton_s1_count += 1
            total_true_links += n_true
            country_true[country] += n_true
            
            captured = true_set & cand_set
            n_cap = len(captured)
            retrieved_true_links += n_cap
            country_retrieved[country] += n_cap
            
            if n_cap == n_true:
                fully_covered_s1_count += 1
                
            # S2 / S3 breakdown
            for tid in true_set:
                if tid.startswith("S2-"):
                    s2_true_links += 1
                    if tid in cand_set:
                        s2_retrieved_links += 1
                elif tid.startswith("S3-"):
                    s3_true_links += 1
                    if tid in cand_set:
                        s3_retrieved_links += 1
                        
            # Multi-match
            if n_true > 1:
                multi_match_s1_count += 1
                multi_match_true_links += n_true
                multi_match_retrieved_links += n_cap
                
            # Bucket
            bucket = "5+" if n_true >= 5 else str(n_true)
            bucket_true[bucket] += n_true
            bucket_retrieved[bucket] += n_cap

    # Compute rates
    overall_recall = (retrieved_true_links / total_true_links) if total_true_links > 0 else 1.0
    s2_recall = (s2_retrieved_links / s2_true_links) if s2_true_links > 0 else 1.0
    s3_recall = (s3_retrieved_links / s3_true_links) if s3_true_links > 0 else 1.0
    s1_coverage = (fully_covered_s1_count / non_singleton_s1_count) if non_singleton_s1_count > 0 else 1.0
    
    country_recall = {
        c: (country_retrieved[c] / country_true[c]) if country_true[c] > 0 else 1.0
        for c in country_true
    }
    
    bucket_recall = {
        b: (bucket_retrieved[b] / bucket_true[b]) if bucket_true[b] > 0 else 1.0
        for b in bucket_true
    }
    
    multi_recall = (multi_match_retrieved_links / multi_match_true_links) if multi_match_true_links > 0 else 1.0
    singleton_zero_rate = (singleton_clean_zero_cands / singleton_s1_count) if singleton_s1_count > 0 else 1.0

    # 3. Reduction Ratio
    # Default target pool size if not provided: ~10,320,219 for train
    target_pool_size = total_target_records if total_target_records is not None else 10_320_219
    theoretical_pairs = total_s1 * target_pool_size
    reduction_ratio = (theoretical_pairs / total_cand_pairs) if total_cand_pairs > 0 else float("inf")
    reduction_pct = (1.0 - (total_cand_pairs / theoretical_pairs)) * 100.0 if theoretical_pairs > 0 else 100.0

    return {
        "candidate_recall_overall": overall_recall,
        "candidate_recall_s2": s2_recall,
        "candidate_recall_s3": s3_recall,
        "s1_coverage": s1_coverage,
        "candidate_recall_by_country": country_recall,
        "candidate_recall_multi_match": multi_recall,
        "candidate_recall_by_bucket": bucket_recall,
        "singleton_s1_count": singleton_s1_count,
        "singleton_pct_clean_zero_cands": singleton_zero_rate * 100.0,
        # Count distributions (combined)
        "candidate_count_mean": float(np.mean(arr_all)),
        "candidate_count_median": float(np.median(arr_all)),
        "candidate_count_min": int(np.min(arr_all)),
        "candidate_count_max": int(np.max(arr_all)),
        "candidate_count_p90": float(np.percentile(arr_all, 90)),
        "candidate_count_p95": float(np.percentile(arr_all, 95)),
        "candidate_count_p99": float(np.percentile(arr_all, 99)),
        # S2 / S3 count means
        "s2_candidate_mean": float(np.mean(arr_s2)),
        "s3_candidate_mean": float(np.mean(arr_s3)),
        # S1 threshold percentages
        "pct_zero_candidates": (zero_cand_s1 / total_s1) * 100.0,
        "pct_gt_100_candidates": (gt_100_s1 / total_s1) * 100.0,
        "pct_gt_1000_candidates": (gt_1000_s1 / total_s1) * 100.0,
        "pct_gt_10000_candidates": (gt_10000_s1 / total_s1) * 100.0,
        # Reduction ratio
        "total_theoretical_pairs": theoretical_pairs,
        "total_final_candidate_pairs": total_cand_pairs,
        "reduction_ratio": reduction_ratio,
        "reduction_percentage": reduction_pct
    }

def analyze_false_negatives(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    s1_records: Dict[str, Dict[str, Any]],
    target_records: Dict[str, Dict[str, Any]],
    max_samples: int = 15
) -> List[Dict[str, Any]]:
    """
    Identifies and categorizes false negative candidate cases
    (true target matches absent from candidate set).
    """
    failures = []
    
    for s1_id, true_set in ground_truth.items():
        if not true_set:
            continue
        cand_set = candidates.get(s1_id, set())
        missed = true_set - cand_set
        
        if missed:
            s1_rec = s1_records.get(s1_id, {})
            for tid in missed:
                t_rec = target_records.get(tid, {})
                
                # Analyze failure reason
                n1 = s1_rec.get("name", {}).get("raw", "")
                n2 = t_rec.get("name", {}).get("raw", "")
                a1 = s1_rec.get("address", {}).get("raw", "")
                a2 = t_rec.get("address", {}).get("raw", "")
                c1 = s1_rec.get("country", "")
                c2 = t_rec.get("country", "")
                
                reason = "other"
                if c1 and c2 and c1 != c2:
                    reason = "country_mismatch"
                elif not a2 or a2.strip() == "":
                    reason = "missing_target_address"
                elif any(ord(c) > 127 for c in n2) and not any(ord(c) > 127 for c in n1):
                    reason = "transliteration_script_difference"
                elif ".com" in n2.lower() and ".com" not in n1.lower():
                    reason = "url_domain_representation"
                elif len(n1) > 0 and len(n2) > 0 and abs(len(n1) - len(n2)) <= 2:
                    reason = "character_typo_spelling_variation"
                else:
                    reason = "token_vocabulary_difference"
                    
                failures.append({
                    "s1_id": s1_id,
                    "target_id": tid,
                    "s1_name": n1,
                    "target_name": n2,
                    "s1_address": a1,
                    "target_address": a2,
                    "country": c1,
                    "category": reason
                })
                
                if len(failures) >= max_samples:
                    return failures
                    
    return failures
