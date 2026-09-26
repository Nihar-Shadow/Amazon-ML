import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

def compute_single_s1_metrics(
    true_set: Set[str],
    pred_set: Set[str],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F_beta (default beta=0.5) for a single Source 1 entity.
    
    Formula for F0.5:
        beta^2 = 0.25
        (1 + beta^2) = 1.25
        F0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
        
    Edge Cases (Singletons & Empty Predictions):
    - True singleton (true_set is empty):
        - If pred_set is empty: precision = 1.0, recall = 1.0, F0.5 = 1.0, exact_match = 1.0
        - If pred_set is non-empty: precision = 0.0, recall = 0.0, F0.5 = 0.0, exact_match = 0.0
    - Non-singleton (true_set is non-empty):
        - If pred_set is empty: precision = 0.0, recall = 0.0, F0.5 = 0.0, exact_match = 0.0
        - If pred_set is non-empty:
            TP = len(true_set & pred_set)
            precision = TP / len(pred_set)
            recall = TP / len(true_set)
            F0.5 = (1.25 * P * R) / (0.25 * P + R) if (0.25 * P + R) > 0 else 0.0
    """
    beta_sq = beta ** 2
    factor = 1.0 + beta_sq
    
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    exact_match = 1.0 if true_set == pred_set else 0.0
    
    if len(true_set) == 0:
        # True singleton
        is_singleton = 1.0
        if len(pred_set) == 0:
            precision = 1.0
            recall = 1.0
            f_beta = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f_beta = 0.0
    else:
        # Non-singleton
        is_singleton = 0.0
        if len(pred_set) == 0:
            precision = 0.0
            recall = 0.0
            f_beta = 0.0
        else:
            precision = tp / len(pred_set)
            recall = tp / len(true_set)
            denom = (beta_sq * precision) + recall
            f_beta = (factor * precision * recall) / denom if denom > 0.0 else 0.0
            
    return {
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "exact_match": exact_match,
        "is_singleton": is_singleton
    }

def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes challenge evaluation metrics across all Source 1 entities in ground truth.
    
    Primary Metric:
        macro_f05: Macro-average of per-S1 F0.5 scores.
        
    Also returns:
        - macro_precision
        - macro_recall
        - total_tp, total_fp, total_fn
        - exact_match_accuracy
        - singleton_count, singleton_accuracy
        - non_singleton_count, non_singleton_macro_f05
    """
    total_s1 = len(ground_truth)
    if total_s1 == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "exact_match_accuracy": 0.0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "singleton_count": 0,
            "singleton_accuracy": 0.0,
            "non_singleton_count": 0,
            "non_singleton_macro_f05": 0.0
        }
        
    sum_f05 = 0.0
    sum_prec = 0.0
    sum_rec = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    exact_matches = 0
    
    singleton_count = 0
    singleton_correct = 0
    
    non_singleton_count = 0
    non_singleton_f05_sum = 0.0
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        m = compute_single_s1_metrics(true_set, pred_set, beta=beta)
        
        sum_f05 += m["f_beta"]
        sum_prec += m["precision"]
        sum_rec += m["recall"]
        total_tp += int(m["tp"])
        total_fp += int(m["fp"])
        total_fn += int(m["fn"])
        if m["exact_match"] == 1.0:
            exact_matches += 1
            
        if m["is_singleton"] == 1.0:
            singleton_count += 1
            if m["exact_match"] == 1.0:
                singleton_correct += 1
        else:
            non_singleton_count += 1
            non_singleton_f05_sum += m["f_beta"]
            
    return {
        "macro_f05": sum_f05 / total_s1,
        "macro_precision": sum_prec / total_s1,
        "macro_recall": sum_rec / total_s1,
        "exact_match_accuracy": exact_matches / total_s1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "singleton_count": singleton_count,
        "singleton_accuracy": (singleton_correct / singleton_count) if singleton_count > 0 else 0.0,
        "non_singleton_count": non_singleton_count,
        "non_singleton_macro_f05": (non_singleton_f05_sum / non_singleton_count) if non_singleton_count > 0 else 0.0
    }

def evaluate_candidate_blocking(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Computes candidate blocking diagnostic metrics:
    - candidate_recall: fraction of all true target links retrieved in candidates
    - s1_coverage: fraction of non-singleton S1 entities whose TRUE targets are ALL captured in candidates
    - breakdown of recall for S2 vs S3 targets
    - candidate count distributions (average, median, P90, P95, P99, max)
    - fraction of S1 with 0 candidates, >100 candidates, >1000 candidates
    - singleton candidate behavior (how many candidates generated for true singletons)
    """
    total_s1 = len(ground_truth)
    if total_s1 == 0:
        return {}
        
    total_true_links = 0
    retrieved_true_links = 0
    
    total_s2_links = 0
    retrieved_s2_links = 0
    
    total_s3_links = 0
    retrieved_s3_links = 0
    
    non_singleton_s1_count = 0
    fully_covered_s1_count = 0
    
    singleton_s1_count = 0
    singleton_zero_candidate_count = 0
    singleton_candidate_counts: List[int] = []
    
    candidate_counts: List[int] = []
    zero_candidate_count = 0
    gt_100_candidate_count = 0
    gt_1000_candidate_count = 0
    
    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        n_cands = len(cand_set)
        candidate_counts.append(n_cands)
        
        if n_cands == 0:
            zero_candidate_count += 1
        if n_cands > 100:
            gt_100_candidate_count += 1
        if n_cands > 1000:
            gt_1000_candidate_count += 1
            
        if len(true_set) == 0:
            # Singleton
            singleton_s1_count += 1
            singleton_candidate_counts.append(n_cands)
            if n_cands == 0:
                singleton_zero_candidate_count += 1
        else:
            # Non-singleton
            non_singleton_s1_count += 1
            n_true = len(true_set)
            total_true_links += n_true
            
            # Intersection
            captured = true_set & cand_set
            retrieved_true_links += len(captured)
            
            if len(captured) == n_true:
                fully_covered_s1_count += 1
                
            for target in true_set:
                if target.startswith("S2-"):
                    total_s2_links += 1
                    if target in cand_set:
                        retrieved_s2_links += 1
                elif target.startswith("S3-"):
                    total_s3_links += 1
                    if target in cand_set:
                        retrieved_s3_links += 1
                        
    arr_counts = np.array(candidate_counts, dtype=np.int32)
    
    overall_recall = (retrieved_true_links / total_true_links) if total_true_links > 0 else 1.0
    s2_recall = (retrieved_s2_links / total_s2_links) if total_s2_links > 0 else 1.0
    s3_recall = (retrieved_s3_links / total_s3_links) if total_s3_links > 0 else 1.0
    s1_coverage = (fully_covered_s1_count / non_singleton_s1_count) if non_singleton_s1_count > 0 else 1.0
    
    return {
        "candidate_recall_overall": overall_recall,
        "candidate_recall_s2": s2_recall,
        "candidate_recall_s3": s3_recall,
        "s1_coverage": s1_coverage,
        "total_true_links": total_true_links,
        "retrieved_true_links": retrieved_true_links,
        "total_s1_evaluated": total_s1,
        "non_singleton_s1_count": non_singleton_s1_count,
        "fully_covered_s1_count": fully_covered_s1_count,
        # Candidate Count Distribution
        "candidate_avg": float(np.mean(arr_counts)),
        "candidate_median": float(np.median(arr_counts)),
        "candidate_min": int(np.min(arr_counts)),
        "candidate_max": int(np.max(arr_counts)),
        "candidate_p90": float(np.percentile(arr_counts, 90)),
        "candidate_p95": float(np.percentile(arr_counts, 95)),
        "candidate_p99": float(np.percentile(arr_counts, 99)),
        # S1 Cardinality Filters
        "pct_zero_candidates": (zero_candidate_count / total_s1) * 100,
        "pct_gt_100_candidates": (gt_100_candidate_count / total_s1) * 100,
        "pct_gt_1000_candidates": (gt_1000_candidate_count / total_s1) * 100,
        # Singleton Diagnostics
        "singleton_s1_count": singleton_s1_count,
        "singleton_pct_clean_zero_cands": (singleton_zero_candidate_count / singleton_s1_count * 100) if singleton_s1_count > 0 else 100.0,
        "singleton_candidate_avg": float(np.mean(singleton_candidate_counts)) if singleton_candidate_counts else 0.0
    }
