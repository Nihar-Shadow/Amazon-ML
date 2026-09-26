from typing import Dict, List, Set, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from src.evaluation.metrics import compute_single_s1_metrics

def evaluate_model_at_threshold(
    val_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    threshold: float = 0.5,
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes challenge-accurate entity-level Macro F0.5 alongside pairwise and diagnostic metrics.
    
    Entity-level mapping:
        For each S1:
            predicted_matches = {target_id | P(target_id) >= threshold}
            Computes precision, recall, F0.5 per S1 according to official competition rules.
            Macro F0.5 is the arithmetic mean across all S1 entities in ground_truth.
    """
    # 1. Group predictions by source1_entity_id
    preds_above = val_df[probabilities >= threshold]
    pred_map: Dict[str, Set[str]] = {}
    for s1_id, t_id in zip(preds_above["source1_entity_id"], preds_above["target_entity_id"]):
        if s1_id not in pred_map:
            pred_map[s1_id] = set()
        pred_map[s1_id].add(t_id)

    # 2. Compute per-S1 metrics across all ground-truth queries
    f_scores: List[float] = []
    p_scores: List[float] = []
    r_scores: List[float] = []

    singleton_f_scores: List[float] = []
    multi_f_scores: List[float] = []
    zero_match_correct = 0
    total_zero_match = 0
    total_predicted_matches = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = pred_map.get(s1_id, set())
        total_predicted_matches += len(pred_set)

        res = compute_single_s1_metrics(true_set, pred_set, beta=beta)
        f_scores.append(res["f_beta"])
        p_scores.append(res["precision"])
        r_scores.append(res["recall"])

        if len(true_set) == 0:
            total_zero_match += 1
            if len(pred_set) == 0:
                zero_match_correct += 1
        elif len(true_set) == 1:
            singleton_f_scores.append(res["f_beta"])
        else:
            multi_f_scores.append(res["f_beta"])

    macro_f05 = float(np.mean(f_scores)) if f_scores else 0.0
    macro_p = float(np.mean(p_scores)) if p_scores else 0.0
    macro_r = float(np.mean(r_scores)) if r_scores else 0.0

    singleton_f05 = float(np.mean(singleton_f_scores)) if singleton_f_scores else 0.0
    multi_f05 = float(np.mean(multi_f_scores)) if multi_f_scores else 0.0
    zero_acc = (zero_match_correct / total_zero_match) if total_zero_match > 0 else 1.0

    # 3. Pairwise classification diagnostics
    y_true = val_df["label"].values
    y_pred_binary = (probabilities >= threshold).astype(int)

    pairwise_p = float(precision_score(y_true, y_pred_binary, zero_division=0))
    pairwise_r = float(recall_score(y_true, y_pred_binary, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, probabilities))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, probabilities))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "threshold": round(threshold, 4),
        "macro_f05": round(macro_f05, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "pairwise_precision": round(pairwise_p, 4),
        "pairwise_recall": round(pairwise_r, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "singleton_f05": round(singleton_f05, 4),
        "multi_match_f05": round(multi_f05, 4),
        "zero_match_accuracy": round(zero_acc, 4),
        "predicted_match_count": total_predicted_matches
    }

def sweep_thresholds(
    val_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Performs a fine-grained threshold sweep to identify the optimal threshold maximizing Macro F0.5.
    """
    if thresholds is None:
        thresholds = [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
        ]

    results = []
    best_res = None
    best_f05 = -1.0

    for t in thresholds:
        res = evaluate_model_at_threshold(val_df, probabilities, ground_truth, threshold=t)
        results.append(res)
        if res["macro_f05"] > best_f05:
            best_f05 = res["macro_f05"]
            best_res = res

    return best_res, results
