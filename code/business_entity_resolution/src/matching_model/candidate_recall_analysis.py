from typing import Dict, Set, Any
import numpy as np
import pandas as pd

def compute_recall_ceilings(
    val_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    threshold: float,
    candidate_generation_recall: float = 0.8832
) -> Dict[str, Any]:
    """
    Computes candidate-generation recall ceiling vs model discrimination recall within candidates.
    """
    total_true_links = sum(len(matches) for matches in ground_truth.values())
    
    # Candidate-representable true links in validation set
    represented_true_df = val_df[val_df["label"] == 1]
    candidate_represented_true_count = len(represented_true_df)
    
    # Model positive predictions at threshold
    preds_above = val_df[probabilities >= threshold]
    pred_map: Dict[str, Set[str]] = {}
    for s1_id, t_id in zip(preds_above["source1_entity_id"], preds_above["target_entity_id"]):
        if s1_id not in pred_map:
            pred_map[s1_id] = set()
        pred_map[s1_id].add(t_id)

    # True links captured by model
    model_captured_true_count = 0
    for s1_id, true_set in ground_truth.items():
        pred_set = pred_map.get(s1_id, set())
        model_captured_true_count += len(true_set & pred_set)

    # Metrics
    cand_recall = candidate_represented_true_count / total_true_links if total_true_links > 0 else 0.0
    model_recall_within_cands = (
        model_captured_true_count / candidate_represented_true_count
        if candidate_represented_true_count > 0 else 0.0
    )
    final_end_to_end_recall = (
        model_captured_true_count / total_true_links
        if total_true_links > 0 else 0.0
    )
    
    blocking_misses = total_true_links - candidate_represented_true_count
    model_discrimination_misses = candidate_represented_true_count - model_captured_true_count

    return {
        "total_ground_truth_links": total_true_links,
        "candidate_represented_true_links": candidate_represented_true_count,
        "model_captured_true_links": model_captured_true_count,
        "candidate_generation_recall_ceiling_pct": round(cand_recall * 100.0, 2),
        "model_recall_within_candidates_pct": round(model_recall_within_cands * 100.0, 2),
        "final_end_to_end_recall_pct": round(final_end_to_end_recall * 100.0, 2),
        "blocking_miss_count": blocking_misses,
        "model_discrimination_miss_count": model_discrimination_misses,
    }
