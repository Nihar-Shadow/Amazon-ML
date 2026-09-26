from typing import Dict, List, Set, Any
import numpy as np
import pandas as pd

def analyze_model_errors(
    val_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    threshold: float
) -> Dict[str, Any]:
    """
    Performs forensic categorization of False Positives and False Negatives.
    """
    preds_binary = (probabilities >= threshold).astype(int)
    labels = val_df["label"].values

    val_df = val_df.copy()
    val_df["pred_prob"] = probabilities
    val_df["pred_binary"] = preds_binary

    fp_df = val_df[(val_df["label"] == 0) & (val_df["pred_binary"] == 1)]
    fn_df = val_df[(val_df["label"] == 1) & (val_df["pred_binary"] == 0)]
    tp_df = val_df[(val_df["label"] == 1) & (val_df["pred_binary"] == 1)]

    # Categorize False Positives
    fp_counts = {
        "same_address_different_business": int(((fp_df["addr_normalized_exact"] == 1.0) & (fp_df["name_token_jaccard"] < 0.3)).sum()),
        "shared_commercial_complex": int(((fp_df["addr_exact_number_match"] == 1.0) & (fp_df["name_token_jaccard"] < 0.5)).sum()),
        "generic_name_collision": int(((fp_df["name_compact_exact"] == 1.0) & (fp_df["addr_token_jaccard"] < 0.2)).sum()),
        "dba_confusion": int((fp_df["view_dba_alias_match"] == 1.0).sum()),
        "address_only_evidence": int(((fp_df["addr_token_jaccard"] >= 0.7) & (fp_df["name_token_jaccard"] < 0.2)).sum()),
        "legal_suffix_collisions": int(((fp_df["qual_has_legal_suffix"] == 1.0) & (fp_df["name_edit_similarity"] >= 0.6) & (fp_df["addr_token_jaccard"] < 0.3)).sum()),
        "other_false_positives": 0
    }
    explained_fp = sum(fp_counts.values())
    fp_counts["other_false_positives"] = max(0, len(fp_df) - explained_fp)

    # Categorize False Negatives
    fn_counts = {
        "cross_script_names": int((fn_df["qual_mixed_script_pair"] == 1.0).sum()),
        "missing_address": int((fn_df["qual_addr_missing_t"] == 1.0).sum()),
        "weak_address": int(((fn_df["addr_token_jaccard"] < 0.2) & (fn_df["qual_addr_missing_t"] == 0.0)).sum()),
        "brand_renames": int(((fn_df["name_token_jaccard"] < 0.2) & (fn_df["addr_token_jaccard"] >= 0.6)).sum()),
        "severe_typos": int(((fn_df["name_token_jaccard"] < 0.5) & (fn_df["name_char_3gram_jaccard"] >= 0.3)).sum()),
        "borderline_refinement_score": int(((fn_df["refine_composite_score"] >= 0.15) & (fn_df["refine_composite_score"] < 0.35)).sum()),
        "other_false_negatives": 0
    }
    explained_fn = sum(fn_counts.values())
    fn_counts["other_false_negatives"] = max(0, len(fn_df) - explained_fn)

    return {
        "total_true_positives": len(tp_df),
        "total_false_positives": len(fp_df),
        "total_false_negatives": len(fn_df),
        "false_positive_categories": fp_counts,
        "false_negative_categories": fn_counts,
        "sample_false_positives": fp_df[["source1_entity_id", "target_entity_id", "pred_prob"]].head(5).to_dict(orient="records"),
        "sample_false_negatives": fn_df[["source1_entity_id", "target_entity_id", "pred_prob"]].head(5).to_dict(orient="records"),
    }
