from typing import Dict, Any

def extract_country_features(q_rec: Dict[str, Any], t_rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts 4 Open-Set Country features.
    Works for arbitrary country strings (US, India, France, or unseen future countries).
    """
    c1 = (q_rec.get("country") or "").strip().upper()
    c2 = (t_rec.get("country") or "").strip().upper()

    missing = 1.0 if (not c1 or not c2) else 0.0
    exact = 1.0 if (c1 and c2 and c1 == c2) else 0.0
    conflict = 1.0 if (c1 and c2 and c1 != c2) else 0.0

    return {
        "country_exact_match": exact,
        "country_normalized_match": exact,
        "country_missing_either": missing,
        "country_conflict": conflict,
    }
