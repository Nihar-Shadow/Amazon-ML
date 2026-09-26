from typing import Dict, Any
from src.matching_features.string_metrics import compute_jaccard, get_char_ngrams

def extract_view_features(q_rec: Dict[str, Any], t_rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts 5 Normalization-View features based on Phase 3 multi-view representations.
    Exposes signals from leetspeak normalization, DBA trade-name extraction, and compact views.
    """
    q_name = q_rec.get("name", {})
    t_name = t_rec.get("name", {})
    q_addr = q_rec.get("address", {})
    t_addr = t_rec.get("address", {})

    q_leet = q_name.get("leet_compact", "")
    t_leet = t_name.get("leet_compact", "")

    # 1. Leet name exact match
    leet_exact = 1.0 if (q_leet and t_leet and q_leet == t_leet) else 0.0

    # 2. Leet 3-gram similarity
    if q_leet and t_leet:
        q_leet_3g = get_char_ngrams(q_leet, 3)
        t_leet_3g = get_char_ngrams(t_leet, 3)
        leet_sim = compute_jaccard(q_leet_3g, t_leet_3g)
    else:
        leet_sim = 0.0

    # 3. DBA trade-name alias match
    q_compact = q_name.get("compact", "")
    t_compact = t_name.get("compact", "")
    q_dbas = set(q_name.get("dba_compacts", []))
    t_dbas = set(t_name.get("dba_compacts", []))

    dba_match = 0.0
    if (q_compact and q_compact in t_dbas) or (t_compact and t_compact in q_dbas) or (q_dbas & t_dbas):
        dba_match = 1.0

    # 4. Compact name match
    compact_name_match = 1.0 if (q_compact and t_compact and q_compact == t_compact) else 0.0

    # 5. Compact address match
    q_ac = q_addr.get("compact", "")
    t_ac = t_addr.get("compact", "")
    compact_addr_match = 1.0 if (q_ac and t_ac and q_ac == t_ac) else 0.0

    return {
        "view_leet_name_exact": leet_exact,
        "view_leet_name_sim": leet_sim,
        "view_dba_alias_match": dba_match,
        "view_compact_name_match": compact_name_match,
        "view_compact_addr_match": compact_addr_match,
    }
