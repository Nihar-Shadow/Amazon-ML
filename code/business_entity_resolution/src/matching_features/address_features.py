import re
from typing import Dict, Any, Set
from src.matching_features.string_metrics import (
    compute_jaccard,
    get_char_ngrams,
    normalized_edit_similarity
)

POSTAL_PATTERN = re.compile(r"\b(\d{5}(?:-\d{4})?|\d{6})\b")

def extract_address_features(q_rec: Dict[str, Any], t_rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts 15 Address Similarity features for a query-target entity pair.
    Safely handles missing or empty addresses without false exact matches.
    """
    q_addr = q_rec.get("address", {})
    t_addr = t_rec.get("address", {})

    q_norm = q_addr.get("normalized", "")
    t_norm = t_addr.get("normalized", "")
    q_compact = q_addr.get("compact", "")
    t_compact = t_addr.get("compact", "")

    q_tokens = q_addr.get("token_set", set())
    t_tokens = t_addr.get("token_set", set())
    q_token_list = q_addr.get("tokens", [])
    t_token_list = t_addr.get("tokens", [])

    is_missing = 1.0 if (not q_norm or not t_norm) else 0.0

    # 1. Exact normalized address equality
    addr_norm_exact = 1.0 if (q_norm and t_norm and q_norm == t_norm) else 0.0

    # 2. Exact compact address equality
    addr_compact_exact = 1.0 if (q_compact and t_compact and q_compact == t_compact) else 0.0

    # 3. Token Jaccard
    addr_token_jaccard = compute_jaccard(q_tokens, t_tokens) if (q_tokens and t_tokens) else 0.0

    # 4, 5. Char 3-gram and 4-gram Jaccards
    if q_compact and t_compact:
        q_3g = get_char_ngrams(q_compact, 3)
        t_3g = get_char_ngrams(t_compact, 3)
        q_4g = get_char_ngrams(q_compact, 4)
        t_4g = get_char_ngrams(t_compact, 4)
        addr_3g = compute_jaccard(q_3g, t_3g)
        addr_4g = compute_jaccard(q_4g, t_4g)
    else:
        addr_3g = 0.0
        addr_4g = 0.0

    # 6. Normalized edit similarity
    addr_edit_sim = normalized_edit_similarity(q_norm, t_norm, max_eval_len=80) if (q_norm and t_norm) else 0.0

    # 7, 8, 9. Number features
    nums_q = q_addr.get("numbers", set())
    nums_t = t_addr.get("numbers", set())
    shared_nums = nums_q & nums_t
    shared_num_count = float(len(shared_nums))
    exact_num_match = 1.0 if (nums_q and nums_t and shared_nums) else 0.0

    # Leading-zero stripped numbers
    norm_nums_q = {n.lstrip("0") or "0" for n in nums_q}
    norm_nums_t = {n.lstrip("0") or "0" for n in nums_t}
    leading_zero_match = 1.0 if (norm_nums_q and norm_nums_t and (norm_nums_q & norm_nums_t)) else 0.0

    # 10. Postal code agreement
    postals_q = set(POSTAL_PATTERN.findall(q_norm))
    postals_t = set(POSTAL_PATTERN.findall(t_norm))
    postal_match = 1.0 if (postals_q and postals_t and (postals_q & postals_t)) else 0.0

    # 11. Salient Locality overlap
    salient_q = set(q_addr.get("salient_tokens", []))
    salient_t = set(t_addr.get("salient_tokens", []))
    locality_overlap = compute_jaccard(salient_q, salient_t) if (salient_q and salient_t) else 0.0

    # 12. State/Region token overlap
    # Check if the last salient tokens (typically state / region in Indian and US addresses) overlap
    state_overlap = 0.0
    if len(q_token_list) >= 1 and len(t_token_list) >= 1:
        tail_q = set(q_token_list[-2:])
        tail_t = set(t_token_list[-2:])
        if tail_q & tail_t:
            state_overlap = 1.0

    # 13. Substring containment
    addr_containment = 0.0
    if q_compact and t_compact:
        if q_compact in t_compact or t_compact in q_compact:
            addr_containment = 1.0

    # 14. Token count difference
    token_diff = float(abs(len(q_token_list) - len(t_token_list)))

    return {
        "addr_normalized_exact": addr_norm_exact,
        "addr_compact_exact": addr_compact_exact,
        "addr_token_jaccard": addr_token_jaccard,
        "addr_char_3gram_jaccard": addr_3g,
        "addr_char_4gram_jaccard": addr_4g,
        "addr_edit_similarity": addr_edit_sim,
        "addr_shared_numbers_count": shared_num_count,
        "addr_exact_number_match": exact_num_match,
        "addr_leading_zero_number_match": leading_zero_match,
        "addr_postal_code_match": postal_match,
        "addr_locality_overlap": locality_overlap,
        "addr_state_overlap": state_overlap,
        "addr_containment": addr_containment,
        "addr_token_count_diff": token_diff,
        "addr_is_missing_either": is_missing,
    }
