from typing import Dict, Any
from src.matching_features.string_metrics import (
    compute_jaccard,
    get_char_ngrams,
    normalized_edit_similarity,
    common_prefix_length,
    common_suffix_length
)

def extract_name_features(q_rec: Dict[str, Any], t_rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts 15 Name Similarity features for a query-target entity pair.
    """
    q_name = q_rec.get("name", {})
    t_name = t_rec.get("name", {})

    q_norm = q_name.get("normalized", "")
    t_norm = t_name.get("normalized", "")
    q_compact = q_name.get("compact", "")
    t_compact = t_name.get("compact", "")

    q_tokens = q_name.get("token_set", set())
    t_tokens = t_name.get("token_set", set())
    q_token_list = q_name.get("tokens", [])
    t_token_list = t_name.get("tokens", [])

    # 1. Exact normalized equality
    name_norm_exact = 1.0 if (q_norm and t_norm and q_norm == t_norm) else 0.0

    # 2. Exact compact equality
    name_compact_exact = 1.0 if (q_compact and t_compact and q_compact == t_compact) else 0.0

    # 3. Token Jaccard
    name_token_jaccard = compute_jaccard(q_tokens, t_tokens)

    # 4. Sorted token equality / Jaccard
    q_sorted = q_name.get("sorted_tokens_str", "")
    t_sorted = t_name.get("sorted_tokens_str", "")
    name_sorted_jaccard = 1.0 if (q_sorted and t_sorted and q_sorted == t_sorted) else name_token_jaccard

    # 5, 6, 7. Character n-gram Jaccards (2-gram, 3-gram, 4-gram on compact names)
    q_2g = get_char_ngrams(q_compact, 2)
    t_2g = get_char_ngrams(t_compact, 2)
    q_3g = q_name.get("ngrams") or get_char_ngrams(q_compact, 3)
    t_3g = t_name.get("ngrams") or get_char_ngrams(t_compact, 3)
    q_4g = get_char_ngrams(q_compact, 4)
    t_4g = get_char_ngrams(t_compact, 4)

    name_2g_jaccard = compute_jaccard(q_2g, t_2g)
    name_3g_jaccard = compute_jaccard(q_3g, t_3g)
    name_4g_jaccard = compute_jaccard(q_4g, t_4g)

    # 8. Normalized edit similarity on normalized strings
    name_edit_sim = normalized_edit_similarity(q_norm, t_norm, max_eval_len=80)

    # 9. Levenshtein similarity on compact names
    name_lev_compact_sim = normalized_edit_similarity(q_compact, t_compact, max_eval_len=60)

    # 10. Compact substring containment
    name_containment = 0.0
    if q_compact and t_compact:
        if q_compact in t_compact or t_compact in q_compact:
            name_containment = 1.0

    # 11, 12. Prefix & Suffix similarity ratios
    max_len = max(len(q_compact), len(t_compact), 1)
    prefix_len = common_prefix_length(q_compact, t_compact)
    suffix_len = common_suffix_length(q_compact, t_compact)
    name_prefix_sim = prefix_len / max_len if (q_compact and t_compact) else 0.0
    name_suffix_sim = suffix_len / max_len if (q_compact and t_compact) else 0.0

    # 13. Token count difference
    name_token_diff = float(abs(len(q_token_list) - len(t_token_list)))

    # 14. Shared token count
    name_shared_tokens = float(len(q_tokens & t_tokens))

    # 15. Longest token match
    q_longest = max(q_token_list, key=len) if q_token_list else ""
    t_longest = max(t_token_list, key=len) if t_token_list else ""
    name_longest_match = 1.0 if (q_longest and t_longest and q_longest == t_longest) else 0.0

    return {
        "name_normalized_exact": name_norm_exact,
        "name_compact_exact": name_compact_exact,
        "name_token_jaccard": name_token_jaccard,
        "name_sorted_token_jaccard": name_sorted_jaccard,
        "name_char_2gram_jaccard": name_2g_jaccard,
        "name_char_3gram_jaccard": name_3g_jaccard,
        "name_char_4gram_jaccard": name_4g_jaccard,
        "name_edit_similarity": name_edit_sim,
        "name_levenshtein_compact_sim": name_lev_compact_sim,
        "name_compact_containment": name_containment,
        "name_prefix_similarity": name_prefix_sim,
        "name_suffix_similarity": name_suffix_sim,
        "name_token_count_diff": name_token_diff,
        "name_shared_token_count": name_shared_tokens,
        "name_longest_token_match": name_longest_match,
    }
