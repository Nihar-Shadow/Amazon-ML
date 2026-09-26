from typing import Dict, Any, Set, Optional
from src.matching_features.string_metrics import compute_jaccard

def extract_refinement_features(
    q_rec: Dict[str, Any],
    t_rec: Dict[str, Any],
    provenance_strategies: Optional[Set[str]] = None
) -> Dict[str, float]:
    """
    Extracts 7 Refinement features exposing Phase 3 heuristic score and its constituent sub-scores.
    """
    strats = provenance_strategies or set()
    
    # Check provenance bypass
    provenance_bypass = 1.0 if (
        "exact_name" in strats
        or "sorted_name_tokens" in strats
        or "name_leet_compact" in strats
        or "name_dba_alias" in strats
    ) else 0.0

    name1 = q_rec.get("name", {})
    name2 = t_rec.get("name", {})
    addr1 = q_rec.get("address", {})
    addr2 = t_rec.get("address", {})

    # 1. Name n-gram sim
    ngram_sim = compute_jaccard(name1.get("ngrams", set()), name2.get("ngrams", set()))

    # 2. Token sim
    token_sim = compute_jaccard(name1.get("token_set", set()), name2.get("token_set", set()))

    # 3. Containment
    compact1 = name1.get("compact", "")
    compact2 = name2.get("compact", "")
    leet1 = name1.get("leet_compact", "")
    leet2 = name2.get("leet_compact", "")
    dba1 = set(name1.get("dba_compacts", []))
    dba2 = set(name2.get("dba_compacts", []))

    containment = 0.0
    if compact1 and compact2:
        if compact1 == compact2 or (leet1 and leet1 == compact2) or (leet2 and leet2 == compact1):
            containment = 1.0
        elif compact1 in dba2 or compact2 in dba1:
            containment = 1.0
        elif compact1 in compact2 or compact2 in compact1:
            containment = 0.6
        elif len(compact1) >= 4 and len(compact2) >= 4 and compact1[:4] == compact2[:4]:
            containment = 0.3

    # 4. Address token sim
    addr_tokens1 = addr1.get("token_set", set())
    addr_tokens2 = addr2.get("token_set", set())
    addr_sim = compute_jaccard(addr_tokens1, addr_tokens2) if (addr_tokens1 and addr_tokens2) else 0.0

    # 5. Number match
    nums1 = addr1.get("numbers", set())
    nums2 = addr2.get("numbers", set())
    num_match = 1.0 if (nums1 and nums2 and len(nums1 & nums2) > 0) else 0.0

    # 6. Composite heuristic score
    if provenance_bypass == 1.0:
        composite = 1.0
    else:
        composite = (
            0.35 * ngram_sim +
            0.30 * token_sim +
            0.20 * containment +
            0.10 * addr_sim +
            0.05 * num_match
        )

    return {
        "refine_composite_score": composite,
        "refine_ngram_sim": ngram_sim,
        "refine_token_sim": token_sim,
        "refine_containment": containment,
        "refine_addr_token_sim": addr_sim,
        "refine_number_match": num_match,
        "refine_provenance_bypass": provenance_bypass,
    }
