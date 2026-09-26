from typing import Dict, List, Set, Tuple, Optional, Any
from src.utils.logging_utils import get_logger

logger = get_logger("candidate_refiner")

def compute_jaccard(set1: Set[Any], set2: Set[Any]) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def score_candidate_pair(
    query_record: Dict[str, Any],
    target_record: Dict[str, Any],
    provenance_strategies: Optional[Set[str]] = None
) -> float:
    """
    Computes a deterministic, lightweight heuristic score in [0.0, 1.0+]
    for refining candidate pairs prior to the ML matching model.
    """
    # 1. Country compatibility: hard filter if both specified and different
    c1 = query_record.get("country", "")
    c2 = target_record.get("country", "")
    if c1 and c2 and c1 != c2:
        return -1.0  # Rejected

    # High-precision strategy bypass
    if provenance_strategies:
        if (
            "exact_name" in provenance_strategies
            or "sorted_name_tokens" in provenance_strategies
            or "name_leet_compact" in provenance_strategies
            or "name_dba_alias" in provenance_strategies
        ):
            return 1.0

    name1 = query_record.get("name", {})
    name2 = target_record.get("name", {})
    
    # 2. Name character n-gram Jaccard
    ngrams1 = name1.get("ngrams", set())
    ngrams2 = name2.get("ngrams", set())
    ngram_sim = compute_jaccard(ngrams1, ngrams2)
    
    # 3. Name token Jaccard
    tokens1 = name1.get("token_set", set())
    tokens2 = name2.get("token_set", set())
    token_sim = compute_jaccard(tokens1, tokens2)
    
    # 4. Compact name containment (handles prefix/acronym/domain/leet/dba variants)
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
            
    # 5. Address signals
    addr1 = query_record.get("address", {})

    addr2 = target_record.get("address", {})
    
    addr_tokens1 = addr1.get("token_set", set())
    addr_tokens2 = addr2.get("token_set", set())
    addr_token_sim = compute_jaccard(addr_tokens1, addr_tokens2) if (addr_tokens1 and addr_tokens2) else 0.0
    
    nums1 = addr1.get("numbers", set())
    nums2 = addr2.get("numbers", set())
    num_match = 1.0 if (nums1 and nums2 and len(nums1 & nums2) > 0) else 0.0

    # Composite weighted score
    score = (
        0.35 * ngram_sim +
        0.30 * token_sim +
        0.20 * containment +
        0.10 * addr_token_sim +
        0.05 * num_match
    )
    return score

class CandidateRefiner:
    """
    Deterministic candidate refinement engine that prunes weak candidates
    while preserving high candidate recall.
    """
    def __init__(
        self,
        min_score_threshold: float = 0.15,
        max_candidates_per_query: int = 50,
        preserve_provenance_bypass: bool = True
    ):
        self.min_score_threshold = min_score_threshold
        self.max_candidates_per_query = max_candidates_per_query
        self.preserve_provenance_bypass = preserve_provenance_bypass

    def refine_candidates_for_query(
        self,
        query_record: Dict[str, Any],
        candidate_target_records: Dict[str, Dict[str, Any]],
        provenance_map: Optional[Dict[str, Set[str]]] = None
    ) -> Set[str]:
        """
        Refines candidate target entities for a single S1 query.
        Returns the filtered and capped set of target entity IDs.
        """
        scored_candidates: List[Tuple[str, float]] = []

        for target_id, target_record in candidate_target_records.items():
            strats = provenance_map.get(target_id) if provenance_map else None
            score = score_candidate_pair(query_record, target_record, provenance_strategies=strats)
            
            if score < 0.0:
                continue  # Country mismatch or hard filter
                
            # If provenance indicates exact match and bypass enabled, retain immediately
            if self.preserve_provenance_bypass and strats and ("exact_name" in strats or "sorted_name_tokens" in strats):
                scored_candidates.append((target_id, max(score, 1.0)))
            elif score >= self.min_score_threshold:
                scored_candidates.append((target_id, score))

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top-K ceiling
        selected_ids = {tid for tid, _ in scored_candidates[:self.max_candidates_per_query]}
        return selected_ids

    def refine_candidate_pool(
        self,
        query_records: Dict[str, Dict[str, Any]],
        target_records: Dict[str, Dict[str, Any]],
        raw_candidates: Dict[str, Set[str]],
        provenance: Optional[Dict[str, Dict[str, Set[str]]]] = None
    ) -> Dict[str, Set[str]]:
        """
        Refines an entire candidate dictionary mapping s1_id -> candidate_set.
        """
        refined: Dict[str, Set[str]] = {}
        for s1_id, cand_set in raw_candidates.items():
            query_rec = query_records.get(s1_id)
            if not query_rec:
                refined[s1_id] = set()
                continue
                
            cand_target_recs = {
                tid: target_records[tid]
                for tid in cand_set
                if tid in target_records
            }
            
            prov_map = provenance.get(s1_id) if provenance else None
            refined[s1_id] = self.refine_candidates_for_query(
                query_rec, cand_target_recs, provenance_map=prov_map
            )
            
        return refined
