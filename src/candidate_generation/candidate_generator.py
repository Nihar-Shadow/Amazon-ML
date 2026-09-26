import os
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator
import pandas as pd
from src.candidate_generation.normalizer import normalize_entity_record
from src.candidate_generation.blocking import BlockingIndex
from src.candidate_generation.candidate_pool import CandidatePool
from src.candidate_generation.candidate_refiner import CandidateRefiner
from src.utils.logging_utils import get_logger, log_step

logger = get_logger("candidate_generator")

class CandidateGenerator:
    """
    End-to-End Scalable Candidate Generation and Blocking Pipeline.
    
    Architecture:
        Raw Records
             ↓
        Normalization (Multi-view representations)
             ↓
        Multi-Block Inverted Indexing
             ↓
        Candidate Pool Union (Multi-Strategy Retrieval)
             ↓
        Candidate Refinement (Deterministic pruning & top-K capping)
             ↓
        FINAL CANDIDATE SET (Written to candidate_pairs.tsv)
    """
    def __init__(
        self,
        blocking_strategies: Optional[List[str]] = None,
        max_block_size: int = 500,
        min_refinement_score: float = 0.15,
        max_candidates_per_s1: int = 50,
        enable_refinement: bool = True
    ):
        self.blocking_strategies = blocking_strategies or [
            "exact_name",
            "sorted_name_tokens",
            "first_two_name_tokens",
            "longest_name_token",
            "name_prefix_6",
            "name_address_combo",
            "address_exact_combo"
        ]
        self.max_block_size = max_block_size
        self.enable_refinement = enable_refinement
        
        self.blocking_index = BlockingIndex(max_block_size=max_block_size)
        self.refiner = CandidateRefiner(
            min_score_threshold=min_refinement_score,
            max_candidates_per_query=max_candidates_per_s1
        )
        
        # In-memory target record cache for refinement feature evaluation
        # Stored compactly
        self.target_record_cache: Dict[str, Dict[str, Any]] = {}

    def index_target_records(
        self,
        records_iterator: Iterator[Tuple[str, str, str, str]],
        store_for_refinement: bool = True
    ):
        """
        Indexes target records (Source 2 and/or Source 3).
        Each record tuple is (entity_id, business_name, business_address, country).
        """
        count = 0
        for eid, bname, baddr, bcountry in records_iterator:
            count += 1
            norm_rec = normalize_entity_record(eid, bname, baddr, bcountry)
            self.blocking_index.add_target_record(norm_rec, strategies=self.blocking_strategies)
            if store_for_refinement and self.enable_refinement:
                self.target_record_cache[eid] = norm_rec
                
        logger.info(f"Indexed {count:,} target records into multi-block index.")

    def generate_candidates(
        self,
        query_records_iterator: Iterator[Tuple[str, str, str, str]]
    ) -> Dict[str, Set[str]]:
        """
        Generates candidate target entity IDs for each query entity.
        Returns:
            Dict[str, Set[str]]: Mapping from s1_entity_id to final set of candidate target IDs.
        """
        final_candidates: Dict[str, Set[str]] = {}
        count = 0
        
        for q_eid, q_bname, q_baddr, q_bcountry in query_records_iterator:
            count += 1
            q_norm = normalize_entity_record(q_eid, q_bname, q_baddr, q_bcountry)
            
            # 1. Multi-strategy retrieval with provenance tracking
            candidates_with_prov = self.blocking_index.retrieve_candidates_for_query(
                q_norm,
                strategies=self.blocking_strategies,
                return_provenance=True
            )
            
            # 2. Refinement step if enabled
            if self.enable_refinement and candidates_with_prov:
                cand_target_recs = {
                    tid: self.target_record_cache[tid]
                    for tid in candidates_with_prov.keys()
                    if tid in self.target_record_cache
                }
                refined_set = self.refiner.refine_candidates_for_query(
                    q_norm,
                    cand_target_recs,
                    provenance_map=candidates_with_prov
                )
                final_candidates[q_eid] = refined_set
            else:
                final_candidates[q_eid] = set(candidates_with_prov.keys())
                
        logger.info(f"Generated final candidates for {count:,} query entities.")
        return final_candidates

    @staticmethod
    def write_candidate_pairs_tsv(
        candidates: Dict[str, Set[str]],
        output_file_path: str = "output/candidate_pairs.tsv"
    ):
        """
        Writes the final candidate set to TSV adhering strictly to official specification:
        Columns:
            source1_entity_id\tcandidate_entity_ids
            
        Rules:
        - Exactly one row per test S1 entity.
        - Comma-separated list of candidate target IDs (empty if none).
        - No duplicate IDs.
        - Preserves exact strings.
        """
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        with open(output_file_path, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in sorted(candidates.keys()):
                target_ids = sorted(candidates[s1_id])
                cand_str = ",".join(target_ids)
                f.write(f"{s1_id}\t{cand_str}\n")
                
        logger.info(f"Saved exact final candidate pairs to {output_file_path} ({len(candidates):,} rows).")
