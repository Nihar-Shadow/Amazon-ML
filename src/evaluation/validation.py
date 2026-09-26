import hashlib
import random
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from src.utils.logging_utils import get_logger

logger = get_logger("validation_framework")

def create_deterministic_split(
    s1_ids: List[str],
    val_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Creates a deterministic 80/20 train/validation split of Source 1 entity IDs.
    
    Anchor-Level Separation Design:
    - In entity resolution, Source 1 acts as query anchors.
    - Splitting at the S1 anchor level ensures that the validation set tests
      the system's ability to resolve completely unseen query entities.
    - Target entities (Source 2 and Source 3) remain accessible in the global target
      candidate pool, but the ground-truth linkages for validation S1 anchors
      are quarantined and strictly hidden during model training / indexing.
    - Fixed seed and stable sort guarantee 100% reproducible splits across environments.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {val_ratio}")
        
    # Sort first for platform-independent stability before shuffling
    sorted_ids = sorted(s1_ids)
    rng = random.Random(seed)
    shuffled_ids = sorted_ids.copy()
    rng.shuffle(shuffled_ids)
    
    n_total = len(shuffled_ids)
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val
    
    val_ids = shuffled_ids[:n_val]
    train_ids = shuffled_ids[n_val:]
    
    logger.info(
        f"Created deterministic split: Train S1 = {len(train_ids):,} ({len(train_ids)/n_total*100:.1f}%), "
        f"Val S1 = {len(val_ids):,} ({len(val_ids)/n_total*100:.1f}%) [Seed: {seed}]"
    )
    return train_ids, val_ids

def generate_candidates(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    max_candidates_per_s1: int = 0
) -> Dict[str, Set[str]]:
    """
    Candidate Generation (Blocking) Interface.
    
    Separates the BLOCKING stage from the MATCHING stage.
    
    In Phase 2, this provides the standardized interface. It returns
    an initial candidate dictionary mapping each S1 ID to a candidate set
    of target IDs (defaults to empty set or trivial baseline).
    
    Args:
        source1_df: DataFrame containing Source 1 records (must have 'entity_id')
        source2_df: DataFrame containing Source 2 records (must have 'entity_id')
        source3_df: DataFrame containing Source 3 records (must have 'entity_id')
        max_candidates_per_s1: Maximum candidates per S1 (0 = empty baseline)
        
    Returns:
        Dict mapping s1_entity_id -> Set of candidate target entity IDs (S2 / S3)
        e.g.:
        {
            "S1-1001": {"S2-5001", "S3-9002"},
            "S1-1002": set()
        }
    """
    if "entity_id" not in source1_df.columns:
        raise ValueError("source1_df must contain 'entity_id' column")
        
    s1_ids = source1_df["entity_id"].tolist()
    candidates: Dict[str, Set[str]] = {s1: set() for s1 in s1_ids}
    
    # Baseline: returns empty set for each S1 entity (no blocking logic implemented yet)
    return candidates
