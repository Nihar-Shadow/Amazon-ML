import re
from typing import Dict, Optional, Set
from src.utils.logging_utils import get_logger

logger = get_logger("ground_truth_parser")

# ID format validation regex: S1-12345, S2-12345, S3-12345
S1_ID_PATTERN = re.compile(r"^S1-\d+$")
TARGET_ID_PATTERN = re.compile(r"^(S2|S3)-\d+$")

def validate_entity_id(entity_id: str, is_s1: bool = True) -> bool:
    """Validates that an entity ID conforms strictly to expected prefix and digits."""
    pattern = S1_ID_PATTERN if is_s1 else TARGET_ID_PATTERN
    return bool(pattern.match(entity_id))

def parse_ground_truth(
    file_path: str = "datasets/train/train_ground_truth.tsv",
    valid_s1_ids: Optional[Set[str]] = None,
    valid_target_ids: Optional[Set[str]] = None,
    strict_id_format: bool = True
) -> Dict[str, Set[str]]:
    """
    Parses train_ground_truth.tsv into an efficient lookup mapping:
        ground_truth[source1_id] -> set(target_ids)
    
    Empty matched_entity_ids becomes an empty set (true singleton).
    
    Validations enforced:
    - Every S1 ID matches format 'S1-digits'.
    - Every matched target ID matches format 'S2-digits' or 'S3-digits'.
    - No duplicate IDs inside any single match list (raises ValueError).
    - If valid_s1_ids is provided, ensures every S1 ID exists in valid_s1_ids.
    - If valid_target_ids is provided, ensures every target ID exists in valid_target_ids.
    
    Does NOT silently repair invalid or duplicate data.
    """
    ground_truth: Dict[str, Set[str]] = {}
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        header = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
        if len(header) < 2 or header[0] != "source1_entity_id" or header[1] != "matched_entity_ids":
            raise ValueError(f"Invalid ground truth header: {header}. Expected ['source1_entity_id', 'matched_entity_ids'].")
            
        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                raise ValueError(f"Malformed row at line {line_num}: expected 2 tab-separated columns, found {len(parts)}.")
                
            s1_id, matched_str = parts[0].strip(), parts[1].strip()
            
            if strict_id_format and not validate_entity_id(s1_id, is_s1=True):
                raise ValueError(f"Malformed S1 entity ID at line {line_num}: '{s1_id}'. Must match pattern S1-<digits>.")
                
            if s1_id in ground_truth:
                raise ValueError(f"Duplicate S1 entity ID '{s1_id}' encountered at line {line_num}.")
                
            if valid_s1_ids is not None and s1_id not in valid_s1_ids:
                raise KeyError(f"Ground truth S1 ID '{s1_id}' at line {line_num} does not exist in train_source1.")
                
            if not matched_str:
                # Singleton / zero matches
                ground_truth[s1_id] = set()
                continue
                
            raw_targets = [t.strip() for t in matched_str.split(",")]
            target_set: Set[str] = set()
            
            for t in raw_targets:
                if not t:
                    raise ValueError(f"Empty target ID in match list for S1 '{s1_id}' at line {line_num}.")
                    
                if strict_id_format and not validate_entity_id(t, is_s1=False):
                    raise ValueError(f"Malformed target entity ID '{t}' for S1 '{s1_id}' at line {line_num}. Must match pattern (S2|S3)-<digits>.")
                    
                if t in target_set:
                    raise ValueError(f"Duplicate target ID '{t}' detected inside match list for S1 '{s1_id}' at line {line_num}.")
                    
                if valid_target_ids is not None and t not in valid_target_ids:
                    raise KeyError(f"Target ID '{t}' for S1 '{s1_id}' at line {line_num} does not exist in target sources.")
                    
                target_set.add(t)
                
            ground_truth[s1_id] = target_set
            
    logger.info(f"Parsed {len(ground_truth):,} ground-truth records successfully from {file_path}.")
    return ground_truth
