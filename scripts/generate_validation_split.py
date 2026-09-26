import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import json
from collections import Counter
from src.data.loader import stream_entity_ids
from src.data.ground_truth import parse_ground_truth
from src.evaluation.validation import create_deterministic_split
from src.utils.logging_utils import get_logger, log_step

logger = get_logger("split_generator")

def generate_and_save_split(
    s1_path: str = "datasets/train/train_source1.tsv",
    gt_path: str = "datasets/train/train_ground_truth.tsv",
    out_dir: str = "eda",
    val_ratio: float = 0.20,
    seed: int = 42
):
    os.makedirs(out_dir, exist_ok=True)
    
    with log_step("Streaming S1 IDs", logger):
        all_s1_ids = list(stream_entity_ids(s1_path))
        
    logger.info(f"Loaded {len(all_s1_ids):,} S1 IDs.")
    
    with log_step("Creating Deterministic Split", logger):
        train_s1_ids, val_s1_ids = create_deterministic_split(
            all_s1_ids, val_ratio=val_ratio, seed=seed
        )
        
    set_train = set(train_s1_ids)
    set_val = set(val_s1_ids)
    assert len(set_train & set_val) == 0, "Train and Val S1 IDs must be strictly disjoint!"
    assert len(set_train) + len(set_val) == len(all_s1_ids), "All S1 IDs must be accounted for!"
    
    # Save split metadata
    train_txt = os.path.join(out_dir, "train_s1_ids.txt")
    val_txt = os.path.join(out_dir, "val_s1_ids.txt")
    
    with log_step("Writing Split ID Files", logger):
        with open(train_txt, "w", encoding="utf-8") as f:
            for s1_id in train_s1_ids:
                f.write(s1_id + "\n")
                
        with open(val_txt, "w", encoding="utf-8") as f:
            for s1_id in val_s1_ids:
                f.write(s1_id + "\n")
                
    summary_path = os.path.join(out_dir, "validation_split_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split_name", "s1_count", "percentage", "seed", "file_path"])
        writer.writerow(["train", len(train_s1_ids), f"{len(train_s1_ids)/len(all_s1_ids)*100:.2f}%", seed, train_txt])
        writer.writerow(["validation", len(val_s1_ids), f"{len(val_s1_ids)/len(all_s1_ids)*100:.2f}%", seed, val_txt])
        writer.writerow(["total", len(all_s1_ids), "100.00%", seed, s1_path])
        
    logger.info(f"Split completed successfully. Saved to {summary_path}")

if __name__ == "__main__":
    generate_and_save_split()
