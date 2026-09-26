import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import pickle
from typing import Dict, List, Set, Any
import pandas as pd

from src.candidate_generation.normalizer import normalize_entity_record
from src.candidate_generation.blocking import BlockingIndex
from src.candidate_generation.candidate_refiner import CandidateRefiner
from src.matching_features.feature_schema import FEATURE_COLUMNS
from src.matching_features.feature_extractor import PairwiseFeatureExtractor
from src.matching_features.frequency_features import FrequencyFeatureStore
from src.matching_model.trainer import predict_probabilities
from src.utils.logging_utils import get_logger, get_memory_usage_mb

logger = get_logger("benchmark_test_throughput")

def benchmark_test_sample(sample_size: int = 2000):
    start_time = time.time()
    logger.info(f"Benchmarking test pipeline on {sample_size:,} test S1 entities...")

    # Load frozen production model and frequency store
    with open("models/production_matching_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/production_frequency_store.pkl", "rb") as f:
        freq_store = pickle.load(f)

    extractor = PairwiseFeatureExtractor(frequency_store=freq_store)

    # 1. Load sample S1 records
    s1_sample = []
    with open("datasets/test/test_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                s1_sample.append(normalize_entity_record(parts[0], parts[1], parts[2], parts[3]))
            if len(s1_sample) >= sample_size:
                break

    # 2. Load target pool for sample (e.g. 50,000 S2 and 50,000 S3)
    target_records = {}
    exp013_strategies = [
        "exact_name", "sorted_name_tokens", "first_two_name_tokens", "longest_name_token",
        "name_prefix_6", "name_address_combo", "address_exact_combo", "name_leet_compact",
        "name_dba_alias", "address_salient_combo"
    ]
    index = BlockingIndex(max_block_size=500)

    t_count = 0
    with open("datasets/test/test_source2.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                rec = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                target_records[parts[0]] = rec
                index.add_target_record(rec, strategies=exp013_strategies)
                t_count += 1
            if t_count >= 50000:
                break

    t3_count = 0
    with open("datasets/test/test_source3.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                rec = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                target_records[parts[0]] = rec
                index.add_target_record(rec, strategies=exp013_strategies)
                t3_count += 1
            if t3_count >= 50000:
                break

    logger.info(f"Indexed {len(target_records):,} target records.")

    # 3. Generate candidates + extract features + infer
    refiner = CandidateRefiner(min_score_threshold=0.15, max_candidates_per_query=50)

    total_candidates = 0
    total_matches = 0
    probs_list = []

    pipe_start = time.time()
    for q_rec in s1_sample:
        candidates_with_prov = index.retrieve_candidates_for_query(
            q_rec, strategies=exp013_strategies, return_provenance=True
        )
        if not candidates_with_prov:
            continue

        cand_targets = {tid: target_records[tid] for tid in candidates_with_prov.keys() if tid in target_records}
        refined_tids = refiner.refine_candidates_for_query(
            q_rec, cand_targets, provenance_map=candidates_with_prov
        )

        total_candidates += len(refined_tids)
        if not refined_tids:
            continue

        # Extract features
        pair_rows = []
        for tid in refined_tids:
            t_rec = target_records[tid]
            prov_strats = candidates_with_prov.get(tid, set())
            feats = extractor.extract_pair_features(q_rec, t_rec, provenance_strategies=prov_strats)
            pair_rows.append(feats)

        df_chunk = pd.DataFrame(pair_rows)
        probs, _ = predict_probabilities(model, df_chunk)
        probs_list.extend(probs)
        matches = [tid for tid, p in zip(refined_tids, probs) if p >= 0.70]
        total_matches += len(matches)

    pipe_time = round(time.time() - pipe_start, 2)
    q_per_sec = round(len(s1_sample) / pipe_time, 1)
    cand_per_sec = round(total_candidates / pipe_time, 1)

    logger.info(f"Processed {len(s1_sample):,} queries in {pipe_time}s ({q_per_sec} queries/s, {cand_per_sec} candidates/s).")
    logger.info(f"Total Candidates: {total_candidates:,} (avg {total_candidates/len(s1_sample):.1f}/S1).")
    logger.info(f"Total Predicted Matches: {total_matches:,} (avg {total_matches/len(s1_sample):.2f}/S1).")
    logger.info(f"Peak RAM: {get_memory_usage_mb():.1f} MB.")

if __name__ == "__main__":
    benchmark_test_sample()
