import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import gc
import pickle
import zipfile
import argparse
from typing import Dict, List, Set, Any, Tuple
import numpy as np
import pandas as pd

from src.candidate_generation.normalizer import normalize_entity_record, normalize_country
from src.candidate_generation.blocking import BlockingIndex
from src.candidate_generation.candidate_refiner import CandidateRefiner
from src.matching_features.feature_schema import FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS, TOTAL_FEATURE_COUNT
from src.matching_features.feature_extractor import PairwiseFeatureExtractor
from src.matching_model.trainer import predict_probabilities
from src.utils.logging_utils import get_logger, get_memory_usage_mb
from utils.validate_submission import validate_submission

logger = get_logger("run_final_submission")

EXP013_STRATEGIES = [
    "exact_name",
    "sorted_name_tokens",
    "first_two_name_tokens",
    "longest_name_token",
    "name_prefix_6",
    "name_address_combo",
    "address_exact_combo",
    "name_leet_compact",
    "name_dba_alias",
    "address_salient_combo",
]

DECISION_THRESHOLD = 0.70

def count_file_lines(file_path: str) -> int:
    """Fast line count excluding header."""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        for _ in f:
            count += 1
    return count

def run_submission_pipeline(
    test_dir: str = "datasets/test",
    model_path: str = "models/production_matching_model.pkl",
    freq_path: str = "models/production_frequency_store.pkl",
    output_dir: str = "output",
    team_name: str = "amazon_ml_submission",
    max_queries: int = None,
    batch_size: int = 5000,
    create_zip: bool = True
) -> Dict[str, Any]:
    """
    Executes the end-to-end production inference and submission assembly pipeline.
    """
    total_pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 6: END-TO-END TEST INFERENCE & SUBMISSION ASSEMBLY")
    logger.info("=" * 60)

    # 1. Load model and frequency store
    logger.info(f"Loading production model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Loading production frequency store from {freq_path}...")
    with open(freq_path, "rb") as f:
        freq_store = pickle.load(f)

    extractor = PairwiseFeatureExtractor(frequency_store=freq_store)
    logger.info(f"Feature Schema Version: {extractor.schema_version} ({len(extractor.feature_columns)} features).")
    assert extractor.schema_version == "phase4-v1", f"Invalid schema version: {extractor.schema_version}"
    assert len(extractor.feature_columns) == 87, f"Expected 87 features, got {len(extractor.feature_columns)}"

    # 2. Count test records
    s1_test_file = os.path.join(test_dir, "test_source1.tsv")
    s2_test_file = os.path.join(test_dir, "test_source2.tsv")
    s3_test_file = os.path.join(test_dir, "test_source3.tsv")

    logger.info("Counting test dataset records...")
    s1_total = count_file_lines(s1_test_file)
    s2_total = count_file_lines(s2_test_file)
    s3_total = count_file_lines(s3_test_file)

    logger.info(f"Test Source1 Count: {s1_total:,}")
    logger.info(f"Test Source2 Count: {s2_total:,}")
    logger.info(f"Test Source3 Count: {s3_total:,}")
    logger.info(f"Total Target Records: {s2_total + s3_total:,}")

    os.makedirs(output_dir, exist_ok=True)
    temp_cand_file = os.path.join(output_dir, "temp_candidate_pairs.tsv")
    temp_match_file = os.path.join(output_dir, "temp_matching_results.tsv")

    # 3. Discover countries in test dataset (O(1) memory pass)
    logger.info("Discovering country distributions in test data...")
    country_query_counts: Dict[str, int] = {}
    processed_queries = 0
    with open(s1_test_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                c = normalize_country(parts[3])
                country_query_counts[c] = country_query_counts.get(c, 0) + 1
                processed_queries += 1
                if max_queries and processed_queries >= max_queries:
                    break

    countries = sorted(list(country_query_counts.keys()))
    logger.info(f"Discovered {len(countries)} countries in Test S1: {countries}")
    for c in countries:
        logger.info(f"  - {c}: {country_query_counts[c]:,} S1 queries")

    refiner = CandidateRefiner(min_score_threshold=0.15, max_candidates_per_query=50)

    total_candidates_all = 0
    candidate_counts_list = []
    zero_candidate_count = 0
    total_matches_all = 0
    match_counts_list = []
    empty_match_count = 0
    probabilities_all = []
    country_diag = {}

    # Open temp output files
    with open(temp_cand_file, "w", encoding="utf-8") as f_cand, \
         open(temp_match_file, "w", encoding="utf-8") as f_match:

        # Process country-by-country for zero cross-country loss & bounded partition memory
        for country in countries:
            expected_queries = country_query_counts[country]
            if expected_queries == 0:
                continue

            logger.info(f"\nProcessing Country: {country} ({expected_queries:,} S1 queries)...")
            country_start = time.time()

            # Index Target records for this country
            index = BlockingIndex(max_block_size=500)
            target_raw = {}  # tid -> (raw_name, raw_addr, raw_country)

            # Load S2 targets for country (normalized country matching)
            t_s2_count = 0
            with open(s2_test_file, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4 and normalize_country(parts[3]) == country:
                        rec = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                        target_raw[parts[0]] = (parts[1], parts[2], parts[3])
                        index.add_target_record(rec, strategies=EXP013_STRATEGIES)
                        t_s2_count += 1

            # Load S3 targets for country (normalized country matching)
            t_s3_count = 0
            with open(s3_test_file, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4 and normalize_country(parts[3]) == country:
                        rec = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                        target_raw[parts[0]] = (parts[1], parts[2], parts[3])
                        index.add_target_record(rec, strategies=EXP013_STRATEGIES)
                        t_s3_count += 1

            logger.info(f"Indexed {len(target_raw):,} targets for {country} (S2={t_s2_count:,}, S3={t_s3_count:,}) in {time.time() - country_start:.1f}s.")
            logger.info(f"Current RAM: {get_memory_usage_mb():.1f} MB.")

            c_candidates = 0
            c_matches = 0

            # Stream S1 queries for this country line-by-line (bounded O(1) S1 query memory)
            q_idx = 0
            with open(s1_test_file, "r", encoding="utf-8") as f_s1:
                f_s1.readline()
                q_total_scanned = 0
                for line in f_s1:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4:
                        q_total_scanned += 1
                        if max_queries and q_total_scanned > max_queries:
                            break
                        if normalize_country(parts[3]) == country:
                            q_idx += 1
                            q_rec = normalize_entity_record(parts[0], parts[1], parts[2], parts[3])
                            q_id = q_rec["entity_id"]

                            cands_prov = index.retrieve_candidates_for_query(
                                q_rec, strategies=EXP013_STRATEGIES, return_provenance=True
                            )

                            if not cands_prov:
                                # Zero candidates
                                f_cand.write(f"{q_id}\t\n")
                                f_match.write(f"{q_id}\t\n")
                                candidate_counts_list.append(0)
                                match_counts_list.append(0)
                                zero_candidate_count += 1
                                empty_match_count += 1
                                if q_idx % 25000 == 0:
                                    logger.info(f"  [{country}] Processed {q_idx:,}/{expected_queries:,} queries... (RAM: {get_memory_usage_mb():.1f} MB)")
                                continue

                            cand_tgts = {}
                            for tid in cands_prov.keys():
                                if tid in target_raw:
                                    rn, ra, rc = target_raw[tid]
                                    cand_tgts[tid] = normalize_entity_record(tid, rn, ra, rc)

                            refined_tids = refiner.refine_candidates_for_query(
                                q_rec, cand_tgts, provenance_map=cands_prov
                            )

                            cand_count = len(refined_tids)
                            candidate_counts_list.append(cand_count)
                            total_candidates_all += cand_count
                            c_candidates += cand_count

                            if not refined_tids:
                                f_cand.write(f"{q_id}\t\n")
                                f_match.write(f"{q_id}\t\n")
                                zero_candidate_count += 1
                                empty_match_count += 1
                                match_counts_list.append(0)
                                if q_idx % 25000 == 0:
                                    logger.info(f"  [{country}] Processed {q_idx:,}/{expected_queries:,} queries... (RAM: {get_memory_usage_mb():.1f} MB)")
                                continue

                            # Write candidate pairs (strictly deterministic order)
                            sorted_cands = sorted(refined_tids)
                            f_cand.write(f"{q_id}\t{','.join(sorted_cands)}\n")

                            # Extract features for all candidates
                            pair_features = []
                            for tid in sorted_cands:
                                t_rec = cand_tgts[tid]
                                prov_strats = cands_prov.get(tid, set())
                                feats = extractor.extract_pair_features(q_rec, t_rec, provenance_strategies=prov_strats)
                                pair_features.append([feats[col] for col in FEATURE_COLUMNS])

                            # Batch model inference
                            X_batch = np.array(pair_features, dtype=np.float32)
                            probs = model.predict_proba(X_batch)[:, 1]
                            probabilities_all.extend(probs)

                            # Filter by frozen threshold 0.70
                            matched_tids = [tid for tid, p in zip(sorted_cands, probs) if p >= DECISION_THRESHOLD]
                            matched_tids = sorted(list(set(matched_tids)))  # Deduplicate & sort deterministically

                            match_count = len(matched_tids)
                            match_counts_list.append(match_count)
                            total_matches_all += match_count
                            c_matches += match_count

                            if match_count == 0:
                                empty_match_count += 1
                                f_match.write(f"{q_id}\t\n")
                            else:
                                f_match.write(f"{q_id}\t{','.join(matched_tids)}\n")

                            if q_idx % 25000 == 0:
                                logger.info(f"  [{country}] Processed {q_idx:,}/{expected_queries:,} queries... (RAM: {get_memory_usage_mb():.1f} MB)")

            country_diag[country] = {
                "queries": q_idx,
                "candidates": c_candidates,
                "matches": c_matches,
                "targets": len(target_raw)
            }

            # Cleanup country index and target records
            del index
            del target_raw
            gc.collect()


    logger.info("\nCountry Processing Complete. Assembling deterministically sorted final submission files...")

    # 4. Sort and assemble final submission files
    final_cand_file = os.path.join(output_dir, "candidate_pairs.tsv")
    final_match_file = os.path.join(output_dir, "matching_results.tsv")

    # Read and sort temp candidate pairs by S1 ID
    logger.info("Sorting candidate_pairs.tsv by source1_entity_id...")
    with open(temp_cand_file, "r", encoding="utf-8") as f_in:
        cand_lines = f_in.readlines()
    cand_lines.sort(key=lambda line: line.split("\t")[0])

    with open(final_cand_file, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        f_out.writelines(cand_lines)

    # Read and sort temp matching results by S1 ID
    logger.info("Sorting matching_results.tsv by source1_entity_id...")
    with open(temp_match_file, "r", encoding="utf-8") as f_in:
        match_lines = f_in.readlines()
    match_lines.sort(key=lambda line: line.split("\t")[0])

    with open(final_match_file, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        f_out.writelines(match_lines)

    # Clean up temp files
    if os.path.exists(temp_cand_file):
        os.remove(temp_cand_file)
    if os.path.exists(temp_match_file):
        os.remove(temp_match_file)

    logger.info(f"Generated {final_cand_file} ({len(cand_lines):,} rows).")
    logger.info(f"Generated {final_match_file} ({len(match_lines):,} rows).")

    # 5. Submission Validation
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING OFFICIAL SUBMISSION VALIDATION")
    logger.info("=" * 60)

    val_start = time.time()
    validation_passed = validate_submission(final_match_file, final_cand_file, test_dir)
    val_time = round(time.time() - val_start, 2)
    logger.info(f"Validation finished in {val_time}s. Result: {'PASS' if validation_passed else 'FAIL'}")

    if not validation_passed:
        logger.error("SUBMISSION VALIDATION FAILED! Halting submission package assembly.")
        sys.exit(1)

    # 6. Package Submission ZIP
    zip_path = None
    if create_zip:
        logger.info("\n" + "=" * 60)
        logger.info("PACKAGING FINAL SUBMISSION ZIP")
        logger.info("=" * 60)

        zip_filename = f"{team_name}_submission.zip"
        zip_path = os.path.join(os.getcwd(), zip_filename)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # 1. output/
            zipf.write(final_match_file, arcname="output/matching_results.tsv")
            zipf.write(final_cand_file, arcname="output/candidate_pairs.tsv")

            # 2. code/business_entity_resolution/src/
            src_dir = os.path.join(os.getcwd(), "src")
            for root, dirs, files in os.walk(src_dir):
                if "__pycache__" in root:
                    continue
                for file in files:
                    if file.endswith(".py"):
                        full_p = os.path.join(root, file)
                        rel_p = os.path.relpath(full_p, os.getcwd())
                        arc_name = os.path.join("code", "business_entity_resolution", rel_p)
                        zipf.write(full_p, arcname=arc_name)

            # 3. code/business_entity_resolution/README.md & requirements.txt
            readme_p = os.path.join(os.getcwd(), "README.md")
            req_p = os.path.join(os.getcwd(), "requirements.txt")
            if os.path.exists(readme_p):
                zipf.write(readme_p, arcname="code/business_entity_resolution/README.md")
            if os.path.exists(req_p):
                zipf.write(req_p, arcname="code/business_entity_resolution/requirements.txt")

            # 4. Documentation_template.md
            doc_p = os.path.join(os.getcwd(), "Documentation_template.md")
            if os.path.exists(doc_p):
                zipf.write(doc_p, arcname="Documentation_template.md")

        zip_size_mb = round(os.path.getsize(zip_path) / (1024 * 1024), 2)
        logger.info(f"Successfully created submission package: {zip_path} ({zip_size_mb} MB).")

    total_pipeline_time = round(time.time() - total_pipeline_start, 2)
    peak_ram = get_memory_usage_mb()

    # Diagnostics Calculation
    cand_arr = np.array(candidate_counts_list)
    match_arr = np.array(match_counts_list)
    prob_arr = np.array(probabilities_all) if probabilities_all else np.array([0.0])

    num_ge_threshold = int(np.sum(prob_arr >= DECISION_THRESHOLD))
    num_lt_threshold = int(np.sum(prob_arr < DECISION_THRESHOLD))

    diagnostics = {
        "production_model": "HistGradientBoostingClassifier",
        "feature_schema": "phase4-v1",
        "feature_count": 87,
        "frozen_threshold": DECISION_THRESHOLD,
        "test_s1_count": s1_total,
        "test_s2_count": s2_total,
        "test_s3_count": s3_total,
        "processed_s1_count": len(candidate_counts_list),
        "total_candidate_pairs": total_candidates_all,
        "candidate_stats": {
            "mean": float(np.mean(cand_arr)) if len(cand_arr) else 0.0,
            "median": float(np.median(cand_arr)) if len(cand_arr) else 0.0,
            "p90": float(np.percentile(cand_arr, 90)) if len(cand_arr) else 0.0,
            "p95": float(np.percentile(cand_arr, 95)) if len(cand_arr) else 0.0,
            "p99": float(np.percentile(cand_arr, 99)) if len(cand_arr) else 0.0,
            "max": int(np.max(cand_arr)) if len(cand_arr) else 0,
            "zero_candidate_count": zero_candidate_count,
            "zero_candidate_pct": round(zero_candidate_count / max(1, len(cand_arr)) * 100, 2)
        },
        "prediction_stats": {
            "total_matches": total_matches_all,
            "mean": float(np.mean(match_arr)) if len(match_arr) else 0.0,
            "median": float(np.median(match_arr)) if len(match_arr) else 0.0,
            "p95": float(np.percentile(match_arr, 95)) if len(match_arr) else 0.0,
            "max": int(np.max(match_arr)) if len(match_arr) else 0,
            "empty_prediction_count": empty_match_count,
            "empty_prediction_pct": round(empty_match_count / max(1, len(match_arr)) * 100, 2)
        },
        "score_distribution": {
            "min": float(np.min(prob_arr)),
            "p01": float(np.percentile(prob_arr, 1)),
            "p05": float(np.percentile(prob_arr, 5)),
            "p25": float(np.percentile(prob_arr, 25)),
            "median": float(np.median(prob_arr)),
            "p75": float(np.percentile(prob_arr, 75)),
            "p95": float(np.percentile(prob_arr, 95)),
            "p99": float(np.percentile(prob_arr, 99)),
            "max": float(np.max(prob_arr)),
            "ge_0_70": num_ge_threshold,
            "lt_0_70": num_lt_threshold
        },
        "country_diagnostics": country_diag,
        "validation_passed": validation_passed,
        "runtime_seconds": total_pipeline_time,
        "peak_ram_mb": peak_ram,
        "output_files": [final_cand_file, final_match_file],
        "zip_path": zip_path
    }

    logger.info("\n" + "=" * 60)
    logger.info("SUBMISSION PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total S1 Queries Processed: {diagnostics['processed_s1_count']:,}")
    logger.info(f"Total Candidate Pairs:      {total_candidates_all:,} (avg {diagnostics['candidate_stats']['mean']:.2f}/S1)")
    logger.info(f"Zero-Candidate Rate:        {diagnostics['candidate_stats']['zero_candidate_pct']}% ({zero_candidate_count:,} queries)")
    logger.info(f"Total Predicted Matches:    {total_matches_all:,} (avg {diagnostics['prediction_stats']['mean']:.2f}/S1)")
    logger.info(f"Empty Prediction Rate:      {diagnostics['prediction_stats']['empty_prediction_pct']}% ({empty_match_count:,} queries)")
    logger.info(f"Candidates >= 0.70:         {num_ge_threshold:,} | < 0.70: {num_lt_threshold:,}")
    logger.info(f"Runtime:                    {total_pipeline_time:.2f}s")
    logger.info(f"Peak RAM:                   {peak_ram:.1f} MB")
    logger.info(f"Submission Package:         {zip_path}")
    logger.info("=" * 60)

    return diagnostics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Production Submission Pipeline")
    parser.add_argument("--test-dir", default="datasets/test", help="Path to test datasets directory")
    parser.add_argument("--model-path", default="models/production_matching_model.pkl", help="Path to trained model")
    parser.add_argument("--freq-path", default="models/production_frequency_store.pkl", help="Path to frequency store")
    parser.add_argument("--output-dir", default="output", help="Directory for final output TSVs")
    parser.add_argument("--team-name", default="amazon_ml_submission", help="Team name prefix for ZIP file")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional query cap for smoke tests/profiling")
    parser.add_argument("--batch-size", type=int, default=5000, help="Inference batch size")
    parser.add_argument("--skip-zip", action="store_true", help="Skip ZIP archive packaging")
    args = parser.parse_args()

    run_submission_pipeline(
        test_dir=args.test_dir,
        model_path=args.model_path,
        freq_path=args.freq_path,
        output_dir=args.output_dir,
        team_name=args.team_name,
        max_queries=args.max_queries,
        batch_size=args.batch_size,
        create_zip=not args.skip_zip
    )
