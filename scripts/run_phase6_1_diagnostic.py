import sys
import os
import time
import psutil
import gc
import pickle
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.candidate_generation.normalizer import normalize_entity_record, normalize_country
from src.candidate_generation.blocking import BlockingIndex
from src.candidate_generation.candidate_refiner import CandidateRefiner
from src.matching_features.feature_schema import FEATURE_COLUMNS
from src.matching_features.feature_extractor import PairwiseFeatureExtractor
from src.utils.logging_utils import get_memory_usage_mb

def run_diagnostic():
    s1_test_file = "datasets/test/test_source1.tsv"
    s2_test_file = "datasets/test/test_source2.tsv"
    s3_test_file = "datasets/test/test_source3.tsv"
    model_path = "models/production_matching_model.pkl"
    freq_path = "models/production_frequency_store.pkl"

    print("Loading production model & frequency store...", flush=True)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(freq_path, "rb") as f:
        freq_store = pickle.load(f)

    extractor = PairwiseFeatureExtractor(frequency_store=freq_store)
    refiner = CandidateRefiner(min_score_threshold=0.15, max_candidates_per_query=50)

    EXP013_STRATEGIES = [
        "exact_name", "sorted_name_tokens", "first_two_name_tokens",
        "longest_name_token", "name_prefix_6", "name_address_combo",
        "address_exact_combo", "name_leet_compact", "name_dba_alias",
        "address_salient_combo",
    ]

    print("Collecting first 100 queries for FRANCE, INDIA, US...", flush=True)
    queries_by_country = {"FRANCE": [], "INDIA": [], "US": []}
    with open(s1_test_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                c = normalize_country(parts[3])
                if c in queries_by_country and len(queries_by_country[c]) < 100:
                    queries_by_country[c].append(normalize_entity_record(parts[0], parts[1], parts[2], parts[3]))
            if all(len(v) == 100 for v in queries_by_country.values()):
                break

    for c, q_list in queries_by_country.items():
        print(f"  {c}: {len(q_list)} queries collected", flush=True)

    diag_results = {}

    for country in ["FRANCE", "INDIA", "US"]:
        print(f"\n--- Testing Country: {country} ---", flush=True)
        t0 = time.time()
        c_queries = queries_by_country[country]
        
        index = BlockingIndex(max_block_size=500)
        target_raw = {}  # tid -> (name, addr, country)
        
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
                    
        print(f"[{country}] Targets loaded: S2={t_s2_count:,}, S3={t_s3_count:,}, Total={len(target_raw):,}", flush=True)
        print(f"[{country}] Current RAM: {get_memory_usage_mb():.1f} MB", flush=True)
        
        c_candidates = 0
        c_matches = 0
        cross_country_violations = 0
        
        for q_rec in c_queries:
            cands_prov = index.retrieve_candidates_for_query(q_rec, strategies=EXP013_STRATEGIES, return_provenance=True)
            if not cands_prov:
                continue
            
            # Materialize normalized records on-the-fly only for candidates
            cand_tgts = {}
            for tid in cands_prov:
                if tid in target_raw:
                    rn, ra, rc = target_raw[tid]
                    cand_tgts[tid] = normalize_entity_record(tid, rn, ra, rc)
                    
            refined = refiner.refine_candidates_for_query(q_rec, cand_tgts, provenance_map=cands_prov)
            c_candidates += len(refined)
            
            # Verify no cross-country
            for tid in refined:
                if cand_tgts[tid]["country"] != country:
                    cross_country_violations += 1
                    
            if not refined:
                continue
                
            pair_features = []
            for tid in refined:
                t_rec = cand_tgts[tid]
                feats = extractor.extract_pair_features(q_rec, t_rec, provenance_strategies=cands_prov.get(tid, set()))
                pair_features.append([feats[col] for col in FEATURE_COLUMNS])
                
            X_batch = np.array(pair_features, dtype=np.float32)
            probs = model.predict_proba(X_batch)[:, 1]
            matches = [tid for tid, p in zip(refined, probs) if p >= 0.70]
            c_matches += len(matches)
            
        peak_rss = get_memory_usage_mb()
        duration = time.time() - t0
        
        diag_results[country] = {
            "s1_queries": len(c_queries),
            "s2_targets": t_s2_count,
            "s3_targets": t_s3_count,
            "total_targets": len(target_raw),
            "candidates": c_candidates,
            "matches": c_matches,
            "cross_country_violations": cross_country_violations,
            "peak_rss_mb": round(peak_rss, 1),
            "runtime_s": round(duration, 2)
        }
        print(f"[{country}] Candidates={c_candidates}, Matches={c_matches}, Violations={cross_country_violations}, Peak RAM={peak_rss:.1f} MB, Time={duration:.2f}s", flush=True)
        
        del index
        del target_raw
        gc.collect()

    print("\n" + "=" * 60, flush=True)
    print("PHASE 6.1 COUNTRY DIAGNOSTIC RESULTS", flush=True)
    print("=" * 60, flush=True)
    for c, r in diag_results.items():
        print(f"Country: {c}", flush=True)
        print(f"  S1 Queries:               {r['s1_queries']}", flush=True)
        print(f"  S2 Targets:               {r['s2_targets']:,}", flush=True)
        print(f"  S3 Targets:               {r['s3_targets']:,}", flush=True)
        print(f"  Total Targets:            {r['total_targets']:,}", flush=True)
        print(f"  Candidates Produced:      {r['candidates']}", flush=True)
        print(f"  Matches Produced (>=0.70):{r['matches']}", flush=True)
        print(f"  Cross-Country Violations: {r['cross_country_violations']}", flush=True)
        print(f"  Peak RSS:                 {r['peak_rss_mb']} MB", flush=True)
        print(f"  Runtime:                  {r['runtime_s']} s", flush=True)
        print("-" * 60, flush=True)

    # Verification assertions
    assert diag_results["FRANCE"]["s2_targets"] > 0, "France S2 targets must be > 0"
    assert diag_results["FRANCE"]["s3_targets"] > 0, "France S3 targets must be > 0"
    assert diag_results["FRANCE"]["candidates"] > 0, "France candidates must be > 0"
    assert diag_results["INDIA"]["s2_targets"] > 0, "India S2 targets must be > 0"
    assert diag_results["INDIA"]["s3_targets"] > 0, "India S3 targets must be > 0"
    assert diag_results["INDIA"]["candidates"] > 0, "India candidates must be > 0"
    assert diag_results["US"]["s2_targets"] > 0, "US S2 targets must be > 0"
    assert diag_results["US"]["candidates"] > 0, "US candidates must be > 0"
    assert all(r["cross_country_violations"] == 0 for r in diag_results.values()), "Cross-country violations must be 0"

    print("ALL DIAGNOSTIC INTEGRITY ASSERTIONS PASSED SUCCESSFULLY!", flush=True)

if __name__ == "__main__":
    run_diagnostic()
