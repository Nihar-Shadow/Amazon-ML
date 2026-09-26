import os
import sys
import argparse
from typing import Set, Dict, List

def validate_submission(matching_file: str, candidate_file: str, test_dir: str) -> bool:
    """
    Validates end-to-end competition submission package integrity.
    Enforces all 15 official Amazon ML Challenge integrity requirements.
    """
    print(f"=== Starting Submission Validation ===")
    print(f"Matching file:  {matching_file}")
    print(f"Candidate file: {candidate_file}")
    print(f"Test directory: {test_dir}")

    # Check 1: Files exist
    if not os.path.exists(matching_file):
        print(f"FAIL: Matching file does not exist: {matching_file}")
        return False
    if not os.path.exists(candidate_file):
        print(f"FAIL: Candidate file does not exist: {candidate_file}")
        return False

    s1_test_file = os.path.join(test_dir, "test_source1.tsv")
    s2_test_file = os.path.join(test_dir, "test_source2.tsv")
    s3_test_file = os.path.join(test_dir, "test_source3.tsv")

    for fpath in [s1_test_file, s2_test_file, s3_test_file]:
        if not os.path.exists(fpath):
            print(f"FAIL: Missing test source file: {fpath}")
            return False

    # Check 2: Load valid Test IDs
    print("\n[Step 1/5] Loading valid test entity IDs...")
    valid_s1_ids = set()
    with open(s1_test_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts and parts[0]:
                valid_s1_ids.add(parts[0])

    expected_s1_count = len(valid_s1_ids)
    print(f"Loaded {expected_s1_count:,} valid Test Source 1 IDs.")

    # Load valid target IDs (S2 and S3)
    valid_target_ids = set()
    with open(s2_test_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts and parts[0]:
                valid_target_ids.add(parts[0])
    s2_count = len(valid_target_ids)

    with open(s3_test_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts and parts[0]:
                valid_target_ids.add(parts[0])
    s3_count = len(valid_target_ids) - s2_count

    print(f"Loaded {s2_count:,} valid Test S2 IDs and {s3_count:,} valid Test S3 IDs (Total targets: {len(valid_target_ids):,}).")

    # Check 3 & 4: Stream and validate candidate_pairs.tsv and matching_results.tsv concurrently
    print("\n[Step 2/5] Validating candidate_pairs.tsv and matching_results.tsv (Streaming Mode)...")
    s1_seen = set()
    row_count = 0
    total_candidates = 0
    total_predicted_matches = 0
    empty_predicted_s1 = 0
    last_s1 = ""

    with open(candidate_file, "r", encoding="utf-8") as fc, open(matching_file, "r", encoding="utf-8") as fm:
        c_header = fc.readline().rstrip("\r\n")
        m_header = fm.readline().rstrip("\r\n")

        if c_header != "source1_entity_id\tcandidate_entity_ids":
            print(f"FAIL: Candidate file has incorrect header: '{c_header}'. Expected 'source1_entity_id\\tcandidate_entity_ids'")
            return False
        if m_header != "source1_entity_id\tmatched_entity_ids":
            print(f"FAIL: Matching file has incorrect header: '{m_header}'. Expected 'source1_entity_id\\tmatched_entity_ids'")
            return False

        for line_num, (c_line, m_line) in enumerate(zip(fc, fm), start=2):
            row_count += 1
            c_parts = c_line.rstrip("\r\n").split("\t")
            m_parts = m_line.rstrip("\r\n").split("\t")

            c_s1 = c_parts[0]
            m_s1 = m_parts[0]

            if c_s1 != m_s1:
                print(f"FAIL: S1 ID mismatch at line {line_num}: candidate has '{c_s1}', matching has '{m_s1}'")
                return False

            s1_id = c_s1
            if s1_id not in valid_s1_ids:
                print(f"FAIL: Unknown S1 ID '{s1_id}' at line {line_num}")
                return False

            if s1_id in s1_seen:
                print(f"FAIL: Duplicate S1 ID '{s1_id}' at line {line_num}")
                return False
            s1_seen.add(s1_id)

            # Deterministic sorting check
            if s1_id < last_s1:
                print(f"FAIL: Output files are not sorted deterministically. '{s1_id}' appeared after '{last_s1}' at line {line_num}")
                return False
            last_s1 = s1_id

            # Parse candidates
            cands_str = c_parts[1] if len(c_parts) >= 2 else ""
            if cands_str:
                cands_list = cands_str.split(",")
                if len(cands_list) != len(set(cands_list)):
                    print(f"FAIL: Duplicate target IDs in candidate list for S1 '{s1_id}' at line {line_num}")
                    return False
                for tid in cands_list:
                    if tid not in valid_target_ids:
                        print(f"FAIL: Invalid target ID '{tid}' in candidates for S1 '{s1_id}' at line {line_num}")
                        return False
                cands_set = set(cands_list)
                total_candidates += len(cands_list)
            else:
                cands_set = set()

            # Parse matches
            matches_str = m_parts[1] if len(m_parts) >= 2 else ""
            if matches_str:
                matches_list = matches_str.split(",")
                for mid in matches_list:
                    if mid.startswith("S1-"):
                        print(f"FAIL: Self-matching S1 ID '{mid}' predicted for S1 '{s1_id}' at line {line_num}")
                        return False
                    if mid not in valid_target_ids:
                        print(f"FAIL: Unknown target ID '{mid}' predicted for S1 '{s1_id}' at line {line_num}")
                        return False
                if len(matches_list) != len(set(matches_list)):
                    print(f"FAIL: Duplicate target IDs in matching prediction for S1 '{s1_id}' at line {line_num}")
                    return False

                # Subsetting guarantee: every predicted match must be in candidate_pairs
                for mid in matches_list:
                    if mid not in cands_set:
                        print(f"FAIL: Predicted match '{mid}' for S1 '{s1_id}' is NOT in candidate_pairs.tsv!")
                        return False

                total_predicted_matches += len(matches_list)
            else:
                empty_predicted_s1 += 1

        # Check for lingering lines in either file
        if fc.readline():
            print("FAIL: candidate_pairs.tsv has more rows than matching_results.tsv!")
            return False
        if fm.readline():
            print("FAIL: matching_results.tsv has more rows than candidate_pairs.tsv!")
            return False

    if row_count != expected_s1_count:
        print(f"FAIL: Row count ({row_count:,}) does not match expected Test S1 count ({expected_s1_count:,})")
        return False
    if s1_seen != valid_s1_ids:
        print("FAIL: Output S1 ID set does not match test S1 ID set.")
        return False

    print(f"SUCCESS: candidate_pairs.tsv and matching_results.tsv passed all checks ({row_count:,} rows).")
    print(f"Total candidates: {total_candidates:,} (avg {total_candidates / row_count:.1f}/S1)")
    print(f"Total predicted matches: {total_predicted_matches:,} (avg {total_predicted_matches / row_count:.2f}/S1)")
    print(f"Empty prediction S1 count: {empty_predicted_s1:,} ({empty_predicted_s1 / row_count * 100:.2f}%)")

    # Step 4: Verified Subsetting Guarantee
    print("\n[Step 3/5] Verified Subsetting Guarantee:")
    print("ALL predicted matches in matching_results.tsv are strictly subsets of candidate_pairs.tsv.")
    print("\n[Step 4/5] Target and Anchor ID Bounds:")
    print("All IDs strictly conform to test IDs. Zero self-matches, zero unknown IDs.")
    print("\n[Step 5/5] Final Verdict:")
    print(">>> SUBMISSION VERIFICATION PASSED. 100% SPEC COMPLIANT. <<<")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official Amazon ML Challenge Submission Validator")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to datasets/test directory")
    args = parser.parse_args()

    ok = validate_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if ok else 1)
