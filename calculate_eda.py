import os
import sys
import csv
import json
import time
from collections import Counter, defaultdict

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

os.makedirs('eda', exist_ok=True)

DATASETS = {
    'train_source1': 'datasets/train/train_source1.tsv',
    'train_source2': 'datasets/train/train_source2.tsv',
    'train_source3': 'datasets/train/train_source3.tsv',
    'train_ground_truth': 'datasets/train/train_ground_truth.tsv',
    'test_source1': 'datasets/test/test_source1.tsv',
    'test_source2': 'datasets/test/test_source2.tsv',
    'test_source3': 'datasets/test/test_source3.tsv'
}

def analyze_entity_file(file_key, file_path):
    print(f"[{file_key}] Analyzing {file_path}...")
    t0 = time.time()
    
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    
    total_rows = 0
    malformed_rows = 0
    first_5_rows = []
    
    missing_counts = {'entity_id': 0, 'business_name': 0, 'business_address': 0, 'country': 0}
    unique_ids = set()
    unique_names = set()
    unique_addrs = set()
    country_counts = Counter()
    
    name_len_min = float('inf')
    name_len_max = 0
    name_len_sum = 0
    name_len_count = 0
    
    addr_len_min = float('inf')
    addr_len_max = 0
    addr_len_sum = 0
    addr_len_count = 0
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        header_line = f.readline()
        header = [c.strip() for c in header_line.rstrip('\r\n').split('\t')]
        
        for line_num, line in enumerate(f, start=2):
            total_rows += 1
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) != 4:
                malformed_rows += 1
                continue
            
            eid, bname, baddr, bcountry = parts
            
            if len(first_5_rows) < 5:
                first_5_rows.append({
                    'entity_id': eid,
                    'business_name': bname,
                    'business_address': baddr,
                    'country': bcountry
                })
            
            # Entity ID
            if not eid or eid.isspace():
                missing_counts['entity_id'] += 1
            else:
                unique_ids.add(hash(eid))
                
            # Business Name
            if not bname or bname.isspace():
                missing_counts['business_name'] += 1
            else:
                unique_names.add(hash(bname))
                l = len(bname)
                name_len_sum += l
                name_len_count += 1
                if l < name_len_min: name_len_min = l
                if l > name_len_max: name_len_max = l
                
            # Business Address
            if not baddr or baddr.isspace():
                missing_counts['business_address'] += 1
            else:
                unique_addrs.add(hash(baddr))
                l = len(baddr)
                addr_len_sum += l
                addr_len_count += 1
                if l < addr_len_min: addr_len_min = l
                if l > addr_len_max: addr_len_max = l
                
            # Country
            if not bcountry or bcountry.isspace():
                missing_counts['country'] += 1
            else:
                country_counts[bcountry] += 1
                
    u_id = len(unique_ids)
    u_name = len(unique_names)
    u_addr = len(unique_addrs)
    
    del unique_ids
    del unique_names
    del unique_addrs
    
    elapsed = round(time.time() - t0, 2)
    print(f"[{file_key}] Completed in {elapsed}s. Rows: {total_rows:,}, Malformed: {malformed_rows}")
    
    return {
        'file_key': file_key,
        'file_path': file_path,
        'file_size_mb': file_size_mb,
        'header': header,
        'total_rows': total_rows,
        'malformed_rows': malformed_rows,
        'first_5_rows': first_5_rows,
        'missing_counts': missing_counts,
        'unique_counts': {
            'entity_id': u_id,
            'business_name': u_name,
            'business_address': u_addr,
            'country': len(country_counts)
        },
        'name_stats': {
            'empty': missing_counts['business_name'],
            'unique': u_name,
            'duplicate': (total_rows - missing_counts['business_name']) - u_name,
            'min_len': 0 if name_len_count == 0 else name_len_min,
            'max_len': name_len_max,
            'avg_len': 0 if name_len_count == 0 else round(name_len_sum / name_len_count, 2)
        },
        'addr_stats': {
            'empty': missing_counts['business_address'],
            'unique': u_addr,
            'duplicate': (total_rows - missing_counts['business_address']) - u_addr,
            'min_len': 0 if addr_len_count == 0 else addr_len_min,
            'max_len': addr_len_max,
            'avg_len': 0 if addr_len_count == 0 else round(addr_len_sum / addr_len_count, 2)
        },
        'country_counts': dict(country_counts),
        'elapsed_sec': elapsed
    }

def analyze_ground_truth(file_path):
    print(f"[train_ground_truth] Analyzing {file_path}...")
    t0 = time.time()
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    
    total_rows = 0
    malformed_rows = 0
    first_5_rows = []
    
    unique_s1 = set()
    match_hist = Counter()
    s2_hist = Counter()
    s3_hist = Counter()
    
    zero_matches = 0
    single_match = 0
    multi_matches = 0
    
    total_s2_matches = 0
    total_s3_matches = 0
    
    # Track target match multiplicity: target_id -> count of s1 that match it
    target_to_s1_count = Counter()
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        header_line = f.readline()
        header = [c.strip() for c in header_line.rstrip('\r\n').split('\t')]
        
        for line_num, line in enumerate(f, start=2):
            total_rows += 1
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) != 2:
                malformed_rows += 1
                continue
                
            s1_id, matches_str = parts[0], parts[1]
            unique_s1.add(hash(s1_id))
            
            if len(first_5_rows) < 5:
                first_5_rows.append({
                    'source1_entity_id': s1_id,
                    'matched_entity_ids': matches_str
                })
                
            matches = [m.strip() for m in matches_str.split(',') if m.strip()]
            num_matches = len(matches)
            match_hist[num_matches] += 1
            
            s2_c = 0
            s3_c = 0
            for m in matches:
                if m.startswith('S2-'):
                    s2_c += 1
                    total_s2_matches += 1
                elif m.startswith('S3-'):
                    s3_c += 1
                    total_s3_matches += 1
                target_to_s1_count[m] += 1
                
            s2_hist[s2_c] += 1
            s3_hist[s3_c] += 1
            
            if num_matches == 0:
                zero_matches += 1
            elif num_matches == 1:
                single_match += 1
            else:
                multi_matches += 1
                
    multi_s1_targets = {k: v for k, v in target_to_s1_count.items() if v > 1}
    unique_s1_count = len(unique_s1)
    del unique_s1
    
    elapsed = round(time.time() - t0, 2)
    print(f"[train_ground_truth] Completed in {elapsed}s. Rows: {total_rows:,}")
    
    return {
        'file_key': 'train_ground_truth',
        'file_path': file_path,
        'file_size_mb': file_size_mb,
        'header': header,
        'total_rows': total_rows,
        'malformed_rows': malformed_rows,
        'first_5_rows': first_5_rows,
        'unique_s1_count': unique_s1_count,
        'zero_matches': zero_matches,
        'single_match': single_match,
        'multi_matches': multi_matches,
        'total_s2_matches': total_s2_matches,
        'total_s3_matches': total_s3_matches,
        'match_hist': dict(match_hist),
        's2_hist': dict(s2_hist),
        's3_hist': dict(s3_hist),
        'total_unique_matched_entities': len(target_to_s1_count),
        'multi_s1_target_count': len(multi_s1_targets),
        'elapsed_sec': elapsed
    }

def main():
    print("=== STARTING COMPLETE DATASET EDA ===")
    
    # 1. Analyze all 6 source files
    source_results = {}
    for key in ['train_source1', 'train_source2', 'train_source3', 'test_source1', 'test_source2', 'test_source3']:
        source_results[key] = analyze_entity_file(key, DATASETS[key])
        
    # 2. Analyze ground truth
    gt_results = analyze_ground_truth(DATASETS['train_ground_truth'])
    
    # Save results as JSON intermediate for reference
    all_results = {
        'source_files': source_results,
        'ground_truth': gt_results
    }
    with open('eda/eda_metrics_raw.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
        
    print("Metrics collected and raw JSON saved to eda/eda_metrics_raw.json")

if __name__ == '__main__':
    main()
