import os
import sys
import csv
import json
import time
from collections import Counter, defaultdict

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

DATASETS = {
    'train_source1': 'datasets/train/train_source1.tsv',
    'train_source2': 'datasets/train/train_source2.tsv',
    'train_source3': 'datasets/train/train_source3.tsv',
    'train_ground_truth': 'datasets/train/train_ground_truth.tsv',
    'test_source1': 'datasets/test/test_source1.tsv',
    'test_source2': 'datasets/test/test_source2.tsv',
    'test_source3': 'datasets/test/test_source3.tsv'
}

def analyze_source_file(file_path, file_key):
    print(f"Analyzing {file_key} from {file_path}...")
    t0 = time.time()
    
    total_rows = 0
    malformed_rows = 0
    malformed_examples = []
    
    first_5_rows = []
    
    # Missing counters
    missing_counts = {'entity_id': 0, 'business_name': 0, 'business_address': 0, 'country': 0}
    
    # Unique value tracking using sets of hash integers
    unique_ids = set()
    unique_names = set()
    unique_addresses = set()
    country_counter = Counter()
    
    # Length tracking
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
        header = [col.strip() for col in header_line.rstrip('\r\n').split('\t')]
        
        for line_num, line in enumerate(f, start=2):
            total_rows += 1
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) != 4:
                malformed_rows += 1
                if len(malformed_examples) < 5:
                    malformed_examples.append((line_num, len(parts), line[:150]))
                continue
            
            e_id, b_name, b_addr, b_country = parts
            
            if len(first_5_rows) < 5:
                first_5_rows.append({
                    'entity_id': e_id,
                    'business_name': b_name,
                    'business_address': b_addr,
                    'country': b_country
                })
            
            # Entity ID
            if not e_id or e_id.isspace():
                missing_counts['entity_id'] += 1
            else:
                unique_ids.add(hash(e_id))
            
            # Business Name
            if not b_name or b_name.isspace():
                missing_counts['business_name'] += 1
            else:
                unique_names.add(hash(b_name))
                l_name = len(b_name)
                name_len_sum += l_name
                name_len_count += 1
                if l_name < name_len_min: name_len_min = l_name
                if l_name > name_len_max: name_len_max = l_name
            
            # Business Address
            if not b_addr or b_addr.isspace():
                missing_counts['business_address'] += 1
            else:
                unique_addresses.add(hash(b_addr))
                l_addr = len(b_addr)
                addr_len_sum += l_addr
                addr_len_count += 1
                if l_addr < addr_len_min: addr_len_min = l_addr
                if l_addr > addr_len_max: addr_len_max = l_addr
            
            # Country
            if not b_country or b_country.isspace():
                missing_counts['country'] += 1
            else:
                country_counter[b_country] += 1
    
    unique_id_count = len(unique_ids)
    unique_name_count = len(unique_names)
    unique_addr_count = len(unique_addresses)
    
    # Clean up large sets
    del unique_ids
    del unique_names
    del unique_addresses
    
    res = {
        'file_key': file_key,
        'file_path': file_path,
        'header': header,
        'total_rows': total_rows,
        'malformed_rows': malformed_rows,
        'malformed_examples': malformed_examples,
        'first_5_rows': first_5_rows,
        'missing_counts': missing_counts,
        'unique_id_count': unique_id_count,
        'unique_name_count': unique_name_count,
        'unique_addr_count': unique_addr_count,
        'name_stats': {
            'empty_count': missing_counts['business_name'],
            'unique_count': unique_name_count,
            'duplicate_count': (total_rows - missing_counts['business_name']) - unique_name_count,
            'min_length': 0 if name_len_count == 0 else name_len_min,
            'max_length': name_len_max,
            'avg_length': 0 if name_len_count == 0 else round(name_len_sum / name_len_count, 2)
        },
        'address_stats': {
            'empty_count': missing_counts['business_address'],
            'unique_count': unique_addr_count,
            'duplicate_count': (total_rows - missing_counts['business_address']) - unique_addr_count,
            'min_length': 0 if addr_len_count == 0 else addr_len_min,
            'max_length': addr_len_max,
            'avg_length': 0 if addr_len_count == 0 else round(addr_len_sum / addr_len_count, 2)
        },
        'country_distribution': dict(country_counter),
        'elapsed_sec': round(time.time() - t0, 2)
    }
    print(f"Finished {file_key} in {res['elapsed_sec']}s (Rows: {total_rows:,})")
    return res

if __name__ == '__main__':
    print("Testing pipeline on train_source1...")
    res = analyze_source_file(DATASETS['train_source1'], 'train_source1')
    print("Header:", res['header'])
    print("Rows:", res['total_rows'])
    print("Missing:", res['missing_counts'])
    print("Country dist:", res['country_distribution'])
    print("Name stats:", res['name_stats'])
    print("Addr stats:", res['address_stats'])
