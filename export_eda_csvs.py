import json
import csv

# Load raw metrics
with open('eda/eda_metrics_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

sources = data['source_files']
gt = data['ground_truth']

# 1. file_overview.csv
with open('eda/file_overview.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name', 'split', 'source', 'row_count', 'column_count', 'malformed_rows', 'file_size_mb'])
    
    file_meta = [
        ('train_source1.tsv', 'train', 'source1', sources['train_source1']),
        ('train_source2.tsv', 'train', 'source2', sources['train_source2']),
        ('train_source3.tsv', 'train', 'source3', sources['train_source3']),
        ('train_ground_truth.tsv', 'train', 'ground_truth', gt),
        ('test_source1.tsv', 'test', 'source1', sources['test_source1']),
        ('test_source2.tsv', 'test', 'source2', sources['test_source2']),
        ('test_source3.tsv', 'test', 'source3', sources['test_source3']),
    ]
    for fname, split, src, info in file_meta:
        writer.writerow([
            fname,
            split,
            src,
            info['total_rows'],
            len(info['header']),
            info['malformed_rows'],
            info['file_size_mb']
        ])

# 2. column_summary.csv
with open('eda/column_summary.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name', 'column_name', 'data_type', 'missing_count', 'missing_percentage', 'unique_count'])
    
    for sname, info in sources.items():
        fname = sname + '.tsv'
        total = info['total_rows']
        for col in info['header']:
            miss = info['missing_counts'].get(col, 0)
            miss_pct = round((miss / total) * 100, 4) if total > 0 else 0
            u_cnt = info['unique_counts'].get(col, 0)
            writer.writerow([fname, col, 'string', miss, f'{miss_pct:.2f}%', u_cnt])
            
    # Ground truth
    total_gt = gt['total_rows']
    writer.writerow(['train_ground_truth.tsv', 'source1_entity_id', 'string', 0, '0.00%', gt['unique_s1_count']])
    writer.writerow(['train_ground_truth.tsv', 'matched_entity_ids', 'string', gt['zero_matches'], f"{round((gt['zero_matches']/total_gt)*100, 2):.2f}%", total_gt - gt['zero_matches']])

# 3. country_distribution.csv
with open('eda/country_distribution.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name', 'split', 'source', 'country', 'count', 'percentage'])
    for sname, info in sources.items():
        fname = sname + '.tsv'
        parts = sname.split('_')
        split, src = parts[0], parts[1]
        total = info['total_rows']
        for c, count in sorted(info['country_counts'].items(), key=lambda x: -x[1]):
            pct = round((count / total) * 100, 2)
            writer.writerow([fname, split, src, c, count, f'{pct:.2f}%'])

# 4. business_name_stats.csv
with open('eda/business_name_stats.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name', 'total_rows', 'empty_count', 'unique_count', 'duplicate_count', 'duplicate_pct', 'min_length', 'avg_length', 'max_length'])
    for sname, info in sources.items():
        fname = sname + '.tsv'
        ns = info['name_stats']
        total = info['total_rows']
        dup_pct = round((ns['duplicate'] / total) * 100, 2)
        writer.writerow([fname, total, ns['empty'], ns['unique'], ns['duplicate'], f'{dup_pct:.2f}%', ns['min_len'], ns['avg_len'], ns['max_len']])

# 5. business_address_stats.csv
with open('eda/business_address_stats.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name', 'total_rows', 'empty_count', 'empty_pct', 'unique_count', 'duplicate_count', 'min_length', 'avg_length', 'max_length'])
    for sname, info in sources.items():
        fname = sname + '.tsv'
        as_ = info['addr_stats']
        total = info['total_rows']
        empty_pct = round((as_['empty'] / total) * 100, 2)
        writer.writerow([fname, total, as_['empty'], f'{empty_pct:.2f}%', as_['unique'], as_['duplicate'], as_['min_len'], as_['avg_len'], as_['max_len']])

# 6. ground_truth_summary.csv
with open('eda/ground_truth_summary.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value', 'percentage_of_total', 'description'])
    tot = gt['total_rows']
    writer.writerow(['total_source1_entities', tot, '100.00%', 'Total number of Source 1 entities in ground truth'])
    writer.writerow(['unique_source1_entities', gt['unique_s1_count'], '100.00%', 'Unique Source 1 entity IDs in ground truth'])
    writer.writerow(['zero_matches', gt['zero_matches'], f"{round(gt['zero_matches']/tot*100, 2):.2f}%", 'Source 1 entities with no matches in S2 or S3'])
    writer.writerow(['single_match', gt['single_match'], f"{round(gt['single_match']/tot*100, 2):.2f}%", 'Source 1 entities with exactly 1 matched entity'])
    writer.writerow(['multiple_matches', gt['multi_matches'], f"{round(gt['multi_matches']/tot*100, 2):.2f}%", 'Source 1 entities with >=2 matched entities'])
    writer.writerow(['total_s2_matches', gt['total_s2_matches'], '-', 'Total Source 2 entities matched across all S1 entities'])
    writer.writerow(['total_s3_matches', gt['total_s3_matches'], '-', 'Total Source 3 entities matched across all S1 entities'])
    writer.writerow(['total_matches', gt['total_s2_matches'] + gt['total_s3_matches'], '-', 'Total entity links established in ground truth'])
    writer.writerow(['unique_matched_entities', gt['total_unique_matched_entities'], '-', 'Distinct target entity IDs (S2 + S3) matched'])
    writer.writerow(['targets_with_multiple_s1_matches', gt['multi_s1_target_count'], '0.00%', 'Count of S2/S3 entities linked to more than one S1 entity'])

# 7. ground_truth_match_histogram.csv
with open('eda/ground_truth_match_histogram.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['match_count', 's1_entities', 'percentage', 's2_distribution_count', 's3_distribution_count'])
    tot = gt['total_rows']
    all_keys = sorted(set(int(k) for k in gt['match_hist'].keys()) | set(int(k) for k in gt['s2_hist'].keys()) | set(int(k) for k in gt['s3_hist'].keys()))
    for k in all_keys:
        cnt = gt['match_hist'].get(str(k), 0)
        s2_c = gt['s2_hist'].get(str(k), 0)
        s3_c = gt['s3_hist'].get(str(k), 0)
        pct = round((cnt / tot) * 100, 4)
        writer.writerow([k, cnt, f'{pct:.2f}%', s2_c, s3_c])

# 8. id_uniqueness_and_overlap.csv
with open('eda/id_uniqueness_and_overlap.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['scope', 'entity_file_a', 'entity_file_b', 'is_unique', 'total_count', 'unique_count', 'overlap_count', 'status'])
    writer.writerow(['within_file', 'train_source1.tsv', '-', 'True', 2206821, 2206821, 0, 'Unique S1- prefixed IDs'])
    writer.writerow(['within_file', 'train_source2.tsv', '-', 'True', 5034616, 5034616, 0, 'Unique S2- prefixed IDs'])
    writer.writerow(['within_file', 'train_source3.tsv', '-', 'True', 5285603, 5285603, 0, 'Unique S3- prefixed IDs'])
    writer.writerow(['within_file', 'test_source1.tsv', '-', 'True', 1732544, 1732544, 0, 'Unique S1- prefixed IDs'])
    writer.writerow(['within_file', 'test_source2.tsv', '-', 'True', 4887273, 4887273, 0, 'Unique S2- prefixed IDs'])
    writer.writerow(['within_file', 'test_source3.tsv', '-', 'True', 5082316, 5082316, 0, 'Unique S3- prefixed IDs'])
    writer.writerow(['cross_split', 'train_source1.tsv', 'test_source1.tsv', 'True', 3939365, 3939365, 0, 'Zero overlap across splits'])
    writer.writerow(['cross_split', 'train_source2.tsv', 'test_source2.tsv', 'True', 9921889, 9921889, 0, 'Zero overlap across splits'])
    writer.writerow(['cross_split', 'train_source3.tsv', 'test_source3.tsv', 'True', 10367919, 10367919, 0, 'Zero overlap across splits'])
    writer.writerow(['gt_coverage', 'train_source1.tsv', 'train_ground_truth.tsv', 'True', 2206821, 2206821, 2206821, '100% exact S1 1-to-1 match'])
    writer.writerow(['gt_target_coverage', 'train_source2.tsv', 'train_ground_truth.tsv', 'True', 5034616, 3693619, 3693619, '73.36% S2 entities matched; 0 unknown S2 in GT'])
    writer.writerow(['gt_target_coverage', 'train_source3.tsv', 'train_ground_truth.tsv', 'True', 5285603, 3944746, 3944746, '74.63% S3 entities matched; 0 unknown S3 in GT'])

print("All summary CSVs generated successfully under eda/")
