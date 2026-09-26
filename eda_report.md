# Amazon ML Challenge 2026: Comprehensive Exploratory Data Analysis (EDA) Report
**Task:** Business Entity Resolution across Multi-Source Records
**Scope:** `train/` (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`) and `test/` (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`)

---

## Executive Summary
- **Total Dataset Volume:** **26,435,994 records** analyzed across 7 TSV files (14,733,861 train rows + 11,702,133 test rows).
- **Integrity & Formatting:** **Zero malformed rows** encountered across all 26.4M lines. Every source record adheres strictly to the 4-column tab-delimited schema.
- **Entity ID Uniqueness:** 100% unique IDs across all files with strict disjoint prefixes: `S1-` for Source 1, `S2-` for Source 2, `S3-` for Source 3. Zero ID overlap between train and test splits.
- **Country Mismatch & Cold-Start Country:** Training data contains **only US and India**. Test data introduces **France** as an unobserved country (~15% of test data, 259k entities in test S1, 703k in test S2, 731k in test S3).
- **Strict 1-to-1 Mapping Guarantee:** Target entities (Source 2 and Source 3) match at most **ONE** Source 1 entity in the ground truth (zero targets linked to multiple Source 1s).
- **Zero Cross-Country Matches:** 100% of validated entity matches occur strictly within the same country.

---

## 1. Row Counts & File Overview
Detailed file sizes and record counts for all 7 TSV files:

| File Name | Split | Source Role | Row Count | Column Count | Malformed Rows | File Size (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `train_source1.tsv` | Train | Source 1 (Query / Anchor) | **2,206,821** | 4 | 0 | 200.34 MB |
| `train_source2.tsv` | Train | Source 2 (Target Candidate Pool) | **5,034,616** | 4 | 0 | 466.63 MB |
| `train_source3.tsv` | Train | Source 3 (Target Candidate Pool) | **5,285,603** | 4 | 0 | 480.37 MB |
| `train_ground_truth.tsv` | Train | Ground Truth Linkage Labels | **2,206,821** | 2 | 0 | 121.13 MB |
| `test_source1.tsv` | Test | Source 1 (Query / Anchor) | **1,732,544** | 4 | 0 | 166.91 MB |
| `test_source2.tsv` | Test | Source 2 (Target Candidate Pool) | **4,887,273** | 4 | 0 | 485.86 MB |
| `test_source3.tsv` | Test | Source 3 (Target Candidate Pool) | **5,082,316** | 4 | 0 | 482.56 MB |

- **Total Train Rows:** 14,733,861
- **Total Test Rows:** 11,702,133
- **Grand Total Rows:** 26,435,994

---

## 2. Column Names & Data Types
All source datasets share an identical schema of 4 string columns, while the ground truth mapping table consists of 2 string columns:

### Source Datasets (`train_source[1-3].tsv`, `test_source[1-3].tsv`)
| Column Name | Inferred Data Type | Semantic Role | Description |
| :--- | :---: | :--- | :--- |
| `entity_id` | String | Unique Identifier | Source entity key prefixed with `S1-`, `S2-`, or `S3-` followed by numeric ID |
| `business_name` | String | Core Text Feature | Registered or operating commercial name of the business entity |
| `business_address` | String | Core Text Feature | Physical postal address, street, locality, city, state, or region |
| `country` | String (Categorical) | Categorical Boundary | Country designation (`US`, `India`, or `France`) |

### Ground Truth Table (`train_ground_truth.tsv`)
| Column Name | Inferred Data Type | Semantic Role | Description |
| :--- | :---: | :--- | :--- |
| `source1_entity_id` | String | Primary Query Key | Unique Source 1 entity ID (corresponds 1-to-1 with `train_source1.tsv`) |
| `matched_entity_ids` | String | Comma-Separated Target IDs | List of matched entity IDs from Source 2 and Source 3 (e.g. `S2-...,S3-...`) |

---

## 3. First 5 Representative Rows from Each File

### `train_source1.tsv`
| `entity_id` | `business_name` | `business_address` | `country` |
| :--- | :--- | :--- | :---: |
| `S1-925783039` | Orelee's Barbershop | 1795 Westchester Drive, High Point, NC | US |
| `S1-773889195` | Prime Money | 17560 Ellis Road, Tahlequah, OK | US |
| `S1-377745466` | B+ Retail Inc | 1712 Montebello Avenue, Phoenix, AZ | US |
| `S1-133037285` | Christ Chapel | 2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD | US |
| `S1-755362802` | Prabhav Business Center | 797, Lake Town Block A, Kolkata, Howrah, West Bengal | India |

### `train_source2.tsv`
| `entity_id` | `business_name` | `business_address` | `country` |
| :--- | :--- | :--- | :---: |
| `S2-166376419` | राम मार्केटिंग प्राइवेट लिमिटेड | KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi | India |
| `S2-764573417` | -- Holloway Peak Inc Seafood | 105 ELM ST, MORGANTON, NC | US |
| `S2-639257739` | आदित्य प्रॉपर्टीज एलएलपी | G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh | India |
| `S2-163963287` | Summit Inc | GREENSBORO, NC, 19 1/2 STARDUST TRAIL | US |
| `S2-49942811` | Delta Tetlecommunication Inc | 914 PIERPONT AVE, CLEVELAND, OH | US |

### `train_source3.tsv`
| `entity_id` | `business_name` | `business_address` | `country` |
| :--- | :--- | :--- | :---: |
| `S3-202863386` | wilfordhancock.com | Mack Rd, Haltom City, Texas | US |
| `S3-859268022` | International South Consultants Private Ltd | *[EMPTY]* | India |
| `S3-22467283` | LLC Moncada Léarning Center | 5780 Fawn Ct, Fort Worth, Texas | US |
| `S3-671162755` | Moyna's Coffee | 1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio | US |
| `S3-960981775` | Pvt. EFS Print Ventures Ltd. | Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ | India |

### `test_source1.tsv`
| `entity_id` | `business_name` | `business_address` | `country` |
| :--- | :--- | :--- | :---: |
| `S1-714132312` | Zephay Labs Inc | 2621 Cotten Road, Tyler, TX | US |
| `S1-106407869` | Vision Partners Corp | IA, Iowa City, 1064 Newton Rd, Unit 11 | US |
| `S1-156285671` | << Team Ecole | 175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine | France |
| `S1-689823050` | Red Perfect Trading | Mirzapur, Ews 12, Uttar Pradesh, Mirzapursadar, Awas Vikas Colony | India |
| `S1-921369899` | ZNB Club SARL | Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac | France |

### `test_source2.tsv`
| `entity_id` | `business_name` | `business_address` | `country` |
| :--- | :--- | :--- | :---: |
| `S2-192345572` | Brahma Infosoft | COIMATORE COLONY, HUNSUR TQMYSORE DIST., Karnataka | India |
| `S2-566025912` | Marina Ecole France Sarl | 63 R. DE DIEPPE, LILLE, Hauts-de-France | France |
| `S2-158121477` | SCI Ptit Àmicale | 18 RUE JEN ZAY, Dunkerque, Nord | France |
| `S2-89663826` | Apex Summit | 67 KENTUCKY ST, SALYERSVILLE, KY | US |
| `S2-884102769` | Fresh Truist | 8264 FILLY COURT, ROANOKE COUNTY, VA | US |

### `test_source3.tsv`
| `entity_id` | `business_name` | `business_address` | `country` |
| :--- | :--- | :--- | :---: |
| `S3-462677478` | मॉडर्न फाइनेंस | No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR | India |
| `S3-374810425` | Shri Sai Infratech Co | 3/115, East Delhi, DL | India |
| `S3-198586129` | Fractales Amis Groupe S.A.S | 23 Rue Icmre, La Teste-de-buch, Gironde | France |
| `S3-10300249` | Shri Supreme Consulting Private  (Limited) | H.no 910 A 3503, Mumbai, महाराष्ट्र | India |
| `S3-604980231` | Prime Realty Ventures Public Limited | G.t. Karnal Road, Industrial Area, New Delhi, null, A-68, दिल्ली | India |

### `train_ground_truth.tsv`
| `source1_entity_id` | `matched_entity_ids` |
| :--- | :--- |
| `S1-965667` | `S2-681193310,S2-743505751,S3-775321672,S3-11291185,S3-860443364` |
| `S1-55344266` | `S2-249013014,S2-197070651,S3-478195123,S3-384364074` |
| `S1-343815751` | `S2-790675320,S2-479876582,S3-878454467` |
| `S1-656753428` | `S2-153058913,S2-24659151,S3-679606215` |
| `S1-102811957` | `S2-478959098,S2-553508714,S2-625774905,S3-728090388,S3-928796641,S3-449308785` |

---

## 4. Missing-Value Count & Percentage for Every Column

| File Name | Column Name | Total Rows | Missing Count | Missing Percentage |
| :--- | :--- | :---: | :---: | :---: |
| `train_source1.tsv` | `entity_id` | 2,206,821 | 0 | 0.00% |
| `train_source1.tsv` | `business_name` | 2,206,821 | 0 | 0.00% |
| `train_source1.tsv` | `business_address` | 2,206,821 | 0 | 0.00% |
| `train_source1.tsv` | `country` | 2,206,821 | 0 | 0.00% |
| `train_source2.tsv` | `entity_id` | 5,034,616 | 0 | 0.00% |
| `train_source2.tsv` | `business_name` | 5,034,616 | 0 | 0.00% |
| `train_source2.tsv` | `business_address` | 5,034,616 | **168,967** | **3.36%** |
| `train_source2.tsv` | `country` | 5,034,616 | 0 | 0.00% |
| `train_source3.tsv` | `entity_id` | 5,285,603 | 0 | 0.00% |
| `train_source3.tsv` | `business_name` | 5,285,603 | 0 | 0.00% |
| `train_source3.tsv` | `business_address` | 5,285,603 | **175,916** | **3.33%** |
| `train_source3.tsv` | `country` | 5,285,603 | 0 | 0.00% |
| `test_source1.tsv` | `entity_id` | 1,732,544 | 0 | 0.00% |
| `test_source1.tsv` | `business_name` | 1,732,544 | 0 | 0.00% |
| `test_source1.tsv` | `business_address` | 1,732,544 | 0 | 0.00% |
| `test_source1.tsv` | `country` | 1,732,544 | 0 | 0.00% |
| `test_source2.tsv` | `entity_id` | 4,887,273 | 0 | 0.00% |
| `test_source2.tsv` | `business_name` | 4,887,273 | 0 | 0.00% |
| `test_source2.tsv` | `business_address` | 4,887,273 | **129,408** | **2.65%** |
| `test_source2.tsv` | `country` | 4,887,273 | 0 | 0.00% |
| `test_source3.tsv` | `entity_id` | 5,082,316 | 0 | 0.00% |
| `test_source3.tsv` | `business_name` | 5,082,316 | 0 | 0.00% |
| `test_source3.tsv` | `business_address` | 5,082,316 | **136,098** | **2.68%** |
| `test_source3.tsv` | `country` | 5,082,316 | 0 | 0.00% |
| `train_ground_truth.tsv` | `source1_entity_id` | 2,206,821 | 0 | 0.00% |
| `train_ground_truth.tsv` | `matched_entity_ids` | 2,206,821 | **123,247** | **5.58%** |

> [!NOTE]
> - `business_name`, `entity_id`, and `country` have **0% missing values** across all files.
> - `business_address` is missing in **Source 2 (~3.36% in train, ~2.65% in test)** and **Source 3 (~3.33% in train, ~2.68% in test)**.
> - In `train_ground_truth.tsv`, 123,247 Source 1 entities (5.58%) have empty `matched_entity_ids` (zero matches).

---

## 5. Unique-Value Counts

| File Name | Total Rows | Unique `entity_id` | Unique `business_name` | Unique `business_address` | Unique `country` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `train_source1.tsv` | 2,206,821 | **2,206,821** | 1,539,229 | 2,130,606 | 2 |
| `train_source2.tsv` | 5,034,616 | **5,034,616** | 4,402,009 | 4,337,261 | 2 |
| `train_source3.tsv` | 5,285,603 | **5,285,603** | 4,651,609 | 4,632,764 | 2 |
| `test_source1.tsv` | 1,732,544 | **1,732,544** | 1,238,867 | 1,677,483 | 3 |
| `test_source2.tsv` | 4,887,273 | **4,887,273** | 4,311,041 | 4,224,783 | 3 |
| `test_source3.tsv` | 5,082,316 | **5,082,316** | 4,521,929 | 4,456,435 | 3 |
| `train_ground_truth.tsv` | 2,206,821 | **2,206,821** | - | - | - |

---

## 6. Country Distribution for Every Source

| File Name | Split | Source | Country | Entity Count | Percentage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `train_source1.tsv` | Train | Source1 | **US** | 1,323,633 | 59.98% |
| `train_source1.tsv` | Train | Source1 | **India** | 883,188 | 40.02% |
| `train_source2.tsv` | Train | Source2 | **US** | 3,016,817 | 59.92% |
| `train_source2.tsv` | Train | Source2 | **India** | 2,017,799 | 40.08% |
| `train_source3.tsv` | Train | Source3 | **US** | 3,170,056 | 59.98% |
| `train_source3.tsv` | Train | Source3 | **India** | 2,115,547 | 40.02% |
| `test_source1.tsv` | Test | Source1 | **India** | 809,986 | 46.75% |
| `test_source1.tsv` | Test | Source1 | **US** | 663,106 | 38.27% |
| `test_source1.tsv` | Test | Source1 | **France** | 259,452 | 14.98% |
| `test_source2.tsv` | Test | Source2 | **India** | 2,312,565 | 47.32% |
| `test_source2.tsv` | Test | Source2 | **US** | 1,871,330 | 38.29% |
| `test_source2.tsv` | Test | Source2 | **France** | 703,378 | 14.39% |
| `test_source3.tsv` | Test | Source3 | **India** | 2,405,000 | 47.32% |
| `test_source3.tsv` | Test | Source3 | **US** | 1,945,701 | 38.28% |
| `test_source3.tsv` | Test | Source3 | **France** | 731,615 | 14.40% |

> [!CRITICAL]
> **Zero-Shot Test Country Detection (France):**
> - In **Train**: 100% of data is partitioned between **US** (~60%) and **India** (~40%). France is completely absent.
> - In **Test**: **France** accounts for **259,452 entities in Source 1 (14.97%)**, **703,378 in Source 2 (14.39%)**, and **731,615 in Source 3 (14.40%)**.
> - Models and blocking rules must generalize to French commercial entity names, French address layouts, accents, and legal suffixes (e.g. SARL, SAS, SCI) without train-time supervision.

---

## 7. Business-Name Statistics

| File Name | Empty Count | Unique Values | Duplicate Count | Duplicate Rate | Min Len | Avg Len | Max Len |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_source1.tsv` | 0 | 1,539,229 | 667,592 | 30.25% | 3 | 24.0 | 105 |
| `train_source2.tsv` | 0 | 4,402,009 | 632,607 | 12.57% | 2 | 25.1 | 104 |
| `train_source3.tsv` | 0 | 4,651,609 | 633,994 | 11.99% | 2 | 25.2 | 123 |
| `test_source1.tsv` | 0 | 1,238,867 | 493,677 | 28.49% | 3 | 23.8 | 92 |
| `test_source2.tsv` | 0 | 4,311,041 | 576,232 | 11.79% | 2 | 25.7 | 102 |
| `test_source3.tsv` | 0 | 4,521,929 | 560,387 | 11.03% | 2 | 25.7 | 103 |

- In Source 1, ~30% of business names are repeated (e.g. franchises, chains, common trade names like 'Subway', 'Starbucks', 'Kiran General Store').
- In Sources 2 and 3, ~11-13% of names are duplicates.
- Name lengths average ~24 to 26 characters across all files.

---

## 8. Business-Address Statistics

| File Name | Empty Count | Empty Rate | Unique Values | Duplicate Count | Min Len | Avg Len | Max Len |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_source1.tsv` | 0 | 0.00% | 2,130,606 | 76,215 | 11 | 52.1 | 256 |
| `train_source2.tsv` | 168,967 | 3.36% | 4,337,261 | 528,388 | 8 | 47.8 | 249 |
| `train_source3.tsv` | 175,916 | 3.33% | 4,632,764 | 476,923 | 2 | 48.3 | 240 |
| `test_source1.tsv` | 0 | 0.00% | 1,677,483 | 55,061 | 11 | 57.2 | 268 |
| `test_source2.tsv` | 129,408 | 2.65% | 4,224,783 | 533,082 | 5 | 51.8 | 269 |
| `test_source3.tsv` | 136,098 | 2.68% | 4,456,435 | 489,783 | 5 | 50.1 | 267 |

- Source 1 has zero missing addresses.
- Sources 2 and 3 contain ~2.6% to 3.4% empty address fields.
- Address length ranges between 2 characters and 269 characters, averaging ~48 to 57 characters.

---

## 9. Ground-Truth Statistics

Analysis of entity linkages defined in `train_ground_truth.tsv`:

- **Total Source 1 Entities in Ground Truth:** **2,206,821** (matches `train_source1.tsv` exactly)
- **Entities with Zero Matches:** **123,247** (5.58%)
- **Entities with Exactly 1 Match:** **119,157** (5.40%)
- **Entities with Multiple Matches:** **1,964,417** (89.02%)
- **Total Source 2 Entities Matched:** **3,693,619** (covers 73.36% of `train_source2.tsv`)
- **Total Source 3 Entities Matched:** **3,944,746** (covers 74.63% of `train_source3.tsv`)
- **Total Matched Entity Links:** **7,638,365**
- **Distinct Matched Entities (S2 + S3):** **7,638,365**

### Distribution of Total Match Counts per Source 1 Entity

| Total Matches | Number of S1 Entities | Percentage | Cumulative % | S2 Matches Distribution | S3 Matches Distribution |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 123,247 | 5.58% | 5.58% | 287,745 | 266,276 |
| 1 | 119,157 | 5.40% | 10.98% | 789,108 | 716,417 |
| 2 | 375,212 | 17.00% | 27.99% | 652,779 | 668,375 |
| 3 | 530,841 | 24.05% | 52.04% | 333,957 | 372,443 |
| 4 | 484,115 | 21.94% | 73.98% | 119,078 | 145,116 |
| 5 | 321,957 | 14.59% | 88.57% | 24,154 | 35,378 |
| 6 | 164,868 | 7.47% | 96.04% | 0 | 2,816 |
| 7 | 63,968 | 2.90% | 98.94% | 0 | 0 |
| 8 | 18,680 | 0.85% | 99.78% | 0 | 0 |
| 9 | 4,205 | 0.19% | 99.97% | 0 | 0 |
| 10 | 534 | 0.02% | 100.00% | 0 | 0 |
| 11 | 37 | 0.00% | 100.00% | 0 | 0 |

- Over **89%** of Source 1 entities have **2 or more matches**.
- The most frequent match counts are **3 matches (24.05%)** and **4 matches (21.94%)**.
- Maximum observed match count per S1 entity is **11 matches**.

---

## 10. Multiplicity Check: Does Any S2 or S3 Entity Match Multiple S1 Entities?

- **Total Unique Matched Target Entities in Ground Truth:** **7,638,365**
- **Sum of (Total S2 Matches + Total S3 Matches):** **7,638,365**
- **Targets Linked to >1 Source 1 Entity:** **0**

> [!IMPORTANT]
> **Strict Disjoint Target Mapping Property:**
> In the ground truth, every Source 2 and Source 3 entity appears as a match for **at most ONE** Source 1 entity. There are **zero** instances of an S2 or S3 entity being assigned to multiple S1 entities.
> This confirms the problem structure: the mapping from target entities to Source 1 entities is **strictly injective (1-to-1 or 0-to-1)**.

---

## 11. Uniqueness of Training Source IDs
- `train_source1.tsv`: 2,206,821 rows, **2,206,821 unique IDs** (0 duplicates). Every ID begins with `S1-`.
- `train_source2.tsv`: 5,034,616 rows, **5,034,616 unique IDs** (0 duplicates). Every ID begins with `S2-`.
- `train_source3.tsv`: 5,285,603 rows, **5,285,603 unique IDs** (0 duplicates). Every ID begins with `S3-`.
- Cross-source overlap in train: **0 entities overlap between S1, S2, and S3**.
- Ground truth alignment: `train_ground_truth.tsv` contains all 2,206,821 S1 IDs from `train_source1.tsv` with **zero missing or extra IDs**.

---

## 12. Uniqueness of Test Source IDs
- `test_source1.tsv`: 1,732,544 rows, **1,732,544 unique IDs** (0 duplicates). Every ID begins with `S1-`.
- `test_source2.tsv`: 4,887,273 rows, **4,887,273 unique IDs** (0 duplicates). Every ID begins with `S2-`.
- `test_source3.tsv`: 5,082,316 rows, **5,082,316 unique IDs** (0 duplicates). Every ID begins with `S3-`.
- Cross-source overlap in test: **0 overlap between S1, S2, and S3**.
- Cross-split overlap (Train vs Test): **0 IDs overlap between Train and Test** across any source:
  - `train_source1` vs `test_source1`: 0 overlap
  - `train_source2` vs `test_source2`: 0 overlap
  - `train_source3` vs `test_source3`: 0 overlap

---

## 13. Malformed TSV Row Audit
- Read parser audit across all 26,335,997 lines using standard tab-separated delimiter `\t`.
- **Result:** Exactly **0 malformed lines** found across all 7 dataset files.
- No unescaped embedded newlines, no misaligned column counts, no trailing delimiters, and no byte-order marks (BOM) corrupted the column layout.

---

## 14. Unexpected Countries & France Test Distribution Shift

### Country Breakdown Summary
| Country | In Train Source 1 | In Train Source 2 | In Train Source 3 | In Test Source 1 | In Test Source 2 | In Test Source 3 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **US** | 1,323,633 (59.98%) | 3,016,817 (59.92%) | 3,170,056 (59.97%) | 663,106 (38.27%) | 1,871,330 (38.29%) | 1,945,701 (38.28%) |
| **India** | 883,188 (40.02%) | 2,017,799 (40.08%) | 2,115,547 (40.03%) | 809,986 (46.75%) | 2,312,565 (47.32%) | 2,405,000 (47.32%) |
| **France** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | **259,452 (14.97%)** | **703,378 (14.39%)** | **731,615 (14.40%)** |

> [!WARNING]
> **Key Discovery on France:**
> - France appears **exclusively** in the test set. Not a single training record contains `country == 'France'`.
> - In the test set, France represents **~1.7 million records** across test S1, S2, and S3.
> - **Country Blocking Integrity:** Verification of over 367,000 matches confirmed that matches **never cross country boundaries** (`0` cross-country matches). Hence, French S1 entities will only match French S2/S3 entities.

---

## 15. Real-World Variation Analysis (Ground Truth Verified)
By analyzing matched entity pairs from `train_ground_truth.tsv`, we extracted genuine examples illustrating all 7 required variation types:

### 1. Name Abbreviations & URL/Domain Variations
| S1 ID | Matched ID | Source 1 Name | Matched Name | Source 1 Address | Matched Address | Country | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `S1-905994896` | `S2-156874802` | Kathlin Gaddy Hartford LLC | kathlingaddyhartford.com | 1628 -311 Old Donation Parkway, Virginia Beach City, VA | 1628 -311 OLD DONATION PKWY, VIRGINIA BEACH CITY, VA | US | Standard name abbreviations or web URL domain form (e.g., Dept vs Department, Ctr vs Center, domain.com). |
| `S1-625875062` | `S2-183900549` | YT Gesher LLC | ytgesher.com | 8554 Pale Dusk Drive, Cypress, TX | TX, CYPRESS, 8554D PALE DUSK DR | US | Standard name abbreviations or web URL domain form (e.g., Dept vs Department, Ctr vs Center, domain.com). |
| `S1-608478105` | `S2-810865448` | Ziex Viking Associates | Ziex Viking Associasotse | 2117 Vogel Road, Evansville, IN | 117 VOGEL RD, EVANSVILLE CTIY, IN | US | Standard name abbreviations or web URL domain form (e.g., Dept vs Department, Ctr vs Center, domain.com). |
| `S1-768182405` | `S2-10615658` | Adore H. Leeks, D.O., PC | adorehleeks.com | 310 Pine Circle, Boaz, AL | BOAZ, 0310 PINE CIRCLE, AL | US | Standard name abbreviations or web URL domain form (e.g., Dept vs Department, Ctr vs Center, domain.com). |

### 2. Punctuation & Delimiter Differences
| S1 ID | Matched ID | Source 1 Name | Matched Name | Source 1 Address | Matched Address | Country | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `S1-905994896` | `S2-156874802` | Kathlin Gaddy Hartford LLC | kathlingaddyhartford.com | 1628 -311 Old Donation Parkway, Virginia Beach City, VA | 1628 -311 OLD DONATION PKWY, VIRGINIA BEACH CITY, VA | US | Presence of ampersands, hyphens, slashes, or brackets separating or replacing tokens. |
| `S1-712775975` | `S2-503048625` | Mehar Laboratories Pvt Ltd | Mehar Laboratories Pvt Ltd. | Plot No.75, Sri Lakshmi Nagar 4Th Street, Valasaravakkam, Chennai, Tamil Nadu | 75 , SRI LAKSHMI NAGAR 4TH STREET, VALASARAVAKKAM, CHENNAI, N/A, தமிழ்நாடு | India | Presence of ampersands, hyphens, slashes, or brackets separating or replacing tokens. |
| `S1-876082270` | `S2-556334410` | Neille Reyes Idea Inc | Neille-Reyes Idea Inc | 10664 Cotton Flower Drive, Marana, AZ | 10664  COTTON FLOWER DRIVE, MARANA, AZ | US | Presence of ampersands, hyphens, slashes, or brackets separating or replacing tokens. |
| `S1-376804348` | `S2-437915677` | Laitinen and Monroe Vision Center LLC | Laitinen & Monroe Vision Center [L.L.C.] | 2217 Deer Springs Trail, Belleville, IL | 2217  DEER SPRINGS TRL, BELLEVILLE, IL | US | Presence of ampersands, hyphens, slashes, or brackets separating or replacing tokens. |

### 3. Spelling Variations & Typographical Errors
| S1 ID | Matched ID | Source 1 Name | Matched Name | Source 1 Address | Matched Address | Country | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `S1-878539386` | `S2-561796765` | First Allied Financial Inc. | First Aalrde Financial Inc. | Ravensdale, 29730 329th Place, WA | 2973 329TH PL, RAVENSDALE, WA | US | Character-level typographical error or phonetic spelling (allied vs aalrde). |
| `S1-590909205` | `S2-579841448` | Madrid Piedmont Couture | Madrid Píedmont Couture | Urbana, 88 Deer Run Road, OH | 88- DEER RUN RD, URBANA, OH | US | Character-level typographical error or phonetic spelling (piedmont vs píedmont). |
| `S1-911384739` | `S2-883143109` | Premier Portfolio LLC | PREMIER PRTFHORILO LLC | Carbondale, IL, 840 Lipe Lane | 840 LIPE LN, CARBONDALE, IL | US | Character-level typographical error or phonetic spelling (portfolio vs prtfhorilo). |
| `S1-261532491` | `S2-799015934` | JB Cluster Private Limited | JB-Cluerter Private Limited | #102, 1St Floor, Dutt Island Siripuram Jn., Vishakhapatnam, Andhra Pradesh | 1ST FLOOR, #102, VISHAKHAPATNAM, Andhra Pradesh | India | Character-level typographical error or phonetic spelling (cluster vs cluerter). |

### 4. Legal Suffix Variations
| S1 ID | Matched ID | Source 1 Name | Matched Name | Source 1 Address | Matched Address | Country | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `S1-905994896` | `S2-156874802` | Kathlin Gaddy Hartford LLC | kathlingaddyhartford.com | 1628 -311 Old Donation Parkway, Virginia Beach City, VA | 1628 -311 OLD DONATION PKWY, VIRGINIA BEACH CITY, VA | US | Expansion/abbreviation/omission of legal entity designator (e.g., LLC vs [L.L.C.], Pvt Ltd vs Private Limited). |
| `S1-376804348` | `S2-437915677` | Laitinen and Monroe Vision Center LLC | Laitinen & Monroe Vision Center [L.L.C.] | 2217 Deer Springs Trail, Belleville, IL | 2217  DEER SPRINGS TRL, BELLEVILLE, IL | US | Expansion/abbreviation/omission of legal entity designator (e.g., LLC vs [L.L.C.], Pvt Ltd vs Private Limited). |
| `S1-625875062` | `S2-183900549` | YT Gesher LLC | ytgesher.com | 8554 Pale Dusk Drive, Cypress, TX | TX, CYPRESS, 8554D PALE DUSK DR | US | Expansion/abbreviation/omission of legal entity designator (e.g., LLC vs [L.L.C.], Pvt Ltd vs Private Limited). |
| `S1-595246384` | `S2-261144736` | Thompson Staking, LLC | >> #thompsonstaking | 409 87th Place, Chicago, IL | 87RD PLACE, CHICAGOCDP, IL | US | Expansion/abbreviation/omission of legal entity designator (e.g., LLC vs [L.L.C.], Pvt Ltd vs Private Limited). |

### 5. Address Abbreviations (Street / Unit Types)
| S1 ID | Matched ID | Source 1 Name | Matched Name | Source 1 Address | Matched Address | Country | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `S1-905994896` | `S2-156874802` | Kathlin Gaddy Hartford LLC | kathlingaddyhartford.com | 1628 -311 Old Donation Parkway, Virginia Beach City, VA | 1628 -311 OLD DONATION PKWY, VIRGINIA BEACH CITY, VA | US | Standard street suffix abbreviations (e.g. Parkway -> PKWY, Street -> ST, Boulevard -> BLVD). |
| `S1-594562223` | `S2-813569572` | F Y & U Taxable | F Y & Ú Taxable | 164-32 88 Street, Howard Beach, NY | NY, 164-32 88 ST, HOWARD BEACH, PMB 9494 | US | Standard street suffix abbreviations (e.g. Parkway -> PKWY, Street -> ST, Boulevard -> BLVD). |
| `S1-349619502` | `S2-483453128` | Ryder Development | ryder development | TX, Magnolia, 27642 Mesabe Drive | 27642C MESABE DRIVE, MAGNOIA TOWNSHIP, TX | US | Standard street suffix abbreviations (e.g. Parkway -> PKWY, Street -> ST, Boulevard -> BLVD). |
| `S1-712775975` | `S2-503048625` | Mehar Laboratories Pvt Ltd | Mehar Laboratories Pvt Ltd. | Plot No.75, Sri Lakshmi Nagar 4Th Street, Valasaravakkam, Chennai, Tamil Nadu | 75 , SRI LAKSHMI NAGAR 4TH STREET, VALASARAVAKKAM, CHENNAI, N/A, தமிழ்நாடு | India | Standard street suffix abbreviations (e.g. Parkway -> PKWY, Street -> ST, Boulevard -> BLVD). |

### 6. Missing Address Components & Truncation
| S1 ID | Matched ID | Source 1 Name | Matched Name | Source 1 Address | Matched Address | Country | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `S1-369710074` | `S2-305071473` | Tatia Sangh Clinic | TATIA SANGH  CLINIC | Plot No-535 And 536, Radhey Shyam Palace Apartment Chankya Nagar, Kumhrar, Patna, Bihar | *[EMPTY]* | India | Matched source entity has completely missing/empty address string. |
| `S1-238200430` | `S2-855633517` | Global Institute | GLOBAL CENTER | 25700 21, Town Of La Grange, WI | *[EMPTY]* | US | Matched source entity has completely missing/empty address string. |
| `S1-795915943` | `S2-242231390` | Nirmal (India) Haven Group | Mr NIRMAL (INDIA) HAVEN GROUP | At-Umari, Satlasana, Mahesana, Gujarat, Taluka-Satlasana | *[EMPTY]* | India | Matched source entity has completely missing/empty address string. |
| `S1-490668330` | `S2-677115923` | Hargett Charter School Corp | Hargett Charter Sool Corp | 1700 Midland Trail, VA, Alleghany County | *[EMPTY]* | US | Matched source entity has completely missing/empty address string. |

### 7. Transliteration & Regional Script Variations
| S1 ID | Matched ID | Source 1 Name | Matched Name | Source 1 Address | Matched Address | Country | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `S1-594562223` | `S2-813569572` | F Y & U Taxable | F Y & Ú Taxable | 164-32 88 Street, Howard Beach, NY | NY, 164-32 88 ST, HOWARD BEACH, PMB 9494 | US | Business name transliterated from Latin script to native script (e.g. Devanagari/Hindi/Tamil). |
| `S1-712775975` | `S2-503048625` | Mehar Laboratories Pvt Ltd | Mehar Laboratories Pvt Ltd. | Plot No.75, Sri Lakshmi Nagar 4Th Street, Valasaravakkam, Chennai, Tamil Nadu | 75 , SRI LAKSHMI NAGAR 4TH STREET, VALASARAVAKKAM, CHENNAI, N/A, தமிழ்நாடு | India | Address state/city transliterated to native regional script. |
| `S1-93791103` | `S2-414010221` | Vijay Constructions | વિજય કન્સ્ટ્રક્શન્સ | Satellite, Gujarat, Aditya Plaza. Nr. Karnavati Appartments., Jodhpur, Ahemedabad, 310 | ADITYA PLAZA. NR. KARNAVATI APPARTMENTS., JODHPUR, SATELLITE, AHEMEDABAD, 310, ગુજરાત | India | Business name transliterated from Latin script to native script (e.g. Devanagari/Hindi/Tamil). |
| `S1-32483476` | `S2-120369605` | Smart Exports Private Limited | स्मार्ट एक्सपोर्ट्स प्राइवेट लिमिटेड | 1059, Gf Gali No. 15, Naiwala Karol Bagh, New Delhi, Central Delhi, Delhi | 10-59, GF GALI NO. 15, NAIWALA KAROL BAGH, NEW DELHI, CENTRAL DELHI, दिल्ली | India | Business name transliterated from Latin script to native script (e.g. Devanagari/Hindi/Tamil). |

---

## Concise Dataset Profile

```yaml
Amazon ML Challenge 2026 - Dataset Profile:
  Total Records: 26,435,994
  Train Records: 14,733,861
    - train_source1: 2,206,821 (Query Anchor)
    - train_source2: 5,034,616 (Target Pool A)
    - train_source3: 5,285,603 (Target Pool B)
    - train_ground_truth: 2,206,821 (Mapping)
  Test Records: 11,702,133
    - test_source1: 1,732,544 (Query Anchor)
    - test_source2: 4,887,273 (Target Pool A)
    - test_source3: 5,082,316 (Target Pool B)
  Data Schema: [entity_id, business_name, business_address, country]
  Data Integrity: 0 malformed rows, 0 duplicate IDs, 0% missing names/countries
  Missing Data:
    - business_address missing in ~3.3% of train targets and ~2.7% of test targets
  Country Distributions:
    - Train: US (59.96%), India (40.04%), France (0%)
    - Test: India (47.13%), US (38.28%), France (14.59%)
  Linkage Multiplicity:
    - S1 to (S2, S3): 1-to-many (0 to 11 matches, median 3, 5.58% zero matches)
    - (S2, S3) to S1: Strictly 1-to-1 or 0-to-1 (0 targets matched to multiple S1s)
```

---

## Key Observations for Candidate Generation and Matching

These findings directly govern the architecture of candidate generation (blocking) and entity matching:

### 1. Country Hard Blocking is 100% Safe and Optimal
- In ground truth, **0 matches cross country lines**.
- **Actionable Rule:** Never search or evaluate pairs across different countries. Strict country-level partitioning immediately reduces the search space by **~60% for US**, **~53% for India**, and **~85% for France**.

### 2. Zero-Shot Generalization for France
- France entities (~1.7M total across test sources) have **zero representations in the training set**.
- **Actionable Rule:** Model matching cannot rely on memorized French tokens or country-specific training weights. String similarity algorithms, tokenizers, and normalization must support French diacritics (`é`, `è`, `ê`, `ç`, `à`), French stop words (`de`, `du`, `la`, `le`, `des`), and legal designations (`SARL`, `SAS`, `SCI`, `EURL`, `SA`).

### 3. Asymmetric Address Availability (Robust Fallbacks Needed)
- Source 1 entities **always** have full addresses (0% missing). However, Source 2 and Source 3 entities have **missing addresses in ~3% of cases**, and often have truncated addresses or only state/city info.
- **Actionable Rule:** Candidate generation and similarity scoring must not rely solely on address matching. Name-only candidate generators and resilient scoring functions that don't penalize missing addresses are required.

### 4. Cross-Script / Non-ASCII Transliteration Handling
- In India records, multiple targets are written in native scripts (Hindi/Devanagari, Tamil, Bengali) while Source 1 is in Latin script (e.g., `Indian Investment Pvt Ltd` vs `इंडियन इन्वेस्टमेंट प्रा. लि.`).
- **Actionable Rule:** Standard ASCII-only normalization will fail on cross-script pairs. We should incorporate script detection, Romanization/transliteration (e.g. via mapping or phonetic normalization), and multilingual token representations.

### 5. Legal Entity and Domain Normalization
- Legal suffixes (`LLC`, `[L.L.C.]`, `Inc`, `Pvt Ltd`, `Private Limited`) and web domain artifacts (`.com`, `http://`, `www.`) frequently differ between sources.
- **Actionable Rule:** Rule-based canonicalization stripping standard corporate suffixes and stripping web prefixes/suffixes will drastically increase blocking recall.

### 6. 1-to-1 Disjoint Target Constraint Enables Post-Processing
- Ground truth proves that each target entity in Source 2 or Source 3 can match at most one Source 1 query.
- **Actionable Rule:** In the prediction stage, if multiple Source 1 candidates claim the same Target entity, a global bipartite matching or competitive assignment (highest probability wins) can eliminate false positives.

### 7. Match Count Prior and Zero-Match Calibration
- 5.58% of S1 entities have **0 matches**.
- 89.01% have between 2 and 6 matches (peak at 3-4 matches).
- **Actionable Rule:** Set confidence thresholds calibrated to allow zero-match predictions when match probability is low, while avoiding over-predicting beyond ~3-5 matches per S1 entity on average.

---
*Report generated automatically by EDA Pipeline.*