from typing import Dict, List, Set, Any, Optional, Tuple
import numpy as np
import pandas as pd
from src.matching_features.feature_schema import FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS, TOTAL_FEATURE_COUNT
from src.matching_features.name_features import extract_name_features
from src.matching_features.address_features import extract_address_features
from src.matching_features.cross_field_features import extract_cross_field_features
from src.matching_features.country_features import extract_country_features
from src.matching_features.view_features import extract_view_features
from src.matching_features.provenance_features import extract_provenance_features
from src.matching_features.refinement_features import extract_refinement_features
from src.matching_features.quality_features import extract_quality_features
from src.matching_features.frequency_features import FrequencyFeatureStore
from src.utils.logging_utils import get_logger

logger = get_logger("feature_extractor")

class PairwiseFeatureExtractor:
    """
    High-Throughput Deterministic Pairwise Feature Extraction Engine.
    
    Transforms (Source1, Candidate S2/S3 Target) pairs into structured 87-dimensional feature vectors.
    """
    def __init__(self, frequency_store: Optional[FrequencyFeatureStore] = None):
        self.frequency_store = frequency_store
        self.schema_version = FEATURE_SCHEMA_VERSION
        self.feature_columns = FEATURE_COLUMNS

    def extract_pair_features(
        self,
        q_rec: Dict[str, Any],
        t_rec: Dict[str, Any],
        provenance_strategies: Optional[Set[str]] = None
    ) -> Dict[str, float]:
        """
        Extracts all 87 pairwise features for a single candidate pair.
        """
        # 1. Name features (15)
        name_feats = extract_name_features(q_rec, t_rec)

        # 2. Address features (15)
        addr_feats = extract_address_features(q_rec, t_rec)

        # 3. Country features (4)
        country_feats = extract_country_features(q_rec, t_rec)

        # 4. Cross-field interaction features (8)
        cross_feats = extract_cross_field_features(name_feats, addr_feats, country_feats)

        # 5. Normalization-view features (5)
        view_feats = extract_view_features(q_rec, t_rec)

        # 6. Blocking provenance features (11)
        prov_feats = extract_provenance_features(provenance_strategies)

        # 7. Refinement features (7)
        refine_feats = extract_refinement_features(q_rec, t_rec, provenance_strategies=provenance_strategies)

        # 8. Record quality features (16)
        qual_feats = extract_quality_features(q_rec, t_rec)

        # 9. Frequency / Cardinality features (6)
        if self.frequency_store:
            freq_feats = self.frequency_store.extract_features(q_rec, t_rec)
        else:
            freq_feats = {
                "freq_q_name": 0.0,
                "freq_t_name": 0.0,
                "freq_q_compact": 0.0,
                "freq_t_compact": 0.0,
                "freq_q_country_name": 0.0,
                "freq_t_country_name": 0.0,
            }

        # Combine into complete dictionary
        features = {}
        features.update(name_feats)
        features.update(addr_feats)
        features.update(cross_feats)
        features.update(country_feats)
        features.update(view_feats)
        features.update(prov_feats)
        features.update(refine_feats)
        features.update(qual_feats)
        features.update(freq_feats)

        # Guard against NaN / inf
        for col in self.feature_columns:
            val = features.get(col, 0.0)
            if np.isnan(val) or np.isinf(val):
                features[col] = 0.0

        return features

    def extract_candidate_dataset(
        self,
        candidate_pairs_map: Dict[str, Set[str]],
        query_records_map: Dict[str, Dict[str, Any]],
        target_records_map: Dict[str, Dict[str, Any]],
        provenance_map: Optional[Dict[str, Dict[str, Set[str]]]] = None,
        ground_truth_map: Optional[Dict[str, Set[str]]] = None,
        chunk_size: int = 100000
    ) -> pd.DataFrame:
        """
        Extracts feature vectors for all candidate pairs in batch.
        If ground_truth_map is provided, attaches the binary 'label' column (1 for match, 0 for non-match).
        """
        rows = []
        total_pairs = sum(len(cands) for cands in candidate_pairs_map.values())
        logger.info(f"Beginning feature extraction for {total_pairs:,} candidate pairs across {len(candidate_pairs_map):,} queries.")

        for q_id, cand_set in candidate_pairs_map.items():
            q_rec = query_records_map.get(q_id)
            if not q_rec:
                continue

            q_prov = provenance_map.get(q_id, {}) if provenance_map else {}
            true_matches = ground_truth_map.get(q_id, set()) if ground_truth_map else None

            for t_id in cand_set:
                t_rec = target_records_map.get(t_id)
                if not t_rec:
                    continue

                target_strats = q_prov.get(t_id)
                feats = self.extract_pair_features(q_rec, t_rec, provenance_strategies=target_strats)

                # Determine target source
                t_source = "S2" if t_id.startswith("S2") else ("S3" if t_id.startswith("S3") else "UNKNOWN")

                row = {
                    "source1_entity_id": q_id,
                    "target_entity_id": t_id,
                    "target_source": t_source,
                }
                row.update(feats)

                if true_matches is not None:
                    row["label"] = 1 if t_id in true_matches else 0

                rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Extracted features for {len(df):,} pairs. Feature schema version: {self.schema_version}.")
        return df
