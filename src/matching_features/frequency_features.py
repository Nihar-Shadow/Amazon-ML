from typing import Dict, Any, Iterator, Tuple
from collections import Counter
import math

class FrequencyFeatureStore:
    """
    Leakage-Safe Frequency / Cardinality Feature Store.
    
    Guarantees:
    - MUST be fitted exclusively on training entities.
    - Validation entities NEVER update or contribute to frequency statistics.
    - Any unseen validation/test entity defaults cleanly to 0.0 frequency.
    """
    def __init__(self):
        self.name_counts: Counter = Counter()
        self.compact_counts: Counter = Counter()
        self.country_compact_counts: Counter = Counter()
        self.is_fitted: bool = False

    def fit(self, records_iterator: Iterator[Dict[str, Any]]):
        """
        Fits frequency statistics on training records ONLY.
        """
        for rec in records_iterator:
            name_info = rec.get("name", {})
            norm_name = name_info.get("normalized", "")
            compact_name = name_info.get("compact", "")
            country = rec.get("country", "")

            if norm_name:
                self.name_counts[norm_name] += 1
            if compact_name:
                self.compact_counts[compact_name] += 1
                if country:
                    self.country_compact_counts[f"{country}|{compact_name}"] += 1

        self.is_fitted = True

    def extract_features(self, q_rec: Dict[str, Any], t_rec: Dict[str, Any]) -> Dict[str, float]:
        """
        Extracts 6 frequency features for a pair using the fitted training statistics.
        """
        q_name = q_rec.get("name", {})
        t_name = t_rec.get("name", {})
        q_norm = q_name.get("normalized", "")
        t_norm = t_name.get("normalized", "")
        q_comp = q_name.get("compact", "")
        t_comp = t_name.get("compact", "")
        q_c = q_rec.get("country", "")
        t_c = t_rec.get("country", "")

        q_name_cnt = float(self.name_counts.get(q_norm, 0))
        t_name_cnt = float(self.name_counts.get(t_norm, 0))
        q_comp_cnt = float(self.compact_counts.get(q_comp, 0))
        t_comp_cnt = float(self.compact_counts.get(t_comp, 0))
        q_c_comp_cnt = float(self.country_compact_counts.get(f"{q_c}|{q_comp}", 0))
        t_c_comp_cnt = float(self.country_compact_counts.get(f"{t_c}|{t_comp}", 0))

        return {
            "freq_q_name": q_name_cnt,
            "freq_t_name": t_name_cnt,
            "freq_q_compact": q_comp_cnt,
            "freq_t_compact": t_comp_cnt,
            "freq_q_country_name": q_c_comp_cnt,
            "freq_t_country_name": t_c_comp_cnt,
        }
