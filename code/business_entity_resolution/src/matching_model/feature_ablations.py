from typing import Dict, List, Any
from src.matching_features.feature_schema import FEATURE_COLUMNS

# Feature groups definitions based on schema
NAME_FEATURES = FEATURE_COLUMNS[0:15]
ADDRESS_FEATURES = FEATURE_COLUMNS[15:30]
CROSS_FIELD_FEATURES = FEATURE_COLUMNS[30:38]
COUNTRY_FEATURES = FEATURE_COLUMNS[38:42]
VIEW_FEATURES = FEATURE_COLUMNS[42:47]
PROVENANCE_FEATURES = FEATURE_COLUMNS[47:58]
REFINEMENT_FEATURES = FEATURE_COLUMNS[58:65]
QUALITY_FEATURES = FEATURE_COLUMNS[65:81]
FREQUENCY_FEATURES = FEATURE_COLUMNS[81:87]

FEATURE_ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "EXP-015": {
        "name": "Name Features Only",
        "columns": NAME_FEATURES,
        "count": len(NAME_FEATURES)
    },
    "EXP-016": {
        "name": "Name + Address",
        "columns": NAME_FEATURES + ADDRESS_FEATURES,
        "count": len(NAME_FEATURES + ADDRESS_FEATURES)
    },
    "EXP-017": {
        "name": "Name + Address + Cross-Field",
        "columns": NAME_FEATURES + ADDRESS_FEATURES + CROSS_FIELD_FEATURES,
        "count": len(NAME_FEATURES + ADDRESS_FEATURES + CROSS_FIELD_FEATURES)
    },
    "EXP-018": {
        "name": "All Similarity Features (Name+Addr+Cross+Country+View)",
        "columns": NAME_FEATURES + ADDRESS_FEATURES + CROSS_FIELD_FEATURES + COUNTRY_FEATURES + VIEW_FEATURES,
        "count": len(NAME_FEATURES + ADDRESS_FEATURES + CROSS_FIELD_FEATURES + COUNTRY_FEATURES + VIEW_FEATURES)
    },
    "EXP-019": {
        "name": "All Similarity + Provenance",
        "columns": (NAME_FEATURES + ADDRESS_FEATURES + CROSS_FIELD_FEATURES + COUNTRY_FEATURES +
                    VIEW_FEATURES + PROVENANCE_FEATURES),
        "count": 58
    },
    "EXP-020": {
        "name": "All Similarity + Provenance + Refinement",
        "columns": (NAME_FEATURES + ADDRESS_FEATURES + CROSS_FIELD_FEATURES + COUNTRY_FEATURES +
                    VIEW_FEATURES + PROVENANCE_FEATURES + REFINEMENT_FEATURES),
        "count": 65
    },
    "EXP-021": {
        "name": "All 87 Features (Full Feature Set)",
        "columns": FEATURE_COLUMNS,
        "count": 87
    },
    "EXP-022": {
        "name": "All Features Minus Frequency Features",
        "columns": [c for c in FEATURE_COLUMNS if c not in FREQUENCY_FEATURES],
        "count": 81
    }
}
