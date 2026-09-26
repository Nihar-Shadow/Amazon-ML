from typing import Dict, Set, Optional

def extract_provenance_features(provenance_strategies: Optional[Set[str]]) -> Dict[str, float]:
    """
    Extracts 11 Blocking Provenance features indicating which strategy generated the pair.
    """
    strats = provenance_strategies or set()

    return {
        "prov_exact_name": 1.0 if "exact_name" in strats else 0.0,
        "prov_sorted_tokens": 1.0 if "sorted_name_tokens" in strats else 0.0,
        "prov_first_two_tokens": 1.0 if "first_two_name_tokens" in strats else 0.0,
        "prov_longest_token": 1.0 if "longest_name_token" in strats else 0.0,
        "prov_name_prefix": 1.0 if "name_prefix_6" in strats else 0.0,
        "prov_name_address": 1.0 if "name_address_combo" in strats else 0.0,
        "prov_address_exact": 1.0 if "address_exact_combo" in strats else 0.0,
        "prov_name_leet": 1.0 if "name_leet_compact" in strats else 0.0,
        "prov_name_dba": 1.0 if "name_dba_alias" in strats else 0.0,
        "prov_address_salient": 1.0 if "address_salient_combo" in strats else 0.0,
        "prov_strategy_count": float(len(strats)),
    }
