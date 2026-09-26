from typing import Dict

def extract_cross_field_features(
    name_feats: Dict[str, float],
    addr_feats: Dict[str, float],
    country_feats: Dict[str, float]
) -> Dict[str, float]:
    """
    Extracts 8 Cross-Field interaction features.
    Distinguishes complementary multi-field evidence from isolated similarities.
    """
    name_jacc = name_feats.get("name_token_jaccard", 0.0)
    addr_jacc = addr_feats.get("addr_token_jaccard", 0.0)
    exact_num = addr_feats.get("addr_exact_number_match", 0.0)
    locality = addr_feats.get("addr_locality_overlap", 0.0)
    country_match = country_feats.get("country_exact_match", 0.0)

    # 1. Strong name + Strong address
    strong_both = 1.0 if (name_jacc >= 0.7 and addr_jacc >= 0.5) else 0.0

    # 2. Strong name + Weak address (e.g., entity moved or address incomplete)
    strong_n_weak_a = 1.0 if (name_jacc >= 0.8 and addr_jacc < 0.2) else 0.0

    # 3. Weak name + Strong address (e.g., cross-script name or rebrand at same location)
    weak_n_strong_a = 1.0 if (name_jacc < 0.3 and addr_jacc >= 0.7) else 0.0

    # 4. Exact building number + Similar name
    num_and_name = 1.0 if (exact_num == 1.0 and name_jacc >= 0.5) else 0.0

    # 5. Locality overlap + Similar name
    loc_and_name = 1.0 if (locality >= 0.5 and name_jacc >= 0.5) else 0.0

    # 6. Same country * Name similarity
    c_name_sim = country_match * name_jacc

    # 7. Same country * Address similarity
    c_addr_sim = country_match * addr_jacc

    # 8. Multi-field agreement count
    agreement_count = (
        name_feats.get("name_normalized_exact", 0.0) +
        name_feats.get("name_compact_exact", 0.0) +
        addr_feats.get("addr_normalized_exact", 0.0) +
        addr_feats.get("addr_exact_number_match", 0.0) +
        country_match +
        (1.0 if locality >= 0.5 else 0.0)
    )

    return {
        "cross_strong_name_strong_addr": strong_both,
        "cross_strong_name_weak_addr": strong_n_weak_a,
        "cross_weak_name_strong_addr": weak_n_strong_a,
        "cross_exact_number_similar_name": num_and_name,
        "cross_exact_locality_similar_name": loc_and_name,
        "cross_same_country_name_sim": c_name_sim,
        "cross_same_country_addr_sim": c_addr_sim,
        "cross_agreement_count": float(agreement_count),
    }
