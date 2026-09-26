from typing import Dict, Any
from src.matching_features.string_metrics import has_non_latin
from src.candidate_generation.normalizer import (
    URL_PREFIX_PATTERN,
    DOMAIN_SUFFIX_PATTERN,
    COMBINED_LEGAL_SUFFIX_RE,
    DBA_PATTERN
)

def extract_quality_features(q_rec: Dict[str, Any], t_rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts 16 Record Quality features describing the noise, lengths, script, and artifact characteristics.
    """
    q_name = q_rec.get("name", {})
    t_name = t_rec.get("name", {})
    q_addr = q_rec.get("address", {})
    t_addr = t_rec.get("address", {})

    q_name_raw = q_name.get("raw", "")
    t_name_raw = t_name.get("raw", "")
    q_addr_raw = q_addr.get("raw", "")
    t_addr_raw = t_addr.get("raw", "")

    q_name_tokens = q_name.get("tokens", [])
    t_name_tokens = t_name.get("tokens", [])
    q_addr_tokens = q_addr.get("tokens", [])
    t_addr_tokens = t_addr.get("tokens", [])

    # Lengths
    q_n_len = float(len(q_name_raw))
    t_n_len = float(len(t_name_raw))
    q_a_len = float(len(q_addr_raw))
    t_a_len = float(len(t_addr_raw))

    # Token counts
    q_n_tc = float(len(q_name_tokens))
    t_n_tc = float(len(t_name_tokens))
    q_a_tc = float(len(q_addr_tokens))
    t_a_tc = float(len(t_addr_tokens))

    # Missing indicators
    addr_miss_q = 1.0 if not q_addr_raw.strip() else 0.0
    addr_miss_t = 1.0 if not t_addr_raw.strip() else 0.0

    # Non-Latin script detection
    non_latin_q = 1.0 if (has_non_latin(q_name_raw) or has_non_latin(q_addr_raw)) else 0.0
    non_latin_t = 1.0 if (has_non_latin(t_name_raw) or has_non_latin(t_addr_raw)) else 0.0
    mixed_script = 1.0 if (non_latin_q != non_latin_t) else 0.0

    # Artifact patterns
    has_url = 1.0 if (
        URL_PREFIX_PATTERN.search(q_name_raw) or URL_PREFIX_PATTERN.search(t_name_raw)
        or DOMAIN_SUFFIX_PATTERN.search(q_name_raw) or DOMAIN_SUFFIX_PATTERN.search(t_name_raw)
    ) else 0.0

    has_legal = 1.0 if (
        COMBINED_LEGAL_SUFFIX_RE.search(q_name_raw) or COMBINED_LEGAL_SUFFIX_RE.search(t_name_raw)
    ) else 0.0

    has_dba = 1.0 if (
        DBA_PATTERN.search(q_name_raw) or DBA_PATTERN.search(t_name_raw)
        or q_name.get("dba_compacts") or t_name.get("dba_compacts")
    ) else 0.0

    return {
        "qual_q_name_len": q_n_len,
        "qual_t_name_len": t_n_len,
        "qual_q_addr_len": q_a_len,
        "qual_t_addr_len": t_a_len,
        "qual_q_name_token_count": q_n_tc,
        "qual_t_name_token_count": t_n_tc,
        "qual_q_addr_token_count": q_a_tc,
        "qual_t_addr_token_count": t_a_tc,
        "qual_addr_missing_q": addr_miss_q,
        "qual_addr_missing_t": addr_miss_t,
        "qual_non_latin_script_q": non_latin_q,
        "qual_non_latin_script_t": non_latin_t,
        "qual_mixed_script_pair": mixed_script,
        "qual_has_url_domain": has_url,
        "qual_has_legal_suffix": has_legal,
        "qual_has_dba": has_dba,
    }
