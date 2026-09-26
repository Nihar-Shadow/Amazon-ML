import re
import unicodedata
from typing import Dict, List, Set, Tuple, Any

# Regex for URL/domain removal
URL_PREFIX_PATTERN = re.compile(r"^https?://(www\.)?|^www\.", re.IGNORECASE)
DOMAIN_SUFFIX_PATTERN = re.compile(
    r"\.(com|org|net|in|fr|co|io|biz|info|gov|edu|us|eu|me|tech|store)(\.[a-z]{2})?$",
    re.IGNORECASE
)

# Common legal suffixes across US, India, France, and international jurisdictions
LEGAL_SUFFIX_PATTERNS = [
    # US / UK / International
    r"\b(l\.?l\.?c\.?|limited liability company)\b",
    r"\b(inc\.?|incorporated)\b",
    r"\b(corp\.?|corporation)\b",
    r"\b(ltd\.?|limited)\b",
    r"\b(co\.?|company)\b",
    r"\b(plc\.?|public limited company)\b",
    r"\b(gmbh|ag|bv|nv)\b",
    # India
    r"\b(pvt\.?\s*ltd\.?|private\s+limited)\b",
    r"\b(l\.?l\.?p\.?|limited liability partnership)\b",
    r"\b(pvt\.?|private)\b",
    # France
    r"\b(s\.?a\.?r\.?l\.?|societe a responsabilite limitee)\b",
    r"\b(s\.?a\.?s\.?|s\.?a\.?s\.?u\.?|societe par actions simplifiee)\b",
    r"\b(s\.?c\.?i\.?|societe civile immobiliere)\b",
    r"\b(e\.?u\.?r\.?l\.?)\b",
    r"\b(s\.?a\.?|societe anonyme)\b",
    r"\b(snc|gie|ei)\b",
]
COMBINED_LEGAL_SUFFIX_RE = re.compile(
    r"|".join(LEGAL_SUFFIX_PATTERNS),
    re.IGNORECASE
)

# Address abbreviations mapping (standardization to canonical full tokens)
ADDRESS_ABBREVIATIONS = {
    "st": "street", "str": "street",
    "rd": "road",
    "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "pkwy": "parkway", "pky": "parkway",
    "trl": "trail",
    "ct": "court",
    "pl": "place",
    "cir": "circle",
    "hwy": "highway",
    "fwy": "freeway",
    "expy": "expressway",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "ctr": "center",
    "sq": "square",
    "ter": "terrace",
    "al": "alley",
    "plz": "plaza",
    "pk": "park",
    "gt": "gate",
    "no": "number"
}

# Generic noise words to exclude from primary name token keys
NAME_STOPWORDS = {
    "the", "and", "of", "for", "in", "at", "by", "to", "on", "a", "an",
    "de", "du", "la", "le", "des", "et", "en", "pour", "par", "sur",
    "ka", "ki", "ke", "aur", "se"
}

def strip_accents(text: str) -> str:
    """Normalizes Unicode text, expands common ligatures, and decomposes accents / diacritics."""
    if not text:
        return ""
    # Expand French / Latin ligatures
    text = (
        text.replace("œ", "oe")
        .replace("Œ", "Oe")
        .replace("æ", "ae")
        .replace("Æ", "Ae")
    )
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))

def normalize_country(country: str) -> str:
    """
    Normalizes country code/name cleanly.
    Open-set: Does NOT filter or restrict to known countries.
    """
    if not country:
        return ""
    return country.strip().upper()

# Common street generic words to distinguish from salient city/locality tokens
COMMON_STREET_WORDS = {
    "street", "road", "avenue", "boulevard", "drive", "lane", "parkway",
    "trail", "court", "place", "circle", "highway", "freeway", "expressway",
    "suite", "apartment", "floor", "building", "department", "center", "square",
    "near", "opp", "opposite", "beside", "behind", "block", "sector", "plot"
}

# DBA and Trade name patterns
DBA_PATTERN = re.compile(
    r"\b(dba|t/a|f/k/a|trading\s+as|doing\s+business\s+as|c/o)\b",
    re.IGNORECASE
)
HONORIFIC_PATTERN = re.compile(r"^\s*(shri|shree|m/s|smt|smts)\s+", re.IGNORECASE)

def normalize_leetspeak(text: str) -> str:
    """
    Normalizes common character/digit substitutions inside words:
    e.g., '@' -> 'a', '0' inside word -> 'o', '8' inside/before word -> 'b', '1' inside word -> 'l'.
    Preserves standalone numbers and strips handle prefix symbols.
    """
    if not text:
        return ""
    # Strip leading handle symbols (@, #)
    s = re.sub(r"^[@#]+", "", text)
    s = s.replace("@", "a").replace("$", "s")
    # Sub 0 inside alphabetic letters
    s = re.sub(r"(?<=[a-zA-Z])0(?=[a-zA-Z])", "o", s)
    s = re.sub(r"(?<=[a-zA-Z])8(?=[a-zA-Z])", "b", s)
    s = re.sub(r"(?<=[a-zA-Z])1(?=[a-zA-Z])", "l", s)
    s = re.sub(r"\b8(?=[a-zA-Z])", "b", s)  # e.g. 8rothers -> brothers
    return s

def extract_dba_names(raw_name: str) -> List[str]:
    """Extracts sub-names if a trade name or DBA clause is present."""
    if not raw_name:
        return []
    aliases = []
    clean_h = HONORIFIC_PATTERN.sub("", raw_name).strip()
    if clean_h and clean_h.lower() != raw_name.lower():
        aliases.append(clean_h)
        
    m = DBA_PATTERN.search(raw_name)
    if m:
        before = raw_name[:m.start()].strip()
        after = raw_name[m.end():].strip()
        if before and len(before) >= 3:
            aliases.append(before)
        if after and len(after) >= 3:
            aliases.append(after)
    return aliases

def normalize_business_name(raw_name: str) -> Dict[str, Any]:
    """
    Produces multi-view representations of a business name:
    - raw: unmodified input
    - normalized: lowercased, accent-stripped, punctuation-normalized, legal suffixes removed
    - compact: normalized with all spaces and symbols removed
    - tokens: list of significant alpha-numeric tokens
    - sorted_tokens_str: whitespace-joined sorted unique tokens (order-invariant)
    - ngrams: set of character 3-grams
    - leet_compact: compact representation after leetspeak normalization
    - dba_compacts: compact representations of extracted trade/DBA names
    """
    if not raw_name:
        return {
            "raw": "",
            "normalized": "",
            "compact": "",
            "tokens": [],
            "token_set": set(),
            "sorted_tokens_str": "",
            "ngrams": set(),
            "leet_compact": "",
            "dba_compacts": []
        }

    # 1. Unicode decomposition
    text = strip_accents(raw_name).lower()
    
    # 2. URL and domain handling
    text = URL_PREFIX_PATTERN.sub("", text)
    text = DOMAIN_SUFFIX_PATTERN.sub("", text)
    
    # 3. Handle symbols like '&' -> 'and', '+' -> 'plus'
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\+", " plus ", text)
    
    # 4. Remove legal entity suffixes
    text = COMBINED_LEGAL_SUFFIX_RE.sub(" ", text)
    
    # 5. Remove punctuation, keep letters and digits
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # 6. Extract tokens
    tokens = [t for t in text.split() if t not in NAME_STOPWORDS and len(t) >= 1]
    
    # 7. Compact form
    compact = re.sub(r"\s+", "", text)
    
    # 8. Order-invariant token string
    sorted_unique = sorted(set(tokens))
    sorted_tokens_str = " ".join(sorted_unique)
    
    # 9. Character 3-grams for fuzzy matching
    ngrams = set()
    if len(compact) >= 3:
        for i in range(len(compact) - 2):
            ngrams.add(compact[i:i+3])
    elif compact:
        ngrams.add(compact)

    # 10. Leetspeak normalized compact view
    leet_text = normalize_leetspeak(strip_accents(raw_name).lower())
    leet_text = COMBINED_LEGAL_SUFFIX_RE.sub(" ", leet_text)
    leet_text = re.sub(r"[^\w\s]", " ", leet_text)
    leet_compact = re.sub(r"\s+", "", leet_text)

    # 11. DBA aliases
    dba_compacts = []
    for alias in extract_dba_names(raw_name):
        alias_norm = strip_accents(alias).lower()
        alias_norm = COMBINED_LEGAL_SUFFIX_RE.sub(" ", alias_norm)
        alias_norm = re.sub(r"[^\w\s]", " ", alias_norm)
        ac = re.sub(r"\s+", "", alias_norm)
        if ac and ac != compact:
            dba_compacts.append(ac)

    return {
        "raw": raw_name,
        "normalized": text,
        "compact": compact,
        "tokens": tokens,
        "token_set": set(tokens),
        "sorted_tokens_str": sorted_tokens_str,
        "ngrams": ngrams,
        "leet_compact": leet_compact,
        "dba_compacts": dba_compacts
    }

def normalize_business_address(raw_address: str) -> Dict[str, Any]:
    """
    Produces multi-view representations of a business address:
    - raw: unmodified input
    - normalized: lowercased, accent-stripped, structural abbreviations expanded
    - compact: normalized without spaces/punctuation
    - tokens: significant word tokens
    - numbers: set of extracted digit sequences (leading zeros stripped)
    - salient_tokens: tokens excluding generic street words (e.g. city/locality names)
    """
    if not raw_address or raw_address.strip() == "":
        return {
            "raw": "",
            "normalized": "",
            "compact": "",
            "tokens": [],
            "token_set": set(),
            "numbers": set(),
            "salient_tokens": []
        }

    # 1. Unicode decomposition
    text = strip_accents(raw_address).lower()
    
    # 2. Extract numeric sequences (strip leading zeros so '00501' == '501')
    raw_nums = re.findall(r"\b\d{1,7}\b", text)
    numbers = set((n.lstrip("0") or "0") for n in raw_nums)
    
    # 3. Replace punctuation with space
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # 4. Standardize abbreviations
    words = text.split()
    normalized_words = [ADDRESS_ABBREVIATIONS.get(w, w) for w in words]
    normalized_text = " ".join(normalized_words)
    
    # 5. Extract significant tokens (excluding single-character noise)
    tokens = [w for w in normalized_words if len(w) >= 2]
    compact = re.sub(r"\s+", "", normalized_text)
    
    # 6. Salient address tokens (exclude generic street words)
    salient_tokens = [w for w in tokens if w not in COMMON_STREET_WORDS and len(w) >= 4]

    return {
        "raw": raw_address,
        "normalized": normalized_text,
        "compact": compact,
        "tokens": tokens,
        "token_set": set(tokens),
        "numbers": numbers,
        "salient_tokens": salient_tokens
    }


def normalize_entity_record(
    entity_id: str,
    business_name: str,
    business_address: str,
    country: str
) -> Dict[str, Any]:
    """Combines all field normalizations into a single structured record."""
    return {
        "entity_id": entity_id,
        "country": normalize_country(country),
        "name": normalize_business_name(business_name),
        "address": normalize_business_address(business_address)
    }
