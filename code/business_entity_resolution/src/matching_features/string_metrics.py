import re
import unicodedata
from typing import Set, Tuple, List, Optional

def compute_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def get_char_ngrams(text: str, n: int) -> Set[str]:
    """Generates set of character n-grams for a string."""
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def fast_levenshtein_distance(s1: str, s2: str, max_eval_len: int = 80) -> int:
    """
    Computes Levenshtein edit distance with two-row DP.
    Truncates strings beyond max_eval_len for high-throughput efficiency.
    """
    if s1 == s2:
        return 0
    if not s1:
        return min(len(s2), max_eval_len)
    if not s2:
        return min(len(s1), max_eval_len)
    
    if len(s1) > max_eval_len:
        s1 = s1[:max_eval_len]
    if len(s2) > max_eval_len:
        s2 = s2[:max_eval_len]
        
    m, n = len(s1), len(s2)
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m

    v0 = list(range(n + 1))
    v1 = [0] * (n + 1)

    for i in range(m):
        v1[0] = i + 1
        c1 = s1[i]
        for j in range(n):
            c = v0[j] if c1 == s2[j] else v0[j] + 1
            a = v1[j] + 1
            b = v0[j + 1] + 1
            v1[j + 1] = c if c <= a and c <= b else (a if a <= b else b)
        v0, v1 = v1, v0

    return v0[n]

def normalized_edit_similarity(s1: str, s2: str, max_eval_len: int = 80) -> float:
    """Computes 1.0 - (levenshtein / max_len). Returns 0.0 for empty inputs."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    max_len = max(min(len(s1), max_eval_len), min(len(s2), max_eval_len))
    if max_len == 0:
        return 0.0
    dist = fast_levenshtein_distance(s1, s2, max_eval_len=max_eval_len)
    return max(0.0, 1.0 - (dist / max_len))

def common_prefix_length(s1: str, s2: str) -> int:
    """Returns the length of the longest common prefix."""
    limit = min(len(s1), len(s2))
    for i in range(limit):
        if s1[i] != s2[i]:
            return i
    return limit

def common_suffix_length(s1: str, s2: str) -> int:
    """Returns the length of the longest common suffix."""
    limit = min(len(s1), len(s2))
    for i in range(1, limit + 1):
        if s1[-i] != s2[-i]:
            return i - 1
    return limit

def has_non_latin(text: str) -> bool:
    """Returns True if text contains characters outside basic Latin/Latin Extended."""
    if not text:
        return False
    for char in text:
        # Codepoints > 0x024F are outside Basic Latin & Latin Extended-A/B
        if ord(char) > 0x024F and char.isalpha():
            return True
    return False
