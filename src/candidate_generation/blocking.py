from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from src.utils.logging_utils import get_logger

logger = get_logger("blocking_engine")

class BlockingIndex:
    """
    Inverted Index for high-performance multi-strategy blocking.
    Maps composite blocking keys to lists of candidate target entity IDs.
    """
    def __init__(self, max_block_size: int = 500):
        self.max_block_size = max_block_size
        # index: strategy_name -> key -> list of target_ids
        self.indices: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.total_target_records = 0

    def add_target_record(self, record: Dict[str, Any], strategies: Optional[List[str]] = None):
        """Extracts keys and adds a target entity record to the index."""
        self.total_target_records += 1
        target_id = record["entity_id"]
        keys_by_strategy = extract_blocking_keys(record, strategies=strategies, is_query=False)
        
        for strategy, keys in keys_by_strategy.items():
            strat_index = self.indices[strategy]
            for key in keys:
                strat_index[key].append(target_id)

    def retrieve_candidates_for_query(
        self,
        query_record: Dict[str, Any],
        strategies: Optional[List[str]] = None,
        return_provenance: bool = False
    ) -> Any:
        """
        Retrieves candidate target IDs for an S1 query record across all active blocking strategies.
        Returns:
            If return_provenance is False: Set[str] of candidate IDs.
            If return_provenance is True: Dict[str, Set[str]] mapping target_id -> set of strategy names.
        """
        keys_by_strategy = extract_blocking_keys(query_record, strategies=strategies, is_query=True)
        
        if return_provenance:
            candidates_with_provenance: Dict[str, Set[str]] = defaultdict(set)
            for strategy, keys in keys_by_strategy.items():
                strat_index = self.indices.get(strategy)
                if not strat_index:
                    continue
                for key in keys:
                    target_ids = strat_index.get(key)
                    if target_ids and len(target_ids) <= self.max_block_size:
                        for tid in target_ids:
                            candidates_with_provenance[tid].add(strategy)
            return dict(candidates_with_provenance)
        else:
            candidates: Set[str] = set()
            for strategy, keys in keys_by_strategy.items():
                strat_index = self.indices.get(strategy)
                if not strat_index:
                    continue
                for key in keys:
                    target_ids = strat_index.get(key)
                    if target_ids and len(target_ids) <= self.max_block_size:
                        candidates.update(target_ids)
            return candidates

def extract_blocking_keys(
    record: Dict[str, Any],
    strategies: Optional[List[str]] = None,
    is_query: bool = False
) -> Dict[str, List[str]]:
    """
    Extracts blocking keys for a normalized entity record.
    
    Strategies implemented:
    1. 'exact_name': Country + exact compact name
    2. 'sorted_name_tokens': Country + sorted unique tokens (order-invariant)
    3. 'first_two_name_tokens': Country + first 2 significant name tokens
    4. 'longest_name_token': Country + longest name token (min length 4)
    5. 'name_prefix_6': Country + first 6 characters of compact name
    6. 'name_address_combo': Country + first name token + first address number
    7. 'address_exact_combo': Country + primary address number + first address token
    """
    all_strategies = {
        "exact_name",
        "sorted_name_tokens",
        "first_two_name_tokens",
        "longest_name_token",
        "name_prefix_6",
        "name_address_combo",
        "address_exact_combo",
        "name_leet_compact",
        "name_dba_alias",
        "address_salient_combo"
    }
    active_strategies = set(strategies) if strategies is not None else all_strategies

    country = record.get("country", "")
    name_info = record.get("name", {})
    addr_info = record.get("address", {})

    compact_name = name_info.get("compact", "")
    leet_compact = name_info.get("leet_compact", "")
    dba_compacts = name_info.get("dba_compacts", [])
    tokens = name_info.get("tokens", [])
    sorted_tokens_str = name_info.get("sorted_tokens_str", "")
    
    addr_tokens = addr_info.get("tokens", [])
    salient_addr_tokens = addr_info.get("salient_tokens", [])
    addr_numbers = sorted(addr_info.get("numbers", []))

    keys: Dict[str, List[str]] = defaultdict(list)

    # Strategy 1: Exact compact name within country
    if "exact_name" in active_strategies and compact_name:
        keys["exact_name"].append(f"{country}|{compact_name}")

    # Strategy 2: Sorted name tokens (handles word reordering)
    if "sorted_name_tokens" in active_strategies and sorted_tokens_str:
        keys["sorted_name_tokens"].append(f"{country}|{sorted_tokens_str}")

    # Strategy 3: First two name tokens
    if "first_two_name_tokens" in active_strategies and len(tokens) >= 2:
        k3 = f"{country}|{tokens[0]}_{tokens[1]}"
        keys["first_two_name_tokens"].append(k3)

    # Strategy 4: Longest name token (salient word, min length 5)
    if "longest_name_token" in active_strategies and tokens:
        longest = max(tokens, key=len)
        if len(longest) >= 5:
            keys["longest_name_token"].append(f"{country}|{longest}")

    # Strategy 5: Name prefix (first 6 characters of compact name)
    if "name_prefix_6" in active_strategies and len(compact_name) >= 6:
        keys["name_prefix_6"].append(f"{country}|{compact_name[:6]}")

    # Strategy 6: Name + Address number combination
    if "name_address_combo" in active_strategies and tokens and addr_numbers:
        primary_num = addr_numbers[0]
        keys["name_address_combo"].append(f"{country}|{tokens[0]}|{primary_num}")

    # Strategy 7: Address number + Address token combination (robust when name varies)
    if "address_exact_combo" in active_strategies and addr_tokens and addr_numbers:
        primary_num = addr_numbers[0]
        primary_word = addr_tokens[0]
        keys["address_exact_combo"].append(f"{country}|{primary_num}|{primary_word}")

    # Strategy 8: Leetspeak-normalized compact name (catches @r0yal -> royal, 8rothers -> brothers)
    if "name_leet_compact" in active_strategies and leet_compact:
        keys["name_leet_compact"].append(f"{country}|{leet_compact}")

    # Strategy 9: DBA / Trade-name alias extraction
    if "name_dba_alias" in active_strategies:
        for dba_c in dba_compacts:
            keys["name_dba_alias"].append(f"{country}|{dba_c}")
        if is_query and compact_name:
            keys["name_dba_alias"].append(f"{country}|{compact_name}")

    # Strategy 10: Address Number + Salient Locality/City Token (handles cross-script / brand corruption)
    if "address_salient_combo" in active_strategies and addr_numbers and salient_addr_tokens:
        primary_num = addr_numbers[0]
        salient_word = salient_addr_tokens[0]
        keys["address_salient_combo"].append(f"{country}|{primary_num}|{salient_word}")

    return dict(keys)

