import os
from typing import Iterator, List, Optional, Set, Union
import pandas as pd
from src.utils.logging_utils import get_logger

logger = get_logger("data_loader")

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

def load_tsv_dataframe(
    file_path: str,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    use_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Loads a TSV file into a pandas DataFrame with strict string dtypes.
    
    Preserves exact string fidelity (keep_default_na=False prevents 'NA' or 'null' 
    from being cast to NaN).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    cols = use_cols if use_cols is not None else columns
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
        usecols=cols,
        engine="c"
    )
    return df

def stream_tsv_chunks(
    file_path: str,
    chunksize: int = 100_000,
    use_cols: Optional[List[str]] = None
) -> Iterator[pd.DataFrame]:
    """
    Streams a TSV file in memory-safe chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    return pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
        usecols=use_cols,
        engine="c"
    )

def stream_entity_ids(file_path: str) -> Iterator[str]:
    """
    High-performance generator that streams only entity IDs from a source TSV
    with minimal memory footprint.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            idx = line.find("\t")
            if idx != -1:
                yield line[:idx]
            else:
                yield line.rstrip("\r\n")

def load_source1(
    path: str = "datasets/train/train_source1.tsv",
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Loads Source 1 dataset (query anchors)."""
    if chunksize is not None:
        return stream_tsv_chunks(path, chunksize=chunksize, use_cols=SOURCE_COLUMNS)
    return load_tsv_dataframe(path, columns=SOURCE_COLUMNS, nrows=nrows)

def load_source2(
    path: str = "datasets/train/train_source2.tsv",
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Loads Source 2 dataset (target candidate pool A)."""
    if chunksize is not None:
        return stream_tsv_chunks(path, chunksize=chunksize, use_cols=SOURCE_COLUMNS)
    return load_tsv_dataframe(path, columns=SOURCE_COLUMNS, nrows=nrows)

def load_source3(
    path: str = "datasets/train/train_source3.tsv",
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Loads Source 3 dataset (target candidate pool B)."""
    if chunksize is not None:
        return stream_tsv_chunks(path, chunksize=chunksize, use_cols=SOURCE_COLUMNS)
    return load_tsv_dataframe(path, columns=SOURCE_COLUMNS, nrows=nrows)

def load_ground_truth(
    path: str = "datasets/train/train_ground_truth.tsv",
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Loads Ground Truth table."""
    if chunksize is not None:
        return stream_tsv_chunks(path, chunksize=chunksize, use_cols=GROUND_TRUTH_COLUMNS)
    return load_tsv_dataframe(path, columns=GROUND_TRUTH_COLUMNS, nrows=nrows)
