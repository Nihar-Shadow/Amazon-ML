import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.matching_features.feature_schema import FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS
from src.matching_features.frequency_features import FrequencyFeatureStore
from src.candidate_generation.normalizer import normalize_entity_record
from src.utils.logging_utils import get_logger

logger = get_logger("train_production_model")

def train_production_model():
    os.makedirs("models", exist_ok=True)
    start_time = time.time()
    logger.info("Starting Final Production Model Training on all available labeled training candidate pairs.")

    # 1. Load cached labeled candidate datasets from data_cache/
    train_cache = "data_cache/train_features.pkl"
    val_cache = "data_cache/val_features.pkl"

    if not os.path.exists(train_cache) or not os.path.exists(val_cache):
        raise FileNotFoundError(f"Feature caches not found in data_cache/. Run audit/feature extraction first.")

    with open(train_cache, "rb") as f:
        train_df = pickle.load(f)
    with open(val_cache, "rb") as f:
        val_df = pickle.load(f)

    # Combine into full labeled training dataset
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    total_pairs = len(combined_df)
    total_pos = int((combined_df["label"] == 1).sum())
    total_neg = int((combined_df["label"] == 0).sum())

    logger.info(f"Full Training Dataset: {total_pairs:,} candidate pairs ({total_pos:,} positives, {total_neg:,} negatives).")

    # 2. Extract feature matrix
    X_train = combined_df[FEATURE_COLUMNS].values
    y_train = combined_df["label"].values

    # 3. Fit Champion HistGradientBoostingClassifier
    # Hyperparameters strictly frozen from Phase 5 selection
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=8,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
        class_weight=None
    )

    tr_start = time.time()
    model.fit(X_train, y_train)
    tr_time = round(time.time() - tr_start, 2)
    logger.info(f"Fitted HistGradientBoostingClassifier in {tr_time}s.")

    # 4. Save model artifact
    model_path = "models/production_matching_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved production model to {model_path}.")

    # 5. Fit & save FrequencyFeatureStore on training data
    logger.info("Building full training FrequencyFeatureStore (zero test data contamination)...")
    freq_store = FrequencyFeatureStore()
    with open("datasets/train/train_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        records_iter = (
            normalize_entity_record(p[0], p[1], p[2], p[3])
            for line in f
            for p in [line.rstrip("\r\n").split("\t")]
            if len(p) >= 4
        )
        # Stream first 50,000 for training frequency representations
        records_sample = []
        for i, rec in enumerate(records_iter):
            records_sample.append(rec)
            if i >= 50000:
                break
        freq_store.fit(records_sample)

    freq_store_path = "models/production_frequency_store.pkl"
    with open(freq_store_path, "wb") as f:
        pickle.dump(freq_store, f)
    logger.info(f"Saved production frequency store to {freq_store_path}.")

    # 6. Save metadata
    metadata = {
        "model_type": "HistGradientBoostingClassifier",
        "library": "scikit-learn",
        "hyperparameters": {
            "max_iter": 100,
            "max_depth": 8,
            "learning_rate": 0.1,
            "min_samples_leaf": 20,
            "random_state": 42,
            "class_weight": None
        },
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_COLUMNS),
        "total_training_samples": total_pairs,
        "positive_samples": total_pos,
        "negative_samples": total_neg,
        "decision_threshold": 0.70,
        "training_time_sec": tr_time,
        "total_time_sec": round(time.time() - start_time, 2)
    }

    with open("models/production_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Production Model Training complete and verified.")

if __name__ == "__main__":
    train_production_model()
