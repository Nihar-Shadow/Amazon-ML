import time
import pickle
from typing import Dict, List, Set, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.matching_features.feature_schema import FEATURE_COLUMNS
from src.utils.logging_utils import get_logger

logger = get_logger("model_trainer")

def train_model(
    model_id: str,
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    class_weight: Optional[str] = None,
    random_state: int = 42
) -> Tuple[Any, float]:
    """
    Trains a classification model on candidate pair features.
    Returns:
        (fitted_model, training_time_sec)
    """
    cols = feature_cols or FEATURE_COLUMNS
    X_train = train_df[cols].values
    y_train = train_df["label"].values

    start_time = time.time()
    
    if model_id == "MODEL-001" or "logistic" in model_id.lower():
        # Standardize for logistic regression stability
        base_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
            solver="lbfgs"
        )
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", base_model)
        ])
        model.fit(X_train, y_train)

    elif model_id == "MODEL-002" or "random_forest" in model_id.lower():
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=5,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

    elif model_id == "MODEL-003" or "hist_gradient_boosting" in model_id.lower():
        cw = "balanced" if class_weight == "balanced" else None
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=8,
            learning_rate=0.1,
            min_samples_leaf=20,
            class_weight=cw,
            random_state=random_state
        )
        model.fit(X_train, y_train)

    elif model_id == "MODEL-004" or "extra_trees" in model_id.lower():
        model = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=5,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    train_time = round(time.time() - start_time, 2)
    logger.info(f"Trained {model_id} in {train_time}s on {len(X_train):,} samples ({len(cols)} features).")
    return model, train_time

def predict_probabilities(
    model: Any,
    eval_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, float]:
    """
    Predicts positive match probability P(target is match | query, target).
    Returns:
        (probabilities_array, inference_time_sec)
    """
    cols = feature_cols or FEATURE_COLUMNS
    X_eval = eval_df[cols].values

    start_time = time.time()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_eval)[:, 1]
    else:
        decision = model.decision_function(X_eval)
        probs = 1.0 / (1.0 + np.exp(-decision))

    inf_time = round(time.time() - start_time, 2)
    return probs, inf_time
