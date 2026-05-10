"""Обучение финальной модели и сохранение всего необходимого для инференса в model.pkl.

Запуск:
    python train.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import (
    CAT_FEATURES_FE,
    NUM_FEATURES_FE,
    build_design_matrix,
    clean_text_numeric_columns,
    clean_torque,
    preprocess_full,
)


CARS_TRAIN = "https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv"
CARS_TEST = "https://github.com/evgpat/datasets/raw/refs/heads/main/cars_test.csv"

RANDOM_STATE = 42
ARTIFACT_PATH = Path(__file__).parent / "model.pkl"
DATA_DIR = Path(__file__).parent / "data"


def load_raw():
    df_train = pd.read_csv(CARS_TRAIN)
    df_test = pd.read_csv(CARS_TEST)
    df_train = df_train.drop_duplicates(
        subset=df_train.columns.drop("selling_price")
    ).reset_index(drop=True)
    return df_train, df_test


def main():
    print("Loading data...")
    df_train_raw, df_test_raw = load_raw()

    df_train_clean = clean_torque(clean_text_numeric_columns(df_train_raw))
    medians = df_train_clean.median(numeric_only=True)

    df_train_clean = df_train_clean.fillna(medians)
    df_train_clean["car_age"] = 2025 - df_train_clean["year"]
    df_train_clean["usage_per_year"] = df_train_clean["km_driven"] / df_train_clean[
        "car_age"
    ].replace(0, 1)
    q20 = float(df_train_clean["usage_per_year"].quantile(0.2))
    q80 = float(df_train_clean["usage_per_year"].quantile(0.8))
    print(f"usage_per_year quantiles: q20={q20:.1f}, q80={q80:.1f}")

    print("Building feature dataframes...")
    df_train_fe = preprocess_full(df_train_raw, medians, q20, q80)
    df_test_fe = preprocess_full(df_test_raw, medians, q20, q80)

    y_train = df_train_raw["selling_price"].reset_index(drop=True)
    y_test = df_test_raw["selling_price"].reset_index(drop=True)
    y_train_log = np.log1p(y_train)

    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    ohe.fit(df_train_fe[CAT_FEATURES_FE])

    X_train = build_design_matrix(df_train_fe, ohe)
    X_test = build_design_matrix(df_test_fe, ohe)
    print(f"Train shape: {X_train.shape}, test: {X_test.shape}")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(random_state=RANDOM_STATE)),
        ]
    )
    grid = GridSearchCV(
        pipe,
        param_grid={"ridge__alpha": np.logspace(-6, 6, 40)},
        cv=10,
        scoring="r2",
        n_jobs=-1,
    )
    print("Fitting Ridge (GridSearch, 40x10)...")
    grid.fit(X_train, y_train_log)

    best_alpha = grid.best_params_["ridge__alpha"]
    print(f"Best alpha: {best_alpha:.4g}")
    print(f"Best CV R2 (log scale): {grid.best_score_:.4f}")

    y_train_pred = np.expm1(grid.predict(X_train))
    y_test_pred = np.expm1(grid.predict(X_test))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test  R2: {test_r2:.4f}")
    print(f"Test  MSE: {test_mse:.4e}")

    artifact = {
        "pipeline": grid.best_estimator_,
        "ohe": ohe,
        "medians": medians,
        "q20": q20,
        "q80": q80,
        "feature_columns": list(X_train.columns),
        "num_features": NUM_FEATURES_FE,
        "cat_features": CAT_FEATURES_FE,
        "target_log": True,
        "metrics": {
            "best_alpha": best_alpha,
            "cv_r2_log": grid.best_score_,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mse": test_mse,
        },
        "categories": {
            col: sorted(map(str, df_train_fe[col].unique())) for col in CAT_FEATURES_FE
        },
        "num_feature_stats": df_train_fe[NUM_FEATURES_FE].describe().to_dict(),
    }
    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved -> {ARTIFACT_PATH}")

    DATA_DIR.mkdir(exist_ok=True)
    df_train_raw.to_csv(DATA_DIR / "cars_train.csv", index=False)
    df_test_raw.to_csv(DATA_DIR / "cars_test.csv", index=False)
    print(f"Saved raw data copies -> {DATA_DIR}")


if __name__ == "__main__":
    main()
