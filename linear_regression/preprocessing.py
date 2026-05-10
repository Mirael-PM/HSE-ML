"""Препроцессинг признаков, идентичный финальному пайплайну из ноутбука.

Все функции применимы и к одной строке, и к датафрейму, поэтому
используются и для обучения (train.py), и для инференса (app.py).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


NUM_FEATURES_RAW = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
    "max_torque_rpm",
]

CAT_FEATURES_RAW = ["name", "fuel", "seller_type", "transmission", "owner"]

NUM_FEATURES_FE = [
    "car_age",
    "car_age_sq",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "max_torque_rpm",
    "power_per_cc",
]

CAT_FEATURES_FE = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "engine_class",
    "seats_class",
    "wear_class",
]

CURRENT_YEAR = 2025


def _extract_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.extract(r"([\d.]+)")[0], errors="coerce"
    )


def clean_text_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """mileage / engine / max_power приходят строками с единицами — вынимаем число."""
    df = df.copy()
    for col in ["mileage", "engine", "max_power"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = _extract_number(df[col])
    return df


def clean_torque(df: pd.DataFrame) -> pd.DataFrame:
    """Из исходного 'torque' получаем число (Nm) и max_torque_rpm."""
    df = df.copy()
    if "torque" not in df.columns:
        return df
    if df["torque"].dtype != object and "max_torque_rpm" in df.columns:
        return df
    is_kgm = df["torque"].astype(str).str.contains("kgm", case=False, na=False)
    torque = pd.to_numeric(
        df["torque"].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce"
    )
    torque[is_kgm] = torque[is_kgm] * 9.8066
    cleaned = df["torque"].astype(str).str.replace(",", "", regex=False)
    rpm = pd.to_numeric(
        cleaned.str.extract(r"(\d+)\s*(?:rpm)", flags=re.IGNORECASE)[0],
        errors="coerce",
    )
    df["torque"] = torque
    df["max_torque_rpm"] = rpm
    return df


def fill_missing(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    df = df.copy()
    return df.fillna(medians)


def categorical_engine(cc: float) -> str:
    cc = float(cc)
    if cc < 1500:
        return "subcompact"
    if cc <= 2000:
        return "compact"
    if cc <= 3000:
        return "midsize"
    if cc <= 4000:
        return "large"
    if cc <= 6000:
        return "performance_or_truck"
    return "heavy_duty_or_exotic"


def categorical_seats(seats: float) -> str:
    seats = float(seats)
    if seats < 2:
        return "small"
    if seats < 6:
        return "average"
    if seats < 9:
        return "mini_bus"
    if seats < 12:
        return "bus"
    return "large"


def add_basic_fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["car_age"] = CURRENT_YEAR - df["year"]
    df["car_age_sq"] = df["car_age"] ** 2
    df["power_per_cc"] = df["max_power"] / df["engine"]
    df["usage_per_year"] = df["km_driven"] / df["car_age"].replace(0, 1)
    return df


def apply_wear_class(df: pd.DataFrame, q20: float, q80: float) -> pd.DataFrame:
    df = df.copy()
    df["wear_class"] = pd.cut(
        df["usage_per_year"],
        bins=[-np.inf, q20, q80, np.inf],
        labels=[0, 1, 2],
    ).astype(int)
    return df


def preprocess_full(
    df: pd.DataFrame,
    medians: pd.Series,
    q20: float,
    q80: float,
) -> pd.DataFrame:
    """Полный пайплайн: сырой df -> датафрейм фич, готовых к OHE."""
    df = clean_text_numeric_columns(df)
    df = clean_torque(df)
    df = fill_missing(df, medians)
    df["seats"] = df["seats"].astype(int)
    df["engine"] = df["engine"].astype(int)
    df["engine_class"] = df["engine"].apply(categorical_engine)
    df["seats_class"] = df["seats"].apply(categorical_seats)
    df = add_basic_fe(df)
    df = apply_wear_class(df, q20, q80)
    df["name"] = df["name"].astype(str).str.split().str[0]
    return df


def build_design_matrix(
    df_features: pd.DataFrame, ohe, num_features=NUM_FEATURES_FE, cat_features=CAT_FEATURES_FE
) -> pd.DataFrame:
    """Применяет обученный OneHotEncoder и собирает финальную матрицу X."""
    ohe_cols = ohe.get_feature_names_out(cat_features)
    ohe_part = pd.DataFrame(
        ohe.transform(df_features[cat_features]),
        columns=ohe_cols,
        index=df_features.index,
    )
    return pd.concat(
        [
            df_features[num_features].reset_index(drop=True),
            ohe_part.reset_index(drop=True),
        ],
        axis=1,
    )
