"""Streamlit-приложение для предсказания цены б/у автомобиля.

Запуск:
    streamlit run app.py
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from preprocessing import (
    CAT_FEATURES_FE,
    NUM_FEATURES_FE,
    build_design_matrix,
    preprocess_full,
)


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.pkl"
DATA_PATH = APP_DIR / "data" / "cars_train.csv"


st.set_page_config(
    page_title="Cars Price Predictor",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_train_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def predict_df(df_raw: pd.DataFrame, artifact: dict) -> np.ndarray:
    df_fe = preprocess_full(
        df_raw, artifact["medians"], artifact["q20"], artifact["q80"]
    )
    X = build_design_matrix(df_fe, artifact["ohe"])
    X = X.reindex(columns=artifact["feature_columns"], fill_value=0.0)
    y_log = artifact["pipeline"].predict(X)
    return np.expm1(y_log)


# ---------------------------------------------------------------------------
# Sidebar / navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🚗 Cars Price")
section = st.sidebar.radio(
    "Раздел",
    [
        "Обзор и EDA",
        "Предсказание цены",
        "Веса модели",
    ],
)
st.sidebar.markdown("---")

if not MODEL_PATH.exists():
    st.error(
        "Не найден `model.pkl`. Сначала запустите обучение:\n\n"
        "```\npython train.py\n```"
    )
    st.stop()

artifact = load_model()
metrics = artifact["metrics"]

st.sidebar.markdown("**Качество модели**")
st.sidebar.metric("Test R²", f"{metrics['test_r2']:.3f}")
st.sidebar.metric("Train R²", f"{metrics['train_r2']:.3f}")
st.sidebar.caption(f"Ridge, α = {metrics['best_alpha']:.3g}")

# ===========================================================================
# 1. EDA
# ===========================================================================
if section == "Обзор и EDA":
    st.title("Анализ данных о подержанных автомобилях")
    st.write(
        "Датасет содержит характеристики автомобилей с площадок Индии. "
        "Цель — предсказать `selling_price` (рупии)."
    )

    df = load_train_df()
    st.subheader("Превью датасета")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Объектов", f"{len(df):,}")
    c2.metric("Признаков", df.shape[1] - 1)
    c3.metric("Средняя цена", f"{df['selling_price'].mean():,.0f}")
    c4.metric("Медианная цена", f"{df['selling_price'].median():,.0f}")

    st.markdown("### Распределение целевой переменной")
    log_axis = st.checkbox("Log-шкала по X", value=True, key="hist_log")
    price_series = df["selling_price"]
    fig = px.histogram(
        np.log1p(price_series) if log_axis else price_series,
        nbins=60,
        labels={
            "value": "log(1+selling_price)" if log_axis else "selling_price",
        },
        title="Гистограмма цены",
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Цена и год выпуска")
    fig = px.scatter(
        df,
        x="year",
        y="selling_price",
        color="fuel",
        opacity=0.5,
        log_y=True,
        title="selling_price vs year (log-Y)",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Последние эксперименты")
    st.caption(
        "Боксплоты `selling_price` по инженерным признакам из части 4: "
        "`engine_class` (категория объёма), `seats_class` (категория числа мест), "
        "`wear_class` (квантили `usage_per_year` по train: q20/q80)."
    )
    df_fe_eda = preprocess_full(
        df, artifact["medians"], artifact["q20"], artifact["q80"]
    )

    engine_order = [
        "subcompact",
        "compact",
        "midsize",
        "large",
        "performance_or_truck",
        "heavy_duty_or_exotic",
    ]
    seats_order = ["small", "average", "mini_bus", "bus", "large"]

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.box(
            df_fe_eda,
            x="engine_class",
            y="selling_price",
            log_y=True,
            category_orders={"engine_class": engine_order},
            title="Цена по engine_class",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(
            df_fe_eda,
            x="seats_class",
            y="selling_price",
            log_y=True,
            category_orders={"seats_class": seats_order},
            title="Цена по seats_class",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.box(
            df_fe_eda,
            x="wear_class",
            y="selling_price",
            log_y=True,
            title="Цена по wear_class (0=низкий износ, 2=высокий)",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Полная карта корреляций")
    st.caption(
        "Корреляция всех числовых признаков (после очистки и FE) с целевой переменной "
        "и между собой. Категориальные признаки (`wear_class`, `engine_class`, "
        "`seats_class`) включены через числовой код / one-hot."
    )

    method = st.radio(
        "Метод",
        ["pearson", "spearman", "kendall"],
        index=0,
        horizontal=True,
    )

    corr_df = df_fe_eda.copy()
    corr_df["wear_class"] = corr_df["wear_class"].astype(int)
    cat_to_dummy = ["engine_class", "seats_class", "fuel", "transmission", "seller_type", "owner"]
    corr_df = pd.get_dummies(
        corr_df.drop(columns=["name"], errors="ignore"),
        columns=[c for c in cat_to_dummy if c in corr_df.columns],
        drop_first=True,
    )
    bool_cols = corr_df.select_dtypes(include="bool").columns
    corr_df[bool_cols] = corr_df[bool_cols].astype(int)
    corr_df = corr_df.select_dtypes(include="number")
    corr_matrix = corr_df.corr(method=method).round(2)

    sort_by_target = st.checkbox(
        "Сортировать по |корр.| с selling_price", value=True
    )
    if sort_by_target and "selling_price" in corr_matrix.columns:
        order = (
            corr_matrix["selling_price"].abs().sort_values(ascending=False).index
        )
        corr_matrix = corr_matrix.loc[order, order]

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        title=f"{method.capitalize()} correlation — {corr_matrix.shape[0]} признаков",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=max(600, 18 * corr_matrix.shape[0]))
    st.plotly_chart(fig, use_container_width=True)

    if "selling_price" in corr_matrix.columns:
        st.markdown("**Топ-10 признаков по |корреляции| с `selling_price`:**")
        target_corr = (
            corr_matrix["selling_price"]
            .drop("selling_price")
            .to_frame("corr")
            .assign(abs_corr=lambda d: d["corr"].abs())
            .sort_values("abs_corr", ascending=False)
            .head(10)
            .drop(columns="abs_corr")
        )
        st.dataframe(target_corr, use_container_width=True)

    with st.expander("Числовые статистики (describe)"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)


# ===========================================================================
# 2. Inference
# ===========================================================================
elif section == "Предсказание цены":
    st.title("Предсказание цены автомобиля")

    st.image(
        "https://www.team-bhp.com/forum/attachments/et-cetera/2109196d1610817369-automotive-memes-thread-img20210115wa0039.jpg",
        caption="Настроение перед инференсом",
        use_container_width=True,
    )

    tab_form, tab_csv = st.tabs(["Ручной ввод", "CSV-файл"])

    cats = artifact["categories"]
    stats = artifact["num_feature_stats"]
    medians = artifact["medians"]

    with tab_form:
        st.write("Заполните характеристики автомобиля:")

        col1, col2, col3 = st.columns(3)
        with col1:
            name_options = sorted(
                {c for c in cats.get("name", []) if c and c != "nan"}
            )
            name = st.selectbox(
                "Бренд (name)",
                options=name_options or ["Maruti"],
                index=name_options.index("Maruti") if "Maruti" in name_options else 0,
            )
            year = st.number_input(
                "Год выпуска",
                min_value=1990,
                max_value=2025,
                value=2015,
                step=1,
            )
            km_driven = st.number_input(
                "Пробег, км",
                min_value=0,
                max_value=1_000_000,
                value=int(medians.get("km_driven", 60000)),
                step=1000,
            )
        with col2:
            fuel = st.selectbox(
                "Тип топлива (fuel)",
                options=sorted({c for c in cats.get("fuel", []) if c}),
            )
            seller_type = st.selectbox(
                "Тип продавца",
                options=sorted({c for c in cats.get("seller_type", []) if c}),
            )
            transmission = st.selectbox(
                "Трансмиссия",
                options=sorted({c for c in cats.get("transmission", []) if c}),
            )
            owner = st.selectbox(
                "Владелец",
                options=sorted({c for c in cats.get("owner", []) if c}),
            )
        with col3:
            mileage = st.number_input(
                "Расход (mileage, km/l)",
                min_value=0.0,
                max_value=50.0,
                value=float(medians.get("mileage", 19.0)),
                step=0.1,
            )
            engine = st.number_input(
                "Объём двигателя (cc)",
                min_value=500,
                max_value=6000,
                value=int(medians.get("engine", 1248)),
                step=10,
            )
            max_power = st.number_input(
                "Мощность (max_power, bhp)",
                min_value=10.0,
                max_value=500.0,
                value=float(medians.get("max_power", 82.0)),
                step=0.5,
            )
            torque_val = st.number_input(
                "Крутящий момент (torque, Nm)",
                min_value=10.0,
                max_value=1000.0,
                value=float(medians.get("torque", 160.0)),
                step=1.0,
            )
            max_torque_rpm = st.number_input(
                "Обороты макс. момента (rpm)",
                min_value=500,
                max_value=10000,
                value=int(medians.get("max_torque_rpm", 2400)),
                step=100,
            )
            seats = st.number_input(
                "Кол-во мест",
                min_value=2,
                max_value=14,
                value=int(medians.get("seats", 5)),
                step=1,
            )

        if st.button("Предсказать цену", type="primary", use_container_width=True):
            row = pd.DataFrame(
                [
                    {
                        "name": name,
                        "year": year,
                        "km_driven": km_driven,
                        "fuel": fuel,
                        "seller_type": seller_type,
                        "transmission": transmission,
                        "owner": owner,
                        "mileage": mileage,
                        "engine": engine,
                        "max_power": max_power,
                        "torque": torque_val,
                        "max_torque_rpm": max_torque_rpm,
                        "seats": seats,
                    }
                ]
            )
            try:
                pred = predict_df(row, artifact)[0]
                st.success(f"Прогноз цены: **{pred:,.0f}** ₹")
                st.caption(
                    "Это значение получено через `expm1` обратного преобразования "
                    "log-таргета."
                )
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")

    with tab_csv:
        st.write(
            "Загрузите CSV с теми же колонками, что и в train (без `selling_price`)."
        )
        uploaded = st.file_uploader("Файл CSV", type=["csv"])
        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                st.write(f"Загружено строк: {len(df_in)}")
                st.dataframe(df_in.head(), use_container_width=True)
                preds = predict_df(df_in, artifact)
                result = df_in.copy()
                result["predicted_selling_price"] = np.round(preds).astype(int)
                st.markdown("**Результаты**")
                st.dataframe(result, use_container_width=True)
                buf = io.StringIO()
                result.to_csv(buf, index=False)
                st.download_button(
                    "Скачать predictions.csv",
                    buf.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
                st.markdown("### Распределение прогнозов")
                fig = px.histogram(preds, nbins=40, log_x=True)
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Не удалось обработать файл: {e}")

# ===========================================================================
# 3. Coefficients
# ===========================================================================
else:
    st.title("Веса обученной модели")
    st.write(
        "Модель — `StandardScaler + Ridge` на log-таргете. Коэффициенты "
        "интерпретируются как влияние стандартизованного признака на `log(price)`."
    )

    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    coefs = pd.Series(pipeline.named_steps["ridge"].coef_, index=feature_columns)
    df_coefs = (
        pd.DataFrame({"feature": coefs.index, "coef": coefs.values})
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
    )

    top_n = st.slider("Сколько топ-признаков показать", 5, len(df_coefs), 25)
    df_top = df_coefs.head(top_n).sort_values("coef")

    fig = px.bar(
        df_top,
        x="coef",
        y="feature",
        orientation="h",
        color="coef",
        color_continuous_scale="RdBu",
        title=f"Top-{top_n} признаков по |коэффициенту|",
    )
    fig.update_layout(height=max(400, 22 * top_n), yaxis={"categoryorder": "array"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Все коэффициенты")
    st.dataframe(
        df_coefs.reset_index(drop=True).style.background_gradient(
            cmap="RdBu_r", subset=["coef"]
        ),
        use_container_width=True,
        height=500,
    )

    pos = df_coefs[df_coefs["coef"] > 0].head(5)
    neg = df_coefs[df_coefs["coef"] < 0].head(5)
    col1, col2 = st.columns(2)
    col1.markdown("**Топ-5 признаков, повышающих цену**")
    col1.dataframe(pos[["feature", "coef"]].reset_index(drop=True))
    col2.markdown("**Топ-5 признаков, снижающих цену**")
    col2.dataframe(neg[["feature", "coef"]].reset_index(drop=True))
