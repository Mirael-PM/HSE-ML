# Cars Price Predictor — Streamlit-приложение

Сервис предсказывает цену подержанного автомобиля по техническим характеристикам.
Модель — `StandardScaler + Ridge` (с GridSearch по α) на log-таргете, обученная на
датасете `cars_train.csv`.

## Структура

```
streamlit_cars_app/
├── app.py              # Streamlit UI: EDA / инференс / веса модели
├── train.py            # обучение и сохранение model.pkl
├── preprocessing.py    # общий препроцессинг для train и inference
├── model.pkl           # обученный pipeline + OHE + статистики
├── requirements.txt
├── data/
│   ├── cars_train.csv
│   └── cars_test.csv
└── results.md          # отчёт по заданию 26
```

## Запуск локально

```bash
pip install -r requirements.txt
python train.py            # (опционально) переобучить и обновить model.pkl
streamlit run app.py
```

После старта откройте `http://localhost:8501`.

## Деплой на Streamlit Community Cloud

1. Залить эту папку в публичный репозиторий GitHub.
2. На <https://share.streamlit.io> создать приложение, указав:
   - репозиторий и ветку,
   - `Main file path: streamlit_cars_app/app.py`.
3. Дождаться сборки. Ссылка вида
   `https://<user>-<repo>-streamlit-cars-app.streamlit.app`
   подставляется в отчёт.

## Возможности приложения

- **Обзор и EDA**: ключевые гистограммы и боксплоты по `selling_price`, scatter
  по `year`, корреляции — то, что отвечает за «информативные графики EDA».
- **Предсказание цены**: два режима — ручной ввод полей и загрузка CSV. Результат
  для CSV можно скачать отдельным файлом.
- **Веса модели**: горизонтальный бар-чарт коэффициентов Ridge (на нормированных
  признаках), таблица всех весов и топ-5 «удорожающих» / «удешевляющих» фич.
