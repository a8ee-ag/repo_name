"""Обучение модели прогнозирования стоимости квартиры.

Запуск:
    python train_model.py

Результат — файл artifacts/model.joblib, который использует интерфейс app.py.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import data

RANDOM_STATE = 42
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def build_model() -> HistGradientBoostingRegressor:
    """Градиентный бустинг с встроенной поддержкой категорий.

    Обычному one-hot тут пришлось бы тяжело: районов больше сотни, и матрица
    признаков раздулась бы. HistGradientBoosting умеет работать с категориями
    напрямую, если столбец имеет тип category.
    """
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.08,
        max_depth=None,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        random_state=RANDOM_STATE,
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Метрики считаем в рублях, а не в логарифмах — так они интерпретируемы."""
    errors = np.abs(y_true - y_pred)
    return {
        "MAE, руб": round(float(mean_absolute_error(y_true, y_pred))),
        "медианная ошибка, руб": round(float(median_absolute_error(y_true, y_pred))),
        "MAPE, %": round(float(np.mean(errors / y_true) * 100), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main() -> None:
    df = data.clean(data.load())

    X = df[data.FEATURES]
    # Цены скошены вправо на два порядка, поэтому учим модель на логарифме:
    # иначе дорогие объекты определяют почти всю ошибку, а типовые квартиры
    # предсказываются плохо. Предсказание возвращаем обратно через exp.
    y = np.log(df[data.TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"обучающая выборка: {len(X_train)}, тестовая: {len(X_test)}")

    model = build_model()
    model.fit(X_train, y_train)

    predictions = np.exp(model.predict(X_test))
    actual = np.exp(y_test.to_numpy())

    metrics = evaluate(actual, predictions)
    print("\nКачество на отложенной выборке:")
    for name, value in metrics.items():
        print(f"  {name:<24} {value:,}".replace(",", " "))

    # Бейзлайн: медианная цена квадратного метра, умноженная на площадь.
    # Без него непонятно, дала ли модель что-то сверх простого правила.
    price_per_square = (np.exp(y_train) / X_train["total_square"]).median()
    baseline = X_test["total_square"] * price_per_square
    baseline_metrics = evaluate(actual, baseline.to_numpy())
    print("\nБейзлайн (медианная цена за м² × площадь):")
    for name, value in baseline_metrics.items():
        print(f"  {name:<24} {value:,}".replace(",", " "))

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "model.joblib", compress=3)

    # Справочники для интерфейса: списки городов и районов, а также координаты
    # по умолчанию, чтобы пользователю не приходилось вводить широту и долготу
    coordinates = (
        df.groupby(["city", "district"], observed=True)[["lat", "lon"]]
        .median().round(6).reset_index()
    )
    meta = {
        "features": data.FEATURES,
        "cities": sorted(df["city"].cat.categories.tolist()),
        "districts_by_city": {
            city: sorted(group["district"].tolist())
            for city, group in coordinates.groupby("city", observed=True)
        },
        "coordinates": {
            f"{row.city}||{row.district}": [row.lat, row.lon]
            for row in coordinates.itertuples()
        },
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "rows_trained": len(X_train),
        "price_range": [int(df[data.TARGET].min()), int(df[data.TARGET].max())],
    }
    (ARTIFACTS / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    size = (ARTIFACTS / "model.joblib").stat().st_size / 1024**2
    print(f"\nмодель сохранена: artifacts/model.joblib ({size:.1f} МБ)")
    print("справочники:     artifacts/meta.json")


if __name__ == "__main__":
    main()
