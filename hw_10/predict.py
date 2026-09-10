"""Получение предсказания обученной моделью.

Модуль отделён от интерфейса намеренно: предсказание можно получить и из
Streamlit, и из консоли, и из другого кода — логика одна и та же.

Пример запуска из консоли:
    python predict.py --square 56 --rooms 2 --floor 7 --city Москва
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
UNKNOWN = "не указан"


@lru_cache(maxsize=1)
def load_artifacts() -> tuple:
    """Читает модель и справочники с диска.

    lru_cache держит их в памяти: Streamlit перезапускает скрипт на каждое
    действие пользователя, и без кеша модель читалась бы с диска каждый раз.
    """
    model_path = ARTIFACTS / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "Модель не найдена. Сначала обучите её: python train_model.py"
        )
    model = joblib.load(model_path)
    meta = json.loads((ARTIFACTS / "meta.json").read_text(encoding="utf-8"))
    return model, meta


def coordinates_for(city: str, district: str) -> tuple[float, float]:
    """Координаты по городу и району: медиана объявлений из обучающих данных.

    Модель обучена в том числе на широте и долготе, а вводить их вручную
    неудобно, поэтому подставляем типичную точку выбранного района.
    """
    _, meta = load_artifacts()
    point = meta["coordinates"].get(f"{city}||{district}")
    if point is None:
        # район не выбран или отсутствует в справочнике — берём центр города
        city_points = [
            value for key, value in meta["coordinates"].items()
            if key.startswith(f"{city}||")
        ]
        if not city_points:
            city_points = list(meta["coordinates"].values())
        point = np.median(np.array(city_points), axis=0).tolist()
    return float(point[0]), float(point[1])


def predict_price(total_square: float, rooms: float, floor: int,
                  city: str, district: str = UNKNOWN,
                  lat: float | None = None, lon: float | None = None) -> float:
    """Возвращает предсказанную стоимость квартиры в рублях."""
    model, meta = load_artifacts()

    if lat is None or lon is None:
        lat, lon = coordinates_for(city, district)

    features = pd.DataFrame([{
        "total_square": float(total_square),
        "rooms": float(rooms),
        "floor": int(floor),
        "lat": lat,
        "lon": lon,
        "city": city,
        "district": district,
    }])[meta["features"]]

    # Категории должны быть именно category: модель обучалась с
    # categorical_features="from_dtype" и по типу столбца понимает, что это категория
    for column in ("city", "district"):
        features[column] = features[column].astype("category")

    # Модель обучена на логарифме цены — возвращаем значение в рубли
    return float(np.exp(model.predict(features)[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Предсказание стоимости квартиры")
    parser.add_argument("--square", type=float, required=True, help="площадь, м²")
    parser.add_argument("--rooms", type=float, default=2, help="число комнат")
    parser.add_argument("--floor", type=int, default=5, help="этаж")
    parser.add_argument("--city", default="Москва", help="город")
    parser.add_argument("--district", default=UNKNOWN, help="район")
    args = parser.parse_args()

    price = predict_price(args.square, args.rooms, args.floor,
                          args.city, args.district)
    print(f"{args.square:g} м², {args.rooms:g} комн., этаж {args.floor}, "
          f"{args.city}, {args.district}")
    print(f"прогноз: {price:,.0f} руб.".replace(",", " "))


if __name__ == "__main__":
    main()
