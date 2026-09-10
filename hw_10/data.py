"""Загрузка и очистка датасета объявлений о продаже недвижимости."""

from pathlib import Path

import pandas as pd

ARCHIVE_NAME = "train_data.zip"

# Столбец description не читаем: это единственный тяжёлый столбец в файле,
# а для модели на табличных признаках он не нужен
COLUMNS = ["price", "lat", "lon", "object_type", "total_square",
           "rooms", "floor", "city", "district"]

TARGET = "price"
NUMERIC = ["total_square", "rooms", "floor", "lat", "lon"]
CATEGORICAL = ["city", "district"]
FEATURES = NUMERIC + CATEGORICAL

# Границы отсечения выбросов. Верхняя граница цены и площади отрезает
# единичные дворцы, на которых модель училась бы шуму: их меньше 0.5%,
# а разброс цен они растягивают в сотню раз
MIN_PRICE, MAX_PRICE = 1_000_000, 300_000_000
MAX_SQUARE = 500
MAX_FLOOR = 100


def find_archive(filename: str = ARCHIVE_NAME) -> Path:
    """Ищет DataSet/<filename>, поднимаясь от текущей папки вверх по дереву."""
    start = Path(__file__).resolve().parent
    for folder in [start, *start.parents]:
        candidate = folder / "DataSet" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Не найден архив {filename} в папке DataSet")


def load(path: Path | None = None) -> pd.DataFrame:
    """Читает csv прямо из zip-архива — распаковывать 185 МБ на диск незачем."""
    return pd.read_csv(path or find_archive(), usecols=COLUMNS)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает выбросы и строки, непригодные для обучения."""
    before = len(df)
    df = df.dropna(subset=[TARGET, "total_square", "lat", "lon"])
    df = df[df[TARGET].between(MIN_PRICE, MAX_PRICE)]
    df = df[df["total_square"].between(8, MAX_SQUARE)]
    df = df[df["floor"].between(1, MAX_FLOOR)]

    # Пропуски в категориях — это «не указано», а не отсутствие данных:
    # объявление без района всё равно пригодно для обучения
    df = df.copy()
    for column in CATEGORICAL:
        df[column] = df[column].fillna("не указан").astype("category")
    df["rooms"] = df["rooms"].fillna(0)      # 0 = студия либо не указано

    print(f"после очистки осталось {len(df)} из {before} строк "
          f"({len(df) / before * 100:.1f}%)")
    return df
