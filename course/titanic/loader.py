"""Сервис загрузки данных: единственное место, где мы читаем файл с диска."""

from pathlib import Path

import pandas as pd

# Корень проекта = папка на уровень выше пакета titanic/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "Data" / "train.csv"


def load(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    """Читает train.csv и возвращает DataFrame.

    index_col="PassengerId" — номер пассажира это уникальный
    идентификатор строки, а не признак, поэтому делаем его индексом.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл с данными не найден: {path}")
    return pd.read_csv(path, index_col="PassengerId")
