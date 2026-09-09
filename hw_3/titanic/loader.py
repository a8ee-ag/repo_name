"""Сервис загрузки данных: единственное место, где мы читаем файл с диска."""

from pathlib import Path

import pandas as pd

# Папка самой работы (hw_3) — на уровень выше пакета titanic/
HW_DIR = Path(__file__).resolve().parent.parent


def find_dataset(filename: str = "train.csv") -> Path:
    """Ищет DataSet/<filename>, поднимаясь от папки работы вверх по дереву.

    Датасет лежит не внутри hw_3, а рядом с ней — он общий для всех
    работ. Поиск вверх избавляет от жёсткого пути вроде "../DataSet":
    код продолжит работать, даже если папку с работой переложат глубже.
    """
    for folder in [HW_DIR, *HW_DIR.parents]:
        candidate = folder / "DataSet" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Не найдена папка DataSet с файлом {filename} — искали от {HW_DIR} вверх"
    )


def load(path: Path | str | None = None) -> pd.DataFrame:
    """Читает train.csv и возвращает DataFrame.

    index_col="PassengerId" — номер пассажира это уникальный
    идентификатор строки, а не признак, поэтому делаем его индексом.
    """
    path = Path(path) if path is not None else find_dataset()
    if not path.exists():
        raise FileNotFoundError(f"Файл с данными не найден: {path}")
    return pd.read_csv(path, index_col="PassengerId")
