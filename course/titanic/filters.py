"""Задача 6: пассажиры, возраст которых больше 44 лет."""

import pandas as pd


def older_than(df: pd.DataFrame, age: float = 44) -> pd.DataFrame:
    """Строки таблицы, где Age строго больше указанного возраста.

    df["Age"] > age даёт булеву маску (True/False на каждую строку),
    а df[маска] оставляет только строки с True.
    Пассажиры с пропущенным возрастом сюда не попадут: сравнение
    с NaN всегда даёт False.
    """
    return df[df["Age"] > age]


def run(df: pd.DataFrame) -> None:
    print("--- Пассажиры старше 44 лет ---")
    print(older_than(df, 44))
