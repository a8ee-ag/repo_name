"""Задача 3: процент выживаемости в каждом классе пассажиров."""

import pandas as pd


def survival_rate(df: pd.DataFrame) -> pd.Series:
    """Доля выживших в каждом классе, в процентах.

    Survived хранит 0 и 1, поэтому среднее по столбцу — это и есть
    доля единиц, то есть доля выживших. Умножаем на 100 для процентов.
    """
    return (df.groupby("Pclass")["Survived"].mean() * 100).round(2)


def run(df: pd.DataFrame) -> None:
    print("--- Процент выживших по классам ---")
    print(survival_rate(df))
