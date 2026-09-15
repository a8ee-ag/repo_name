"""Задача 2: основная информация о датасете — типы данных, пропуски, средние."""

import pandas as pd


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица пропусков: сколько NaN в каждом столбце и какой это процент."""
    count = df.isna().sum()
    return pd.DataFrame({
        "пропусков": count,
        "процент": (count / len(df) * 100).round(2),
    })


def run(df: pd.DataFrame) -> None:
    print("--- Типы данных и заполненность ---")
    df.info()

    print("\n--- Пропуски ---")
    print(missing_report(df))

    print("\n--- Средние и прочая статистика по числовым столбцам ---")
    print(df.describe().round(2))
