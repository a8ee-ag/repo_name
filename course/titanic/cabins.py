"""Задача 8: количество n-местных кабин.

В столбце Cabin у 24 пассажиров записано несколько кают сразу
(«C23 C25 C27») — это семьи, выкупившие смежные каюты одной бронью.
Такую запись считаем единым размещением: если разбивать её на
отдельные каюты, каждый член семьи засчитается в каждую из них,
и сумма проживающих превысит число пассажиров с известной каютой.
"""

import pandas as pd


def people_per_cabin(df: pd.DataFrame) -> pd.Series:
    """Сколько человек приходится на каждую запись о каюте.

    dropna() убирает 687 пассажиров без указанной каюты,
    value_counts() считает, сколько раз встречается каждая запись.
    """
    return df["Cabin"].dropna().value_counts()


def cabin_size_distribution(df: pd.DataFrame, min_people: int = 2) -> pd.Series:
    """Сколько существует кают на 2, 3, 4, ... человека.

    Второй value_counts() применяется уже к числам из первого:
    он считает, сколько кают имеют одинаковое число жильцов.
    """
    sizes = people_per_cabin(df).value_counts().sort_index()
    sizes.index.name = "человек в каюте"
    sizes.name = "количество кают"
    return sizes[sizes.index >= min_people]


def run(df: pd.DataFrame) -> None:
    print("--- Количество n-местных кают ---")
    print(cabin_size_distribution(df, min_people=2))
