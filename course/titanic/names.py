"""Задача 4: самое популярное мужское и женское имя на корабле.

Полные имена в датасете уникальны (891 разное значение на 891 строку),
поэтому «популярное имя» приходится извлекать из строки вида
    Фамилия, Титул. Имя Отчество
"""

import re

import pandas as pd

# Часть в скобках: у замужних женщин там их собственное имя
PARENTHESES = re.compile(r"\(([^)]*)\)")
# Прозвище в кавычках: Anna "Annie"
QUOTED = re.compile(r'"[^"]*"')


def extract_first_name(full_name: str) -> str:
    """Достаёт личное имя из полного имени пассажира.

    Логика по шагам:
      1. отрезаем фамилию (всё до первой запятой) и титул (всё до точки);
      2. если дальше есть скобки — берём их содержимое: замужние женщины
         записаны под именем мужа, а своё имя стоит в скобках
         («Cumings, Mrs. John Bradley (Florence Briggs Thayer)» -> Florence);
      3. скобки с прозвищем в кавычках («Mr. Pastcho ("Pentcho")») пропускаем;
      4. из оставшейся строки берём первое слово.
    """
    # 1. после первой точки остаётся всё, что идёт за титулом
    rest = full_name.split(".", 1)[1] if "." in full_name else full_name

    # 2-3. содержимое скобок, если это не прозвище в кавычках
    match = PARENTHESES.search(rest)
    if match:
        inside = match.group(1).strip()
        if not inside.startswith('"'):
            rest = inside

    # убираем прозвища в кавычках и скобки, если они остались
    rest = QUOTED.sub("", rest)
    rest = PARENTHESES.sub("", rest)

    # 4. первое слово и есть имя
    words = rest.split()
    return words[0] if words else ""


def first_names(df: pd.DataFrame) -> pd.Series:
    """Столбец с личными именами всех пассажиров."""
    return df["Name"].apply(extract_first_name)


def most_popular(df: pd.DataFrame, sex: str) -> tuple[str, int]:
    """Самое частое имя среди пассажиров указанного пола и сколько раз встречается."""
    counts = first_names(df[df["Sex"] == sex]).value_counts()
    return counts.index[0], int(counts.iloc[0])


def run(df: pd.DataFrame) -> None:
    print("--- Самые популярные имена на корабле ---")
    for sex, label in (("male", "мужское"), ("female", "женское")):
        name, count = most_popular(df, sex)
        print(f"{label}: {name} ({count} чел.)")
