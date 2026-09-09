"""Точка входа.

Задача 1: считать датасет из файла train.csv.
"""

from titanic import loader

if __name__ == "__main__":
    df = loader.load()
    print(df)
