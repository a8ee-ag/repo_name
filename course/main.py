"""Точка входа: загружает данные и запускает выбранную задачу.

Использование:
    python main.py            — список доступных задач
    python main.py overview   — задача 2: информация о датасете
    python main.py survival   — задача 3: выживаемость по классам
    python main.py names      — задача 4: популярные имена
"""

import sys

from titanic import loader, names, overview, survival

# Реестр задач: имя команды -> функция, принимающая DataFrame
TASKS = {
    "overview": overview.run,
    "survival": survival.run,
    "names": names.run,
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in TASKS:
        print("Доступные задачи:")
        for name in TASKS:
            print(f"  python main.py {name}")
        return 1

    df = loader.load()
    TASKS[argv[0]](df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
