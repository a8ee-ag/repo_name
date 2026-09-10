"""Интерфейс для прогнозирования стоимости квартиры.

Запуск:
    streamlit run app.py
"""

import streamlit as st

from predict import UNKNOWN, load_artifacts, predict_price

st.set_page_config(page_title="Оценка стоимости квартиры", page_icon="🏠")

model, meta = load_artifacts()

st.title("🏠 Оценка стоимости квартиры")
st.caption(
    f"Модель обучена на {meta['rows_trained']:,} объявлениях "
    "о продаже квартир в Москве и Московской области."
    .replace(",", " ")
)

# --- Ввод признаков -------------------------------------------------------

left, right = st.columns(2)

with left:
    city = st.selectbox(
        "Город", meta["cities"],
        index=meta["cities"].index("Москва") if "Москва" in meta["cities"] else 0,
        help="Населённый пункт, в котором находится квартира",
    )
    # Список районов зависит от выбранного города: показывать районы Химок
    # при выбранной Москве бессмысленно
    districts = meta["districts_by_city"].get(city, [UNKNOWN])
    district = st.selectbox("Район", districts,
                            help="Если район неизвестен, выберите «не указан»")

with right:
    total_square = st.number_input("Площадь, м²", min_value=8.0, max_value=500.0,
                                   value=56.0, step=1.0)
    rooms = st.number_input("Комнат", min_value=0, max_value=15, value=2, step=1,
                            help="0 — студия или количество комнат не указано")
    floor = st.number_input("Этаж", min_value=1, max_value=100, value=7, step=1)

# --- Результат ------------------------------------------------------------

if st.button("Рассчитать стоимость", type="primary", use_container_width=True):
    price = predict_price(total_square, rooms, floor, city, district)

    st.divider()
    first, second = st.columns(2)
    first.metric("Прогноз стоимости", f"{price:,.0f} ₽".replace(",", " "))
    second.metric("Цена за квадратный метр",
                  f"{price / total_square:,.0f} ₽/м²".replace(",", " "))

    # Интервал по средней ошибке модели: одно число без оценки точности
    # создаёт ложное впечатление, будто прогноз абсолютно точен
    error = meta["metrics"]["MAPE, %"] / 100
    low, high = price * (1 - error), price * (1 + error)
    st.info(
        f"Вероятный диапазон: **{low:,.0f} — {high:,.0f} ₽**. "
        f"Средняя ошибка модели составляет {meta['metrics']['MAPE, %']}%."
        .replace(",", " ")
    )

    st.caption(
        f"Параметры расчёта: {total_square:g} м², "
        f"{'студия' if rooms == 0 else f'{rooms:g} комн.'}, "
        f"этаж {floor}, {city}, {district}."
    )

# --- Справка о модели -----------------------------------------------------

with st.expander("Качество модели"):
    metrics, baseline = meta["metrics"], meta["baseline_metrics"]
    st.write("Оценка на отложенной выборке, которую модель не видела при обучении:")
    st.table({
        "Метрика": ["Средняя ошибка (MAE)", "Медианная ошибка",
                    "Средняя ошибка в процентах (MAPE)", "R²"],
        "Модель": [f"{metrics['MAE, руб']:,} ₽".replace(",", " "),
                   f"{metrics['медианная ошибка, руб']:,} ₽".replace(",", " "),
                   f"{metrics['MAPE, %']}%", str(metrics["R2"])],
        "Бейзлайн": [f"{baseline['MAE, руб']:,} ₽".replace(",", " "),
                     f"{baseline['медианная ошибка, руб']:,} ₽".replace(",", " "),
                     f"{baseline['MAPE, %']}%", str(baseline["R2"])],
    })
    st.caption(
        "Бейзлайн — медианная цена квадратного метра, умноженная на площадь. "
        "Модель — градиентный бустинг на площади, числе комнат, этаже, "
        "координатах, городе и районе."
    )
