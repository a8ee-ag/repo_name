"""API для модели прогнозирования стоимости квартиры.

Запуск:
    uvicorn main:app --reload

Документация Swagger: http://127.0.0.1:8000/docs
"""

from typing import Annotated

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from predict import UNKNOWN, load_artifacts, predict_price

app = FastAPI(
    title="Оценка стоимости квартиры",
    description=(
        "Прогноз стоимости квартиры в Москве и Московской области "
        "по площади, числу комнат, этажу, городу и району."
    ),
    version="1.0.0",
)


class Apartment(BaseModel):
    """Признаки квартиры для post-запроса.

    Field(...) задаёт границы и примеры: они попадают в схему OpenAPI,
    поэтому Swagger сразу показывает заполненный образец запроса,
    а некорректные значения отсекаются до вызова модели.
    """

    total_square: float = Field(..., gt=8, le=500, description="Площадь, м²",
                                examples=[56.0])
    rooms: float = Field(2, ge=0, le=15, description="Число комнат, 0 — студия",
                         examples=[2])
    floor: int = Field(5, ge=1, le=100, description="Этаж", examples=[7])
    city: str = Field("Москва", description="Город", examples=["Москва"])
    district: str = Field(UNKNOWN, description="Район",
                          examples=["Тверской район"])


class Prediction(BaseModel):
    """Ответ с прогнозом."""

    price: float = Field(..., description="Прогноз стоимости, рублей")
    price_per_square: float = Field(..., description="Цена за квадратный метр")
    currency: str = "RUB"
    features: dict = Field(..., description="Признаки, по которым сделан прогноз")


def make_prediction(total_square: float, rooms: float, floor: int,
                    city: str, district: str) -> Prediction:
    """Общая логика для обоих эндпоинтов: считаем один раз, зовём из двух мест."""
    price = predict_price(total_square, rooms, floor, city, district)
    return Prediction(
        price=round(price, 2),
        price_per_square=round(price / total_square, 2),
        features={
            "total_square": total_square, "rooms": rooms, "floor": floor,
            "city": city, "district": district,
        },
    )


@app.get("/health", tags=["Служебные"])
def health() -> dict:
    """Liveness-проба.

    Отвечает 200, только если модель действительно загружена: сервис,
    поднявшийся без модели, жив как процесс, но бесполезен как API.
    """
    model, meta = load_artifacts()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "rows_trained": meta["rows_trained"],
        "mape_percent": meta["metrics"]["MAPE, %"],
    }


@app.get("/predict_get", response_model=Prediction, tags=["Прогноз"])
def predict_get(
    total_square: Annotated[float, Query(gt=8, le=500, description="Площадь, м²")] = 56.0,
    rooms: Annotated[float, Query(ge=0, le=15, description="Комнат, 0 — студия")] = 2,
    floor: Annotated[int, Query(ge=1, le=100, description="Этаж")] = 7,
    city: Annotated[str, Query(description="Город")] = "Москва",
    district: Annotated[str, Query(description="Район")] = UNKNOWN,
) -> Prediction:
    """Прогноз через get-запрос: признаки передаются параметрами строки запроса."""
    return make_prediction(total_square, rooms, floor, city, district)


@app.post("/predict_post", response_model=Prediction, tags=["Прогноз"])
def predict_post(apartment: Apartment) -> Prediction:
    """Прогноз через post-запрос: признаки передаются телом запроса в JSON."""
    return make_prediction(apartment.total_square, apartment.rooms,
                           apartment.floor, apartment.city, apartment.district)


@app.get("/", include_in_schema=False)
def root() -> dict:
    """Корень отдаёт подсказку, куда идти: пустая страница сбивает с толку."""
    return {"docs": "/docs", "health": "/health",
            "endpoints": ["/predict_get", "/predict_post"]}
