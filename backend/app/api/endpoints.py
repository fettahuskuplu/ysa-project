from fastapi import APIRouter, Query

from app.core.config import settings
from app.schemas.dashboard import DashboardResponseSchema
from app.services.dashboard_service import get_dashboard_data, MODEL_SLUG_MAP

router = APIRouter()


@router.get("/dashboard", response_model=DashboardResponseSchema)
def get_dashboard_metrics(
    model_name: str = Query("Dueling DQN", description="Seçilen Model"),
    symbol: str = Query("THYAO", description="Hisse Senedi Kodu"),
):
    return get_dashboard_data(
        results_dir=settings.dashboard_results_path,
        model_name=model_name,
        symbol=symbol,
    )


@router.get("/models")
def list_models():
    return [{"id": name, "name": name} for name in MODEL_SLUG_MAP]