from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.endpoints import router as api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0",
    description="DeepQuant Projesi için Yapay Zeka Model Çıktılarını Besleyen Merkezi Backend Sunucusu"
)

# CORS İzinlerini Uygulamaya Enjekte Ediyoruz (Frontend rahatça bağlansın diye)
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Yazdığımız router yapısını uygulamaya kaydediyoruz
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {"status": "online", "message": "DeepQuant DRL API çalışıyor!"}