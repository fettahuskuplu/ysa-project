import os
import sys
from pydantic_settings import BaseSettings

# Backend içinden üst dizindeki 'src' modüllerine erişebilmek için sistem yolunu ayarlıyoruz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

class Settings(BaseSettings):
    PROJECT_NAME: str = "DeepQuant: DRL Trading Dashboard API"
    API_V1_STR: str = "/api"
    
    # Frontend ekibinin lokalde çalışırken (örn: localhost:5173 veya 3000) engellenmemesi için izin verilen adresler
    BACKEND_CORS_ORIGINS: list = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

settings = Settings()