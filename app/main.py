from fastapi import FastAPI

from app.core.config import settings
from app.api.routes import router


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)


app.include_router(router)


@app.get("/")
def root():

    return {
        "message": (
            "Credit Document AI API Running"
        )
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "gemini_configured": bool(
            settings.gemini_api_key
        ),
        "models": settings.gemini_models,
    }