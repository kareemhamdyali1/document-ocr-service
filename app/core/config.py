# ============================================================
# app/core/config.py
# ============================================================

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


class Settings(BaseSettings):

    # ========================================================
    # APPLICATION
    # ========================================================

    app_name: str = "Credit Document AI API"

    app_version: str = "1.0.0"

    environment: str = "development"

    # ========================================================
    # GEMINI
    # ========================================================

    gemini_api_key: str = Field(
        default="",
        alias="GEMINI_API_KEY",
    )

    # Primary Gemini model
    #
    # You can override this from .env:
    #
    # GEMINI_MODEL=gemini-3.5-flash-lite
    #
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite",
        alias="GEMINI_MODEL",
    )

    # Optional fallback models.
    #
    # Example:
    #
    # GEMINI_FALLBACK_MODELS=gemini-3.5-flash,gemini-3.6-flash
    #
    gemini_fallback_models: str = Field(
        default="",
        alias="GEMINI_FALLBACK_MODELS",
    )

    # Generation temperature
    #
    # 0.0 is recommended for document extraction
    # because we want deterministic structured output.
    #
    gemini_temperature: float = Field(
        default=0.0,
        alias="GEMINI_TEMPERATURE",
    )

    # ========================================================
    # OPENROUTER
    # ========================================================
    #
    # Kept here in case you need to switch back later.
    #

    openrouter_api_key: str = Field(
        default="",
        alias="OPENROUTER_API_KEY",
    )

    openrouter_model: str = Field(
        default="openrouter/free",
        alias="OPENROUTER_MODEL",
    )

    openrouter_fallback_models: str = Field(
        default="",
        alias="OPENROUTER_FALLBACK_MODELS",
    )

    openrouter_temperature: float = Field(
        default=0.0,
        alias="OPENROUTER_TEMPERATURE",
    )

    openrouter_max_tokens: int = Field(
        default=8192,
        alias="OPENROUTER_MAX_TOKENS",
    )

    # ========================================================
    # REQUEST SETTINGS
    # ========================================================

    max_file_size_mb: int = Field(
        default=15,
        alias="MAX_FILE_SIZE_MB",
    )

    # ========================================================
    # HTTP / OPENROUTER METADATA
    # ========================================================

    openrouter_http_referer: str = Field(
        default="",
        alias="OPENROUTER_HTTP_REFERER",
    )

    openrouter_app_name: str = Field(
        default="Credit Document AI API",
        alias="OPENROUTER_APP_NAME",
    )

    # ========================================================
    # PYDANTIC SETTINGS
    # ========================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ========================================================
    # GEMINI MODEL LIST
    # ========================================================

    @property
    def gemini_models(self) -> list[str]:
        """
        Return the configured Gemini model list.

        Primary model:
            GEMINI_MODEL

        Optional fallback models:
            GEMINI_FALLBACK_MODELS=model1,model2
        """

        models = [self.gemini_model]

        for model in self.gemini_fallback_models.split(","):
            model = model.strip()

            if model and model not in models:
                models.append(model)

        return models

    # ========================================================
    # OPENROUTER MODEL LIST
    # ========================================================

    @property
    def openrouter_models(self) -> list[str]:
        """
        Return the configured OpenRouter model list.

        Primary:
            OPENROUTER_MODEL

        Optional:
            OPENROUTER_FALLBACK_MODELS=model1,model2
        """

        models = [self.openrouter_model]

        for model in self.openrouter_fallback_models.split(","):
            model = model.strip()

            if model and model not in models:
                models.append(model)

        return models


# ============================================================
# SETTINGS SINGLETON
# ============================================================

@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()