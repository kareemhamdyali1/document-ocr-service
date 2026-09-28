import os
import re
import time
from typing import Any, Dict, List, Tuple, Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.core.config import settings


T = TypeVar("T", bound=BaseModel)


class GeminiServiceError(Exception):
    """Custom exception raised when all configured Gemini models fail or initialization fails."""
    pass


class GeminiService:

    def __init__(self):
        # ============================================================
        # GEMINI API KEY
        # ============================================================

        self.api_key = (
            getattr(settings, "gemini_api_key", "")
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GEMINI", "")
        )

        if not self.api_key:
            raise GeminiServiceError(
                "GEMINI API key is not configured."
            )

        # ============================================================
        # CLIENT
        # ============================================================

        self.client = genai.Client(
            api_key=self.api_key
        )

        # ============================================================
        # MODELS
        # ============================================================

        primary_model = (
            getattr(settings, "gemini_model", "")
            or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        )

        fallback_models_raw = (
            getattr(settings, "gemini_fallback_models", "")
            or os.getenv(
                "GEMINI_FALLBACK_MODELS",
                "gemini-3.5-flash,gemini-3.5-flash-lite",
            )
        )

        if isinstance(fallback_models_raw, str):
            fallback_list = [
                m.strip()
                for m in fallback_models_raw.split(",")
                if m.strip()
            ]

        elif isinstance(fallback_models_raw, (list, tuple)):
            fallback_list = [
                str(m).strip()
                for m in fallback_models_raw
                if str(m).strip()
            ]

        else:
            fallback_list = []

        models_list = [primary_model]

        for model in fallback_list:
            if model and model not in models_list:
                models_list.append(model)

        self.models = models_list

        # ============================================================
        # TEMPERATURE
        # ============================================================

        self.temperature = float(
            getattr(settings, "gemini_temperature", 0.0)
            or os.getenv("GEMINI_TEMPERATURE", 0.0)
        )

    # ================================================================
    # JSON SCHEMA CLEANING
    # ================================================================

    @staticmethod
    def _clean_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Pydantic's JSON schema into a Gemini Developer API
        compatible JSON schema.

        Important:
        - Remove additionalProperties.
        - Keep properties.
        - Keep required.
        - Keep $defs / $ref.
        - Recursively clean nested objects.
        - Preserve supported numeric/string constraints.
        """

        if not isinstance(schema, dict):
            return schema

        cleaned: Dict[str, Any] = {}

        # ------------------------------------------------------------
        # Supported JSON schema keys
        # ------------------------------------------------------------

        supported_keys = {
            "$id",
            "$defs",
            "$ref",
            "$anchor",
            "type",
            "format",
            "title",
            "description",
            "enum",
            "items",
            "prefixItems",
            "minItems",
            "maxItems",
            "minimum",
            "maximum",
            "properties",
            "required",
            "anyOf",
            "oneOf",
        }

        for key, value in schema.items():

            # --------------------------------------------------------
            # Explicitly remove this problematic key
            # --------------------------------------------------------

            if key == "additionalProperties":
                continue

            if key not in supported_keys:
                continue

            # --------------------------------------------------------
            # Recursive dictionaries
            # --------------------------------------------------------

            if isinstance(value, dict):
                cleaned[key] = GeminiService._clean_json_schema(value)

            # --------------------------------------------------------
            # Lists
            # --------------------------------------------------------

            elif isinstance(value, list):

                cleaned[key] = [
                    GeminiService._clean_json_schema(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]

            else:
                cleaned[key] = value

        # ------------------------------------------------------------
        # Properties
        # ------------------------------------------------------------

        if isinstance(schema.get("properties"), dict):

            cleaned["properties"] = {
                property_name: GeminiService._clean_json_schema(
                    property_schema
                )
                for property_name, property_schema
                in schema["properties"].items()
            }

        # ------------------------------------------------------------
        # $defs
        # ------------------------------------------------------------

        if isinstance(schema.get("$defs"), dict):

            cleaned["$defs"] = {
                definition_name: GeminiService._clean_json_schema(
                    definition_schema
                )
                for definition_name, definition_schema
                in schema["$defs"].items()
            }

        return cleaned

    @classmethod
    def _build_gemini_schema(
        cls,
        response_schema: Type[T],
    ) -> Dict[str, Any]:
        """
        Build a Gemini-compatible JSON schema from a Pydantic model.
        """

        raw_schema = response_schema.model_json_schema()

        cleaned_schema = cls._clean_json_schema(
            raw_schema
        )

        return cleaned_schema

    # ================================================================
    # RESPONSE VALIDATION
    # ================================================================

    @staticmethod
    def _parse_response(
        response: Any,
        response_schema: Type[T],
    ) -> T:

        # ------------------------------------------------------------
        # Preferred path: SDK parsed response
        # ------------------------------------------------------------

        parsed = getattr(response, "parsed", None)

        if parsed is not None:

            if isinstance(parsed, response_schema):
                return parsed

            if isinstance(parsed, BaseModel):
                return response_schema.model_validate(
                    parsed.model_dump()
                )

            if isinstance(parsed, dict):
                return response_schema.model_validate(parsed)

        # ------------------------------------------------------------
        # Fallback: raw JSON text
        # ------------------------------------------------------------

        text = getattr(response, "text", None)

        if text:

            clean_text = text.strip()

            # Remove markdown code fences if Gemini returns them
            clean_text = re.sub(
                r"^```json\s*",
                "",
                clean_text,
                flags=re.IGNORECASE,
            )

            clean_text = re.sub(
                r"\s*```$",
                "",
                clean_text,
                flags=re.IGNORECASE,
            )

            return response_schema.model_validate_json(
                clean_text
            )

        raise ValueError(
            "Gemini returned an empty response."
        )

    # ================================================================
    # GENERATION WITH FALLBACK
    # ================================================================

    def _generate_with_fallback(
        self,
        contents: list,
        response_schema: Type[T],
    ) -> T:

        last_error = None

        max_retries_per_model = 4

        # Build schema ONCE.
        # We don't want to regenerate it for every retry.
        gemini_schema = self._build_gemini_schema(
            response_schema
        )

        for model in self.models:

            for attempt in range(
                1,
                max_retries_per_model + 1,
            ):

                try:

                    # ====================================================
                    # IMPORTANT:
                    # Use response_json_schema instead of passing the
                    # Pydantic class directly as response_schema.
                    #
                    # This avoids the problematic Pydantic
                    # additionalProperties generated schema.
                    # ====================================================

                    config = types.GenerateContentConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json",
                        response_json_schema=gemini_schema,
                    )

                    response = self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )

                    return self._parse_response(
                        response=response,
                        response_schema=response_schema,
                    )

                except Exception as exc:

                    last_error = exc

                    err_msg = str(exc)

                    # ------------------------------------------------
                    # Retry temporary Gemini availability/rate issues
                    # ------------------------------------------------

                    is_retryable = (
                        "503" in err_msg
                        or "429" in err_msg
                        or "UNAVAILABLE" in err_msg
                        or "RESOURCE_EXHAUSTED" in err_msg
                        or "DEADLINE_EXCEEDED" in err_msg
                        or "INTERNAL" in err_msg
                    )

                    if (
                        is_retryable
                        and attempt < max_retries_per_model
                    ):

                        wait_time = (
                            (2 ** attempt) + 1
                        )

                        print(
                            f"\n[Gemini Warning] "
                            f"Model '{model}' "
                            f"temporary failure "
                            f"(attempt "
                            f"{attempt}/{max_retries_per_model}). "
                            f"Waiting {wait_time}s...\n"
                        )

                        time.sleep(wait_time)

                        continue

                    # ------------------------------------------------
                    # Non-retryable error
                    # ------------------------------------------------

                    print(
                        f"\n[Gemini Error] "
                        f"Model: '{model}' "
                        f"failed on attempt {attempt}: "
                        f"{type(exc).__name__}: {exc}\n"
                    )

                    break

        raise GeminiServiceError(
            "All Gemini models failed to process request. "
            f"Last error: {last_error}"
        ) from last_error

    # ================================================================
    # SINGLE DOCUMENT EXTRACTION
    # ================================================================

    def extract(
        self,
        file_bytes: bytes,
        mime_type: str,
        prompt: str,
        response_schema: Type[T],
    ) -> T:
        """
        Extract structured data from a single document.
        """

        file_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type=mime_type,
        )

        contents = [
            file_part,
            prompt,
        ]

        return self._generate_with_fallback(
            contents=contents,
            response_schema=response_schema,
        )

    # ================================================================
    # MULTIPLE DOCUMENT EXTRACTION
    # ================================================================

    def extract_multiple(
        self,
        files: List[Tuple[bytes, str]],
        prompt: str,
        response_schema: Type[T],
    ) -> T:
        """
        Extract structured data from multiple document parts.
        """

        contents = []

        for file_bytes, mime_type in files:

            contents.append(
                types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=mime_type,
                )
            )

        contents.append(prompt)

        return self._generate_with_fallback(
            contents=contents,
            response_schema=response_schema,
        )