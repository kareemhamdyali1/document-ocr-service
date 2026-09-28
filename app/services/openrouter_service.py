# ============================================================
# app/services/openrouter_service.py
# ============================================================

import base64
import json
import time
from typing import Type, TypeVar, Optional, Any

import requests
from pydantic import BaseModel

from app.core.config import settings


# ============================================================
# TYPES
# ============================================================

T = TypeVar("T", bound=BaseModel)


# ============================================================
# ERRORS
# ============================================================

class OpenRouterServiceError(Exception):
    """
    Raised when OpenRouter extraction fails.
    """

    pass


# ============================================================
# SERVICE
# ============================================================

class OpenRouterService:
    """
    OpenRouter multimodal document extraction service.

    Supports:
        - JPEG
        - PNG
        - WEBP
        - PDF

    Designed as a replacement for GeminiService.

    Main responsibilities:
        - Convert files to image inputs
        - Send multimodal requests to OpenRouter
        - Support Pydantic structured outputs
        - Fallback to normal JSON prompting if needed
        - Parse model responses safely
        - Validate output with Pydantic
        - Handle provider/model failures
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    # ========================================================
    # INIT
    # ========================================================

    def __init__(self):

        if not settings.openrouter_api_key:
            raise OpenRouterServiceError(
                "OPENROUTER_API_KEY is not configured."
            )

        self.api_key = settings.openrouter_api_key

        self.models = settings.openrouter_models

        self.temperature = settings.openrouter_temperature

        self.max_tokens = settings.openrouter_max_tokens

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if settings.openrouter_http_referer:
            self.headers["HTTP-Referer"] = (
                settings.openrouter_http_referer
            )

        if settings.openrouter_app_name:
            self.headers["X-Title"] = (
                settings.openrouter_app_name
            )

    # ========================================================
    # FILE HELPERS
    # ========================================================

    @staticmethod
    def _mime_to_data_uri(
        file_bytes: bytes,
        mime_type: str,
    ) -> str:
        """
        Convert raw bytes into a base64 data URI.
        """

        encoded = base64.b64encode(
            file_bytes
        ).decode("utf-8")

        return f"data:{mime_type};base64,{encoded}"

    # ========================================================
    # PDF -> IMAGES
    # ========================================================

    @staticmethod
    def _pdf_to_images(
        file_bytes: bytes,
        max_pages: int = 5,
    ) -> list[bytes]:
        """
        Convert PDF pages into PNG images.
        """

        try:
            import fitz

        except ImportError as exc:

            raise OpenRouterServiceError(
                "PyMuPDF is required for PDF processing. "
                "Install it using: pip install pymupdf"
            ) from exc

        try:

            document = fitz.open(
                stream=file_bytes,
                filetype="pdf",
            )

            images = []

            page_count = min(
                len(document),
                max_pages,
            )

            for page_index in range(page_count):

                page = document.load_page(
                    page_index
                )

                matrix = fitz.Matrix(
                    2.0,
                    2.0,
                )

                pixmap = page.get_pixmap(
                    matrix=matrix,
                    alpha=False,
                )

                png_bytes = pixmap.tobytes(
                    "png"
                )

                images.append(
                    png_bytes
                )

            document.close()

            if not images:

                raise OpenRouterServiceError(
                    "PDF contains no readable pages."
                )

            return images

        except OpenRouterServiceError:
            raise

        except Exception as exc:

            raise OpenRouterServiceError(
                f"Failed to convert PDF to images: {exc}"
            ) from exc

    # ========================================================
    # BUILD IMAGE PARTS
    # ========================================================

    def _build_image_parts(
        self,
        file_bytes: bytes,
        mime_type: str,
    ) -> list[dict]:
        """
        Convert document bytes into OpenRouter image parts.
        """

        mime_type = (
            mime_type or "application/octet-stream"
        ).lower()

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if mime_type == "application/pdf":

            pdf_images = self._pdf_to_images(
                file_bytes=file_bytes,
                max_pages=5,
            )

            parts = []

            for image_bytes in pdf_images:

                data_uri = self._mime_to_data_uri(
                    image_bytes,
                    "image/png",
                )

                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                        },
                    }
                )

            return parts

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if mime_type.startswith("image/"):

            data_uri = self._mime_to_data_uri(
                file_bytes,
                mime_type,
            )

            return [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_uri,
                    },
                }
            ]

        raise OpenRouterServiceError(
            f"Unsupported document MIME type: {mime_type}"
        )

    # ========================================================
    # SCHEMA CLEANING
    # ========================================================

    @staticmethod
    def _clean_schema(
        schema: dict,
    ) -> dict:
        """
        Clean Pydantic JSON schema for providers
        that do not accept some JSON Schema metadata.
        """

        if not isinstance(schema, dict):
            return schema

        schema = dict(schema)

        schema.pop(
            "$schema",
            None,
        )

        schema.pop(
            "additionalProperties",
            None,
        )

        def clean(value: Any):

            if isinstance(value, dict):

                value.pop(
                    "$schema",
                    None,
                )

                value.pop(
                    "additionalProperties",
                    None,
                )

                for child in value.values():
                    clean(child)

            elif isinstance(value, list):

                for child in value:
                    clean(child)

        clean(schema)

        return schema

    # ========================================================
    # RESPONSE FORMAT
    # ========================================================

    @classmethod
    def _build_response_format(
        cls,
        response_schema: Type[BaseModel],
    ) -> dict:

        schema = response_schema.model_json_schema()

        schema = cls._clean_schema(
            schema
        )

        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.__name__,
                "strict": True,
                "schema": schema,
            },
        }

    # ========================================================
    # FALLBACK JSON PROMPT
    # ========================================================

    @classmethod
    def _build_json_fallback_prompt(
        cls,
        prompt: str,
        response_schema: Type[BaseModel],
    ) -> str:
        """
        Build a strict JSON-only prompt.

        This is used when a provider/model does not support
        response_format=json_schema.
        """

        schema = response_schema.model_json_schema()

        schema = cls._clean_schema(
            schema
        )

        schema_text = json.dumps(
            schema,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
{prompt}

============================================================
STRICT OUTPUT REQUIREMENTS
============================================================

You are performing document OCR and structured extraction.

IMPORTANT:

1. Extract ONLY information actually visible in the provided
   document image(s).

2. DO NOT guess.

3. DO NOT infer information from:
   - names
   - gender
   - ID structure
   - common Egyptian ID formats
   - external knowledge
   - previous examples

4. If a field cannot be read reliably, return null.

5. Preserve numbers exactly as visible before normalization.

6. For Arabic digits, read them carefully and convert them
   to Western digits only in the final JSON if required.

7. Do NOT explain your answer.

8. Do NOT provide reasoning.

9. Do NOT return Markdown.

10. Return ONLY one valid JSON object.

The JSON object MUST follow this schema:

{schema_text}

============================================================
END REQUIREMENTS
============================================================
""".strip()

    # ========================================================
    # ERROR CLASSIFICATION
    # ========================================================

    @staticmethod
    def _classify_error(
        status_code: int,
        message: str,
    ) -> str:

        text = message.lower()

        # ----------------------------------------------------
        # Authentication
        # ----------------------------------------------------

        if status_code in (401, 403):
            return "auth"

        if any(
            token in text
            for token in [
                "invalid api key",
                "authentication",
                "unauthorized",
                "forbidden",
            ]
        ):
            return "auth"

        # ----------------------------------------------------
        # Quota
        # ----------------------------------------------------

        if status_code == 429:
            return "quota"

        if any(
            token in text
            for token in [
                "rate limit",
                "rate_limit",
                "quota exceeded",
                "too many requests",
            ]
        ):
            return "quota"

        # ----------------------------------------------------
        # Provider unavailable
        # ----------------------------------------------------

        if status_code in (502, 503, 504):
            return "unavailable"

        if any(
            token in text
            for token in [
                "temporarily unavailable",
                "service unavailable",
                "high demand",
                "overloaded",
            ]
        ):
            return "unavailable"

        # ----------------------------------------------------
        # Bad request
        # ----------------------------------------------------

        if status_code == 400:
            return "bad_request"

        # ----------------------------------------------------
        # Network
        # ----------------------------------------------------

        if any(
            token in text
            for token in [
                "timeout",
                "timed out",
                "connection",
                "network",
            ]
        ):
            return "network"

        return "unknown"

    # ========================================================
    # REQUEST
    # ========================================================

    def _request(
        self,
        model: str,
        messages: list[dict],
        response_format: Optional[dict] = None,
    ) -> dict:

        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,

            # Tell OpenRouter that reasoning is not required
            # for this extraction request.
            "reasoning": {
                "exclude": True,
            },
        }

        if response_format is not None:

            payload["response_format"] = (
                response_format
            )

        response = requests.post(
            self.BASE_URL,
            headers=self.headers,
            json=payload,
            timeout=180,
        )

        # ----------------------------------------------------
        # DEBUG INFORMATION
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "OPENROUTER RESPONSE"
        )

        print(
            "=" * 70
        )

        print(
            "HTTP STATUS:",
            response.status_code,
        )

        # Only print the response when debugging is useful.
        # Limit output to avoid huge logs.
        print(
            "RESPONSE PREVIEW:"
        )

        print(
            response.text[:5000]
        )

        print(
            "=" * 70
            + "\n"
        )

        # ----------------------------------------------------
        # HTTP ERROR
        # ----------------------------------------------------

        if not response.ok:

            try:

                error_json = response.json()

                error_message = json.dumps(
                    error_json,
                    ensure_ascii=False,
                )

            except Exception:

                error_message = (
                    response.text
                )

            error_type = (
                self._classify_error(
                    response.status_code,
                    error_message,
                )
            )

            raise OpenRouterServiceError(
                f"{response.status_code} "
                f"{error_type}: "
                f"{error_message}"
            )

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        try:

            response_json = response.json()

        except Exception as exc:

            raise OpenRouterServiceError(
                "OpenRouter returned invalid JSON: "
                f"{exc}"
            ) from exc

        return response_json

    # ========================================================
    # RESPONSE TEXT
    # ========================================================

    @staticmethod
    def _extract_response_text(
        response_json: dict,
    ) -> str:
        """
        Extract final assistant content.

        Important:
        We intentionally ignore:
            - reasoning
            - reasoning_details
            - provider metadata

        and only use message.content.
        """

        # ----------------------------------------------------
        # Validate response
        # ----------------------------------------------------

        if not isinstance(
            response_json,
            dict,
        ):

            raise OpenRouterServiceError(
                "OpenRouter returned a non-object response."
            )

        choices = response_json.get(
            "choices"
        )

        if not choices:

            # Include a useful preview so future provider
            # errors are immediately diagnosable.
            raise OpenRouterServiceError(
                "OpenRouter response contains no choices.\n"
                f"Response preview: "
                f"{json.dumps(response_json, ensure_ascii=False)[:5000]}"
            )

        first_choice = choices[0]

        if not isinstance(
            first_choice,
            dict,
        ):

            raise OpenRouterServiceError(
                "OpenRouter returned an invalid choice object."
            )

        message = first_choice.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):

            raise OpenRouterServiceError(
                "OpenRouter response contains no valid message."
            )

        # ----------------------------------------------------
        # Content
        # ----------------------------------------------------

        content = message.get(
            "content"
        )

        if content is None:

            raise OpenRouterServiceError(
                "OpenRouter response contains no message content."
            )

        # ----------------------------------------------------
        # Normal string response
        # ----------------------------------------------------

        if isinstance(
            content,
            str,
        ):

            return content.strip()

        # ----------------------------------------------------
        # Structured content
        # ----------------------------------------------------

        if isinstance(
            content,
            list,
        ):

            texts = []

            for item in content:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                text = item.get(
                    "text"
                )

                if text:
                    texts.append(
                        text
                    )

            if texts:

                return "".join(
                    texts
                ).strip()

        raise OpenRouterServiceError(
            "Unable to extract final text "
            "from OpenRouter response."
        )

    # ========================================================
    # JSON PARSING
    # ========================================================

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict:

        if not text:

            raise OpenRouterServiceError(
                "Model returned an empty response."
            )

        text = text.strip()

        # ----------------------------------------------------
        # Direct JSON
        # ----------------------------------------------------

        try:

            result = json.loads(
                text
            )

            if isinstance(
                result,
                dict,
            ):
                return result

        except json.JSONDecodeError:
            pass

        # ----------------------------------------------------
        # Markdown JSON block
        # ----------------------------------------------------

        if text.startswith(
            "```"
        ):

            lines = text.splitlines()

            if len(lines) >= 3:

                lines = lines[1:]

                if lines[-1].strip().startswith(
                    "```"
                ):

                    lines = lines[:-1]

                cleaned = "\n".join(
                    lines
                ).strip()

                try:

                    result = json.loads(
                        cleaned
                    )

                    if isinstance(
                        result,
                        dict,
                    ):
                        return result

                except json.JSONDecodeError:
                    pass

        # ----------------------------------------------------
        # Find JSON object
        # ----------------------------------------------------

        first = text.find(
            "{"
        )

        last = text.rfind(
            "}"
        )

        if (
            first != -1
            and last != -1
            and last > first
        ):

            candidate = text[
                first:last + 1
            ]

            try:

                result = json.loads(
                    candidate
                )

                if isinstance(
                    result,
                    dict,
                ):
                    return result

            except json.JSONDecodeError:
                pass

        raise OpenRouterServiceError(
            "Model returned invalid JSON.\n"
            f"Response:\n{text[:5000]}"
        )

    # ========================================================
    # PYDANTIC VALIDATION
    # ========================================================

    @staticmethod
    def _validate_schema(
        data: dict,
        response_schema: Type[T],
    ) -> T:

        try:

            return response_schema.model_validate(
                data
            )

        except Exception as exc:

            raise OpenRouterServiceError(
                "OpenRouter response failed "
                "Pydantic validation.\n"
                f"Validation error: {exc}\n"
                "Response: "
                f"{json.dumps(data, ensure_ascii=False)[:5000]}"
            ) from exc

    # ========================================================
    # GENERATE
    # ========================================================

    def _generate(
        self,
        files: list[tuple[bytes, str]],
        prompt: str,
        response_schema: Type[T],
        max_retries_per_model: int = 1,
    ) -> T:

        # ====================================================
        # BUILD MULTIMODAL CONTENT
        # ====================================================

        content = []

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        # ----------------------------------------------------
        # Files
        # ----------------------------------------------------

        for (
            file_bytes,
            mime_type,
        ) in files:

            image_parts = (
                self._build_image_parts(
                    file_bytes=file_bytes,
                    mime_type=mime_type,
                )
            )

            content.extend(
                image_parts
            )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # ----------------------------------------------------
        # Structured output
        # ----------------------------------------------------

        response_format = (
            self._build_response_format(
                response_schema
            )
        )

        # ----------------------------------------------------
        # Fallback prompt
        # ----------------------------------------------------

        fallback_prompt = (
            self._build_json_fallback_prompt(
                prompt=prompt,
                response_schema=response_schema,
            )
        )

        fallback_content = []

        fallback_content.append(
            {
                "type": "text",
                "text": fallback_prompt,
            }
        )

        for (
            file_bytes,
            mime_type,
        ) in files:

            fallback_content.extend(
                self._build_image_parts(
                    file_bytes=file_bytes,
                    mime_type=mime_type,
                )
            )

        fallback_messages = [
            {
                "role": "user",
                "content": fallback_content,
            }
        ]

        errors = []

        # ====================================================
        # MODEL LOOP
        # ====================================================

        for model in self.models:

            # ------------------------------------------------
            # Attempt 1:
            # Structured JSON schema
            # ------------------------------------------------

            for attempt in range(
                1,
                max_retries_per_model + 1,
            ):

                try:

                    print(
                        f"[OpenRouter] "
                        f"model={model} "
                        f"structured=True "
                        f"attempt={attempt}"
                    )

                    response_json = (
                        self._request(
                            model=model,
                            messages=messages,
                            response_format=response_format,
                        )
                    )

                    response_text = (
                        self._extract_response_text(
                            response_json
                        )
                    )

                    data = (
                        self._parse_json(
                            response_text
                        )
                    )

                    result = (
                        self._validate_schema(
                            data,
                            response_schema,
                        )
                    )

                    print(
                        f"[OpenRouter] "
                        f"SUCCESS "
                        f"model={model} "
                        f"structured=True"
                    )

                    return result

                except OpenRouterServiceError as exc:

                    error_message = str(
                        exc
                    )

                    errors.append(
                        {
                            "model": model,
                            "attempt": attempt,
                            "mode": "structured",
                            "error": error_message,
                        }
                    )

                    print(
                        f"[OpenRouter] "
                        f"FAILED "
                        f"model={model} "
                        f"structured=True "
                        f"attempt={attempt} "
                        f"error={error_message}"
                    )

                    # ----------------------------------------
                    # Authentication
                    # ----------------------------------------

                    if (
                        " auth:" in error_message
                    ):

                        raise OpenRouterServiceError(
                            "OpenRouter authentication failed. "
                            "Check OPENROUTER_API_KEY."
                        )

                    # ----------------------------------------
                    # Quota
                    # ----------------------------------------

                    if (
                        " quota:" in error_message
                    ):

                        break

                    # ----------------------------------------
                    # Retry unavailable/network
                    # ----------------------------------------

                    if (
                        " unavailable:"
                        in error_message
                    ):

                        if (
                            attempt
                            < max_retries_per_model
                        ):

                            time.sleep(3)
                            continue

                        break

                    if (
                        " network:"
                        in error_message
                    ):

                        if (
                            attempt
                            < max_retries_per_model
                        ):

                            time.sleep(2)
                            continue

                        break

                    # ----------------------------------------
                    # Unknown
                    # ----------------------------------------

                    if (
                        attempt
                        < max_retries_per_model
                    ):

                        time.sleep(2)
                        continue

                    break

            # =================================================
            # FALLBACK:
            # PLAIN JSON
            # =================================================

            try:

                print(
                    f"[OpenRouter] "
                    f"model={model} "
                    f"structured=False "
                    f"fallback=True"
                )

                response_json = (
                    self._request(
                        model=model,
                        messages=fallback_messages,
                        response_format=None,
                    )
                )

                response_text = (
                    self._extract_response_text(
                        response_json
                    )
                )

                data = (
                    self._parse_json(
                        response_text
                    )
                )

                result = (
                    self._validate_schema(
                        data,
                        response_schema,
                    )
                )

                print(
                    f"[OpenRouter] "
                    f"SUCCESS "
                    f"model={model} "
                    f"structured=False"
                )

                return result

            except OpenRouterServiceError as exc:

                error_message = str(
                    exc
                )

                errors.append(
                    {
                        "model": model,
                        "attempt": 1,
                        "mode": "fallback",
                        "error": error_message,
                    }
                )

                print(
                    f"[OpenRouter] "
                    f"FAILED "
                    f"model={model} "
                    f"structured=False "
                    f"error={error_message}"
                )

                if (
                    " auth:"
                    in error_message
                ):

                    raise OpenRouterServiceError(
                        "OpenRouter authentication failed. "
                        "Check OPENROUTER_API_KEY."
                    )

                if (
                    " quota:"
                    in error_message
                ):

                    continue

                if (
                    " unavailable:"
                    in error_message
                ):

                    time.sleep(2)

                continue

        # ====================================================
        # EVERYTHING FAILED
        # ====================================================

        summary = []

        for error in errors:

            summary.append(
                f"model={error['model']} "
                f"attempt={error['attempt']} "
                f"mode={error['mode']} "
                f"{error['error']}"
            )

        raise OpenRouterServiceError(
            "All OpenRouter models failed.\n"
            + "\n".join(summary)
        )

    # ========================================================
    # SINGLE DOCUMENT
    # ========================================================

    def extract(
        self,
        file_bytes: bytes,
        mime_type: str,
        prompt: str,
        response_schema: Type[T],
    ) -> T:

        return self._generate(
            files=[
                (
                    file_bytes,
                    mime_type,
                )
            ],
            prompt=prompt,
            response_schema=response_schema,
        )

    # ========================================================
    # MULTIPLE DOCUMENTS
    # ========================================================

    def extract_multiple(
        self,
        files: list[tuple[bytes, str]],
        prompt: str,
        response_schema: Type[T],
    ) -> T:

        return self._generate(
            files=files,
            prompt=prompt,
            response_schema=response_schema,
        )