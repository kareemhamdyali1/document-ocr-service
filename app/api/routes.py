# ============================================================
# app/api/routes.py
# ============================================================

import json

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import Response

from app.services.credit_pipeline import CreditPipeline
from app.services.gemini_service import GeminiServiceError


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/v1/ocr",
    tags=["OCR"],
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = CreditPipeline()


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

@router.post("/process-documents")
async def process_documents(
    national_id_front_file: UploadFile = File(...),
    national_id_back_file: UploadFile = File(...),
    salary_certificate_file: UploadFile = File(...),
    bank_statement_file: UploadFile = File(...),
    iscore_file: UploadFile = File(...),
    application_id: str = Form(...),
):
    """
    Process the five required credit documents.

    Request body:

        national_id_front_file
        national_id_back_file
        salary_certificate_file
        bank_statement_file
        iscore_file
        application_id
    """

    # ========================================================
    # READ NATIONAL ID FRONT
    # ========================================================

    try:
        national_id_front_bytes = (
            await national_id_front_file.read()
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to read national ID front file: "
                f"{exc}"
            ),
        )

    if not national_id_front_bytes:
        raise HTTPException(
            status_code=400,
            detail="National ID front file is empty.",
        )

    # ========================================================
    # READ NATIONAL ID BACK
    # ========================================================

    try:
        national_id_back_bytes = (
            await national_id_back_file.read()
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to read national ID back file: "
                f"{exc}"
            ),
        )

    if not national_id_back_bytes:
        raise HTTPException(
            status_code=400,
            detail="National ID back file is empty.",
        )

    # ========================================================
    # READ SALARY CERTIFICATE
    # ========================================================

    try:
        salary_certificate_bytes = (
            await salary_certificate_file.read()
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to read salary certificate file: "
                f"{exc}"
            ),
        )

    if not salary_certificate_bytes:
        raise HTTPException(
            status_code=400,
            detail="Salary certificate file is empty.",
        )

    # ========================================================
    # READ BANK STATEMENT
    # ========================================================

    try:
        bank_statement_bytes = (
            await bank_statement_file.read()
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to read bank statement file: "
                f"{exc}"
            ),
        )

    if not bank_statement_bytes:
        raise HTTPException(
            status_code=400,
            detail="Bank statement file is empty.",
        )

    # ========================================================
    # READ I-SCORE
    # ========================================================

    try:
        iscore_bytes = await iscore_file.read()

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to read I-Score file: "
                f"{exc}"
            ),
        )

    if not iscore_bytes:
        raise HTTPException(
            status_code=400,
            detail="I-Score file is empty.",
        )

    # ========================================================
    # MIME TYPES
    # ========================================================

    national_id_front_mime = (
        national_id_front_file.content_type
        or "application/octet-stream"
    )

    national_id_back_mime = (
        national_id_back_file.content_type
        or "application/octet-stream"
    )

    salary_certificate_mime = (
        salary_certificate_file.content_type
        or "application/octet-stream"
    )

    bank_statement_mime = (
        bank_statement_file.content_type
        or "application/octet-stream"
    )

    iscore_mime = (
        iscore_file.content_type
        or "application/octet-stream"
    )

    # ========================================================
    # RUN CREDIT PIPELINE
    # ========================================================

    try:

        result = pipeline.process_documents(
            application_id=application_id,

            national_id_front=(
                national_id_front_bytes,
                national_id_front_mime,
            ),

            national_id_back=(
                national_id_back_bytes,
                national_id_back_mime,
            ),

            salary_certificate=(
                salary_certificate_bytes,
                salary_certificate_mime,
            ),

            bank_statement=(
                bank_statement_bytes,
                bank_statement_mime,
            ),

            iscore=(
                iscore_bytes,
                iscore_mime,
            ),
        )

    except GeminiServiceError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {exc}",
        )

    # ========================================================
    # CONVERT PYDANTIC MODEL TO PYTHON DICT
    # ========================================================

    response_data = result.model_dump(
        mode="python"
    )

    # ========================================================
    # DEBUG STATEMENT PERIOD
    # ========================================================

    try:

        statement_value = (
            response_data
            .get(
                "bank_statement_fields",
                {},
            )
            .get(
                "statement_period_months",
                {},
            )
            .get(
                "value"
            )
        )

        print(
            "DEBUG statement_period_months:",
            statement_value,
            type(statement_value),
        )

    except Exception as exc:

        print(
            "DEBUG statement_period_months inspection failed:",
            exc,
        )

    # ========================================================
    # FORCE JSON SERIALIZATION
    # ========================================================

    try:

        json_content = json.dumps(
            response_data,
            ensure_ascii=False,
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to serialize response: "
                f"{exc}"
            ),
        )

    # ========================================================
    # RETURN RAW JSON RESPONSE
    # ========================================================

    return Response(
        content=json_content,
        status_code=200,
        media_type="application/json",
    )