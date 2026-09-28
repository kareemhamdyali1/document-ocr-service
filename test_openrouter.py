import json
from pathlib import Path

from app.services.openrouter_service import OpenRouterService
from app.schemas.documents import NationalIDExtraction
from app.prompts.prompts import NATIONAL_ID_PROMPT


# ============================================================
# CHANGE THIS PATH
# ============================================================

IMAGE_PATH = Path(
    r"C:\Users\USER\Downloads\Front.jpeg"
)


# ============================================================
# CHECK FILE
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


# ============================================================
# READ IMAGE
# ============================================================

file_bytes = IMAGE_PATH.read_bytes()

suffix = IMAGE_PATH.suffix.lower()

mime_types = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

mime_type = mime_types.get(
    suffix,
    "image/jpeg",
)


print("=" * 70)
print("OPENROUTER OCR TEST")
print("=" * 70)

print("Image:", IMAGE_PATH)
print("Size:", len(file_bytes), "bytes")
print("MIME:", mime_type)


# ============================================================
# CREATE SERVICE
# ============================================================

service = OpenRouterService()


# ============================================================
# EXTRACT
# ============================================================

try:

    result = service.extract(
        file_bytes=file_bytes,
        mime_type=mime_type,
        prompt=NATIONAL_ID_PROMPT,
        response_schema=NationalIDExtraction,
    )

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        json.dumps(
            result.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )

except Exception as e:

    print("\n" + "=" * 70)
    print("FAILED")
    print("=" * 70)

    print(type(e).__name__)
    print(str(e))

    raise