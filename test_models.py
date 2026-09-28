import os
from dotenv import load_dotenv
from google import genai

# 1. تحميل المتغيرات من ملف .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("X Error: GEMINI_API_KEY not found in .env file!")
    exit(1)

client = genai.Client(api_key=api_key)

# 2. قائمة الموديلات التي سنختبرها
candidate_models = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-1.5-pro",
]

print("--- Testing Available Models ---")

working_model = None

for model in candidate_models:
    print(f"Testing model: '{model}'...")
    try:
        response = client.models.generate_content(
            model=model,
            contents="Hello, reply with OK"
        )
        print(f"SUCCESS with '{model}'! Response: {response.text.strip()}\n")
        working_model = model
        break
    except Exception as e:
        err_msg = str(e)
        if "503" in err_msg or "UNAVAILABLE" in err_msg:
            print(f"  -> High demand (503) on {model}")
        elif "404" in err_msg or "NOT_FOUND" in err_msg:
            print(f"  -> Model {model} not found for this key")
        elif "429" in err_msg:
            print(f"  -> Quota limit (429) on {model}")
        else:
            print(f"  -> Failed: {err_msg[:100]}...")

if working_model:
    print(f"==========================================")
    print(f"Use this model in your .env file: {working_model}")
    print(f"==========================================")
else:
    print("\nAll models failed on Free Tier. Please check if your API key has Billing enabled.")