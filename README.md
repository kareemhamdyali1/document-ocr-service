# Credit Document AI API

Gemini-powered document OCR and extraction microservice for an Egyptian credit-financing platform.

The service extracts structured information from:

- Egyptian National ID
- Salary Certificate
- Bank Statement

The extracted information is normalized, validated, and assembled into a canonical credit-document contract.

---

## Architecture

```text
Client
  |
  | multipart/form-data
  v
FastAPI
  |
  v
Credit Pipeline
  |
  v
Document Service
  |
  v
Gemini API
  |
  +---------------------+
  |                     |
  v                     v
National ID        Salary Certificate
Parser             Parser
  |
  v
Validation

  +
  
Bank Statement
Parser
  |
  v
Deterministic
Financial Calculations

  |
  v
Canonical Contract
  |
  v
JSON Response