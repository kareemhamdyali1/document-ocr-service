# ============================================================
# app/schemas/documents.py
# ============================================================

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# BASE
# ============================================================

class ExtractionBaseModel(BaseModel):

    model_config = ConfigDict(
        extra="ignore"
    )


# ============================================================
# NATIONAL ID
# ============================================================

class NationalIDExtraction(ExtractionBaseModel):

    document_type: str = "national_id"

    # Core identity
    national_id: Optional[str] = None

    full_name_ar: Optional[str] = None
    full_name_en: Optional[str] = None

    date_of_birth: Optional[str] = None

    gender: Optional[str] = None
    nationality: Optional[str] = None
    religion: Optional[str] = None
    marital_status: Optional[str] = None

    address_ar: Optional[str] = None
    governorate: Optional[str] = None
    police_station: Optional[str] = None
    profession: Optional[str] = None

    husband_or_wife_name: Optional[str] = None

    id_issue_date: Optional[str] = None
    id_expiry_date: Optional[str] = None

    age_years: Optional[float] = None

    governorate_code: Optional[str] = None

    # Validation
    validation_passed: Optional[bool] = None
    validation_errors: List[str] = Field(
        default_factory=list
    )

    front_image_processed: bool = False
    back_image_processed: bool = False

    cross_validation_passed: Optional[bool] = None

    # Quality / confidence
    overall_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    ocr_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    field_confidences: Dict[str, float] = Field(
        default_factory=dict
    )

    is_tampered_suspected: Optional[bool] = None


# ============================================================
# SALARY EMPLOYMENT
# ============================================================

class SalaryEmployment(ExtractionBaseModel):

    employer_name: Optional[str] = None
    employee_name: Optional[str] = None
    employee_id: Optional[str] = None
    national_id: Optional[str] = None

    job_title: Optional[str] = None
    department: Optional[str] = None

    employment_start_date: Optional[str] = None


# ============================================================
# SALARY INFORMATION
# ============================================================

class SalaryInformation(ExtractionBaseModel):

    basic_salary: Optional[float] = None
    housing_allowance: Optional[float] = None
    transport_allowance: Optional[float] = None
    other_allowances: Optional[float] = None

    gross_salary: Optional[float] = None
    deductions: Optional[float] = None
    net_salary: Optional[float] = None

    currency: Optional[str] = None


# ============================================================
# SALARY CERTIFICATE
# ============================================================

class SalaryCertificateExtraction(ExtractionBaseModel):

    document_type: str = "salary_certificate"

    employment: SalaryEmployment = Field(
        default_factory=SalaryEmployment
    )

    salary: SalaryInformation = Field(
        default_factory=SalaryInformation
    )

    certificate_date: Optional[str] = None
    salary_period: Optional[str] = None
    certificate_reference: Optional[str] = None

    payment_method: Optional[str] = None
    bank_name: Optional[str] = None

    company_address: Optional[str] = None
    company_phone: Optional[str] = None
    company_email: Optional[str] = None

    hr_contact: Optional[str] = None

    has_company_stamp: Optional[bool] = None
    has_signature: Optional[bool] = None

    signatory_name: Optional[str] = None
    signatory_position: Optional[str] = None

    # Quality / confidence
    overall_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    ocr_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    field_confidences: Dict[str, float] = Field(
        default_factory=dict
    )

    is_tampered_suspected: Optional[bool] = None


# ============================================================
# STATEMENT PERIOD
# ============================================================

class StatementPeriod(ExtractionBaseModel):

    from_date: Optional[str] = None
    to_date: Optional[str] = None


# ============================================================
# BANK ACCOUNT INFORMATION
# ============================================================

class BankAccountInformation(ExtractionBaseModel):

    bank_name: Optional[str] = None
    customer_name: Optional[str] = None

    account_number: Optional[str] = None
    iban: Optional[str] = None

    currency: Optional[str] = None

    statement_period: StatementPeriod = Field(
        default_factory=StatementPeriod
    )

    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None


# ============================================================
# BANK TRANSACTION
# ============================================================

class BankTransaction(ExtractionBaseModel):

    date: Optional[str] = Field(
        default=None,
        description="Transaction date in YYYY-MM-DD format"
    )

    transaction: Optional[str] = None

    reference_no: Optional[str] = None

    debit: Optional[float] = None
    credit: Optional[float] = None
    balance: Optional[float] = None


# ============================================================
# BANK STATEMENT
# ============================================================

class BankStatementExtraction(ExtractionBaseModel):

    document_type: str = "bank_statement"

    account_information: BankAccountInformation = Field(
        default_factory=BankAccountInformation
    )

    transactions: List[BankTransaction] = Field(
        default_factory=list
    )

    # Gemini may extract these directly.
    # Pipeline still validates / derives when needed.
    statement_months: Optional[float] = None

    statement_period_months: Optional[float] = None

    payroll_depositor: Optional[str] = None
    payroll_channel_type: Optional[str] = None

    returned_cheques_count: Optional[int] = None
    overdraft_frequency: Optional[int] = None

    # Quality / confidence
    overall_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    ocr_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    field_confidences: Dict[str, float] = Field(
        default_factory=dict
    )

    is_tampered_suspected: Optional[bool] = None


# ============================================================
# CREDIT FACILITY
# ============================================================

class CreditFacility(ExtractionBaseModel):

    lender_name: Optional[str] = None

    facility_type: Optional[str] = None

    account_number: Optional[str] = None

    status: Optional[str] = None

    # Possible source representations
    credit_limit: Optional[float] = None
    original_amount: Optional[float] = None

    outstanding_balance: Optional[float] = None
    monthly_payment: Optional[float] = None

    # Canonical compatibility fields
    granted_amount: Optional[float] = None
    outstanding_amount: Optional[float] = None
    installment_amount: Optional[float] = None

    overdue_amount: Optional[float] = None

    days_past_due: Optional[int] = None
    overdue_days: Optional[int] = None

    start_date: Optional[str] = None
    maturity_date: Optional[str] = None

    legal_action_flag: Optional[bool] = None

    # Row-level confidence
    ocr_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    field_confidences: Dict[str, float] = Field(
        default_factory=dict
    )


# ============================================================
# I-SCORE
# ============================================================

class IScoreExtraction(ExtractionBaseModel):

    document_type: str = "i_score"

    customer_name: Optional[str] = None
    national_id: Optional[str] = None

    report_date: Optional[str] = None

    credit_score: Optional[float] = None
    score_range: Optional[str] = None

    number_of_facilities: Optional[int] = None

    total_credit_limit: Optional[float] = None

    total_active_loans_limit: Optional[float] = None

    total_outstanding_balance: Optional[float] = None

    total_overdue_amount: Optional[float] = None

    max_days_past_due: Optional[int] = None

    active_credit_cards_count: Optional[int] = None

    total_credit_card_utilization: Optional[float] = None

    facilities: List[CreditFacility] = Field(
        default_factory=list
    )

    payment_history_summary: Optional[str] = None
    adverse_information: Optional[str] = None

    # Quality / confidence
    overall_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    ocr_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    field_confidences: Dict[str, float] = Field(
        default_factory=dict
    )

    is_tampered_suspected: Optional[bool] = None