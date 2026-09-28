# ============================================================
# app/schemas/responses.py
# ============================================================

from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
)


# ============================================================
# BASE
# ============================================================

class ResponseBaseModel(BaseModel):

    model_config = ConfigDict(
        extra="ignore"
    )


# ============================================================
# VALUE + CONFIDENCE
# ============================================================

class FieldValue(ResponseBaseModel):

    value: Any = None

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# FLOAT VALUE + CONFIDENCE
# ============================================================

class FloatFieldValue(ResponseBaseModel):

    value: Optional[float] = None

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    @field_validator(
        "value",
        mode="before",
    )
    @classmethod
    def force_float(
        cls,
        value: Any,
    ) -> Optional[float]:

        if value is None:
            return None

        try:

            return float(value)

        except (
            ValueError,
            TypeError,
        ):

            return None


# ============================================================
# DOCUMENT OUTPUT
# ============================================================

class DocumentOutput(ResponseBaseModel):

    document_type: str

    overall_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    is_tampered_suspected: bool = False

    extracted_fields: Dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# NATIONAL ID FIELDS
# ============================================================

class NationalIDFields(ResponseBaseModel):

    full_name: FieldValue = Field(
        default_factory=FieldValue
    )

    national_id: FieldValue = Field(
        default_factory=FieldValue
    )

    date_of_birth: FieldValue = Field(
        default_factory=FieldValue
    )

    age_years: FieldValue = Field(
        default_factory=FieldValue
    )

    gender: FieldValue = Field(
        default_factory=FieldValue
    )

    governorate: FieldValue = Field(
        default_factory=FieldValue
    )

    address: FieldValue = Field(
        default_factory=FieldValue
    )

    profession_on_id: FieldValue = Field(
        default_factory=FieldValue
    )

    marital_status_on_id: FieldValue = Field(
        default_factory=FieldValue
    )

    id_expiry_date: FieldValue = Field(
        default_factory=FieldValue
    )


# ============================================================
# SALARY FIELDS
# ============================================================

class SalaryCertificateFields(ResponseBaseModel):

    employer_name: FieldValue = Field(
        default_factory=FieldValue
    )

    employer_sector: FieldValue = Field(
        default_factory=FieldValue
    )

    job_title: FieldValue = Field(
        default_factory=FieldValue
    )

    declared_net_salary: FieldValue = Field(
        default_factory=FieldValue
    )

    declared_gross_salary: FieldValue = Field(
        default_factory=FieldValue
    )

    employment_date: FieldValue = Field(
        default_factory=FieldValue
    )

    employment_tenure_years: FieldValue = Field(
        default_factory=FieldValue
    )

    issue_date: FieldValue = Field(
        default_factory=FieldValue
    )


# ============================================================
# BANK STATEMENT FIELDS
# ============================================================

class BankStatementFields(ResponseBaseModel):

    bank_name: FieldValue = Field(
        default_factory=FieldValue
    )

    account_number: FieldValue = Field(
        default_factory=FieldValue
    )

    statement_period_months: FloatFieldValue = Field(
        default_factory=FloatFieldValue
    )

    avg_monthly_net_inflow: FieldValue = Field(
        default_factory=FieldValue
    )

    avg_monthly_balance: FieldValue = Field(
        default_factory=FieldValue
    )

    min_monthly_balance: FieldValue = Field(
        default_factory=FieldValue
    )

    max_monthly_balance: FieldValue = Field(
        default_factory=FieldValue
    )

    balance_volatility_std: FieldValue = Field(
        default_factory=FieldValue
    )

    overdraft_frequency: FieldValue = Field(
        default_factory=FieldValue
    )

    returned_cheques_count: FieldValue = Field(
        default_factory=FieldValue
    )

    income_regularity_score: FieldValue = Field(
        default_factory=FieldValue
    )


# ============================================================
# I-SCORE FACILITY
# ============================================================

class BureauFacilityOutput(ResponseBaseModel):

    lender_name: Optional[str] = None

    facility_type: Optional[str] = None

    granted_amount: Optional[float] = None

    outstanding_amount: Optional[float] = None

    installment_amount: Optional[float] = None

    status: Optional[str] = None

    overdue_days: Optional[int] = None

    overdue_amount: Optional[float] = None

    start_date: Optional[str] = None

    legal_action_flag: Optional[bool] = None


# ============================================================
# I-SCORE FIELDS
# ============================================================

class IScoreReportFields(ResponseBaseModel):

    is_available: bool = False

    credit_score: FieldValue = Field(
        default_factory=FieldValue
    )

    score_tier: FieldValue = Field(
        default_factory=FieldValue
    )

    score_date: FieldValue = Field(
        default_factory=FieldValue
    )

    total_active_loans_limit: FieldValue = Field(
        default_factory=FieldValue
    )

    total_outstanding_balance: FieldValue = Field(
        default_factory=FieldValue
    )

    total_overdue_amount: FieldValue = Field(
        default_factory=FieldValue
    )

    max_days_past_due: FieldValue = Field(
        default_factory=FieldValue
    )

    active_credit_cards_count: FieldValue = Field(
        default_factory=FieldValue
    )

    total_credit_card_utilization: FieldValue = Field(
        default_factory=FieldValue
    )

    # Raw facility list.
    bureau_facilities: List[
        BureauFacilityOutput
    ] = Field(
        default_factory=list
    )


# ============================================================
# OCR CONTRACT
# ============================================================

class CanonicalContract(ResponseBaseModel):

    application_id: str

    applicant_type: str = "individual"

    submission_timestamp: str

    is_returning_customer: Optional[bool] = None

    documents: List[
        DocumentOutput
    ] = Field(
        default_factory=list
    )

    national_id_fields: NationalIDFields = Field(
        default_factory=NationalIDFields
    )

    salary_certificate_fields: SalaryCertificateFields = Field(
        default_factory=SalaryCertificateFields
    )

    bank_statement_fields: BankStatementFields = Field(
        default_factory=BankStatementFields
    )

    iscore_report_fields: IScoreReportFields = Field(
        default_factory=IScoreReportFields
    )

    form_data: Dict[str, Any] = Field(
        default_factory=dict
    )

    internal_history: Dict[str, Any] = Field(
        default_factory=dict
    )

    consistency_checks: Dict[str, Any] = Field(
        default_factory=dict
    )