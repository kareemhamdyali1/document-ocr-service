# ============================================================
# app/services/credit_pipeline.py
# ============================================================

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.responses import CanonicalContract

from app.services.document_service import (
    DocumentService,
    EGYPT_GOVERNORATES,
)


# ============================================================
# CREDIT PIPELINE
# ============================================================

class CreditPipeline:

    def __init__(self):
        self.document_service = DocumentService()

    # ========================================================
    # GENERIC HELPERS
    # ========================================================

    @staticmethod
    def get_attr(
        obj: Any,
        name: str,
        default: Any = None,
    ) -> Any:

        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(name, default)

        return getattr(
            obj,
            name,
            default,
        )

    @staticmethod
    def get_nested(
        obj: Any,
        path: str,
        default: Any = None,
    ) -> Any:

        current = obj

        for part in path.split("."):

            if current is None:
                return default

            if isinstance(current, dict):

                current = current.get(
                    part,
                    default,
                )

            else:

                current = getattr(
                    current,
                    part,
                    default,
                )

        return current

    @staticmethod
    def safe_float(
        value: Any,
    ) -> Optional[float]:

        if value is None:
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def safe_int(
        value: Any,
    ) -> Optional[int]:

        if value is None:
            return None

        try:
            return int(float(value))

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def safe_round(
        value: Any,
        digits: int = 2,
    ):

        if value is None:
            return None

        try:
            return round(
                float(value),
                digits,
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    # ========================================================
    # FIELD CONFIDENCE
    # ========================================================

    @staticmethod
    def get_field_confidences(
        result: Any,
    ) -> Dict[str, float]:

        value = CreditPipeline.get_attr(
            result,
            "field_confidences",
            None,
        )

        # ----------------------------------------------------
        # Dict format
        # ----------------------------------------------------
        #
        # {
        #     "field_name": 0.97
        # }
        #

        if isinstance(value, dict):

            output: Dict[str, float] = {}

            for key, confidence in value.items():

                try:

                    confidence = float(
                        confidence
                    )

                    confidence = max(
                        0.0,
                        min(
                            1.0,
                            confidence,
                        ),
                    )

                    output[str(key)] = confidence

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            return output

        # ----------------------------------------------------
        # Gemini-compatible list format
        # ----------------------------------------------------
        #
        # [
        #     {
        #         "field": "field_name",
        #         "confidence": 0.97
        #     }
        # ]
        #

        if isinstance(value, list):

            output: Dict[str, float] = {}

            for item in value:

                if isinstance(item, dict):

                    key = item.get(
                        "field"
                    )

                    confidence = item.get(
                        "confidence"
                    )

                else:

                    key = getattr(
                        item,
                        "field",
                        None,
                    )

                    confidence = getattr(
                        item,
                        "confidence",
                        None,
                    )

                if key is None:
                    continue

                try:

                    confidence = float(
                        confidence
                    )

                    confidence = max(
                        0.0,
                        min(
                            1.0,
                            confidence,
                        ),
                    )

                    output[str(key)] = confidence

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            return output

        return {}

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def get_confidence(
        self,
        obj: Any,
        field_name: str,
        aliases: Optional[List[str]] = None,
        allow_overall_fallback: bool = True,
        parent: Any = None,
    ) -> float:
        """
        Resolve confidence for a field.

        Priority:

        1. field_confidences[field_name] if > 0
        2. field_confidences[alias] if > 0
        3. field-specific '<field>_confidence' if > 0
        4. alias-specific confidence if > 0
        5. obj.ocr_confidence
        6. parent.ocr_confidence
        7. obj.overall_quality_score
        8. parent.overall_quality_score
        9. 0.0

        Important:
        A confidence value of 0 is treated as "no useful
        confidence information" and therefore does not block
        fallback to the document-level confidence.

        This is important for Gemini outputs where a field may
        be present but its individual confidence was omitted
        or returned as 0.
        """

        if obj is None and parent is None:
            return 0.0

        aliases = aliases or []

        field_names = [
            field_name,
            *aliases,
        ]

        # ----------------------------------------------------
        # 1 + 2. field_confidences
        # ----------------------------------------------------

        for source in (
            obj,
            parent,
        ):

            if source is None:
                continue

            field_confidences = (
                self.get_field_confidences(
                    source
                )
            )

            for name in field_names:

                if name not in field_confidences:
                    continue

                confidence = field_confidences[name]

                # A positive field confidence is authoritative.
                if confidence > 0.0:
                    return confidence

                # If it is exactly zero, continue searching
                # for a useful fallback.
                continue

        # ----------------------------------------------------
        # 3 + 4. Field-specific confidence attributes
        # ----------------------------------------------------

        for source in (
            obj,
            parent,
        ):

            if source is None:
                continue

            for name in field_names:

                value = self.get_attr(
                    source,
                    f"{name}_confidence",
                    None,
                )

                if value is None:
                    continue

                try:

                    confidence = max(
                        0.0,
                        min(
                            1.0,
                            float(value),
                        ),
                    )

                    if confidence > 0.0:
                        return confidence

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

        # ----------------------------------------------------
        # If overall fallback is disabled
        # ----------------------------------------------------

        if not allow_overall_fallback:
            return 0.0

        # ----------------------------------------------------
        # 5 + 6. OCR confidence
        # ----------------------------------------------------

        for source in (
            obj,
            parent,
        ):

            if source is None:
                continue

            ocr_confidence = self.get_attr(
                source,
                "ocr_confidence",
                None,
            )

            if ocr_confidence is None:
                continue

            try:

                confidence = max(
                    0.0,
                    min(
                        1.0,
                        float(ocr_confidence),
                    ),
                )

                if confidence > 0.0:
                    return confidence

            except (
                TypeError,
                ValueError,
            ):
                continue

        # ----------------------------------------------------
        # 7 + 8. Overall document quality
        # ----------------------------------------------------

        for source in (
            obj,
            parent,
        ):

            if source is None:
                continue

            quality = self.get_attr(
                source,
                "overall_quality_score",
                None,
            )

            if quality is None:
                continue

            try:

                confidence = max(
                    0.0,
                    min(
                        1.0,
                        float(quality),
                    ),
                )

                if confidence > 0.0:
                    return confidence

            except (
                TypeError,
                ValueError,
            ):
                continue

        return 0.0

    # ========================================================
    # FIELD VALUE
    # ========================================================

    def make_field(
        self,
        value: Any,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:

        if value is None:

            return {
                "value": None,
                "confidence": 0.0,
            }

        if confidence is None:
            confidence = 0.0

        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        return {
            "value": value,
            "confidence": confidence,
        }

    # ========================================================
    # DOCUMENT QUALITY
    # ========================================================

    def get_document_quality(
        self,
        result: Any,
    ) -> float:

        value = self.get_attr(
            result,
            "overall_quality_score",
            None,
        )

        if value is None:

            value = self.get_attr(
                result,
                "ocr_confidence",
                None,
            )

        if value is None:
            return 0.0

        try:

            return max(
                0.0,
                min(
                    1.0,
                    float(value),
                ),
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    # ========================================================
    # TAMPERING
    # ========================================================

    @staticmethod
    def get_tampering_flag(
        result: Any,
    ) -> bool:

        value = CreditPipeline.get_attr(
            result,
            "is_tampered_suspected",
            None,
        )

        if value is None:
            return False

        return bool(value)

    # ========================================================
    # NATIONAL ID
    # ========================================================

    def normalize_national_id(
        self,
        result,
    ) -> Dict[str, Any]:

        national_id = (
            self.document_service
            .normalize_national_id(
                self.get_attr(
                    result,
                    "national_id",
                )
            )
        )

        date_of_birth = self.get_attr(
            result,
            "date_of_birth",
        )

        if date_of_birth:

            date_of_birth = (
                self.document_service
                .normalize_date(
                    date_of_birth
                )
            )

        age_years = self.get_attr(
            result,
            "age_years",
        )

        if age_years is None and date_of_birth:

            age_years = (
                self.document_service
                .calculate_age(
                    date_of_birth
                )
            )

        gender = self.get_attr(
            result,
            "gender",
        )

        if (
            gender is None
            and national_id
            and len(national_id) == 14
        ):

            gender_digit = int(
                national_id[12]
            )

            gender = (
                "female"
                if gender_digit % 2 == 0
                else "male"
            )

        governorate = self.get_attr(
            result,
            "governorate",
        )

        if national_id:

            code = national_id[7:9]

            governorate = (
                EGYPT_GOVERNORATES.get(
                    code,
                    governorate,
                )
            )

        expiry_date = (
            self.document_service
            .normalize_date(
                self.get_attr(
                    result,
                    "id_expiry_date",
                )
            )
        )

        return {

            "full_name": self.make_field(
                self.get_attr(
                    result,
                    "full_name_ar",
                ),
                self.get_confidence(
                    result,
                    "full_name_ar",
                    aliases=["full_name"],
                ),
            ),

            "national_id": self.make_field(
                national_id,
                self.get_confidence(
                    result,
                    "national_id",
                ),
            ),

            "date_of_birth": self.make_field(
                date_of_birth,
                self.get_confidence(
                    result,
                    "date_of_birth",
                ),
            ),

            "age_years": self.make_field(
                age_years,
                self.get_confidence(
                    result,
                    "age_years",
                ),
            ),

            "gender": self.make_field(
                gender,
                self.get_confidence(
                    result,
                    "gender",
                ),
            ),

            "governorate": self.make_field(
                governorate,
                self.get_confidence(
                    result,
                    "governorate",
                ),
            ),

            "address": self.make_field(
                self.get_attr(
                    result,
                    "address_ar",
                ),
                self.get_confidence(
                    result,
                    "address_ar",
                    aliases=["address"],
                ),
            ),

            "profession_on_id": self.make_field(
                self.get_attr(
                    result,
                    "profession",
                ),
                self.get_confidence(
                    result,
                    "profession",
                ),
            ),

            "marital_status_on_id": self.make_field(
                self.get_attr(
                    result,
                    "marital_status",
                ),
                self.get_confidence(
                    result,
                    "marital_status",
                ),
            ),

            "id_expiry_date": self.make_field(
                expiry_date,
                self.get_confidence(
                    result,
                    "id_expiry_date",
                ),
            ),
        }

    # ========================================================
    # SALARY
    # ========================================================

    def normalize_salary(
        self,
        result,
    ) -> Dict[str, Any]:

        employment = self.get_attr(
            result,
            "employment",
        )

        salary = self.get_attr(
            result,
            "salary",
        )

        employer_name = self.get_attr(
            employment,
            "employer_name",
        )

        job_title = self.get_attr(
            employment,
            "job_title",
        )

        employment_date = (
            self.document_service
            .normalize_date(
                self.get_attr(
                    employment,
                    "employment_start_date",
                )
            )
        )

        gross_salary = self.safe_float(
            self.get_attr(
                salary,
                "gross_salary",
            )
        )

        # ----------------------------------------------------
        # NET SALARY
        #
        # Priority:
        # 1. Explicit extracted net salary
        # 2. 80% of gross salary if net is missing
        # ----------------------------------------------------

        raw_net_salary = self.get_attr(
            salary,
            "net_salary",
        )

        net_salary = self.safe_float(
            raw_net_salary
        )

        net_salary_is_calculated = False

        if (
            net_salary is None
            and gross_salary is not None
        ):

            net_salary = round(
                gross_salary * 0.80,
                2,
            )

            net_salary_is_calculated = True

        # ----------------------------------------------------
        # EMPLOYMENT TENURE
        # ----------------------------------------------------

        employment_tenure_years = None

        if employment_date:

            try:

                start_date = datetime.strptime(
                    employment_date,
                    "%Y-%m-%d",
                ).date()

                today = datetime.now(
                    timezone.utc
                ).date()

                days = (
                    today - start_date
                ).days

                if days >= 0:

                    employment_tenure_years = round(
                        days / 365.25,
                        1,
                    )

            except (
                ValueError,
                TypeError,
            ):

                employment_tenure_years = None

        issue_date = (
            self.document_service
            .normalize_date(
                self.get_attr(
                    result,
                    "certificate_date",
                )
            )
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        employer_confidence = self.get_confidence(
            result,
            "employer_name",
            aliases=[
                "employment_employer_name"
            ],
            parent=employment,
        )

        job_title_confidence = self.get_confidence(
            result,
            "job_title",
            parent=employment,
        )

        gross_confidence = self.get_confidence(
            result,
            "gross_salary",
            parent=salary,
        )

        net_confidence = self.get_confidence(
            result,
            "net_salary",
            parent=salary,
        )

        employment_date_confidence = self.get_confidence(
            result,
            "employment_start_date",
            aliases=[
                "employment_date"
            ],
            parent=employment,
        )

        issue_date_confidence = self.get_confidence(
            result,
            "certificate_date",
            aliases=[
                "issue_date"
            ],
        )

        # ----------------------------------------------------
        # Calculated net salary does not have OCR confidence.
        # ----------------------------------------------------

        if net_salary_is_calculated:
            net_confidence = 0.0

        return {

            "employer_name": self.make_field(
                employer_name,
                employer_confidence,
            ),

            "employer_sector": self.make_field(
                None,
                0.0,
            ),

            "job_title": self.make_field(
                job_title,
                job_title_confidence,
            ),

            "declared_net_salary": self.make_field(
                net_salary,
                net_confidence,
            ),

            "declared_gross_salary": self.make_field(
                gross_salary,
                gross_confidence,
            ),

            "employment_date": self.make_field(
                employment_date,
                employment_date_confidence,
            ),

            "employment_tenure_years": self.make_field(
                employment_tenure_years,
                0.0,
            ),

            "issue_date": self.make_field(
                issue_date,
                issue_date_confidence,
            ),
        }

    # ========================================================
    # BANK STATEMENT
    # ========================================================

    def normalize_bank(
        self,
        result,
    ) -> Dict[str, Any]:

        account_information = self.get_attr(
            result,
            "account_information",
        )

        account_number = self.get_attr(
            account_information,
            "account_number",
        )

        bank_name = self.get_attr(
            account_information,
            "bank_name",
        )

        features = (
            self.document_service
            .calculate_bank_features(
                result
            )
        )

        # ----------------------------------------------------
        # Resolve statement period.
        #
        # Priority:
        # 1. Explicit statement_period_months
        # 2. Explicit statement_months
        # 3. Calculated document feature
        # ----------------------------------------------------

        explicit_period = self.get_attr(
            result,
            "statement_period_months",
            None,
        )

        period_field_name = (
            "statement_period_months"
        )

        if explicit_period is None:

            explicit_period = self.get_attr(
                result,
                "statement_months",
                None,
            )

            period_field_name = (
                "statement_months"
            )

        explicit_period = self.safe_float(
            explicit_period
        )

        if explicit_period is not None:

            statement_period_months = float(
                explicit_period
            )

            statement_period_confidence = (
                self.get_confidence(
                    result,
                    period_field_name,
                    aliases=[
                        (
                            "statement_months"
                            if period_field_name
                            == "statement_period_months"
                            else
                            "statement_period_months"
                        )
                    ],
                )
            )

        else:

            statement_period_months = (
                self.safe_float(
                    features.get(
                        "statement_period_months"
                    )
                )
            )

            if statement_period_months is not None:

                statement_period_months = float(
                    statement_period_months
                )

            # Calculated from statement dates.
            statement_period_confidence = 0.0

        # ----------------------------------------------------
        # Derived bank metrics
        # ----------------------------------------------------

        avg_monthly_net_inflow = features.get(
            "avg_monthly_net_inflow"
        )

        avg_monthly_balance = features.get(
            "avg_monthly_balance"
        )

        min_monthly_balance = features.get(
            "min_monthly_balance"
        )

        max_monthly_balance = features.get(
            "max_monthly_balance"
        )

        balance_volatility_std = features.get(
            "balance_volatility_std"
        )

        overdraft_frequency = features.get(
            "overdraft_frequency"
        )

        returned_cheques_count = features.get(
            "returned_cheques_count"
        )

        income_regularity_score = features.get(
            "income_regularity_score"
        )

        return {

            "bank_name": self.make_field(
                bank_name,
                self.get_confidence(
                    result,
                    "bank_name",
                    parent=account_information,
                ),
            ),

            "account_number": self.make_field(
                account_number,
                self.get_confidence(
                    result,
                    "account_number",
                    parent=account_information,
                ),
            ),

            "statement_period_months": self.make_field(
                statement_period_months,
                statement_period_confidence,
            ),

            # These are APPLICATION-DERIVED.
            "avg_monthly_net_inflow": self.make_field(
                avg_monthly_net_inflow,
                0.0,
            ),

            "avg_monthly_balance": self.make_field(
                avg_monthly_balance,
                0.0,
            ),

            "min_monthly_balance": self.make_field(
                min_monthly_balance,
                0.0,
            ),

            "max_monthly_balance": self.make_field(
                max_monthly_balance,
                0.0,
            ),

            "balance_volatility_std": self.make_field(
                balance_volatility_std,
                0.0,
            ),

            "overdraft_frequency": self.make_field(
                overdraft_frequency,
                0.0,
            ),

            "returned_cheques_count": self.make_field(
                returned_cheques_count,
                0.0,
            ),

            "income_regularity_score": self.make_field(
                income_regularity_score,
                0.0,
            ),
        }

    # ========================================================
    # I-SCORE FACILITY
    # ========================================================

    def normalize_facility(
        self,
        facility,
    ) -> Dict[str, Any]:

        granted_amount = self.safe_float(
            self.get_attr(
                facility,
                "granted_amount",
            )
        )

        if granted_amount is None:

            granted_amount = self.safe_float(
                self.get_attr(
                    facility,
                    "original_amount",
                )
            )

        if granted_amount is None:

            granted_amount = self.safe_float(
                self.get_attr(
                    facility,
                    "credit_limit",
                )
            )

        outstanding_amount = self.safe_float(
            self.get_attr(
                facility,
                "outstanding_amount",
            )
        )

        if outstanding_amount is None:

            outstanding_amount = self.safe_float(
                self.get_attr(
                    facility,
                    "outstanding_balance",
                )
            )

        installment_amount = self.safe_float(
            self.get_attr(
                facility,
                "installment_amount",
            )
        )

        if installment_amount is None:

            installment_amount = self.safe_float(
                self.get_attr(
                    facility,
                    "monthly_payment",
                )
            )

        overdue_days = self.safe_int(
            self.get_attr(
                facility,
                "overdue_days",
            )
        )

        if overdue_days is None:

            overdue_days = self.safe_int(
                self.get_attr(
                    facility,
                    "days_past_due",
                )
            )

        return {

            "lender_name": self.get_attr(
                facility,
                "lender_name",
            ),

            "facility_type": self.get_attr(
                facility,
                "facility_type",
            ),

            "granted_amount": granted_amount,

            "outstanding_amount": outstanding_amount,

            "installment_amount": installment_amount,

            "status": self.get_attr(
                facility,
                "status",
            ),

            "overdue_days": overdue_days,

            "overdue_amount": self.safe_float(
                self.get_attr(
                    facility,
                    "overdue_amount",
                )
            ),

            "start_date": (
                self.document_service
                .normalize_date(
                    self.get_attr(
                        facility,
                        "start_date",
                    )
                )
            ),

            "legal_action_flag": self.get_attr(
                facility,
                "legal_action_flag",
            ),
        }

    # ========================================================
    # EXACT FACILITY DEDUPLICATION
    # ========================================================

    @staticmethod
    def deduplicate_facilities(
        facilities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        seen = set()
        output = []

        for facility in facilities:

            signature = (
                facility.get("lender_name"),
                facility.get("facility_type"),
                facility.get("granted_amount"),
                facility.get("outstanding_amount"),
                facility.get("installment_amount"),
                facility.get("status"),
                facility.get("overdue_days"),
                facility.get("overdue_amount"),
                facility.get("start_date"),
                facility.get("legal_action_flag"),
            )

            if signature in seen:
                continue

            seen.add(signature)
            output.append(facility)

        return output

    # ========================================================
    # SUSPICIOUS CROSS-ROW VALUE REUSE
    # ========================================================

    @staticmethod
    def remove_suspicious_facility_reuse(
        facilities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if len(facilities) < 3:
            return facilities

        tuple_counts: Dict[
            Tuple[Any, ...],
            int,
        ] = {}

        for facility in facilities:

            financial_tuple = (
                facility.get(
                    "granted_amount"
                ),
                facility.get(
                    "outstanding_amount"
                ),
                facility.get(
                    "installment_amount"
                ),
            )

            if all(
                value is not None
                for value in financial_tuple
            ):

                tuple_counts[
                    financial_tuple
                ] = (
                    tuple_counts.get(
                        financial_tuple,
                        0,
                    )
                    + 1
                )

        suspicious_tuples = {
            signature
            for signature, count
            in tuple_counts.items()
            if count >= 3
        }

        if not suspicious_tuples:
            return facilities

        cleaned = []

        for facility in facilities:

            financial_tuple = (
                facility.get(
                    "granted_amount"
                ),
                facility.get(
                    "outstanding_amount"
                ),
                facility.get(
                    "installment_amount"
                ),
            )

            if financial_tuple in suspicious_tuples:

                facility = dict(
                    facility
                )

                facility[
                    "granted_amount"
                ] = None

                facility[
                    "outstanding_amount"
                ] = None

                facility[
                    "installment_amount"
                ] = None

            cleaned.append(
                facility
            )

        return cleaned

    # ========================================================
    # I-SCORE
    # ========================================================

    def normalize_iscore(
        self,
        result,
    ) -> Dict[str, Any]:

        facilities = (
            self.get_attr(
                result,
                "facilities",
                [],
            )
            or []
        )

        normalized_facilities = [
            self.normalize_facility(
                facility
            )
            for facility in facilities
        ]

        normalized_facilities = (
            self.deduplicate_facilities(
                normalized_facilities
            )
        )

        normalized_facilities = (
            self.remove_suspicious_facility_reuse(
                normalized_facilities
            )
        )

        # ----------------------------------------------------
        # Top-level DPD
        # ----------------------------------------------------

        max_days_past_due = self.safe_int(
            self.get_attr(
                result,
                "max_days_past_due",
            )
        )

        if (
            max_days_past_due is None
            and normalized_facilities
        ):

            actual_dpds = [
                item["overdue_days"]
                for item in normalized_facilities
                if item["overdue_days"] is not None
            ]

            if actual_dpds:

                max_days_past_due = max(
                    actual_dpds
                )

        # ----------------------------------------------------
        # Active credit cards
        # ----------------------------------------------------

        active_credit_cards_count = self.safe_int(
            self.get_attr(
                result,
                "active_credit_cards_count",
            )
        )

        if active_credit_cards_count is None:

            active_credit_cards_count = sum(
                1
                for facility in normalized_facilities
                if (
                    str(
                        facility.get(
                            "facility_type"
                        )
                        or ""
                    )
                    .strip()
                    .lower()
                    in {
                        "credit card",
                        "credit cards",
                        "بطاقة ائتمانية",
                        "البطاقة الائتمانية",
                    }
                )
            )

        return {

            "is_available": True,

            "credit_score": self.make_field(
                self.safe_float(
                    self.get_attr(
                        result,
                        "credit_score",
                    )
                ),
                self.get_confidence(
                    result,
                    "credit_score",
                ),
            ),

            "score_tier": self.make_field(
                self.get_attr(
                    result,
                    "score_range",
                ),
                self.get_confidence(
                    result,
                    "score_range",
                    aliases=[
                        "score_tier"
                    ],
                ),
            ),

            "score_date": self.make_field(
                self.document_service
                .normalize_date(
                    self.get_attr(
                        result,
                        "report_date",
                    )
                ),
                self.get_confidence(
                    result,
                    "report_date",
                    aliases=[
                        "score_date"
                    ],
                ),
            ),

            "total_active_loans_limit":
                self.make_field(
                    self.safe_float(
                        self.get_attr(
                            result,
                            "total_active_loans_limit",
                        )
                    ),
                    self.get_confidence(
                        result,
                        "total_active_loans_limit",
                    ),
                ),

            "total_outstanding_balance":
                self.make_field(
                    self.safe_float(
                        self.get_attr(
                            result,
                            "total_outstanding_balance",
                        )
                    ),
                    self.get_confidence(
                        result,
                        "total_outstanding_balance",
                    ),
                ),

            "total_overdue_amount":
                self.make_field(
                    self.safe_float(
                        self.get_attr(
                            result,
                            "total_overdue_amount",
                        )
                    ),
                    self.get_confidence(
                        result,
                        "total_overdue_amount",
                    ),
                ),

            "max_days_past_due":
                self.make_field(
                    max_days_past_due,
                    self.get_confidence(
                        result,
                        "max_days_past_due",
                    ),
                ),

            "active_credit_cards_count":
                self.make_field(
                    active_credit_cards_count,
                    self.get_confidence(
                        result,
                        "active_credit_cards_count",
                    ),
                ),

            "total_credit_card_utilization":
                self.make_field(
                    self.safe_float(
                        self.get_attr(
                            result,
                            "total_credit_card_utilization",
                        )
                    ),
                    self.get_confidence(
                        result,
                        "total_credit_card_utilization",
                    ),
                ),

            "bureau_facilities":
                normalized_facilities,
        }

    # ========================================================
    # DOCUMENT OUTPUTS
    # ========================================================

    def build_document_outputs(
        self,
        national_id,
        salary_certificate,
        bank_statement,
        iscore,
    ) -> List[Dict[str, Any]]:

        # ====================================================
        # BANK FEATURES
        # ====================================================

        bank_features = (
            self.document_service
            .calculate_bank_features(
                bank_statement
            )
        )

        # ----------------------------------------------------
        # Resolve statement period ONCE.
        #
        # Priority:
        # 1. statement_period_months
        # 2. statement_months
        # 3. calculated statement_period_months
        # ----------------------------------------------------

        raw_statement_period = self.get_attr(
            bank_statement,
            "statement_period_months",
            None,
        )

        if raw_statement_period is None:

            raw_statement_period = self.get_attr(
                bank_statement,
                "statement_months",
                None,
            )

        if raw_statement_period is None:

            raw_statement_period = bank_features.get(
                "statement_period_months"
            )

        statement_period_months = self.safe_float(
            raw_statement_period
        )

        if statement_period_months is not None:

            statement_period_months = float(
                statement_period_months
            )

        # ====================================================
        # SALARY VALUES
        # ====================================================

        salary = self.get_attr(
            salary_certificate,
            "salary",
        )

        gross_salary = self.safe_float(
            self.get_attr(
                salary,
                "gross_salary",
            )
        )

        raw_net_salary = self.safe_float(
            self.get_attr(
                salary,
                "net_salary",
            )
        )

        # If Gemini did not extract net salary,
        # use the same fixed 80% calculation.
        if (
            raw_net_salary is None
            and gross_salary is not None
        ):

            net_salary = round(
                gross_salary * 0.80,
                2,
            )

        else:

            net_salary = raw_net_salary

        # ====================================================
        # EMPLOYMENT TENURE
        # ====================================================

        employment_date = (
            self.document_service
            .normalize_date(
                self.get_nested(
                    salary_certificate,
                    "employment.employment_start_date",
                )
            )
        )

        employment_tenure_years = None

        if employment_date:

            try:

                start_date = datetime.strptime(
                    employment_date,
                    "%Y-%m-%d",
                ).date()

                today = datetime.now(
                    timezone.utc
                ).date()

                days = (
                    today - start_date
                ).days

                if days >= 0:

                    employment_tenure_years = round(
                        days / 365.25,
                        1,
                    )

            except (
                ValueError,
                TypeError,
            ):

                employment_tenure_years = None

        # ====================================================
        # I-SCORE FACILITIES
        # ====================================================

        iscore_facilities = [
            self.normalize_facility(
                facility
            )
            for facility in (
                self.get_attr(
                    iscore,
                    "facilities",
                    [],
                )
                or []
            )
        ]

        iscore_facilities = (
            self.deduplicate_facilities(
                iscore_facilities
            )
        )

        iscore_facilities = (
            self.remove_suspicious_facility_reuse(
                iscore_facilities
            )
        )

        # ====================================================
        # DOCUMENT OUTPUTS
        # ====================================================

        return [

            # =================================================
            # NATIONAL ID
            # =================================================

            {
                "document_type":
                    "national_id",

                "overall_quality_score":
                    self.get_document_quality(
                        national_id
                    ),

                "is_tampered_suspected":
                    self.get_tampering_flag(
                        national_id
                    ),

                "extracted_fields": {

                    "full_name":
                        self.get_attr(
                            national_id,
                            "full_name_ar",
                        ),

                    "national_id":
                        self.get_attr(
                            national_id,
                            "national_id",
                        ),

                    "date_of_birth":
                        self.get_attr(
                            national_id,
                            "date_of_birth",
                        ),

                    "governorate":
                        self.get_attr(
                            national_id,
                            "governorate",
                        ),

                    "address":
                        self.get_attr(
                            national_id,
                            "address_ar",
                        ),

                    "profession_on_id":
                        self.get_attr(
                            national_id,
                            "profession",
                        ),

                    "marital_status_on_id":
                        self.get_attr(
                            national_id,
                            "marital_status",
                        ),

                    "id_expiry_date":
                        self.get_attr(
                            national_id,
                            "id_expiry_date",
                        ),
                },
            },

            # =================================================
            # SALARY CERTIFICATE
            # =================================================

            {
                "document_type":
                    "salary_certificate",

                "overall_quality_score":
                    self.get_document_quality(
                        salary_certificate
                    ),

                "is_tampered_suspected":
                    self.get_tampering_flag(
                        salary_certificate
                    ),

                "extracted_fields": {

                    "employer_name":
                        self.get_nested(
                            salary_certificate,
                            "employment.employer_name",
                        ),

                    "employer_sector":
                        None,

                    "job_title":
                        self.get_nested(
                            salary_certificate,
                            "employment.job_title",
                        ),

                    "declared_net_salary":
                        net_salary,

                    "declared_gross_salary":
                        gross_salary,

                    "employment_date":
                        employment_date,

                    "employment_tenure_years":
                        employment_tenure_years,

                    "issue_date":
                        self.document_service
                        .normalize_date(
                            self.get_attr(
                                salary_certificate,
                                "certificate_date",
                            )
                        ),
                },
            },

            # =================================================
            # BANK STATEMENT
            # =================================================

            {
                "document_type":
                    "bank_statement",

                "overall_quality_score":
                    self.get_document_quality(
                        bank_statement
                    ),

                "is_tampered_suspected":
                    self.get_tampering_flag(
                        bank_statement
                    ),

                "extracted_fields": {

                    "bank_name":
                        self.get_nested(
                            bank_statement,
                            "account_information.bank_name",
                        ),

                    "account_number":
                        self.get_nested(
                            bank_statement,
                            "account_information.account_number",
                        ),

                    "statement_period_months":
                        statement_period_months,

                    "avg_monthly_net_inflow":
                        bank_features.get(
                            "avg_monthly_net_inflow"
                        ),

                    "avg_monthly_balance":
                        bank_features.get(
                            "avg_monthly_balance"
                        ),

                    "min_monthly_balance":
                        bank_features.get(
                            "min_monthly_balance"
                        ),

                    "max_monthly_balance":
                        bank_features.get(
                            "max_monthly_balance"
                        ),

                    "balance_volatility_std":
                        bank_features.get(
                            "balance_volatility_std"
                        ),

                    "overdraft_frequency":
                        bank_features.get(
                            "overdraft_frequency"
                        ),

                    "returned_cheques_count":
                        bank_features.get(
                            "returned_cheques_count"
                        ),

                    "income_regularity_score":
                        bank_features.get(
                            "income_regularity_score"
                        ),
                },
            },

            # =================================================
            # I-SCORE
            # =================================================

            {
                "document_type":
                    "iscore",

                "overall_quality_score":
                    self.get_document_quality(
                        iscore
                    ),

                "is_tampered_suspected":
                    self.get_tampering_flag(
                        iscore
                    ),

                "extracted_fields": {

                    "credit_score":
                        self.get_attr(
                            iscore,
                            "credit_score",
                        ),

                    "score_range":
                        self.get_attr(
                            iscore,
                            "score_range",
                        ),

                    "report_date":
                        self.get_attr(
                            iscore,
                            "report_date",
                        ),

                    "total_credit_limit":
                        self.get_attr(
                            iscore,
                            "total_credit_limit",
                        ),

                    "total_outstanding_balance":
                        self.get_attr(
                            iscore,
                            "total_outstanding_balance",
                        ),

                    "total_overdue_amount":
                        self.get_attr(
                            iscore,
                            "total_overdue_amount",
                        ),

                    "max_days_past_due":
                        self.get_attr(
                            iscore,
                            "max_days_past_due",
                        ),

                    "active_credit_cards_count":
                        self.get_attr(
                            iscore,
                            "active_credit_cards_count",
                        ),

                    "total_credit_card_utilization":
                        self.get_attr(
                            iscore,
                            "total_credit_card_utilization",
                        ),

                    "facilities":
                        iscore_facilities,
                },
            },
        ]

    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(
        self,
        application_id: str,
        national_id,
        salary_certificate,
        bank_statement,
        iscore,
    ) -> CanonicalContract:

        national_id_fields = (
            self.normalize_national_id(
                national_id
            )
        )

        salary_fields = (
            self.normalize_salary(
                salary_certificate
            )
        )

        bank_fields = (
            self.normalize_bank(
                bank_statement
            )
        )

        iscore_fields = (
            self.normalize_iscore(
                iscore
            )
        )

        documents = (
            self.build_document_outputs(
                national_id,
                salary_certificate,
                bank_statement,
                iscore,
            )
        )

        return CanonicalContract(

            application_id=application_id,

            applicant_type="individual",

            submission_timestamp=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),

            is_returning_customer=None,

            documents=documents,

            national_id_fields=
                national_id_fields,

            salary_certificate_fields=
                salary_fields,

            bank_statement_fields=
                bank_fields,

            iscore_report_fields=
                iscore_fields,

            form_data={},

            internal_history={},

            consistency_checks={},
        )

    # ========================================================
    # ROUTE-COMPATIBLE PROCESS DOCUMENTS
    # ========================================================

    def process_documents(
        self,
        application_id: str,

        national_id_front: Tuple[
            bytes,
            str,
        ],

        national_id_back: Tuple[
            bytes,
            str,
        ],

        salary_certificate: Tuple[
            bytes,
            str,
        ],

        bank_statement: Tuple[
            bytes,
            str,
        ],

        iscore: Tuple[
            bytes,
            str,
        ],
    ) -> CanonicalContract:

        national_id_result = (
            self.document_service
            .extract_national_id(
                front_bytes=
                    national_id_front[0],

                front_mime_type=
                    national_id_front[1],

                back_bytes=
                    national_id_back[0],

                back_mime_type=
                    national_id_back[1],
            )
        )

        salary_result = (
            self.document_service
            .extract_salary_certificate(
                file_bytes=
                    salary_certificate[0],

                mime_type=
                    salary_certificate[1],
            )
        )

        bank_result = (
            self.document_service
            .extract_bank_statement(
                file_bytes=
                    bank_statement[0],

                mime_type=
                    bank_statement[1],
            )
        )

        iscore_result = (
            self.document_service
            .extract_iscore(
                file_bytes=
                    iscore[0],

                mime_type=
                    iscore[1],
            )
        )

        return self.process(
            application_id=
                application_id,

            national_id=
                national_id_result,

            salary_certificate=
                salary_result,

            bank_statement=
                bank_result,

            iscore=
                iscore_result,
        )