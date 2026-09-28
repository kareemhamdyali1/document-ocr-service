# ============================================================
# app/services/document_service.py
# ============================================================

import re

from collections import defaultdict

from datetime import date, datetime

from typing import List, Optional, Tuple


from app.prompts.prompts import (
    NATIONAL_ID_PROMPT,
    SALARY_CERTIFICATE_PROMPT,
    BANK_STATEMENT_PROMPT,
    ISCORE_PROMPT,
)

from app.schemas.documents import (
    NationalIDExtraction,
    SalaryCertificateExtraction,
    BankStatementExtraction,
    IScoreExtraction,
)

from app.services.gemini_service import GeminiService


# ============================================================
# ARABIC DIGIT NORMALIZATION
# ============================================================

ARABIC_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


# ============================================================
# EGYPTIAN GOVERNORATE CODES
# ============================================================

EGYPT_GOVERNORATES = {
    "01": "Cairo",
    "02": "Alexandria",
    "03": "Port Said",
    "04": "Suez",
    "11": "Damietta",
    "12": "Dakahlia",
    "13": "Ash Sharqia",
    "14": "Qalyubia",
    "15": "Kafr El Sheikh",
    "16": "Gharbia",
    "17": "Monufia",
    "18": "Beheira",
    "19": "Ismailia",
    "21": "Giza",
    "22": "Beni Suef",
    "23": "Faiyum",
    "24": "Minya",
    "25": "Asyut",
    "26": "Sohag",
    "27": "Qena",
    "28": "Aswan",
    "29": "Luxor",
    "31": "Red Sea",
    "32": "New Valley",
    "33": "Matrouh",
    "34": "North Sinai",
    "35": "South Sinai",
}


# ============================================================
# TRANSACTION KEYWORDS
# ============================================================

INCOME_KEYWORDS = [
    "salary",
    "payroll",
    "wages",
    "income",
    "monthly salary",
    "salary transfer",
    "payroll credit",
    "salary payment",
    "credit salary",
    "راتب",
    "مرتب",
    "المرتب",
    "الراتب",
    "مرتب شهر",
    "تحويل راتب",
    "تحويل مرتب",
    "إيداع راتب",
    "ايداع راتب",
    "راتب شهر",
]


NON_INCOME_KEYWORDS = [
    "fee",
    "fees",
    "charge",
    "commission",
    "service charge",
    "bank charge",
    "monthly fee",
    "atm fee",
    "interest",
    "refund",
    "reversal",
    "cash deposit",
    "رسوم",
    "عمولة",
    "مصروفات",
    "فوائد",
    "رد",
    "استرداد",
]


# ============================================================
# DOCUMENT SERVICE
# ============================================================

class DocumentService:

    def __init__(self):

        self.gemini = GeminiService()

    # ========================================================
    # BASIC NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_digits(
        value: Optional[str],
    ) -> Optional[str]:

        if value is None:
            return None

        return str(value).translate(
            ARABIC_DIGIT_MAP
        )

    # ========================================================
    # NATIONAL ID NORMALIZATION
    # ========================================================

    def normalize_national_id(
        self,
        national_id: Optional[str],
    ) -> Optional[str]:

        if not national_id:
            return None

        normalized = self.normalize_digits(
            national_id
        )

        normalized = re.sub(
            r"[^0-9]",
            "",
            normalized,
        )

        return normalized

    # ========================================================
    # DATE NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_date(
        value: Optional[str],
    ) -> Optional[str]:

        if not value:
            return None

        value = str(value).strip()

        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d.%m.%Y",
            "%d-%b-%Y",
            "%d-%B-%Y",
        ]

        for fmt in formats:

            try:

                return datetime.strptime(
                    value,
                    fmt,
                ).strftime("%Y-%m-%d")

            except ValueError:

                continue

        return value

    # ========================================================
    # CALCULATE AGE
    # ========================================================

    @staticmethod
    def calculate_age(
        birth_date: Optional[str],
    ) -> Optional[float]:

        if not birth_date:
            return None

        normalized = (
            DocumentService.normalize_date(
                birth_date
            )
        )

        if not normalized:
            return None

        try:

            dob = datetime.strptime(
                normalized,
                "%Y-%m-%d",
            ).date()

            today = date.today()

            days = (
                today - dob
            ).days

            return round(
                days / 365.25,
                1,
            )

        except Exception:

            return None

    # ========================================================
    # NATIONAL ID VALIDATION
    # ========================================================

    def validate_national_id(
        self,
        national_id: Optional[str],
    ) -> Tuple[bool, List[str]]:

        errors = []

        if not national_id:

            errors.append(
                "National ID is missing."
            )

            return False, errors

        if len(national_id) != 14:

            errors.append(
                "National ID must contain exactly 14 digits."
            )

        if not national_id.isdigit():

            errors.append(
                "National ID must contain digits only."
            )

        if national_id and national_id[0] not in {
            "2",
            "3",
        }:

            errors.append(
                "Invalid Egyptian National ID century digit."
            )

        if len(national_id) >= 9:

            governorate_code = national_id[7:9]

            if governorate_code not in EGYPT_GOVERNORATES:

                errors.append(
                    f"Invalid governorate code: "
                    f"{governorate_code}"
                )

        if len(national_id) == 14:

            try:

                century = (
                    1900
                    if national_id[0] == "2"
                    else 2000
                )

                year = (
                    century
                    + int(national_id[1:3])
                )

                month = int(
                    national_id[3:5]
                )

                day = int(
                    national_id[5:7]
                )

                birth_date = date(
                    year,
                    month,
                    day,
                )

                if birth_date > date.today():

                    errors.append(
                        "Birth date is in the future."
                    )

            except ValueError:

                errors.append(
                    "Invalid birth date encoded in National ID."
                )

        return (
            len(errors) == 0,
            errors,
        )

    # ========================================================
    # NATIONAL ID
    # ========================================================

    def extract_national_id(
        self,
        front_bytes: bytes,
        front_mime_type: str,
        back_bytes: bytes,
        back_mime_type: str,
    ) -> NationalIDExtraction:

        result = self.gemini.extract_multiple(
            files=[
                (
                    front_bytes,
                    front_mime_type,
                ),
                (
                    back_bytes,
                    back_mime_type,
                ),
            ],
            prompt=NATIONAL_ID_PROMPT,
            response_schema=NationalIDExtraction,
        )

        result.national_id = (
            self.normalize_national_id(
                result.national_id
            )
        )

        result.date_of_birth = (
            self.normalize_date(
                result.date_of_birth
            )
        )

        result.id_issue_date = (
            self.normalize_date(
                result.id_issue_date
            )
        )

        result.id_expiry_date = (
            self.normalize_date(
                result.id_expiry_date
            )
        )

        validation_passed, errors = (
            self.validate_national_id(
                result.national_id
            )
        )

        result.validation_passed = (
            validation_passed
        )

        result.validation_errors = errors

        result.front_image_processed = True
        result.back_image_processed = True

        result.cross_validation_passed = (
            validation_passed
        )

        if result.date_of_birth:

            result.age_years = (
                self.calculate_age(
                    result.date_of_birth
                )
            )

        if result.national_id:

            governorate_code = (
                result.national_id[7:9]
            )

            result.governorate_code = (
                governorate_code
            )

            if governorate_code in EGYPT_GOVERNORATES:

                result.governorate = (
                    EGYPT_GOVERNORATES[
                        governorate_code
                    ]
                )

        if (
            result.national_id
            and len(result.national_id) == 14
        ):

            gender_digit = int(
                result.national_id[12]
            )

            result.gender = (
                "female"
                if gender_digit % 2 == 0
                else "male"
            )

        return result

    # ========================================================
    # SALARY CERTIFICATE
    # ========================================================

    def extract_salary_certificate(
        self,
        file_bytes: bytes,
        mime_type: str,
    ) -> SalaryCertificateExtraction:

        result = self.gemini.extract(
            file_bytes=file_bytes,
            mime_type=mime_type,
            prompt=SALARY_CERTIFICATE_PROMPT,
            response_schema=SalaryCertificateExtraction,
        )

        result.employment.employment_start_date = (
            self.normalize_date(
                result.employment.employment_start_date
            )
        )

        result.certificate_date = (
            self.normalize_date(
                result.certificate_date
            )
        )

        # IMPORTANT:
        # Do NOT estimate net salary from gross salary.
        #
        # If net salary is not explicitly visible,
        # it remains None.

        return result

    # ========================================================
    # BANK STATEMENT
    # ========================================================

    def extract_bank_statement(
        self,
        file_bytes: bytes,
        mime_type: str,
    ) -> BankStatementExtraction:

        result = self.gemini.extract(
            file_bytes=file_bytes,
            mime_type=mime_type,
            prompt=BANK_STATEMENT_PROMPT,
            response_schema=BankStatementExtraction,
        )

        for transaction in result.transactions:

            transaction.date = (
                self.normalize_date(
                    transaction.date
                )
            )

        result.account_information.statement_period.from_date = (
            self.normalize_date(
                result.account_information
                .statement_period
                .from_date
            )
        )

        result.account_information.statement_period.to_date = (
            self.normalize_date(
                result.account_information
                .statement_period
                .to_date
            )
        )

        if result.account_information.bank_name:

            result.account_information.bank_name = (
                result.account_information.bank_name.strip()
            )

        # ----------------------------------------------------
        # Normalize explicit month values to FLOAT
        # ----------------------------------------------------

        if result.statement_months is not None:

            try:

                result.statement_months = float(
                    result.statement_months
                )

            except (
                TypeError,
                ValueError,
            ):

                result.statement_months = None

        if result.statement_period_months is not None:

            try:

                result.statement_period_months = float(
                    result.statement_period_months
                )

            except (
                TypeError,
                ValueError,
            ):

                result.statement_period_months = None

        return result

    # ========================================================
    # I-SCORE
    # ========================================================

    def extract_iscore(
        self,
        file_bytes: bytes,
        mime_type: str,
    ) -> IScoreExtraction:

        result = self.gemini.extract(
            file_bytes=file_bytes,
            mime_type=mime_type,
            prompt=ISCORE_PROMPT,
            response_schema=IScoreExtraction,
        )

        result.report_date = (
            self.normalize_date(
                result.report_date
            )
        )

        # ----------------------------------------------------
        # Normalize facility dates
        # ----------------------------------------------------

        for facility in result.facilities:

            facility.start_date = (
                self.normalize_date(
                    facility.start_date
                )
            )

            facility.maturity_date = (
                self.normalize_date(
                    facility.maturity_date
                )
            )

        # ----------------------------------------------------
        # Normalize facility aliases
        # ----------------------------------------------------

        for facility in result.facilities:

            if (
                facility.granted_amount is None
                and facility.original_amount is not None
            ):

                facility.granted_amount = (
                    facility.original_amount
                )

            if (
                facility.outstanding_amount is None
                and facility.outstanding_balance is not None
            ):

                facility.outstanding_amount = (
                    facility.outstanding_balance
                )

            if (
                facility.installment_amount is None
                and facility.monthly_payment is not None
            ):

                facility.installment_amount = (
                    facility.monthly_payment
                )

            if (
                facility.overdue_days is None
                and facility.days_past_due is not None
            ):

                facility.overdue_days = (
                    facility.days_past_due
                )

        # ----------------------------------------------------
        # Remove only exact duplicate rows
        # ----------------------------------------------------

        result.facilities = (
            self.dedupe_facilities(
                result.facilities
            )
        )

        return result

    # ========================================================
    # TRANSACTION TEXT
    # ========================================================

    @staticmethod
    def _transaction_text(
        transaction,
    ) -> str:

        return (
            str(
                transaction.transaction
                or ""
            )
            .strip()
            .lower()
        )

    # ========================================================
    # POSITIVE CREDIT
    # ========================================================

    @staticmethod
    def _is_positive_credit(
        transaction,
    ) -> bool:

        credit = transaction.credit

        if credit is None:
            return False

        try:

            return float(credit) > 0

        except (
            TypeError,
            ValueError,
        ):

            return False

    # ========================================================
    # EXPLICIT INCOME
    # ========================================================

    @staticmethod
    def _is_explicit_income(
        transaction,
    ) -> bool:

        text = (
            DocumentService
            ._transaction_text(
                transaction
            )
        )

        if not text:
            return False

        return any(
            keyword in text
            for keyword in INCOME_KEYWORDS
        )

    # ========================================================
    # NON-INCOME CREDIT
    # ========================================================

    @staticmethod
    def _is_non_income_credit(
        transaction,
    ) -> bool:

        text = (
            DocumentService
            ._transaction_text(
                transaction
            )
        )

        if not text:
            return False

        return any(
            keyword in text
            for keyword in NON_INCOME_KEYWORDS
        )

    # ========================================================
    # MONTHLY INCOMING CREDITS
    # ========================================================

    def calculate_monthly_incoming_credits(
        self,
        transactions,
    ):

        monthly_totals = defaultdict(float)

        all_credit_transactions = []

        for transaction in transactions:

            if not self._is_positive_credit(
                transaction
            ):
                continue

            if self._is_non_income_credit(
                transaction
            ):
                continue

            if not transaction.date:
                continue

            try:

                month_key = (
                    str(transaction.date)[:7]
                )

                amount = float(
                    transaction.credit
                )

                monthly_totals[
                    month_key
                ] += amount

                all_credit_transactions.append(
                    transaction
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

        return (
            monthly_totals,
            all_credit_transactions,
        )

    # ========================================================
    # AVG MONTHLY NET INFLOW
    # ========================================================

    def calculate_avg_monthly_net_inflow(
        self,
        transactions,
    ) -> Optional[float]:

        monthly_totals, credits = (
            self.calculate_monthly_incoming_credits(
                transactions
            )
        )

        if not monthly_totals:
            return None

        salary_monthly = defaultdict(float)

        for transaction in credits:

            if self._is_explicit_income(
                transaction
            ):

                try:

                    month_key = (
                        str(transaction.date)[:7]
                    )

                    salary_monthly[
                        month_key
                    ] += float(
                        transaction.credit
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

        if salary_monthly:

            monthly_values = list(
                salary_monthly.values()
            )

        else:

            monthly_values = list(
                monthly_totals.values()
            )

        if not monthly_values:
            return None

        return round(
            sum(monthly_values)
            / len(monthly_values),
            2,
        )

    # ========================================================
    # INCOME REGULARITY
    # ========================================================

    def calculate_income_regularity(
        self,
        transactions,
    ) -> Optional[float]:

        monthly_totals, _ = (
            self.calculate_monthly_incoming_credits(
                transactions
            )
        )

        if len(monthly_totals) < 2:
            return None

        values = list(
            monthly_totals.values()
        )

        average = (
            sum(values)
            / len(values)
        )

        if average <= 0:
            return None

        deviations = [
            abs(value - average)
            / average
            for value in values
        ]

        instability = (
            sum(deviations)
            / len(deviations)
        )

        score = max(
            0.0,
            min(
                1.0,
                1.0 - instability,
            ),
        )

        return round(
            score,
            4,
        )

    # ========================================================
    # I-SCORE DEDUPLICATION
    # ========================================================

    @staticmethod
    def dedupe_facilities(
        facilities,
    ):

        seen = set()

        deduped = []

        for facility in facilities:

            # ------------------------------------------------
            # Prefer a strong identity
            # ------------------------------------------------

            account_number = (
                str(
                    facility.account_number
                    or ""
                )
                .strip()
                .lower()
            )

            lender = (
                str(
                    facility.lender_name
                    or ""
                )
                .strip()
                .lower()
            )

            facility_type = (
                str(
                    facility.facility_type
                    or ""
                )
                .strip()
                .lower()
            )

            start_date = (
                str(
                    facility.start_date
                    or ""
                )
                .strip()
            )

            maturity_date = (
                str(
                    facility.maturity_date
                    or ""
                )
                .strip()
            )

            # ------------------------------------------------
            # Exact row signature
            # ------------------------------------------------

            signature = (
                lender,
                facility_type,
                account_number,
                str(
                    facility.granted_amount
                    if facility.granted_amount is not None
                    else facility.original_amount
                    if facility.original_amount is not None
                    else facility.credit_limit
                    if facility.credit_limit is not None
                    else ""
                ),
                str(
                    facility.outstanding_amount
                    if facility.outstanding_amount is not None
                    else facility.outstanding_balance
                    if facility.outstanding_balance is not None
                    else ""
                ),
                str(
                    facility.installment_amount
                    if facility.installment_amount is not None
                    else facility.monthly_payment
                    if facility.monthly_payment is not None
                    else ""
                ),
                str(
                    facility.overdue_amount
                    or ""
                ),
                str(
                    facility.overdue_days
                    if facility.overdue_days is not None
                    else facility.days_past_due
                    if facility.days_past_due is not None
                    else ""
                ),
                start_date,
                maturity_date,
                str(
                    facility.status
                    or ""
                )
                .strip()
                .lower(),
                str(
                    facility.legal_action_flag
                    if facility.legal_action_flag is not None
                    else ""
                ),
            )

            if signature in seen:
                continue

            seen.add(signature)

            deduped.append(
                facility
            )

        return deduped

    # ========================================================
    # BANK FEATURES
    # ========================================================

    def calculate_bank_features(
        self,
        result: BankStatementExtraction,
    ) -> dict:

        transactions = (
            result.transactions
        )

        avg_monthly_net_inflow = (
            self.calculate_avg_monthly_net_inflow(
                transactions
            )
        )

        income_regularity_score = (
            self.calculate_income_regularity(
                transactions
            )
        )

        balances = []

        for transaction in transactions:

            if transaction.balance is not None:

                try:

                    balances.append(
                        float(
                            transaction.balance
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    pass

        avg_monthly_balance = None
        min_monthly_balance = None
        max_monthly_balance = None
        balance_std = None

        if balances:

            avg_monthly_balance = round(
                sum(balances)
                / len(balances),
                2,
            )

            min_monthly_balance = round(
                min(balances),
                2,
            )

            max_monthly_balance = round(
                max(balances),
                2,
            )

            if len(balances) > 1:

                mean = (
                    sum(balances)
                    / len(balances)
                )

                variance = (
                    sum(
                        (
                            value - mean
                        ) ** 2
                        for value in balances
                    )
                    / len(balances)
                )

                balance_std = round(
                    variance ** 0.5,
                    6,
                )

        # ----------------------------------------------------
        # Overdraft
        # ----------------------------------------------------

        overdraft_frequency = 0

        for transaction in transactions:

            if (
                transaction.balance is not None
                and transaction.balance < 0
            ):

                overdraft_frequency += 1

        # ----------------------------------------------------
        # Returned cheques
        # ----------------------------------------------------

        returned_cheques_count = 0

        for transaction in transactions:

            text = (
                self._transaction_text(
                    transaction
                )
            )

            if (
                "returned cheque" in text
                or "returned check" in text
                or "شيك مرتجع" in text
                or "شيك مرفوض" in text
            ):

                returned_cheques_count += 1

        # ----------------------------------------------------
        # Statement period
        # ----------------------------------------------------

        statement_period_months = (
            result.statement_period_months
        )

        if statement_period_months is None:

            statement_period_months = (
                result.statement_months
            )

        if statement_period_months is None:

            statement_period_months = (
                self.calculate_statement_months(
                    result.account_information
                    .statement_period
                )
            )

        if statement_period_months is not None:

            try:

                statement_period_months = float(
                    statement_period_months
                )

            except (
                TypeError,
                ValueError,
            ):

                statement_period_months = None

        return {

            "statement_period_months":
                statement_period_months,

            "avg_monthly_net_inflow":
                avg_monthly_net_inflow,

            "avg_monthly_balance":
                avg_monthly_balance,

            "min_monthly_balance":
                min_monthly_balance,

            "max_monthly_balance":
                max_monthly_balance,

            "balance_volatility_std":
                balance_std,

            "overdraft_frequency":
                overdraft_frequency,

            "returned_cheques_count":
                returned_cheques_count,

            "income_regularity_score":
                income_regularity_score,
        }

    # ========================================================
    # STATEMENT MONTHS
    # ========================================================

    @staticmethod
    def calculate_statement_months(
        statement_period,
    ) -> Optional[float]:

        start = (
            statement_period.from_date
        )

        end = (
            statement_period.to_date
        )

        if not start or not end:
            return None

        try:

            start_dt = datetime.strptime(
                start,
                "%Y-%m-%d",
            )

            end_dt = datetime.strptime(
                end,
                "%Y-%m-%d",
            )

            months = (
                (
                    end_dt.year
                    - start_dt.year
                )
                * 12
                + (
                    end_dt.month
                    - start_dt.month
                )
            )

            return float(
                max(
                    months,
                    1,
                )
            )

        except Exception:

            return None