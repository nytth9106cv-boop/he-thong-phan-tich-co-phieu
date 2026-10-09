"""
financial_ratios.py
TV3 - Fundamental Analysis

Lấy dữ liệu tài chính từ Vnstock, chuẩn hóa BCTC và tính/chuẩn hóa
các chỉ số tài chính cho doanh nghiệp phi tài chính và ngân hàng.
"""

import math
import re
import unicodedata
import numpy as np
import pandas as pd
from vnstock import Fundamental



BANK_RATIO_IDS = {
    "loan_growth": "deposits_from_customers",
    "deposit_growth": "deposits_from_customers_2",
    "profit_growth": "profit_after_tax_for_shareholders_of_the_parent_company",
    "roe": "roe",
    "roa": "roa",
    "nim": "net_interest_margin_nim",
    "ldr": "outstanding_loans_customer_deposits",
    "eps": "trailing_eps",
    "pe": "pe_ratio",
    "pb": "pb_ratio",
    # Vnstock ratio table tested with TCB does not expose these directly.
    "npl": "npl_ratio",
    "llr": "loan_loss_coverage",
    "car": "capital_adequacy_ratio",
    "casa": "casa_ratio",
}


def get_financial_data(ticker, period="year"):

    ticker = ticker.upper().strip()

    try:

        fundamental = Fundamental()

        company = fundamental.equity(
            ticker
        )

        income = company.income_statement(
            period=period
        )

        balance = company.balance_sheet(
            period=period
        )

        cashflow = company.cash_flow(
            period=period
        )

        ratios = company.ratio(
            period=period
        )

        return {
            "ticker": ticker,
            "income": income,
            "balance": balance,
            "cashflow": cashflow,
            "ratios": ratios
        }

    except Exception as error:

        raise RuntimeError(
            f"Không thể lấy dữ liệu tài chính "
            f"của {ticker}: {error}"
        )


def get_period_columns(df):

    metadata_columns = {
        "item",
        "item_en",
        "item_id",
        "unit",
        "levels",
        "level",
        "row_number"
    }

    return [
        column
        for column in df.columns
        if column not in metadata_columns
    ]


def extract_item(df, item_id):

    if df is None or df.empty:

        return {}

    if "item_id" not in df.columns:

        return {}

    rows = df[
        df["item_id"] == item_id
    ].copy()

    if rows.empty:

        return {}

    periods = get_period_columns(
        df
    )

    if not periods:

        return {}

    numeric_values = rows[
        periods
    ].apply(
        lambda column:
        pd.to_numeric(
            column,
            errors="coerce"
        )
    )

    rows["_valid_count"] = (
        numeric_values
        .notna()
        .sum(axis=1)
    )

    best_index = rows[
        "_valid_count"
    ].idxmax()

    row = rows.loc[
        best_index
    ]

    result = {}

    for period in periods:

        result[str(period)] = (
            pd.to_numeric(
                row[period],
                errors="coerce"
            )
        )

    return result


def period_sort_key(value):

    text = str(value)

    try:

        return int(
            text[:4]
        )

    except Exception:

        return text


def reverse_period_values(data):

    """
    Ví dụ:

    trước:
    2022 = 70
    2023 = 62
    2024 = 52
    2025 = 44

    sau:
    2022 = 44
    2023 = 52
    2024 = 62
    2025 = 70

    Chỉ đảo VALUE, không đổi tên kỳ.
    """

    if not data:

        return data

    periods = sorted(
        data.keys(),
        key=period_sort_key
    )

    values = [
        data[p]
        for p in periods
    ]

    values.reverse()

    return dict(
        zip(
            periods,
            values
        )
    )


def calculate_series_growth(data):

    if not data:

        return {}

    periods = sorted(
        data.keys(),
        key=period_sort_key
    )

    result = {}

    previous_value = None

    for period in periods:

        current_value = data.get(
            period,
            np.nan
        )

        if (
            previous_value is None
            or pd.isna(previous_value)
            or pd.isna(current_value)
            or previous_value == 0
        ):

            result[period] = np.nan

        else:

            result[period] = (
                current_value
                / previous_value
                - 1
            )

        previous_value = (
            current_value
        )

    return result


def growth_match_error(
    statement_growth,
    reference_growth
):

    """
    Tính sai số trung bình tuyệt đối giữa:

    - growth tính từ BCTC
    - growth do Vnstock cung cấp

    Reference growth của Vnstock đang ở dạng %,
    ví dụ 19.56 -> chuyển thành 0.1956.
    """

    errors = []

    common_periods = (
        set(statement_growth.keys())
        &
        set(reference_growth.keys())
    )

    for period in common_periods:

        statement_value = (
            statement_growth.get(
                period
            )
        )

        reference_value = (
            reference_growth.get(
                period
            )
        )

        if (
            pd.isna(statement_value)
            or pd.isna(reference_value)
        ):

            continue

        reference_decimal = (
            reference_value / 100
        )

        errors.append(
            abs(
                statement_value
                - reference_decimal
            )
        )

    if not errors:

        return np.inf

    return float(
        np.mean(errors)
    )


def detect_statement_period_mapping(
    revenue,
    revenue_growth_vnstock,
    tolerance=0.03
):

    """
    So sánh 2 trường hợp:

    A. Giữ nguyên BCTC
    B. Đảo giá trị BCTC theo kỳ

    Nếu trường hợp B giảm sai số đáng kể và
    khớp reference growth tốt thì mới đảo.

    Không hard-code rằng Vnstock luôn bị đảo.
    """

    if (
        not revenue
        or not revenue_growth_vnstock
    ):

        return {
            "reverse_statements": False,
            "original_error": np.nan,
            "reversed_error": np.nan,
            "reason":
                "Không đủ dữ liệu để kiểm tra mapping kỳ."
        }

    # Growth khi giữ nguyên
    original_growth = (
        calculate_series_growth(
            revenue
        )
    )

    original_error = (
        growth_match_error(
            original_growth,
            revenue_growth_vnstock
        )
    )

    # Growth khi đảo
    reversed_revenue = (
        reverse_period_values(
            revenue
        )
    )

    reversed_growth = (
        calculate_series_growth(
            reversed_revenue
        )
    )

    reversed_error = (
        growth_match_error(
            reversed_growth,
            revenue_growth_vnstock
        )
    )

    # Chỉ đảo nếu:
    # 1. reversed_error nhỏ hơn tolerance
    # 2. reversed_error tốt hơn original rõ rệt
    should_reverse = (
        np.isfinite(reversed_error)
        and
        reversed_error <= tolerance
        and
        (
            not np.isfinite(original_error)
            or
            reversed_error
            < original_error * 0.5
        )
    )

    if should_reverse:

        reason = (
            "Phát hiện giá trị BCTC có khả năng "
            "bị đảo theo kỳ; hệ thống đã tự "
            "chuẩn hóa lại dựa trên đối chiếu "
            "tăng trưởng doanh thu."
        )

    else:

        reason = (
            "Không phát hiện bằng chứng đủ mạnh "
            "để đảo mapping kỳ BCTC."
        )

    return {
        "reverse_statements":
            should_reverse,

        "original_error":
            original_error,

        "reversed_error":
            reversed_error,

        "reason":
            reason
    }


def create_financial_summary(
    financial_data
):

    ticker = financial_data[
        "ticker"
    ]

    income = financial_data[
        "income"
    ]

    balance = financial_data[
        "balance"
    ]

    ratios = financial_data[
        "ratios"
    ]

    # ========================================================
    # INCOME STATEMENT
    # ========================================================

    revenue = extract_item(
        income,
        "revenue"
    )

    gross_profit = extract_item(
        income,
        "gross_profit"
    )

    operating_profit = extract_item(
        income,
        "operating_profit"
    )

    net_income = extract_item(
        income,
        "net_profit"
    )

    # ========================================================
    # BALANCE SHEET
    # ========================================================

    total_assets = extract_item(
        balance,
        "total_assets"
    )

    current_assets = extract_item(
        balance,
        "current_assets"
    )

    equity = extract_item(
        balance,
        "owners_equity_2"
    )

    liabilities = extract_item(
        balance,
        "total_liabilities"
    )

    short_term_debt = extract_item(
        balance,
        "short_term_borrowings_and_financial_leases"
    )

    long_term_debt = extract_item(
        balance,
        "long_term_borrowings_and_financial_leases"
    )

    # ========================================================
    # RATIOS
    # ========================================================

    roe = extract_item(
        ratios,
        "roe"
    )

    roa = extract_item(
        ratios,
        "roa"
    )

    eps = extract_item(
        ratios,
        "trailing_eps"
    )

    pe = extract_item(
        ratios,
        "pe_ratio"
    )

    pb = extract_item(
        ratios,
        "pb_ratio"
    )

    current_ratio = extract_item(
        ratios,
        "short_term_ratio"
    )

    quick_ratio = extract_item(
        ratios,
        "quick_ratio"
    )

    interest_coverage = extract_item(
        ratios,
        "interest_coverage"
    )

    # ========================================================
    # GROWTH TỪ VNSTOCK
    # ========================================================

    revenue_growth_vnstock = (
        extract_item(
            ratios,
            "net_revenue"
        )
    )

    profit_growth_vnstock = (
        extract_item(
            ratios,
            "profit_after_tax_for_shareholders_of_the_parent_company"
        )
    )

    # ========================================================
    # KIỂM TRA MAPPING KỲ
    # ========================================================

    mapping_check = (
        detect_statement_period_mapping(
            revenue,
            revenue_growth_vnstock
        )
    )

    # ========================================================
    # NẾU PHÁT HIỆN STATEMENT BỊ ĐẢO
    # THÌ ĐẢO TẤT CẢ STATEMENT CÙNG CÁCH
    #
    # KHÔNG ĐẢO:
    # - ROE
    # - ROA
    # - EPS
    # - P/E
    # - P/B
    # - growth Vnstock
    # ========================================================

    if mapping_check[
        "reverse_statements"
    ]:

        revenue = (
            reverse_period_values(
                revenue
            )
        )

        gross_profit = (
            reverse_period_values(
                gross_profit
            )
        )

        operating_profit = (
            reverse_period_values(
                operating_profit
            )
        )

        net_income = (
            reverse_period_values(
                net_income
            )
        )

        total_assets = (
            reverse_period_values(
                total_assets
            )
        )

        current_assets = (
            reverse_period_values(
                current_assets
            )
        )

        equity = (
            reverse_period_values(
                equity
            )
        )

        liabilities = (
            reverse_period_values(
                liabilities
            )
        )

        short_term_debt = (
            reverse_period_values(
                short_term_debt
            )
        )

        long_term_debt = (
            reverse_period_values(
                long_term_debt
            )
        )

    # ========================================================
    # TẬP HỢP CÁC KỲ
    # ========================================================

    datasets = [
        revenue,
        gross_profit,
        operating_profit,
        net_income,
        total_assets,
        current_assets,
        equity,
        liabilities,
        short_term_debt,
        long_term_debt,
        roe,
        roa,
        eps,
        pe,
        pb,
        current_ratio,
        quick_ratio,
        interest_coverage,
        revenue_growth_vnstock,
        profit_growth_vnstock
    ]

    all_periods = set()

    for dataset in datasets:

        all_periods.update(
            dataset.keys()
        )

    sorted_periods = sorted(
        all_periods,
        key=period_sort_key
    )

    # ========================================================
    # DATAFRAME
    # ========================================================

    rows = []

    for period in sorted_periods:

        short_debt = (
            short_term_debt.get(
                period,
                np.nan
            )
        )

        long_debt = (
            long_term_debt.get(
                period,
                np.nan
            )
        )

        if (
            pd.isna(short_debt)
            and
            pd.isna(long_debt)
        ):

            total_debt = np.nan

        else:

            total_debt = (
                (
                    0
                    if pd.isna(short_debt)
                    else short_debt
                )
                +
                (
                    0
                    if pd.isna(long_debt)
                    else long_debt
                )
            )

        rows.append({

            "ticker":
                ticker,

            "period":
                str(period),

            "revenue":
                revenue.get(
                    period,
                    np.nan
                ),

            "gross_profit":
                gross_profit.get(
                    period,
                    np.nan
                ),

            "operating_profit":
                operating_profit.get(
                    period,
                    np.nan
                ),

            "net_income":
                net_income.get(
                    period,
                    np.nan
                ),

            "total_assets":
                total_assets.get(
                    period,
                    np.nan
                ),

            "current_assets":
                current_assets.get(
                    period,
                    np.nan
                ),

            "equity":
                equity.get(
                    period,
                    np.nan
                ),

            "liabilities":
                liabilities.get(
                    period,
                    np.nan
                ),

            "total_debt":
                total_debt,

            "roe_vnstock":
                roe.get(
                    period,
                    np.nan
                ),

            "roa_vnstock":
                roa.get(
                    period,
                    np.nan
                ),

            "eps":
                eps.get(
                    period,
                    np.nan
                ),

            "pe":
                pe.get(
                    period,
                    np.nan
                ),

            "pb":
                pb.get(
                    period,
                    np.nan
                ),

            "current_ratio":
                current_ratio.get(
                    period,
                    np.nan
                ),

            "quick_ratio":
                quick_ratio.get(
                    period,
                    np.nan
                ),

            "interest_coverage":
                interest_coverage.get(
                    period,
                    np.nan
                ),

            "revenue_growth_vnstock":
                revenue_growth_vnstock.get(
                    period,
                    np.nan
                ),

            "profit_growth_vnstock":
                profit_growth_vnstock.get(
                    period,
                    np.nan
                ),

            # Metadata kiểm tra mapping
            "statement_period_adjusted":
                mapping_check[
                    "reverse_statements"
                ]
        })

    df = pd.DataFrame(
        rows
    )

    # Lưu metadata vào attrs
    df.attrs[
        "period_mapping_check"
    ] = mapping_check

    return df.reset_index(
        drop=True
    )


def load_company_financials(
    ticker,
    period="year"
):

    raw_data = (
        get_financial_data(
            ticker=ticker,
            period=period
        )
    )

    return (
        create_financial_summary(
            raw_data
        )
    )


def safe_divide(a, b):

    if pd.isna(a):

        return np.nan

    if pd.isna(b):

        return np.nan

    if b == 0:

        return np.nan

    return a / b


def calculate_growth(df):

    df = df.copy()

    # Vnstock:
    # 11.56 = 11.56%
    #
    # Chuẩn hóa:
    # 0.1156

    df["revenue_growth"] = (
        pd.to_numeric(
            df[
                "revenue_growth_vnstock"
            ],
            errors="coerce"
        )
        / 100
    )

    df["profit_growth"] = (
        pd.to_numeric(
            df[
                "profit_growth_vnstock"
            ],
            errors="coerce"
        )
        / 100
    )

    return df


def calculate_margins(df):

    df = df.copy()

    df["gross_margin"] = (
        df.apply(
            lambda row:
            safe_divide(
                row[
                    "gross_profit"
                ],
                row[
                    "revenue"
                ]
            ),
            axis=1
        )
    )

    df["operating_margin"] = (
        df.apply(
            lambda row:
            safe_divide(
                row[
                    "operating_profit"
                ],
                row[
                    "revenue"
                ]
            ),
            axis=1
        )
    )

    df["net_margin"] = (
        df.apply(
            lambda row:
            safe_divide(
                row[
                    "net_income"
                ],
                row[
                    "revenue"
                ]
            ),
            axis=1
        )
    )

    return df


def calculate_roa_roe(df):

    df = df.copy()

    df["average_assets"] = (
        (
            df[
                "total_assets"
            ]
            +
            df[
                "total_assets"
            ].shift(1)
        )
        / 2
    )

    df["average_equity"] = (
        (
            df[
                "equity"
            ]
            +
            df[
                "equity"
            ].shift(1)
        )
        / 2
    )

    df["roa_calculated"] = (
        df.apply(
            lambda row:
            safe_divide(
                row[
                    "net_income"
                ],
                row[
                    "average_assets"
                ]
            ),
            axis=1
        )
    )

    df["roe_calculated"] = (
        df.apply(
            lambda row:
            safe_divide(
                row[
                    "net_income"
                ],
                row[
                    "average_equity"
                ]
            ),
            axis=1
        )
    )

    return df


def calculate_leverage(df):

    df = df.copy()

    # Tổng nợ phải trả / tài sản
    df["liabilities_to_assets"] = (
        df.apply(
            lambda row:
            safe_divide(
                row[
                    "liabilities"
                ],
                row[
                    "total_assets"
                ]
            ),
            axis=1
        )
    )

    # Tổng nợ phải trả / VCSH
    df["liabilities_to_equity"] = (
        df.apply(
            lambda row:
            safe_divide(
                row[
                    "liabilities"
                ],
                row[
                    "equity"
                ]
            ),
            axis=1
        )
    )

    # Nợ vay có lãi / tài sản
    df["debt_to_assets"] = (
        df.apply(
            lambda row:
            safe_divide(
                row[
                    "total_debt"
                ],
                row[
                    "total_assets"
                ]
            ),
            axis=1
        )
    )

    # Nợ vay có lãi / VCSH
    df["debt_to_equity"] = (
        df.apply(
            lambda row:
            safe_divide(
                row[
                    "total_debt"
                ],
                row[
                    "equity"
                ]
            ),
            axis=1
        )
    )

    return df


def calculate_financial_ratios(df):

    # Giữ metadata
    mapping_metadata = (
        df.attrs.get(
            "period_mapping_check",
            {}
        )
    )

    df = df.copy()

    df = calculate_growth(
        df
    )

    df = calculate_margins(
        df
    )

    df = calculate_roa_roe(
        df
    )

    df = calculate_leverage(
        df
    )

    df.attrs[
        "period_mapping_check"
    ] = mapping_metadata

    return df


def normalize_bank_percentage(value):
    """Convert Vnstock percentage points (e.g. 21.51) to decimal (0.2151)."""
    if pd.isna(value):
        return np.nan
    return float(value) / 100


def detect_bank_statement_mapping(total_assets, asset_growth_reference, tolerance=0.03):
    """Detect reversed bank statement periods using total-asset growth from Vnstock ratios."""
    result = detect_statement_period_mapping(
        total_assets,
        asset_growth_reference,
        tolerance=tolerance,
    )
    if result.get("reason"):
        result["reason"] = result["reason"].replace("doanh thu", "tổng tài sản")
    return result


def create_bank_financial_summary(financial_data):
    """Create a bank-specific standardized statement table; never uses regular-company revenue logic."""
    ticker = financial_data["ticker"]
    income = financial_data["income"]
    balance = financial_data["balance"]
    ratios = financial_data["ratios"]

    net_income = extract_item(income, "net_profit")
    total_assets = extract_item(balance, "total_assets")
    liabilities = extract_item(balance, "total_liabilities")
    equity = extract_item(balance, "capital_and_reserves")
    customer_loans = extract_item(balance, "loans_advances_and_finance_leases_to_customers")
    customer_deposits = extract_item(balance, "deposits_from_customers")

    roe = extract_item(ratios, "roe")
    roa = extract_item(ratios, "roa")
    eps = extract_item(ratios, "trailing_eps")
    pe = extract_item(ratios, "pe_ratio")
    pb = extract_item(ratios, "pb_ratio")
    asset_growth_ref = extract_item(ratios, "total_assets")

    mapping = detect_bank_statement_mapping(total_assets, asset_growth_ref)

    statement_series = {
        "net_income": net_income,
        "total_assets": total_assets,
        "equity": equity,
        "liabilities": liabilities,
        "customer_loans": customer_loans,
        "customer_deposits": customer_deposits,
    }
    if mapping["reverse_statements"]:
        statement_series = {
            key: reverse_period_values(values)
            for key, values in statement_series.items()
        }

    datasets = list(statement_series.values()) + [roe, roa, eps, pe, pb]
    periods = set()
    for data in datasets:
        periods.update(data.keys())
    periods = sorted(periods, key=period_sort_key)

    rows = []
    for period in periods:
        rows.append({
            "ticker": ticker,
            "period": str(period),
            "net_income": statement_series["net_income"].get(period, np.nan),
            "total_assets": statement_series["total_assets"].get(period, np.nan),
            "equity": statement_series["equity"].get(period, np.nan),
            "liabilities": statement_series["liabilities"].get(period, np.nan),
            "customer_loans": statement_series["customer_loans"].get(period, np.nan),
            "customer_deposits": statement_series["customer_deposits"].get(period, np.nan),
            "roe": roe.get(period, np.nan),
            "roa": roa.get(period, np.nan),
            "eps": eps.get(period, np.nan),
            "pe": pe.get(period, np.nan),
            "pb": pb.get(period, np.nan),
            "statement_period_adjusted": mapping["reverse_statements"],
        })

    df = pd.DataFrame(rows).reset_index(drop=True)
    df.attrs["period_mapping_check"] = mapping
    return df


def create_bank_ratio_dataframe(raw_financial_data, bank_financial_df):
    ratios = raw_financial_data["ratios"]
    extracted = {
        metric: extract_item(ratios, item_id)
        for metric, item_id in BANK_RATIO_IDS.items()
    }

    periods = set(bank_financial_df["period"].astype(str))
    for data in extracted.values():
        periods.update(data.keys())
    periods = sorted(periods, key=period_sort_key)

    rows = []
    decimal_metrics = {
        "loan_growth", "deposit_growth", "profit_growth",
        "nim", "npl", "llr", "car", "casa", "ldr",
    }

    for period in periods:
        base_match = bank_financial_df[
            bank_financial_df["period"].astype(str) == str(period)
        ]
        base = {} if base_match.empty else base_match.iloc[0].to_dict()

        row = {"ticker": raw_financial_data["ticker"], "period": str(period)}
        for metric in BANK_RATIO_IDS:
            value = extracted[metric].get(str(period), np.nan)
            if metric in decimal_metrics and not pd.isna(value):
                value = normalize_bank_percentage(value)
            row[metric] = value

        # Fallback only for metrics already standardized in bank statement table.
        for metric in ["roe", "roa", "eps", "pe", "pb"]:
            if pd.isna(row.get(metric, np.nan)):
                row[metric] = base.get(metric, np.nan)
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)
