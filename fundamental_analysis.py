"""
fundamental_analysis.py
TV3 - Fundamental Analysis

Phân tích tăng trưởng, khả năng sinh lời, sức khỏe tài chính/chất lượng tài sản,
định giá, xu hướng, điểm mạnh/yếu và chất lượng dữ liệu.
"""

import math
import numpy as np
import pandas as pd

from financial_ratios import (
    get_financial_data,
    create_financial_summary,
    calculate_financial_ratios,
    create_bank_financial_summary,
    create_bank_ratio_dataframe,
)



def get_trend(
    series,
    tolerance=0.05
):

    values = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )

    if len(values) < 2:

        return "insufficient_data"

    first = values.iloc[0]

    last = values.iloc[-1]

    if first == 0:

        return "insufficient_data"

    change = (
        last - first
    ) / abs(first)

    if change > tolerance:

        return "increasing"

    if change < -tolerance:

        return "decreasing"

    return "stable"


def calculate_cagr_from_growth(
    df,
    growth_column
):

    """
    Với dữ liệu 2022-2025:

    bỏ growth 2022 vì nó là:
    2021 -> 2022

    Chỉ compound:
    2022 -> 2023
    2023 -> 2024
    2024 -> 2025
    """

    if growth_column not in df.columns:

        return np.nan

    growth = pd.to_numeric(
        df[
            growth_column
        ],
        errors="coerce"
    )

    # Bỏ kỳ đầu
    growth = (
        growth
        .iloc[1:]
        .dropna()
    )

    growth = growth[
        growth > -1
    ]

    if growth.empty:

        return np.nan

    compounded = np.prod(
        1 + growth
    )

    return (
        compounded
        ** (
            1 / len(growth)
        )
        - 1
    )


def analyze_growth(df):

    latest = df.iloc[-1]

    return {

        "revenue_growth":
            latest[
                "revenue_growth"
            ],

        "profit_growth":
            latest[
                "profit_growth"
            ],

        "revenue_cagr":
            calculate_cagr_from_growth(
                df,
                "revenue_growth"
            ),

        "profit_cagr":
            calculate_cagr_from_growth(
                df,
                "profit_growth"
            ),

        "revenue_growth_trend":
            get_trend(
                df[
                    "revenue_growth"
                ]
            ),

        "profit_growth_trend":
            get_trend(
                df[
                    "profit_growth"
                ]
            )
    }


def analyze_profitability(df):

    latest = df.iloc[-1]

    return {

        "gross_margin":
            latest[
                "gross_margin"
            ],

        "operating_margin":
            latest[
                "operating_margin"
            ],

        "net_margin":
            latest[
                "net_margin"
            ],

        # Dùng Vnstock làm chỉ tiêu chính
        "roe":
            latest[
                "roe_vnstock"
            ],

        "roa":
            latest[
                "roa_vnstock"
            ],

        "gross_margin_trend":
            get_trend(
                df[
                    "gross_margin"
                ]
            ),

        "operating_margin_trend":
            get_trend(
                df[
                    "operating_margin"
                ]
            ),

        "net_margin_trend":
            get_trend(
                df[
                    "net_margin"
                ]
            ),

        "roe_trend":
            get_trend(
                df[
                    "roe_vnstock"
                ]
            ),

        "roa_trend":
            get_trend(
                df[
                    "roa_vnstock"
                ]
            )
    }


def analyze_financial_health(df):

    latest = df.iloc[-1]

    return {

        "current_ratio":
            latest[
                "current_ratio"
            ],

        "quick_ratio":
            latest[
                "quick_ratio"
            ],

        "interest_coverage":
            latest[
                "interest_coverage"
            ],

        "liabilities_to_assets":
            latest[
                "liabilities_to_assets"
            ],

        "liabilities_to_equity":
            latest[
                "liabilities_to_equity"
            ],

        "debt_to_assets":
            latest[
                "debt_to_assets"
            ],

        "debt_to_equity":
            latest[
                "debt_to_equity"
            ],

        "leverage_trend":
            get_trend(
                df[
                    "debt_to_equity"
                ]
            )
    }


def analyze_valuation(df):

    latest = df.iloc[-1]

    valid_pe = (
        pd.to_numeric(
            df[
                "pe"
            ],
            errors="coerce"
        )
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .dropna()
    )

    valid_pb = (
        pd.to_numeric(
            df[
                "pb"
            ],
            errors="coerce"
        )
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .dropna()
    )

    return {

        "eps":
            latest[
                "eps"
            ],

        "pe":
            latest[
                "pe"
            ],

        "pb":
            latest[
                "pb"
            ],

        "historical_pe_median":
            (
                valid_pe.median()
                if not valid_pe.empty
                else np.nan
            ),

        "historical_pb_median":
            (
                valid_pb.median()
                if not valid_pb.empty
                else np.nan
            )
    }


def generate_observations(
    growth,
    profitability,
    health,
    valuation
):

    strengths = []

    weaknesses = []

    valuation_notes = []

    # --------------------------------------------------------
    # Revenue growth
    # --------------------------------------------------------

    revenue_growth = (
        growth[
            "revenue_growth"
        ]
    )

    if not pd.isna(
        revenue_growth
    ):

        if revenue_growth > 0:

            strengths.append(
                "Doanh thu kỳ gần nhất "
                "tiếp tục tăng trưởng."
            )

        elif revenue_growth < 0:

            weaknesses.append(
                "Doanh thu kỳ gần nhất "
                "suy giảm."
            )

    # --------------------------------------------------------
    # Profit growth
    # --------------------------------------------------------

    profit_growth = (
        growth[
            "profit_growth"
        ]
    )

    if not pd.isna(
        profit_growth
    ):

        if profit_growth > 0:

            strengths.append(
                "Lợi nhuận kỳ gần nhất "
                "tiếp tục tăng trưởng."
            )

        elif profit_growth < 0:

            weaknesses.append(
                "Lợi nhuận kỳ gần nhất "
                "suy giảm."
            )

    # --------------------------------------------------------
    # Growth momentum
    # --------------------------------------------------------

    if (
        growth[
            "revenue_growth_trend"
        ]
        == "decreasing"
        and
        revenue_growth > 0
    ):

        weaknesses.append(
            "Doanh thu vẫn tăng nhưng "
            "tốc độ tăng trưởng đang chậm lại."
        )

    if (
        growth[
            "profit_growth_trend"
        ]
        == "decreasing"
        and
        profit_growth > 0
    ):

        weaknesses.append(
            "Lợi nhuận vẫn tăng nhưng "
            "tốc độ tăng trưởng đang chậm lại."
        )

    # --------------------------------------------------------
    # Gross margin
    # --------------------------------------------------------

    if (
        profitability[
            "gross_margin_trend"
        ]
        == "increasing"
    ):

        strengths.append(
            "Biên lợi nhuận gộp "
            "có xu hướng cải thiện."
        )

    elif (
        profitability[
            "gross_margin_trend"
        ]
        == "decreasing"
    ):

        weaknesses.append(
            "Biên lợi nhuận gộp "
            "có xu hướng thu hẹp."
        )

    # --------------------------------------------------------
    # Net margin
    # --------------------------------------------------------

    if (
        profitability[
            "net_margin_trend"
        ]
        == "increasing"
    ):

        strengths.append(
            "Biên lợi nhuận ròng "
            "có xu hướng cải thiện."
        )

    elif (
        profitability[
            "net_margin_trend"
        ]
        == "decreasing"
    ):

        weaknesses.append(
            "Biên lợi nhuận ròng "
            "có xu hướng thu hẹp."
        )

    # --------------------------------------------------------
    # ROE
    # --------------------------------------------------------

    roe = profitability[
        "roe"
    ]

    if not pd.isna(roe):

        if roe >= 20:

            strengths.append(
                "Khả năng sinh lời trên "
                "vốn chủ sở hữu ở mức cao."
            )

        elif roe < 10:

            weaknesses.append(
                "Khả năng sinh lời trên "
                "vốn chủ sở hữu ở mức thấp."
            )

    # --------------------------------------------------------
    # Current ratio
    # --------------------------------------------------------

    current_ratio = (
        health[
            "current_ratio"
        ]
    )

    if not pd.isna(
        current_ratio
    ):

        if current_ratio >= 1:

            strengths.append(
                "Khả năng thanh toán "
                "ngắn hạn ở mức "
                "tương đối an toàn."
            )

        else:

            weaknesses.append(
                "Khả năng thanh toán "
                "ngắn hạn cần được "
                "theo dõi."
            )

    # --------------------------------------------------------
    # Interest coverage
    # --------------------------------------------------------

    interest_coverage = (
        health[
            "interest_coverage"
        ]
    )

    if not pd.isna(
        interest_coverage
    ):

        if interest_coverage >= 5:

            strengths.append(
                "Khả năng thanh toán "
                "chi phí lãi vay "
                "ở mức tốt."
            )

        elif interest_coverage < 2:

            weaknesses.append(
                "Khả năng thanh toán "
                "lãi vay ở mức thấp."
            )

    # --------------------------------------------------------
    # Debt / Equity
    # --------------------------------------------------------

    debt_to_equity = (
        health[
            "debt_to_equity"
        ]
    )

    if not pd.isna(
        debt_to_equity
    ):

        if debt_to_equity > 1:

            weaknesses.append(
                "Nợ vay cao so với "
                "vốn chủ sở hữu."
            )

        elif debt_to_equity < 0.5:

            strengths.append(
                "Nợ vay ở mức tương đối thấp "
                "so với vốn chủ sở hữu."
            )

    # --------------------------------------------------------
    # P/E
    # --------------------------------------------------------

    pe = valuation[
        "pe"
    ]

    median_pe = valuation[
        "historical_pe_median"
    ]

    if (
        not pd.isna(pe)
        and
        not pd.isna(median_pe)
        and
        median_pe > 0
    ):

        difference = (
            pe / median_pe
            - 1
        )

        if difference < -0.10:

            valuation_notes.append(
                "P/E hiện tại thấp hơn "
                "đáng kể so với trung vị "
                "lịch sử."
            )

        elif difference > 0.10:

            valuation_notes.append(
                "P/E hiện tại cao hơn "
                "đáng kể so với trung vị "
                "lịch sử."
            )

        else:

            valuation_notes.append(
                "P/E hiện tại không "
                "chênh lệch lớn so với "
                "trung vị lịch sử."
            )

    # --------------------------------------------------------
    # P/B
    # --------------------------------------------------------

    pb = valuation[
        "pb"
    ]

    median_pb = valuation[
        "historical_pb_median"
    ]

    if (
        not pd.isna(pb)
        and
        not pd.isna(median_pb)
        and
        median_pb > 0
    ):

        difference = (
            pb / median_pb
            - 1
        )

        if difference < -0.10:

            valuation_notes.append(
                "P/B hiện tại thấp hơn "
                "đáng kể so với trung vị "
                "lịch sử."
            )

        elif difference > 0.10:

            valuation_notes.append(
                "P/B hiện tại cao hơn "
                "đáng kể so với trung vị "
                "lịch sử."
            )

        else:

            valuation_notes.append(
                "P/B hiện tại không "
                "chênh lệch lớn so với "
                "trung vị lịch sử."
            )

    return {

        "strengths":
            strengths,

        "weaknesses":
            weaknesses,

        "valuation_notes":
            valuation_notes
    }


def check_data_quality(df):

    warnings = []

    notes = []

    # ========================================================
    # REQUIRED DATA
    # ========================================================

    required_columns = [
        "revenue_growth",
        "profit_growth",
        "roe_vnstock",
        "roa_vnstock",
        "pe",
        "pb"
    ]

    missing = []

    for column in required_columns:

        if column not in df.columns:

            missing.append(
                column
            )

        elif df[
            column
        ].isna().all():

            missing.append(
                column
            )

    if missing:

        warnings.append(
            "Thiếu dữ liệu cho các chỉ tiêu: "
            + ", ".join(missing)
        )

    # ========================================================
    # DUPLICATE PERIOD
    # ========================================================

    if df[
        "period"
    ].duplicated().any():

        warnings.append(
            "Phát hiện kỳ báo cáo bị trùng."
        )

    # ========================================================
    # P/E
    # ========================================================

    if "pe" in df.columns:

        pe_values = pd.to_numeric(
            df[
                "pe"
            ],
            errors="coerce"
        )

        if (
            pe_values < 0
        ).any():

            warnings.append(
                "Có kỳ P/E âm; cần "
                "thận trọng khi diễn giải "
                "định giá."
            )

    # ========================================================
    # P/B
    # ========================================================

    if "pb" in df.columns:

        pb_values = pd.to_numeric(
            df[
                "pb"
            ],
            errors="coerce"
        )

        if (
            pb_values < 0
        ).any():

            warnings.append(
                "Có kỳ P/B âm; cần "
                "kiểm tra giá trị sổ sách."
            )

    # ========================================================
    # KIỂM TRA REVENUE GROWTH SAU CHUẨN HÓA
    # ========================================================

    revenue = pd.to_numeric(
        df[
            "revenue"
        ],
        errors="coerce"
    )

    reported_growth = pd.to_numeric(
        df[
            "revenue_growth"
        ],
        errors="coerce"
    )

    calculated_growth = (
        revenue.pct_change(
            fill_method=None
        )
    )

    mismatch = (
        calculated_growth.notna()
        &
        reported_growth.notna()
        &
        (
            (
                calculated_growth
                - reported_growth
            ).abs()
            > 0.03
        )
    )

    if mismatch.any():

        warnings.append(
            "Tăng trưởng doanh thu tính từ "
            "BCTC sau chuẩn hóa vẫn không "
            "khớp với dữ liệu tăng trưởng "
            "tham chiếu."
        )

    # ========================================================
    # PERIOD MAPPING INFORMATION
    # ========================================================

    mapping_check = (
        df.attrs.get(
            "period_mapping_check",
            {}
        )
    )

    if mapping_check.get(
        "reverse_statements",
        False
    ):

        notes.append(
            "Hệ thống phát hiện và đã tự "
            "điều chỉnh mapping kỳ của BCTC "
            "dựa trên đối chiếu tăng trưởng "
            "doanh thu."
        )

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "status":
            (
                "OK"
                if len(warnings) == 0
                else "WARNING"
            ),

        "warnings":
            warnings,

        "notes":
            notes,

        "statement_period_adjusted":
            mapping_check.get(
                "reverse_statements",
                False
            )
    }


def analyze_fundamentals(df):

    if df.empty:

        raise ValueError(
            "Không có dữ liệu "
            "để phân tích."
        )

    growth = (
        analyze_growth(
            df
        )
    )

    profitability = (
        analyze_profitability(
            df
        )
    )

    financial_health = (
        analyze_financial_health(
            df
        )
    )

    valuation = (
        analyze_valuation(
            df
        )
    )

    observations = (
        generate_observations(
            growth,
            profitability,
            financial_health,
            valuation
        )
    )

    data_quality = (
        check_data_quality(
            df
        )
    )

    return {

        "ticker":
            df.iloc[-1][
                "ticker"
            ],

        "latest_period":
            df.iloc[-1][
                "period"
            ],

        "growth":
            growth,

        "profitability":
            profitability,

        "financial_health":
            financial_health,

        "valuation":
            valuation,

        "observations":
            observations,

        "data_quality":
            data_quality
    }


def create_period_comparison(df):

    columns = [

        "ticker",
        "period",

        "revenue_growth",
        "profit_growth",

        "gross_margin",
        "operating_margin",
        "net_margin",

        "roe_vnstock",
        "roa_vnstock",

        "current_ratio",
        "quick_ratio",

        "interest_coverage",

        "debt_to_assets",
        "debt_to_equity",

        "eps",
        "pe",
        "pb"
    ]

    available_columns = [

        column

        for column in columns

        if column in df.columns
    ]

    return (
        df[
            available_columns
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def convert_to_native(value):

    if isinstance(
        value,
        dict
    ):

        return {
            key:
                convert_to_native(
                    item
                )

            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list
    ):

        return [
            convert_to_native(
                item
            )

            for item in value
        ]

    if isinstance(
        value,
        np.generic
    ):

        return value.item()

    return value


def analyze_bank_fundamentals(df):
    if df.empty:
        raise ValueError("Không có dữ liệu ngân hàng để phân tích.")

    latest = df.iloc[-1]
    valid_pe = pd.to_numeric(df["pe"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    valid_pb = pd.to_numeric(df["pb"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    growth = {
        "loan_growth": latest.get("loan_growth", np.nan),
        "deposit_growth": latest.get("deposit_growth", np.nan),
        "profit_growth": latest.get("profit_growth", np.nan),
        "loan_growth_trend": get_trend(df["loan_growth"]),
        "deposit_growth_trend": get_trend(df["deposit_growth"]),
        "profit_growth_trend": get_trend(df["profit_growth"]),
    }
    profitability = {
        "roe": latest.get("roe", np.nan),
        "roa": latest.get("roa", np.nan),
        "nim": latest.get("nim", np.nan),
        "roe_trend": get_trend(df["roe"]),
        "roa_trend": get_trend(df["roa"]),
        "nim_trend": get_trend(df["nim"]),
    }
    asset_quality = {
        "npl": latest.get("npl", np.nan),
        "llr": latest.get("llr", np.nan),
        "npl_trend": get_trend(df["npl"]),
        "llr_trend": get_trend(df["llr"]),
    }
    capital_funding = {
        "car": latest.get("car", np.nan),
        "casa": latest.get("casa", np.nan),
        "ldr": latest.get("ldr", np.nan),
        "car_trend": get_trend(df["car"]),
        "casa_trend": get_trend(df["casa"]),
        "ldr_trend": get_trend(df["ldr"]),
    }
    valuation = {
        "eps": latest.get("eps", np.nan),
        "pe": latest.get("pe", np.nan),
        "pb": latest.get("pb", np.nan),
        "historical_pe_median": valid_pe.median() if not valid_pe.empty else np.nan,
        "historical_pb_median": valid_pb.median() if not valid_pb.empty else np.nan,
    }

    strengths, weaknesses, valuation_notes = [], [], []
    if not pd.isna(growth["loan_growth"]) and growth["loan_growth"] > 0:
        strengths.append("Dư nợ cho vay kỳ gần nhất tiếp tục tăng trưởng.")
    if not pd.isna(growth["deposit_growth"]) and growth["deposit_growth"] > 0:
        strengths.append("Huy động vốn khách hàng kỳ gần nhất tiếp tục tăng trưởng.")
    if not pd.isna(growth["profit_growth"]):
        (strengths if growth["profit_growth"] > 0 else weaknesses).append(
            "Lợi nhuận kỳ gần nhất tiếp tục tăng trưởng." if growth["profit_growth"] > 0
            else "Lợi nhuận kỳ gần nhất suy giảm."
        )
    if not pd.isna(profitability["roe"]) and profitability["roe"] >= 20:
        strengths.append("ROE ở mức cao.")
    if profitability["nim_trend"] == "increasing":
        strengths.append("Biên lãi ròng NIM có xu hướng cải thiện.")
    elif profitability["nim_trend"] == "decreasing":
        weaknesses.append("Biên lãi ròng NIM có xu hướng thu hẹp.")
    if not pd.isna(asset_quality["npl"]):
        (strengths if asset_quality["npl"] < 0.03 else weaknesses).append(
            "Tỷ lệ nợ xấu dưới 3%." if asset_quality["npl"] < 0.03
            else "Tỷ lệ nợ xấu từ 3% trở lên cần được theo dõi."
        )

    pe, med_pe = valuation["pe"], valuation["historical_pe_median"]
    pb, med_pb = valuation["pb"], valuation["historical_pb_median"]
    if not pd.isna(pe) and not pd.isna(med_pe) and med_pe > 0:
        diff = pe / med_pe - 1
        valuation_notes.append(
            "P/E hiện tại thấp hơn đáng kể so với trung vị lịch sử." if diff < -0.10
            else "P/E hiện tại cao hơn đáng kể so với trung vị lịch sử." if diff > 0.10
            else "P/E hiện tại không chênh lệch lớn so với trung vị lịch sử."
        )
    if not pd.isna(pb) and not pd.isna(med_pb) and med_pb > 0:
        diff = pb / med_pb - 1
        valuation_notes.append(
            "P/B hiện tại thấp hơn đáng kể so với trung vị lịch sử." if diff < -0.10
            else "P/B hiện tại cao hơn đáng kể so với trung vị lịch sử." if diff > 0.10
            else "P/B hiện tại không chênh lệch lớn so với trung vị lịch sử."
        )

    optional = ["npl", "llr", "car", "casa"]
    missing_optional = [c for c in optional if c not in df.columns or df[c].isna().all()]
    required = ["loan_growth", "deposit_growth", "profit_growth", "roe", "roa", "nim", "ldr", "eps", "pe", "pb"]
    missing_required = [c for c in required if c not in df.columns or df[c].isna().all()]
    warnings = []
    notes = []

    if missing_required:
        warnings.append(
            "Thiếu dữ liệu cho các chỉ tiêu chính: "
            + ", ".join(missing_required)
        )

    if missing_optional:
        notes.append(
            "Nguồn Vnstock hiện chưa có/không nhận diện được "
            "các chỉ tiêu bổ sung: "
            + ", ".join(missing_optional)
            + "."
        )

    notes.append(
        "Không tự điền dữ liệu thiếu bằng 0."
    )

    return {
        "ticker": latest["ticker"],
        "company_type": "bank",
        "latest_period": latest["period"],
        "growth": growth,
        "profitability": profitability,
        "asset_quality": asset_quality,
        "capital_funding": capital_funding,
        "valuation": valuation,
        "observations": {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "valuation_notes": valuation_notes,
        },
        "data_quality": {
            "status": "OK" if not missing_required else "WARNING",
            "warnings": warnings,
            "notes": notes,
        },
    }


def run_fundamental_analysis(ticker, period="year", company_type="regular"):
    ticker = ticker.upper().strip()
    company_type = str(company_type).lower().strip()
    if company_type not in ["regular", "bank"]:
        raise ValueError("company_type chỉ nhận 'regular' hoặc 'bank'.")

    raw_data = get_financial_data(ticker=ticker, period=period)

    if company_type == "regular":
        financial_df = create_financial_summary(raw_data)
        mapping_metadata = financial_df.attrs.get("period_mapping_check", {})
        ratio_df = calculate_financial_ratios(financial_df)
        ratio_df.attrs["period_mapping_check"] = mapping_metadata
        comparison_df = create_period_comparison(ratio_df)
        analysis = analyze_fundamentals(ratio_df)
        analysis["company_type"] = "regular"
    else:
        financial_df = create_bank_financial_summary(raw_data)
        mapping_metadata = financial_df.attrs.get("period_mapping_check", {})
        ratio_df = create_bank_ratio_dataframe(raw_data, financial_df)
        comparison_df = ratio_df.copy()
        analysis = analyze_bank_fundamentals(ratio_df)

    return {
        "company_type": company_type,
        "financial_data": financial_df,
        "ratio_data": ratio_df,
        "comparison_data": comparison_df,
        "analysis": convert_to_native(analysis),
        "period_mapping": convert_to_native(mapping_metadata),
    }
