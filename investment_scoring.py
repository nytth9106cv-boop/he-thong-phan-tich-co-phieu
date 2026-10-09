"""
investment_scoring.py  -  TV4 (Chấm điểm và rủi ro)
===================================================
Thang điểm hấp dẫn đầu tư 0-100 = tổng hợp 4 trụ cột:

    Kỹ thuật (TV2) | Tài chính (TV3) | Định giá (TV3) | Rủi ro (risk_analysis.py)

Nguyên tắc thiết kế (công khai để kiểm tra):
  1. MỖI CHỈ SỐ được quy về điểm 0-100 bằng bảng mốc tuyến tính (ANCHORS trong file này / risk_analysis.py).
     50 điểm = mức trung tính.
  2. Điểm trụ cột = trung bình có trọng số các chỉ số có dữ liệu.
  3. Điểm tổng = trung bình có trọng số các trụ cột. Thiếu dữ liệu -> trọng số được chia lại cho phần còn lại
     và ĐỘ TIN CẬY bị hạ; không bao giờ tự điền 0 hay giá trị giả.
  4. Điểm tổng = 50 + Σ đóng góp của từng chỉ số, nên giải thích được CHÍNH XÁC chỉ số nào kéo điểm lên/xuống.
  5. Có "chốt chặn": rủi ro quá cao / thanh khoản quá thấp / nền tảng tài chính quá yếu thì nhãn không vượt "Trung lập".

Dùng nhanh:
    from investment_scoring import score_stock, score_many, explain_text
    res = score_stock("ACB", price_df, benchmark=bench, fundamentals=tv3_output_or_snapshot)
    res["total_score"], res["label"], res["confidence"]
    res["pillar_table"], res["metric_table"]            # DataFrame cho dashboard / PDF
    res["drivers_up"], res["drivers_down"], res["summary"]
    summary_df, results, errors = score_many(["ACB", "FPT", "HPG"], benchmark="auto")

Đây là công cụ hỗ trợ phân tích, KHÔNG phải khuyến nghị mua/bán.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

import risk_analysis as ra

# =========================================================================== #
# 1. CẤU HÌNH CÔNG KHAI: trọng số, nhãn, bảng mốc
# =========================================================================== #
PILLARS = ["technical", "fundamental", "valuation", "risk"]
PILLAR_LABELS = {"technical": "Kỹ thuật", "fundamental": "Tài chính", "valuation": "Định giá", "risk": "Rủi ro"}

WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "balanced":     {"technical": 0.25, "fundamental": 0.30, "valuation": 0.20, "risk": 0.25},
    "growth":       {"technical": 0.35, "fundamental": 0.25, "valuation": 0.15, "risk": 0.25},   # ưu tiên động lượng ngắn hạn
    "conservative": {"technical": 0.15, "fundamental": 0.30, "valuation": 0.20, "risk": 0.35},   # ưu tiên an toàn
}
PROFILE_LABELS = {"balanced": "Cân bằng", "growth": "Thiên về xu hướng/tăng trưởng", "conservative": "Phòng thủ"}

WEIGHT_RATIONALE = {
    "technical": "Phản ánh xu hướng, động lượng và tâm lý thị trường hiện tại; hữu ích cho thời điểm nhưng "
                 "nhiễu và dễ đảo chiều nên không để chi phối (25%).",
    "fundamental": "Chất lượng doanh nghiệp (sinh lời, tăng trưởng, an toàn tài chính) là nền tảng của lợi nhuận "
                   "dài hạn nên có trọng số lớn nhất (30%).",
    "valuation": "Mua tốt doanh nghiệp nhưng giá quá cao vẫn kém hấp dẫn; định giá điều chỉnh kỳ vọng sinh lời (20%). "
                 "Thấp hơn tài chính vì P/E, P/B dễ nhiễu theo chu kỳ.",
    "risk": "Rủi ro (biến động, drawdown, beta, thanh khoản) quyết định mức lỗ có thể gặp và khả năng thoát vị thế; "
            "được tính như một trụ cột riêng để cổ phiếu 'đẹp' nhưng quá rủi ro không được điểm cao (25%).",
}

# Ngưỡng nhãn (điểm tổng -> mức hấp dẫn)
LABEL_ORDER = ["Không hấp dẫn", "Kém hấp dẫn", "Trung lập", "Hấp dẫn", "Rất hấp dẫn"]
LABEL_THRESHOLDS = [(80, "Rất hấp dẫn"), (65, "Hấp dẫn"), (50, "Trung lập"), (35, "Kém hấp dẫn"), (0, "Không hấp dẫn")]
LABEL_MEANING = {
    "Rất hấp dẫn": "Các trụ cột đồng thuận tích cực - ứng viên ưu tiên nghiên cứu sâu.",
    "Hấp dẫn": "Điểm số tích cực, có thể đưa vào danh sách theo dõi/nghiên cứu thêm.",
    "Trung lập": "Ưu và nhược điểm xen kẽ - chưa có lợi thế rõ rệt, nên theo dõi thêm.",
    "Kém hấp dẫn": "Nhiều yếu tố bất lợi hơn thuận lợi ở thời điểm hiện tại.",
    "Không hấp dẫn": "Phần lớn chỉ số ở mức yếu - rủi ro cao so với kỳ vọng.",
}

# Chốt chặn
RISK_CAP_BELOW = 30            # điểm rủi ro < 30  => nhãn tối đa "Trung lập"
LIQUIDITY_CAP_BELOW = 20       # điểm thanh khoản < 20 => tối đa "Trung lập"
FUNDAMENTAL_CAP_BELOW = 30     # điểm tài chính < 30 => tối đa "Trung lập"
MIN_PILLAR_COVERAGE = 0.40     # trụ cột cần >= 40% trọng số chỉ số có dữ liệu mới được tính
MIN_PILLARS = 2                # cần tối thiểu 2 trụ cột

BANK_TICKERS = {"VCB", "BID", "CTG", "TCB", "MBB", "VPB", "ACB", "STB", "HDB", "TPB", "VIB", "SHB", "LPB", "MSB",
                "OCB", "EIB", "SSB", "NAB", "BAB", "ABB", "BVB", "KLB", "PGB", "SGB", "VBB", "VAB", "NVB"}
NONBANK_FINANCIAL_TICKERS = {"SSI", "VND", "VCI", "HCM", "SHS", "MBS", "FTS", "BSI", "CTS", "AGR", "VIX", "ORS",
                             "APG", "BVH", "BMI", "PVI", "MIG"}

# --- Bảng mốc: (key, nhãn, trọng số trong trụ cột, [(giá trị, điểm)...], định dạng, giải thích) -------------
_ROE_NOTE = "ROE: hiệu quả sinh lời trên vốn chủ sở hữu, càng cao càng tốt"
FUNDAMENTAL_SPECS: dict[str, list[tuple]] = {
    "regular": [
        ("roe", "ROE", 20, [(-0.05, 0), (0.05, 35), (0.10, 55), (0.15, 75), (0.22, 100)], "pct", _ROE_NOTE),
        ("roa", "ROA", 10, [(0.0, 20), (0.03, 50), (0.06, 80), (0.10, 100)], "pct", "ROA: lợi nhuận trên tổng tài sản"),
        ("net_margin", "Biên lợi nhuận ròng", 10, [(0.0, 15), (0.05, 45), (0.10, 70), (0.20, 100)], "pct",
         "Biên lợi nhuận ròng: phần lợi nhuận giữ lại trên mỗi đồng doanh thu"),
        ("revenue_growth", "Tăng trưởng doanh thu", 12, [(-0.20, 0), (0.0, 35), (0.10, 65), (0.25, 100)], "pct",
         "Tăng trưởng doanh thu kỳ gần nhất"),
        ("profit_growth", "Tăng trưởng lợi nhuận", 15, [(-0.30, 0), (0.0, 35), (0.15, 70), (0.40, 100)], "pct",
         "Tăng trưởng lợi nhuận sau thuế kỳ gần nhất"),
        ("debt_to_equity", "Nợ vay/Vốn chủ", 15, [(0.0, 100), (0.5, 85), (1.0, 65), (2.0, 30), (3.5, 0)], "x",
         "Đòn bẩy tài chính, càng thấp càng an toàn"),
        ("current_ratio", "Thanh toán hiện hành", 8, [(0.6, 0), (1.0, 45), (1.5, 80), (2.0, 100)], "x",
         "Khả năng trả nợ ngắn hạn"),
        ("interest_coverage", "Khả năng trả lãi vay", 10, [(1.0, 0), (2.0, 40), (4.0, 75), (8.0, 100)], "x",
         "Số lần lợi nhuận trước lãi vay/thuế bù đắp lãi vay"),
    ],
    # Chứng khoán / bảo hiểm: bỏ nhóm đòn bẩy & thanh toán (không so sánh được với doanh nghiệp sản xuất)
    "financial": [
        ("roe", "ROE", 25, [(-0.05, 0), (0.05, 35), (0.10, 55), (0.15, 75), (0.22, 100)], "pct", _ROE_NOTE),
        ("roa", "ROA", 15, [(0.0, 20), (0.02, 50), (0.04, 80), (0.07, 100)], "pct", "ROA: lợi nhuận trên tổng tài sản"),
        ("net_margin", "Biên lợi nhuận ròng", 15, [(0.0, 15), (0.10, 45), (0.25, 75), (0.40, 100)], "pct",
         "Biên lợi nhuận ròng"),
        ("revenue_growth", "Tăng trưởng doanh thu", 15, [(-0.20, 0), (0.0, 35), (0.10, 65), (0.25, 100)], "pct",
         "Tăng trưởng doanh thu kỳ gần nhất"),
        ("profit_growth", "Tăng trưởng lợi nhuận", 30, [(-0.30, 0), (0.0, 35), (0.15, 70), (0.40, 100)], "pct",
         "Tăng trưởng lợi nhuận sau thuế kỳ gần nhất"),
    ],
    "bank": [
        ("roe", "ROE", 20, [(0.05, 0), (0.08, 30), (0.12, 60), (0.17, 85), (0.22, 100)], "pct", _ROE_NOTE),
        ("roa", "ROA", 10, [(0.003, 0), (0.005, 30), (0.010, 65), (0.017, 100)], "pct", "ROA: lợi nhuận trên tổng tài sản"),
        ("nim", "Biên lãi ròng (NIM)", 10, [(0.02, 20), (0.03, 60), (0.04, 90), (0.05, 100)], "pct",
         "NIM: chênh lệch lãi suất cho vay và huy động"),
        ("npl", "Nợ xấu (NPL)", 20, [(0.01, 100), (0.02, 80), (0.03, 55), (0.05, 20), (0.08, 0)], "pct",
         "Tỷ lệ nợ xấu, càng thấp càng tốt"),
        ("car", "Hệ số an toàn vốn (CAR)", 10, [(0.08, 30), (0.10, 60), (0.12, 85), (0.14, 100)], "pct",
         "CAR: đệm vốn chống đỡ tổn thất (tối thiểu theo quy định 8%)"),
        ("profit_growth", "Tăng trưởng lợi nhuận", 15, [(-0.30, 0), (0.0, 35), (0.15, 70), (0.40, 100)], "pct",
         "Tăng trưởng lợi nhuận sau thuế kỳ gần nhất"),
        ("loan_growth", "Tăng trưởng tín dụng", 5, [(-0.05, 0), (0.0, 30), (0.10, 65), (0.18, 90), (0.25, 100)], "pct",
         "Tăng trưởng dư nợ cho vay"),
        ("casa", "Tỷ lệ CASA", 10, [(0.05, 10), (0.15, 40), (0.25, 75), (0.35, 100)], "pct",
         "CASA cao = vốn huy động rẻ, ổn định"),
    ],
}

VALUATION_SPECS: dict[str, list[tuple]] = {
    "regular": [
        ("pe", "P/E", 30, [(6, 100), (10, 85), (14, 65), (20, 40), (30, 15), (45, 0)], "x", "P/E thấp = rẻ hơn (với lợi nhuận dương)"),
        ("pb", "P/B", 20, [(0.8, 100), (1.5, 80), (2.5, 55), (4.0, 25), (6.0, 0)], "x", "P/B thấp = giá gần giá trị sổ sách hơn"),
        ("pe_rel", "P/E so với trung vị lịch sử", 25, [(0.7, 100), (0.9, 75), (1.0, 55), (1.2, 30), (1.5, 0)], "x",
         "<1: đang rẻ hơn mức thường thấy của chính doanh nghiệp"),
        ("pb_rel", "P/B so với trung vị lịch sử", 25, [(0.7, 100), (0.9, 75), (1.0, 55), (1.2, 30), (1.5, 0)], "x",
         "<1: đang rẻ hơn mức thường thấy của chính doanh nghiệp"),
    ],
    "bank": [
        ("pe", "P/E", 30, [(6, 100), (9, 85), (12, 62), (16, 35), (22, 0)], "x", "P/E thấp = rẻ hơn (ngân hàng thường 6-12)"),
        ("pb", "P/B", 25, [(0.8, 100), (1.3, 80), (2.0, 50), (3.0, 15), (4.0, 0)], "x", "P/B là thước đo chính của ngân hàng"),
        ("pe_rel", "P/E so với trung vị lịch sử", 20, [(0.7, 100), (0.9, 75), (1.0, 55), (1.2, 30), (1.5, 0)], "x",
         "<1: đang rẻ hơn mức thường thấy của chính doanh nghiệp"),
        ("pb_rel", "P/B so với trung vị lịch sử", 25, [(0.7, 100), (0.9, 75), (1.0, 55), (1.2, 30), (1.5, 0)], "x",
         "<1: đang rẻ hơn mức thường thấy của chính doanh nghiệp"),
    ],
}
VALUATION_SPECS["financial"] = VALUATION_SPECS["regular"]

# Kỹ thuật: 6 quy tắc của TV2 chiếm 75%, động lượng giá 6 và 3 tháng chiếm 25%
TECH_RULE_TOTAL_WEIGHT = 75.0
TECH_RULE_SCORE = {1: 85.0, 0: 50.0, -1: 15.0}      # quy tắc +1/0/-1 của TV2 -> điểm
MOMENTUM_SPECS = [
    ("return_6m", "Lợi suất 6 tháng", 15, 126, [(-0.25, 0), (0.0, 45), (0.15, 70), (0.40, 100)]),
    ("return_3m", "Lợi suất 3 tháng", 10, 63, [(-0.15, 0), (0.0, 45), (0.10, 70), (0.25, 100)]),
]

METHODOLOGY = (
    "Điểm hấp dẫn 0-100 gồm 4 trụ cột (Kỹ thuật, Tài chính, Định giá, Rủi ro). Mỗi chỉ số được quy về điểm 0-100 "
    "theo bảng mốc công khai (50 = trung tính); điểm trụ cột là trung bình có trọng số của các chỉ số có dữ liệu; "
    "điểm tổng là trung bình có trọng số của các trụ cột. Khi thiếu dữ liệu, trọng số được chia lại cho phần còn lại "
    "và độ tin cậy giảm (không điền giá trị giả). Nhãn: ≥80 Rất hấp dẫn; 65-80 Hấp dẫn; 50-65 Trung lập; "
    "35-50 Kém hấp dẫn; <35 Không hấp dẫn. Chốt chặn: rủi ro < 30, thanh khoản < 20 hoặc tài chính < 30 điểm thì "
    "nhãn tối đa là Trung lập. Kết quả chỉ mang tính tham khảo, không phải khuyến nghị đầu tư."
)
DISCLAIMER = "Kết quả do mô hình chấm điểm tự động dựa trên dữ liệu công khai, chỉ mang tính tham khảo, không phải khuyến nghị mua/bán."


class ScoringError(Exception):
    """Không đủ dữ liệu để chấm điểm."""


# =========================================================================== #
# 2. Tiện ích
# =========================================================================== #
def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fmt(value: Any, fmt: str) -> str:
    v = _num(value)
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if fmt == "pct" else f"{v:.2f}"


def _quality(score: float) -> str:
    return ("rất tốt" if score >= 85 else "tốt" if score >= 70 else "khá" if score >= 55 else "trung bình" if score >= 40
            else "yếu" if score >= 25 else "kém")


def classify_score(score: float) -> str:
    for th, lab in LABEL_THRESHOLDS:
        if score >= th:
            return lab
    return LABEL_THRESHOLDS[-1][1]


def _cap_label(label: str, max_label: str) -> str:
    return label if LABEL_ORDER.index(label) <= LABEL_ORDER.index(max_label) else max_label


def detect_company_type(symbol: str, hint: Optional[str] = None) -> str:
    """'bank' | 'financial' (chứng khoán/bảo hiểm) | 'regular'."""
    if hint:
        h = str(hint).lower()
        if h in ("bank", "financial", "regular"):
            return h
    s = (symbol or "").upper()
    return "bank" if s in BANK_TICKERS else "financial" if s in NONBANK_FINANCIAL_TICKERS else "regular"


# =========================================================================== #
# 3. Chuẩn hóa dữ liệu cơ bản từ TV3 / snapshot của TV1
# =========================================================================== #
_SECTIONS = ["growth", "profitability", "financial_health", "valuation", "asset_quality", "capital_funding"]
_RENAME = {"historical_pe_median": "pe_median", "historical_pb_median": "pb_median", "npl_ratio": "npl"}
# (khóa, ngưỡng): nếu |giá trị| > ngưỡng thì coi là điểm phần trăm (21.5) và chia 100
_PERCENT_KEYS = {"roe": 1.5, "roa": 1.0, "net_margin": 1.5, "gross_margin": 1.5, "operating_margin": 1.5,
                 "nim": 0.3, "npl": 0.3, "car": 1.0, "casa": 1.0, "ldr": 3.0, "llr": 20.0}


def normalize_fundamentals(obj: Any) -> tuple[dict, list[str]]:
    """
    Chấp nhận 3 dạng đầu vào và trả về (dict phẳng, ghi chú):
      - đầu ra của run_fundamental_analysis() (TV3) - có khóa "analysis"
      - dict `analysis` của TV3 (growth/profitability/financial_health/valuation/...)
      - snapshot phẳng của get_fundamentals_snapshot() (TV1)
    Giá trị NaN/inf/không phải số bị bỏ (thiếu chứ không phải 0). Chỉ số % ở dạng điểm phần trăm
    (21.5) được đổi về thập phân (0.215) theo ngưỡng _PERCENT_KEYS.
    """
    notes: list[str] = []
    if not obj:
        return {}, notes
    meta: dict[str, Any] = {}
    root = obj
    if isinstance(root, dict) and isinstance(root.get("analysis"), dict):
        meta["company_type"] = root.get("company_type")
        root = root["analysis"]
    if not isinstance(root, dict):
        return {}, ["Dữ liệu cơ bản không đúng định dạng (cần dict)."]
    raw: dict[str, Any] = {}
    for k, v in root.items():
        if k in _SECTIONS and isinstance(v, dict):
            raw.update(v)
        elif not isinstance(v, (dict, list)):
            raw[k] = v
    meta["company_type"] = meta.get("company_type") or root.get("company_type")
    if root.get("is_financial") is not None:
        meta["is_financial"] = bool(root["is_financial"])

    flat: dict[str, Any] = {}
    for k, v in raw.items():
        k2 = _RENAME.get(k, k)
        n = _num(v)
        if n is not None:
            flat[k2] = n
        elif isinstance(v, str) and k2.endswith("_trend"):
            flat[k2] = v
    converted = []
    for k, th in _PERCENT_KEYS.items():
        if k in flat and abs(flat[k]) > th:
            flat[k] = flat[k] / 100.0
            converted.append(k)
    if converted:
        notes.append("Đã đổi từ điểm phần trăm sang thập phân: " + ", ".join(sorted(converted)) + ".")
    flat["_meta"] = meta
    return flat, notes


def default_fundamentals_loader(symbol: str, company_type: Optional[str] = None) -> tuple[Any, list[str]]:
    """Lấy dữ liệu cơ bản: ưu tiên TV3 (run_fundamental_analysis), dự phòng snapshot của TV1. Trả (obj, cảnh báo)."""
    warns: list[str] = []
    ctype = detect_company_type(symbol, company_type)
    try:
        from fundamental_analysis import run_fundamental_analysis
        return run_fundamental_analysis(symbol, "year", "bank" if ctype == "bank" else "regular"), warns
    except Exception as exc:                                                    # noqa: BLE001
        warns.append(f"Không lấy được phân tích cơ bản từ TV3 ({type(exc).__name__}: {exc}).")
    try:
        from data_loader import load_financial_data, get_fundamentals_snapshot
        return get_fundamentals_snapshot(load_financial_data(symbol)), warns
    except Exception as exc:                                                    # noqa: BLE001
        warns.append(f"Không lấy được snapshot tài chính từ TV1 ({type(exc).__name__}: {exc}).")
    return None, warns


# =========================================================================== #
# 4. Chấm từng chỉ số và gộp trụ cột
# =========================================================================== #
def _blank(pillar: str, key: str, label: str, weight: float, note: str) -> dict:
    return {"pillar": pillar, "key": key, "label": label, "weight": float(weight), "value": None, "display": "n/a",
            "score": None, "note": note, "status": "missing", "w_in_pillar": 0.0, "w_effective": 0.0,
            "contribution": 0.0, "used": False}


def _score_specs(pillar: str, specs: list[tuple], data: dict) -> list[dict]:
    out = []
    for key, label, weight, anchors, fmt, note in specs:
        rec = _blank(pillar, key, label, weight, note)
        v = _num(data.get(key))
        if key == "pe" and data.get("pe_loss"):
            rec.update(value=None, display="Lỗ / không có ý nghĩa", score=10.0, status="ok",
                       note="Doanh nghiệp đang lỗ (EPS ≤ 0) nên P/E không có ý nghĩa: chấm thấp.")
        elif v is not None:
            rec.update(value=v, display=_fmt(v, fmt), score=ra.score_from_anchors(v, anchors), status="ok")
            if key.endswith("_rel"):
                rec["display"] = f"{v:.2f} lần"
        out.append(rec)
    return out


def _derive_valuation_inputs(data: dict, last_close: Optional[float]) -> list[str]:
    """Bổ sung pe_rel/pb_rel, nhận diện lỗ, tự tính P/E từ giá nếu thiếu. Trả về ghi chú."""
    notes = []
    eps, pe, pb = _num(data.get("eps")), _num(data.get("pe")), _num(data.get("pb"))
    if (eps is not None and eps <= 0) or (pe is not None and pe <= 0):
        data["pe_loss"] = True
        data.pop("pe", None)
        pe = None
    elif pe is None and eps and eps > 0 and last_close:
        calc = last_close / eps
        if 0 < calc < 200:
            data["pe"] = pe = calc
            notes.append("P/E được tự tính = giá đóng cửa / EPS vì nguồn không có P/E.")
    if pb is not None and pb <= 0:
        data.pop("pb", None)
    for base in ("pe", "pb"):
        cur, med = _num(data.get(base)), _num(data.get(f"{base}_median"))
        if cur and med and med > 0:
            data[f"{base}_rel"] = cur / med
    return notes


def _aggregate(metrics: list[dict]) -> dict:
    """Điểm trụ cột từ các chỉ số có dữ liệu."""
    total_w = sum(m["weight"] for m in metrics) or 1.0
    avail = [m for m in metrics if m["score"] is not None]
    wa = sum(m["weight"] for m in avail)
    coverage = wa / total_w
    if coverage < MIN_PILLAR_COVERAGE:
        return {"score": None, "status": "missing", "coverage": coverage}
    for m in avail:
        m["w_in_pillar"] = m["weight"] / wa
    score = sum(m["score"] * m["weight"] for m in avail) / wa
    return {"score": float(score), "status": "ok" if coverage >= 0.999 else "partial", "coverage": coverage}


# --- Kỹ thuật ---------------------------------------------------------------
def _technical_metrics(price_df: Optional[pd.DataFrame], technical: Optional[dict],
                       warnings: list[str]) -> list[dict]:
    metrics: list[dict] = []
    tech = technical
    if tech is None and price_df is not None:
        try:
            import technical_analysis as ta
            tech = ta.evaluate_signals(ta.compute_indicators(price_df))
        except Exception as exc:                                                # noqa: BLE001
            warnings.append(f"Không tính được tín hiệu kỹ thuật từ TV2 ({type(exc).__name__}: {exc}).")
    reasons = (tech or {}).get("reasons") or []
    for r in reasons:
        s = int(r.get("score", 0))
        rec = _blank("technical", "ta_" + str(r.get("rule", "")).lower().replace(" ", "_").replace("/", "_"),
                     str(r.get("rule", "Quy tắc kỹ thuật")), TECH_RULE_TOTAL_WEIGHT / len(reasons), str(r.get("explain", "")))
        rec.update(value=s, display=str(r.get("label", {1: "Tích cực", 0: "Trung lập", -1: "Tiêu cực"}[max(-1, min(1, s))])),
                   score=TECH_RULE_SCORE[max(-1, min(1, s))], status="ok")
        metrics.append(rec)
    if not reasons:
        for m in MOMENTUM_SPECS:
            pass
    # động lượng giá
    if price_df is not None:
        try:
            close = ra.prepare_prices(price_df)["close"]
        except ValueError as exc:
            close = None
            warnings.append(f"Không đọc được giá cho động lượng: {exc}")
        for key, label, weight, lag, anchors in MOMENTUM_SPECS:
            rec = _blank("technical", key, label, weight, f"Thay đổi giá đóng cửa sau {lag} phiên")
            if close is not None and len(close) > lag:
                v = float(close.iloc[-1] / close.iloc[-lag - 1] - 1)
                rec.update(value=v, display=_fmt(v, "pct"), score=ra.score_from_anchors(v, anchors), status="ok")
            metrics.append(rec)
    return metrics


def _risk_metrics(risk: dict) -> list[dict]:
    out = []
    for m in risk.get("metrics", []):
        rec = _blank("risk", m["key"], m["label"], m["weight"], m["note"])
        rec.update(value=m["value"], display=m["display"], score=m["score"], status="ok")
        out.append(rec)
    return out


# =========================================================================== #
# 5. Hàm chính
# =========================================================================== #
def _default_price_loader(symbol: str) -> pd.DataFrame:
    from data_loader import load_stock_data
    return load_stock_data(symbol)


def load_default_benchmark(index_symbol: str = "VNINDEX") -> pd.Series:
    """Tải VN-Index qua TV1 và đổi sang Series để tính Beta."""
    from data_loader import load_index_data, benchmark_series
    return benchmark_series(load_index_data(index_symbol))


def _resolve_weights(profile: str, weights: Optional[dict]) -> tuple[dict, str]:
    if weights:
        w = {k: float(v) for k, v in weights.items() if k in PILLARS and _num(v) is not None and float(v) > 0}
        if not w:
            raise ValueError("`weights` phải có ít nhất một trụ cột hợp lệ (technical/fundamental/valuation/risk).")
        s = sum(w.values())
        return {k: w.get(k, 0.0) / s for k in PILLARS}, "custom"
    if profile not in WEIGHT_PROFILES:
        raise ValueError(f"profile phải là một trong {list(WEIGHT_PROFILES)}")
    return dict(WEIGHT_PROFILES[profile]), profile


def score_stock(symbol: str, price_df: Optional[pd.DataFrame] = None, benchmark: Any = None,
                fundamentals: Any = None, technical: Optional[dict] = None, profile: str = "balanced",
                weights: Optional[dict] = None, company_type: Optional[str] = None,
                price_loader: Optional[Callable[[str], pd.DataFrame]] = None) -> dict:
    """
    Chấm điểm đầu tư 0-100 cho một mã.

    price_df     DataFrame giá của TV1 (Date/Close/Volume...). None -> tự tải bằng price_loader / data_loader.
    benchmark    Series giá VN-Index (benchmark_series() của TV1) hoặc "auto" để tự tải; None -> bỏ Beta.
    fundamentals đầu ra run_fundamental_analysis() | dict analysis của TV3 | snapshot của TV1 | "auto" | None.
                 None -> bỏ trụ cột Tài chính và Định giá (độ tin cậy giảm).
    technical    kết quả evaluate_signals()/analyze_symbol() của TV2 nếu dashboard đã có (tránh tính lại).
    profile      "balanced" | "growth" | "conservative"; hoặc truyền `weights` tùy chỉnh.
    Raise ScoringError nếu có ít hơn 2 trụ cột đủ dữ liệu.
    """
    symbol = str(symbol).strip().upper()
    if not symbol:
        raise ScoringError("Chưa có mã cổ phiếu.")
    nominal, profile_used = _resolve_weights(profile, weights)
    warnings: list[str] = []
    notes: list[str] = []

    # ---- dữ liệu đầu vào
    if price_df is None:
        try:
            price_df = (price_loader or _default_price_loader)(symbol)
        except Exception as exc:                                                # noqa: BLE001
            raise ScoringError(f"Không tải được giá của {symbol}: {exc}") from exc
    if isinstance(benchmark, str) and benchmark == "auto":
        try:
            benchmark = load_default_benchmark()
        except Exception as exc:                                                # noqa: BLE001
            benchmark = None
            warnings.append(f"Không tải được VN-Index để tính Beta ({type(exc).__name__}).")
    if isinstance(fundamentals, str) and fundamentals == "auto":
        fundamentals, w = default_fundamentals_loader(symbol, company_type)
        warnings += w

    fdata, fnotes = normalize_fundamentals(fundamentals)
    notes += fnotes
    meta = fdata.pop("_meta", {})
    ctype = detect_company_type(symbol, company_type or meta.get("company_type")
                                or ("financial" if meta.get("is_financial") and symbol not in BANK_TICKERS else None))
    ctype_label = {"bank": "ngân hàng", "financial": "tài chính phi ngân hàng", "regular": "doanh nghiệp phi tài chính"}[ctype]

    # ---- rủi ro
    risk: dict = {"available": False, "metrics": [], "warnings": [], "notes": []}
    last_close, as_of = None, None
    try:
        risk = ra.assess_risk(price_df, benchmark)
        warnings += risk["warnings"]
        notes += risk["notes"]
        pp = ra.prepare_prices(price_df)
        last_close, as_of = float(pp["close"].iloc[-1]) * pp.attrs["price_multiplier"], pp.index[-1]
    except ValueError as exc:
        warnings.append(f"Không đánh giá được rủi ro: {exc}")

    # ---- các chỉ số theo trụ cột
    pillar_metrics: dict[str, list[dict]] = {
        "technical": _technical_metrics(price_df, technical, warnings),
        "risk": _risk_metrics(risk),
    }
    if fdata:
        vnotes = _derive_valuation_inputs(fdata, last_close)
        notes += vnotes
    else:
        warnings.append("Không có dữ liệu tài chính/định giá: bỏ qua hai trụ cột này.")
    pillar_metrics["fundamental"] = _score_specs("fundamental", FUNDAMENTAL_SPECS[ctype], fdata)
    pillar_metrics["valuation"] = _score_specs("valuation", VALUATION_SPECS[ctype], fdata)

    # ---- gộp trụ cột
    pillars: dict[str, dict] = {}
    for p in PILLARS:
        agg = _aggregate(pillar_metrics[p])
        pillars[p] = {"label": PILLAR_LABELS[p], "weight_nominal": nominal[p], "weight_used": 0.0, **agg,
                      "metrics_missing": [m["label"] for m in pillar_metrics[p] if m["score"] is None]}
        if agg["status"] == "missing" and nominal[p] > 0:
            if p in ("fundamental", "valuation") and not fdata:
                pass                                    # đã cảnh báo ở trên
            else:
                warnings.append(f"Trụ cột {PILLAR_LABELS[p]} thiếu dữ liệu (độ phủ {agg['coverage']:.0%}) nên bị bỏ qua.")
        elif agg["status"] == "partial":
            warnings.append(f"Trụ cột {PILLAR_LABELS[p]} chỉ có {agg['coverage']:.0%} chỉ số: "
                            f"thiếu {', '.join(pillars[p]['metrics_missing'])}.")
    usable = [p for p in PILLARS if pillars[p]["status"] != "missing" and nominal[p] > 0]
    if len(usable) < MIN_PILLARS:
        raise ScoringError(f"Chỉ có {len(usable)} trụ cột đủ dữ liệu cho {symbol} (cần ≥ {MIN_PILLARS}). "
                           + " ".join(warnings[-3:]))
    wsum = sum(nominal[p] for p in usable)
    for p in usable:
        pillars[p]["weight_used"] = nominal[p] / wsum
        for m in pillar_metrics[p]:
            if m["score"] is not None:
                m["used"] = True
                m["w_effective"] = pillars[p]["weight_used"] * m["w_in_pillar"]
                m["contribution"] = m["w_effective"] * (m["score"] - 50.0)
    total = sum(pillars[p]["weight_used"] * pillars[p]["score"] for p in usable)
    coverage_weight = wsum / sum(nominal.values())

    # ---- nhãn + chốt chặn
    label = classify_score(total)
    caps: list[str] = []
    risk_sc = pillars["risk"]["score"]
    if risk_sc is not None and risk_sc < RISK_CAP_BELOW:
        caps.append(f"Điểm rủi ro {risk_sc:.0f} < {RISK_CAP_BELOW} (rủi ro rất cao)")
    liq = risk.get("liquidity_score")
    if liq is not None and liq < LIQUIDITY_CAP_BELOW:
        caps.append(f"Điểm thanh khoản {liq:.0f} < {LIQUIDITY_CAP_BELOW} (thanh khoản rất thấp)")
    fsc = pillars["fundamental"]["score"]
    if fsc is not None and fsc < FUNDAMENTAL_CAP_BELOW:
        caps.append(f"Điểm tài chính {fsc:.0f} < {FUNDAMENTAL_CAP_BELOW} (nền tảng tài chính yếu)")
    if coverage_weight < 0.5:
        caps.append(f"Chỉ {coverage_weight:.0%} trọng số có dữ liệu")
    raw_label = label
    if caps:
        label = _cap_label(label, "Trung lập")
    if label != raw_label:
        notes.append(f"Nhãn được hạ từ '{raw_label}' xuống '{label}' do chốt chặn: " + "; ".join(caps) + ".")

    all_ok = all(pillars[p]["status"] == "ok" for p in usable)
    confidence = ("Cao" if coverage_weight >= 0.99 and all_ok and not any("cũ hơn 7 ngày" in w for w in warnings)
                  else "Trung bình" if coverage_weight >= 0.7 else "Thấp")

    # ---- bảng + giải thích
    metrics_flat = [m for p in PILLARS for m in pillar_metrics[p]]
    for m in metrics_flat:
        m["assessment"] = _quality(m["score"]) if m["score"] is not None else "thiếu dữ liệu"
    ups = sorted([m for m in metrics_flat if m["used"] and m["contribution"] > 0], key=lambda m: -m["contribution"])
    downs = sorted([m for m in metrics_flat if m["used"] and m["contribution"] < 0], key=lambda m: m["contribution"])

    def _driver(m: dict) -> dict:
        return {"pillar": PILLAR_LABELS[m["pillar"]], "metric": m["label"], "value": m["display"],
                "score": round(m["score"], 1), "contribution": round(m["contribution"], 2),
                "text": f"{m['label']} {m['display']} - mức {m['assessment']} ({m['score']:.0f}/100): "
                        f"{'+' if m['contribution'] > 0 else ''}{m['contribution']:.2f} điểm vào tổng. {m['note']}"}

    result = {
        "symbol": symbol, "company_type": ctype, "as_of": as_of.strftime("%Y-%m-%d") if as_of is not None else None,
        "profile": profile_used, "weights_nominal": nominal,
        "weights_used": {p: pillars[p]["weight_used"] for p in PILLARS},
        "total_score": float(total), "label": label, "raw_label": raw_label, "label_meaning": LABEL_MEANING[label],
        "confidence": confidence, "coverage_weight": float(coverage_weight), "caps": caps,
        "pillars": pillars,
        "metrics": metrics_flat,
        "drivers_up": [_driver(m) for m in ups[:5]],
        "drivers_down": [_driver(m) for m in downs[:5]],
        "risk": risk, "warnings": list(dict.fromkeys(warnings)), "notes": list(dict.fromkeys(notes)),
        "weight_rationale": WEIGHT_RATIONALE, "methodology": METHODOLOGY, "disclaimer": DISCLAIMER,
    }
    result["summary"] = _build_summary(result, ctype_label)
    result["pillar_table"] = pillar_table(result)
    result["metric_table"] = metric_table(result)
    return result


# =========================================================================== #
# 6. Bảng & diễn giải
# =========================================================================== #
def pillar_table(res: dict) -> pd.DataFrame:
    rows = []
    for p in PILLARS:
        d = res["pillars"][p]
        rows.append({
            "Trụ cột": d["label"], "Điểm (0-100)": None if d["score"] is None else round(d["score"], 1),
            "Trọng số danh nghĩa": f"{d['weight_nominal']:.0%}", "Trọng số áp dụng": f"{d['weight_used']:.0%}",
            "Độ phủ dữ liệu": f"{d['coverage']:.0%}",
            "Trạng thái": {"ok": "Đủ", "partial": "Thiếu một phần", "missing": "Bỏ qua (thiếu dữ liệu)"}[d["status"]],
            "Đóng góp vào tổng (điểm)": round(sum(m["contribution"] for m in res["metrics"] if m["pillar"] == p), 2),
        })
    return pd.DataFrame(rows)


def metric_table(res: dict) -> pd.DataFrame:
    rows = []
    for m in res["metrics"]:
        rows.append({
            "Trụ cột": PILLAR_LABELS[m["pillar"]], "Chỉ số": m["label"], "Giá trị": m["display"],
            "Điểm (0-100)": None if m["score"] is None else round(m["score"], 1), "Đánh giá": m["assessment"],
            "Trọng số hiệu dụng": f"{m['w_effective']:.1%}" if m["used"] else "-",
            "Đóng góp (điểm)": round(m["contribution"], 2) if m["used"] else None, "Diễn giải": m["note"],
        })
    return pd.DataFrame(rows)


def _build_summary(res: dict, ctype_label: str) -> str:
    ps = {p: d for p, d in res["pillars"].items() if d["score"] is not None}
    best = max(ps, key=lambda p: ps[p]["score"])
    worst = min(ps, key=lambda p: ps[p]["score"])
    s = (f"{res['symbol']} ({ctype_label}) đạt {res['total_score']:.1f}/100 - mức \"{res['label']}\" "
         f"(độ tin cậy {res['confidence'].lower()}, hồ sơ trọng số: {PROFILE_LABELS.get(res['profile'], 'tùy chỉnh')}). ")
    if best != worst:
        s += (f"Trụ cột mạnh nhất là {PILLAR_LABELS[best]} ({ps[best]['score']:.0f}), yếu nhất là "
              f"{PILLAR_LABELS[worst]} ({ps[worst]['score']:.0f}). ")
    if res["drivers_up"]:
        s += "Kéo điểm lên: " + "; ".join(f"{d['metric']} {d['value']}" for d in res["drivers_up"][:3]) + ". "
    if res["drivers_down"]:
        s += "Kéo điểm xuống: " + "; ".join(f"{d['metric']} {d['value']}" for d in res["drivers_down"][:3]) + ". "
    if res["caps"]:
        s += f"Nhãn bị giới hạn ở '{res['label']}' do: " + "; ".join(res["caps"]) + ". "
    missing = [PILLAR_LABELS[p] for p in PILLARS if res["pillars"][p]["status"] == "missing"]
    if missing:
        s += "Thiếu dữ liệu nên bỏ qua trụ cột: " + ", ".join(missing) + ". "
    return s.strip()


def explain_text(res: dict) -> str:
    """Báo cáo giải thích dạng Markdown (dùng cho dashboard/PDF)."""
    L = [f"## {res['symbol']}: {res['total_score']:.1f}/100 - {res['label']}", "", res["summary"], "",
         f"*{res['label_meaning']}*", "", "### Các trụ cột"]
    for p in PILLARS:
        d = res["pillars"][p]
        sc = "n/a" if d["score"] is None else f"{d['score']:.0f}/100"
        L.append(f"- **{d['label']}**: {sc} (trọng số {d['weight_used']:.0%}) - {WEIGHT_RATIONALE[p]}")
    L += ["", "### Yếu tố kéo điểm LÊN"] + [f"- {d['text']}" for d in res["drivers_up"]]
    L += ["", "### Yếu tố kéo điểm XUỐNG"] + [f"- {d['text']}" for d in res["drivers_down"]]
    if res["warnings"]:
        L += ["", "### Cảnh báo dữ liệu"] + [f"- {w}" for w in res["warnings"]]
    if res["notes"]:
        L += ["", "### Ghi chú"] + [f"- {w}" for w in res["notes"]]
    L += ["", f"> {res['disclaimer']}"]
    return "\n".join(L)


def to_jsonable(res: dict) -> dict:
    """Bỏ DataFrame/đối tượng không serialize được - dùng khi lưu JSON hoặc truyền sang TV6."""
    out = {k: v for k, v in res.items() if k not in ("pillar_table", "metric_table")}
    return _clean(out)


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# =========================================================================== #
# 7. Chấm nhiều mã
# =========================================================================== #
def score_many(symbols: list[str], benchmark: Any = "auto", profile: str = "balanced",
               weights: Optional[dict] = None, use_fundamentals: bool = True,
               price_loader: Optional[Callable[[str], pd.DataFrame]] = None,
               fundamentals_loader: Optional[Callable[[str], Any]] = None,
               company_types: Optional[dict[str, str]] = None) -> tuple[pd.DataFrame, dict, dict]:
    """
    Chấm nhiều mã; mã lỗi không làm hỏng cả lô.
    Trả về (bảng xếp hạng, {mã: kết quả chi tiết}, {mã: thông báo lỗi}).
    fundamentals_loader(symbol) -> obj cơ bản (hoặc (obj, warnings)); mặc định dùng TV3 -> snapshot TV1.
    """
    if isinstance(benchmark, str) and benchmark == "auto":
        try:
            benchmark = load_default_benchmark()
        except Exception:                                                       # noqa: BLE001
            benchmark = None
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for raw_sym in symbols:
        sym = str(raw_sym).strip().upper()
        if not sym or sym in results:
            continue
        try:
            ctype = (company_types or {}).get(sym)
            fund, extra = None, []
            if use_fundamentals:
                loaded = (fundamentals_loader or (lambda s: default_fundamentals_loader(s, ctype)))(sym)
                fund, extra = loaded if isinstance(loaded, tuple) and len(loaded) == 2 and isinstance(loaded[1], list) else (loaded, [])
            res = score_stock(sym, benchmark=benchmark, fundamentals=fund, profile=profile, weights=weights,
                              company_type=ctype, price_loader=price_loader)
            res["warnings"] = list(dict.fromkeys(extra + res["warnings"]))
            results[sym] = res
        except Exception as exc:                                                # noqa: BLE001
            errors[sym] = f"{type(exc).__name__}: {exc}"
    return ranking_table(results), results, errors


def ranking_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for sym, r in results.items():
        row = {"Mã": sym, "Điểm tổng": round(r["total_score"], 1), "Phân loại": r["label"], "Độ tin cậy": r["confidence"]}
        for p in PILLARS:
            sc = r["pillars"][p]["score"]
            row[PILLAR_LABELS[p]] = None if sc is None else round(sc, 1)
        row["Kéo lên chính"] = r["drivers_up"][0]["metric"] if r["drivers_up"] else ""
        row["Kéo xuống chính"] = r["drivers_down"][0]["metric"] if r["drivers_down"] else ""
        row["Số cảnh báo"] = len(r["warnings"])
        row["Dữ liệu đến"] = r["as_of"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("Điểm tổng", ascending=False).reset_index(drop=True)
    df.insert(0, "Hạng", df.index + 1)
    return df


# =========================================================================== #
# 8. CLI:  python investment_scoring.py ACB FPT HPG --profile balanced --out scores.csv
# =========================================================================== #
def _main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="TV4 - chấm điểm đầu tư 0-100")
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--profile", default="balanced", choices=list(WEIGHT_PROFILES))
    ap.add_argument("--no-fundamentals", action="store_true")
    ap.add_argument("--out", default="investment_scores.csv")
    a = ap.parse_args()
    table, results, errors = score_many(a.symbols, profile=a.profile, use_fundamentals=not a.no_fundamentals)
    if not table.empty:
        table.to_csv(a.out, index=False, encoding="utf-8-sig")
        print(table.to_string(index=False))
        with open(a.out.rsplit(".", 1)[0] + "_details.json", "w", encoding="utf-8") as f:
            json.dump({s: to_jsonable(r) for s, r in results.items()}, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu {a.out} và file chi tiết *_details.json")
    for s, e in errors.items():
        print(f"LỖI {s}: {e}")


if __name__ == "__main__":
    _main()