# -*- coding: utf-8 -*-
"""
data_cleaning.py  -  TV1: Làm sạch, chuẩn hóa và kiểm tra chất lượng dữ liệu
=============================================================================
Đầu ra chuẩn của dự án (mọi thành viên TV2-TV6 đều nhận đúng định dạng này):

    DataFrame giá:  Date | Symbol | Open | High | Low | Close | Volume
        - Date   : datetime64[ns] (không múi giờ), tăng dần, không trùng, không cuối tuần
        - Symbol : mã viết hoa, ví dụ "VRE"
        - Giá    : ĐƠN VỊ ĐỒNG (VND). Nguồn trả về nghìn đồng (25.5) sẽ được tự đổi thành 25,500
        - Volume : số cổ phiếu (int)
        - df.attrs chứa nguồn, thời điểm cập nhật, báo cáo chất lượng

Nguyên tắc: KHÔNG tự bịa số liệu. Chỉ sửa những lỗi có thể suy ra chắc chắn
(High < Close, thiếu Open...), mọi thay đổi đều được ghi vào QualityReport.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Ngoại lệ dùng chung (TV5 chỉ cần bắt DataError là đủ)
# --------------------------------------------------------------------------
class DataError(Exception):
    """Lỗi dữ liệu nói chung (thông điệp tiếng Việt, hiển thị thẳng cho người dùng)."""


class SymbolError(DataError):
    """Mã chứng khoán sai hoặc không có dữ liệu."""


class DateRangeError(DataError):
    """Khoảng thời gian không hợp lệ."""


class DataSourceError(DataError):
    """Không kết nối/đọc được nguồn dữ liệu."""


PRICE_COLUMNS = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
VN_TZ = "Asia/Ho_Chi_Minh"


# --------------------------------------------------------------------------
# Tiện ích văn bản
# --------------------------------------------------------------------------
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def normalize_key(s: str) -> str:
    """'Giá đóng cửa (VND)' -> 'giadongcuavnd' (dùng để so khớp tên cột bất kể dấu/ký tự lạ)."""
    return re.sub(r"[^a-z0-9]", "", strip_accents(s).lower())


_ALIASES = {
    "Date": ["date", "time", "ngay", "ngaygiaodich", "tradingdate", "datetime", "thoigian", "t"],
    "Open": ["open", "giamocua", "giamo", "mocua", "o", "openprice"],
    "High": ["high", "giacaonhat", "caonhat", "cao", "h", "highprice"],
    "Low": ["low", "giathapnhat", "thapnhat", "thap", "l", "lowprice"],
    "Close": ["close", "giadongcua", "dongcua", "giadong", "c", "closeprice"],
    "Volume": ["volume", "khoiluong", "klgd", "khoiluonggiaodich", "kl", "v", "tongkl",
               "klkhoplenh", "khoiluongkhoplenh"],
    "Symbol": ["symbol", "ticker", "ma", "macp", "mack", "macophieu", "code", "cp"],
}
_ALIAS_LOOKUP = {a: std for std, lst in _ALIASES.items() for a in lst}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột (Anh/Việt, hoa/thường, có dấu) về Date/Open/High/Low/Close/Volume/Symbol."""
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = ["_".join(str(x) for x in c if str(x) != "") for c in d.columns]
    mapping, used = {}, set()
    for c in d.columns:
        std = _ALIAS_LOOKUP.get(normalize_key(c))
        if std and std not in used:
            mapping[c] = std
            used.add(std)
    if "Close" not in used:   # Chỉ dùng giá điều chỉnh khi không có giá đóng cửa thường
        for c in d.columns:
            if normalize_key(c) in ("adjclose", "giadieuchinh", "closeadj"):
                mapping[c] = "Close"
                break
    d = d.rename(columns=mapping)
    return d.loc[:, ~d.columns.duplicated()]


# --------------------------------------------------------------------------
# Chuyển kiểu: số và ngày
# --------------------------------------------------------------------------
_NA_TEXT = {"", "-", "--", "nan", "none", "n/a", "na", "null"}


def parse_number(x) -> float:
    """Đọc số từ nhiều kiểu: 1,234.5 | 1.234,5 | 25.500 (nhiều dấu chấm) | (123) | 12% | ' 1 234 '."""
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("\xa0", "").replace(" ", "")
    if s.lower() in _NA_TEXT:
        return float("nan")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("%", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", "") if (len(parts) > 2 or len(parts[-1]) == 3) else s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    return -v if neg else v


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.map(parse_number).astype(float)


def parse_dates(series: pd.Series) -> pd.Series:
    """Chuẩn hóa về datetime không múi giờ, đặt về 00:00. Hỗ trợ epoch giây/mili-giây, dd/mm/yyyy, ISO."""
    s = series
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, errors="coerce")
    elif pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.dropna().abs().median() > 1e11 else "s"
        out = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
        out = out.dt.tz_convert(VN_TZ)
    else:
        txt = s.astype(str).str.strip()
        sample = txt.dropna().head(50)
        slash = sample.str.contains(r"^\d{1,2}[/-]\d{1,2}[/-]\d{4}", regex=True).mean() > 0.5
        out = pd.to_datetime(txt, errors="coerce", dayfirst=bool(slash))
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert(VN_TZ).dt.tz_localize(None)
    return out.dt.normalize()


# --------------------------------------------------------------------------
# Báo cáo chất lượng
# --------------------------------------------------------------------------
@dataclass
class QualityReport:
    symbol: str = ""
    source: str = "unknown"
    fetched_at: str = ""
    n_raw: int = 0
    n_clean: int = 0
    start: str = ""
    end: str = ""
    price_unit_detected: str = ""
    issues: List[dict] = field(default_factory=list)   # level: info | warning | error
    score: float = 100.0
    grade: str = "Tốt"

    def add(self, level: str, code: str, message: str, count: int = 0, examples: Optional[list] = None):
        self.issues.append({"level": level, "code": code, "message": message, "count": int(count),
                            "examples": [str(e) for e in (examples or [])][:5]})

    def finalize(self, penalty: float) -> "QualityReport":
        self.score = float(max(0.0, min(100.0, 100.0 - penalty)))
        self.grade = "Tốt" if self.score >= 90 else "Khá" if self.score >= 75 else "Trung bình" if self.score >= 60 else "Kém"
        return self

    @property
    def warnings(self) -> List[str]:
        return [i["message"] for i in self.issues if i["level"] in ("warning", "error")]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QualityReport":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.issues, columns=["level", "code", "message", "count", "examples"])

    def to_markdown(self) -> str:
        """Dùng cho báo cáo PDF (TV6): nguồn, thời điểm cập nhật, chất lượng dữ liệu."""
        lines = [f"- Mã: {self.symbol} | Nguồn: {self.source} | Cập nhật lúc: {self.fetched_at}",
                 f"- Khoảng dữ liệu: {self.start} → {self.end} ({self.n_clean:,} phiên; thô {self.n_raw:,} dòng)",
                 f"- Điểm chất lượng dữ liệu: {self.score:.0f}/100 ({self.grade})"]
        for i in self.issues:
            if i["level"] != "info" or i["count"]:
                lines.append(f"  • [{i['level']}] {i['message']}")
        return "\n".join(lines)


def _now_str() -> str:
    return pd.Timestamp.now(tz=VN_TZ).strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# Làm sạch dữ liệu giá
# --------------------------------------------------------------------------
def detect_price_unit(close: pd.Series) -> str:
    """'thousand_vnd' nếu giá trung vị < 1000 (ví dụ 25.5 = 25,500 đồng), ngược lại 'vnd'."""
    med = float(close.tail(250).median())
    return "thousand_vnd" if med < 1000 else "vnd"


def clean_price_data(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    source: str = "unknown",
    fetched_at: Optional[str] = None,
    price_unit: str = "auto",
    drop_weekends: bool = True,
) -> Tuple[pd.DataFrame, QualityReport]:
    """Làm sạch DataFrame giá thô -> (DataFrame chuẩn, QualityReport). Ném DataError nếu không dùng được."""
    if df is None or len(df) == 0:
        raise DataError("Dữ liệu giá rỗng.")
    rep = QualityReport(symbol=(symbol or "").upper(), source=source, fetched_at=fetched_at or _now_str(),
                        n_raw=len(df))
    penalty = 0.0

    d = standardize_columns(df)
    for need in ("Date", "Close"):
        if need not in d.columns:
            raise DataError(f"Dữ liệu thiếu cột bắt buộc '{need}'. Các cột hiện có: {list(df.columns)[:10]}")

    # ---- 1. Ngày ----
    d["Date"] = parse_dates(d["Date"])
    bad = int(d["Date"].isna().sum())
    if bad:
        rep.add("warning", "BAD_DATE", f"Loại {bad} dòng có ngày không đọc được.", bad)
        penalty += min(10, bad / len(d) * 100)
        d = d.dropna(subset=["Date"])

    # ---- 2. Số ----
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in d.columns:
            d[c] = to_numeric(d[c])
    nonpos = int((d["Close"] <= 0).sum())
    nan_close = int(d["Close"].isna().sum())
    if nonpos or nan_close:
        rep.add("warning", "BAD_CLOSE", f"Loại {nonpos + nan_close} dòng có giá đóng cửa thiếu hoặc ≤ 0.", nonpos + nan_close)
        penalty += min(20, (nonpos + nan_close) / len(d) * 100)
    d = d[d["Close"] > 0]

    # ---- 3. Trùng lặp, thứ tự ----
    dups = int(d["Date"].duplicated().sum())
    if dups:
        rep.add("warning", "DUPLICATE", f"Loại {dups} dòng trùng ngày (giữ dòng cuối).", dups)
        penalty += min(5, dups * 0.5)
    d = d.drop_duplicates(subset="Date", keep="last").sort_values("Date")

    if drop_weekends:
        wk = d["Date"].dt.dayofweek >= 5
        if wk.any():
            rep.add("info", "WEEKEND", f"Loại {int(wk.sum())} dòng rơi vào thứ Bảy/Chủ nhật (thị trường nghỉ).", int(wk.sum()))
            d = d[~wk]

    if len(d) == 0:
        raise DataError("Không còn dòng dữ liệu hợp lệ sau khi làm sạch.")

    # ---- 4. Thiếu Open/High/Low/Volume, OHLC không nhất quán ----
    for c in ["Open", "High", "Low"]:
        if c not in d.columns:
            d[c] = np.nan
    miss = int(d[["Open", "High", "Low"]].isna().any(axis=1).sum())
    if miss:
        rep.add("warning", "MISSING_OHL", f"{miss} dòng thiếu Open/High/Low: điền bằng giá đóng cửa của chính phiên đó.", miss)
        penalty += min(10, miss / len(d) * 100)
        for c in ["Open", "High", "Low"]:
            d[c] = d[c].fillna(d["Close"])
    hi = d[["Open", "High", "Low", "Close"]].max(axis=1)
    lo = d[["Open", "High", "Low", "Close"]].min(axis=1)
    inconsistent = int(((d["High"] < hi) | (d["Low"] > lo)).sum())
    if inconsistent:
        rep.add("warning", "OHLC_INCONSISTENT", f"Sửa {inconsistent} dòng có High < max(O,C) hoặc Low > min(O,C).", inconsistent)
        penalty += min(10, inconsistent / len(d) * 100)
        d["High"], d["Low"] = hi, lo
    if "Volume" not in d.columns:
        d["Volume"] = 0.0
        rep.add("warning", "NO_VOLUME", "Nguồn không có khối lượng: đặt Volume = 0.", len(d))
        penalty += 5
    else:
        nv = int((d["Volume"].isna() | (d["Volume"] < 0)).sum())
        if nv:
            rep.add("warning", "BAD_VOLUME", f"{nv} dòng khối lượng thiếu/âm: đặt = 0.", nv)
            penalty += min(5, nv / len(d) * 100)
        d["Volume"] = d["Volume"].where(d["Volume"] >= 0).fillna(0.0)

    # ---- 5. Đơn vị giá -> VND ----
    unit = detect_price_unit(d["Close"]) if price_unit == "auto" else price_unit
    if unit not in ("vnd", "thousand_vnd"):
        raise DataError("price_unit phải là 'auto', 'vnd' hoặc 'thousand_vnd'.")
    rep.price_unit_detected = unit
    if unit == "thousand_vnd":
        for c in ["Open", "High", "Low", "Close"]:
            d[c] = d[c] * 1000.0
        rep.add("info", "UNIT", "Nguồn trả giá theo nghìn đồng: đã nhân 1,000 để chuẩn hóa về đồng (VND).", 0)

    d = d.reset_index(drop=True)
    if symbol:
        sym_value = str(symbol).upper()
    elif "Symbol" in d.columns and d["Symbol"].notna().any():
        sym_value = str(d["Symbol"].dropna().iloc[0]).strip().upper()
    else:
        sym_value = ""
    d["Symbol"] = sym_value
    rep.symbol = rep.symbol or sym_value
    d["Volume"] = d["Volume"].round().astype("int64")
    d = d[PRICE_COLUMNS]

    # ---- 6. Kiểm tra bất thường (chỉ cảnh báo, không tự xóa) ----
    ret = d["Close"].pct_change()
    big = d.loc[ret.abs() > 0.15, "Date"].dt.strftime("%Y-%m-%d").tolist()
    if big:
        rep.add("warning", "BIG_MOVE", f"{len(big)} phiên biến động >15%: có thể do chia tách/cổ tức bằng cổ phiếu chưa điều chỉnh "
                "hoặc lỗi dữ liệu.", len(big), big)
        penalty += min(20, len(big) * 5)
    ticks = [d["Date"].iloc[i].strftime("%Y-%m-%d") for i in range(1, len(d) - 1)
             if abs(ret.iloc[i]) > 0.3 and abs(d["Close"].iloc[i + 1] / d["Close"].iloc[i - 1] - 1) < 0.05]
    if ticks:
        rep.add("error", "BAD_TICK", f"{len(ticks)} phiên nghi lỗi giá (tăng/giảm >30% rồi quay lại ngay).", len(ticks), ticks)
        penalty += min(20, len(ticks) * 10)

    gaps = d["Date"].diff().dt.days
    gap_rows = d.loc[gaps > 10, "Date"].dt.strftime("%Y-%m-%d").tolist()
    if gap_rows:
        rep.add("warning", "GAP", f"{len(gap_rows)} khoảng trống >10 ngày lịch giữa hai phiên (thiếu dữ liệu hoặc tạm ngừng giao dịch).",
                len(gap_rows), gap_rows)
        penalty += min(10, len(gap_rows) * 2)

    zero = (d["Volume"] == 0).astype(int)
    streak = zero.groupby((zero == 0).cumsum()).cumsum().max() if len(zero) else 0
    if streak >= 5:
        rep.add("info", "ZERO_VOLUME", f"Có chuỗi {int(streak)} phiên liên tiếp không có khối lượng (nghi tạm ngừng giao dịch).", int(streak))

    n = len(d)
    if n < 60:
        rep.add("warning", "SHORT_HISTORY", f"Chỉ có {n} phiên: quá ít để tính chỉ báo (SMA50, rủi ro...).", n)
        penalty += 30
    elif n < 252:
        rep.add("info", "SHORT_HISTORY", f"Chỉ có {n} phiên (<1 năm): chỉ số rủi ro kém tin cậy hơn.", n)
        penalty += 8

    today = pd.Timestamp.now(tz=VN_TZ).tz_localize(None).normalize()
    lag = (today - d["Date"].iloc[-1]).days
    if lag > 7:
        rep.add("warning", "STALE", f"Phiên cuối cùng là {d['Date'].iloc[-1]:%d/%m/%Y}, cách hôm nay {lag} ngày: dữ liệu có thể chưa cập nhật.", lag)
        penalty += 10

    dropped = rep.n_raw - n
    if dropped > 0:
        penalty += min(20, dropped / max(rep.n_raw, 1) * 100 * 0.5)
    rep.n_clean, rep.start, rep.end = n, f"{d['Date'].iloc[0]:%Y-%m-%d}", f"{d['Date'].iloc[-1]:%Y-%m-%d}"
    rep.finalize(penalty)
    d.attrs.update({"symbol": rep.symbol, "source": source, "fetched_at": rep.fetched_at,
                    "quality": rep.to_dict(), "price_unit": "VND"})
    return d, rep


def validate_for_analysis(df: pd.DataFrame, min_rows: int = 60) -> None:
    """Các thành viên khác gọi hàm này đầu hàm của mình để có thông báo lỗi rõ ràng."""
    missing = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
    if missing:
        raise DataError(f"DataFrame giá thiếu cột: {missing}")
    if len(df) < min_rows:
        raise DataError(f"Chỉ có {len(df)} phiên, cần tối thiểu {min_rows} phiên.")


# --------------------------------------------------------------------------
# Báo cáo tài chính (đầu ra của vnstock có thể là MultiIndex, wide theo kỳ)
# --------------------------------------------------------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = ["|".join(str(x).strip() for x in c if str(x).strip() and not str(x).startswith("Unnamed"))
                     for c in d.columns]
    else:
        d.columns = [str(c).strip() for c in d.columns]
    return d.loc[:, ~d.columns.duplicated()]


def clean_financial_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Làm phẳng cột, bỏ dòng/cột rỗng hoàn toàn, ép kiểu số cho cột nào đọc được >=80% giá trị."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = flatten_columns(df.reset_index(drop=True) if not isinstance(df.index, pd.RangeIndex) and df.index.name is None else df)
    d = d.dropna(how="all").dropna(axis=1, how="all")
    for c in d.columns:
        if d[c].dtype == object or str(d[c].dtype).startswith(("str", "string")):
            parsed = d[c].map(parse_number)
            filled = d[c].notna().sum()
            if filled and parsed.notna().sum() / filled >= 0.8:
                d[c] = parsed.astype(float)
    return d.reset_index(drop=True)


def find_column(df: pd.DataFrame, candidates: List[str], exact: bool = False) -> Optional[str]:
    """Tìm cột theo danh sách tên ứng viên (so khớp không dấu). Ưu tiên theo thứ tự ứng viên."""
    keys = {c: normalize_key(str(c).split("|")[-1]) for c in df.columns}
    for cand in candidates:
        k = normalize_key(cand)
        for col, nk in keys.items():
            if nk == k or (not exact and k in nk):
                return col
    return None
