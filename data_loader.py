# -*- coding: utf-8 -*-
"""
data_loader.py  -  TV1: Thu thập dữ liệu
=========================================
Nguồn dữ liệu:
    Giá OHLCV      : DNSE (Entrade) https://services.entrade.com.vn/chart-api/v2/ohlcs/stock
                     -> dự phòng 1: vnstock (VCI/TCBS)  -> dự phòng 2: file CSV/Excel nội bộ
                     -> dự phòng 3: cache cũ trên đĩa (có cảnh báo)
    Báo cáo tài chính, chỉ số, thông tin doanh nghiệp : vnstock

Hàm chính cho các thành viên khác:
    load_stock_data(symbol, start, end)            -> DataFrame chuẩn (xem data_cleaning.py)
    load_stock_data(..., return_report=True)       -> (DataFrame, QualityReport)
    load_index_data("VNINDEX", start, end)         -> DataFrame chỉ số (làm benchmark tính Beta)
    benchmark_series(index_df)                     -> Series Close theo ngày, đưa thẳng cho TV4
    load_financial_data(symbol, period="year")     -> FinancialData (4 bảng + overview)
    get_fundamentals_snapshot(fin)                 -> dict chỉ số mới nhất (cầu nối cho TV3/TV4)

Mọi lỗi người dùng nhìn thấy đều là DataError (SymbolError, DateRangeError, DataSourceError)
với thông điệp tiếng Việt; TV5 chỉ cần `except DataError as e: st.error(str(e))`.

Chạy thử/kiểm tra kết nối:   python data_loader.py --check
Tạo stock_prices.csv:        python data_loader.py --symbols VRE VCB FPT --start 2021-01-01 --out stock_prices.csv
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # báo lỗi thân thiện khi thực sự cần gọi mạng
    requests = None

from data_cleaning import (
    DataError, DataSourceError, DateRangeError, SymbolError, PRICE_COLUMNS, VN_TZ,
    QualityReport, clean_financial_df, clean_price_data, find_column, normalize_key,
    standardize_columns,
)

log = logging.getLogger("data_loader")

DNSE_URL = "https://services.entrade.com.vn/chart-api/v2/ohlcs/{kind}"   # kind = stock | index
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StockAnalysisBot/1.0)", "Accept": "application/json"}
CACHE_DIR = Path("data_cache")
CHUNK_DAYS = 730            # chia khoảng thời gian dài thành các đoạn ~2 năm để tránh bị cắt bớt dữ liệu
DEFAULT_YEARS = 5
TIMEOUT = (5, 25)           # (kết nối, đọc) giây

# Mã thuộc nhóm tài chính (ngân hàng/chứng khoán/bảo hiểm): TV3/TV4 dùng bộ chỉ số riêng.
FINANCIAL_TICKERS = {
    "VCB", "BID", "CTG", "TCB", "MBB", "VPB", "ACB", "STB", "HDB", "TPB", "VIB", "SHB", "LPB", "MSB", "OCB",
    "EIB", "SSB", "NAB", "BAB", "ABB", "BVB", "KLB", "PGB", "SGB", "VBB", "VAB", "NVB",
    "SSI", "VND", "VCI", "HCM", "SHS", "MBS", "FTS", "BSI", "CTS", "AGR", "VIX", "ORS", "APG",
    "BVH", "BMI", "PVI", "MIG",
}


# ==========================================================================
# 1. Chuẩn hóa tham số đầu vào
# ==========================================================================
def normalize_symbol(symbol: str) -> str:
    """' vre.vn ' / 'HOSE:VRE' -> 'VRE'. Ném SymbolError nếu không hợp lệ."""
    if symbol is None or not str(symbol).strip():
        raise SymbolError("Chưa nhập mã chứng khoán.")
    s = str(symbol).strip().upper()
    s = re.sub(r"^(HOSE|HSX|HNX|UPCOM|UPC)[:\-\s]+", "", s)
    s = re.sub(r"\.(VN|HM|HN)$", "", s)
    if not re.fullmatch(r"[A-Z0-9]{3,10}", s):
        raise SymbolError(f"Mã chứng khoán '{symbol}' không hợp lệ (chỉ gồm chữ/số, 3-10 ký tự, ví dụ VRE, VCB, FPT).")
    return s


def is_financial_ticker(symbol: str) -> bool:
    return normalize_symbol(symbol) in FINANCIAL_TICKERS


def today_vn() -> pd.Timestamp:
    return pd.Timestamp.now(tz=VN_TZ).tz_localize(None).normalize()


def parse_date(x, label: str = "ngày") -> pd.Timestamp:
    """Nhận 'YYYY-MM-DD', 'DD/MM/YYYY', date, datetime, Timestamp."""
    if x is None or (isinstance(x, str) and not x.strip()):
        raise DateRangeError(f"Thiếu {label}.")
    if isinstance(x, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return pd.Timestamp(datetime.strptime(x.strip(), fmt))
            except ValueError:
                continue
        raise DateRangeError(f"{label.capitalize()} '{x}' sai định dạng. Dùng YYYY-MM-DD hoặc DD/MM/YYYY.")
    try:
        ts = pd.Timestamp(x)
    except Exception:
        raise DateRangeError(f"{label.capitalize()} '{x}' không hợp lệ.")
    if ts.tzinfo is not None:
        ts = ts.tz_convert(VN_TZ).tz_localize(None)
    return ts.normalize()


def resolve_range(start=None, end=None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    t = today_vn()
    e = parse_date(end, "ngày kết thúc") if end else t
    if e > t:
        e = t
    s = parse_date(start, "ngày bắt đầu") if start else e - pd.DateOffset(years=DEFAULT_YEARS)
    if s >= e:
        raise DateRangeError(f"Ngày bắt đầu ({s:%d/%m/%Y}) phải trước ngày kết thúc ({e:%d/%m/%Y}).")
    if s < pd.Timestamp("2000-07-28"):
        s = pd.Timestamp("2000-07-28")   # Sở GDCK TP.HCM bắt đầu giao dịch
    return s.normalize(), e.normalize()


# ==========================================================================
# 2. Gọi mạng có retry
# ==========================================================================
def _http_get_json(url: str, params: dict, retries: int = 3, session=None) -> object:
    if requests is None:
        raise DataSourceError("Thiếu thư viện 'requests'. Cài bằng: pip install requests")
    getter = session or requests
    last = "không rõ"
    for i in range(retries):
        try:
            r = getter.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        except requests.RequestException as e:           # mất mạng, timeout, DNS...
            last = f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    raise DataSourceError("DNSE trả về dữ liệu không phải JSON (có thể bị chặn hoặc đổi API).")
            last = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):   # lỗi 4xx khác: thử lại vô ích
                raise DataSourceError(f"DNSE từ chối yêu cầu ({last}).")
        if i < retries - 1:
            time.sleep(0.8 * (2 ** i))
    raise DataSourceError(f"Không kết nối được DNSE sau {retries} lần thử ({last}).")


# ==========================================================================
# 3. Các nguồn giá
# ==========================================================================
_KEYMAP = {"t": ["t", "time", "timestamp"], "o": ["o", "open"], "h": ["h", "high"],
           "l": ["l", "low"], "c": ["c", "close"], "v": ["v", "volume"]}


def parse_dnse_payload(payload) -> pd.DataFrame:
    """JSON DNSE dạng {t:[...], o:[...], h:[...], l:[...], c:[...], v:[...]} -> DataFrame thô."""
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = payload["data"]
    if not isinstance(payload, dict):
        raise DataSourceError("Phản hồi DNSE có định dạng không như mong đợi.")
    lower = {str(k).lower(): v for k, v in payload.items()}
    arrays = {}
    for std, alts in _KEYMAP.items():
        for a in alts:
            if a in lower and isinstance(lower[a], list):
                arrays[std] = lower[a]
                break
    if "t" not in arrays or "c" not in arrays:
        raise DataSourceError(f"Phản hồi DNSE thiếu trường thời gian/giá (các khóa nhận được: {list(payload)[:8]}).")
    n = len(arrays["t"])
    if n == 0:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    if any(len(v) != n for v in arrays.values()):
        raise DataSourceError("Phản hồi DNSE có các mảng dữ liệu không cùng độ dài.")
    return pd.DataFrame({"Date": arrays["t"], "Open": arrays.get("o"), "High": arrays.get("h"),
                         "Low": arrays.get("l"), "Close": arrays["c"], "Volume": arrays.get("v")})


def fetch_dnse_ohlc(symbol: str, start: pd.Timestamp, end: pd.Timestamp, kind: str = "stock",
                    resolution: str = "1D", session=None) -> pd.DataFrame:
    """Gọi API DNSE theo từng đoạn thời gian, ghép lại. Trả DataFrame thô (chưa làm sạch)."""
    url = DNSE_URL.format(kind=kind)
    frames, cur = [], start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=CHUNK_DAYS), end)
        params = {"symbol": symbol, "resolution": resolution,
                  "from": int(cur.tz_localize(VN_TZ).timestamp()),
                  "to": int((nxt + pd.Timedelta(days=1)).tz_localize(VN_TZ).timestamp())}
        part = parse_dnse_payload(_http_get_json(url, params, session=session))
        if len(part):
            frames.append(part)
        cur = nxt + pd.Timedelta(days=1)
    if not frames:
        raise SymbolError(f"DNSE không có dữ liệu cho '{symbol}' trong {start:%d/%m/%Y} - {end:%d/%m/%Y}. "
                          "Kiểm tra lại mã hoặc khoảng thời gian.")
    return pd.concat(frames, ignore_index=True)


def _import_vnstock():
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from vnstock import Vnstock
        return Vnstock
    except ImportError:
        raise DataSourceError("Chưa cài thư viện vnstock. Cài bằng: pip install -U vnstock")
    except Exception as e:
        raise DataSourceError(f"Không nạp được vnstock ({type(e).__name__}: {e}).")


def _call_flex(fn: Callable, **kwargs):
    """Gọi hàm vnstock chỉ với các tham số mà phiên bản đang cài hỗ trợ (API vnstock đổi theo phiên bản)."""
    try:
        params = inspect.signature(fn).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        pass
    with contextlib.redirect_stdout(io.StringIO()):     # vnstock hay in banner quảng cáo
        return fn(**kwargs)


def fetch_vnstock_ohlc(symbol: str, start: pd.Timestamp, end: pd.Timestamp, **_) -> pd.DataFrame:
    Vnstock = _import_vnstock()
    last = None
    for src in ("VCI", "TCBS"):
        try:
            quote = Vnstock().stock(symbol=symbol, source=src).quote
            raw = _call_flex(quote.history, start=f"{start:%Y-%m-%d}", end=f"{end:%Y-%m-%d}", interval="1D")
            if raw is not None and len(raw):
                return raw
            last = SymbolError(f"vnstock ({src}) không có dữ liệu cho '{symbol}'.")
        except Exception as e:
            last = e
    if isinstance(last, DataError):
        raise last
    raise DataSourceError(f"vnstock không lấy được giá '{symbol}' ({type(last).__name__}: {last}).")


def _find_header_row(raw: pd.DataFrame) -> int:
    for i in range(min(15, len(raw))):
        keys = {normalize_key(x) for x in raw.iloc[i].tolist()}
        if keys & {"close", "giadongcua", "dongcua", "closeprice"}:
            return i
    return -1


def read_local_file(path: Union[str, Path], symbol: Optional[str] = None) -> pd.DataFrame:
    """Đọc CSV/Excel giá (kể cả khi tiêu đề không nằm ở dòng đầu, cột tiếng Việt)."""
    p = Path(path)
    if not p.exists():
        raise DataSourceError(f"Không tìm thấy file '{path}'.")
    try:
        if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
            raw = pd.read_excel(p, header=None)
        else:
            for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
                try:
                    raw = pd.read_csv(p, header=None, encoding=enc, sep=None, engine="python")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise DataSourceError("Không đọc được bảng mã của file CSV.")
    except DataError:
        raise
    except Exception as e:
        raise DataSourceError(f"Không đọc được file '{p.name}': {type(e).__name__}: {e}")
    h = _find_header_row(raw)
    if h < 0:
        raise DataSourceError(f"File '{p.name}' không có cột giá đóng cửa (Close/Đóng cửa).")
    df = raw.iloc[h + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[h].tolist()]
    df = df.reset_index(drop=True)
    std = standardize_columns(df)
    if symbol and "Symbol" in std.columns:
        sub = std[std["Symbol"].astype(str).str.upper().str.strip() == symbol]
        if len(sub):
            return sub
    return std


# ==========================================================================
# 4. Cache
# ==========================================================================
def _cache_paths(kind: str, symbol: str, start, end, cache_dir: Path) -> Tuple[Path, Path]:
    stem = f"{kind}_{symbol}_{start:%Y%m%d}_{end:%Y%m%d}"
    return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.json"


def _read_cache(csv_p: Path, json_p: Path) -> Optional[pd.DataFrame]:
    try:
        if not csv_p.exists():
            return None
        df = pd.read_csv(csv_p, parse_dates=["Date"])
        rep = QualityReport.from_dict(json.loads(json_p.read_text(encoding="utf-8"))) if json_p.exists() else QualityReport()
        df.attrs.update({"symbol": rep.symbol, "source": rep.source, "fetched_at": rep.fetched_at,
                         "quality": rep.to_dict(), "price_unit": "VND"})
        return df
    except Exception as e:
        log.warning("Cache hỏng (%s): %s", csv_p.name, e)
        return None


def _write_cache(df: pd.DataFrame, rep: QualityReport, csv_p: Path, json_p: Path) -> None:
    try:
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_p, index=False)
        json_p.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as e:     # không ghi được cache không được làm hỏng luồng chính
        log.warning("Không ghi được cache: %s", e)


def _is_fresh(p: Path, end: pd.Timestamp, ttl_hours: float) -> bool:
    if not p.exists():
        return False
    age_h = (time.time() - p.stat().st_mtime) / 3600
    historical = (today_vn() - end).days > 3          # dữ liệu quá khứ không đổi: giữ cache lâu
    return age_h < (24 * 7 if historical else ttl_hours)


# ==========================================================================
# 5. Hàm chính: giá cổ phiếu / chỉ số
# ==========================================================================
def _load_prices(kind: str, symbol: str, start, end, source: str, local_file, use_cache: bool,
                 cache_dir, cache_ttl_hours: float, price_unit: str, session):
    sym = normalize_symbol(symbol)
    s, e = resolve_range(start, end)
    cache_dir = Path(cache_dir)
    csv_p, json_p = _cache_paths(kind, sym, s, e, cache_dir)

    if use_cache and _is_fresh(csv_p, e, cache_ttl_hours):
        cached = _read_cache(csv_p, json_p)
        if cached is not None and len(cached):
            return cached, QualityReport.from_dict(cached.attrs["quality"])

    providers = {"dnse": ["dnse"], "vnstock": ["vnstock"], "local": ["local"],
                 "auto": ["dnse", "vnstock"] + (["local"] if local_file else [])}.get(source)
    if providers is None:
        raise DataError("source phải là 'auto', 'dnse', 'vnstock' hoặc 'local'.")
    if "local" in providers and not local_file:
        raise DataError("source='local' cần truyền local_file=<đường dẫn CSV/Excel>.")

    errors: List[Tuple[str, Exception]] = []
    for prov in providers:
        try:
            if prov == "dnse":
                raw = fetch_dnse_ohlc(sym, s, e, kind=kind, session=session)
                label = f"DNSE (Entrade) - {DNSE_URL.format(kind=kind)}"
            elif prov == "vnstock":
                raw = fetch_vnstock_ohlc(sym, s, e)
                label = "vnstock (VCI/TCBS)"
            else:
                raw = read_local_file(local_file, sym)
                label = f"File nội bộ: {Path(local_file).name}"
            df, rep = clean_price_data(raw, symbol=sym, source=label, price_unit=price_unit)
            df = df[(df["Date"] >= s) & (df["Date"] <= e)].reset_index(drop=True)
            if df.empty:
                raise SymbolError(f"Nguồn {prov} không có dữ liệu '{sym}' trong khoảng đã chọn.")
            rep.n_clean, rep.start, rep.end = len(df), f"{df['Date'].iloc[0]:%Y-%m-%d}", f"{df['Date'].iloc[-1]:%Y-%m-%d}"
            if errors:
                rep.add("warning", "FALLBACK", "Nguồn chính lỗi, đã dùng nguồn dự phòng: " +
                        "; ".join(f"{p}: {ex}" for p, ex in errors), len(errors))
            df.attrs.update({"quality": rep.to_dict(), "source": label})
            if use_cache:
                _write_cache(df, rep, csv_p, json_p)
            return df, rep
        except DataError as ex:
            errors.append((prov, ex))
            log.warning("Nguồn %s lỗi: %s", prov, ex)

    # Tất cả nguồn thất bại -> thử cache cũ (kể cả đã hết hạn)
    stale = _read_cache(csv_p, json_p) if use_cache else None
    if stale is not None and len(stale):
        rep = QualityReport.from_dict(stale.attrs["quality"])
        rep.add("warning", "STALE_CACHE", f"Không kết nối được nguồn dữ liệu; đang dùng bản lưu cũ cập nhật lúc {rep.fetched_at}.", 1)
        rep.finalize(100 - rep.score + 10)
        stale.attrs["quality"] = rep.to_dict()
        return stale, rep

    if errors and all(isinstance(ex, SymbolError) for _, ex in errors):
        raise errors[0][1]
    detail = " | ".join(f"{p}: {ex}" for p, ex in errors)
    raise DataSourceError(f"Không lấy được dữ liệu '{sym}'. {detail}. "
                          "Gợi ý: kiểm tra mạng, cài vnstock (pip install -U vnstock) hoặc truyền local_file=<file Excel/CSV giá>.")


def load_stock_data(symbol: str, start=None, end=None, source: str = "auto",
                    local_file: Optional[Union[str, Path]] = None, use_cache: bool = True,
                    cache_dir: Union[str, Path] = CACHE_DIR, cache_ttl_hours: float = 6.0,
                    price_unit: str = "auto", return_report: bool = False, session=None):
    """
    Tải giá cổ phiếu và trả DataFrame chuẩn: Date | Symbol | Open | High | Low | Close | Volume (giá: VND).
    `df.attrs["source"]`, `df.attrs["fetched_at"]`, `df.attrs["quality"]` ghi nguồn, thời điểm cập nhật, chất lượng.
    Gói bằng st.cache_data ở TV5 để dashboard nhanh hơn.
    """
    df, rep = _load_prices("stock", symbol, start, end, source, local_file, use_cache, cache_dir,
                           cache_ttl_hours, price_unit, session)
    return (df, rep) if return_report else df


def load_index_data(index_symbol: str = "VNINDEX", start=None, end=None, source: str = "auto",
                    use_cache: bool = True, cache_dir: Union[str, Path] = CACHE_DIR,
                    return_report: bool = False, session=None):
    """Tải chỉ số thị trường (VNINDEX, VN30, HNX...) - dùng làm benchmark tính Beta cho TV4."""
    sym = str(index_symbol).strip().upper().replace("-", "").replace(" ", "")
    if not re.fullmatch(r"[A-Z0-9]{3,12}", sym):
        raise SymbolError(f"Mã chỉ số '{index_symbol}' không hợp lệ.")
    df, rep = _load_prices("index", sym, start, end, source, None, use_cache, cache_dir, 6.0, "vnd", session)
    return (df, rep) if return_report else df


def benchmark_series(index_df: pd.DataFrame) -> pd.Series:
    """Series giá đóng cửa chỉ số, index = Date. Truyền thẳng vào score_stock(..., benchmark=...) của TV4."""
    s = index_df.set_index("Date")["Close"].astype(float)
    s.name = "Close"
    return s


# ==========================================================================
# 6. Báo cáo tài chính, chỉ số, thông tin doanh nghiệp (vnstock)
# ==========================================================================
@dataclass
class FinancialData:
    symbol: str
    period: str
    source: str = ""
    fetched_at: str = ""
    income_statement: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash_flow: pd.DataFrame = field(default_factory=pd.DataFrame)
    ratios: pd.DataFrame = field(default_factory=pd.DataFrame)
    overview: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return all(len(x) == 0 for x in (self.income_statement, self.balance_sheet, self.cash_flow, self.ratios))

    def coverage(self) -> Dict[str, bool]:
        return {k: len(getattr(self, k)) > 0 for k in ("income_statement", "balance_sheet", "cash_flow", "ratios", "overview")}


def _finance_accessors(Vnstock, symbol: str, source: str):
    stock = Vnstock().stock(symbol=symbol, source=source)
    return stock.finance, stock.company


def load_financial_data(symbol: str, period: str = "year", lang: str = "vi",
                        sources: Tuple[str, ...] = ("VCI", "TCBS")) -> FinancialData:
    """
    Lấy báo cáo KQKD, CĐKT, LCTT, chỉ số tài chính và thông tin doanh nghiệp từ vnstock.
    Lỗi từng phần chỉ tạo cảnh báo; chỉ ném DataSourceError khi KHÔNG lấy được bảng nào.
    """
    sym = normalize_symbol(symbol)
    if period not in ("year", "quarter"):
        raise DataError("period phải là 'year' hoặc 'quarter'.")
    Vnstock = _import_vnstock()
    out = FinancialData(sym, period, fetched_at=pd.Timestamp.now(tz=VN_TZ).strftime("%Y-%m-%d %H:%M:%S"))
    tables = {"income_statement": "income_statement", "balance_sheet": "balance_sheet",
              "cash_flow": "cash_flow", "ratios": "ratio"}
    used = []
    for src in sources:
        try:
            fin, comp = _finance_accessors(Vnstock, sym, src)
        except Exception as e:
            out.warnings.append(f"vnstock({src}) không khởi tạo được: {type(e).__name__}: {e}")
            continue
        for attr, meth in tables.items():
            if len(getattr(out, attr)):
                continue
            fn = getattr(fin, meth, None)
            if fn is None:
                out.warnings.append(f"Phiên bản vnstock hiện tại không có hàm finance.{meth}().")
                continue
            try:
                df = clean_financial_df(_call_flex(fn, period=period, lang=lang, dropna=True))
                if len(df):
                    setattr(out, attr, df)
                    if src not in used:
                        used.append(src)
            except Exception as e:
                out.warnings.append(f"Không lấy được {attr} từ {src}: {type(e).__name__}: {str(e)[:120]}")
        if len(out.overview) == 0:
            for meth in ("overview", "profile"):
                try:
                    df = clean_financial_df(_call_flex(getattr(comp, meth)))
                    if len(df):
                        out.overview = df
                        break
                except Exception:
                    continue
        if all(len(getattr(out, a)) for a in tables):
            break
    out.source = "vnstock (" + "/".join(used) + ")" if used else "vnstock"
    if out.is_empty():
        raise DataSourceError(f"Không lấy được báo cáo tài chính của '{sym}' từ vnstock. "
                              + (" | ".join(out.warnings[:3]) or "Kiểm tra mạng/phiên bản vnstock (pip install -U vnstock)."))
    for k, ok in out.coverage().items():
        if not ok:
            out.warnings.append(f"Thiếu bảng '{k}': các chỉ số liên quan sẽ bị bỏ qua.")
    return out


def _latest_row(df: pd.DataFrame) -> Optional[pd.Series]:
    """Dòng kỳ mới nhất (vnstock thường sắp mới nhất ở trên; vẫn sắp theo cột năm/kỳ nếu có)."""
    if df is None or len(df) == 0:
        return None
    ycol = find_column(df, ["yearReport", "Năm", "year"], exact=True)
    qcol = find_column(df, ["lengthReport", "Kỳ", "quarter"], exact=True)
    d = df
    if ycol is not None:
        keys = [ycol] + ([qcol] if qcol is not None else [])
        d = df.sort_values(keys, ascending=False)
    return d.iloc[0]


def _growth(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
    col = find_column(df, candidates)
    if col is None or len(df) < 2:
        return None
    ycol = find_column(df, ["yearReport", "Năm", "year"], exact=True)
    d = df.sort_values(ycol, ascending=False) if ycol is not None else df
    vals = pd.to_numeric(d[col], errors="coerce").dropna().tolist()
    if len(vals) < 2 or vals[1] <= 0:
        return None
    return float(vals[0] / vals[1] - 1)


def get_fundamentals_snapshot(fin: FinancialData) -> dict:
    """
    Gom chỉ số kỳ mới nhất thành dict cho TV3/TV4 (khóa trùng với score_stock của TV4).
    LƯU Ý: tên cột của vnstock thay đổi theo phiên bản nên đây là bước 'best effort':
    chỉ số nào không tìm thấy sẽ không có trong dict (không bao giờ tự điền số). TV3 nên kiểm tra
    lại bằng fin.ratios trước khi dùng cho kết luận.
    """
    snap: dict = {"is_financial": is_financial_ticker(fin.symbol), "source": fin.source, "period_type": fin.period}
    row = _latest_row(fin.ratios)
    spec = {   # khóa -> (danh sách tên cột ứng viên, so khớp chính xác?)
        "roe": (["ROE"], True), "roa": (["ROA"], True), "pe": (["P/E", "PE"], True), "pb": (["P/B", "PB"], True),
        "eps": (["EPS"], False),
        "net_margin": (["Net Profit Margin", "Biên lợi nhuận ròng", "Biên lợi nhuận sau thuế"], False),
        "debt_to_equity": (["Debt/Equity", "Nợ/Vốn chủ sở hữu", "Nợ trên vốn chủ"], False),
        "current_ratio": (["Current Ratio", "Chỉ số thanh toán hiện thời", "Chỉ số thanh toán hiện hành"], False),
        "interest_coverage": (["Interest Coverage", "Khả năng thanh toán lãi vay"], False),
        "npl_ratio": (["NPL", "Nợ xấu"], False), "car": (["CAR", "Hệ số an toàn vốn"], True),
    }
    if row is not None:
        for key, (cands, exact) in spec.items():
            col = find_column(fin.ratios, cands, exact=exact)
            if col is not None:
                v = pd.to_numeric(row.get(col), errors="coerce")
                if pd.notna(v):
                    snap[key] = float(v)
        ycol = find_column(fin.ratios, ["yearReport", "Năm"], exact=True)
        if ycol is not None:
            snap["latest_year"] = row.get(ycol)
    rg = _growth(fin.income_statement, ["Doanh thu thuần", "Net Revenue", "Revenue", "Doanh thu"])
    pg = _growth(fin.income_statement, ["Lợi nhuận sau thuế", "Net Profit", "Profit after tax", "Lợi nhuận ròng"])
    if rg is not None:
        snap["revenue_growth"] = rg
    if pg is not None:
        snap["profit_growth"] = pg
    return snap


# ==========================================================================
# 7. CLI: tạo stock_prices.csv, kiểm tra kết nối
# ==========================================================================
def build_stock_prices_csv(symbols: List[str], start, end, out: Union[str, Path], source: str = "auto",
                           local_file: Optional[str] = None, report_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    frames, reports = [], {}
    for sym in symbols:
        try:
            df, rep = load_stock_data(sym, start, end, source=source, local_file=local_file, return_report=True)
            frames.append(df)
            reports[rep.symbol] = rep.to_dict()
            print(f"[OK]  {rep.symbol}: {rep.n_clean} phiên, {rep.start} → {rep.end}, chất lượng {rep.score:.0f}/100 ({rep.grade}) - {rep.source}")
        except DataError as e:
            print(f"[LỖI] {sym}: {e}")
            reports[sym.upper()] = {"error": str(e)}
    if not frames:
        raise DataError("Không tải được mã nào.")
    allp = pd.concat(frames, ignore_index=True)[PRICE_COLUMNS]
    allp.to_csv(out, index=False, encoding="utf-8-sig")
    rp = Path(report_path) if report_path else Path(out).with_name("data_quality_report.json")
    rp.write_text(json.dumps(reports, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"Đã ghi {len(allp):,} dòng vào {out} và báo cáo chất lượng vào {rp}")
    return allp


def _check() -> int:
    print("== Kiểm tra kết nối ==")
    ok = True
    end = today_vn()
    try:
        df, rep = load_stock_data("VRE", end - pd.Timedelta(days=45), end, use_cache=False, return_report=True)
        print(f"Giá VRE: OK ({len(df)} phiên, nguồn {rep.source}); đóng cửa gần nhất {df['Close'].iloc[-1]:,.0f} đ")
    except DataError as e:
        ok = False
        print("Giá VRE: LỖI -", e)
    try:
        fin = load_financial_data("VRE")
        print("Tài chính VRE: OK", fin.coverage())
        print("Snapshot:", get_fundamentals_snapshot(fin))
    except DataError as e:
        ok = False
        print("Tài chính VRE: LỖI -", e)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TV1 - Tải và làm sạch dữ liệu chứng khoán")
    ap.add_argument("--symbols", nargs="+", default=["VRE"])
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="stock_prices.csv")
    ap.add_argument("--source", default="auto", choices=["auto", "dnse", "vnstock", "local"])
    ap.add_argument("--local-file", default=None, help="CSV/Excel giá có sẵn (dùng khi mất mạng)")
    ap.add_argument("--check", action="store_true", help="Kiểm tra kết nối DNSE + vnstock rồi thoát")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if a.check:
        sys.exit(_check())
    try:
        build_stock_prices_csv(a.symbols, a.start, a.end, a.out, a.source, a.local_file)
    except DataError as e:
        print("LỖI:", e)
        sys.exit(1)
