"""
risk_analysis.py  -  TV4 (Chấm điểm và rủi ro)
==============================================
Tính rủi ro của một cổ phiếu từ chuỗi giá/khối lượng (không gọi mạng):

    - Biến động lợi suất (độ lệch chuẩn năm hóa), VaR 95% 1 ngày, độ lệch chuẩn phía giảm
    - Maximum drawdown (MDD) + drawdown hiện tại + thời gian phục hồi
    - Beta so với VN-Index (nếu có benchmark)
    - Rủi ro thanh khoản: giá trị giao dịch/ngày, tỷ lệ phiên không có khối lượng, Amihud

Mỗi chỉ số được quy về "điểm an toàn" 0-100 (100 = ít rủi ro nhất) bằng bảng mốc tuyến tính
từng đoạn (ANCHORS bên dưới - công khai để cả nhóm kiểm tra / chỉnh).

Dùng nhanh:
    from risk_analysis import assess_risk
    r = assess_risk(df_gia, benchmark=benchmark_series(load_index_data("VNINDEX")))
    r["risk_score"], r["risk_level"], r["liquidity_level"], r["metrics"], r["warnings"]

Đầu vào `df_gia`: DataFrame của TV1 (cột Date/Close/Volume, không phân biệt hoa thường) hoặc
DataFrame có index là ngày. Giá đơn vị VND (tự nhận ra nếu là nghìn đồng).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_WINDOW = 756          # ~3 năm phiên giao dịch dùng cho MDD
MIN_RETURNS = 60              # tối thiểu số quan sát lợi suất để tính vol/VaR/beta
MIN_SESSIONS_MDD = 30
MIN_SESSIONS_LIQ = 20
BETA_WINDOW = 504             # ~2 năm

# --------------------------------------------------------------------------- #
# Bảng mốc quy đổi chỉ số -> điểm an toàn 0-100 (x tăng dần; ngoài biên thì giữ nguyên điểm biên)
# --------------------------------------------------------------------------- #
ANCHORS: dict[str, list[tuple[float, float]]] = {
    # Độ biến động năm hóa (1 năm gần nhất): VN-Index thường ~15-25%
    "volatility": [(0.15, 100), (0.25, 80), (0.35, 60), (0.50, 30), (0.70, 0)],
    # Max drawdown (độ lớn, 3 năm): sụt >50% từ đỉnh là rủi ro rất cao
    "max_drawdown": [(0.10, 100), (0.20, 80), (0.35, 55), (0.50, 25), (0.65, 0)],
    # VaR 95% 1 ngày (độ lớn, 1 năm): biên độ HOSE ±7%/ngày
    "var_95": [(0.015, 100), (0.025, 75), (0.035, 45), (0.050, 0)],
    # Mức giảm hiện tại so với đỉnh trong cửa sổ
    "current_drawdown": [(0.0, 100), (0.10, 80), (0.25, 40), (0.40, 0)],
    # Beta: thấp = phòng thủ hơn thị trường
    "beta": [(0.6, 100), (1.0, 75), (1.3, 50), (1.8, 15), (2.2, 0)],
    # Thanh khoản theo log10(giá trị khớp trung vị 60 phiên, VND/ngày): 0.5 tỷ -> 0; 100 tỷ -> 100
    "liquidity_log10_value": [(8.7, 0), (9.3, 30), (10.0, 60), (10.48, 85), (11.0, 100)],
}

# Trọng số trong "điểm rủi ro" (tự chuẩn hóa lại khi thiếu chỉ số, vd. không có benchmark)
RISK_WEIGHTS: dict[str, float] = {
    "volatility": 25, "max_drawdown": 25, "var_95": 10,
    "current_drawdown": 10, "beta": 10, "liquidity": 20,
}

RISK_WEIGHT_RATIONALE = {
    "volatility": "Biến động giá là thước đo rủi ro phổ biến nhất, phản ánh mức dao động lợi suất hằng ngày.",
    "max_drawdown": "MDD cho biết nhà đầu tư từng phải chịu mức lỗ tạm thời lớn nhất từ đỉnh - rủi ro 'đuôi'.",
    "var_95": "VaR 95% là mức lỗ 1 ngày mà 95% số phiên không vượt quá, bổ sung cho vol ở các phiên xấu.",
    "current_drawdown": "Vị thế hiện tại so với đỉnh: cổ phiếu đang ở vùng đáy sâu mang rủi ro xu hướng.",
    "beta": "Beta đo độ nhạy với thị trường chung; beta cao khuếch đại lỗ khi thị trường giảm.",
    "liquidity": "Thanh khoản thấp khiến khó vào/ra lệnh và trượt giá - đặc biệt rủi ro ở cổ phiếu vốn hóa nhỏ.",
}

_ALIASES = {
    "date": ["date", "time", "datetime", "ngay", "ngày", "tradingdate", "trading_date"],
    "close": ["close", "gia_dong_cua", "giá đóng cửa", "adj close", "adj_close", "c"],
    "volume": ["volume", "vol", "khoi_luong", "khối lượng", "klgd", "v"],
}


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def score_from_anchors(x: float, anchors: list[tuple[float, float]]) -> float:
    """Nội suy tuyến tính từng đoạn; ngoài biên giữ điểm biên."""
    xs, ys = zip(*anchors)
    return float(np.interp(x, xs, ys))


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}%"


def _money(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:,.1f} tỷ đồng"
    return f"{v / 1e6:,.0f} triệu đồng"


def classify_risk(score: float) -> str:
    return ("Rủi ro thấp" if score >= 75 else "Rủi ro trung bình" if score >= 55
            else "Rủi ro cao" if score >= 35 else "Rủi ro rất cao")


def classify_liquidity(score: float) -> str:
    return ("Thanh khoản tốt" if score >= 75 else "Thanh khoản khá" if score >= 50
            else "Thanh khoản thấp" if score >= 25 else "Thanh khoản rất thấp")


def prepare_prices(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    Chuẩn hóa về DataFrame index=ngày, cột close, volume (volume có thể NaN).
    Nhận DataFrame của TV1 (Date/Close/Volume) hoặc DataFrame/Series index là ngày.
    attrs["price_multiplier"] = 1000 nếu giá ở đơn vị nghìn đồng (để tính giá trị giao dịch đúng).
    """
    if df is None or len(df) == 0:
        raise ValueError("Dữ liệu giá rỗng.")
    unit = str(getattr(df, "attrs", {}).get("price_unit", "")).lower()
    d = df.to_frame("close") if isinstance(df, pd.Series) else df.copy()
    d = d.rename(columns={c: str(c).strip().lower() for c in d.columns})
    for std, names in _ALIASES.items():
        if std not in d.columns:
            for a in names:
                if a in d.columns:
                    d = d.rename(columns={a: std})
                    break
    if "close" not in d.columns:
        raise ValueError(f"Thiếu cột giá đóng cửa. Các cột hiện có: {list(d.columns)}")
    idx = pd.to_datetime(d["date"], errors="coerce") if "date" in d.columns else pd.to_datetime(d.index, errors="coerce")
    d.index = pd.DatetimeIndex(idx).normalize()
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce") if "volume" in d.columns else np.nan
    d = d[d.index.notna() & (d["close"] > 0)]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    if d.empty:
        raise ValueError("Không còn dòng giá hợp lệ sau khi làm sạch.")
    mult = 1000.0 if (unit == "thousand_vnd" or (unit != "vnd" and float(d["close"].tail(250).median()) < 1000)) else 1.0
    out = d[["close", "volume"]].copy()
    out.attrs["price_multiplier"] = mult
    return out


def _bench_close(benchmark: pd.Series | pd.DataFrame | None) -> Optional[pd.Series]:
    if benchmark is None:
        return None
    if isinstance(benchmark, pd.DataFrame):
        b = prepare_prices(benchmark)["close"]
    else:
        s = benchmark.copy()
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index, errors="coerce")).normalize()
        b = pd.to_numeric(s, errors="coerce")
        b = b[b.index.notna() & (b > 0)]
        b = b[~b.index.duplicated(keep="last")].sort_index()
    return b if len(b) else None


# --------------------------------------------------------------------------- #
# Các thước đo rủi ro
# --------------------------------------------------------------------------- #
def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def annualized_volatility(returns: pd.Series) -> Optional[float]:
    if len(returns) < 2:
        return None
    return _finite(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def drawdown_series(close: pd.Series) -> pd.Series:
    """Chuỗi drawdown (<= 0) so với đỉnh lũy kế - dùng vẽ biểu đồ ở dashboard."""
    return close / close.cummax() - 1


def rolling_volatility(close: pd.Series, window: int = 60) -> pd.Series:
    return daily_returns(close).rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)


def max_drawdown_info(close: pd.Series) -> dict:
    """MDD (độ lớn dương), ngày đỉnh/đáy/phục hồi, drawdown hiện tại."""
    dd = drawdown_series(close)
    trough = dd.idxmin()
    peak_date = close.loc[:trough].idxmax()
    after = close.loc[trough:]
    rec = after[after >= close.loc[peak_date]]
    return {
        "max_drawdown": float(-dd.min()),
        "peak_date": peak_date.strftime("%Y-%m-%d"),
        "trough_date": trough.strftime("%Y-%m-%d"),
        "recovery_date": rec.index[0].strftime("%Y-%m-%d") if len(rec) else None,
        "recovered": bool(len(rec)),
        "current_drawdown": float(-dd.iloc[-1]),
        "sessions_since_peak": int(len(close.loc[close.loc[:close.index[-1]].idxmax():]) - 1),
    }


def compute_beta(stock_close: pd.Series, bench_close: pd.Series, min_obs: int = MIN_RETURNS,
                 window: int = BETA_WINDOW) -> Optional[dict]:
    """Beta = cov(r_cp, r_chỉ số) / var(r_chỉ số) trên các ngày trùng nhau; None nếu quá ít quan sát."""
    r = pd.concat([daily_returns(stock_close).rename("s"), daily_returns(bench_close).rename("b")],
                  axis=1, join="inner").dropna().tail(window)
    if len(r) < min_obs or r["b"].var(ddof=1) == 0:
        return None
    cov = r["s"].cov(r["b"])
    corr = r["s"].corr(r["b"])
    return {"beta": float(cov / r["b"].var(ddof=1)), "correlation": _finite(corr),
            "r_squared": _finite(corr ** 2) if corr == corr else None, "n_obs": int(len(r))}


def liquidity_stats(p: pd.DataFrame, returns: Optional[pd.Series] = None) -> Optional[dict]:
    """Giá trị giao dịch/ngày (VND) = close * volume * hệ số đơn vị giá."""
    vol = p["volume"]
    if vol.notna().sum() < MIN_SESSIONS_LIQ:
        return None
    mult = float(p.attrs.get("price_multiplier", 1.0))
    value = (p["close"] * vol * mult)
    last60, last20, last120 = value.tail(60), value.tail(20), vol.tail(120)
    stats = {
        "adv_value_20d": _finite(last20.mean()),
        "adv_value_60d": _finite(last60.mean()),
        "median_value_60d": _finite(last60.median()),
        "avg_volume_20d": _finite(vol.tail(20).mean()),
        "avg_volume_60d": _finite(vol.tail(60).mean()),
        "zero_volume_share": float((last120.fillna(0) <= 0).mean()),
        "volume_cv_60d": _finite(vol.tail(60).std(ddof=1) / vol.tail(60).mean()) if vol.tail(60).mean() > 0 else None,
    }
    if returns is not None and len(returns):
        v = value.reindex(returns.index).tail(250)
        ami = (returns.reindex(v.index).abs() / v.replace(0, np.nan)).dropna()
        stats["amihud_1e9"] = _finite(ami.mean() * 1e9) if len(ami) else None   # % thay đổi giá / 1 tỷ đồng GD
    return stats


# --------------------------------------------------------------------------- #
# Hàm chính
# --------------------------------------------------------------------------- #
def _metric(key: str, label: str, value: Any, display: str, score: float, note: str,
            weight: Optional[float] = None) -> dict:
    return {"key": key, "label": label, "value": value, "display": display,
            "score": float(np.clip(score, 0, 100)), "weight": float(weight if weight is not None else RISK_WEIGHTS[key]),
            "note": note}


def assess_risk(price_df: pd.DataFrame, benchmark: pd.Series | pd.DataFrame | None = None,
                window: int = DEFAULT_WINDOW, risk_free: float = 0.03) -> dict:
    """
    Đánh giá rủi ro. Trả về dict (JSON-friendly):
        available       False nếu không đủ dữ liệu để chấm
        risk_score      0-100 (100 = ít rủi ro) | risk_level  (Rủi ro thấp/trung bình/cao/rất cao)
        liquidity_score | liquidity_level
        metrics         list {key,label,value,display,score,weight,note} - dùng để giải thích / chấm điểm
        stats           số liệu chi tiết (vol, MDD, VaR, beta, thanh khoản, Sharpe...)
        warnings, notes
    `risk_free` (mặc định 3%/năm) chỉ dùng cho Sharpe tham khảo, KHÔNG ảnh hưởng điểm.
    """
    warnings: list[str] = []
    notes: list[str] = []
    p_all = prepare_prices(price_df)
    mult = p_all.attrs["price_multiplier"]
    p = p_all.tail(window + 1)
    p.attrs["price_multiplier"] = mult
    close = p["close"]
    rets = daily_returns(close)
    metrics: list[dict] = []
    stats: dict[str, Any] = {}
    as_of = close.index[-1]

    if (pd.Timestamp.today().normalize() - as_of).days > 7:
        warnings.append(f"Dữ liệu giá mới nhất là {as_of:%d/%m/%Y}, đã cũ hơn 7 ngày.")

    # --- biến động & VaR (1 năm gần nhất)
    r1y = rets.tail(TRADING_DAYS)
    if len(r1y) >= MIN_RETURNS:
        vol = annualized_volatility(r1y)
        stats["volatility_1y"] = vol
        stats["volatility_60d"] = annualized_volatility(rets.tail(60))
        stats["volatility_window"] = annualized_volatility(rets)
        metrics.append(_metric(
            "volatility", "Biến động lợi suất (1 năm)", vol, _pct(vol),
            score_from_anchors(vol, ANCHORS["volatility"]),
            f"Độ lệch chuẩn lợi suất ngày năm hóa; 60 phiên gần nhất: {_pct(stats['volatility_60d'])}."))
        var95 = float(-r1y.quantile(0.05))
        stats["var_95_1d"] = var95
        stats["worst_day"] = float(r1y.min())
        stats["best_day"] = float(r1y.max())
        metrics.append(_metric(
            "var_95", "VaR 95% (1 ngày)", var95, _pct(var95, 2),
            score_from_anchors(max(var95, 0), ANCHORS["var_95"]),
            f"95% số phiên lỗ không quá {_pct(var95, 2)}; phiên xấu nhất 1 năm: {_pct(stats['worst_day'], 2)}."))
        neg = r1y[r1y < 0]
        stats["downside_deviation"] = _finite(math.sqrt((neg ** 2).sum() / len(r1y)) * math.sqrt(TRADING_DAYS))
        years = len(r1y) / TRADING_DAYS
        total = float(close.iloc[-1] / close.iloc[-len(r1y) - 1] - 1) if len(close) > len(r1y) else None
        if total is not None:
            stats["return_1y"] = total
            stats["annual_return"] = (1 + total) ** (1 / years) - 1 if total > -1 else None
            if vol and stats["annual_return"] is not None:
                stats["sharpe"] = (stats["annual_return"] - risk_free) / vol
                notes.append(f"Sharpe tham khảo dùng lãi suất phi rủi ro giả định {_pct(risk_free)}/năm.")
    else:
        warnings.append(f"Chỉ có {len(r1y)} quan sát lợi suất (< {MIN_RETURNS}): bỏ qua biến động và VaR.")

    # --- drawdown
    if len(close) >= MIN_SESSIONS_MDD:
        dd = max_drawdown_info(close)
        stats.update(dd)
        stats["drawdown_window_sessions"] = int(len(close))
        rec_txt = (f"đã phục hồi ngày {dd['recovery_date']}" if dd["recovered"] else "CHƯA phục hồi về đỉnh cũ")
        metrics.append(_metric(
            "max_drawdown", f"Max drawdown ({len(close)} phiên)", dd["max_drawdown"], _pct(dd["max_drawdown"]),
            score_from_anchors(dd["max_drawdown"], ANCHORS["max_drawdown"]),
            f"Từ đỉnh {dd['peak_date']} xuống đáy {dd['trough_date']}, {rec_txt}."))
        metrics.append(_metric(
            "current_drawdown", "Mức giảm hiện tại so với đỉnh", dd["current_drawdown"], _pct(dd["current_drawdown"]),
            score_from_anchors(dd["current_drawdown"], ANCHORS["current_drawdown"]),
            f"Giá đóng cửa hiện thấp hơn đỉnh {dd['peak_date'] if dd['sessions_since_peak'] == 0 else 'trong cửa sổ'} "
            f"{_pct(dd['current_drawdown'])}."))
    else:
        warnings.append(f"Chỉ có {len(close)} phiên (< {MIN_SESSIONS_MDD}): bỏ qua drawdown.")

    # --- beta
    bench = _bench_close(benchmark)
    if bench is not None:
        b = compute_beta(close, bench)
        if b:
            stats.update({"beta": b["beta"], "beta_correlation": b["correlation"], "beta_obs": b["n_obs"]})
            metrics.append(_metric(
                "beta", "Beta so với thị trường", b["beta"], f"{b['beta']:.2f}",
                score_from_anchors(b["beta"], ANCHORS["beta"]),
                f"Tính trên {b['n_obs']} phiên trùng với benchmark, tương quan {b['correlation']:.2f}."))
        else:
            warnings.append("Benchmark có quá ít phiên trùng ngày với cổ phiếu: bỏ qua Beta.")
    else:
        notes.append("Không có benchmark (VN-Index): không tính Beta, trọng số được chia lại cho các chỉ số còn lại.")

    # --- thanh khoản
    liq = liquidity_stats(p, rets)
    liq_score = None
    if liq and liq["median_value_60d"] and liq["median_value_60d"] > 0:
        stats.update({f"liq_{k}": v for k, v in liq.items()})
        base = score_from_anchors(math.log10(liq["median_value_60d"]), ANCHORS["liquidity_log10_value"])
        penalty = min(40.0, liq["zero_volume_share"] * 200)
        liq_score = float(np.clip(base - penalty, 0, 100))
        note = f"Giá trị khớp trung vị 60 phiên ~ {_money(liq['median_value_60d'])}/ngày"
        if liq["zero_volume_share"] > 0:
            note += f"; {_pct(liq['zero_volume_share'])} số phiên (120 gần nhất) không có khối lượng (trừ {penalty:.0f} điểm)"
        metrics.append(_metric("liquidity", "Thanh khoản", liq["median_value_60d"],
                               _money(liq["median_value_60d"]) + "/ngày", liq_score, note + "."))
    else:
        warnings.append("Thiếu khối lượng giao dịch: không đánh giá được rủi ro thanh khoản.")

    total_w = sum(RISK_WEIGHTS.values())
    avail_w = sum(m["weight"] for m in metrics)
    result: dict[str, Any] = {
        "available": False, "risk_score": None, "risk_level": None,
        "liquidity_score": liq_score, "liquidity_level": classify_liquidity(liq_score) if liq_score is not None else None,
        "metrics": metrics, "stats": {k: v for k, v in stats.items()},
        "weights": dict(RISK_WEIGHTS), "weight_rationale": dict(RISK_WEIGHT_RATIONALE),
        "coverage": avail_w / total_w, "n_sessions": int(len(p_all)), "window_sessions": int(len(close)),
        "as_of": as_of.strftime("%Y-%m-%d"), "warnings": warnings, "notes": notes,
    }
    if avail_w / total_w >= 0.4:
        score = sum(m["score"] * m["weight"] for m in metrics) / avail_w
        result.update(available=True, risk_score=float(score), risk_level=classify_risk(score))
    else:
        warnings.append("Không đủ chỉ số rủi ro (độ phủ < 40%) để chấm điểm rủi ro.")
    return result


def risk_table(risk: dict) -> pd.DataFrame:
    """Bảng chỉ số rủi ro cho dashboard/PDF."""
    rows = [{"Chỉ số": m["label"], "Giá trị": m["display"], "Điểm an toàn (0-100)": round(m["score"], 1),
             "Trọng số": m["weight"], "Diễn giải": m["note"]} for m in risk.get("metrics", [])]
    return pd.DataFrame(rows)