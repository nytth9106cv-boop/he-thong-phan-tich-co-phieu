"""
technical_analysis.py
=====================
Module phân tích kỹ thuật cổ phiếu cho dashboard.

Cách dùng nhanh (trong dashboard của bạn):

    from technical_analysis import analyze_symbol
    result = analyze_symbol("VNM")          # hoặc analyze_symbol("VNM", df=df_cua_ban)
    fig = result["figure"]                  # plotly Figure  -> st.plotly_chart(fig) / dcc.Graph(figure=fig)
    result["signal"]                        # "Tích cực" | "Trung lập" | "Tiêu cực"
    result["trend"], result["momentum"]     # xu hướng, động lượng (kèm giải thích)
    result["reasons"]                       # list các quy tắc + giải thích

Yêu cầu: pip install pandas numpy plotly
Dữ liệu đầu vào: DataFrame có các cột (không phân biệt hoa/thường, tự nhận diện tên tiếng Việt/Anh phổ biến):
    date, open, high, low, close, volume
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# 1. NẠP DỮ LIỆU 
# ----------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "date": ["date", "time", "datetime", "ngay", "ngày", "tradingdate", "trading_date"],
    "open": ["open", "gia_mo_cua", "giá mở cửa", "o"],
    "high": ["high", "gia_cao_nhat", "giá cao nhất", "h"],
    "low": ["low", "gia_thap_nhat", "giá thấp nhất", "l"],
    "close": ["close", "gia_dong_cua", "giá đóng cửa", "c", "adj close", "adj_close"],
    "volume": ["volume", "vol", "khoi_luong", "khối lượng", "klgd", "v"],
}


def load_price_data(symbol: str) -> pd.DataFrame:
    """Dùng bộ nạp dữ liệu của TV1 (có cache, nguồn dự phòng, báo lỗi tiếng Việt)."""
    from data_loader import load_stock_data
    return load_stock_data(symbol)          # mặc định lấy 5 năm gần nhất


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hoá tên cột, kiểu dữ liệu, sắp xếp theo ngày."""
    df = df.copy()
    lower = {c: str(c).strip().lower() for c in df.columns}
    df.rename(columns=lower, inplace=True)
    rename = {}
    for std, aliases in _COLUMN_ALIASES.items():
        for a in aliases:
            if a in df.columns and std not in df.columns and a not in rename:
                rename[a] = std
                break
    df.rename(columns=rename, inplace=True)

    missing = [c for c in ["date", "open", "high", "low", "close", "volume"] if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}. Các cột hiện có: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.dropna(subset=["open", "high", "low", "close"])
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True))
    df["volume"] = df["volume"].fillna(0)
    return df


# ----------------------------------------------------------------------------
# 2. TÍNH CHỈ BÁO
# ----------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm SMA20/50, EMA12/26, RSI14, MACD(12,26,9), Bollinger(20,2) vào DataFrame."""
    df = _normalize(df)
    c = df["close"]

    # Trung bình động
    df["sma20"] = c.rolling(20).mean()
    df["sma50"] = c.rolling(50).mean()
    df["ema12"] = c.ewm(span=12, adjust=False).mean()
    df["ema26"] = c.ewm(span=26, adjust=False).mean()

    # RSI 14 (Wilder smoothing)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100

    # MACD
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20, 2)
    std20 = c.rolling(20).std(ddof=0)
    df["bb_mid"] = df["sma20"]
    df["bb_upper"] = df["bb_mid"] + 2 * std20
    df["bb_lower"] = df["bb_mid"] - 2 * std20
    band = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pctb"] = (c - df["bb_lower"]) / band            # %B: 0 = chạm băng dưới, 1 = chạm băng trên
    df["bb_width"] = band / df["bb_mid"]                   # độ rộng băng

    # Khối lượng trung bình 20 phiên (để so sánh thanh khoản)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df


# ----------------------------------------------------------------------------
# 3. XU HƯỚNG, ĐỘNG LƯỢNG VÀ TÍN HIỆU
# ----------------------------------------------------------------------------
def _recent_cross(a: pd.Series, b: pd.Series, lookback: int = 5) -> Optional[str]:
    """Trả 'up' nếu a cắt lên b trong `lookback` phiên gần nhất, 'down' nếu cắt xuống, None nếu không."""
    diff = (a - b).dropna()
    if len(diff) < lookback + 1:
        return None
    seg = diff.iloc[-(lookback + 1):]
    sign = np.sign(seg)
    if sign.iloc[0] < 0 and sign.iloc[-1] > 0:
        return "up"
    if sign.iloc[0] > 0 and sign.iloc[-1] < 0:
        return "down"
    return None


def evaluate_signals(df: pd.DataFrame) -> dict:
    """
    Đánh giá xu hướng, động lượng và tín hiệu tổng hợp từ DataFrame đã có chỉ báo.

    Mỗi quy tắc cho điểm +1 (tích cực) / 0 (trung lập) / -1 (tiêu cực) kèm giải thích.
    Tổng điểm >= +2 -> Tích cực; <= -2 -> Tiêu cực; còn lại -> Trung lập.
    """
    d = df.dropna(subset=["sma50", "rsi14", "macd_signal", "bb_pctb"])
    if len(d) < 5:
        raise ValueError("Không đủ dữ liệu (cần tối thiểu ~60 phiên) để đánh giá tín hiệu.")

    last, prev = d.iloc[-1], d.iloc[-2]
    price = last["close"]
    reasons: list[dict] = []

    def add(rule: str, score: int, text: str):
        reasons.append({"rule": rule, "score": score,
                        "label": {1: "Tích cực", 0: "Trung lập", -1: "Tiêu cực"}[score],
                        "explain": text})

    # --- Quy tắc 1: Giá so với SMA20 & SMA50
    if price > last["sma20"] > last["sma50"]:
        add("Giá vs SMA20/SMA50", 1,
            f"Giá ({price:,.2f}) nằm trên cả SMA20 ({last['sma20']:,.2f}) và SMA50 ({last['sma50']:,.2f}) -> xu hướng tăng ngắn và trung hạn.")
    elif price < last["sma20"] < last["sma50"]:
        add("Giá vs SMA20/SMA50", -1,
            f"Giá ({price:,.2f}) nằm dưới cả SMA20 ({last['sma20']:,.2f}) và SMA50 ({last['sma50']:,.2f}) -> xu hướng giảm.")
    else:
        add("Giá vs SMA20/SMA50", 0,
            f"Giá ({price:,.2f}) đang đan xen giữa SMA20 ({last['sma20']:,.2f}) và SMA50 ({last['sma50']:,.2f}) -> chưa rõ xu hướng.")

    # --- Quy tắc 2: Giao cắt SMA20/SMA50 (Golden/Death cross) trong 5 phiên
    cross = _recent_cross(d["sma20"], d["sma50"], 5)
    if cross == "up":
        add("Giao cắt SMA20/SMA50", 1, "SMA20 vừa cắt lên SMA50 trong 5 phiên gần nhất (Golden Cross ngắn hạn).")
    elif cross == "down":
        add("Giao cắt SMA20/SMA50", -1, "SMA20 vừa cắt xuống SMA50 trong 5 phiên gần nhất (Death Cross ngắn hạn).")
    else:
        pos = "trên" if last["sma20"] > last["sma50"] else "dưới"
        add("Giao cắt SMA20/SMA50", 0, f"Không có giao cắt mới; SMA20 đang {pos} SMA50.")

    # --- Quy tắc 3: EMA12 vs EMA26
    if last["ema12"] > last["ema26"]:
        add("EMA12 vs EMA26", 1, f"EMA12 ({last['ema12']:,.2f}) > EMA26 ({last['ema26']:,.2f}) -> đà tăng ngắn hạn chiếm ưu thế.")
    else:
        add("EMA12 vs EMA26", -1, f"EMA12 ({last['ema12']:,.2f}) < EMA26 ({last['ema26']:,.2f}) -> đà giảm ngắn hạn chiếm ưu thế.")

    # --- Quy tắc 4: MACD
    macd_cross = _recent_cross(d["macd"], d["macd_signal"], 3)
    hist_up = last["macd_hist"] > prev["macd_hist"]
    if macd_cross == "up" or (last["macd"] > last["macd_signal"] and hist_up):
        add("MACD", 1, f"MACD ({last['macd']:.2f}) trên đường tín hiệu ({last['macd_signal']:.2f})"
                       + (", vừa cắt lên" if macd_cross == "up" else "") + ", histogram đang cải thiện.")
    elif macd_cross == "down" or (last["macd"] < last["macd_signal"] and not hist_up):
        add("MACD", -1, f"MACD ({last['macd']:.2f}) dưới đường tín hiệu ({last['macd_signal']:.2f})"
                        + (", vừa cắt xuống" if macd_cross == "down" else "") + ", histogram đang suy yếu.")
    else:
        add("MACD", 0, "MACD và đường tín hiệu đang ở trạng thái trái chiều (đảo chiều/chưa xác nhận).")

    # --- Quy tắc 5: RSI14
    rsi = last["rsi14"]
    if rsi >= 70:
        add("RSI14", 0, f"RSI = {rsi:.1f} >= 70: vùng quá mua, đà tăng mạnh nhưng rủi ro điều chỉnh -> thận trọng.")
    elif rsi >= 55:
        add("RSI14", 1, f"RSI = {rsi:.1f} trong vùng 55-70: động lượng tăng lành mạnh, chưa quá mua.")
    elif rsi > 45:
        add("RSI14", 0, f"RSI = {rsi:.1f} quanh 50: động lượng cân bằng.")
    elif rsi > 30:
        add("RSI14", -1, f"RSI = {rsi:.1f} trong vùng 30-45: động lượng yếu, phe bán chiếm ưu thế.")
    else:
        add("RSI14", 0, f"RSI = {rsi:.1f} <= 30: vùng quá bán, có thể xuất hiện nhịp hồi kỹ thuật nhưng xu hướng giảm vẫn mạnh.")

    # --- Quy tắc 6: Bollinger Bands (%B)
    b = last["bb_pctb"]
    if b > 1:
        add("Bollinger Bands", 0, f"Giá vượt băng trên (%B = {b:.2f}): bứt phá mạnh hoặc quá mua ngắn hạn.")
    elif b >= 0.5:
        add("Bollinger Bands", 1, f"Giá nằm nửa trên dải Bollinger (%B = {b:.2f}) -> phe mua kiểm soát.")
    elif b >= 0:
        add("Bollinger Bands", -1, f"Giá nằm nửa dưới dải Bollinger (%B = {b:.2f}) -> phe bán kiểm soát.")
    else:
        add("Bollinger Bands", 0, f"Giá thủng băng dưới (%B = {b:.2f}): bán tháo mạnh hoặc quá bán ngắn hạn.")

    # --- Xu hướng (dựa SMA + độ dốc SMA50)
    slope = (d["sma50"].iloc[-1] / d["sma50"].iloc[-6] - 1) * 100 if len(d) >= 6 else 0.0
    if price > last["sma50"] and last["sma20"] > last["sma50"] and slope > 0:
        trend = "Xu hướng tăng"
        trend_exp = f"Giá > SMA50, SMA20 > SMA50 và SMA50 dốc lên ({slope:+.2f}%/5 phiên)."
    elif price < last["sma50"] and last["sma20"] < last["sma50"] and slope < 0:
        trend = "Xu hướng giảm"
        trend_exp = f"Giá < SMA50, SMA20 < SMA50 và SMA50 dốc xuống ({slope:+.2f}%/5 phiên)."
    else:
        trend = "Đi ngang / chưa rõ xu hướng"
        trend_exp = f"Các đường trung bình chưa đồng thuận (độ dốc SMA50 {slope:+.2f}%/5 phiên)."

    # --- Động lượng (RSI + MACD histogram)
    mom_score = (1 if rsi > 55 else -1 if rsi < 45 else 0) + (1 if last["macd_hist"] > 0 else -1)
    if mom_score >= 2:
        momentum, mom_exp = "Động lượng mạnh (tăng)", f"RSI {rsi:.1f} > 55 và MACD histogram dương ({last['macd_hist']:.2f})."
    elif mom_score <= -2:
        momentum, mom_exp = "Động lượng yếu (giảm)", f"RSI {rsi:.1f} < 45 và MACD histogram âm ({last['macd_hist']:.2f})."
    else:
        momentum, mom_exp = "Động lượng trung tính", f"RSI {rsi:.1f} và MACD histogram ({last['macd_hist']:.2f}) chưa đồng thuận."

    total = sum(r["score"] for r in reasons)
    signal = "Tích cực" if total >= 2 else "Tiêu cực" if total <= -2 else "Trung lập"

    # Ghi chú thêm: thanh khoản & nén biến động
    notes = []
    if last["vol_ma20"] and last["volume"] > 1.5 * last["vol_ma20"]:
        notes.append("Khối lượng phiên gần nhất cao hơn 1.5 lần trung bình 20 phiên (thanh khoản tăng đột biến).")
    bw = d["bb_width"].dropna()
    if len(bw) >= 60 and bw.iloc[-1] <= bw.iloc[-120:].quantile(0.1):
        notes.append("Dải Bollinger đang co hẹp bất thường (squeeze) -> có thể sắp có biến động mạnh.")

    return {
        "signal": signal,
        "score": int(total),
        "max_score": len(reasons),
        "trend": trend, "trend_explain": trend_exp,
        "momentum": momentum, "momentum_explain": mom_exp,
        "reasons": reasons,
        "notes": notes,
        "last": {k: (float(last[k]) if pd.notna(last[k]) else None) for k in
                 ["close", "sma20", "sma50", "ema12", "ema26", "rsi14", "macd", "macd_signal",
                  "macd_hist", "bb_upper", "bb_mid", "bb_lower", "volume"]},
        "as_of": last["date"].strftime("%d/%m/%Y"),
    }


# ----------------------------------------------------------------------------
# 4. VẼ BIỂU ĐỒ (trả về Figure, KHÔNG xuất file)
# ----------------------------------------------------------------------------
def build_figure(df: pd.DataFrame, symbol: str = "", last_n: Optional[int] = 250) -> go.Figure:
    """Biểu đồ 4 panel: Nến + SMA/EMA/Bollinger | Khối lượng | RSI | MACD."""
    d = df.tail(last_n) if last_n else df
    up, down = "#16a34a", "#dc2626"

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.14, 0.17, 0.19],
        subplot_titles=(f"{symbol.upper()} - Giá & đường trung bình", "Khối lượng", "RSI (14)", "MACD (12, 26, 9)"),
    )

    # Panel 1: nến + MA + Bollinger
    fig.add_trace(go.Candlestick(
        x=d["date"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        increasing_line_color=up, decreasing_line_color=down, name="Giá"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["bb_upper"], line=dict(color="rgba(120,120,120,.6)", width=1, dash="dot"),
                             name="BB trên"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["bb_lower"], line=dict(color="rgba(120,120,120,.6)", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(120,120,120,.08)", name="BB dưới"), row=1, col=1)
    for col, color, name in [("sma20", "#f59e0b", "SMA20"), ("sma50", "#2563eb", "SMA50"),
                             ("ema12", "#a855f7", "EMA12"), ("ema26", "#0d9488", "EMA26")]:
        fig.add_trace(go.Scatter(x=d["date"], y=d[col], line=dict(color=color, width=1.3), name=name), row=1, col=1)

    # Panel 2: khối lượng
    vol_colors = np.where(d["close"] >= d["open"], up, down)
    fig.add_trace(go.Bar(x=d["date"], y=d["volume"], marker_color=vol_colors, name="Khối lượng", showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["vol_ma20"], line=dict(color="#f59e0b", width=1), name="KL TB20",
                             showlegend=False), row=2, col=1)

    # Panel 3: RSI
    fig.add_trace(go.Scatter(x=d["date"], y=d["rsi14"], line=dict(color="#7c3aed", width=1.4), name="RSI14"), row=3, col=1)
    for lvl, clr in [(70, down), (50, "gray"), (30, up)]:
        fig.add_hline(y=lvl, line=dict(color=clr, width=1, dash="dash"), row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # Panel 4: MACD
    hist_colors = np.where(d["macd_hist"] >= 0, up, down)
    fig.add_trace(go.Bar(x=d["date"], y=d["macd_hist"], marker_color=hist_colors, name="Histogram"), row=4, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["macd"], line=dict(color="#2563eb", width=1.3), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["macd_signal"], line=dict(color="#f59e0b", width=1.3), name="Signal"),
                  row=4, col=1)

    # Ẩn ngày không có giao dịch (cuối tuần + ngày nghỉ) để nến liền mạch
    all_days = pd.date_range(d["date"].min(), d["date"].max())
    missing = all_days.difference(pd.DatetimeIndex(d["date"]))
    fig.update_xaxes(rangebreaks=[dict(values=missing.strftime("%Y-%m-%d").tolist())])

    fig.update_layout(
        height=900, margin=dict(l=40, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    return fig


# ----------------------------------------------------------------------------
# 5. HÀM CHÍNH CHO DASHBOARD
# ----------------------------------------------------------------------------
def analyze_symbol(symbol: str, df: Optional[pd.DataFrame] = None, last_n: Optional[int] = 250) -> dict:
    """
    Điểm vào duy nhất cho dashboard.
    - symbol: mã cổ phiếu người dùng nhập
    - df: (tuỳ chọn) DataFrame giá nếu dashboard đã có sẵn; nếu không sẽ gọi load_price_data(symbol)
    - last_n: số phiên hiển thị trên biểu đồ (None = toàn bộ)
    """
    raw = df if df is not None else load_price_data(symbol)
    data = compute_indicators(raw)
    result = evaluate_signals(data)
    result["symbol"] = symbol.upper()
    result["data"] = data                                   # DataFrame đã có đủ chỉ báo (nếu cần dùng thêm)
    result["figure"] = build_figure(data, symbol, last_n)   # plotly Figure
    return result


# ----------------------------------------------------------------------------
# 6. VÍ DỤ TÍCH HỢP (chạy: streamlit run technical_analysis.py)
# ----------------------------------------------------------------------------
def _streamlit_demo():
    import streamlit as st
    st.set_page_config(page_title="Phân tích kỹ thuật", layout="wide")
    symbol = st.text_input("Nhập mã cổ phiếu", "VNM").strip().upper()
    if not symbol:
        return
    try:
        r = analyze_symbol(symbol)
    except Exception as e:
        st.error(str(e))
        return

    icon = {"Tích cực": "🟢", "Trung lập": "🟡", "Tiêu cực": "🔴"}[r["signal"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tín hiệu tổng hợp", f"{icon} {r['signal']}", f"{r['score']:+d}/{r['max_score']} điểm")
    c2.metric("Xu hướng", r["trend"])
    c3.metric("Động lượng", r["momentum"])
    st.caption(f"Dữ liệu đến ngày {r['as_of']}")
    st.plotly_chart(r["figure"], use_container_width=True)

    st.subheader("Giải thích tín hiệu")
    st.write(f"**Xu hướng:** {r['trend_explain']}")
    st.write(f"**Động lượng:** {r['momentum_explain']}")
    st.dataframe(pd.DataFrame(r["reasons"])[["rule", "label", "explain"]], use_container_width=True, hide_index=True)
    for n in r["notes"]:
        st.info(n)


if __name__ == "__main__":
    try:
        _streamlit_demo()
    except ModuleNotFoundError:
        print("Hãy import analyze_symbol() vào dashboard của bạn.")