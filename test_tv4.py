import pandas as pd
import numpy as np
from investment_scoring import score_stock, ranking_table, explain_text

def create_mock_price_df(start_price, days=252, vol_mean=0.001, vol_std=0.02):
    """Tạo DataFrame giá giả lập có thể điều chỉnh độ biến động (volatility)"""
    np.random.seed(42)
    returns = np.random.normal(vol_mean, vol_std, days)
    price = start_price * np.cumprod(1 + returns)
    vol = np.random.randint(1_000_000, 10_000_000, days)
    
    df = pd.DataFrame({
        "Date": pd.date_range(end="2026-10-09", periods=days, freq="B"),
        "Close": price,
        "Volume": vol
    })
    df.attrs["price_unit"] = "vnd"
    return df

def test_different_tickers():
    print("="*75)
    print("CHẠY THỬ NGHIỆM HỆ THỐNG CHẤM ĐIỂM - TCB, VNM, DXG")
    print("="*75)

    # 1. TCB (Ngân hàng) - Chỉ số cơ bản tốt, nợ xấu thấp
    df_tcb = create_mock_price_df(24000)
    fund_tcb = {"roe": 0.20, "pb": 1.1, "car": 0.15, "npl": 0.011, "nim": 0.045, "profit_growth": 0.18}
    tech_tcb = {"score": 4, "max_score": 6, "reasons": [{"rule": "SMA", "score": 1, "label": "Tích cực", "explain": "Giá duy trì trên SMA50."}]}

    # 2. VNM (Sản xuất/Phòng thủ) - Ổn định, độ lệch chuẩn thấp (vol_std=0.01)
    df_vnm = create_mock_price_df(65000, vol_std=0.01) 
    fund_vnm = {"roe": 0.28, "pe": 16, "debt_to_equity": 0.2, "revenue_growth": 0.05, "profit_growth": 0.08}
    tech_vnm = {"score": 0, "max_score": 6, "reasons": [{"rule": "RSI", "score": 0, "label": "Trung lập", "explain": "RSI quanh 50, động lượng đi ngang."}]}

    # 3. DXG (Bất động sản) - Biến động mạnh (vol_std=0.04), nợ vay cao
    df_dxg = create_mock_price_df(15000, vol_std=0.04) 
    fund_dxg = {"roe": 0.05, "pe": 25, "debt_to_equity": 1.8, "profit_growth": -0.2}
    tech_dxg = {"score": -4, "max_score": 6, "reasons": [{"rule": "MACD", "score": -1, "label": "Tiêu cực", "explain": "MACD cắt xuống đường tín hiệu."}]}

    results = {}
    print("Đang tính toán cho TCB...")
    results["TCB"] = score_stock("TCB", price_df=df_tcb, fundamentals=fund_tcb, technical=tech_tcb)
    
    print("Đang tính toán cho VNM...")
    results["VNM"] = score_stock("VNM", price_df=df_vnm, fundamentals=fund_vnm, technical=tech_vnm)
    
    print("Đang tính toán cho DXG...")
    results["DXG"] = score_stock("DXG", price_df=df_dxg, fundamentals=fund_dxg, technical=tech_dxg)

    # In bảng xếp hạng tổng hợp
    df_rank = ranking_table(results)
    print("\n" + "="*75)
    print("BẢNG XẾP HẠNG & ĐIỂM TỔNG HỢP")
    print("="*75)
    print(df_rank.to_markdown(index=False))

    # In báo cáo giải thích chi tiết cho DXG để kiểm tra bắt lỗi rủi ro
    print("\n" + "="*75)
    print("XUẤT BÁO CÁO GIẢI THÍCH MẪU: DXG")
    print("="*75)
    print(explain_text(results["DXG"]))

if __name__ == "__main__":
    test_different_tickers()