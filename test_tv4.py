import pandas as pd
from investment_scoring import InvestmentScorer

def test_investment_scoring():
    vpb_fund = {'is_financial': True, 'roe': 0.18, 'pb': 1.2, 'car': 12.5}
    vpb_tech = {'score': 4, 'max_score': 6}
    vpb_risk = {'volatility': 0.25, 'max_drawdown': -0.15, 'avg_volume_20d': 15000000}

    mwg_fund = {'is_financial': False, 'roe': 0.12, 'pe': None}
    mwg_tech = {'score': -2, 'max_score': 6}
    mwg_risk = {'volatility': 0.45, 'max_drawdown': -0.35, 'avg_volume_20d': 5000000}

    danh_sach_test = [
        ("VPB", vpb_fund, vpb_tech, vpb_risk),
        ("MWG", mwg_fund, mwg_tech, mwg_risk)
    ]

    ket_qua = []
    print("="*50)
    print("CHẠY THỬ NGHIỆM HỆ THỐNG CHẤM ĐIỂM (TV4)")
    print("="*50)

    for ma_cp, fund, tech, risk in danh_sach_test:
        scorer = InvestmentScorer(fund, tech, risk)
        result = scorer.evaluate()
        
        ket_qua.append({
            "Mã CP": ma_cp,
            "Tổng điểm": result['total_score'],
            "Phân loại": result['classification'],
            "Cơ bản": result['details']['Điểm cơ bản'],
            "Kỹ thuật": result['details']['Điểm kỹ thuật'],
            "Rủi ro": result['details']['Điểm quản trị rủi ro']
        })

    bang_diem_df = pd.DataFrame(ket_qua)
    print("\nBẢNG ĐIỂM TỔNG HỢP:")
    print(bang_diem_df.to_markdown(index=False))

if __name__ == "__main__":
    test_investment_scoring()