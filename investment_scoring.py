import pandas as pd

class InvestmentScorer:
    def __init__(self, fundamental_snap: dict, tech_result: dict, risk_metrics: dict):
        self.fund = fundamental_snap
        self.tech = tech_result
        self.risk = risk_metrics
        self.details = {}

    def _scale(self, value, min_v, max_v, inverse=False):
        if pd.isna(value) or value is None: return 50.0 
        val = max(min_v, min(value, max_v))
        score = ((val - min_v) / (max_v - min_v)) * 100
        return 100 - score if inverse else score

    def evaluate(self) -> dict:
        is_fin = self.fund.get('is_financial', False)
        roe_score = self._scale(self.fund.get('roe', 0), 0, 0.25)
        
        if is_fin:
            pb_score = self._scale(self.fund.get('pb', 1.5), 0.5, 3.0, inverse=True)
            fund_score = (roe_score * 0.5) + (pb_score * 0.5)
        else:
            pe_score = self._scale(self.fund.get('pe', 15), 5, 25, inverse=True)
            fund_score = (roe_score * 0.5) + (pe_score * 0.5)

        t_score_raw = self.tech.get('score', 0)
        t_max = self.tech.get('max_score', 6)
        tech_score = self._scale(t_score_raw, -t_max, t_max)

        vol_score = self._scale(self.risk.get('volatility', 0.3), 0.1, 0.6, inverse=True)
        mdd_score = self._scale(abs(self.risk.get('max_drawdown', -0.2)), 0.05, 0.5, inverse=True)
        risk_score = (vol_score * 0.5) + (mdd_score * 0.5)

        total = (fund_score * 0.4) + (tech_score * 0.3) + (risk_score * 0.3)
        
        self.details = {
            "Điểm cơ bản": round(fund_score, 1),
            "Điểm kỹ thuật": round(tech_score, 1),
            "Điểm quản trị rủi ro": round(risk_score, 1)
        }
        
        return {
            "total_score": round(total, 1),
            "classification": self._classify(total),
            "explanation": self._explain(),
            "details": self.details
        }

    def _classify(self, score: float) -> str:
        if score >= 75: return "Rất hấp dẫn"
        if score >= 60: return "Khả quan"
        if score >= 45: return "Trung lập"
        return "Rủi ro cao"

    def _explain(self) -> str:
        best = max(self.details, key=self.details.get)
        worst = min(self.details, key=self.details.get)
        return (f"Cổ phiếu nhận được lực kéo chính từ {best} ({self.details[best]}/100 điểm), "
                f"tuy nhiên {worst} đang là yếu tố kìm hãm ({self.details[worst]}/100 điểm).")