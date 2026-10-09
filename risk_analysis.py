import numpy as np
import pandas as pd

class RiskAnalyzer:
    def __init__(self, price_df: pd.DataFrame, benchmark_series: pd.Series = None):
        """
        price_df: DataFrame giá từ TV1 (cần cột Date, Close, Volume)[cite: 5].
        benchmark_series: Series giá đóng cửa của VNINDEX từ TV1 để tính Beta.
        """
        self.df = price_df.sort_values('Date').copy()
        self.df['Return'] = self.df['Close'].pct_change()
        self.benchmark = benchmark_series

    def calculate_volatility(self, window: int = 252) -> float:
        if len(self.df) < 2: return np.nan
        return float(self.df['Return'].std() * np.sqrt(window))

    def calculate_max_drawdown(self) -> float:
        cum_ret = (1 + self.df['Return'].fillna(0)).cumprod()
        peak = cum_ret.expanding(min_periods=1).max()
        drawdown = (cum_ret / peak) - 1
        return float(drawdown.min())

    def calculate_liquidity_risk(self, window: int = 20) -> float:
        return float(self.df['Volume'].tail(window).mean())

    def calculate_beta(self) -> float:
        if self.benchmark is None or len(self.df) < 2: return 1.0
        stock_ret = self.df.set_index('Date')['Return'].dropna()
        bench_ret = self.benchmark.pct_change().dropna()
        aligned = pd.concat([stock_ret, bench_ret], axis=1, join='inner').dropna()
        if len(aligned) < 30: return 1.0
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        var = np.var(aligned.iloc[:, 1])
        return float(cov / var) if var > 0 else 1.0

    def get_risk_metrics(self) -> dict:
        return {
            "volatility": self.calculate_volatility(),
            "max_drawdown": self.calculate_max_drawdown(),
            "avg_volume_20d": self.calculate_liquidity_risk(),
            "beta": self.calculate_beta()
        }