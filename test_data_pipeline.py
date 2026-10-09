# -*- coding: utf-8 -*-
"""
test_data_pipeline.py - Kiểm thử TV1 (không cần mạng: dùng phản hồi giả lập của DNSE).
Chạy:  python -m unittest test_data_pipeline -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import data_loader as dl
from data_cleaning import (DataError, DataSourceError, DateRangeError, SymbolError, clean_financial_df,
                           clean_price_data, find_column, parse_number, validate_for_analysis)


def make_payload(n=400, start="2024-01-02", base=25.5, seed=1):
    """Phản hồi DNSE giả: giá theo nghìn đồng, t là epoch giây (00:00 giờ VN)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n)
    c = base * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    t = [int(d.tz_localize("Asia/Ho_Chi_Minh").timestamp()) for d in days]
    return {"t": t, "o": list(c * 0.999), "h": list(c * 1.01), "l": list(c * 0.99), "c": list(c),
            "v": [int(x) for x in rng.integers(2e5, 2e6, n)]}


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        if self._p is None:
            raise ValueError("no json")
        return self._p


class FakeSession:
    def __init__(self, handler):
        self.handler, self.calls = handler, 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        return self.handler(url, params, self.calls)


class TestInputs(unittest.TestCase):
    def test_symbol(self):
        self.assertEqual(dl.normalize_symbol(" vre.vn "), "VRE")
        self.assertEqual(dl.normalize_symbol("HOSE:vcb"), "VCB")
        for bad in ["", None, "V", "VRE!!", "ABCDEFGHIJKLM"]:
            with self.assertRaises(SymbolError):
                dl.normalize_symbol(bad)

    def test_dates(self):
        self.assertEqual(dl.parse_date("31/12/2025"), pd.Timestamp("2025-12-31"))
        with self.assertRaises(DateRangeError):
            dl.parse_date("2025-13-45")
        with self.assertRaises(DateRangeError):
            dl.resolve_range("2026-05-01", "2026-01-01")
        s, e = dl.resolve_range(None, None)
        self.assertLess(s, e)
        _, e2 = dl.resolve_range("2020-01-01", "2099-01-01")
        self.assertLessEqual(e2, dl.today_vn())

    def test_parse_number(self):
        self.assertEqual(parse_number("1,234.5"), 1234.5)
        self.assertEqual(parse_number("1.234,5"), 1234.5)
        self.assertEqual(parse_number("1.234.567"), 1234567.0)
        self.assertEqual(parse_number("(12)"), -12.0)
        self.assertTrue(np.isnan(parse_number("-")))
        self.assertTrue(np.isnan(parse_number("abc")))


class TestCleaning(unittest.TestCase):
    def test_messy_vietnamese_columns_and_issues(self):
        raw = pd.DataFrame({
            "Ngày": ["02/01/2025", "03/01/2025", "03/01/2025", "06/01/2025", "07/01/2025", "08/01/2025", "xx"],
            "Giá mở cửa": ["25,5", "25.6", "25.6", None, "25.7", "25.8", "1"],
            "Giá cao nhất": ["26", "25.5", "25.5", "26", "26", "26", "1"],      # dòng 2: High < Close
            "Giá thấp nhất": ["25", "25", "25", "25", "25", "25", "1"],
            "Giá đóng cửa": ["25.8", "25.9", "25.9", "26", "25.9", "-5", "1"],   # có giá âm
            "Khối lượng": ["1,000", "2,000", "2,000", "", "3,000", "1,000", "1"],
        })
        df, rep = clean_price_data(raw, symbol="vre", source="test")
        self.assertEqual(list(df.columns), ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"])
        self.assertTrue(df["Date"].is_monotonic_increasing)
        self.assertFalse(df["Date"].duplicated().any())
        self.assertTrue((df["Symbol"] == "VRE").all())
        self.assertTrue((df["Close"] > 1000).all())                       # đã đổi về đồng
        self.assertTrue((df["High"] >= df[["Open", "Close"]].max(axis=1)).all())
        self.assertTrue((df["Low"] <= df[["Open", "Close"]].min(axis=1)).all())
        codes = {i["code"] for i in rep.issues}
        for c in ["BAD_DATE", "BAD_CLOSE", "DUPLICATE", "MISSING_OHL", "OHLC_INCONSISTENT", "UNIT", "SHORT_HISTORY"]:
            self.assertIn(c, codes)
        self.assertLess(rep.score, 80)

    def test_weekend_removed_and_bigmove_flagged(self):
        days = pd.date_range("2025-01-01", periods=120, freq="D")           # có cả cuối tuần
        close = np.full(len(days), 30000.0)
        close[60:] = 45000.0                                                  # nhảy +50%
        raw = pd.DataFrame({"Date": days, "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000})
        df, rep = clean_price_data(raw, symbol="ABC")
        self.assertTrue((df["Date"].dt.dayofweek < 5).all())
        self.assertIn("BIG_MOVE", {i["code"] for i in rep.issues})
        self.assertEqual(rep.price_unit_detected, "vnd")

    def test_empty_and_missing_close(self):
        with self.assertRaises(DataError):
            clean_price_data(pd.DataFrame())
        with self.assertRaises(DataError):
            clean_price_data(pd.DataFrame({"Date": ["2025-01-01"], "Open": [1]}))
        with self.assertRaises(DataError):
            validate_for_analysis(pd.DataFrame({"Date": [1]}))

    def test_financial_cleaning(self):
        cols = pd.MultiIndex.from_tuples([("Meta", "CP"), ("Meta", "Năm"), ("Chỉ tiêu định giá", "P/E"), ("Chỉ tiêu", "ROE (%)")])
        df = pd.DataFrame([["VRE", 2025, "12,5", "0.14"], ["VRE", 2024, "10,0", "0.12"]], columns=cols)
        out = clean_financial_df(df)
        self.assertIn("Chỉ tiêu định giá|P/E", out.columns)
        pe = find_column(out, ["P/E"], exact=True)
        self.assertEqual(float(out[pe].iloc[0]), 12.5)


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.end = dl.today_vn()
        self.start = self.end - pd.Timedelta(days=700)

    def _payload_ending_today(self):
        n = 480
        days = pd.bdate_range(end=self.end, periods=n)
        p = make_payload(n)
        p["t"] = [int(d.tz_localize("Asia/Ho_Chi_Minh").timestamp()) for d in days]
        return p

    def test_dnse_happy_path_and_cache(self):
        payload = self._payload_ending_today()
        sess = FakeSession(lambda u, p, n: FakeResp(payload if n == 1 else {"t": [], "c": [], "o": [], "h": [], "l": [], "v": []}))
        df, rep = dl.load_stock_data("vre", self.start, self.end, cache_dir=self.tmp, session=sess, return_report=True)
        self.assertGreater(len(df), 300)
        self.assertGreater(df["Close"].iloc[-1], 1000)
        self.assertIn("DNSE", rep.source)
        self.assertEqual(df.attrs["source"], rep.source)
        calls_after_first = sess.calls
        df2 = dl.load_stock_data("VRE", self.start, self.end, cache_dir=self.tmp, session=sess)
        self.assertEqual(sess.calls, calls_after_first)                     # lần 2 dùng cache, không gọi mạng
        self.assertEqual(len(df2), len(df))

    def test_retry_then_success(self):
        payload = self._payload_ending_today()
        sess = FakeSession(lambda u, p, n: FakeResp(None, 503) if n < 3 else FakeResp(payload))
        with mock.patch("data_loader.time.sleep"):
            df = dl.load_stock_data("VRE", self.start, self.end, use_cache=False, session=sess, source="dnse")
        self.assertGreater(len(df), 100)

    def test_network_down_falls_back_to_local_file(self):
        import requests
        def boom(u, p, n):
            raise requests.ConnectionError("no network")
        path = Path(self.tmp) / "VRE.xlsx"
        days = pd.bdate_range(end=self.end, periods=300)
        c = np.linspace(25000, 30000, len(days))
        with pd.ExcelWriter(path) as w:       # tiêu đề nằm ở dòng thứ 3 như file Excel tự làm
            pd.DataFrame([["Dữ liệu VRE"], [None]]).to_excel(w, header=False, index=False, startrow=0)
            pd.DataFrame({"Ngày": days.strftime("%d/%m/%Y"), "Mở cửa": c, "Cao nhất": c * 1.01, "Thấp nhất": c * .99,
                          "Đóng cửa": c, "Khối lượng": 1000}).to_excel(w, index=False, startrow=2)
        with mock.patch("data_loader.time.sleep"), \
             mock.patch("data_loader.fetch_vnstock_ohlc", side_effect=DataSourceError("vnstock giả lập: lỗi")):
            df, rep = dl.load_stock_data("VRE", self.start, self.end, cache_dir=self.tmp, local_file=path,
                                         session=FakeSession(boom), return_report=True)
        self.assertGreater(len(df), 200)
        self.assertIn("File nội bộ", rep.source)
        self.assertIn("FALLBACK", {i["code"] for i in rep.issues})

    def test_all_sources_fail_raises_vietnamese_error(self):
        import requests
        def boom(u, p, n):
            raise requests.ConnectionError("no network")
        with mock.patch("data_loader.time.sleep"), \
             mock.patch("data_loader.fetch_vnstock_ohlc", side_effect=DataSourceError("vnstock lỗi")):
            with self.assertRaises(DataSourceError) as cm:
                dl.load_stock_data("VRE", self.start, self.end, cache_dir=self.tmp, session=FakeSession(boom))
        self.assertIn("Không lấy được dữ liệu", str(cm.exception))

    def test_stale_cache_used_when_offline(self):
        payload = self._payload_ending_today()
        ok = FakeSession(lambda u, p, n: FakeResp(payload if n == 1 else {"t": [], "c": []}))
        dl.load_stock_data("VRE", self.start, self.end, cache_dir=self.tmp, session=ok)
        import requests
        def boom(u, p, n):
            raise requests.ConnectionError("x")
        with mock.patch("data_loader.time.sleep"), mock.patch("data_loader._is_fresh", return_value=False), \
             mock.patch("data_loader.fetch_vnstock_ohlc", side_effect=DataSourceError("off")):
            df, rep = dl.load_stock_data("VRE", self.start, self.end, cache_dir=self.tmp,
                                         session=FakeSession(boom), return_report=True)
        self.assertIn("STALE_CACHE", {i["code"] for i in rep.issues})
        self.assertGreater(len(df), 100)

    def test_unknown_symbol_gives_symbol_error(self):
        sess = FakeSession(lambda u, p, n: FakeResp({"t": [], "o": [], "h": [], "l": [], "c": [], "v": []}))
        with mock.patch("data_loader.fetch_vnstock_ohlc", side_effect=SymbolError("không có")):
            with self.assertRaises(SymbolError):
                dl.load_stock_data("ZZZ", self.start, self.end, use_cache=False, session=sess)

    def test_bad_payloads(self):
        with self.assertRaises(DataSourceError):
            dl.parse_dnse_payload({"x": 1})
        with self.assertRaises(DataSourceError):
            dl.parse_dnse_payload({"t": [1, 2], "c": [1]})
        with self.assertRaises(DataSourceError):
            dl.parse_dnse_payload("hello")

    def test_index_and_benchmark_series(self):
        payload = self._payload_ending_today()
        payload["c"] = [1200 + i for i in range(len(payload["c"]))]
        for k in "ohl":
            payload[k] = payload["c"]
        sess = FakeSession(lambda u, p, n: FakeResp(payload if n == 1 else {"t": [], "c": []}))
        idx = dl.load_index_data("VNINDEX", self.start, self.end, cache_dir=self.tmp, session=sess)
        self.assertIn("/ohlcs/index", "/ohlcs/index")
        s = dl.benchmark_series(idx)
        self.assertIsInstance(s.index, pd.DatetimeIndex)

    def test_integration_with_tv4(self):
        """DataFrame của TV1 phải chạy được thẳng vào module chấm điểm của TV4 (nếu có trong thư mục)."""
        try:
            from investment_scoring import score_stock
        except ImportError:
            self.skipTest("Không có investment_scoring.py trong thư mục")
        payload = self._payload_ending_today()
        sess = FakeSession(lambda u, p, n: FakeResp(payload if n == 1 else {"t": [], "c": []}))
        df = dl.load_stock_data("VRE", self.start, self.end, cache_dir=self.tmp, session=sess)
        res = score_stock("VRE", df, {"roe": 0.12, "pe": 15, "pb": 1.5, "debt_to_equity": 1.0})
        self.assertTrue(0 <= res.total_score <= 100)

    def test_financial_snapshot_with_fake_vnstock(self):
        ratio = pd.DataFrame({("Meta", "Năm"): [2025, 2024], ("Chỉ tiêu định giá", "P/E"): [14.2, 12.0],
                              ("Chỉ tiêu định giá", "P/B"): [1.6, 1.4], ("Chỉ tiêu sinh lời", "ROE (%)"): [0.13, 0.11],
                              ("Chỉ tiêu sinh lời", "ROA (%)"): [0.05, 0.04]})
        inc = pd.DataFrame({"Năm": [2025, 2024], "Doanh thu thuần": [1200.0, 1000.0], "Lợi nhuận sau thuế": [300.0, 200.0]})

        class Fin:
            def income_statement(self, period="year", lang="vi", dropna=True): return inc
            def balance_sheet(self, period="year"): raise RuntimeError("boom")
            def cash_flow(self, period="year", lang="vi", dropna=True): return pd.DataFrame()
            def ratio(self, period="year", lang="vi", dropna=True): return ratio

        class Comp:
            def overview(self): return pd.DataFrame({"symbol": ["VRE"]})

        class Stock:
            finance, company = Fin(), Comp()

        class FakeVnstock:
            def stock(self, symbol, source): return Stock()

        with mock.patch("data_loader._import_vnstock", return_value=FakeVnstock):
            fin = dl.load_financial_data("VRE")
        self.assertTrue(any("balance_sheet" in w for w in fin.warnings))      # lỗi từng phần chỉ cảnh báo
        snap = dl.get_fundamentals_snapshot(fin)
        self.assertAlmostEqual(snap["pe"], 14.2)
        self.assertAlmostEqual(snap["roe"], 0.13)
        self.assertAlmostEqual(snap["revenue_growth"], 0.2)
        self.assertAlmostEqual(snap["profit_growth"], 0.5)
        self.assertFalse(snap["is_financial"])

    def test_financial_total_failure(self):
        class FakeVnstock:
            def stock(self, symbol, source): raise RuntimeError("API đổi rồi")
        with mock.patch("data_loader._import_vnstock", return_value=FakeVnstock):
            with self.assertRaises(DataSourceError):
                dl.load_financial_data("VRE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
