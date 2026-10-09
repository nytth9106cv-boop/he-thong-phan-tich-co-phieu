from pprint import pprint
from fundamental_analysis import run_fundamental_analysis

# Đổi 2 dòng này để test.
ticker = "TCB"
company_type = "bank"  # "bank" hoặc "regular"

result = run_fundamental_analysis(
    ticker=ticker,
    company_type=company_type,
)

print("\n===== PERIOD MAPPING CHECK =====")
pprint(result["period_mapping"])

print("\n===== BCTC CHUẨN HÓA =====")
print(result["financial_data"].to_string(index=False))

print("\n===== FINANCIAL RATIOS =====")
print(result["ratio_data"].to_string(index=False))

print("\n===== PERIOD COMPARISON =====")
print(result["comparison_data"].to_string(index=False))

print("\n===== FUNDAMENTAL ANALYSIS =====")
pprint(result["analysis"])

print("\n===== DATA QUALITY =====")
pprint(result["analysis"]["data_quality"])
