import os
import io
import math
from datetime import datetime
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")  # Chế độ headless cho server/script không mở cửa sổ GUI
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)

from report_template import (
    ReportTheme, NumberedCanvas, get_report_styles
)

def generate_technical_chart(price_history: Optional[Dict[str, list]] = None) -> io.BytesIO:
    """
    Sinh biểu đồ kỹ thuật (Giá, MA20, MA50 và Khối lượng).
    Nếu thiếu dữ liệu, tự động sinh dữ liệu mẫu hợp lý hoặc biểu đồ cảnh báo.
    """
    buf = io.BytesIO()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 3.2), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )
    plt.subplots_adjust(hspace=0.08, top=0.92, bottom=0.15, left=0.1, right=0.95)

    if not price_history or len(price_history.get("dates", [])) < 5:
        # Trường hợp thiếu dữ liệu: thông báo rõ trên đồ thị
        ax1.text(0.5, 0.5, "Dữ liệu lịch sử giá chưa đủ để dựng biểu đồ kỹ thuật", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes, color="#EF4444", fontsize=11, fontweight='bold')
        ax2.set_visible(False)
    else:
        dates = price_history["dates"]
        closes = np.array(price_history["closes"])
        volumes = np.array(price_history.get("volumes", [0] * len(dates)))
        x_indices = np.arange(len(dates))

        # Đường giá và Moving Average
        ax1.plot(x_indices, closes, label="Giá đóng cửa", color="#0B2F64", linewidth=1.8)
        if len(closes) >= 20:
            ma20 = np.convolve(closes, np.ones(20)/20, mode='valid')
            ax1.plot(x_indices[19:], ma20, label="MA20", color="#F59E0B", linestyle="--", linewidth=1.2)
        if len(closes) >= 50:
            ma50 = np.convolve(closes, np.ones(50)/50, mode='valid')
            ax1.plot(x_indices[49:], ma50, label="MA50", color="#10B981", linestyle=":", linewidth=1.3)

        ax1.set_ylabel("Giá (VNĐ)", fontsize=8, color="#333333")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.8)
        ax1.tick_params(labelsize=7.5)

        # Cột khối lượng
        bar_colors = ["#10B981" if i > 0 and closes[i] >= closes[i-1] else "#EF4444" for i in range(len(closes))]
        ax2.bar(x_indices, volumes / 1e6, color=bar_colors, alpha=0.75, width=0.8)
        ax2.set_ylabel("KL (Tr.cp)", fontsize=8, color="#333333")
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.tick_params(labelsize=7.5)

        # X-tick dates (hiển thị 6 mốc cách đều)
        step = max(1, len(dates) // 6)
        tick_pos = list(x_indices[::step])
        tick_labels = [dates[i] for i in tick_pos]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_labels, rotation=0, fontsize=7.5)

    fig.patch.set_facecolor("#FFFFFF")
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_financial_chart(financials: Optional[Dict[str, list]] = None) -> io.BytesIO:
    """Sinh biểu đồ doanh thu & lợi nhuận 4 quý gần nhất."""
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    plt.subplots_adjust(top=0.88, bottom=0.18, left=0.1, right=0.95)

    if not financials or len(financials.get("quarters", [])) == 0:
        ax.text(0.5, 0.5, "Chưa cập nhật dữ liệu tài chính quý",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color="#666666", fontsize=10)
    else:
        quarters = financials["quarters"]
        revenue = np.array(financials["revenue"]) / 1e9  # quy đổi sang Tỷ VNĐ
        net_profit = np.array(financials["net_profit"]) / 1e9

        x = np.arange(len(quarters))
        width = 0.35

        ax.bar(x - width/2, revenue, width, label="Doanh thu thuần (Tỷ)", color="#1A5F7A", alpha=0.85)
        ax.bar(x + width/2, net_profit, width, label="Lợi nhuận ròng (Tỷ)", color="#D4AF37", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(quarters, fontsize=8)
        ax.set_ylabel("Tỷ đồng", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=7.5)
        ax.tick_params(labelsize=7.5)

    fig.patch.set_facecolor("#FFFFFF")
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def build_header_card(data: Dict[str, Any], styles: Dict[str, Any]) -> Table:
    """Xây dựng Header Banner chứa Mã CK, Tên công ty, Khuyến nghị và Giá mục tiêu"""
    ticker = data.get("ticker", "UNKNOWN").upper()
    company_name = data.get("company_name", "Doanh nghiệp chưa xác định")
    sector = data.get("sector", "Chưa phân loại")
    report_date = data.get("report_date", datetime.now().strftime("%d/%m/%Y"))
    
    recommendation = data.get("recommendation", "THEO DÕI").upper()
    target_price = data.get("target_price", "N/A")
    current_price = data.get("current_price", "N/A")
    upside = data.get("upside", "N/A")

    # Xác định màu sắc khuyến nghị
    rec_color = ReportTheme.SUCCESS if "MUA" in recommendation else (
        ReportTheme.DANGER if "BÁN" in recommendation else ReportTheme.WARNING
    )

    left_content = [
        Paragraph(f"<b>BÁO CÁO PHÂN TÍCH DOANH NGHIỆP: {ticker}</b>", styles["ReportTitle"]),
        Paragraph(f"<b>{company_name}</b> | Ngành: {sector} | Ngày lập: {report_date}", styles["CompanySub"])
    ]

    rec_badge = f"""
    <font color="{rec_color.hexval()}"><b>{recommendation}</b></font><br/>
    <font size="8" color="#555555">Giá hiện tại: <b>{current_price}</b></font><br/>
    <font size="8" color="#555555">Giá mục tiêu: <b>{target_price}</b> ({upside})</font>
    """
    right_content = Paragraph(rec_badge, styles["TableCellRight"])

    header_table = Table(
        [[left_content, right_content]],
        colWidths=[12.5 * cm, 5.5 * cm]
    )
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    return header_table

def build_metrics_table(metrics: Dict[str, Any], styles: Dict[str, Any]) -> Table:
    """Tạo bảng chỉ số định giá và cơ bản then chốt (P/E, P/B, ROE, Vốn hóa...)"""
    def format_val(val, suffix=""):
        if val is None or val == "" or str(val).strip().upper() == "N/A":
            return "N/A"
        return f"{val}{suffix}"

    data_matrix = [
        [
            Paragraph("Chỉ số", styles["TableHead"]),
            Paragraph("Giá trị", styles["TableHead"]),
            Paragraph("Chỉ số", styles["TableHead"]),
            Paragraph("Giá trị", styles["TableHead"]),
            Paragraph("Chỉ số", styles["TableHead"]),
            Paragraph("Giá trị", styles["TableHead"]),
        ],
        [
            Paragraph("Vốn hóa", styles["TableCell"]),
            Paragraph(format_val(metrics.get("market_cap"), " Tỷ"), styles["TableCellRight"]),
            Paragraph("P/E (TTM)", styles["TableCell"]),
            Paragraph(format_val(metrics.get("pe"), "x"), styles["TableCellRight"]),
            Paragraph("ROE", styles["TableCell"]),
            Paragraph(format_val(metrics.get("roe"), "%"), styles["TableCellRight"]),
        ],
        [
            Paragraph("Số CP lưu hành", styles["TableCell"]),
            Paragraph(format_val(metrics.get("shares_outstanding"), " Tr"), styles["TableCellRight"]),
            Paragraph("P/B", styles["TableCell"]),
            Paragraph(format_val(metrics.get("pb"), "x"), styles["TableCellRight"]),
            Paragraph("ROA", styles["TableCell"]),
            Paragraph(format_val(metrics.get("roa"), "%"), styles["TableCellRight"]),
        ],
        [
            Paragraph("KLGD TB 10 phiên", styles["TableCell"]),
            Paragraph(format_val(metrics.get("avg_volume_10d"), " cp"), styles["TableCellRight"]),
            Paragraph("EPS (VNĐ)", styles["TableCell"]),
            Paragraph(format_val(metrics.get("eps")), styles["TableCellRight"]),
            Paragraph("Tỷ suất cổ tức", styles["TableCell"]),
            Paragraph(format_val(metrics.get("dividend_yield"), "%"), styles["TableCellRight"]),
        ]
    ]

    t = Table(data_matrix, colWidths=[2.8*cm, 3.2*cm, 2.8*cm, 3.2*cm, 2.8*cm, 3.2*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), ReportTheme.PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, ReportTheme.BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, ReportTheme.BG_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    return t

def generate_pdf_report(report_data: Dict[str, Any], output_target: Any = "sample_report.pdf") -> str:
    """
    Hàm sinh tài liệu PDF hoàn chỉnh.
    output_target có thể là đường dẫn file string hoặc io.BytesIO để tích hợp Streamlit download.
    """
    doc = SimpleDocTemplate(
        output_target,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=55
    )

    styles = get_report_styles()
    story = []

    # 1. Header Card & Khuyến nghị
    story.append(build_header_card(report_data, styles))
    story.append(HRFlowable(width="100%", thickness=1, color=ReportTheme.PRIMARY, spaceAfter=8, spaceBefore=4))

    # 2. Bảng tổng quan chỉ số
    story.append(Paragraph("1. TỔNG QUAN CHỈ SỐ TÀI CHÍNH VÀ ĐỊNH GIÁ", styles["SectionHeading"]))
    metrics = report_data.get("metrics", {})
    story.append(build_metrics_table(metrics, styles))
    story.append(Spacer(1, 10))

    # 3. Phân tích Kỹ thuật & Biểu đồ giá
    story.append(Paragraph("2. PHÂN TÍCH KỸ THUẬT & DỮ LIỆU GIAO DỊCH", styles["SectionHeading"]))
    tech_comment = report_data.get("technical_summary", "Chưa có nhận định kỹ thuật.")
    story.append(Paragraph(tech_comment, styles["BodyTextCustom"]))

    tech_img_buf = generate_technical_chart(report_data.get("price_history"))
    story.append(Image(tech_img_buf, width=18*cm, height=8*cm))
    story.append(Spacer(1, 10))

    # 4. Phân tích Cơ bản & Kết quả kinh doanh
    story.append(Paragraph("3. PHÂN TÍCH CƠ BẢN & HIỆU QUẢ HOẠT ĐỘNG", styles["SectionHeading"]))
    fundamental_comment = report_data.get("fundamental_summary", "Chưa có nhận định cơ bản.")
    story.append(Paragraph(fundamental_comment, styles["BodyTextCustom"]))

    fin_img_buf = generate_financial_chart(report_data.get("financials"))
    story.append(Image(fin_img_buf, width=18*cm, height=6.5*cm))
    story.append(Spacer(1, 10))

    # Đảm bảo sang trang sạch đẹp cho luận điểm và kết luận
    story.append(PageBreak())

    # 5. Điểm nhấn đầu tư (Investment Highlights)
    story.append(Paragraph("4. LUẬN ĐIỂM ĐẦU TƯ CỐT LÕI", styles["SectionHeading"]))
    investment_points = report_data.get("investment_points", [])
    if investment_points:
        for pt in investment_points:
            story.append(Paragraph(f"• <b>{pt.get('title', '')}:</b> {pt.get('detail', '')}", styles["BulletPoint"]))
    else:
        story.append(Paragraph("<i>Chưa có dữ liệu luận điểm đầu tư ghi nhận.</i>", styles["BodyTextCustom"]))
    story.append(Spacer(1, 10))

    # 6. Rủi ro đầu tư (Investment Risks)
    story.append(Paragraph("5. RỦI RO ĐẦU TƯ CẦN THEO DÕI", styles["SectionHeading"]))
    risks = report_data.get("risks", [])
    if risks:
        for rk in risks:
            story.append(Paragraph(f"• <b>{rk.get('title', '')}:</b> {rk.get('detail', '')}", styles["BulletPoint"]))
    else:
        story.append(Paragraph("<i>Không có rủi ro bất thường được ghi nhận.</i>", styles["BodyTextCustom"]))
    story.append(Spacer(1, 10))

    # 7. Kết luận & Định giá
    story.append(Paragraph("6. KẾT LUẬN & ĐỊNH GIÁ MỤC TIÊU", styles["SectionHeading"]))
    conclusion = report_data.get("conclusion", "Chưa có kết luận định giá.")
    story.append(Paragraph(conclusion, styles["BodyTextCustom"]))
    story.append(Spacer(1, 10))

    # 8. Nguồn dữ liệu & Miễn trừ trách nhiệm
    story.append(Paragraph("7. NGUỒN DỮ LIỆU & TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM", styles["SectionHeading"]))
    sources = report_data.get("sources", "Nguồn: Sở Giao dịch Chứng khoán (HOSE/HNX), BCTC quý và kiểm toán độc lập, Tổng hợp hệ thống.")
    story.append(Paragraph(f"<b>Nguồn tham khảo:</b> {sources}", styles["BodyTextCustom"]))
    
    disclaimer_text = (
        "Báo cáo này được tự động trích xuất bởi Hệ thống Phân tích Cơ hội Đầu tư Cổ phiếu phục vụ mục đích nghiên cứu "
        "và học thuật. Mọi số liệu và nhận định không cấu thành lời mời chào mua hay bán bất kỳ chứng khoán nào. "
        "Nhà đầu tư tự chịu trách nhiệm đối với quyết định đầu tư của mình."
    )
    story.append(Paragraph(disclaimer_text, styles["Disclaimer"]))

    # Build PDF qua custom canvas NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return str(output_target)

if __name__ == "__main__":
    sample_stock_data = {
        "ticker": "FPT",
        "company_name": "Công ty Cổ phần FPT",
        "sector": "Công nghệ thông tin",
        "report_date": datetime.now().strftime("%d/%m/%Y"),
        "recommendation": "MUA",
        "current_price": "135,000 đ",
        "target_price": "162,000 đ",
        "upside": "+20.0%",
        "metrics": {
            "market_cap": "198,420",
            "pe": "24.5",
            "pb": "5.6",
            "roe": "28.4",
            "roa": "14.2",
            "shares_outstanding": "1,469",
            "avg_volume_10d": "3,450,000",
            "eps": "5,510",
            "dividend_yield": "2.2"
        },
        "technical_summary": (
            "Cổ phiếu FPT duy trì xu hướng tăng trung và dài hạn vững chắc. Đường giá vận động trên các đường MA20 "
            "và MA50 ngày. Chỉ báo RSI dao động quanh mức 58 điểm, thể hiện xung lực mua ổn định và không rơi vào "
            "vùng quá mua. Khối lượng giao dịch gia tăng trong các phiên phục hồi cho thấy dòng tiền tổ chức tiếp tục nâng đỡ."
        ),
        "price_history": {
            "dates": ["T05/26", "T06/26", "T07/26", "T08/26", "T09/26", "T10/26"],
            "closes": [115000, 120000, 128000, 126000, 131000, 135000],
            "volumes": [3200000, 2900000, 4100000, 2500000, 3800000, 4200000]
        },
        "fundamental_summary": (
            "Mảng xuất khẩu phần mềm tiếp tục là động lực tăng trưởng cốt lõi với doanh thu từ thị trường Nhật Bản và APAC "
            "tăng trưởng trên 30% YoY. Doanh thu chuyển đổi số tăng tốc nhờ các hợp đồng ký mới quy mô lớn liên quan đến GenAI "
            "và điện toán đám mây."
        ),
        "financials": {
            "quarters": ["Q3/25", "Q4/25", "Q1/26", "Q2/26"],
            "revenue": [13800e9, 15100e9, 14200e9, 16300e9],
            "net_profit": [2450e9, 2750e9, 2580e9, 2980e9]
        },
        "investment_points": [
            {
                "title": "Tăng trưởng xuất khẩu phần mềm bền vững",
                "detail": "Doanh thu ký mới tại thị trường nước ngoài đạt hơn 1 tỷ USD, duy trì biên lợi nhuận trước thuế trên 18%."
            },
            {
                "title": "Hệ sinh thái AI & Bán dẫn đón đầu chu kỳ mới",
                "detail": "Hợp tác chiến lược xây dựng AI Factory thúc đẩy dịch vụ Cloud và đào tạo nhân lực bán dẫn quy mô lớn."
            },
            {
                "title": "Sức khỏe tài chính lành mạnh",
                "detail": "Tỷ lệ đòn bẩy tài chính thấp, dòng tiền thuần từ hoạt động kinh doanh duy trì dương và dồi dào."
            }
        ],
        "risks": [
            {
                "title": "Rủi ro biến động tỷ giá (JPY, USD)",
                "detail": "Đồng Yên Nhật biến động có thể ảnh hưởng đến doanh thu quy đổi từ thị trường trọng điểm Nhật Bản."
            },
            {
                "title": "Chi phí đầu tư mở rộng hạ tầng Data Center",
                "detail": "Chi phí khấu hao gia tăng trong giai đoạn đầu vận hành các trung tâm dữ liệu mới."
            }
        ],
        "conclusion": (
            "Chúng tôi duy trì khuyến nghị MUA đối với cổ phiếu FPT với giá mục tiêu 162,000 VNĐ/cổ phiếu, tương ứng mức P/E "
            "mục tiêu 26.5x cho dự phóng lợi nhuận năm 2026. FPT tiếp tục là cổ phiếu phòng thủ và tăng trưởng hàng đầu thị trường."
        ),
        "sources": "BCTC hợp nhất FPT kiểm toán, Bloomberg, Vietstock, Ước tính hệ thống phân tích."
    }

    output_filename = "sample_report.pdf"
    print(f"Đang sinh báo cáo mẫu: {output_filename}...")
    generate_pdf_report(sample_stock_data, output_filename)
    print(f"Hoàn thành sinh file: {output_filename}")