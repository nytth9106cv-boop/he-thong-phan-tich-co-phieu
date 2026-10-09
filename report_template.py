import os
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)

def setup_vietnamese_fonts():
    """
    Đăng ký font Times New Roman hỗ trợ đầy đủ tiếng Việt UTF-8.
    Tự động tìm kiếm đường dẫn font theo từng hệ điều hành.
    """
    font_registered = False
    
    # Danh sách các đường dẫn font phổ biến trên Windows, macOS, Ubuntu/Linux
    candidates = [
        # Windows
        {"regular": "C:/Windows/Fonts/times.ttf", "bold": "C:/Windows/Fonts/timesbd.ttf", "italic": "C:/Windows/Fonts/timesi.ttf"},
        # macOS
        {"regular": "/System/Library/Fonts/Supplemental/Times New Roman.ttf", "bold": "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf", "italic": "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"},
        {"regular": "/Library/Fonts/Times New Roman.ttf", "bold": "/Library/Fonts/Times New Roman Bold.ttf", "italic": "/Library/Fonts/Times New Roman Italic.ttf"},
        # Linux / Ubuntu
        {"regular": "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", "bold": "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf", "italic": "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf"},
        {"regular": "/usr/share/fonts/truetype/freefont/FreeSerif.ttf", "bold": "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf", "italic": "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf"}
    ]

    for cand in candidates:
        if os.path.exists(cand["regular"]):
            try:
                pdfmetrics.registerFont(TTFont("TimesNewRoman", cand["regular"]))
                bold_path = cand["bold"] if os.path.exists(cand.get("bold", "")) else cand["regular"]
                italic_path = cand["italic"] if os.path.exists(cand.get("italic", "")) else cand["regular"]
                bold_italic_path = cand.get("bold_italic") if os.path.exists(cand.get("bold_italic", "")) else bold_path

                pdfmetrics.registerFont(TTFont("TimesNewRoman", regular_path))
                pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", bold_path))
                pdfmetrics.registerFont(TTFont("TimesNewRoman-Italic", italic_path))
                pdfmetrics.registerFont(TTFont("TimesNewRoman-BoldItalic", bold_italic_path))

                pdfmetrics.registerFontFamily(
                    "TimesNewRoman",
                    normal="TimesNewRoman",
                    bold="TimesNewRoman-Bold",
                    italic="TimesNewRoman-Italic",
                    boldItalic="TimesNewRoman-BoldItalic"
                )
                font_registered = True
                break
            except Exception as e:
                continue

    if not font_registered:
        # Fallback về Helvetica nếu không tìm thấy font cục bộ (chấp nhận hạn chế font Latin)
        pass

# Gọi đăng ký font ngay khi import module
setup_vietnamese_fonts()

class ReportTheme:
    """Bảng màu chủ đạo theo phong cách báo cáo tài chính chuyên nghiệp (Investment Banking / Securities)"""
    PRIMARY = colors.HexColor("#0B2F64")       # Xanh navy đậm chủ đạo
    SECONDARY = colors.HexColor("#1A5F7A")     # Xanh biển phụ trợ
    ACCENT_GOLD = colors.HexColor("#D4AF37")   # Màu vàng kim tạo điểm nhấn
    TEXT_DARK = colors.HexColor("#222222")     # Chữ nội dung
    TEXT_MUTED = colors.HexColor("#666666")    # Chữ ghi chú phụ
    BG_LIGHT = colors.HexColor("#F4F6F9")      # Màu nền bảng
    BORDER_COLOR = colors.HexColor("#D1D5DB")  # Đường viền bảng
    SUCCESS = colors.HexColor("#10B981")       # Màu khuyến nghị Mua / Tăng
    DANGER = colors.HexColor("#EF4444")        # Màu khuyến nghị Bán / Giảm
    WARNING = colors.HexColor("#F59E0B")       # Màu khuyến nghị Theo dõi / Trung lập

class NumberedCanvas(canvas.Canvas):
    """Canvas tùy chỉnh hỗ trợ tính toán tổng số trang và in Header/Footer tự động (Trang X/Y)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, total_pages: int):
        self.saveState()
        font_name = "TimesNewRoman" if "TimesNewRoman" in pdfmetrics.getRegisteredFontNames() else "Times-Roman"
        font_italic = "TimesNewRoman-Italic" if "TimesNewRoman-Italic" in pdfmetrics.getRegisteredFontNames() else "Times-Italic"
        
        # --- Top Header line (Chỉ vẽ từ trang 2 trở đi để giữ bìa/trang 1 thông thoáng) ---
        if self._pageNumber > 1:
            self.setStrokeColor(ReportTheme.SECONDARY)
            self.setLineWidth(0.5)
            self.line(40, A4[1] - 40, A4[0] - 40, A4[1] - 40)
            
            self.setFont(font_italic, 8)
            self.setFillColor(ReportTheme.TEXT_MUTED)
            self.drawString(40, A4[1] - 35, "HỆ THỐNG PHÂN TÍCH CƠ HỘI ĐẦU TƯ CỔ PHIẾU | BÁO CÁO PHÂN TÍCH")

        # --- Bottom Footer ---
        self.setStrokeColor(ReportTheme.BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(40, 45, A4[0] - 40, 45)

        self.setFont(font_italic, 8)
        self.setFillColor(ReportTheme.TEXT_MUTED)
        self.drawString(40, 32, "Nguồn: Dữ liệu giao dịch & BCTC kiểm toán. Báo cáo phục vụ mục đích nghiên cứu học thuật.")
        
        page_text = f"Trang {self._pageNumber} / {total_pages}"
        self.drawRightString(A4[0] - 40, 32, page_text)
        self.restoreState()

def get_report_styles():
    """Tạo bộ styles dựa trên font Times New Roman chuẩn"""
    styles = getSampleStyleSheet()
    font_main = "TimesNewRoman" if "TimesNewRoman" in pdfmetrics.getRegisteredFontNames() else "Times-Roman"
    font_bold = "TimesNewRoman-Bold" if "TimesNewRoman-Bold" in pdfmetrics.getRegisteredFontNames() else "Times-Bold"
    font_italic = "TimesNewRoman-Italic" if "TimesNewRoman-Italic" in pdfmetrics.getRegisteredFontNames() else "Times-Italic"

    custom_styles = {
        "ReportTitle": ParagraphStyle(
            "ReportTitle",
            parent=styles["Normal"],
            fontName=font_bold,
            fontSize=22,
            leading=26,
            textColor=ReportTheme.PRIMARY,
            spaceAfter=6
        ),
        "CompanySub": ParagraphStyle(
            "CompanySub",
            parent=styles["Normal"],
            fontName=font_main,
            fontSize=11,
            leading=15,
            textColor=ReportTheme.TEXT_MUTED,
            spaceAfter=12
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading",
            parent=styles["Normal"],
            fontName=font_bold,
            fontSize=13,
            leading=17,
            textColor=ReportTheme.PRIMARY,
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True
        ),
        "BodyTextCustom": ParagraphStyle(
            "BodyTextCustom",
            parent=styles["Normal"],
            fontName=font_main,
            fontSize=10,
            leading=14.5,
            textColor=ReportTheme.TEXT_DARK,
            spaceAfter=6
        ),
        "BulletPoint": ParagraphStyle(
            "BulletPoint",
            parent=styles["Normal"],
            fontName=font_main,
            fontSize=9.5,
            leading=13.5,
            textColor=ReportTheme.TEXT_DARK,
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=4
        ),
        "TableHead": ParagraphStyle(
            "TableHead",
            parent=styles["Normal"],
            fontName=font_bold,
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
            alignment=1
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName=font_main,
            fontSize=8.5,
            leading=11,
            textColor=ReportTheme.TEXT_DARK,
            alignment=0
        ),
        "TableCellCenter": ParagraphStyle(
            "TableCellCenter",
            parent=styles["Normal"],
            fontName=font_main,
            fontSize=8.5,
            leading=11,
            textColor=ReportTheme.TEXT_DARK,
            alignment=1
        ),
        "TableCellRight": ParagraphStyle(
            "TableCellRight",
            parent=styles["Normal"],
            fontName=font_main,
            fontSize=8.5,
            leading=11,
            textColor=ReportTheme.TEXT_DARK,
            alignment=2
        ),
        "Disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontName=font_italic,
            fontSize=8,
            leading=11,
            textColor=ReportTheme.TEXT_MUTED,
            spaceBefore=8
        )
    }
    
    for name, style in custom_styles.items():
        styles.add(style)
        
    return styles