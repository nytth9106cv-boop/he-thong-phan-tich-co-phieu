import io
   import streamlit as st
   from pdf_generator import generate_pdf_report

   # Lấy dữ liệu phân tích của cổ phiếu hiện tại
   current_stock_data = get_stock_analysis_data(ticker)

   # Tạo buffer bộ nhớ thay vì ghi trực tiếp ra đĩa
   pdf_buffer = io.BytesIO()
   generate_pdf_report(current_stock_data, pdf_buffer)
   pdf_bytes = pdf_buffer.getvalue()

   # Nút tải PDF chuẩn UI
   st.download_button(
       label=f"📥 Tải báo cáo PDF ({ticker})",
       data=pdf_bytes,
       file_name=f"{ticker}_BaoCaoPhanTich.pdf",
       mime="application/pdf",
       use_container_width=True
   )