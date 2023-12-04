from fpdf import FPDF


file = open("/data/summ.txt")
pdf = FPDF()
pdf.add_page()
for text in file:
    pdf.set_font("Arial",size=20)
    pdf.multi_cell(w=200,h=10, txt=text,align="L",)
    pdf.output("/data/summ_to_pdf.pdf")