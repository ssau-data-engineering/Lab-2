from fpdf import FPDF

file = open("/data/summary_text.txt","r")
pdf = FPDF()
pdf.add_page()
for text in file:
    pdf.set_font("Times","BI", size=15)
    pdf.multi_cell(w=0,h=10, txt=text, align="J")
    pdf.output("/data/result.pdf")