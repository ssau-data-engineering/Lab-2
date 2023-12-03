from fpdf import FPDF

file = open("/data/summ.txt","r")
pdf = FPDF()
pdf.add_page()
for text in file:
    pdf.set_font("Arial", size=15)
    pdf.multi_cell(0, 5, txt=text, align="J")
    pdf.output("/data/output.pdf")