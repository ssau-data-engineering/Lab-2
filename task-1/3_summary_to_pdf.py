from fpdf import FPDF

file = open("/data/summary.txt","r")
pdf = FPDF()
pdf.add_page()
for text in file:
    pdf.set_font('Helvetica', '', size=14)
    pdf.multi_cell(w=100, h=10, border = 1, txt = text, align = 'J')
    pdf.output("/data/pdf_result.pdf")