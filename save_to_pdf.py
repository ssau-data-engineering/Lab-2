from fpdf import FPDF

file = open("/data/summ.txt","r")
pdf = FPDF()
pdf.add_page()
for text in file:
    if len(text) <= 20:
        pdf.set_font("Arial","B",size=18) # For title text
        pdf .cell(w=200,h=10, txt=text, Ln=1,align="C")
    else:
        pdf.set_font("Arial",size=15) # For paragraph text
        pdf .multi_cell(w=0,h=10, txt=text,align="L")
    pdf. output ( "/data/output.pdf")