from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def save_text_to_pdf(text, pdf_filename):
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    
    c.setFont("Helvetica", 10)

    lines = '.\n'.join(text.split('.')).split('\n')

    for line in lines:
        c.drawString(50, 700, line) 

        c.translate(0, -15)

    c.save()

if __name__ == "__main__":
    
    with open("/data/audio_summary.txt") as f:
        text = f.read()
        pdf_filename = "/data/output.pdf"
        save_text_to_pdf(text, pdf_filename)
