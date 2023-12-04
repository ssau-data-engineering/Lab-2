import whisper
from transformers import pipeline
from fpdf import FPDF

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
model = whisper.load_model("small")

result = model.transcribe('/data/audio.mp3',language = 'en')

summary = summarizer(result["text"], max_length=130, min_length=30, do_sample=False)
 
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size = 15)
pdf.multi_cell(200, 10, txt = summary[0]['summary_text'],
          align = 'C')
 
pdf.output("summary.pdf","/data/")  