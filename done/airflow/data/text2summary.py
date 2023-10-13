from transformers import pipeline
import glob
import os

for txt_file_path in glob.glob('/data/lab2_output/text/*.txt'):
    _, filename = os.path.split(txt_file_path)
    output_summary_file_path = f'/data/lab2_output/summary/{filename}'
    if os.path.isfile(output_summary_file_path):
        continue # skip, already exist

    with open(txt_file_path, 'r') as fr:
        text = fr.read()

    summarizer = pipeline("summarization", max_length=9) # Since our input small
    text_summ = summarizer(text)
    
    with open(output_summary_file_path, "w") as text_file:
        text_file.write(text_summ[0]['summary_text'])


