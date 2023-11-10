import os
from tqdm.auto import tqdm
from transformers import pipeline
from text2pdf import text_to_pdf


SUPPORTED_EXT = [".txt"]

MODEL_PATH = "model"
REPORT_PATH = "reports"
TEXT_PATH = "texts"


def is_valid_ext(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext in SUPPORTED_EXT

def read_text_file(path: str) -> str:
    with open(path, "r") as f:
        return "\n".join(f.readlines())


MIN_PERCENT = 0.25
MAX_PERCENT = 0.75


text_files = [it for it in os.listdir(TEXT_PATH) if is_valid_ext(it)]

if len(text_files) == 0:
    exit()

summarizer = pipeline("summarization", model=MODEL_PATH)

for file_name in tqdm(text_files):
    file_path = os.path.join(TEXT_PATH, file_name)
    clean_file_name, _ = os.path.splitext(file_name)
    
    text = read_text_file(file_path)
    
    token_num = len(text.split())
    summary = summarizer(text, 
                        max_length=int(token_num * MAX_PERCENT), 
                        min_length=int(token_num * MIN_PERCENT), 
                        do_sample=False)

    summary = summary[0]["summary_text"]

    text_to_pdf(summary, os.path.join(REPORT_PATH, clean_file_name + ".pdf"))
