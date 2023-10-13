from transformers import pipeline
import glob
import os

for audio_file_path in glob.glob('/data/lab2_output/*.wav'):
    _, filename = os.path.split(audio_file_path)
    output_txt_file_path = f'/data/lab2_output/text/{filename}.txt'
    if os.path.isfile(output_txt_file_path):
        continue # skip, already exist

    pipe = pipeline("automatic-speech-recognition", "openai/whisper-tiny")
    res = pipe(audio_file_path)

    with open(output_txt_file_path, "w") as text_file:
        text_file.write(res['text'])

