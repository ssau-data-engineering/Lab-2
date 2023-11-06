import os
from tqdm.auto import tqdm
from faster_whisper import WhisperModel


SUPPORTED_EXT = [".wav"]

MODEL_PATH = "model"
AUDIO_PATH = "audios"
TEXT_PATH = "texts"


#FOR TESTING
#MODEL_PATH = "/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/model"
#AUDIO_PATH = "/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/inference_data/audios"
#TEXT_PATH = "/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/inference_data/texts"


def is_valid_ext(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext in SUPPORTED_EXT

def extract_text(audio_path: str, model: WhisperModel) -> str:
    segments, info = model.transcribe(audio_path)
    fragments = [segment.text for segment in segments]
    
    return ". ".join(fragments)

def save_text(text: str, path: str) -> None:
    with open(path, "w") as f:
        f.writelines(text)


audio_files = [it for it in os.listdir(AUDIO_PATH) if is_valid_ext(it)]

if len(audio_files) == 0:
    exit(0)

model = WhisperModel(MODEL_PATH)

for file_name in tqdm(audio_files):
    
    file_path = os.path.join(AUDIO_PATH, file_name)
    clean_file_name, _ = os.path.splitext(file_name)
    
    text = extract_text(file_path, model)

    save_text(text, os.path.join(TEXT_PATH, clean_file_name + ".txt"))
