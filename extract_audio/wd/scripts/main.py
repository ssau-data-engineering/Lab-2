import ffmpeg
import os
from tqdm.auto import tqdm


SUPPORTED_EXT = [".mp4"]
AUDIO_EXT = ".wav"
VIDEO_PATH = "videos"
AUDIO_PATH = "audios"


def extract_audio(src: str, dest: str) -> None:
    stream = ffmpeg.input(src)
    audio_stream = stream['a']
    ffmpeg.output(audio_stream, dest).run()

def is_valid_ext(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext in SUPPORTED_EXT


video_files = os.listdir(VIDEO_PATH)

for file_name in tqdm(filter(lambda x: is_valid_ext, video_files), total=len(video_files)):
    
    file_path = os.path.join(VIDEO_PATH, file_name)
    clean_file_name, _ = os.path.splitext(file_name)
    print(os.path.join(AUDIO_PATH, clean_file_name + AUDIO_EXT))
    extract_audio(file_path, os.path.join(AUDIO_PATH, clean_file_name + AUDIO_EXT))