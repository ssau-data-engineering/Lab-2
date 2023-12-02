import torch
import torchaudio
from transformers import pipeline


def transcribe_audio(audio_path, asr_model):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    transcript = asr_model(waveform.numpy()[0], sample_rate=sample_rate)

    return transcript


if __name__ == "__main__":

    asr_model = pipeline('automatic-speech-recognition')

    audio_path = '/data/my_audio.aac'

    transcript = transcribe_audio(audio_path, asr_model)

    with open("/data/audio_text.txt", "w+") as out_file:
        out_file.write(transcript)
