from transformers import pipeline


def generate_summary(text, summarization_model):
    summary = summarization_model(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text']
    return summary

if __name__ == "__main__":

    summarization_model = pipeline('summarization', model='t5-base', device=0)

    audio_path = '/data/my_audio.aac'

    with open('/data/audio_text.txt') as f:
        transcript = f.read()

    summary = generate_summary(transcript, summarization_model)

    with open("/data/audio_summary.txt", "w+") as out_file:
        out_file.write(summary)
