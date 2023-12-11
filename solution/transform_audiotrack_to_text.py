import requests
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
API_TOKEN = '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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}

with open('/data/audio.aac', "rb") as f:
    data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    result = response.json()
    text_from_audio = result['text']
    text_file = open("/data/text.txt", "w+")
    text_file.write(text_from_audio)
    text_file.close()