import requests
url = "https://api-inference.huggingface.co/models/openai/whisper-small"
headers = {"Authorization": "Bearer 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"}

with open('/data/audio.aac', "rb") as file:
    data = file.read()
    response = requests.post(url, headers=headers, data=data)
    result = response.json()
    text_from_audio = result['text']
    text_file = open("/data/text.txt", "w+")
    text_file.write(text_from_audio)
    text_file.close()