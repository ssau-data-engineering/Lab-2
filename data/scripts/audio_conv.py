import requests
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
API_TOKEN = 'hf_*********lefNpiI'
headers = {"Authorization": f"Bearer {API_TOKEN}" }

with open('data/output/audio.aac', "rb") as f:
    data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)

    if response.status_code == 200:
        try:
            result = response.json()
            text_file = open("data/output/text.txt", "w+")
            text_file.write(result.get('text', ''))
            text_file.close()
        except requests.exceptions.JSONDecodeError:
            print("Invalid JSON in the response")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)