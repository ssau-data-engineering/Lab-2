import requests
API_URL = "https://api-inference.huggingface.co/models/slauw87/bart_summarisation"
API_TOKEN = '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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}

with open('/data/text.txt', "rb") as f:
    data = f.read()
    response = requests.post(API_URL, headers=headers, json={'inputs': f"{data}",})
    result = response.json()
    summary_text = result[0]['summary_text']
    text_file = open("/data/summary.txt", "w+")
    text_file.write(summary_text)
    text_file.close() 