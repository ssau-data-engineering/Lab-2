import requests
url = "https://api-inference.huggingface.co/models/slauw87/bart_summarisation"
headers = {"Authorization": "Bearer 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"}

with open('/data/text.txt', "rb") as file:
    data = file.read()
    response = requests.post(url, headers=headers, json={'inputs': f"{data}",})
    result = response.json()
    summary = result[0]['summary_text']
    text_file = open("/data/summary.txt", "w+")
    text_file.write(summary)
    text_file.close() 