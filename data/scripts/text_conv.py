import requests
API_URL = "https://api-inference.huggingface.co/models/slauw87/bart_summarisation"
API_TOKEN = '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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}

with open('/data/output/text.txt', "rb") as f:
    data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)

    if response.status_code == 200:
        try:
            result = response.json()
            text_file = open("/data/output/summ.txt", "w+")
            text_file.write(result[0]['summary_text'])
            text_file.close()  

        except requests.exceptions.JSONDecodeError:
            print("Invalid JSON in the response")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)