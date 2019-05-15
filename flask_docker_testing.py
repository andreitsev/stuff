# Код для тесторования flask/docker приложений
import requests
import json
url = 'http://127.0.0.1:8118/eva'
# url = 'http://bigdata.doctoroncall.ru/eva'
r = requests.post(url, json={"question_text": text})
print(json.dumps(r.json(), ensure_ascii=False, indent=2))
