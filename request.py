import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'movie':"The Kids Are All Right"})

print(r.json())