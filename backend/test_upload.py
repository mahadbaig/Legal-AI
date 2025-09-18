import requests

url = "http://127.0.0.1:8000/parse"
file_path = "C:/Mahad/Agentic AI Bootcamp/Capstone Project/Legal AI/Agreement_TilesSupply_20150726.pdf"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "application/pdf")}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
