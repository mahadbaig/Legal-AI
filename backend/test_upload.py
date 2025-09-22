# import requests
#
# url = "http://127.0.0.1:8000/parse"
# file_path = "C:/Mahad/Agentic AI Bootcamp/Capstone Project/Legal AI/Agreement_TilesSupply_20150726.pdf"
#
# with open(file_path, "rb") as f:
#     files = {"file": (file_path, f, "application/pdf")}
#     response = requests.post(url, files=files)
#
# print(response.status_code)
# print(response.json())


from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

resp = client.chat.completions.create(
    model="mixtral-8x7b-32768",  # known Groq model
    messages=[{"role": "user", "content": "Say hello from Groq"}]
)

print(resp.choices[0].message["content"])

import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

load_dotenv()


def test_sendgrid():
    message = Mail(
        from_email=os.getenv("FROM_EMAIL"),
        to_emails="your-test-email@gmail.com",
        subject="Test Email",
        plain_text_content="This is a test email."
    )

    try:
        sg = SendGridAPIClient(api_key=os.getenv("SENDGRID_API_KEY"))
        response = sg.send(message)
        print(f"Email sent! Status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_sendgrid()