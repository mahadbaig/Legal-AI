import streamlit as st
import requests

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

st.title("⚖️ AI Legal Contract Analyzer")

# File Upload
uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])

if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    response = requests.post(f"{BACKEND_URL}/parse", files=files)

    if response.status_code == 200:
        st.success(f"File '{uploaded_file.name}' parsed successfully!")
    else:
        st.error("Failed to parse the PDF.")

# Question Answering
st.subheader("Ask a Question")
user_query = st.text_input("Enter your question about the document")

if st.button("Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        response = requests.post(f"{BACKEND_URL}/query", json={"query": user_query})
        if response.status_code == 200:
            st.write("### Answer:")
            st.write(response.json().get("answer"))
        else:
            st.error("Error fetching answer from backend.")
