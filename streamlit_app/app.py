import streamlit as st
import requests

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

st.title("⚖️ AI Legal Contract Analyzer")

# File Upload
uploaded_file = st.file_uploader("Upload a legal document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    st.info(f"📄 Uploading file: {uploaded_file.name} ({uploaded_file.size} bytes)")

    # Detect MIME type
    if uploaded_file.type:
        mime_type = uploaded_file.type
    else:
        # fallback by extension
        if uploaded_file.name.endswith(".pdf"):
            mime_type = "application/pdf"
        elif uploaded_file.name.endswith(".docx"):
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            mime_type = "text/plain"

    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔄 Processing file...")
        progress_bar.progress(25)

        # Read file bytes
        file_bytes = uploaded_file.read()
        files = {"file": (uploaded_file.name, file_bytes, mime_type)}

        status_text.text("📤 Uploading to backend...")
        progress_bar.progress(50)

        response = requests.post(f"{BACKEND_URL}/parse", files=files, timeout=300)

        progress_bar.progress(75)

        if response.status_code == 200:
            result = response.json()
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")

            st.success(f"File '{uploaded_file.name}' parsed successfully!")
            st.info("📊 **Processing Results:**")
            st.write(f"- **Chunks created:** {result.get('chunks_stored', 'N/A')}")

        else:
            progress_bar.progress(0)
            status_text.text("❌ Processing failed")
            st.error(f"Failed to parse the file. Status: {response.status_code}")
            if response.text:
                st.error(f"Error details: {response.text}")

    except requests.exceptions.Timeout:
        progress_bar.progress(0)
        status_text.text("⏰ Request timed out")
        st.error("⏰ The file processing took too long.")

    except requests.exceptions.ConnectionError:
        progress_bar.progress(0)
        status_text.text("🔌 Connection error")
        st.error("🔌 Could not connect to the backend server.")

    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Unexpected error")
        st.error(f"❌ An unexpected error occurred: {str(e)}")


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
