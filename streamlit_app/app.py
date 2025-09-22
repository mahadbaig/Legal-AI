import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:9000"

st.title("⚖️ AI Legal Contract Analyzer")

# ---------------- Session States ----------------
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Upload Mode ----------------
if not st.session_state.doc_uploaded:
    uploaded_file = st.file_uploader("Upload a legal document (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        response = requests.post(f"{BACKEND_URL}/parse", files=files)

        if response.status_code == 200:
            st.success("✅ Document uploaded & parsed")
            st.session_state.doc_uploaded = True
        else:
            st.error("❌ Upload failed")

# ---------------- After Upload: Show Start Chat ----------------
if st.session_state.doc_uploaded and not st.session_state.chat_mode:
    if st.button("💬 Start Chat"):
        st.session_state.chat_mode = True
        st.session_state.messages = []  # reset chat history

        # Call backend to auto-generate summary
        summary_prompt = "Please provide a concise but detailed legal summary of the uploaded document."
        res = requests.post(f"{BACKEND_URL}/query", json={"query": summary_prompt})
        if res.status_code == 200:
            summary_text = res.json().get("answer", "⚠️ Failed to summarize document.")
        else:
            summary_text = "⚠️ Error fetching summary."

        # Add AI summary as first message
        st.session_state.messages.append({
            "user": None,
            "ai": summary_text
        })

        st.rerun()

# ---------------- Chat Mode ----------------
if st.session_state.chat_mode:
    # Top Bar
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        user_email = st.text_input("📧 Enter your email for report")
    with col2:
        if st.button("📨 Send Report"):
            if user_email.strip():
                res = requests.post(f"{BACKEND_URL}/email-report", json={"email": user_email})
                if res.status_code == 200:
                    st.success(res.json().get("status"))
                else:
                    st.error("❌ Failed to send report")
    with col3:
        if st.button("📤 Upload New Document"):
            st.session_state.doc_uploaded = False
            st.session_state.chat_mode = False
            st.session_state.messages = []
            st.rerun()

    st.subheader("💬 Chat with your contract")

    for msg in st.session_state.messages:
        if msg["user"]:
            with st.chat_message("user"):
                st.write(msg["user"])
        with st.chat_message("assistant"):
            st.write(msg["ai"])

    if user_input := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"user": user_input, "ai": "..."})
        response = requests.post(f"{BACKEND_URL}/query", json={"query": user_input})
        if response.status_code == 200:
            ai_msg = response.json()["answer"]
        else:
            ai_msg = "⚠️ Error fetching response"
        st.session_state.messages[-1]["ai"] = ai_msg
        st.rerun()
