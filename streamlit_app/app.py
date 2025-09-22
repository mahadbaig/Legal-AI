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
if "filename" not in st.session_state:
    st.session_state.filename = ""


# Debug function to check backend status
def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/document-status")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


# ---------------- Upload Mode ----------------
if not st.session_state.doc_uploaded:
    st.write("Upload a legal document to get started with AI-powered analysis.")

    uploaded_file = st.file_uploader(
        "Upload a legal document (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            try:
                # Prepare file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{BACKEND_URL}/parse", files=files)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success", False):
                        st.success(f"✅ Document '{result['filename']}' uploaded & parsed successfully!")
                        st.success(f"📄 Processed {result.get('length', 0)} characters")

                        # Show vector database status
                        if result.get("vector_db_enabled", False):
                            st.success("🚀 AI-powered semantic search enabled!")
                        else:
                            st.info("📝 Using keyword-based search (ChromaDB not available)")

                        # Update session state
                        st.session_state.doc_uploaded = True
                        st.session_state.filename = result['filename']

                        # Show preview
                        if result.get('text'):
                            st.write("**Document Preview:**")
                            st.text_area("First 500 characters:", result['text'], height=150)
                    else:
                        st.error("❌ Document parsing failed - document may be empty")
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    st.error(f"❌ Upload failed: {error_detail}")

            except Exception as e:
                st.error(f"❌ Upload error: {str(e)}")

# ---------------- After Upload: Show Start Chat ----------------
if st.session_state.doc_uploaded and not st.session_state.chat_mode:
    st.write(f"📄 **Loaded:** {st.session_state.filename}")

    # Check backend status
    status = check_backend_status()
    if status:
        if status["has_document"]:
            st.success(f"✅ Backend has document: {status['filename']} ({status['text_length']} chars)")

            # Show vector database status
            if status.get("vector_store_ready", False):
                st.success("🚀 AI semantic search ready")
            elif status.get("vector_db_available", False):
                st.info("⏳ Vector database available but not loaded")
            else:
                st.info("📝 Using keyword search only")
        else:
            st.error("❌ Backend doesn't have the document - please re-upload")
            if st.button("🔄 Reset and Try Again"):
                st.session_state.doc_uploaded = False
                st.session_state.filename = ""
                st.rerun()

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("💬 Start Chatting", use_container_width=True):
            st.session_state.chat_mode = True

            # Auto-generate summary
            with st.spinner("Generating document summary..."):
                summary_prompt = "Please provide a concise but detailed legal summary of the uploaded document, highlighting key terms, obligations, and potential risks."
                try:
                    res = requests.post(f"{BACKEND_URL}/query", json={"query": summary_prompt})
                    if res.status_code == 200:
                        summary_text = res.json().get("answer", "⚠️ Failed to summarize document.")
                    else:
                        error_detail = res.json().get('detail', 'Unknown error')
                        summary_text = f"⚠️ Error fetching summary: {error_detail}"
                except Exception as e:
                    summary_text = f"⚠️ Error connecting to backend: {str(e)}"

                # Add AI summary as first message
                st.session_state.messages.append({
                    "user": None,
                    "ai": summary_text
                })

            st.rerun()

    with col2:
        if st.button("📤 Upload Different Document", use_container_width=True):
            st.session_state.doc_uploaded = False
            st.session_state.chat_mode = False
            st.session_state.messages = []
            st.session_state.filename = ""
            st.rerun()

# ---------------- Chat Mode ----------------
if st.session_state.chat_mode:
    # Top Bar
    st.write(f"💬 **Chatting with:** {st.session_state.filename}")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user_email = st.text_input("📧 Enter your email for detailed report")
    with col2:
        if st.button("📨 Send Report"):
            if user_email.strip():
                with st.spinner("Generating and sending report..."):
                    try:
                        res = requests.post(f"{BACKEND_URL}/email-report", json={"email": user_email})
                        if res.status_code == 200:
                            st.success(res.json().get("status"))
                        else:
                            error_detail = res.json().get('detail', 'Unknown error')
                            st.error(f"❌ Failed to send report: {error_detail}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.error("Please enter a valid email address")
    with col3:
        if st.button("📤 New Document"):
            st.session_state.doc_uploaded = False
            st.session_state.chat_mode = False
            st.session_state.messages = []
            st.session_state.filename = ""
            st.rerun()

    st.divider()

    # Chat Messages
    for msg in st.session_state.messages:
        if msg["user"]:  # User message
            with st.chat_message("user"):
                st.write(msg["user"])

        # AI message
        with st.chat_message("assistant"):
            st.write(msg["ai"])

    # Chat Input
    if user_input := st.chat_input("Ask a question about the document..."):
        # Add user message to chat
        st.session_state.messages.append({"user": user_input, "ai": "⏳ Analyzing..."})

        # Re-run to show the user message immediately
        st.rerun()

# Handle the API call outside of the chat_input context
if (st.session_state.chat_mode and
        st.session_state.messages and
        st.session_state.messages[-1]["ai"] == "⏳ Analyzing..."):

    user_query = st.session_state.messages[-1]["user"]

    with st.spinner("Getting AI response..."):
        try:
            response = requests.post(f"{BACKEND_URL}/query", json={"query": user_query})
            if response.status_code == 200:
                ai_msg = response.json()["answer"]
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                ai_msg = f"⚠️ Error: {error_detail}"
        except Exception as e:
            ai_msg = f"⚠️ Connection error: {str(e)}"

    # Update the last message with the AI response
    st.session_state.messages[-1]["ai"] = ai_msg
    st.rerun()

# Sidebar with debug info (optional - remove in production)
with st.sidebar:
    st.subheader("🔧 Debug Info")
    if st.button("Check Backend Status"):
        status = check_backend_status()
        if status:
            st.json(status)
        else:
            st.error("Cannot connect to backend")

    # Vector search test
    if st.session_state.doc_uploaded:
        st.subheader("🔍 Test Vector Search")
        test_query = st.text_input("Test query:")
        if test_query and st.button("Search"):
            try:
                response = requests.post(f"{BACKEND_URL}/vector-search", json={"query": test_query})
                if response.status_code == 200:
                    results = response.json()
                    st.write("**Vector Search Results:**")
                    st.json(results)
                else:
                    st.error("Vector search failed")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("Clear All Data"):
        # Also clear backend data
        try:
            requests.post(f"{BACKEND_URL}/clear-document")
        except:
            pass

        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()