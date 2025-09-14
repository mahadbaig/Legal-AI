# streamlit_app/app.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Legal Contract Analyzer", layout="wide")
st.title("📑 AI Legal Contract Analyzer")

uploaded = st.file_uploader("Upload a contract (PDF or DOCX)", type=["pdf", "docx", "doc"])

if uploaded:
    with st.spinner("Uploading and parsing..."):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        resp = requests.post(f"{BACKEND_URL}/parse", files=files, timeout=60)

        if resp.status_code != 200:
            st.error(f"Parsing failed: {resp.text}")
        else:
            data = resp.json()
            st.subheader("Parsed Text (Preview)")
            st.text_area("Contract text", data.get("text", "")[:2000], height=300)

            if st.button("Analyze Contract"):
                with st.spinner("Analyzing..."):
                    analyze_resp = requests.post(
                        f"{BACKEND_URL}/analyze",
                        json={"text": data.get("text", "")},
                        timeout=120
                    )
                    if analyze_resp.status_code != 200:
                        st.error(f"Analysis failed: {analyze_resp.text}")
                    else:
                        result = analyze_resp.json()
                        clauses = result.get("clauses", [])

                        if not clauses:
                            st.success("✅ No risky clauses detected (stub analysis).")
                        else:
                            st.subheader("⚠️ Detected Clauses")
                            for c in clauses:
                                st.markdown(
                                    f"""
                                    **Keyword:** {c['clause_keyword']}  
                                    **Risk:** {c['risk_level']}  
                                    **Explanation:** {c['explanation']}  
                                    """
                                )
                                st.info(c["snippet"])
