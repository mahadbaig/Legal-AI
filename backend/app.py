# backend/app.py
import io
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Legal Capstone - Backend")

# Allow frontend (Streamlit) to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

@app.get("/")
def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/parse")
async def parse_contract(file: UploadFile = File(...)):
    """
    Accepts PDF or DOCX, extracts raw text.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = file.filename.lower()
    try:
        contents = await file.read()
        text = ""
        if filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(contents))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif filename.endswith(".docx") or filename.endswith(".doc"):
            doc = Document(io.BytesIO(contents))
            for p in doc.paragraphs:
                text += p.text + "\n"
        else:
            text = contents.decode(errors="ignore")
    except Exception as e:
        logger.exception("Parsing error")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {e}")

    return {"filename": file.filename, "text": text[:20000]}

@app.post("/analyze")
async def analyze(payload: dict):
    """
    Stub analysis endpoint.
    Later: will use Groq + LangChain.
    """
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    logger.info("Running stub analysis")

    # Temporary keyword search
    keywords = ["termination", "liability", "non-compete", "confidential", "payment", "governing law"]
    clauses = []
    lower = text.lower()
    for kw in keywords:
        if kw in lower:
            idx = lower.find(kw)
            start = max(0, idx - 80)
            end = min(len(text), idx + 80)
            snippet = text[start:end].replace("\n", " ")
            clauses.append({
                "clause_keyword": kw,
                "risk_level": "medium",
                "explanation": f"Found keyword '{kw}' — review this clause.",
                "snippet": snippet
            })

    return {"clauses": clauses}
