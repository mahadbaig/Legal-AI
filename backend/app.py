# backend/app.py
import io
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq
import os

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
async def parse_file(file: UploadFile = File(...)):
    """
    Accepts PDF, DOCX, or TXT and extracts raw text.
    Stores globally for demo.
    """
    global parsed_text
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

        elif filename.endswith((".docx", ".doc")):
            doc = Document(io.BytesIO(contents))
            for p in doc.paragraphs:
                text += p.text + "\n"

        else:
            # assume text file
            text = contents.decode(errors="ignore")

        parsed_text = text  # Save for later querying

    except Exception as e:
        logger.exception("Parsing error")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {e}")

    # return preview
    return {"filename": file.filename, "text": text[:500]}

# Input model for query
class QueryRequest(BaseModel):
    query: str


# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@app.post("/query")
async def query_doc(req: QueryRequest):
    global parsed_text
    if not parsed_text:
        return {"answer": "❌ No document uploaded yet. Please upload a PDF first."}

    # Prepare prompt for LLM

    prompt = f"""
    You are an AI Legal Assistant specializing in contract analysis and legal reasoning. 
    Your role is to carefully review the document context and provide clear, accurate, and practical legal insights.

    ### Context:
    The following is an excerpt of a legal/contractual document (partial text for analysis, not full contract):

    {parsed_text[:2000]}

    ### Instructions:
    - ONLY use information present in the document context. 
    - If the question cannot be fully answered from the provided text, say so explicitly and suggest what additional sections would be required. 
    - Provide your answer in a structured format:
      1. **Direct Answer** – address the user’s query clearly.
      2. **Relevant Clauses or Evidence** – cite exact snippets from the text that support your answer.
      3. **Risks / Ambiguities** – highlight any legal risks, unclear language, or missing information.
      4. **Practical Implications** – explain what this means for the user in plain English.
    - Do NOT invent laws or clauses not present in the document.
    - Maintain a professional, precise, and objective legal tone.

    ### Question:
    {req.query}
    """

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="moonshotai/kimi-k2-instruct-0905"  # Groq’s free fast model
        )

        answer = response.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        return {"answer": f"⚠️ Error: {str(e)}"}


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
