# backend/app.py
import io
import logging
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Chroma client & embeddings
chroma_client = chromadb.PersistentClient(path="C:/Mahad/Agentic AI Bootcamp/Capstone Project/Legal AI/Chroma DB")
logger.info("Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
logger.info("Embedding model loaded successfully")

# Create / get collection
collection = chroma_client.get_or_create_collection(name="legal_docs")

app = FastAPI(title="AI Legal Capstone - Backend")

# Allow frontend (Streamlit) to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/parse")
async def parse_file(file: UploadFile = File(...)):
    """
    Accepts PDF, DOCX, or TXT and extracts raw text.
    Splits into chunks, embeds, and stores in Chroma.
    """
    logger.info(f"Starting to parse file: {file.filename}")
    
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = file.filename.lower()
    try:
        # Read file contents
        logger.info("Reading file contents...")
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        text = ""

        if filename.endswith(".pdf"):
            logger.info("Processing PDF file...")
            reader = PdfReader(io.BytesIO(contents))
            logger.info(f"PDF has {len(reader.pages)} pages")
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                logger.info(f"Processed page {i+1}/{len(reader.pages)}")

        elif filename.endswith((".docx", ".doc")):
            logger.info("Processing DOCX file...")
            doc = Document(io.BytesIO(contents))
            for p in doc.paragraphs:
                text += p.text + "\n"

        else:
            # assume text file
            logger.info("Processing text file...")
            text = contents.decode(errors="ignore")

        logger.info(f"Extracted text length: {len(text)} characters")

        # Clear previous docs from collection (fresh start per upload)
        logger.info("Clearing previous documents...")
        existing_ids = collection.get()["ids"]
        if existing_ids and len(existing_ids) > 0:
            collection.delete(ids=existing_ids)

        # Split into chunks for embedding
        logger.info("Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks")

        # Embed and store in Chroma
        logger.info("Generating embeddings...")
        embeddings = embedding_model.embed_documents(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        logger.info("Storing chunks in database...")
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                embeddings=[embeddings[i]],
                ids=[f"{file.filename}_{i}"]
            )
            if (i + 1) % 10 == 0:  # Log every 10 chunks
                logger.info(f"Stored {i+1}/{len(chunks)} chunks")

        logger.info("File processing completed successfully")

    except Exception as e:
        logger.exception(f"Parsing error for file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

    # return preview
    return {"filename": file.filename, "chunks_stored": len(chunks), "text_length": len(text)}

# Input model for query
class QueryRequest(BaseModel):
    query: str

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.post("/query")
async def query_doc(req: QueryRequest):
    if collection.count() == 0:
        return {"answer": "❌ No document uploaded yet. Please upload a file first."}

    try:
        # Embed the query
        query_embedding = embedding_model.embed_query(req.query)

        # Retrieve top 3 relevant chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        context = " ".join(results["documents"][0])

        prompt = f"""
        You are an AI Legal Assistant specializing in contract analysis and legal reasoning. 
        Use the following document context to answer the question.

        ### Context:
        {context}

        ### Question:
        {req.query}

        ### Instructions:
        - ONLY use information present in the document context. 
        - If the question cannot be fully answered, say so explicitly.
        - Provide your answer in this format:
          1. **Direct Answer**
          2. **Relevant Clauses**
          3. **Risks / Ambiguities**
          4. **Practical Implications**
        """

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="moonshotai/kimi-k2-instruct-0905"  # adjust if needed
        )

        answer = response.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        logger.exception("Query error")
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
