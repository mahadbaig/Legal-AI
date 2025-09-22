import io
import logging
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.openai import OpenAI
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers.langchain import LangChainTracer

# LangChain + Groq + Tools
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
import certifi
import ssl
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import urllib3

# HuggingFace fallback
from transformers import pipeline

# Vector Database imports
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import pickle
    import os

    VECTOR_DB_AVAILABLE = True
    print("✅ FAISS vector database available")
except ImportError as e:
    VECTOR_DB_AVAILABLE = False
    print(f"⚠️ FAISS not installed: {e}. Vector search will be disabled.")

# SSL context configuration for Windows/Python 3.13 compatibility
ssl_context_ = ssl.create_default_context(cafile=certifi.where())

# Configure SSL for SendGrid compatibility
import os

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Disable SSL warnings for development (remove in production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Legal Capstone - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# Globals - Initialize properly
parsed_text = ""
chat_history = []
current_filename = ""
# FAISS vector store components
faiss_index = None
document_chunks = []
embeddings_model = None


@app.get("/")
def health():
    return {"status": "ok"}


# Add a debug endpoint to check document status
@app.get("/document-status")
def get_document_status():
    global parsed_text, current_filename, faiss_index
    return {
        "has_document": bool(parsed_text),
        "filename": current_filename,
        "text_length": len(parsed_text) if parsed_text else 0,
        "text_preview": parsed_text[:200] if parsed_text else "No document loaded",
        "vector_db_available": VECTOR_DB_AVAILABLE,
        "vector_store_ready": faiss_index is not None,
        "chunks_count": len(document_chunks) if document_chunks else 0
    }


# FAISS Vector Database Setup Function
def setup_faiss_vector_store(text_content: str, filename: str):
    """Create and populate FAISS vector store"""
    global faiss_index, document_chunks, embeddings_model

    if not VECTOR_DB_AVAILABLE:
        logger.warning("FAISS not available, skipping vector store setup")
        return False

    try:
        # Initialize embeddings model (lightweight sentence transformer)
        if embeddings_model is None:
            embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embeddings model loaded")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
        )

        # Create chunks from the document
        chunks = text_splitter.split_text(text_content)
        logger.info(f"Split document into {len(chunks)} chunks")

        if not chunks:
            return False

        # Create embeddings for all chunks
        logger.info("Creating embeddings...")
        chunk_embeddings = embeddings_model.encode(chunks, show_progress_bar=True)

        # Create FAISS index
        dimension = chunk_embeddings.shape[1]  # Usually 384 for all-MiniLM-L6-v2
        faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(chunk_embeddings)

        # Add embeddings to index
        faiss_index.add(chunk_embeddings.astype('float32'))

        # Store chunks for retrieval
        document_chunks = [
            {
                "text": chunk,
                "source": filename,
                "chunk_id": i
            }
            for i, chunk in enumerate(chunks)
        ]

        # Save to disk (optional)
        save_path = "./faiss_store"
        os.makedirs(save_path, exist_ok=True)

        faiss.write_index(faiss_index, f"{save_path}/index.faiss")
        with open(f"{save_path}/chunks.pkl", "wb") as f:
            pickle.dump(document_chunks, f)

        logger.info(f"FAISS vector store created successfully with {len(chunks)} chunks, dimension {dimension}")
        return True

    except Exception as e:
        logger.error(f"Error setting up FAISS vector store: {e}")
        faiss_index = None
        document_chunks = []
        return False


# Enhanced search function using FAISS
def enhanced_search_document(query: str) -> str:
    """Search document using both traditional text search and FAISS vector similarity"""
    global parsed_text, faiss_index, document_chunks, embeddings_model

    logger.info(
        f"Enhanced search. Text length: {len(parsed_text)}, FAISS ready: {faiss_index is not None}, Query: {query}")

    if not parsed_text:
        return "No document has been uploaded yet. Please upload a legal document first."

    if not query.strip():
        return "Please provide a search query."

    results = []

    # 1. Traditional keyword search (existing logic)
    traditional_results = []
    lines = parsed_text.split("\n")
    query_words = query.lower().split()

    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue

        if any(word in line_clean.lower() for word in query_words):
            context_start = max(0, i - 1)
            context_end = min(len(lines), i + 2)
            context = " ".join(lines[context_start:context_end]).strip()
            traditional_results.append(context)

    # 2. FAISS vector similarity search (if available)
    vector_results = []
    if faiss_index is not None and embeddings_model is not None and document_chunks:
        try:
            # Create embedding for query
            query_embedding = embeddings_model.encode([query])
            faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity

            # Search for similar chunks
            k = min(3, len(document_chunks))  # Top 3 or less if fewer chunks exist
            similarities, indices = faiss_index.search(query_embedding.astype('float32'), k)

            # Get results with similarity scores
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1:  # Valid index
                    chunk_data = document_chunks[idx]
                    vector_results.append({
                        "text": chunk_data["text"],
                        "similarity": float(similarity),
                        "chunk_id": chunk_data["chunk_id"]
                    })

            logger.info(f"FAISS search found {len(vector_results)} results")
        except Exception as e:
            logger.error(f"FAISS search error: {e}")

    # Combine results
    if vector_results:
        results.append("🚀 **Most Relevant Sections (AI Similarity Search):**")
        for i, result in enumerate(vector_results, 1):
            similarity_percent = int(result["similarity"] * 100)
            results.append(f"{i}. [{similarity_percent}% match] {result['text'].strip()}")

    if traditional_results:
        results.append("\n📝 **Keyword Matches:**")
        unique_traditional = list(set(traditional_results))[:3]  # Remove duplicates, limit to 3
        for i, result in enumerate(unique_traditional, 1):
            results.append(f"{i}. {result.strip()}")

    if not results:
        # Fallback to broader search
        broader_results = []
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) > 20:
                broader_results.append(line_clean)

        if broader_results:
            return f"No direct matches found for '{query}'. Here are some key sections from the document:\n\n" + "\n\n".join(
                broader_results[:3])
        else:
            return f"No relevant content found for '{query}' in the uploaded document."

    return "\n\n".join(results)


# ---------------- Parse ----------------
@app.post("/parse")
async def parse_file(file: UploadFile = File(...)):
    global parsed_text, chat_history, current_filename, vector_store

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        contents = await file.read()
        text = ""

        if file.filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(contents))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        elif file.filename.endswith((".docx", ".doc")):
            doc = Document(io.BytesIO(contents))
            for p in doc.paragraphs:
                text += p.text + "\n"

        else:
            # For TXT and other text files
            text = contents.decode('utf-8', errors="ignore")

        # Store the parsed text globally
        parsed_text = text.strip()
        current_filename = file.filename

        # Clear chat history when new document is uploaded
        chat_history = []

        # Log for debugging
        logger.info(f"Document parsed: {file.filename}, length: {len(parsed_text)}")

        if not parsed_text:
            raise HTTPException(status_code=400, detail="Document appears to be empty or could not be parsed")

        # Setup vector store in background (don't block the response)
        vector_setup_success = False
        if VECTOR_DB_AVAILABLE and len(parsed_text) > 100:  # Only if document is substantial
            try:
                vector_setup_success = setup_faiss_vector_store(parsed_text, file.filename)
            except Exception as e:
                logger.warning(f"FAISS setup failed (non-blocking): {e}")

        return {
            "filename": file.filename,
            "text": text[:500],
            "success": True,
            "length": len(parsed_text),
            "vector_db_enabled": vector_setup_success
        }

    except Exception as e:
        logger.exception("Parsing error")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT"))
callback_manager = CallbackManager([tracer])


# ---------------- Query ----------------
class QueryRequest(BaseModel):
    query: str


llm = ChatGroq(
    model=os.getenv("MODEL_NAME", "mixtral-8x7b-32768"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    callbacks=callback_manager,
)

# Web Search Tool
tavily_client = TavilySearchResults(max_results=3, api_key=os.getenv("TAVILY_API_KEY"))
web_search_tool = Tool(
    name="WebSearch",
    func=tavily_client.run,
    description="Search the web for external legal references, case law, statutes, or resources."
)


# Fixed Search inside doc function
def search_document(query: str) -> str:
    global parsed_text

    # Debug logging
    logger.info(f"Searching document. Text length: {len(parsed_text)}, Query: {query}")

    if not parsed_text:
        return "No document has been uploaded yet. Please upload a legal document first."

    if not query.strip():
        return "Please provide a search query."

    # Search for relevant content
    results = []
    lines = parsed_text.split("\n")

    # Search for lines containing the query terms
    query_words = query.lower().split()

    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue

        # Check if any query words are in this line
        if any(word in line_clean.lower() for word in query_words):
            # Include some context (previous and next lines)
            context_start = max(0, i - 1)
            context_end = min(len(lines), i + 2)
            context = " ".join(lines[context_start:context_end]).strip()
            results.append(context)

    if not results:
        # If no direct matches, try a broader search
        broader_results = []
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) > 20:  # Only consider substantial lines
                broader_results.append(line_clean)

        if broader_results:
            return f"No direct matches found for '{query}'. Here are some key sections from the document:\n\n" + "\n\n".join(
                broader_results[:3])
        else:
            return f"No relevant content found for '{query}' in the uploaded document."

    # Return top results with clear separation
    unique_results = list(set(results))[:5]  # Remove duplicates and limit
    return "\n\n--- Relevant sections ---\n" + "\n\n".join(unique_results)


search_tool = Tool(
    name="SearchLegalText",
    func=enhanced_search_document,  # Use the enhanced search function
    description="Search the uploaded legal document for relevant clauses, terms, or content using both keyword matching and AI-powered semantic similarity search."
)

# Agent
agent = initialize_agent(
    tools=[search_tool, web_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    handle_unknown_errors=True,
    callbacks=callback_manager,
)

STRUCTURED_PROMPT = """
You are an AI Legal Assistant with expertise in analyzing contracts, agreements, and other legal documents. 
Your task is to provide a clear, structured, and practical legal analysis.

Instructions:
1. **Direct Answer** – Provide a precise and plain-language response to the user's question. Avoid unnecessary legal jargon unless essential.  
2. **Relevant Clauses** – Quote or summarize the specific clauses, sections, or provisions from the provided context that support your answer.  
3. **Risks & Ambiguities** – Identify any unclear wording, conflicting terms, missing details, or potential risks the user should be aware of.  
4. **Practical Implications** – Explain how this affects the user in real-world terms (e.g., rights, obligations, liabilities, financial impact, timelines).  
5. **Additional Notes (if applicable)** – Suggest follow-up actions, clarifications to seek, or common legal practices related to this scenario.  
6. **References** - list of websites the web search tool visited

Important:
- Stay neutral and objective.  
- Include web search links in the final answer
- Keep the tone and language simple
- Don't use too much legal jargon, a normal person should be able to understand
- You are talking to a regular person, not a lawyer
- Do not provide speculative advice beyond the provided context.  
- If the context does not contain enough information, state what is missing and suggest what additional details would be required for a full analysis.  

IMPORTANT: 
Always use the following response structure:
Thought: <your reasoning>
Action: <tool name, if any>
Action Input: <input>
OR
Final Answer: <your answer to the user>
"""

reasoning_summary = """
I searched the uploaded document for relevant clauses and also cross-checked with online legal resources. 
Here's what I found:
- Document insights were retrieved with `SearchLegalText`.
- External references were gathered with `WebSearch`.
"""


@app.post("/query")
async def query_doc(req: QueryRequest):
    global parsed_text, chat_history

    # Debug check
    logger.info(f"Query received. Document available: {bool(parsed_text)}, Length: {len(parsed_text)}")

    if not parsed_text:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a document first.")

    try:
        query_with_instructions = f"{STRUCTURED_PROMPT}\n\nUser Question: {req.query}\n\nContext (extracted from uploaded document):\n{parsed_text[:2000]}"
        answer = agent.run(query_with_instructions)
        chat_history.append({"user": req.query, "ai": answer})
        return {"answer": f"{reasoning_summary}\n\n{answer}"}
    except Exception as e:
        logger.exception(f"Agent failed: {e}")
        # Fallback without agent
        try:
            fallback_prompt = f"Based on this legal document:\n\n{parsed_text[:2000]}\n\nUser question: {req.query}\n\nProvide a helpful legal analysis in plain language."
            fallback = llm.predict(fallback_prompt)
            return {"answer": f"⚠️ Used fallback mode due to agent error.\n\n{fallback}"}
        except Exception as fallback_error:
            logger.exception(f"Fallback also failed: {fallback_error}")
            return {
                "answer": "❌ Sorry, I encountered an error processing your request. Please try again or upload a different document."}


# ---------------- Email Report ----------------
class EmailRequest(BaseModel):
    email: str


@app.post("/email-report")
async def email_report(req: EmailRequest):
    global parsed_text, chat_history
    if not parsed_text:
        raise HTTPException(status_code=400, detail="No document uploaded")

    try:
        chat_summary = "\n".join([f"User: {c['user']}\nAI: {c['ai']}" for c in chat_history])
        doc_preview = parsed_text[:4000] if parsed_text else "No document uploaded yet."
        query_with_instructions = f"""
        {STRUCTURED_PROMPT}

        Task: Create a comprehensive **legal report email**.
        Include:
        - Summary of document obligations, risks, and clauses
        - Sketchy or harmful terms
        - Practical implications
        - Citations from doc (via SearchLegalText)
        - 2–3 external references (via WebSearch)
        - Summary of chat history insights

        Document:
        {doc_preview}

        Chat History:
        {chat_summary}
        """

        logger.info("Generating email report...")
        report_text = agent.run(query_with_instructions)

        # Create SendGrid message
        message = Mail(
            from_email=os.getenv("FROM_EMAIL"),
            to_emails=req.email,
            subject="Your Legal Contract Analysis Report",
            plain_text_content=report_text
        )

        # Initialize SendGrid client with SSL context
        sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        if not sendgrid_api_key:
            raise ValueError("SENDGRID_API_KEY environment variable not set")

        logger.info(f"Sending email to: {req.email}")

        try:
            # Try with SSL context first
            sg = SendGridAPIClient(api_key=sendgrid_api_key)
            response = sg.send(message)
            logger.info(f"Email sent successfully. Status: {response.status_code}")
            return {"status": "✅ Report sent successfully"}

        except Exception as ssl_error:
            logger.warning(f"SSL error with SendGrid: {ssl_error}")

            # Fallback: Try with modified SSL settings
            try:
                import ssl
                import urllib3

                # Create unverified SSL context as fallback
                ssl._create_default_https_context = ssl._create_unverified_context

                sg = SendGridAPIClient(api_key=sendgrid_api_key)
                response = sg.send(message)
                logger.info(f"Email sent with fallback SSL. Status: {response.status_code}")
                return {"status": "✅ Report sent successfully (with SSL fallback)"}

            except Exception as fallback_error:
                logger.error(f"Both SSL methods failed: {fallback_error}")
                # Try alternative email method or return detailed error
                return await send_email_alternative(req.email, report_text)

    except Exception as e:
        logger.exception("Email generation/sending error")
        return {"status": f"⚠️ Error sending email: {str(e)}"}


# Alternative email sending method using smtplib
async def send_email_alternative(email: str, report_text: str):
    """Alternative email sending using SMTP (Gmail/Outlook)"""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # Gmail SMTP configuration (you can modify for other providers)
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        sender_email = os.getenv("SENDER_EMAIL")  # Your Gmail address
        sender_password = os.getenv("SENDER_APP_PASSWORD")  # App-specific password

        if not sender_email or not sender_password:
            return {"status": "⚠️ SMTP credentials not configured. Please set SENDER_EMAIL and SENDER_APP_PASSWORD"}

        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = "Your Legal Contract Analysis Report"

        # Attach report
        msg.attach(MIMEText(report_text, 'plain'))

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, email, text)

        logger.info(f"Email sent via SMTP to: {email}")
        return {"status": "✅ Report sent successfully via SMTP"}

    except Exception as smtp_error:
        logger.error(f"SMTP fallback also failed: {smtp_error}")
        return {
            "status": f"⚠️ All email methods failed. Last error: {str(smtp_error)}. Please check your email configuration."}


# Add endpoint to clear document (useful for testing)
@app.post("/clear-document")
async def clear_document():
    global parsed_text, chat_history, current_filename, faiss_index, document_chunks, embeddings_model
    parsed_text = ""
    chat_history = []
    current_filename = ""
    faiss_index = None
    document_chunks = []
    # Keep embeddings_model loaded for performance

    # Clear FAISS files if available
    if VECTOR_DB_AVAILABLE:
        try:
            import shutil
            if os.path.exists("./faiss_store"):
                shutil.rmtree("./faiss_store")
                logger.info("FAISS store directory cleared")
        except Exception as e:
            logger.warning(f"Could not clear FAISS store: {e}")

    return {"status": "Document and FAISS vector store cleared"}


# Add endpoint to test vector search directly
@app.post("/vector-search")
async def vector_search(req: QueryRequest):
    """Direct FAISS vector search endpoint for testing"""
    global faiss_index, document_chunks, embeddings_model

    if not faiss_index or not embeddings_model:
        return {"error": "FAISS vector store not available"}

    try:
        # Create embedding for query
        query_embedding = embeddings_model.encode([req.query])
        faiss.normalize_L2(query_embedding)

        # Search for similar chunks
        k = min(5, len(document_chunks))
        similarities, indices = faiss_index.search(query_embedding.astype('float32'), k)

        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx != -1:
                chunk_data = document_chunks[idx]
                results.append({
                    "chunk_id": chunk_data["chunk_id"],
                    "content": chunk_data["text"][:200] + "..." if len(chunk_data["text"]) > 200 else chunk_data[
                        "text"],
                    "similarity": float(similarity),
                    "similarity_percent": int(similarity * 100)
                })

        return {"results": results, "count": len(results)}
    except Exception as e:
        return {"error": str(e)}