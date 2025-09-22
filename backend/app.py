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

# HuggingFace fallback
from transformers import pipeline

ssl_context_ = ssl.create_default_context(cafile=certifi.where())

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


@app.get("/")
def health():
    return {"status": "ok"}


# Add a debug endpoint to check document status
@app.get("/document-status")
def get_document_status():
    global parsed_text, current_filename
    return {
        "has_document": bool(parsed_text),
        "filename": current_filename,
        "text_length": len(parsed_text) if parsed_text else 0,
        "text_preview": parsed_text[:200] if parsed_text else "No document loaded"
    }


# ---------------- Parse ----------------
@app.post("/parse")
async def parse_file(file: UploadFile = File(...)):
    global parsed_text, chat_history, current_filename

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

        return {
            "filename": file.filename,
            "text": text[:500],
            "success": True,
            "length": len(parsed_text)
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
    func=search_document,
    description="Search the uploaded legal document for relevant clauses, terms, or content. Use this to find specific information within the uploaded document."
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
- Always Include web search links in the final answer
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
        report_text = agent.run(query_with_instructions)

        message = Mail(
            from_email=os.getenv("FROM_EMAIL"),
            to_emails=req.email,
            subject="Your Legal Contract Analysis Report",
            plain_text_content=report_text
        )
        sg = SendGridAPIClient(api_key=os.getenv("SENDGRID_API_KEY"))
        sg.send(message)

        return {"status": "✅ Report sent successfully"}
    except Exception as e:
        logger.exception("Email error")
        return {"status": f"⚠️ Error sending email: {str(e)}"}


# Add endpoint to clear document (useful for testing)
@app.post("/clear-document")
async def clear_document():
    global parsed_text, chat_history, current_filename
    parsed_text = ""
    chat_history = []
    current_filename = ""
    return {"status": "Document cleared"}