# backend/app.py
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

# LangChain + Groq + Tools
from langchain.agents import initialize_agent, Tool, AgentType
# from langchain_groq import ChatGroq
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

# Globals
parsed_text = ""
chat_history = []  # store user + AI messages

@app.get("/")
def health():
    return {"status": "ok"}

# ---------------- Parse ----------------
@app.post("/parse")
async def parse_file(file: UploadFile = File(...)):
    global parsed_text, chat_history
    chat_history = []  # reset chat
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
            text = contents.decode(errors="ignore")

        parsed_text = text

    except Exception as e:
        logger.exception("Parsing error")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {e}")

    # return {"text": text}
    return {"filename": file.filename, "text": text[:500]}

# ---------------- Query ----------------
class QueryRequest(BaseModel):
    query: str

llm = ChatGroq(
    model=os.getenv("MODEL_NAME", "mixtral-8x7b-32768"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Web Search Tool
tavily_client = TavilySearchResults(max_results=3, api_key=os.getenv("TAVILY_API_KEY"))
web_search_tool = Tool(
    name="WebSearch",
    func=tavily_client.run,
    description="Search the web for external legal references, case law, statutes, or resources."
)

# Search inside doc
def search_document(query: str) -> str:
    global parsed_text
    if not parsed_text:
        return "❌ No document uploaded yet."
    results = []
    for line in parsed_text.split("\n"):
        if query.lower() in line.lower():
            results.append(line.strip())
    if not results:
        return "No directly relevant clause found."
    return "\n".join(results[:5])

search_tool = Tool(
    name="SearchLegalText",
    func=search_document,
    description="Search the uploaded legal document for relevant clauses."
)

# Agent
agent = initialize_agent(
    tools=[search_tool, web_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    handle_unknown_errors=True,
)

STRUCTURED_PROMPT = f"""
        You are an AI Legal Assistant with expertise in analyzing contracts, agreements, and other legal documents. 
        Your task is to provide a clear, structured, and practical legal analysis.

        Instructions:
        1. **Direct Answer** – Provide a precise and plain-language response to the user’s question. Avoid unnecessary legal jargon unless essential.  
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

reasoning_summary = f"""
I searched the uploaded document for relevant clauses and also cross-checked with online legal resources. 
Here’s what I found:
- Document insights were retrieved with `SearchLegalText`.
- External references were gathered with `WebSearch`.
"""

@app.post("/query")
async def query_doc(req: QueryRequest):
    global parsed_text, chat_history
    if not parsed_text:
        return {"answer": "❌ No document uploaded yet."}

    try:
        query_with_instructions = f"{STRUCTURED_PROMPT}\n\nUser Question: {req.query}\n\n Context (extracted from uploaded document):{(parsed_text[:2000])}"
        answer = agent.invoke({"input": query_with_instructions})
        chat_history.append({"user": req.query, "ai": answer})
        return {"answer": f"{reasoning_summary}\n\n{answer}"}
    except Exception as e:
        logger.warning(f"Agent failed, falling back: {e}")
        fallback = llm.predict(f"Context: {parsed_text[:2000]}\n\nQuestion: {req.query}")
        return {"answer": fallback}

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
        doc_preview = parsed_text[:4000] if parsed_text else "❌ No document uploaded yet."
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
        sg = SendGridAPIClient(api_key=os.getenv("SENDGRID_API_KEY"), ssl_context=ssl_context_)
        sg.send(message)

        return {"status": "✅ Report sent successfully"}
    except Exception as e:
        logger.exception("Email error")
        return {"status": f"⚠️ Error sending email: {str(e)}"}