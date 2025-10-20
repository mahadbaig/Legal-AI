# ⚖️ AI Legal Contract Analyzer

An intelligent legal document analysis system powered by AI that helps users understand complex contracts, agreements, and legal documents through natural language conversations.

## 🌟 Features

- **Document Upload & Parsing**: Support for PDF, DOCX, and TXT files
- **AI-Powered Analysis**: Uses Groq's Mixtral model for intelligent legal document analysis
- **Semantic Search**: FAISS-powered vector database for context-aware document search
- **Interactive Chat Interface**: Natural language Q&A about uploaded documents
- **Automated Reporting**: Email comprehensive legal analysis reports via SendGrid
- **Web Search Integration**: External legal references using Tavily API
- **Plain Language Explanations**: Converts legal jargon into understandable language

## 🏗️ Architecture

### Backend (FastAPI)
- Document parsing and text extraction
- LangChain agent with multiple tools:
  - **SearchLegalText**: Searches uploaded documents using FAISS vector similarity
  - **WebSearch**: Fetches external legal references and case law
- FAISS vector store for semantic search
- SendGrid email integration for reports

### Frontend (Streamlit)
- Clean, user-friendly chat interface
- Real-time document processing status
- Interactive Q&A sessions
- Email report generation

## 📋 Prerequisites

- Python 3.13+
- API Keys:
  - Groq API (for LLM)
  - Tavily API (for web search)
  - SendGrid API (for email reports)
  - LangChain API (optional, for tracing)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mahadbaig/Legal-AI
cd legal-ai
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:

```env
# Groq API (Required)
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=mixtral-8x7b-32768

# Tavily Search API (Required)
TAVILY_API_KEY=your_tavily_api_key

# SendGrid Email (Required for email reports)
SENDGRID_API_KEY=your_sendgrid_api_key
FROM_EMAIL=your_verified_sendgrid_email@example.com

# Alternative SMTP (Optional fallback)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_APP_PASSWORD=your_app_password

# LangChain Tracing (Optional)
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=legal-ai-project
LANGCHAIN_TRACING_V2=true
```

## 🎯 Usage

### Start the Backend
```bash
cd backend
uvicorn app:app --reload --port 8000
```

The backend will be available at `http://127.0.0.1:8000`

### Start the Frontend
In a separate terminal:
```bash
streamlit run streamlit_app/app.py
```

The Streamlit app will open automatically in your browser at `http://localhost:8501`

### Using the Application

1. **Upload Document**: Upload a PDF, DOCX, or TXT legal document
2. **Start Chat**: Click "Start Chatting" to begin analysis
3. **Ask Questions**: Type questions in natural language about the document
4. **Get Report**: Enter your email to receive a comprehensive analysis report

## 🔧 API Endpoints

### Backend Endpoints

- `GET /` - Health check
- `POST /parse` - Upload and parse document
- `POST /query` - Query the document with natural language
- `POST /email-report` - Generate and email comprehensive report
- `GET /document-status` - Check document upload status (debug)
- `POST /vector-search` - Test FAISS vector search (debug)
- `POST /clear-document` - Clear uploaded document and reset state

## 🧠 How It Works

### Document Processing
1. Document is uploaded and parsed using PyPDF2/python-docx
2. Text is split into chunks using RecursiveCharacterTextSplitter
3. FAISS vector embeddings are created using SentenceTransformer
4. Document is stored both as raw text and vector embeddings

### Query Processing
1. User query is processed by LangChain agent
2. Agent decides whether to:
   - Search the document (semantic + keyword search)
   - Search the web for external references
   - Or both
3. Results are formatted in plain language with:
   - Direct answers
   - Relevant clauses
   - Risk analysis
   - Practical implications
   - External references

### Email Reports
- Comprehensive analysis combining document insights and web research
- Includes chat history summary
- Highlights risks, obligations, and key terms
- Provides actionable recommendations

## 📦 Dependencies

### Core
- FastAPI - Web framework
- Streamlit - Frontend interface
- LangChain - AI orchestration
- Groq - LLM provider

### Document Processing
- PyPDF2 - PDF parsing
- python-docx - Word document parsing

### Vector Search
- FAISS - Vector similarity search
- SentenceTransformer - Text embeddings

### Integrations
- SendGrid - Email delivery
- Tavily - Web search

## 🐳 Deployment

### Using Railway/Heroku

The project includes configuration files for easy deployment:

- `Procfile` - Heroku deployment
- `railway.toml` - Railway deployment
- `requirements.txt` - All dependencies

Set environment variables in your deployment platform's dashboard.

### Docker (Optional)
```bash
docker-compose up
```

## 🛠️ Development

### Run Tests
```bash
python test_backend.py
```

### Debug Mode
The Streamlit sidebar includes debug tools:
- Backend status checker
- Vector search tester
- Document state viewer

## ⚠️ Important Notes

### FAISS Vector Database
- Requires `faiss-cpu` or `faiss-gpu` to be installed
- Falls back to keyword search if FAISS is unavailable
- Vector store is saved to `./faiss_store/` directory

### SSL/Certificate Issues
- The app handles SSL certificate issues with multiple fallback methods
- Uses `certifi` for proper certificate chain validation
- SMTP fallback available if SendGrid has SSL issues

### Browser Storage
- **DO NOT** use localStorage or sessionStorage in artifacts
- Not supported in Claude.ai environment
- Use React state or JavaScript variables instead

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- Groq for fast LLM inference
- LangChain for AI orchestration
- FAISS for vector similarity search
- Anthropic Claude for assistance in development

## 📞 Support

For issues or questions:
- Check the debug endpoints (`/document-status`)
- Review backend logs for detailed error messages
- Ensure all API keys are correctly configured

---

**Built with ❤️ using AI-powered tools**
