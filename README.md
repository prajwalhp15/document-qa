# Document Q&A System

A full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload documents (PDF) and ask natural language questions.  
The system retrieves context from the document and provides citation-grounded answers using local embeddings and a Local LLM via Ollama.

## 🚀 Features

- 📄 Upload PDF documents
- 🧠 Local embeddings using SentenceTransformers
- 📚 Citation-grounded retrieval using FAISS
- 💬 Local LLM inference (e.g., Phi-3 mini via Ollama)
- 🖥️ Streamlit UI frontend
- ⚡ Lightning-fast local document search and answers

##  Tech Stack

- **Backend:** FastAPI
- **Retrieval:** FAISS + SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM:** Local (Ollama / Phi-3 mini)
- **Frontend:** Streamlit
- **PDF parsing:** PyPDF

##  Installation

```bash
git clone https://github.com/prajwalhp15/document-qa
cd document-qa
pip install -r requirements.txt

 Author

Prajwal H. P.