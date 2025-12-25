import numpy as np
import faiss
import requests
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# -------- Embeddings (local) --------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384

index = faiss.IndexFlatL2(EMBED_DIM)
chunks = []

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

# -------- Document Processing --------
def process_document(pdf_path: str):
    global chunks

    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    embeddings = embedder.encode(chunks).astype("float32")
    index.reset()
    index.add(embeddings)

    print("Document processed successfully!")

# -------- Question Answering (Ollama) --------
def ask_question(query: str):
    if not chunks:
        return {
            "answer": "No document uploaded yet.",
            "citations": []
        }

    q_emb = embedder.encode([query]).astype("float32")
    _, I = index.search(q_emb, 3)

    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a document analysis assistant.

Using ONLY the context below, answer the question clearly.
If the question asks for a summary or list, infer it from the context.
Do not hallucinate.

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    result = response.json()

    return {
        "answer": result.get("response", "").strip(),
        "citations": retrieved_chunks
    }
