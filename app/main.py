from fastapi import FastAPI, UploadFile, File
from app.rag import process_document, ask_question

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Document Q&A API is running!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = f"data/uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    process_document(file_path)
    return {"message": "Document processed successfully!"}

@app.get("/ask")
async def ask(query: str):
    return ask_question(query)
