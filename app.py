import os
import io
import shutil
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

QA_CHAIN = None

def initialize_retriever(file_path):
    print(f"Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    loaded_doc = loader.load()
    
    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(loaded_doc)
    
    print("Creating in-memory vector database...")
    # Using OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    # Using FAISS in-memory so it works perfectly on Vercel/Render free tier without disk access
    vectordb = FAISS.from_documents(chunks, embeddings)
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever

@asynccontextmanager
async def lifespan(app_instance):
    yield
    global QA_CHAIN
    QA_CHAIN = None

app = FastAPI(lifespan=lifespan)

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = os.path.join(TEMPLATE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/init")
async def init_model(
    file: UploadFile = File(...),
):
    global QA_CHAIN
    
    if not os.environ.get("OPENAI_API_KEY"):
        return JSONResponse({
            "status": "error", 
            "message": "OPENAI_API_KEY environment variable is missing! Please add it in your Vercel/Render dashboard."
        })
        
    try:
        # Save uploaded file to /tmp (Vercel/Render compatible)
        os.makedirs("/tmp/temp_uploads", exist_ok=True)
        file_path = f"/tmp/temp_uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        retriever = initialize_retriever(file_path)
        
        # Use GPT-3.5-Turbo (fast, cheap, works on cloud)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

        system_prompt = (
            "You are a precise, professional assistant. "
            "Answer the user's question based ONLY on the context provided. "
            "Keep your answer concise. Do not repeat sentences. "
            "If the context does not contain the answer, say "
            "\"I cannot find the answer in the provided document.\"\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        document_chain = create_stuff_documents_chain(llm, prompt)
        QA_CHAIN = create_retrieval_chain(retriever, document_chain)
        
        return {"status": "success", "message": f"Loaded: {file.filename}. Cloud AI Engine Ready!"}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

@app.post("/api/query")
async def query_model(request: Request):
    global QA_CHAIN
    data = await request.json()
    query = data.get("query", "")
    
    if QA_CHAIN is None:
        return JSONResponse({"status": "error", "message": "Document not initialized yet."})
        
    if not query.strip():
        return {"status": "success", "answer": "", "sources": []}
        
    try:
        response = QA_CHAIN.invoke({"input": query})
        answer = response.get("answer", "")
        
        sources_out = []
        context_docs = response.get("context", [])
        if context_docs:
            for doc in context_docs:
                snippet = doc.page_content.strip().replace("\n", " ")[:200]
                page = doc.metadata.get("page", doc.metadata.get("page_number", "N/A"))
                sources_out.append({"snippet": snippet, "page": page})
                
        return {"status": "success", "answer": answer, "sources": sources_out}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "model_loaded": QA_CHAIN is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
