import os
import uuid
import shutil
import json
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_core.prompts import ChatPromptTemplate

TEMP_CHROMA_DIR = "./chroma_db"
os.makedirs(TEMP_CHROMA_DIR, exist_ok=True)

QA_CHAIN = None
LLM_INSTANCE = None

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def document_loader(file_path):
    print(f"Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    print("Document loaded.")
    return loaded_document

def text_splitter(data):
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def vector_database(chunks):
    print("Creating vector database...")
    embedding_model = get_embedding_model()
    session_id = uuid.uuid4().hex
    persist_directory = os.path.join(TEMP_CHROMA_DIR, f"vectordb_{session_id}")
    os.makedirs(persist_directory, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print(f"Vector database stored at: {persist_directory}")
    return vectordb

def initialize_retriever(file_path):
    print("Initializing retriever...")
    loaded_doc = document_loader(file_path)
    chunks = text_splitter(loaded_doc)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    print("Retriever initialized.")
    return retriever

def load_gguf_model(model_path, temperature, max_tokens):
    print(f"Loading GGUF model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    llm = LlamaCpp(
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        n_ctx=2048,
        n_gpu_layers = -1,
        stop=["<|im_end|>", "<|im_start|>user", "Question:"],
        verbose=False,
    )
    return llm

def load_hf_model(model_name_or_path, temperature, max_tokens):
    print(f"Loading HuggingFace model from {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

@asynccontextmanager
async def lifespan(app_instance):
    yield
    global QA_CHAIN, LLM_INSTANCE
    if LLM_INSTANCE is not None:
        del LLM_INSTANCE
        LLM_INSTANCE = None
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
    model_type: str = Form(...),
    model_path: str = Form(...),
    temperature: float = Form(0.1),
    max_tokens: int = Form(512)
):
    global QA_CHAIN, LLM_INSTANCE
    try:
        if not model_path:
            return JSONResponse({"status": "error", "message": "Please provide a valid model path."})
            
        model_path = model_path.strip().strip('"').strip("'")
        
        # Save uploaded file
        os.makedirs("./temp_uploads", exist_ok=True)
        file_path = f"./temp_uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"Processing file: {file_path}")
        retriever = initialize_retriever(file_path)
        
        if model_type == "GGUF":
            llm = load_gguf_model(model_path, temperature, max_tokens)
        else:
            llm = load_hf_model(model_path, temperature, max_tokens)

        LLM_INSTANCE = llm

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
        
        return {"status": "success", "message": f"Loaded: {file.filename}. Engine: {model_type}. Ready for questions!"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)})

@app.post("/api/query")
async def query_model(request: Request):
    global QA_CHAIN
    data = await request.json()
    query = data.get("query", "")
    
    if QA_CHAIN is None:
        return JSONResponse({"status": "error", "message": "Model and document not initialized yet."})
        
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
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)})

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "model_loaded": QA_CHAIN is not None}

if __name__ == "__main__":
    print("Starting HTML interface backend...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
