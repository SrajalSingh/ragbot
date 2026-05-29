# ☁️ Cloud RAG Chatbot

**Live Demo:** [View on Render](https://ragbot.onrender.com/) *(Note: Replace this URL with your actual Render link!)*

A modern, web-based **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload PDF documents and instantly chat with them. Powered by **OpenAI's GPT-3.5-Turbo**, this application is designed for seamless cloud deployment on Render or Vercel.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.0+-1C3C3C?logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT_3.5-412991?logo=openai&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render&logoColor=white)

---

## ✨ Features

*   **Cloud Deployment Ready:** Optimized for one-click deployment on Render with Docker.
*   **In-Memory Vector Search:** Uses `FAISS` to run 100% in memory, completely bypassing Read-Only File System errors on cloud platforms.
*   **OpenAI Integration:** Leverages `GPT-3.5-Turbo` and `OpenAI Embeddings` for blazing fast, highly accurate responses.
*   **Modern RAG Pipeline:** Built utilizing LangChain's advanced `create_retrieval_chain`.
*   **Beautiful UI:** Features a sleek Glassmorphism design, typing indicators, toast notifications, and auto-scrolling.
*   **Markdown Support:** The bot dynamically renders code blocks, lists, and formatted text.

---

## 🏗️ Architecture

1.  **Document Ingestion:** PDFs are temporarily uploaded and parsed using `PyPDFLoader`.
2.  **Chunking:** Text is split using `RecursiveCharacterTextSplitter` to maintain contextual boundaries.
3.  **Embedding:** `OpenAIEmbeddings` convert text chunks into high-dimensional vectors.
4.  **Vector Store:** `FAISS` stores the vectors entirely in-memory for instant retrieval.
5.  **Generation:** `GPT-3.5-Turbo` synthesizes the retrieved context into a natural, conversational answer.

---

## 🚀 Quick Start (Local Development)

### 1. Clone the repository
```bash
git clone https://github.com/SrajalSingh/ragbot.git
cd ragbot
```

### 2. Set up Environment
Create a virtual environment and install the lightweight cloud dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your API Key
Set your OpenAI API key in your terminal before running:
```bash
export OPENAI_API_KEY="sk-your-secret-key"
```

### 4. Run the Application
```bash
python app.py
```
*The app will be available at http://localhost:8000*

## Screenshot

> Upload a PDF, pick your local model, and start chatting.

---

## Quick Start

### Prerequisites

- Python 3.10+
- A GGUF model file (e.g., [Qwen2.5-3B-Instruct-Q8](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)) **or** a HuggingFace model ID

### Installation

```bash
# Clone the repo
git clone https://github.com/SrajalSingh/ragbot.git
cd ragbot

# Install dependencies
pip install fastapi uvicorn langchain langchain-classic langchain-community langchain-huggingface langchain-text-splitters chromadb sentence-transformers llama-cpp-python torch transformers
