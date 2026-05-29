# Cloud RAG Chatbot

**Live Demo:** [View on Render](https://ragbot.onrender.com/) *(Note: Replace this URL with your actual Render link!)*

A web-based Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and chat with them using OpenAI's powerful language models. Built for seamless cloud deployment on Render.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.0+-1C3C3C?logo=langchain&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render&logoColor=white)

---

##  Features

- **Cloud Deployment Ready** — Pre-configured for one-click deployment on Render with Docker
- **PDF Upload** — Upload any PDF and start querying instantly
- **OpenAI Integration** — Uses GPT-3.5-Turbo and OpenAI Embeddings for fast, accurate responses
- **Modern RAG Pipeline** — Built with LangChain's `create_retrieval_chain` and FAISS in-memory vector store
- **Markdown Responses** — Bot answers render with proper formatting (code blocks, lists, bold, etc.)
- **Beautiful UI** — Glassmorphism design with typing indicators, toast notifications, and responsive layout
- **Configurable** — Adjust temperature, max tokens, and model engine from the sidebar

---

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
