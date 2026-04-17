#  Local RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and chat with them using local LLMs. No data leaves your machine — everything runs privately on your hardware.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.0+-1C3C3C?logo=langchain&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  Features

- **100% Local** — No API keys, no cloud, no data leaks
- **PDF Upload** — Upload any PDF and start querying instantly
- **Multiple Model Backends** — Supports GGUF (llama.cpp) and HuggingFace Transformers
- **Modern RAG Pipeline** — Built with LangChain's `create_retrieval_chain` and ChromaDB vector store
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
