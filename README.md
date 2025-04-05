# local-rag-llm-app

A local Retrieval-Augmented Generation (RAG) app that lets you chat with your own documents using:

- High-quality embeddings (`BAAI/bge-base-en-v1.5`)
- Local LLM via [Ollama](https://ollama.com) (e.g. `llama3`, `mistral`)
- Document chunk retrieval with context and metadata
- Streamlit UI with live chat, source previews, and fallback knowledge

---

## Features

- Load and index PDFs, DOCX, TXT, and HTML files
- Show retrieved document source, page number, and content
- RAG-powered QA using Chroma and Ollama
- Fallback answers from local "knowledge snippets"
- Fully offline (no OpenAI API required)

---

## Quick Start

### 1. Set up your environment

```bash
conda create -n local-rag python=3.10
conda activate local-rag
pip install -r requirements.txt
```

Or use `environment.yml` if available:

```bash
conda env create -f environment.yml
conda activate local-rag
```

---

### 2. Pull your LLM with Ollama

```bash
ollama pull llama3
```

Other supported models:

```bash
ollama pull mistral
ollama pull deepseek-llm
```

---

### 3. Add your documents

Place `.pdf`, `.docx`, `.txt`, or `.html` files into the `uploaded_docs/` folder.

---

### 4. Run the app

```bash
python main.py
```

Or run directly with Streamlit:

```bash
streamlit run src/app.py --server.runOnSave=false
```

---

## Folder Structure

```
local-rag-llm-app/
├── src/
│   ├── app.py                  # Streamlit UI
│   └── rag_local.py            # RAG backend logic
├── uploaded_docs/              # Your document corpus
├── fallback_knowledge/         # Optional fallback content
├── chroma_db/                  # Vector store (auto-generated)
├── requirements.txt
├── environment.yml             # (optional) for conda setup
├── main.py                     # Entry point
└── README.md
```

---

## Fallback Knowledge (optional)

If a question can't be answered from the documents, the app looks in `fallback_knowledge/`.

Example files:

```
fallback_knowledge/bayesian_hypothesis.txt
fallback_knowledge/llm.txt
```

Each file should contain a short summary or explanation of that topic.

---

## Configuration & Tuning

| Parameter            | File            | Description                      |
|---------------------|------------------|----------------------------------|
| Embedding model      | `rag_local.py`   | `BAAI/bge-base-en-v1.5`          |
| LLM model            | `rag_local.py`   | `llama3.2`, `mistral`, etc.      |
| Chunk size / overlap | `rag_local.py`   | 1500 / 300 recommended           |
| Retrieval depth      | `rag_local.py`   | `search_kwargs={"k": 5}`         |

---

## Possible Future Enhancements

- Reranking with cross-encoders
- Displaying answer confidence scores
- Summarization or topic tagging of context
- Support for CSV and Markdown files
- UI-based document or fallback upload

---

## Acknowledgments

This project uses:

- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Ollama](https://ollama.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io/)

---

## Privacy & Local Use

This app runs 100% locally. Your documents and queries never leave your machine.
