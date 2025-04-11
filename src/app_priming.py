# RAG Option 1: Continuous Chat with Smart Document Handling
# File: src/app_priming.py

import streamlit as st
import os
from rag_local import load_documents, split_documents, create_vectorstore
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredHTMLLoader, UnstructuredWordDocumentLoader
)

st.set_page_config(page_title="RAG Option 1 - Continuous Chat")
st.title("RAG: Continuous Chat with Smart Document Handling")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "all_docs" not in st.session_state:
    st.session_state.all_docs = []
if "full_docs" not in st.session_state:
    st.session_state.full_docs = []

MODEL_NAME = "llama3.2"
MAX_CHARS = 20000
MAX_CHUNKS = 25


def get_full_documents(folder):
    docs = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        elif filename.endswith(".html"):
            loader = UnstructuredHTMLLoader(path)
        else:
            continue

        pages = loader.load()
        full_content = "\n\n".join([p.page_content for p in pages])
        doc = {"filename": filename, "content": full_content, "pages": pages}
        docs.append(doc)
    return docs


def chop_and_append(content, filename, messages):
    chunks = [content[i:i + MAX_CHARS] for i in range(0, len(content), MAX_CHARS)]
    merged_text = "\n\n".join(chunks)
    msg = f"Here is the full content of the document '{filename}':\n\n{merged_text}"
    messages.append(HumanMessage(content=msg))
    st.code(msg[:3000], language="text")


uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt", "html"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("uploaded_docs", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("uploaded_docs", file.name), "wb") as f:
            f.write(file.read())
    st.success("Documents uploaded.")

    raw_docs = load_documents("uploaded_docs")
    chunks = split_documents(raw_docs)

    if os.path.exists("chroma_db/index"):
        vectordb = Chroma(persist_directory="chroma_db")
    else:
        vectordb = create_vectorstore(chunks)

    st.session_state.vectordb = vectordb
    st.session_state.all_docs = raw_docs
    st.session_state.full_docs = get_full_documents("uploaded_docs")
    st.success("Vector store and full document memory initialized.")

if st.session_state.vectordb and st.session_state.full_docs:
    st.markdown("---")
    st.subheader("Ask a question")
    query = st.text_input("Your question")
    if query:
        messages = []
        for role, content in st.session_state.chat_history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        results = st.session_state.vectordb.similarity_search(query, k=5)
        print(f"results: {results}")
        retrieved_filenames = list(set(doc.metadata.get("source") for doc in results))
        full_docs = [doc for doc in st.session_state.full_docs if doc["filename"] in retrieved_filenames]

        for doc in full_docs:
            filename = doc["filename"]
            content = doc["content"].strip()
            pages = doc["pages"]
            doc_chunks = [content[i:i + MAX_CHARS] for i in range(0, len(content), MAX_CHARS)]

            if len(doc_chunks) <= MAX_CHUNKS:
                chop_and_append(content, filename, messages)
            else:
                matches = [m for m in results if m.metadata.get("source") == filename]
                retrieved_pages = set(m.metadata.get("page") for m in matches if m.metadata.get("page") is not None)
                if not retrieved_pages:
                    continue
                for page_idx in retrieved_pages:
                    if 0 <= page_idx < len(pages):
                        page = pages[page_idx]
                        page_text = page.page_content.strip()
                        chop_and_append(page_text, f"{filename} (page {page_idx + 1})", messages)

        question_prompt = f"""
You have read the documents above.

Now, answer this question strictly based on the content of those documents only.
If the answer is not present in the documents, say \"I don't know.\"
Do not use any outside knowledge or assumptions.

Question: {query}
""".strip()
        messages.append(HumanMessage(content=question_prompt))
        st.code(question_prompt, language="text")

        chat = ChatOllama(model=MODEL_NAME)
        response = chat(messages)
        answer = response.content

        st.subheader("Answer")
        st.markdown(answer)

        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("ai", answer))

if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Chat History")
    for role, content in st.session_state.chat_history:
        st.markdown(f"**{role.capitalize()}:** {content}")
