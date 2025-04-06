# RAG Option 1: Continuous Chat with ChatOllama and Full Document Streaming
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
st.title("RAG: Continuous Chat with Full Document Priming")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "all_docs" not in st.session_state:
    st.session_state.all_docs = []
if "full_docs" not in st.session_state:
    st.session_state.full_docs = []

MAX_CHARS = 20000  # Limit per message to avoid context overflow


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
        doc = {"filename": filename, "content": full_content}
        docs.append(doc)
    return docs

uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt", "html"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("uploaded_docs", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("uploaded_docs", file.name), "wb") as f:
            f.write(file.read())
    st.success("Documents uploaded.")

    raw_docs = load_documents("uploaded_docs")
    chunks = split_documents(raw_docs)
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
        # Build message history
        messages = []
        for role, content in st.session_state.chat_history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        # Step 1: Retrieve relevant documents
        results = st.session_state.vectordb.similarity_search(query, k=5)
        retrieved_filenames = list(set(
            doc.metadata.get("source") for doc in results
        ))

        # Step 2: Load full documents and chunk them if too long
        full_docs = [doc for doc in st.session_state.full_docs if doc["filename"] in retrieved_filenames]

        for doc in full_docs:
            filename = doc["filename"]
            content = doc["content"].strip()
            chunks = [content[i:i+MAX_CHARS] for i in range(0, len(content), MAX_CHARS)]
            for idx, chunk in enumerate(chunks):
                prompt = f"Here is part {idx+1} of the document '{filename}':\n\n{chunk}"
                messages.append(HumanMessage(content=prompt))
                st.code(prompt, language="text")

        # Step 3: Add the user query
        question_prompt = f"""
You have read the documents above.

Now, answer this question strictly based on the content of those documents only.
If the answer is not present in the documents, say "I don't know."
Do not use any outside knowledge or assumptions.

Question: {query}
""".strip()
        messages.append(HumanMessage(content=question_prompt))
        st.code(question_prompt, language="text")

        chat = ChatOllama(model="llama3.2")
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