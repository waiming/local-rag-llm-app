# RAG Option 2: Conversational Retrieval with Memory
# File: src/app_conversational.py

import streamlit as st
import os
from rag_local import load_documents, split_documents, create_vectorstore, load_vectorstore
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="RAG Option 2 - Conversational")
st.title("RAG: Conversational Retrieval with Memory")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

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

    llm = OllamaLLM(model="llama3.2")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    st.session_state.qa_chain = qa_chain
    st.success("Retrieval chain initialized.")

if st.session_state.qa_chain:
    st.markdown("---")
    query = st.text_input("Ask a question about your documents")
    if query:
        result = st.session_state.qa_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append((query, result["answer"]))
        st.subheader("Answer")
        st.markdown(result["answer"])
        st.subheader("Sources")
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "unknown")
            st.markdown(f"**{source}**")
            st.code(doc.page_content.strip()[:1000])

for q, a in st.session_state.chat_history:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
