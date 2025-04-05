import streamlit as st
import os
from rag_local import (
    load_documents, split_documents,
    create_vectorstore, load_vectorstore, create_qa_chain
)

st.set_page_config(page_title="📚 Local RAG App", layout="wide")
st.title("🧠 Chat with Your Documents (Offline)")

if "qa" not in st.session_state:
    st.session_state.qa = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs, DOCX, TXT, HTML", type=["pdf", "docx", "txt", "html"], accept_multiple_files=True)
    if st.button("🔍 Process & Index"):
        os.makedirs("uploaded_docs", exist_ok=True)
        for file in uploaded_files:
            with open(os.path.join("uploaded_docs", file.name), "wb") as f:
                f.write(file.read())
        docs = load_documents("uploaded_docs")
        chunks = split_documents(docs)
        create_vectorstore(chunks)
        st.session_state.qa = create_qa_chain(load_vectorstore())
        st.success("✅ Documents processed and indexed.")

st.subheader("💬 Ask a Question")

query = st.text_input("Enter your question here:")

if query and st.session_state.qa:
    with st.spinner("Thinking..."):
        # Get retriever manually to inspect context
        retriever = st.session_state.qa.retriever
        docs = retriever.get_relevant_documents(query)

        if docs:
            context_text = "\n\n".join([doc.page_content for doc in docs])
            result = st.session_state.qa.invoke({"question": query})

            st.subheader("🧠 Answer")
            st.markdown(result["answer"])

            st.subheader("📚 Retrieved Context")
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                source = metadata.get("source", "Unknown file")
                page = metadata.get("page", "Unknown page")
                st.markdown(f"**Source {i}** — `{source}`, page {page}")
                st.code(doc.page_content.strip()[:1000])  # first 1000 chars

            st.session_state.chat_history.append((query, result["answer"], [doc.metadata for doc in docs]))

        else:
            st.warning("🤖 No relevant context found in your documents. Falling back to general knowledge.")
            from rag_local import OllamaLLM  # quick local fallback
            fallback_llm = OllamaLLM(model="llama3.2")
            fallback_response = fallback_llm.invoke(query)

            st.subheader("🧠 Answer (fallback)")
            st.markdown(fallback_response)

            st.session_state.chat_history.append((query, fallback_response, []))

if st.session_state.chat_history:
    st.subheader("🗂️ Chat History")
    for i, (q, a, s) in enumerate(reversed(st.session_state.chat_history), 1):
        with st.expander(f"Q{i}: {q}"):
            st.markdown(f"**Answer:** {a}")
            st.markdown(f"📎 **Sources:** {s}")