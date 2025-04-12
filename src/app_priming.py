import streamlit as st
import os
import hashlib
from rag_local import load_documents, split_documents, create_vectorstore, load_vectorstore, get_full_documents
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage

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

def chop_and_append(content, filename, messages):
    chunks = [content[i:i + MAX_CHARS] for i in range(0, len(content), MAX_CHARS)]
    merged_text = "\n\n".join(chunks)
    msg = f"Here is the full content of the document '{filename}':\n\n{merged_text}"
    messages.append(HumanMessage(content=msg))
    st.code(msg, language="text")

uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt", "html"], accept_multiple_files=True)

# Upload and process new files
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

# Auto-load saved vectorstore if it exists and wasn't loaded already
elif not st.session_state.vectordb and os.path.exists("faiss_index"):
    st.session_state.vectordb = load_vectorstore()
    st.session_state.full_docs = get_full_documents("uploaded_docs")
    st.info("Loaded existing FAISS index and documents.")

# Display chat interface if vectorstore is available
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

        retrieved_filenames = list(set(doc.metadata.get("source") for doc in results))

        doc_lookup = {doc["filename"]: doc for doc in st.session_state.full_docs}
        for filename in retrieved_filenames:
            if filename not in doc_lookup:
                continue
            doc = doc_lookup[filename]
            content = doc["content"].strip()
            pages = doc["pages"]
            num_chunks = len(content) // MAX_CHARS + 1

            if num_chunks <= MAX_CHUNKS:
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
If the answer is not present in the documents, say "I don't know."
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

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Chat History")
    for role, content in st.session_state.chat_history:
        st.markdown(f"**{role.capitalize()}:** {content}")