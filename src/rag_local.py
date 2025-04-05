import os
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQAWithSourcesChain


def load_documents(folder_path):
    """Load all supported document types from the given folder."""
    all_docs = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        elif filename.endswith(".html") or filename.endswith(".htm"):
            loader = UnstructuredHTMLLoader(path)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            continue
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        all_docs.extend(docs)
    return all_docs


def split_documents(documents):
    """Split documents into chunks with overlap for better context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    """Embed and store the document chunks in a Chroma vector database."""
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    return vectordb


def load_vectorstore():
    """Load the persisted vector store from disk."""
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )


def create_qa_chain(vectordb):
    """Create a RetrievalQA chain with local Ollama LLM."""
    llm = OllamaLLM(model="llama3.2")
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    return RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)