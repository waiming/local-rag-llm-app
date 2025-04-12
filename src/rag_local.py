import os
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQAWithSourcesChain


def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
        elif filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename))
        elif filename.endswith(".html"):
            loader = UnstructuredHTMLLoader(os.path.join(folder_path, filename))
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(os.path.join(folder_path, filename))
        else:
            continue
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        documents.extend(docs)
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local("faiss_index")
    return vectordb


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vectordb


def create_qa_chain(vectordb):
    llm = OllamaLLM(model="llama3.2")
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever()
    )


def get_full_documents(folder):
    from langchain_community.document_loaders import (
        PyPDFLoader, TextLoader, UnstructuredHTMLLoader, UnstructuredWordDocumentLoader
    )

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
        docs.append({
            "filename": filename,
            "content": full_content,
            "pages": pages
        })
    return docs