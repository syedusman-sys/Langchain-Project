
import os
import streamlit as st
import time
import requests
import tempfile
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)
from langchain_core.prompts import ChatPromptTemplate

# ✅ Load environment variables from .env file
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Streamlit UI ---
st.title("News Research Assistant 📰🤖")
st.sidebar.header("Configuration")

# --- Input Section ---
url1 = st.sidebar.text_input("Enter URL 1")
url2 = st.sidebar.text_input("Enter URL 2")
url3 = st.sidebar.text_input("Enter URL 3")

uploaded_files = st.sidebar.file_uploader(
    "📎 Upload PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

process_button = st.sidebar.button("🚀 Process Documents")
query = st.text_input("Ask a question about the content:")

FAISS_INDEX_PATH = "faiss_index"

# --- Process URLs and Files ---
if process_button:
    urls = [u.strip() for u in [url1, url2, url3] if u.strip()]
    all_docs = []

    with st.spinner("🔄 Loading and processing documents..."):
        # Handle URLs
        for url in urls:
            try:
                if url.lower().endswith(".pdf"):
                    response = requests.get(url)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(response.content)
                        loader = PyPDFLoader(tmp_file.name)
                        all_docs.extend(loader.load())

                elif url.lower().endswith(".docx"):
                    response = requests.get(url)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                        tmp_file.write(response.content)
                        loader = UnstructuredWordDocumentLoader(tmp_file.name)
                        all_docs.extend(loader.load())

                else:
                    loader = UnstructuredURLLoader(urls=[url])
                    all_docs.extend(loader.load())

                st.success(f"✅ Loaded: {url}")
            except Exception as e:
                st.warning(f"⚠️ Skipped {url} due to error: {e}")

        # Handle uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    if uploaded_file.name.lower().endswith(".pdf"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            loader = PyPDFLoader(tmp_file.name)
                            all_docs.extend(loader.load())

                    elif uploaded_file.name.lower().endswith(".docx"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            loader = UnstructuredWordDocumentLoader(tmp_file.name)
                            all_docs.extend(loader.load())

                    st.success(f"✅ Uploaded: {uploaded_file.name}")
                except Exception as e:
                    st.warning(f"⚠️ Could not load {uploaded_file.name}: {e}")

        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(all_docs)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=os.getenv("GOOGLE_API_KEY"),credentials_path=None)
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)
            st.success("✅ Documents processed and FAISS index saved!")
        else:
            st.error("No valid documents found.")

# --- Question Answering ---
if query:
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=os.getenv("GOOGLE_API_KEY"),credentials_path=None)
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant with access to context from uploaded documents and online sources.\nIf no relevant context is available, use your general knowledge to answer clearly.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
        )

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, max_output_tokens=500, google_api_key=os.getenv("GOOGLE_API_KEY"))

        retrieved_docs = retriever.get_relevant_documents(query)

        if retrieved_docs:
            context_text = "\n".join([d.page_content for d in retrieved_docs])
        else:
            st.info("No relevant documents found — Gemini will answer from general knowledge.")
            context_text = ""

        response = llm.invoke(prompt.format(context=context_text, question=query))

        st.header("🧩 Answer")
        st.write(response.content)

        # --- Show referenced URLs ---
        st.markdown("### 🔗 Sources used:")
        if retrieved_docs:
            urls = list(set([doc.metadata.get("source", "Uploaded File or Unknown") for doc in retrieved_docs]))
            for i, url in enumerate(urls, 1):
                st.markdown(f"**{i}.** [{url}]({url})")
        else:
            st.write("No document sources found.")
    else:
        st.warning("⚠️ No FAISS index found. Please process URLs or upload documents first.")