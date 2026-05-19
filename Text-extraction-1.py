import streamlit as st
import os
import tempfile
import hashlib
import time
import requests
from pathlib import Path
from typing import List

# -------- OCR + Image --------
import cv2
import numpy as np
from PIL import Image
import pypdfium2
import torch

# -------- Surya OCR --------
from surya.models import load_predictors

# -------- LlamaIndex --------
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext

import chromadb

# -------- CONFIG --------
OLLAMA_BASE_URL = "http://localhost:8890/v1"
OLLAMA_MODEL = "meta/llama-3.1-8b-instruct"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

st.set_page_config(page_title="Chat with PDF/Image", layout="wide")
st.title("💬 Intelligent Document Extraction")
st.title("TCS NVIDIA Capability Centre.... Welcome!")

# -------- SESSION STATE --------
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("file_hash", None)

# -------- DEVICE DETECTION --------
def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()
st.sidebar.write(f"🖥 Device: **{DEVICE}**")

# -------- LOAD SURYA --------
@st.cache_resource
def load_surya():
    return load_predictors(device=DEVICE)

predictors = load_surya()

# -------- TEXT CLEANING --------
def clean_text(text: str):

    text = text.replace("\x0c", " ")
    text = text.replace("\ufeff", " ")

    text = text.encode("utf-8", "ignore").decode("utf-8")

    return " ".join(text.split())

# -------- WAIT FOR OLLAMA --------
def wait_for_server(url=f"{OLLAMA_BASE_URL}/health/ready", timeout=60):

    start = time.time()

    while True:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except:
            pass

        if time.time() - start > timeout:
            raise TimeoutError("Ollama server not ready")

        time.sleep(1)

# -------- LLM --------
def ask_llama(prompt: str):

    url = f"{OLLAMA_BASE_URL}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """
You are a multilingual assistant.
Answer ONLY using the provided context.
Support English, Hindi, and Arabic.
Return the answer in English if user didn't mention any language like hindi or arabic language in the question asked.
"""
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]

# -------- PDF → IMAGES --------
def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:

    images = []

    doc = pypdfium2.PdfDocument(pdf_path)

    try:
        for page in range(len(doc)):

            renderer = doc.render(
                pypdfium2.PdfBitmap.to_pil,
                page_indices=[page],
                scale=dpi / 72
            )

            img = list(renderer)[0].convert("RGB")

            img_np = np.array(img)

            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

            denoise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            processed = Image.fromarray(denoise)

            images.append(processed)

    finally:
        doc.close()

    return images

# -------- SURYA OCR --------
def surya_ocr(images):

    recognitions = predictors["recognition"](
        images=images,
        task_names=["ocr_with_boxes"] * len(images),
        det_predictor=predictors["detection"]
    )

    texts = []

    for rec in recognitions:

        page_text = " ".join([line.text for line in rec.text_lines])

        page_text = clean_text(page_text)

        texts.append(page_text)

    return texts

# -------- VECTOR DATABASE --------
@st.cache_resource
def build_vectorstore(text_pages, persist_dir):

    splitter = SentenceSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    documents = []

    for page_num, page_text in enumerate(text_pages, start=1):

        chunks = splitter.split_text(page_text)

        for chunk in chunks:

            documents.append(
                Document(
                    text=chunk,
                    metadata={"page": page_num}
                )
            )

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device=DEVICE
    )

    client = chromadb.PersistentClient(path=persist_dir)

    collection = client.get_or_create_collection("doc_collection")

    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )

    return index

# -------- QUERY DOCUMENT --------
def query_document(question):

    retriever = st.session_state.vectorstore.as_retriever(
        similarity_top_k=4
    )

    nodes = retriever.retrieve(question)

    context = "\n\n".join(
        [
            f"(Page {n.metadata.get('page')}) {n.text}"
            for n in nodes
        ]
    )

    prompt = f"""
Context:
{context}

Question:
{question}

Answer accurately using ONLY the context and cite page numbers.
"""

    return ask_llama(prompt)

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "jpg", "jpeg", "png"]
)

if uploaded_file:

    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    uploaded_file.seek(0)

    if st.session_state.file_hash != file_hash:

        st.session_state.file_hash = file_hash
        st.session_state.messages = []

        suffix = Path(uploaded_file.name).suffix.lower()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(file_bytes)
        tmp.close()

        file_path = tmp.name

        with st.spinner("🔍 Processing file..."):

            if suffix == ".pdf":

                st.info("📄 PDF detected → Converting pages to images")

                images = pdf_to_images(file_path)

                st.info(f"🔎 Running Surya OCR on {len(images)} pages")

                text_pages = surya_ocr(images)

            else:

                st.info("🖼 Image detected → Using Surya OCR")

                image = Image.open(file_path).convert("RGB")

                text_pages = surya_ocr([image])

        persist_dir = f"./chroma_{file_hash}"

        with st.spinner("📚 Building vector database..."):

            st.session_state.vectorstore = build_vectorstore(
                text_pages,
                persist_dir
            )

        os.remove(file_path)

        st.success("✅ File processed successfully!")

# -------- CHAT UI --------
if st.session_state.vectorstore:

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about the document...")

    if user_input:

        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            with st.spinner("🤖 Thinking..."):

                answer = query_document(user_input)

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
