import streamlit as st
import os
import shutil
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

REQUEST_TIMEOUT = 60          # seconds, for LLM calls
MAX_LLM_RETRIES = 2
SERVER_READY_TIMEOUT = 60     # seconds, for startup health check

OCR_DPI_DEFAULT = 200          # was 300 -- 200 is plenty for typical scans/text
OCR_DPI_MAX_PAGES_THRESHOLD = 30   # drop dpi further for very large documents
OCR_DPI_LOW = 150
MAX_PAGES_ALLOWED = 150        # hard guard against runaway uploads

CHROMA_ROOT = "./chroma_stores"
CHROMA_TTL_SECONDS = 7 * 24 * 60 * 60  # delete stores untouched for 7 days

st.set_page_config(page_title="Chat with PDF/Image", layout="wide")
st.title("💬 Intelligent Document Extraction")
st.title("TCS NVIDIA Capability Centre.... Welcome!")

# -------- SESSION STATE --------
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("file_hash", None)
st.session_state.setdefault("server_ready", False)


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


# -------- LOAD EMBEDDING MODEL (own cache, independent of document) --------
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device=DEVICE,
    )


embed_model = load_embedding_model()


# -------- TEXT CLEANING --------
def clean_text(text: str):
    text = text.replace("\x0c", " ")
    text = text.replace("\ufeff", " ")
    text = text.encode("utf-8", "ignore").decode("utf-8")
    return " ".join(text.split())


# -------- WAIT FOR OLLAMA (now actually called at startup) --------
def wait_for_server(url: str = f"{OLLAMA_BASE_URL}/health/ready", timeout: int = SERVER_READY_TIMEOUT) -> bool:
    start = time.time()
    last_error = None
    while True:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except requests.RequestException as e:
            last_error = e
        if time.time() - start > timeout:
            raise TimeoutError(f"Ollama server not ready after {timeout}s: {last_error}")
        time.sleep(1)


if not st.session_state.server_ready:
    with st.spinner("⏳ Waiting for inference server..."):
        try:
            wait_for_server()
            st.session_state.server_ready = True
        except TimeoutError as e:
            st.error(f"❌ Inference server unavailable: {e}")
            st.stop()


# -------- LLM (timeout + retry + error handling) --------
def ask_llama(prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
    }
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """
You are a multilingual assistant. Answer ONLY using the provided context.
Support English, Hindi, and Arabic.
Return the answer in English if user didn't mention any language
like hindi or arabic language in the question asked.
""",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    last_error = None
    for attempt in range(1, MAX_LLM_RETRIES + 2):  # initial try + retries
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.Timeout as e:
            last_error = e
            st.warning(f"⏱ LLM request timed out (attempt {attempt}). Retrying...")
        except requests.RequestException as e:
            last_error = e
            st.warning(f"⚠️ LLM request failed (attempt {attempt}): {e}. Retrying...")
        except (KeyError, IndexError, ValueError) as e:
            # malformed response -- no point retrying with same payload structure issue,
            # but transient server hiccups can still cause this, so retry a couple times
            last_error = e
            st.warning(f"⚠️ Unexpected LLM response format (attempt {attempt}): {e}. Retrying...")
        time.sleep(1.5 * attempt)  # simple backoff

    return f"❌ Sorry, I couldn't get a response from the model after several attempts. ({last_error})"


# -------- PDF → IMAGES (generator, page-by-page to bound memory) --------
def pdf_to_images(pdf_path: str, dpi: int):
    """Yields one processed PIL image per page instead of holding the whole
    document in memory at once."""
    doc = pypdfium2.PdfDocument(pdf_path)
    try:
        n_pages = len(doc)
        for page in range(n_pages):
            renderer = doc.render(
                pypdfium2.PdfBitmap.to_pil,
                page_indices=[page],
                scale=dpi / 72,
            )
            img = list(renderer)[0].convert("RGB")
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            denoise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            processed = Image.fromarray(denoise)
            yield processed
    finally:
        doc.close()


def get_pdf_page_count(pdf_path: str) -> int:
    doc = pypdfium2.PdfDocument(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def choose_dpi(n_pages: int) -> int:
    """Scale down DPI for very large documents to keep memory/CPU bounded."""
    if n_pages > OCR_DPI_MAX_PAGES_THRESHOLD:
        return OCR_DPI_LOW
    return OCR_DPI_DEFAULT


# -------- SURYA OCR (works page-by-page to keep memory bounded) --------
def surya_ocr_pages(image_iter) -> List[str]:
    texts = []
    for img in image_iter:
        try:
            recognitions = predictors["recognition"](
                images=[img],
                task_names=["ocr_with_boxes"],
                det_predictor=predictors["detection"],
            )
            page_text = " ".join(line.text for line in recognitions[0].text_lines)
            texts.append(clean_text(page_text))
        except Exception as e:
            st.warning(f"⚠️ OCR failed on a page, skipping it: {e}")
            texts.append("")
    return texts


# -------- CHROMA STORE HOUSEKEEPING --------
def cleanup_old_chroma_stores(root: str = CHROMA_ROOT, ttl_seconds: int = CHROMA_TTL_SECONDS):
    """Remove chroma persist directories that haven't been touched in a while,
    so disk usage doesn't grow unbounded across many uploads."""
    if not os.path.isdir(root):
        return
    now = time.time()
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            last_modified = os.path.getmtime(path)
            if now - last_modified > ttl_seconds:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def touch_dir(path: str):
    try:
        os.utime(path, None)
    except OSError:
        pass


# -------- VECTOR DATABASE (dedup-aware, no re-embedding on repeat runs) --------
def build_vectorstore(text_pages: List[str], persist_dir: str, file_hash: str):
    """Builds (or loads) the vector index for this document.

    Uses `file_hash` as the effective cache/identity key instead of the raw
    OCR'd text, and checks whether the Chroma collection already has content
    for this document before re-embedding, so re-running the app (or hitting
    a Streamlit rerun) never duplicates chunks.
    """
    os.makedirs(persist_dir, exist_ok=True)
    touch_dir(persist_dir)

    collection_name = f"doc_{file_hash}"
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if collection.count() > 0:
        # Already embedded in a previous run/session -- just load it, no re-embedding.
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        return index

    splitter = SentenceSplitter(chunk_size=1500, chunk_overlap=300)
    documents = []
    for page_num, page_text in enumerate(text_pages, start=1):
        if not page_text:
            continue
        for chunk in splitter.split_text(page_text):
            documents.append(Document(text=chunk, metadata={"page": page_num}))

    if not documents:
        raise ValueError("No extractable text found in the uploaded document.")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


@st.cache_resource
def get_vectorstore_cached(persist_dir: str, file_hash: str, _text_pages: List[str]):
    """Thin cache wrapper. Cache key is `persist_dir`/`file_hash` (cheap to hash);
    `_text_pages` is prefixed with an underscore so Streamlit does NOT hash it
    (avoids hashing large OCR text on every rerun)."""
    return build_vectorstore(_text_pages, persist_dir, file_hash)


# -------- QUERY DOCUMENT --------
def query_document(question: str) -> str:
    try:
        retriever = st.session_state.vectorstore.as_retriever(similarity_top_k=4)
        nodes = retriever.retrieve(question)
    except Exception as e:
        return f"❌ Retrieval failed: {e}"

    if not nodes:
        return "I couldn't find anything relevant to that question in the document."

    context = "\n\n".join(
        f"(Page {n.metadata.get('page')}) {n.text}" for n in nodes
    )
    prompt = f"""
Context:
{context}

Question: {question}

Answer accurately using ONLY the context and cite page numbers.
"""
    return ask_llama(prompt)


# -------- FILE UPLOAD --------
cleanup_old_chroma_stores()

uploaded_file = st.file_uploader(
    "Upload PDF or Image", type=["pdf", "jpg", "jpeg", "png"]
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
        file_path = tmp.name

        try:
            tmp.write(file_bytes)
            tmp.close()

            with st.spinner("🔍 Processing file..."):
                if suffix == ".pdf":
                    n_pages = get_pdf_page_count(file_path)
                    if n_pages > MAX_PAGES_ALLOWED:
                        st.error(
                            f"❌ This PDF has {n_pages} pages, which exceeds the "
                            f"{MAX_PAGES_ALLOWED}-page limit. Please split it and "
                            f"upload a smaller file."
                        )
                        st.stop()

                    dpi = choose_dpi(n_pages)
                    st.info(f"📄 PDF detected ({n_pages} pages) → OCR at {dpi} DPI")
                    text_pages = surya_ocr_pages(pdf_to_images(file_path, dpi=dpi))
                else:
                    st.info("🖼 Image detected → Using Surya OCR")
                    image = Image.open(file_path).convert("RGB")
                    text_pages = surya_ocr_pages([image])

                if not any(t.strip() for t in text_pages):
                    st.error("❌ No text could be extracted from this file.")
                    st.stop()

            persist_dir = os.path.join(CHROMA_ROOT, file_hash)
            with st.spinner("📚 Building vector database..."):
                st.session_state.vectorstore = get_vectorstore_cached(
                    persist_dir, file_hash, text_pages
                )

            st.success("✅ File processed successfully!")

        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")
            st.session_state.vectorstore = None
            st.session_state.file_hash = None
        finally:
            # Always clean up the temp file, even if processing failed.
            if os.path.exists(file_path):
                os.remove(file_path)

# -------- CHAT UI --------
if st.session_state.vectorstore:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about the document...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    answer = query_document(user_input)
                except Exception as e:
                    answer = f"❌ Something went wrong while answering: {e}"
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
