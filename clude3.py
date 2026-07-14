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
import pdfplumber  # native text/table extraction for born-digital PDFs

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

OCR_DPI_DEFAULT = 300          # dense small-font tables need more resolution than plain prose
OCR_DPI_MAX_PAGES_THRESHOLD = 30   # drop dpi further for very large documents
OCR_DPI_LOW = 220
MAX_PAGES_ALLOWED = 150        # hard guard against runaway uploads

# Pages are only sent to OCR if native text extraction yields too little text
# (i.e. the page is an actual scan) OR the extracted text looks like mojibake
# (garbled non-Unicode font mapping -- common in older Indian government PDFs
# with custom Hindi fonts). Otherwise native extraction is used: it's faster
# and far more accurate for born-digital tables/numbers than OCR.
MIN_NATIVE_CHARS_PER_PAGE = 40
DEVANAGARI_RANGE = (0x0900, 0x097F)
MOJIBAKE_RATIO_THRESHOLD = 0.05

OCR_LANGS = ["en", "hi"]  # this document is bilingual English/Hindi

# GPU tuning -- lets you pin a specific GPU on multi-GPU boxes and batch
# OCR calls to actually use GPU parallelism instead of one image at a time.
GPU_DEVICE_INDEX = int(os.getenv("GPU_DEVICE_INDEX", "0"))
OCR_BATCH_SIZE_GPU = int(os.getenv("OCR_BATCH_SIZE", "4"))
OCR_BATCH_SIZE_CPU = 1
EMBED_BATCH_SIZE_GPU = int(os.getenv("EMBED_BATCH_SIZE", "64"))
EMBED_BATCH_SIZE_CPU = 8

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
    if torch.cuda.is_available():
        if GPU_DEVICE_INDEX >= torch.cuda.device_count():
            st.sidebar.warning(
                f"⚠️ GPU_DEVICE_INDEX={GPU_DEVICE_INDEX} is out of range "
                f"({torch.cuda.device_count()} GPU(s) visible); using cuda:0."
            )
            return "cuda:0"
        return f"cuda:{GPU_DEVICE_INDEX}"
    return "cpu"


DEVICE = get_device()
USE_GPU = DEVICE.startswith("cuda")

if USE_GPU:
    # cudnn.benchmark picks the fastest conv algorithms for the actual input
    # sizes seen at runtime (worth it here since page images are a fairly
    # consistent size); TF32 matmul is a solid throughput win on Ampere+ GPUs
    # with negligible precision cost for this workload.
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    OCR_BATCH_SIZE = OCR_BATCH_SIZE_GPU
    EMBED_BATCH_SIZE = EMBED_BATCH_SIZE_GPU
else:
    OCR_BATCH_SIZE = OCR_BATCH_SIZE_CPU
    EMBED_BATCH_SIZE = EMBED_BATCH_SIZE_CPU

st.sidebar.write(f"🖥 Device: **{DEVICE}**")
if USE_GPU:
    gpu_name = torch.cuda.get_device_name(int(DEVICE.split(":")[1]))
    st.sidebar.write(f"🎮 GPU: **{gpu_name}**")
    st.sidebar.write(f"📦 OCR batch size: **{OCR_BATCH_SIZE}**, Embed batch size: **{EMBED_BATCH_SIZE}**")
else:
    st.sidebar.warning("⚠️ No GPU detected -- running on CPU will be significantly slower.")


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
        embed_batch_size=EMBED_BATCH_SIZE,
    )


embed_model = load_embedding_model()


# -------- TEXT CLEANING --------
def clean_text(text: str):
    text = text.replace("\x0c", " ")
    text = text.replace("\ufeff", " ")
    text = text.encode("utf-8", "ignore").decode("utf-8")
    return " ".join(text.split())


def clean_text_preserve_rows(text: str):
    """Like clean_text, but keeps line breaks intact -- used for OCR output
    where newlines encode reconstructed table rows (see
    reconstruct_rows_from_ocr). Collapsing everything to spaces would erase
    the row structure right after building it."""
    text = text.replace("\x0c", " ")
    text = text.replace("\ufeff", " ")
    text = text.encode("utf-8", "ignore").decode("utf-8")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


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


def _line_bbox(line) -> tuple:
    """Best-effort extraction of a text line's (x0, y0) position from a
    Surya OCR result. Different surya versions expose this as `.bbox`
    (x0,y0,x1,y1) or `.polygon` (list of [x,y] points); this tries both
    and falls back to (0, 0) -- which just disables row reconstruction
    for that line rather than crashing."""
    bbox = getattr(line, "bbox", None)
    if bbox and len(bbox) >= 2:
        return (bbox[0], bbox[1])
    polygon = getattr(line, "polygon", None)
    if polygon:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs), min(ys))
    return (0, 0)


def reconstruct_rows_from_ocr(text_lines, y_tolerance: int = 12) -> str:
    """Approximates table structure from raw OCR line detections by
    grouping lines into rows based on vertical (y) position, then
    ordering left-to-right within each row.

    This matters a lot for documents like this one: dense multi-column
    financial tables where naively space-joining every detected line
    (the original behaviour) scrambles which number belongs to which
    row/column. This is a heuristic, not real table detection (Surya's
    dedicated table-recognition model would be more robust), but it's
    far better than flattening everything into one blob.
    """
    items = []
    for line in text_lines:
        x0, y0 = _line_bbox(line)
        text = getattr(line, "text", "") or ""
        if text.strip():
            items.append((y0, x0, text.strip()))

    if not items:
        return ""

    items.sort(key=lambda t: (t[0], t[1]))
    rows = [[items[0]]]
    for item in items[1:]:
        if abs(item[0] - rows[-1][-1][0]) <= y_tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])

    lines_out = []
    for row in rows:
        row_sorted = sorted(row, key=lambda t: t[1])
        lines_out.append(" | ".join(t[2] for t in row_sorted))
    return "\n".join(lines_out)


# -------- SURYA OCR (GPU-batched for throughput) --------
def surya_ocr_pages(images: List[Image.Image]) -> List[str]:
    """Fallback OCR path -- used only for pages that don't have a usable
    native text layer (i.e. actual scans, or pages whose font encoding
    makes native extraction unreliable). Passes explicit language hints
    since this app supports bilingual English/Hindi/Arabic documents, and
    reconstructs row/column layout from line positions instead of just
    space-joining every detected line (see reconstruct_rows_from_ocr).

    Images are processed in batches (OCR_BATCH_SIZE, GPU-only) rather than
    one at a time, so the GPU actually gets parallel work instead of
    running a single-image forward pass per page. torch.inference_mode()
    disables autograd bookkeeping we don't need, saving GPU memory and
    time; cache is cleared after each document to avoid fragmentation
    across repeated uploads in a long-running session.
    """
    texts: List[str] = []
    if not images:
        return texts

    with torch.inference_mode():
        for start in range(0, len(images), OCR_BATCH_SIZE):
            batch = images[start:start + OCR_BATCH_SIZE]
            try:
                recognitions = predictors["recognition"](
                    images=batch,
                    task_names=["ocr_with_boxes"] * len(batch),
                    det_predictor=predictors["detection"],
                    langs=[OCR_LANGS] * len(batch),
                )
                for rec in recognitions:
                    page_text = reconstruct_rows_from_ocr(rec.text_lines)
                    texts.append(clean_text_preserve_rows(page_text))
            except Exception as e:
                st.warning(f"⚠️ OCR failed on a batch of {len(batch)} page(s), skipping: {e}")
                texts.extend([""] * len(batch))

    if USE_GPU:
        torch.cuda.empty_cache()

    return texts


# -------- NATIVE TEXT/TABLE EXTRACTION (for born-digital PDFs) --------
def looks_like_mojibake(text: str, threshold: float = MOJIBAKE_RATIO_THRESHOLD) -> bool:
    """Detects text extracted through a broken font-encoding mapping --
    common in Indian government PDFs that embed a custom (non-Unicode)
    Hindi font. Symptom: lots of non-ASCII characters that fall outside the
    Devanagari Unicode block (e.g. stray Latin-Extended glyphs like
    'ÉÊE', 'ºÉ®BÉE') rather than real Hindi script."""
    if not text:
        return False
    non_ascii = 0
    suspicious = 0
    for ch in text:
        code = ord(ch)
        if code > 127:
            non_ascii += 1
            if not (DEVANAGARI_RANGE[0] <= code <= DEVANAGARI_RANGE[1]):
                suspicious += 1
    if non_ascii == 0:
        return False
    return (suspicious / non_ascii) > threshold


def chunk_rows(text: str, rows_per_chunk: int = 25) -> List[str]:
    """Chunks reconstructed OCR row-text (see reconstruct_rows_from_ocr) by
    grouping N rows per chunk, instead of running it through a
    sentence-based splitter that has no concept of table rows and could
    cut a chunk in the middle of one."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []
    return ["\n".join(lines[i:i + rows_per_chunk]) for i in range(0, len(lines), rows_per_chunk)]


def table_to_markdown(table: List[List[str]]) -> str:
    """Render an extracted table as a markdown table, keeping row/column
    relationships intact instead of flattening cells into prose."""
    if not table or len(table) < 1:
        return ""
    rows = [[("" if c is None else str(c).strip()) for c in row] for row in table]
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_native_page_content(pdf_path: str):
    """Extracts each page's prose text and tables directly from the PDF's
    text layer, using pdfplumber. Tables are kept as markdown blocks (not
    flattened) so row/column alignment survives into chunking. Returns a
    list of dicts: {page, text_ok, prose, tables: [markdown, ...]}.

    `text_ok` is False for pages that need to fall back to OCR (too little
    native text, or text that looks like a mojibake font-mapping issue)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                prose = page.extract_text() or ""
                tables = page.extract_tables() or []
            except Exception:
                prose, tables = "", []

            md_tables = [table_to_markdown(t) for t in tables if t]
            md_tables = [t for t in md_tables if t]

            has_enough_text = len(prose.strip()) >= MIN_NATIVE_CHARS_PER_PAGE or md_tables
            mojibake = looks_like_mojibake(prose)

            pages.append({
                "page": i,
                "text_ok": has_enough_text and not mojibake,
                "prose": clean_text(prose),
                "tables": md_tables,
            })
    return pages


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


# -------- VECTOR DATABASE (dedup-aware, table-aware chunking) --------
def build_vectorstore(pages: List[dict], persist_dir: str, file_hash: str):
    """Builds (or loads) the vector index for this document.

    Uses `file_hash` as the effective cache/identity key instead of the raw
    extracted text, and checks whether the Chroma collection already has
    content for this document before re-embedding, so re-running the app
    (or hitting a Streamlit rerun) never duplicates chunks.

    Table-aware chunking: each extracted table is kept as its own document
    (never split mid-row), and prose is chunked separately with the
    sentence splitter. This preserves row/column relationships in tabular
    documents like financial statements, instead of flattening everything
    into one blob of text that gets cut at an arbitrary character offset.
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
    for entry in pages:
        page_num = entry["page"]

        # Tables become their own chunks, kept intact (up to a size cap --
        # a single table rarely needs splitting, but very long ones still
        # go through the splitter as a unit-preserving fallback).
        for table_md in entry.get("tables", []):
            if len(table_md) <= 1800:
                documents.append(Document(
                    text=table_md,
                    metadata={"page": page_num, "content_type": "table"},
                ))
            else:
                for chunk in splitter.split_text(table_md):
                    documents.append(Document(
                        text=chunk,
                        metadata={"page": page_num, "content_type": "table"},
                    ))

        prose = entry.get("prose", "")
        if prose:
            if entry.get("ocr_reconstructed"):
                # Row-reconstructed OCR text -- chunk by table rows, not
                # sentences, so a chunk never splits a row in half.
                for chunk in chunk_rows(prose):
                    documents.append(Document(
                        text=chunk,
                        metadata={"page": page_num, "content_type": "table"},
                    ))
            else:
                for chunk in splitter.split_text(prose):
                    documents.append(Document(
                        text=chunk,
                        metadata={"page": page_num, "content_type": "prose"},
                    ))

    if not documents:
        raise ValueError("No extractable text found in the uploaded document.")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


@st.cache_resource
def get_vectorstore_cached(persist_dir: str, file_hash: str, _pages: List[dict]):
    """Thin cache wrapper. Cache key is `persist_dir`/`file_hash` (cheap to hash);
    `_pages` is prefixed with an underscore so Streamlit does NOT hash it
    (avoids hashing large extracted content on every rerun)."""
    return build_vectorstore(_pages, persist_dir, file_hash)


# -------- QUERY DOCUMENT --------
def query_document(question: str) -> str:
    try:
        retriever = st.session_state.vectorstore.as_retriever(similarity_top_k=6)
        nodes = retriever.retrieve(question)
    except Exception as e:
        return f"❌ Retrieval failed: {e}"

    if not nodes:
        return "I couldn't find anything relevant to that question in the document."

    context_blocks = []
    for n in nodes:
        page = n.metadata.get("page")
        kind = n.metadata.get("content_type", "text")
        label = f"(Page {page}, table)" if kind == "table" else f"(Page {page})"
        context_blocks.append(f"{label}\n{n.text}")
    context = "\n\n".join(context_blocks)

    prompt = f"""
Context (tables are given in markdown -- read column headers carefully,
values in different columns often represent different years/estimate types
such as Actuals, Budget Estimates, or Revised Estimates):
{context}

Question: {question}

Answer accurately using ONLY the context above. When citing a number from a
table, name the exact column/row it came from, not just the page number.
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

                    # Step 1: try native text/table extraction first -- fast
                    # and far more accurate for born-digital PDFs (preserves
                    # table structure, no risk of OCR digit misreads).
                    st.info(f"📄 PDF detected ({n_pages} pages) → reading native text/tables")
                    native_pages = extract_native_page_content(file_path)

                    ocr_page_indices = [p["page"] for p in native_pages if not p["text_ok"]]
                    pages = native_pages

                    # Step 2: OCR fallback, but ONLY for pages that didn't
                    # yield usable native text (real scans) or that look
                    # like a broken font-encoding mapping (mojibake).
                    if ocr_page_indices:
                        dpi = choose_dpi(n_pages)
                        st.info(
                            f"🔎 {len(ocr_page_indices)} page(s) need OCR "
                            f"(scanned or garbled text layer) at {dpi} DPI, "
                            f"batched {OCR_BATCH_SIZE} at a time"
                        )
                        # Render all flagged pages to images first, then send
                        # them through surya_ocr_pages as one call so the GPU
                        # gets OCR_BATCH_SIZE pages of parallel work per
                        # forward pass instead of one page at a time.
                        doc = pypdfium2.PdfDocument(file_path)
                        try:
                            fallback_images = []
                            for page_num in ocr_page_indices:
                                idx = page_num - 1
                                renderer = doc.render(
                                    pypdfium2.PdfBitmap.to_pil,
                                    page_indices=[idx],
                                    scale=dpi / 72,
                                )
                                img = list(renderer)[0].convert("RGB")
                                img_np = np.array(img)
                                gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                                # Note: no fastNlMeansDenoising here -- it can
                                # blur thin digit strokes in small dense-table
                                # fonts, increasing the risk of digit misreads
                                # in financial figures.
                                fallback_images.append(Image.fromarray(gray))
                        finally:
                            doc.close()

                        ocr_results = surya_ocr_pages(fallback_images)

                        for page_num, ocr_text in zip(ocr_page_indices, ocr_results):
                            # OCR result replaces the unusable native prose;
                            # any native tables already extracted for this
                            # page (rare when text_ok is False) are kept.
                            for p in pages:
                                if p["page"] == page_num:
                                    p["prose"] = ocr_text
                                    p["ocr_reconstructed"] = True
                else:
                    st.info("🖼 Image detected → Using Surya OCR")
                    image = Image.open(file_path).convert("RGB")
                    ocr_text = surya_ocr_pages([image])[0]
                    pages = [{
                        "page": 1,
                        "prose": ocr_text,
                        "tables": [],
                        "ocr_reconstructed": True,
                    }]

                if not any(p.get("prose") or p.get("tables") for p in pages):
                    st.error("❌ No text could be extracted from this file.")
                    st.stop()

            persist_dir = os.path.join(CHROMA_ROOT, file_hash)
            with st.spinner("📚 Building vector database..."):
                st.session_state.vectorstore = get_vectorstore_cached(
                    persist_dir, file_hash, pages
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
