import asyncio
import time
from pathlib import Path
import os
import requests

import streamlit as st
import inngest
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="RAG PDF Analyzer",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG PDF Analyzer")
st.write(
    "Upload PDFs, ingest them into your local RAG system, "
    "and ask questions about your documents."
)

# -------------------------------
# Async Helpers for Streamlit
# -------------------------------
def run_async(func, *args, **kwargs):
    """
    Safely run an async function in Streamlit for Python 3.12+
    in threads where there is no event loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(func(*args, **kwargs))
    finally:
        loop.close()

# -------------------------------
# Inngest Client
# -------------------------------
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    """
    Creates and caches the Inngest client.
    Reads APP_ID and ENV from environment variables.
    """
    app_id = os.getenv("INNGEST_APP_ID", "rag_app")
    return inngest.Inngest(app_id=app_id, is_production=False)

# -------------------------------
# PDF Upload Helper
# -------------------------------
def save_uploaded_pdf(file) -> Path:
    """
    Saves uploaded PDF to the 'uploads' directory and returns its path.
    """
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path

# -------------------------------
# Send RAG Ingest Event
# -------------------------------
async def send_rag_ingest_event(pdf_path: Path) -> str:
    """
    Sends an event to Inngest to ingest a PDF.
    Returns the event ID for polling.
    """
    client = get_inngest_client()
    event = inngest.Event(
        name="rag/ingest_pdf",
        data={
            "pdf_path": str(pdf_path.resolve()),
            "source_id": pdf_path.name
        },
    )
    event_id = await client.send(event)
    return event_id

# -------------------------------
# Send RAG Query Event
# -------------------------------
async def send_rag_query_event(question: str, top_k: int = 5) -> str:
    """
    Sends an event to Inngest to query PDFs.
    Returns the event ID for polling.
    """
    client = get_inngest_client()
    event = inngest.Event(
        name="rag/query_pdf_ai",
        data={
            "question": question,
            "top_k": top_k
        }
    )
    event_id = await client.send(event)
    return event_id

# -------------------------------
# Polling Helpers
# -------------------------------
def _inngest_api_base() -> str:
    """
    Base URL for local Inngest dev server.
    Configurable via environment variable INNGEST_API_BASE.
    """
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

def fetch_runs(event_id: str) -> list[dict]:
    """
    Fetches runs for a given Inngest event ID.
    """
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("data", [])

def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    """
    Polls Inngest for the run's output until it is complete or failed.
    """
    start = time.time()
    last_status = None

    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)

# -------------------------------
# PDF Upload Section
# -------------------------------
st.header("1️⃣ Upload a PDF for Ingestion")
uploaded_file = st.file_uploader(
    "Choose a PDF file", type=["pdf"], accept_multiple_files=False
)

if uploaded_file is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        pdf_path = save_uploaded_pdf(uploaded_file)
        event_id = run_async(send_rag_ingest_event, pdf_path)
        # Small pause for UX
        time.sleep(0.3)
    st.success(f"Ingestion triggered for: {pdf_path.name}")
    st.caption("You can upload another PDF if needed.")

st.divider()

# -------------------------------
# PDF Query Section
# -------------------------------
st.header("2️⃣ Ask a question about your PDFs")

with st.form("query_form"):
    question = st.text_input("Enter your question:")
    top_k = st.number_input("How many chunks to retrieve?", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Querying and generating answer..."):
            event_id = run_async(send_rag_query_event, question.strip(), int(top_k))
            output = wait_for_run_output(event_id)

            answer = output.get("answer", "(No answer)")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.caption("Sources used:")
            for src in sources:
                st.write(f"- {src}")