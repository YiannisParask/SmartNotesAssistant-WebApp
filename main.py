"""Flask application entry-point for SmartDocsAssistant.

This refactors the previous FastAPI implementation to Flask while keeping:
 - RAG pipeline initialization (Milvus + HuggingFace models)
 - htmx-driven endpoints for chat (/send) and dataset ingestion (/create_db)
 - Jinja2 templates & static file serving

Run with:
    python main.py
or (production example):
    gunicorn -w 1 -b 0.0.0.0:8080 main:app
"""

from __future__ import annotations

import os
from threading import Lock
from typing import cast

import torch
from flask import Flask, render_template, request, abort

from src.perform_rag import RagSearch
from src.load_dataset import LoadAndVectorizeData


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
MILVUS_URI: str = os.environ.get("MILVUS_URI", os.path.join(BASE_DIR, "milvus_demo.db"))
COLLECTION_NAME: str = os.environ.get("MILVUS_COLLECTION", "MilvusDocs")
DEVICE: str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EMBED_MODEL: str = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"
)
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
#/home/yiannisparask/Projects/Personal-Cheat-Sheets

# ---------------------------------------------------------------------------
# Flask app factory (simple singleton pattern here)
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)


# ---------------------------------------------------------------------------
# Lazy RAG pipeline globals (initialized only after DB creation or first need)
# ---------------------------------------------------------------------------
rag_lock: Lock = Lock()
rag: RagSearch | None = None
embeddings = None
retriever = None
prompt = None
llm = None
qa_chain = None
NOT_READY_MESSAGE: str = (
    "Vector database not initialized. Open settings (⚙️) and create it by supplying a markdown folder path."
)


def build_rag_pipeline(force: bool = False) -> bool:
    """Build or rebuild the RAG pipeline lazily.

    This avoids loading models or opening Milvus until data has been ingested.

    Args:
        force (bool): Force rebuild even if already initialized.

    Returns:
        bool: True if pipeline ready; False otherwise.
    """
    global rag, embeddings, retriever, prompt, llm, qa_chain  # noqa: PLW0603

    if not force and qa_chain is not None:
        return True

    # If using local Milvus Lite file path, skip initialization until file exists.
    if not (MILVUS_URI.startswith("http://") or MILVUS_URI.startswith("https://")):
        if not os.path.exists(MILVUS_URI):  # DB file not yet created via ingestion
            return False

    try:
        local_rag = RagSearch(
            milvus_uri=MILVUS_URI, device=DEVICE, collection_name=COLLECTION_NAME
        )
        local_embeddings = local_rag.get_embeddings_model(EMBED_MODEL)
        local_retriever = local_rag.get_retriever()
        local_prompt = local_rag.build_prompt_template()
        local_llm = local_rag.get_hg_llm(MODEL_NAME)
        local_qa_chain = local_rag.get_qa_chain(local_retriever, local_prompt)

        rag = local_rag
        embeddings = local_embeddings
        retriever = local_retriever
        prompt = local_prompt
        llm = local_llm
        qa_chain = local_qa_chain
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[RAG INIT] Delayed (will retry later): {exc}")
        return False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def run_rag_query(message: str) -> str:
    """Execute a RAG query.

    If vector DB not yet initialized, return guidance instead of raising.
    """
    ready = build_rag_pipeline()
    if not ready or qa_chain is None or rag is None:
        return NOT_READY_MESSAGE
    return cast(str, rag.perform_rag_query(qa_chain, message))


def build_milvus_collection(data_path: str) -> int:
    """Ingest markdown documents from a directory and create/update Milvus collection.

    After ingestion, force a rebuild of the pipeline so queries work immediately.

    Args:
        data_path (str): Directory containing markdown files.

    Returns:
        int: Number of chunks inserted.
    """
    loader = LoadAndVectorizeData(
        data_path=data_path,
        collection_name=COLLECTION_NAME,
        device=DEVICE,
        milvus_uri=MILVUS_URI,
    )
    docs = loader.load_md_data()
    chunks = loader.split_docs(docs)
    embeddings_model = loader.get_embeddings_model(EMBED_MODEL)
    loader.save_to_milvus(chunks, embeddings_model)

    # Force rebuild of RAG pipeline now that vectors exist.
    with rag_lock:
        build_rag_pipeline(force=True)
    return len(chunks)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index() -> str:
    """Render the main chat page with readiness flag."""
    db_ready = build_rag_pipeline()  # Attempt lazy init (no-op if not created yet)
    return render_template("index.html", db_ready=db_ready)


@app.route("/send", methods=["POST"])
def send() -> str:
    """Handle chat message via htmx, returning an HTML fragment."""
    raw_message = request.form.get("message")
    if not raw_message:
        abort(400, "Missing message")
    message = cast(str, raw_message)

    # For now run inline (blocking). For better responsiveness you could
    # offload to a thread pool and stream partial output (SSE) later.
    answer = run_rag_query(message)
    return render_template("message_fragment.html", user=message, server=answer)


@app.route("/create_db", methods=["POST"])
def create_db() -> str:
    """Ingest PDFs and rebuild Milvus collection. Returns status fragment.

    Heavy work runs in a background thread; we immediately return a 'started'
    state and rely on htmx polling (future enhancement) or a second click.
    For simplicity here we block until completion (synchronous).
    """
    raw_path = request.form.get("data_path")
    if not raw_path:
        abort(400, "Missing data_path")
    data_path = cast(str, raw_path)

    # Basic server-side validation
    if not os.path.isdir(data_path):
        return render_template(
            "create_db_fragment.html",
            status="error",
            error=f"Path not found: {data_path}",
            data_path=data_path,
            milvus_uri=MILVUS_URI,
        )

    try:
        count = build_milvus_collection(data_path)
        return render_template(
            "create_db_fragment.html",
            status="success",
            count=count,
            data_path=data_path,
            milvus_uri=MILVUS_URI,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return render_template(
            "create_db_fragment.html",
            status="error",
            error=str(exc),
            data_path=data_path,
            milvus_uri=MILVUS_URI,
        )


# ---------------------------------------------------------------------------
# Graceful teardown / GPU memory cleanup hook (Flask variant)
# ---------------------------------------------------------------------------
@app.teardown_appcontext
def _teardown(_exception):  # type: ignore[unused-argument]
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
