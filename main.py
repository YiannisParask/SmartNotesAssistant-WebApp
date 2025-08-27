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
from typing import cast
from flask import Flask, render_template, request, abort
import torch

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


# ---------------------------------------------------------------------------
# Flask app factory (simple singleton pattern here)
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)


# ---------------------------------------------------------------------------
# RAG initialization (performed at import; heavy models could optionally be
# lazy-loaded on first request to reduce cold-start time).
# ---------------------------------------------------------------------------
rag = RagSearch(milvus_uri=MILVUS_URI, device=DEVICE, collection_name=COLLECTION_NAME)
embeddings = rag.get_embeddings_model(EMBED_MODEL)
retriever = rag.get_retriever()
prompt = rag.build_prompt_template()
llm = rag.get_hg_llm(MODEL_NAME)
qa_chain = rag.get_qa_chain(retriever, prompt)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def run_rag_query(message: str) -> str:
    """Execute a RAG query and return the answer text."""
    answer = rag.perform_rag_query(qa_chain, message)
    return answer


def build_milvus_collection(data_path: str) -> int:
    """Ingest PDFs from data_path and (re)build Milvus collection.

    Returns the number of chunks inserted.
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
    return len(chunks)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index() -> str:
    """Render the main chat page."""
    return render_template("index.html")


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
