"""Flask application entry-point for SmartDocsAssistant.
Run with:
    python main.py
or (production example):
    gunicorn -w 1 -b 0.0.0.0:8080 main:app
"""

from __future__ import annotations
import os
from threading import Lock
from typing import cast
from flask_bootstrap import Bootstrap5
import torch
from flask import Flask, render_template, request, abort
from src.perform_rag import RagSearch


# ---------------------------------------------------------------------------
# Configuration & mutable runtime state
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
# Device / models (can still be overridden by env)
DEVICE: str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")

# Remote Milvus connection (required). Start empty unless env provides.
current_milvus_uri: str | None = os.environ.get("MILVUS_URI") or None
current_collection_name: str | None = os.environ.get("MILVUS_COLLECTION") or None


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512MB limit for archives
bootstrap = Bootstrap5(app)

# ---------------------------------------------------------------------------
# Lazy RAG pipeline globals
# ---------------------------------------------------------------------------
rag_lock: Lock = Lock()
rag: RagSearch | None = None
embeddings = None
retriever = None
prompt = None
llm = None
qa_chain = None
NOT_READY_MESSAGE: str = (
    "Milvus not configured or collection empty. Provide the Milvus URI and Collection above. Ensure the collection already contains embeddings."
)


def _milvus_configured() -> bool:
    """Return True if user has supplied remote Milvus URI & collection name."""
    return bool(current_milvus_uri and current_collection_name)


def build_rag_pipeline(force: bool = False) -> bool:
    """Build or rebuild the RAG pipeline lazily if Milvus config present.

    Avoids loading models or opening Milvus until data exists & config supplied.
    """
    global rag, embeddings, retriever, prompt, llm, qa_chain  # noqa: PLW0603

    if not _milvus_configured():
        return False

    if not force and qa_chain is not None:
        return True

    try:
        assert current_milvus_uri is not None and current_collection_name is not None
        uri: str = current_milvus_uri  # cast after assert
        coll: str = current_collection_name
        local_rag = RagSearch(
            milvus_uri=uri,
            device=DEVICE,
            collection_name=coll,
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
    """Execute a RAG query or return guidance if not ready."""
    ready = build_rag_pipeline()
    if not ready or qa_chain is None or rag is None:
        return NOT_READY_MESSAGE
    return cast(str, rag.perform_rag_query(qa_chain, message))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index() -> str:
    """Render main chat page with flags for configuration & readiness."""
    db_ready = build_rag_pipeline()  # Attempt lazy init if config present
    milvus_configured = _milvus_configured()
    return render_template(
        "index.html",
        db_ready=db_ready,
        milvus_configured=milvus_configured,
        milvus_uri=current_milvus_uri or "",
        collection_name=current_collection_name or "",
    )


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


## /create_db endpoint removed: application expects existing remote Milvus collection.
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
# Milvus remote configuration route
# ---------------------------------------------------------------------------
@app.route("/configure_milvus", methods=["POST"])
def configure_milvus() -> str:
    """Store Milvus URI & collection name; attempt immediate pipeline init."""
    uri = (request.form.get("milvus_uri") or "").strip()
    collection = (request.form.get("collection_name") or "").strip()

    if not uri or not collection:
        return render_template(
            "milvus_config_fragment.html",
            status="error",
            error="Both URI and Collection are required.",
            milvus_uri=uri,
            collection_name=collection,
        )

    # Basic sanity check for remote URI (Milvus stand-alone typical: http://host:19530)
    if not (uri.startswith("http://") or uri.startswith("https://")):
        return render_template(
            "milvus_config_fragment.html",
            status="error",
            error="URI should start with http:// or https://",
            milvus_uri=uri,
            collection_name=collection,
        )

    global current_milvus_uri, current_collection_name, rag, embeddings, retriever, prompt, llm, qa_chain

    # Reset pipeline state so lazy builder re-initializes with new settings.
    current_milvus_uri = uri
    current_collection_name = collection
    rag = None
    embeddings = retriever = prompt = llm = qa_chain = None

    ready = build_rag_pipeline(force=True)
    return render_template(
        "milvus_config_fragment.html",
        status="success" if ready else "pending",
        milvus_uri=uri,
        collection_name=collection,
        ready=ready,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
