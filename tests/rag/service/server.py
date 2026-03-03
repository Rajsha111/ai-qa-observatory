"""
RAG Microservice — Flask + LlamaIndex + ChromaDB + Ragas
Port: 8001

━━━ ARCHITECTURE LESSON ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The RAG pipeline has 3 phases. This service implements all 3:

  INGEST PHASE  (POST /ingest)
    Document text
        │
        ▼
    Embedding model (text-embedding-3-small)
        │  turns text → 1536-dimension vector
        ▼
    ChromaDB (vector store)
        │  stores vector + original text + metadata
        ▼
    Indexed and ready for retrieval

  RETRIEVAL PHASE  (POST /query, step 1)
    User query
        │
        ▼
    Embed the query (same embedding model — MUST match!)
        │
        ▼
    Cosine similarity search in ChromaDB
        │  finds top-K most similar vectors
        ▼
    Retrieved chunks (text + similarity score)

  GENERATION PHASE  (POST /query, step 2)
    Retrieved chunks (the "context")
        │
        ▼
    Prompt: "Answer using ONLY this context"
        │
        ▼
    Claude (LLM) generates answer
        │
        ▼
    Return: answer + retrieved chunks (both needed for testing!)

━━━ WHY RETURN THE CHUNKS? ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Most production RAG APIs only return the answer. That's fine for users.
For testing, you NEED the chunks because:
  - Faithfulness test: compare answer claims against chunk content
  - Precision test: score each chunk's relevance
  - Retrieval-unit test: verify the right doc was retrieved

This is the design decision that makes RAG testable.

━━━ ENDPOINTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  POST   /ingest      — load documents into vector store
  POST   /query       — retrieve + generate answer (returns chunks too)
  POST   /evaluate    — run Ragas faithfulness score
  DELETE /collection  — wipe the collection (used by stale-data tests)

━━━ START ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cd tests/rag/service
  pip install -r requirements.txt
  python server.py

"""

import os
import asyncio
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import anthropic
import json
import chromadb
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic as AnthropicLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

# Load .env from project root (three levels up from tests/rag/service/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env"))

app = Flask(__name__)

# ── Global error handler — return JSON instead of Flask's default HTML error page.
# Without this, any unhandled exception returns <!doctype html> which the TypeScript
# tests parse as JSON and get "Unexpected token '<'" — hiding the real error.
@app.errorhandler(Exception)
def handle_exception(e):
    from werkzeug.exceptions import HTTPException
    # Flask routing errors (404, 405, etc.) keep their original status code.
    # Only unexpected Python exceptions get wrapped as 500.
    if isinstance(e, HTTPException):
        return jsonify({"error": str(e)}), e.code
    import traceback
    tb = traceback.format_exc()
    print(f"\n[SERVER ERROR]\n{tb}", flush=True)
    return jsonify({"error": str(e), "traceback": tb}), 500


# ── CORS — allow the UI (port 3000) to call this service directly if needed.
# The Express server already proxies /rag/* here, so CORS is mainly for
# standalone use (e.g. testing the service without the Express wrapper).
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path=""):  # noqa: ARG001
    return jsonify({}), 200

# ── Configurable model names and collection (override via environment variables)
# These mirror the values in tests/rag/config/rag.config.json so the Python
# service and TypeScript tests always use the same models.
EMBED_MODEL      = os.getenv("EMBED_MODEL",       "text-embedding-3-small")
GENERATOR_MODEL  = os.getenv("GENERATOR_MODEL",   "claude-haiku-4-5-20251001")
COLLECTION_NAME  = os.getenv("CHROMA_COLLECTION", "qa_policies")

# ── Paths to config files — resolved relative to this file so the server
# works regardless of where it's launched from.
_SERVICE_DIR  = os.path.dirname(__file__)
PROMPTS_DIR   = os.path.join(_SERVICE_DIR, "../config/prompts")
CONFIG_FILE   = os.path.join(_SERVICE_DIR, "../config/rag.config.json")

# ── Generation prompt — read from disk on every /query request so that
# edits made through the UI take effect immediately without a server restart.
# Override the default path via RAG_GENERATION_PROMPT_PATH env var.
_DEFAULT_PROMPT_PATH = os.path.join(PROMPTS_DIR, "rag-generation.txt")


def _load_generation_prompt(context: str, question: str) -> str:
    """Read the prompt template from disk and fill in {{context}}/{{question}}."""
    prompt_path = os.getenv("RAG_GENERATION_PROMPT_PATH", _DEFAULT_PROMPT_PATH)
    if prompt_path and os.path.isfile(prompt_path):
        with open(prompt_path, "r") as f:
            template = f.read()
        return template.replace("{{context}}", context).replace("{{question}}", question)
    # Fallback — should only hit this if the prompts directory is missing
    return (
        f"Answer using ONLY this context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

# ── ChromaDB client ────────────────────────────────────────────────────────────
# Using in-memory client: data lives as long as the Python process runs.
# This is correct for tests — all test steps happen within one server session.
#
# For production: chromadb.PersistentClient(path="./chroma_data")
# That would survive restarts, which matters in prod but complicates test isolation.
chroma_client = chromadb.EphemeralClient()  # in-memory; use PersistentClient() for production


def get_collection():
    """Get-or-create the ChromaDB collection."""
    return chroma_client.get_or_create_collection(COLLECTION_NAME)


def configure_settings():
    """
    LlamaIndex uses a global Settings object for the embedding model and LLM.

    CRITICAL: The embedding model used at INGEST TIME must be the same as at
    QUERY TIME. If you change models between ingest and query, the vectors are
    in different spaces — similarity scores become meaningless.

    We use:
    - text-embedding-3-small: cheap (100x cheaper than large), fast, good quality
    - claude-haiku: fast and cheap for generation; haiku is sufficient for
      question-answering tasks where the answer is constrained to retrieved context
    """
    Settings.embed_model = OpenAIEmbedding(
        model=EMBED_MODEL,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    Settings.llm = AnthropicLLM(
        model=GENERATOR_MODEL,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )


# ── POST /ingest ───────────────────────────────────────────────────────────────
@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Load documents into the vector store.

    What LlamaIndex does under the hood:
    1. Splits each Document into nodes (chunks) based on chunk_size setting
    2. Calls the embedding model on each chunk → vector
    3. Stores (vector, text, metadata) in ChromaDB

    Body: {
      "documents": [
        { "id": "doc-001", "content": "...", "metadata": { "topic": "..." } }
      ]
    }

    Response: { "ingested": 5, "ids": ["doc-001", ...] }
    """
    body = request.get_json()
    docs = body.get("documents", [])

    if not docs:
        return jsonify({"error": "No documents provided"}), 400

    configure_settings()

    collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Convert to LlamaIndex Document objects
    # doc_id = the ID you assign; used later in retrieval-unit tests to identify source
    # metadata["source"] = same as doc_id; returned in query results for assertions
    llama_docs = [
        Document(
            text=doc["content"],
            doc_id=doc["id"],
            metadata={**doc.get("metadata", {}), "source": doc["id"]},
        )
        for doc in docs
    ]

    VectorStoreIndex.from_documents(
        llama_docs,
        storage_context=storage_context,
        show_progress=False,
    )

    return jsonify({"ingested": len(llama_docs), "ids": [d["id"] for d in docs]})


# ── POST /query ────────────────────────────────────────────────────────────────
@app.route("/query", methods=["POST"])
def query():
    """
    Retrieve relevant chunks and generate a grounded answer.

    The grounding instruction in the prompt ("using ONLY the context below") is
    your first line of defense against hallucination. It's a soft constraint —
    the LLM can still hallucinate if it ignores the instruction — which is
    exactly why we need the faithfulness and negative tests.

    Body: { "query": "...", "topK": 3 }

    Response: {
      "answer": "...",
      "retrievedChunks": [{ "id": "...", "content": "...", "source": "...", "score": 0.87 }],
      "latencyMs": 234
    }
    """
    start = time.time()
    body = request.get_json()
    query_text = body.get("query")
    top_k = body.get("topK", 3)

    if not query_text:
        return jsonify({"error": "query is required"}), 400

    configure_settings()

    collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # VectorStoreIndex.from_vector_store() creates an index view over
    # an existing ChromaDB collection — it does NOT re-embed anything.
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query_text)

    if not nodes:
        return jsonify({
            "answer": "I don't have information about that in the knowledge base.",
            "retrievedChunks": [],
            "latencyMs": int((time.time() - start) * 1000),
        })

    # Build context string — the "C" in RAG
    context = "\n\n---\n\n".join([node.text for node in nodes])

    # Generate answer grounded in retrieved context
    # Using direct Anthropic SDK (not LlamaIndex's query engine) so we have
    # full control over the prompt — important for testing boundary conditions.
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = _load_generation_prompt(context, query_text)

    message = client.messages.create(
        model=GENERATOR_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = message.content[0].text

    # Return chunks alongside the answer — this is what makes the pipeline testable
    retrieved_chunks = [
        {
            "id": node.node.node_id,
            "content": node.text,
            "source": node.node.metadata.get("source", "unknown"),
            "score": float(node.score) if node.score is not None else 0.0,
        }
        for node in nodes
    ]

    return jsonify({
        "answer": answer,
        "retrievedChunks": retrieved_chunks,
        "latencyMs": int((time.time() - start) * 1000),
    })


# ── POST /evaluate ─────────────────────────────────────────────────────────────
@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Run Ragas faithfulness evaluation.

    ━━━ RAGAS VS LLM-AS-JUDGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Both methods evaluate the same thing (answer grounding), but differently:

    LLM-as-Judge (in your TypeScript tests):
    - You write the prompt → full control over what "faithful" means
    - Easier to customize for your domain
    - Output varies slightly per run (temperature > 0)
    - Good for: fine-grained control, domain-specific criteria

    Ragas:
    - Standardized algorithm, comparable across teams and projects
    - Reproducible scores (uses deterministic NLI under the hood)
    - Widely cited in research; interviewers recognize it
    - Good for: benchmarking, CI quality gates, cross-team comparisons

    Use both: LLM-as-Judge for fine-grained test assertions,
    Ragas for CI thresholds and reporting.

    Note: Ragas uses OpenAI by default for its internal LLM judge.

    Body: { "question": "...", "answer": "...", "contexts": ["chunk1", "chunk2"] }
    Response: { "faithfulness": 0.92, "threshold": 0.85, "passed": true }
    """
    body = request.get_json()

    sample = SingleTurnSample(
        user_input=body["question"],
        response=body["answer"],
        retrieved_contexts=body["contexts"],
    )

    faithfulness_metric = Faithfulness()

    # Ragas metrics are async — run in a fresh event loop
    loop = asyncio.new_event_loop()
    try:
        faith_score = loop.run_until_complete(
            faithfulness_metric.single_turn_ascore(sample)
        )
    finally:
        loop.close()

    faith_score = float(faith_score) if faith_score is not None else 0.0
    threshold   = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.85"))

    return jsonify({
        "faithfulness": round(faith_score, 4),
        "threshold": threshold,
        "passed": faith_score >= threshold,
    })


# ── DELETE /collection ─────────────────────────────────────────────────────────
@app.route("/collection", methods=["DELETE"])
def clear_collection():
    """
    Wipe the ChromaDB collection and recreate it empty.

    ━━━ WHY THIS EXISTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ChromaDB does NOT automatically replace old embeddings when you re-ingest
    a document. If you ingest doc-v1 then doc-v2 with the same ID, you get BOTH
    embeddings in the store. Queries might retrieve either version — this is
    exactly the staleness bug we test for.

    In production, the correct approach is:
      collection.delete(where={"source": "doc-004"})  # delete old
      then re-ingest the updated document               # add new

    In tests, clearing the whole collection gives a clean slate for
    the stale-data test's "before" and "after" phases.
    """
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        chroma_client.get_or_create_collection(COLLECTION_NAME)
        return jsonify({"cleared": True, "collection": COLLECTION_NAME})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── GET /status ────────────────────────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def collection_status():
    """Return the collection name and current document count."""
    collection = get_collection()
    return jsonify({
        "collection": COLLECTION_NAME,
        "documentCount": collection.count(),
    })


# ── GET /prompts ────────────────────────────────────────────────────────────────
@app.route("/prompts", methods=["GET"])
def list_prompts():
    """
    Return all prompt templates as { name: content } from the prompts directory.

    This lets the UI display and edit every prompt without knowing their names
    in advance — the list is driven entirely by what .txt files exist on disk.
    """
    prompts = {}
    if os.path.isdir(PROMPTS_DIR):
        for fname in sorted(os.listdir(PROMPTS_DIR)):
            if fname.endswith(".txt"):
                name = fname[:-4]
                with open(os.path.join(PROMPTS_DIR, fname), "r") as f:
                    prompts[name] = f.read()
    return jsonify(prompts)


# ── POST /prompts/<name> ───────────────────────────────────────────────────────
@app.route("/prompts/<name>", methods=["POST"])
def save_prompt(name):
    """
    Save updated prompt content to <name>.txt.

    Because _load_generation_prompt() reads from disk on every /query request,
    changes take effect immediately — no server restart needed.
    """
    fpath = os.path.join(PROMPTS_DIR, f"{name}.txt")
    if not os.path.isfile(fpath):
        return jsonify({"error": f"Prompt '{name}' not found"}), 404
    body = request.get_json()
    content = body.get("content", "")
    with open(fpath, "w") as f:
        f.write(content)
    return jsonify({"saved": True, "name": name})


# ── GET /config ────────────────────────────────────────────────────────────────
@app.route("/config", methods=["GET"])
def get_config():
    """Return the current rag.config.json as parsed JSON."""
    with open(CONFIG_FILE, "r") as f:
        return jsonify(json.load(f))


# ── POST /config ───────────────────────────────────────────────────────────────
@app.route("/config", methods=["POST"])
def save_config():
    """
    Persist an updated rag.config.json.

    The TypeScript tests call loadRagConfig() synchronously on startup, so
    config changes only affect test runs started after the save. The Python
    service itself reads config at request time where relevant (e.g., FAITHFULNESS_THRESHOLD),
    so those values update on the next request.
    """
    body = request.get_json()
    with open(CONFIG_FILE, "w") as f:
        json.dump(body, f, indent=2)
    return jsonify({"saved": True})


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  RAG Microservice — http://localhost:8001")
    print("  Endpoints:")
    print("    POST   /ingest      — load documents")
    print("    POST   /query       — retrieve + generate")
    print("    POST   /evaluate    — Ragas faithfulness score")
    print("    DELETE /collection  — wipe for stale-data tests")
    print("═" * 60 + "\n")
    app.run(port=8001, debug=False)
