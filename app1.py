import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import time
import pickle
import hashlib
import re
from datetime import datetime

import numpy as np
import torch
import faiss
import ollama
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
app.secret_key = "change-this-secret"
CORS(app)

# -----------------------------
# CONFIG (FAST MODE)
# -----------------------------
DATA_FOLDER = "data"
CACHE_FOLDER = "cache"
MODEL_NAME = "llama3"

TOP_K = 5
MAX_CONTEXT_DOCS = 3

USE_SEMANTIC_CACHE = True
USE_EMBEDDING_CACHE = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {DEVICE}")

os.makedirs(CACHE_FOLDER, exist_ok=True)

# -----------------------------
# GLOBALS
# -----------------------------
documents = []
doc_embeddings = None
faiss_index = None
embedder = None

embedding_cache = {}
semantic_cache = {}

# -----------------------------
# UTILS
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.,;:()\- ]", "", text)
    return text.strip()

# -----------------------------
# EMBEDDING CACHE
# -----------------------------
def get_embedding(text: str):
    key = hashlib.md5(text.encode()).hexdigest()
    if USE_EMBEDDING_CACHE and key in embedding_cache:
        return embedding_cache[key]

    with torch.no_grad():
        emb = embedder.encode(
            [text],
            convert_to_tensor=True,
            device=DEVICE,
            normalize_embeddings=True
        )

    if USE_EMBEDDING_CACHE:
        embedding_cache[key] = emb

    return emb

# -----------------------------
# SEMANTIC CACHE
# -----------------------------
def semantic_lookup(query_emb):
    for data in semantic_cache.values():
        sim = np.dot(query_emb, data["embedding"])
        if sim > 0.95:
            return data["answer"], data["sources"]
    return None, None

def semantic_store(query_emb, answer, sources):
    key = hashlib.md5(query_emb.tobytes()).hexdigest()
    semantic_cache[key] = {
        "embedding": query_emb,
        "answer": answer,
        "sources": sources
    }

# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
def load_documents():
    docs = []
    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)

        if file.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    q = item.get("question", "")
                    a = item.get("answer", "")
                    if q and a:
                        text = clean_text(f"Q: {q} A: {a}")
                        docs.append({
                            "content": text,
                            "source": f"{file} ({i+1})"
                        })

        elif file.endswith(".csv"):
            df = pd.read_csv(path)
            for i, row in df.iterrows():
                row_text = clean_text(" ".join(map(str, row.values)))
                docs.append({
                    "content": row_text,
                    "source": f"{file} ({i+1})"
                })

    return docs

# -----------------------------
# INIT SYSTEM
# -----------------------------
def initialize():
    global documents, doc_embeddings, faiss_index, embedder

    print("🧠 Loading embedding model...")
    embedder = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device=DEVICE
    )
    embedder.eval()

    print("📄 Loading documents...")
    documents = load_documents()
    texts = [d["content"] for d in documents]

    print("🔢 Creating embeddings...")
    with torch.no_grad():
        doc_embeddings = embedder.encode(
            texts,
            batch_size=64,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=DEVICE
        )

    embeddings_np = doc_embeddings.cpu().numpy().astype("float32")
    dim = embeddings_np.shape[1]

    print("⚡ Building FAISS index...")
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings_np)

    print(f"✅ Ready with {len(documents)} documents")

# -----------------------------
# SEARCH
# -----------------------------
def search(query: str):
    q_emb = get_embedding(query)
    q_np = q_emb.cpu().numpy().astype("float32")

    # semantic cache
    if USE_SEMANTIC_CACHE:
        cached_ans, cached_src = semantic_lookup(q_np[0])
        if cached_ans:
            return cached_ans, cached_src

    scores, idxs = faiss_index.search(q_np, TOP_K)

    contexts = []
    sources = []

    for i, idx in enumerate(idxs[0][:MAX_CONTEXT_DOCS]):
        contexts.append(documents[idx]["content"])
        sources.append({
            "source": documents[idx]["source"],
            "score": float(scores[0][i])
        })

    prompt = f"""
Answer ONLY using the information below.

INFO:
{chr(10).join(contexts)}

Q: {query}
A:
"""

    start = time.time()
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": 2048,
            "num_predict": 256,
            "temperature": 0.1,
            "num_thread": 8
        }
    )
    print(f"⚡ LLM time: {time.time() - start:.3f}s")

    answer = response["message"]["content"]

    if USE_SEMANTIC_CACHE:
        semantic_store(q_np[0], answer, sources)

    return answer, sources

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"reply": "Empty query", "sources": []})

    answer, sources = search(query)
    return jsonify({
        "reply": answer.replace("\n", "<br>"),
        "sources": sources
    })

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    initialize()
    app.run(host="0.0.0.0", port=5000, debug=False)
