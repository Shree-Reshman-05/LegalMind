import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import numpy as np
import json
from functools import lru_cache
import hashlib
from datetime import datetime
import re
from collections import Counter
from typing import List, Dict, Tuple
import pickle
import time

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import pandas as pd
import ollama
import faiss

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this'
CORS(app, supports_credentials=True)

# -----------------------------
# Configuration
# -----------------------------
DATA_FOLDER = "data"
CACHE_FOLDER = "cache"
MODEL_NAME = "llama3"

# Thresholds - LOWERED for better recall
SIMILARITY_THRESHOLD = 0.20  # Very low
RERANK_THRESHOLD = 0.30      # Very low

# Performance
BATCH_SIZE = 64
NUM_CTX = 8192
MAX_CONVERSATION_HISTORY = 10

# JSON processing
JSON_COMBINE_QA = True  # Combine question + answer for better context

# Cache settings
USE_SEMANTIC_CACHE = True
USE_EMBEDDING_CACHE = True

# -----------------------------
# GPU / CPU DEVICE
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"🟢 GPU detected: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -----------------------------
# Global variables
# -----------------------------
documents = []
doc_embeddings = None
embedder = None
reranker = None
faiss_index = None
conversation_histories = {}

semantic_cache = {}
embedding_cache = {}

# -----------------------------
# Cache Setup
# -----------------------------
os.makedirs(CACHE_FOLDER, exist_ok=True)
EMBEDDING_CACHE_FILE = os.path.join(CACHE_FOLDER, "embedding_cache.pkl")
SEMANTIC_CACHE_FILE = os.path.join(CACHE_FOLDER, "semantic_cache.pkl")
DOCUMENTS_CACHE_FILE = os.path.join(CACHE_FOLDER, "documents_cache.pkl")
FAISS_INDEX_FILE = os.path.join(CACHE_FOLDER, "faiss_index.bin")

# -----------------------------
# Performance Monitoring
# -----------------------------
class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
    
    def start(self, name):
        self.timings[name] = time.time()
    
    def end(self, name):
        if name in self.timings:
            elapsed = time.time() - self.timings[name]
            print(f"  ⏱️  {name}: {elapsed:.3f}s")
            del self.timings[name]
            return elapsed
        return 0

perf = PerformanceMonitor()

# -----------------------------
# Semantic Caching
# -----------------------------
class SemanticCache:
    def __init__(self, threshold=0.95):
        self.cache = {}
        self.threshold = threshold
        self.hits = 0
        self.misses = 0
    
    def _get_hash(self, query_embedding):
        return hashlib.md5(query_embedding.tobytes()).hexdigest()
    
    def get(self, query_embedding, embedder_model):
        if not self.cache:
            self.misses += 1
            return None
        
        if torch.is_tensor(query_embedding):
            query_embedding = query_embedding.cpu().numpy()
        
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        for cache_key, cache_data in self.cache.items():
            cached_embedding = cache_data["embedding"]
            similarity = np.dot(query_embedding.flatten(), cached_embedding.flatten())
            
            if similarity >= self.threshold:
                self.hits += 1
                print(f"  ✅ CACHE HIT (similarity: {similarity:.3f})")
                return {
                    "answer": cache_data["answer"],
                    "sources": cache_data["sources"]
                }
        
        self.misses += 1
        return None
    
    def set(self, query_embedding, answer, sources):
        if torch.is_tensor(query_embedding):
            query_embedding = query_embedding.cpu().numpy()
        
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        cache_key = self._get_hash(query_embedding)
        
        self.cache[cache_key] = {
            "embedding": query_embedding.flatten(),
            "answer": answer,
            "sources": sources,
            "timestamp": time.time()
        }
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.cache, f)
        print(f"  💾 Saved cache ({len(self.cache)} entries)")
    
    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.cache = pickle.load(f)
            print(f"  📂 Loaded cache ({len(self.cache)} entries)")
    
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.cache)
        }

semantic_cache_system = SemanticCache() if USE_SEMANTIC_CACHE else None

# -----------------------------
# Embedding Cache
# -----------------------------
def get_cached_embedding(text, embedder_model):
    cache_key = hashlib.md5(text.encode()).hexdigest()
    
    if USE_EMBEDDING_CACHE and cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    with torch.no_grad():
        embedding = embedder_model.encode(
            [text],
            convert_to_tensor=True,
            device=DEVICE,
            normalize_embeddings=True
        )
    
    if USE_EMBEDDING_CACHE:
        embedding_cache[cache_key] = embedding
    
    return embedding

def save_embedding_cache():
    if USE_EMBEDDING_CACHE and embedding_cache:
        with open(EMBEDDING_CACHE_FILE, 'wb') as f:
            cache_to_save = {}
            for k, v in embedding_cache.items():
                if torch.is_tensor(v):
                    cache_to_save[k] = v.cpu().numpy()
                else:
                    cache_to_save[k] = v
            pickle.dump(cache_to_save, f)
        print(f"  💾 Saved embedding cache ({len(embedding_cache)} entries)")

def load_embedding_cache():
    global embedding_cache
    if USE_EMBEDDING_CACHE and os.path.exists(EMBEDDING_CACHE_FILE):
        with open(EMBEDDING_CACHE_FILE, 'rb') as f:
            loaded = pickle.load(f)
            for k, v in loaded.items():
                if isinstance(v, np.ndarray):
                    embedding_cache[k] = torch.from_numpy(v).to(DEVICE)
                else:
                    embedding_cache[k] = v
        print(f"  📂 Loaded embedding cache ({len(embedding_cache)} entries)")

# -----------------------------
# Legal Pattern Extraction
# -----------------------------
def extract_legal_keywords(text: str) -> List[str]:
    """Extract IPC sections, chapters, and legal terms"""
    keywords = []
    
    # Legal patterns - COMPREHENSIVE
    patterns = [
        r'IPC[- ]?(?:Section|Sec\.?|§)?\s*(\d+[A-Z]?)',  # IPC Section 278, IPC 278
        r'Section\s+(\d+[A-Z]?)',  # Section 278
        r'Chapter\s+(\d+)',  # Chapter 14
        r'Article\s+(\d+)',
        r'₹\s*(\d+(?:,\d{3})*)',  # ₹500
        r'\$\s*(\d+(?:,\d{3})*)',  # $500
        r'(\d+)\s*(?:years?|months?|days?)',  # 10 years
        r'fine',
        r'imprisonment',
        r'penalty',
        r'punishment',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple):
                keywords.extend([m for m in matches if m])
            else:
                keywords.extend(matches)
    
    # Extract significant words
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stopwords = {'shall', 'must', 'being', 'been', 'have', 'with', 'that', 'this', 'from'}
    filtered = [w for w in words if w not in stopwords and len(w) > 3]
    
    counter = Counter(filtered)
    keywords.extend([word for word, _ in counter.most_common(10)])
    
    return list(set(keywords))[:25]

# -----------------------------
# Conversation Memory
# -----------------------------
def get_conversation_history(session_id):
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    return conversation_histories[session_id]

def add_to_conversation(session_id, user_message, bot_response):
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    conversation_histories[session_id].append({
        "user": user_message,
        "assistant": bot_response,
        "timestamp": datetime.now().isoformat()
    })
    
    if len(conversation_histories[session_id]) > MAX_CONVERSATION_HISTORY:
        conversation_histories[session_id] = conversation_histories[session_id][-MAX_CONVERSATION_HISTORY:]

def clear_conversation(session_id):
    if session_id in conversation_histories:
        conversation_histories[session_id] = []

def format_conversation_context(session_id):
    history = get_conversation_history(session_id)
    if not history:
        return ""
    
    formatted = "\n\nPrevious:\n"
    for msg in history[-2:]:
        formatted += f"Q: {msg['user']}\nA: {msg['assistant']}\n"
    
    return formatted

# -----------------------------
# Query Expansion - ENHANCED
# -----------------------------
def expand_query(query: str) -> List[str]:
    """Aggressively expand query"""
    variations = [query, query.lower()]
    query_lower = query.lower()
    
    # Legal synonyms
    synonyms = {
        'polluting': ['pollution', 'pollute', 'contaminating', 'fouling'],
        'atmosphere': ['air', 'environment', 'sky'],
        'results in': ['leads to', 'causes', 'penalty', 'punishment', 'fine'],
        'killed': ['murder', 'death', 'homicide', 'killing'],
        'baby': ['infant', 'newborn', 'child'],
        'punishment': ['penalty', 'sentence', 'fine', 'imprisonment'],
    }
    
    for key, syns in synonyms.items():
        if key in query_lower:
            for syn in syns[:2]:
                variations.append(query_lower.replace(key, syn))
    
    # Remove question words
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'results', 'in', 'the']
    words = query_lower.split()
    core_words = [w for w in words if w not in question_words and len(w) > 2]
    
    if core_words:
        variations.append(" ".join(core_words))
    
    # Add partial phrases
    if len(core_words) >= 2:
        variations.append(f"{core_words[0]} {core_words[-1]}")
    
    return list(set(variations))[:6]

# -----------------------------
# Load Documents - OPTIMIZED FOR JSON
# -----------------------------
def load_documents(folder=DATA_FOLDER):
    docs = []

    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 Created {folder} folder.")
        return docs

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            # JSON Files - PRIMARY FORMAT
            if file.endswith(".json"):
                print(f"\n📋 Loading JSON: {file}")
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    count = 0
                    for item in data:
                        if isinstance(item, dict):
                            # Try different JSON structures
                            question = item.get("question") or item.get("Query") or item.get("q") or ""
                            answer = item.get("answer") or item.get("Response") or item.get("a") or ""
                            
                            if question and answer:
                                # Create rich content with Q&A combined
                                if JSON_COMBINE_QA:
                                    content = f"{question} {answer}"
                                    summary = f"Q: {question[:100]}... A: {answer[:100]}..."
                                else:
                                    content = f"Question: {question}\nAnswer: {answer}"
                                    summary = content[:200]
                                
                                docs.append({
                                    "content": content,
                                    "summary": summary,
                                    "source": f"{file} (Entry {count + 1})",
                                    "keywords": extract_legal_keywords(content),
                                    "question": question,
                                    "answer": answer,
                                    "type": "json"
                                })
                                count += 1
                    
                    print(f"    ✓ Loaded {count} Q&A pairs")
            
            # CSV Files
            elif file.endswith(".csv"):
                print(f"\n📊 Loading CSV: {file}")
                try:
                    df = pd.read_csv(path, encoding='utf-8')
                except:
                    df = pd.read_csv(path, encoding='latin-1')
                
                # Process CSV rows
                for idx, row in df.iterrows():
                    row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                    
                    if row_text.strip():
                        docs.append({
                            "content": row_text,
                            "summary": row_text[:200],
                            "source": f"{file} (Row {idx + 1})",
                            "keywords": extract_legal_keywords(row_text),
                            "type": "csv"
                        })
                
                print(f"    ✓ Loaded {len(df)} rows")
            
            # PDF Files
            elif file.endswith(".pdf"):
                print(f"\n📄 Loading PDF: {file}")
                reader = PdfReader(path)
                full_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                
                if full_text.strip():
                    # Split into chunks
                    sentences = re.split(r'(?<=[.!?])\s+', full_text)
                    
                    chunk_size = 5
                    for i in range(0, len(sentences), chunk_size):
                        chunk = " ".join(sentences[i:i+chunk_size])
                        
                        if chunk.strip():
                            docs.append({
                                "content": chunk,
                                "summary": chunk[:200],
                                "source": f"{file} (Page {i//chunk_size + 1})",
                                "keywords": extract_legal_keywords(chunk),
                                "type": "pdf"
                            })
                    
                    print(f"    ✓ Created {len(docs)} chunks")

        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    return docs

# -----------------------------
# FAISS Index
# -----------------------------
def create_faiss_index(embeddings):
    global faiss_index
    
    if torch.is_tensor(embeddings):
        embeddings_np = embeddings.cpu().numpy().astype('float32')
    else:
        embeddings_np = np.array(embeddings).astype('float32')
    
    dimension = embeddings_np.shape[1]
    
    # HNSW for fast search
    faiss_index = faiss.IndexHNSWFlat(dimension, 32)
    faiss_index.hnsw.efConstruction = 40
    faiss_index.hnsw.efSearch = 20  # Increased for better recall
    
    faiss_index.add(embeddings_np)
    
    print(f"  ✅ FAISS index created ({len(embeddings_np)} vectors)")

def save_faiss_index():
    if faiss_index is not None:
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)
        print(f"  💾 Saved FAISS index")

def load_faiss_index():
    global faiss_index
    if os.path.exists(FAISS_INDEX_FILE):
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"  📂 Loaded FAISS index")
        return True
    return False

# -----------------------------
# Cache Management
# -----------------------------
def save_documents_cache():
    cache_data = {
        "documents": documents,
        "embeddings": doc_embeddings.cpu().numpy() if torch.is_tensor(doc_embeddings) else doc_embeddings
    }
    with open(DOCUMENTS_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"  💾 Saved documents cache")

def load_documents_cache():
    global documents, doc_embeddings
    if os.path.exists(DOCUMENTS_CACHE_FILE):
        with open(DOCUMENTS_CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
            documents = cache_data["documents"]
            doc_embeddings = torch.from_numpy(cache_data["embeddings"]).to(DEVICE)
        print(f"  📂 Loaded cache ({len(documents)} docs)")
        return True
    return False

# -----------------------------
# Initialize System
# -----------------------------
def initialize_system():
    global documents, doc_embeddings, embedder, reranker

    print("\n" + "="*70)
    print("🚀 INITIALIZING LEGAL RAG SYSTEM")
    print("="*70)
    
    load_embedding_cache()
    if semantic_cache_system:
        semantic_cache_system.load(SEMANTIC_CACHE_FILE)
    
    if load_documents_cache() and load_faiss_index():
        print("  ✅ Loaded from cache!")
    else:
        print("\n  🔄 Processing documents...")
        documents = load_documents()
        
        if not documents:
            print("⚠️ No documents found!")
            return False
        
        print(f"\n✅ Total: {len(documents)} chunks")
        
        print("\n🧠 Loading models...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        embedder.eval()
        
        print("🔢 Creating embeddings...")
        contents = [f"{d.get('summary', '')} {d['content']}" for d in documents]
        
        with torch.no_grad():
            doc_embeddings = embedder.encode(
                contents,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=DEVICE,
                normalize_embeddings=True
            )
        
        print("\n⚡ Building FAISS index...")
        create_faiss_index(doc_embeddings)
        
        save_documents_cache()
        save_faiss_index()
    
    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        embedder.eval()
    
    print("🎯 Loading re-ranker...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE)

    print("\n✅ READY!")
    print("="*70 + "\n")
    return True

# -----------------------------
# Hybrid Search
# -----------------------------
def hybrid_search_faiss(query, top_k=25):
    perf.start("search")
    
    query_variations = expand_query(query)
    print(f"  🔍 Variations: {query_variations[:3]}")
    
    # Get embeddings
    all_embeddings = []
    for q_var in query_variations:
        q_embedding = get_cached_embedding(q_var, embedder)
        all_embeddings.append(q_embedding.cpu().numpy())
    
    avg_embedding = np.mean(all_embeddings, axis=0).astype('float32')
    
    # FAISS search
    distances, indices = faiss_index.search(avg_embedding.reshape(1, -1), top_k)
    semantic_scores = 1 / (1 + distances[0])
    
    # Keyword boost - AGGRESSIVE
    query_lower = query.lower()
    query_keywords = set(query_lower.split())
    
    keyword_boost = np.zeros(len(indices[0]))
    for i, idx in enumerate(indices[0]):
        doc = documents[idx]
        content_lower = doc["content"].lower()
        
        # Exact phrase matching
        if query_lower in content_lower:
            keyword_boost[i] += 0.5
        
        # Keyword matching
        doc_keywords = set(doc.get("keywords", []))
        matches = len(query_keywords & doc_keywords)
        keyword_boost[i] += matches * 0.2
        
        # Word matching
        for word in query_keywords:
            if len(word) > 2 and word in content_lower:
                keyword_boost[i] += 0.15
    
    # Combine: 50% semantic + 50% keyword
    combined_scores = semantic_scores * 0.5 + keyword_boost * 0.5
    
    perf.end("search")
    
    return indices[0], combined_scores

# -----------------------------
# Re-ranking
# -----------------------------
def rerank_results_fast(query, candidate_indices, candidate_scores, top_k=7):
    perf.start("rerank")
    
    rerank_count = min(15, len(candidate_indices))
    
    pairs = [[query, documents[idx]["content"][:700]] for idx in candidate_indices[:rerank_count]]
    rerank_scores = reranker.predict(pairs)
    
    if rerank_count < len(candidate_indices):
        rerank_scores = np.concatenate([
            rerank_scores,
            candidate_scores[rerank_count:] * 0.3
        ])
    
    final_scores = 0.3 * candidate_scores + 0.7 * rerank_scores
    sorted_indices = np.argsort(final_scores)[::-1]
    
    perf.end("rerank")
    
    return candidate_indices[sorted_indices][:top_k], final_scores[sorted_indices][:top_k]

# -----------------------------
# RAG Query
# -----------------------------
def rag_query(question, session_id, top_k=7):
    if embedder is None or faiss_index is None:
        return {"answer": "System not initialized.", "sources": []}

    perf.start("total")
    print(f"\n{'='*70}")
    print(f"🔍 Q: {question}")
    
    # Check cache
    if semantic_cache_system:
        query_embedding = get_cached_embedding(question, embedder)
        cached_result = semantic_cache_system.get(query_embedding, embedder)
        if cached_result:
            perf.end("total")
            print(f"{'='*70}\n")
            return cached_result
    
    # Search
    candidate_indices, candidate_scores = hybrid_search_faiss(question, top_k=25)
    print(f"  ✓ Found {len(candidate_indices)} candidates")
    
    # Re-rank
    ranked_indices, ranked_scores = rerank_results_fast(
        question, 
        candidate_indices,
        candidate_scores,
        top_k=top_k
    )
    
    print(f"  ✓ Best score: {ranked_scores[0]:.3f}")
    
    # Check threshold
    if ranked_scores[0] < RERANK_THRESHOLD:
        print(f"  ⚠️ Score too low ({ranked_scores[0]:.3f} < {RERANK_THRESHOLD})")
        result = {
            "answer": "I couldn't find a confident answer in the documents. The information might not be in the loaded files, or try rephrasing your question.",
            "sources": []
        }
        perf.end("total")
        return result

    # Gather contexts
    contexts = []
    sources = []

    for idx, score in zip(ranked_indices, ranked_scores):
        if float(score) >= RERANK_THRESHOLD:
            doc = documents[idx]
            
            # For JSON, use the answer directly if available
            if doc.get("type") == "json" and doc.get("answer"):
                contexts.append(f"Q: {doc.get('question', '')}\nA: {doc.get('answer', '')}")
            else:
                contexts.append(doc['content'])
            
            sources.append({
                "source": doc["source"],
                "score": float(score)
            })

    if not sources:
        result = {"answer": "No relevant information found.", "sources": []}
        perf.end("total")
        return result

    print(f"  ✓ Using {len(sources)} sources")

    combined_context = "\n\n".join(contexts)
    conversation_context = format_conversation_context(session_id)

    prompt = f"""Answer using ONLY the information below. Be direct and specific.

{conversation_context}

INFORMATION:
{combined_context}

QUESTION: {question}

Answer clearly and directly based on the information above.

ANSWER:"""

    perf.start("llm")
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_ctx": 4096,
                "num_predict": 512,
                "temperature": 0.1,
                "top_k": 10,
                "top_p": 0.6,
                "num_thread": os.cpu_count(),
            }
        )
        
        answer = response["message"]["content"]
        perf.end("llm")
        
        result = {"answer": answer, "sources": sources}
        
        if semantic_cache_system:
            semantic_cache_system.set(query_embedding, answer, sources)
        
        add_to_conversation(session_id, question, answer)
        
        total = perf.end("total")
        print(f"  🎯 TOTAL: {total:.3f}s")
        print(f"{'='*70}\n")
        
        return result
        
    except Exception as e:
        perf.end("llm")
        perf.end("total")
        return {"answer": f"Error: {str(e)}", "sources": sources}

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not user_msg:
        return jsonify({"reply": "Empty message", "sources": []})

    result = rag_query(user_msg, session_id)

    return jsonify({
        "reply": result["answer"].replace("\n", "<br>"),
        "sources": result["sources"]
    })

@app.route("/status")
def status():
    cache_stats = semantic_cache_system.get_stats() if semantic_cache_system else {}
    
    doc_types = {}
    for doc in documents:
        doc_type = doc.get("type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    return jsonify({
        "device": DEVICE,
        "documents": len(documents),
        "document_types": doc_types,
        "model": MODEL_NAME,
        "cache_stats": cache_stats,
        "thresholds": {
            "similarity": SIMILARITY_THRESHOLD,
            "rerank": RERANK_THRESHOLD
        }
    })

@app.route("/clear-conversation", methods=["POST"])
def clear_conversation_route():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    clear_conversation(session_id)
    return jsonify({"status": "Cleared", "session_id": session_id})

@app.route("/clear-cache", methods=["POST"])
def clear_cache_route():
    global embedding_cache
    embedding_cache = {}
    if semantic_cache_system:
        semantic_cache_system.cache = {}
        semantic_cache_system.hits = 0
        semantic_cache_system.misses = 0
    return jsonify({"status": "Cache cleared"})

# -----------------------------
# Shutdown
# -----------------------------
import atexit

def save_on_exit():
    print("\n💾 Saving...")
    save_embedding_cache()
    if semantic_cache_system:
        semantic_cache_system.save(SEMANTIC_CACHE_FILE)
    print("✅ Saved")

atexit.register(save_on_exit)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    success = initialize_system()
    
    if not success:
        exit(1)

    print("\n" + "="*70)
    print("🚀 LEGAL RAG SERVER")
    print("="*70)
    print(f"📍 http://localhost:5000")
    print(f"⚡ {DEVICE}")
    print(f"📚 {len(documents)} documents")
    
    doc_types = {}
    for doc in documents:
        doc_type = doc.get("type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    for dtype, count in doc_types.items():
        print(f"   - {dtype.upper()}: {count}")
    
    print(f"\n⚡ OPTIMIZATIONS:")
    print(f"   ✓ JSON Q&A optimized")
    print(f"   ✓ Aggressive keyword matching")
    print(f"   ✓ Low thresholds ({RERANK_THRESHOLD})")
    print(f"   ✓ FAISS fast search")
    print(f"   ✓ Semantic caching")
    print("="*70 + "\n")

    app.run(debug=False, port=5000, host="0.0.0.0", threaded=True)