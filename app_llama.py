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
import openpyxl
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
app.secret_key = 'legal-mind-secret-key-2025'
CORS(app, supports_credentials=True)

# -----------------------------
# Configuration
# -----------------------------
DATA_FOLDER = "data"
CACHE_FOLDER = "cache"
MODEL_NAME = "llama3"

# INTELLIGENT THRESHOLDS - Balanced for accuracy
SIMILARITY_THRESHOLD = 0.08  # Very low for recall
RERANK_THRESHOLD = 0.12      # Low but not too low
ULTRA_FALLBACK = 0.06        # Last resort

# Performance
BATCH_SIZE = 64
NUM_CTX = 4096
MAX_CONVERSATION_HISTORY = 10

# Chunking - Optimized for legal text
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 50

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
os.makedirs(DATA_FOLDER, exist_ok=True)

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
# INTELLIGENT QUERY REWRITING
# -----------------------------
def rewrite_query_intelligent(query: str) -> str:
    """
    Uses LLM to understand twisted/slang queries and convert to formal legal terms.
    This is THE KEY to handling natural language questions.
    """
    # Skip for very short queries
    if len(query.split()) < 3:
        return query
    
    try:
        prompt = f"""You are a legal expert. Rewrite this user question into a formal legal search query.
Focus on key legal terms, sections, and concepts. Keep it concise.

User Question: "{query}"

Formal Legal Query:"""
        
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": 40,
                "temperature": 0.1,
                "num_ctx": 512  # Small context for speed
            }
        )
        
        rewritten = response["message"]["content"].strip()
        # Clean up
        rewritten = rewritten.replace('"', '').replace("'", "").strip()
        
        if rewritten and rewritten != query:
            print(f"  🧠 Query Rewrite: '{query}' → '{rewritten}'")
            return rewritten
        
    except Exception as e:
        print(f"  ⚠️  Rewrite failed: {e}")
    
    return query

def expand_query_smart(query: str) -> List[str]:
    """
    Combines original query with intelligent rewriting and synonyms.
    """
    variations = [query, query.lower()]
    
    # Get smart rewrite
    smart_query = rewrite_query_intelligent(query)
    if smart_query != query and smart_query not in variations:
        variations.append(smart_query)
        variations.append(smart_query.lower())
    
    # Legal synonyms
    synonyms = {
        'punishment': ['penalty', 'sentence', 'imprisonment', 'fine', 'punishable'],
        'killed': ['murder', 'death', 'homicide', 'causing death', 'killing'],
        'baby': ['infant', 'child', 'newborn', 'unborn child', 'quick unborn'],
        'quick unborn child': ['fetus', 'unborn baby', 'infant in womb'],
        'culpable homicide': ['murder', 'manslaughter', 'homicide'],
        'theft': ['stealing', 'robbery', 'larceny'],
        'hurt': ['injury', 'harm', 'grievous hurt'],
    }
    
    query_lower = query.lower()
    for key, syns in synonyms.items():
        if key in query_lower:
            for syn in syns[:3]:  # Top 3 synonyms only
                variations.append(query_lower.replace(key, syn))
    
    # Extract IPC section if mentioned
    section_match = re.search(r'(?:IPC|Section)\s*(\d+)', query, re.IGNORECASE)
    if section_match:
        section_num = section_match.group(1)
        variations.append(f"IPC Section {section_num}")
        variations.append(f"Section {section_num}")
    
    return list(set(variations))[:20]  # Max 20 variations

# -----------------------------
# SMART CSV PROCESSING - The Accuracy Fix
# -----------------------------
class DocumentPreprocessor:
    """Handles different file formats with intelligence"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\x00', '').strip()
        return text
    
    def process_csv(self, filepath: str) -> List[Dict]:
        """
        INTELLIGENT CSV PROCESSING
        - Detects schema (Q&A, FIR, Laws)
        - Front-loads important info (Punishment, Section)
        - Preserves section numbers for accuracy
        """
        print(f"\n📊 Processing CSV: {os.path.basename(filepath)}")
        chunks = []
        
        try:
            # Try multiple encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    print(f"    ✓ Loaded with {encoding}")
                    break
                except:
                    continue
            
            if df is None:
                print(f"    ❌ Could not read CSV")
                return chunks
            
            source = os.path.basename(filepath)
            columns_lower = [c.lower() for c in df.columns]
            
            # Detect dataset type
            is_fir = any(col in columns_lower for col in ['punishment', 'offense', 'description'])
            is_qa = any(col in columns_lower for col in ['instruction', 'output'])
            is_laws = any(col in columns_lower for col in ['ipc chapter', 'ipc section'])
            
            print(f"    ✓ Detected: {'FIR' if is_fir else 'Q&A' if is_qa else 'Laws' if is_laws else 'Generic'}")
            
            for idx, row in df.iterrows():
                content = ""
                section_num = ""
                
                # STRATEGY 1: FIR Dataset (Offense/Punishment/Description)
                if is_fir:
                    offense = str(row.get('Offense', '')).strip()
                    punishment = str(row.get('Punishment', '')).strip()
                    description = str(row.get('Description', '')).strip()
                    
                    # Extract section from description
                    section_match = re.search(r'(?:IPC|Section)\s*(\d+)', description)
                    if section_match:
                        section_num = section_match.group(1)
                    
                    # Truncate long descriptions
                    if len(description) > 1500:
                        description = description[:1500] + "..."
                    
                    # FRONT-LOAD critical info
                    content = f"""IPC SECTION: {section_num if section_num else 'N/A'}
OFFENSE: {offense}
PUNISHMENT: {punishment}
DETAILS: {description}"""
                
                # STRATEGY 2: Laws Dataset (instruction/input/output)
                elif is_qa:
                    instruction = str(row.get('instruction', '')).strip()
                    input_text = str(row.get('input', '')).strip()
                    output = str(row.get('output', '')).strip()
                    
                    # Extract section number
                    section_match = re.search(r'(?:IPC|Section)\s*(\d+)', instruction + input_text)
                    if section_match:
                        section_num = section_match.group(1)
                    
                    question = f"{instruction} {input_text}".strip()
                    
                    content = f"""SECTION: {section_num if section_num else 'N/A'}
TOPIC: {question}
LAW: {output}"""
                
                # STRATEGY 3: Constitution Dataset (IPC Chapter/Section)
                elif is_laws:
                    chapter = str(row.get('IPC Chapter', '')).strip()
                    section = str(row.get('IPC Section', '')).strip()
                    details = str(row.get('Details', '')).strip()
                    
                    # Extract section number
                    section_match = re.search(r'(\d+)', section)
                    if section_match:
                        section_num = section_match.group(1)
                    
                    content = f"""IPC SECTION: {section_num if section_num else section}
CHAPTER: {chapter}
DETAILS: {details}"""
                
                # STRATEGY 4: Generic
                else:
                    parts = []
                    for col, val in row.items():
                        if pd.notna(val) and str(val).strip():
                            parts.append(f"{col}: {val}")
                    content = " | ".join(parts)
                
                # Only add if meaningful content
                if content.strip() and len(content) > MIN_CHUNK_SIZE:
                    chunks.append({
                        "content": self.clean_text(content),
                        "section": section_num,  # Store for accuracy
                        "type": "csv_intelligent",
                        "source": f"{source} (Row {idx + 1})",
                        "char_count": len(content)
                    })
            
            print(f"    ✓ Processed {len(chunks)} rows intelligently")
            
        except Exception as e:
            print(f"    ❌ CSV error: {e}")
        
        return chunks
    
    def process_json(self, filepath: str) -> List[Dict]:
        """Process JSON files"""
        print(f"\n📋 Processing JSON: {os.path.basename(filepath)}")
        chunks = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            source = os.path.basename(filepath)
            
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        q = item.get("question") or item.get("Query") or ""
                        a = item.get("answer") or item.get("Response") or ""
                        
                        if q and a:
                            content = f"Q: {q}\nA: {a}"
                            chunks.append({
                                "content": content,
                                "type": "json",
                                "source": f"{source} #{idx + 1}"
                            })
            
            print(f"    ✓ Extracted {len(chunks)} chunks")
            
        except Exception as e:
            print(f"    ❌ JSON error: {e}")
        
        return chunks
    
    def process_pdf(self, filepath: str) -> List[Dict]:
        """Process PDF files"""
        print(f"\n📄 Processing PDF: {os.path.basename(filepath)}")
        chunks = []
        
        try:
            reader = PdfReader(filepath)
            full_text = ""
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
            
            full_text = self.clean_text(full_text)
            
            # Chunk with LangChain
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            text_chunks = splitter.split_text(full_text)
            
            for chunk in text_chunks:
                if len(chunk) > MIN_CHUNK_SIZE:
                    chunks.append({
                        "content": chunk,
                        "type": "pdf",
                        "source": os.path.basename(filepath)
                    })
            
            print(f"    ✓ Extracted {len(chunks)} chunks")
            
        except Exception as e:
            print(f"    ❌ PDF error: {e}")
        
        return chunks

preprocessor = DocumentPreprocessor()

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
                print(f"  ✅ CACHE HIT ({similarity:.3f})")
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
            normalize_embeddings=True,
            show_progress_bar=False
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
        print(f"  📂 Loaded embedding cache")

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
# Load Documents
# -----------------------------
def load_documents(folder=DATA_FOLDER):
    """Load all documents"""
    docs = []

    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 Created {folder} folder.")
        return docs

    files = os.listdir(folder)
    if not files:
        print(f"⚠️  No files in {folder}")
        return docs

    print(f"\n{'='*70}")
    print(f"📚 INTELLIGENT DOCUMENT LOADING")
    print(f"{'='*70}")

    for file in files:
        filepath = os.path.join(folder, file)
        
        if not os.path.isfile(filepath):
            continue
        
        ext = os.path.splitext(file)[1].lower()

        try:
            if ext == '.csv':
                chunks = preprocessor.process_csv(filepath)
                docs.extend(chunks)
            elif ext == '.json':
                chunks = preprocessor.process_json(filepath)
                docs.extend(chunks)
            elif ext == '.pdf':
                chunks = preprocessor.process_pdf(filepath)
                docs.extend(chunks)

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    print(f"\n{'='*70}")
    print(f"✅ Total chunks loaded: {len(docs)}")
    print(f"{'='*70}\n")

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
    
    # Use Inner Product (cosine similarity) for better semantic matching
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings_np)
    
    print(f"  ✅ FAISS index ({len(embeddings_np)} vectors, dim={dimension})")

def save_faiss_index():
    if faiss_index is not None:
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)

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

def load_documents_cache():
    global documents, doc_embeddings
    if os.path.exists(DOCUMENTS_CACHE_FILE):
        with open(DOCUMENTS_CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
            documents = cache_data["documents"]
            doc_embeddings = torch.from_numpy(cache_data["embeddings"]).to(DEVICE)
        print(f"  📂 Loaded docs ({len(documents)})")
        return True
    return False

# -----------------------------
# Initialize System
# -----------------------------
def initialize_system():
    global documents, doc_embeddings, embedder, reranker, faiss_index

    print("\n" + "="*70)
    print("🚀 INTELLIGENT RAG SYSTEM - LEGAL AI")
    print("="*70)
    
    load_embedding_cache()
    if semantic_cache_system:
        semantic_cache_system.load(SEMANTIC_CACHE_FILE)
    
    if load_documents_cache() and load_faiss_index():
        print("  ✅ Loaded from cache!")
    else:
        print("\n  🔄 Processing with intelligence...")
        documents = load_documents()
        
        if not documents:
            print("⚠️ No documents!")
            return False
        
        print(f"\n✅ Total: {len(documents)}")
        
        print("\n🧠 Loading models...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        embedder.eval()
        
        print("🔢 Generating embeddings...")
        contents = [d['content'] for d in documents]
        
        with torch.no_grad():
            doc_embeddings = embedder.encode(
                contents,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=DEVICE,
                normalize_embeddings=True
            )
        
        print("\n⚡ Creating FAISS index...")
        create_faiss_index(doc_embeddings)
        
        save_documents_cache()
        save_faiss_index()
    
    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        embedder.eval()
    
    print("🎯 Loading re-ranker...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE)

    print("\n✅ SYSTEM READY!")
    print("="*70 + "\n")
    return True

# -----------------------------
# INTELLIGENT HYBRID SEARCH
# -----------------------------
def hybrid_search_intelligent(query, top_k=30):
    """
    Combines semantic search with keyword matching and section number extraction.
    """
    perf.start("search")
    
    # Get smart query variations
    query_variations = expand_query_smart(query)
    print(f"  🔍 Searching with {len(query_variations)} variations")
    
    # Get embeddings
    all_embeddings = []
    for q_var in query_variations:
        q_embedding = get_cached_embedding(q_var, embedder)
        all_embeddings.append(q_embedding.cpu().numpy())
    
    avg_embedding = np.mean(all_embeddings, axis=0).astype('float32')
    
    # FAISS search
    distances, indices = faiss_index.search(avg_embedding.reshape(1, -1), top_k)
    semantic_scores = distances[0]  # Already normalized with IndexFlatIP
    
    # Keyword matching
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Extract section number if present
    query_section = None
    section_match = re.search(r'(?:IPC|Section)\s*(\d+)', query, re.IGNORECASE)
    if section_match:
        query_section = section_match.group(1)
        print(f"  🎯 Detected Section: {query_section}")
    
    keyword_boost = np.zeros(len(indices[0]))
    for i, idx in enumerate(indices[0]):
        doc = documents[idx]
        content_lower = doc["content"].lower()
        
        boost = 0
        
        # CRITICAL: Section number exact match gets HUGE boost
        if query_section and doc.get("section") == query_section:
            boost += 5.0
            print(f"  ✓ Exact section match: {doc.get('section')}")
        
        # Exact phrase match
        if query_lower in content_lower:
            boost += 1.5
        
        # Word overlap
        for word in query_words:
            if len(word) > 2 and word in content_lower:
                boost += 0.4
        
        keyword_boost[i] = boost
    
    # WEIGHTS: 60% semantic + 40% keyword (balanced)
    combined_scores = semantic_scores * 0.6 + keyword_boost * 0.4
    
    perf.end("search")
    
    return indices[0], combined_scores

# -----------------------------
# Re-ranking with Intelligence
# -----------------------------
def rerank_results_smart(query, candidate_indices, candidate_scores, top_k=10):
    """Smart re-ranking that preserves section accuracy"""
    perf.start("rerank")
    
    # Extract section from query
    query_section = None
    section_match = re.search(r'(?:IPC|Section)\s*(\d+)', query, re.IGNORECASE)
    if section_match:
        query_section = section_match.group(1)
    
    rerank_count = min(15, len(candidate_indices))
    
    pairs = [[query, documents[idx]["content"][:800]] for idx in candidate_indices[:rerank_count]]
    rerank_scores = reranker.predict(pairs)
    
    # Apply section boost to rerank scores
    if query_section:
        for i, idx in enumerate(candidate_indices[:rerank_count]):
            if documents[idx].get("section") == query_section:
                rerank_scores[i] += 0.3  # Boost section matches
    
    if rerank_count < len(candidate_indices):
        rerank_scores = np.concatenate([
            rerank_scores,
            candidate_scores[rerank_count:] * 0.2
        ])
    
    # 70% rerank + 30% initial
    final_scores = 0.7 * rerank_scores + 0.3 * candidate_scores
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
    print(f"🔍 QUERY: {question}")
    
    # Check cache
    if semantic_cache_system:
        query_embedding = get_cached_embedding(question, embedder)
        cached_result = semantic_cache_system.get(query_embedding, embedder)
        if cached_result:
            perf.end("total")
            return cached_result
    
    # Search with intelligence
    candidate_indices, candidate_scores = hybrid_search_intelligent(question, top_k=30)
    print(f"  ✓ Found {len(candidate_indices)} candidates")
    
    # Smart re-rank
    ranked_indices, ranked_scores = rerank_results_smart(
        question, 
        candidate_indices,
        candidate_scores,
        top_k=top_k
    )
    
    print(f"  ✓ Top score: {ranked_scores[0]:.3f}")
    print(f"  ✓ Threshold: {RERANK_THRESHOLD}")
    
    # Check confidence
    if ranked_scores[0] < RERANK_THRESHOLD:
        if ranked_scores[0] >= ULTRA_FALLBACK:
            print(f"  ⚠️  Using fallback (score: {ranked_scores[0]:.3f})")
        else:
            result = {
                "answer": "I couldn't find specific information on that. Please try:\n" +
                         "• Being more specific about the IPC section\n" +
                         "• Using legal terminology\n" +
                         "• Rephrasing your question",
                "sources": []
            }
            perf.end("total")
            return result

    # Gather contexts
    contexts = []
    sources = []

    for idx, score in zip(ranked_indices, ranked_scores):
        if float(score) >= ULTRA_FALLBACK:
            doc = documents[idx]
            contexts.append(doc['content'])
            
            sources.append({
                "source": doc.get("source", "Unknown"),
                "score": float(score),
                "type": doc.get("type", "unknown"),
                "section": doc.get("section", "N/A")
            })

    if not sources:
        result = {"answer": "No relevant information found.", "sources": []}
        perf.end("total")
        return result

    print(f"  ✓ Using {len(sources)} sources")

    combined_context = "\n\n---\n\n".join(contexts)
    conversation_context = format_conversation_context(session_id)

    prompt = f"""You are an expert legal assistant. Answer the question based ONLY on the provided legal information.

{conversation_context}

LEGAL INFORMATION:
{combined_context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer accurately and cite specific IPC sections when available
- Mention the punishment clearly
- Be concise but complete
- If unsure, say so

ANSWER:"""

    perf.start("llm")
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_ctx": NUM_CTX,
                "num_predict": 500,
                "temperature": 0.1,
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
def welcome():
    return render_template("index1.html") 
@app.route("/chatbot")
def chatbot():
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
        "types": doc_types,
        "model": MODEL_NAME,
        "cache_stats": cache_stats,
        "thresholds": {
            "similarity": SIMILARITY_THRESHOLD,
            "rerank": RERANK_THRESHOLD,
            "ultra_fallback": ULTRA_FALLBACK
        }
    })

@app.route("/clear-conversation", methods=["POST"])
def clear_conversation_route():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    clear_conversation(session_id)
    return jsonify({"status": "Cleared"})

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

atexit.register(save_on_exit)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    success = initialize_system()
    
    if not success:
        exit(1)

    print("\n" + "="*70)
    print("🚀 INTELLIGENT LEGAL RAG SYSTEM")
    print("="*70)
    print(f"📍 http://localhost:5000")
    print(f"⚡ {DEVICE}")
    print(f"📚 {len(documents)} chunks")
    
    doc_types = {}
    for doc in documents:
        doc_type = doc.get("type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    for dtype, count in doc_types.items():
        print(f"   • {dtype.upper()}: {count}")
    
    print(f"\n🧠 INTELLIGENCE FEATURES:")
    print(f"   ✓ LLM Query Rewriting (handles twisted questions)")
    print(f"   ✓ Schema Detection (FIR/Q&A/Laws auto-detected)")
    print(f"   ✓ Section Number Extraction & Matching")
    print(f"   ✓ Smart Front-Loading (Punishment/Section first)")
    print(f"   ✓ Intelligent Re-ranking")
    print(f"   ✓ 20 Query variations with legal synonyms")
    
    print(f"\n⚡ ACCURACY FIXES:")
    print(f"   ✓ Section-specific matching (IPC 122 ≠ IPC 123)")
    print(f"   ✓ Keyword boost for exact section matches (5x)")
    print(f"   ✓ Balanced thresholds (not too high, not too low)")
    print("="*70 + "\n")

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))