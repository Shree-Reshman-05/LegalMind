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

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import pandas as pd
import ollama

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
MODEL_NAME = "llama3"
SIMILARITY_THRESHOLD = 0.25  # Even lower - agentic chunks are better
RERANK_THRESHOLD = 0.4

# Agentic chunking parameters
PROPOSITION_CHUNK_SIZE = 3  # Number of propositions per chunk
USE_AGENTIC_CHUNKING = True  # Set to False to use simple chunking

# Performance
BATCH_SIZE = 32
NUM_CTX = 8192
MAX_CONVERSATION_HISTORY = 10

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
query_cache = {}
conversation_histories = {}
chunk_cache = {}  # Cache for agentic chunks

# -----------------------------
# Agentic Chunking System
# -----------------------------
class AgenticChunker:
    """
    Intelligent chunking that uses LLM to:
    1. Break text into atomic propositions
    2. Group related propositions
    3. Add contextual metadata
    4. Create semantic summaries
    """
    
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        
    def extract_propositions(self, text: str, source: str) -> List[str]:
        """Break text into atomic propositions using LLM"""
        
        prompt = f"""You are a legal document analyzer. Your task is to break down the following legal text into clear, standalone propositions (facts/statements).

RULES:
1. Each proposition must be self-contained and understandable on its own
2. Include ALL key legal information (sections, penalties, timeframes, conditions)
3. Maintain legal accuracy and specificity
4. Number each proposition
5. Keep propositions concise but complete

TEXT TO ANALYZE:
{text}

OUTPUT FORMAT:
1. [First proposition]
2. [Second proposition]
3. [Third proposition]
...

PROPOSITIONS:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": 4096,
                    "temperature": 0.1,  # Very deterministic
                    "num_predict": 2048,
                }
            )
            
            content = response["message"]["content"]
            
            # Parse propositions
            propositions = []
            for line in content.split('\n'):
                line = line.strip()
                # Match numbered lines like "1. proposition" or "1) proposition"
                match = re.match(r'^(\d+)[.)]\s*(.+)$', line)
                if match:
                    prop = match.group(2).strip()
                    if prop:
                        propositions.append(prop)
            
            return propositions if propositions else [text]
            
        except Exception as e:
            print(f"⚠️ Proposition extraction failed: {e}")
            # Fallback to sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def create_contextual_summary(self, propositions: List[str], source: str) -> str:
        """Create a contextual summary of propositions for better retrieval"""
        
        props_text = "\n".join([f"- {p}" for p in propositions[:10]])  # Limit for token efficiency
        
        prompt = f"""Summarize the key legal concepts, sections, and topics covered in these propositions in 2-3 sentences. Focus on searchable keywords.

PROPOSITIONS:
{props_text}

SUMMARY:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": 2048,
                    "temperature": 0.1,
                    "num_predict": 256,
                }
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            print(f"⚠️ Summary generation failed: {e}")
            return " ".join(propositions[:3])
    
    def group_propositions(self, propositions: List[str], chunk_size: int = PROPOSITION_CHUNK_SIZE) -> List[List[str]]:
        """Group propositions into semantically coherent chunks"""
        chunks = []
        
        for i in range(0, len(propositions), chunk_size):
            chunk = propositions[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def chunk_document(self, text: str, source: str) -> List[Dict]:
        """Main agentic chunking method"""
        print(f"  🤖 Agentic chunking: {source}")
        
        # Step 1: Extract atomic propositions
        propositions = self.extract_propositions(text, source)
        print(f"    ✓ Extracted {len(propositions)} propositions")
        
        # Step 2: Group propositions
        prop_groups = self.group_propositions(propositions)
        print(f"    ✓ Created {len(prop_groups)} proposition groups")
        
        # Step 3: Create chunks with context
        chunks = []
        for idx, prop_group in enumerate(prop_groups):
            # Combine propositions
            chunk_text = " ".join(prop_group)
            
            # Create contextual summary
            summary = self.create_contextual_summary(prop_group, source)
            
            # Extract keywords
            keywords = self.extract_keywords(chunk_text)
            
            chunks.append({
                "content": chunk_text,
                "summary": summary,
                "source": f"{source} (Chunk {idx + 1})",
                "keywords": keywords,
                "proposition_count": len(prop_group),
                "propositions": prop_group  # Store for reference
            })
        
        print(f"    ✓ Created {len(chunks)} agentic chunks")
        return chunks
    
    def extract_keywords(self, text: str, top_n: int = 15) -> List[str]:
        """Extract important keywords with legal focus"""
        # Legal-specific patterns
        legal_patterns = [
            r'Section \d+[A-Z]?',
            r'Article \d+',
            r'Chapter \d+',
            r'Rule \d+',
            r'\b\d+\s*(?:years?|months?|days?)\b',
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
        ]
        
        keywords = []
        
        # Extract legal patterns
        for pattern in legal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend(matches)
        
        # Extract significant words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Legal stopwords to exclude
        legal_stopwords = {
            'shall', 'must', 'section', 'article', 'whereas', 'hereby',
            'herein', 'therefore', 'pursuant', 'aforementioned', 'such',
            'said', 'same', 'following', 'foregoing', 'provided', 'without'
        }
        
        # Filter and count
        filtered_words = [w for w in words if w not in legal_stopwords and len(w) > 3]
        counter = Counter(filtered_words)
        
        # Combine with legal patterns
        keywords.extend([word for word, _ in counter.most_common(top_n)])
        
        return list(set(keywords))[:top_n]

# Initialize agentic chunker
agentic_chunker = AgenticChunker() if USE_AGENTIC_CHUNKING else None

# -----------------------------
# Simple Text Processing (fallback)
# -----------------------------
def clean_text(text):
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]§$]', '', text)
    return text.strip()

def simple_chunk_text(text, chunk_size=500, overlap=100):
    """Simple overlapping chunks (fallback)"""
    chunks = []
    text = clean_text(text)
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap//10:]) if len(words) > overlap//10 else ""
            current_chunk = overlap_text + " " + sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

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
    
    formatted = "\n\nPrevious conversation context:\n"
    for msg in history[-3:]:
        formatted += f"User: {msg['user']}\n"
        formatted += f"Assistant: {msg['assistant']}\n"
    
    return formatted

# -----------------------------
# Query Expansion with Legal Focus
# -----------------------------
def expand_query_legal(query: str) -> List[str]:
    """Expand query with legal-specific variations"""
    variations = [query]
    query_lower = query.lower()
    
    # Legal synonyms
    legal_synonyms = {
        'killed': ['murder', 'homicide', 'death', 'killing', 'killed'],
        'baby': ['infant', 'newborn', 'child', 'minor'],
        'punishment': ['penalty', 'sentence', 'punishment', 'sanction'],
        'section': ['article', 'section', 'provision', 'clause'],
        'how much': ['what is', 'duration', 'length', 'period'],
        'day': ['days', 'day', 'daily']
    }
    
    # Replace with synonyms
    for key, synonyms in legal_synonyms.items():
        if key in query_lower:
            for syn in synonyms[:2]:  # Limit to 2 synonyms per term
                variant = query_lower.replace(key, syn)
                variations.append(variant)
    
    # Extract core concepts
    # Remove question words
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'be', 'will', 'much']
    words = query_lower.split()
    core_words = [w for w in words if w not in question_words and len(w) > 2]
    
    if len(core_words) >= 2:
        # Core concept query
        variations.append(" ".join(core_words))
        
        # Reordered core concepts
        if len(core_words) >= 3:
            variations.append(f"{core_words[-1]} {core_words[0]} {core_words[1]}")
    
    # Limit variations
    return list(set(variations))[:5]

# -----------------------------
# Load documents with Agentic Chunking
# -----------------------------
def load_documents(folder=DATA_FOLDER):
    docs = []

    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 Created {folder} folder. Please add documents there.")
        return docs

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            if file.endswith(".pdf"):
                print(f"\n📄 Loading PDF: {file}")
                reader = PdfReader(path)
                full_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                
                # Use agentic chunking
                if USE_AGENTIC_CHUNKING and agentic_chunker:
                    chunks = agentic_chunker.chunk_document(full_text, file)
                    docs.extend(chunks)
                else:
                    # Fallback to simple chunking
                    simple_chunks = simple_chunk_text(full_text)
                    for idx, chunk in enumerate(simple_chunks):
                        if chunk.strip():
                            docs.append({
                                "content": chunk.strip(),
                                "summary": chunk[:200],
                                "source": f"{file} (Chunk {idx + 1})",
                                "keywords": agentic_chunker.extract_keywords(chunk) if agentic_chunker else []
                            })

            elif file.endswith((".xlsx", ".xls")):
                print(f"\n📊 Loading Excel: {file}")
                df = pd.read_excel(path)
                for idx, row in df.iterrows():
                    row_text = " ".join(str(v) for v in row.values if pd.notna(v))
                    if row_text.strip():
                        docs.append({
                            "content": row_text,
                            "summary": row_text[:200],
                            "source": f"{file} (Row {idx + 1})",
                            "keywords": agentic_chunker.extract_keywords(row_text) if agentic_chunker else []
                        })

            elif file.endswith(".json"):
                print(f"\n📋 Loading JSON: {file}")
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        if "question" in item and "answer" in item:
                            content = f"Question: {item['question']}\nAnswer: {item['answer']}"
                            docs.append({
                                "content": content,
                                "summary": content[:200],
                                "source": file,
                                "keywords": agentic_chunker.extract_keywords(content) if agentic_chunker else []
                            })

        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    return docs

# -----------------------------
# Initialize system
# -----------------------------
def initialize_system():
    global documents, doc_embeddings, embedder, reranker

    print("\n" + "="*70)
    print("🚀 Initializing LegalMind with AGENTIC CHUNKING")
    print("="*70)
    
    documents = load_documents()

    if not documents:
        print("⚠️ No documents found.")
        return False

    print(f"\n✅ Total chunks created: {len(documents)}")

    print("\n🧠 Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    embedder.eval()

    print("🎯 Loading re-ranking model (ms-marco-MiniLM)...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE)

    # Create embeddings for content + summary
    print("\n🔢 Creating embeddings...")
    contents_with_summary = []
    for d in documents:
        # Combine content and summary for richer embeddings
        combined = f"{d.get('summary', '')} {d['content']}"
        contents_with_summary.append(combined)

    with torch.no_grad():
        doc_embeddings = embedder.encode(
            contents_with_summary,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=DEVICE,
            normalize_embeddings=True
        )

    print("✅ System ready!")
    print("="*70 + "\n")
    return True

# -----------------------------
# Hybrid Search
# -----------------------------
def keyword_search(query, documents, top_k=10):
    """Enhanced keyword search with legal patterns"""
    query_keywords = set(agentic_chunker.extract_keywords(query, top_n=15) if agentic_chunker else query.split())
    
    scores = []
    for doc in documents:
        doc_keywords = set(doc.get("keywords", []))
        
        # Calculate overlap
        overlap = len(query_keywords & doc_keywords)
        
        # Bonus for exact phrase matches in content
        exact_bonus = 0
        query_lower = query.lower()
        content_lower = doc["content"].lower()
        
        # Check for important query terms
        for word in query_keywords:
            if len(word) > 3 and word.lower() in content_lower:
                exact_bonus += 0.1
        
        score = (overlap / max(len(query_keywords), 1)) + exact_bonus
        scores.append(min(score, 1.0))  # Cap at 1.0
    
    return np.array(scores)

def hybrid_search(query, top_k=25):
    """Hybrid search with query expansion"""
    query_variations = expand_query_legal(query)
    
    print(f"  🔍 Query variations: {len(query_variations)}")
    for qv in query_variations[:3]:
        print(f"     - {qv}")
    
    # Semantic search
    all_scores = []
    with torch.no_grad():
        for q_var in query_variations:
            q_embedding = embedder.encode(
                [q_var],
                convert_to_tensor=True,
                device=DEVICE,
                normalize_embeddings=True
            )
            
            if DEVICE == "cuda":
                similarities = torch.mm(q_embedding, doc_embeddings.T).squeeze(0)
                scores = similarities.cpu().numpy()
            else:
                scores = cosine_similarity(
                    q_embedding.cpu().numpy(),
                    doc_embeddings.cpu().numpy()
                )[0]
            
            all_scores.append(scores)
    
    # Average semantic scores
    semantic_scores = np.mean(all_scores, axis=0)
    
    # Keyword search
    keyword_scores = keyword_search(query, documents, top_k)
    
    # Combine: 60% semantic + 40% keyword (keyword more important for legal)
    combined_scores = 0.6 * semantic_scores + 0.4 * keyword_scores
    
    # Get top candidates
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    top_scores = combined_scores[top_indices]
    
    return top_indices, top_scores

# -----------------------------
# Re-ranking
# -----------------------------
def rerank_results(query, candidate_indices, candidate_scores, top_k=10):
    """Re-rank with cross-encoder"""
    candidates = [(documents[idx]["content"], float(candidate_scores[i])) 
                  for i, idx in enumerate(candidate_indices)]
    
    # Cross-encoder
    pairs = [[query, doc] for doc, _ in candidates]
    rerank_scores = reranker.predict(pairs)
    
    # Combine: 30% original + 70% rerank
    final_scores = 0.3 * candidate_scores + 0.7 * rerank_scores
    
    # Sort
    sorted_indices = np.argsort(final_scores)[::-1]
    
    return candidate_indices[sorted_indices][:top_k], final_scores[sorted_indices][:top_k]

# -----------------------------
# RAG Query
# -----------------------------
def rag_query(question, session_id, top_k=7):
    if embedder is None or doc_embeddings is None:
        return {
            "answer": "System not initialized.",
            "sources": []
        }

    print(f"\n{'='*70}")
    print(f"🔍 QUERY: {question}")
    print(f"{'='*70}")
    
    # Hybrid search
    candidate_indices, candidate_scores = hybrid_search(question, top_k=25)
    
    print(f"  ✓ Retrieved {len(candidate_indices)} candidates")
    print(f"  ✓ Top score: {candidate_scores[0]:.3f}")
    
    # Re-rank
    ranked_indices, ranked_scores = rerank_results(
        question, 
        candidate_indices[:20],
        candidate_scores[:20],
        top_k=top_k
    )
    
    print(f"  ✓ Re-ranked to top {top_k}")
    print(f"  ✓ Best re-rank score: {ranked_scores[0]:.3f}")
    
    # Check threshold
    if ranked_scores[0] < RERANK_THRESHOLD:
        print(f"  ⚠️ Best score {ranked_scores[0]:.3f} below threshold {RERANK_THRESHOLD}")
        return {
            "answer": "I apologize, but I could not find sufficiently relevant information in the legal documents to answer your question confidently. Please try:\n\n1. Rephrasing your question with different words\n2. Adding more specific details (section numbers, legal terms)\n3. Breaking complex questions into simpler parts\n\nOr ask about topics that are explicitly covered in the uploaded documents.",
            "sources": []
        }

    # Gather contexts
    contexts = []
    sources = []

    for idx, score in zip(ranked_indices, ranked_scores):
        score_val = float(score)
        if score_val >= RERANK_THRESHOLD:
            doc = documents[idx]
            
            # Use summary if available for context
            context_text = doc['content']
            if 'summary' in doc and doc['summary']:
                context_text = f"[Summary: {doc['summary']}]\n\n{doc['content']}"
            
            contexts.append(context_text)
            sources.append({
                "source": doc["source"],
                "score": score_val
            })

    if not sources:
        return {
            "answer": "I found some potentially relevant information, but the relevance scores are too low. Please rephrase your question with more specific legal terms.",
            "sources": []
        }

    print(f"  ✓ Using {len(sources)} high-quality sources")
    
    combined_context = "\n\n" + "="*50 + "\n\n".join(contexts)
    conversation_context = format_conversation_context(session_id)

    # Enhanced prompt
    prompt = f"""You are an expert legal AI assistant analyzing legal documents. You must provide accurate, precise answers based ONLY on the context provided.

STRICT RULES:
1. Answer using ONLY information from the Context below
2. If the Context doesn't fully answer the question, state what you found and what's missing
3. Cite specific sections, articles, or provisions mentioned in the Context
4. Use precise legal language
5. If uncertain or if Context is insufficient, explicitly state this
6. Never infer penalties, timeframes, or legal consequences not stated in the Context

{conversation_context}

CONTEXT FROM LEGAL DOCUMENTS:
{combined_context}

QUESTION: {question}

Provide a clear, accurate answer based solely on the Context above. If the Context contains relevant information, answer directly. If not, state what information is missing.

ANSWER:"""

    print(f"  🤖 Generating response...\n")
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_ctx": NUM_CTX,
                "num_predict": 1024,
                "temperature": 0.1,  # Very low for legal accuracy
                "top_k": 5,
                "top_p": 0.5,
                "repeat_penalty": 1.4,
                "num_thread": os.cpu_count(),
            }
        )
        
        answer = response["message"]["content"]
        
        result = {
            "answer": answer,
            "sources": sources
        }
        
        add_to_conversation(session_id, question, answer)
        
        print(f"  ✅ Response generated\n")
        print(f"{'='*70}\n")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Ollama error: {e}\n")
        return {
            "answer": f"Error generating response: {str(e)}. Please ensure Ollama is running with 'ollama run llama3'.",
            "sources": sources
        }

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
    return jsonify({
        "device": DEVICE,
        "documents": len(documents),
        "model": MODEL_NAME,
        "active_sessions": len(conversation_histories),
        "features": ["agentic_chunking", "query_expansion", "hybrid_search", "cross_encoder_reranking"]
    })

@app.route("/clear-conversation", methods=["POST"])
def clear_conversation_route():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    clear_conversation(session_id)
    return jsonify({"status": "Conversation cleared", "session_id": session_id})

@app.route("/get-conversation", methods=["POST"])
def get_conversation():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    history = get_conversation_history(session_id)
    return jsonify({"history": history, "session_id": session_id})

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    success = initialize_system()
    
    if not success:
        print("\n⚠️ System initialization failed. Please add documents to the 'data' folder.")
        exit(1)

    print("\n" + "="*70)
    print("🌐 LEGALMIND AGENTIC RAG SERVER")
    print("="*70)
    print(f"📍 URL: http://localhost:5000")
    print(f"🦙 Ollama: Ensure 'ollama run llama3' is running")
    print(f"⚡ Device: {DEVICE}")
    print(f"📚 Documents: {len(documents)} agentic chunks")
    print(f"\n🚀 AGENTIC FEATURES ENABLED:")
    print(f"   ✓ Proposition-based chunking")
    print(f"   ✓ Contextual summaries")
    print(f"   ✓ Legal-aware query expansion")
    print(f"   ✓ Hybrid search (semantic + keyword)")
    print(f"   ✓ Cross-encoder re-ranking")
    print(f"   ✓ Multi-turn conversation memory")
    print("="*70 + "\n")

    app.run(debug=False, port=5000, host="0.0.0.0", threaded=True)