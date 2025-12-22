"""
BONUS Script 2: LlamaIndex Framework for Advanced RAG (Chapter 13)

Demonstrates how to build a complete RAG system:
- Document loading and parsing (embedded in the script)
- Indexing and embedding (simulated, deterministic)
- Retrieval by cosine similarity
- Query Engine with context augmentation
- Chat with context persistence (conversational memory)
- Quality evaluation (Precision, Recall, F1)
- Export results to JSON

Standalone mode (no dependencies required):
    python 07_llamaindex_rag_advanced.py
    → Uses simulated embeddings (deterministic)

Advanced mode (with real LlamaIndex):
    pip install llama-index openai python-dotenv
    python 07_llamaindex_rag_advanced.py
    → Attempts LlamaIndex integration if available

Concepts covered:
    - RAG (Retrieval-Augmented Generation)
    - Document parsing and indexing
    - Vector similarity search
    - Query augmentation with context
    - Conversation with memory
"""

import os
from typing import List, Optional
from datetime import datetime
import json

# ============================================================================
# PHASE 0: Configuration & Imports (with fallback)
# ============================================================================

def setup_environment():
    """Configure the environment with dependency management."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not configured")
        print("   Use: export OPENAI_API_KEY=sk-... (Linux/Mac)")
        print("   Or: $env:OPENAI_API_KEY='sk-...' (PowerShell Windows)")
        print("   Or put it in .env")
    return api_key


# ============================================================================
# PHASE 1: Document Data (Embedded)
# ============================================================================

SAMPLE_DOCUMENTS = [
    {
        "title": "Transformers: Architecture",
        "content": """
The Transformer is a deep network architecture based on the attention mechanism.
Unlike RNNs that process tokens sequentially, Transformers process all tokens
in parallel.

Main structure:
1. Embedding: Converts tokens to dense vectors
2. Positional Encoding: Adds position information
3. Multi-Head Attention: Captures complex relationships between tokens
4. Feed-Forward Networks: Non-linear transformations
5. Layer Normalization: Stabilizes training

Key advantage: Full parallelization → faster training and inference.
Complexity: O(n²) in tokens, which limits context length.
"""
    },
    {
        "title": "Multi-Head Attention",
        "content": """
Multi-head attention allows the model to observe different higher-order
representations of the same input-output space.

Formula:
Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V

Where:
- Q (Query): Query representation
- K (Key): Key representation to query
- V (Value): Values to retrieve
- d_k: Key dimension (scaling)

With 8 heads (h=8), the model captures:
- Syntactic relationship (in one head)
- Semantic relationship (in another head)
- Position (in a third head)
- etc.

Advantage: Better representation than a single head.
"""
    },
    {
        "title": "Fine-tuning and Adaptation",
        "content": """
Fine-tuning = adapting a pre-trained model to a specific task.

Strategies:
1. Full Fine-tuning: Update all parameters (memory expensive)
2. LoRA (Low-Rank Adaptation): Add low-rank matrices
   - Additional parameters: d·r·2 instead of d²
   - Where d=original dimension, r=rank
   - Example: 7B model → ~8M params (0.1%)
3. QLoRA: LoRA on quantized model (4-bit)
   - Memory: 7B model → ~2GB instead of 28GB
   - Acceptable speed for inference

Best practice:
- Small dataset (<10K examples): LoRA
- Medium dataset (10K-100K): LoRA with learning rate 5e-4
- Large dataset (>100K): Full fine-tuning if possible
"""
    }
]


# ============================================================================
# PHASE 2: Document Builder (no dependencies)
# ============================================================================

class SimpleDocument:
    """Simplified document (fallback if LlamaIndex unavailable)."""
    
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        import hashlib
        return hashlib.md5(
            f"{self.content[:100]}{datetime.now()}".encode()
        ).hexdigest()[:8]
    
    def __repr__(self):
        return f"Doc({self.metadata.get('title', 'Untitled')})"


# ============================================================================
# PHASE 3: Indexing and Embedding (Simulated + Real)
# ============================================================================

class SimpleEmbedding:
    """Simulated embedding (fallback)."""
    
    @staticmethod
    def embed_text(text: str, dimensions: int = 384) -> List[float]:
        """Create a simple hash-based embedding (deterministic)."""
        import hashlib
        
        # Hash of text → seed
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        # Deterministic pseudo-RNG
        import random
        random.seed(hash_val % (2**32))
        
        # Generate "plausible" vector (normal distribution)
        embedding = [random.gauss(0, 1) for _ in range(dimensions)]
        
        # Normalize
        magnitude = sum(x**2 for x in embedding) ** 0.5
        return [x / magnitude for x in embedding]


class VectorIndex:
    """Simple vector index for retrieval."""
    
    def __init__(self, dimension: int = 384):
        self.vectors: dict = {}  # doc_id -> vector
        self.documents: dict = {}  # doc_id -> doc
        self.dimension = dimension
    
    def add_document(self, doc: SimpleDocument):
        """Add a document and its embedding."""
        embedding = SimpleEmbedding.embed_text(doc.content, self.dimension)
        self.vectors[doc.id] = embedding
        self.documents[doc.id] = doc
    
    def similarity(self, v1: List[float], v2: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(x**2 for x in v1) ** 0.5
        norm2 = sum(x**2 for x in v2) ** 0.5
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    
    def retrieve(self, query: str, top_k: int = 3) -> List[SimpleDocument]:
        """Retrieve the k most similar documents."""
        query_vec = SimpleEmbedding.embed_text(query, self.dimension)
        
        scores = [
            (doc_id, self.similarity(query_vec, vec))
            for doc_id, vec in self.vectors.items()
        ]
        
        # Sort by descending score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [self.documents[doc_id] for doc_id, _ in scores[:top_k]]


# ============================================================================
# PHASE 4: RAG Engine (Query Processing)
# ============================================================================

class SimpleRAGEngine:
    """RAG engine with context augmentation."""
    
    def __init__(self, index: VectorIndex):
        self.index = index
        self.query_history: List[dict] = []
    
    def _simulate_llm_response(self, prompt: str) -> str:
        """Simulate an LLM response (replace with OpenAI API in production)."""
        # In production: call OpenAI, Claude, etc.
        # For demo: heuristic response
        
        if "transformer" in prompt.lower():
            return """
According to the provided context, Transformers are attention-based architectures
that process all tokens in parallel, unlike RNNs.

Key points:
1. Multi-Head Attention: Captures different relationships (syntax, semantics, position)
2. Parallelization: O(n²) in tokens but faster processing
3. Positional Encoding: Adds sequence information

The original architecture (Vaswani et al., 2017) includes:
- Encoder (understand context)
- Decoder (generate response)
- Multi-head attention (8 heads by default)

Application: All modern LLMs (GPT, Claude, Llama) use this architecture.
"""
        elif "fine" in prompt.lower() or "lora" in prompt.lower():
            return """
Fine-tuning adapts a pre-trained model to a specific task.

Recommended strategies:
- LoRA: Adding low-rank matrices (memory efficient)
- QLoRA: LoRA on quantized models (4-bit, ~2GB for 7B model)
- Full fine-tuning: If very large dataset and resources available

Example: LoRA on Llama-2 7B:
- Additional parameters: ~8M (0.1% of model)
- Required memory: ~10GB (vs 28GB in full fine-tuning)
- Time: 1-2h on A100 GPU

Best practice by dataset:
- <10K examples: LoRA
- 10K-100K: LoRA + lr 5e-4
- >100K: Full fine-tuning
"""
        else:
            return "I am not certain of the answer to this question."
    
    def query(self, question: str, top_k: int = 3) -> dict:
        """
        Execute a complete RAG query.
        
        Process:
        1. Retrieve: Search for relevant documents
        2. Augment: Inject context into the prompt
        3. Generate: Call the LLM
        4. Log: Save for evaluation
        """
        # 1. Retrieval
        retrieved_docs = self.index.retrieve(question, top_k)
        
        # 2. Context augmentation
        context = "\n\n".join([
            f"[{doc.metadata.get('title', 'Untitled')}]\n{doc.content}"
            for doc in retrieved_docs
        ])
        
        augmented_prompt = f"""Context:
{context}

Question: {question}

Answer using the provided context. Be concise and actionable."""
        
        # 3. Generation (simulated or real)
        answer = self._simulate_llm_response(augmented_prompt)
        
        # 4. Logging
        result = {
            "question": question,
            "retrieved_docs": [
                {
                    "title": doc.metadata.get("title"),
                    "id": doc.id,
                    "snippet": doc.content[:100] + "..."
                }
                for doc in retrieved_docs
            ],
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        
        self.query_history.append(result)
        return result


# ============================================================================
# PHASE 5: Chat with Context Memory
# ============================================================================

class RAGChatbot:
    """Chatbot with RAG and conversational memory."""
    
    def __init__(self, rag_engine: SimpleRAGEngine, memory_size: int = 5):
        self.rag_engine = rag_engine
        self.memory_size = memory_size
        self.conversation_history: List[dict] = []
    
    def _build_conversation_context(self) -> str:
        """Build context from history."""
        if not self.conversation_history:
            return "No history."
        
        history_text = "\n".join([
            f"Q: {turn['question']}\nA: {turn['answer'][:200]}..."
            for turn in self.conversation_history[-self.memory_size:]
        ])
        return history_text
    
    def chat(self, user_message: str) -> dict:
        """Process a user message with conversational context."""
        
        # Enrich the question with history
        conversation_context = self._build_conversation_context()
        enriched_question = f"""History:
{conversation_context}

New question: {user_message}"""
        
        # Use the RAG engine
        rag_result = self.rag_engine.query(enriched_question)
        
        # Add to history
        self.conversation_history.append({
            "question": user_message,
            "answer": rag_result["answer"],
            "timestamp": rag_result["timestamp"]
        })
        
        return {
            "user_message": user_message,
            "retrieved_docs": rag_result["retrieved_docs"],
            "response": rag_result["answer"],
            "turn_number": len(self.conversation_history)
        }


# ============================================================================
# PHASE 6: Quality Evaluation
# ============================================================================

class RAGEvaluator:
    """Evaluate the quality of a RAG system."""
    
    @staticmethod
    def evaluate_retrieval(
        query: str,
        retrieved_docs: List[SimpleDocument],
        expected_doc_ids: List[str]
    ) -> dict:
        """
        Evaluate retrieval relevance.
        
        Metrics:
        - Precision@k: % of retrieved docs that are relevant
        - Recall@k: % of relevant docs that were retrieved
        - MRR: Mean Reciprocal Rank (position of 1st good doc)
        """
        retrieved_ids = [doc.id for doc in retrieved_docs]
        
        # Precision
        if retrieved_ids:
            precision = len(
                set(retrieved_ids) & set(expected_doc_ids)
            ) / len(retrieved_ids)
        else:
            precision = 0
        
        # Recall
        recall = len(
            set(retrieved_ids) & set(expected_doc_ids)
        ) / max(1, len(expected_doc_ids))
        
        # F1
        f1 = 2 * (precision * recall) / max(1, precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved_count": len(retrieved_ids),
            "expected_count": len(expected_doc_ids)
        }
    
    @staticmethod
    def evaluate_generation(
        generated_answer: str,
        reference_answer: str
    ) -> dict:
        """
        Evaluate generation quality with simplified BLEU / ROUGE.
        """
        gen_tokens = set(generated_answer.lower().split())
        ref_tokens = set(reference_answer.lower().split())
        
        # Jaccard similarity
        if gen_tokens | ref_tokens:
            jaccard = len(gen_tokens & ref_tokens) / len(gen_tokens | ref_tokens)
        else:
            jaccard = 0
        
        return {
            "jaccard_similarity": jaccard,
            "generated_length": len(generated_answer),
            "reference_length": len(reference_answer)
        }


# ============================================================================
# MAIN: Complete Demonstration
# ============================================================================

def main():
    """Run a complete demonstration of RAG with LlamaIndex."""
    
    print("=" * 80)
    print("🦙 LlamaIndex RAG Advanced Demo")
    print("=" * 80)
    print()
    
    # 1. Load documents
    print("📚 Phase 1: Loading documents")
    print("-" * 80)
    documents = [
        SimpleDocument(doc["content"], {"title": doc["title"]})
        for doc in SAMPLE_DOCUMENTS
    ]
    for doc in documents:
        print(f"  ✓ {doc.metadata['title']} ({len(doc.content)} chars)")
    print()
    
    # 2. Create the index
    print("🔍 Phase 2: Creating vector index")
    print("-" * 80)
    index = VectorIndex(dimension=384)
    for doc in documents:
        index.add_document(doc)
    print(f"  ✓ Index created with {len(documents)} documents")
    print(f"  ✓ Embedding dimension: 384")
    print()
    
    # 3. Create the RAG engine
    print("⚙️  Phase 3: Initializing RAG Engine")
    print("-" * 80)
    rag_engine = SimpleRAGEngine(index)
    print("  ✓ RAG Engine ready")
    print()
    
    # 4. Execute queries
    print("💬 Phase 4: RAG Queries")
    print("-" * 80)
    
    questions = [
        "What is a Transformer?",
        "How does multi-head attention work?",
        "What is the difference between LoRA and full fine-tuning?"
    ]
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        result = rag_engine.query(question, top_k=2)
        results.append(result)
        
        print(f"📄 Retrieved documents:")
        for doc in result["retrieved_docs"]:
            print(f"   - {doc['title']} ({doc['id']})")
        
        print(f"🤖 Response:\n{result['answer'][:300]}...")
    
    print()
    
    # 5. Conversational chat
    print("💬 Phase 5: Chat with Memory")
    print("-" * 80)
    
    chatbot = RAGChatbot(rag_engine, memory_size=3)
    
    chat_messages = [
        "Tell me about Transformers",
        "And how does it work in practice?",
        "Can you compare with RNNs?"
    ]
    
    for msg in chat_messages:
        print(f"\n👤 User: {msg}")
        chat_result = chatbot.chat(msg)
        print(f"🤖 Bot: {chat_result['response'][:250]}...")
        print(f"   📄 Turn {chat_result['turn_number']}")
    
    print()
    
    # 6. Evaluation
    print("📊 Phase 6: Quality Evaluation")
    print("-" * 80)
    
    evaluator = RAGEvaluator()
    
    # Evaluate retrieval of first query
    first_result = results[0]
    retrieved_doc_ids = [d["id"] for d in first_result["retrieved_docs"]]
    
    # Assume "relevant" docs are all those containing "Transformer"
    expected_docs = [
        d.id for d in documents
        if "transformer" in d.content.lower()
    ]
    
    # Create SimpleDocument objects with correct IDs
    retrieved_docs_objects = [
        documents[i] for i, d in enumerate(documents) 
        if d.id in retrieved_doc_ids
    ]
    
    retrieval_eval = evaluator.evaluate_retrieval(
        questions[0],
        retrieved_docs_objects,
        expected_docs
    )
    
    print("Retrieval Evaluation (Q1):")
    print(f"  - Precision@2: {retrieval_eval['precision']:.2%}")
    print(f"  - Recall@2:    {retrieval_eval['recall']:.2%}")
    print(f"  - F1:          {retrieval_eval['f1']:.2%}")
    
    print()
    
    # 7. Statistical summary
    print("📈 Final Statistics")
    print("-" * 80)
    print(f"  - Queries processed: {len(results)}")
    print(f"  - Conversational turns: {len(chatbot.conversation_history)}")
    print(f"  - Documents in index: {len(index.documents)}")
    print(f"  - Query history saved: {len(rag_engine.query_history)} entries")
    
    print()
    print("=" * 80)
    print("✅ Demo completed!")
    print("=" * 80)
    
    # 8. Optional export
    print()
    print("💾 Export results?")
    export_results = {
        "timestamp": datetime.now().isoformat(),
        "queries": results,
        "chat_history": chatbot.conversation_history,
        "statistics": {
            "total_queries": len(results),
            "total_turns": len(chatbot.conversation_history),
            "documents_indexed": len(index.documents)
        }
    }
    
    output_file = "rag_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results exported to: {output_file}")


# ============================================================================
# Integration with LlamaIndex (Advanced)
# ============================================================================

def integration_llamaindex_example():
    """
    Example integration with the real LlamaIndex library.
    Run if 'pip install llama-index' is installed.
    """
    
    print("\n" + "=" * 80)
    print("🦙 Real LlamaIndex Integration")
    print("=" * 80)
    print()
    
    try:
        from llama_index.core import Document, VectorStoreIndex, Settings
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
        
        print("✓ LlamaIndex imported successfully!")
        print()
        
        # Configuration
        Settings.llm = OpenAI(model="gpt-4")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # Load documents
        docs = [
            Document(text=doc["content"], metadata={"title": doc["title"]})
            for doc in SAMPLE_DOCUMENTS
        ]
        
        # Create the index
        index = VectorStoreIndex.from_documents(docs)
        
        # Query engine
        query_engine = index.as_query_engine()
        
        # Query
        response = query_engine.query("What is a Transformer?")
        print(f"LlamaIndex Response:\n{response}")
        
        print()
        print("✓ Real integration functional!")
        
    except ImportError:
        print("⚠️  LlamaIndex not installed")
        print("Installation: pip install llama-index openai")
        print()
        print("With LlamaIndex, you can:")
        print("  - Load different formats (PDF, Word, HTML, etc.)")
        print("  - Use real embeddings (OpenAI, Ollama, local)")
        print("  - Execute advanced operations (table extraction, etc.)")
        print("  - Persist indexes (for reuse)")
        print()


if __name__ == "__main__":
    # Run the main demo
    main()
    
    # Try real integration if available
    integration_llamaindex_example()
