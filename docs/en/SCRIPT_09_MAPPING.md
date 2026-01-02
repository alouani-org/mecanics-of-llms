# 🔗 Code ↔ Concept Mapping: Script 09

🌍 **English** | 📖 [Français](../fr/SCRIPT_09_MAPPING.md) | 🇪🇸 [Español](../es/SCRIPT_09_MAPPING.md) | 🇧🇷 [Português](../pt/SCRIPT_09_MAPPING.md) | 🇸🇦 [العربية](../ar/SCRIPT_09_MAPPING.md)

> **Understand which code implements which concept**  
> Line-by-line learning guide

---

## 📍 Quick Navigation

- **📖 See: [Pedagogical Journey](PEDAGOGICAL_JOURNEY.md)** - Theory
- **🏗️ See: [Architecture](INDEX_SCRIPT_09.md)** - Structure
- **⚡ See: [Quick Start](QUICKSTART_SCRIPT_09.md)** - Run it
- **🌍 Français: [Version Française](../fr/SCRIPT_09_MAPPING.md)**

---

## 🎯 Section 1: Imports & Setup

### Concept: Environment Preparation

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
```

**What it teaches:**
- `numpy`: Numerical computing (embeddings, softmax)
- `cosine_similarity`: Computing similarity between documents
- `defaultdict`: Data structure for knowledge base
- `re`: Text processing

---

## 🎯 Section 2: Knowledge Base

### Concept: Data Storage

```python
KNOWLEDGE_BASE = {
    'doc_1': "An LLM is a large language model...",
    'doc_2': "Transformers use attention mechanisms...",
    'doc_3': "RAG combines retrieval with generation...",
    # ... more documents
}
```

**What it teaches:**
- How to store domain knowledge
- Simple dictionary structure
- Scalable to thousands of documents

---

## 🎯 Section 3: Embeddings

### Concept: Text → Vector Representation

```python
def create_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Convert text to vector using deterministic hash"""
    hash_val = hash(text)
    np.random.seed(abs(hash_val) % 2**32)
    return np.random.randn(dim)
```

**What it teaches:**
- **Real production:** Use SentenceTransformer
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('all-MiniLM-L6-v2')
  embedding = model.encode(text)
  ```
- **In this demo:** Simplified hash-based approach for speed
- **Key concept:** Text → fixed-size vector (128 dimensions)
- **Property:** Similar text → similar vectors

**Real-world analogy:**
```
Imagine: Each document is a point in 128-dimensional space
Close points = similar meaning
```

---

## 🎯 Section 4: Retrieval (RAG Part 1)

### Concept: Find Relevant Documents

```python
def retrieve_documents(query: str, k: int = 3) -> list:
    """Step 1: Embed query
       Step 2: Compare with all documents
       Step 3: Return top-k most similar
    """
    query_embedding = create_embedding(query)
    
    # Create matrix of all document embeddings
    doc_embeddings = np.array([
        create_embedding(doc) 
        for doc in KNOWLEDGE_BASE.values()
    ])
    
    # Compute cosine similarity
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1), 
        doc_embeddings
    )[0]
    
    # Get top-k
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    results = []
    for idx in top_indices:
        doc_name = list(KNOWLEDGE_BASE.keys())[idx]
        results.append({
            'doc': doc_name,
            'content': KNOWLEDGE_BASE[doc_name],
            'similarity': similarities[idx]
        })
    
    return results
```

**What it teaches:**
- **Embedding:** Convert text to vector
- **Similarity:** Cosine similarity = how aligned are two vectors?
  ```
  cosine_similarity = (A · B) / (||A|| * ||B||)
  Range: -1 (opposite) to 1 (identical)
  ```
- **Selection:** Return top-k (most similar) documents
- **Complexity:** O(n*d) where n=docs, d=dimensions

**Real-world analogy:**
```
Like a librarian:
1. Reads your question
2. Mentally compares to all books
3. Brings you the 3 most relevant books
```

---

## 🎯 Section 5: Reasoning (Chain-of-Thought)

### Concept: Structured Problem-Solving

```python
def reasoning_phase(question: str, contexts: list) -> str:
    """Think step-by-step with retrieved context"""
    
    reasoning = f"""
    Step 1: Analyze the Question
    The user is asking about: {question}
    
    Step 2: Key Concepts
    Extract main concepts from the question
    
    Step 3: Retrieve Relevant Context
    From the retrieved documents:
    """
    
    for i, ctx in enumerate(contexts, 1):
        reasoning += f"\n- From {ctx['doc']}: {ctx['content'][:100]}..."
    
    reasoning += f"""
    
    Step 4: Synthesize an Answer
    Combining the knowledge:
    - Point 1: [from context 1]
    - Point 2: [from context 2]
    - Point 3: [from context 3]
    
    Conclusion: Based on the above, we can conclude...
    """
    
    return reasoning
```

**What it teaches:**
- **Chain-of-Thought:** Breaking problem into steps
- **Context Integration:** Using retrieved documents
- **Reproducibility:** Each step is visible
- **Transparency:** Easy to debug reasoning

**Real-world analogy:**
```
Like showing your work in math:
Not just "answer: 42"
But "Step 1: ... Step 2: ... Step 3: ... Answer: 42"
```

---

## 🎯 Section 6: Generation with Temperature

### Concept: Softmax & Temperature Sampling

```python
def generate_with_temperature(
    prompt: str, 
    temperature: float = 1.0
) -> str:
    """
    Simulate token generation with temperature control
    
    Temperature:
    - 0.1: Very focused (deterministic)
    - 1.0: Balanced (normal softmax)
    - 2.0: Very creative (diverse)
    """
    
    # Simulate logits (unnormalized scores)
    prompt_hash = hash(prompt)
    np.random.seed(abs(prompt_hash) % 2**32)
    logits = np.random.randn(100) * 2
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Softmax to get probabilities
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Sample token
    selected_idx = np.random.choice(100, p=probabilities)
    
    # Generate text
    vocab = ["an", "LLM", "is", "a", "model", "that", 
             "generates", "text", "using", "neural", "networks"]
    response = " ".join([vocab[i % len(vocab)] for i in range(selected_idx % 20)])
    
    return response
```

**What it teaches:**

**Softmax Formula:**
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
Result: probability distribution (sum = 1)
```

**Temperature Effect:**
```
T = 0.1  →  [0.01, 0.98, 0.01]  ← Sharp (deterministic)
T = 1.0  →  [0.15, 0.70, 0.15]  ← Balanced
T = 2.0  →  [0.30, 0.40, 0.30]  ← Flat (diverse)
```

**Key insight:**
- Low T: Model repeats most likely token (boring)
- High T: Model explores alternatives (creative)

**Real-world analogy:**
```
Like choosing a menu item:
T=0.1: Always choose the most popular item
T=1.0: Choose popular, but sometimes try others
T=2.0: Choose randomly, explore everything equally
```

---

## 🎯 Section 7: Agent Loop (ReAct)

### Concept: Autonomous Decision-Making

```python
def agent_loop(
    initial_query: str, 
    max_turns: int = 3
) -> dict:
    """
    ReAct Pattern:
    THINK → ACT → OBSERVE → (repeat)
    """
    
    context = initial_query
    trace = []
    turn = 0
    
    while turn < max_turns:
        turn += 1
        
        # THINK: Analyze current state
        thought = f"Turn {turn}: Analyzing '{context[:50]}...'"
        trace.append(f"THINK: {thought}")
        
        # Decide: Continue or Stop?
        should_continue = turn < max_turns and len(context) < 500
        
        if not should_continue:
            trace.append("STOP: Sufficient information gathered")
            break
        
        # ACT: Retrieve documents
        documents = retrieve_documents(context, k=2)
        trace.append(f"ACT: Retrieved {len(documents)} documents")
        
        # OBSERVE: Process results
        context += f" [Retrieved: {documents[0]['doc']}]"
        trace.append(f"OBSERVE: Added context from {documents[0]['doc']}")
    
    return {
        'answer': context,
        'turns': turn,
        'trace': trace
    }
```

**What it teaches:**

**ReAct Loop:**
```
┌─────────────────────────────────┐
│ THINK (analyze state)           │
└────────────────┬────────────────┘
                 ↓
┌─────────────────────────────────┐
│ ACT (take action/retrieve)      │
└────────────────┬────────────────┘
                 ↓
┌─────────────────────────────────┐
│ OBSERVE (process results)       │
└────────────────┬────────────────┘
                 ↓
        Repeat or Stop?
```

**Key properties:**
- Autonomous: Makes decisions independently
- Observable: Each step is traced
- Iterative: Improves with each turn
- Stoppable: Knows when to stop

**Real-world analogy:**
```
Like a student solving a problem:
1. THINK: "What does the question ask?"
2. ACT: "Let me look up relevant information"
3. OBSERVE: "I found useful info, let me continue"
4. THINK: "Do I have enough to answer?"
5. ACT: "Yes, let me formulate the answer"
6. OBSERVE: "Complete!"
```

---

## 🎯 Section 8: Evaluation Metrics

### Concept: Quality Assessment

```python
def evaluate_response(response: str, context: str) -> dict:
    """Compute multiple quality metrics"""
    
    # Metric 1: Length Ratio
    length_ratio = min(len(response), 500) / 500
    
    # Metric 2: BLEU-like (vocabulary overlap)
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    overlap = len(response_words & context_words)
    vocabulary_overlap = overlap / max(len(response_words), 1)
    
    # Metric 3: Embedding Similarity
    response_emb = create_embedding(response)
    context_emb = create_embedding(context)
    similarity = cosine_similarity(
        response_emb.reshape(1, -1),
        context_emb.reshape(1, -1)
    )[0][0]
    
    # Metric 4: Coherence (token diversity)
    tokens = response.lower().split()
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    coherence = 0.5 + 0.5 * (1 - unique_ratio)  # Balanced
    
    # Metric 5: Overall Quality
    quality_score = (
        length_ratio * 0.2 +
        vocabulary_overlap * 0.3 +
        similarity * 0.25 +
        coherence * 0.25
    ) * 100
    
    return {
        'metrics': {
            'length_ratio': length_ratio,
            'vocabulary_overlap': vocabulary_overlap,
            'embedding_similarity': similarity,
            'coherence': coherence
        },
        'quality_score': quality_score,
        'interpretation': interpret_score(quality_score)
    }
```

**What it teaches:**

**Metric Types:**

1. **Length Ratio**: 0-1
   - Ensures response isn't too short/long
   
2. **BLEU Score**: 0-1
   - How many words overlap with context?
   
3. **Embedding Similarity**: -1 to 1
   - Are response and context semantically similar?
   
4. **Coherence**: 0-1
   - Does response avoid repetition?
   
5. **Overall Quality**: 0-100
   - Weighted combination of above

**Why multiple metrics?**
```
Single metric = incomplete picture
Example: A short, generic response might score high on 
         vocabulary_overlap but low on length_ratio
```

---

## 🎯 Section 9: Main Loop Integration

### Concept: Orchestrating Everything

```python
def main():
    """Combine all components"""
    
    # 1. INITIALIZE
    print("Loading knowledge base...")
    
    # 2. USER INTERACTION
    while True:
        choice = input("\n1. Ask\n2. Examples\n3. Exit\n> ")
        
        if choice == "1":
            # USER QUESTION
            query = input("Your question: ")
            
            # RETRIEVE
            contexts = retrieve_documents(query, k=3)
            
            # REASON
            reasoning = reasoning_phase(query, contexts)
            
            # GENERATE
            response = generate_with_temperature(reasoning, temp=1.0)
            
            # EVALUATE
            metrics = evaluate_response(response, query)
            
            # DISPLAY
            print(f"\nResponse: {response}")
            print(f"Quality: {metrics['quality_score']:.1f}/100")
```

**What it teaches:**
- **Pipeline Architecture:** Each step feeds into next
- **Separation of Concerns:** Each function does one thing
- **Composability:** Easy to swap components
- **Debugging:** Can test each component independently

---

## 📊 Data Flow Diagram

```
User Input
    ↓
embed_documents() → Document vectors (128-dim)
    ↓
retrieve_documents() → Top-k similar docs
    ↓
reasoning_phase() → Structured thinking
    ↓
generate_with_temperature() → Text generation
    ↓
agent_loop() → Autonomous iteration
    ↓
evaluate_response() → Quality metrics
    ↓
Output to User
```

---

## 🎓 Learning Checklist

After reading this, you should understand:

- [ ] How text becomes vectors (embeddings)
- [ ] How similarity is computed (cosine similarity)
- [ ] How documents are retrieved (k-NN search)
- [ ] How reasoning is structured (Chain-of-Thought)
- [ ] How temperature affects randomness (softmax scaling)
- [ ] How agents make decisions (ReAct loop)
- [ ] How quality is measured (multiple metrics)
- [ ] How components integrate (pipeline)

---

## 🔬 Experimentation Ideas

Try modifying:

```python
# 1. Change embedding dimension
EMBEDDING_DIM = 256  # More dimensions = more precise

# 2. Change temperature
temperature = 0.1    # More focused
temperature = 2.0    # More creative

# 3. Change k_documents
k = 5                # More context = slower but richer

# 4. Add more documents
KNOWLEDGE_BASE['doc_4'] = "Your new document..."

# 5. Change evaluation weights
quality_score = (
    length_ratio * 0.1 +
    vocabulary_overlap * 0.5 +  # More emphasis here
    similarity * 0.2 +
    coherence * 0.2
) * 100
```

---

## 📚 Further Reading

- **Chapter 11:** Temperature & Generation
- **Chapter 12:** Chain-of-Thought Reasoning
- **Chapter 13:** RAG Architecture
- **Chapter 14:** Agent Patterns (ReAct)
- **Chapter 15:** Evaluation

---

**Now you understand the code! 🎓**
