# Practical Scripts: Experimenting with LLM Concepts

🌍 **English** | 📖 [Français](../fr/README.md) | 🇪🇸 [Español](../es/README.md) | 🇧🇷 [Português](../pt/README.md) | 🇸🇦 [العربية](../ar/README.md)

Collection of **10 executable Python scripts** to experiment with the key concepts from the book **"The Mechanics of LLMs"**.

> 📚 **About** : These scripts accompany the book chapters. See [Pedagogical Journey](PEDAGOGICAL_JOURNEY.md) for detailed correspondences.

**📕 Buy the Book:**
- **Paperback** : [Amazon](https://amzn.eu/d/3oREERI)
- **Kindle** : [Amazon](https://amzn.eu/d/b7sG5iw)

---

## 📋 Scripts Overview

| # | Script | Chapter(s) | Concepts | Status |
|---|--------|-----------|----------|--------|
| 1 | `01_tokenization_embeddings.py` | 2 | Tokenization, sequence length impact | ✅ |
| 2 | `02_multihead_attention.py` | 3 | Self-attention, multi-head, attention weights | ✅ |
| 3 | `03_temperature_softmax.py` | 7, 11 | Temperature, softmax, entropy | ✅ |
| 4 | `04_rag_minimal.py` | 13 | RAG pipeline, retrieval, cosine similarity | ✅ |
| 5 | `05_pass_at_k_evaluation.py` | 12 | Pass@k, Pass^k, model evaluation | ✅ |
| 🎁 6 | `06_react_agent_bonus.py` | 14, 15 | **ReAct Agents, tool registration, MCP** | ✅ BONUS |
| 🎁 7 | `07_llamaindex_rag_advanced.py` | 13, 14 | **Advanced RAG, indexing, persistent chat** | ✅ BONUS |
| 🎁 8 | `08_lora_finetuning_example.py` | 9, 10 | **LoRA, QLoRA, fine-tuning comparison** | ✅ BONUS |
| 🏆 **9** | `09_mini_assistant_complet.py` | **11-15** | **🎯 Final Integrator Project** | ✅ FLAGSHIP |
| 🎁 10 | `10_activation_steering_demo.py` | 10 | **Activation Steering, 3SO, concept vectors** | ✅ BONUS |

---

## � Detailed Script Descriptions

### 📌 Script 01: Tokenization and Embeddings
**File:** `01_tokenization_embeddings.py` | **Chapter:** 2

**What the script does:**
- Loads a tokenizer (GPT-2 or LLaMA-2) and analyzes different texts
- Compares token counts between French and English
- Demonstrates sequence length impact on computational cost

**What you learn:**
- How text is split into tokens (BPE, WordPiece)
- Why "Bonjour" may become 2-3 tokens while "Hello" is just one
- The direct impact: more tokens = higher O(n²) cost for attention

**Expected output:**
```
Text: L'IA est utile
  Token count: 5
  Tokens: ['L', "'", 'IA', 'est', 'utile']
```

---

### 📌 Script 02: Multi-Head Attention
**File:** `02_multihead_attention.py` | **Chapter:** 3

**What the script does:**
- Simulates a multi-head attention layer with PyTorch tensors
- Computes Q, K, V projections and attention weights
- Shows how each head "looks" at the sentence differently

**What you learn:**
- The Q (Query), K (Key), V (Value) mechanism
- Why multiple heads capture different dependencies
- That attention weights always sum to 1 (probability distribution)

**Expected output:**
```
Sentence: The cat sleeps well
Head 1: Attention weights from 'cat' → 'sleeps': 0.42
Head 2: Attention weights from 'cat' → 'The': 0.38
```

---

### 📌 Script 03: Temperature and Softmax
**File:** `03_temperature_softmax.py` | **Chapters:** 7, 11

**What the script does:**
- Applies softmax with different temperatures (0.1, 0.5, 1.0, 2.0)
- Calculates Shannon entropy for each distribution
- Generates graphs (if matplotlib is installed)

**What you learn:**
- T < 1: "sharp" distribution → deterministic generation (greedy)
- T > 1: "flat" distribution → creative/diverse generation
- Entropy increases with temperature (more uncertainty)

**Expected output:**
```
Temperature 0.5: Token 'Paris' = 85% (sharp, deterministic)
Temperature 2.0: Token 'Paris' = 35% (flat, creative)
```

---

### 📌 Script 04: Minimal RAG
**File:** `04_rag_minimal.py` | **Chapter:** 13

**What the script does:**
- Creates a mini knowledge base (7 documents about LLMs)
- Vectorizes documents with TF-IDF
- Performs cosine similarity search
- Simulates generation augmented by retrieved context

**What you learn:**
- The complete RAG pipeline: Retrieval → Augmentation → Generation
- How cosine similarity finds relevant documents
- Why RAG enables answering questions about private data

**Expected output:**
```
Question: "How does attention work in the Transformer?"
→ Retrieved documents: [doc_1: 0.72, doc_4: 0.65]
→ Response generated with context
```

---

### 📌 Script 05: Pass@k Evaluation
**File:** `05_pass_at_k_evaluation.py` | **Chapter:** 12

**What the script does:**
- Simulates 100 generation attempts with 30% success rate
- Calculates Pass@k (at least 1 success in k attempts)
- Calculates Pass^k (all k attempts succeed)

**What you learn:**
- Pass@k = 1 - (1-p)^k: probability of at least one success
- Pass^k = p^k: probability all succeed (very strict)
- Why Pass@10 ≈ 97% even with p=30% (you get 10 chances)

**Expected output:**
```
Pass@1  = 30%  (chance with 1 attempt)
Pass@5  = 83%  (chance with 5 attempts)
Pass@10 = 97%  (almost certain with 10 attempts)
```

---

### 🎁 Script 06: ReAct Agent (BONUS)
**File:** `06_react_agent_bonus.py` | **Chapters:** 14, 15

**What the script does:**
- Implements a mini autonomous agent framework
- Demonstrates the ReAct loop: Thought → Action → Observation → ...
- Includes simulated tools: calculator, web search, weather

**What you learn:**
- The ReAct pattern (Reasoning + Acting)
- How an agent decides which action to take
- Self-correction: the agent can retry if an action fails
- The foundation for understanding MCP agents (Model Context Protocol)

**Expected output:**
```
Thought: I need to calculate 15% of $250
Action: calculator(250 * 0.15)
Observation: 37.5
Final Answer: The tip is $37.50
```

---

### 🎁 Script 07: Advanced RAG with LlamaIndex (BONUS)
**File:** `07_llamaindex_rag_advanced.py` | **Chapters:** 13, 14

**What the script does:**
- Complete RAG system with document parsing
- Indexing and embeddings (simulated or real with OpenAI)
- Chat with conversational memory
- Quality evaluation (Precision, Recall, F1)

**What you learn:**
- Production RAG architecture: ingestion → indexing → retrieval → generation
- How to maintain context across multiple conversation turns
- How to evaluate RAG system quality

**Expected output:**
```
[Chat Mode]
User: What is a Transformer?
Assistant: [Context: 3 docs] A Transformer is...
User: And multi-head attention?
Assistant: [Memory: previous question + 2 docs] ...
```

---

### 🎁 Script 08: LoRA/QLoRA Fine-tuning (BONUS)
**File:** `08_lora_finetuning_example.py` | **Chapters:** 9, 10

**What the script does:**
- Compares Full Fine-tuning vs LoRA vs QLoRA (numerical calculations)
- Displays VRAM and trainable parameter savings
- Use case: adapting LLaMA-7B for a business domain (railway)

**What you learn:**
- LoRA: adds ~0.1% parameters vs full fine-tuning
- QLoRA: 4-bit quantization + LoRA = 24GB GPU instead of 140GB
- Why efficient fine-tuning democratizes LLMs

**Expected output:**
```
LLaMA-7B:
  Full Fine-tuning: 28 GB VRAM, 7B params
  LoRA (rank=8):    8 GB VRAM, 4.2M params (0.06%)
  QLoRA:            6 GB VRAM, 4.2M params + 4-bit base
```

---

### � Script 10: Activation Steering & 3SO (BONUS)
**File:** `10_activation_steering_demo.py` | **Chapter:** 10

**What the script does:**
- Demonstrates activation steering: injecting concept vectors into hidden states
- Implements contrastive activation extraction for steering vectors
- Simulates a Sparse Autoencoder (SAE) for concept decomposition
- Implements a finite state machine for 3SO (guaranteed JSON outputs)
- Compares RLHF/DPO vs Steering with detailed table

**What you learn:**
- Steering modifies activations at inference: $X_{steered} = X + (c \times V)$
- How to extract concept vectors (contrastive method, SAE)
- Impact of steering coefficient (too low → none, optimal → effective, too high → derailment)
- 3SO mathematically guarantees valid JSON syntax
- When to use alignment vs steering

**Expected output:**
```
STEP 3: Analyzing Coefficient Effect
   Coeff   Direction Δ     Perturbation    Stability
   1.0     12.5°           8.2%            ✅ stable
   5.0     45.3°           35.1%           ⚠️ moderate
   15.0    78.2°           89.4%           ❌ unstable
```

---

### �🏆 Script 09: Complete Mini-Assistant (FINAL PROJECT)
**File:** `09_mini_assistant_complet.py` | **Chapters:** 11-15

**What the script does:**
- Integrates ALL concepts: RAG + Agents + Temperature + Evaluation
- Complete system with knowledge base, retrieval, reasoning
- Interactive mode to test different questions

**What you learn:**
- How to assemble a complete AI assistant from A to Z
- Layered architecture: Data → Retrieval → Reasoning → Generation
- End-to-end evaluation of a system

**Dedicated documentation:**
- [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md): Complete architecture
- [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md): 5-min quick start
- [SCRIPT_09_MAPPING.md](SCRIPT_09_MAPPING.md): Code ↔ concept mapping

---

## �🚀 Quick Start

### 1. Create a Virtual Environment (recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Basic installation (for scripts 1-5)
pip install torch transformers numpy scikit-learn

# Full installation (with visualizations)
pip install torch transformers numpy scikit-learn matplotlib

# For bonus scripts (optional, work in demo mode without)
pip install llama-index openai python-dotenv peft bitsandbytes
```

**Note:** Bonus scripts (06, 07, 08) work **without external dependencies** in demo mode.

### 3. Run a Script

```bash
python 01_tokenization_embeddings.py
python 02_multihead_attention.py
python 03_temperature_softmax.py
python 04_rag_minimal.py
python 05_pass_at_k_evaluation.py
python 06_react_agent_bonus.py
python 07_llamaindex_rag_advanced.py
python 08_lora_finetuning_example.py
python 09_mini_assistant_complet.py    # ← Final integrator project
```

---

## 🏆 Integrator Project: Complete Mini-Assistant

**THE flagship script** : integrates ALL concepts from chapters 11-15.

- **Script:** `09_mini_assistant_complet.py`
- **Documentation:** [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md)
- **Quick Start:** [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md)
- **Architecture:** [SCRIPT_09_MAPPING.md](SCRIPT_09_MAPPING.md)

---

## 📖 Complete Documentation

- **[Pedagogical Journey](PEDAGOGICAL_JOURNEY.md)** : Chapter-by-chapter book ↔ scripts correspondence
- **[ReAct Agents](REACT_AGENT_INTEGRATION.md)** : ReAct pattern and integration
- **[LlamaIndex RAG](LLAMAINDEX_GUIDE.md)** : Advanced RAG framework

---

## 📝 Notes

- **No GPU required** : all scripts run on CPU (slower)
- **Educational code** : prioritizes clarity over optimization
- **Compatible Python 3.9+**

---

**Happy learning! 🚀**
