# 🗺️ Complete Pedagogical Journey: Book → Scripts → Concepts

🌍 **English** | 📖 [Français](../fr/PEDAGOGICAL_JOURNEY.md) | 🇪🇸 [Español](../es/PEDAGOGICAL_JOURNEY.md) | 🇧🇷 [Português](../pt/PEDAGOGICAL_JOURNEY.md) | 🇸🇦 [العربية](../ar/PEDAGOGICAL_JOURNEY.md)

> **Complete Guide** to navigate the "The Mechanics of LLMs" project  
> Detailed correspondence: book chapters ↔ Python scripts ↔ practical concepts

---

## 📍 Getting Started...

### If you're new ✨

```
1. Read this page (you are here)
   ↓
2. Check README.md (general navigation)
   ↓
3. Open PEDAGOGICAL_JOURNEY.md (scripts guide)
   ↓
4. Run your first script
```

### If you've already read the book 📖

```
1. Find your chapter below
   ↓
2. Click on the corresponding script
   ↓
3. Execute and experiment
```

### If you want to code immediately 💻

```
1. Go directly to: 09_mini_assistant_complet.py
   ↓
2. Read: INDEX_SCRIPT_09.md (architecture)
   ↓
3. Understand then adapt
```

---

## 📚 Pathway By Book Chapter

### Chapter 1: NLP Introduction

**Book Content:**
- What is NLP?
- History: from rules to learning to LLMs
- Where we are in 2025

**Code Link:**
- ❌ No dedicated script (theoretical)
- ✅ Continue to Chapter 2

---

### Chapter 2: Text Representation & Sequential Models

**Book Content:**
- How do models see text?
- Tokens and tokenizers (BPE, WordPiece, SentencePiece)
- Impact on sequence length
- RNNs, LSTMs, GRUs (the ancestors)

**👉 Corresponding Script:**

#### [`01_tokenization_embeddings.py`](../../01_tokenization_embeddings.py)

**What you learn by running:**
```python
python 01_tokenization_embeddings.py
```

- Tokenization with different tokenizers
- Impact of tokenization on sequence length
- Differences French vs English
- Embeddings and their dimensions
- Computational cost based on tokens

**Key Concepts Demonstrated:**
- BPE (Byte Pair Encoding) tokenizers
- Vocabulary and subwords
- Tokens ↔ O(n²) attention cost relationship

**Runtime:** ~5 seconds  
**Prerequisites:** Python, `transformers`

---

### Chapter 3: Transformer Architecture

**Book Content:**
- The invention of the attention mechanism
- Self-attention and multi-head attention
- Encoder-decoder structure
- Positional encoding
- The position problem

**👉 Corresponding Script:**

#### [`02_multihead_attention.py`](../../02_multihead_attention.py)

**What you learn by running:**
```python
python 02_multihead_attention.py
```

- Architecture of an attention layer
- Q, K, V projections (Query, Key, Value)
- Attention score computation
- Multi-head: how each head focuses differently
- Visualization: who attends to whom?

**Key Concepts Demonstrated:**
- Softmax and score normalization
- Embedding dimension vs number of heads
- Each head learns different relationships

**Runtime:** ~2 seconds  
**Prerequisites:** Python, `numpy`

---

### Chapters 4-8: Architecture, Optimization, Pre-training

**Book Content:**
- Ch. 4: Transformer-derived models (BERT, GPT, T5...)
- Ch. 5: Architecture optimization (linear attention, RoPE...)
- Ch. 6: MoE architecture (Mixture of Experts)
- Ch. 7: LLM pre-training
- Ch. 8: Training optimizations (gradient accumulation...)

**Code Link:**
- 📖 Theoretical + concepts
- ⚡ Integrated in Script 03 (temperature during pre-training)
- 🏆 Enhanced in Script 09 (mini-assistant)

---

### Chapter 9: Supervised Fine-tuning (SFT)

**Book Content:**
- From prediction to assistance
- Supervised fine-tuning (SFT)
- Quality over quantity
- Evaluation of fine-tuned models
- Case study: adapting LLaMA 7B

**👉 Corresponding Bonus Script:**

#### [`08_lora_finetuning_example.py`](../../08_lora_finetuning_example.py) 🎁

**What you learn by running:**
```python
python 08_lora_finetuning_example.py
```

- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Comparison: full fine-tuning vs LoRA
- Efficiency in terms of memory/speed
- Real case SNCF (from book text)

**Key Concepts Demonstrated:**
- Adapt models without retraining everything
- Memory vs quality trade-off
- Additional parameters vs gain

**Runtime:** ~3 seconds  
**Prerequisites:** Python, `numpy` (demo without external LLM)

---

### Chapter 11: Generation & Inference Strategies

**Book Content:**
- Prompting: guiding the model through examples
- Temperature control
- Sampling strategies (top-k, top-p, nucleus sampling)
- Optimize latency: KV-cache, speculation

**👉 Corresponding Scripts:**

#### [`03_temperature_softmax.py`](../../03_temperature_softmax.py)

**What you learn by running:**
```python
python 03_temperature_softmax.py
```

- Effect of temperature on softmax
- Low T = deterministic (greedy)
- High T = diversity (creative)
- Relationship with entropy
- Temperature effect graphs

**Key Concepts Demonstrated:**
- Softmax and probabilistic interpretation
- Temperature as scaling factor
- Determinism vs creativity trade-off

**Runtime:** ~2 seconds  
**Prerequisites:** Python, `matplotlib` (optional)

#### [`09_mini_assistant_complet.py`](../../09_mini_assistant_complet.py) 🏆

**Your first assistant with:**
- Prompting (Chain-of-Thought)
- Temperature sampling
- Generation strategies

---

### Chapter 12: Reasoning Models

**Book Content:**
- Chain-of-Thought (CoT) prompting
- Tree-of-Thought (ToT)
- Code and mathematics (reasoning demonstration)
- Reinforcement Learning (RL) for thinking

**👉 Corresponding Scripts:**

#### [`05_pass_at_k_evaluation.py`](../../05_pass_at_k_evaluation.py)

**What you learn by running:**
```python
python 05_pass_at_k_evaluation.py
```

- Pass@k metric for evaluation
- Pass^k (different from Pass@k)
- Why these metrics for reasoning?
- Empirics on code tasks

**Key Concepts Demonstrated:**
- Evaluation beyond simple accuracy
- Multiple attempts vs single attempt
- Metrics specific to reasoning

**Runtime:** ~1 second  
**Prerequisites:** Python, `numpy`

---

### Chapter 13: Augmented Systems & Agents (RAG)

**Book Content:**
- RAG: Retrieval-Augmented Generation
- The M:N integration problem
- Under the hood: technical implementation
- Progressive tool discovery

**👉 Corresponding Scripts:**

#### [`04_rag_minimal.py`](../../04_rag_minimal.py)

**What you learn by running:**
```python
python 04_rag_minimal.py
```

- Minimal RAG pipeline (understand the steps)
- Cosine similarity for retrieval
- Context augmentation
- Quality vs latency

**Key Concepts Demonstrated:**
- Document chunking
- Embeddings and search
- Hallucination reduction

**Runtime:** ~3 seconds  
**Prerequisites:** Python, `numpy`, `scikit-learn`

#### [`07_llamaindex_rag_advanced.py`](../../07_llamaindex_rag_advanced.py) 🎁

**What you learn by running:**
```python
python 07_llamaindex_rag_advanced.py
```

- Complete RAG framework (LlamaIndex)
- 6 phases: Load → Index → RAG → Chat → Eval → Export
- Document ingestion
- Chat with persistence
- Automatic evaluation

**Key Concepts Demonstrated:**
- Production RAG architecture
- Indexing strategies
- Persistence layer

**Runtime:** ~5 seconds  
**Prerequisites:** Python (demo), optional: `llama-index`, `openai`

---

### Chapter 14: Agentic Protocols (MCP)

**Book Content:**
- Agents: autonomy and decision
- Agent definition
- Patterns: ReAct, Tool Use, Function Calling
- Model Context Protocol (MCP)
- Limitations and difficulties

**👉 Corresponding Bonus Script:**

#### [`06_react_agent_bonus.py`](../../06_react_agent_bonus.py) 🎁

**What you learn by running:**
```python
python 06_react_agent_bonus.py
```

- ReAct pattern (Reasoning + Acting)
- Generic framework for creating agents
- Tool registration (tool registration)
- 3 example tools
- Loop: think → act → observe

**Key Concepts Demonstrated:**
- Autonomous agent loop
- Decision making
- Tool composition

**Runtime:** ~4 seconds  
**Prerequisites:** Python, `numpy`

**See also:** [REACT_AGENT_INTEGRATION.md](REACT_AGENT_INTEGRATION.md)

---

### Chapter 15: Critical Evaluation of Agentic Flows

**Book Content:**
- The measurement challenge
- Evaluating agents: from words to deeds
- Quantitative & qualitative metrics
- Case studies

**👉 Complete Integrator Script:**

#### [`09_mini_assistant_complet.py`](../../09_mini_assistant_complet.py) 🏆

**What you learn by running:**
```python
python 09_mini_assistant_complet.py
```

- Evaluation of a complete system
- Metrics: BLEU, embedding similarity, coherence
- Traces and debugging
- Iterative improvement

**Key Concepts Demonstrated:**
- Multi-criteria evaluation
- Feedback loops
- Execution quality

**Runtime:** ~10 seconds  
**Prerequisites:** Python (all provided)

**See also:**
- [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md) - Architecture
- [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md) - Quick start

**CONGRATULATIONS** 🎉 You've completed the journey!

---

## 🎯 Accelerated Pathways

### "I want to understand LLMs quickly" (2-3 hours)

```
Read Chapters 1-3        (30 min)
   ↓
Run Scripts 01-02        (15 min)
   ↓
Read Chapters 11-12      (45 min)
   ↓
Run Scripts 03-05        (30 min)
   ↓
Read Chapters 13-14      (45 min)
   ↓
Run Script 09            (15 min)
```

**Result:** Solid understanding of key concepts ✅

### "I want to code a RAG + Agents application" (4-6 hours)

```
Understand RAG           (Chapter 13)  (30 min)
   ↓
Run Scripts 04, 07       (30 min)
   ↓
Understand Agents        (Chapter 14)  (30 min)
   ↓
Run Script 06            (20 min)
   ↓
Study Script 09          (60 min)
   ↓
Adapt for your case      (variable)
```

**Result:** Functional RAG + Agents application ✅

---

## 📝 Notes

- **No GPU required**: all scripts run on CPU (slower)
- **Minimal dependencies**: only `numpy`, `torch`, `transformers`, `scikit-learn`
- **Educational code**: prioritizes clarity over optimization
- **Compatible Python 3.9+**
- **Bonus scripts** demonstrate advanced concepts, work without external LLM (simulation mode)

---

**Happy learning! 🎓**
