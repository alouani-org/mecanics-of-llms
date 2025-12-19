# ⚡ 5-Minute Quick Start: Script 09

> **Get running in 5 minutes**  
> No theory. Just code.

---

## 📍 Quick Navigation

- **📖 See: [Pedagogical Journey](PEDAGOGICAL_JOURNEY.md)** - Learn the concepts
- **🏗️ See: [Architecture](INDEX_SCRIPT_09.md)** - How it's built
- **🌍 Français: [Version Française](../fr/QUICKSTART_SCRIPT_09.md)**

---

## Step 1️⃣: Requirements (30 seconds)

```bash
# Already installed? You're good!
# You only need these (likely already on your system):
- Python 3.9+
- numpy
- scikit-learn (for cosine similarity)
```

**Check if installed:**
```bash
python --version
python -c "import numpy; print('numpy OK')"
python -c "from sklearn.metrics.pairwise import cosine_similarity; print('sklearn OK')"
```

---

## Step 2️⃣: Navigate & Run (1 minute)

```bash
# Go to script directory
cd c:\dev\IA-Eductation\examples

# Run the script
python 09_mini_assistant_complet.py
```

---

## Step 3️⃣: Try It Out (3 minutes)

You'll see a menu:

```
========================================
   Mini LLM Assistant - Complete Demo
========================================

Choose an option:
1. Ask a question
2. Test with examples
3. See evaluation metrics
4. Understand architecture
5. Exit

Enter your choice (1-5): 
```

**Try this:**

```
Enter your choice: 2

=== Running Examples ===

Example 1: "What is an LLM?"
Question: What is an LLM?

📥 RETRIEVAL PHASE
Found 3 documents:
- doc_1 (similarity: 0.85)
- doc_3 (similarity: 0.78)
- doc_2 (similarity: 0.72)

💭 REASONING PHASE
[Shows step-by-step thinking]

🤖 GENERATION PHASE
Response: "An LLM is a large language..."

🎯 EVALUATION
Quality Score: 82/100
- BLEU score: 0.78
- Embedding similarity: 0.84
- Coherence: 0.79

...more examples...
```

---

## 💡 What Just Happened?

Your script:

1. **📥 Retrieved** documents from knowledge base
2. **💭 Reasoned** step-by-step through the problem
3. **🤖 Generated** a response using temperature sampling
4. **🎯 Evaluated** the quality using 5 metrics

All in `09_mini_assistant_complet.py` ✅

---

## 🎮 Interactive Mode

Choose option 1 to ask your own questions:

```
Enter your choice: 1

Ask your question: What is transformers?
Temperature (0.1=focused, 1.0=balanced, 2.0=creative) [default 1.0]: 1.0

📥 RETRIEVAL: Found relevant documents
💭 REASONING: Thinking step-by-step...
🤖 GENERATION: Creating response...
🎯 EVALUATION: Assessing quality...

Response: [Your answer here]
Quality Score: 78/100
```

---

## 🔧 Customization (Advanced)

Want to change behavior? Edit in the script:

```python
# Change these constants at the top of the file:

TEMPERATURE = 1.0        # 0.1 (focused) to 2.0 (creative)
K_DOCUMENTS = 3          # How many documents to retrieve
MAX_TURNS = 3            # Agent iterations
EMBEDDING_DIM = 128      # Embedding dimension
```

Then run again.

---

## 🏆 What You're Learning

By running this script, you're practicing:

✅ **RAG** - Retrieve relevant documents  
✅ **Temperature Sampling** - Control randomness  
✅ **Chain-of-Thought** - Step-by-step reasoning  
✅ **ReAct Agents** - Autonomous loops  
✅ **Evaluation** - Measure quality  

All with educational code you can read and modify.

---

## 🆘 Troubleshooting

**"Module not found: numpy"**
```bash
pip install numpy scikit-learn
```

**"Script doesn't run"**
```bash
# Check Python version
python --version

# Should be 3.9 or higher
```

**"Slow execution"**
- Normal! Demo code prioritizes clarity over speed
- Real systems would use GPU acceleration

---

## 🚀 Next Steps

1. ✅ You've run the script
2. 📖 [Read the architecture](INDEX_SCRIPT_09.md)
3. 🔗 [Map code to concepts](SCRIPT_09_MAPPING.md)
4. 💻 Modify and experiment
5. 🌟 Integrate into your project

---

## 📚 More Resources

- **Understanding concepts?** → [Pedagogical Journey](PEDAGOGICAL_JOURNEY.md)
- **How is it built?** → [Architecture](INDEX_SCRIPT_09.md)
- **Which code teaches what?** → [Code Mapping](SCRIPT_09_MAPPING.md)
- **Agents in detail?** → [ReAct Guide](REACT_AGENT_INTEGRATION.md)
- **RAG in detail?** → [RAG Guide](LLAMAINDEX_GUIDE.md)

---

**Congratulations! 🎉 You're running a mini LLM assistant.**

Try experimenting with different questions and temperature values. See how the system responds differently!

**Questions? Check [Pedagogical Journey](PEDAGOGICAL_JOURNEY.md) for detailed explanations.**
