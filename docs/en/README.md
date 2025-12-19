# Practical Scripts: Experimenting with LLM Concepts

🌍 **English** | 📖 **[Version Française](../fr/README.md)**

Collection of **9 executable Python scripts** to experiment with the key concepts from the book **"The Mechanics of LLMs"**.

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

---

## 🚀 Quick Start

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
