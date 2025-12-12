# LlamaIndex Guide Complet - RAG Avancé

Ce guide explique comment utiliser le script `07_llamaindex_rag_advanced.py` et intégrer LlamaIndex dans vos projets.

## 📋 Table des Matières

1. [Qu'est-ce que LlamaIndex?](#intro)
2. [Installation](#installation)
3. [Concepts Clés](#concepts)
4. [Exécution du Script](#execution)
5. [Cas d'Usage Avancés](#usecases)
6. [Intégration avec OpenAI](#openai)
7. [Troubleshooting](#troubleshooting)

---

## <a name="intro"></a>1️⃣ Qu'est-ce que LlamaIndex?

**LlamaIndex** (anciennement GPT Index) est un framework pour construire des applications LLM avancées.

**Cas d'usage:**
- 🔄 **RAG (Retrieval-Augmented Generation)**: Augmenter LLMs avec données externes
- 📄 **Document Q&A**: Poser des questions sur des documents
- 🤖 **Chatbots intelligents**: Agents avec mémoire et contexte
- 📊 **Data analysis**: Extraire insights de données non-structurées
- 🔗 **Knowledge graphs**: Construire des graphes de connaissances

**Architectures supportées:**
```
Data Sources (PDF, Web, Database)
         ↓
    LlamaIndex
         ↓
    (Parsing + Embedding)
         ↓
Vector Store (Pinecone, Weaviate, Chroma, etc.)
         ↓
Query Engine / Retriever
         ↓
LLM (OpenAI, Claude, Ollama, etc.)
         ↓
Réponse final
```

---

## <a name="installation"></a>2️⃣ Installation

### Installation Minimale (Démo)

```bash
# Script démo sans dépendances externes
python examples/07_llamaindex_rag_advanced.py
```

La démo fonctionne **sans aucune installation supplémentaire** (embeddings simulés).

### Installation Complète (Production)

```bash
# Installation de base
pip install llama-index

# Avec OpenAI
pip install llama-index openai

# Avec support de documents (PDF, Word, etc.)
pip install llama-index-readers-file python-pptx openpyxl

# Avec vector stores
pip install llama-index-vector-stores-pinecone
pip install llama-index-vector-stores-weaviate

# Avec autres LLMs
pip install llama-index-llms-anthropic
pip install llama-index-llms-groq
pip install llama-index-llms-ollama
```

### Configuration OpenAI

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-..."

# PowerShell Windows
$env:OPENAI_API_KEY="sk-..."

# Ou dans un fichier .env
OPENAI_API_KEY=sk-...
```

---

## <a name="concepts"></a>3️⃣ Concepts Clés

### A) Document Loading

```python
from llama_index.core import Document, SimpleDirectoryReader

# Charger depuis fichiers
documents = SimpleDirectoryReader("./data").load_data()

# Ou créer manuellement
doc = Document(
    text="Contenu du document",
    metadata={"source": "manual", "date": "2025-01"}
)
```

### B) Vector Index

```python
from llama_index.core import VectorStoreIndex

# Créer un index à partir de documents
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model  # Choisir modèle d'embedding
)

# Sauvegarder pour réutilisation
index.storage_context.persist("./storage")

# Charger
from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

### C) Query Engine

```python
# Créer un query engine
query_engine = index.as_query_engine(
    similarity_top_k=3  # Retriever les 3 documents les plus pertinents
)

# Exécuter une requête
response = query_engine.query("Qu'est-ce qu'un Transformer?")
print(response)
```

### D) Chat Engine (avec mémoire)

```python
# Chat avec historique conversationnel
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",  # Reformuler Q avec contexte
    memory=ChatMemoryBuffer(token_limit=3900)
)

# Messages
response = chat_engine.chat("Parle-moi des Transformers")
response = chat_engine.chat("Et les mécanismes d'attention?")  # Avec contexte!
```

### E) Hybrid Search (BM25 + Vector)

```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

# Combiner keyword (BM25) + semantic (vector) search
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore, 
    nodes=nodes
)
vector_retriever = index.as_retriever(similarity_top_k=3)

fusion_retriever = QueryFusionRetriever(
    [bm25_retriever, vector_retriever],
    similarity_top_k=3,
    query_gen_prompt="Reformule cette question de 3 façons..."
)

# Retriever les meilleurs results
nodes = fusion_retriever.retrieve("Qu'est-ce qu'un Transformer?")
```

---

## <a name="execution"></a>4️⃣ Exécution du Script

### Exécution Basique

```bash
cd examples
python 07_llamaindex_rag_advanced.py
```

**Output attendu:**

```
================================================================================
🦙 LlamaIndex RAG Advanced Demo
================================================================================

📚 Phase 1: Chargement des documents
────────────────────────────────────────────────────────────────────────────────
  ✓ Transformers : Architecture (1234 chars)
  ✓ Attention Multi-Tête (890 chars)
  ✓ Fine-tuning et Adaptation (1050 chars)

🔍 Phase 2: Création de l'index vectoriel
────────────────────────────────────────────────────────────────────────────────
  ✓ Index créé avec 3 documents
  ✓ Dimension embedding: 384

⚙️  Phase 3: Initialisation du RAG Engine
────────────────────────────────────────────────────────────────────────────────
  ✓ RAG Engine prêt

💬 Phase 4: Requêtes RAG
────────────────────────────────────────────────────────────────────────────────

Q1: Qu'est-ce qu'un Transformer?
📄 Documents retrievés:
   - Transformers : Architecture (a1b2c3d4)
   - Attention Multi-Tête (e5f6g7h8)

🤖 Réponse:
D'après le contexte fourni, les Transformers sont des architectures basées
sur l'attention qui traitent tous les tokens en parallèle, contrairement aux RNNs...

[Suite de la démonstration...]

✅ Démo complétée!
💾 Résultats exportés dans: rag_results.json
```

### Résultat JSON

Le script génère `rag_results.json`:

```json
{
  "timestamp": "2025-01-10T14:23:45.123456",
  "queries": [
    {
      "question": "Qu'est-ce qu'un Transformer?",
      "retrieved_docs": [
        {
          "title": "Transformers : Architecture",
          "id": "a1b2c3d4",
          "snippet": "Le Transformer est une architecture de réseau profond..."
        }
      ],
      "answer": "D'après le contexte...",
      "timestamp": "2025-01-10T14:23:45.123456"
    }
  ],
  "statistics": {
    "total_queries": 3,
    "total_turns": 3,
    "documents_indexed": 3
  }
}
```

---

## <a name="usecases"></a>5️⃣ Cas d'Usage Avancés

### A) RAG sur PDFs

```python
from llama_index.core import SimpleDirectoryReader

# Charger tous les PDFs
documents = SimpleDirectoryReader("./pdfs/").load_data()

# Créer l'index
index = VectorStoreIndex.from_documents(documents)

# Requête
query_engine = index.as_query_engine()
response = query_engine.query("Quel est le chapitre 3?")
```

### B) Chat Multi-Tour avec Mémoire

```python
from llama_index.core.memory import ChatMemoryBuffer

chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    memory=ChatMemoryBuffer(token_limit=3900),
    system_prompt="Tu es un expert en IA..."
)

# Tour 1
response1 = chat_engine.chat("Parle des Transformers")
# Chat conserve le contexte

# Tour 2
response2 = chat_engine.chat("Qu'en est-il des MoE?")
# La question est augmentée avec le contexte du tour 1
```

### C) Evaluation RAG

```python
from llama_index.core.evaluation import (
    RelevancyEvaluator,
    FaithfulnessEvaluator
)

evaluator_relevancy = RelevancyEvaluator()
evaluator_faithfulness = FaithfulnessEvaluator()

# Évaluer une réponse
eval_result = evaluator_relevancy.evaluate_response(
    query="Qu'est-ce qu'un Transformer?",
    response=response
)

print(f"Pertinence: {eval_result.score}")
print(f"Raison: {eval_result.feedback}")
```

### D) Agents avec Outils

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# Définir des outils
def calculator(a: int, b: int, op: str) -> int:
    if op == "+": return a + b
    if op == "*": return a * b
    return 0

tools = [
    FunctionTool.from_defaults(fn=calculator),
    # ... autres outils
]

# Créer un agent ReAct
agent = ReActAgent.from_tools(tools, llm=llm)

# Exécuter
response = agent.chat("Calcule 5 + 3 puis multiplie par 2")
```

### E) Hybrid Search pour Meilleures Résultats

```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

# Créer les retrievers
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes)
vector_retriever = index.as_retriever(similarity_top_k=3)

# Fusionner
fusion_retriever = QueryFusionRetriever(
    [bm25_retriever, vector_retriever],
    similarity_top_k=3
)

# Utiliser
nodes = fusion_retriever.retrieve("Transformer attention")
```

---

## <a name="openai"></a>6️⃣ Intégration avec OpenAI

### Configuration Recommandée

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# LLM
Settings.llm = OpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=2048
)

# Embeddings
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",  # Moins cher, bon pour RAG
)

# Ou GPT-4o pour meilleure qualité
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
```

### Coûts Estimés

| Modèle | Input (1K tokens) | Output (1K tokens) |
|--------|-------------------|-------------------|
| GPT-4o mini | $0.00015 | $0.0006 |
| text-embedding-3-small | $0.00002 | - |
| text-embedding-3-large | $0.00013 | - |

**Exemple pour 1000 requêtes:**
- LLM (gpt-4o-mini): ~$0.75
- Embeddings (small): ~$0.02
- **Total: ~$0.77 pour 1000 requêtes** ✅

---

## <a name="troubleshooting"></a>7️⃣ Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'llama_index'"

**Solution:**
```bash
pip install llama-index
```

### ❌ "OpenAI API key not found"

**Solution:**
```bash
# Vérifier que la clé est définie
echo $OPENAI_API_KEY  # Linux/Mac
echo $env:OPENAI_API_KEY  # PowerShell

# Ou créer un .env
echo 'OPENAI_API_KEY=sk-...' > .env
```

### ❌ "Embedding model is required"

**Solution:**
```python
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding()
```

### ❌ "No documents found in directory"

**Solution:**
```bash
# Vérifier le répertoire
ls -la ./data/

# Ou spécifier le chemin absolu
from llama_index.core import SimpleDirectoryReader
docs = SimpleDirectoryReader("/absolute/path/to/data").load_data()
```

### ⚠️ Embeddings lents

**Solution:**
```python
# Utiliser un modèle plus rapide
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"  # Plus rapide que large
)

# Ou Ollama en local
from llama_index.embeddings.ollama import OllamaEmbedding
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
```

### 🐢 Query Engine lent

**Solutions:**
1. Réduire `similarity_top_k`:
```python
query_engine = index.as_query_engine(similarity_top_k=2)  # au lieu de 5
```

2. Utiliser cache:
```python
from llama_index.core.cache import GPTCache
from gptcache import Cache

gptcache = GPTCache()
Settings.cache = gptcache
```

3. Batch plusieurs requêtes:
```python
responses = []
for query in queries:
    response = query_engine.query(query)
    responses.append(response)
```

---

## 📚 Ressources Avancées

- **Docs officielles**: https://docs.llamaindex.ai/
- **Community**: https://discord.gg/dGcwcsnxhU
- **GitHub**: https://github.com/run-llama/llama_index
- **Exemples**: https://github.com/run-llama/llama_index/tree/main/examples

---

## 🎯 Prochaines Étapes

1. ✅ Exécuter le script démo: `python examples/07_llamaindex_rag_advanced.py`
2. ✅ Installer LlamaIndex: `pip install llama-index openai`
3. ✅ Charger vos propres documents: `SimpleDirectoryReader("./data")`
4. ✅ Configurer OpenAI: `export OPENAI_API_KEY=sk-...`
5. ✅ Intégrer dans votre app: Voir Cas d'Usage Avancés
6. ✅ Évaluer la qualité: Utiliser `RelevancyEvaluator`, `FaithfulnessEvaluator`

---

**Bon développement!** 🚀
