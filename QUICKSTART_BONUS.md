# 🚀 Quick Start Guide - Bonus Scripts

Démonstration rapide des deux nouveaux bonus scripts. **Aucune installation requise pour les démos.**

---

## 1️⃣ ReAct Agent - Agent Autonome

**Fichier:** `examples/06_react_agent_bonus.py` (380 lignes)  
**Concept:** Agents autonomes avec pattern ReAct (Thought → Action → Observation)  
**Chapitres:** 13, 14

### Exécution (30 secondes)

```bash
cd examples
python 06_react_agent_bonus.py
```

### Sortie attendue

```
================================================================================
🤖 ReAct Agent Demo
================================================================================

📋 Agents Registrés:
  ✓ Calculator Agent
  ✓ Tool-Based Agent

💬 Task 1: Calcule 15 + 27, puis multiplie par 2
[Iteration 1]
  Thought: L'utilisateur me demande de calculer 15 + 27...
  Action: calculator(a=15, b=27, operation=+)
  Observation: 42
  
[Iteration 2]
  Thought: J'ai maintenant 42, je dois le multiplier par 2...
  Action: calculator(a=42, b=2, operation=*)
  Observation: 84

✅ Final Answer: 84

[2 itérations | 0.045 secondes]
```

### Code Clé

```python
# Créer un agent
agent = Agent(name="MyAgent", max_iterations=10)

# Enregistrer un outil
agent.register_tool(
    name="calculator",
    description="Effectue des calculs simples",
    parameters={"a": float, "b": float, "operation": str},
    func=calculator_function
)

# Exécuter une tâche
response = agent.run("Calcule 15 + 27")
```

### Intégration avec un vrai LLM

**Voir:** `examples/REACT_AGENT_INTEGRATION.md`

```python
from openai import OpenAI

class OpenAIAgent(Agent):
    def __init__(self, model="gpt-4", **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI()
        self.model = model
    
    def _simulate_llm_reasoning(self, task, context):
        # Appeler OpenAI au lieu de simuler
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Utiliser
agent = OpenAIAgent(model="gpt-4")
```

---

## 2️⃣ LlamaIndex RAG - Retrieval-Augmented Generation

**Fichier:** `examples/07_llamaindex_rag_advanced.py` (380+ lignes)  
**Concept:** Système RAG complet avec document indexing, chat persistant, évaluation  
**Chapitres:** 13

### Exécution (30 secondes)

```bash
cd examples
python 07_llamaindex_rag_advanced.py
```

### Sortie attendue

```
================================================================================
🦙 LlamaIndex RAG Advanced Demo
================================================================================

📚 Phase 1: Chargement des documents
────────────────────────────────────────────────────────────────────────────────
  ✓ Transformers : Architecture (675 chars)
  ✓ Attention Multi-Tête (571 chars)
  ✓ Fine-tuning et Adaptation (653 chars)

🔍 Phase 2: Création de l'index vectoriel
────────────────────────────────────────────────────────────────────────────────
  ✓ Index créé avec 3 documents
  ✓ Dimension embedding: 384

💬 Phase 4: Requêtes RAG
────────────────────────────────────────────────────────────────────────────────

Q1: Qu'est-ce qu'un Transformer?
📄 Documents retrievés:
   - Transformers : Architecture (0f44208b)
   - Attention Multi-Tête (90ba7a80)
🤖 Réponse:
D'après le contexte fourni, les Transformers sont des architectures basées
sur l'attention qui traitent tous les tokens en parallèle...

💬 Phase 5: Chat avec Mémoire
────────────────────────────────────────────────────────────────────────────────

👤 Utilisateur: Parle-moi des Transformers
🤖 Bot: D'après le contexte...

📊 Phase 6: Évaluation de Qualité
────────────────────────────────────────────────────────────────────────────────
Évaluation du Retrieval:
  - Precision@2: 66.67%
  - Recall@2:    75.00%
  - F1:          70.59%

💾 Résultats exportés dans: rag_results.json
```

### Code Clé

```python
# 1. Créer des documents
docs = [
    SimpleDocument(content, metadata={"title": "Transformers"}),
    SimpleDocument(content, metadata={"title": "Attention"})
]

# 2. Indexer
index = VectorIndex(dimension=384)
for doc in docs:
    index.add_document(doc)

# 3. Requête RAG
rag = SimpleRAGEngine(index)
result = rag.query("Qu'est-ce qu'un Transformer?", top_k=2)

# 4. Chat avec mémoire
chatbot = RAGChatbot(rag)
response1 = chatbot.chat("Parle des Transformers")
response2 = chatbot.chat("Et les RNNs?")  # Avec contexte!

# 5. Évaluation
evaluator = RAGEvaluator()
metrics = evaluator.evaluate_retrieval(query, docs, expected)
```

### Intégration avec LlamaIndex réel

**Voir:** `examples/LLAMAINDEX_GUIDE.md`

```bash
# Installation
pip install llama-index openai

# Utilisation
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configuration
Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Charger documents réels
documents = SimpleDirectoryReader("./documents").load_data()

# Créer index
index = VectorStoreIndex.from_documents(documents)

# Query engine
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("Votre question ici")
```

---

## 📚 Comparaison des Deux Bonus

| Aspect | ReAct Agent | LlamaIndex RAG |
|--------|-------------|----------------|
| **Cas d'usage** | Autonomisation, reasoning | Augmentation avec contexte |
| **Pattern** | Thought → Action → Observation | Retrieval → Augmentation → Generation |
| **Outils** | Fonction, calculatrice, API | Documents, base de connaissances |
| **Mémoire** | Historique d'itérations | Contexte conversationnel |
| **Évaluation** | Itérations, actions | Precision/Recall, similarité |
| **Complexité** | Moyenne | Moyenne → Avancée |

---

## 🎯 Quand utiliser quoi?

### Utiliser ReAct Agent si:
- ✅ Vous avez besoin d'**agents autonomes**
- ✅ Le modèle doit **faire plusieurs actions** (calcul, API call)
- ✅ Vous voulez du **raisonnement pas-à-pas**
- ✅ Exemple: Assistant qui peut calculer, chercher une date, etc.

### Utiliser LlamaIndex RAG si:
- ✅ Vous avez des **documents à rechercher** (PDFs, articles)
- ✅ Vous besoin de **réduire les hallucinations**
- ✅ Vous voulez une **conversation multi-tour** sur des docs
- ✅ Exemple: Chatbot sur votre documentation

### Combiner les deux si:
- ✅ Vous besoin d'un **agent** qui cherche aussi dans des **documents**
- ✅ Pattern: Agent (ReAct) + Outil (RAG query_engine)
- ✅ Exemple: Agent intelligent qui peut raisonner ET chercher

---

## 🔗 Ressources Complètes

| Ressource | Fichier | Contenu |
|-----------|---------|---------|
| **ReAct Intégrations** | `REACT_AGENT_INTEGRATION.md` | OpenAI, Claude, Groq, Ollama |
| **LlamaIndex Complet** | `LLAMAINDEX_GUIDE.md` | Installation, concepts, cas d'usage |
| **Liste Scripts** | `examples/README.md` | Tous les 7 scripts avec descriptions |
| **Changelog** | `BONUS_SCRIPTS_CHANGELOG.md` | Détails des ajouts |

---

## ⚡ Commandes Rapides

```bash
# Vérifier l'installation Python
python --version

# Exécuter Bonus 1
python examples/06_react_agent_bonus.py

# Exécuter Bonus 2
python examples/07_llamaindex_rag_advanced.py

# Installer LlamaIndex pour version réelle
pip install llama-index openai

# Vérifier les résultats exportés
cat examples/rag_results.json | head -20
```

---

## 📝 Notes

- ✅ **Aucune dépendance pour les démos**
- ✅ **Tous les scripts sont testés**
- ✅ **Bien documentés avec commentaires**
- ✅ **Prêts pour GitHub**

---

**Bon code!** 🚀
