# Scripts Pratiques : Expérimenter les Concepts LLM

🌍 **[English Version](#english-version)** | 📖 **Français**

Collection de **9 scripts Python exécutables** (+ documentation) pour expérimenter les concepts clés présentés dans le livre **"La Mécanique des LLM"**.

> 📚 **À propos** : Ces scripts accompagnent les chapitres du livre. Voir [Correspondance Livre ↔ Scripts](docs/fr/PEDAGOGICAL_JOURNEY.md) pour les liens détaillés.

**📕 Acheter le livre :**
- **Broché** : [Amazon](https://amzn.eu/d/3oREERI)
- **Kindle** : [Amazon](https://amzn.eu/d/b7sG5iw)

---

## 📋 Vue d'Ensemble des Scripts

| # | Script | Chapitre(s) | Concepts | Status |
|---|--------|-----------|----------|--------|
| 1 | [01_tokenization_embeddings.py](#script-1--tokenisation-et-embeddings) | 2 | Tokenisation, impact sur la longueur de séquence | ✅ |
| 2 | [02_multihead_attention.py](#script-2--multi-head-attention) | 3 | Self-attention, multi-head, poids d'attention | ✅ |
| 3 | [03_temperature_softmax.py](#script-3--température-et-softmax) | 7, 11 | Température, softmax, entropie | ✅ |
| 4 | [04_rag_minimal.py](#script-4--pipeline-rag-minimal) | 13 | Pipeline RAG, retrieval, similarité cosinus | ✅ |
| 5 | [05_pass_at_k_evaluation.py](#script-5--évaluation-pass-k) | 12 | Pass@k, Pass^k, évaluation de modèles | ✅ |
| 🎁 6 | [06_react_agent_bonus.py](#bonus-1--react-agent-avec-framework-générique) | 14, 15 | **Agents ReAct, tool registration, MCP** | ✅ BONUS |
| 🎁 7 | [07_llamaindex_rag_advanced.py](#bonus-2--rag-avancé-avec-llamaindex) | 13, 14 | **RAG avancé, indexing, chat persistant** | ✅ BONUS |
| 🎁 8 | [08_lora_finetuning_example.py](#bonus-3--lora-et-fine-tuning) | 9, 10 | **LoRA, QLoRA, fine-tuning comparatif** | ✅ BONUS |
| 🏆 **9** | [09_mini_assistant_complet.py](#-projet-intégrateur--mini-assistant-complet) | **11-15** | **🎯 Projet Final Intégrateur** | ✅ FLAGSHIP |

## 🚀 Démarrage Rapide

### 1. Créer un environnement virtuel (recommandé)

```bash
# Sur Windows
python -m venv venv
venv\Scripts\activate

# Sur macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
# Installation basique (pour les scripts 1-5)
pip install torch transformers numpy scikit-learn

# Installation complète (avec visualisations)
pip install torch transformers numpy scikit-learn matplotlib

# Pour les bonus (optionnel, scripts fonctionnent aussi sans)
pip install llama-index openai python-dotenv peft bitsandbytes
```

**Note:** Les scripts bonus (06, 07, 08) fonctionnent **sans dépendances externes**
en mode démo. Ils utilisent des simulations/calculs pour illustrer les concepts.

### 3. Exécuter un script

```bash
python 01_tokenization_embeddings.py
python 02_multihead_attention.py
python 03_temperature_softmax.py
python 04_rag_minimal.py
python 05_pass_at_k_evaluation.py
python 06_react_agent_bonus.py
python 07_llamaindex_rag_advanced.py
python 08_lora_finetuning_example.py
python 09_mini_assistant_complet.py    # ← Projet intégrateur final
```

## 📖 Détails par Script

### Script 1 : Tokenisation et Embeddings (Chapitre 2)

**Voir:** `02-representation-texte-modeles-sequentiels.md`

Illustre :
- Comment les tokenizers (BPE, WordPiece) fragmentent le texte.
- L'impact du nombre de tokens sur le coût computationnel.
- Les différences entre langues (français vs anglais).

```bash
python 01_tokenization_embeddings.py
```

**Exemple de sortie:**
```
Texte: L'IA est utile
  Nombre de tokens: 6
  Token IDs: [43, 6, 3539, 1556, 3384, 576]
  Tokens (texte): ['L', "'", 'IA', 'Ġest', 'Ġut', 'ile']

Texte court (7 caractères) → 3 tokens
Texte long (700 caractères) → 300 tokens
Facteur: 100.0x

⚠️ IMPLICATIONS:
  • Plus de tokens = plus de VRAM
  • Plus de tokens = latence plus élevée
  • Coût d'inférence ∝ O(n²) pour l'attention
```

---

### Script 2 : Multi-Head Attention (Chapitre 3)

**Voir:** `03-architecture-transformer.md`

Simule une couche d'attention multi-tête minimale :
- Projections Q, K, V.
- Calcul des scores d'attention.
- Visualisation de comment chaque tête focalise différemment.

```bash
python 02_multihead_attention.py
```

**Exemple de sortie:**
```
Entrée x shape: (1, 4, 64)
  (batch=1, seq_len=4, d_model=64)

Tête 0:
  Poids d'attention (après softmax):
    [[0.25 0.35 0.25 0.15]  # Le "regarde" 35% vers "chat"
     [0.10 0.60 0.20 0.10]  # "chat" regarde 60% vers "dort"
     ...

💡 INTUITION:
  • Chaque tête capture DIFFÉRENTES dépendances.
  • Tête 0 peut se concentrer sur sujet-verbe.
  • Tête 1 peut se concentrer sur verbe-adverbe.
```

---

### Script 3 : Température et Softmax (Chapitres 7 & 11)

**Voir:** `07-preentrainement-llms.md`, `11-strategies-generation-inference.md`

Montre l'effet de la température sur la distribution softmax :
- Basse T → distribution pointue (greedy, déterministe).
- Haute T → distribution plate (diversité, créativité).
- Lien avec l'entropie.

```bash
python 03_temperature_softmax.py
```

**Exemple de sortie:**
```
Température = 0.1
  Probabilités:
    chat    : 1.000  █████████████████████████████████████████████████
    chien   : 0.000
    souris  : 0.000
    oiseau  : 0.000
  Entropie: 0.001

Température = 5.0
  Probabilités:
    chat    : 0.308  ███████████████
    chien   : 0.252  ████████████
    souris  : 0.228  ███████████
    oiseau  : 0.211  ██████████
  Entropie: 1.376

✓ À T=0.1 → Déterministe (distribution pointue, 'chat' domine à 100%).
✓ À T=5.0 → Quasi-uniforme (distribution plate, tous les tokens similaires).
✓ T=0.7-0.9 → Bon compromis créativité/stabilité.
```

Génère optionnellement un graphique : `temperature_effect.png`

---

### Script 4 : RAG Minimaliste (Chapitre 13)

**Voir:** `13-systemes-augmentes-agents.md`

Simule un pipeline RAG complet :
1. **Retrieval** : chercher les 3 documents les plus pertinents.
2. **Augmentation** : injecter le contexte dans le prompt.
3. **Génération** : le LLM répond en s'appuyant sur le contexte.

```bash
python 04_rag_minimal.py
```

**Exemple de sortie:**
```
Question: "Comment fonctionne l'attention dans le Transformer?"

Top 3 documents récupérés:
1. Score: 0.223
   Le Transformer est une architecture basée sur l'attention multi-tête.

2. Score: 0.102
   Le Transformer a été introduit en 2017 par Vaswani et ses collègues.

Prompt augmenté envoyé au LLM:
---
Vous êtes un assistant expert.

Voici des documents pertinents:
- L'attention multi-tête permet...
- Le Transformer est une architecture...
- ...

Question: Comment fonctionne l'attention dans le Transformer?

Réponse basée sur les documents:
---

COMPARAISON:
❌ SANS RAG:
  → Hallucination possible
  → Connaissances figées
  → Pas de sources à vérifier

✅ AVEC RAG:
  → Réponses basées sur des sources
  → Accès aux données externes
  → Utilisateur peut vérifier les sources
```

---

### Script 5 : Évaluation Pass@k (Chapitre 12)

**Voir:** `12-modeles-raisonnement.md`

Évalue la fiabilité d'un modèle sur des tâches vérifiables :
- **Pass@k** : probabilité d'au moins **une** réussite en k tentatives.
- **Pass^k** : probabilité que **toutes** les k tentatives réussissent.

```bash
python 05_pass_at_k_evaluation.py
```

**Exemple de sortie:**
```
Paramètres:
  • Nombre de tentatives: 100
  • Probabilité de succès: 30%

PASS@K (Au moins UNE réussite en k tentatives):
Pass@1  = 34.0% (1 tentative)
Pass@3  = 71.3% (3 tentatives)
Pass@5  = 87.5% (5 tentatives)
Pass@10 = 98.4% (10 tentatives)

PASS^K (TOUTES les k tentatives réussissent) — STRICT:
Pass^1 = 34.0% empirique / 30.0% théorique (0.3^1)
Pass^3 =  0.0% empirique /  2.7% théorique (0.3^3)
Pass^5 =  0.0% empirique /  0.2% théorique (0.3^5)

APPLICATION:
  ✓ Recherche (HumanEval): Pass@k (diversité)
  ✓ Agents critiques: Pass^k (fiabilité totale)
```

---

## 🎯 Comment Utiliser Ces Scripts

### Pour les Étudiants

1. **Lisez le chapitre pertinent du livre.**
2. **Exécutez le script associé.**
3. **Modifiez les paramètres** pour voir les effets :
   - Changez `seq_len`, `num_heads`, `temperatures`, etc.
   - Ajoutez vos propres textes/documents.
4. **Ajoutez des `print()`** pour déboguer et comprendre les dimensions.

### Pour les Ingénieurs

- Utilisez ces scripts comme **point de départ** pour vos implémentations.
- Intégrez-les dans des **pipelines de production** (RAG, évaluation, etc.).
- Adaptez le code à votre **infrastructure** (GPUs, APIs, bases de données).

## � Projet Intégrateur : Mini-Assistant Complet (`09_mini_assistant_complet.py`)

**Voir:** Chapitres 11-15 du livre

Ce script final **assemble TOUS les concepts du livre** en un système cohérent :
- **RAG (Ch. 13)** : Indexation vectorielle TF-IDF et recherche par similarité
- **Agents ReAct (Ch. 14)** : Boucle Thought→Action→Observation avec tool calling
- **Prompting (Ch. 11)** : Zero-shot, Few-shot, Chain-of-Thought pour structures les réponses
- **Évaluation (Ch. 12, 15)** : Confiance, self-consistency, métriques de qualité
- **Outils** : Calculatrice, recherche, horloge, résumé

**Parcours pédagogique du chapitre 11 au 15 :**

1. **Chapitre 11 (Prompting)** → Structurer les demandes avec Chain-of-Thought
2. **Chapitre 12 (Évaluation)** → Mesurer la qualité avec Pass@k et confiance
3. **Chapitre 13 (RAG)** → Augmenter le contexte avec documents pertinents
4. **Chapitre 14 (Agents)** → Boucle autonome avec tool calling et réactions
5. **Chapitre 15 (Mise en production)** → Assembler tout cela en système robuste

**Phases d'exécution :**
1. Initialisation de la base de connaissances (5 documents démo)
2. Création de l'agent avec 4 outils enregistrés
3. Traitement de 3 questions test
4. Évaluation des réponses (itérations, confiance, succès)
5. **Bonus** : Test de self-consistency (même question, 3 essais)
6. Rapport global des performances

**Exécution :**
```bash
python 09_mini_assistant_complet.py
```

**Exemple de sortie :**
```
🚀 MINI-ASSISTANT COMPLET - PROJET INTÉGRATEUR

📚 Phase 1 : Initialisation de la base de connaissances
✓ Index créé : 5 documents indexés

🤖 Phase 2 : Création de l'agent
✓ Agent créé avec 4 outils

💬 Phase 3 : Questions de test

🤖 Question : Qu'est-ce qu'un Transformer ?
💭 Pensée : Je dois chercher des informations sur transformer
🔧 Action : search(query='transformer')
📊 Observation : Documents trouvés:
  [Architecture Transformer] (score: 0.89)
  Les Transformers sont une architecture...

✅ Réponse finale : Les Transformers sont une architecture...

📊 Phase 4 : Rapport d'évaluation
Question 1 : Qu'est-ce qu'un Transformer ?
  • Itérations : 1
  • Confiance : 100.00%
  • Succès : ✅

📈 Statistiques globales
  • Nombre de questions : 3
  • Itérations moyennes : 1.3
  • Confiance moyenne : 88.33%
  • Taux de succès : 100.00%
```

**Points d'extension pour les étudiants :**
1. Intégrer OpenAI, Claude ou un modèle local (Ollama)
2. Ajouter de nouveaux outils (météo, API, base de données)
3. Persister les conversations (SQLite, PostgreSQL)
4. Créer une interface web (Streamlit, Gradio, FastAPI)
5. Implémenter des métriques avancées (ROUGE, BERTScore)
6. Gérer les contexts longs et la pagination
7. Déployer en production (Docker, Kubernetes)

---

## 🎁 Autres Bonus Scripts

### BONUS 1 : ReAct Agent Framework (`06_react_agent_bonus.py`)

**Voir:** `REACT_AGENT_INTEGRATION.md`

Framework complet pour construire des **agents autonomes avec pattern ReAct**.

**Caractéristiques:**
- ✅ Classe `Agent` réutilisable et extensible
- ✅ Système de registration d'outils (tool definition)
- ✅ Boucle Thought → Action → Observation
- ✅ LLM simulation (prêt pour OpenAI, Claude, Groq, Ollama)
- ✅ Historique et gestion d'itérations
- ✅ Exemple avec 3 outils (calculator, date, knowledge base)

**Concepts couverts:**
- Agents autonomes (Chapitre 13)
- Protocoles standards agentiques (Chapitre 14)
- Pattern ReAct (Reasoning + Acting)
- Tool calling et execution

**Exécution:**
```bash
python 06_react_agent_bonus.py
```

**Intégration avec LLMs réels:**
```python
from openai import OpenAI

class OpenAIAgent(Agent):
    def _simulate_llm_reasoning(self, task, context):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

Voir `REACT_AGENT_INTEGRATION.md` pour intégrations complètes (OpenAI, Claude, Groq, Ollama).

---

### BONUS 2 : LlamaIndex RAG Avancé (`07_llamaindex_rag_advanced.py`)

**Voir:** `LLAMAINDEX_GUIDE.md`

Framework complet pour construire des **systèmes RAG avancés avec LlamaIndex**.

**Caractéristiques:**
- ✅ Indexation vectorielle d'documents
- ✅ Retrieval avancé (similarity search, hybrid BM25+vector)
- ✅ RAG Engine avec augmentation de contexte
- ✅ Chatbot avec mémoire conversationnelle
- ✅ Évaluation de qualité (Precision, Recall, F1)
- ✅ Export des résultats en JSON
- ✅ Fallback embeddings simulés (pas de dépendances requises)

**Concepts couverts:**
- RAG (Retrieval-Augmented Generation) - Chapitre 13
- Document parsing et indexing
- Vector similarity search
- Query augmentation avec contexte
- Conversation avec persistance

**Phases d'exécution:**
1. Chargement des documents
2. Création de l'index vectoriel
3. Initialisation du RAG Engine
4. Requêtes RAG avec retrieval
5. Chat avec mémoire
6. Évaluation de qualité
7. Export des résultats

**Exécution sans dépendances:**
```bash
python 07_llamaindex_rag_advanced.py
```
⚠️ Utilise embeddings simulés (déterministes).

**Exécution avec LlamaIndex réel:**
```bash
pip install llama-index openai
python 07_llamaindex_rag_advanced.py
```
✓ Utilise OpenAI embeddings (text-embedding-3-small).

**Production avec documents réels:**
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Charger PDFs, Word, HTML, etc.
documents = SimpleDirectoryReader("./docs").load_data()

# Créer index
index = VectorStoreIndex.from_documents(documents)

# Query engine
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("Votre question ici")
```

Voir `LLAMAINDEX_GUIDE.md` pour :
- Installation complète (LlamaIndex, vector stores, readers)
- Intégration OpenAI, Claude, Groq
- Hybrid search (BM25 + vectoriel)
- Agents avec outils
- Évaluation de qualité
- Cas d'usage avancés

---

### Script 8 : LoRA & QLoRA Fine-tuning (Chapitre 9) 🎁 BONUS 3

**Voir:** `09-affinage-supervise-sft.md`

Démontre les techniques de fine-tuning efficace en ressources :
- LoRA (Low-Rank Adaptation) : réduction des paramètres entraînables.
- QLoRA (Quantized LoRA) : quantification + LoRA pour VRAM ultra-faible.
- Comparaison chiffrée : Full Fine-tuning vs LoRA vs QLoRA.
- Cas réel : adaptation d'un modèle LLaMA-7B pour le domaine ferroviaire (SNCF).

```bash
python 08_lora_finetuning_example.py
```

**Exemple de sortie:**
```
================================================================================
EXEMPLE 1 : Fine-tuner LLaMA-7B
================================================================================

Modèle : LLaMA-7B (7.0B paramètres)
LoRA rank : 8

Comparaison des méthodes :
Méthode              Params          VRAM      Temps   Cas d'usage
full_fine_tuning     7000.0M        26.1GB     1.0x  → Meilleure performance
lora                    2.1M         6.5GB     0.3x  → Bon compromis
qlora                   2.1M         1.6GB     0.4x  → RÉVOLUTION : fine-tune sur GPU basic

INSIGHT :
  • Full fine-tuning : 28 GB VRAM → nécessite A100 ou RTX 6000
  • LoRA : 8 GB VRAM → entraînable sur RTX 4090 (24 GB)
  • QLoRA : 2 GB VRAM → entraînable sur RTX 3090 ✅ RÉVOLUTION!
```

Concepts abordés :
- W = W₀ + BA (décomposition LoRA)
- Effet du rank (8, 16, 32, 64) sur taille vs performance
- Quantisation 8-bit et économies de mémoire
- Peft library integration (transformers + peft)
- Pseudo-code pour adapter modèles multilingues

---

## 📚 Correspondance Livre ↔ Scripts

| Chapitre | Topic | Script |
|----------|-------|--------|
| 2 | Tokenisation, Embeddings | `01_tokenization_embeddings.py` |
| 3 | Architecture Transformer, Attention | `02_multihead_attention.py` |
| 7 | Pré-entraînement, Loss | `03_temperature_softmax.py` |
| 9 | Affinage supervisé, LoRA, QLoRA | **`08_lora_finetuning_example.py`** |
| 11 | Stratégies de génération, Température | `03_temperature_softmax.py` |
| 12 | Modèles de raisonnement, Évaluation | `05_pass_at_k_evaluation.py` |
| 13 | Systèmes augmentés, RAG, Agents | `04_rag_minimal.py`, **`06_react_agent_bonus.py`**, **`07_llamaindex_rag_advanced.py`** |
| 14 | Protocoles standards agentiques | **`06_react_agent_bonus.py`** |
| **11-15** | **Projet Intégrateur Complet** | **`09_mini_assistant_complet.py`** |

### Parcours Pédagogique Recommandé

**Phase 1 : Fondamentaux (Chapitres 1-7)**
→ Exécutez les scripts 1, 2, 3 pour comprendre les mécaniques de base

**Phase 2 : Évaluation et RAG (Chapitres 9-13)**
→ Scripts 5, 4, 6, 7, 8 pour maîtriser évaluation, retrieval, agents avancés

**Phase 3 : Intégration (Chapitres 11-15)** ← **Vous êtes ici**
→ **Script 9** : Assembler tous les concepts en un mini-assistant cohérent
→ Comprendre comment RAG + Agents + Prompting + Évaluation travaillent ensemble
→ Point d'ancrage pour vos propres extensions en production

---

## 🏆 Projet Intégrateur : Mini-Assistant Complet (`09_mini_assistant_complet.py`)

**LE script phare** : intègre TOUS les concepts des chapitres 11-15 en un seul projet exécutable.

### 📍 Localisation dans le parcours pédagogique

| Chapitre | Sujet | Utilisé Dans ? |
|----------|-------|---|
| **11** | Stratégies de génération et inférence | ✅ Tempérture, top-k, top-p |
| **12** | Modèles de raisonnement (CoT, ToT) | ✅ Chain-of-Thought prompt |
| **13** | Systèmes augmentés et agents (RAG) | ✅ Retrieval + indexing |
| **14** | Protocoles standards agentiques (MCP) | ✅ Tool registration, agents |
| **15** | Évaluation critique des flux agentiques | ✅ Métriques + évaluation |

### 🎯 Fonction du script

L'assistant démontre :
1. **Contexte Enrichi** : RAG pour mémoire externe
2. **Raisonnement** : Chain-of-Thought reasoning
3. **Agentivité** : Agent auto-suffisant prenant des décisions
4. **Évaluation** : Métriques BLEU, embedding similarity, cohérence

### 🚀 Exécuter

```bash
python 09_mini_assistant_complet.py
```

**Voir aussi:**
- [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md) - Vue d'ensemble architecture
- [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md) - Guide démarrage rapide
- [SCRIPT_09_MAPPING.md](SCRIPT_09_MAPPING.md) - Correspondance concepts ↔ code

---

## 🎁 Autres Bonus Scripts

### Bonus 1 : ReAct Agent (`06_react_agent_bonus.py`)

Pattern **ReAct** (Reasoning + Acting) avec framework générique, tool registration et 3 outils d'exemple.

**Voir:** [REACT_AGENT_INTEGRATION.md](REACT_AGENT_INTEGRATION.md)

### Bonus 2 : RAG Avancé (`07_llamaindex_rag_advanced.py`)

Framework RAG complet : document ingestion, indexing, 6 phases d'exécution, export JSON.

**Voir:** [LLAMAINDEX_GUIDE.md](LLAMAINDEX_GUIDE.md)

### Bonus 3 : LoRA Fine-Tuning (`08_lora_finetuning_example.py`)

Techniques d'optimisation : LoRA, QLoRA, comparaison fine-tuning.

---

## 📖 Correspondance Livre ↔ Scripts (Parcours Pédagogique)

```
📖 Chapitres du Livre                  →  💻 Scripts Correspondants
────────────────────────────────────────────────────────────────

Ch. 2  : Représentation texte         →  01_tokenization_embeddings.py
Ch. 3  : Architecture Transformer     →  02_multihead_attention.py
Ch. 7  : Pré-entraînement             →  03_temperature_softmax.py
Ch. 9  : Fine-tuning                  →  08_lora_finetuning_example.py 🎁
Ch. 11 : Génération & Inférence       →  03_temperature_softmax.py (bis)
                                       →  09_mini_assistant_complet.py 🏆
Ch. 12 : Raisonnement & Évaluation    →  05_pass_at_k_evaluation.py
                                       →  09_mini_assistant_complet.py 🏆
Ch. 13 : Systèmes Augmentés (RAG)     →  04_rag_minimal.py
                                       →  07_llamaindex_rag_advanced.py 🎁
                                       →  09_mini_assistant_complet.py 🏆
Ch. 14 : Protocoles Agentiques (MCP)  →  06_react_agent_bonus.py 🎁
                                       →  09_mini_assistant_complet.py 🏆
Ch. 15 : Évaluation Critique          →  09_mini_assistant_complet.py 🏆
```

---

## 🛠️ Troubleshooting

### "ModuleNotFoundError: No module named 'transformers'"

```bash
pip install transformers
```

### "ModuleNotFoundError: No module named 'torch'"

```bash
# CPU
pip install torch

# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Matplotlib non installé

```bash
pip install matplotlib
# Script 3 continuera à fonctionner sans, mais pas de graphique.
```

### Script trop lent (transformers qui télécharge un modèle)

- Les modèles se téléchargent automatiquement à la première exécution (~3 GB pour LLaMA).
- Les prochaines exécutions seront plus rapides (cache local).
- Alternative : utiliser un modèle plus petit (`distilbert-base-multilingual-cased`).

## 📝 Notes

- **Pas de GPU requis** : tous les scripts tournent sur CPU (plus lentement).
- **Dépendances minimales** : seulement `numpy`, `torch`, `transformers`, `scikit-learn`.
- **Code éducatif** : les scripts privilégient la clarté sur l'optimisation.
- **Compatible Python 3.9+**.
- **Scripts bonus** : démontrent des concepts avancés, fonctionnent sans LLM externe (mode simulation).

---

**Bon apprentissage! 🚀**
